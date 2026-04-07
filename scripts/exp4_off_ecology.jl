"""
Experiment 4: Off-Ecology Error

Tests: the proposition `off-ecology excess bound` and an empirical
proxy for the proposition `off-ecology non-identifiability`.

Train multiple models on a reduced ecology (English, French, German), then probe
on all 5 Alice languages plus the Voynich manuscript (EVA transcription).

On-ecology: English, French, German (training languages).
Off-ecology (related): Italian, Finnish (same script, unseen).
Off-ecology (alien): Voynich.

Off-ecology excess bound: compute excess loss (L_hat - H_irred) for
on-ecology vs off-ecology. The on-ecology excess should be small; the
off-ecology excess should be larger because the model's encoding does not
preserve off-ecology distinctions.

Off-ecology non-identifiability proxy: show that models achieving
near-equal on-ecology held-out fit produce divergent off-ecology predictions
(high inter-model JS off-ecology despite low inter-model JS on-ecology).

Voynich caveat: the EVA transcription is projected through the Alice-derived
character tokenizer, so EVA characters outside a-z are dropped. This weakens
the "maximally alien" interpretation but the probe remains informative as
an extreme out-of-distribution test.

Run:  julia --project=. scripts/exp4_off_ecology.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "microgpt.jl"))
include(joinpath(@__DIR__, "..", "src", "data.jl"))
include(joinpath(@__DIR__, "..", "src", "measurement.jl"))

using Random, LinearAlgebra

BLAS.set_num_threads(1)

# ── Configuration ─────────────────────────────────────────────────────────────

const ALICE_DIR    = joinpath(@__DIR__, "..", "data", "alice")
const VOYNICH_DIR  = joinpath(@__DIR__, "..", "data", "voynich")
const OUTPUT_DIR   = joinpath(@__DIR__, "..", "output")
const N_MODELS     = 10
const N_TRAIN_STEPS = 15000
const LR           = 0.008
const PREFIX_LENGTHS = [3, 5, 8]
const CHUNK_SIZE   = 30
const CHUNK_OVERLAP = 5
const PREFIX_SAMPLES = 160
const SPECIALIST_RESTARTS = 1

const TRAIN_LANGUAGES = ["english", "french", "german"]
const ALL_ALICE_LANGUAGES = ["english", "french", "german", "italian", "finnish"]

const ALICE_FILE_MAP = Dict(
    "english" => "alice_english.txt",
    "french"  => "alice_french.txt",
    "german"  => "alice_german.txt",
    "italian" => "alice_italian.txt",
    "finnish" => "alice_finnish.txt",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

function train_model(train_tokens::Vector{Vector{Int}}, cfg::ModelConfig,
                     seed::Int; n_steps::Int=N_TRAIN_STEPS, lr::Float64=LR,
                     verbose::Bool=false)
    ms = init_model(cfg; seed=seed)
    rng = MersenneTwister(seed + 10000)
    loss_sum = 0.0
    loss_count = 0
    t0 = time()

    for step in 1:n_steps
        idx = rand(rng, 1:length(train_tokens))
        tokens = train_tokens[idx]
        length(tokens) < 2 && continue
        lr_t = lr * max(0.0, 1.0 - step / n_steps)
        l = train_step_lr!(ms, tokens, step, cfg, lr_t)
        loss_sum += l
        loss_count += 1
        if verbose && step % (n_steps ÷ 4) == 0
            avg = loss_sum / loss_count
            println("    " * progress_str(step, n_steps, "train";
                    start_time=t0) * " avg_loss=" * string(round(avg, digits=4)))
            loss_sum = 0.0
            loss_count = 0
        end
    end
    ms
end

function best_specialist_ce(train_tokens::Vector{Vector{Int}},
                            eval_tokens::Vector{Vector{Int}},
                            cfg::ModelConfig,
                            base_seed::Int;
                            n_restarts::Int=SPECIALIST_RESTARTS)
    best_ce = Inf
    for r in 1:n_restarts
        ms = train_model(train_tokens, cfg, base_seed + 1000 * (r - 1);
                         n_steps=N_TRAIN_STEPS, verbose=false)
        ce = mean_cross_entropy(ms.W, eval_tokens, cfg)
        best_ce = min(best_ce, ce)
    end
    best_ce
end

"""
    inter_model_js(models, prefixes, cfg) -> (mean_js, all_js_values)

Compute mean pairwise JS divergence between models' averaged output
distributions on the given prefixes. Returns mean and all pairwise values.
"""
function inter_model_js(models::Vector{ModelState},
                        prefixes::Dict{Int, Vector{Vector{Int}}},
                        cfg::ModelConfig)
    n = length(models)
    # Pre-compute p_bar for each model at each prefix length
    model_pbars = Vector{Dict{Int, Vector{Float64}}}(undef, n)
    Threads.@threads for i in 1:n
        ms = models[i]
        pbars = Dict{Int, Vector{Float64}}()
        for (r, seqs) in prefixes
            if !isempty(seqs)
                pbars[r] = mean_output_distribution(ms.W, seqs, cfg)
            end
        end
        model_pbars[i] = pbars
    end

    # Pairwise JS between all model pairs
    js_values = Float64[]
    for i in 1:n, j in (i + 1):n
        total_js = 0.0
        n_lengths = 0
        for r in sort(collect(keys(model_pbars[i])))
            haskey(model_pbars[j], r) || continue
            total_js += js_div(model_pbars[i][r], model_pbars[j][r])
            n_lengths += 1
        end
        d = n_lengths > 0 ? total_js / n_lengths : 0.0
        push!(js_values, d)
    end

    mean_js = isempty(js_values) ? 0.0 : sum(js_values) / length(js_values)
    (mean_js, js_values)
end

function _quantile95(xs::Vector{Float64})
    isempty(xs) && return NaN
    ys = sort(xs)
    idx = ceil(Int, 0.95 * length(ys))
    ys[min(idx, length(ys))]
end

function calibrate_fit_tolerance(models::Vector{ModelState},
                                 eval_tokens_by_lang::Dict{String, Vector{Vector{Int}}},
                                 languages::Vector{String},
                                 cfg::ModelConfig;
                                 n_splits::Int=12,
                                 rng::AbstractRNG=Random.default_rng())
    diffs = Float64[]
    for _ in 1:n_splits
        split_a = Dict{String, Vector{Vector{Int}}}()
        split_b = Dict{String, Vector{Vector{Int}}}()
        for lang in languages
            toks = eval_tokens_by_lang[lang]
            n = length(toks)
            n < 2 && continue
            perm = randperm(rng, n)
            mid = max(1, n ÷ 2)
            split_a[lang] = toks[perm[1:mid]]
            split_b[lang] = toks[perm[(mid + 1):end]]
        end
        isempty(split_a) && continue
        for ms in models
            ce_a = mean_cross_entropy_balanced(ms.W, split_a, cfg)
            ce_b = mean_cross_entropy_balanced(ms.W, split_b, cfg)
            push!(diffs, abs(ce_a - ce_b))
        end
    end
    _quantile95(diffs)
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    t_start = time()
    println("=" ^ 60)
    println("Experiment 4: Off-Ecology Error")
    println("=" ^ 60)
    println()

    # Check Alice data files
    for lang in ALL_ALICE_LANGUAGES
        path = joinpath(ALICE_DIR, ALICE_FILE_MAP[lang])
        if !isfile(path)
            println("ERROR: Missing data file: " * path)
            return
        end
    end

    # Check Voynich
    voynich_path = joinpath(VOYNICH_DIR, "voynich_eva.txt")
    if !isfile(voynich_path)
        println("ERROR: Missing Voynich file: " * voynich_path)
        return
    end

    # Load Alice languages
    chunks_by_lang, tok = load_text_ecology(ALICE_DIR, ALICE_FILE_MAP;
                                             chunk_size=CHUNK_SIZE,
                                             overlap=CHUNK_OVERLAP)
    println("Tokenizer: " * string(tok.vocab_size) * " tokens (" *
            string(length(tok.chars)) * " chars + BOS)")
    for lang in ALL_ALICE_LANGUAGES
        println("  " * lang * ": " * string(length(chunks_by_lang[lang])) *
                " chunks")
    end

    # Load and chunk Voynich (using the same tokenizer — Voynich EVA chars
    # that happen to overlap with a-z will be tokenized; others dropped)
    voynich_text = load_text(voynich_path)
    voynich_chunks = chunk_text(voynich_text, CHUNK_SIZE; overlap=CHUNK_OVERLAP)
    println("  voynich: " * string(length(voynich_chunks)) * " chunks (" *
            string(length(voynich_text)) * " chars)")
    println()

    # Model config
    cfg = ModelConfig(; vocab_size=tok.vocab_size, n_layer=2, n_embd=32,
                       n_head=4, block_size=32)
    println("Model: " * string(n_params(init_model(cfg))) * " params")
    println("Training on: " * join(TRAIN_LANGUAGES, ", "))
    println("N_MODELS=" * string(N_MODELS))
    println()

    # Split chunks into train/test per language (80/20)
    rng_split = MersenneTwister(12345)
    train_chunks = Dict{String, Vector{String}}()
    test_chunks  = Dict{String, Vector{String}}()
    for lang in ALL_ALICE_LANGUAGES
        chunks = chunks_by_lang[lang]
        n = length(chunks)
        perm = randperm(rng_split, n)
        split_idx = round(Int, n * 0.8)
        train_chunks[lang] = chunks[perm[1:split_idx]]
        test_chunks[lang]  = chunks[perm[(split_idx + 1):end]]
    end

    # Voynich: use all chunks as test (model never trains on it)
    test_chunks["voynich"] = voynich_chunks

    # Pre-tokenize train and eval tokens
    train_toks = Vector{Int}[]
    for lang in TRAIN_LANGUAGES
        for chunk in train_chunks[lang]
            push!(train_toks, tokenize_chunk(tok, chunk))
        end
    end
    println(string(length(train_toks)) * " training sequences (T=3)")

    eval_tokens_by_lang = Dict{String, Vector{Vector{Int}}}()
    for lang in ALL_ALICE_LANGUAGES
        eval_tokens_by_lang[lang] = [tokenize_chunk(tok, c) for c in test_chunks[lang]]
    end
    eval_tokens_by_lang["voynich"] = [tokenize_chunk(tok, c)
                                      for c in voynich_chunks]

    # Build prefixes for all probe languages + Voynich
    prefixes_by_lang = Dict{String, Dict{Int, Vector{Vector{Int}}}}()
    for lang in ALL_ALICE_LANGUAGES
        prefixes_by_lang[lang] = make_prefixes(test_chunks[lang], tok,
                                                PREFIX_LENGTHS;
                                                max_per_length=PREFIX_SAMPLES,
                                                rng=MersenneTwister(99))
    end
    prefixes_by_lang["voynich"] = make_prefixes(voynich_chunks, tok,
                                                 PREFIX_LENGTHS;
                                                 max_per_length=PREFIX_SAMPLES,
                                                 rng=MersenneTwister(99))

    println()

    # ── Train per-language specialists (for excess loss, Prop 10.2) ──────────
    println("Training per-language specialists for excess loss decomposition...")
    specialist_seeds = Dict("english" => 9001, "french" => 9002,
                            "german" => 9003, "italian" => 9004,
                            "finnish" => 9005)
    specialist_ces = Dict{String, Float64}()
    for lang in ALL_ALICE_LANGUAGES
        t_spec = time()
        # Train specialist on its own language only
        lang_train = [tokenize_chunk(tok, c) for c in train_chunks[lang]]
        specialist_ces[lang] = best_specialist_ce(lang_train,
                                                  eval_tokens_by_lang[lang],
                                                  cfg,
                                                  specialist_seeds[lang])
        println("  " * lang * ": CE=" *
                string(round(specialist_ces[lang], digits=4)) *
                " (" * string(round(time() - t_spec, digits=1)) * "s)")
    end
    println()

    # ── Train models ──────────────────────────────────────────────────────────
    println("Training " * string(N_MODELS) * " models on " *
            join(TRAIN_LANGUAGES, ", ") * "...")
    models = ModelState[]
    for i in 1:N_MODELS
        t_model = time()
        ms = train_model(train_toks, cfg, i * 100; verbose=(i == 1))
        push!(models, ms)
        println("  model " * string(i) * "/" * string(N_MODELS) *
                " done (" * string(round(time() - t_model, digits=1)) * "s)")
    end
    println()

    # ── Per-model cross-entropy on each probe ─────────────────────────────────
    println("Evaluating cross-entropy on each probe category...")
    println()

    # Categories for reporting
    on_ecology   = TRAIN_LANGUAGES                      # en, fr, de
    off_related  = ["italian", "finnish"]               # same script, unseen
    off_alien    = ["voynich"]                           # EVA

    all_probes = vcat(ALL_ALICE_LANGUAGES, ["voynich"])

    # Per-model per-language CE
    model_ces = Vector{Dict{String, Float64}}(undef, N_MODELS)
    Threads.@threads for idx in 1:N_MODELS
        ms = models[idx]
        ces = Dict{String, Float64}()
        for lang in all_probes
            if haskey(eval_tokens_by_lang, lang) && !isempty(eval_tokens_by_lang[lang])
                ce = mean_cross_entropy(ms.W, eval_tokens_by_lang[lang], cfg)
                ces[lang] = ce
            end
        end
        model_ces[idx] = ces
    end

    # Print table
    println("Per-model CE by language:")
    header = "model | " * join([l[1:min(4, end)] for l in all_probes], " | ")
    println(header)
    println("-" ^ length(header))
    for idx in 1:N_MODELS
        row = lpad(string(idx), 5) * " | "
        for lang in all_probes
            ce = get(model_ces[idx], lang, NaN)
            row *= lpad(string(round(ce, digits=3)), 4) * " | "
        end
        println(row)
    end
    println()

    # Category means
    function category_mean_ce(category_langs)
        vals = Float64[]
        for idx in 1:N_MODELS
            cat_ce = Float64[]
            for lang in category_langs
                if haskey(model_ces[idx], lang)
                    push!(cat_ce, model_ces[idx][lang])
                end
            end
            if !isempty(cat_ce)
                push!(vals, sum(cat_ce) / length(cat_ce))
            end
        end
        isempty(vals) ? NaN : sum(vals) / length(vals)
    end

    ce_on    = category_mean_ce(on_ecology)
    ce_off_r = category_mean_ce(off_related)
    ce_off_a = category_mean_ce(off_alien)

    println("Category mean CE:")
    println("  on-ecology (" * join(on_ecology, ",") * "): " *
            string(round(ce_on, digits=4)))
    println("  off-ecology related (" * join(off_related, ",") * "): " *
            string(round(ce_off_r, digits=4)))
    println("  off-ecology alien (voynich): " *
            string(round(ce_off_a, digits=4)))
    println()

    # ── Inter-model agreement ─────────────────────────────────────────────────
    println("Computing inter-model agreement (pairwise JS on averaged outputs)...")
    println()

    # On-ecology: pool prefixes from training languages
    on_prefixes = Dict{Int, Vector{Vector{Int}}}()
    for lang in on_ecology
        for (r, seqs) in prefixes_by_lang[lang]
            if !haskey(on_prefixes, r)
                on_prefixes[r] = Vector{Int}[]
            end
            append!(on_prefixes[r], seqs)
        end
    end

    # Off-ecology (related): pool from Italian + Finnish
    off_r_prefixes = Dict{Int, Vector{Vector{Int}}}()
    for lang in off_related
        for (r, seqs) in prefixes_by_lang[lang]
            if !haskey(off_r_prefixes, r)
                off_r_prefixes[r] = Vector{Int}[]
            end
            append!(off_r_prefixes[r], seqs)
        end
    end

    # Off-ecology (alien): Voynich prefixes
    off_a_prefixes = prefixes_by_lang["voynich"]

    mean_js_on, js_vals_on = inter_model_js(models, on_prefixes, cfg)
    mean_js_off_r, js_vals_off_r = inter_model_js(models, off_r_prefixes, cfg)
    mean_js_off_a, js_vals_off_a = inter_model_js(models, off_a_prefixes, cfg)

    println("Inter-model JS divergence (lower = more agreement):")
    println("  on-ecology:           " * string(round(mean_js_on, digits=6)))
    println("  off-ecology (related): " * string(round(mean_js_off_r, digits=6)))
    println("  off-ecology (alien):   " * string(round(mean_js_off_a, digits=6)))
    println()

    # ── Excess loss decomposition (Prop 10.2) ─────────────────────────────────
    println("Excess loss (L_hat - H_irred) by category:")
    println()

    # H_irred per category: average specialist CE
    h_irred_on = sum(specialist_ces[l] for l in on_ecology) / length(on_ecology)
    h_irred_off_r = sum(specialist_ces[l] for l in off_related) / length(off_related)

    # Excess = multi-model CE - specialist CE (averaged across models)
    excess_on   = ce_on - h_irred_on
    excess_off_r = ce_off_r - h_irred_off_r

    println("  on-ecology:      H_irred=" * string(round(h_irred_on, digits=4)) *
            "  L_hat=" * string(round(ce_on, digits=4)) *
            "  excess=" * string(round(excess_on, digits=4)))
    println("  off-ecology rel: H_irred=" * string(round(h_irred_off_r, digits=4)) *
            "  L_hat=" * string(round(ce_off_r, digits=4)) *
            "  excess=" * string(round(excess_off_r, digits=4)))
    println("  off-ecology alien: L_hat=" * string(round(ce_off_a, digits=4)) *
            "  (no specialist available for Voynich)")
    println()

    # ── Non-identifiability proxy (Prop 10.3) ─────────────────────────────────
    # Calibrate a near-equal held-out fit tolerance from evaluation noise.
    # Then test whether model pairs within that tolerance still disagree
    # off-ecology.

    # Per-model on-ecology CE (mean across training languages)
    on_ces_per_model = Vector{Float64}(undef, N_MODELS)
    for idx in 1:N_MODELS
        vals = Float64[]
        for l in on_ecology
            haskey(model_ces[idx], l) && push!(vals, model_ces[idx][l])
        end
        on_ces_per_model[idx] = isempty(vals) ? NaN :
                                sum(vals) / length(vals)
    end

    # Build pairwise: on-ecology CE difference vs off-ecology JS
    pair_on_ce_diff  = Float64[]
    pair_on_js       = Float64[]
    pair_off_js_rel  = Float64[]
    pair_off_js_ali  = Float64[]
    js_on_idx = 0
    js_off_r_idx = 0
    js_off_a_idx = 0
    for i in 1:N_MODELS, j in (i + 1):N_MODELS
        push!(pair_on_ce_diff, abs(on_ces_per_model[i] - on_ces_per_model[j]))
        js_on_idx += 1
        js_off_r_idx += 1
        js_off_a_idx += 1
        push!(pair_on_js, js_vals_on[js_on_idx])
        push!(pair_off_js_rel, js_vals_off_r[js_off_r_idx])
        push!(pair_off_js_ali, js_vals_off_a[js_off_a_idx])
    end
    n_pairs_total = length(pair_on_ce_diff)

    eps_fit = calibrate_fit_tolerance(models, eval_tokens_by_lang, on_ecology, cfg;
                                      n_splits=12, rng=MersenneTwister(2026))
    close_pairs = [i for i in 1:n_pairs_total if pair_on_ce_diff[i] <= eps_fit]
    far_pairs   = [i for i in 1:n_pairs_total if pair_on_ce_diff[i] > eps_fit]

    close_on_js = isempty(close_pairs) ? NaN :
                  sum(pair_on_js[i] for i in close_pairs) / length(close_pairs)
    close_off_js_rel = isempty(close_pairs) ? NaN :
                       sum(pair_off_js_rel[i] for i in close_pairs) / length(close_pairs)
    close_off_js_ali = isempty(close_pairs) ? NaN :
                       sum(pair_off_js_ali[i] for i in close_pairs) / length(close_pairs)
    far_off_js_rel   = isempty(far_pairs) ? NaN :
                       sum(pair_off_js_rel[i] for i in far_pairs) / length(far_pairs)

    on_ce_mean = sum(on_ces_per_model) / N_MODELS
    on_ce_std = sqrt(sum((x - on_ce_mean)^2 for x in on_ces_per_model) / N_MODELS)

    println("Non-identifiability proxy (Prop 10.3):")
    println("  On-ecology CE across models: mean=" *
            string(round(on_ce_mean, digits=4)) *
            " std=" * string(round(on_ce_std, digits=4)))
    println("  Near-equal fit tolerance eps_fit = " *
            string(round(eps_fit, digits=6)) *
            " (95th percentile held-out CE split noise)")
    println("  Pairwise analysis (" * string(n_pairs_total) * " model pairs):")
    println("    close pairs (CE diff <= eps_fit): n=" *
            string(length(close_pairs)))
    println("      mean on-ecology JS          = " *
            string(round(close_on_js, digits=6)))
    println("      mean off-ecology JS (related) = " *
            string(round(close_off_js_rel, digits=6)))
    println("      mean off-ecology JS (alien)   = " *
            string(round(close_off_js_ali, digits=6)))
    println("    far pairs (CE diff > eps_fit):  n=" *
            string(length(far_pairs)))
    println("      mean off-ecology JS (related) = " *
            string(round(far_off_js_rel, digits=6)))
    println("  Non-identifiability: close pairs should still have high off-ecology JS")
    println("  Inter-model JS on-ecology:  " * string(round(mean_js_on, digits=6)))
    println("  Inter-model JS off-related: " * string(round(mean_js_off_r, digits=6)))
    println("  Inter-model JS off-alien:   " * string(round(mean_js_off_a, digits=6)))
    println()

    # ── Predictions check ─────────────────────────────────────────────────────
    println("=" ^ 60)
    println("PREDICTIONS CHECK")
    println("=" ^ 60)
    println()

    println("1. Prop 10.2: Off-ecology excess loss > on-ecology excess loss")
    println("   excess on=" * string(round(excess_on, digits=4)) *
            " excess off_related=" * string(round(excess_off_r, digits=4)))
    pass_excess = excess_off_r > excess_on
    println("   " * (pass_excess ? "PASS" : "CHECK"))
    println()

    println("2. Prop 10.3 proxy: Near-equal on-ecology fit with divergent off-ecology predictions")
    println("   Close-fit pairs (on-ecology CE diff <= " *
            string(round(eps_fit, digits=6)) * "):")
    println("     off-ecology JS (related) = " *
            string(round(close_off_js_rel, digits=6)))
    println("     on-ecology JS            = " *
            string(round(close_on_js, digits=6)))
    pass_nonident = !isnan(close_off_js_rel) && !isnan(close_on_js) &&
                    close_off_js_rel > close_on_js
    println("   close-pair off JS > close-pair on JS: " *
            (pass_nonident ? "PASS" : "CHECK"))
    println()

    println("3. Voynich = maximally off-ecology")
    pass_voynich = mean_js_off_a > mean_js_off_r
    println("   JS alien > JS related: " *
            (pass_voynich ? "YES (PASS)" : "NO (CHECK)"))
    println()

    # ── Save results ──────────────────────────────────────────────────────────
    mkpath(OUTPUT_DIR)
    outpath = joinpath(OUTPUT_DIR, "exp4_off_ecology_results.txt")
    open(outpath, "w") do io
        println(io, "# Experiment 4: Off-Ecology Error")
        println(io, "# N_MODELS=" * string(N_MODELS) *
                    " N_TRAIN_STEPS=" * string(N_TRAIN_STEPS) *
                    " LR=" * string(LR))
        println(io, "# TRAIN_LANGUAGES=" * join(TRAIN_LANGUAGES, ","))
        println(io, "# CHUNK_SIZE=" * string(CHUNK_SIZE) *
                    " CHUNK_OVERLAP=" * string(CHUNK_OVERLAP))
        println(io, "# PREFIX_LENGTHS=" * string(PREFIX_LENGTHS))
        println(io, "# block_size=" * string(cfg.block_size) *
                    " n_embd=" * string(cfg.n_embd) *
                    " n_layer=" * string(cfg.n_layer))
        println(io, "")

        # Per-model per-language CE
        println(io, "model," * join(all_probes, ","))
        for idx in 1:N_MODELS
            row = string(idx)
            for lang in all_probes
                ce = get(model_ces[idx], lang, NaN)
                row *= "," * string(round(ce, digits=6))
            end
            println(io, row)
        end
        println(io, "")

        # Category summaries
        println(io, "# Category CE: on=" * string(round(ce_on, digits=6)) *
                    " off_related=" * string(round(ce_off_r, digits=6)) *
                    " off_alien=" * string(round(ce_off_a, digits=6)))
        println(io, "# H_irred: on=" * string(round(h_irred_on, digits=6)) *
                    " off_related=" * string(round(h_irred_off_r, digits=6)))
        println(io, "# Excess loss: on=" * string(round(excess_on, digits=6)) *
                    " off_related=" * string(round(excess_off_r, digits=6)))
        println(io, "# Non-identifiability proxy:")
        println(io, "#   on-ecology CE std=" * string(round(on_ce_std, digits=6)))
        println(io, "#   near-equal fit tolerance eps_fit=" *
                    string(round(eps_fit, digits=6)))
        println(io, "#   close-pair on-ecology JS=" *
                    string(round(close_on_js, digits=8)))
        println(io, "#   close-pair off-ecology JS (related)=" *
                    string(round(close_off_js_rel, digits=8)))
        println(io, "#   close-pair off-ecology JS (alien)=" *
                    string(round(close_off_js_ali, digits=8)))
        println(io, "# Inter-model JS: on=" * string(round(mean_js_on, digits=8)) *
                    " off_related=" * string(round(mean_js_off_r, digits=8)) *
                    " off_alien=" * string(round(mean_js_off_a, digits=8)))
        println(io, "# Specialist CEs:")
        for lang in ALL_ALICE_LANGUAGES
            println(io, "#   " * lang * "=" *
                        string(round(specialist_ces[lang], digits=6)))
        end
        println(io, "")

        # Pairwise JS values
        println(io, "# Pairwise inter-model JS (on-ecology)")
        println(io, join([string(round(v, digits=8)) for v in js_vals_on], ","))
        println(io, "# Pairwise inter-model JS (off-ecology related)")
        println(io, join([string(round(v, digits=8)) for v in js_vals_off_r], ","))
        println(io, "# Pairwise inter-model JS (off-ecology alien)")
        println(io, join([string(round(v, digits=8)) for v in js_vals_off_a], ","))
    end
    println("Results saved to " * outpath)
    println("Total time: " * string(round(time() - t_start, digits=1)) * "s")
end

main()
