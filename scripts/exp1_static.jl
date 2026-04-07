"""
Experiment 1: Static Optimality on Held-Out Corpora

Tests the static theory in two layers:

  1. Exact theorem layer:
     Build a finite ecology directly from held-out corpora, with contexts given
     by prefix lengths and world states given by languages. For the partition
     induced by a trained model, evaluate the exact finite-ecology quantities
     from the exact excess-loss decomposition theorem:
         excess_loss = JS_excess

  2. Neural approximation layer:
     Compare the model's actual context-level loss on those same held-out
     prefix contexts to the Bayes-optimal decoder loss for the induced
     partition. This isolates the optimisation/approximation gap from the exact
     theorem itself.

Run: julia --project=. scripts/exp1_static.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "microgpt.jl"))
include(joinpath(@__DIR__, "..", "src", "data.jl"))
include(joinpath(@__DIR__, "..", "src", "measurement.jl"))

using Random, LinearAlgebra
using LlmPaper

BLAS.set_num_threads(1)

# ── Configuration ─────────────────────────────────────────────────────────────

const OUTPUT_DIR = joinpath(@__DIR__, "..", "output")
const N_SEEDS = 5
const N_TRAIN_STEPS = 15000
const LR = 0.008
const PREFIX_LENGTHS = [3, 5, 8]
const CHUNK_SIZE = 30
const CHUNK_OVERLAP = 5
const N_LAYER = 2
const N_EMBD = 32
const N_HEAD = 4
const BLOCK_SIZE = 32

const ALL_LANGUAGES = ["english", "french", "german", "italian", "finnish"]

const CONDITIONS = [
    (1, ["english"]),
    (2, ["english", "french"]),
    (3, ["english", "french", "german"]),
    (5, ALL_LANGUAGES),
]

const CORPORA = [
    (name="alice",
     data_dir=joinpath(@__DIR__, "..", "data", "alice"),
     file_map=Dict(
         "english" => "alice_english.txt",
         "french"  => "alice_french.txt",
         "german"  => "alice_german.txt",
         "italian" => "alice_italian.txt",
         "finnish" => "alice_finnish.txt",
     )),
    (name="manifesto",
     data_dir=joinpath(@__DIR__, "..", "data", "manifesto"),
     file_map=Dict(
         "english" => "manifesto_english.txt",
         "french"  => "manifesto_french.txt",
         "german"  => "manifesto_german.txt",
         "italian" => "manifesto_italian.txt",
         "finnish" => "manifesto_finnish.txt",
     )),
]

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

function build_all_prefixes(chunks::Vector{String}, tok::Tokenizer,
                            prefix_lengths::Vector{Int})
    make_prefixes(chunks, tok, prefix_lengths; max_per_length=typemax(Int),
                  rng=MersenneTwister(99))
end

function model_context_loss(W::Dict{String,Matrix{Float64}},
                            prefixes_by_lang::Dict{String, Dict{Int, Vector{Vector{Int}}}},
                            languages::Vector{String},
                            fe::FiniteEcology,
                            valid_lengths::Vector{Int},
                            cfg::ModelConfig)
    p_bars = compute_p_bars(W, prefixes_by_lang, languages, cfg)
    total = 0.0
    for (wi, lang) in enumerate(languages), (ci, r) in enumerate(valid_lengths)
        haskey(p_bars, lang) || error("missing model cache for " * lang)
        haskey(p_bars[lang], r) || error("missing prefix length " * string(r) *
                                         " for " * lang)
        total += fe.pi[wi] * fe.d_c[ci] *
                 cross_entropy(view(fe.p_wcv, wi, ci, :), p_bars[lang][r])
    end
    total
end

function run_corpus(spec)
    name = spec.name
    data_dir = spec.data_dir
    file_map = spec.file_map

    println("=" ^ 70)
    println("Experiment 1: " * uppercasefirst(name))
    println("=" ^ 70)
    println()

    chunks_by_lang, tok = load_text_ecology(data_dir, file_map;
                                            chunk_size=CHUNK_SIZE,
                                            overlap=CHUNK_OVERLAP)
    cfg = ModelConfig(; vocab_size=tok.vocab_size, n_layer=N_LAYER,
                       n_embd=N_EMBD, n_head=N_HEAD, block_size=BLOCK_SIZE)
    println("Tokenizer: " * string(tok.vocab_size) * " tokens (" *
            string(length(tok.chars)) * " chars + BOS)")
    for lang in ALL_LANGUAGES
        println("  " * lang * ": " * string(length(chunks_by_lang[lang])) *
                " chunks")
    end
    println()
    println("Model: " * string(n_params(init_model(cfg))) * " params")
    println()

    rng_split = MersenneTwister(12345)
    train_chunks = Dict{String, Vector{String}}()
    test_chunks = Dict{String, Vector{String}}()
    for lang in ALL_LANGUAGES
        chunks = chunks_by_lang[lang]
        n = length(chunks)
        perm = randperm(rng_split, n)
        split_idx = round(Int, n * 0.8)
        train_chunks[lang] = chunks[perm[1:split_idx]]
        test_chunks[lang] = chunks[perm[(split_idx + 1):end]]
        println("  " * lang * ": " * string(length(train_chunks[lang])) *
                " train, " * string(length(test_chunks[lang])) * " test chunks")
    end
    println()

    train_tokens_by_lang = Dict{String, Vector{Vector{Int}}}()
    eval_tokens_by_lang = Dict{String, Vector{Vector{Int}}}()
    prefixes_by_lang = Dict{String, Dict{Int, Vector{Vector{Int}}}}()
    for lang in ALL_LANGUAGES
        train_tokens_by_lang[lang] = [tokenize_chunk(tok, c) for c in train_chunks[lang]]
        eval_tokens_by_lang[lang] = [tokenize_chunk(tok, c) for c in test_chunks[lang]]
        prefixes_by_lang[lang] = build_all_prefixes(test_chunks[lang], tok, PREFIX_LENGTHS)
    end

    println("Calibrating threshold from T=1 baseline...")
    ms_calib = train_model(train_tokens_by_lang["english"], cfg, 9999;
                           n_steps=N_TRAIN_STEPS)
    tau = calibrate_threshold(ms_calib.W, test_chunks["english"], tok, cfg;
                              n_splits=30, prefix_lengths=PREFIX_LENGTHS,
                              rng=MersenneTwister(777))
    println("  tau = " * string(round(tau, digits=6)))
    println()

    Exp1Result = NamedTuple{(:corpus, :T, :seed, :exact_ecology_k, :s, :k, :seq_ce_balanced,
                             :model_context_loss, :exact_bayes_loss,
                             :exact_entropy_floor, :exact_js_excess,
                             :exact_excess_gap, :exact_excess_ratio,
                             :approx_gap, :n_classes),
                            Tuple{String, Int, Int, Int, Int, Int, Float64,
                                  Float64, Float64, Float64, Float64,
                                  Float64, Float64, Float64, Int}}
    results = Exp1Result[]
    d_full_seed1 = Dict{Int, Matrix{Float64}}()

    for (T, languages) in CONDITIONS
        t_cond = time()
        println("--- corpus=" * name * " T=" * string(T) * " languages: " *
                join(languages, ", ") * " ---")

        train_toks = Vector{Int}[]
        for lang in languages
            append!(train_toks, train_tokens_by_lang[lang])
        end
        cond_eval_tokens = Dict(l => eval_tokens_by_lang[l] for l in languages)
        cond_prefixes = Dict(l => prefixes_by_lang[l] for l in languages)
        fe, valid_lengths, _ = empirical_prefix_ecology(cond_eval_tokens, languages,
                                                        PREFIX_LENGTHS, cfg.vocab_size;
                                                        uniform_worlds=true,
                                                        context_weighting=:uniform)
        exact_ecology_k = n_classes(ecology_partition(fe))

        println("  " * string(length(train_toks)) * " training sequences")
        println("  exact contexts: prefix lengths " * string(valid_lengths))
        println("  exact ecology k=" * string(exact_ecology_k))

        for seed in 1:N_SEEDS
            t_seed = time()
            ms = train_model(train_toks, cfg, seed * 100 + T;
                             verbose=(seed == 1))

            D_train = distance_matrix(ms.W, cond_prefixes, languages, cfg)
            s, k = separation_summary(D_train, tau)
            labels = partition_labels_from_distance(D_train, tau)

            if seed == 1
                d_full_seed1[T] = distance_matrix(ms.W, prefixes_by_lang,
                                                  ALL_LANGUAGES, cfg)
            end

            seq_ce_bal = mean_cross_entropy_balanced(ms.W, cond_eval_tokens, cfg)
            exact_floor = conditional_entropy_VCW(fe)
            exact_bayes = bayes_opt_loss(fe, labels)
            exact_js = js_excess(fe, labels)
            exact_gap = exact_bayes - exact_floor
            exact_ratio = exact_gap > 1e-12 ? exact_js / exact_gap : NaN
            ctx_loss = model_context_loss(ms.W, cond_prefixes, languages,
                                          fe, valid_lengths, cfg)
            approx_gap = ctx_loss - exact_bayes

            push!(results, (corpus=name, T=T, seed=seed,
                            exact_ecology_k=exact_ecology_k, s=s, k=k,
                            seq_ce_balanced=seq_ce_bal,
                            model_context_loss=ctx_loss,
                            exact_bayes_loss=exact_bayes,
                            exact_entropy_floor=exact_floor,
                            exact_js_excess=exact_js,
                            exact_excess_gap=exact_gap,
                            exact_excess_ratio=exact_ratio,
                            approx_gap=approx_gap,
                            n_classes=n_classes(labels)))

            if seed == 1
                println("  seed 1 exact: bayes=" *
                        string(round(exact_bayes, digits=6)) *
                        " floor=" * string(round(exact_floor, digits=6)) *
                        " js=" * string(round(exact_js, digits=6)) *
                        " ratio=" * string(round(exact_ratio, digits=6)))
                println("  seed 1 model: seq_CE=" *
                        string(round(seq_ce_bal, digits=4)) *
                        " ctx_loss=" * string(round(ctx_loss, digits=6)) *
                        " approx_gap=" * string(round(approx_gap, digits=6)) *
                        " k=" * string(k))
            end
            println("  seed " * string(seed) * " done (" *
                    string(round(time() - t_seed, digits=1)) * "s)")
        end

        subset = [r for r in results if r.corpus == name && r.T == T]
        k_mean = sum(r.k for r in subset) / length(subset)
        eco_k = subset[1].exact_ecology_k
        seq_ce_mean = sum(r.seq_ce_balanced for r in subset) / length(subset)
        ratios = [r.exact_excess_ratio for r in subset if !isnan(r.exact_excess_ratio)]
        approx_mean = sum(r.approx_gap for r in subset) / length(subset)
        println("  exact ecology k=" * string(eco_k))
        println("  k mean=" * string(round(k_mean, digits=2)))
        println("  seq_CE mean=" * string(round(seq_ce_mean, digits=4)))
        if !isempty(ratios)
            println("  exact excess ratio mean=" *
                    string(round(sum(ratios) / length(ratios), digits=6)))
        end
        println("  model approximation gap mean=" *
                string(round(approx_mean, digits=6)))
        println("  condition time: " *
                string(round(time() - t_cond, digits=1)) * "s")
        println()
    end

    (results=results, tau=tau, d_full_seed1=d_full_seed1)
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    t_start = time()
    println("=" ^ 70)
    println("Experiment 1: Static Optimality on Held-Out Corpora")
    println("=" ^ 70)
    println()

    all_results = NamedTuple[]
    taus = Dict{String, Float64}()
    d_full_seed1 = Dict{Tuple{String, Int}, Matrix{Float64}}()

    for spec in CORPORA
        corpus_results = run_corpus(spec)
        append!(all_results, corpus_results.results)
        taus[spec.name] = corpus_results.tau
        for (T, D) in corpus_results.d_full_seed1
            d_full_seed1[(spec.name, T)] = D
        end
    end

    println("=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println()
    println("corpus,T,exact_ecology_k,k_mean,seq_CE_mean,exact_ratio_mean,approx_gap_mean")
    for spec in CORPORA, (T, _) in CONDITIONS
        subset = [r for r in all_results if r.corpus == spec.name && r.T == T]
        eco_k = subset[1].exact_ecology_k
        k_mean = sum(r.k for r in subset) / length(subset)
        ce_mean = sum(r.seq_ce_balanced for r in subset) / length(subset)
        ratios = [r.exact_excess_ratio for r in subset if !isnan(r.exact_excess_ratio)]
        ratio_mean = isempty(ratios) ? NaN : sum(ratios) / length(ratios)
        approx_mean = sum(r.approx_gap for r in subset) / length(subset)
        println(spec.name * "," * string(T) * "," *
                string(eco_k) * "," *
                string(round(k_mean, digits=2)) * "," *
                string(round(ce_mean, digits=6)) * "," *
                string(round(ratio_mean, digits=6)) * "," *
                string(round(approx_mean, digits=6)))
    end
    println()

    println("Exact theorem check: excess_ratio should be ~1 whenever exact_excess_gap > 0")
    for spec in CORPORA
        ratios = [r.exact_excess_ratio for r in all_results
                  if r.corpus == spec.name && !isnan(r.exact_excess_ratio)]
        mean_ratio = isempty(ratios) ? NaN : sum(ratios) / length(ratios)
        println("  " * spec.name * ": mean exact_ratio=" *
                string(round(mean_ratio, digits=6)))
    end
    println()

    mkpath(OUTPUT_DIR)
    outpath = joinpath(OUTPUT_DIR, "exp1_static_results.txt")
    open(outpath, "w") do io
        println(io, "# Experiment 1: Static Optimality on Held-Out Corpora")
        println(io, "# N_SEEDS=" * string(N_SEEDS) *
                    " N_TRAIN_STEPS=" * string(N_TRAIN_STEPS) *
                    " LR=" * string(LR))
        println(io, "# CHUNK_SIZE=" * string(CHUNK_SIZE) *
                    " CHUNK_OVERLAP=" * string(CHUNK_OVERLAP))
        println(io, "# PREFIX_LENGTHS=" * string(PREFIX_LENGTHS))
        println(io, "# block_size=" * string(BLOCK_SIZE) *
                    " n_embd=" * string(N_EMBD) *
                    " n_layer=" * string(N_LAYER) *
                    " n_head=" * string(N_HEAD))
        for spec in CORPORA
            println(io, "# tau[" * spec.name * "]=" * string(taus[spec.name]))
        end
        println(io, "")
        println(io, "corpus,T,seed,exact_ecology_k,k,s,seq_ce_balanced,model_context_loss," *
                    "exact_bayes_loss,exact_entropy_floor,exact_js_excess," *
                    "exact_excess_gap,exact_excess_ratio,approx_gap")
        for r in all_results
            println(io, r.corpus * "," * string(r.T) * "," * string(r.seed) * "," *
                        string(r.exact_ecology_k) * "," *
                        string(r.k) * "," * string(r.s) * "," *
                        string(round(r.seq_ce_balanced, digits=6)) * "," *
                        string(round(r.model_context_loss, digits=6)) * "," *
                        string(round(r.exact_bayes_loss, digits=6)) * "," *
                        string(round(r.exact_entropy_floor, digits=6)) * "," *
                        string(round(r.exact_js_excess, digits=8)) * "," *
                        string(round(r.exact_excess_gap, digits=8)) * "," *
                        string(round(r.exact_excess_ratio, digits=6)) * "," *
                        string(round(r.approx_gap, digits=6)))
        end
        println(io, "")
        println(io, "# Full distance matrices (seed 1 only)")
        for spec in CORPORA, (T, _) in CONDITIONS
            key = (spec.name, T)
            haskey(d_full_seed1, key) || continue
            println(io, "# corpus=" * spec.name * " T=" * string(T) * " D_full:")
            println(io, "# " * join(ALL_LANGUAGES, ","))
            D = d_full_seed1[key]
            for i in 1:length(ALL_LANGUAGES)
                println(io, join([string(round(D[i, j], digits=6))
                                  for j in 1:length(ALL_LANGUAGES)], ","))
            end
            println(io, "")
        end
    end

    println("Results saved to " * outpath)
    println("Total time: " * string(round(time() - t_start, digits=1)) * "s")
end

main()
