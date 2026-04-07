"""
Experiment 3: Population Dynamics and Price Equation

Validates: the proposition `when the ecological-veridicality dynamic theorems
apply to idealized model populations`, plus the expected selection-stage Price
term:
E[Delta_R_bar_sel | risk, fitness] = Cov(fitness, risk) / w_bar.

Data: Alice corpus, 5 languages.
Design: K=20 models, G=50 generations, N_RUNS=3 independent runs.
Each generation also computes the exact conditional Wright-Fisher selection
expectation and its exact sampling variance.

Each generation uses run_generation! from population.jl which handles:
  1. Evaluate pre-selection risk and fitness (balanced CE across languages)
  2. Wright-Fisher selection
  3. Development (20 steps training from fresh optimizer)
  4. Mutation (Gaussian noise, reset optimizer)
  5. Records stagewise means, Price equation terms, separation summaries

Run:  julia --project=. scripts/exp3_population.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "microgpt.jl"))
include(joinpath(@__DIR__, "..", "src", "data.jl"))
include(joinpath(@__DIR__, "..", "src", "measurement.jl"))
include(joinpath(@__DIR__, "..", "src", "population.jl"))

using Random, LinearAlgebra

BLAS.set_num_threads(1)

# ── Configuration ─────────────────────────────────────────────────────────────

const DATA_DIR        = joinpath(@__DIR__, "..", "data", "alice")
const OUTPUT_DIR      = joinpath(@__DIR__, "..", "output")
const K               = 20          # population size
const G               = 50          # generations
const N_DEV_STEPS     = 20          # development steps per generation
const LR_DEV          = 0.005       # learning rate during development
const SIGMA_MUT       = 0.001       # mutation noise std
const N_RUNS          = 3           # independent runs
const N_INIT_STEPS    = 100         # pre-training steps for initialization
const LR_INIT         = 0.008       # learning rate for initialization
const PREFIX_LENGTHS  = [3, 5, 8]
const CHUNK_SIZE      = 30
const CHUNK_OVERLAP   = 5
const PREFIX_SAMPLES  = 160

const ALL_LANGUAGES = ["english", "french", "german", "italian", "finnish"]

const FILE_MAP = Dict(
    "english" => "alice_english.txt",
    "french"  => "alice_french.txt",
    "german"  => "alice_german.txt",
    "italian" => "alice_italian.txt",
    "finnish" => "alice_finnish.txt",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

"""
    initialize_population(cfg, train_chunks, tok, run_seed; K, verbose)

Create a diverse initial population. Each model is initialized with a unique
seed, then trained for N_INIT_STEPS on a single randomly chosen language.
This produces diverse, non-veridical starting models.
"""
function initialize_population(cfg::ModelConfig,
                               train_chunks::Dict{String, Vector{String}},
                               tok::Tokenizer,
                               run_seed::Int;
                               K::Int=K,
                               verbose::Bool=false)
    rng = MersenneTwister(run_seed)
    pop = ModelState[]
    for i in 1:K
        # 1. Init model with unique seed
        model_seed = run_seed * 1000 + i
        ms = init_model(cfg; seed=model_seed)

        # 2. Pick a random language for this model
        lang = ALL_LANGUAGES[rand(rng, 1:length(ALL_LANGUAGES))]

        # 3. Pre-tokenize that language's training chunks
        lang_tokens = [tokenize_chunk(tok, c) for c in train_chunks[lang]]

        # 4. Train for N_INIT_STEPS on that language
        train_rng = MersenneTwister(model_seed + 50000)
        for step in 1:N_INIT_STEPS
            idx = rand(train_rng, 1:length(lang_tokens))
            tokens = lang_tokens[idx]
            length(tokens) < 2 && continue
            lr_t = LR_INIT * max(0.0, 1.0 - step / N_INIT_STEPS)
            train_step_lr!(ms, tokens, step, cfg, lr_t)
        end

        push!(pop, ms)
        if verbose && (i == 1 || i == K)
            println("  model " * string(i) * ": seed=" * string(model_seed) *
                    " init_lang=" * lang)
        end
    end
    pop
end

"""
    fraction_full_separation(pop, prefixes_by_lang, languages, cfg, tau) -> Float64

Compute the fraction of models in the population that distinguish all
languages (k == length(languages)).
"""
function fraction_full_separation(pop::Vector{ModelState},
                                  prefixes_by_lang::Dict{String, Dict{Int, Vector{Vector{Int}}}},
                                  languages::Vector{String},
                                  cfg::ModelConfig,
                                  tau::Float64)
    n_lang = length(languages)
    hits = Vector{Int}(undef, length(pop))
    Threads.@threads for i in eachindex(pop)
        D = distance_matrix(pop[i].W, prefixes_by_lang, languages, cfg)
        _, k = separation_summary(D, tau)
        hits[i] = (k == n_lang) ? 1 : 0
    end
    Float64(sum(hits)) / length(pop)
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    t_start = time()
    println("=" ^ 60)
    println("Experiment 3: Population Dynamics and Price Equation")
    println("=" ^ 60)
    println()

    # Check data files
    for lang in ALL_LANGUAGES
        path = joinpath(DATA_DIR, FILE_MAP[lang])
        if !isfile(path)
            println("ERROR: Missing data file: " * path)
            return
        end
    end

    # Load all languages via continuous-text ecology
    chunks_by_lang, tok = load_text_ecology(DATA_DIR, FILE_MAP;
                                             chunk_size=CHUNK_SIZE,
                                             overlap=CHUNK_OVERLAP)
    println("Tokenizer: " * string(tok.vocab_size) * " tokens (" *
            string(length(tok.chars)) * " chars + BOS)")
    for lang in ALL_LANGUAGES
        println("  " * lang * ": " * string(length(chunks_by_lang[lang])) *
                " chunks")
    end
    println()

    # Model config
    cfg = ModelConfig(; vocab_size=tok.vocab_size, n_layer=2, n_embd=32,
                       n_head=4, block_size=32)
    println("Model: " * string(n_params(init_model(cfg))) * " params")
    println("Population: K=" * string(K) * " G=" * string(G) *
            " N_RUNS=" * string(N_RUNS))
    println("Development: n_dev_steps=" * string(N_DEV_STEPS) *
            " lr_dev=" * string(LR_DEV))
    println("Mutation: sigma_mut=" * string(SIGMA_MUT))
    println("Selection diagnostics: exact conditional mean and sd")
    println()

    # Split chunks into train/test per language (80/20)
    rng_split = MersenneTwister(12345)
    train_chunks = Dict{String, Vector{String}}()
    test_chunks  = Dict{String, Vector{String}}()
    for lang in ALL_LANGUAGES
        chunks = chunks_by_lang[lang]
        n = length(chunks)
        perm = randperm(rng_split, n)
        split_idx = round(Int, n * 0.8)
        train_chunks[lang] = chunks[perm[1:split_idx]]
        test_chunks[lang]  = chunks[perm[(split_idx + 1):end]]
        println("  " * lang * ": " * string(length(train_chunks[lang])) *
                " train, " * string(length(test_chunks[lang])) * " test chunks")
    end
    println()

    # Pre-tokenize evaluation tokens
    eval_tokens_by_lang = Dict{String, Vector{Vector{Int}}}()
    for lang in ALL_LANGUAGES
        eval_tokens_by_lang[lang] = [tokenize_chunk(tok, c) for c in test_chunks[lang]]
    end

    # Build per-language training tokens (for balanced development)
    train_tokens_by_lang = Dict{String, Vector{Vector{Int}}}()
    total_train = 0
    for lang in ALL_LANGUAGES
        train_tokens_by_lang[lang] = [tokenize_chunk(tok, c)
                                       for c in train_chunks[lang]]
        total_train += length(train_tokens_by_lang[lang])
    end
    println("Total training sequences: " * string(total_train) *
            " (balanced across " * string(length(ALL_LANGUAGES)) * " languages)")

    # Build test prefixes for evaluation
    prefixes_by_lang = Dict{String, Dict{Int, Vector{Vector{Int}}}}()
    for lang in ALL_LANGUAGES
        prefixes_by_lang[lang] = make_prefixes(test_chunks[lang], tok,
                                                PREFIX_LENGTHS;
                                                max_per_length=PREFIX_SAMPLES,
                                                rng=MersenneTwister(99))
    end

    # Calibrate threshold from T=1 baseline
    println("Calibrating threshold from T=1 baseline...")
    train_toks_en = [tokenize_chunk(tok, c) for c in train_chunks["english"]]
    ms_calib = init_model(cfg; seed=9999)
    calib_rng = MersenneTwister(10009)
    for step in 1:N_INIT_STEPS
        idx = rand(calib_rng, 1:length(train_toks_en))
        tokens = train_toks_en[idx]
        length(tokens) < 2 && continue
        lr_t = LR_INIT * max(0.0, 1.0 - step / N_INIT_STEPS)
        train_step_lr!(ms_calib, tokens, step, cfg, lr_t)
    end
    tau = calibrate_threshold(ms_calib.W, test_chunks["english"], tok, cfg;
                               n_splits=30, prefix_lengths=PREFIX_LENGTHS,
                               rng=MersenneTwister(777))
    println("  tau = " * string(round(tau, digits=6)))
    println()

    # ── Run experiments ────────────────────────────────────────────────────────
    # Collect all per-generation records across runs
    Exp3Row = NamedTuple{(:run, :gen, :rec), Tuple{Int, Int, GenerationRecord}}
    all_records = Exp3Row[]

    for run in 1:N_RUNS
        t_run = time()
        run_seed = run * 42
        println("=" ^ 50)
        println("Run " * string(run) * "/" * string(N_RUNS) *
                " (seed=" * string(run_seed) * ")")
        println("=" ^ 50)

        # Initialize population
        println("Initializing population (K=" * string(K) *
                ", " * string(N_INIT_STEPS) * " steps each)...")
        pop = initialize_population(cfg, train_chunks, tok, run_seed;
                                    K=K, verbose=true)

        # Initial evaluation
        risks_init, fitnesses_init = evaluate_population(pop, eval_tokens_by_lang, cfg)
        r_bar_init = sum(risks_init) / length(risks_init)
        w_bar_init = sum(fitnesses_init) / length(fitnesses_init)
        println("  Initial: R_bar=" * string(round(r_bar_init, digits=4)) *
                " w_bar=" * string(round(w_bar_init, digits=6)))
        println()

        # Run generations
        gen_rng = MersenneTwister(run_seed + 7777)
        for gen in 1:G
            t_gen = time()

            rec = run_generation!(pop, train_tokens_by_lang, eval_tokens_by_lang,
                                  prefixes_by_lang, ALL_LANGUAGES, cfg,
                                  gen, tau;
                                  K=K, n_dev_steps=N_DEV_STEPS,
                                  lr_dev=LR_DEV, sigma_mut=SIGMA_MUT,
                                  rng=gen_rng,
                                  verbose=false)

            push!(all_records, (run=run, gen=gen, rec=rec))

            # One-line summary per generation
            r_bar = sum(rec.risk_before) / length(rec.risk_before)
            w_bar = sum(rec.fitness) / length(rec.fitness)
            elapsed_gen = round(time() - t_gen, digits=1)
            println("  gen " * string(gen) * "/" * string(G) *
                    "  R_bar=" * string(round(r_bar, digits=4)) *
                    "  w_bar=" * string(round(w_bar, digits=6)) *
                    "  delta_sel=" * string(round(rec.delta_sel, digits=6)) *
                    "  price=" * string(round(rec.price_predicted, digits=6)) *
                    "  k=" * string(round(rec.mean_k, digits=1)) *
                    "  (" * string(elapsed_gen) * "s)")
        end

        # End-of-run separation check
        frac_full = fraction_full_separation(pop, prefixes_by_lang,
                                              ALL_LANGUAGES, cfg, tau)
        println()
        println("  Run " * string(run) * " complete: " *
                string(round(time() - t_run, digits=1)) * "s")
        println("  Final fraction with k=5: " *
                string(round(frac_full, digits=2)))
        println()
    end

    # ── Analysis ──────────────────────────────────────────────────────────────
    println("=" ^ 60)
    println("ANALYSIS")
    println("=" ^ 60)
    println()

    # 1. Selection-stage expectation validation under Wright-Fisher sampling
    println("Selection-stage expectation check (realized vs conditional expectation):")
    println("  gen | delta_sel_mean | expected_mean | exact 95% band")
    println("  ----|----------------|---------------|------------------")
    for gen in 1:G
        gen_recs = [r.rec for r in all_records if r.gen == gen]
        ds_vals = [rec.delta_sel for rec in gen_recs]
        pp_vals = [rec.price_predicted for rec in gen_recs]
        lo_vals = [rec.delta_sel_lo95 for rec in gen_recs]
        hi_vals = [rec.delta_sel_hi95 for rec in gen_recs]
        ds_mean = sum(ds_vals) / length(ds_vals)
        pp_mean = sum(pp_vals) / length(pp_vals)
        lo_mean = sum(lo_vals) / length(lo_vals)
        hi_mean = sum(hi_vals) / length(hi_vals)
        if gen <= 10 || gen % 10 == 0 || gen == G
            println("  " * lpad(string(gen), 3) * " | " *
                    lpad(string(round(ds_mean, digits=6)), 14) * " | " *
                    lpad(string(round(pp_mean, digits=6)), 13) * " | [" *
                    string(round(lo_mean, digits=6)) * ", " *
                    string(round(hi_mean, digits=6)) * "]")
        end
    end
    println()

    all_ds = [r.rec.delta_sel for r in all_records]
    all_pp = [r.rec.price_predicted for r in all_records]
    all_sd = [r.rec.delta_sel_sd for r in all_records]
    all_lo = [r.rec.delta_sel_lo95 for r in all_records]
    all_hi = [r.rec.delta_sel_hi95 for r in all_records]
    n_total = length(all_ds)
    valid_z = [i for i in 1:n_total if all_sd[i] > 1e-12]
    z_vals = [(all_ds[i] - all_pp[i]) / all_sd[i] for i in valid_z]
    mean_abs_gap = sum(abs(all_ds[i] - all_pp[i]) for i in 1:n_total) / n_total
    mean_abs_z = isempty(z_vals) ? NaN :
                 sum(abs(z) for z in z_vals) / length(z_vals)
    rms_z = isempty(z_vals) ? NaN :
            sqrt(sum(z^2 for z in z_vals) / length(z_vals))
    interval_coverage = count(all_lo[i] <= all_ds[i] <= all_hi[i]
                              for i in 1:n_total) / n_total
    z_within_2 = isempty(z_vals) ? NaN :
                 count(abs(z) <= 2.0 for z in z_vals) / length(z_vals)

    println("Conditional Wright-Fisher diagnostics:")
    println("  mean |delta_sel - expected| = " *
            string(round(mean_abs_gap, digits=6)))
    println("  mean |z| = " * string(round(mean_abs_z, digits=4)))
    println("  rms z = " * string(round(rms_z, digits=4)))
    println("  coverage in exact 95% band = " *
            string(round(interval_coverage, digits=4)))
    println("  fraction |z| <= 2 = " * string(round(z_within_2, digits=4)))
    println()

    # Secondary global regression on realized vs expected delta_sel.
    pp_bar = sum(all_pp) / n_total
    ds_bar = sum(all_ds) / n_total
    ss_xx = sum((all_pp[i] - pp_bar)^2 for i in 1:n_total)
    ss_xy = sum((all_pp[i] - pp_bar) * (all_ds[i] - ds_bar) for i in 1:n_total)
    price_slope = ss_xx > 1e-15 ? ss_xy / ss_xx : NaN
    price_intercept = ds_bar - price_slope * pp_bar
    # R^2
    ss_tot = sum((all_ds[i] - ds_bar)^2 for i in 1:n_total)
    ss_res = sum((all_ds[i] - price_intercept - price_slope * all_pp[i])^2
                  for i in 1:n_total)
    price_r2 = ss_tot > 1e-15 ? 1.0 - ss_res / ss_tot : NaN
    price_rmse = sqrt(ss_res / n_total)

    println("Price equation regression: delta_sel = a + b * price_predicted")
    println("  slope b = " * string(round(price_slope, digits=4)) *
            "  (expect ~1.0)")
    println("  intercept a = " * string(round(price_intercept, digits=6)) *
            "  (expect ~0.0)")
    println("  R^2 = " * string(round(price_r2, digits=4)))
    println("  RMSE = " * string(round(price_rmse, digits=6)))
    sign_match = count(((all_ds[i] <= 0 && all_pp[i] <= 0) ||
                        (all_ds[i] >= 0 && all_pp[i] >= 0)) for i in 1:n_total)
    println("  mean |delta_sel - price_predicted| = " *
            string(round(mean_abs_gap, digits=6)))
    println("  sign match rate = " *
            string(round(sign_match / n_total, digits=4)))
    println()

    # 2. Mean risk and fitness over generations (averaged across runs)
    println("Mean risk and fitness trajectory:")
    println("  gen | R_bar   | w_bar")
    println("  ----|---------|--------")
    for gen in 1:G
        gen_recs = [r.rec for r in all_records if r.gen == gen]
        r_bars = [sum(rec.risk_before) / length(rec.risk_before) for rec in gen_recs]
        w_bars = [sum(rec.fitness) / length(rec.fitness) for rec in gen_recs]
        r_mean = sum(r_bars) / length(r_bars)
        w_mean = sum(w_bars) / length(w_bars)
        if gen <= 5 || gen % 10 == 0 || gen == G
            println("  " * lpad(string(gen), 3) * " | " *
                    lpad(string(round(r_mean, digits=4)), 7) * " | " *
                    lpad(string(round(w_mean, digits=6)), 8))
        end
    end
    println()

    # 3. Mean separation across generations
    # mean_k is now computed from ALL K models per generation (not a sample),
    # so it directly gives the population-level mean class count.
    println("Mean separation (k) across generations:")
    for gen in [1, 10, 25, G]
        gen_recs = [r.rec for r in all_records if r.gen == gen]
        k_vals = [rec.mean_k for rec in gen_recs]
        k_mean = sum(k_vals) / length(k_vals)
        println("  gen " * lpad(string(gen), 3) * ": mean_k=" *
                string(round(k_mean, digits=2)))
    end
    println()

    # 4. Stagewise decomposition: delta_sel, delta_dev, delta_mut
    println("Stagewise decomposition (mean across runs):")
    println("  gen | delta_sel | delta_dev | delta_mut | total")
    println("  ----|-----------|-----------|-----------|------")
    for gen in 1:G
        gen_recs = [r.rec for r in all_records if r.gen == gen]
        ds = sum(rec.delta_sel for rec in gen_recs) / length(gen_recs)
        dd = sum(rec.delta_dev for rec in gen_recs) / length(gen_recs)
        dm = sum(rec.delta_mut for rec in gen_recs) / length(gen_recs)
        dt = ds + dd + dm
        if gen <= 5 || gen % 10 == 0 || gen == G
            println("  " * lpad(string(gen), 3) * " | " *
                    lpad(string(round(ds, digits=6)), 9) * " | " *
                    lpad(string(round(dd, digits=6)), 9) * " | " *
                    lpad(string(round(dm, digits=6)), 9) * " | " *
                    lpad(string(round(dt, digits=6)), 9))
        end
    end
    println()

    # ── Predictions check ─────────────────────────────────────────────────────
    println("=" ^ 60)
    println("PREDICTIONS CHECK")
    println("=" ^ 60)
    println()

    println("1. Selection-stage expectation matches Wright-Fisher sampling noise")
    println("   coverage in exact 95% band = " *
            string(round(interval_coverage, digits=4)))
    println("   fraction |z| <= 2 = " * string(round(z_within_2, digits=4)))
    pass_replay = !isnan(interval_coverage) && interval_coverage >= 0.9
    pass_z      = !isnan(z_within_2) && z_within_2 >= 0.9
    println("   exact 95% coverage >= 0.9: " *
            (pass_replay ? "PASS" : "CHECK"))
    println("   fraction |z| <= 2 >= 0.9: " *
            (pass_z ? "PASS" : "CHECK"))
    println("   secondary OLS: slope=" * string(round(price_slope, digits=4)) *
            " intercept=" * string(round(price_intercept, digits=6)) *
            " R^2=" * string(round(price_r2, digits=4)))
    println()

    # Check risk decreases over generations
    first_gen_recs = [r.rec for r in all_records if r.gen == 1]
    last_gen_recs  = [r.rec for r in all_records if r.gen == G]
    r_first = sum(sum(rec.risk_before) / length(rec.risk_before)
                  for rec in first_gen_recs) / length(first_gen_recs)
    r_last  = sum(sum(rec.risk_before) / length(rec.risk_before)
                  for rec in last_gen_recs) / length(last_gen_recs)
    println("2. Mean risk decreases over generations")
    println("   R_bar(gen=1)=" * string(round(r_first, digits=4)) *
            " R_bar(gen=" * string(G) * ")=" * string(round(r_last, digits=4)))
    println("   " * (r_last < r_first ? "PASS" : "CHECK") *
            " (expect decrease)")
    println()

    # Check separation increases
    k_first = sum(rec.mean_k for rec in first_gen_recs) / length(first_gen_recs)
    k_last  = sum(rec.mean_k for rec in last_gen_recs) / length(last_gen_recs)
    println("3. Behavioural separation increases")
    println("   mean_k(gen=1)=" * string(round(k_first, digits=2)) *
            " mean_k(gen=" * string(G) * ")=" * string(round(k_last, digits=2)))
    println("   " * (k_last >= k_first ? "PASS" : "CHECK") *
            " (expect k increases toward 5)")
    println()

    # Check delta_sel is negative on average (selection lowers risk)
    mean_ds_all = sum(all_ds) / length(all_ds)
    println("4. Selection lowers risk (delta_sel < 0 on average)")
    println("   mean delta_sel = " * string(round(mean_ds_all, digits=6)))
    println("   " * (mean_ds_all < 0 ? "PASS" : "CHECK") *
            " (expect negative)")
    println()

    # ── Save results ──────────────────────────────────────────────────────────
    mkpath(OUTPUT_DIR)
    outpath = joinpath(OUTPUT_DIR, "exp3_population_results.txt")
    open(outpath, "w") do io
        println(io, "# Experiment 3: Population Dynamics and Price Equation")
        println(io, "# K=" * string(K) * " G=" * string(G) *
                    " N_RUNS=" * string(N_RUNS))
        println(io, "# n_dev_steps=" * string(N_DEV_STEPS) *
                    " lr_dev=" * string(LR_DEV) *
                    " sigma_mut=" * string(SIGMA_MUT))
        println(io, "# N_INIT_STEPS=" * string(N_INIT_STEPS) *
                    " LR_INIT=" * string(LR_INIT))
        println(io, "# CHUNK_SIZE=" * string(CHUNK_SIZE) *
                    " CHUNK_OVERLAP=" * string(CHUNK_OVERLAP))
        println(io, "# PREFIX_LENGTHS=" * string(PREFIX_LENGTHS))
        println(io, "# tau=" * string(tau))
        println(io, "# block_size=" * string(cfg.block_size) *
                    " n_embd=" * string(cfg.n_embd) *
                    " n_layer=" * string(cfg.n_layer) *
                    " n_head=" * string(cfg.n_head))
        println(io, "# Price equation OLS: slope=" *
                    string(round(price_slope, digits=4)) *
                    " intercept=" *
                    string(round(price_intercept, digits=6)) *
                    " R^2=" * string(round(price_r2, digits=4)))
        println(io, "")
        println(io, "run,gen,R_bar_before,w_bar,delta_sel,price_predicted," *
                    "delta_sel_sd,delta_sel_lo95,delta_sel_hi95," *
                    "delta_dev,delta_mut,mean_k,mean_s," *
                    "risk_selected,risk_selected_expected," *
                    "risk_after_dev,risk_after_mut")
        for r in all_records
            rec = r.rec
            r_bar = sum(rec.risk_before) / length(rec.risk_before)
            w_bar = sum(rec.fitness) / length(rec.fitness)
            println(io, string(r.run) * "," *
                        string(r.gen) * "," *
                        string(round(r_bar, digits=6)) * "," *
                        string(round(w_bar, digits=8)) * "," *
                        string(round(rec.delta_sel, digits=8)) * "," *
                        string(round(rec.price_predicted, digits=8)) * "," *
                        string(round(rec.delta_sel_sd, digits=8)) * "," *
                        string(round(rec.delta_sel_lo95, digits=8)) * "," *
                        string(round(rec.delta_sel_hi95, digits=8)) * "," *
                        string(round(rec.delta_dev, digits=8)) * "," *
                        string(round(rec.delta_mut, digits=8)) * "," *
                        string(round(rec.mean_k, digits=2)) * "," *
                        string(round(rec.mean_s, digits=2)) * "," *
                        string(round(rec.risk_selected, digits=6)) * "," *
                        string(round(rec.risk_selected_expected, digits=6)) * "," *
                        string(round(rec.risk_after_dev, digits=6)) * "," *
                        string(round(rec.risk_after_mut, digits=6)))
        end
    end
    println("Results saved to " * outpath)
    println("Total time: " * string(round(time() - t_start, digits=1)) * "s")
end

main()
