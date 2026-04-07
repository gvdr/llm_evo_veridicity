"""
Experiment 7: Two-Ecology Transfer (Bracket Discrimination)

Properly separates the token and evaluation ecologies:

  Token ecology: next-token prediction on Lisp source code.
    Recipe trait alpha in [0,1] controls the fraction of bracket-containing
    text in the training data. At alpha=0, training sees only bracket-scrubbed
    code from the same training split. At alpha=1, training includes the
    bracket-rich originals. No balance labels or summary tokens appear during
    training.

  Evaluation ecology: held-out NLL discrimination between balanced and
    unbalanced bracket chunks. The model is never trained on balance labels.
    A model that has learned bracket structure from real Lisp should assign
    lower NLL to balanced code (matching its learned patterns) and higher
    NLL to permuted-bracket code (violating those patterns).

  Fitness: discrimination = mean_NLL(unbalanced) - mean_NLL(balanced).
    Higher discrimination = better bracket tracking = higher fitness.

  Selection: Wright-Fisher on fitness. Selects for recipes whose data
    composition produces models that happen to discriminate bracket balance,
    even though no recipe was designed for that purpose.

Run: julia --project=. -t 16 scripts/exp7_transfer_ecology.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "microgpt.jl"))

using Random, LinearAlgebra

BLAS.set_num_threads(1)

# ── Configuration ─────────────────────────────────────────────────────────────

const OUTPUT_DIR = joinpath(@__DIR__, "..", "output")
const DATA_DIR = joinpath(@__DIR__, "..", "data", "lisp")
const BALANCED_PATH = joinpath(DATA_DIR, "lisp_balanced.txt")
const UNBALANCED_PATH = joinpath(DATA_DIR, "lisp_unbalanced.txt")

const N_LAYER = 2
const N_EMBD = 32
const N_HEAD = 4
const BLOCK_SIZE = 32

# Charset: a-z (1-26) + '(' (27) + ')' (28) + ' ' (29) + BOS (30)
# No summary tokens — evaluation is NLL-based only
const VOCAB_SIZE = 30
const BOS_TOKEN = 30

const LR = 0.005
const N_TRAIN_STEPS = 12000
const N_EVAL = 300           # chunks per type for evaluation
const TRAIN_EVAL_SPLIT = 0.8 # fraction of corpus used for training

# Stage 2
const K_POP = 14
const G_GEN = 25
const N_TRAIN_POP = 6000
const SIGMA_MUT = 0.12

# ── Tokenizer ─────────────────────────────────────────────────────────────────

const CHAR_TO_IDX = let
    d = Dict{Char,Int}()
    for (i, c) in enumerate('a':'z')
        d[c] = i
    end
    d['('] = 27
    d[')'] = 28
    d[' '] = 29
    d
end

function tokenize_chunk(chunk::AbstractString)
    tokens = Vector{Int}(undef, length(chunk))
    for (i, ch) in enumerate(chunk)
        tokens[i] = get(CHAR_TO_IDX, ch, 29)  # unknown -> space
    end
    tokens
end

function make_sequence(chunk::AbstractString)
    vcat([BOS_TOKEN], tokenize_chunk(chunk))
end

# ── Data loading and splitting ────────────────────────────────────────────────

function has_brackets(chunk::AbstractString)
    for ch in chunk
        if ch == '(' || ch == ')'
            return true
        end
    end
    false
end

function scrub_brackets(chunk::AbstractString)
    chars = collect(chunk)
    for i in eachindex(chars)
        if chars[i] == '(' || chars[i] == ')'
            chars[i] = ' '
        end
    end
    String(chars)
end

function load_and_split(rng::AbstractRNG)
    bal_lines = readlines(BALANCED_PATH)
    unbal_lines = readlines(UNBALANCED_PATH)
    n_pairs = min(length(bal_lines), length(unbal_lines))

    # Shuffle and split into train / eval portions
    perm = randperm(rng, n_pairs)
    n_train = round(Int, n_pairs * TRAIN_EVAL_SPLIT)
    n_eval_use = min(N_EVAL, n_pairs - n_train)

    train_idx = perm[1:n_train]
    eval_idx = perm[(n_train + 1):(n_train + n_eval_use)]

    # Training pools: bracket-rich originals and bracket-scrubbed controls.
    # Both are built from the training split only, so alpha changes bracket
    # exposure without changing the underlying text source or leaking eval data.
    train_bracket_rich = Vector{Vector{Int}}()
    train_bracket_free = Vector{Vector{Int}}()

    for i in train_idx
        line = bal_lines[i]
        if has_brackets(line)
            push!(train_bracket_rich, make_sequence(line))
            push!(train_bracket_free, make_sequence(scrub_brackets(line)))
        end
    end

    # Held-out evaluation sets (never in training)
    eval_balanced = [make_sequence(bal_lines[i]) for i in eval_idx]
    eval_unbalanced = [make_sequence(unbal_lines[i]) for i in eval_idx]

    println("Training pools:")
    println("  bracket-rich: " * string(length(train_bracket_rich)) * " sequences")
    println("  bracket-scrubbed: " * string(length(train_bracket_free)) * " sequences")
    println("Evaluation (held-out):")
    println("  balanced:   " * string(length(eval_balanced)) * " sequences")
    println("  unbalanced: " * string(length(eval_unbalanced)) * " sequences")

    (bracket_rich=train_bracket_rich,
     bracket_free=train_bracket_free,
     eval_balanced=eval_balanced,
     eval_unbalanced=eval_unbalanced)
end

# ── Training ──────────────────────────────────────────────────────────────────

function train_with_alpha(alpha::Float64,
                          bracket_rich::Vector{Vector{Int}},
                          bracket_free::Vector{Vector{Int}},
                          cfg::ModelConfig, n_steps::Int;
                          seed::Int=42)
    ms = init_model(cfg; seed=seed)
    rng = MersenneTwister(seed + 1000)
    nr = length(bracket_rich)
    nf = length(bracket_free)

    if nr == 0 && alpha > 0.0
        error("No bracket-rich sequences but alpha > 0")
    end
    if nf == 0 && alpha < 1.0
        error("No bracket-free sequences but alpha < 1")
    end

    for step in 1:n_steps
        if rand(rng) < alpha && nr > 0
            seq = bracket_rich[rand(rng, 1:nr)]
        else
            seq = bracket_free[rand(rng, 1:nf)]
        end
        train_step_lr!(ms, seq, step, cfg, LR)
    end
    ms
end

# ── Evaluation ────────────────────────────────────────────────────────────────

function mean_nll(W::Dict{String,Matrix{Float64}},
                  seqs::Vector{Vector{Int}},
                  cfg::ModelConfig)
    total = 0.0
    n_tok = 0
    for seq in seqs
        nll, nt = sequence_nll(W, seq, cfg)
        total += nll
        n_tok += nt
    end
    n_tok > 0 ? total / n_tok : 0.0
end

function eval_discrimination(W::Dict{String,Matrix{Float64}},
                             eval_balanced::Vector{Vector{Int}},
                             eval_unbalanced::Vector{Vector{Int}},
                             cfg::ModelConfig)
    nll_bal = mean_nll(W, eval_balanced, cfg)
    nll_unbal = mean_nll(W, eval_unbalanced, cfg)
    discrimination = nll_unbal - nll_bal
    (nll_bal=nll_bal, nll_unbal=nll_unbal, discrimination=discrimination)
end

# ── Stage 1: Static alpha sweep ──────────────────────────────────────────────

function run_stage1(data, cfg::ModelConfig; base_seed::Int=42)
    println("=" ^ 60)
    println("Stage 1: Static alpha sweep")
    println("=" ^ 60)
    println()

    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    results = []

    for alpha in alphas
        t1 = time()
        ms = train_with_alpha(alpha, data.bracket_rich, data.bracket_free,
                              cfg, N_TRAIN_STEPS; seed=base_seed)
        ev = eval_discrimination(ms.W, data.eval_balanced,
                                 data.eval_unbalanced, cfg)
        dt = round(time() - t1; digits=1)

        push!(results, (alpha=alpha, nll_bal=ev.nll_bal,
                        nll_unbal=ev.nll_unbal,
                        discrimination=ev.discrimination))

        println("  alpha=" * lpad(string(alpha), 4) *
                "  NLL_bal=" * string(round(ev.nll_bal; digits=4)) *
                "  NLL_unbal=" * string(round(ev.nll_unbal; digits=4)) *
                "  discrim=" * string(round(ev.discrimination; digits=4)) *
                "  (" * string(dt) * "s)")
    end

    println()
    results
end

# ── Stage 2: Population selection ─────────────────────────────────────────────

function run_stage2(data, cfg::ModelConfig; base_seed::Int=42)
    println("=" ^ 60)
    println("Stage 2: Population selection on recipe trait alpha")
    println("=" ^ 60)
    println()

    rng_pop = MersenneTwister(base_seed + 5000)
    alphas = [rand(rng_pop) for _ in 1:K_POP]

    println("Initial alphas: " *
            join([string(round(a; digits=3)) for a in alphas], ", "))
    println()

    records = []

    for gen in 1:G_GEN
        t1 = time()

        fitnesses = Vector{Float64}(undef, K_POP)
        discriminations = Vector{Float64}(undef, K_POP)

        Threads.@threads for i in 1:K_POP
            ms = train_with_alpha(alphas[i], data.bracket_rich,
                                  data.bracket_free, cfg, N_TRAIN_POP;
                                  seed=base_seed + gen * 1000 + i)
            ev = eval_discrimination(ms.W, data.eval_balanced,
                                     data.eval_unbalanced, cfg)
            discriminations[i] = ev.discrimination
            # Fitness: exp(discrimination). If discrimination <= 0, fitness is low
            fitnesses[i] = exp(ev.discrimination)
        end

        alpha_bar = sum(alphas) / K_POP
        d_bar = sum(discriminations) / K_POP
        f_bar = sum(fitnesses) / K_POP

        # Wright-Fisher selection
        cum_fit = cumsum(fitnesses)
        total_fit = cum_fit[end]
        new_alphas = Vector{Float64}(undef, K_POP)
        for i in 1:K_POP
            r = rand(rng_pop) * total_fit
            parent = 1
            while parent < K_POP && cum_fit[parent] < r
                parent += 1
            end
            new_alphas[i] = clamp(alphas[parent] + SIGMA_MUT * randn(rng_pop),
                                  0.0, 1.0)
        end

        alpha_bar_sel = 0.0
        for i in 1:K_POP
            alpha_bar_sel += fitnesses[i] * alphas[i]
        end
        alpha_bar_sel /= total_fit

        dt = round(time() - t1; digits=1)
        push!(records, (gen=gen, alpha_bar=alpha_bar,
                        alpha_bar_sel=alpha_bar_sel,
                        d_bar=d_bar, f_bar=f_bar))

        println("  gen " * lpad(string(gen), 2) *
                "  alpha_bar=" * string(round(alpha_bar; digits=3)) *
                "  alpha_sel=" * string(round(alpha_bar_sel; digits=3)) *
                "  discrim=" * string(round(d_bar; digits=4)) *
                "  (" * string(dt) * "s)")

        alphas = new_alphas
    end

    println()
    println("Final alphas: " *
            join([string(round(a; digits=3)) for a in alphas], ", "))
    println()

    records
end

# ── Output ────────────────────────────────────────────────────────────────────

function write_results(stage1, stage2)
    path = joinpath(OUTPUT_DIR, "exp7_transfer_ecology_results.txt")
    open(path, "w") do io
        println(io, "# Experiment 7: Two-Ecology Transfer (Bracket Discrimination)")
        println(io, "# Corpus: Practical Common Lisp (normalized)")
        println(io, "# N_TRAIN_STEPS=" * string(N_TRAIN_STEPS) *
                     " LR=" * string(LR))
        println(io, "# VOCAB_SIZE=" * string(VOCAB_SIZE) *
                     " BLOCK_SIZE=" * string(BLOCK_SIZE))
        println(io, "# K_POP=" * string(K_POP) *
                     " G_GEN=" * string(G_GEN) *
                     " N_TRAIN_POP=" * string(N_TRAIN_POP) *
                     " SIGMA_MUT=" * string(SIGMA_MUT))
        println(io, "# Evaluation: NLL discrimination (never trained on balance labels)")
        println(io)

        println(io, "# Stage 1: Static alpha sweep")
        println(io, "alpha,nll_bal,nll_unbal,discrimination")
        for r in stage1
            println(io, string(r.alpha) * "," *
                        string(round(r.nll_bal; digits=6)) * "," *
                        string(round(r.nll_unbal; digits=6)) * "," *
                        string(round(r.discrimination; digits=6)))
        end
        println(io)

        println(io, "# Stage 2: Population selection")
        println(io, "gen,alpha_bar,alpha_bar_sel,discrimination_bar,f_bar")
        for r in stage2
            println(io, string(r.gen) * "," *
                        string(round(r.alpha_bar; digits=6)) * "," *
                        string(round(r.alpha_bar_sel; digits=6)) * "," *
                        string(round(r.d_bar; digits=6)) * "," *
                        string(round(r.f_bar; digits=6)))
        end
    end
    println("Results saved to " * path)
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    cfg = ModelConfig(N_LAYER, N_EMBD, N_HEAD, BLOCK_SIZE, VOCAB_SIZE,
                      4, 0.02, LR, 0.9, 0.999, 1e-8)

    println("Model: " * string(n_params(init_model(cfg))) * " params")
    println()

    rng_data = MersenneTwister(20260317)
    data = load_and_split(rng_data)
    println()

    stage1 = run_stage1(data, cfg)
    stage2 = run_stage2(data, cfg)
    write_results(stage1, stage2)
end

main()
