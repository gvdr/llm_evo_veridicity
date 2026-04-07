"""
Experiment 6: Two-Ecology Bracket Balance (Design B)

Two world states:
  A (balanced): Lisp source chunks with verified properly nested parentheses
  B (unbalanced): same chunks with bracket pattern permuted to break nesting

Local character statistics are nearly identical (same filler text, same bracket
positions and counts, only open/close ordering differs). The global property —
bracket balance — is invisible to short-context next-token prediction but
testable by an evaluation ecology.

Each sequence:  BOS + chunk(30 chars) + summary token = 32 tokens
Summary token:  'y' if balanced,  'z' if unbalanced

Token ecology (alpha=0): train on body-only sequences (chunk without summary).
  sigma_tok(A,B) is small because local character statistics are similar.

Evaluation ecology: CE on the summary token.
  sigma_eval(A,B) > 0 because summaries differ deterministically.

The summary token is injected directly during post-training, so this remains a
deliberately leaky two-ecology design. But evaluation is held out: the
balanced/unbalanced chunks used for reporting are disjoint from those used for
training.

Stage 1: static alpha sweep (base train + post-train at varying alpha)
Stage 2: population selection on recipe trait alpha

Run: julia --project=. -t 16 scripts/exp6_bracket_ecology.jl
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

# Charset: a-z (1-26) + '(' (27) + ')' (28) + ' ' (29) + summary_y (30) +
#          summary_z (31) + BOS (32)
# Total vocab: 32
const VOCAB_SIZE = 32
const BOS_TOKEN = 32
const SUMMARY_BAL = 30    # 'y' — balanced
const SUMMARY_UNBAL = 31  # 'z' — unbalanced

const LR = 0.005
const N_BASE_STEPS = 10000
const N_POST_STEPS = 4000
const N_EVAL = 200   # sequences per type for evaluation
const TRAIN_EVAL_SPLIT = 0.8

# Stage 2
const K_POP = 12
const G_GEN = 20
const N_POST_POP = 1500
const SIGMA_MUT = 0.10

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
        tokens[i] = get(CHAR_TO_IDX, ch, 29)  # unknown → space
    end
    tokens
end

# ── Data loading ──────────────────────────────────────────────────────────────

function load_corpus(rng::AbstractRNG)
    bal_lines = readlines(BALANCED_PATH)
    unbal_lines = readlines(UNBALANCED_PATH)
    n_pairs = min(length(bal_lines), length(unbal_lines))

    perm = randperm(rng, n_pairs)
    n_train = round(Int, n_pairs * TRAIN_EVAL_SPLIT)
    n_eval_use = min(N_EVAL, n_pairs - n_train)
    train_idx = perm[1:n_train]
    eval_idx = perm[(n_train + 1):(n_train + n_eval_use)]

    function make_body(line::AbstractString)
        vcat([BOS_TOKEN], tokenize_chunk(line))
    end

    function make_full(line::AbstractString, summary::Int)
        vcat([BOS_TOKEN], tokenize_chunk(line), [summary])
    end

    body_bal_train = [make_body(bal_lines[i]) for i in train_idx]
    body_unbal_train = [make_body(unbal_lines[i]) for i in train_idx]
    full_bal_train = [make_full(bal_lines[i], SUMMARY_BAL) for i in train_idx]
    full_unbal_train = [make_full(unbal_lines[i], SUMMARY_UNBAL) for i in train_idx]

    body_bal_eval = [make_body(bal_lines[i]) for i in eval_idx]
    body_unbal_eval = [make_body(unbal_lines[i]) for i in eval_idx]
    full_bal_eval = [make_full(bal_lines[i], SUMMARY_BAL) for i in eval_idx]
    full_unbal_eval = [make_full(unbal_lines[i], SUMMARY_UNBAL) for i in eval_idx]

    body_all_train = vcat(body_bal_train, body_unbal_train)
    full_all_train = vcat(full_bal_train, full_unbal_train)

    println("Loaded: " * string(n_pairs) * " balanced/unbalanced pairs")
    println("Training pairs: " * string(length(train_idx)))
    println("Held-out eval pairs: " * string(length(eval_idx)))
    println("Sequence lengths: body=" * string(length(body_bal_train[1])) *
            " full=" * string(length(full_bal_train[1])))

    (body_all_train=body_all_train,
     full_all_train=full_all_train,
     full_bal_train=full_bal_train,
     full_unbal_train=full_unbal_train,
     body_bal_train=body_bal_train,
     body_unbal_train=body_unbal_train,
     body_bal_eval=body_bal_eval,
     body_unbal_eval=body_unbal_eval,
     full_bal_eval=full_bal_eval,
     full_unbal_eval=full_unbal_eval)
end

# ── Training ──────────────────────────────────────────────────────────────────

function train_base(body_seqs::Vector{Vector{Int}}, cfg::ModelConfig,
                    n_steps::Int; seed::Int=42)
    ms = init_model(cfg; seed=seed)
    rng = MersenneTwister(seed + 1000)
    n = length(body_seqs)
    t0 = time()
    for step in 1:n_steps
        seq = body_seqs[rand(rng, 1:n)]
        train_step_lr!(ms, seq, step, cfg, LR)
        if step % 2500 == 0
            elapsed = time() - t0
            eta = elapsed / step * (n_steps - step)
            println("    base step " * string(step) * "/" * string(n_steps) *
                    " (" * string(round(elapsed; digits=1)) * "s, ETA " *
                    string(round(eta; digits=0)) * "s)")
        end
    end
    ms
end

function post_train!(ms::ModelState, alpha::Float64,
                     body_seqs::Vector{Vector{Int}},
                     full_seqs::Vector{Vector{Int}},
                     cfg::ModelConfig, n_steps::Int;
                     rng::AbstractRNG=MersenneTwister(99),
                     step_offset::Int=0)
    nb = length(body_seqs)
    nf = length(full_seqs)
    for step in 1:n_steps
        if rand(rng) < alpha
            seq = full_seqs[rand(rng, 1:nf)]
        else
            seq = body_seqs[rand(rng, 1:nb)]
        end
        train_step_lr!(ms, seq, step + step_offset, cfg, LR)
    end
    ms
end

# ── Evaluation ────────────────────────────────────────────────────────────────

function summary_nll_at(W::Dict{String,Matrix{Float64}},
                        full_seq::Vector{Int}, cfg::ModelConfig,
                        target_token::Int)
    n = length(full_seq) - 1
    Kc = [Vector{Float64}[] for _ in 1:cfg.n_layer]
    Vc = [Vector{Float64}[] for _ in 1:cfg.n_layer]
    ws = ForwardWorkspace(cfg)
    local logits::Vector{Float64}
    for t in 1:n
        logits = gpt_forward(full_seq[t], t, Kc, Vc, W, cfg, ws)
    end
    mx = maximum(logits)
    lse = 0.0
    for i in eachindex(logits)
        lse += exp(logits[i] - mx)
    end
    lse = log(lse)
    -((logits[target_token] - mx) - lse)
end

function eval_summary_ce(W::Dict{String,Matrix{Float64}},
                         full_bal::Vector{Vector{Int}},
                         full_unbal::Vector{Vector{Int}},
                         cfg::ModelConfig;
                         n_eval::Int=N_EVAL)
    ne_b = min(n_eval, length(full_bal))
    ne_u = min(n_eval, length(full_unbal))

    ce_bal = 0.0
    for i in 1:ne_b
        ce_bal += summary_nll_at(W, full_bal[i], cfg, SUMMARY_BAL)
    end
    ce_bal /= ne_b

    ce_unbal = 0.0
    for i in 1:ne_u
        ce_unbal += summary_nll_at(W, full_unbal[i], cfg, SUMMARY_UNBAL)
    end
    ce_unbal /= ne_u

    (ce_bal + ce_unbal) / 2.0
end

function eval_body_ce(W::Dict{String,Matrix{Float64}},
                      body_seqs::Vector{Vector{Int}},
                      cfg::ModelConfig;
                      n_eval::Int=200)
    ne = min(n_eval, length(body_seqs))
    total = 0.0
    n_tok = 0
    for i in 1:ne
        nll, nt = sequence_nll(W, body_seqs[i], cfg)
        total += nll
        n_tok += nt
    end
    n_tok > 0 ? total / n_tok : 0.0
end

function summary_output_distribution(W::Dict{String,Matrix{Float64}},
                                     full_seqs::Vector{Vector{Int}},
                                     cfg::ModelConfig;
                                     n_eval::Int=100)
    ne = min(n_eval, length(full_seqs))
    dist = zeros(VOCAB_SIZE)
    for i in 1:ne
        seq = full_seqs[i]
        n = length(seq) - 1
        Kc = [Vector{Float64}[] for _ in 1:cfg.n_layer]
        Vc = [Vector{Float64}[] for _ in 1:cfg.n_layer]
        ws = ForwardWorkspace(cfg)
        local logits::Vector{Float64}
        for t in 1:n
            logits = gpt_forward(seq[t], t, Kc, Vc, W, cfg, ws)
        end
        mx = maximum(logits)
        probs = exp.(logits .- mx)
        probs ./= sum(probs)
        dist .+= probs
    end
    dist ./= ne
    dist
end

function js_divergence(p::Vector{Float64}, q::Vector{Float64})
    m = 0.5 .* (p .+ q)
    kl_pm = 0.0
    kl_qm = 0.0
    for i in eachindex(p)
        if p[i] > 0.0 && m[i] > 0.0
            kl_pm += p[i] * log(p[i] / m[i])
        end
        if q[i] > 0.0 && m[i] > 0.0
            kl_qm += q[i] * log(q[i] / m[i])
        end
    end
    0.5 * kl_pm + 0.5 * kl_qm
end

# ── Stage 1: Static alpha sweep ──────────────────────────────────────────────

function run_stage1(data, cfg::ModelConfig; base_seed::Int=42)
    println("=" ^ 60)
    println("Stage 1: Static alpha sweep")
    println("=" ^ 60)
    println()

    println("Training base model (body-only, " * string(N_BASE_STEPS) * " steps)...")
    base_ms = train_base(data.body_all_train, cfg, N_BASE_STEPS; seed=base_seed)

    base_body_ce = eval_body_ce(base_ms.W, vcat(data.body_bal_eval, data.body_unbal_eval), cfg)
    println("  base body CE: " * string(round(base_body_ce; digits=4)))

    # Check base model sigma_tok: body CE on balanced vs unbalanced
    ce_body_bal = eval_body_ce(base_ms.W, data.body_bal_eval, cfg)
    ce_body_unbal = eval_body_ce(base_ms.W, data.body_unbal_eval, cfg)
    println("  base body CE (balanced chunks): " * string(round(ce_body_bal; digits=4)))
    println("  base body CE (unbalanced chunks): " * string(round(ce_body_unbal; digits=4)))
    println("  body CE gap |bal - unbal|: " *
            string(round(abs(ce_body_bal - ce_body_unbal); digits=6)))
    println()

    alphas = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    results = []

    for alpha in alphas
        t1 = time()
        ms = copy_model(base_ms; reset_optimizer=true)
        rng_post = MersenneTwister(base_seed + 2000 + round(Int, alpha * 1000))
        post_train!(ms, alpha, data.body_all_train, data.full_all_train, cfg, N_POST_STEPS;
                    rng=rng_post, step_offset=N_BASE_STEPS)

        body_ce = eval_body_ce(ms.W, vcat(data.body_bal_eval, data.body_unbal_eval), cfg)
        summ_ce = eval_summary_ce(ms.W, data.full_bal_eval, data.full_unbal_eval, cfg)
        dist_bal = summary_output_distribution(ms.W, data.full_bal_eval, cfg)
        dist_unbal = summary_output_distribution(ms.W, data.full_unbal_eval, cfg)
        js = js_divergence(dist_bal, dist_unbal)
        dt = round(time() - t1; digits=1)

        push!(results, (alpha=alpha, body_ce=body_ce, summary_ce=summ_ce,
                        js_AB=js))

        println("  alpha=" * lpad(string(alpha), 4) *
                "  body_ce=" * string(round(body_ce; digits=4)) *
                "  summary_ce=" * string(round(summ_ce; digits=4)) *
                "  JS(bal,unbal)=" * string(round(js; digits=6)) *
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

    println("Training shared base model (" * string(N_BASE_STEPS) * " steps)...")
    base_ms = train_base(data.body_all_train, cfg, N_BASE_STEPS; seed=base_seed)
    println()

    rng_pop = MersenneTwister(base_seed + 5000)
    alphas = [rand(rng_pop) for _ in 1:K_POP]

    println("Initial alphas: " * join([string(round(a; digits=3)) for a in alphas], ", "))
    println()

    records = []

    for gen in 1:G_GEN
        t1 = time()

        fitnesses = Vector{Float64}(undef, K_POP)
        summary_ces = Vector{Float64}(undef, K_POP)

        Threads.@threads for i in 1:K_POP
            ms = copy_model(base_ms; reset_optimizer=true)
            rng_i = MersenneTwister(base_seed + gen * 1000 + i)
            post_train!(ms, alphas[i], data.body_all_train, data.full_all_train,
                        cfg, N_POST_POP; rng=rng_i, step_offset=N_BASE_STEPS)
            sce = eval_summary_ce(ms.W, data.full_bal_eval, data.full_unbal_eval, cfg;
                                  n_eval=100)
            summary_ces[i] = sce
            fitnesses[i] = exp(-sce)
        end

        alpha_bar = sum(alphas) / K_POP
        f_bar = sum(fitnesses) / K_POP
        ce_bar = sum(summary_ces) / K_POP

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
        push!(records, (gen=gen, alpha_bar=alpha_bar, alpha_bar_sel=alpha_bar_sel,
                        ce_bar=ce_bar, f_bar=f_bar))

        println("  gen " * lpad(string(gen), 2) *
                "  alpha_bar=" * string(round(alpha_bar; digits=3)) *
                "  alpha_sel=" * string(round(alpha_bar_sel; digits=3)) *
                "  ce_bar=" * string(round(ce_bar; digits=3)) *
                "  (" * string(dt) * "s)")

        alphas = new_alphas
    end

    println()
    println("Final alphas: " * join([string(round(a; digits=3)) for a in alphas], ", "))
    println()

    records
end

# ── Output ────────────────────────────────────────────────────────────────────

function write_results(stage1, stage2)
    path = joinpath(OUTPUT_DIR, "exp6_bracket_ecology_results.txt")
    open(path, "w") do io
        println(io, "# Experiment 6: Two-Ecology Bracket Balance")
        println(io, "# Corpus: Practical Common Lisp (normalized)")
        println(io, "# N_BASE_STEPS=" * string(N_BASE_STEPS) *
                     " N_POST_STEPS=" * string(N_POST_STEPS) *
                     " LR=" * string(LR))
        println(io, "# VOCAB_SIZE=" * string(VOCAB_SIZE) *
                     " BLOCK_SIZE=" * string(BLOCK_SIZE))
        println(io, "# K_POP=" * string(K_POP) *
                     " G_GEN=" * string(G_GEN) *
                     " N_POST_POP=" * string(N_POST_POP) *
                     " SIGMA_MUT=" * string(SIGMA_MUT))
        println(io)

        println(io, "# Stage 1: Static alpha sweep")
        println(io, "alpha,body_ce,summary_ce,js_bal_unbal")
        for r in stage1
            println(io, string(r.alpha) * "," *
                        string(round(r.body_ce; digits=6)) * "," *
                        string(round(r.summary_ce; digits=6)) * "," *
                        string(round(r.js_AB; digits=8)))
        end
        println(io)

        println(io, "# Stage 2: Population selection")
        println(io, "gen,alpha_bar,alpha_bar_sel,ce_bar,f_bar")
        for r in stage2
            println(io, string(r.gen) * "," *
                        string(round(r.alpha_bar; digits=6)) * "," *
                        string(round(r.alpha_bar_sel; digits=6)) * "," *
                        string(round(r.ce_bar; digits=6)) * "," *
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
    data = load_corpus(rng_data)
    println()

    stage1 = run_stage1(data, cfg)
    stage2 = run_stage2(data, cfg)
    write_results(stage1, stage2)
end

main()
