# ── Refactored microgpt v2 — parameterized, no globals ────────────────────────
#
# Based on microgpt_v2_array.jl (Karpathy's microgpt ported to Julia).
# Changes: ModelConfig + ModelState structs, explicit parameter threading,
# forward-only evaluation path (no Tensor), copy/mutate utilities.
#
# Dependencies: Random, LinearAlgebra (stdlib only).

using Random, LinearAlgebra

# ── Configuration ─────────────────────────────────────────────────────────────

struct ModelConfig
    n_layer    :: Int
    n_embd     :: Int
    n_head     :: Int
    block_size :: Int
    vocab_size :: Int
    mlp_mult   :: Int      # MLP hidden = mlp_mult * n_embd
    init_std   :: Float64  # weight initialization std
    lr         :: Float64
    beta1      :: Float64
    beta2      :: Float64
    eps        :: Float64
end

function ModelConfig(; n_layer=1, n_embd=16, n_head=4, block_size=16,
                       vocab_size=27, mlp_mult=4, init_std=0.08,
                       lr=0.01, beta1=0.85, beta2=0.99, eps=1e-8)
    @assert n_embd % n_head == 0 "n_embd must be divisible by n_head"
    ModelConfig(n_layer, n_embd, n_head, block_size, vocab_size,
                mlp_mult, init_std, lr, beta1, beta2, eps)
end

# ── Model state ───────────────────────────────────────────────────────────────

struct ModelState
    W  :: Dict{String, Matrix{Float64}}   # parameters
    G  :: Dict{String, Matrix{Float64}}   # gradient accumulators
    M  :: Dict{String, Matrix{Float64}}   # Adam first moment
    Va :: Dict{String, Matrix{Float64}}   # Adam second moment
end

function init_model(cfg::ModelConfig; seed::Int=42)
    rng = MersenneTwister(seed)
    rmat(r, c) = randn(rng, r, c) .* cfg.init_std

    W = Dict{String, Matrix{Float64}}(
        "wte"     => rmat(cfg.vocab_size, cfg.n_embd),
        "wpe"     => rmat(cfg.block_size, cfg.n_embd),
        "lm_head" => rmat(cfg.vocab_size, cfg.n_embd))

    for l in 0:(cfg.n_layer - 1)
        sl = string(l)
        W[sl * ".attn_wq"]  = rmat(cfg.n_embd, cfg.n_embd)
        W[sl * ".attn_wk"]  = rmat(cfg.n_embd, cfg.n_embd)
        W[sl * ".attn_wv"]  = rmat(cfg.n_embd, cfg.n_embd)
        W[sl * ".attn_wo"]  = rmat(cfg.n_embd, cfg.n_embd)
        W[sl * ".mlp_fc1"]  = rmat(cfg.mlp_mult * cfg.n_embd, cfg.n_embd)
        W[sl * ".mlp_fc2"]  = rmat(cfg.n_embd, cfg.mlp_mult * cfg.n_embd)
    end

    G  = Dict(k => zeros(size(v)) for (k, v) in W)
    M  = Dict(k => zeros(size(v)) for (k, v) in W)
    Va = Dict(k => zeros(size(v)) for (k, v) in W)

    ModelState(W, G, M, Va)
end

"""
    copy_model(ms; reset_optimizer=false) -> ModelState

Deep-copy model. If reset_optimizer=true, zero out Adam moments
(use this for offspring in evolutionary experiments).
"""
function copy_model(ms::ModelState; reset_optimizer::Bool=false)
    W_new = Dict(k => copy(v) for (k, v) in ms.W)
    G_new = Dict(k => zeros(size(v)) for (k, v) in ms.W)
    if reset_optimizer
        M_new  = Dict(k => zeros(size(v)) for (k, v) in ms.W)
        Va_new = Dict(k => zeros(size(v)) for (k, v) in ms.W)
    else
        M_new  = Dict(k => copy(v) for (k, v) in ms.M)
        Va_new = Dict(k => copy(v) for (k, v) in ms.Va)
    end
    ModelState(W_new, G_new, M_new, Va_new)
end

function n_params(ms::ModelState)
    sum(length(v) for v in values(ms.W))
end

"""
    mutate_weights!(ms, sigma; rng, reset_optimizer=true) -> ModelState

Add Gaussian noise to weights. Resets Adam moments by default since
the old moments are stale after perturbation.
"""
function mutate_weights!(ms::ModelState, sigma::Float64;
                         rng::AbstractRNG=Random.default_rng(),
                         reset_optimizer::Bool=true)
    for k in keys(ms.W)
        ms.W[k] .+= sigma .* randn(rng, size(ms.W[k]))
    end
    if reset_optimizer
        for k in keys(ms.M)
            fill!(ms.M[k], 0.0)
            fill!(ms.Va[k], 0.0)
        end
    end
    ms
end

function zero_grad!(ms::ModelState)
    for k in keys(ms.G)
        fill!(ms.G[k], 0.0)
    end
end

# ── Tensor type (unchanged from microgpt v2) ─────────────────────────────────

mutable struct Tensor
    data  :: Vector{Float64}
    grad  :: Vector{Float64}
    back! :: Function
    prev  :: Vector{Tensor}
end
# No copy — callers already materialize fresh vectors
Tensor(d::Vector) = Tensor(d, zeros(length(d)), () -> nothing, Tensor[])

function backward!(root::Tensor)
    order, seen = Tensor[], IdDict{Tensor, Nothing}()
    visit!(v) = haskey(seen, v) ? nothing :
                    (seen[v] = nothing; foreach(visit!, v.prev); push!(order, v))
    visit!(root)
    root.grad .= 1.0
    foreach(v -> v.back!(), Iterators.reverse(order))
end

# ── Primitive ops (Tensor-based, for training) ────────────────────────────────

function t_embed(W::Matrix, G::Matrix, i::Int)
    out = Tensor(W[i, :])
    out.back! = () -> @views G[i, :] .+= out.grad
    out
end

function t_linear(x::Tensor, W::Matrix, G::Matrix)
    out = Tensor(W * x.data)
    out.prev = [x]
    out.back! = () -> (x.grad .+= W' * out.grad;
                       G      .+= out.grad * x.data')
    out
end

function t_add(a::Tensor, b::Tensor)
    out = Tensor(a.data .+ b.data)
    out.prev = [a, b]
    out.back! = () -> (a.grad .+= out.grad; b.grad .+= out.grad)
    out
end

function t_relu(a::Tensor)
    out = Tensor(max.(0.0, a.data))
    out.prev = [a]
    out.back! = () -> a.grad .+= out.grad .* (a.data .> 0)
    out
end

function t_rmsnorm(x::Tensor)
    rms = sqrt(dot(x.data, x.data) / length(x.data) + 1e-5)
    y = x.data ./ rms
    out = Tensor(y)
    out.prev = [x]
    out.back! = () -> x.grad .+= (out.grad .- y .* (dot(out.grad, y) / length(y))) ./ rms
    out
end

function t_mha(q::Tensor, Kc::Vector{Tensor}, Vc::Vector{Tensor}, n_head::Int)
    T   = length(Kc)
    d   = length(q.data) ÷ n_head
    scl = 1.0 / sqrt(d)

    saves = map(0:(n_head - 1)) do h
        r  = (h * d + 1):((h + 1) * d)
        ls = @views [dot(q.data[r], Kc[t].data[r]) for t in 1:T] .* scl
        ls .-= maximum(ls)
        e = exp.(ls)
        p = e ./ sum(e)
        ho = @views sum(p[t] .* Vc[t].data[r] for t in 1:T)
        (r, p, ho)
    end

    out      = Tensor(vcat((ho for (_, _, ho) in saves)...))
    out.prev = vcat([q], Kc, Vc)
    out.back! = () -> @views begin
        for (r, p, _) in saves
            go = out.grad[r]
            dp = [dot(Vc[t].data[r], go) for t in 1:T]
            dl = p .* (dp .- dot(p, dp))
            q.grad[r] .+= scl .* sum(dl[t] .* Kc[t].data[r] for t in 1:T)
            for t in 1:T
                Kc[t].grad[r] .+= scl .* dl[t] .* q.data[r]
                Vc[t].grad[r] .+=        p[t]  .* go
            end
        end
    end
    out
end

function t_ce_loss(logits::Tensor, target::Int)
    mx = maximum(logits.data)
    z = logits.data .- mx
    e = exp.(z)
    p = e ./ sum(e)
    out = Tensor([-log(p[target])])
    out.prev = [logits]
    out.back! = () -> (g = copy(p); g[target] -= 1.0; logits.grad .+= out.grad[1] .* g)
    out
end

function t_mean_loss(losses::Vector{Tensor})
    n   = length(losses)
    out = Tensor([sum(l.data[1] for l in losses) / n])
    out.prev = losses
    out.back! = () -> for l in losses; l.grad[1] += out.grad[1] / n; end
    out
end

# ── GPT forward (with gradients, for training) ───────────────────────────────

function gpt(tid::Int, pos::Int, Kc, Vc, ms::ModelState, cfg::ModelConfig)
    W, G = ms.W, ms.G
    x = t_rmsnorm(t_add(t_embed(W["wte"], G["wte"], tid),
                         t_embed(W["wpe"], G["wpe"], pos)))
    for l in 0:(cfg.n_layer - 1)
        sl = string(l)
        r  = x
        x  = t_rmsnorm(x)
        q  = t_linear(x, W[sl * ".attn_wq"], G[sl * ".attn_wq"])
        k  = t_linear(x, W[sl * ".attn_wk"], G[sl * ".attn_wk"])
        v  = t_linear(x, W[sl * ".attn_wv"], G[sl * ".attn_wv"])
        push!(Kc[l + 1], k)
        push!(Vc[l + 1], v)
        x  = t_add(t_linear(t_mha(q, Kc[l + 1], Vc[l + 1], cfg.n_head),
                             W[sl * ".attn_wo"], G[sl * ".attn_wo"]), r)
        r  = x
        x  = t_rmsnorm(x)
        x  = t_relu(t_linear(x, W[sl * ".mlp_fc1"], G[sl * ".mlp_fc1"]))
        x  = t_add(t_linear(x, W[sl * ".mlp_fc2"], G[sl * ".mlp_fc2"]), r)
    end
    t_linear(x, W["lm_head"], G["lm_head"])
end

# ── Forward-only workspace (pre-allocated buffers for evaluation) ─────────────

"""
    ForwardWorkspace

Pre-allocated buffers for the forward-only evaluation path.
Eliminates per-call vector allocations in gpt_forward.
"""
struct ForwardWorkspace
    x      :: Vector{Float64}   # n_embd — current activations
    r      :: Vector{Float64}   # n_embd — residual connection
    q      :: Vector{Float64}   # n_embd — query vector
    attn   :: Vector{Float64}   # n_embd — attention output
    mlp    :: Vector{Float64}   # mlp_mult * n_embd — MLP hidden
    scores :: Vector{Float64}   # block_size — attention scores per head
end

function ForwardWorkspace(cfg::ModelConfig)
    ne = cfg.n_embd
    ForwardWorkspace(
        Vector{Float64}(undef, ne),
        Vector{Float64}(undef, ne),
        Vector{Float64}(undef, ne),
        Vector{Float64}(undef, ne),
        Vector{Float64}(undef, cfg.mlp_mult * ne),
        Vector{Float64}(undef, cfg.block_size))
end

# ── GPT forward-only (no Tensor, no gradients — for evaluation) ──────────────

"""
    _rmsnorm!(x) -> x

In-place RMS normalization. Uses dot(x,x) to avoid allocating x.^2.
"""
function _rmsnorm!(x::Vector{Float64})
    s = 1.0 / sqrt(dot(x, x) / length(x) + 1e-5)
    @inbounds @simd for i in eachindex(x)
        x[i] *= s
    end
    x
end

"""
    _mha_forward!(out, q, Kc, Vc, n_head)

In-place multi-head attention. Writes result into `out`.
Uses scalar indexing to avoid slice copies.
"""
function _mha_forward!(out::Vector{Float64}, q::Vector{Float64},
                       Kc::Vector{Vector{Float64}},
                       Vc::Vector{Vector{Float64}}, n_head::Int,
                       scores::Vector{Float64})
    T   = length(Kc)
    d   = length(q) ÷ n_head
    scl = 1.0 / sqrt(d)
    fill!(out, 0.0)

    for h in 0:(n_head - 1)
        lo = h * d + 1
        hi = (h + 1) * d

        # Compute attention scores into pre-allocated buffer
        mx = -Inf
        @inbounds for t in 1:T
            s = 0.0
            for j in lo:hi
                s += q[j] * Kc[t][j]
            end
            scores[t] = s * scl
            if scores[t] > mx
                mx = scores[t]
            end
        end

        # Stable softmax in-place on scores[1:T]
        s_sum = 0.0
        @inbounds for t in 1:T
            scores[t] = exp(scores[t] - mx)
            s_sum += scores[t]
        end
        inv_sum = 1.0 / s_sum

        # Weighted sum of values
        @inbounds for t in 1:T
            p_t = scores[t] * inv_sum
            for j in lo:hi
                out[j] += p_t * Vc[t][j]
            end
        end
    end
    out
end

"""
    gpt_forward(tid, pos, Kc, Vc, W, cfg, ws) -> Vector{Float64}

Forward-only GPT pass using pre-allocated workspace.
Returns logits vector (newly allocated — the only allocation per call
besides KV cache entries).
"""
function gpt_forward(tid::Int, pos::Int, Kc, Vc,
                     W::Dict{String,Matrix{Float64}},
                     cfg::ModelConfig, ws::ForwardWorkspace)
    ne = cfg.n_embd
    wte = W["wte"]
    wpe = W["wpe"]

    # Embedding: x = rmsnorm(wte[tid,:] + wpe[pos,:])
    @inbounds for i in 1:ne
        ws.x[i] = wte[tid, i] + wpe[pos, i]
    end
    _rmsnorm!(ws.x)

    for l in 0:(cfg.n_layer - 1)
        sl = string(l)
        wq  = W[sl * ".attn_wq"]
        wk  = W[sl * ".attn_wk"]
        wv  = W[sl * ".attn_wv"]
        wo  = W[sl * ".attn_wo"]
        fc1 = W[sl * ".mlp_fc1"]
        fc2 = W[sl * ".mlp_fc2"]

        # Save residual, normalize
        copyto!(ws.r, ws.x)
        _rmsnorm!(ws.x)

        # Attention: q uses workspace, k/v allocate (go to cache)
        mul!(ws.q, wq, ws.x)
        k = wk * ws.x
        v = wv * ws.x
        push!(Kc[l + 1], k)
        push!(Vc[l + 1], v)

        _mha_forward!(ws.attn, ws.q, Kc[l + 1], Vc[l + 1], cfg.n_head, ws.scores)
        mul!(ws.x, wo, ws.attn)
        ws.x .+= ws.r

        # MLP with residual
        copyto!(ws.r, ws.x)
        _rmsnorm!(ws.x)
        mul!(ws.mlp, fc1, ws.x)
        @inbounds @simd for i in eachindex(ws.mlp)
            ws.mlp[i] = max(0.0, ws.mlp[i])
        end
        mul!(ws.x, fc2, ws.mlp)
        ws.x .+= ws.r
    end

    W["lm_head"] * ws.x
end

"""
    softmax!(out, logits) -> out

In-place numerically stable softmax. Writes into `out`.
"""
function softmax!(out::Vector{Float64}, logits::Vector{Float64})
    mx = maximum(logits)
    s = 0.0
    @inbounds for i in eachindex(logits)
        out[i] = exp(logits[i] - mx)
        s += out[i]
    end
    inv_s = 1.0 / s
    @inbounds @simd for i in eachindex(out)
        out[i] *= inv_s
    end
    out
end

"""
    softmax(logits) -> Vector{Float64}

Allocating softmax (convenience wrapper).
"""
function softmax(logits::Vector{Float64})
    softmax!(similar(logits), logits)
end

"""
    output_probs(W, tokens, cfg) -> Vector{Vector{Float64}}

Run forward-only pass on a token sequence. Returns the softmax probability
vector for each position 1:length(tokens)-1 (predicting the next token).
"""
function output_probs(W::Dict{String,Matrix{Float64}}, tokens::Vector{Int},
                      cfg::ModelConfig)
    n = min(cfg.block_size, length(tokens) - 1)
    Kc = [Vector{Float64}[] for _ in 1:cfg.n_layer]
    Vc = [Vector{Float64}[] for _ in 1:cfg.n_layer]
    ws = ForwardWorkspace(cfg)
    probs = Vector{Vector{Float64}}(undef, n)
    for t in 1:n
        logits = gpt_forward(tokens[t], t, Kc, Vc, W, cfg, ws)
        probs[t] = softmax(logits)
    end
    probs
end

"""
    last_position_distribution(W, tokens, cfg) -> Vector{Float64}

Run forward-only pass and return ONLY the softmax distribution at the
last position.
"""
function last_position_distribution(W::Dict{String,Matrix{Float64}},
                                     tokens::Vector{Int}, cfg::ModelConfig)
    n = min(cfg.block_size, length(tokens) - 1)
    n < 1 && return Float64[]
    Kc = [Vector{Float64}[] for _ in 1:cfg.n_layer]
    Vc = [Vector{Float64}[] for _ in 1:cfg.n_layer]
    ws = ForwardWorkspace(cfg)
    probs = Vector{Float64}(undef, cfg.vocab_size)
    local logits
    for t in 1:n
        logits = gpt_forward(tokens[t], t, Kc, Vc, W, cfg, ws)
    end
    softmax!(probs, logits)
end

"""
    sequence_nll(W, tokens, cfg) -> (total_nll, n_tokens)

Compute total negative log-likelihood over a token sequence.
Uses scalar log-softmax to avoid allocating probability vectors.
"""
function sequence_nll(W::Dict{String,Matrix{Float64}},
                      tokens::Vector{Int}, cfg::ModelConfig)
    n = min(cfg.block_size, length(tokens) - 1)
    n < 1 && return (0.0, 0)
    Kc = [Vector{Float64}[] for _ in 1:cfg.n_layer]
    Vc = [Vector{Float64}[] for _ in 1:cfg.n_layer]
    ws = ForwardWorkspace(cfg)
    total = 0.0
    for t in 1:n
        logits = gpt_forward(tokens[t], t, Kc, Vc, W, cfg, ws)
        target = tokens[t + 1]
        # Numerically stable log-softmax without allocating
        mx = maximum(logits)
        lse = 0.0
        @inbounds for i in eachindex(logits)
            lse += exp(logits[i] - mx)
        end
        lse = log(lse)
        total -= (logits[target] - mx) - lse
    end
    (total, n)
end

# ── Training utilities ────────────────────────────────────────────────────────

"""
    _adam_update!(W_k, M_k, V_k, G_k, lr_t, beta1, beta2, bc1, bc2, eps, wd_lr)

Fused Adam update for a single weight matrix. Avoids allocating temporary
m_hat, v_hat arrays by computing element-wise in a single pass.
"""
function _adam_update!(W_k::Matrix{Float64}, M_k::Matrix{Float64},
                       V_k::Matrix{Float64}, G_k::Matrix{Float64},
                       lr_t::Float64, beta1::Float64, beta2::Float64,
                       bc1::Float64, bc2::Float64, eps::Float64,
                       wd_lr::Float64)
    @inbounds @simd for i in eachindex(W_k)
        M_k[i] = beta1 * M_k[i] + (1 - beta1) * G_k[i]
        V_k[i] = beta2 * V_k[i] + (1 - beta2) * G_k[i] * G_k[i]
        m_hat = M_k[i] / bc1
        v_hat = V_k[i] / bc2
        W_k[i] -= lr_t * m_hat / (sqrt(v_hat) + eps) + wd_lr * W_k[i]
    end
end

"""
    train_step!(ms, tokens, step, cfg; weight_decay=0.0) -> Float64

One gradient step (Adam). Returns the loss value.
"""
function train_step!(ms::ModelState, tokens::Vector{Int}, step::Int,
                     cfg::ModelConfig; weight_decay::Float64=0.0)
    zero_grad!(ms)

    n  = min(cfg.block_size, length(tokens) - 1)
    Kc = [Tensor[] for _ in 1:cfg.n_layer]
    Vc = [Tensor[] for _ in 1:cfg.n_layer]

    losses = [t_ce_loss(gpt(tokens[t], t, Kc, Vc, ms, cfg), tokens[t + 1])
              for t in 1:n]
    loss = t_mean_loss(losses)

    backward!(loss)

    lr_t = cfg.lr * max(0.0, 1 - step / 1000)
    bc1 = 1 - cfg.beta1^step
    bc2 = 1 - cfg.beta2^step
    wd_lr = lr_t * weight_decay
    for k in keys(ms.W)
        _adam_update!(ms.W[k], ms.M[k], ms.Va[k], ms.G[k],
                      lr_t, cfg.beta1, cfg.beta2, bc1, bc2, cfg.eps, wd_lr)
    end

    loss.data[1]
end

"""
    train_step_lr!(ms, tokens, step, cfg, lr_t; weight_decay=0.0) -> Float64

Like train_step! but with an explicit learning rate (no built-in schedule).
"""
function train_step_lr!(ms::ModelState, tokens::Vector{Int}, step::Int,
                        cfg::ModelConfig, lr_t::Float64;
                        weight_decay::Float64=0.0)
    zero_grad!(ms)

    n  = min(cfg.block_size, length(tokens) - 1)
    Kc = [Tensor[] for _ in 1:cfg.n_layer]
    Vc = [Tensor[] for _ in 1:cfg.n_layer]

    losses = [t_ce_loss(gpt(tokens[t], t, Kc, Vc, ms, cfg), tokens[t + 1])
              for t in 1:n]
    loss = t_mean_loss(losses)

    backward!(loss)

    bc1 = 1 - cfg.beta1^step
    bc2 = 1 - cfg.beta2^step
    wd_lr = lr_t * weight_decay
    for k in keys(ms.W)
        _adam_update!(ms.W[k], ms.M[k], ms.Va[k], ms.G[k],
                      lr_t, cfg.beta1, cfg.beta2, bc1, bc2, cfg.eps, wd_lr)
    end

    loss.data[1]
end
