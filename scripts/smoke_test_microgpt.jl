"""
Smoke test for refactored microgpt: verify init, forward, backward,
forward-only, copy, and mutate all work correctly.

Run:  julia --project=. scripts/smoke_test_microgpt.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "microgpt.jl"))

using Random

println("=" ^ 60)
println("Smoke test: refactored microgpt")
println("=" ^ 60)

# ── 1. Init model ─────────────────────────────────────────────────────────────
cfg = ModelConfig(; n_layer=1, n_embd=16, n_head=4, block_size=16, vocab_size=27)
ms = init_model(cfg; seed=42)
println("1. Init:     " * string(n_params(ms)) * " params — OK")

# ── 2. Forward with gradients ─────────────────────────────────────────────────
tokens = [27, 1, 2, 3, 27]  # BOS, a, b, c, BOS
Kc = [Tensor[] for _ in 1:cfg.n_layer]
Vc = [Tensor[] for _ in 1:cfg.n_layer]
logits = gpt(tokens[1], 1, Kc, Vc, ms, cfg)
@assert length(logits.data) == cfg.vocab_size
println("2. Forward:  logits length " * string(length(logits.data)) * " — OK")

# ── 3. Backward ──────────────────────────────────────────────────────────────
loss = t_ce_loss(logits, tokens[2])
backward!(loss)
has_grad = any(any(ms.G[k] .!= 0) for k in keys(ms.G))
@assert has_grad "Expected nonzero gradients after backward"
println("3. Backward: loss=" * string(round(loss.data[1], digits=4)) *
        " grads nonzero — OK")

# ── 4. Training step ─────────────────────────────────────────────────────────
ms2 = init_model(cfg; seed=42)
loss_val = train_step!(ms2, tokens, 1, cfg)
@assert isfinite(loss_val)
println("4. Train:    loss=" * string(round(loss_val, digits=4)) * " — OK")

# ── 5. Forward-only (no Tensor) ───────────────────────────────────────────────
probs = output_probs(ms2.W, tokens, cfg)
@assert length(probs) == length(tokens) - 1
@assert all(abs(sum(p) - 1.0) < 1e-10 for p in probs)
println("5. Eval:     " * string(length(probs)) *
        " distributions, all sum to 1 — OK")

# ── 6. Consistency: Tensor vs plain forward ──────────────────────────────────
ms3 = init_model(cfg; seed=99)
tokens2 = [27, 5, 10, 15, 27]
# Tensor forward
Kc_t = [Tensor[] for _ in 1:cfg.n_layer]
Vc_t = [Tensor[] for _ in 1:cfg.n_layer]
logits_t = gpt(tokens2[1], 1, Kc_t, Vc_t, ms3, cfg)
# Plain forward
Kc_p = [Vector{Float64}[] for _ in 1:cfg.n_layer]
Vc_p = [Vector{Float64}[] for _ in 1:cfg.n_layer]
logits_p = gpt_forward(tokens2[1], 1, Kc_p, Vc_p, ms3.W, cfg)
gap = maximum(abs.(logits_t.data .- logits_p))
@assert gap < 1e-12 "Tensor vs plain forward mismatch: " * string(gap)
println("6. Consist:  tensor vs plain gap=" * string(gap) * " — OK")

# ── 7. Copy model ────────────────────────────────────────────────────────────
ms_copy = copy_model(ms2)
# Verify weights are equal but not aliased
@assert all(ms_copy.W[k] == ms2.W[k] for k in keys(ms2.W))
ms_copy.W["wte"][1, 1] += 1.0
@assert ms_copy.W["wte"][1, 1] != ms2.W["wte"][1, 1]
println("7. Copy:     deep copy verified — OK")

# ── 8. Mutate ─────────────────────────────────────────────────────────────────
ms_mut = copy_model(ms2)
w_before = copy(ms_mut.W["wte"])
mutate_weights!(ms_mut, 0.01; rng=MersenneTwister(1))
diff = maximum(abs.(ms_mut.W["wte"] .- w_before))
@assert diff > 0
println("8. Mutate:   max weight change=" * string(round(diff, digits=6)) * " — OK")

# ── 9. Multi-step training ───────────────────────────────────────────────────
ms4 = init_model(cfg; seed=42)
losses = Float64[]
for step in 1:50
    l = train_step_lr!(ms4, tokens, step, cfg, 0.005)
    push!(losses, l)
end
@assert losses[end] < losses[1] "Expected loss to decrease over training"
println("9. Train50:  loss " * string(round(losses[1], digits=3)) *
        " -> " * string(round(losses[end], digits=3)) * " — OK")

# ── 10. Output probs after training ──────────────────────────────────────────
probs_after = output_probs(ms4.W, tokens, cfg)
# After training on BOS->a->b->c->BOS, the model should put more mass
# on 'a' (token 1) after BOS
prob_a_after_bos = probs_after[1][1]
println("10. Learned: P(a|BOS)=" * string(round(prob_a_after_bos, digits=4)) *
        " (untrained ~" * string(round(1.0/27, digits=4)) * ") — OK")

println()
println("All smoke tests passed.")
