"""Test all 7 fixes."""

include(joinpath(@__DIR__, "..", "src", "microgpt.jl"))
include(joinpath(@__DIR__, "..", "src", "data.jl"))
include(joinpath(@__DIR__, "..", "src", "measurement.jl"))
include(joinpath(@__DIR__, "..", "src", "population.jl"))

using Random

passed = 0
failed = 0

function check(name, cond)
    global passed, failed
    if cond
        println("  PASS: " * name)
        passed += 1
    else
        println("  FAIL: " * name)
        failed += 1
    end
end

println("=== Testing fixes ===")

# Fix 1: chunk_text forward progress
println("\n1. chunk_text safety")
text = repeat("a b c d e f g h ", 1000)
chunks = chunk_text(text, 30; overlap=5)
check("chunk_text terminates", length(chunks) < 2000)
check("chunk_text non-empty", length(chunks) > 100)
# Pathological case: very short words
text2 = repeat("a b c ", 5000)
chunks2 = chunk_text(text2, 30; overlap=5)
check("short-word chunking terminates", length(chunks2) < 10000)

# Fix 2: token-level prefixes
println("\n2. Token-level prefixes")
tok = Tokenizer(collect('a':'z'))
chunks_with_spaces = ["hello world foo bar", "the quick brown fox"]
prefs = make_prefixes(chunks_with_spaces, tok, [3, 5]; max_per_length=10)
# With token-level prefixes, length 3 means [BOS, h, e, l] or [BOS, t, h, e]
check("prefix length 3 has 4 tokens", all(length(s) == 4 for s in prefs[3]))
check("prefix length 5 has 6 tokens", all(length(s) == 6 for s in prefs[5]))
# Spaces should NOT appear in tokens (tokenizer only has a-z)
all_tok_ids = vcat(prefs[3]...)
check("no space tokens", all(id -> id <= 27, all_tok_ids))  # 26 chars + BOS

# Fix 3: copy_model reset_optimizer
println("\n3. Optimizer reset")
cfg = ModelConfig(; vocab_size=27, n_layer=1, n_embd=16, n_head=4, block_size=16)
ms = init_model(cfg)
# Do a training step to populate Adam moments
tokens = [27, 1, 12, 5, 27]
train_step_lr!(ms, tokens, 1, cfg, 0.01)
check("moments populated", any(ms.M["wte"] .!= 0))
ms_copy = copy_model(ms; reset_optimizer=true)
check("copy weights preserved", ms_copy.W["wte"] == ms.W["wte"])
check("copy moments zeroed", all(ms_copy.M["wte"] .== 0))
check("copy va zeroed", all(ms_copy.Va["wte"] .== 0))

# Fix 4: token-weighted CE
println("\n4. Token-weighted CE")
cfg2 = ModelConfig(; vocab_size=27, n_layer=1, n_embd=16, n_head=4, block_size=16)
ms2 = init_model(cfg2)
short_seq = [27, 1, 2, 27]     # 3 tokens
long_seq  = [27, 1, 2, 3, 4, 5, 6, 7, 8, 27]  # 10 tokens
# Token-weighted: long seq contributes more
ce_tw = mean_cross_entropy(ms2.W, [short_seq, long_seq], cfg2; token_weighted=true)
# Sequence-weighted: equal weight
ce_sw = mean_cross_entropy(ms2.W, [short_seq, long_seq], cfg2; token_weighted=false)
check("token vs seq weighting differ", abs(ce_tw - ce_sw) > 1e-6)

# Fix 5: balanced evaluation
println("\n5. Balanced population eval")
# Create imbalanced token sets
toks_en = [tokenize(tok, "hello"), tokenize(tok, "world")]
toks_fr = [tokenize(tok, "bonjour")]  # fewer sequences
eval_by_lang = Dict("en" => toks_en, "fr" => toks_fr)
ms3 = init_model(cfg)
ce_bal = mean_cross_entropy_balanced(ms3.W, eval_by_lang, cfg)
check("balanced CE computed", ce_bal > 0)

# Fix 6: efficient eval (sequence_nll)
println("\n6. Efficient eval")
ms4 = init_model(cfg)
tokens2 = [27, 1, 12, 5, 3, 27]
nll, nt = sequence_nll(ms4.W, tokens2, cfg)
ce_old = cross_entropy_loss(ms4.W, tokens2, cfg)
check("sequence_nll matches CE", abs(nll / nt - ce_old) < 1e-10)

dist = last_position_distribution(ms4.W, tokens2, cfg)
check("last_position_distribution sums to 1", abs(sum(dist) - 1.0) < 1e-10)

# Fix 7: train_step! no negative LR
println("\n7. train_step! LR clamp")
ms5 = init_model(cfg)
w_before = copy(ms5.W["wte"])
# Step 2000 — would have negative LR without the fix
train_step!(ms5, tokens, 2000, cfg)
# With clamped LR=0, weights should not change (no gradient applied)
check("step 2000 no weight change", ms5.W["wte"] == w_before)

# Progress reporting
println("\n8. Progress reporting")
msg = progress_str(50, 100, "test"; start_time=time() - 10.0)
check("progress_str works", occursin("50/100", msg) && occursin("ETA", msg))

println("\n=== " * string(passed) * " passed, " * string(failed) * " failed ===")
