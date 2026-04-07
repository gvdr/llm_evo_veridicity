"""Quick pipeline test: data loading + model + training + evaluation."""

include(joinpath(@__DIR__, "..", "src", "microgpt.jl"))
include(joinpath(@__DIR__, "..", "src", "data.jl"))
include(joinpath(@__DIR__, "..", "src", "measurement.jl"))

using Random

println("=== Full pipeline test ===")
println("RSS after includes: " * string(Sys.maxrss() ÷ 1024^2) * " MB")

DATA_DIR = joinpath(@__DIR__, "..", "data", "alice")
FILE_MAP = Dict(
    "english" => "alice_english.txt", "french" => "alice_french.txt",
    "german" => "alice_german.txt", "italian" => "alice_italian.txt",
    "finnish" => "alice_finnish.txt",
)

chunks_by_lang, tok = load_text_ecology(DATA_DIR, FILE_MAP; chunk_size=30, overlap=5)
println("RSS after load: " * string(Sys.maxrss() ÷ 1024^2) * " MB")
println("vocab: " * string(tok.vocab_size))
for lang in ["english", "french", "german", "italian", "finnish"]
    println("  " * lang * ": " * string(length(chunks_by_lang[lang])) * " chunks")
end

cfg = ModelConfig(; vocab_size=tok.vocab_size, n_layer=2, n_embd=32,
                   n_head=4, block_size=32)
ms = init_model(cfg)
println("Model: " * string(n_params(ms)) * " params")

# Quick train test (10 steps)
rng = MersenneTwister(42)
train_toks = [tokenize_chunk(tok, c) for c in chunks_by_lang["english"][1:100]]
for step in 1:10
    idx = rand(rng, 1:length(train_toks))
    l = train_step_lr!(ms, train_toks[idx], step, cfg, 0.008)
    if step == 1 || step == 5 || step == 10
        println("  step " * string(step) * " loss=" * string(round(l, digits=4)) *
                " RSS=" * string(Sys.maxrss() ÷ 1024^2) * "MB")
    end
end

# Eval test: output_probs
probs = output_probs(ms.W, train_toks[1], cfg)
println("output_probs: " * string(length(probs)) * " positions")

# Eval test: behavioural distance
pref_en = make_prefixes(chunks_by_lang["english"][1:50], tok, [3, 5, 8];
                         max_per_length=20, rng=MersenneTwister(99))
pref_fr = make_prefixes(chunks_by_lang["french"][1:50], tok, [3, 5, 8];
                         max_per_length=20, rng=MersenneTwister(99))
d = behavioural_distance(ms.W, pref_en, pref_fr, cfg)
println("D_beh(en, fr) = " * string(round(d, digits=6)))

println("RSS final: " * string(Sys.maxrss() ÷ 1024^2) * " MB")
println("=== PASS ===")
