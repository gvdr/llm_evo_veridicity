"""
Generate synthetic name data with known phonotactic patterns.
Each "language" has a distinct character distribution and syllable structure.
This gives us ground-truth separation: languages with different generators
should be separated, languages with similar generators (spanish/italian)
should be hard to separate.

Run:  julia scripts/generate_synthetic_names.jl
"""

using Random

const DATA_DIR = joinpath(@__DIR__, "..", "data", "names")
const N_NAMES = 500

# ── Phonotactic generators ────────────────────────────────────────────────────

# Each generator produces one name as a string

function gen_english(rng)
    # Consonant clusters, varied endings, "th", "sh", "ck"
    onsets = ["b", "br", "ch", "cl", "d", "dr", "f", "fl", "fr", "g", "gr",
              "h", "j", "k", "l", "m", "n", "p", "pr", "r", "s", "sh", "sl",
              "sm", "sn", "sp", "st", "str", "sw", "t", "th", "tr", "v", "w"]
    vowels = ["a", "e", "i", "o", "u", "ai", "ea", "ee", "oo", "ou"]
    codas  = ["", "b", "ck", "d", "f", "g", "k", "l", "ll", "m", "n", "nd",
              "ng", "nk", "nt", "p", "r", "rd", "rn", "rt", "s", "sh", "ss",
              "st", "t", "th", "x"]

    n_syl = rand(rng, 1:3)
    parts = String[]
    for _ in 1:n_syl
        push!(parts, onsets[rand(rng, 1:end)])
        push!(parts, vowels[rand(rng, 1:end)])
        if rand(rng) < 0.5
            push!(parts, codas[rand(rng, 1:end)])
        end
    end
    join(parts)
end

function gen_japanese(rng)
    # CV syllable structure, vowel endings, no clusters
    consonants = ["k", "s", "t", "n", "h", "m", "y", "r", "w", "g", "z",
                  "d", "b", "p"]
    vowels = ["a", "i", "u", "e", "o"]
    special = ["n"]  # syllabic n

    n_syl = rand(rng, 2:4)
    parts = String[]
    for i in 1:n_syl
        if rand(rng) < 0.15 && i > 1
            push!(parts, "n")  # syllabic n
        end
        if rand(rng) < 0.8
            push!(parts, consonants[rand(rng, 1:end)])
        end
        push!(parts, vowels[rand(rng, 1:end)])
    end
    join(parts)
end

function gen_arabic(rng)
    # Triconsonantal roots, strong consonants, -a/-i/-ah endings
    consonants = ["b", "d", "f", "g", "h", "j", "k", "kh", "l", "m", "n",
                  "q", "r", "s", "sh", "t", "th", "w", "y", "z", "dh", "gh"]
    vowels = ["a", "i", "u"]
    patterns = [
        # CaCiC, CaCuC, CaCaC, etc
        (c, v1, c2, v2, c3) -> c * v1 * c2 * v2 * c3,
        (c, v1, c2, v2, c3) -> c * v1 * c2 * v2 * c3 * "ah",
        (c, v1, c2, v2, c3) -> c * v1 * c2 * c3 * v2,
        (c, v1, c2, v2, c3) -> c * v1 * c2 * v2 * c3 * "i",
    ]

    c1 = consonants[rand(rng, 1:end)]
    c2 = consonants[rand(rng, 1:end)]
    c3 = consonants[rand(rng, 1:end)]
    v1 = vowels[rand(rng, 1:end)]
    v2 = vowels[rand(rng, 1:end)]
    pat = patterns[rand(rng, 1:end)]
    pat(c1, v1, c2, v2, c3)
end

function gen_finnish(rng)
    # Double vowels, k/t/p heavy, -nen/-la/-ri endings
    onsets = ["h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v"]
    vowels = ["a", "e", "i", "o", "u", "aa", "ee", "ii", "oo", "uu",
              "ai", "ei", "oi", "ui", "au"]
    endings = ["nen", "la", "ri", "ki", "ti", "na", "ta", "ka", "kka",
               "tti", "ppi", "li", "ni", "ma", "va"]

    n_syl = rand(rng, 1:2)
    parts = String[]
    for _ in 1:n_syl
        push!(parts, onsets[rand(rng, 1:end)])
        push!(parts, vowels[rand(rng, 1:end)])
    end
    push!(parts, endings[rand(rng, 1:end)])
    join(parts)
end

function gen_swahili(rng)
    # M-/N- prefixes, vowel harmony, -wa/-ji/-ni endings
    prefixes = ["m", "n", "ki", "bi", "u", "ma", "wa", "mi"]
    consonants = ["b", "ch", "d", "f", "g", "h", "j", "k", "l", "m",
                  "n", "ng", "ny", "p", "r", "s", "sh", "t", "v", "w", "z"]
    vowels = ["a", "e", "i", "o", "u"]
    endings = ["a", "i", "u", "wa", "ji", "ni", "si", "zi", "li", "ki"]

    parts = String[]
    if rand(rng) < 0.4
        push!(parts, prefixes[rand(rng, 1:end)])
    end
    n_syl = rand(rng, 2:3)
    for _ in 1:n_syl
        push!(parts, consonants[rand(rng, 1:end)])
        push!(parts, vowels[rand(rng, 1:end)])
    end
    push!(parts, endings[rand(rng, 1:end)])
    join(parts)
end

function gen_spanish(rng)
    # Romance patterns, -o/-a/-ez endings, similar to Italian
    onsets = ["b", "c", "ch", "d", "f", "g", "gr", "h", "j", "l", "ll",
              "m", "n", "p", "pr", "r", "rr", "s", "t", "v", "z"]
    vowels = ["a", "e", "i", "o", "u"]
    endings = ["o", "a", "ez", "es", "os", "as", "io", "ia", "al",
               "on", "an", "ero", "era", "ino", "ina"]

    n_syl = rand(rng, 1:2)
    parts = String[]
    for _ in 1:n_syl
        push!(parts, onsets[rand(rng, 1:end)])
        push!(parts, vowels[rand(rng, 1:end)])
    end
    push!(parts, endings[rand(rng, 1:end)])
    join(parts)
end

function gen_italian(rng)
    # Very similar to Spanish: Romance patterns, -o/-a/-i endings
    onsets = ["b", "c", "ch", "d", "f", "g", "gr", "l", "m", "n",
              "p", "pr", "r", "s", "sc", "t", "v", "z"]
    vowels = ["a", "e", "i", "o", "u"]
    endings = ["o", "a", "i", "e", "io", "ia", "ino", "ina", "ello",
               "ella", "etti", "otti", "acci", "ucci", "one", "oni"]

    n_syl = rand(rng, 1:2)
    parts = String[]
    for _ in 1:n_syl
        push!(parts, onsets[rand(rng, 1:end)])
        push!(parts, vowels[rand(rng, 1:end)])
    end
    push!(parts, endings[rand(rng, 1:end)])
    join(parts)
end

# ── Generate and write ────────────────────────────────────────────────────────

function generate_names(gen_fn, n::Int, rng; min_len=3, max_len=12)
    names = Set{String}()
    attempts = 0
    while length(names) < n && attempts < n * 10
        name = gen_fn(rng)
        attempts += 1
        length(name) < min_len && continue
        length(name) > max_len && continue
        # Only lowercase ascii
        all(c -> 'a' <= c <= 'z', name) || continue
        push!(names, name)
    end
    sort(collect(names))
end

function main()
    mkpath(DATA_DIR)

    generators = [
        ("english",  gen_english,  42),
        ("japanese", gen_japanese, 43),
        ("arabic",   gen_arabic,   44),
        ("finnish",  gen_finnish,  45),
        ("swahili",  gen_swahili,  46),
        ("spanish",  gen_spanish,  47),
        ("italian",  gen_italian,  48),
    ]

    for (lang, gen_fn, seed) in generators
        rng = MersenneTwister(seed)
        names = generate_names(gen_fn, N_NAMES, rng)
        path = joinpath(DATA_DIR, "names_" * lang * ".txt")
        open(path, "w") do io
            for name in names
                println(io, name)
            end
        end
        println(lang * ": " * string(length(names)) * " names -> " * path)

        # Show first 5
        for i in 1:min(5, length(names))
            println("  " * names[i])
        end
    end
end

main()
