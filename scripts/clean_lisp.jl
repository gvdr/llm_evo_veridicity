"""
Prepare the bracket-balance corpus from Practical Common Lisp source.

Input:  raw concatenated Lisp source (provided as argument or default path)
Output: data/lisp/lisp_balanced.txt   — chunks with verified balanced parens
        data/lisp/lisp_unbalanced.txt — same chunks with permuted bracket pattern

Each chunk is CHUNK_SIZE characters. For balanced chunks, parentheses are
verified to be properly nested (not just equal counts). For unbalanced
chunks, the bracket characters at the same positions are randomly permuted,
which almost always breaks nesting.

Run: julia --project=. scripts/clean_lisp.jl
"""

using Random

const BASE_DIR = joinpath(@__DIR__, "..", "data", "lisp")
const RAW_PATH = length(ARGS) >= 1 ? ARGS[1] : "/tmp/all_pcl_lisp.txt"
const BALANCED_PATH = joinpath(BASE_DIR, "lisp_balanced.txt")
const UNBALANCED_PATH = joinpath(BASE_DIR, "lisp_unbalanced.txt")

const CHUNK_SIZE = 30
const CHUNK_STRIDE = 5
const KEEP_CHARS = Set("abcdefghijklmnopqrstuvwxyz() ")

function normalize(raw::String)
    out = Char[]
    sizehint!(out, length(raw))
    for ch in lowercase(raw)
        if ch in KEEP_CHARS
            push!(out, ch)
        end
    end
    # Collapse runs of spaces
    result = Char[]
    sizehint!(result, length(out))
    prev_space = false
    for ch in out
        if ch == ' '
            if !prev_space
                push!(result, ch)
            end
            prev_space = true
        else
            push!(result, ch)
            prev_space = false
        end
    end
    String(result)
end

function is_balanced(chunk::AbstractString)
    depth = 0
    for ch in chunk
        if ch == '('
            depth += 1
        elseif ch == ')'
            depth -= 1
            if depth < 0
                return false
            end
        end
    end
    depth == 0
end

function count_brackets(chunk::AbstractString)
    n_open = 0
    n_close = 0
    for ch in chunk
        if ch == '('
            n_open += 1
        elseif ch == ')'
            n_close += 1
        end
    end
    (n_open, n_close)
end

function bracket_positions(chunk::AbstractString)
    positions = Int[]
    for (i, ch) in enumerate(chunk)
        if ch == '(' || ch == ')'
            push!(positions, i)
        end
    end
    positions
end

function permute_brackets(chunk::String, rng::AbstractRNG)
    chars = collect(chunk)
    positions = bracket_positions(chunk)
    isempty(positions) && return chunk

    # Extract bracket chars, shuffle, replace
    bracket_chars = [chars[p] for p in positions]
    shuffle!(rng, bracket_chars)
    for (i, p) in enumerate(positions)
        chars[p] = bracket_chars[i]
    end
    String(chars)
end

function main()
    mkpath(BASE_DIR)

    println("Reading raw Lisp source from: " * RAW_PATH)
    raw = read(RAW_PATH, String)
    println("  raw size: " * string(length(raw)) * " bytes")

    text = normalize(raw)
    println("  normalized size: " * string(length(text)) * " chars")

    # Chunk with overlapping stride
    n_chunks = div(length(text) - CHUNK_SIZE, CHUNK_STRIDE) + 1
    println("  total candidate chunks (size " * string(CHUNK_SIZE) *
            ", stride " * string(CHUNK_STRIDE) * "): " * string(n_chunks))

    # Filter for balanced chunks with at least 2 brackets
    balanced_chunks = String[]
    for i in 1:n_chunks
        start = (i - 1) * CHUNK_STRIDE + 1
        start + CHUNK_SIZE - 1 > length(text) && break
        chunk = text[start:(start + CHUNK_SIZE - 1)]
        n_open, n_close = count_brackets(chunk)
        if n_open >= 1 && n_close >= 1 && is_balanced(chunk)
            push!(balanced_chunks, chunk)
        end
    end

    println("  balanced chunks (>= 2 brackets): " * string(length(balanced_chunks)))

    if isempty(balanced_chunks)
        println("ERROR: no balanced chunks found")
        return
    end

    # Verify every chunk is truly balanced
    n_verified = 0
    for chunk in balanced_chunks
        if !is_balanced(chunk)
            error("Verification failed: chunk is not balanced: " * chunk)
        end
        n_verified += 1
    end
    println("  all " * string(n_verified) * " chunks verified balanced")

    # Report bracket statistics
    total_brackets = 0
    min_brackets = typemax(Int)
    max_brackets = 0
    for chunk in balanced_chunks
        nb = length(bracket_positions(chunk))
        total_brackets += nb
        min_brackets = min(min_brackets, nb)
        max_brackets = max(max_brackets, nb)
    end
    mean_brackets = total_brackets / length(balanced_chunks)
    println("  brackets per chunk: min=" * string(min_brackets) *
            " max=" * string(max_brackets) *
            " mean=" * string(round(mean_brackets; digits=1)))

    # Generate unbalanced versions by permuting brackets
    rng = MersenneTwister(20260317)
    unbalanced_chunks = String[]
    n_attempts = 0
    for chunk in balanced_chunks
        # Try permutations until we get one that is NOT balanced
        # (with >= 2 brackets, random permutation is almost always unbalanced)
        found = false
        for attempt in 1:50
            n_attempts += 1
            candidate = permute_brackets(chunk, rng)
            if !is_balanced(candidate)
                push!(unbalanced_chunks, candidate)
                found = true
                break
            end
        end
        if !found
            # Skip this chunk — all permutations happen to be balanced
            # (only possible with very few brackets in symmetric positions)
            continue
        end
    end

    # Trim to same length
    n_pairs = min(length(balanced_chunks), length(unbalanced_chunks))
    balanced_chunks = balanced_chunks[1:n_pairs]
    unbalanced_chunks = unbalanced_chunks[1:n_pairs]

    println("  paired chunks: " * string(n_pairs))
    println("  mean permutation attempts: " *
            string(round(n_attempts / length(balanced_chunks); digits=2)))

    # Verify no unbalanced chunk is accidentally balanced
    for (i, chunk) in enumerate(unbalanced_chunks)
        if is_balanced(chunk)
            error("Unbalanced chunk " * string(i) * " is actually balanced: " * chunk)
        end
    end
    println("  all " * string(n_pairs) * " unbalanced chunks verified NOT balanced")

    # Write output
    open(BALANCED_PATH, "w") do io
        for chunk in balanced_chunks
            println(io, chunk)
        end
    end
    open(UNBALANCED_PATH, "w") do io
        for chunk in unbalanced_chunks
            println(io, chunk)
        end
    end

    println()
    println("Written:")
    println("  " * BALANCED_PATH * " (" * string(n_pairs) * " lines)")
    println("  " * UNBALANCED_PATH * " (" * string(n_pairs) * " lines)")

    # Show a few examples
    println()
    println("Sample balanced / unbalanced pairs:")
    for i in 1:min(5, n_pairs)
        println("  BAL: " * balanced_chunks[i])
        println("  UNB: " * unbalanced_chunks[i])
        println()
    end
end

main()
