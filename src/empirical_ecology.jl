# ── Empirical finite ecologies from token corpora ────────────────────────────
#
# Bridges the neural experiments to the exact finite-ecology theory by
# estimating a discrete task ecology directly from held-out token sequences.

"""
    partition_labels_from_distance(D, tau) -> Vector{Int}

Return a label vector for the connected components of the graph where
`D[i,j] <= tau`.
"""
function partition_labels_from_distance(D::Matrix{Float64}, tau::Float64)
    n = size(D, 1)
    parent = collect(1:n)

    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        x
    end

    function uf_union!(x, y)
        px, py = find(x), find(y)
        px != py && (parent[px] = py)
    end

    for i in 1:n, j in (i + 1):n
        if D[i, j] <= tau
            uf_union!(i, j)
        end
    end

    labels = [find(i) for i in 1:n]
    mapping = Dict{Int, Int}()
    next = 1
    out = similar(labels)
    for i in eachindex(labels)
        lab = labels[i]
        if !haskey(mapping, lab)
            mapping[lab] = next
            next += 1
        end
        out[i] = mapping[lab]
    end
    out
end

"""
    empirical_prefix_ecology(tokens_by_lang, languages, prefix_lengths, vocab_size;
                             uniform_worlds=true,
                             context_weighting=:uniform)
        -> (fe, valid_prefix_lengths, counts_by_lang)

Construct a finite ecology whose contexts are prefix lengths `r`.

For each language `w` and prefix length `r`, the empirical next-token law
`P_w(.|r)` is estimated from held-out token sequences by counting the token
that follows the first `r` in-vocabulary tokens. Contexts with no observations
for any language are dropped; with `context_weighting=:uniform`, the remaining
contexts are equally weighted.
"""
function empirical_prefix_ecology(tokens_by_lang::Dict{String, Vector{Vector{Int}}}, 
                                  languages::Vector{String},
                                  prefix_lengths::Vector{Int},
                                  vocab_size::Int;
                                  uniform_worlds::Bool=true,
                                  context_weighting::Symbol=:uniform)
    valid_lengths = Int[]
    counts_by_lang = Dict{String, Dict{Int, Vector{Float64}}}()
    totals_by_lang = Dict{String, Dict{Int, Int}}()

    for lang in languages
        counts_by_lang[lang] = Dict{Int, Vector{Float64}}()
        totals_by_lang[lang] = Dict{Int, Int}()
        for r in prefix_lengths
            counts = zeros(Float64, vocab_size)
            total = 0
            for tokens in tokens_by_lang[lang]
                target_pos = r + 2
                target_pos <= length(tokens) || continue
                target = tokens[target_pos]
                counts[target] += 1.0
                total += 1
            end
            counts_by_lang[lang][r] = counts
            totals_by_lang[lang][r] = total
        end
    end

    for r in prefix_lengths
        if all(totals_by_lang[lang][r] > 0 for lang in languages)
            push!(valid_lengths, r)
        end
    end
    isempty(valid_lengths) && error("no shared empirical prefix contexts found")

    n_w = length(languages)
    n_c = length(valid_lengths)
    p_wcv = Array{Float64}(undef, n_w, n_c, vocab_size)
    total_counts = zeros(Float64, n_c)

    for (wi, lang) in enumerate(languages), (ci, r) in enumerate(valid_lengths)
        counts = counts_by_lang[lang][r]
        total = totals_by_lang[lang][r]
        total_counts[ci] += total
        p_wcv[wi, ci, :] .= counts ./ total
    end

    pi = if uniform_worlds
        fill(1.0 / n_w, n_w)
    else
        world_totals = [sum(totals_by_lang[lang][r] for r in valid_lengths) for lang in languages]
        world_totals ./ sum(world_totals)
    end

    d_c = if context_weighting == :uniform
        fill(1.0 / n_c, n_c)
    elseif context_weighting == :count
        total_counts ./ sum(total_counts)
    else
        error("unsupported context_weighting: " * string(context_weighting))
    end

    fe = FiniteEcology(pi, d_c, p_wcv)
    (fe, valid_lengths, totals_by_lang)
end
