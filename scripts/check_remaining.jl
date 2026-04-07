"""
Remaining theorem checks:
  - prop:ling-bound
  - cor:topological-convergence
  - prop:icl-bound
  - prop:llm-wf

Run:  julia --project=. scripts/check_remaining.jl
"""

using Random, LinearAlgebra
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using LlmPaper

const TOL = 1e-10

mutable struct CheckResult
    name::String
    cases::Int
    worst_gap::Float64
    passed::Bool
    worst_detail::String
end

CheckResult(name) = CheckResult(name, 0, 0.0, true, "")

function record!(cr::CheckResult, gap::Float64, detail::String="")
    cr.cases += 1
    absgap = abs(gap)
    if absgap > cr.worst_gap
        cr.worst_gap = absgap
        cr.worst_detail = detail
    end
    if absgap > TOL
        cr.passed = false
    end
end

function report(cr::CheckResult)
    status = cr.passed ? "PASS" : "FAIL"
    println("check_name: " * cr.name)
    println("cases_checked: " * string(cr.cases))
    println("status: " * status)
    println("worst_abs_gap: " * string(cr.worst_gap))
    if !cr.passed && cr.worst_detail != ""
        println("worst_detail: " * cr.worst_detail)
    end
    println()
end

function renumber_labels(labels::Vector{Int})
    mapping = Dict{Int, Int}()
    next_label = 1
    out = similar(labels)
    for (idx, label) in enumerate(labels)
        if !haskey(mapping, label)
            mapping[label] = next_label
            next_label += 1
        end
        out[idx] = mapping[label]
    end
    out
end

function linguistic_equivalent(fe::FiniteEcology, w1::Int, w2::Int)
    for c in 1:n_contexts(fe), v in 1:n_vocab(fe)
        if fe.p_wcv[w1, c, v] != fe.p_wcv[w2, c, v]
            return false
        end
    end
    true
end

function linguistic_partition(fe::FiniteEcology)
    nw = n_worlds(fe)
    labels = collect(1:nw)
    for i in 1:nw, j in (i + 1):nw
        if linguistic_equivalent(fe, i, j)
            old_label = labels[j]
            new_label = labels[i]
            for k in 1:nw
                if labels[k] == old_label
                    labels[k] = new_label
                end
            end
        end
    end
    renumber_labels(labels)
end

function with_context_weights(fe::FiniteEcology, weights::AbstractVector{Float64})
    d_c = collect(weights ./ sum(weights))
    FiniteEcology(fe.pi, d_c, fe.p_wcv)
end

function centered_gram_rank(code_matrix::AbstractMatrix{Float64})
    centered = code_matrix .- (1.0 / size(code_matrix, 1)) .* ones(size(code_matrix, 1)) * sum(code_matrix, dims=1)
    gram = centered * centered'
    eigs = eigvals(Symmetric(gram))
    count(x -> x > TOL, eigs)
end

function check_ling_bound(rng::AbstractRNG; seed=0, n_w=4, n_c=3, n_v=3)
    cr = CheckResult("prop_ling_bound")

    # Part (a): a linguistically equivalent pair has zero ecology distance
    fe_eq = ecology_with_merged_pair(rng, n_w, n_c, n_v; merged=(1, 2))
    for alt in 1:5
        weights = rand(rng, n_c)
        fe_alt = with_context_weights(fe_eq, weights)
        sigma = task_distance(fe_alt, 1, 2)
        record!(cr, sigma, "linguistically equivalent pair has positive sigma, seed=" *
                string(seed) * " alt=" * string(alt))

        ling = linguistic_partition(fe_alt)
        eco = ecology_partition(fe_alt)
        if !is_refinement(ling, eco)
            record!(cr, 1.0, "linguistic partition not refined by ecology partition")
        else
            record!(cr, 0.0)
        end
    end

    # Parts (b) and (c): a non-equivalent pair can be separated by a witness
    fe_sep = ecology_with_partial_separation(rng, n_w, n_c, n_v;
                                             pair=(1, 2), sep_context=1)

    # Witness distribution concentrated on the one separating context.
    witness = zeros(Float64, n_c)
    witness[1] = 1.0
    fe_witness = with_context_weights(fe_sep, witness)
    sigma_witness = task_distance(fe_witness, 1, 2)
    if sigma_witness <= TOL
        record!(cr, 1.0, "witness context failed to separate non-equivalent pair")
    else
        record!(cr, 0.0)
    end

    # Full support gives equality of linguistic and ecology partitions.
    fe_full = with_context_weights(fe_sep, ones(n_c))
    ling_full = linguistic_partition(fe_full)
    eco_full = ecology_partition(fe_full)
    if !same_partition(ling_full, eco_full)
        record!(cr, 1.0, "full-support ecology partition differs from linguistic partition")
    else
        record!(cr, 0.0)
    end

    # Zero mass on the only separating context yields strict inclusion.
    zero_sep = ones(Float64, n_c)
    zero_sep[1] = 0.0
    fe_zero = with_context_weights(fe_sep, zero_sep)
    ling_zero = linguistic_partition(fe_zero)
    eco_zero = ecology_partition(fe_zero)
    if !is_refinement(ling_zero, eco_zero)
        record!(cr, 1.0, "general inclusion violated when separating context has zero mass")
    else
        record!(cr, 0.0)
    end
    if same_partition(ling_zero, eco_zero)
        record!(cr, 1.0, "expected strict inclusion when witness context has zero mass")
    else
        record!(cr, 0.0)
    end

    cr
end

function check_topological_convergence(rng::AbstractRNG; seed=0, n_w=5, n_c=3, n_v=3)
    cr = CheckResult("cor_topological_convergence")
    fe = random_ecology(rng, n_w, n_c, n_v)
    partitions = all_partitions(n_w)
    eco = ecology_partition(fe)
    eco_k = n_classes(eco)

    zero_excess = Vector{Tuple{Vector{Int}, Float64}}()
    for labels in partitions
        if is_zero_excess(fe, labels)
            push!(zero_excess, (labels, partition_complexity(fe, labels)))
        end
    end
    min_complexity = minimum(last.(zero_excess))
    minimizers = [labels for (labels, comp) in zero_excess if abs(comp - min_complexity) < TOL]

    for labels in minimizers
        if !same_partition(labels, eco)
            record!(cr, 1.0, "minimum-complexity zero-excess partition differs from ecology partition")
        else
            record!(cr, 0.0)
        end
    end

    # Rank bound for any kernel on k_D distinct class codes.
    line_codes = reshape(collect(1.0:eco_k), eco_k, 1)
    H_line = line_codes[eco, :]
    rank_line = centered_gram_rank(H_line)
    if rank_line > eco_k - 1
        record!(cr, Float64(rank_line - (eco_k - 1)), "line-code kernel rank bound violated")
    else
        record!(cr, 0.0)
    end

    if eco_k > 1
        simplex_codes = zeros(eco_k, eco_k - 1)
        for i in 1:(eco_k - 1)
            simplex_codes[i, i] = 1.0
        end
        H_simplex = simplex_codes[eco, :]
        rank_simplex = centered_gram_rank(H_simplex)
        gap = Float64(rank_simplex - (eco_k - 1))
        record!(cr, gap, "simplex-code rank mismatch")
    else
        record!(cr, 0.0)
    end

    cr
end

function separation_pairs(outputs::Matrix{Int}, contexts::AbstractVector{Int})
    n_c, n_w = size(outputs)
    mask = falses(n_w, n_w)
    for i in 1:n_w, j in (i + 1):n_w
        sep = false
        for c in contexts
            if outputs[c, i] != outputs[c, j]
                sep = true
                break
            end
        end
        mask[i, j] = sep
        mask[j, i] = sep
    end
    mask
end

function check_icl_bound(rng::AbstractRNG; seed=0, n_w=5, n_c=4, n_codes=4)
    cr = CheckResult("prop_icl_bound")
    outputs = rand(rng, 1:n_codes, n_c, n_w)
    full_contexts = collect(1:n_c)
    full_sep = separation_pairs(outputs, full_contexts)

    for mask in 0:(2^n_c - 1)
        chosen = Int[]
        for c in 1:n_c
            if ((mask >> (c - 1)) & 1) == 1
                push!(chosen, c)
            end
        end
        subset_sep = separation_pairs(outputs, chosen)
        for i in 1:n_w, j in (i + 1):n_w
            if subset_sep[i, j] && !full_sep[i, j]
                record!(cr, 1.0, "subset of contexts enlarged separation set, mask=" *
                        string(mask) * " seed=" * string(seed))
            else
                record!(cr, 0.0)
            end
        end
    end

    cr
end

function random_stochastic_matrix(rng::AbstractRNG, n::Int)
    q = rand(rng, n, n)
    q ./ sum(q, dims=2)
end

function check_llm_wf(rng::AbstractRNG; seed=0, n_enc=3, n_lineages=6)
    cr = CheckResult("prop_llm_wf")

    # Ensure each encoding appears at least once and at least one is duplicated.
    lineage_encodings = [1, 2, 3, 1, rand(rng, 1:n_enc), rand(rng, 1:n_enc)]
    fitness = rand(rng, n_enc) .+ 0.2
    Q = random_stochastic_matrix(rng, n_enc)

    # Direct lineage-level expectation.
    lineage_fitness = [fitness[z] for z in lineage_encodings]
    p_line = lineage_fitness ./ sum(lineage_fitness)
    direct = zeros(Float64, n_enc)
    for (idx, z) in enumerate(lineage_encodings), j in 1:n_enc
        direct[j] += p_line[idx] * Q[z, j]
    end

    # Reduced encoding-level replicator-mutator expectation.
    x = zeros(Float64, n_enc)
    for z in lineage_encodings
        x[z] += 1.0 / n_lineages
    end
    mean_fitness = dot(x, fitness)
    reduced = zeros(Float64, n_enc)
    for i in 1:n_enc, j in 1:n_enc
        reduced[j] += (x[i] * fitness[i] / mean_fitness) * Q[i, j]
    end

    for j in 1:n_enc
        record!(cr, direct[j] - reduced[j],
                "lineage-level and encoding-level expectations differ for state " *
                string(j) * ", seed=" * string(seed))
    end

    record!(cr, sum(direct) - 1.0, "direct expectation does not normalize")
    record!(cr, sum(reduced) - 1.0, "reduced expectation does not normalize")

    cr
end

function main()
    base_seed = 20260312
    results = CheckResult[]

    for offset in 0:19
        seed = base_seed + offset
        push!(results, check_ling_bound(MersenneTwister(seed); seed=seed))
        push!(results, check_topological_convergence(MersenneTwister(seed + 1000); seed=seed))
        push!(results, check_icl_bound(MersenneTwister(seed + 2000); seed=seed))
        push!(results, check_llm_wf(MersenneTwister(seed + 3000); seed=seed))
    end

    for name in ["prop_ling_bound", "cor_topological_convergence",
                 "prop_icl_bound", "prop_llm_wf"]
        grouped = filter(r -> r.name == name, results)
        summary = CheckResult(name)
        summary.cases = sum(r.cases for r in grouped)
        summary.worst_gap = maximum(r.worst_gap for r in grouped)
        summary.passed = all(r.passed for r in grouped)
        if !summary.passed
            worst = grouped[argmax([r.worst_gap for r in grouped])]
            summary.worst_detail = worst.worst_detail
        end
        report(summary)
    end
end

main()
