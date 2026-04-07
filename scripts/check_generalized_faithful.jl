"""
Generalized faithful-loss checker on finite ecologies.

Checks the generalized Bayes decomposition theorem in the Brier-score
case, where the Bayes report is still the full categorical law.

Run:  julia --project=. scripts/check_generalized_faithful.jl
"""

using Random
using LinearAlgebra: dot
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

function brier_entropy(p)
    1.0 - dot(p, p)
end

function brier_expected_loss(p, q)
    1.0 + dot(q, q) - 2.0 * dot(p, q)
end

function brier_bayes_opt_loss(fe::FiniteEcology, labels::AbstractVector{Int})
    cells = partition_cells(labels)
    cell_of = Dict{Int, Int}()
    for (idx, cell) in enumerate(cells), w in cell
        cell_of[w] = idx
    end

    loss = 0.0
    for w in 1:n_worlds(fe), c in 1:n_contexts(fe)
        p_w = @view fe.p_wcv[w, c, :]
        q_x = cell_average(fe, cells[cell_of[w]], c)
        loss += fe.pi[w] * fe.d_c[c] * brier_expected_loss(p_w, q_x)
    end
    loss
end

function brier_floor(fe::FiniteEcology)
    total = 0.0
    for w in 1:n_worlds(fe), c in 1:n_contexts(fe)
        total += fe.pi[w] * fe.d_c[c] * brier_entropy(@view fe.p_wcv[w, c, :])
    end
    total
end

function brier_jensen_excess(fe::FiniteEcology, labels::AbstractVector{Int})
    cells = partition_cells(labels)
    total = 0.0
    for c in 1:n_contexts(fe), cell in cells
        length(cell) == 1 && continue
        pi_x = sum(fe.pi[w] for w in cell)
        avg = cell_average(fe, cell, c)
        inside = 0.0
        for w in cell
            alpha_w = fe.pi[w] / pi_x
            inside += alpha_w * brier_entropy(@view fe.p_wcv[w, c, :])
        end
        total += fe.d_c[c] * pi_x * (brier_entropy(avg) - inside)
    end
    total
end

function check_generalized_bayes_brier(fe::FiniteEcology, partitions; seed=0)
    cr = CheckResult("thm_generalized_bayes_brier")
    floor = brier_floor(fe)

    for labels in partitions
        direct = brier_bayes_opt_loss(fe, labels) - floor
        jensen = brier_jensen_excess(fe, labels)
        record!(cr, direct - jensen, "identity seed=" * string(seed) *
                " labels=" * string(labels))

        has_merged_sep = merges_separated_pair(fe, labels)
        is_zero = direct < TOL
        if is_zero && has_merged_sep
            record!(cr, 1.0, "zero excess but merges separated pair, seed=" *
                    string(seed) * " labels=" * string(labels))
        elseif !is_zero && !has_merged_sep
            record!(cr, direct, "nonzero excess without merged separated pair, seed=" *
                    string(seed) * " labels=" * string(labels))
        else
            record!(cr, 0.0)
        end
    end

    cr
end

function main()
    println("=" ^ 60)
    println("Generalized faithful-loss checks")
    println("=" ^ 60)
    println()

    n_random = 12
    sizes = [(3, 2, 2), (3, 3, 3), (4, 2, 3)]
    all_results = CheckResult[]

    for (n_w, n_c, n_v) in sizes
        parts = all_partitions(n_w)
        for trial in 1:n_random
            seed = 3000 * n_w + 100 * n_c + trial
            rng = MersenneTwister(seed)
            fe = random_ecology(rng, n_w, n_c, n_v)
            push!(all_results, check_generalized_bayes_brier(fe, parts; seed=seed))
        end

        for trial in 1:3
            seed = 4000 * n_w + trial
            rng = MersenneTwister(seed)
            fe_merged = ecology_with_merged_pair(rng, n_w, n_c, n_v)
            push!(all_results, check_generalized_bayes_brier(fe_merged, parts; seed=seed))
            if n_c >= 2
                rng2 = MersenneTwister(seed + 10000)
                fe_partial = ecology_with_partial_separation(rng2, n_w, n_c, n_v)
                push!(all_results, check_generalized_bayes_brier(fe_partial, parts;
                                                                 seed=seed + 10000))
            end
        end
    end

    relevant = filter(r -> r.name == "thm_generalized_bayes_brier", all_results)
    total_cases = sum(r.cases for r in relevant)
    worst = maximum(r.worst_gap for r in relevant)
    all_pass = all(r.passed for r in relevant)
    failed = filter(r -> !r.passed, relevant)

    cr = CheckResult("thm_generalized_bayes_brier")
    cr.cases = total_cases
    cr.worst_gap = worst
    cr.passed = all_pass
    if !isempty(failed)
        cr.worst_detail = failed[1].worst_detail
    end
    report(cr)
end

main()
