"""
Two-ecology theorem checks:
  - prop: ecology injection threshold
  - cor: post-training refines developmental resolution
  - prop: selection on recipe traits

Run:  julia --project=. scripts/check_two_ecology.jl
"""

using Random
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

function with_context_weights(fe::FiniteEcology, weights::AbstractVector{Float64})
    d_c = collect(weights ./ sum(weights))
    FiniteEcology(fe.pi, d_c, fe.p_wcv)
end

function mixed_ecology(fe0::FiniteEcology, fe1::FiniteEcology, alpha::Float64)
    n_worlds(fe0) == n_worlds(fe1) || error("world mismatch")
    n_vocab(fe0) == n_vocab(fe1) || error("vocab mismatch")

    # Represent the mixed ecology as a disjoint union of task/context
    # coordinates. This works because task_distance is linear in the
    # context weights d_c, so concatenating contexts with weights
    # (1-alpha) d0 and alpha d1 is exactly the same as mixing measures.
    d0 = (1.0 - alpha) .* fe0.d_c
    d1 = alpha .* fe1.d_c
    d_mix = vcat(d0, d1)
    p_mix = cat(fe0.p_wcv, fe1.p_wcv; dims=2)
    FiniteEcology(fe0.pi, d_mix, p_mix)
end

function strong_ecology(n_w::Int, n_c::Int, n_v::Int)
    n_v >= n_w || error("need n_v >= n_w for one-hot strong ecology")
    pi = fill(1.0 / n_w, n_w)
    d_c = fill(1.0 / n_c, n_c)
    p_wcv = zeros(n_w, n_c, n_v)
    for w in 1:n_w, c in 1:n_c
        p_wcv[w, c, w] = 1.0
    end
    FiniteEcology(pi, d_c, p_wcv)
end

function threshold_base_ecology()
    n_w, n_c, n_v = 4, 3, 4
    pi = fill(0.25, n_w)
    d_c = [0.90, 0.05, 0.05]
    p_wcv = zeros(n_w, n_c, n_v)

    # Pair (1,2) differs only weakly on one context.
    p_wcv[1, 1, :] .= [0.51, 0.49, 0.0, 0.0]
    p_wcv[2, 1, :] .= [0.49, 0.51, 0.0, 0.0]
    p_wcv[1, 2, :] .= [0.25, 0.25, 0.25, 0.25]
    p_wcv[2, 2, :] .= [0.25, 0.25, 0.25, 0.25]
    p_wcv[1, 3, :] .= [0.40, 0.10, 0.30, 0.20]
    p_wcv[2, 3, :] .= [0.40, 0.10, 0.30, 0.20]

    # Other worlds are generic.
    p_wcv[3, 1, :] .= [0.0, 0.0, 1.0, 0.0]
    p_wcv[4, 1, :] .= [0.0, 0.0, 0.0, 1.0]
    p_wcv[3, 2, :] .= [0.2, 0.3, 0.4, 0.1]
    p_wcv[4, 2, :] .= [0.3, 0.1, 0.2, 0.4]
    p_wcv[3, 3, :] .= [0.1, 0.2, 0.6, 0.1]
    p_wcv[4, 3, :] .= [0.2, 0.2, 0.1, 0.5]

    FiniteEcology(pi, d_c, p_wcv)
end

function rescued_pairs(fe0::FiniteEcology, fe1::FiniteEcology, alpha::Float64, epsilon::Float64)
    fe_mix = mixed_ecology(fe0, fe1, alpha)
    out = Set{Tuple{Int,Int}}()
    nw = n_worlds(fe0)
    for i in 1:nw, j in (i + 1):nw
        if task_distance(fe_mix, i, j) > epsilon
            push!(out, (i, j))
        end
    end
    out
end

function check_ecology_injection_threshold(rng::AbstractRNG; seed=0)
    cr = CheckResult("prop_ecology_injection_threshold")

    # Exact interpolation on random ecology pairs.
    for trial in 1:20
        fe0 = random_ecology(rng, 4, 3, 4)
        fe1 = random_ecology(rng, 4, 2, 4)
        for alpha in (0.0, 0.1, 0.25, 0.5, 0.8, 1.0)
            fe_mix = mixed_ecology(fe0, fe1, alpha)
            for i in 1:4, j in (i + 1):4
                lhs = task_distance(fe_mix, i, j)
                rhs = (1.0 - alpha) * task_distance(fe0, i, j) +
                      alpha * task_distance(fe1, i, j)
                record!(cr, lhs - rhs, "interpolation seed=" * string(seed) *
                        " trial=" * string(trial) * " alpha=" * string(alpha) *
                        " pair=" * string((i, j)))
            end
        end
    end

    # Explicit threshold check on a planted gap pair.
    fe0 = threshold_base_ecology()
    fe1 = strong_ecology(4, 2, 4)
    sigma0 = task_distance(fe0, 1, 2)
    sigma1 = task_distance(fe1, 1, 2)
    epsilon = 0.5 * (sigma0 + sigma1)
    alpha_star = (epsilon - sigma0) / (sigma1 - sigma0)

    for alpha in (0.0, 0.25 * alpha_star, 0.75 * alpha_star, alpha_star,
                  min(1.0, alpha_star + 0.05), min(1.0, alpha_star + 0.2))
        fe_mix = mixed_ecology(fe0, fe1, alpha)
        sigma = task_distance(fe_mix, 1, 2)
        predicted = alpha > alpha_star + TOL
        observed = sigma > epsilon + TOL
        if predicted != observed
            record!(cr, 1.0, "threshold mismatch alpha=" * string(alpha) *
                    " alpha*=" * string(alpha_star) *
                    " sigma=" * string(sigma) *
                    " epsilon=" * string(epsilon))
        else
            record!(cr, 0.0)
        end
    end

    cr
end

function check_post_training_refinement(rng::AbstractRNG; seed=0)
    cr = CheckResult("cor_post_training_refinement")

    # Part 1: refinement holds for alpha < 1 for arbitrary injected ecology.
    for trial in 1:20
        fe0 = random_ecology(rng, 4, 3, 4)
        fe1 = random_ecology(rng, 4, 2, 4)
        base = ecology_partition(fe0)
        for alpha in (0.05, 0.2, 0.5, 0.8, 0.95)
            mix = ecology_partition(mixed_ecology(fe0, fe1, alpha))
            if !is_refinement(mix, base)
                record!(cr, 1.0, "refinement failed seed=" * string(seed) *
                        " trial=" * string(trial) * " alpha=" * string(alpha))
            else
                record!(cr, 0.0)
            end
        end
    end

    # Parts 2 and 3: rescue and monotone rescued set under stronger injected ecology.
    fe0 = threshold_base_ecology()
    fe1 = strong_ecology(4, 2, 4)
    nw = n_worlds(fe0)

    epsilons = Dict{Tuple{Int,Int},Float64}()
    for i in 1:nw, j in (i + 1):nw
        s0 = task_distance(fe0, i, j)
        s1 = task_distance(fe1, i, j)
        epsilons[(i, j)] = 0.5 * (s0 + s1)
    end

    # Gap-pair rescue on the planted pair (1,2).
    epsilon = epsilons[(1, 2)]
    sigma0 = task_distance(fe0, 1, 2)
    sigma1 = task_distance(fe1, 1, 2)
    alpha_star = (epsilon - sigma0) / (sigma1 - sigma0)
    for alpha in (0.0, 0.5 * alpha_star, min(1.0, alpha_star + 0.05))
        fe_mix = mixed_ecology(fe0, fe1, alpha)
        observed = task_distance(fe_mix, 1, 2) > epsilon + TOL
        predicted = alpha > alpha_star + TOL
        if observed != predicted
            record!(cr, 1.0, "gap rescue mismatch alpha=" * string(alpha))
        else
            record!(cr, 0.0)
        end
    end

    # Monotone rescued set for pairwise epsilons under strong ecology.
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8]
    previous = Set{Tuple{Int,Int}}()
    for alpha in alphas
        current = Set{Tuple{Int,Int}}()
        fe_mix = mixed_ecology(fe0, fe1, alpha)
        for i in 1:nw, j in (i + 1):nw
            epsilon_ij = epsilons[(i, j)]
            if task_distance(fe_mix, i, j) > epsilon_ij + TOL
                push!(current, (i, j))
            end
        end
        if !issubset(previous, current)
            record!(cr, 1.0, "rescued set not monotone at alpha=" * string(alpha))
        else
            record!(cr, 0.0)
        end
        previous = current
    end

    cr
end

function check_recipe_trait_selection(rng::AbstractRNG; seed=0)
    cr = CheckResult("prop_recipe_trait_selection")

    for trial in 1:40
        # Three alpha levels crossed with two background recipe states.
        alphas = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
        zetas  = [1, 1, 1, 2, 2, 2]
        x_raw = rand(rng, length(alphas))
        x = x_raw ./ sum(x_raw)

        # Δ_sel(α, ζ) weakly decreases in α for each ζ.
        base = Dict(1 => 1.4, 2 => 1.1)
        slope = Dict(1 => 0.35, 2 => 0.15)
        delta = [base[zetas[i]] - slope[zetas[i]] * alphas[i]
                 for i in eachindex(alphas)]
        fitness = exp.(-delta)

        alpha_bar = sum(x[i] * alphas[i] for i in eachindex(alphas))
        f_bar = sum(x[i] * fitness[i] for i in eachindex(alphas))
        alpha_bar_sel = sum(x[i] * fitness[i] * alphas[i] for i in eachindex(alphas)) / f_bar
        cov_x = sum(x[i] * (fitness[i] - f_bar) * (alphas[i] - alpha_bar)
                    for i in eachindex(alphas))

        lhs = alpha_bar_sel - alpha_bar
        rhs = cov_x / f_bar
        record!(cr, lhs - rhs, "identity trial=" * string(trial) *
                " seed=" * string(seed))

        if cov_x < -TOL
            record!(cr, 1.0, "covariance negative under monotone selective excess")
        else
            record!(cr, 0.0)
        end
        if alpha_bar_sel + TOL < alpha_bar
            record!(cr, 1.0, "selected mean alpha decreased despite monotone setup")
        else
            record!(cr, 0.0)
        end
    end

    # Strict binary-trait case: mean increase implies stronger injected trait selected.
    x = [0.6, 0.4]
    alphas = [0.0, 1.0]
    delta = [1.2, 0.4]
    fitness = exp.(-delta)
    alpha_bar = sum(x[i] * alphas[i] for i in eachindex(alphas))
    f_bar = sum(x[i] * fitness[i] for i in eachindex(alphas))
    alpha_bar_sel = sum(x[i] * fitness[i] * alphas[i] for i in eachindex(alphas)) / f_bar
    if alpha_bar_sel <= alpha_bar + TOL
        record!(cr, 1.0, "strict binary-trait case failed to increase mean alpha")
    else
        record!(cr, 0.0)
    end

    cr
end

function main()
    println("=" ^ 60)
    println("Two-ecology theorem checks")
    println("=" ^ 60)
    println()

    rng = MersenneTwister(20260317)

    results = CheckResult[
        check_ecology_injection_threshold(rng; seed=20260317),
        check_post_training_refinement(rng; seed=20260317),
        check_recipe_trait_selection(rng; seed=20260317),
    ]

    for cr in results
        report(cr)
    end
end

main()
