"""
Finite-class checks for:
  - thm:sgd-conditional
  - thm:sgd-conditional (prior-weighted form)
  - cor:sgd-near

This script does not model SGD in parameter space. It validates the finite-class
uniform-convergence theorem, and its prior-weighted refinement, exactly on tiny
ecologies by exhaustive enumeration of all datasets of length n over the joint
support.

Run: julia --project=. scripts/check_finite_sample.jl
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

function build_decoder(fe::FiniteEcology, labels::AbstractVector{Int})
    decoder = Dict{Tuple{Int,Int}, Vector{Float64}}()
    for (label, cell) in enumerate(partition_cells(labels)), c in 1:n_contexts(fe)
        decoder[(label, c)] = cell_average(fe, cell, c)
    end
    decoder
end

function joint_support(fe::FiniteEcology)
    support = Tuple{Int,Int,Int}[]
    probs = Float64[]
    for w in 1:n_worlds(fe), c in 1:n_contexts(fe), v in 1:n_vocab(fe)
        p = fe.pi[w] * fe.d_c[c] * fe.p_wcv[w, c, v]
        if p > 0
            push!(support, (w, c, v))
            push!(probs, p)
        end
    end
    support, probs
end

function decoder_sample_loss(decoder, labels, sample)
    w, c, v = sample
    -log(decoder[(labels[w], c)][v])
end

function canonical_partition_key(labels::AbstractVector{Int})
    Tuple(labels)
end

function empirical_losses(dataset, decoders, labels_list)
    n = length(dataset)
    [sum(decoder_sample_loss(decoder, labels, sample) for sample in dataset) / n
     for (decoder, labels) in zip(decoders, labels_list)]
end

function make_uniform_prior(partitions)
    m = length(partitions)
    Dict(canonical_partition_key(labels) => 1.0 / m for labels in partitions)
end

function make_entropic_prior(fe::FiniteEcology, partitions, beta0::Float64)
    weights = Dict{Tuple{Vararg{Int}}, Float64}()
    z = 0.0
    for labels in partitions
        key = canonical_partition_key(labels)
        w = exp(-beta0 * partition_complexity(fe, labels))
        weights[key] = w
        z += w
    end
    Dict(key => w / z for (key, w) in weights)
end

function theorem_prior_threshold(fe::FiniteEcology, partitions, true_losses,
                                 prior, alpha::Float64, eps_opt::Float64)
    h_vcw = conditional_entropy_VCW(fe)
    veridical_idx = findfirst(labels -> !merges_separated_pair(fe, labels), partitions)
    veridical_idx === nothing && error("no veridical partition in family")
    gamma = minimum(true_losses[i] - h_vcw for i in eachindex(partitions)
                    if merges_separated_pair(fe, partitions[i]))

    tau = Inf
    decoders = [build_decoder(fe, labels) for labels in partitions]
    for (decoder, labels) in zip(decoders, partitions)
        for w in 1:n_worlds(fe), c in 1:n_contexts(fe), v in 1:n_vocab(fe)
            if fe.pi[w] * fe.d_c[c] * fe.p_wcv[w, c, v] > 0
                tau = min(tau, decoder[(labels[w], c)][v])
            end
        end
    end
    C_tau = log(1 / tau)

    p_v = prior[canonical_partition_key(partitions[veridical_idx])]
    max_term = log(1 / p_v) + log(2 / alpha)
    for i in eachindex(partitions)
        merges_separated_pair(fe, partitions[i]) || continue
        delta = true_losses[i] - h_vcw
        rho = prior[canonical_partition_key(partitions[i])]
        term = (log(1 / rho) + log(2 / alpha)) * (gamma / delta)^2
        max_term = max(max_term, term)
    end

    threshold = 2.0 * C_tau^2 * max_term / (gamma - eps_opt)^2
    threshold, gamma, C_tau, veridical_idx
end

function find_full_separation_ecology(seed::Int; min_gap::Float64=1e-3)
    for attempt in 1:500
        rng = MersenneTwister(seed + attempt)
        fe = random_ecology(rng, 3, 2, 2; floor=0.05)
        labels = ecology_partition(fe)
        n_classes(labels) == 3 || continue
        parts = all_partitions(3)
        h_vcw = conditional_entropy_VCW(fe)
        nonveridical_gaps = [
            bayes_opt_loss(fe, part) - h_vcw
            for part in parts if merges_separated_pair(fe, part)
        ]
        isempty(nonveridical_gaps) && continue
        minimum(nonveridical_gaps) > min_gap || continue
        return fe
    end
    error("failed to find a full-separation ecology with positive gap")
end

function check_sgd_conditional_exact(fe::FiniteEcology, partitions, n_values; seed=0)
    cr = CheckResult("thm_sgd_conditional")
    h_vcw = conditional_entropy_VCW(fe)
    M = length(partitions)
    decoders = [build_decoder(fe, labels) for labels in partitions]
    true_losses = [bayes_opt_loss(fe, labels) for labels in partitions]
    optimal_loss = minimum(true_losses)
    gamma = minimum(true_losses[i] - h_vcw for i in eachindex(partitions)
                    if merges_separated_pair(fe, partitions[i]))

    tau = Inf
    for (decoder, labels) in zip(decoders, partitions)
        for w in 1:n_worlds(fe), c in 1:n_contexts(fe), v in 1:n_vocab(fe)
            if fe.pi[w] * fe.d_c[c] * fe.p_wcv[w, c, v] > 0
                tau = min(tau, decoder[(labels[w], c)][v])
            end
        end
    end
    C_tau = log(1 / tau)

    support, probs = joint_support(fe)
    eps_opt = 0.25 * gamma
    eta = 0.5 * (gamma - eps_opt)
    hoeffding_upper = n -> min(1.0, 2.0 * M * exp(-2.0 * n * eta^2 / C_tau^2))

    for n in n_values
        datasets_checked = 0
        event_prob = 0.0
        failure_prob = 0.0

        for idxs in Iterators.product(ntuple(_ -> 1:length(support), n)...)
            datasets_checked += 1
            dataset = [support[idx] for idx in idxs]
            dataset_prob = prod(probs[idx] for idx in idxs)
            emp_losses = empirical_losses(dataset, decoders, partitions)
            max_dev = maximum(abs.(emp_losses .- true_losses))
            min_emp = minimum(emp_losses)
            near_emp = [i for i in eachindex(partitions)
                        if emp_losses[i] <= min_emp + eps_opt + TOL]

            if max_dev < eta
                event_prob += dataset_prob
                for i in near_emp
                    if merges_separated_pair(fe, partitions[i])
                        record!(cr, 1.0, "non-veridical empirical near-minimizer on E_eta," *
                                " n=" * string(n) * " seed=" * string(seed))
                    else
                        record!(cr, 0.0)
                    end
                end
            else
                if any(merges_separated_pair(fe, partitions[i]) for i in near_emp)
                    failure_prob += dataset_prob
                end
            end
        end

        complement_prob = 1.0 - event_prob
        record!(cr, max(0.0, complement_prob - hoeffding_upper(n)),
                "Hoeffding event bound violated, n=" * string(n) *
                " seed=" * string(seed))
        record!(cr, max(0.0, failure_prob - complement_prob),
                "failure occurs on E_eta, n=" * string(n) * " seed=" * string(seed))
        record!(cr, abs(optimal_loss - h_vcw),
                "expected veridical optimum mismatch, n=" * string(n) *
                " seed=" * string(seed))

        # Count the exhaustive dataset family as one case.
        record!(cr, 0.0, "datasets=" * string(datasets_checked) * " n=" * string(n))
    end

    cr
end

function check_sgd_conditional_prior_exact(fe::FiniteEcology, partitions, n_values; seed=0)
    cr = CheckResult("thm_sgd_conditional_prior")
    h_vcw = conditional_entropy_VCW(fe)
    decoders = [build_decoder(fe, labels) for labels in partitions]
    true_losses = [bayes_opt_loss(fe, labels) for labels in partitions]
    support, probs = joint_support(fe)

    priors = [
        ("uniform", make_uniform_prior(partitions), 0.25),
        ("entropic_beta_0.5", make_entropic_prior(fe, partitions, 0.5), 0.25),
        ("entropic_beta_1.5", make_entropic_prior(fe, partitions, 1.5), 0.25),
    ]

    for (prior_name, prior, alpha) in priors
        gamma = minimum(true_losses[i] - h_vcw for i in eachindex(partitions)
                        if merges_separated_pair(fe, partitions[i]))
        eps_opt = 0.25 * gamma
        threshold, _, C_tau, veridical_idx = theorem_prior_threshold(
            fe, partitions, true_losses, prior, alpha, eps_opt
        )
        p_v = partitions[veridical_idx]

        for n in n_values
            etas = [
                C_tau * sqrt((log(1 / prior[canonical_partition_key(labels)]) + log(2 / alpha)) / (2.0 * n))
                for labels in partitions
            ]

            event_prob = 0.0
            failure_prob = 0.0
            guarantee_active = n + TOL >= threshold

            for idxs in Iterators.product(ntuple(_ -> 1:length(support), n)...)
                dataset = [support[idx] for idx in idxs]
                dataset_prob = prod(probs[idx] for idx in idxs)
                emp_losses = empirical_losses(dataset, decoders, partitions)
                on_event = all(abs(emp_losses[i] - true_losses[i]) < etas[i] - TOL
                               for i in eachindex(partitions))

                if on_event
                    event_prob += dataset_prob
                    min_emp = minimum(emp_losses)
                    near_emp = [i for i in eachindex(partitions)
                                if emp_losses[i] <= min_emp + eps_opt + TOL]
                    if any(merges_separated_pair(fe, partitions[i]) for i in near_emp)
                        failure_prob += dataset_prob
                        if guarantee_active
                            record!(cr, 1.0,
                                    "prior guarantee violated on event, prior=" * prior_name *
                                    " n=" * string(n) * " seed=" * string(seed))
                        else
                            record!(cr, 0.0)
                        end
                    else
                        record!(cr, 0.0)
                    end
                end
            end

            complement_prob = 1.0 - event_prob
            record!(cr, max(0.0, complement_prob - alpha),
                    "weighted event bound violated, prior=" * prior_name *
                    " n=" * string(n) * " seed=" * string(seed))

            if guarantee_active
                record!(cr, failure_prob,
                        "non-veridical near-minimizer on weighted event, prior=" * prior_name *
                        " n=" * string(n) * " seed=" * string(seed))
            else
                record!(cr, 0.0)
            end

            record!(cr, 0.0,
                    "prior=" * prior_name * " threshold=" * string(threshold) *
                    " n=" * string(n) * " veridical=" * string(p_v))
        end
    end

    cr
end

function check_sgd_near_exact(fe::FiniteEcology, partitions, n_values; seed=0)
    cr = CheckResult("cor_sgd_near")
    decoders = [build_decoder(fe, labels) for labels in partitions]
    true_losses = [bayes_opt_loss(fe, labels) for labels in partitions]
    optimal_loss = minimum(true_losses)
    gamma = minimum(true_losses[i] - optimal_loss for i in eachindex(partitions)
                    if merges_separated_pair(fe, partitions[i]))
    eps_opt = 0.25 * gamma
    eta = 0.5 * (gamma - eps_opt)

    support, probs = joint_support(fe)

    for n in n_values
        complement_prob = 0.0
        bad_prob = 0.0

        for idxs in Iterators.product(ntuple(_ -> 1:length(support), n)...)
            dataset = [support[idx] for idx in idxs]
            dataset_prob = prod(probs[idx] for idx in idxs)
            emp_losses = empirical_losses(dataset, decoders, partitions)
            max_dev = maximum(abs.(emp_losses .- true_losses))
            min_emp = minimum(emp_losses)
            near_emp = [i for i in eachindex(partitions)
                        if emp_losses[i] <= min_emp + eps_opt + TOL]
            rhs = optimal_loss + eps_opt + 2.0 * eta

            if max_dev >= eta
                complement_prob += dataset_prob
            end

            if any(true_losses[i] > rhs + TOL for i in near_emp)
                bad_prob += dataset_prob
                if max_dev < eta
                    record!(cr, 1.0, "near-optimality bound violated on E_eta, n=" *
                            string(n) * " seed=" * string(seed))
                else
                    record!(cr, 0.0)
                end
            else
                record!(cr, 0.0)
            end
        end

        record!(cr, max(0.0, bad_prob - complement_prob),
                "bad near-optimal event exceeds complement, n=" * string(n) *
                " seed=" * string(seed))
    end

    cr
end

function report(cr::CheckResult)
    println("check_name: " * cr.name)
    println("cases_checked: " * string(cr.cases))
    println("status: " * (cr.passed ? "PASS" : "FAIL"))
    println("worst_abs_gap: " * string(cr.worst_gap))
    if !cr.passed && cr.worst_detail != ""
        println("worst_detail: " * cr.worst_detail)
    end
    println()
end

function main()
    println("=" ^ 60)
    println("Finite-class concentration checks")
    println("=" ^ 60)
    println()

    partitions = all_partitions(3)
    n_values = 1:4
    ecologies = [find_full_separation_ecology(8000 + 100 * i) for i in 1:3]

    results = CheckResult[]
    for (i, fe) in enumerate(ecologies)
        seed = 8000 + i
        push!(results, check_sgd_conditional_exact(fe, partitions, n_values; seed=seed))
        push!(results, check_sgd_conditional_prior_exact(fe, partitions, n_values; seed=seed))
        push!(results, check_sgd_near_exact(fe, partitions, n_values; seed=seed))
    end

    println("SUMMARY")
    println()
    for name in ["thm_sgd_conditional", "thm_sgd_conditional_prior", "cor_sgd_near"]
        relevant = filter(r -> r.name == name, results)
        total_cases = sum(r.cases for r in relevant)
        worst = maximum(r.worst_gap for r in relevant)
        all_pass = all(r.passed for r in relevant)
        failed = filter(r -> !r.passed, relevant)

        println("check_name: " * name)
        println("cases_checked: " * string(total_cases))
        println("status: " * (all_pass ? "PASS" : "FAIL"))
        println("worst_abs_gap: " * string(worst))
        if !isempty(failed)
            println("worst_detail: " * failed[1].worst_detail)
        end
        println()
    end
end

main()
