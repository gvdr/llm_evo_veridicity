"""
Experiment 0: Synthetic Finite Ecology

Direct theorem-aligned calibration experiment on tiny fully discrete ecologies.
All partitions of W are enumerated exactly, so the key quantities from:
  - the exact excess-loss decomposition theorem,
  - the minimum-complexity ecological veridicality theorem, and
  - the split-versus-merge threshold theorem
can be checked without SGD, neural proxies, or thresholded behavioural
distances.

Run: julia --project=. scripts/exp0_synthetic_exact.jl
"""

using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using LlmPaper

const OUTPUT_DIR = joinpath(@__DIR__, "..", "output")
const N_SWEEP = 24
const TOL = 1e-10

function merge_states(labels::Vector{Int}, i::Int, j::Int)
    out = copy(labels)
    li = out[i]
    lj = out[j]
    for idx in eachindex(out)
        if out[idx] == lj
            out[idx] = li
        end
    end
    cells = partition_cells(out)
    renumbered = similar(out)
    for (label, cell) in enumerate(cells), idx in cell
        renumbered[idx] = label
    end
    renumbered
end

function canonical_renumber(labels::Vector{Int})
    cells = partition_cells(labels)
    renumbered = similar(labels)
    for (label, cell) in enumerate(cells), idx in cell
        renumbered[idx] = label
    end
    renumbered
end

function best_partition(fe::FiniteEcology, partitions, beta::Float64)
    vals = [regularized_objective(fe, labels, beta) for labels in partitions]
    idx = argmin(vals)
    partitions[idx], vals[idx]
end

function find_full_separation_ecology(seed::Int; min_beta::Float64=1e-3,
                                      max_beta::Float64=0.5)
    rng = MersenneTwister(seed)
    for attempt in 1:5000
        fe = random_ecology(rng, 4, 3, 3; floor=0.05)
        labels = ecology_partition(fe)
        n_classes(labels) == 4 || continue
        D = task_distance_matrix(fe)
        best_pair = nothing
        best_d = Inf
        for i in 1:4, j in (i + 1):4
            if D[i, j] > 0 && D[i, j] < best_d
                best_d = D[i, j]
                best_pair = (i, j)
            end
        end
        best_pair === nothing && continue
        i, j = best_pair
        pi_i = fe.pi[i]
        pi_j = fe.pi[j]
        lam = pi_i / (pi_i + pi_j)
        gain = split_gain(fe, [i], [j])
        h = binary_entropy(lam)
        h <= 0 && continue
        beta_star = gain / h
        (min_beta <= beta_star <= max_beta) || continue
        return fe, best_pair, beta_star
    end
    error("failed to find a full-separation ecology with a usable split threshold")
end

function run_exact_sweep()
    partitions = all_partitions(4)
    max_decomp_gap = 0.0
    max_split_gap = 0.0
    min_complexity_failures = 0
    zero_excess_failures = 0

    for seed in 1:N_SWEEP
        fe = random_ecology(MersenneTwister(10_000 + seed), 4, 3, 3; floor=0.05)
        h_vcw = conditional_entropy_VCW(fe)
        eco_labels = ecology_partition(fe)
        eco_complexity = partition_complexity(fe, eco_labels)

        for labels in partitions
            gap = abs(excess_loss(fe, labels) - js_excess(fe, labels))
            max_decomp_gap = max(max_decomp_gap, gap)

            has_merged = merges_separated_pair(fe, labels)
            is_zero = excess_loss(fe, labels) < TOL
            if is_zero == has_merged
                zero_excess_failures += 1
            end

            for cell in partition_cells(labels)
                length(cell) < 2 && continue
                for (A, B) in all_binary_splits(cell)
                    refined = copy(labels)
                    old_label = labels[A[1]]
                    new_label = maximum(labels) + 1
                    for idx in B
                        refined[idx] = new_label
                    end
                    refined = canonical_renumber(refined)
                    for beta in (0.0, 0.05, 0.1, 0.2, 0.4)
                        lhs = regularized_objective(fe, labels, beta) -
                              regularized_objective(fe, refined, beta)
                        rhs = split_threshold_rhs(fe, A, B, beta)
                        max_split_gap = max(max_split_gap, abs(lhs - rhs))
                    end
                end
            end
        end

        zero_parts = [labels for labels in partitions if is_zero_excess(fe, labels)]
        if !isempty(zero_parts)
            min_zero_complexity = minimum(partition_complexity(fe, labels)
                                          for labels in zero_parts)
            if abs(min_zero_complexity - eco_complexity) > TOL
                min_complexity_failures += 1
            end
        end
    end

    (max_decomp_gap=max_decomp_gap,
     max_split_gap=max_split_gap,
     zero_excess_failures=zero_excess_failures,
     min_complexity_failures=min_complexity_failures)
end

function main()
    println("=" ^ 70)
    println("Experiment 0: Synthetic Finite Ecology")
    println("=" ^ 70)
    println()

    sweep = run_exact_sweep()
    println("Exact random-ecology sweep (" * string(N_SWEEP) * " ecologies, |W|=4):")
    println("  max decomposition gap      = " * string(round(sweep.max_decomp_gap, digits=14)))
    println("  max split-threshold gap    = " * string(round(sweep.max_split_gap, digits=14)))
    println("  zero-excess failures       = " * string(sweep.zero_excess_failures))
    println("  min-complexity failures    = " * string(sweep.min_complexity_failures))
    println()

    fe, pair, beta_star = find_full_separation_ecology(2026)
    i, j = pair
    full = ecology_partition(fe)
    merged = merge_states(full, i, j)
    lam = fe.pi[i] / (fe.pi[i] + fe.pi[j])
    gain = split_gain(fe, [i], [j])
    h = binary_entropy(lam)

    println("Illustrative full-separation ecology:")
    println("  ecology partition = " * string(full))
    println("  chosen weak pair  = (" * string(i) * ", " * string(j) * ")")
    println("  lambda            = " * string(round(lam, digits=6)))
    println("  Delta_pred        = " * string(round(gain, digits=6)))
    println("  h(lambda)         = " * string(round(h, digits=6)))
    println("  beta_crit         = " * string(round(beta_star, digits=6)))
    println()

    println("Local split-threshold check around beta_crit:")
    println("  beta      | J(merge)-J(split) | theorem rhs | preferred")
    println("  ----------|-------------------|-------------|----------")
    for beta in [0.5 * beta_star, beta_star, 1.5 * beta_star]
        lhs = regularized_objective(fe, merged, beta) -
              regularized_objective(fe, full, beta)
        rhs = split_threshold_rhs(fe, [i], [j], beta)
        pref = lhs > TOL ? "split" : (lhs < -TOL ? "merge" : "tie")
        println("  " * rpad(string(round(beta, digits=6)), 9) * " | " *
                lpad(string(round(lhs, digits=8)), 17) * " | " *
                lpad(string(round(rhs, digits=8)), 11) * " | " * pref)
    end
    println()

    partitions = all_partitions(4)
    println("Exact optimum under J_{D,beta} for a small beta grid:")
    println("  beta      | k | excess     | complexity | labels")
    println("  ----------|---|------------|------------|----------------")
    for beta in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
        labels, _ = best_partition(fe, partitions, beta)
        k = n_classes(labels)
        ex = excess_loss(fe, labels)
        comp = partition_complexity(fe, labels)
        println("  " * rpad(string(round(beta, digits=3)), 9) * " | " *
                string(k) * " | " *
                lpad(string(round(ex, digits=8)), 10) * " | " *
                lpad(string(round(comp, digits=8)), 10) * " | " *
                string(labels))
    end
    println()

    mkpath(OUTPUT_DIR)
    outpath = joinpath(OUTPUT_DIR, "exp0_synthetic_exact_results.txt")
    open(outpath, "w") do io
        println(io, "# Experiment 0: Synthetic Finite Ecology")
        println(io, "# N_SWEEP=" * string(N_SWEEP))
        println(io, "# max_decomposition_gap=" * string(sweep.max_decomp_gap))
        println(io, "# max_split_threshold_gap=" * string(sweep.max_split_gap))
        println(io, "# zero_excess_failures=" * string(sweep.zero_excess_failures))
        println(io, "# min_complexity_failures=" * string(sweep.min_complexity_failures))
        println(io, "# illustrative_pair=" * string(pair))
        println(io, "# beta_crit=" * string(beta_star))
        println(io, "# full_partition=" * string(full))
        println(io, "# merged_partition=" * string(merged))
        println(io, "beta,J_merge_minus_split,theorem_rhs,preferred")
        for beta in [0.5 * beta_star, beta_star, 1.5 * beta_star]
            lhs = regularized_objective(fe, merged, beta) -
                  regularized_objective(fe, full, beta)
            rhs = split_threshold_rhs(fe, [i], [j], beta)
            pref = lhs > TOL ? "split" : (lhs < -TOL ? "merge" : "tie")
            println(io, string(beta) * "," * string(lhs) * "," * string(rhs) * "," * pref)
        end
        println(io, "")
        println(io, "beta,k,excess,complexity,labels")
        for beta in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
            labels, _ = best_partition(fe, partitions, beta)
            println(io, string(beta) * "," *
                        string(n_classes(labels)) * "," *
                        string(excess_loss(fe, labels)) * "," *
                        string(partition_complexity(fe, labels)) * "," *
                        "\"" * string(labels) * "\"")
        end
    end
    println("Results saved to " * outpath)
end

main()
