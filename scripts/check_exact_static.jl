"""
Exact finite static checks — the minimal first deliverable.

For n_w ≤ 5, exhaustively checks:
  1. thm:ce-decomposition  (excess = JS excess, identity)
  2. thm:llm-veridicality  (zero excess ⟺ no merged separated pair)
  3. thm:min-complexity     (min-complexity zero-excess = ecology partition)
  4. thm:split-threshold    (J(p) - J(p^{A|B}) = π_C (Δ_pred - β h(λ)))

Run:  julia --project=. scripts/check_exact_static.jl
"""

using Random
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using LlmPaper

const TOL = 1e-10

# ── Reporting ────────────────────────────────────────────────────────────────

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

# ── Check 1: thm:ce-decomposition ───────────────────────────────────────────
#
# For every partition of W:
#   (a) L*_D(p) - H(V|C,W) == JS excess  (identity)
#   (b) excess == 0  ⟺  every cell contains only ecology-equivalent states

function check_ce_decomposition(fe::FiniteEcology, partitions; seed=0)
    cr = CheckResult("thm_ce_decomposition")
    h_vcw = conditional_entropy_VCW(fe)
    eco_labels = ecology_partition(fe)

    for labels in partitions
        # (a) Identity: excess via direct loss minus entropy = JS excess
        excess_direct = bayes_opt_loss(fe, labels) - h_vcw
        excess_js = js_excess(fe, labels)
        gap_a = excess_direct - excess_js
        record!(cr, gap_a, "identity gap, seed=" * string(seed) *
                " labels=" * string(labels))

        # (b) Zero-excess characterization
        has_merged_sep = merges_separated_pair(fe, labels)
        is_zero = excess_direct < TOL

        if is_zero && has_merged_sep
            # Zero excess but merges a separated pair: theorem violated
            record!(cr, 1.0, "zero excess but merges separated pair, labels=" *
                    string(labels))
        elseif !is_zero && !has_merged_sep
            # Nonzero excess but no merged separated pair: theorem violated
            record!(cr, excess_direct, "nonzero excess=" *
                    string(excess_direct) * " but no merged separated pair, labels=" *
                    string(labels))
        else
            record!(cr, 0.0)
        end
    end
    cr
end

# ── Check 2: thm:llm-veridicality ───────────────────────────────────────────
#
# Among all partitions:
#   (a) minimizers of L*_D(p) have zero excess iff they merge no separated pair
#   (b) if all partitions merge some separated pair, inf L*_D > H(V|C,W)

function check_llm_veridicality(fe::FiniteEcology, partitions; seed=0)
    cr = CheckResult("thm_llm_veridicality")
    h_vcw = conditional_entropy_VCW(fe)

    losses = [bayes_opt_loss(fe, lab) for lab in partitions]
    min_loss = minimum(losses)

    # Find all minimizers
    for (i, lab) in enumerate(partitions)
        if abs(losses[i] - min_loss) < TOL
            # This is a minimizer
            has_merged = merges_separated_pair(fe, lab)
            is_at_floor = abs(min_loss - h_vcw) < TOL

            if is_at_floor && has_merged
                record!(cr, 1.0, "minimizer at floor but merges separated, seed=" *
                        string(seed))
            elseif is_at_floor && !has_merged
                record!(cr, 0.0)  # correct: at floor and no merged sep pair
            elseif !is_at_floor
                record!(cr, 0.0)  # correct: not at floor (model class too small or all merge)
            end
        end
    end

    # Check: if every partition merges a separated pair, min loss > floor
    all_merge = all(merges_separated_pair(fe, lab) for lab in partitions)
    if all_merge
        gap = h_vcw - min_loss  # should be negative (min_loss > h_vcw)
        if gap > -TOL
            # min_loss <= h_vcw despite all partitions merging: violation
            record!(cr, gap + TOL, "all partitions merge but min_loss <= H(V|C,W)")
        else
            record!(cr, 0.0)
        end
    else
        # There exists a non-merging partition; check it attains the floor
        non_merging_losses = [losses[i] for (i, lab) in enumerate(partitions)
                              if !merges_separated_pair(fe, lab)]
        best_non_merging = minimum(non_merging_losses)
        gap = abs(best_non_merging - h_vcw)
        record!(cr, gap, "best non-merging partition loss vs H(V|C,W), seed=" *
                string(seed))
    end

    cr
end

# ── Check 3: thm:min-complexity ──────────────────────────────────────────────
#
# Among zero-excess partitions:
#   (a) the minimum I(W;p(W)) is H(W/∼_μ)
#   (b) achieved exactly by the ecology partition
#   (c) any finer zero-excess partition has strictly larger entropy

function check_min_complexity(fe::FiniteEcology, partitions; seed=0)
    cr = CheckResult("thm_min_complexity")
    eco_labels = ecology_partition(fe)
    eco_complexity = partition_complexity(fe, eco_labels)

    # Find all zero-excess partitions
    zero_excess = [(lab, partition_complexity(fe, lab))
                   for lab in partitions
                   if is_zero_excess(fe, lab)]

    if isempty(zero_excess)
        # No zero-excess partition exists (all merge some separated pair)
        # This is fine — the theorem is vacuously true
        record!(cr, 0.0)
        return cr
    end

    min_complexity = minimum(c for (_, c) in zero_excess)

    # (a) Minimum equals ecology partition complexity
    gap_a = abs(min_complexity - eco_complexity)
    record!(cr, gap_a, "min zero-excess complexity vs ecology complexity, seed=" *
            string(seed))

    # (b) Ecology partition itself is zero-excess and achieves the minimum
    eco_is_zero = is_zero_excess(fe, eco_labels)
    if !eco_is_zero
        record!(cr, 1.0, "ecology partition is not zero-excess, seed=" * string(seed))
    else
        record!(cr, 0.0)
    end

    # (c) All zero-excess partitions that achieve minimum have same partition
    for (lab, comp) in zero_excess
        if abs(comp - min_complexity) < TOL
            if !same_partition(lab, eco_labels)
                record!(cr, 1.0, "min-complexity zero-excess not same as ecology, " *
                        "labels=" * string(lab) * " seed=" * string(seed))
            else
                record!(cr, 0.0)
            end
        else
            # Finer zero-excess: should have strictly larger complexity
            if comp <= eco_complexity + TOL
                record!(cr, eco_complexity - comp + TOL,
                        "finer zero-excess not strictly larger complexity, seed=" *
                        string(seed))
            else
                record!(cr, 0.0)
            end
        end
    end

    cr
end

# ── Check 4: thm:split-threshold ────────────────────────────────────────────
#
# For each partition and each cell, for each binary split A|B:
#   J_{D,β}(p) - J_{D,β}(p^{A|B}) == π_C (Δ_pred(A,B) - β h(λ))

function check_split_threshold(fe::FiniteEcology, partitions; seed=0,
                               betas=[0.0, 0.01, 0.1, 0.5, 1.0, 5.0])
    cr = CheckResult("thm_split_threshold")

    for labels in partitions
        cells = partition_cells(labels)
        for (cell_idx, cell) in enumerate(cells)
            length(cell) < 2 && continue
            splits = all_binary_splits(cell)
            for (A, B) in splits, beta in betas
                # Direct computation: J(p) - J(p^{A|B})
                # Build the refined partition
                labels_refined = copy(labels)
                new_label = maximum(labels) + 1
                for w in B
                    labels_refined[w] = new_label
                end

                j_before = regularized_objective(fe, labels, beta)
                j_after  = regularized_objective(fe, labels_refined, beta)
                lhs = j_before - j_after

                # Theorem RHS
                rhs = split_threshold_rhs(fe, A, B, beta)

                gap = lhs - rhs
                record!(cr, gap, "split gap, beta=" * string(beta) *
                        " cell=" * string(cell) * " A=" * string(A) *
                        " B=" * string(B) * " seed=" * string(seed))
            end
        end
    end

    cr
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    println("=" ^ 60)
    println("Exact finite static checks")
    println("=" ^ 60)
    println()

    n_random = 20    # random ecologies per size
    sizes = [(3, 2, 2), (3, 3, 3), (4, 2, 2), (4, 3, 3), (5, 2, 2)]

    all_results = CheckResult[]

    for (n_w, n_c, n_v) in sizes
        println("--- n_w=" * string(n_w) * " n_c=" * string(n_c) *
                " n_v=" * string(n_v) * " ---")

        parts = all_partitions(n_w)
        println("  partitions: " * string(length(parts)))

        for trial in 1:n_random
            seed = 1000 * n_w + 100 * n_c + trial
            rng = MersenneTwister(seed)
            fe = random_ecology(rng, n_w, n_c, n_v)

            r1 = check_ce_decomposition(fe, parts; seed=seed)
            r2 = check_llm_veridicality(fe, parts; seed=seed)
            r3 = check_min_complexity(fe, parts; seed=seed)
            r4 = check_split_threshold(fe, parts; seed=seed)

            append!(all_results, [r1, r2, r3, r4])
        end

        # Also test structured ecologies
        for trial in 1:5
            seed = 2000 * n_w + trial
            rng = MersenneTwister(seed)

            # Merged pair ecology
            fe_merged = ecology_with_merged_pair(rng, n_w, n_c, n_v)
            r1 = check_ce_decomposition(fe_merged, parts; seed=seed)
            r3 = check_min_complexity(fe_merged, parts; seed=seed)
            append!(all_results, [r1, r3])

            # Partial separation (differs on one context only)
            if n_c >= 2
                rng2 = MersenneTwister(seed + 10000)
                fe_partial = ecology_with_partial_separation(rng2, n_w, n_c, n_v)
                r1p = check_ce_decomposition(fe_partial, parts; seed=seed + 10000)
                push!(all_results, r1p)
            end
        end
    end

    # Aggregate results per check name
    println()
    println("=" ^ 60)
    println("SUMMARY")
    println("=" ^ 60)
    println()

    for name in ["thm_ce_decomposition", "thm_llm_veridicality",
                  "thm_min_complexity", "thm_split_threshold"]
        relevant = filter(r -> r.name == name, all_results)
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
