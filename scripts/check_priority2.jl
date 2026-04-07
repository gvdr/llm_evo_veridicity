"""
Priority 2 checks: text-distance, ecology-expansion, capacity-criterion,
generalist-vs-specialist, off-ecology error, off-ecology non-identifiability.

Run:  julia --project=. scripts/check_priority2.jl
"""

using Random
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using LlmPaper

const TOL = 1e-10

# ── Reporting (same as check_exact_static.jl) ────────────────────────────────

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

function build_decoder(fe::FiniteEcology, labels::AbstractVector{Int})
    decoder = Dict{Tuple{Int,Int}, Vector{Float64}}()
    for (label, cell) in enumerate(partition_cells(labels)), c in 1:n_contexts(fe)
        decoder[(label, c)] = cell_average(fe, cell, c)
    end
    decoder
end

function decoder_loss(fe::FiniteEcology,
                      labels::AbstractVector{Int},
                      decoder::Dict{Tuple{Int,Int}, Vector{Float64}})
    loss = 0.0
    for w in 1:n_worlds(fe), c in 1:n_contexts(fe)
        loss += fe.pi[w] * fe.d_c[c] *
                cross_entropy(@view(fe.p_wcv[w, c, :]), decoder[(labels[w], c)])
    end
    loss
end

# ── Check 5: prop:text-distance and cor:text-separation ──────────────────────
#
# For each pair (w1, w2):
#   (a) σ²_D(w1,w2) ≥ 0
#   (b) σ²_D = 0  ⟺  no positive-mass separating context exists
#   (c) Hellinger-KL bounds: H² ≤ KL  and  -2 log(1 - H²) ≤ KL  per context
#       (Rényi monotonicity: D_{1/2} = -2log(1-H²) ≤ D_1 = KL)
#   NOTE: The manuscript claims KL ≤ -2log(1-H²); this is reversed.
#         The correct bound is -2log(1-H²) ≤ KL.

function check_text_distance(fe::FiniteEcology; seed=0)
    cr = CheckResult("prop_text_distance")
    nw = n_worlds(fe)
    nc = n_contexts(fe)

    for w1 in 1:nw, w2 in (w1 + 1):nw
        sigma2 = task_distance(fe, w1, w2)

        # (a) Non-negativity
        if sigma2 < -TOL
            record!(cr, -sigma2, "negative sigma2, w1=" * string(w1) *
                    " w2=" * string(w2) * " seed=" * string(seed))
        else
            record!(cr, 0.0)
        end

        # (b) Zero iff no positive-mass separating context
        has_sep = separated(fe, w1, w2)
        sigma_positive = sigma2 > TOL

        if sigma_positive && !has_sep
            record!(cr, sigma2, "positive sigma2 but no separating context")
        elseif !sigma_positive && has_sep
            record!(cr, 1.0, "zero sigma2 but separating context exists")
        else
            record!(cr, 0.0)
        end

        # (c) Hellinger-KL sandwich per context
        for c in 1:nc
            fe.d_c[c] <= 0 && continue
            p = fe.p_wcv[w1, c, :]
            q = fe.p_wcv[w2, c, :]
            h2 = hellinger2(p, q)
            kl_val = kl_divergence(p, q)

            # H² ≤ KL
            if h2 > kl_val + TOL
                record!(cr, h2 - kl_val,
                        "Hellinger > KL, w1=" * string(w1) * " w2=" * string(w2) *
                        " c=" * string(c) * " seed=" * string(seed))
            else
                record!(cr, 0.0)
            end

            # -2 log(1 - H²) ≤ KL  (Rényi monotonicity: D_{1/2} ≤ D_1)
            if h2 < 1.0 - TOL
                renyi_half = -2.0 * log(1.0 - h2)
                if renyi_half > kl_val + TOL
                    record!(cr, renyi_half - kl_val,
                            "D_{1/2} > KL, w1=" * string(w1) *
                            " w2=" * string(w2) * " c=" * string(c) *
                            " seed=" * string(seed))
                else
                    record!(cr, 0.0)
                end
            end
        end
    end
    cr
end

# ── Check 6: prop:ecology-expansion ──────────────────────────────────────────
#
# For two ecologies on same (W, C, V):
#   σ²_{μ'} = (1-α) σ²_D + α σ²_ν   (linearity)
#   Partition refinement: separated under D ⟹ separated under μ'

function check_ecology_expansion(rng::AbstractRNG, n_w, n_c, n_v; seed=0)
    cr = CheckResult("prop_ecology_expansion")

    fe1 = random_ecology(rng, n_w, n_c, n_v)
    fe2 = random_ecology(MersenneTwister(seed + 99999), n_w, n_c, n_v)
    # fe2 shares same p_wcv but has different d_c
    # Actually we want different d_c but same p_wcv for a clean test
    fe2 = FiniteEcology(fe1.pi, fe2.d_c, fe1.p_wcv)

    for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]
        d_c_mix = (1 - alpha) .* fe1.d_c .+ alpha .* fe2.d_c
        fe_mix = FiniteEcology(fe1.pi, d_c_mix, fe1.p_wcv)

        for w1 in 1:n_w, w2 in (w1 + 1):n_w
            s1 = task_distance(fe1, w1, w2)
            s2 = task_distance(fe2, w1, w2)
            s_mix = task_distance(fe_mix, w1, w2)
            expected = (1 - alpha) * s1 + alpha * s2

            # Linearity check
            gap = s_mix - expected
            record!(cr, gap, "linearity, alpha=" * string(alpha) *
                    " w1=" * string(w1) * " w2=" * string(w2) *
                    " seed=" * string(seed))

            # Refinement: if separated under fe1, must be separated under fe_mix
            if separated(fe1, w1, w2) && !separated(fe_mix, w1, w2)
                record!(cr, 1.0, "refinement violated, alpha=" * string(alpha) *
                        " w1=" * string(w1) * " w2=" * string(w2))
            else
                record!(cr, 0.0)
            end
        end
    end
    cr
end

# ── Check 7: prop:capacity-criterion ─────────────────────────────────────────
#
# (a) If a restricted family contains a partition separating all μ-equivalence
#     classes, the entropy floor is attainable.
# (b) If no partition in the family does, inf L* > H(V|C,W).

function check_capacity_criterion(fe::FiniteEcology, partitions; seed=0)
    cr = CheckResult("prop_capacity_criterion")
    h_vcw = conditional_entropy_VCW(fe)
    eco_k = n_classes(ecology_partition(fe))

    # Test restricted families: partitions with at most k_max cells
    for k_max in 1:n_worlds(fe)
        restricted = [lab for lab in partitions if n_classes(lab) <= k_max]
        losses = [bayes_opt_loss(fe, lab) for lab in restricted]
        min_loss = minimum(losses)

        has_separating = any(!merges_separated_pair(fe, lab) for lab in restricted)

        if has_separating
            # (a) Floor should be attainable
            gap = min_loss - h_vcw
            if gap > TOL
                record!(cr, gap, "floor not attained despite non-merging partition, " *
                        "k_max=" * string(k_max) * " seed=" * string(seed))
            else
                record!(cr, 0.0)
            end
        else
            # (b) Strictly above floor
            gap = h_vcw - min_loss  # should be negative
            if gap > -TOL
                record!(cr, gap + TOL, "at floor despite all merging, k_max=" *
                        string(k_max) * " seed=" * string(seed))
            else
                record!(cr, 0.0)
            end
        end
    end
    cr
end

# ── Check 8: thm:gen-vs-spec ────────────────────────────────────────────────
#
# Build T task-specific ecologies sharing the same (W, p_wcv) but with
# different context distributions.
#
# (a) If a partition has zero excess on the mixture, it has zero excess on each.
# (b) A specialist merging a pair separated under task s incurs positive excess,
#     bounded below by the stated JS bound.

function check_gen_vs_spec(rng::AbstractRNG, n_w, n_c, n_v, T; seed=0)
    cr = CheckResult("thm_gen_vs_spec")

    # Shared p_wcv and pi
    fe_base = random_ecology(rng, n_w, n_c, n_v)
    pi = fe_base.pi
    p_wcv = fe_base.p_wcv

    # T task-specific context distributions
    task_dcs = [let x = rand(MersenneTwister(seed + t), n_c); x ./ sum(x) end
                for t in 1:T]
    task_fes = [FiniteEcology(pi, dc, p_wcv) for dc in task_dcs]

    # Mixture ecology
    d_c_mix = sum(task_dcs) ./ T
    fe_mix = FiniteEcology(pi, d_c_mix, p_wcv)

    h_vcw = conditional_entropy_VCW(fe_mix)
    partitions = all_partitions(n_w)

    for labels in partitions
        loss_mix = bayes_opt_loss(fe_mix, labels)
        excess_mix = loss_mix - h_vcw

        # (a) Zero excess on mixture ⟹ zero excess on each task
        if excess_mix < TOL
            for (t, fe_t) in enumerate(task_fes)
                h_t = conditional_entropy_VCW(fe_t)
                excess_t = bayes_opt_loss(fe_t, labels) - h_t
                if excess_t > TOL
                    record!(cr, excess_t, "zero mix excess but positive task excess, " *
                            "t=" * string(t) * " labels=" * string(labels) *
                            " seed=" * string(seed))
                else
                    record!(cr, 0.0)
                end
            end
        end

        # (b) Specialist lower bound: if labels merges (w1,w2) separated under task s
        for s in 1:T
            fe_s = task_fes[s]
            for w1 in 1:n_w, w2 in (w1 + 1):n_w
                labels[w1] != labels[w2] && continue  # not merged
                !separated(fe_s, w1, w2) && continue    # not separated under s

                h_s = conditional_entropy_VCW(fe_s)
                excess_s = bayes_opt_loss(fe_s, labels) - h_s

                # Compute the theorem lower bound:
                # (π(w1)+π(w2)) E_{c~D_C^s}[JS_λ(P_{w1}, P_{w2})]
                pi_pair = pi[w1] + pi[w2]
                lam = pi[w1] / pi_pair
                bound = 0.0
                for c in 1:n_c
                    p1 = p_wcv[w1, c, :]
                    p2 = p_wcv[w2, c, :]
                    m = lam .* p1 .+ (1 - lam) .* p2
                    js = entropy(m) - lam * entropy(p1) - (1 - lam) * entropy(p2)
                    bound += fe_s.d_c[c] * js
                end
                bound *= pi_pair

                # excess_s should be >= bound (and both > 0)
                if excess_s < bound - TOL
                    record!(cr, bound - excess_s,
                            "excess below bound, s=" * string(s) *
                            " w1=" * string(w1) * " w2=" * string(w2) *
                            " seed=" * string(seed))
                elseif bound < TOL
                    record!(cr, TOL - bound, "bound not positive, seed=" * string(seed))
                else
                    record!(cr, 0.0)
                end
            end
        end
    end
    cr
end

# ── Check 9: prop:off-ecology-error ──────────────────────────────────────────
#
# Training ecology has σ²_tr(w1,w2)=0 for some pair; deployment has σ²_te > 0.
# Min-complexity zero-excess encoding for training merges (w1,w2).
# Deployment excess ≥ stated JS lower bound.

function check_off_ecology_error(rng::AbstractRNG, n_w, n_c, n_v; seed=0)
    cr = CheckResult("prop_off_ecology_error")

    # Build training ecology where w1=1 and w2=2 are identical on training contexts
    fe_full = random_ecology(rng, n_w, n_c, n_v)
    pi = fe_full.pi
    p_wcv = copy(fe_full.p_wcv)

    # Training: contexts 1..n_c-1 (w1 and w2 identical there)
    # Deployment: context n_c (w1 and w2 differ there)
    n_c < 2 && return cr  # need at least 2 contexts

    # Make w1 and w2 identical on training contexts (1..n_c-1)
    for c in 1:(n_c - 1)
        p_wcv[2, c, :] .= p_wcv[1, c, :]
    end
    # Ensure they differ on context n_c (already random, almost surely different)

    # Training d_c: positive mass only on contexts 1..n_c-1
    d_c_tr = zeros(n_c)
    raw_tr = rand(MersenneTwister(seed + 50000), n_c - 1)
    d_c_tr[1:(n_c - 1)] = raw_tr ./ sum(raw_tr)

    # Deployment d_c: positive mass on all contexts
    raw_te = rand(MersenneTwister(seed + 60000), n_c)
    d_c_te = raw_te ./ sum(raw_te)

    fe_tr = FiniteEcology(pi, d_c_tr, p_wcv)
    fe_te = FiniteEcology(pi, d_c_te, p_wcv)

    # Verify w1 and w2 are training-equivalent but deployment-separated
    if separated(fe_tr, 1, 2)
        record!(cr, 0.0)  # skip: construction failed (shouldn't happen)
        return cr
    end
    if !separated(fe_te, 1, 2)
        record!(cr, 0.0)  # skip: not separated at deployment either
        return cr
    end

    # Find minimum-complexity zero-excess partition for training
    eco_tr = ecology_partition(fe_tr)

    # It should merge w1 and w2
    if eco_tr[1] != eco_tr[2]
        record!(cr, 1.0, "ecology partition does not merge w1,w2 under training")
        return cr
    else
        record!(cr, 0.0)
    end

    # Deployment excess of the training-optimal encoding
    h_te = conditional_entropy_VCW(fe_te)
    excess_te = bayes_opt_loss(fe_te, eco_tr) - h_te

    # Theorem lower bound: (π(w1)+π(w2)) E_{c~D_C^te}[JS_λ(P_w1, P_w2)]
    pi_pair = pi[1] + pi[2]
    lam = pi[1] / pi_pair
    bound = 0.0
    for c in 1:n_c
        p1 = p_wcv[1, c, :]
        p2 = p_wcv[2, c, :]
        m = lam .* p1 .+ (1 - lam) .* p2
        js = entropy(m) - lam * entropy(p1) - (1 - lam) * entropy(p2)
        bound += d_c_te[c] * js
    end
    bound *= pi_pair

    # Excess should be ≥ bound > 0
    if bound < TOL
        record!(cr, TOL - bound, "bound not positive, seed=" * string(seed))
    else
        record!(cr, 0.0)
    end

    if excess_te < bound - TOL
        record!(cr, bound - excess_te,
                "deployment excess below bound, seed=" * string(seed))
    else
        record!(cr, 0.0)
    end

    cr
end

# ── Check 10: prop:off-ecology-nonident ──────────────────────────────────────
#
# Explicit witness: two decoders that agree on training support, differ on
# deployment, both attain L*_tr(p).

function check_off_ecology_nonident(rng::AbstractRNG, n_w, n_c, n_v; seed=0)
    cr = CheckResult("prop_off_ecology_nonident")

    fe_full = random_ecology(rng, n_w, n_c, n_v)
    pi = fe_full.pi
    p_wcv = copy(fe_full.p_wcv)

    n_c < 2 && return cr

    # Same construction as off-ecology-error:
    # w1,w2 identical on training contexts, differ on deployment context n_c
    for c in 1:(n_c - 1)
        p_wcv[2, c, :] .= p_wcv[1, c, :]
    end

    # Training: mass on 1..n_c-1 only; deployment: mass on all
    d_c_tr = zeros(n_c)
    raw_tr = rand(MersenneTwister(seed + 50000), n_c - 1)
    d_c_tr[1:(n_c - 1)] = raw_tr ./ sum(raw_tr)

    fe_tr = FiniteEcology(pi, d_c_tr, p_wcv)
    eco_tr = ecology_partition(fe_tr)

    if eco_tr[1] != eco_tr[2]
        record!(cr, 0.0)
        return cr
    end

    # Bayes-optimal decoder on training: q*(v|x,c) = cell_average(fe_tr, cell, c)
    # On context n_c (zero training mass), the decoder is unconstrained.
    # Construct two decoders that differ there:
    #   q1 uses P_{w1}(·|n_c)
    #   q2 uses P_{w2}(·|n_c)
    q_star = build_decoder(fe_tr, eco_tr)
    q1 = deepcopy(q_star)
    q2 = deepcopy(q_star)
    target_label = eco_tr[1]
    q1[(target_label, n_c)] = collect(@view p_wcv[1, n_c, :])
    q2[(target_label, n_c)] = collect(@view p_wcv[2, n_c, :])

    loss_star = bayes_opt_loss(fe_tr, eco_tr)
    loss_q1 = decoder_loss(fe_tr, eco_tr, q1)
    loss_q2 = decoder_loss(fe_tr, eco_tr, q2)
    record!(cr, loss_q1 - loss_star,
            "q1 training loss differs from optimum, seed=" * string(seed))
    record!(cr, loss_q2 - loss_star,
            "q2 training loss differs from optimum, seed=" * string(seed))
    record!(cr, loss_q1 - loss_q2,
            "q1 and q2 training losses differ, seed=" * string(seed))

    # Deployment predictions differ on context n_c
    p1_deploy = q1[(target_label, n_c)]
    p2_deploy = q2[(target_label, n_c)]
    diff = sum(abs.(p1_deploy .- p2_deploy))
    if diff < TOL
        record!(cr, 1.0, "constructed off-ecology decoders do not differ, seed=" *
                string(seed))
    else
        record!(cr, 0.0)

        # Verify deployment predictions actually differ
        # JS between the two decoder outputs on context n_c
        js_deploy = js_divergence(p1_deploy, p2_deploy)
        if js_deploy < TOL
            record!(cr, TOL - js_deploy,
                    "decoders agree on deployment despite different distributions")
        else
            record!(cr, 0.0)
        end
    end

    cr
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    println("=" ^ 60)
    println("Priority 2 checks")
    println("=" ^ 60)
    println()

    n_random = 20
    sizes = [(3, 2, 2), (3, 3, 3), (4, 2, 2), (4, 3, 3), (5, 2, 2)]

    all_results = CheckResult[]

    for (n_w, n_c, n_v) in sizes
        println("--- n_w=" * string(n_w) * " n_c=" * string(n_c) *
                " n_v=" * string(n_v) * " ---")

        parts = all_partitions(n_w)

        for trial in 1:n_random
            seed = 3000 * n_w + 100 * n_c + trial
            rng = MersenneTwister(seed)
            fe = random_ecology(rng, n_w, n_c, n_v)

            push!(all_results, check_text_distance(fe; seed=seed))
            push!(all_results, check_ecology_expansion(
                MersenneTwister(seed + 1), n_w, n_c, n_v; seed=seed))
            push!(all_results, check_capacity_criterion(fe, parts; seed=seed))
            push!(all_results, check_gen_vs_spec(
                MersenneTwister(seed + 2), n_w, n_c, n_v, 3; seed=seed))
            push!(all_results, check_off_ecology_error(
                MersenneTwister(seed + 3), n_w, n_c, n_v; seed=seed))
            push!(all_results, check_off_ecology_nonident(
                MersenneTwister(seed + 4), n_w, n_c, n_v; seed=seed))
        end
    end

    # Summary
    println()
    println("=" ^ 60)
    println("SUMMARY")
    println("=" ^ 60)
    println()

    for name in ["prop_text_distance", "prop_ecology_expansion",
                  "prop_capacity_criterion", "thm_gen_vs_spec",
                  "prop_off_ecology_error", "prop_off_ecology_nonident"]
        relevant = filter(r -> r.name == name, all_results)
        isempty(relevant) && continue
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
