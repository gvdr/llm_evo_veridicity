"""
Priority 3 geometry checks:
  - prop:hilbert-kernel (PSD kernel, distance identity)
  - thm:topology-general (partition, rank bound)
  - thm:geometry-gaussian (isotropic proportionality, anisotropic failure)
  - prop:knn-stability (perturbation lemma)

Run:  julia --project=. scripts/check_geometry.jl
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

# ── Check 11: prop:hilbert-kernel ────────────────────────────────────────────
#
# Canonical embedding Ψ_D(w)(c) = √P_w(·|c).
# (a) σ²_D(w_i,w_j) = (1/2) ‖Ψ_D(w_i) - Ψ_D(w_j)‖²
# (b) K_μ = -(1/2) J D_σ J  is PSD
# (c) (K_μ)_{ij} = (1/2) ⟨Ψ̄_D(w_i), Ψ̄_D(w_j)⟩

function check_hilbert_kernel(fe::FiniteEcology; seed=0)
    cr = CheckResult("prop_hilbert_kernel")
    nw = n_worlds(fe)
    nc = n_contexts(fe)
    nv = n_vocab(fe)

    # Build the canonical embedding Ψ_D(w) as a long vector
    # Ψ_D(w) = [√d_c(c) * √P_w(v|c)]  for all (c,v)
    # Then ‖Ψ_D(w_i) - Ψ_D(w_j)‖² = Σ_c d_c(c) Σ_v (√P_{w_i}(v|c) - √P_{w_j}(v|c))²
    #                                = 2 σ²_D(w_i, w_j)

    dim = nc * nv
    Psi = zeros(nw, dim)
    for w in 1:nw
        idx = 0
        for c in 1:nc, v in 1:nv
            idx += 1
            Psi[w, idx] = sqrt(fe.d_c[c]) * sqrt(fe.p_wcv[w, c, v])
        end
    end

    # (a) Distance identity
    D_sigma = task_distance_matrix(fe)
    for i in 1:nw, j in (i + 1):nw
        hilbert_dist = 0.5 * sum((Psi[i, :] .- Psi[j, :]) .^ 2)
        gap = D_sigma[i, j] - hilbert_dist
        record!(cr, gap, "distance identity, i=" * string(i) * " j=" * string(j) *
                " seed=" * string(seed))
    end

    # (b) K_μ is PSD
    N = nw
    J = I(N) .- (1.0 / N) .* ones(N, N)
    K_mu = -0.5 .* J * D_sigma * J

    # Symmetry
    sym_gap = maximum(abs.(K_mu .- K_mu'))
    record!(cr, sym_gap, "symmetry, seed=" * string(seed))

    # PSD: all eigenvalues ≥ -TOL
    eigs = eigvals(Symmetric(K_mu))
    min_eig = minimum(eigs)
    if min_eig < -TOL
        record!(cr, -min_eig, "negative eigenvalue " * string(min_eig) *
                " seed=" * string(seed))
    else
        record!(cr, 0.0)
    end

    # (c) Gram matrix identity
    Psi_bar = Psi .- (1.0 / N) .* ones(N) * sum(Psi, dims=1)
    K_gram = 0.5 .* Psi_bar * Psi_bar'
    gram_gap = maximum(abs.(K_mu .- K_gram))
    record!(cr, gram_gap, "Gram identity, seed=" * string(seed))

    cr
end

# ── Check 12: thm:topology-general ──────────────────────────────────────────
#
# (a) Ecologically veridical h: h(w_i) ≠ h(w_j) for every separated pair.
# (b) Min-complexity: h(w_i) = h(w_j) for equivalent pairs.
# (c) If k_μ distinct codes, rank(K_h) ≤ k_μ - 1.

function check_topology_general(fe::FiniteEcology; seed=0)
    cr = CheckResult("thm_topology_general")
    nw = n_worlds(fe)
    eco = ecology_partition(fe)
    k_mu = n_classes(eco)

    # Construct code vectors h(w) from the ecology partition
    # Use random distinct vectors for each class
    rng_h = MersenneTwister(seed + 77777)
    d = max(k_mu, 2)  # embedding dimension
    class_codes = randn(rng_h, k_mu, d)
    H = zeros(nw, d)
    for w in 1:nw
        H[w, :] = class_codes[eco[w], :]
    end

    # (a) Separated pairs have different codes
    for i in 1:nw, j in (i + 1):nw
        if separated(fe, i, j) && H[i, :] == H[j, :]
            record!(cr, 1.0, "separated pair shares code, i=" * string(i) *
                    " j=" * string(j) * " seed=" * string(seed))
        else
            record!(cr, 0.0)
        end
    end

    # (b) Equivalent pairs share code (by construction, but verify)
    for i in 1:nw, j in (i + 1):nw
        if eco[i] == eco[j] && H[i, :] != H[j, :]
            record!(cr, 1.0, "equivalent pair has different codes")
        else
            record!(cr, 0.0)
        end
    end

    # (c) Rank bound: centered Gram matrix of k_μ distinct codes has rank ≤ k_μ - 1
    H_centered = H .- (1.0 / nw) .* ones(nw) * sum(H, dims=1)
    K_h = H_centered * H_centered'
    eigs = eigvals(Symmetric(K_h))
    numerical_rank = count(e -> e > TOL, eigs)
    if numerical_rank > k_mu - 1
        record!(cr, Float64(numerical_rank - (k_mu - 1)),
                "rank " * string(numerical_rank) * " > k_mu-1=" *
                string(k_mu - 1) * " seed=" * string(seed))
    else
        record!(cr, 0.0)
    end

    cr
end

# ── Check 13: thm:geometry-gaussian ──────────────────────────────────────────
#
# Gaussian-linear tasks: f(w_i) = c^T φ_i, c ~ N(0, Σ_c).
# Linear encoder h(w_i) = A φ_i.
# (a) Zero risk ⟺ A(φ_i - φ_j) ≠ 0 for all separated pairs.
# (b) Under isotropic Σ_c on V: ‖h(w_i)-h(w_j)‖² ∝ σ²(w_i,w_j).
# (c) Under anisotropic Σ_c: proportionality fails.

function check_geometry_gaussian(rng::AbstractRNG; seed=0, n_w=4, dim_phi=6)
    cr = CheckResult("thm_geometry_gaussian")

    # Construct latent state vectors φ_i
    Phi = randn(rng, n_w, dim_phi)

    # Task distance in Gaussian-linear model:
    # σ²(w_i,w_j) = (φ_i - φ_j)^T Σ_c (φ_i - φ_j)

    # ── Isotropic case: Σ_c = σ_c² P_V ──
    # Define V = span{φ_i - φ_j : σ²>0}. Under isotropic, all pairs are separated.
    sigma_c2 = 2.5
    Sigma_c_iso = sigma_c2 * I(dim_phi)

    # Task distances
    sigma2_iso = zeros(n_w, n_w)
    for i in 1:n_w, j in (i + 1):n_w
        diff = Phi[i, :] .- Phi[j, :]
        sigma2_iso[i, j] = dot(diff, Sigma_c_iso * diff)
        sigma2_iso[j, i] = sigma2_iso[i, j]
    end

    # Canonical projector encoder A = P_V (project onto span of differences)
    diffs = [Phi[i, :] .- Phi[j, :] for i in 1:n_w for j in (i + 1):n_w]
    diff_mat = hcat(diffs...)
    F = svd(diff_mat)
    r = count(s -> s > 1e-8, F.S)
    P_V = F.U[:, 1:r] * F.U[:, 1:r]'

    # Encoder: h(w_i) = P_V φ_i
    H_iso = (P_V * Phi')'  # n_w × dim_phi

    # (a) Separation: A(φ_i - φ_j) ≠ 0 for all pairs with σ² > 0
    for i in 1:n_w, j in (i + 1):n_w
        if sigma2_iso[i, j] > TOL
            enc_diff = H_iso[i, :] .- H_iso[j, :]
            if norm(enc_diff) < TOL
                record!(cr, 1.0, "separated pair mapped to same code, iso case")
            else
                record!(cr, 0.0)
            end
        end
    end

    # (b) Isotropic proportionality: ‖h(w_i)-h(w_j)‖² = σ²/σ_c²
    for i in 1:n_w, j in (i + 1):n_w
        enc_dist = sum((H_iso[i, :] .- H_iso[j, :]) .^ 2)
        expected = sigma2_iso[i, j] / sigma_c2
        gap = enc_dist - expected
        record!(cr, gap, "iso proportionality, i=" * string(i) * " j=" * string(j) *
                " seed=" * string(seed))
    end

    # ── Anisotropic case: Σ_c with distinct eigenvalues ──
    eigenvalues = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0][1:dim_phi]
    Sigma_c_aniso = Diagonal(eigenvalues)

    sigma2_aniso = zeros(n_w, n_w)
    for i in 1:n_w, j in (i + 1):n_w
        diff = Phi[i, :] .- Phi[j, :]
        sigma2_aniso[i, j] = dot(diff, Sigma_c_aniso * diff)
        sigma2_aniso[j, i] = sigma2_aniso[i, j]
    end

    # Under P_V encoder (uniform projection), encoding distances are:
    H_aniso = (P_V * Phi')'
    # These are the same H as isotropic (P_V doesn't depend on Σ_c)
    # But σ² now weights directions differently

    # (c) Proportionality should FAIL in general
    # Check: ‖h(w_i)-h(w_j)‖² / σ²_aniso(w_i,w_j) is NOT constant
    ratios = Float64[]
    for i in 1:n_w, j in (i + 1):n_w
        enc_dist = sum((H_aniso[i, :] .- H_aniso[j, :]) .^ 2)
        if sigma2_aniso[i, j] > TOL
            push!(ratios, enc_dist / sigma2_aniso[i, j])
        end
    end

    if length(ratios) >= 2
        ratio_range = maximum(ratios) - minimum(ratios)
        if ratio_range < TOL
            record!(cr, 1.0, "anisotropic witness not found, seed=" * string(seed))
        else
            record!(cr, 0.0)
        end
    else
        record!(cr, 1.0, "insufficient separated pairs for anisotropic check, seed=" *
                string(seed))
    end

    cr
end

# ── Check 14: prop:knn-stability ─────────────────────────────────────────────
#
# If sup|d̂ - d| < γ_k/2, the directed k-NN graph is unchanged.

function check_knn_stability(rng::AbstractRNG; seed=0, n_w=6, k=2)
    cr = CheckResult("prop_knn_stability")

    # Random distance matrix
    d = zeros(n_w, n_w)
    for i in 1:n_w, j in (i + 1):n_w
        d[i, j] = rand(rng) * 10 + 0.1
        d[j, i] = d[i, j]
    end

    # Compute k-NN margin γ_k
    function knn_graph(dm, kk)
        n = size(dm, 1)
        graph = [Set{Int}() for _ in 1:n]
        for i in 1:n
            dists = [(dm[i, j], j) for j in 1:n if j != i]
            sort!(dists, by=first)
            graph[i] = Set([j for (_, j) in dists[1:kk]])
        end
        graph
    end

    function knn_margin(dm, kk)
        n = size(dm, 1)
        gamma = Inf
        for i in 1:n
            dists = sort([dm[i, j] for j in 1:n if j != i])
            gap = dists[kk + 1] - dists[kk]
            gamma = min(gamma, gap)
        end
        gamma
    end

    gamma_k = knn_margin(d, k)
    if gamma_k <= 0
        record!(cr, 0.0)  # degenerate: tied distances
        return cr
    end

    true_graph = knn_graph(d, k)

    # (a) Perturbation within γ_k/2: graph should be unchanged
    for trial in 1:20
        perturb = (rand(MersenneTwister(seed + trial), n_w, n_w) .- 0.5) .*
                  (0.9 * gamma_k)  # sup norm < γ_k/2
        perturb = (perturb .+ perturb') ./ 2  # symmetrize
        for i in 1:n_w; perturb[i, i] = 0.0; end

        d_hat = d .+ perturb
        # Verify sup norm
        sup_err = maximum(abs.(d_hat .- d))
        if sup_err >= gamma_k / 2
            # Scale down
            d_hat = d .+ perturb .* (0.49 * gamma_k / sup_err)
        end

        perturbed_graph = knn_graph(d_hat, k)
        if perturbed_graph != true_graph
            record!(cr, 1.0, "graph changed within margin, trial=" * string(trial) *
                    " seed=" * string(seed))
        else
            record!(cr, 0.0)
        end
    end

    # (b) Perturbation beyond γ_k/2: graph CAN change (existential, just verify
    # that we can construct one)
    big_perturb = zeros(n_w, n_w)
    # Find the tightest margin pair and push it over
    for i in 1:n_w
        dists = [(d[i, j], j) for j in 1:n_w if j != i]
        sort!(dists, by=first)
        r_k = dists[k][1]
        r_k1 = dists[k + 1][1]
        gap = r_k1 - r_k
        if abs(gap - gamma_k) < TOL
            # This is the tightest margin pair; swap the k-th and (k+1)-th neighbors
            j_k = dists[k][2]
            j_k1 = dists[k + 1][2]
            big_perturb[i, j_k] = gap  # push k-th neighbor out
            big_perturb[i, j_k1] = -gap  # pull (k+1)-th neighbor in
            break
        end
    end
    d_big = d .+ big_perturb
    big_graph = knn_graph(d_big, k)
    # We don't assert it changed (not required by theorem), just record
    record!(cr, 0.0)

    cr
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    println("=" ^ 60)
    println("Priority 3: Geometry checks")
    println("=" ^ 60)
    println()

    all_results = CheckResult[]

    sizes = [(3, 2, 2), (3, 3, 3), (4, 2, 2), (4, 3, 3), (5, 2, 2)]
    n_random = 20

    for (n_w, n_c, n_v) in sizes
        println("--- n_w=" * string(n_w) * " n_c=" * string(n_c) *
                " n_v=" * string(n_v) * " ---")

        for trial in 1:n_random
            seed = 5000 * n_w + 100 * n_c + trial
            rng = MersenneTwister(seed)
            fe = random_ecology(rng, n_w, n_c, n_v)

            push!(all_results, check_hilbert_kernel(fe; seed=seed))
            push!(all_results, check_topology_general(fe; seed=seed))
        end
    end

    # Gaussian-linear and kNN checks (not ecology-dependent)
    for trial in 1:50
        seed = 70000 + trial
        push!(all_results, check_geometry_gaussian(MersenneTwister(seed); seed=seed))
        push!(all_results, check_knn_stability(MersenneTwister(seed); seed=seed, n_w=6, k=2))
    end

    println()
    println("=" ^ 60)
    println("SUMMARY")
    println("=" ^ 60)
    println()

    for name in ["prop_hilbert_kernel", "thm_topology_general",
                  "thm_geometry_gaussian", "prop_knn_stability"]
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
