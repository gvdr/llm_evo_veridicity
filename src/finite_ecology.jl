# ── Finite ecology framework ─────────────────────────────────────────────────
#
# A fully discrete ecology for exact theorem checking.
# All quantities can be computed exhaustively when |W| ≤ 6.

"""
    FiniteEcology(pi, d_c, p_wcv)

A finite task ecology with:
- `pi[w]`:        prior over world states,  length n_w, sums to 1
- `d_c[c]`:       context marginal,         length n_c, sums to 1
- `p_wcv[w,c,v]`: P_w(v|c),                size (n_w, n_c, n_v)

Each slice `p_wcv[w, c, :]` must be a valid probability distribution.
"""
struct FiniteEcology
    pi::Vector{Float64}
    d_c::Vector{Float64}
    p_wcv::Array{Float64, 3}
end

n_worlds(fe::FiniteEcology)   = length(fe.pi)
n_contexts(fe::FiniteEcology) = length(fe.d_c)
n_vocab(fe::FiniteEcology)    = size(fe.p_wcv, 3)

# ── Generators ───────────────────────────────────────────────────────────────

"""
    random_ecology(rng, n_w, n_c, n_v; floor=1e-6)

Sample a random FiniteEcology in generic position (no accidental equalities).
"""
function random_ecology(rng::AbstractRNG, n_w::Int, n_c::Int, n_v::Int;
                        floor::Float64=1e-6)
    _rand_simplex(rng, n) = (x = rand(rng, n); x ./ sum(x))

    pi  = _rand_simplex(rng, n_w)
    d_c = _rand_simplex(rng, n_c)

    p_wcv = Array{Float64}(undef, n_w, n_c, n_v)
    for w in 1:n_w, c in 1:n_c
        raw = rand(rng, n_v) .+ floor
        p_wcv[w, c, :] = raw ./ sum(raw)
    end

    FiniteEcology(pi, d_c, p_wcv)
end

"""
    ecology_with_merged_pair(rng, n_w, n_c, n_v; merged=(1,2))

Random ecology where states merged[1] and merged[2] have identical
conditional distributions on all contexts. Tests the zero-separation case.
"""
function ecology_with_merged_pair(rng::AbstractRNG, n_w::Int, n_c::Int, n_v::Int;
                                  merged::Tuple{Int,Int}=(1,2), floor::Float64=1e-6)
    fe = random_ecology(rng, n_w, n_c, n_v; floor=floor)
    w1, w2 = merged
    for c in 1:n_c
        fe.p_wcv[w2, c, :] .= fe.p_wcv[w1, c, :]
    end
    fe
end

"""
    ecology_with_partial_separation(rng, n_w, n_c, n_v; pair=(1,2), sep_context=1)

Random ecology where pair[1] and pair[2] differ only on context sep_context.
Useful for testing separation on a single context.
"""
function ecology_with_partial_separation(rng::AbstractRNG, n_w::Int, n_c::Int, n_v::Int;
                                         pair::Tuple{Int,Int}=(1,2), sep_context::Int=1,
                                         floor::Float64=1e-6)
    fe = random_ecology(rng, n_w, n_c, n_v; floor=floor)
    w1, w2 = pair
    # Make w1 and w2 identical on all contexts except sep_context
    for c in 1:n_c
        c == sep_context && continue
        fe.p_wcv[w2, c, :] .= fe.p_wcv[w1, c, :]
    end
    fe
end

# ── Core quantities ──────────────────────────────────────────────────────────

"""
    conditional_entropy_VCW(fe)

H(V | C, W) = Σ_w π(w) Σ_c d_c(c) H(P_w(·|c)).
The irreducible entropy floor.
"""
function conditional_entropy_VCW(fe::FiniteEcology)
    h = 0.0
    for w in 1:n_worlds(fe), c in 1:n_contexts(fe)
        h += fe.pi[w] * fe.d_c[c] * entropy(@view fe.p_wcv[w, c, :])
    end
    h
end

"""
    task_distance(fe, w1, w2)

σ²_D(w1, w2) = E_{c~D_C}[H²(P_{w1}(·|c), P_{w2}(·|c))].
"""
function task_distance(fe::FiniteEcology, w1::Int, w2::Int)
    d = 0.0
    for c in 1:n_contexts(fe)
        d += fe.d_c[c] * hellinger2(@view(fe.p_wcv[w1, c, :]),
                                    @view(fe.p_wcv[w2, c, :]))
    end
    d
end

"""
    task_distance_matrix(fe)

Full pairwise task-distance matrix.
"""
function task_distance_matrix(fe::FiniteEcology)
    nw = n_worlds(fe)
    D = zeros(nw, nw)
    for i in 1:nw, j in (i + 1):nw
        d = task_distance(fe, i, j)
        D[i, j] = d
        D[j, i] = d
    end
    D
end

"""
    separated(fe, w1, w2)

True iff σ²_D(w1, w2) > 0, i.e., there exists a positive-mass context
where the next-token distributions differ.
"""
function separated(fe::FiniteEcology, w1::Int, w2::Int)
    for c in 1:n_contexts(fe)
        fe.d_c[c] <= 0 && continue
        for v in 1:n_vocab(fe)
            if fe.p_wcv[w1, c, v] != fe.p_wcv[w2, c, v]
                return true
            end
        end
    end
    false
end

"""
    ecology_partition(fe)

Compute the ecology partition W/∼_μ as a label vector.
Two states share a label iff they are NOT separated.
"""
function ecology_partition(fe::FiniteEcology)
    nw = n_worlds(fe)
    labels = collect(1:nw)
    for i in 1:nw, j in (i + 1):nw
        if !separated(fe, i, j)
            # Merge j's class into i's class
            old_label = labels[j]
            new_label = labels[i]
            for k in 1:nw
                if labels[k] == old_label
                    labels[k] = new_label
                end
            end
        end
    end
    # Renumber contiguously
    _renumber(labels)
end

function _renumber(labels::Vector{Int})
    mapping = Dict{Int,Int}()
    next = 1
    out = similar(labels)
    for (i, l) in enumerate(labels)
        if !haskey(mapping, l)
            mapping[l] = next
            next += 1
        end
        out[i] = mapping[l]
    end
    out
end

# ── Encoding quantities ──────────────────────────────────────────────────────

"""
    cell_average(fe, cell, c)

P̄_x(·|c) = Σ_{w∈cell} α_x(w) P_w(·|c),  where α_x(w) = π(w) / π_x.
"""
function cell_average(fe::FiniteEcology, cell::AbstractVector{Int}, c::Int)
    nv = n_vocab(fe)
    pi_x = sum(fe.pi[w] for w in cell)
    avg = zeros(nv)
    for w in cell
        alpha_w = fe.pi[w] / pi_x
        for v in 1:nv
            avg[v] += alpha_w * fe.p_wcv[w, c, v]
        end
    end
    avg
end

"""
    bayes_opt_loss(fe, labels)

L*_D(p) = Σ_w π(w) Σ_c d_c(c) CE(P_w(·|c), P̄_{p(w)}(·|c)).
"""
function bayes_opt_loss(fe::FiniteEcology, labels::AbstractVector{Int})
    cells = partition_cells(labels)
    nw = n_worlds(fe)
    nc = n_contexts(fe)

    # Precompute cell averages
    cell_of = Dict{Int, Int}()
    for (idx, cell) in enumerate(cells), w in cell
        cell_of[w] = idx
    end

    loss = 0.0
    for w in 1:nw, c in 1:nc
        p_w = @view fe.p_wcv[w, c, :]
        q_x = cell_average(fe, cells[cell_of[w]], c)
        loss += fe.pi[w] * fe.d_c[c] * cross_entropy(p_w, q_x)
    end
    loss
end

"""
    excess_loss(fe, labels)

L*_D(p) - H(V|C,W).
"""
function excess_loss(fe::FiniteEcology, labels::AbstractVector{Int})
    bayes_opt_loss(fe, labels) - conditional_entropy_VCW(fe)
end

"""
    js_excess(fe, labels)

The weighted JS decomposition of the excess:
Σ_c d_c(c) Σ_x π_x JS_{α_x}({P_w(·|c)}_{w∈C_x}).
Should equal `excess_loss(fe, labels)` exactly by the exact excess-loss
decomposition theorem.
"""
function js_excess(fe::FiniteEcology, labels::AbstractVector{Int})
    cells = partition_cells(labels)
    nc = n_contexts(fe)

    total = 0.0
    for c in 1:nc, cell in cells
        length(cell) == 1 && continue  # singleton cells contribute 0
        pi_x = sum(fe.pi[w] for w in cell)
        alpha = [fe.pi[w] / pi_x for w in cell]
        ps = [fe.p_wcv[w, c, :] for w in cell]
        total += fe.d_c[c] * pi_x * weighted_js(ps, alpha)
    end
    total
end

"""
    partition_complexity(fe, labels)

I(W; p(W)) = H(p(W)) for a deterministic encoding.
H(p(W)) = -Σ_x π_x log π_x  where π_x = Σ_{w∈C_x} π(w).
"""
function partition_complexity(fe::FiniteEcology, labels::AbstractVector{Int})
    cells = partition_cells(labels)
    h = 0.0
    for cell in cells
        pi_x = sum(fe.pi[w] for w in cell)
        if pi_x > 0
            h -= pi_x * log(pi_x)
        end
    end
    h
end

"""
    regularized_objective(fe, labels, beta)

J_{D,β}(p) = L*_D(p) + β H(p(W)).
"""
function regularized_objective(fe::FiniteEcology, labels::AbstractVector{Int},
                               beta::Float64)
    bayes_opt_loss(fe, labels) + beta * partition_complexity(fe, labels)
end

"""
    merges_separated_pair(fe, labels)

True iff the partition merges at least one μ-separated pair.
"""
function merges_separated_pair(fe::FiniteEcology, labels::AbstractVector{Int})
    nw = n_worlds(fe)
    for i in 1:nw, j in (i + 1):nw
        if labels[i] == labels[j] && separated(fe, i, j)
            return true
        end
    end
    false
end

"""
    is_zero_excess(fe, labels; tol=1e-10)

Check whether the partition has zero excess loss (up to tolerance).
"""
function is_zero_excess(fe::FiniteEcology, labels::AbstractVector{Int};
                        tol::Float64=1e-10)
    excess_loss(fe, labels) < tol
end

# ── Split-threshold quantities (split-versus-merge theorem) ─────────────────

"""
    split_gain(fe, cell_A, cell_B)

Δ_pred(A,B) = E_{c~D_C}[JS_λ(P̄_A(·|c), P̄_B(·|c))],
where λ = π_A / (π_A + π_B).
"""
function split_gain(fe::FiniteEcology,
                    cell_A::AbstractVector{Int},
                    cell_B::AbstractVector{Int})
    nc = n_contexts(fe)
    pi_A = sum(fe.pi[w] for w in cell_A)
    pi_B = sum(fe.pi[w] for w in cell_B)
    lam = pi_A / (pi_A + pi_B)

    gain = 0.0
    for c in 1:nc
        p_A = cell_average(fe, cell_A, c)
        p_B = cell_average(fe, cell_B, c)
        # JS_λ(P̄_A, P̄_B) = H(λ P̄_A + (1-λ) P̄_B) - λ H(P̄_A) - (1-λ) H(P̄_B)
        m = lam .* p_A .+ (1 - lam) .* p_B
        js = entropy(m) - lam * entropy(p_A) - (1 - lam) * entropy(p_B)
        gain += fe.d_c[c] * js
    end
    gain
end

"""
    split_threshold_rhs(fe, cell_A, cell_B, beta)

The RHS from the split-versus-merge threshold theorem:
π_C * (Δ_pred(A,B) - β h(λ)),
where C = A ∪ B, λ = π_A / π_C.
"""
function split_threshold_rhs(fe::FiniteEcology,
                             cell_A::AbstractVector{Int},
                             cell_B::AbstractVector{Int},
                             beta::Float64)
    pi_A = sum(fe.pi[w] for w in cell_A)
    pi_B = sum(fe.pi[w] for w in cell_B)
    pi_C = pi_A + pi_B
    lam = pi_A / pi_C
    delta_pred = split_gain(fe, cell_A, cell_B)
    pi_C * (delta_pred - beta * binary_entropy(lam))
end
