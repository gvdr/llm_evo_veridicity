# ── Information-theoretic primitives ──────────────────────────────────────────
#
# Pure functions on probability vectors (Vector{Float64} summing to 1).
# Shared by finite_ecology.jl (theorem checks) and measurement.jl (experiments).

"""
    entropy(p)

Shannon entropy H(p) = -Σ pᵢ log pᵢ  (nats).
Treats 0 log 0 = 0.
"""
function entropy(p::AbstractVector{<:Real})
    h = 0.0
    for pᵢ in p
        if pᵢ > 0
            h -= pᵢ * log(pᵢ)
        end
    end
    h
end

"""
    kl_divergence(p, q)

KL(p ‖ q) = Σ pᵢ log(pᵢ/qᵢ)  (nats).
Requires supp(p) ⊆ supp(q).  Returns Inf if violated.
"""
function kl_divergence(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})
    d = 0.0
    for (pᵢ, qᵢ) in zip(p, q)
        if pᵢ > 0
            qᵢ <= 0 && return Inf
            d += pᵢ * log(pᵢ / qᵢ)
        end
    end
    d
end

"""
    hellinger2(p, q)

Squared Hellinger distance: H²(p,q) = (1/2) Σ (√pᵢ - √qᵢ)².
"""
function hellinger2(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})
    h = 0.0
    for (pᵢ, qᵢ) in zip(p, q)
        d = sqrt(pᵢ) - sqrt(qᵢ)
        h += d * d
    end
    0.5 * h
end

"""
    js_divergence(p, q)

Symmetric Jensen-Shannon divergence with equal weights:
JS(p,q) = (1/2) KL(p‖m) + (1/2) KL(q‖m),  m = (p+q)/2.
Equivalently: JS(p,q) = H(m) - (H(p) + H(q))/2.
"""
function js_divergence(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})
    m = 0.5 .* (p .+ q)
    entropy(m) - 0.5 * entropy(p) - 0.5 * entropy(q)
end

"""
    weighted_js(ps, alpha)

Weighted Jensen-Shannon divergence:
JS_α(p₁,…,pₖ) = H(Σ αᵢ pᵢ) - Σ αᵢ H(pᵢ).

`ps` is a vector of probability vectors, `alpha` is a weight vector summing to 1.
"""
function weighted_js(ps::AbstractVector{<:AbstractVector{<:Real}},
                     alpha::AbstractVector{<:Real})
    mixture = zeros(length(ps[1]))
    h_components = 0.0
    for (αᵢ, pᵢ) in zip(alpha, ps)
        mixture .+= αᵢ .* pᵢ
        h_components += αᵢ * entropy(pᵢ)
    end
    entropy(mixture) - h_components
end

"""
    binary_entropy(lam)

h(λ) = -λ log λ - (1-λ) log(1-λ).  Returns 0 for λ ∈ {0, 1}.
"""
function binary_entropy(lam::Real)
    (lam <= 0 || lam >= 1) && return 0.0
    -lam * log(lam) - (1 - lam) * log(1 - lam)
end

"""
    cross_entropy(p, q)

CE(p,q) = -Σ pᵢ log qᵢ = H(p) + KL(p‖q).
"""
function cross_entropy(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})
    ce = 0.0
    for (pᵢ, qᵢ) in zip(p, q)
        if pᵢ > 0
            qᵢ <= 0 && return Inf
            ce -= pᵢ * log(qᵢ)
        end
    end
    ce
end
