# ── Measurement functions for ecological veridicality experiments ──────────────
#
# Behavioural distance, separation counting, excess loss decomposition,
# and Price equation terms.

"""
    js_div(p, q)

Jensen-Shannon divergence (symmetric, in nats).
"""
function js_div(p::Vector{Float64}, q::Vector{Float64})
    d = 0.0
    @inbounds for i in eachindex(p)
        m_i = 0.5 * (p[i] + q[i])
        if p[i] > 0 && m_i > 0
            d += 0.5 * p[i] * log(p[i] / m_i)
        end
        if q[i] > 0 && m_i > 0
            d += 0.5 * q[i] * log(q[i] / m_i)
        end
    end
    d
end

"""
    mean_output_distribution(W, prefix_tokens_list, cfg) -> Vector{Float64}

Average next-token distribution over a list of prefix token sequences.
Uses last_position_distribution for efficiency (no intermediate allocation).
"""
function mean_output_distribution(W::Dict{String,Matrix{Float64}},
                                  prefix_tokens_list::Vector{Vector{Int}},
                                  cfg::ModelConfig)
    avg = zeros(cfg.vocab_size)
    count = 0
    for tokens in prefix_tokens_list
        length(tokens) < 2 && continue
        dist = last_position_distribution(W, tokens, cfg)
        isempty(dist) && continue
        avg .+= dist
        count += 1
    end
    if count > 0
        avg ./= count
    else
        fill!(avg, 1.0 / cfg.vocab_size)
    end
    avg
end

"""
    make_prefixes(chunks, tok, prefix_lengths; ...) -> Dict{Int, Vector{Vector{Int}}}

Build prefix token sequences for evaluation.
Tokenizes each chunk first, then takes the first r+1 tokens (BOS + r tokens)
as a prefix of token-length r. This ensures "prefix length" is a stable
token-level notion regardless of spaces or other dropped characters.
"""
function make_prefixes(chunks::Vector{String}, tok::Tokenizer,
                       prefix_lengths::Vector{Int};
                       max_per_length::Int=100,
                       rng::AbstractRNG=Random.default_rng())
    result = Dict{Int, Vector{Vector{Int}}}()
    for r in prefix_lengths
        seqs = Vector{Int}[]
        for chunk in chunks
            # Tokenize chunk: [BOS, c1, c2, ..., cn, BOS]
            tokens = Int[tok.bos]
            for c in chunk
                if haskey(tok.char2id, c)
                    push!(tokens, tok.char2id[c])
                end
            end
            # Take first r+1 tokens: [BOS, t1, ..., tr]
            # Need at least r+1 tokens (r characters that survived tokenization)
            length(tokens) >= r + 1 || continue
            prefix = tokens[1:(r + 1)]
            # Need at least 2 tokens for the model to produce a distribution
            length(prefix) >= 2 && push!(seqs, prefix)
        end
        # Subsample if needed
        if length(seqs) > max_per_length
            perm = randperm(rng, length(seqs))
            seqs = seqs[perm[1:max_per_length]]
        end
        result[r] = seqs
    end
    result
end

"""
    behavioural_distance(W, prefixes_i, prefixes_j, cfg)

Compute D_beh(w_i, w_j) = mean over prefix lengths of
JS(p_bar(·|w_i, r), p_bar(·|w_j, r)).
"""
function behavioural_distance(W::Dict{String,Matrix{Float64}},
                              prefixes_i::Dict{Int, Vector{Vector{Int}}},
                              prefixes_j::Dict{Int, Vector{Vector{Int}}},
                              cfg::ModelConfig)
    total_js = 0.0
    n_lengths = 0
    for r in sort(collect(keys(prefixes_i)))
        haskey(prefixes_j, r) || continue
        isempty(prefixes_i[r]) && continue
        isempty(prefixes_j[r]) && continue

        p_bar_i = mean_output_distribution(W, prefixes_i[r], cfg)
        p_bar_j = mean_output_distribution(W, prefixes_j[r], cfg)
        total_js += js_div(p_bar_i, p_bar_j)
        n_lengths += 1
    end
    n_lengths > 0 ? total_js / n_lengths : 0.0
end

"""
    compute_p_bars(W, prefixes_by_lang, languages, cfg)

Pre-compute all mean output distributions for every language and prefix length.
Returns `Dict{String, Dict{Int, Vector{Float64}}}` mapping
language -> prefix_length -> p_bar distribution.
"""
function compute_p_bars(W::Dict{String,Matrix{Float64}},
                        prefixes_by_lang::Dict{String, Dict{Int, Vector{Vector{Int}}}},
                        languages::Vector{String},
                        cfg::ModelConfig)
    cache = Dict{String, Dict{Int, Vector{Float64}}}()
    for lang in languages
        lang_cache = Dict{Int, Vector{Float64}}()
        for (r, seqs) in prefixes_by_lang[lang]
            if !isempty(seqs)
                lang_cache[r] = mean_output_distribution(W, seqs, cfg)
            end
        end
        cache[lang] = lang_cache
    end
    cache
end

"""
    _cached_behavioural_distance(cache_i, cache_j) -> Float64

Compute D_beh from pre-computed p_bar caches for two languages.
Averages JS divergence across all shared prefix lengths.
"""
function _cached_behavioural_distance(cache_i::Dict{Int, Vector{Float64}},
                                      cache_j::Dict{Int, Vector{Float64}})
    total_js = 0.0
    n_lengths = 0
    for r in sort(collect(keys(cache_i)))
        haskey(cache_j, r) || continue
        total_js += js_div(cache_i[r], cache_j[r])
        n_lengths += 1
    end
    n_lengths > 0 ? total_js / n_lengths : 0.0
end

"""
    distance_matrix(W, prefixes_by_lang, languages, cfg) -> Matrix{Float64}

Compute the full pairwise behavioural distance matrix.
Pre-computes all p_bar distributions so each language/prefix combination
is evaluated only once (instead of once per pair).
"""
function distance_matrix(W::Dict{String,Matrix{Float64}},
                         prefixes_by_lang::Dict{String, Dict{Int, Vector{Vector{Int}}}},
                         languages::Vector{String},
                         cfg::ModelConfig)
    # Pre-compute all p_bar distributions once
    p_bars = compute_p_bars(W, prefixes_by_lang, languages, cfg)

    n = length(languages)
    D = zeros(n, n)
    for i in 1:n, j in (i + 1):n
        d = _cached_behavioural_distance(p_bars[languages[i]],
                                         p_bars[languages[j]])
        D[i, j] = d
        D[j, i] = d
    end
    D
end

"""
    separation_summary(D, tau) -> (s, k)

From a distance matrix D and threshold tau:
- s: number of separated pairs (D[i,j] > tau)
- k: number of equivalence classes (connected components where D[i,j] <= tau)
"""
function separation_summary(D::Matrix{Float64}, tau::Float64)
    n = size(D, 1)
    # Count separated pairs
    s = 0
    for i in 1:n, j in (i + 1):n
        if D[i, j] > tau
            s += 1
        end
    end

    # Connected components via union-find
    parent = collect(1:n)
    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        x
    end
    function uf_union!(x, y)
        px, py = find(x), find(y)
        if px != py
            parent[px] = py
        end
    end

    for i in 1:n, j in (i + 1):n
        if D[i, j] <= tau
            uf_union!(i, j)
        end
    end

    k = length(Set(find(i) for i in 1:n))
    (s, k)
end

"""
    calibrate_threshold(W, chunks, tok, cfg; n_splits, rng) -> Float64

Calibrate the separation threshold tau from the noise floor.
Split one language's chunks randomly, compute D_beh between splits,
return the 95th percentile.
"""
function calibrate_threshold(W::Dict{String,Matrix{Float64}},
                             chunks::Vector{String},
                             tok::Tokenizer,
                             cfg::ModelConfig;
                             n_splits::Int=20,
                             prefix_lengths::Vector{Int}=[3, 5, 8],
                             rng::AbstractRNG=Random.default_rng())
    null_distances = Float64[]
    for _ in 1:n_splits
        perm = randperm(rng, length(chunks))
        mid = length(chunks) ÷ 2
        split_a = chunks[perm[1:mid]]
        split_b = chunks[perm[(mid + 1):end]]
        pref_a = make_prefixes(split_a, tok, prefix_lengths; rng=rng)
        pref_b = make_prefixes(split_b, tok, prefix_lengths; rng=rng)
        d = behavioural_distance(W, pref_a, pref_b, cfg)
        push!(null_distances, d)
    end
    sort!(null_distances)
    # 95th percentile
    idx = ceil(Int, 0.95 * length(null_distances))
    null_distances[min(idx, length(null_distances))]
end

# ── Cross-entropy evaluation (efficient, token-weighted) ─────────────────────

"""
    cross_entropy_loss(W, tokens, cfg) -> Float64

Mean cross-entropy loss on a token sequence (per-token, forward-only).
Uses sequence_nll for efficient evaluation.
"""
function cross_entropy_loss(W::Dict{String,Matrix{Float64}},
                            tokens::Vector{Int}, cfg::ModelConfig)
    total_nll, n_tokens = sequence_nll(W, tokens, cfg)
    n_tokens > 0 ? total_nll / n_tokens : 0.0
end

"""
    mean_cross_entropy(W, tokens_list, cfg; token_weighted=true) -> Float64

Mean cross-entropy across a list of token sequences.
If token_weighted=true (default), weight by number of prediction targets
so each token contributes equally. Otherwise each sequence contributes equally.
"""
function mean_cross_entropy(W::Dict{String,Matrix{Float64}},
                            tokens_list::Vector{Vector{Int}},
                            cfg::ModelConfig;
                            token_weighted::Bool=true)
    total_nll = 0.0
    total_tokens = 0
    n_seqs = 0
    seq_sum = 0.0
    for tokens in tokens_list
        length(tokens) < 2 && continue
        nll, nt = sequence_nll(W, tokens, cfg)
        total_nll += nll
        total_tokens += nt
        if nt > 0
            seq_sum += nll / nt
            n_seqs += 1
        end
    end
    if token_weighted
        total_tokens > 0 ? total_nll / total_tokens : 0.0
    else
        n_seqs > 0 ? seq_sum / n_seqs : 0.0
    end
end

"""
    mean_cross_entropy_balanced(W, tokens_by_lang, cfg; token_weighted=true) -> Float64

Mean cross-entropy balanced across languages (world states).
Computes per-language CE first, then averages across languages so each
language contributes equally regardless of corpus size.
"""
function mean_cross_entropy_balanced(W::Dict{String,Matrix{Float64}},
                                     tokens_by_lang::Dict{String,Vector{Vector{Int}}},
                                     cfg::ModelConfig;
                                     token_weighted::Bool=true)
    lang_ces = Float64[]
    for (_, toks) in tokens_by_lang
        ce = mean_cross_entropy(W, toks, cfg; token_weighted=token_weighted)
        push!(lang_ces, ce)
    end
    isempty(lang_ces) ? 0.0 : sum(lang_ces) / length(lang_ces)
end

"""
    price_equation_terms(risks, fitnesses)

Compute the expected selection-stage Price equation term.
Returns (cov_wR, w_bar, predicted_delta) where:
- cov_wR = Cov(fitness, risk)
- w_bar = mean fitness
- predicted_delta = Cov(fitness, risk) / w_bar
"""
function price_equation_terms(risks::Vector{Float64}, fitnesses::Vector{Float64})
    n = length(risks)
    w_bar = sum(fitnesses) / n
    r_bar = sum(risks) / n
    cov_wR = sum((fitnesses[i] - w_bar) * (risks[i] - r_bar) for i in 1:n) / n
    pred = w_bar > 0 ? cov_wR / w_bar : NaN
    (cov_wR, w_bar, pred)
end

"""
    selection_stage_stats(risks, fitnesses, K_selected)

Exact conditional selection-stage moments under Wright-Fisher sampling with
replacement. Returns

- `cov_wR`: Cov(fitness, risk)
- `w_bar`: mean fitness
- `delta_expected`: expected `R̄_sel - R̄_before`
- `risk_selected_expected`: expected post-selection mean risk
- `delta_sd`: exact conditional standard deviation of `R̄_sel - R̄_before`

The expectation agrees with the Price term: `delta_expected = Cov(w, R) / w̄`.
"""
function selection_stage_stats(risks::Vector{Float64},
                               fitnesses::Vector{Float64},
                               K_selected::Int)
    cov_wR, w_bar, delta_expected = price_equation_terms(risks, fitnesses)
    total_fitness = sum(fitnesses)
    if !(isfinite(total_fitness) && total_fitness > 0.0)
        error("selection_stage_stats requires positive finite total fitness")
    end
    probs = fitnesses ./ total_fitness
    risk_selected_expected = sum(probs[i] * risks[i] for i in eachindex(risks))
    var_single = sum(probs[i] * (risks[i] - risk_selected_expected)^2
                     for i in eachindex(risks))
    delta_sd = sqrt(max(var_single / K_selected, 0.0))
    (cov_wR, w_bar, delta_expected, risk_selected_expected, delta_sd)
end

# ── Excess loss decomposition (exact decomposition theorem validation) ───────

"""
    _shannon_entropy(p)

Shannon entropy H(p) = -sum(p_i log p_i) in nats, treating 0 log 0 = 0.
Local helper to avoid dispatch ambiguity with information.jl's `entropy`.
"""
function _shannon_entropy(p::Vector{Float64})
    h = 0.0
    @inbounds for i in eachindex(p)
        pi = p[i]
        if pi > 0
            h -= pi * log(pi)
        end
    end
    h
end

"""
    _multiway_js(distributions)

Multi-way Jensen-Shannon divergence with uniform weights:
JS(p1, ..., pn) = H(mean(pi)) - mean(H(pi)).
Returns 0.0 for a single distribution or empty input.
"""
function _multiway_js(distributions::Vector{Vector{Float64}})
    n = length(distributions)
    n <= 1 && return 0.0
    # Compute the uniform mixture
    d = length(distributions[1])
    mixture = zeros(d)
    for p in distributions
        mixture .+= p
    end
    mixture ./= n
    # H(mixture) - mean of H(p_i)
    h_mix = _shannon_entropy(mixture)
    h_avg = 0.0
    for p in distributions
        h_avg += _shannon_entropy(p)
    end
    h_avg /= n
    max(0.0, h_mix - h_avg)
end

"""
    _partition_from_distance(D, tau) -> Vector{Vector{Int}}

Compute connected components of the graph where D[i,j] <= tau.
Returns a vector of cells, each cell being a vector of 1-based indices.
"""
function _partition_from_distance(D::Matrix{Float64}, tau::Float64)
    n = size(D, 1)
    # Union-find
    parent = collect(1:n)
    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        x
    end
    function uf_union!(x, y)
        px, py = find(x), find(y)
        if px != py
            parent[px] = py
        end
    end

    for i in 1:n, j in (i + 1):n
        if D[i, j] <= tau
            uf_union!(i, j)
        end
    end

    # Group indices by their root
    groups = Dict{Int, Vector{Int}}()
    for i in 1:n
        root = find(i)
        if !haskey(groups, root)
            groups[root] = Int[]
        end
        push!(groups[root], i)
    end
    collect(values(groups))
end

"""
    excess_loss_decomposition(W, eval_tokens_by_lang, prefixes_by_lang,
                              languages, cfg, tau, specialist_ces)

Validate the exact excess-loss decomposition theorem empirically for a trained
microgpt model.

The excess loss decomposition says:
    L_hat - H_irred ≈ JS_excess
where:
- L_hat: language-balanced cross-entropy of the multi-language model.
- H_irred: irreducible entropy — the average CE of per-language specialist
  models, each evaluated on its own language. This is the best achievable
  loss for an oracle that knows which language it is processing.
- JS_excess: for each partition cell (languages merged at threshold tau),
  the multi-way JS divergence among those languages' mean output
  distributions, averaged over prefix lengths and weighted by cell size.

`specialist_ces` is a Dict mapping language name to the CE of a
single-language specialist model evaluated on that language's test data.

Returns a NamedTuple with fields:
- `observed_loss`:       L_hat
- `irreducible_entropy`: H_irred
- `js_excess`:           total JS excess from the partition
- `excess_gap`:          L_hat - H_irred
- `excess_ratio`:        js_excess / (L_hat - H_irred), should be close to 1.0
- `partition_cells`:     the cells of the induced partition (indices into `languages`)
- `n_classes`:           number of equivalence classes k
"""
function excess_loss_decomposition(W::Dict{String,Matrix{Float64}},
                                   eval_tokens_by_lang::Dict{String,Vector{Vector{Int}}},
                                   prefixes_by_lang::Dict{String,Dict{Int,Vector{Vector{Int}}}},
                                   languages::Vector{String},
                                   cfg::ModelConfig,
                                   tau::Float64,
                                   specialist_ces::Dict{String,Float64})
    n_lang = length(languages)

    # ── Step 1: distance matrix and partition ────────────────────────────────
    D = distance_matrix(W, prefixes_by_lang, languages, cfg)
    cells = _partition_from_distance(D, tau)
    k = length(cells)

    # ── Step 2: observed loss (language-balanced CE) ─────────────────────────
    observed_loss = mean_cross_entropy_balanced(W, eval_tokens_by_lang, cfg)

    # ── Step 3: irreducible entropy from per-language specialists ────────────
    # H_irred = average of specialist CEs. Each specialist is optimal for its
    # own language, so this is the best achievable loss given perfect separation.
    spec_vals = Float64[]
    for lang in languages
        if haskey(specialist_ces, lang)
            push!(spec_vals, specialist_ces[lang])
        end
    end
    irreducible_entropy = isempty(spec_vals) ? 0.0 : sum(spec_vals) / length(spec_vals)

    # ── Step 4: JS excess from partition ─────────────────────────────────────
    # For each cell, for each prefix length, compute the multi-way JS
    # divergence among the languages in that cell's mean output distributions.
    # Then average over prefix lengths and weight by (cell size / n_lang).

    # Pre-compute all p_bar distributions (reuses the caching infrastructure)
    p_bars = compute_p_bars(W, prefixes_by_lang, languages, cfg)

    # Collect the set of prefix lengths available across all languages
    all_prefix_lengths = Set{Int}()
    for lang in languages
        if haskey(p_bars, lang)
            for r in keys(p_bars[lang])
                push!(all_prefix_lengths, r)
            end
        end
    end
    sorted_lengths = sort(collect(all_prefix_lengths))

    js_excess = 0.0
    n_valid_lengths = 0

    for r in sorted_lengths
        js_at_r = 0.0
        has_contribution = false

        for cell in cells
            length(cell) <= 1 && continue  # singleton cells contribute 0

            # Gather cached mean output distributions for each language in this cell
            distributions = Vector{Float64}[]
            for idx in cell
                lang = languages[idx]
                if haskey(p_bars, lang) && haskey(p_bars[lang], r)
                    push!(distributions, p_bars[lang][r])
                end
            end

            if length(distributions) >= 2
                cell_js = _multiway_js(distributions)
                # Weight by cell fraction (uniform prior over languages)
                js_at_r += (length(cell) / n_lang) * cell_js
                has_contribution = true
            end
        end

        if has_contribution
            js_excess += js_at_r
            n_valid_lengths += 1
        end
    end

    if n_valid_lengths > 0
        js_excess /= n_valid_lengths
    end

    # ── Step 5: assemble results ─────────────────────────────────────────────
    excess_gap = observed_loss - irreducible_entropy
    excess_ratio = excess_gap > 1e-12 ? js_excess / excess_gap : NaN

    (observed_loss       = observed_loss,
     irreducible_entropy = irreducible_entropy,
     js_excess           = js_excess,
     excess_gap          = excess_gap,
     excess_ratio        = excess_ratio,
     partition_cells     = cells,
     n_classes           = k)
end

# ── Progress reporting ────────────────────────────────────────────────────────

"""
    progress_str(current, total, label; start_time)

Format a progress string with ETA.
"""
function progress_str(current::Int, total::Int, label::String;
                      start_time::Float64=0.0)
    pct = round(100.0 * current / total, digits=1)
    msg = label * " " * string(current) * "/" * string(total) *
          " (" * string(pct) * "%)"
    if start_time > 0 && current > 0
        elapsed = time() - start_time
        eta = elapsed / current * (total - current)
        if eta < 60
            msg *= " ETA " * string(round(Int, eta)) * "s"
        else
            msg *= " ETA " * string(round(eta / 60, digits=1)) * "m"
        end
    end
    msg
end
