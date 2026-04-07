# ── Population dynamics for Wright-Fisher selection experiments ────────────────
#
# Wright-Fisher selection on a population of microgpt models.
# Each generation: evaluate → select → develop → mutate.

"""
    wright_fisher_select(fitnesses, K; rng) -> Vector{Int}

Sample K parents with replacement, probability proportional to fitness.
Returns a vector of parent indices.
"""
function wright_fisher_select(fitnesses::Vector{Float64}, K::Int;
                              rng::AbstractRNG=Random.default_rng())
    # Normalize to probabilities
    total = sum(fitnesses)
    if !(isfinite(total) && total > 0.0)
        error("wright_fisher_select requires positive finite total fitness")
    end
    probs = fitnesses ./ total
    # Sample with replacement using binary search
    cumprobs = cumsum(probs)
    parents = Vector{Int}(undef, K)
    @inbounds for i in 1:K
        r = rand(rng)
        parents[i] = searchsortedfirst(cumprobs, r)
    end
    parents
end

"""
    GenerationRecord

Records for one generation of the population experiment.
"""
struct GenerationRecord
    gen              :: Int
    risk_before      :: Vector{Float64}  # pre-selection risk per model
    fitness          :: Vector{Float64}  # fitness per model
    parent_indices   :: Vector{Int}      # selected parents
    risk_selected    :: Float64          # mean risk of selected parents
    risk_selected_expected :: Float64    # exact conditional E[R_bar_sel]
    risk_after_dev   :: Float64          # mean risk after development
    risk_after_mut   :: Float64          # mean risk after mutation
    # Price equation terms
    delta_sel        :: Float64          # R_bar_sel - R_bar_before
    price_predicted  :: Float64          # expected Delta_R_bar_sel = Cov(w, R) / w_bar
    delta_sel_sd     :: Float64          # exact conditional sd under WF sampling
    delta_sel_lo95   :: Float64          # normal-approx 95% conditional lower bound
    delta_sel_hi95   :: Float64          # normal-approx 95% conditional upper bound
    delta_dev        :: Float64          # R_bar_dev - R_bar_sel
    delta_mut        :: Float64          # R_bar_mut - R_bar_dev
    # Separation summaries
    mean_k           :: Float64          # mean behavioural class count
    mean_s           :: Float64          # mean separated-pair count
end

"""
    evaluate_population(pop, eval_tokens_by_lang, cfg) -> (risks, fitnesses)

Evaluate each model, balanced across languages. Computes per-language
cross-entropy first, then averages across languages so each world state
contributes equally regardless of corpus size.
"""
function evaluate_population(pop::Vector{ModelState},
                             eval_tokens_by_lang::Dict{String,Vector{Vector{Int}}},
                             cfg::ModelConfig)
    K = length(pop)
    risks = Vector{Float64}(undef, K)
    Threads.@threads for i in 1:K
        risks[i] = mean_cross_entropy_balanced(pop[i].W, eval_tokens_by_lang, cfg;
                                                token_weighted=true)
    end
    fitnesses = exp.(-risks)
    (risks, fitnesses)
end

"""
    develop!(ms, train_tokens, n_steps, cfg, lr; weight_decay, rng, step_offset)

Continue training a model for n_steps on the given token sequences.
step_offset shifts the Adam step counter (for bias correction).
"""
function develop!(ms::ModelState, train_tokens::Vector{Vector{Int}},
                  n_steps::Int, cfg::ModelConfig, lr::Float64;
                  weight_decay::Float64=0.0,
                  rng::AbstractRNG=Random.default_rng(),
                  step_offset::Int=0)
    for step in 1:n_steps
        idx = rand(rng, 1:length(train_tokens))
        tokens = train_tokens[idx]
        length(tokens) < 2 && continue
        train_step_lr!(ms, tokens, step_offset + step, cfg, lr;
                       weight_decay=weight_decay)
    end
    ms
end

"""
    develop_balanced!(ms, train_tokens_by_lang, languages, n_steps, cfg, lr;
                      weight_decay, rng, step_offset)

Balanced development: each step first picks a language uniformly, then picks
a random chunk from that language. This ensures each language contributes
equally to development regardless of corpus size.
"""
function develop_balanced!(ms::ModelState,
                           train_tokens_by_lang::Dict{String,Vector{Vector{Int}}},
                           languages::Vector{String},
                           n_steps::Int, cfg::ModelConfig, lr::Float64;
                           weight_decay::Float64=0.0,
                           rng::AbstractRNG=Random.default_rng(),
                           step_offset::Int=0)
    n_lang = length(languages)
    for step in 1:n_steps
        lang = languages[rand(rng, 1:n_lang)]
        toks = train_tokens_by_lang[lang]
        idx = rand(rng, 1:length(toks))
        tokens = toks[idx]
        length(tokens) < 2 && continue
        train_step_lr!(ms, tokens, step_offset + step, cfg, lr;
                       weight_decay=weight_decay)
    end
    ms
end

"""
    run_generation!(pop, ...) -> GenerationRecord

Execute one generation of Wright-Fisher selection.
Offspring get fresh optimizer state (reset_optimizer=true in copy_model)
and mutation resets moments by default.
"""
function run_generation!(pop::Vector{ModelState},
                         train_tokens_by_lang::Dict{String,Vector{Vector{Int}}},
                         eval_tokens_by_lang::Dict{String,Vector{Vector{Int}}},
                         prefixes_by_lang::Dict{String,Dict{Int,Vector{Vector{Int}}}},
                         languages::Vector{String},
                         cfg::ModelConfig,
                         gen::Int,
                         tau::Float64;
                         K::Int=length(pop),
                         n_dev_steps::Int=20,
                         lr_dev::Float64=0.005,
                         sigma_mut::Float64=0.001,
                         weight_decay::Float64=0.0,
                         rng::AbstractRNG=Random.default_rng(),
                         step_offset::Int=0,
                         verbose::Bool=false)
    t0 = time()
    # 1. Evaluate pre-selection
    risks, fitnesses = evaluate_population(pop, eval_tokens_by_lang, cfg)
    r_bar_before = sum(risks) / length(risks)
    verbose && println("  gen " * string(gen) * " eval: " *
                       string(round(time() - t0, digits=1)) * "s, R_bar=" *
                       string(round(r_bar_before, digits=4)))

    # 2. Selection
    parents = wright_fisher_select(fitnesses, K; rng=rng)
    r_bar_sel = sum(risks[p] for p in parents) / K

    # Price equation terms
    cov_wR, w_bar, price_pred, r_sel_expected, delta_sel_sd =
        selection_stage_stats(risks, fitnesses, K)
    delta_sel = r_bar_sel - r_bar_before
    delta_sel_lo95 = price_pred - 1.96 * delta_sel_sd
    delta_sel_hi95 = price_pred + 1.96 * delta_sel_sd

    # 3. Create offspring (deep copy, fresh optimizer state)
    offspring = [copy_model(pop[p]; reset_optimizer=true) for p in parents]

    # 4. Development (balanced across languages, fresh optimizer state)
    for ms in offspring
        develop_balanced!(ms, train_tokens_by_lang, languages,
                          n_dev_steps, cfg, lr_dev;
                          weight_decay=weight_decay, rng=rng,
                          step_offset=step_offset)
    end
    risks_dev, _ = evaluate_population(offspring, eval_tokens_by_lang, cfg)
    r_bar_dev = sum(risks_dev) / length(risks_dev)
    delta_dev = r_bar_dev - r_bar_sel

    # 5. Mutation (resets optimizer by default)
    for ms in offspring
        mutate_weights!(ms, sigma_mut; rng=rng, reset_optimizer=true)
    end
    risks_mut, _ = evaluate_population(offspring, eval_tokens_by_lang, cfg)
    r_bar_mut = sum(risks_mut) / length(risks_mut)
    delta_mut = r_bar_mut - r_bar_dev

    # 6. Separation summaries (all models)
    ks = Vector{Float64}(undef, K)
    ss = Vector{Float64}(undef, K)
    Threads.@threads for idx in 1:K
        D = distance_matrix(offspring[idx].W, prefixes_by_lang, languages, cfg)
        s_val, k_val = separation_summary(D, tau)
        ks[idx] = Float64(k_val)
        ss[idx] = Float64(s_val)
    end

    verbose && println("  gen " * string(gen) * " done: " *
                       string(round(time() - t0, digits=1)) * "s, k=" *
                       string(round(sum(ks) / length(ks), digits=1)))

    # Replace population
    for i in 1:K
        pop[i] = offspring[i]
    end

    GenerationRecord(gen, risks, fitnesses, parents,
                     r_bar_sel, r_sel_expected, r_bar_dev, r_bar_mut,
                     delta_sel, price_pred, delta_sel_sd,
                     delta_sel_lo95, delta_sel_hi95,
                     delta_dev, delta_mut,
                     sum(ks) / length(ks), sum(ss) / length(ss))
end
