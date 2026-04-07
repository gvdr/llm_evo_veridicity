"""
Experiment 2: Exact Split-Merge Threshold on Empirical Corpora

Directly evaluates the explicit encoding-level objective

    J_{D,beta}(p) = L*_D(p) + beta H(p(W))

on corpus-induced finite ecologies. This replaces the earlier weight-decay
proxy with the theorem's own object.

For each corpus:
  - build a held-out finite ecology from empirical next-token laws;
  - enumerate all partitions of the language set exactly;
  - compute the exact global optimum path as beta increases;
  - compare each global transition to the theorem's local merge prediction
    from the split-versus-merge threshold theorem.

Run: julia --project=. scripts/exp2_split_merge.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Random
include(joinpath(@__DIR__, "..", "src", "data.jl"))

using LlmPaper

const OUTPUT_DIR = joinpath(@__DIR__, "..", "output")
const PREFIX_LENGTHS = [3, 5, 8]
const CHUNK_SIZE = 30
const CHUNK_OVERLAP = 5
const TOL = 1e-10

const ALL_LANGUAGES = ["english", "french", "german", "italian",
                       "spanish", "finnish", "portuguese"]

const CORPORA = [
    (name="dante",
     data_dir=joinpath(@__DIR__, "..", "data", "dante"),
     file_map=Dict(
         "english"    => "dante_english.txt",
         "french"     => "dante_french.txt",
         "german"     => "dante_german.txt",
         "italian"    => "dante_italian.txt",
         "spanish"    => "dante_spanish.txt",
         "finnish"    => "dante_finnish.txt",
         "portuguese" => "dante_portuguese.txt",
     )),
    (name="manifesto",
     data_dir=joinpath(@__DIR__, "..", "data", "manifesto"),
     file_map=Dict(
         "english"    => "manifesto_english.txt",
         "french"     => "manifesto_french.txt",
         "german"     => "manifesto_german.txt",
         "italian"    => "manifesto_italian.txt",
         "spanish"    => "manifesto_spanish.txt",
         "finnish"    => "manifesto_finnish.txt",
         "portuguese" => "manifesto_portuguese.txt",
     )),
]

function canonical_renumber(labels::Vector{Int})
    cells = partition_cells(labels)
    out = similar(labels)
    for (lab, cell) in enumerate(cells), idx in cell
        out[idx] = lab
    end
    out
end

function merge_cells(labels::Vector{Int},
                     cell_a::Vector{Int},
                     cell_b::Vector{Int})
    out = copy(labels)
    lab_a = labels[cell_a[1]]
    lab_b = labels[cell_b[1]]
    for i in eachindex(out)
        if out[i] == lab_b
            out[i] = lab_a
        end
    end
    canonical_renumber(out)
end

function best_partition_index(losses::Vector{Float64},
                              complexities::Vector{Float64},
                              beta::Float64)
    vals = losses .+ beta .* complexities
    argmin(vals)
end

function global_transition(current_idx::Int,
                           losses::Vector{Float64},
                           complexities::Vector{Float64})
    loss_c = losses[current_idx]
    comp_c = complexities[current_idx]
    best_beta = Inf
    best_idx = 0
    for i in eachindex(losses)
        i == current_idx && continue
        comp_o = complexities[i]
        comp_o < comp_c - TOL || continue
        loss_o = losses[i]
        beta = max(0.0, (loss_o - loss_c) / (comp_c - comp_o))
        if beta < best_beta - TOL
            best_beta = beta
            best_idx = i
        elseif abs(beta - best_beta) <= TOL && best_idx != 0
            val_i = loss_o + beta * comp_o
            val_best = losses[best_idx] + beta * complexities[best_idx]
            if val_i < val_best - TOL
                best_idx = i
            end
        end
    end
    (best_beta, best_idx)
end

function local_threshold_rows(fe::FiniteEcology,
                              labels::Vector{Int},
                              languages::Vector{String})
    rows = NamedTuple[]
    cells = partition_cells(labels)
    for i in 1:length(cells), j in (i + 1):length(cells)
        A = cells[i]
        B = cells[j]
        pi_A = sum(fe.pi[w] for w in A)
        pi_B = sum(fe.pi[w] for w in B)
        lam = pi_A / (pi_A + pi_B)
        h = binary_entropy(lam)
        gain = split_gain(fe, A, B)
        beta_star = h > 0 ? gain / h : Inf
        merged = merge_cells(labels, A, B)
        push!(rows, (cell_a=copy(A),
                     cell_b=copy(B),
                     names_a=[languages[w] for w in A],
                     names_b=[languages[w] for w in B],
                     lambda=lam,
                     gain=gain,
                     entropy_cost=h,
                     beta_star=beta_star,
                     merged=merged))
    end
    sort(rows, by=row -> row.beta_star)
end

function partition_label(languages::Vector{String}, labels::Vector{Int})
    cells = partition_cells(labels)
    rendered = String[]
    for cell in cells
        push!(rendered, "{" * join(languages[cell], "+") * "}")
    end
    join(rendered, " | ")
end

function run_corpus(spec)
    println("=" ^ 70)
    println("Experiment 2: " * uppercasefirst(spec.name))
    println("=" ^ 70)
    println()

    chunks_by_lang, tok = load_text_ecology(spec.data_dir, spec.file_map;
                                            chunk_size=CHUNK_SIZE,
                                            overlap=CHUNK_OVERLAP)
    rng_split = MersenneTwister(12345)
    test_chunks = Dict{String, Vector{String}}()
    eval_tokens_by_lang = Dict{String, Vector{Vector{Int}}}()
    for lang in ALL_LANGUAGES
        chunks = chunks_by_lang[lang]
        n = length(chunks)
        perm = randperm(rng_split, n)
        split_idx = round(Int, n * 0.8)
        test_chunks[lang] = chunks[perm[(split_idx + 1):end]]
        eval_tokens_by_lang[lang] = [tokenize_chunk(tok, c) for c in test_chunks[lang]]
        println("  " * lang * ": " * string(length(eval_tokens_by_lang[lang])) *
                " held-out chunks")
    end
    println()

    fe, valid_lengths, _ = empirical_prefix_ecology(eval_tokens_by_lang,
                                                    ALL_LANGUAGES,
                                                    PREFIX_LENGTHS,
                                                    tok.vocab_size;
                                                    uniform_worlds=true,
                                                    context_weighting=:uniform)
    partitions = all_partitions(length(ALL_LANGUAGES))
    losses = [bayes_opt_loss(fe, labels) for labels in partitions]
    complexities = [partition_complexity(fe, labels) for labels in partitions]
    eco_labels = ecology_partition(fe)
    eco_idx = findfirst(labels -> same_partition(labels, eco_labels), partitions)
    eco_idx === nothing && error("failed to find ecology partition in enumeration")

    println("Exact contexts: prefix lengths " * string(valid_lengths))
    println("Ecology partition: " * partition_label(ALL_LANGUAGES, eco_labels))
    println("Ecology classes k=" * string(n_classes(eco_labels)) *
            " excess=" * string(round(excess_loss(fe, eco_labels), digits=8)) *
            " complexity=" * string(round(partition_complexity(fe, eco_labels), digits=8)))
    println()

    steps = NamedTuple[]
    current_idx = eco_idx
    visited = Set([current_idx])
    while n_classes(partitions[current_idx]) > 1
        labels = partitions[current_idx]
        local_rows = local_threshold_rows(fe, labels, ALL_LANGUAGES)
        isempty(local_rows) && break
        predicted = local_rows[1]
        beta_next, next_idx = global_transition(current_idx, losses, complexities)
        next_idx == 0 && break
        next_labels = partitions[next_idx]
        local_match = same_partition(predicted.merged, next_labels)
        push!(steps, (from=copy(labels),
                      to=copy(next_labels),
                      beta_star=beta_next,
                      predicted_beta=predicted.beta_star,
                      predicted_merge=copy(predicted.merged),
                      predicted_names=(predicted.names_a, predicted.names_b),
                      local_match=local_match))
        println("Transition " * string(length(steps)) * ":")
        println("  global beta* = " * string(round(beta_next, digits=8)))
        println("  current      = " * partition_label(ALL_LANGUAGES, labels))
        println("  next global  = " * partition_label(ALL_LANGUAGES, next_labels))
        println("  local best   = " *
                "{" * join(predicted.names_a, "+") * "} vs {" *
                join(predicted.names_b, "+") * "}, beta*=" *
                string(round(predicted.beta_star, digits=8)))
        println("  local match  = " * string(local_match))
        println()
        next_idx in visited && break
        push!(visited, next_idx)
        current_idx = next_idx
    end

    singleton_thresholds = local_threshold_rows(fe, eco_labels, ALL_LANGUAGES)
    println("Singleton/surface split thresholds from the ecology partition:")
    for (rank, row) in enumerate(singleton_thresholds)
        println("  " * string(rank) * ". {" * join(row.names_a, "+") * "} vs {" *
                join(row.names_b, "+") * "} gain=" *
                string(round(row.gain, digits=8)) * " beta*=" *
                string(round(row.beta_star, digits=8)))
    end
    println()

    grid_betas = sort(unique(vcat([0.0],
                                  [0.5 * s.beta_star for s in steps if isfinite(s.beta_star)],
                                  [s.beta_star for s in steps if isfinite(s.beta_star)],
                                  [1.5 * s.beta_star for s in steps if isfinite(s.beta_star)])))
    frontier = NamedTuple[]
    for beta in grid_betas
        idx = best_partition_index(losses, complexities, beta)
        labels = partitions[idx]
        push!(frontier, (beta=beta,
                         k=n_classes(labels),
                         excess=excess_loss(fe, labels),
                         complexity=partition_complexity(fe, labels),
                         labels=copy(labels)))
    end

    (fe=fe,
     valid_lengths=valid_lengths,
     eco_labels=eco_labels,
     singleton_thresholds=singleton_thresholds,
     steps=steps,
     frontier=frontier)
end

function main()
    t_start = time()
    println("=" ^ 70)
    println("Experiment 2: Exact Split-Merge Threshold on Empirical Corpora")
    println("=" ^ 70)
    println()

    outputs = Dict{String, Any}()
    for spec in CORPORA
        outputs[spec.name] = run_corpus(spec)
    end

    println("=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println()
    for spec in CORPORA
        out = outputs[spec.name]
        n_match = count(s.local_match for s in out.steps)
        println(spec.name * ": " * string(n_match) * "/" *
                string(length(out.steps)) *
                " global transitions matched the theorem's local best merge")
    end
    println()

    mkpath(OUTPUT_DIR)
    outpath = joinpath(OUTPUT_DIR, "exp2_split_merge_results.txt")
    open(outpath, "w") do io
        println(io, "# Experiment 2: Exact Split-Merge Threshold on Empirical Corpora")
        println(io, "# PREFIX_LENGTHS=" * string(PREFIX_LENGTHS))
        println(io, "# CHUNK_SIZE=" * string(CHUNK_SIZE) *
                    " CHUNK_OVERLAP=" * string(CHUNK_OVERLAP))
        println(io, "# Languages: " * join(ALL_LANGUAGES, ", "))
        println(io, "")
        for spec in CORPORA
            out = outputs[spec.name]
            println(io, "# corpus=" * spec.name)
            println(io, "# valid_prefix_lengths=" * string(out.valid_lengths))
            println(io, "# ecology_partition=" * string(out.eco_labels))
            println(io, "# singleton thresholds from ecology partition")
            println(io, "rank,pair,gain,beta_star")
            for (rank, row) in enumerate(out.singleton_thresholds)
                println(io, string(rank) * "," *
                            "\"" * "{" * join(row.names_a, "+") * "} vs {" *
                            join(row.names_b, "+") * "}" * "\"" * "," *
                            string(round(row.gain, digits=10)) * "," *
                            string(round(row.beta_star, digits=10)))
            end
            println(io, "")
            println(io, "# exact global transition path")
            println(io, "step,beta_star,current_labels,next_labels,predicted_local_merge,predicted_beta,local_match")
            for (step_idx, step) in enumerate(out.steps)
                println(io, string(step_idx) * "," *
                            string(round(step.beta_star, digits=10)) * "," *
                            "\"" * string(step.from) * "\"" * "," *
                            "\"" * string(step.to) * "\"" * "," *
                            "\"" * string(step.predicted_merge) * "\"" * "," *
                            string(round(step.predicted_beta, digits=10)) * "," *
                            string(step.local_match))
            end
            println(io, "")
            println(io, "# exact frontier samples")
            println(io, "beta,k,excess,complexity,labels")
            for row in out.frontier
                println(io, string(round(row.beta, digits=10)) * "," *
                            string(row.k) * "," *
                            string(round(row.excess, digits=10)) * "," *
                            string(round(row.complexity, digits=10)) * "," *
                            "\"" * string(row.labels) * "\"")
            end
            println(io, "")
        end
    end

    println("Results saved to " * outpath)
    println("Total time: " * string(round(time() - t_start, digits=1)) * "s")
end

main()
