"""
Figures for Experiment 2 / exact split-merge path on empirical corpora.

Generates one figure with:
  1. Exact frontier k(beta) for each corpus
  2. Global transition steps with local-match indicator

Run:
  julia --project=. scripts/fig_exp2_split_merge.jl
"""

using CairoMakie

const ROOT = joinpath(@__DIR__, "..")
const RESULT_PATH = joinpath(ROOT, "output", "exp2_split_merge_results.txt")
const FIG_DIR = joinpath(ROOT, "figures")
const CB_BLUE = colorant"#0072B2"
const CB_ORANGE = colorant"#E69F00"
const CB_MAGENTA = colorant"#CC79A7"

struct FrontierRow
    corpus::String
    beta::Float64
    k::Int
end

struct TransitionRow
    corpus::String
    step::Int
    beta::Float64
    local_match::Bool
end

function parse_exp2_results(path::String)
    frontier = FrontierRow[]
    transitions = TransitionRow[]
    current_corpus = ""
    mode = :none
    for line in eachline(path)
        isempty(line) && continue
        if startswith(line, "# corpus=")
            current_corpus = split(split(line, "=")[2])[1]
            mode = :none
            continue
        elseif startswith(line, "# exact global transition path")
            mode = :transitions
            continue
        elseif startswith(line, "# exact frontier samples")
            mode = :frontier
            continue
        elseif startswith(line, "#")
            continue
        end
        if mode == :transitions
            startswith(line, "step,") && continue
            m = match(r"^(\d+),([^,]+),.*,(true|false)$", line)
            m === nothing && continue
            push!(transitions, TransitionRow(current_corpus,
                                             parse(Int, m.captures[1]),
                                             parse(Float64, m.captures[2]),
                                             parse(Bool, m.captures[3])))
        elseif mode == :frontier
            startswith(line, "beta,") && continue
            parts = split(line, ",", limit=3)
            length(parts) == 3 || continue
            push!(frontier, FrontierRow(current_corpus,
                                        parse(Float64, parts[1]),
                                        parse(Int, parts[2])))
        end
    end
    frontier, transitions
end

function main()
    mkpath(FIG_DIR)
    frontier, transitions = parse_exp2_results(RESULT_PATH)
    corpora = ["dante", "manifesto"]
    colors = Dict("dante" => CB_BLUE, "manifesto" => CB_ORANGE)
    labels = Dict("dante" => "Commedia", "manifesto" => "Manifesto")

    fig = Figure(size=(980, 420))

    ax1 = Axis(fig[1, 1];
               xlabel="Complexity weight  β",
               ylabel="Optimal number of classes  k")
    for corpus in corpora
        rows = sort([r for r in frontier if r.corpus == corpus], by=r -> r.beta)
        betas = [r.beta for r in rows]
        ks = [r.k for r in rows]
        lines!(ax1, betas, ks; color=colors[corpus], linewidth=3, label=labels[corpus])
        scatter!(ax1, betas, ks; color=colors[corpus], markersize=9)
    end
    axislegend(ax1; position=:rt)

    ax2 = Axis(fig[1, 2];
               xlabel="Transition step",
               ylabel="β* of realized global transition")
    for corpus in corpora
        rows = sort([r for r in transitions if r.corpus == corpus], by=r -> r.step)
        steps = [r.step for r in rows]
        betas = [r.beta for r in rows]
        match_idx = findall(r -> r.local_match, rows)
        mismatch_idx = findall(r -> !r.local_match, rows)
        lines!(ax2, steps, betas; color=(colors[corpus], 0.45), linewidth=2)
        if !isempty(match_idx)
            scatter!(ax2, steps[match_idx], betas[match_idx];
                     color=colors[corpus], markersize=14,
                     label=labels[corpus] * " local match")
        end
        if !isempty(mismatch_idx)
            scatter!(ax2, steps[mismatch_idx], betas[mismatch_idx];
                     color=CB_MAGENTA, marker=:utriangle, markersize=16,
                     label=labels[corpus] * " mismatch")
        end
    end
    axislegend(ax2; position=:lt)
    save(joinpath(FIG_DIR, "exp2_split_merge_exact_path.pdf"), fig)
    save(joinpath(FIG_DIR, "exp2_split_merge_exact_path.png"), fig)
    println("Saved figures to " * FIG_DIR)
end

main()
