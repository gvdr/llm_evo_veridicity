"""
Figures for Experiment 1 / static empirical ecology.

Generates one figure with:
  1. Exact ecology partition size vs ecology size
  2. Exact excess gap and exact JS excess vs ecology size

Run:
  julia --project=. scripts/fig_exp1_static.jl
"""

using CairoMakie

const ROOT = joinpath(@__DIR__, "..")
const RESULT_PATH = joinpath(ROOT, "output", "exp1_static_results.txt")
const FIG_DIR = joinpath(ROOT, "figures")
const CB_BLUE = colorant"#0072B2"
const CB_ORANGE = colorant"#E69F00"

struct Exp1Row
    corpus::String
    t::Int
    seed::Int
    exact_ecology_k::Int
    exact_js_excess::Float64
    exact_excess_gap::Float64
end

function parse_exp1_results(path::String)
    rows = Exp1Row[]
    in_table = false
    for line in eachline(path)
        isempty(line) && continue
        startswith(line, "# Full distance matrices") && break
        if startswith(line, "corpus,")
            in_table = true
            continue
        end
        in_table || continue
        startswith(line, "#") && continue
        parts = split(line, ",")
        length(parts) == 14 || continue
        push!(rows, Exp1Row(parts[1],
                            parse(Int, parts[2]),
                            parse(Int, parts[3]),
                            parse(Int, parts[4]),
                            parse(Float64, parts[11]),
                            parse(Float64, parts[12])))
    end
    rows
end

function grouped_mean(rows::Vector{Exp1Row}, field::Symbol)
    out = Dict{Tuple{String, Int}, Float64}()
    corpora = sort(unique(r.corpus for r in rows))
    ts = sort(unique(r.t for r in rows))
    for corpus in corpora, t in ts
        subset = [r for r in rows if r.corpus == corpus && r.t == t]
        vals = [Float64(getproperty(r, field)) for r in subset]
        out[(corpus, t)] = sum(vals) / length(vals)
    end
    out
end

function grouped_exact_k(rows::Vector{Exp1Row})
    out = Dict{Tuple{String, Int}, Int}()
    corpora = sort(unique(r.corpus for r in rows))
    ts = sort(unique(r.t for r in rows))
    for corpus in corpora, t in ts
        subset = [r for r in rows if r.corpus == corpus && r.t == t]
        out[(corpus, t)] = subset[1].exact_ecology_k
    end
    out
end

function main()
    mkpath(FIG_DIR)
    rows = parse_exp1_results(RESULT_PATH)
    corpora = ["alice", "manifesto"]
    labels = Dict("alice" => "Alice", "manifesto" => "Manifesto")
    ts = sort(unique(r.t for r in rows))
    x = 1:length(ts)

    mean_js = grouped_mean(rows, :exact_js_excess)
    mean_gap = grouped_mean(rows, :exact_excess_gap)
    exact_k = grouped_exact_k(rows)

    fig = Figure(size=(980, 420))

    ax1 = Axis(fig[1, 1];
               xlabel="Ecology size  T",
               ylabel="Partition size")
    colors = Dict("alice" => CB_BLUE, "manifesto" => CB_ORANGE)
    exact_vals_ref = [exact_k[("alice", t)] for t in ts]
    lines!(ax1, x, exact_vals_ref; color=:black, linewidth=3,
           label="Exact ecology k (both corpora)")
    scatter!(ax1, x, exact_vals_ref; color=:black, markersize=14)
    ax1.xticks = (collect(x), string.(ts))
    axislegend(ax1; position=:lt)

    ax2 = Axis(fig[1, 2];
               xlabel="Ecology size  T",
               ylabel="Exact loss term")
    for corpus in corpora
        gap_vals = [mean_gap[(corpus, t)] for t in ts]
        js_vals = [mean_js[(corpus, t)] for t in ts]
        lines!(ax2, x, gap_vals; color=colors[corpus], linewidth=3,
               label=labels[corpus] * " exact excess")
        scatter!(ax2, x, js_vals; color=(colors[corpus], 0.6),
                 marker=:circle, markersize=12,
                 label=labels[corpus] * " exact JS excess")
    end
    ax2.xticks = (collect(x), string.(ts))
    axislegend(ax2; position=:rb)

    save(joinpath(FIG_DIR, "exp1_static_empirical_ecology.pdf"), fig)
    save(joinpath(FIG_DIR, "exp1_static_empirical_ecology.png"), fig)
    println("Saved figures to " * FIG_DIR)
end

main()
