"""
Figure for Experiments 6-7 / two-ecology neural validation.

Two panels:
  Left:  Static sweep — normalized evaluation signal vs alpha for both tasks
  Right: Population selection — mean alpha trajectories for both tasks

Run:
  julia --project=. scripts/fig_exp67_two_ecology.jl
"""

using CairoMakie

const ROOT = joinpath(@__DIR__, "..")
const FIG_DIR = joinpath(ROOT, "figures")
const EXP6_PATH = joinpath(ROOT, "output", "exp6_bracket_ecology_results.txt")
const EXP7_PATH = joinpath(ROOT, "output", "exp7_transfer_ecology_results.txt")

const CB_ORANGE = colorant"#E69F00"
const CB_GREEN = colorant"#009E73"

struct StaticRow
    alpha::Float64
    value::Float64
end

struct PopRow
    gen::Int
    alpha_bar::Float64
end

function parse_static_rows(path::String, value_col::Int)
    rows = StaticRow[]
    in_table = false
    for line in eachline(path)
        isempty(line) && continue
        if startswith(line, "# Stage 1")
            in_table = false
            continue
        end
        if startswith(line, "alpha,")
            in_table = true
            continue
        end
        if startswith(line, "# Stage 2")
            break
        end
        in_table || continue
        startswith(line, "#") && continue
        parts = split(line, ",")
        push!(rows, StaticRow(parse(Float64, parts[1]),
                              parse(Float64, parts[value_col])))
    end
    rows
end

function parse_pop_rows(path::String)
    rows = PopRow[]
    in_table = false
    for line in eachline(path)
        isempty(line) && continue
        if startswith(line, "gen,")
            in_table = true
            continue
        end
        in_table || continue
        startswith(line, "#") && continue
        parts = split(line, ",")
        push!(rows, PopRow(parse(Int, parts[1]),
                           parse(Float64, parts[2])))
    end
    rows
end

function alpha_vals(rows::Vector{StaticRow})
    [row.alpha for row in rows]
end

function metric_vals(rows::Vector{StaticRow})
    [row.value for row in rows]
end

function gen_vals(rows::Vector{PopRow})
    [row.gen for row in rows]
end

function alpha_bar_vals(rows::Vector{PopRow})
    [row.alpha_bar for row in rows]
end

function normalize_decreasing(vals::Vector{Float64})
    # For metrics where lower is better (summary CE):
    # score = (val_0 - val) / (val_0 - val_best)
    v0 = vals[1]
    vbest = minimum(vals)
    denom = v0 - vbest
    denom < 1e-15 && return zeros(length(vals))
    [(v0 - v) / denom for v in vals]
end

function normalize_increasing(vals::Vector{Float64})
    # For metrics where higher is better (discrimination):
    # score = (val - val_0) / (val_best - val_0)
    v0 = vals[1]
    vbest = maximum(vals)
    denom = vbest - v0
    denom < 1e-15 && return zeros(length(vals))
    [(v - v0) / denom for v in vals]
end

function main()
    mkpath(FIG_DIR)

    exp6_static = parse_static_rows(EXP6_PATH, 3)  # summary_ce (lower = better)
    exp7_static = parse_static_rows(EXP7_PATH, 4)  # discrimination (higher = better)
    exp6_pop = parse_pop_rows(EXP6_PATH)
    exp7_pop = parse_pop_rows(EXP7_PATH)

    # Normalize both static sweeps to [0, 1]
    exp6_norm = normalize_decreasing(metric_vals(exp6_static))
    exp7_norm = normalize_increasing(metric_vals(exp7_static))

    fig = Figure(size=(1000, 400))

    # Left: normalized evaluation signal vs alpha
    ax1 = Axis(fig[1, 1];
               xlabel=L"\alpha",
               ylabel="Evaluation signal captured",
               title="Static sweep")
    lines!(ax1, alpha_vals(exp6_static), exp6_norm;
           color=CB_ORANGE, linewidth=3, label="Balance checking")
    scatter!(ax1, alpha_vals(exp6_static), exp6_norm;
             color=CB_ORANGE, markersize=12)
    lines!(ax1, alpha_vals(exp7_static), exp7_norm;
           color=CB_GREEN, linewidth=3, label="Code validation")
    scatter!(ax1, alpha_vals(exp7_static), exp7_norm;
             color=CB_GREEN, markersize=12)
    hlines!(ax1, [0.0]; color=(:black, 0.3), linewidth=1)
    axislegend(ax1; position=:rb)

    # Right: population alpha trajectories
    ax2 = Axis(fig[1, 2];
               xlabel="Generation",
               ylabel=L"\bar{\alpha}",
               title="Population selection")
    lines!(ax2, gen_vals(exp6_pop), alpha_bar_vals(exp6_pop);
           color=CB_ORANGE, linewidth=3, label="Balance checking")
    scatter!(ax2, gen_vals(exp6_pop), alpha_bar_vals(exp6_pop);
             color=CB_ORANGE, markersize=8)
    lines!(ax2, gen_vals(exp7_pop), alpha_bar_vals(exp7_pop);
           color=CB_GREEN, linewidth=3, label="Code validation")
    scatter!(ax2, gen_vals(exp7_pop), alpha_bar_vals(exp7_pop);
             color=CB_GREEN, markersize=8)
    axislegend(ax2; position=:rb)

    save(joinpath(FIG_DIR, "exp67_two_ecology.pdf"), fig)
    save(joinpath(FIG_DIR, "exp67_two_ecology.png"), fig)
    println("Saved figures to " * FIG_DIR)
end

main()
