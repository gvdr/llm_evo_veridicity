"""
Figure for Experiment 3 / evolutionary dynamics.

Generates one figure with:
  1. Mean risk over generations (runs shown faintly, mean shown boldly)
  2. Mean behavioural separation k over generations (runs shown faintly, mean shown boldly)

Run:
  julia --project=. scripts/fig_exp3_dynamics.jl
"""

using CairoMakie

const ROOT = joinpath(@__DIR__, "..")
const RESULT_PATH = joinpath(ROOT, "output", "exp3_population_results.txt")
const FIG_DIR = joinpath(ROOT, "figures")
const CB_BLUE = colorant"#0072B2"
const CB_ORANGE = colorant"#E69F00"

struct Exp3Row
    run::Int
    gen::Int
    r_bar_before::Float64
    mean_k::Float64
end

function parse_exp3_results(path::String)
    rows = Exp3Row[]
    for line in eachline(path)
        isempty(line) && continue
        startswith(line, "#") && continue
        startswith(line, "run,") && continue
        parts = split(line, ",")
        length(parts) >= 12 || continue
        push!(rows, Exp3Row(parse(Int, parts[1]),
                            parse(Int, parts[2]),
                            parse(Float64, parts[3]),
                            parse(Float64, parts[12])))
    end
    rows
end

function series_by_run(rows::Vector{Exp3Row}, field::Symbol)
    runs = sort(unique(row.run for row in rows))
    gens = sort(unique(row.gen for row in rows))
    series = Dict{Int, Vector{Float64}}()
    for run in runs
        vals = Vector{Float64}(undef, length(gens))
        for (idx, gen) in enumerate(gens)
            row = only(filter(r -> r.run == run && r.gen == gen, rows))
            vals[idx] = getproperty(row, field)
        end
        series[run] = vals
    end
    gens, runs, series
end

function mean_series(runs::Vector{Int}, series::Dict{Int, Vector{Float64}})
    n = length(first(values(series)))
    out = zeros(n)
    for run in runs
        out .+= series[run]
    end
    out ./ length(runs)
end

function main()
    mkpath(FIG_DIR)
    rows = parse_exp3_results(RESULT_PATH)
    gens, runs, risk_series = series_by_run(rows, :r_bar_before)
    _, _, k_series = series_by_run(rows, :mean_k)
    risk_mean = mean_series(runs, risk_series)
    k_mean = mean_series(runs, k_series)

    fig = Figure(size=(980, 420))

    ax1 = Axis(fig[1, 1];
               xlabel="Generation",
               ylabel="Mean risk  R̄")
    for run in runs
        lines!(ax1, gens, risk_series[run]; color=(:gray50, 0.35), linewidth=1.5)
    end
    lines!(ax1, gens, risk_mean; color=CB_BLUE, linewidth=3)

    ax2 = Axis(fig[1, 2];
               xlabel="Generation",
               ylabel="Mean behavioural separation  k")
    for run in runs
        lines!(ax2, gens, k_series[run]; color=(:gray50, 0.35), linewidth=1.5)
    end
    lines!(ax2, gens, k_mean; color=CB_ORANGE, linewidth=3)

    save(joinpath(FIG_DIR, "exp3_population_dynamics.pdf"), fig)
    save(joinpath(FIG_DIR, "exp3_population_dynamics.png"), fig)
    println("Saved figures to " * FIG_DIR)
end

main()
