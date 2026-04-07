"""
Paper-oriented figure for Experiment 3 / selection-stage diagnostics.

Generates a two-panel figure with:
  1. Cumulative selection-stage change in mean risk
  2. Histogram of standardized residuals z

Run:
  julia --project=. scripts/fig_exp3_selection_summary.jl
"""

using CairoMakie

const ROOT = joinpath(@__DIR__, "..")
const RESULT_PATH = joinpath(ROOT, "output", "exp3_population_results.txt")
const FIG_DIR = joinpath(ROOT, "figures")

const CB_BLUE = colorant"#0072B2"
const CB_GREEN = colorant"#009E73"

struct Exp3Row
    run::Int
    gen::Int
    delta_sel::Float64
    price_predicted::Float64
    delta_sel_sd::Float64
end

function parse_exp3_results(path::String)
    rows = Exp3Row[]
    for line in eachline(path)
        isempty(line) && continue
        startswith(line, "#") && continue
        startswith(line, "run,") && continue
        parts = split(line, ",")
        length(parts) >= 7 || continue
        push!(rows, Exp3Row(parse(Int, parts[1]),
                            parse(Int, parts[2]),
                            parse(Float64, parts[5]),
                            parse(Float64, parts[6]),
                            parse(Float64, parts[7])))
    end
    rows
end

function runs_and_gens(rows::Vector{Exp3Row})
    runs = sort(unique(row.run for row in rows))
    gens = sort(unique(row.gen for row in rows))
    runs, gens
end

function series_by_run(rows::Vector{Exp3Row}, field::Symbol)
    runs, gens = runs_and_gens(rows)
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

function cumulative_series(vals::Vector{Float64})
    out = similar(vals)
    s = 0.0
    for i in eachindex(vals)
        s += vals[i]
        out[i] = s
    end
    out
end

function flattened_values(runs::Vector{Int}, series::Dict{Int, Vector{Float64}})
    out = Float64[]
    for run in runs
        append!(out, series[run])
    end
    out
end

function main()
    mkpath(FIG_DIR)
    rows = parse_exp3_results(RESULT_PATH)

    gens, runs, delta_series = series_by_run(rows, :delta_sel)
    _, _, expected_series = series_by_run(rows, :price_predicted)
    _, _, sd_series = series_by_run(rows, :delta_sel_sd)

    cumulative = Dict{Int, Vector{Float64}}()
    z_series = Dict{Int, Vector{Float64}}()
    for run in runs
        cumulative[run] = cumulative_series(delta_series[run])
        zvals = Vector{Float64}(undef, length(gens))
        for i in eachindex(gens)
            sd = sd_series[run][i]
            zvals[i] = sd > 0 ? (delta_series[run][i] - expected_series[run][i]) / sd : 0.0
        end
        z_series[run] = zvals
    end

    cumulative_mean = mean_series(runs, cumulative)
    z_all = flattened_values(runs, z_series)

    fig = Figure(size=(980, 420))

    ax1 = Axis(fig[1, 1];
               xlabel="Generation",
               ylabel="Cumulative selection ΔR̄")
    for run in runs
        lines!(ax1, gens, cumulative[run]; color=(:gray45, 0.35), linewidth=1.4)
    end
    lines!(ax1, gens, cumulative_mean; color=CB_BLUE, linewidth=3)
    hlines!(ax1, [0.0]; color=(:black, 0.4), linewidth=1.2)

    ax2 = Axis(fig[1, 2];
               xlabel="Standardized residual  z",
               ylabel="Count")
    hist!(ax2, z_all; bins=-3.5:0.35:3.5, color=(CB_GREEN, 0.78), strokewidth=1.0)
    vlines!(ax2, [0.0]; color=(:black, 0.55), linewidth=1.5)
    vlines!(ax2, [-2.0, 2.0]; color=(:gray40, 0.6), linestyle=:dash, linewidth=1.5)

    save(joinpath(FIG_DIR, "exp3_selection_summary.pdf"), fig)
    save(joinpath(FIG_DIR, "exp3_selection_summary.png"), fig)
    println("Saved figures to " * FIG_DIR)
end

main()
