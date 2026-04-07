"""
Figure for Experiment 3 / redesigned evolutionary diagnostics.

Generates a four-panel figure with:
  1. Mean risk over generations
  2. Mean behavioural separation k over generations
  3. Selection-stage delta_sel with exact conditional expectation and 95% band
  4. Standardized selection residual z = (delta_sel - E[delta_sel]) / sd

Run:
  julia --project=. scripts/fig_exp3_diagnostics.jl
"""

using CairoMakie

const ROOT = joinpath(@__DIR__, "..")
const RESULT_PATH = joinpath(ROOT, "output", "exp3_population_results.txt")
const FIG_DIR = joinpath(ROOT, "figures")

const CB_BLUE = colorant"#0072B2"
const CB_ORANGE = colorant"#E69F00"
const CB_GREEN = colorant"#009E73"
const CB_RED = colorant"#D55E00"

struct Exp3Row
    run::Int
    gen::Int
    r_bar_before::Float64
    delta_sel::Float64
    price_predicted::Float64
    delta_sel_sd::Float64
    delta_sel_lo95::Float64
    delta_sel_hi95::Float64
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
                            parse(Float64, parts[5]),
                            parse(Float64, parts[6]),
                            parse(Float64, parts[7]),
                            parse(Float64, parts[8]),
                            parse(Float64, parts[9]),
                            parse(Float64, parts[12])))
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

function main()
    mkpath(FIG_DIR)
    rows = parse_exp3_results(RESULT_PATH)

    gens, runs, risk_series = series_by_run(rows, :r_bar_before)
    _, _, k_series = series_by_run(rows, :mean_k)
    _, _, delta_series = series_by_run(rows, :delta_sel)
    _, _, expected_series = series_by_run(rows, :price_predicted)
    _, _, lo_series = series_by_run(rows, :delta_sel_lo95)
    _, _, hi_series = series_by_run(rows, :delta_sel_hi95)
    _, _, sd_series = series_by_run(rows, :delta_sel_sd)

    risk_mean = mean_series(runs, risk_series)
    k_mean = mean_series(runs, k_series)
    delta_mean = mean_series(runs, delta_series)
    expected_mean = mean_series(runs, expected_series)
    lo_mean = mean_series(runs, lo_series)
    hi_mean = mean_series(runs, hi_series)

    z_series = Dict{Int, Vector{Float64}}()
    for run in runs
        zvals = Vector{Float64}(undef, length(gens))
        for idx in eachindex(gens)
            sd = sd_series[run][idx]
            zvals[idx] = sd > 0 ? (delta_series[run][idx] - expected_series[run][idx]) / sd : 0.0
        end
        z_series[run] = zvals
    end
    z_mean = mean_series(runs, z_series)

    fig = Figure(size=(1100, 760))

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

    ax3 = Axis(fig[2, 1];
               xlabel="Generation",
               ylabel="Selection-stage ΔR̄")
    band!(ax3, gens, lo_mean, hi_mean; color=(CB_GREEN, 0.18))
    lines!(ax3, gens, expected_mean; color=CB_GREEN, linewidth=3)
    for run in runs
        lines!(ax3, gens, delta_series[run]; color=(:gray45, 0.28), linewidth=1.2)
    end
    lines!(ax3, gens, delta_mean; color=CB_RED, linewidth=3)

    ax4 = Axis(fig[2, 2];
               xlabel="Generation",
               ylabel="Standardized residual  z")
    hlines!(ax4, [0.0]; color=(:black, 0.55), linewidth=1.5)
    hlines!(ax4, [-2.0, 2.0]; color=(:gray40, 0.55), linestyle=:dash, linewidth=1.5)
    for run in runs
        lines!(ax4, gens, z_series[run]; color=(:gray45, 0.28), linewidth=1.2)
    end
    lines!(ax4, gens, z_mean; color=CB_BLUE, linewidth=3)

    Label(fig[0, 1:2],
          "Experiment 3 Diagnostics: Trajectories and Exact Conditional Selection Checks";
          fontsize=18, font=:bold)

    save(joinpath(FIG_DIR, "exp3_population_diagnostics.pdf"), fig)
    save(joinpath(FIG_DIR, "exp3_population_diagnostics.png"), fig)
    println("Saved figures to " * FIG_DIR)
end

main()
