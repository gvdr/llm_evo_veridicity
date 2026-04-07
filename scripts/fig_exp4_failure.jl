"""
Figure for Experiment 4 / off-ecology failure.

Generates one figure with:
  1. Per-model cross-entropy by probe category
  2. Pairwise inter-model JS by probe category

Run:
  julia --project=. scripts/fig_exp4_failure.jl
"""

using CairoMakie

const ROOT = joinpath(@__DIR__, "..")
const RESULT_PATH = joinpath(ROOT, "output", "exp4_off_ecology_results.txt")
const FIG_DIR = joinpath(ROOT, "figures")
const CB_BLUE = colorant"#0072B2"
const CB_ORANGE = colorant"#E69F00"
const CB_MAGENTA = colorant"#CC79A7"

function parse_model_rows(path::String)
    rows = Dict{String, Vector{Float64}}()
    in_models = false
    for line in eachline(path)
        if startswith(line, "model,")
            headers = split(line, ",")
            for h in headers[2:end]
                rows[h] = Float64[]
            end
            in_models = true
            continue
        end
        in_models || continue
        isempty(line) && break
        startswith(line, "#") && break
        parts = split(line, ",")
        for (idx, h) in enumerate(keys(rows))
            # preserve insertion order of parsed columns
        end
    end

    headers = String[]
    in_models = false
    for line in eachline(path)
        if startswith(line, "model,")
            headers = split(line, ",")[2:end]
            for h in headers
                rows[h] = Float64[]
            end
            in_models = true
            continue
        end
        in_models || continue
        isempty(line) && break
        startswith(line, "#") && break
        parts = split(line, ",")
        for (j, h) in enumerate(headers)
            push!(rows[h], parse(Float64, parts[j + 1]))
        end
    end
    rows
end

function parse_js_block(path::String, header::String)
    capture = false
    for line in eachline(path)
        if startswith(line, header)
            capture = true
            continue
        end
        capture || continue
        isempty(line) && return Float64[]
        startswith(line, "#") && return Float64[]
        return [parse(Float64, strip(x)) for x in split(line, ",") if !isempty(strip(x))]
    end
    Float64[]
end

function category_means(model_rows::Dict{String, Vector{Float64}})
    n = length(model_rows["english"])
    on = Vector{Float64}(undef, n)
    related = Vector{Float64}(undef, n)
    alien = copy(model_rows["voynich"])
    for i in 1:n
        on[i] = (model_rows["english"][i] + model_rows["french"][i] + model_rows["german"][i]) / 3
        related[i] = (model_rows["italian"][i] + model_rows["finnish"][i]) / 2
    end
    Dict("on" => on, "off-related" => related, "off-alien" => alien)
end

function strip_with_mean!(ax, xpos::Int, values::Vector{Float64}, color)
    jitter = range(-0.10, 0.10, length=length(values))
    scatter!(ax, fill(xpos, length(values)) .+ collect(jitter), values;
             color=(color, 0.55), markersize=11)
    mean_val = sum(values) / length(values)
    scatter!(ax, [xpos], [mean_val]; color=color, markersize=18)
    hlines!(ax, [mean_val]; xmin=xpos - 0.22, xmax=xpos + 0.22, color=color, linewidth=3)
end

function main()
    mkpath(FIG_DIR)
    model_rows = parse_model_rows(RESULT_PATH)
    ce_by_cat = category_means(model_rows)
    js_on = parse_js_block(RESULT_PATH, "# Pairwise inter-model JS (on-ecology)")
    js_related = parse_js_block(RESULT_PATH, "# Pairwise inter-model JS (off-ecology related)")
    js_alien = parse_js_block(RESULT_PATH, "# Pairwise inter-model JS (off-ecology alien)")

    fig = Figure(size=(980, 420))

    ax1 = Axis(fig[1, 1];
               xlabel="Probe category",
               ylabel="Cross-entropy")
    strip_with_mean!(ax1, 1, ce_by_cat["on"], CB_BLUE)
    strip_with_mean!(ax1, 2, ce_by_cat["off-related"], CB_ORANGE)
    strip_with_mean!(ax1, 3, ce_by_cat["off-alien"], CB_MAGENTA)
    ax1.xticks = ([1, 2, 3], ["On", "Off-related", "Voynich"])

    ax2 = Axis(fig[1, 2];
               xlabel="Probe category",
               ylabel="Pairwise inter-model JS")
    strip_with_mean!(ax2, 1, js_on, CB_BLUE)
    strip_with_mean!(ax2, 2, js_related, CB_ORANGE)
    strip_with_mean!(ax2, 3, js_alien, CB_MAGENTA)
    ax2.xticks = ([1, 2, 3], ["On", "Off-related", "Voynich"])

    save(joinpath(FIG_DIR, "exp4_off_ecology_failure.pdf"), fig)
    save(joinpath(FIG_DIR, "exp4_off_ecology_failure.png"), fig)
    println("Saved figures to " * FIG_DIR)
end

main()
