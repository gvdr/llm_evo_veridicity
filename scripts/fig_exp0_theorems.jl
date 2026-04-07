"""
Figures for Experiment 0 / exact finite-ecology theorem support.

Generates three theorem-facing figures:
  1. Exact decomposition: excess loss vs JS excess (exact decomposition theorem)
  2. Local split threshold crossing under J_{D,beta} (split-versus-merge theorem)
  3. Complexity-distortion frontier over all partitions (minimum-complexity theorem)

Run:
  julia --project=. scripts/fig_exp0_theorems.jl
"""

using Random
using CairoMakie

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using LlmPaper

const FIG_DIR = joinpath(@__DIR__, "..", "figures")
const CB_BLUE = colorant"#0072B2"
const CB_ORANGE = colorant"#E69F00"
const CB_GREEN = colorant"#009E73"
const CB_MAGENTA = colorant"#CC79A7"
const CB_VERMILION = colorant"#D55E00"
const CB_SKY = colorant"#56B4E9"

function merge_states(labels::Vector{Int}, i::Int, j::Int)
    out = copy(labels)
    li = out[i]
    lj = out[j]
    for idx in eachindex(out)
        if out[idx] == lj
            out[idx] = li
        end
    end
    cells = partition_cells(out)
    renumbered = similar(out)
    for (label, cell) in enumerate(cells), idx in cell
        renumbered[idx] = label
    end
    renumbered
end

function find_full_separation_ecology(seed::Int; min_beta::Float64=1e-3,
                                      max_beta::Float64=0.5)
    rng = MersenneTwister(seed)
    for attempt in 1:5000
        fe = random_ecology(rng, 4, 3, 3; floor=0.05)
        labels = ecology_partition(fe)
        n_classes(labels) == 4 || continue
        D = task_distance_matrix(fe)
        best_pair = nothing
        best_d = Inf
        for i in 1:4, j in (i + 1):4
            if D[i, j] > 0 && D[i, j] < best_d
                best_d = D[i, j]
                best_pair = (i, j)
            end
        end
        best_pair === nothing && continue
        i, j = best_pair
        pi_i = fe.pi[i]
        pi_j = fe.pi[j]
        lam = pi_i / (pi_i + pi_j)
        gain = split_gain(fe, [i], [j])
        h = binary_entropy(lam)
        h <= 0 && continue
        beta_star = gain / h
        (min_beta <= beta_star <= max_beta) || continue
        return fe, best_pair, beta_star
    end
    error("failed to find a full-separation ecology with a usable split threshold")
end

function theorem1_points(n_ecologies::Int=32)
    xs = Float64[]
    ys = Float64[]
    ks = Int[]
    for seed in 1:n_ecologies
        fe = random_ecology(MersenneTwister(20_000 + seed), 4, 3, 3; floor=0.05)
        for labels in all_partitions(4)
            push!(xs, excess_loss(fe, labels))
            push!(ys, js_excess(fe, labels))
            push!(ks, n_classes(labels))
        end
    end
    xs, ys, ks
end

function figure_decomposition()
    xs, ys, ks = theorem1_points()
    lim_hi = 1.02 * max(maximum(xs), maximum(ys))
    fig = Figure(size=(760, 520))
    ax = Axis(fig[1, 1];
              xlabel="Excess loss  L*_D(p) - H(V | C, W)",
              ylabel="JS excess")
    scatter!(ax, xs, ys; color=ks, colormap=:viridis, markersize=10)
    lines!(ax, [0.0, lim_hi], [0.0, lim_hi]; color=:black, linewidth=2, linestyle=:dash)
    xlims!(ax, 0.0, lim_hi)
    ylims!(ax, 0.0, lim_hi)
    Colorbar(fig[1, 2], limits=(minimum(ks), maximum(ks)), colormap=:viridis,
             label="Number of partition cells")
    fig
end

function figure_split_threshold()
    fe, pair, beta_star = find_full_separation_ecology(2026)
    i, j = pair
    full = ecology_partition(fe)
    merged = merge_states(full, i, j)

    betas = collect(range(0.0, stop=2.0 * beta_star, length=200))
    lhs = [regularized_objective(fe, merged, beta) -
           regularized_objective(fe, full, beta) for beta in betas]
    rhs = [split_threshold_rhs(fe, [i], [j], beta) for beta in betas]

    fig = Figure(size=(760, 520))
    ax = Axis(fig[1, 1];
              xlabel="Complexity weight  β",
              ylabel="J(merge) - J(split)")
    lines!(ax, betas, rhs; color=CB_ORANGE, linewidth=3,
           label="Split-threshold prediction")
    sample_idx = 1:8:length(betas)
    scatter!(ax, betas[sample_idx], lhs[sample_idx]; color=CB_BLUE, markersize=10,
             label="Exact objective difference")
    hlines!(ax, [0.0]; color=:black, linewidth=1)
    vlines!(ax, [beta_star]; color=:gray40, linewidth=2, linestyle=:dot)
    axislegend(ax; position=:rb)
    fig
end

function figure_frontier()
    fe, pair, beta_star = find_full_separation_ecology(2026)
    partitions = all_partitions(4)
    complexities = [partition_complexity(fe, labels) for labels in partitions]
    excesses = [excess_loss(fe, labels) for labels in partitions]
    classes = [n_classes(labels) for labels in partitions]

    frontier = Int[]
    for i in eachindex(partitions)
        dominated = any(j != i &&
                        complexities[j] <= complexities[i] + 1e-12 &&
                        excesses[j] <= excesses[i] + 1e-12 &&
                        (complexities[j] < complexities[i] - 1e-12 ||
                         excesses[j] < excesses[i] - 1e-12)
                        for j in eachindex(partitions))
        dominated || push!(frontier, i)
    end
    frontier = sort(frontier, by=i -> complexities[i])

    eco_labels = ecology_partition(fe)
    eco_idx = findfirst(labels -> same_partition(labels, eco_labels), partitions)
    merged_idx = findfirst(labels -> same_partition(labels, merge_states(eco_labels, pair[1], pair[2])), partitions)
    full_merge_idx = findfirst(labels -> n_classes(labels) == 1, partitions)

    fig = Figure(size=(760, 520))
    ax = Axis(fig[1, 1];
              xlabel="Encoding complexity  H(p(W))",
              ylabel="Excess loss")
    scatter!(ax, complexities, excesses; color=classes, colormap=:plasma, markersize=16)
    lines!(ax, complexities[frontier], excesses[frontier]; color=:black, linewidth=2)
    scatter!(ax, [complexities[eco_idx]], [excesses[eco_idx]];
             color=CB_GREEN, markersize=22, label="Ecology partition")
    scatter!(ax, [complexities[merged_idx]], [excesses[merged_idx]];
             color=CB_VERMILION, markersize=22, label="One informative merge")
    scatter!(ax, [complexities[full_merge_idx]], [excesses[full_merge_idx]];
             color=:gray30, markersize=22, label="Full merge")
    axislegend(ax; position=:rt)
    Colorbar(fig[1, 2], limits=(minimum(classes), maximum(classes)), colormap=:plasma,
             label="Number of partition cells")
    fig
end

function main()
    mkpath(FIG_DIR)

    fig1 = figure_decomposition()
    save(joinpath(FIG_DIR, "exp0_theorem31_decomposition.pdf"), fig1)
    save(joinpath(FIG_DIR, "exp0_theorem31_decomposition.png"), fig1)

    fig2 = figure_split_threshold()
    save(joinpath(FIG_DIR, "exp0_theorem73_split_threshold.pdf"), fig2)
    save(joinpath(FIG_DIR, "exp0_theorem73_split_threshold.png"), fig2)

    fig3 = figure_frontier()
    save(joinpath(FIG_DIR, "exp0_theorem71_frontier.pdf"), fig3)
    save(joinpath(FIG_DIR, "exp0_theorem71_frontier.png"), fig3)

    println("Saved figures to " * FIG_DIR)
end

main()
