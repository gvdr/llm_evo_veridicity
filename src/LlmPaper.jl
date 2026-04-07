module LlmPaper

using Random: AbstractRNG, MersenneTwister
using LinearAlgebra: dot

# Source files in dependency order
include("information.jl")
include("partitions.jl")
include("finite_ecology.jl")
include("empirical_ecology.jl")

# ── Information primitives ───────────────────────────────────────────────────
export entropy, kl_divergence, hellinger2, js_divergence,
       weighted_js, binary_entropy, cross_entropy

# ── Partitions ───────────────────────────────────────────────────────────────
export all_partitions, partition_cells, n_classes,
       is_refinement, same_partition, all_binary_splits

# ── Finite ecology ───────────────────────────────────────────────────────────
export FiniteEcology, n_worlds, n_contexts, n_vocab,
       random_ecology, ecology_with_merged_pair, ecology_with_partial_separation,
       conditional_entropy_VCW, task_distance, task_distance_matrix,
       separated, ecology_partition,
       cell_average, bayes_opt_loss, excess_loss, js_excess,
       partition_complexity, regularized_objective,
       merges_separated_pair, is_zero_excess,
       split_gain, split_threshold_rhs,
       partition_labels_from_distance, empirical_prefix_ecology

end # module
