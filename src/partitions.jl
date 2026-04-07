# ── Partition enumeration ─────────────────────────────────────────────────────
#
# Enumerate all set partitions of {1,…,n} via restricted growth strings.
# A restricted growth string (RGS) is a vector a[1:n] where:
#   a[1] = 0
#   a[i] <= max(a[1], ..., a[i-1]) + 1
# Each RGS encodes a unique partition (up to label ordering).

"""
    all_partitions(n)

Return a vector of all partitions of {1,…,n}, where each partition is
represented as a label vector `labels[1:n]` with labels in 1:k.
The number of partitions is the Bell number B(n).
"""
function all_partitions(n::Int)
    n <= 0 && return [Int[]]
    result = Vector{Vector{Int}}()
    rgs = zeros(Int, n)
    maxval = zeros(Int, n)
    # Generate all restricted growth strings
    _enumerate_rgs!(result, rgs, maxval, n, 1)
    result
end

function _enumerate_rgs!(result, rgs, maxval, n, pos)
    if pos > n
        push!(result, rgs .+ 1)  # shift from 0-based to 1-based labels
        return
    end
    upper = pos == 1 ? 0 : maxval[pos - 1] + 1
    for val in 0:upper
        rgs[pos] = val
        maxval[pos] = pos == 1 ? val : max(maxval[pos - 1], val)
        _enumerate_rgs!(result, rgs, maxval, n, pos + 1)
    end
end

"""
    partition_cells(labels)

Convert a label vector to a list of cells, where each cell is a sorted
vector of indices sharing that label.

    partition_cells([1,1,2,1,3]) → [[1,2,4], [3], [5]]
"""
function partition_cells(labels::AbstractVector{Int})
    k = maximum(labels)
    cells = [Int[] for _ in 1:k]
    for (i, l) in enumerate(labels)
        push!(cells[l], i)
    end
    filter!(!isempty, cells)
    cells
end

"""
    n_classes(labels)

Number of distinct labels (classes) in a partition.
"""
n_classes(labels::AbstractVector{Int}) = length(unique(labels))

"""
    is_refinement(finer, coarser)

Check whether partition `finer` refines `coarser`: every cell of `finer`
is contained in some cell of `coarser`.
Equivalently: if finer[i] == finer[j] then coarser[i] == coarser[j].
"""
function is_refinement(finer::AbstractVector{Int}, coarser::AbstractVector{Int})
    n = length(finer)
    for i in 1:n, j in (i + 1):n
        if finer[i] == finer[j] && coarser[i] != coarser[j]
            return false
        end
    end
    true
end

"""
    same_partition(a, b)

Check whether two label vectors encode the same partition (possibly with
different label names).
"""
function same_partition(a::AbstractVector{Int}, b::AbstractVector{Int})
    is_refinement(a, b) && is_refinement(b, a)
end

"""
    all_binary_splits(cell)

Given a cell (vector of indices), return all non-trivial splits (A, B)
with A ∪ B = cell, A ∩ B = ∅, |A| ≥ 1, |B| ≥ 1.
We enumerate only splits where min(A) < min(B) to avoid duplicates.
"""
function all_binary_splits(cell::AbstractVector{Int})
    n = length(cell)
    n < 2 && return Tuple{Vector{Int}, Vector{Int}}[]
    splits = Tuple{Vector{Int}, Vector{Int}}[]
    # Enumerate subsets of size 1..n-1 via bitmask
    for mask in 1:(2^n - 2)
        A = Int[]
        B = Int[]
        for i in 1:n
            if (mask >> (i - 1)) & 1 == 1
                push!(A, cell[i])
            else
                push!(B, cell[i])
            end
        end
        # Canonical form: the subset containing cell[1] is always A
        if cell[1] in A
            push!(splits, (A, B))
        end
    end
    splits
end
