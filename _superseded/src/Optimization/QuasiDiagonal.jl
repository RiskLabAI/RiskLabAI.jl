using DataFrames
using LinearAlgebra

"""
    quasiDiagonal(linkageMatrix::AbstractMatrix)

Create a sorted list of original items to reshape the correlation matrix.

# Arguments
- `linkageMatrix::AbstractMatrix`: The linkage matrix from hierarchical clustering.

# Returns
- `Vector{Int}`: A sorted list of original items.

# Methodology
1. Sort clustered items by distance.
2. Initialize a sorted array with the last row of the linkage matrix.
3. Loop through the clusters, replacing them with the original items.
4. Sort and re-index the list.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 16.2, Page 229

"""
function quasiDiagonal(linkageMatrix::AbstractMatrix)::Vector{Int}
    linkageMatrix = Int.(floor.(linkageMatrix)) # Convert to integers
    sortedItems = DataFrame(index = [1, 2], value = [linkageMatrix[end, 1], linkageMatrix[end, 2]]) # Initialize sorted array
    nItems = linkageMatrix[end, 3] # Number of original items

    while maximum(sortedItems.value) >= nItems
        clusters = filter(row -> row.value >= nItems, sortedItems) # Find clusters
        for cluster in eachrow(clusters)
            # Replace clusters with their members
            members = linkageMatrix[cluster.value - nItems + 1, 1:2]
            idx = cluster.index
            sortedItems[idx, :value] = members[1]
            push!(sortedItems, (index = idx + 1, value = members[2]))
        end
        sort!(sortedItems, by = x -> x.index) # Re-sort
    end

    return Vector{Int}(sortedItems.value)
end
