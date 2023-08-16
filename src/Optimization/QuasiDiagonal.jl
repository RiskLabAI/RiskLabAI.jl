using LinearAlgebra
using DataFrames

"""
    quasiDiagonal(linkageMatrix::AbstractMatrix)::Vector{Int}

Create a sorted list of original items to reshape the correlation matrix.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 16.2, Page 229

# Arguments
- `linkageMatrix::AbstractMatrix`: The linkage matrix from hierarchical clustering.

# Returns
- `Vector{Int}`: A sorted list of original items.

# Methodology
1. Sort clustered items by distance.
2. Initialize a sorted array with the last row of the linkage matrix.
3. Loop through the clusters, replacing them with the original items.
4. Sort and re-index the list.
"""
function quasiDiagonal(linkageMatrix::AbstractMatrix)::Vector{Int}
    linkageMatrix = Int.(floor.(linkageMatrix)) # Convert to integers
    sortedItems = DataFrame(index = [1, 2], value = [linkageMatrix[end, 1], linkageMatrix[end, 2]]) # Initialize sorted array
    nItems = linkageMatrix[end, 4] # Number of original items

    while maximum(sortedItems.value) >= nItems
        sortedItems.index .= range(0, stop = nrow(sortedItems) * 2 - 1, step = 2) # Make space
        dataframe = sortedItems[sortedItems.value .>= nItems, :] # Find clusters
        index = dataframe.index
        value = dataframe.value .- nItems
        sortedItems[in.(sortedItems.index, (index,)), :value] = linkageMatrix[value .+ 1, 1] # Item 1
        
        dataframe = DataFrame(index = index .+ 1, value = linkageMatrix[value .+ 1, 2]) # Create DataFrame for item 2
        sortedItems = vcat(sortedItems, dataframe) # Append item 2
        sort!(sortedItems, by = x -> x[1]) # Re-sort
        sortedItems.index .= range(0, length = nrow(sortedItems)) # Re-index
    end
    
    return sortedItems.value |> Vector
end
