"""----------------------------------------------------------------------
function: Calculates hedging weights using cov, risk distribution(risk_dist) and σ
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: page 36
----------------------------------------------------------------------"""
function PCAWeights(cov, # covariance matrix
                    riskdisturbution = nothing,  # risk distribution
                    σ = 1.0) # risk target
    Λ = eigvals(cov)
    V = eigvecs(cov)
    indices = reverse(sortperm(Λ)) # arguments for sorting eVal descending
    Λ = Λ[indices] # sort eigen values
    V = V[:, indices] # sort eigen vectors
    # if riskdisturbution is nothing, it will assume all risk must be allocated to the principal component with
    # smallest eigenvalue, and the weights will be the last eigenvector re-scaled to match σ
    if riskdisturbution == nothing
        riskdisturbution = zeros(size(cov)[1])
        riskdisturbution[end] = 1.0
    end
    loads = σ*(riskdisturbution./Λ).^0.5 # represent the allocation in the new (orthogonal) basis
    weights = V*loads # calculate weights
    return weights
end

"""----------------------------------------------------------------------
function: Implementation of the symmetric CUSUM filter
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: page 39
----------------------------------------------------------------------"""
function events(input, # dataframe of prices and dates
                threshold) # threshold
    timeevents, shiftpositive, shiftnegative = [], 0, 0
    # dataframe with price differences
    deltaprice = DataFrames.DataFrame(hcat(input[2:end, 1], diff(input[:, 2])), :auto) 
    for i ∈ deltaprice[:, 1]
        # compute shiftnegative/shiftpositive with min/max of 0 and ΔPRICE in each day
        shiftpositive = max(0, shiftpositive+deltaprice[deltaprice[:, 1] .== i, 2][1]) # compare price diff with zero
        shiftnegative = min(0, shiftnegative+deltaprice[deltaprice[:, 1] .== i, 2][1]) # compare price diff with zero
        if shiftnegative < -threshold
            shiftnegative = 0 # reset shiftnegative to 0
            append!(timeevents, [i]) # append this time into timeevents
        elseif shiftpositive > threshold
            shiftpositive = 0 # reset shiftpositive to 0
            append!(timeevents, [i])  # append this time into timeevents
        end
    end
    return timeevents
end
