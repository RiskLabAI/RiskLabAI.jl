"""
function: Calculates accuracy of bagging classifier
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: page 96 Improved Accuracy section
"""
function baggingClassifierAccuracy(
    N::Int, # number of independent classifers
    p::Float64, # The accuracy of a classifier is the probability p of labeling a prediction as 1
    k::Int, # number of classes
)::Float64
    probabilitySum = 0
    for i ∈ 0:Int(trunc(N / k + 1))    
        probabilitySum += binomial(N |> BigInt, i |> BigInt) * p^i * (1 - p)^(N - i)
    end

    1 - probabilitySum    
end