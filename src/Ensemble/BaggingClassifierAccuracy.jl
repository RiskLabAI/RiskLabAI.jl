using Combinatorics, BigInts

"""
    baggingClassifierAccuracy(
        n::Int,
        p::Float64,
        k::Int
    )::Float64

Calculates the accuracy of a bagging classifier.

This function calculates the accuracy of a bagging classifier based on the
improved accuracy formula from De Prado (2018), Advances in financial
machine learning, page 96, Improved Accuracy section.

# Parameters
- `n`: Int, Number of independent classifiers.
- `p`: Float64, The accuracy of a classifier is the probability p of labeling a prediction as 1.
- `k`: Int, Number of classes.

# Returns
- Float64: Accuracy of the bagging classifier.

# Formula
The formula for calculating the accuracy of a bagging classifier is as follows:

.. math::

    1 - \sum_{i=0}^{\lfloor n / k + 1 \rfloor} \binom{n}{i} p^i (1-p)^{n-i}

where:
- :math:`n` is the number of independent classifiers
- :math:`p` is the accuracy of a classifier
- :math:`k` is the number of classes

"""
function baggingClassifierAccuracy(
        n::Int,
        p::Float64,
        k::Int
    )::Float64

    probabilitySum = 0.0
    for i in 0:Int(trunc(n / k + 1))
        probabilitySum += binomial(BigInt(n), BigInt(i)) * p^i * (1 - p)^(n - i)
    end

    return 1 - probabilitySum
end
