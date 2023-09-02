using Combinatorics

"""
    backtestPathsNumber(
        nTrainSplits::Int,
        nTestSplits::Int
    )::Int

Calculate the number of combinatorial backtest paths based on the number of train and test splits in the sample.

.. math::
    \\text{Number of backtest paths} = \\frac{\\binom{nTrainSplits}{nTrainSplits - nTestSplits} \\times nTestSplits}{nTrainSplits}

:param nTrainSplits: Number of train splits in the sample.
:type nTrainSplits: Int
:param nTestSplits: Number of test splits in the sample.
:type nTestSplits: Int

:return: The number of combinatorial backtest paths.
:rtype: Int
"""
function backtestPathsNumber(
        nTrainSplits::Int,
        nTestSplits::Int
    )::Int

    return Int64(binomial(nTrainSplits, nTrainSplits - nTestSplits) * nTestSplits ÷ nTrainSplits)
end
