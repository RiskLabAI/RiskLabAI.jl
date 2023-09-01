function initializeExpectedDollarImbalanceBars(;
    windowSizeForExpectedNTicksEstimation::Int=3,
    expectedImbalanceWindow::Int=10000,
    initialEstimateOfExpectedNTicksInBar::Float64=20000.0,
    expectedTicksNumberBounds::Union{Tuple,Nothing}=nothing,
    analyseThresholds::Bool=false,
)::ExpectedImbalanceBars{Dollar}
    """
    Initialize the expected dollar imbalance bars.

    :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
    :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
    :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
    :param expected_ticks_number_bounds lower and upper bound of possible number of expected ticks that used to force bars sampling convergence.
    :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    :return: initialized bar
    """

    bars = ExpectedImbalanceBars{Dollar}(;
        barType="dollar_imbalance",
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTicksEstimation,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        expectedTicksNumberBounds=expectedTicksNumberBounds,
        doesAnalyseThresholds=analyseThresholds
    )

    return bars
end

# the expected type does not exist anywhere but only one source. 
# double check and provide a new name or structure
function initializeExpectedVolumeImbalanceBars(;
    windowSizeForExpectedNTicksEstimation::Int=3,
    expectedImbalanceWindow::Int=10000,
    initialEstimateOfExpectedNTicksInBar::Float64=20000.0,
    expectedTicksNumberBounds::Union{Tuple,Nothing}=nothing,
    analyseThresholds::Bool=false,
)
    """
    Initialize the expected volume imbalance bars.

    :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
    :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
    :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
    :param expected_ticks_number_bounds lower and upper bound of possible number of expected ticks that used to force bars sampling convergence.
    :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    :return: initialized bar
    """

    bars = ExpectedImbalanceBars{Volume}(;
        barType="volume_imbalance",
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTicksEstimation,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        expectedTicksNumberBounds=expectedTicksNumberBounds,
        doesAnalyseThresholds=analyseThresholds
    )

    return bars

end

function initializeExpectedTickImbalanceBars(;
    windowSizeForExpectedNTicksEstimation::Int=3,
    expectedImbalanceWindow::Int=10000,
    initialEstimateOfExpectedNTicksInBar::Float64=20000.0,
    expectedTicksNumberBounds::Union{Tuple,Nothing}=nothing,
    analyseThresholds::Bool=false,
)
    """
    Initialize the expected tick imbalance bars.

    :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
    :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
    :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
    :param expected_ticks_number_bounds lower and upper bound of possible number of expected ticks that used to force bars sampling convergence.
    :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    :return: initialized bar
    """

    bars = ExpectedImbalanceBars{Tick}(;
        barType="tick_imbalance",
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTicksEstimation,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        expectedTicksNumberBounds=expectedTicksNumberBounds,
        doesAnalyseThresholds=analyseThresholds
    )

    return bars
end

function initializeFixedDollarImbalanceBars(;
    expectedImbalanceWindow::Int=10000,
    initialEstimateOfExpectedNTicksInBar::Float64=20000.0,
    analyseThresholds::Bool=false,
)
    """
    Initialize the fixed dollar imbalance bars.

    :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
    :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
    :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    :return: initialized bar
    """

    bars = FixedImbalanceBars{Dollar}(;
        barType="dollar_imbalance",
        windowSizeForExpectedNTicksEstimation=nothing,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        doesAnalyseThresholds=analyseThresholds
    )

    return bars
end

function initializeFixedVolumeImbalanceBars(;
    expectedImbalanceWindow::Int=10000,
    initialEstimateOfExpectedNTicksInBar::Float64=20000.0,
    analyseThresholds::Bool=false,
)
    """
    Initialize the fixed volume imbalance bars.

    :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
    :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
    :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    :return: initialized bar
    """


    bars = FixedImbalanceBars{Volume}(
        barType="volume_imbalance",
        windowSizeForExpectedNTicksEstimation=nothing,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        doesAnalyseThresholds=analyseThresholds
    )

    return bars
end


function initializeFixedTickImbalanceBars(;
    expectedImbalanceWindow::Int=10000,
    initialEstimateOfExpectedNTicksInBar::Float64=20000.0,
    analyseThresholds::Bool=false,
)
    """
    Initialize the fixed tick imbalance bars.

    :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
    :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
    :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    :return: initialized bar
    """

    bars = FixedImbalanceBars{Tick}(;
        barType="tick_imbalance",
        windowSizeForExpectedNTicksEstimation=nothing,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        doesAnalyseThresholds=analyseThresholds
    )

    return bars
end

function initializeExpectedDollarRunBars(;
    windowSizeForExpectedNTicksEstimation::Int=3,
    expectedImbalanceWindow::Int=10000,
    initialEstimateOfExpectedNTicksInBar::Float64=20000.0,
    expectedTicksNumberBounds::Union{Tuple,Nothing}=nothing,
    analyseThresholds::Bool=false,
)
    """
    Initialize the expected dollar run bars.

    :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
    :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
    :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
    :param expected_ticks_number_bounds lower and upper bound of possible number of expected ticks that used to force bars sampling convergence.
    :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    :return: initialized bar
    """

    bars = ExpectedRunBars{Dollar}(
        barType="dollar_run",
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTicksEstimation,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        expectedTicksNumberBounds=expectedTicksNumberBounds,
        doesAnalyseThresholds=analyseThresholds
    )

    return bars

end


function initializeExpectedVolumeRunBars(;
    windowSizeForExpectedNTicksEstimation::Int=3,
    expectedImbalanceWindow::Int=10000,
    initialEstimateOfExpectedNTicksInBar::Float64=20000.0,
    expectedTicksNumberBounds::Union{Tuple,Nothing}=nothing,
    analyseThresholds::Bool=false,
)
    """
    Initialize the expected volume run bars.

    :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
    :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
    :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
    :param expected_ticks_number_bounds lower and upper bound of possible number of expected ticks that used to force bars sampling convergence.
    :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    :return: initialized bar
    """

    bars = ExpectedRunBars{Volume}(;
        barType="volume_run",
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTicksEstimation,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        expectedTicksNumberBounds=expectedTicksNumberBounds,
        doesAnalyseThresholds=analyseThresholds
    )

    return bars

end

function initializeExpectedTickRunBars(;
    windowSizeForExpectedNTicksEstimation::Int=3,
    expectedImbalanceWindow::Int=10000,
    initialEstimateOfExpectedNTicksInBar::Float64=20000.0,
    expectedTicksNumberBounds::Union{Tuple,Nothing}=nothing,
    analyseThresholds::Bool=false,
)
    """
    Initialize the expected tick run bars.

    :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
    :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
    :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
    :param expected_ticks_number_bounds lower and upper bound of possible number of expected ticks that used to force bars sampling convergence.
    :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    :return: initialized bar
    """

    bars = ExpectedRunBars{Tick}(;
        barType="tick_run",
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTicksEstimation,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        expectedTicksNumberBounds=expectedTicksNumberBounds,
        doesAnalyseThresholds=analyseThresholds
    )

    return bars
end

function initializeFixedDollarRunBars(;
    windowSizeForExpectedNTticksEstimation::Int=3,
    expectedImbalanceWindow::Int=10000,
    initialEstimateOfExpectedNTicksInBar::Float64=20000.0,
    analyseThresholds::Bool=false,
)
    """
    Initialize the fixed dollar run bars.

    :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
    :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
    :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
    :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    :return: initialized bar
    """

    bars = FixedRunBars{Dollar}(;
        barType="dollar_run",
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTticksEstimation,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        doesAnalyseThresholds=analyseThresholds
    )

    return bars
end

function initializeFixedVolumeRunBars(;
    windowSizeForExpectedNTticksEstimation::Int=3,
    expectedImbalanceWindow::Int=10000,
    initialEstimateOfExpectedNTicksInBar::Float64=20000.0,
    analyseThresholds::Bool=false,
)
    """
    Initialize the fixed volume run bars.

    :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
    :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
    :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
    :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    :return: initialized bar
    """

    bars = FixedRunBars{Volume}(;
        barType="volume_run",
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTticksEstimation,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        doesAnalyseThresholds=analyseThresholds
    )

    return bars
end


function initializeFixedTickRunBars(;
    windowSizeForExpectedNTticksEstimation::Int=3,
    expectedImbalanceWindow::Int=10000,
    initialEstimateOfExpectedNTicksInBar::Float64=20000.0,
    analyseThresholds::Bool=false,
)
    """
    Initialize the fixed tick run bars.

    :param window_size_for_expected_n_ticks_estimation: window size used to estimate number of ticks expectation
    :param window_size_for_expected_imbalance_estimation: window size used to estimate imbalance expectation
    :param initial_estimate_of_expected_n_ticks_in_bar: initial estimate of number of ticks expectation window size
    :param analyse_thresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    :return: initialized bar
    """

    bars = FixedRunBars{Tick}(;
        barType="tick_run",
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTticksEstimation,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        doesAnalyseThresholds=analyseThresholds
    )

    return bars
end

function initializeDollarStandardBars(;
    threshold::Float64=70_000_000.0
)
    """
    Initialize the dollar bars.

    :param threshold: threshold that used to sampling process
    :return: initialized bar
    """

    bars = StandardBars{Dollar}(
        barType=CUMULATIVE_DOLLAR,
        threshold=threshold
    )

    return bars
end


function initializeVolumeStandardBars(;
    threshold::Float64=30_000.0
)
    """
    Initialize the volume bars.

    :param threshold: threshold that used to sampling process
    :return: initialized bar
    """

    bars = StandardBars{Volume}(
        barType=CUMULATIVE_VOLUME,
        threshold=threshold
    )

    return bars
end


function initializeTickStandardBars(;
    threshold::Float64=6_000.0
)
    """
    Initialize the tick bars.

    :param threshold: threshold that used to sampling process
    :return: initialized bar
    """

    bars = StandardBars{Tick}(
        barType=CUMULATIVE_TICKS,
        threshold=threshold
    )
    return bars
end

function initializeTimeBars(;
    resolutionType::String="D",
    resolutionUnits::Int=1
)
    """
    Initialize the time bars.

    :param resolution_type: (str) Type of bar resolution: ['D', 'H', 'MIN', 'S'].
    :param resolution_units: (int) Number of days, minutes, etc.
    :return: initialized bar
    """

    bars = TimeBars(;
        resolutionType=resolutionType,
        resolutionUnits=resolutionUnits,
    )

    return bars
end

methodNameToMethod = Dict(
    "ExpectedDollarImbalanceBars" => initializeExpectedDollarImbalanceBars,
    "ExpectedVolumeImbalanceBars" => initializeExpectedVolumeImbalanceBars,
    "ExpectedTickImbalanceBars" => initializeExpectedTickImbalanceBars, "FixedDollarImbalanceBars" => initializeFixedDollarImbalanceBars,
    "FixedVolumeImbalanceBars" => initializeFixedVolumeImbalanceBars,
    "FixedTickImbalanceBars" => initializeFixedTickImbalanceBars, "ExpectedDollarRunBars" => initializeExpectedDollarRunBars,
    "ExpectedVolumeRunBars" => initializeExpectedVolumeRunBars,
    "ExpectedTickRunBars" => initializeExpectedTickRunBars, "FixedDollarRunBars" => initializeFixedDollarRunBars,
    "FixedVolumeRunBars" => initializeFixedVolumeRunBars,
    "FixedTickRunBars" => initializeFixedTickRunBars, "DollarStandardBars" => initializeDollarStandardBars,
    "VolumeStandardBars" => initializeVolumeStandardBars,
    "TickStandardBars" => initializeTickStandardBars, "TimeBars" => initializeTimeBars
)