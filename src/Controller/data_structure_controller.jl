using DataFrames
using ResumableFunctions
using CSV
using Parameters
using Dates

include("../model/types.jl")

include("../utils/ewma.jl")
include("../utils/constants.jl")
include("../utils/field_inheritance.jl")

include("bars_initializer.jl")

include("../model/abstract_bars.jl")

include("../model/abstract_information_driven_bars.jl")
include("../model/abstract_imbalance_bars.jl")
include("../model/imbalance_bars.jl")

include("../model/abstract_run_bars.jl")
include("../model/run_bars.jl")

include("../model/standard_bars.jl")

include("../model/time_bars.jl")

include("bars_initializer.jl")

function handleInput(
    methodName::String,
    methodArguments::Dict,
    inputData::Union{DataFrame,String};
    outputPath::Union{String,Nothing}=nothing,
    batchSize::Int=1_000_000
)::DataFrame
    """
    This function handle input command and can be used as api with back-end

    :param methodName: desired type of bars to construct
    :param methodArguments: hyper-parameters used to bar's construction process
    :param inputData: input that can be either file path string or dataframe
    :param outputPath: output path of append results after bar's construction using batch data
    :param batchSize: batch size approximation
    :return: the desired sampled bars 
    """

    method = methodNameToMethod[methodName]
    initializedBars = method(; methodArguments...)
    bars = runOnBatches(
        initializedBars,
        inputData,
        outputPath,
        batchSize,
    )

    return bars
end

function runOnBatches(
    abstractBars::AbstractBarsType,
    inputData::String,
    outputPath::Union{String,Nothing}=nothing,
    batchSize::Int=1_000_000,)::DataFrame
    """
    Run on batches of data and construct the desired type of bars
    :param abstractBars: bars
    :param inputData: batch of data
    :param outputPath: output path of resulted dataframe
    :param batchSize: batch size
    :return: desired type of bars
    """

    dataframe = CSV.File(inputData) |> DataFrame
    dataframe.date = DateTime.(dataframe.date,)

    return runOnBatches(abstractBars, dataframe, outputPath, batchSize)
end

function runOnBatches(
    abstractBars::AbstractBarsType,
    inputData::Union{DataFrame,String},
    outputPath::Union{String,Nothing}=nothing,
    batchSize::Int=1_000_000;
    # returnDataStructureObject::Bool=true
)::DataFrame
    """
    Run on batches of data and construct the desired type of bars
    :param abstractBars: bars
    :param inputData: batch of data
    :param outputPath: output path of resulted dataframe
    :param batchSize: batch size
    :return: desired type of bars
    """

    saveResultsInCSV = !isnothing(outputPath)

    #todo: this part is very close
    # I changed toCSV to saveResultsInCSV
    if saveResultsInCSV
        header = true
        open(outputPath, "w") do io
        end
    end

    columnNames = [
        DATE_TIME,
        TICK_NUMBER,
        OPEN_PRICE,
        HIGH_PRICE,
        LOW_PRICE,
        CLOSE_PRICE,
        CUMULATIVE_VOLUME,
        CUMULATIVE_BUY_VOLUME,
        CUMULATIVE_SELL_VOLUME,
        CUMULATIVE_TICKS,
        CUMULATIVE_DOLLAR,
        THRESHOLD,
    ]

    batchesAggregationConstructedBars = []
    batchesGenerator = readInBatches(inputData, batchSize=batchSize,)
    for (k, batch) ∈ batchesGenerator |> enumerate
        println("batch $k with size $(size(batch)[1])")

        batchConstructedBars::Vector = runOnBatch(abstractBars, batch)
        append!(batchesAggregationConstructedBars, batchConstructedBars)

        if saveResultsInCSV
            CSV.write(outputPath, DataFrame(batchConstructedBars, columnNames), writeheader=header, append=true)
            header = false
        end
    end

    aggregationDataFrame = rowsVectorToDataFrame(batchesAggregationConstructedBars, columnNames)

    return aggregationDataFrame
end;

function rowsVectorToDataFrame(rowsVector::Vector, columnNames::Vector)::DataFrame
    """
    :param rowsVector sampled bars vectors of vectors
    :param columnNames column names of final dataframe of constructed bars
    :return dataframe of constructed bars
    """

    nRows = length(rowsVector)
    if nRows == 0
        nColumns = length(columnNames)
        return DataFrame([[[] for row ∈ rowsVector] for columnIndex ∈ 1:nColumns], columnNames)
    else
        nColumns = length(rowsVector[1])
        return DataFrame([[row[columnIndex] for row ∈ rowsVector] for columnIndex ∈ 1:nColumns], columnNames)
    end
end

function runOnBatch(
    abstractBars::AbstractBarsType,
    data::DataFrame
)::Vector
    """
    read a batch of data and construct the desired type of bars
    :param bars: bars
    :param data: batch of data
    :return: desired type of bars
    """

    # data[:, :volume] = float.(data[!, :volume])
    data.volume = convert.(Float64, data.volume)

    barsList = constructBarsFromData(abstractBars, data=Array(data))
    abstractBars.tickCounter += size(data)[1]
    return barsList
end

function countCSVFileRows(inputPath::String)
    n = 0
    for row in CSV.Rows(inputPath; reusebuffer=true)
    end
    return n
end

#todo: read about resumable functions (arian)
@resumable function readInBatches(
    inputPath::String;
    batchSize::Int=1_000_000,
)
    """
    Read input data (csv file using input path) and split it to the batches
    :param inputPath: input data path
    :param batchSize: batch size
    :return yield batches
    """

    dateFormat = "yyyy/mm/dd H:M:S.s"

    nRows = countCSVFileRows(inputPath)
    nBatches = max(1, round(Int, nRows / batchSize))

    if nBatches == 1
        data = CSV.File("data/micro_raw_tick_data.csv") |> DataFrame
        data.date = DateTime.(data.date, dateFormat)
        @yield data
    else
        # this should be changed to chunks most likely
        for batch ∈ CSV.Chunks(inputPath; ntasks=nBatches)
            batch = batch |> DataFrame
            batch.date = DateTime.(batch.date, dateFormat)
            checkConditionsOnRow(batch)
            @yield batch
        end
    end
end


@resumable function readInBatches(
    dataframe::DataFrame;
    batchSize::Int=1_000_000,
)
    """
    Read input data (dataframe) and split it to the batches
    :param dataframe: input dataframe
    :param batchSize: batch size
    :return yield batches
    """

    dateFormat = "yyyy/mm/dd H:M:S.s"

    nRows = size(dataframe)[1]
    nBatches = max(1, round(Int, nRows / batchSize))

    checkConditionsOnRow(dataframe)

    if nBatches == 1
        @yield dataframe
    else
        for k ∈ 1:nBatches
            batchSize = floor(Int, nRows / nBatches)
            l = (k - 1) * batchSize + 1
            r = min(nRows, k * batchSize)
            batch = dataframe[l:r, :]
            @yield batch
        end
    end
end

function checkConditionsOnRow(
    testData::DataFrame
)
    """
    check input dataframe conditions and throw exception if any miss
    :param testData: sample row from the dataset (e.g first row)
    :return 
    """
    @assert size(testData)[2] == 3 "Data should have three columns"
    @assert typeof(testData[1, 1]) == DateTime "First column should be date time"
    @assert typeof(testData[1, 2]) == Float64 "Second column should be float"
    @assert typeof(testData[1, 3]) != String "Third column should be string"
end