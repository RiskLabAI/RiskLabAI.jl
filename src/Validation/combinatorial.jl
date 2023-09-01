using Pandas
using DataFrames
using PyCall
using Dates
using Combinatorics
KFold = pyimport("sklearn.model_selection").KFold

function ml_get_train_times(samples_info_sets::Pandas.Series, test_times::Pandas.Series)::Pandas.Series

    train = deepcopy(samples_info_sets)
    
    for (start_ix, end_ix) in zip(index(test_times), values(test_times))
        df0 = index(get(train, (start_ix <= index(train)) .* (index(train) <= end_ix), -1)) # Train starts within test
        df1 = index(get(train, (start_ix <= train) .* (train <= end_ix), -1)) # Train ends within test
        df2 = index(get(train, (index(train) <= start_ix) .* (end_ix <= train), -1)) # Train envelops test
        train = drop(train, union(df2, union(df0, df1)))
    end

    return train
end

function _get_number_of_backtest_paths(n_train_splits::Int64, n_test_splits::Int64)::Int64
    return Int64(binomial(n_train_splits, n_train_splits - n_test_splits) * n_test_splits / n_train_splits)
end

mutable struct CombinatorialPurgedKFold

    n_splits::Int64
    n_test_splits::Int64
    samples_info_sets::Pandas.Series
    pct_embargo::Float64
    num_backtest_paths::Int64
    backtest_paths::Array
    function CombinatorialPurgedKFold(n_splits::Int64 = 3,
                                      n_test_splits::Int64 = 2,
                                      samples_info_sets::Pandas.Series = nothing,
                                      pct_embargo::Float64 = 0.)

        typeof(samples_info_sets) != Pandas.Series ? 
        error("The samples_info_sets param must be a Pandas.Series.") : 
        new(n_splits, n_test_splits, samples_info_sets, pct_embargo, _get_number_of_backtest_paths(n_splits, n_test_splits), [])
    end
end    
    
function _generate_combinatorial_test_ranges(CombinatorialPurgedKFold::CombinatorialPurgedKFold, splits_indices::Dict)::Array

    # Possible test splits for each fold
    combinatorial_splits = Array(collect(combinations(Array(keys(splits_indices)), CombinatorialPurgedKFold.n_test_splits)))
    combinatorial_test_ranges = Array([])  # Array of test indices formed from combinatorial splits
    for combination in combinatorial_splits
        temp_test_indices = Array([])  # Array of test indices for current split combination
        for int_index in combination
            append!(temp_test_indices, [splits_indices[int_index]])
        end
        append!(combinatorial_test_ranges, [temp_test_indices])
    end
    
    return combinatorial_test_ranges
end

function _fill_backtest_paths(CombinatorialPurgedKFold::CombinatorialPurgedKFold, train_indices::Array, test_splits::Array)

    # Fill backtest paths using train/test splits from CPCV
    for split in test_splits
        found = false  # Flag indicating that split was found and filled in one of backtest paths
        for path in CombinatorialPurgedKFold.backtest_paths
            for path_el in path
                if path_el["train"] == nothing && split == path_el["test"] && found == false
                    path_el["train"] = Array(train_indices)
                    path_el["test"] = Array(collect(split[1]:split[end]))
                    found = true
                end
            end
        end
    end
    
    return CombinatorialPurgedKFold
end

function split(CombinatorialPurgedKFold::CombinatorialPurgedKFold,
    X:: DataFrames.DataFrame,
    y:: Pandas.Series = nothing,
    groups= nothing) :: Tuple


    CombinatorialPurgedKFold.backtest_paths = Array([])  # Reset backtest paths

    size(X, 1) != size(CombinatorialPurgedKFold.samples_info_sets, 1) ? error("X and the 'samples_info_sets' series param must be the same length.") : Nothing()

    test_ranges::Vector = [(ix[1], ix[end]) for ix in Iterators.partition(collect(size(X, 1)), ceil(Int64, size(X, 1)/CombinatorialPurgedKFold.n_splits))]
    splits_indices = Dict() 
    for (index, (start_ix, end_ix)) in enumerate(test_ranges)
        splits_indices[index] = [start_ix, end_ix]
    end

    combinatorial_test_ranges = _generate_combinatorial_test_ranges(CombinatorialPurgedKFold, splits_indices)
    # Prepare backtest paths
    for _ in collect(1:CombinatorialPurgedKFold.num_backtest_paths)
        path = Array([])
        for split_idx in values(splits_indices)
            append!(path, [Dict("train"=> nothing, "test"=> split_idx)])
        end
        append!(CombinatorialPurgedKFold.backtest_paths , [path])
    end

    embargo::Int64 = floor(Int64, size(X, 1) * CombinatorialPurgedKFold.pct_embargo)

    final_test::Vector = []
    final_train::Vector = []

    for test_splits in combinatorial_test_ranges

        # Embargo
        test_times = Pandas.Series(index=[CombinatorialPurgedKFold.samples_info_sets[ix[1]] for ix in test_splits], data=[  
        begin 
            if ix[2] - 1 + embargo >= size(X,1)
            max(CombinatorialPurgedKFold.samples_info_sets[ix[1]:ix[2]]) 
            else
            max(CombinatorialPurgedKFold.samples_info_sets[ix[1]:ix[2] + embargo])
            end
        end     
        for ix in test_splits])

        test_indices = Array([])
        for (start_ix, end_ix) in test_splits
            append!(test_indices, [collect(start_ix:end_ix)])
        end
        # Purge
        train_times = ml_get_train_times(CombinatorialPurgedKFold.samples_info_sets, test_times)
  
        # Get indices
        train_indices = Array([])
        for train_ix in index(train_times)
            append!(train_indices, findall(x -> x == train_ix ,index(CombinatorialPurgedKFold.samples_info_sets)))
        end
  
        CombinatorialPurgedKFold = _fill_backtest_paths(CombinatorialPurgedKFold, train_indices, test_splits)
  
        append!(final_test, [test_indices])
        append!(final_train, [train_indices])
    end

    return (Tuple(zip(final_train, final_test)), CombinatorialPurgedKFold)
end

mutable struct StackedCombinatorialPurgedKFold

    n_splits::Int64
    n_test_splits::Int64
    samples_info_sets::Dict
    pct_embargo::Float64
    num_backtest_paths::Int64
    backtest_paths::Dict  # Dict of Arrays of backtest paths
   
   
    function StackedCombinatorialPurgedKFold(n_splits::Int64 = 3,
                      n_test_splits::Int64 = 2,
                      samples_info_sets_dict::Dict = nothing,
                      pct_embargo::Float64 = 0.)

        new(n_splits, n_test_splits, samples_info_sets, pct_embargo, _get_number_of_backtest_paths(n_splits, n_test_splits), Dict())
    end
end

function _fill_backtest_paths(StackedCombinatorialPurgedKFold::StackedCombinatorialPurgedKFold, asset, train_indices::Array , test_splits::Array)

    # Fill backtest paths using train/test splits from CPCV
    for split in test_splits
        found = false  # Flag indicating that split was found and filled in one of backtest paths
        for path in StackedCombinatorialPurgedKFold.backtest_paths[asset]
            for path_el in path
                if path_el["train"] == nothing && split == path_el["test"] && found == false
                    path_el["train"] = Array(train_indices)
                    path_el["test"] = Array(collect(split[1]:split[end]))
                    found = true
                end
            end
        end
    end

    return StackedCombinatorialPurgedKFold
end

function _generate_combinatorial_test_ranges(StackedCombinatorialPurgedKFold, splits_indices::Dict)::Array   

    # Possible test splits for each fold
    combinatorial_splits = Array(collect(combinations(Array(keys(splits_indices)), StackedCombinatorialPurgedKFold.n_test_splits)))
    combinatorial_test_ranges = Array([])  # Array of test indices formed from combinatorial splits
    for combination in combinatorial_splits
        temp_test_indices = Array([])  # Array of test indices for current split combination
        for int_index in combination
            append!(temp_test_indices, [splits_indices[int_index]])
        end
        append!(combinatorial_test_ranges, [temp_test_indices])
    end
    
    return combinatorial_test_ranges
end


function split(StackedCombinatorialPurgedKFold::StackedCombinatorialPurgedKFold, X_dict::Dict,
               y_dict::Dict = nothing,
               groups = nothing)::Tuple

    first_asset = Array(keys(X_dict))[1]
    for asset in keys(X_dict)
        size(X_dict[asset], 1) != size(StackedCombinatorialPurgedKFold.samples_info_sets[asset], 1) ? error("X and the 'samples_info_sets' series param must be the same length.") : Nothing()
    end

    test_ranges_assets = Dict()
    for asset in X_dict
        test_ranges_assets[asset] = [(ix[1], ix[end]) for ix in Iterators.partition(collect(size(X_dict[asset], 1)), ceil(Int64, size(X_dict[asset], 1)/StackedCombinatorialPurgedKFold.n_splits))]
        StackedCombinatorialPurgedKFold.backtest_paths[asset] = Array([])
    end

    split_indices_assets = Dict()
    combinatorial_test_ranges_assets = Dict()
    for asset in keys(X_dict)
        splits_indices = Dict()
        for (index, (start_ix, end_ix)) in enumerate(test_ranges_assets[asset])
            splits_indices[index] = [start_ix, end_ix]
        end
        split_indices_assets[asset] = splits_indices
        combinatorial_test_ranges_assets[asset] = _generate_combinatorial_test_ranges(StackedCombinatorialPurgedKFold, splits_indices)
    end

    # Prepare backtest paths
    for asset in keys(X_dict)
        for _ in collect(1:StackedCombinatorialPurgedKFold.num_backtest_paths)
            path = Array([])
            for split_idx in values(split_indices_assets[asset])
                append!(path, [Dict("train"=> nothing, "test"=> split_idx)])
            end
            append!(StackedCombinatorialPurgedKFold.backtest_paths[asset] , [path])
        end
    end

    final_test::Vector = []
    final_train::Vector = []

    for i in collect(1:size(combinatorial_test_ranges_assets[first_asset],1))
        train_indices_dict = Dict()  # Dictionary of asset: [train indices]
        test_indices_dict = Dict()
        for (asset, X_asset) in X_dict
            embargo::Int64 = floor(Int64, size(X_asset, 1) * StackedCombinatorialPurgedKFold.pct_embargo)
            test_splits = combinatorial_test_ranges_assets[asset][i]
            # Embargo
            test_times = Pandas.Series(index=[StackedCombinatorialPurgedKFold.samples_info_sets_dict[asset][ix[1]] for ix in test_splits], data=[  
            begin 
                if ix[2] - 1 + embargo >= size(X_asset,1)
                max(StackedCombinatorialPurgedKFold.samples_info_sets_dict[asset][ix[1]:ix[2]]) 
                else
                max(StackedCombinatorialPurgedKFold.samples_info_sets_dict[asset][ix[1]:ix[2] + embargo])
                end
            end     
            for ix in test_splits])

            test_indices = Array([])
            for (start_ix, end_ix) in test_splits
                append!(test_indices, [collect(start_ix:end_ix)])
            end

            # Purge
            train_times = ml_get_train_times(StackedCombinatorialPurgedKFold.samples_info_sets_dict[asset], test_times)

            # Get indices
            train_indices = Array([])
            for train_ix in index(train_times)
                append!(train_indices, findall(x -> x == train_ix ,index(StackedCombinatorialPurgedKFold.samples_info_sets_dict[asset])))
            end
        
            train_indices_dict[asset] = train_indices
            test_indices_dict[asset] = test_indices

            StackedCombinatorialPurgedKFold = _fill_backtest_paths(StackedCombinatorialPurgedKFold, asset, train_indices, test_splits)

        end       
        append!(final_test, [test_indices_dict])
        append!(final_train, [train_indices_dict])
    end

    return (Tuple(zip(final_train, final_test)), StackedCombinatorialPurgedKFold)
end            