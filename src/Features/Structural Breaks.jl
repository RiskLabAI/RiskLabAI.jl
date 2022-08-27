"""----------------------------------------------------------------------
    function: Compute β 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: snippet 17.4
----------------------------------------------------------------------"""
function ComputeBeta(X, # matrix of independent variable 
                     y) # dependent variable
    β = inv(transpose(X) * X) * transpose(X) * y # Compute β with OLS estimator 
    ϵ = y .- X * β # compute error 
    BetaVariance = transpose(ϵ) * ϵ /(size(X)[1] - size(X)[2]) * inv(transpose(X) * X) # compute variance of β
    return β,BetaVariance
end

"""----------------------------------------------------------------------
    function: Prepare Data for test
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: snippet 17.2 & 17.3
----------------------------------------------------------------------"""
function PrepareData(data, # data of price or log price
                     constant, # string thant must be "nc" or "ct" or "ctt"
                     lags) # arrays of lag or integer that show number of lags 
    if isinteger(lags) #check that lag is integer of array 
        lags = 1:lags
    else
        lags = [Int(lag) for lag in lags]
    end
    diffData = diff(data) #compute data difference 
    X_index = [Int.((zeros(length(lags)) .+ i) .- lags) for i in maximum(lags)+1:length(diffData)] # index of each row . each row show featurse 
    X = [diffData[xindex] for xindex in X_index] # create feature matrix
    X = hcat(X...)'
    y=diffData[maximum(lags)+1:length(diffData)] # create dependent variable array respect to X 
    X = hcat(data[(length(data)-length(y)):(end-1)],X) # add actual data columns to feature matrix 
    if constant  != "nc" # check for time trend and add 1,t or t^2 columns
        X = hcat(X,ones(size(X)[1])) #append columns with 1
        trend = 1:size(X)[1]
        if constant[1:2] == "ct"
            X = hcat(X,trend) # append columns t 
        end
        if constant == "ctt"
            X = hcat(X,trend .^ 2) # append columns t^2
        end
    end
    return X,y
end

"""----------------------------------------------------------------------
    function: SADF inner loop
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: snippet 17.1
----------------------------------------------------------------------"""
function ADF(X, # feature matrix that contain data with lags 
             y, # dependent variable
             minSampleLength) # minimum sample length for computing OLS 
    start_points = 1:(length(y) - minSampleLength + 1) # create start_points array 
    ADF_ = zeros(length(start_points)) # create ADF array that contain 
    for (i,start) in enumerate(start_points)
        y_, X_ = y[start:end], X[start:end,:]
        β_mean, β_std = ComputeBeta(X_, y_) #compute β and and its variance 
        if !isnan(β_mean[1])
            β_mean_,β_std_ = β_mean[1], β_std[1, 1] ^ 0.5 # get coefficient of first independent variable
            ADF_[i] = β_mean_ / β_std_ # compute augmented dicky fuller statatistics
        end
    end
    return ADF_
end
"""----------------------------------------------------------------------
    function: SADF test statatistics
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: -
----------------------------------------------------------------------"""

function ADFTestType(data, #dataframe of price that have date indxe 
                     minSampleLength, # minimum sample length for ols estimator
                     constant, # string thant must be "nc" or "ct" or "ctt"
                     lags,# arrays of lag or integer that show number of lags 
                     type; # type of ADF test SADF or QADF or CADF
                     quantile = nothing, # quantile for QADF
                     probability = ones(size(data)[1])) # probability or weight for CADF
    X,y = PrepareData(data.price,constant,lags) # Prepare data for test 
    maxlag = 0 # initial maxlag with zero 
    if isinteger(lags)
        maxlag = lags 
    else
        maxlag = maximum(lags)
    end
    indexs = (minSampleLength - maxlag) : length(y) # initial index 
    result = zeros(length(indexs))
    for (i,index) in enumerate(indexs)
        X_ ,y_= X[1:index,:],y[1:index] 
        adf_ = ADF(X_,y_,minSampleLength) # compute adf statatistics 
        if length(adf_) ==0
            continue
        end
        if type == "SADF" # check typ of ADF and then add it to result
            result[i] = maximum(adf_)
        elseif type =="QADF"
            result[i] = sort(adf_)[ floor(quantile * length(adf_))]
        elseif  type == "CADF"
            perm = sortperm(adf_)
            perm = perm[floor(quantile * length(adf_)) : end] #choose quantile  of perm
            result[i] = adf_[perm] .* probability[perm] / sum(probability[perm]) # compute weighted average for CADF
        else 
            println("type must be SADF or QADF or CADF")
        end
    end
    ADFStatistics = DataFrame(index = data.index[minSampleLength:length(y) + maxlag ] , statistics = result)
    return ADFStatistics
end

"""----------------------------------------------------------------------
    function: impement Brow-Durbin-Evans cumsum test 
    reference: Techniques for Testing the Constancy of Regression Relationships over Time(1975)
    methodology: -
----------------------------------------------------------------------"""

function BrownDurbinEvansTest(X, # feature matrix(price with lags )
                              y, # dependent variable
                              lags, # integer lags
                              k, # first index that statistics compute after that 
                              index) # index of feature matrix it is usuallt date 
    # compure variance of error of OLS estimator
    β,_ = ComputeBeta(X,y) 
    ϵ = y - X*β
    σ  = ϵ' * ϵ /(length(y) - 1 + lags - k)

    startindex = k - lags+1
    cumsum = 0  # initial cumsum with zero
    BDEcumsumstatistics = zeros(length(y) - startindex)
    for i in startindex:length(y) -1 
        X_,y_ = X[1:i,:] , y[1:i]
        β,_ = ComputeBeta(X_,y_)

        #compute statatistics 
        ω = (y[i+1] - X[i+1,:]' * β)/sqrt(1+X[i+1,:]' * inv(X_' * X_)* X[i+1,:]) # eq (2)
        cumsum += ω
        BDEcumsumstatistics[i-startindex + 1] = cumsum/ sqrt(σ)
    end
    BDECstatistics_ = DataFrame(index = index[k:length(y) + lags - 2] ,BDECstatistics = BDEcumsumstatistics )
    return BDECstatistics_
end
"""----------------------------------------------------------------------
    function: implementing of Chu-Stinchcombe-White test 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.251 
----------------------------------------------------------------------"""
function ChuStinchcombeWhiteTest(data, # dataframe of log price with date index  
                                 testType, # type of test it must be one_side or two_sided
                                 interval) # interval that test statatistics compute on it
    CSWstatistics , CSWtreashold = zeros(length(interval)),zeros(length(interval))
    for (idx,i) in enumerate(interval)
        price = data[data.index .<= i,:].price # initial price with date befor i 
        diffLogPrice = diff(price) # compute difference of log price 
        σ = sum(diffLogPrice .^ 2)/length(diffLogPrice) # compute variance of diffLogPrice assume mean is zero 
        max_S_n ,max_s_n_rejecting_threshold = -1 * Inf , 0 
        for j in 1:length(price) - 1
            S_nt = 0  
            # equation on page 251 
            if testType == "one_sided"
                S_nt = price[end] - data.price[j]
            elseif testType =="two_sided"
                S_nt = abs(price[end] - data.price[j])
            end
            S_nt /= (sqrt(σ *(length(price)-j)))
            if S_nt > max_S_n 
                max_S_n = S_nt
                max_s_n_rejecting_threshold = sqrt(4.6 + log(length(price)-j))
            end
        end
        CSWstatistics[idx] = max_S_n
        CSWtreashold[idx] = max_s_n_rejecting_threshold
    end
    CSWtest = DataFrame(index = interval , S_n = CSWstatistics , threshold = CSWtreashold)
    return CSWtest    
end
"""----------------------------------------------------------------------
    function: implementing of Chow-Type Dickey-Fuller test inner loop 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.252
----------------------------------------------------------------------"""
function ChowTypeDickeyFullerTestInnderLoop(data, # dataframe of log price with date index  
                                            interval)  # interval that test statatistics compute on it
    DFC = zeros(length(interval))
    for (i,dates) in enumerate(interval)
        δy = diff(data.price) # compute difference of data price or logprice 
        y = [(data.index[j] <= dates) ? 0.0 : data.price[j] for j in 1:length(δy)] #initial dependent variable with zero one index that less than dates
        y = reshape(y,(length(y),1))
        δy = reshape(δy ,(length(y),1))
        β,βstd = ComputeBeta(y,δy) 
        DFC[i] = β[1] / sqrt(βstd[1,1]) # compute statistics
    end
    DFCDataframe = DataFrame(index = interval , DFC = DFC)
    return DFCDataframe
end
"""----------------------------------------------------------------------
    function: implementing of Chow-Type Dickey-Fuller test 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.252
----------------------------------------------------------------------"""
function SDFC(data, # dataframe of price or log price with date index 
              interval, # interval that statatistics compute on it 
              τ, # number between 0,1 that we compute interval of ChowTypeDickeyFullerTestInnderLoop function interval based on it 
              T) # maximum lags price for estimating model 
    sdfc_ = zeros(length(interval))
    for (idx,i) in enumerate(interval)
        selectdata = data[(data.index .>= (i - Dates.Day(T))) .& (data.index .<= i),:] # select specifice interval of data 
        integertau = floor(T * τ)
        selectinterval = data.index[(data.index .>= (i - Dates.Day(T-integertau))) .& (data.index .<= (i - Dates.Day(integertau)))] # select interval for ChowTypeDickeyFullerTestInnderLoo function
        dfc = ChowTypeDickeyFullerTestInnderLoop(selectdata,selectinterval) # compute Chow-Type Dickey-Fuller statatistics
        sdfc_[idx] = maximum(dfc.DFC)
    end
    SDFC_ = DataFrame(index = interval , SDFC = sdfc_)
    return SDFC_
end