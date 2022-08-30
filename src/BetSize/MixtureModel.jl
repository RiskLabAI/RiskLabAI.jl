# Mixture model object defination 
Base.@kwdef mutable struct M2N
    moments 
    parameters::Array = zeros(5)
    error::Float64= sum(moments.^2)
end

#Update error of Mixture model 
function UpdateEror!(M2Nobject)
    M2Nobject.error = sum(M2Nobject.moments.^2)
end

"""----------------------------------------------------------------------
    function: Compute moments of Mixture model with parameters of model
    reference: De Prado, M. & Foreman M. (2013) A Mixture of gaussina approch to mathematical portfolio oversight: the EF3M algorithm
    methodology: p22
----------------------------------------------------------------------"""
function computemoments(parameters #parameters of gaussian mixture model
                        )
        μ_1, μ_2, σ_1, σ_2, p_1 = parameters  # initial parameters
        p_2 = 1 - p_1  # calculate p_2
        firstMomemnt = p_1 * μ_1 + p_2 * μ_2  # Eq. (6) of paper
        seccondMoments  = p_1 * (σ_1^2 + μ_1^2) + p_2 *(σ_2^2 + μ_2^2)  # Eq. (7) of paper
        thirdMoments = p_1 * (3 * σ_1^2 *μ_1  + μ_1^3) + p_2 * (3 * σ_2^2 * μ_2 + μ_2^3)  # Eq. (8) paper
        forthMoments = p_1 * (3 * σ_1^4 + 6 * σ_1^2 * μ_1^2 + μ_1^4) + p_2 * (3 * σ_2^4 + 6 * σ_2^2 * μ_2^2 + μ_2^4)  # Eq. (9) paper
        fifthMoments = p_1 * (15 * σ_1^4 * μ_1 + 10 * σ_1^2 * μ_1^3 + μ_1^5) + p_2 *(15 * σ_2^4 * μ_2 + 10 * σ_2^2 * μ_2^3 + μ_2^5) # Eq. (10) paper
        return [firstMomemnt, seccondMoments , thirdMoments, forthMoments, fifthMoments] # return moments of model
 end

 """----------------------------------------------------------------------
 function: Estimate parameters of model with 4 moments
 reference: De Prado, M. & Foreman M. (2013) A Mixture of gaussina approch to mathematical portfolio oversight: the EF3M algorithm
 methodology: p22
----------------------------------------------------------------------"""

 function iterationwith4moments(μ_2, # mean of seccond normal model
                                p1, # probability of choosing first normal model 
                                moments) # 4 moments of gaussian mixture model 
    μ_1 = (moments[1]-(1-p1)*μ_2)/p1   # Eq. (22) 
    σ_2 = ((moments[3]+2*p1*μ_1^3 + (p1-1)*μ_2^3 - 3*μ_1*(moments[2] + μ_2^2*(p1-1)))/(3*(1-p1)*(μ_2 - μ_1)))^(.5) # Eq. (23) 
    σ_1 = ((moments[2]- σ_2^2 - μ_2^2)/p1  +  σ_2^2 + μ_2^2 - μ_1^2)^(.5) # Eq. (24) 
    p1 = (moments[4]-3*σ_2^4-6*σ_2^2*μ_2^2-μ_2^4)/(3*(σ_1^4- σ_2^4)+ 6*(σ_1^2*μ_2^2-σ_2^2*μ_2^2)+μ_1^4-μ_2^4) # Eq. (25) 

    return [μ_1,μ_2,σ_1,σ_2,p1]
 end


 """----------------------------------------------------------------------
 function: Estimate parameters of model with 5 moments
 reference: De Prado, M. & Foreman M. (2013) A Mixture of gaussina approch to mathematical portfolio oversight: the EF3M algorithm
 methodology: p22
----------------------------------------------------------------------"""
 function iterationwith5moments(μ_2, # mean of seccond normal model
                                p1, # probability of choosing first normal model 
                                moments) # 5 moments of gaussian mixture model 

    μ_1 = (moments[1]-(1-p1)*μ_2)/p1 # Eq. (22) 
    σ_2 = ((moments[3]+2*p1*μ_1^3 + (p1-1)*μ_2^3 - 3*μ_1*(moments[2] + μ_2^2*(p1-1)))/(3*(1-p1)*(μ_2 - μ_1)))^(.5) # Eq. (23) 
    σ_1 = ((moments[2]- σ_2^2 - μ_2^2)/p1  +  σ_2^2 + μ_2^2 - μ_1^2)^(.5) # Eq. (24) 

    # Eq. (27)
    a = (6*σ_2^4+(moments[4]-p1*(3*σ_1^4+6*σ_1^2*μ_1^2+ μ_1^4))/ (1-p1))^.5 
    μ_2 = (a-3*σ_2^2)^.5

    # Eq. (29)
    a = 15*σ_1^4 *μ_1 + 10*σ_1^2*μ_1^3+μ_1^5
    b = 15*σ_2^4*μ_2+10*σ_2^2 *μ_2^3 + μ_2^5

    p1 = (moments[5]-b)/(a-b) # Eq. (28)
   
    return [μ_1,μ_2,σ_1, σ_2, p1]
 end

 """----------------------------------------------------------------------
 function: Update other parameters of model based on μ_2 and random p1
 reference: De Prado, M. & Foreman M. (2013) A Mixture of gaussina approch to mathematical portfolio oversight: the EF3M algorithm
 methodology: p21
----------------------------------------------------------------------"""

function firstfitloop!(guassianmixturemodel::M2N, # M2N object
                       μ_2, # mean of seccond normal model
                       ϵ) # ϵ for stoping algorithm 

    p1 = rand(1)[1] # initial p1 with random number between 0 and 1 
    NumberOfIteration = 0 # initial number of iteration to zero
    parameters = zeros(5) # initial parameters to zeros 
    while  true   
        NumberOfIteration += 1 # increase NumberOfIteration by 1 
        try # if computed σ is negative finish loop
            #parameters =  iterationwith4moments(μ_2,p1,guassianmixturemodel.moments) # estimate other parameters of model based on fixed μ_2 and p_1 with 4 moments
            parameters =  iterationwith5moments(μ_2,p1,guassianmixturemodel.moments) # estimate other parameters of model based on fixed μ_2 and p_1 with 5 moments
        catch
            break
        end

        moments = computemoments(parameters) # compute moments of model based on estimated parameters
        error = sum((moments - guassianmixturemodel.moments).^2) # calculate error 
        
        if error < guassianmixturemodel.error # if error is less than all previous error update error and parameters
            guassianmixturemodel.parameters = parameters # update parameters
            guassianmixturemodel.error = error # update error
        end

        if abs(p1 - parameters[5]) <ϵ  # check if differences between p1 and estimated p1 is less than ϵ then break
            return 
        end 
        
        if NumberOfIteration > 1.0 / ϵ  # check if number of iteration is more than 1/ϵ then break
            return 
        end
        p1 = parameters[5] # update parameters
        μ_2 = parameters[2]   # update parameters
    end
end

"""----------------------------------------------------------------------
function: Fit gaussian mixture model on data 
reference: De Prado, M. & Foreman M. (2013) A Mixture of gaussina approch to mathematical portfolio oversight: the EF3M algorithm
methodology: p23
----------------------------------------------------------------------"""

function fit!(guassianmixturemodel::M2N, # M2N object
              data; # data for fitting mixture model
              ϵ = 1e-4, # step size for searching best μ2 
              λ = 5) # coefficient of step Size 
  
    guassianmixturemodel.moments = [mean(data .^ i) for i in 1:5] # calculate moments of model based on data

    μ_2 = [i * ϵ * λ * Statistics.std(data) + guassianmixturemodel.moments[1] for i in 1:Int(round(1 / ϵ) + 1)] # define array of μ_2 for choosing best μ_2
    UpdateEror!(guassianmixturemodel) # update error of guassianmixturemodel 
    MinimumEror = guassianmixturemodel.error # initial minimum error with guassianmixturemodel error
    BestParameters = [] # initial BestParameters with nothing!
    for i in μ_2 #loop in candidate for μ_2
        firstfitloop!(guassianmixturemodel,i,0.001) # update parameters of model based on candidate μ_2
        if guassianmixturemodel.error < MinimumEror # check if model error is less than all previous errors then update error and parameters
            BestParameters  = guassianmixturemodel.parameters # update parameters
            MinimumEror = guassianmixturemodel.error # update error
        end
    end

    guassianmixturemodel.parameters = BestParameters ; # choose best parameters
    guassianmixturemodel.error = MinimumEror ; # choose best error
end




