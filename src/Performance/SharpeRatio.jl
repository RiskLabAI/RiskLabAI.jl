"""
function: Implementation of the symmetric CUSUM filter
reference: De Prado, M. (2018) Advances in financial machine learning.
methodology: page 213, snippet 15.1
"""
function sharpeRatioTrials(p, # probability of success
						   nRun) # number of runs
  
    result=[] # 
    for i in 1:nRun
        b = Binomial(1, p)
        random = rand(b, 1)
        if random[1] == 1
            x = 1
        else 
            x = -1
        end
        append!(result, [x])
    end
    return (mean(result), std(result), mean(result)/std(result))
end

"""
function: Uses the SymPy library for symbolic operations 
reference: De Prado, M. (2018) Advances in financial machine learning.
methodology: page 214, snippet 15.2
"""
function targetSharpeRatioSymbolic()
    p,u,d = symbols("p u d") # Create symbols
    m2 = p * u ^ 2 + (1 - p) * d ^ 2
    m1 = p * u + (1 - p) * d
    v = m2 - m1 ^ 2
    factor(v)
end

"""
function: computes implied precision 
reference: De Prado, M. (2018) Advances in financial machine learning.
methodology: page 214, snippet 15.3
"""
function impliedPrecision(stopLoss, # stop loss threshold
			   			  profitTaking, # profit taking threshold
			   			  freq, # number of bets per year
			   	 		  targetSharpeRatio) # target annual Sharpe ratio

	a = (freq + targetSharpeRatio^2)*(profitTaking - stopLoss)^2
	b = (2*freq*stopLoss - targetSharpeRatio^2*(profitTaking - stopLoss))*(profitTaking - stopLoss)
	c = freq*stopLoss^2
	p = (-b + (b^2 - 4*a*c)^0.5)/(2*a)
	return p
end
