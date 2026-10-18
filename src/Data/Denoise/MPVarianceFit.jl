"""
    _mp_bounded_local_min(objective; derivative=nothing, max_iter=15000, max_evals=15000)

Fit the scalar MP variance from 0.5 on [1e-5, 1-1e-5] with projected BFGS steps,
an optional analytic derivative or forward differences and the reference line-search and stopping
settings. In one dimension the secant curvature defines the BFGS approximation.
More–Thuente trial intervals retain their extrapolated limits before applying
the absolute step bound. Warning steps are accepted before testing the outer
stopping rules. This is a local fit; nonconvergence is returned explicitly for
the public variance-one fallback. Floating-point differences near stationary
points can change line-search termination and therefore the fallback outcome.
"""
function _mp_bounded_local_min(objective; derivative = nothing, max_iter::Integer = 15000, max_evals::Integer = 15000)
    low, high, x = 1e-5, 1-1e-5, .5
    evaluations = Ref(0)
    evaluate = value -> begin
        evaluations[] += 1
        Float64(objective(clamp(value, low, high)))
    end
    gradient = (value, result) -> begin
        derivative !== nothing && return Float64(derivative(value))
        next = value+1e-8 <= high ? value+1e-8 : value-1e-8
        (evaluate(next)-result)/(next-value)
    end
    f = evaluate(x)
    g = gradient(x, f)
    curvature = 1.
    result = (value, objective_value, converged, iteration) ->
        (variance = value, objective = objective_value, converged = converged,
            iterations = iteration, evaluations = evaluations[])
    for iteration in 1:max_iter
        (isfinite(f) && isfinite(g)) || return result(x, f, false, iteration-1)
        abs(x-clamp(x-g, low, high)) <= 1e-5 && return result(x, f, true, iteration-1)
        direction = clamp(x-g/curvature, low, high)-x
        maximum_step = direction > 0 ? (high-x)/direction : (low-x)/direction
        objective_line = step -> begin
            trial = clamp(x+step*direction, low, high)
            value = evaluate(trial)
            return value, gradient(trial, value)*direction
        end
        search = _mp_line_search(objective_line, f, g*direction,
            iteration == 1 ? 1. : min(maximum_step, 1e10))
        if !search.converged
            if curvature != 1. && evaluations[] < max_evals
                curvature = 1.
                continue
            else
                return result(x, f, false, iteration-1)
            end
        end
        next_x = clamp(x+search.step*direction, low, high)
        next_f = evaluate(next_x)
        next_g = gradient(next_x, next_f)
        evaluations[] > max_evals && return result(next_x, next_f, false, iteration)
        if abs(next_x-clamp(next_x-next_g, low, high)) <= 1e-5 || f-next_f <= 2.220446049250313e-9*max(abs(f), abs(next_f), 1.)
            return result(next_x, next_f, true, iteration)
        end
        displacement = next_x-x
        change = next_g-g
        change*displacement > eps(Float64)*(-g*displacement) && (curvature = change/displacement)
        x, f, g = next_x, next_f, next_g
    end
    return result(x, f, false, max_iter)
end

"""Safeguarded cubic and secant interpolation for the scalar MP line search."""
function _mp_interpolate(best, opposite, trial, bracketed, limits)
    x, fx, gx = best
    y, fy, gy = opposite
    t, ft, gt = trial
    lower, upper = limits
    sign_change = sign(gt)*sign(gx)<0
    if ft>fx
        theta = 3*(fx-ft)/(t-x)+gx+gt
        scale = max(abs(theta),abs(gx),abs(gt))
        gamma = scale*sqrt(max(0.,(theta/scale)^2-(gx/scale)*(gt/scale)))
        t<x && (gamma=-gamma)
        cubic = x+(gamma-gx+theta)/(2gamma-gx+gt)*(t-x)
        quadratic = x+(gx/((fx-ft)/(t-x)+gx)/2)*(t-x)
        next = abs(cubic-x)<=abs(quadratic-x) ? cubic : cubic+(quadratic-cubic)/2
        bracketed = true
    elseif sign_change
        theta = 3*(fx-ft)/(t-x)+gx+gt
        scale = max(abs(theta),abs(gx),abs(gt))
        gamma = scale*sqrt(max(0.,(theta/scale)^2-(gx/scale)*(gt/scale)))
        t>x && (gamma=-gamma)
        cubic = t+(gamma-gt+theta)/(2gamma-gt+gx)*(x-t)
        secant = t+gt/(gt-gx)*(x-t)
        next = abs(cubic-t)>abs(secant-t) ? cubic : secant
        bracketed = true
    elseif abs(gt)<abs(gx)
        theta = 3*(fx-ft)/(t-x)+gx+gt
        scale = max(abs(theta),abs(gx),abs(gt))
        gamma = scale*sqrt(max(0.,(theta/scale)^2-(gx/scale)*(gt/scale)))
        t>x && (gamma=-gamma)
        ratio = (gamma-gt+theta)/(gamma+(gx-gt)+gamma)
        cubic = ratio<0 && gamma!=0 ? t+ratio*(x-t) : (t>x ? upper : lower)
        secant = t+gt/(gt-gx)*(x-t)
        if bracketed
            next = abs(cubic-t)<abs(secant-t) ? cubic : secant
            next = t>x ? min(t+.66*(y-t),next) : max(t+.66*(y-t),next)
        else
            next = abs(cubic-t)>abs(secant-t) ? cubic : secant
            next = clamp(next,lower,upper)
        end
    elseif bracketed
        theta = 3*(ft-fy)/(y-t)+gy+gt
        scale = max(abs(theta),abs(gy),abs(gt))
        gamma = scale*sqrt(max(0.,(theta/scale)^2-(gy/scale)*(gt/scale)))
        t>y && (gamma=-gamma)
        next = t+(gamma-gt+theta)/(2gamma-gt+gy)*(y-t)
    else
        next = t>x ? upper : lower
    end
    if ft>fx
        opposite = trial
    else
        sign_change && (opposite=best)
        best = trial
    end
    return best,opposite,next,bracketed
end

"""Use the public bounded fit's sufficient decrease, curvature and warning rules."""
function _mp_line_search(evaluate, initial_value, initial_gradient, maximum_step)
    best = (0.,initial_value,initial_gradient)
    opposite = best
    step, lower, upper = 1.,0.,5.
    width, previous_width = maximum_step,2maximum_step
    bracketed, modified = false,true
    decrease_gradient = 1e-3*initial_gradient
    for evaluation in 1:20
        value, gradient = evaluate(step)
        isfinite(value) && isfinite(gradient) || return (step=step,converged=false)
        sufficient = initial_value+step*decrease_gradient
        modified && value<=sufficient && gradient>=0 && (modified=false)
        warning = bracketed && (step<=lower || step>=upper || upper-lower<=.1upper)
        warning |= step==maximum_step && value<=sufficient && gradient<=decrease_gradient
        warning |= step==0 && (value>sufficient || gradient>=decrease_gradient)
        wolfe = value<=sufficient && abs(gradient)<=-.9initial_gradient
        (warning || wolfe) && return (step=step,converged=true)
        if modified && value<=best[2] && value>sufficient
            transformed(point) = (point[1],point[2]-point[1]*decrease_gradient,point[3]-decrease_gradient)
            recover(point) = (point[1],point[2]+point[1]*decrease_gradient,point[3]+decrease_gradient)
            best,opposite,step,bracketed = _mp_interpolate(transformed(best),transformed(opposite),transformed((step,value,gradient)),bracketed,(lower,upper))
            best,opposite = recover(best),recover(opposite)
        else
            best,opposite,step,bracketed = _mp_interpolate(best,opposite,(step,value,gradient),bracketed,(lower,upper))
        end
        if bracketed
            abs(opposite[1]-best[1])>=.66previous_width && (step=best[1]+.5*(opposite[1]-best[1]))
            previous_width,width = width,abs(opposite[1]-best[1])
            lower,upper = minmax(best[1],opposite[1])
        else
            lower,upper = step+1.1*(step-best[1]),step+4*(step-best[1])
        end
        step = clamp(step,0.,maximum_step)
        bracketed && (step<=lower || step>=upper || upper-lower<=.1upper) && (step=best[1])
    end
    return (step=step,converged=false)
end
