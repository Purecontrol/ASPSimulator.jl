################################################################################################################################

"""
This scripts is a modified version of the PeriodicCallback function in DiffEqCallbacks.jl so that is it possible to add an offset.
"""

using Parameters

struct PeriodicCallbackAffect{A, dT, Ref1, Ref2}
    affect!::A
    Δt::dT
    t0::Ref1
    index::Ref2
end
    
function (S::PeriodicCallbackAffect)(integrator)
    @unpack affect!, Δt, t0, index = S

    affect!(integrator)

    tstops = integrator.opts.tstops

    # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever
    tnew = t0[] + (index[] + 1) * Δt
    tstops = integrator.opts.tstops
    #=
    Okay yeah, this is nasty
    the comparer is always less than for type stability, so in order
    for this to actually check the correct direction we multiply by
    tdir
    =#
    tdir_tnew = integrator.tdir * tnew
    for i in length(tstops):-1:1 # reverse iterate to encounter large elements earlier
        if tdir_tnew < tstops.valtree[i] # TODO: relying on implementation details
            index[] += 1
            add_tstop!(integrator, tnew)
            break
        end
    end
end

function PeriodicOffsetCallback(f, Δt::Number;
                          offset::Number = 0,
                          initial_affect = false,
                          initialize = (cb, u, t, integrator) -> u_modified!(integrator,
                                                                             initial_affect),
                          kwargs...)

    # Value of `t` at which `f` should be called next:
    t0 = Ref(typemax(Δt))
    index = Ref(0)
    condition = (u, t, integrator) -> t == (t0[] + index[] * Δt)

    # Call f, update tnext, and make sure we stop at the new tnext
    affect! = PeriodicCallbackAffect(f, Δt, t0, index)

    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize_periodic = function (c, u, t, integrator)
        @assert integrator.tdir == sign(Δt)
        initialize(c, u, t, integrator)
        t0[] = t + offset
        if initial_affect
            index[] = 0
            affect!(integrator)
        else
            index[] = 1
            add_tstop!(integrator, t0[] + Δt)
        end
    end

    DiscreteCallback(condition, affect!; initialize = initialize_periodic, kwargs...)
end

################################################################################################################################

"""
According to the input, get the number or evaluate the function.
"""
@inline evaluate(v::Function, t) = v(t)
@inline evaluate(v::Number, t) = v


"""
Adapt the value of the parameter ρ according to the temperature T and the coefficient a following the Van 't Hoff equation.
"""
function T_var(T, ρ, a)
    return ρ * exp((log2(ρ/a)/5)*(T-15))
end  


"""
Return the inflow specified in a source file coming from the Benchmark Simulation model.
"""
function get_inflow_from_bsm_files(influent_file_path)

    # Read file
    inflow_generator = readdlm(influent_file_path)

    # Set up interpolation function with data provided in the file
    itp = interpolate((inflow_generator[: ,1], ), inflow_generator[: ,10], Gridded(Linear()))

    # Define a function giving the inflow
    T_max = maximum(inflow_generator[: ,1])
    function Q_in(t) 
         return itp(abs(t) % T_max)
    end

    return Q_in
end


"""
Return the inlet concentrations specified in a source file coming from the Benchmark Simulation model.
"""
function get_inlet_concentrations_from_src_files(influent_file_path; output_index = collect(1:13))

    # Read file
    inflow_generator = readdlm(influent_file_path)

    # Set up interpolation function with data provided in the file
    # Real usage of the file but don't work whithout using a clarifier
    # list_order = [7, 2, 5, 4, 3, 0.0, 0.0, 0.0, 0.0, 6, 8, 9, 7.0]
    # constant_value = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
    # Our modified usage
    list_order = [28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699, 964.8992, 0.0093, 3.9350, 6, 0.9580, 3.8453, 5.4213]
    constant_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    divide_by = [1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1]
    list_itp = [(constant_value[i] ==  0) ? interpolate((inflow_generator[: ,1], ), inflow_generator[: ,Int(list_order[i])]/divide_by[i], Gridded(Linear())) :  interpolate((inflow_generator[: ,1], ), list_order[i] .* ones(size(inflow_generator, 1))/divide_by[i], Gridded(Linear())) for i in 1:13]
    
    # Define a function giving the inlet concentration
    T_max = maximum(inflow_generator[: ,1])
    function X_in(t) 
        return [sum([list_itp[j](abs(t) .% T_max) for j in vcat([i]...)]) for i in output_index] 
    end

    return X_in
end