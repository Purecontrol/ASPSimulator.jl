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