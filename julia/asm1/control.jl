using DifferentialEquations
include("../add_on_DiffEqCallbacks/periodic.jl")

"""
Returns a CallbackSet that simulates the redox control of a bioreactor.
"""
function redox_control()

    # If the S_{NO} is near 0, then the aeration is turned on
    S_NO_low(u,t,integrator) = u[9] < 0.5
    aeration_on!(integrator) = integrator.u[14] = 1
    redox_c1 = DiscreteCallback(S_NO_low, aeration_on!; save_positions=(false, false))

    # If the S_{NH} is near 0, then the aeration is turned off
    S_NH_low(u,t,integrator) = u[10] < 0.5
    aeration_off!(integrator) = integrator.u[14] = 0
    redox_c2 = DiscreteCallback(S_NH_low, aeration_off!; save_positions=(false, false))

    redox_callback = CallbackSet(redox_c1, redox_c2)

    return redox_callback

end

"""
Returns a CallbackSet that simulates the clock control of a bioreactor. The time are given in minutes.
"""
function clock_control(;t_aerating = 60.0, t_waiting = 60.0)

    # Convert time given in minutes to days
    t_aerating = t_aerating / 60 / 24
    t_waiting = t_waiting / 60 / 24

    # Periodically turn on and off the aeration
    stop_aerating = PeriodicCallback( (integrator) -> integrator.u[14] = 0, t_waiting + t_aerating, save_positions=(false, false))
    start_aerating = PeriodicOffsetCallback( (integrator) -> integrator.u[14] = 1, t_waiting + t_aerating, offset=t_waiting, initial_affect=true, save_positions=(false, false))

    clock_callback = CallbackSet(stop_aerating, start_aerating)

    return clock_callback

end

