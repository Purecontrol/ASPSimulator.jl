using DifferentialEquations, Interpolations
include("../utils.jl")

"""
Returns a CallbackSet that simulates the redox control of a bioreactor.
"""
function redox_control(;index_no3 = 9, index_nh4 = 10, index_u = -1)

    # If the S_{NO} is near 0, then the aeration is turned on
    S_NO_low(u,t,integrator) = u[index_no3] < 0.5
    aeration_on!(integrator) = index_u == -1 ? integrator.u[end] = 1 : integrator.u[index_u] = 1
    redox_c1 = DiscreteCallback(S_NO_low, aeration_on!; save_positions=(false, false))

    # If the S_{NH} is near 0, then the aeration is turned off
    S_NH_low(u,t,integrator) = u[index_nh4] < 0.5
    aeration_off!(integrator) = index_u == -1 ? integrator.u[end] = 0 : integrator.u[index_u] = 0
    redox_c2 = DiscreteCallback(S_NH_low, aeration_off!; save_positions=(false, false))

    redox_callback = CallbackSet(redox_c1, redox_c2)

    return redox_callback

end


"""
Returns a CallbackSet that simulates the clock control of a bioreactor. The time are given in minutes.
"""
function clock_control(;t_aerating = 60.0, t_waiting = 60.0, index_u = -1)

    # Convert time given in minutes to days
    t_aerating = t_aerating / 60 / 24
    t_waiting = t_waiting / 60 / 24

    # Periodically turn on and off the aeration
    start_aerating = PeriodicCallback( (integrator) -> index_u == -1 ? integrator.u[end] = 1 : integrator.u[index_u] = 1, t_waiting + t_aerating, initial_affect=true, save_positions=(false, false))
    stop_aerating = PeriodicOffsetCallback( (integrator) -> index_u == -1 ? integrator.u[end] = 0 : integrator.u[index_u] = 0, t_waiting + t_aerating, offset=t_aerating, initial_affect=false, save_positions=(false, false))

    clock_callback = CallbackSet(start_aerating, stop_aerating)

    return clock_callback

end

"""
Returns a ContinuousCallback that simulates the external control of a bioreactor given as a time array and a control array.
"""
function external_control(array_t, array_u; index_u = -1)

    # Create a function that returns the control value at a given time
    control = Interpolations.interpolate((array_t,), vcat(array_u[2:end], array_u[end]), Gridded(Linear()))

    # Search for the change in the control value in the array_u
    event_times = [0.0]
    for i in 1:length(array_u)-1
        if abs(array_u[i] - array_u[i+1]) > 0.5
            push!(event_times, array_t[i])
        end
    end

    # Create a callback that changes the control value at the given times
    external_control_callback = PresetTimeCallback(event_times,(integrator) -> index_u == -1 ? integrator.u[end] = abs(control(integrator.t)) : integrator.u[index_u] = abs(control(integrator.t)), save_positions=(false, false))

    return external_control_callback

end