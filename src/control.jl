using DiffEqCallbacks: DiscreteCallback, CallbackSet, PeriodicCallback, PresetTimeCallback
using Interpolations: interpolate, Gridded, Linear

"""
    $(TYPEDSIGNATURES)

Returns a CallbackSet that simulates the redox control of a bioreactor.
"""
function redox_control(; index_no3 = 9, index_nh4 = 10, index_u = -1)

    # If the S_{NO} is near 0, then the aeration is turned on
    S_NO_low(u, t, integrator) = u[index_no3] < 0.5
    function aeration_on!(integrator)
        index_u == -1 ? integrator.u[end] = 1 : integrator.u[index_u] = 1
    end
    redox_c1 = DiscreteCallback(S_NO_low, aeration_on!; save_positions = (false, false))

    # If the S_{NH} is near 0, then the aeration is turned off
    S_NH_low(u, t, integrator) = u[index_nh4] < 0.5
    function aeration_off!(integrator)
        index_u == -1 ? integrator.u[end] = 0 : integrator.u[index_u] = 0
    end
    redox_c2 = DiscreteCallback(S_NH_low, aeration_off!; save_positions = (false, false))

    redox_callback = CallbackSet(redox_c1, redox_c2)

    return redox_callback
end
@inline redox_control(s::Symbol) = redox_control(
    index_no3 = get_no3_index(s), index_nh4 = get_nh4_index(s),
    index_u = get_control_index(s))

"""
    $(TYPEDSIGNATURES)

Returns a CallbackSet that simulates the clock control of a bioreactor. The time are given in minutes.
"""
function clock_control(; t_aerating = 60.0, t_waiting = 60.0, index_u = -1)

    # Convert time given in minutes to days
    t_aerating = t_aerating / 60 / 24
    t_waiting = t_waiting / 60 / 24

    # Periodically turn on and off the aeration
    start_aerating = PeriodicCallback(
        (integrator) -> index_u == -1 ? integrator.u[end] = 1 : integrator.u[index_u] = 1,
        t_waiting + t_aerating, initial_affect = true, save_positions = (false, false))
    stop_aerating = PeriodicCallback(
        (integrator) -> index_u == -1 ? integrator.u[end] = 0 : integrator.u[index_u] = 0,
        t_waiting + t_aerating, phase = t_aerating,
        initial_affect = false, save_positions = (false, false))

    clock_callback = CallbackSet(start_aerating, stop_aerating)

    return clock_callback
end

"""
    $(TYPEDSIGNATURES)

Returns a ContinuousCallback that simulates the external control of a bioreactor given as a time array and a control array.
"""
function external_control(array_t, array_u; index_u = -1)

    # Create a function that returns the control value at a given time
    control = interpolate(
        (array_t,), vcat(array_u[2:end], array_u[end]), Gridded(Linear()))

    # Search for the change in the control value in the array_u
    event_times = [0.0]
    for i in 1:(length(array_u) - 1)
        if abs(array_u[i] - array_u[i + 1]) > 0.5
            push!(event_times, array_t[i])
        end
    end

    # Create a callback that changes the control value at the given times
    external_control_callback = PresetTimeCallback(event_times,
        (integrator) -> index_u == -1 ? integrator.u[end] = abs(control(integrator.t)) :
                        integrator.u[index_u] = abs(control(integrator.t)),
        save_positions = (false, false))

    return external_control_callback
end

"""
    $(TYPEDSIGNATURES)

Returns a CallbackSet that simulates the redox control of a bioreactor with additional real time constraints.
"""
function timed_redox_control(;
        index_no3 = 9,
        index_nh4 = 10,
        index_u = -1,
        t_initial = 0.0,
        min_aeration_time_minutes = 30.0,
        min_non_aeration_time_minutes = 30.0,
        max_aeration_time_minutes = 120.0,
        max_non_aeration_time_minutes = 120.0,
        NO_threshold = 0.5,
        NH_threshold = 0.5
)

    # Convert times from minutes to days
    min_aeration_time = min_aeration_time_minutes / (60.0 * 24.0)
    min_non_aeration_time = min_non_aeration_time_minutes / (60.0 * 24.0)
    max_aeration_time = max_aeration_time_minutes / (60.0 * 24.0)
    max_non_aeration_time = max_non_aeration_time_minutes / (60.0 * 24.0)

    # Stores the time of the last aeration state change
    t_last_aeration_change = Ref(t_initial)

    get_u_idx(uu) = (index_u == -1) ? lastindex(uu) : index_u

    function condition_aeration_on(u, t, integrator)
        idx = get_u_idx(u)
        current_aeration_state = u[idx]

        (current_aeration_state == 1) && return false # Already ON

        time_since_last_change = t - t_last_aeration_change[]
        s_no_low = u[index_no3] < NO_threshold

        return (time_since_last_change >= max_non_aeration_time) ||
               (s_no_low && (time_since_last_change >= min_non_aeration_time))
    end

    function affect_aeration_on!(integrator)
        idx = get_u_idx(integrator.u)
        integrator.u[idx] = 1
        t_last_aeration_change[] = integrator.t
    end

    callback_aeration_on = DiscreteCallback(
        condition_aeration_on, affect_aeration_on!; save_positions = (false, false))

    function condition_aeration_off(u, t, integrator)
        idx = get_u_idx(u)
        current_aeration_state = u[idx]

        (current_aeration_state == 0) && return false # Already OFF

        time_since_last_change = t - t_last_aeration_change[]
        s_nh_low = u[index_nh4] < NH_threshold

        return (time_since_last_change >= max_aeration_time) ||
               (s_nh_low && (time_since_last_change >= min_aeration_time))
    end

    function affect_aeration_off!(integrator)
        idx = get_u_idx(integrator.u)
        integrator.u[idx] = 0
        t_last_aeration_change[] = integrator.t
    end

    callback_aeration_off = DiscreteCallback(
        condition_aeration_off, affect_aeration_off!; save_positions = (false, false))

    return CallbackSet(callback_aeration_on, callback_aeration_off)
end
@inline timed_redox_control(s::Symbol) = timed_redox_control(
    index_no3 = get_no3_index(s), index_nh4 = get_nh4_index(s),
    index_u = get_control_index(s))