using Dates
using OrdinaryDiffEq

"""
$(TYPEDEF)

$(TYPEDFIELDS)

A mutable structure designed to simulate systems using ordinary differential equations (ODEs).
This struct contains all the necessary information for defining and solving an ODE system.
"""
mutable struct ODECore
    "The current time of the simulation."
    current_t::Real
    "The fixed timestep for simulation in days."
    fixed_dt::Real
    "The ODE function to simulate."
    ode_fct!::Function
    "The current state of the system."
    current_state::AbstractVector
    "Parameters used in the ODE function."
    parameters::NamedTuple
    "Indices of the state vector representing control inputs."
    index_u::AbstractVector{Int}

    """
    Constructor with full arguments.
    """
    function ODECore(current_t::Real, fixed_dt::Real, ode_fct!::Function,
            current_state::AbstractVector, parameters::NamedTuple, index_u::AbstractVector{Int})
        return new(current_t, fixed_dt, ode_fct!, current_state, parameters, index_u)
    end

    """
    Default constructor for initializing based on a specific system configuration.
    """
    function ODECore(system::Symbol; kwargs...)
        return new(set_ode_core(system; kwargs...)...)
    end
end

"""
    $(TYPEDSIGNATURES)

Performs a single time step simulation over an `ODECore` instance.
"""
function step!(env::ODECore, action::Union{AbstractVector, CallbackSet}; alg=nothing, kwargs...)

    # Copy the initial state to preserve the original
    init_state = deepcopy(env.current_state)

    # Apply control action if the action is a vector
    if isa(action, AbstractVector)
        @assert size(action, 1)==size(env.index_u, 1) "Action size must match the size of `index_u` in `env`."
        init_state[env.index_u] .= action
    end

    # Define the ODEProblem
    ode_problem = ODEProblem(
        env.ode_fct!, init_state, (env.current_t, env.current_t + env.fixed_dt), env.parameters)

    # Solve the ODEProblem with or without callbacks
    sim_results = isa(action, CallbackSet) ?
                  solve(ode_problem, alg; saveat = [env.current_t + env.fixed_dt],
        alg_hints = [:stiff], callback = action, kwargs...) :
                  solve(ode_problem, alg;
                   saveat = [env.current_t + env.fixed_dt], alg_hints = [:stiff], kwargs...)

    # Update the environment's state and time
    env.current_t += env.fixed_dt
    env.current_state = sim_results.u[1]

    return TSConcentrations([env.current_t], hcat(env.current_state...))
end

"""
    $(TYPEDSIGNATURES)

Performs a multi-step simulation over an `ODECore` instance.
"""
function multi_step!(env::ODECore, action::Union{Vector{<:AbstractVector}, CallbackSet},
        n_steps::Union{Int, Period}; alg=nothing, kwargs...)

    # Convert Period to number of steps if applicable
    n_steps = isa(n_steps, Period) ?
              Int(Dates.value(convert(Dates.Second, n_steps)) /
                  (env.fixed_dt * 24 * 60 * 60)) : n_steps

    # Handle vectorized control actions
    if isa(action, Vector{<:AbstractVector})
        @assert size(action, 1)==n_steps "The number of actions must match the number of steps."
        @assert all(size(a_t, 1) == size(env.index_u, 1) for a_t in action) "Each action must match the size of `index_u`."

        # Combine actions for external control
        action = external_control(
            [env.current_t + (i - 1) * env.fixed_dt for i in 1:n_steps],
            vcat(action...); index_u = env.index_u)
    end

    # Define the ODEProblem for multi-step simulation
    ode_problem = ODEProblem(env.ode_fct!, env.current_state,
        (env.current_t, env.current_t + env.fixed_dt * n_steps), env.parameters)

    # Solve the ODEProblem
    sim_results = solve(ode_problem, alg;
        saveat = [env.current_t + i * env.fixed_dt for i in 1:n_steps],
        alg_hints = [:stiff], callback = action, kwargs...)

    # Update the environment's state and time
    env.current_t += env.fixed_dt * n_steps
    env.current_state = sim_results.u[end]

    return TSConcentrations(sim_results.t, stack(sim_results.u, dims = 1))
end
