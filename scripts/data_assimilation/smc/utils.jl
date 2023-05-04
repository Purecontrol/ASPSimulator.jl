using LinearAlgebra, DifferentialEquations, Distributions, ProgressMeter, Interpolations

include("time_series.jl")


mutable struct StateSpaceModel

    model::Function
    dt_integration::Float64
    dt_states::Int64
    dt_obs::Int64
    params::Vector{Any}
    var_obs::Vector{Int64}
    Σ_model::Union{Vector{Float64}, Float64}
    Σ_obs::Union{Vector{Float64}, Float64}
    SNR_model::Float64
    SNR_obs::Float64

    # Constructor
    function StateSpaceModel(model::Function, dt_integration::Float64, dt_states::Int64, dt_obs::Int64, params::Vector{Any}, var_obs::Vector{Int64}; Σ_model::Union{Vector{Float64}, Float64} = -1.0, Σ_obs::Union{Vector{Float64}, Float64} = -1.0, SNR_model::Float64 = -1.0, SNR_obs::Float64 = -1.0)

        # Check if the parameters are correct
        try 
            @assert Σ_model != -1 || SNR_model != -1
        catch
            @error "Σ_model or SNR_model must be defined"
        end

        try 
            @assert Σ_obs != -1 || SNR_obs != -1
        catch
            @error "Σ_obs or SNR_obs must be defined"
        end
        
        new(model, dt_integration, dt_states, dt_obs, params, var_obs, Σ_model, Σ_obs, SNR_model, SNR_obs)

    end

end


function generate_data(ssm::StateSpaceModel, u0::Vector{Float64}, T::Float64, seed = 42; control = nothing)

    rng = MersenneTwister(seed)

    try
        @assert ssm.dt_states <= ssm.dt_obs
    catch
        @error " ssm.dt_obs must be bigger or equal to ssm.dt_states"
    end

    try
        @assert mod(ssm.dt_obs, ssm.dt_states) == 0.0
    catch
        @error " ssm.dt_obs must be a multiple of ssm.dt_states "
    end

    ################################################################
    ### Solve the ODE to generate the true state and get control ###
    ################################################################

    # Define the parameters
    tspan = (0,T)
    u0_prob = u0

    # Define ODE problem
    prob = ODEProblem(ssm.model, u0_prob, tspan, ssm.params)

    # Solve the problem
    if isnothing(control)
        sol_true = solve(prob; saveat = ssm.dt_integration)
    else
        sol_true = solve(prob; saveat = ssm.dt_integration, callback=control)
    end
    xt = TimeSeries(sol_true.t[1:ssm.dt_states:end], sol_true.u[1:ssm.dt_states:end])

    # Get the control vector of the system
    U = ASM1Simulator.Models.external_control(sol_true.t, getindex.(sol_true.u, 14))

    ########################################################
    ### Compute Σ_model and Σ_obs if not defined by user ###
    ########################################################

    if ssm.Σ_model == -1.0
        ssm.Σ_model = (var(xt.u) + (mean(xt.u).^2))./ssm.SNR_model
        ssm.Σ_model[end] = 0.0
    end

    if ssm.Σ_obs == -1.0
        ssm.Σ_obs = (var(xt.u) + (mean(xt.u).^2))./ssm.SNR_obs
    end

    #######################################################
    ### Solve the ODE to generate the approximate model ###
    #######################################################

    x̂t = TimeSeries(xt.t, xt.u .* NaN)
    x̂t.u[1] = u0
    for i in 2:length(x̂t.t)

        # Create parameters problem
        u0_prob = vcat(x̂t.u'...)[i-1,:]
        tspan = (x̂t.t[i-1], x̂t.t[i])

        # Define ODE problem
        prob = ODEProblem(ssm.model, u0_prob, tspan, ssm.params)

        # Solve the problem
        if isnothing(control)
            sol_model = solve(prob; save_everystep = false)
        else
            sol_model = solve(prob; save_everystep = false, callback=U)
        end

        # Get the state
        x̂t.u[i] .= max.(last(sol_model.u) + rand(rng, Normal(), x̂t.nv).*ssm.Σ_model.^(1/2), 10e-9)
    end

    #################################################
    ### generate  partial/noisy observations (ŷt) ###
    #################################################

    nt = xt.nt
    nv = xt.nv

    ŷt = TimeSeries(xt.t, xt.u .* NaN)
    step = ssm.dt_obs ÷ ssm.dt_states
    nt = length(xt.t)

    for j = 1:step:nt
        ŷt.u[j][ssm.var_obs] .= max.(xt.u[j][ssm.var_obs] .+ rand(rng, Normal(), size(ssm.var_obs, 1)).*ssm.Σ_obs[ssm.var_obs].^(1/2), 10e-9)
    end

    xt, x̂t, ŷt, U

end


struct DataAssimilation

    xb::Vector{Float64}
    B::Array{Float64,2}
    H::Array{Bool,2}
    R::Array{Float64,2}
    m::StateSpaceModel

    function DataAssimilation(m::StateSpaceModel, xt::TimeSeries)

        xb = xt.u[1]
        B = m.Σ_model .* Matrix(I, xt.nv, xt.nv)
        H = Matrix(I, xt.nv, xt.nv)
        R = m.Σ_obs .* H

        new(xb, B, H, R, m)

    end

end


function (ssm::StateSpaceModel)(x::Array{Float64,2}, T_start::Float64; control = nothing)

    nv, np = size(x)
    xf = similar(x)
    p = ssm.params
    tspan = (T_start + 0.0, T_start +  ssm.dt_states * ssm.dt_integration)
    u0 = zeros(Float64, nv)

    prob = ODEProblem(ssm.model, u0, tspan, p)

    function prob_func(prob, i, repeat)
        remake(prob, u0 = vec(x[:, i]))
    end

    monte_prob = EnsembleProblem(prob, prob_func = prob_func)

    if isnothing(control)
        sim = solve(monte_prob; trajectories = np, save_everystep = false)
    else
        sim = solve(monte_prob; trajectories = np, save_everystep = false, callback = control)
    end

    for i = 1:np
        xf[:, i] .= last(sim[i].u)
    end

    xf

end


function resample_multinomial(w::Vector{Float64})

    m = length(w)
    q = cumsum(w)
    q[end] = 1.0 # Just in case...
    i = 1
    indx = Int64[]
    while i <= m
        sampl = rand()
        j = 1
        while q[j] < sampl
            j = j + 1
        end
        push!(indx, j)
        i = i + 1
    end
    indx
end


""" 
    resample!( indx, w )

Multinomial resampler.
"""
function resample!(indx::Vector{Int64}, w::Vector{Float64})

    m = length(w)
    q = cumsum(w)
    i = 1
    while i <= m
        sampl = rand()
        j = 1
        while q[j] < sampl
            j = j + 1
        end
        indx[i] = j
        i = i + 1
    end
end