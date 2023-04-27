using LinearAlgebra, DifferentialEquations, Distributions, ProgressMeter, Interpolations

import AnalogDataAssimilation: TimeSeries


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

    if !isnothing(control)
        sim = solve(monte_prob; trajectories = np, save_everystep = false, callback = control)
    else
        sim = solve(monte_prob; trajectories = np, save_everystep = false, callback = control)
    end

    for i = 1:np
        xf[:, i] .= last(sim[i].u)
    end

    xf

end


function EnKF(da::DataAssimilation, yo::TimeSeries, np::Int64; progress = true, control = nothing)

    # dimensions
    nt = yo.nt        # number of observations
    nv = yo.nv        # number of variables (dimensions of problem)

    # initialization
    x̂ = TimeSeries(nt, nv)

    xa = [zeros(Float64, (nv, np)) for i = 1:nt]
    pf = [zeros(Float64, (nv, nv)) for i = 1:nt]
    xf = [zeros(Float64, (nv, np)) for i = 1:nt]
    ef = similar(xf[1])

    if progress
        p = Progress(nt)
    end

    for k = 1:nt

        if progress
            next!(p)
        end

        # update step (compute forecasts)            
        if k == 1
            xf[k] .= max.(da.xb .+ da.B.^(1/2)*rand(Normal(), nv, np), 0)
        else
            xf[k] = max.(da.m(xa[k-1], yo.t[k-1]; control = control) + da.m.Σ_model.^(1/2).*rand(Normal(), nv, np), 0)
        end

        ef .= xf[k] * (Matrix(I, np, np) .- 1 / np)
        pf[k] .= (ef * ef') ./ (np - 1)

        # analysis step (correct forecasts with observations)          
        ivar_obs = findall(.!isnan.(yo.u[k]))
        n = length(ivar_obs)

        if n > 0
            μ = zeros(Float64, n)
            σ = da.R[ivar_obs, ivar_obs]
            yf = max.(da.H[ivar_obs, :] * xf[k] .+ σ.^(1/2)*rand(Normal(), n, np) .+ μ, 0)
            Σ = (da.H[ivar_obs, :] * pf[k]) * da.H[ivar_obs, :]'
            Σ .+= da.R[ivar_obs, ivar_obs]
            invΣ = inv(Σ)
            K = (pf[k] * da.H[ivar_obs, :]') * invΣ
            d = yo.u[k][ivar_obs] .- yf
            xa[k] .= xf[k] .+ K * d
            # compute likelihood
            # innov_ll = mean(yo.u[k][ivar_obs] .- yf, dims = 2)
        else
            xa[k] .= xf[k]
        end

        x̂.u[k] .= vec(sum(xa[k] ./ np, dims = 2))

    end

    x̂

end


function PF(da::DataAssimilation, yo::TimeSeries, np::Int64; progress = true, control = nothing)

    # dimensions
    nt = yo.nt        # number of observations
    nv = yo.nv        # number of variables (dimensions of problem)

    # initialization
    x̂ = TimeSeries(nt, nv)

    # special case for k=1
    k = 1
    m_xa_traj = Array{Float64,2}[]
    xf = max.(da.xb .+ da.B.^(1/2)*rand(Normal(), nv, np), 0)
    ivar_obs = findall(.!isnan.(yo.u[k]))
    nobs = length(ivar_obs)
    weights = zeros(Float64, np)
    indic = zeros(Int64, np)
    part = [zeros(Float64, (nv, np)) for i = 1:nt]

    if nobs > 0

        for ip = 1:np
            μ = vec(da.H[ivar_obs, :] * xf[:, ip])
            σ = Matrix(da.R[ivar_obs, ivar_obs])
            d = MvNormal(μ, σ)
            weights[ip] = pdf(d, yo.u[k][ivar_obs])
        end
        # normalization
        weights ./= sum(weights)
        # resampling
        resample!(indic, weights)
        part[k] .= xf[:, indic]
        weights .= weights[indic] ./ sum(weights[indic])
        x̂.u[k] .= vec(sum(part[k] .* weights', dims = 2))

        # find number of iterations before new observation
        # todo: try the findnext function
        # findnext(.!isnan.(vcat(yo.u'...)), k+1)
        knext = 1
        while knext + k <= nt && all(isnan.(yo.u[k+knext]))
            knext += 1
        end

    else

        weights .= 1.0 / np # weights
        resample!(indic, weights) # resampling

    end

    kcount = 1

    if progress
        p = Progress(nt)
    end

    for k = 2:nt
        if progress
            next!(p)
        end

        # update step (compute forecasts) and add small Gaussian noise
        xf = max.(da.m(part[k-1], yo.t[k-1]; control = control) .+ da.m.Σ_model.^(1/2).*rand(Normal(), nv, np), 0)
        if kcount <= length(m_xa_traj)
            m_xa_traj[kcount] .= xf
        else
            push!(m_xa_traj, xf)
        end
        kcount += 1

        # analysis step (correct forecasts with observations)
        ivar_obs = findall(.!isnan.(yo.u[k]))

        if length(ivar_obs) > 0
            # weights
            σ = Symmetric(da.R[ivar_obs, ivar_obs])
            for ip = 1:np
                μ = vec(da.H[ivar_obs, :] * xf[:, ip])
                d = MvNormal(μ, σ)
                weights[ip] = pdf(d, yo.u[k][ivar_obs])
            end
            # normalization
            weights ./= sum(weights)
            # resampling
            resample!(indic, weights)
            weights .= weights[indic] ./ sum(weights[indic])
            # stock results
            for j = 1:knext
                jm = k - knext + j
                for ip = 1:np
                    part[jm][:, ip] .= m_xa_traj[j][:, indic[ip]]
                end
                x̂.u[jm] .= vec(sum(part[jm] .* weights', dims = 2))
            end
            kcount = 1
            # find number of iterations  before new observation
            knext = 1
            while knext + k <= nt && all(isnan.(yo.u[k+knext]))
                knext += 1
            end
        else
            # stock results
            part[k] .= xf
            x̂.u[k] .= vec(sum(xf .* weights', dims = 2))
        end

    end

    x̂

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