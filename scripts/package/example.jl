using Plots

include("time_series.jl")
include("systems/base.jl")
include("models.jl")
include("systems/gaussian_linear_state_space.jl")
include("filters/base.jl")
include("smoothers/base.jl")
include("filter_smoother.jl")
include("filters/kalman_filter.jl")
include("filters/ensemble_kalman_filter.jl")
include("filters/particle_filter.jl")
include("smoothers/kalman_smoother.jl")
include("smoothers/ensemble_kalman_smoother.jl")
include("fit.jl")

function A_t(exogenous, params, t)

    # Get Q_in and V
    Q_in = exogenous[1]
    V = params[1]

    # Define A
    A = hcat([[1 - Q_in(t)/params[1]*(1/1440)]]...)

    return A

end

function B_t(exogenous, params, t)

    # β
    β = params[2]

    # Define B
    B = hcat([[- β*(1/1440)]]...)

    return B

end

H_t(exogenous, params, t) = Matrix{Float64}(I, 1, 1)

function R_t(exogenous, params, t)

    # Get params
    η = exp(params[3])

    # Define R
    R = hcat([[η]]...)

    return R

end

function Q_t(exogenous, params, t)

    # Get params
    ϵ = exp(params[4])

    # Define Q
    Q = hcat([[ϵ]]...)

    return Q

end

function c_t(exogenous, params, t)

    # Get Q_in, X_in and β
    Q_in = exogenous[1]
    X_in = exogenous[2]
    V = params[1]

    # Define B
    c = [(X_in(t)*Q_in(t))/V*(1/1440)]

    return c

end

function d_t(exogenous, params, t)

    # Define d
    d = zeros(1)

    return d

end

# Define the system
n_X = 1
n_Y = 1
glss = GaussianLinearStateSpaceSystem(A_t, B_t, c_t, H_t, d_t, R_t, Q_t, n_X, n_Y, sim.fixed_dt)

# Define init state
init_P_0 = zeros(1, 1) .+ 0.001
init_state = GaussianStateStochasticProcess(T_steady_state, [x_train[10,1]], init_P_0)

# Define model
parameters = [1333.0, 200.0, -2.30, -2.30]
model = ForecastingModel{GaussianLinearStateSpaceSystem}(glss, init_state, parameters)

# Set exogenous predictors
Q_in = sim.params_vec[5]
function X_in(t)
    return sim.params_vec[4][10]
end
exogenous_variables = [Q_in, X_in]

# Optimize model with numerical maximization of likelihood
y_t = transpose(y_train)
@timed sol = numerical_MLE(model, y_t, exogenous_variables, U_train)
opt_params_nmle = sol.minimizer

# Optimize model with EM
@timed optim_params_em = EM(model, y_t, exogenous_variables, U_train)

@timed optim_params_em = speed_EM(model, y_t, exogenous_variables, U_train)

# Optimize model with EnKS EM
optim_params_enks_em = EM_EnKS(model, y_t, exogenous_variables, U_train; n_particles = 300)

println("--------------------------------------------------------------------")
println("----------------------- OPTIMIZATION RESULTS -----------------------")
println("--------------------------------------------------------------------\n")


println("   V    | Estimated NMLE = ", round(opt_params_nmle[1], digits=3), " | Estimated EM = ", round(optim_params_em[1], digits=3), " | Estimated EnKS-EM = ", round(optim_params_enks_em[1], digits=3))
println("   β    | Estimated NMLE = ", round(opt_params_nmle[2], digits=3), " | Estimated EM = ", round(optim_params_em[2], digits=3), " | Estimated EnKS-EM = ", round(optim_params_enks_em[2], digits=3))
println("σ_model | Estimated NMLE = ", round(sqrt(exp(opt_params_nmle[3])), digits=3), " | Estimated EM = ", round(sqrt(exp(optim_params_em[3])), digits=3), " | Estimated EnKS-EM = ", round(sqrt(exp(optim_params_enks_em[3])), digits=3)," | Real = ", 0.0)
println("σ_obs   | Estimated NMLE = ", round(sqrt(exp(opt_params_nmle[4])), digits=3), " | Estimated EM = ", round(sqrt(exp(optim_params_em[4])), digits=3), " | Estimated EnKS-EM = ", round(sqrt(exp(optim_params_enks_em[4])), digits=3)," | Real = ", σ_ϵ)

println("\n--------------------------------------------------------------------")
println("--------------------------------------------------------------------")
println("--------------------------------------------------------------------")


# Kalman filtering on y_t
model.parameters = opt_params_nmle
@timed filter_output = filter(model, y_t, exogenous_variables, U_train)

# Kalman smoothing on y_t ( 2 possible constructors )
# smoother_output = smoother(model, y_t, exogenous_variables, U_train)
@timed smoother_output = smoother(model, y_t, exogenous_variables, U_train, filter_output)

# Multistep ahead forecast of the model
model_state, predicted_obs = forecast(model, exogenous_variables, U_train; n_steps_ahead=1440)

# Plot timeseries
plot(model_state, label="predictor NH4")
plot!(smoother_output.smoothed_state, label="smoothed predictor NH4")
plot!(filter_output.predicted_state, label="test NH4")
plot!((H*x_train)', label = "true NH4")
scatter!(y_t, markersize=0.5, label = "Observations")


################################################
################ ParticleFilter ################
################################################



##### TEST PF-AS #####

# AT : Ancestor-Tracking
# AS : Ancestor-Sampling CPF (filter with conditionning particle) -> AT AS
# PF -> AT, BS CPF -> AT or AS, BS


function sample_discrete(prob, n_particules)

    # this speedup is due to Peter Acklam
    cumprob = cumsum(prob)
    N = size(cumprob, 1)
    R = rand(n_particules)

    ind = ones(Int64, (n_particules))
    for i = 1:N-1
        ind .+= R .> cumprob[i]
    end
    ind

end


n_obs = size(y_t, 1)
function PF_AT(sampling_weight, predicted_particles_swarm, ancestor_indices, n_smoothing)

    t_index = [model.current_state.t + (model.system.dt)*(t-1) for t in 1:(n_obs+1)]
    smoothed_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs+1, 1, t_index; n_particles=n_smoothing)

    Xs = zeros(Float64, n_obs+1, n_X, n_smoothing)
    
    ind_smoothing = sample_discrete(sampling_weight, n_smoothing)

    Xs[end, :, :] .= predicted_particles_swarm[end].particles_state[:, ind_smoothing]
    smoothed_particles_swarm[end].particles_state = Xs[end, :, :]

    @inbounds for t in (n_obs):-1:1

        ind_smoothing = ancestor_indices[t, ind_smoothing]

        Xs[t, :, :] .= predicted_particles_swarm[t].particles_state[:, ind_smoothing]

        smoothed_particles_swarm[t].particles_state = Xs[t, :, :]

    end

    return smoothed_particles_swarm

end


function PF_BS(sampling_weight, predicted_particles_swarm, predicted_particles_swarm_mean, n_smoothing, n_filtering, parameters)

    t_index = [model.current_state.t + (model.system.dt)*(t-1) for t in 1:(n_obs+1)]
    smoothed_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs+1, 1, t_index; n_particles=n_smoothing)

    Xs = zeros(Float64, n_obs+1, n_X, n_smoothing)
    
    ind_smoothing = sample_discrete((1/n_filtering).*ones(n_filtering), n_smoothing)

    Xs[end, :, :] .= predicted_particles_swarm[end].particles_state[:, ind_smoothing]
    smoothed_particles_swarm[end].particles_state = Xs[end, :, :]

    @inbounds for t in (n_obs):-1:1

        # println("Ite $t")

        # Get current t_step
        t_step = init_state.t + (t-1)*glss.dt

        σ = Matrix(glss.R_t(exogenous_variables, parameters, t_step))

        C = ((2*pi)^(-n_X/2))*(det(σ)^(-1/2))
        for i in 1:n_smoothing

            v = Xs[t+1, :, i] .- predicted_particles_swarm_mean[t+1, :, :]
            smoothing_weight = dropdims(C.*exp.(-(1/2)*v'*pinv(σ).*v').*sampling_weight[t+1, :], dims=2) # sampling_weight[t, :] or sampling_weight[t+1, :]

            smoothing_weight ./= sum(smoothing_weight) 

            ind_smoothing = sample_discrete(smoothing_weight, 1)

            Xs[t, :, i] .= predicted_particles_swarm[t].particles_state[:, ind_smoothing]

        end

        smoothed_particles_swarm[t].particles_state = Xs[t, :, :]

    end

    return smoothed_particles_swarm

end


function SEM(model::ForecastingModel{GaussianLinearStateSpaceSystem}, y_t, exogenous_variables, control_variables; n_particles=30)

    # Fixed values
    n_obs = size(y_t, 1)
    t_start = model.current_state.t
    dt = model.system.dt

    # Q function
    function Q(parameters, smoothed_values)

        L = 0
        for t in 1:n_obs

            # Get current t_step
            t_step = t_start + (t-1)*dt

            ivar_obs = findall(.!isnan.(y_t[t, :]))

            R_i = model.system.R_t(exogenous_variables, parameters, t_step)
            Q_i = model.system.Q_t(exogenous_variables, parameters, t_step)

            M_i = transition(model.system, smoothed_values[t].particles_state, exogenous_variables, control_variables[t, :], parameters, t_step)
            H_i = observation(model.system, smoothed_values[t].particles_state, exogenous_variables, parameters, t_step)[ivar_obs, :]

            ϵ_i = y_t[t, ivar_obs] .- H_i
            Σ = (ϵ_i*ϵ_i') ./ (n_particles - 1)
    
            η_i = smoothed_values[t+1].particles_state - M_i
            Ω = (η_i*η_i') ./ (n_particles - 1)

            if size(ivar_obs, 1) > 0
                L += -(1/2)*(log(2*pi) + sum(log(det(Q_i[ivar_obs, ivar_obs]))) +  tr(Σ*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L += -(1/2)*(log(2*pi) + sum(log(det(R_i))) + tr(Ω*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # M function
    optprob = OptimizationFunction(Q, Optimization.AutoForwardDiff())

    llk_array = []
    parameters = model.parameters
    for i in 1:50

        filter_output , filtered_state, filtered_state_var= filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=ParticleFilter(model.current_state, model.system.n_X, model.system.n_Y, n_particles))

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        smoothed_particles_swarm = PF_BS(filter_output.sampling_weights, filter_output.predicted_particles_swarm, filter_output.predicted_particles_swarm_mean, n_particles, n_particles, parameters)

        prob = Optimization.OptimizationProblem(optprob, parameters, smoothed_particles_swarm)
        sol = solve(prob, Optim.Newton(), maxiters = 20)
        parameters = sol.minimizer
    
    end

    filter_output , filtered_state, filtered_state_var = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=ParticleFilter(model.current_state, model.system.n_X, model.system.n_Y, n_particles))

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters

end


model.parameters = [1333.0, 200.0, -2.30, -2.30]
test = SEM(model, y_t, exogenous_variables, U_train; n_particles = 300)




################################################
################ Comparison methods ################
################################################

n_particles = 1000
filter_output = filter(model, y_t, exogenous_variables, U_train)
filter_output_pf, filtered_state, filtered_state_var = filter(model, y_t, exogenous_variables, U_train; filter=ParticleFilter(model.current_state, 1, 1, n_particles))
filter_output_enkf = filter(model, y_t, exogenous_variables, U_train; filter=EnsembleKalmanFilter(model.current_state, model.system.n_X, model.system.n_Y, n_particles))

# Predicted values
plot(filter_output_enkf.predicted_particles_swarm, label = "ENKF")
plot!(filter_output_pf.predicted_particles_swarm, label = "PF")
plot!(filter_output.predicted_state, label = "KF")
plot!(title="Predicted values")

# Filtered values
plot(filter_output_enkf.filtered_particles_swarm, label = "EnKF")
plot!(filter_output_pf.filtered_particles_swarm, label = "PF")
plot!(filter_output.filtered_state, label = "KF")
plot!(title="Filtered values")


n_smoothing = 200
smoother_output = smoother(model, y_t, exogenous_variables, U_train, filter_output)
smoother_output_pf = PF_BS(filter_output_pf.sampling_weights, filter_output_pf.predicted_particles_swarm, filter_output_pf.predicted_particles_swarm_mean, n_smoothing, n_particles, model.parameters)
smoother_output_enkf = smoother(model, y_t, exogenous_variables, U_train, filter_output_enkf; smoother_method=EnsembleKalmanSmoother(model.system.n_X, model.system.n_Y, n_particles))

# Smoothed values
plot(smoother_output_enkf.smoothed_state, label="EnKF")
plot!(smoother_output_pf, label="PF-BS")
plot!(smoother_output.smoothed_state, label="KF")
plot!(title="Smoothing")