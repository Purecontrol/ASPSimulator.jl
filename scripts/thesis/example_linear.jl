using Plots
using StateSpaceIdentification
using SparseArrays
using Optim, OptimizationNLopt
using PDMats

function A_t(exogenous, params)

    # Get Q_in and V
    Q_in = exogenous[1]
    V = params[1]

    # Define A
    A = sparse(hcat([[1 - Q_in/params[1]*(dt_model/1440)]]...))

    return A

end

function B_t(exogenous, params)

    # β
    β = params[2]

    # Define B
    B = sparse(hcat([[- β*(dt_model/1440)]]...))

    return B

end

H_t(exogenous, params) = sparse(Matrix{Float64}(I, 1, 1))

function R_t(exogenous, params)

    # Get params
    η = exp(params[3])

    # Define R
    # R = hcat([[η]]...)

    return PDiagMat([η])

end

function Q_t(exogenous, params)

    # Get params
    ϵ = exp(params[4])

    # Define Q
    # Q = hcat([[ϵ]]...)

    return PDiagMat([ϵ])

end

function c_t(exogenous, params)

    # Get Q_in, X_in and β
    Q_in = exogenous[1]
    X_in = exogenous[2]
    V = params[1]

    # Define B
    c = sparse([(X_in*Q_in)/V*(dt_model/1440)])

    return c

end

function d_t(exogenous, params)

    # Define d
    d = sparse(zeros(1))

    return d

end

# Define the system
n_X = 1
n_Y = 1
glss = GaussianLinearStateSpaceSystem(A_t, B_t, c_t, H_t, d_t, R_t, Q_t, n_X, n_Y, dt_model/1440)

# Define init state
init_P_0 = zeros(1, 1) .+   1 #0.001
init_state = GaussianStateStochasticProcess(T_steady_state, [x_train[10,1]], init_P_0)

# Define model
parameters = [1333.0, 200.0, -2.30, -2.30]
model = ForecastingModel{GaussianLinearStateSpaceSystem}(glss, init_state, parameters)

# Set exogenous predictors
Q_in = sim.params_vec.Q_in
X_in = sim.params_vec.X_in[10]

Q_in_t = [Q_in(T_steady_state + t*dt_model/(24*60)) for t in 1:size(y_train, 1)]
X_in_t = [X_in for t in 1:size(y_train, 1)]
E_train = hcat([Q_in_t, X_in_t]...)

# Optimize model with numerical maximization of likelihood
sol = numerical_MLE(model, y_train, E_train, U_train, optim_method=Opt(:LD_TNEWTON, 4), verbose=true)
opt_params_nmle = sol.minimizer

# Optimize model with EM
optim_params_em = StateSpaceIdentification.EM(model, y_train, E_train, U_train, maxiters_em=200, maxiters=10, optim_method=Opt(:LD_LBFGS, 4))

# Optimize with EM using approximate EnKS smoother
optim_params_enks_em, results_enks = EM_EnKS(model, y_train, E_train, U_train; n_particles = 100, maxiters_em=50, reltol_em=1e-4, maxiters=50, optim_method=Opt(:LD_LBFGS, 4));

# Optimize with EM using approximate Backward Smoothing smoother
optim_params_pfbs_em, results_pfbs = SEM(model, y_train, E_train, U_train; n_filtering = 500, n_smoothing = 500, maxiters_em=50, reltol_em=1e-4, maxiters=50, optim_method=Opt(:LD_LBFGS, 4))


println("--------------------------------------------------------------------")
println("----------------------- OPTIMIZATION RESULTS -----------------------")
println("--------------------------------------------------------------------\n")


println("   V    | Estimated NMLE = ", round(opt_params_nmle[1], digits=3), " | Estimated EM = ", round(optim_params_em[1], digits=3), " | Estimated EnKS-EM = ", round(optim_params_enks_em[1], digits=3), " | Estimated PFBS-EM = ", round(optim_params_pfbs_em[1], digits=3))
println("   β    | Estimated NMLE = ", round(opt_params_nmle[2], digits=3), " | Estimated EM = ", round(optim_params_em[2], digits=3), " | Estimated EnKS-EM = ", round(optim_params_enks_em[2], digits=3), " | Estimated PFBS-EM = ", round(optim_params_pfbs_em[2], digits=3))
println("σ_model | Estimated NMLE = ", round(sqrt(exp(opt_params_nmle[3])), digits=3), " | Estimated EM = ", round(sqrt(exp(optim_params_em[3])), digits=3), " | Estimated EnKS-EM = ", round(sqrt(exp(optim_params_enks_em[3])), digits=3), " | Estimated PFBS-EM = ", round(sqrt(exp(optim_params_pfbs_em[3])), digits=3)," | Real = ", 0.0)
println("σ_obs   | Estimated NMLE = ", round(sqrt(exp(opt_params_nmle[4])), digits=3), " | Estimated EM = ", round(sqrt(exp(optim_params_em[4])), digits=3), " | Estimated EnKS-EM = ", round(sqrt(exp(optim_params_enks_em[4])), digits=3), " | Estimated PFBS-EM = ", round(sqrt(exp(optim_params_pfbs_em[4])), digits=3)," | Real = ", σ_ϵ)

println("\n--------------------------------------------------------------------")
println("--------------------------------------------------------------------")
println("--------------------------------------------------------------------")


# Kalman filtering on y_t
model.parameters = opt_params_nmle
@timed filter_output = StateSpaceIdentification.filter(model, y_train, E_train, U_train)

# Kalman smoothing on y_t ( 2 possible constructors )
# smoother_output = smoother(model, y_t, exogenous_variables, U_train)
@timed smoother_output = smoother(model, y_train, E_train, U_train, filter_output)

# Multistep ahead forecast of the model
model_state, predicted_obs = forecast(model, E_train, U_train; n_steps_ahead=Int(T_training*1440/dt_model))

# Plot timeseries
# plot(model_state, label="predictor NH4")
plot(size=(900, 500))
plot!(smoother_output.smoothed_state, label="smoothed predictor NH4")
plot!(filter_output.predicted_state, label="filtered NH4")
plot!(T_steady_state:(1/1439.5):(T_steady_state+T_training), (H*x_train)', label = "true NH4")
scatter!(T_steady_state:(dt_model/1439):(T_steady_state+T_training), y_train, markersize=1.5, label = "Observations")



#########################################################
################ Analysis ParticleFilter ################
#########################################################

# Filtering Particle Filter
n_filtering = 50
filter_output_pf, _, _ = StateSpaceIdentification.filter(model, y_train, E_train, U_train; filter=ParticleFilter(model, n_particles=n_filtering))


filter_output_cpf, _, _ = StateSpaceIdentification.filter(model, y_train, E_train, U_train; filter=ConditionalParticleFilter(model, n_particles=n_filtering, conditional_particle=Xcond))

# Smoothing 2
n_filtering = 1000
n_smoothing = 300
filter_output_pf, _, _ = StateSpaceIdentification.filter(model, y_train, E_train, U_train; filter=ParticleFilter(model, n_particles=n_filtering))
smoother_output_bs1 = backward_smoothing(y_train, E_train, filter_output_pf, model, model.parameters; n_smoothing=n_smoothing)



# Smoothing 1
n_filtering = 300
n_smoothing = 300
Xcond = zeros(Float64, 289, 1)
smoother_output_bs2 = nothing
for i in 1:10
    filter_output_cpf, _, _ = StateSpaceIdentification.filter(model, y_train, E_train, U_train; filter=ConditionalParticleFilter(model, n_particles=n_filtering, conditional_particle=Xcond))
    # smoother_output_at = ancestor_tracking_smoothing(y_train, E_train, filter_output_cpf, model, model.parameters; n_smoothing=n_smoothing)
    smoother_output_bs2 = backward_smoothing(y_train, E_train, filter_output_cpf, model, model.parameters; n_smoothing=n_smoothing)
    Xs = vcat([i.particles_state for i in smoother_output_at.state]...)
    Xcond = Xs[:, end]
end


plot(smoother_output_bs1, label=["PF-BS"])
plot!(smoother_output_bs2, label=["CPF-BS"])
plot!(smoother_output.smoothed_state, label="Kalman Filter")
plot!(title="Comparison smoothing distribution")



plot(filter_output_pf.filtered_particles_swarm, label=["Particle Filter"])
plot!(filter_output_cpf.filtered_particles_swarm, label=["Conditional Particle Filter"])

plot!(filter_output.filtered_state, label="Kalman Filter")
plot!(title="Comparison filtered distribution")

plot(filter_output_pf.predicted_particles_swarm, label=["Particle Filter"])
plot!(filter_output.predicted_state, label="Kalman Filter")
plot!(title="Comparison predicted distribution")

# Smoothing particle filter
n_smoothing = 300
smoother_output_bs = backward_smoothing(y_train, E_train, filter_output_pf, model, model.parameters; n_smoothing=n_smoothing)
smoother_output_at = ancestor_tracking_smoothing(y_train, E_train, filter_output_pf, model, model.parameters; n_smoothing=n_smoothing)

plot(smoother_output_bs, label=["Backward Smoothing"])
plot!(smoother_output_at, label=["Ancestor Tracking"])
plot!(smoother_output.smoothed_state, label="Kalman Filter")
plot!(title="Comparison smoothing distribution")