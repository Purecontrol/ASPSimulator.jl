using Plots
using StateSpaceIdentification
using SparseArrays
using DifferentialEquations.EnsembleAnalysis
using Optim, OptimizationNLopt, PDMats

influent_file_path = "/home/victor/Documents/code/asm1-simulator/data/external/influent_files/dryinfluent.ascii"
whitebox_params, _ = ASM1Simulator.Models.get_default_parameters_simplified_asm1(influent_file_path=influent_file_path)

# Initialize greybox problem
greybox_problem = ODEProblem(ASM1Simulator.Models.simplified_asm1_opt!, zeros(6),  (0, 1), whitebox_params)

# de = modelingtoolkitize(greybox_problem)
# greybox_problem = ODEProblem(de, jac=true)


# x = filter_output.predicted_particles_swarm[1].particles_state
# x_bis = x[:, 1:2]
# u = U_train[1, :]
# e = E_train[1, :]

# @benchmark M_t(x, e, u, params)

dt_euler = 5/(60*24*60)
function M_t(x, exogenous, u, params)

    params = max.(params, 0.0001)

    # ode_params = vcat(params[1:19], exogenous[1], [exogenous[2:6]])
    ode_params = vcat(params[1:19], exogenous[1:6])

    problem_ite = remake(greybox_problem, u0 = zeros(6), tspan=(exogenous[7], exogenous[7] + exogenous[8]), p=ode_params)

    n_particules = size(x, 2)
    states = vcat(x, repeat(u, n_particules)')

    function prob_func(prob, i, repeat)
        remake(prob, u0 = states[:, i])
    end
    monte_prob = EnsembleProblem(problem_ite, prob_func = prob_func)

    sim_results = solve(monte_prob, AutoTsit5(Rosenbrock23()), trajectories = n_particules, saveat=[exogenous[7] + exogenous[8]], maxiters=10e4)
    # sim_results = solve(monte_prob, Euler(), dt = dt_euler, trajectories = n_particules, saveat=[exogenous[7] + exogenous[8]], maxiters=10e3)

    return hcat([max.(sim_results[i].u[1][1:5], 0.0) for i in 1:n_particules]...) #(hcat(componentwise_vectors_timestep(sim, 1)...)[:, 1:5])' #

end

function H_t(x, exogenous, params)

    #return Matrix([0 0 0 1 0])*x

    return sparse(Matrix([0 0 0 1 0 ;])*x)


    # Matrix([0 1 0 0 0 ; 0 0 1 0 0  ; 0 0 0 1 0 ])

end

function R_t(exogenous, params)

    return PDiagMat(exp.(params[20:24]))

end

function Q_t(exogenous, params)

    return PDiagMat([exp(params[25])])

end

# Params
model_params = convert(Array{Float64}, collect(whitebox_params)[1:19])
cov_params = [-5.0, -5.0, -5.0, -5.0, -5.0, -0.6]
params = vcat([model_params, cov_params]...)

# Model
gnlss = GaussianNonLinearStateSpaceSystem(M_t, H_t, R_t, Q_t, 5, nb_obs_var, dt_model/(1440))
init_P_0 = Matrix(I(5).*0.001) #.+ 0.1
x_init = [x_train[2, 1] + x_train[4, 1], x_train[8, 1], x_train[9, 1], x_train[10, 1], x_train[11, 1]]
init_state = GaussianStateStochasticProcess(T_steady_state, x_init, init_P_0)
model2 = ForecastingModel(gnlss, init_state, params)

# Exogenous variables
Q_in = whitebox_params.Q_in
Q_in_vec = repeat([Q_in(T_steady_state + dt_model/(1440)*(t-1)) for t in 1:Int(T_training*1440/dt_model)], 1, 1)
X_in_vec = repeat(whitebox_params[end], 1, Int(T_training*1440/dt_model))'
T_start_vec = repeat([T_steady_state + dt_model/(1440)*(i-1) for i in 1:Int(T_training*1440/dt_model)], 1, 1)
dt_vec = repeat([dt_model/(1440)], 1, Int(T_training*1440/dt_model))'
E_train = hcat(Q_in_vec, X_in_vec, T_start_vec, dt_vec)

######################################################
################## Particle Filter ###################
######################################################

n_filtering = 100
@timed filter_output, filtered_state_mean,  filtered_state_var = StateSpaceIdentification.filter(model2, y_train, E_train, U_train, filter=ParticleFilter(init_state, 5, nb_obs_var, n_filtering))

######################################################
############### Ensemble Kalman Filter ###############
######################################################

@timed filter_output = StateSpaceIdentification.filter(model2, y_train, E_train, U_train, filter=EnsembleKalmanFilter(init_state, 5, nb_obs_var, n_filtering));

index_t_training = [filter_output.predicted_particles_swarm[t].t for t in 1:filter_output.predicted_particles_swarm.n_t]
index_t_sim = [T_steady_state + (1/1440)*t for t in 1:Int(T_training*1440)]
plot(filter_output.predicted_particles_swarm, index = [4], label= ["NH4"])
scatter!(index_t_training[1:(end-1)], y_train[:, 1], label="Observations")
plot!(index_t_sim, x_train[10, :], label="True NH4")

@timed smoother_output = smoother(model2, y_train, E_train, U_train, filter_output; smoother_method=EnsembleKalmanSmoother(5, nb_obs_var, n_filtering))

plot(smoother_output.smoothed_state, index = [4], label= ["NH4"])
plot!(index_t_sim, x_train[10, :], label="True NH4")
size(model_params, 1)

lb = vcat([repeat([1e-2], size(model_params, 1)), repeat([-Inf], size(cov_params, 1))]...)
ub = repeat([Inf], size(params, 1))
sol_em_enks = EM_EnKS(model2, y_train, E_train, U_train; lb=lb, ub=ub, n_particles = 100, maxiters_em=10, optim_method=Opt(:LD_LBFGS, 25), maxiters=10)

params_enks, results_enks = sol_em_enks
# => solution : améliorer la stabilité de la résolution de l'EDO ! car la ca marche pas du tout
model2.parameters = params_enks 

sort(filter_output.predicted_particles_swarm[2].particles_state[2, :])
variance_1 = [var(filter_output.predicted_particles_swarm[t].particles_state, dims=2)[1] for t in 1:filter_output.predicted_particles_swarm.n_t]

plot(filter_output.predicted_particles_swarm, index = [1], label= ["NH4"])
plot!(index_t_training, variance_1*3)


M_t(filter_output.predicted_particles_swarm[3].particles_state, E_train[3, :], U_train[3, :], model2.parameters)

# COnnard d'oxygene qui devient negatif ...

######################################################
######################################################
######################################################

x_tab = [x_init]
@timed for i in 1:Int(T_training*1440)

    x_i = M_t(x_tab[end], U_train[i], exogenous, params_bis)

    push!(x_tab, x_i[1:5])

end


