using Plots
using SciMLSensitivity

include("time_series.jl")
include("systems/base.jl")
include("models.jl")
include("systems/gaussian_linear_state_space.jl")
include("systems/non_linear_gaussian_state_space.jl")
include("filters/base.jl")
include("smoothers/base.jl")
include("filter_smoother.jl")
include("filters/kalman_filter.jl")
include("filters/ensemble_kalman_filter.jl")
include("filters/particle_filter.jl")
include("smoothers/kalman_smoother.jl")
include("smoothers/ensemble_kalman_smoother.jl")
include("fit.jl")



x_init = [x_train[2, 1] + x_train[4, 1], x_train[8, 1], x_train[9, 1], x_train[10, 1], x_train[11, 1]]
u = 0.0

exogenous = [20.0, 1/(24*60)]

influent_file_path = "/home/victor/Documents/code/asm1-simulator/data/external/influent_files/dryinfluent.ascii"
simplified_params, _ = ASM1Simulator.Models.get_default_parameters_simplified_asm1(influent_file_path=influent_file_path, get_R=false)

simplified_params_tot, _ = ASM1Simulator.Models.get_default_parameters_simplified_asm1(influent_file_path=influent_file_path)

simplified_params_unfold = vcat(simplified_params...)
init_params = vcat(simplified_params_unfold[1:17], zeros(6), simplified_params_unfold[24:25])
greybox_problem = ODEProblem(ASM1Simulator.Models.simplified_asm1!, zeros(6),  (0, 1), ode_params)

# de = modelingtoolkitize(greybox_problem)
# greybox_problem = ODEProblem(de, jac=true)

function M_t(x, exogenous, u, params)

    # params = max.(params, 0.0000001)

    ode_params = vcat(params[1:17], exogenous[1:6], params[18:19])

    problem_ite = remake(greybox_problem, u0 = vcat(x[:, 1], u), tspan=(exogenous[7], exogenous[7] + exogenous[8]), p=ode_params)

    n_particules = size(x, 2)

    function prob_func(prob, i, repeat)
        remake(problem_ite, u0 = vcat(x[:, i], u))
    end
    monte_prob = EnsembleProblem(problem_ite, prob_func = prob_func)

    #AutoTsit5(Rosenbrock23())
    sim_results = solve(monte_prob, Euler(), dt = 1/(6*24*60), trajectories = n_particules, saveat=[exogenous[7] + exogenous[8]], maxiters=10e3, sensealg = ForwardSensitivity())

    return hcat([max.(sim_results[i].u[1][1:5], 0.0) for i in 1:n_particules]...)

end

# @btime M_t(x_init, exogenous_matrix[1, :], U_train[1, :], params);

function H_t(x, exogenous, params)

    #return Matrix([0 0 0 1 0])*x

    return Matrix([0 1 0 0 0  ; 0 0 0 1 0 ])*x


    # Matrix([0 1 0 0 0 ; 0 0 1 0 0  ; 0 0 0 1 0 ])

end

function R_t(exogenous, params)

    return params[20:24].*Matrix(I, 5, 5)

end

function Q_t(exogenous, params)

    return params[25:26].*Matrix(I, nb_obs_var, nb_obs_var)

end

params_bis = vcat([simplified_params[1], simplified_params[2], simplified_params[3], simplified_params[4], simplified_params[7], simplified_params[8]]...)

params_ter = [0.005064213536696635,0.005064213536696635, 0.005064213536696635,0.005064213536696635, 0.005064213536696635, 0.21351570163527817, 0.21351570163527817]

params = vcat([params_bis, params_ter]...)

# Model
gnlss = GaussianNonLinearStateSpaceSystem(M_t, H_t, R_t, Q_t, 5, nb_obs_var, 1/(24*60))

init_P_0 = zeros(5, 5) .+ 0.001
init_state = GaussianStateStochasticProcess(T_steady_state, x_init, init_P_0)
model2 = ForecastingModel(gnlss, init_state, params)

Q_in = simplified_params[6]
Q_in_vec = [Q_in(T_steady_state + 1/(24*60)*(t-1)) for t in 1:Int(T_training*1440)]
exogenous_vec = []
for i in 1:Int(T_training*1440)

    exogenous = vcat([simplified_params[5], Q_in_vec[i], [T_steady_state + 1/(24*60)*(i-1), 1/(24*60)]]...)  

    push!(exogenous_vec, exogenous)

end
exogenous_matrix = hcat(exogenous_vec...)'

y_t = transpose(y_train)

######################################################
################## Particle Filter ###################
######################################################


@timed filter_output, filtered_state_mean,  filtered_state_var = filter(model2, y_t, exogenous_matrix, U_train)


######################################################
############### Ensemble Kalman Filter ###############
######################################################

@timed filter_output = filter(model2, y_t, exogenous_matrix, U_train, filter=EnsembleKalmanFilter(init_state, 5, nb_obs_var, 30))

plot(filter_output.predicted_particles_swarm, index = [2], label= ["NH4"])
scatter!(y_t[:, 2], label="Observations")
plot!(x_train[10, :], label="True NH4")

@timed smoother_output = smoother(model2, y_t, exogenous_matrix, U_train, filter_output; smoother_method=EnsembleKalmanSmoother(5, nb_obs_var, 30))

plot(smoother_output.smoothed_state, index = [4], label= ["NH4"])
plot!(x_train[10, :], label="True NH4")

sol_em_enks = EM_EnKS(model2, y_t, exogenous_matrix, U_train)

# => solution : améliorer la stabilité de la résolution de l'EDO ! car la ca marche pas du tout

######################################################
######################################################
######################################################

x_tab = [x_init]
@timed for i in 1:Int(T_training*1440)

    x_i = M_t(x_tab[end], U_train[i], exogenous, params_bis)

    push!(x_tab, x_i[1:5])

end


