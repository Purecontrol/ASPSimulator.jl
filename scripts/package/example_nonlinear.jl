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



ode_fct! = ASM1Simulator.Models.simplified_asm1!
ode_params = simplified_params_tot

function M_t(x, exogenous, u, params)

    # params = max.(params, 0.0000001)

    # ode_params = Array{Any, 1}(undef, 8)
    # ode_params[1] = params[1:8]
    # ode_params[2] = params[9:13]
    # ode_params[3] = ASM1Simulator.Models.get_stoichiometric_matrix_simplified_asm1(params[14:16])
    # ode_params[4] = params[17]
    # ode_params[5] = exogenous[1:5]
    # ode_params[6] = exogenous[6]
    # ode_params[7] = params[18]
    # ode_params[8] = params[19]

    # ode_params = simplified_params_tot

    # ode_fct! = ASM1Simulator.Models.simplified_asm1!

    ode_problem = ODEProblem(ode_fct!, vcat(x[:, 1], u),  (exogenous[7], exogenous[7] + exogenous[8]), ode_params)

    n_particules = size(x, 2)
    function prob_func(prob, i, repeat)
        remake(ode_problem, u0 = vcat(x[:, i], u))
    end
    monte_prob = EnsembleProblem(ode_problem, prob_func = prob_func)


    sim_results = solve(monte_prob, Euler(), dt = 1/(6*24*60), trajectories = n_particules, saveat=[exogenous[7] + exogenous[8]], maxiters=10e3, sensealg = ForwardSensitivity())

    return hcat([max.(sim_results[i].u[1][1:5], 0.0) for i in 1:n_particules]...)

end

function H_t(x, exogenous, params)

    #return Matrix([0 0 0 1 0])*x

    return Matrix([0 1 0 0 0  ; 0 0 0 1 0 ])*x


    # Matrix([0 1 0 0 0 ; 0 0 1 0 0  ; 0 0 0 1 0 ])

end

function R_t(exogenous, params)

    return params[1:5].*Matrix(I, 5, 5)

end


@timed M_t(filter_output.predicted_particles_swarm[1].particles_state ,exogenous_matrix[1, :], U_train[1, :], params_ter)

function Q_t(exogenous, params)

    return params[6:7].*Matrix(I, nb_obs_var, nb_obs_var)

end

params_bis = vcat([simplified_params[1], simplified_params[2], simplified_params[3], simplified_params[4], simplified_params[7], simplified_params[8]]...)

params_ter = [0.005064213536696635,0.005064213536696635, 0.005064213536696635,0.005064213536696635, 0.005064213536696635, 0.21351570163527817, 0.21351570163527817]

Q_in = simplified_params[6]
Q_in_vec = [Q_in(T_steady_state + 1/(24*60)*(t-1)) for t in 1:Int(T_training*1440)]


# Model
gnlss = GaussianNonLinearStateSpaceSystem(M_t, H_t, R_t, Q_t, 5, nb_obs_var, 1/(24*60))


init_P_0 = zeros(5, 5) .+ 0.001
init_state = GaussianStateStochasticProcess(T_steady_state, x_init, init_P_0)
model2 = ForecastingModel(gnlss, init_state, params_ter)


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

model2.parameters = [0.06371248296204134, 0.034015180660661, 0.01292750778748952, 0.009201129271155834, 0.005696603787443115, 0.21352161780768125, 0.21356337466892983]


# model2.parameters = [643.1145521564094, 21805.83611671609, 18054.661182990676, 15008.915309647797, 621.788246425763, 0.10429976738619016, 0.46362388179344455]
@timed filter_output, filtered_state_mean,  filtered_state_var = filter(model2, y_t, exogenous_matrix, U_train)


######################################################
############### Ensemble Kalman Filter ###############
######################################################

@timed filter_output = filter(model2, y_t, exogenous_matrix, U_train, filter=EnsembleKalmanFilter(init_state, 5, nb_obs_var, 30))

plot(filter_output.predicted_particles_swarm, index = [2], label= ["NH4"])
# scatter!(y_t[:, 2], label="Observations")
plot!(x_train[10, :], label="True NH4")

@timed smoother_output = smoother(model2, y_t, exogenous_matrix, U_train, filter_output; smoother_method=EnsembleKalmanSmoother(5, nb_obs_var, 30))

plot(smoother_output.smoothed_state, index = [2], label= ["NH4"])
plot!(x_train[8, :], label="True NH4")

sol_em_enks = EM_EnKS(model2, y_t, exogenous_matrix, U_train)

# => solution : améliorer la stabilité de la résolution de l'EDO ! car la ca marche pas du tout

######################################################
######################################################
######################################################

# x_tab = [x_init]
# @timed for i in 1:Int(T_training*1440)

#     exogenous = vcat([params[5], Q_in_vec[i], [20.0 + 1/(24*60)*(i-1), 1/(24*60)]]...)

#     x_i = M_t(x_tab[end], U_train[i], exogenous, params_bis)

#     push!(x_tab, x_i[1:5])

# end


