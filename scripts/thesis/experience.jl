using ASM1Simulator
using DifferentialEquations, Distributions
using LinearAlgebra:I
using Statistics:mean
using Plots
using StateSpaceIdentification
using SparseArrays
using Optim, OptimizationNLopt, PDMats
using JLD

include("../StateSpace/env.jl")

# User-defined parameters
T_steady_state = 20 #(in days)
T_training = 5.0 #(in days)
T_testing = 1.0 #(in days)
σ_ϵ = 0.2
# WARNING : dt_obs must be a multiple of dt_model
dt_obs = 5 #(in minutes)
dt_model = 5 #(in minutes)

# Fixed parameters
nb_var = 14
index_obs_var = [10]
nb_obs_var = size(index_obs_var, 1)


####################################################################################################################
############################################### Variation parameters ###############################################
####################################################################################################################

X_in_tab = [6.8924, 7.8924, 8.8924, 9.8924]

σ_ϵ_tab = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

X_in_res = Array{Any, 1}(undef, size(X_in_tab, 1)*size(σ_ϵ_tab, 1))
σ_ϵ_res = Array{Any, 1}(undef, size(X_in_tab, 1)*size(σ_ϵ_tab, 1))
params_res = Array{Any, 1}(undef, size(X_in_tab, 1)*size(σ_ϵ_tab, 1))
true_nh4_res = Array{Any, 1}(undef, size(X_in_tab, 1)*size(σ_ϵ_tab, 1))
predicted_nh4_res = Array{Any, 1}(undef, size(X_in_tab, 1)*size(σ_ϵ_tab, 1))

###############################################################################################
########################################## Experience #########################################
###############################################################################################

Threads.@threads for i_x_in in 1:size(X_in_tab, 1)

    X_in_ite = X_in_tab[i_x_in]

    for i_σ in 1:size(σ_ϵ_tab, 1)

        σ_ϵ = σ_ϵ_tab[i_σ]

        ite_nb = (i_x_in-1)*size(X_in_tab, 1) + i_σ

        X_in_res[ite_nb] = X_in_ite
        σ_ϵ_res[ite_nb] = σ_ϵ

        ####################################################################################################
        ########################################## Generation data #########################################
        ####################################################################################################
        
        sim = ODECore()

        sim.params_vec.X_in[10] = X_in_ite

        # Let the system evolve during T_steady_state days in order to be stable and get new X_init
        res_steady = multi_step!(sim, ASM1Simulator.Models.redox_control(), Day(T_steady_state))
        X_init_real = sim.state

        # Let the system evolve during T_training + T_testing days to get training and testing data
        sol_real  = multi_step!(sim, ASM1Simulator.Models.redox_control(), Hour((T_training + T_testing)*24))

        # Define H
        H = zeros(nb_var, nb_var)
        H[index_obs_var, index_obs_var] = Matrix(I, nb_obs_var, nb_obs_var)
        H = H[index_obs_var, :]

        # Get training and test set for the state of the system
        x_train = hcat(sol_real.u...)[:, 1:Int(T_training*1440)]
        x_test = hcat(sol_real.u...)[:, (Int(T_training*1440)+1):end]

        # Get training and test set for the observation of the system
        y_train = max.((H*x_train + rand(Normal(0, σ_ϵ), (nb_obs_var, size(x_train, 2))) + vcat([reshape([(i-1)%Int(dt_obs) == 0 ? 0 : NaN for i in 1:size(x_train, 2)], (1,size(x_train, 2))) for i in 1:nb_obs_var]...))', 0)
        y_test = max.((H*x_test + rand(Normal(0, σ_ϵ), (nb_obs_var, size(x_test, 2))) + vcat([reshape([(i-1)%Int(dt_obs) == 0 ? 0 : NaN for i in 1:size(x_test, 2)], (1,size(x_test, 2)))  for i in 1:nb_obs_var]...))', 0)

        # Get control variables of the system
        U_train = transpose(hcat(getindex.(sol_real.u, 14)...))[1:Int(T_training*1440), :]
        U_test = transpose(hcat(getindex.(sol_real.u, 14)...))[(Int(T_training*1440)+1):end, :]

        # Adapt y_train and U_train according to dt_assimilation
        y_train = y_train[1:dt_model:end, 1:end]
        y_test = y_test[1:dt_model:end, 1:end]

        U_train = U_train[1:dt_model:end, :]
        U_test = U_test[1:dt_model:end, :]

        ####################################################################################################
        ########################################## Model estimation ########################################
        ####################################################################################################



        function M_t(x, exogenous, u, params)

            # Get Q_in and V
            Q_in = exogenous[1]
            X_in = exogenous[2]
            V = params[1]
            β = params[2]
            K = params[5]

            # Define A
            # A = sparse(hcat([[1 - Q_in/V*(dt_model/1440)]]...))
            # B = sparse(hcat([[- β*(dt_model/1440)]]...))

            A = sparse([1 - Q_in/V*(dt_model/1440);;])
            B = sparse([- β*(dt_model/1440);;])

            return A*x + B*u.*(x./(x .+ K)) .+ sparse([(X_in*Q_in)/V*(dt_model/1440)])

        end

        H_t(x, exogenous, params) = x

        function R_t(exogenous, params)

            # Get params
            # η = exp(params[3])
            η = exp(params[3])


            # Define R
            # R = hcat([[η]]...)

            return PDiagMat([η])

        end

        function Q_t(exogenous, params)

            # Get params
            # ϵ = exp(params[4])
            ϵ = exp(params[4])

            # Define Q
            # Q = hcat([[ϵ]]...)

            return PDiagMat([ϵ])

        end


        # Define the system
        n_X = 1
        n_Y = 1
        gnlss = GaussianNonLinearStateSpaceSystem(M_t, H_t, R_t, Q_t, n_X, n_Y, dt_model/(1440))

        # Define init state
        init_P_0 = zeros(1, 1) .+   1 #0.001
        init_state = GaussianStateStochasticProcess(T_steady_state, [x_train[10,1]], init_P_0)

        # Define model
        # parameters = [1333.0, 200.0, -2.30, -2.30, -1]
        parameters = [1333.0, 200.0, -2.30, -2.30, 0.36]
        # parameters = [1333.0, 200.0, 1.0, 1.0, 0.36]
        model = ForecastingModel{GaussianNonLinearStateSpaceSystem}(gnlss, init_state, parameters)

        # Set exogenous predictors
        Q_in = sim.params_vec.Q_in
        X_in = sim.params_vec.X_in[10]

        Q_in_t = [Q_in(T_steady_state + t*dt_model/(24*60)) for t in 1:size(y_train, 1)]
        X_in_t = [X_in for t in 1:size(y_train, 1)]
        E_train = hcat([Q_in_t, X_in_t]...)

        # Optimize with EM using approximate EnKS smoother
        lb = [1e-2, 1e-2, -Inf, -Inf, 1e-2]
        ub = [Inf, Inf, Inf, Inf, Inf]
        @timed optim_params_pfbs_em, results_pfbs = SEM(model, y_train, E_train, U_train; lb=lb, ub=ub, n_filtering = 300, n_smoothing = 300, maxiters_em=25, optim_method=Opt(:LD_LBFGS, 5), maxiters=100);
        model.parameters = optim_params_pfbs_em

        params_res[ite_nb] = optim_params_pfbs_em

        println("   V    | Estimated PFBS = ", round(optim_params_pfbs_em[1], digits=3))
        println("   β    | Estimated PFBS = ", round(optim_params_pfbs_em[2], digits=3))
        println("   K    | Estimated PFBS = ", round(optim_params_pfbs_em[5], digits=3))
        println("σ_model | Estimated PFBS = ", round(sqrt(exp(optim_params_pfbs_em[3])), digits=3), " | Real = ", 0.0)
        println("σ_obs   | Estimated PFBS = ", round(sqrt(exp(optim_params_pfbs_em[4])), digits=3), " | Real = ", σ_ϵ)

        ####################################################################################################
        ############################################ Get results ###########################################
        ####################################################################################################


        index_x = [T_steady_state + (1/1440)*t for t in 1:Int((T_training+T_testing)*1440)]
        index_y = [T_steady_state + (1/1440)*t*dt_model for t in 1:Int((T_training)*1440/dt_model)]
        
        Q_in = sim.params_vec.Q_in
        Q_in_t = [Q_in(T_steady_state + t*dt_model/(24*60)) for t in 1:(size(y_train, 1)+size(y_test, 1))]
        X_in_t = [X_in for t in 1:(size(y_train, 1)+size(y_test, 1))]
        E_graph = hcat([Q_in_t, X_in_t]...)
        
        y_graph = vcat(y_train, similar(y_test).*NaN)
        u_graph = vcat(U_train, U_test)
        x_graph = hcat(x_train, x_test)
        
        n_smoothing = 300
        filter_output, _, _ = StateSpaceIdentification.filter(model, y_graph, E_graph, u_graph, filter=ParticleFilter(model, n_particles = 300))
        smoother_output_bs1 = backward_smoothing(y_graph, E_graph, filter_output, model, model.parameters; n_smoothing=n_smoothing)
        
        true_nh4 = x_graph[10, Int((1440)*(T_training)):end][1:dt_model:end]

        true_nh4_res[ite_nb] = true_nh4
        predicted_nh4_res[ite_nb] = filter_output.predicted_particles_swarm[Int((1440/dt_model)*(T_training)+1):end]
        
    end
end

# @save "experience.jld" X_in_res, σ_ϵ_res, params_res, true_nh4_res, predicted_nh4_res
test = load("experience.jld")
results = test["(X_in_res, σ_ϵ_res, params_res, true_nh4_res, predicted_nh4_res)"]


params

X_in_res = results[1][1:22]
σ_ϵ_res = results[2][1:22]
params_res = results[3][1:22]
true_nh4_res = results[4][1:22]
predicted_nh4_res = results[5][1:22]
# Plots.backend(:gr)
# violin(hcat(results...)'[:, 1:5])
# violin(hcat([log10.(hcat(results...)'[:, 1:2]), log10.(sqrt.(exp.(hcat(results...)'[:, 3:4]))), log10.(hcat(results...)'[:, 5:5])]...))

#################################################################################################################
############################# Paper results #####################################################################
#################################################################################################################
rmse(H, true_val, pred) = mean(sqrt.((true_val[1:Int(H/(dt_model/60))] - pred[1:Int(H/(dt_model/60))]).^2))
ic_function(H, true_val, q_inf, q_sup) = mean(q_inf[1:Int(H/(dt_model/60))] .< true_val[1:Int(H/(dt_model/60))] .< q_sup[1:Int(H/(dt_model/60))])



nb_exp = 22 #size(true_nh4_res, 1)

mean_process_res = [hcat([mean(t[i].particles_state, dims=2) for i in 1:t.n_t]...)' for t in predicted_nh4_res]
q_low_res = [hcat([[quantile(t[i].particles_state[j, :], 0.025) for j in 1:t.n_state] for i in 1:t.n_t]...)' for t in predicted_nh4_res]
q_high_res = [hcat([[quantile(t[i].particles_state[j, :], 0.975) for j in 1:t.n_state] for i in 1:t.n_t]...)' for t in predicted_nh4_res]

IC_res = [mean(q_low_res[i] .< true_nh4_res[i] .< q_high_res[i]) for i in 1:nb_exp] 
NW_res = [mean(q_high_res[i] - q_low_res[i]) for i in 1:nb_exp] 
rmse_res = [rmse(24, true_nh4_res[i], mean_process_res[i]) for i in 1:nb_exp]
var_res = var(hcat([(hcat(params_res...)'[:, 1:2]), (sqrt.(exp.(hcat(params_res...)'[:, 3:4]))), (hcat(params_res...)'[:, 5:5])]...), dims=1)'

### RMSE
rmse_tab = hcat([[rmse(H, true_nh4_res[i], mean_process_res[i]) for i in 1:nb_exp] for H in 1:24]...)'
q_rmse_low = [quantile(rmse_tab[i, :], 0.025) for i in 1:24]
q_rmse_high = [quantile(rmse_tab[i, :], 0.975) for i in 1:24]
mean_rmse = [mean(rmse_tab[i, :]) for i in 1:24]
plot(rmse_tab)
plot(q_rmse_low, fillrange = q_rmse_high, alpha=0.3, label="IC 95% RMSE")#, label = hcat("IC 95 % ".*label...))
plot!(mean_rmse, label="Mean RMSE")#, label = hcat("Mean ".*label...))
plot!(xlabel="Hours of prediction", ylabel="RMSE")
savefig("global_results.pdf")

ic_tab = hcat([[ic_function(H, true_nh4_res[i], q_low_res[i], q_high_res[i]) for i in 1:nb_exp] for H in 1:24]...)'
q_ic_low = [quantile(ic_tab[i, :], 0.025) for i in 1:24]
q_ic_high = [quantile(ic_tab[i, :], 0.975) for i in 1:24]
mean_ic = [mean(ic_tab[i, :]) for i in 1:24]
plot(ic_tab)
plot(q_ic_low, fillrange = q_ic_high, alpha=0.3)#, label = hcat("IC 95 % ".*label...))
plot!(mean_ic)#, label = hcat("Mean ".*label...))

