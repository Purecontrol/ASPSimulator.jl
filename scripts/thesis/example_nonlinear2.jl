using Plots
using StateSpaceIdentification
using SparseArrays
using Optim, OptimizationNLopt, PDMats

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
# lb = [1e-2, 1e-2, 1e-4, 1e-4, 1e-2]
ub = [Inf, Inf, Inf, Inf, Inf]
@timed optim_params_enks_em, results_enks = EM_EnKS(model, y_train, E_train, U_train; lb=lb, ub=ub, n_particles = 300, maxiters_em=10, optim_method=Opt(:LD_LBFGS, 5), maxiters=100);
@timed optim_params_pfbs_em, results_pfbs = SEM(model, y_train, E_train, U_train; lb=lb, ub=ub, n_filtering = 300, n_smoothing = 300, maxiters_em=15, optim_method=Opt(:LD_LBFGS, 5), maxiters=100);
model.parameters = optim_params_pfbs_em

println("--------------------------------------------------------------------")
println("----------------------- OPTIMIZATION RESULTS -----------------------")
println("--------------------------------------------------------------------\n")


println("   V    | Estimated EnKS = ", round(optim_params_enks_em[1], digits=3), " | Estimated PFBS = ", round(optim_params_pfbs_em[1], digits=3))
println("   β    | Estimated EnKS = ", round(optim_params_enks_em[2], digits=3), " | Estimated PFBS = ", round(optim_params_pfbs_em[2], digits=3))
println("   K    | Estimated EnKS = ", round(optim_params_enks_em[5], digits=3), " | Estimated PFBS = ", round(optim_params_pfbs_em[5], digits=3))
println("σ_model | Estimated EnKS = ", round(sqrt(exp(optim_params_enks_em[3])), digits=3), " | Estimated PFBS = ", round(sqrt(exp(optim_params_pfbs_em[3])), digits=3), " | Real = ", 0.0)
println("σ_obs   | Estimated EnKS = ", round(sqrt(exp(optim_params_enks_em[4])), digits=3), " | Estimated PFBS = ", round(sqrt(exp(optim_params_pfbs_em[4])), digits=3), " | Real = ", σ_ϵ)

println("\n--------------------------------------------------------------------")
println("--------------------------------------------------------------------")
println("--------------------------------------------------------------------")

#######################################################
### Show results with iterative step ahead forecast ###
#######################################################

# Filtering
y_bis = similar(y_train)
y_bis .= NaN
filter_output, _, _ = StateSpaceIdentification.filter(model, y_train, E_train, U_train, filter=ParticleFilter(model, n_particles = 1000))

# Show results
index_t_training = [filter_output.filtered_particles_swarm[t].t for t in 1:filter_output.filtered_particles_swarm.n_t]
index_t_sim = [T_steady_state + (1/1440)*t for t in 1:Int(T_training*1440)]
plot(filter_output.predicted_particles_swarm, label= ["NH4"])
scatter!(index_t_training, y_train, label="Observations", markersize=0.7)
plot(index_t_sim, x_train[10, :], label="True NH4")

#####################################################################
################ Graph for the papers ###############################
#####################################################################

index_x = [T_steady_state + (1/1440)*t for t in 1:Int((T_training+T_testing)*1440)]
index_y = [T_steady_state + (1/1440)*t*dt_model for t in 1:Int((T_training)*1440/dt_model)]


Q_in_t = [Q_in(T_steady_state + t*dt_model/(24*60)) for t in 1:(size(y_train, 1)+size(y_test, 1))]
X_in_t = [X_in for t in 1:(size(y_train, 1)+size(y_test, 1))]
E_graph = hcat([Q_in_t, X_in_t]...)

y_graph = vcat(y_train, similar(y_test).*NaN)
u_graph = vcat(U_train, U_test)
x_graph = hcat(x_train, x_test)

n_smoothing = 300
filter_output, _, _ = StateSpaceIdentification.filter(model, y_graph, E_graph, u_graph, filter=ParticleFilter(model, n_particles = 300))
smoother_output_bs1 = backward_smoothing(y_graph, E_graph, filter_output, model, model.parameters; n_smoothing=n_smoothing)

Plots.backend(:pythonplot)

using LaTeXStrings
using Measures
plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=false)
# scalefontsizes(0.9)


plot(filter_output.predicted_particles_swarm[577:721], label= ["Model"])
plot!(size=(1000, 350), xlabel="Time (in days)", ylabel=L"S_{NH}"*" (mg/L)", margin=6mm)
plot!(filter_output.predicted_particles_swarm[721:end], label= ["Predicted NH4"])
plot!(index_x[5760:end], x_graph[10, 5760:end], label="True NH4")
scatter!(index_y[577:end], y_train[577:end], label="Observations", markersize=1.0)
# plot!(legend=:outerbottom, legend_columns=3)
vline!([25.0])
annotate!(24.02,7.5,text("Past",plot_font,15))
annotate!(25.1,7.5,text("Futur",plot_font,15))
savefig("model_prediction.pdf")