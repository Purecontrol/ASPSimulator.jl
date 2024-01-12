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
# parameters = [1333.0, 200.0, -2.30, -2.30, 0.36] 
parameters = [1099, 262, -2.30, -2.30, 0.71]
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
@timed optim_params_enks_em, results_enks = EM_EnKS(model, y_train, E_train, U_train; lb=lb, ub=ub, n_particles = 300, maxiters_em=1, optim_method=Opt(:LD_LBFGS, 5), maxiters=100);
@timed optim_params_pfbs_em, results_pfbs = SEM(model, y_train, E_train, U_train; lb=lb, ub=ub, n_filtering = 300, n_smoothing = 300, maxiters_em=30, optim_method=Opt(:LD_LBFGS, 5), maxiters=10);
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
plot!(index_t_sim, x_train[10, :], label="True NH4")

#####################################################################
################ Graph for the papers ###############################
#####################################################################

index_x = [T_steady_state + (1/1440)*t for t in 1:Int((T_training+T_testing)*1440)]
index_y = [T_steady_state + (1/1440)*t*dt_model for t in 1:Int((T_training)*1440/dt_model)]
index_u = [T_steady_state + (1/1440)*t*dt_model for t in 1:Int((T_training + T_testing)*1440/dt_model)]

Q_in_t = [Q_in(T_steady_state + t*dt_model/(24*60)) for t in 1:(size(y_train, 1)+size(y_test, 1))]
X_in_t = [X_in for t in 1:(size(y_train, 1)+size(y_test, 1))]
E_graph = hcat([Q_in_t, X_in_t]...)

y_graph = vcat(y_train, similar(y_test).*NaN)
u_graph = vcat(U_train, U_test)
x_graph = hcat(x_train, x_test)

n_smoothing = 300
filter_output, _, _ = StateSpaceIdentification.filter(model, y_graph, E_graph, u_graph, filter=ParticleFilter(model, n_particles = 300))
smoother_output_bs1 = backward_smoothing(y_graph, E_graph, filter_output, model, model.parameters; n_smoothing=n_smoothing)

Plots.backend(:gr)

using LaTeXStrings
using Measures
plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=false)
# scalefontsizes(0.9)


plot(index_u[Int((1440/dt_model)*(T_training-1)):end], u_graph[Int((1440/dt_model)*(T_training-1)):end, :], alpha=0.5, color=:grey, label=L"u(t)")
plot!(filter_output.predicted_particles_swarm[Int((1440/dt_model)*(T_training-1)+1):end], label= ["Model"])
plot!(size=(1000, 320), xlabel="Time (in days)", ylabel=L"S_{NH}"*" (mg/L)", margin=6mm)
plot!(index_x[Int((1440)*(T_training-1)):end], x_graph[10, Int((1440)*(T_training-1)):end], label="True NH4", linestyle=:dashdot)
scatter!(index_y[Int((1440/dt_model)*(T_training-1)+1):end], y_train[Int((1440/dt_model)*(T_training-1)+1):end], label="Observations", markersize=1.0)
plot!(legend=:topright)
vline!([25.0], color=:black)
annotate!(24.02,7.5,text("Past",plot_font,15))
annotate!(25.1,7.5,text("Future",plot_font,15))
plot!(legend=:bottomleft)


savefig("model_prediction.pdf")



#####################################################################
################## RMSE Computation #################################
#####################################################################

true_nh4 = x_graph[10, Int((1440)*(T_training)):end][1:dt_model:end]
t = filter_output.predicted_particles_swarm[Int((1440/dt_model)*(T_training)+1):end]
mean_process = hcat([mean(t[i].particles_state, dims=2) for i in 1:t.n_t]...)'
q_low = hcat([[quantile(t[i].particles_state[j, :], 0.025) for j in 1:t.n_state] for i in 1:t.n_t]...)'
q_high = hcat([[quantile(t[i].particles_state[j, :], 0.975) for j in 1:t.n_state] for i in 1:t.n_t]...)'

max_nh4 = max(maximum(mean_process), maximum(true_nh4))
plot(LinRange(0, max_nh4, 100), LinRange(0, max_nh4, 100), lw=3, xlabel=L"\hat{S}_{NH}", ylabel=L"S_{NH}", legend=false)
scatter!(true_nh4, mean_process, markersize=2.5)



IC = mean(q_low .< true_nh4 .< q_high)
NW = mean(q_high - q_low)

H = 2
rmse(H, true_val, pred; interval=H) = mean(sqrt.((true_val[Int((max(1, H-interval)/(dt_model/60))):Int(H/(dt_model/60))] - pred[Int(max(1, H-interval)/(dt_model/60)):Int(H/(dt_model/60))]).^2))

plot([rmse(h, true_nh4, mean_process) for h in 1:24])

#################################################################################
################## Estimation using classical ML ################################
#################################################################################

function model_nh4(params, x, u, exogenous)

    return x .+ ((exogenous[1:1, :]./params[1]).*(exogenous[2:2, :] .- x) .- params[2].*u.*(x./(x .+ params[3]))).*(dt_model/1440)

end

function loss(params, y, x, u, exogenous)
    return mean((y .- model_nh4(params, x, u, exogenous)).^2)
end


callback = function (p, l)
    println("Loss : ", l)
    return false
end

# Downsample x_train
x_true = x_train[10, 1:dt_model:end]
x_true = y_train

y = reshape(x_true[2:end], 1, 1439)# - x_true[1:(end-1)], 1, 1439)
x = reshape(x_true[1:(end-1)], 1, 1439)
u = U_train[1:(end-1), :]'
e = E_train[1:(end-1), :]'



k = 1439
train_loader = Flux.Data.DataLoader((y, x, u, e), batchsize = k)

optfun = OptimizationFunction((θ, params, y, x, u, exogenous) -> loss(θ, y, x, u, exogenous), Optimization.AutoForwardDiff())

init_p = [800.0, 100.0, 2.0]
optprob = OptimizationProblem(optfun, init_p)

numEpochs = 50000

using OptimizationOptimisers
using IterTools: ncycle
res1 = Optimization.solve(optprob, Optimisers.ADAM(0.1), ncycle(train_loader, numEpochs), callback = callback)

mean(y - model_nh4(res1, x, u, e))
sqrt(var(y - model_nh4(res1, x, u, e)))

plot(model_nh4(res1, x, u, e)')
plot!(y')


Q_in_t = [Q_in(T_steady_state + T_training + t*dt_model/(24*60)) for t in 1:size(y_test, 1)]
X_in_t = [X_in for t in 1:size(y_test, 1)]
E_test = hcat([Q_in_t, X_in_t]...)

x_pred = zeros(289)
x_pred[1] = y_train[end, 1]
for i in 1:Int(1440/dt_model)
    x_pred[i+1] = model_nh4(res1, x_pred[i], U_test[i, :], E_test[i, :])[1, 1]
end
###### Comparison ###########

interval=24
plot([rmse(h, true_nh4, mean_process, interval=interval) for h in 1:24], label="EM")
plot!([rmse(h, true_nh4, x_pred, interval=interval) for h in 1:24], label="PEM")
plot!(xlabel="Horizon (H)", ylabel="RMSE")

plot([t[i].t for i in 1:t.n_t], mean_process, label="EM")
plot!([t[i].t for i in 1:t.n_t], x_pred, label="PEM")
plot!([t[i].t for i in 1:t.n_t], true_nh4, label="True")

