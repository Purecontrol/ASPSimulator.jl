using Optimization, OptimizationOptimJL, LinearAlgebra, OptimizationOptimisers

include("utils.jl")
include("env.jl")

#####################################################
#### Define the structure of the grey-box models ####
#####################################################

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

init_P_0 = zeros(1, 1) .+ 0.001
Q_in = sim.params_vec[5]
function X_in(t)
    return sim.params_vec[4][10]
end
exogenous_variables = [Q_in, X_in]

ssm = StateSpaceModel(
    A_t,
    B_t,
    H_t,   
    R_t,
    Q_t,
    c_t, 
    d_t,
    [x_train[10,1]],
    init_P_0,
    sim.fixed_dt,
    T_steady_state,
)

#################################################
#### Optimized parameters using EM algorithm ####
#################################################

function llh(params_vec, ssm_model)

    return - kalman_filter_optim(y_train, ssm_model, exogenous_variables, U_train, params_vec)

end

x0 = [1333.0, 200.0, -2.30, -2.30]
x0 = [10, 10.0, 5.30, 5.30]
x0 = [5000.0, 10.0, 5.30, 5.30]
p = ssm
optprob = OptimizationFunction(llh, Optimization.AutoForwardDiff())
prob = Optimization.OptimizationProblem(optprob, x0, p)
sol = solve(prob, Optim.BFGS(), maxiters = 1000, progress= true, store_trace=true) #Adam(0.01) #, show_trace=true, show_every=1, extended_trace=true

print(sol.original)

plot(Optim.f_trace(sol.original), title="Log-likelihood", xlabel="Iteration", ylabel="Log-likelihood", legend=false)

println("### OPTIMIZED PARAMETERS ###")
println("σ_model | Estimated = ", round(exp(sol.minimizer[3]), digits=3), " | Real = ", 0.0)
println("σ_obs   | Estimated = ", round(sqrt(exp(sol.minimizer[4])), digits=3), " | Real = ", σ_ϵ)

###################################################
#### Plot the models with optimized parameters ####
###################################################

# Update parameter vectors with optimized values
ssm.parameters = sol.minimizer#[817.3783503399302, 185.8007718508432, -2.512790998088006, -3.9909505087348265]#sol_test.minimizer

X_hat_train, Y_hat_train, P_hat_train = forecast(ssm, exogenous_variables, U_train; steps_ahead=Int(1440*T_training))

filtered_X̂t, filtered_P̂t, predicted_X̂t, predicted_P̂t, K, L, S, v = kalman_filter(ssm, y_train, exogenous_variables, U_train)

smoothed_X̂t, smoothed_P̂t, N = kalman_smoother(ssm, y_train, exogenous_variables, U_train, predicted_X̂t, predicted_P̂t, L, S, v)

smoothed_ϵt, smoothed_ηt, smoothed_var_ϵt, smoothed_var_ηt = kalman_disturbance_smoother(ssm, y_train, exogenous_variables, U_train, K, L, S, v)

smoothed_pairwise_P̂t = kalman_pair_smoother(ssm, y_train, exogenous_variables, U_train, N, predicted_P̂t, L)

l_plot = 722

# EM(y_train, ssm, exogenous_variables, U_train, sol.minimizer)

@timed EM(y_train, ssm, exogenous_variables, U_train, sol.minimizer)

@timed loss_tab = EM2(y_train, ssm, exogenous_variables, U_train, [826.8134453448635, 187.7418534868474, -2.30, -2.30])


x0 = minimizer #[5000.0, 10.0, 5.30, 5.30]
@timed loss_tab = EM3(y_train, ssm, exogenous_variables, U_train, x0)

@timed loss_tab = EM4(y_train, ssm, exogenous_variables, U_train, x0)

@timed loss_tab, minimizer = EM5(y_train, ssm, exogenous_variables, U_train, x0)



@timed llh(sol.minimizer, ssm)

plot(collect(1:(Int(1440*T_training) + 1))[1:l_plot], hcat(smoothed_X̂t...)[1,1:l_plot] - 0.96*sqrt.(abs.(hcat(smoothed_P̂t...)[1,1:l_plot])), fillrange = hcat(smoothed_X̂t...)[1,1:l_plot] + 0.96*sqrt.(abs.(hcat(smoothed_P̂t...)[1,1:l_plot])), fillalpha = 0.35, c = 1, label = "IC 95%", lw=0)
plot!(collect(1:(Int(1440*T_training)))[1:l_plot], hcat(filtered_X̂t...)[1,1:l_plot] - 0.96*sqrt.(hcat(filtered_P̂t...)[1,1:l_plot]), fillrange = hcat(filtered_X̂t...)[1,1:l_plot] + 0.96*sqrt.(hcat(filtered_P̂t...)[1,1:l_plot]), fillalpha = 0.35, c = 2, label = "IC 95%", lw=0)
scatter!(collect(1:(Int(1440*T_training)))[1:l_plot], y_train[1, 1:l_plot], label="Observations", c=3)
plot!(collect(1:(Int(1440*T_training) + 1))[1:l_plot], hcat(X_hat_train...)[1,1:l_plot], label="Multistep ahead forecast", c=4)
plot!(collect(1:(Int(1440*T_training) + 1))[1:l_plot], hcat(smoothed_X̂t...)[1,1:l_plot], label="Smoothing", c=1)
plot!(collect(1:(Int(1440*T_training) + 1))[1:l_plot], x_train[10,1:l_plot], label="True state", c=7, linestyle = :dash, linewidth = 2)
plot!(collect(1:(Int(1440*T_training) + 1))[1:l_plot], hcat(filtered_X̂t...)[1,1:l_plot], label="Filtering", c=2)
plot!(legend=:topright)
plot!(size=(750,550))

lwb_smoothing = reshape((hcat(smoothed_X̂t...)[1,:] - 0.96*sqrt.(abs.(hcat(smoothed_P̂t...)[1,:])))[1:end-1], (1, Int(T_training*1440)))
upb_smoothing = reshape((hcat(smoothed_X̂t...)[1,:] + 0.96*sqrt.(abs.(hcat(smoothed_P̂t...)[1,:])))[1:end-1], (1, Int(T_training*1440)))
target = H*x_train
lwb_filtered = reshape((hcat(predicted_X̂t...)[1,:] - 0.96*sqrt.(abs.(hcat(predicted_P̂t...)[1,:])))[1:end-1], (1, Int(T_training*1440)))
upb_filtered = reshape((hcat(predicted_X̂t...)[1,:] + 0.96*sqrt.(abs.(hcat(predicted_P̂t...)[1,:])))[1:end-1], (1, Int(T_training*1440)))

println("### METRICS RESULTS ###")
println("CP | Smoothed = ", round.(get_coverage_probability(target, lwb_smoothing, upb_smoothing), digits=3), " | Real = ", 0.95)
println("CP | Filtered = ", round.(get_coverage_probability(target, lwb_filtered, upb_filtered), digits=3), " | Real = ", 0.95)
println("RMSE | Filtered = ", round.(rmse(hcat(predicted_X̂t...)[1, 1:end-1], target), digits=3), " | Smoothed = ", round.(rmse(hcat(smoothed_X̂t...)[1, 1:end-1], target), digits=3))


###################################################
###### Compute the performance of the models ######
###################################################

H_horizon_prediction = 12 #(in H)

# Define table to store the results
X_hat_vec = Array{Array{Array{Float64, 1}, 1}, 1}(undef, Int(T_testing*1440 - H_horizon_prediction*60))
P_hat_vec = Array{Array{Array{Float64, 2}, 1}, 1}(undef, Int(T_testing*1440 - H_horizon_prediction*60))

# Set up intial values
ssm.X_0 = predicted_X̂t[end-1]
ssm.P_0 = predicted_P̂t[end-1]
ssm.t_start = T_training + T_steady_state
ssm.parameters = sol.minimizer

for i in 1:Int(T_testing*1440 - 720)

    println("Iteration number: ", i)

    # Apply the Kalman filter
    _, _, predicted_X̂t_ite, predicted_P̂t_ite, _, _, _, _ = kalman_filter(ssm, y_test[:,i], exogenous_variables, U_test[i,:])

    # Increment t_start
    ssm.t_start += ssm.dt

    # Set up new X_0 and P_0
    ssm.X_0 = predicted_X̂t_ite[end]
    ssm.P_0 = predicted_P̂t_ite[end]

    # Make a forecast of 12h
    X_hat, Y_hat, P_hat = forecast(ssm, exogenous_variables, U_test[(i+1):end,:]; steps_ahead=Int(H_horizon_prediction*60))

    # Store the results
    X_hat_vec[i] = X_hat
    P_hat_vec[i] = P_hat

end

# Compute the RMSE for each 12h forecast
RMSE = zeros(Int(T_testing*1440 - H_horizon_prediction*60))
CP = zeros(Int(T_testing*1440 - H_horizon_prediction*60))

for i in 1:Int(T_testing*1440 - H_horizon_prediction*60)

    RMSE[i] = sqrt(mean((x_test[10,(i+1):(i+Int(H_horizon_prediction*60))] - hcat(X_hat_vec[i][2:end]...)[1,:]).^2))
    CP[i] = get_coverage_probability(H*x_test[:,(i+1):(i+Int(H_horizon_prediction*60))], hcat(X_hat_vec[i][2:end]...) - 0.96*sqrt.(abs.(hcat(P_hat_vec[i][2:end]...))), hcat(X_hat_vec[i][2:end]...) + 0.96*sqrt.(abs.(hcat(P_hat_vec[i][2:end]...))))[1]

end

println("### PREDICTION RESULTS ###")
println("RMSE = ", mean(RMSE))
println("CP = ", mean(CP))

plot(RMSE, label="RMSE")
plot!(CP, label="CP")

anim = @animate for i in 1:10:Int(T_testing*1440 - Int(H_horizon_prediction*60))
    plot(1:Int(H_horizon_prediction*60), hcat(X_hat_vec[i]...)'[2:end, :] - 0.96*sqrt.(hcat(P_hat_vec[i]...)[1,:])[2:end, :], fillrange = hcat(X_hat_vec[i]...)'[2:end, :] + 0.96*sqrt.(hcat(P_hat_vec[i]...)[1,:])[2:end, :], fillalpha = 0.35, c = 1, label = "IC 95%", lw=0)
    plot!(hcat(X_hat_vec[i]...)', c=1, label="Forecast")
    plot!(x_test[10,(i+1):(Int(H_horizon_prediction*60)+i)], label="True state", c=7, linestyle = :dash, linewidth = 2)
end

gif(anim, fps=7)