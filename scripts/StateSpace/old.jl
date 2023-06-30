using ASM1Simulator, DifferentialEquations, Distributions
using LinearAlgebra:I
using Statistics:mean

# User-defined parameters
T_steady_state = 20 #(in days)
T_training = 1.0 #(in days)
T_testing = 1.5 #(in days)
σ_ϵ = 0.5
dt_obs = 5 #(in minutes)

# Fixed parameters
nb_var = 14
index_obs_var = [10]
nb_obs_var = size(index_obs_var, 1)

####################################################################################################################
########################################## REFERENCE DATA FOR OPTIMIZATION #########################################
####################################################################################################################

# Get true init_vector and p_vector
influent_file_path = "/home/victor/Documents/code/asm1-simulator/data/external/influent_files/dryinfluent.ascii"
real_p, X_init = ASM1Simulator.Models.get_default_parameters_asm1(influent_file_path=influent_file_path)
real_p_vec, ~ = ASM1Simulator.Models.get_default_parameters_asm1(get_R=false, influent_file_path=influent_file_path)

# Let the system evolve during 20 days in order to be stable and get new X_init
tspan = (0, T_steady_state)
steady_prob = ODEProblem(ASM1Simulator.Models.asm1!, X_init, tspan, real_p)
sol_steady = solve(steady_prob, Tsit5(), callback=ASM1Simulator.Models.redox_control())
X_init_real = sol_steady.u[end]

# Define the real ODE problem
tspan = (T_steady_state, (T_steady_state + T_training + T_testing))
tsave = LinRange(T_steady_state, (T_steady_state + T_training + T_testing), Int((T_training + T_testing)*1440))
ode_prob = ODEProblem(ASM1Simulator.Models.asm1!, X_init_real, tspan, real_p)

# Generate the real solution
sol_real = solve(ode_prob, Tsit5(), saveat = tsave, callback=ASM1Simulator.Models.redox_control())

# Define H
H = zeros(nb_var, nb_var)
H[index_obs_var, index_obs_var] = Matrix(I, nb_obs_var, nb_obs_var)
H = H[index_obs_var, :]

# Get training and test set for the state of the system
x_train = hcat(sol_real.u...)[:, 1:Int(T_training*1440)]
x_test = hcat(sol_real.u...)[:, (Int(T_training*1440)+1):end]

# Get training and test set for the observation of the system
y_train = H*x_train + rand(Normal(0, σ_ϵ), (nb_obs_var, size(x_train, 2))) + reshape([(i-1)%Int(dt_obs) == 0 ? 0 : NaN for i in 1:size(x_train, 2)], (1,size(x_train, 2)))
y_test = H*x_test + rand(Normal(0, σ_ϵ), (nb_obs_var, size(x_test, 2))) + reshape([(i-1)%Int(dt_obs) == 0 ? 0 : NaN for i in 1:size(x_test, 2)], (1,size(x_test, 2)))

# Get control variables of the system
U_train = transpose(hcat(getindex.(sol_real.u, 14)...))[1:Int(T_training*1440), :]
U_test = transpose(hcat(getindex.(sol_real.u, 14)...))[(Int(T_training*1440)+1):end, :]

# function kalman_filter_old(ssm, yt, exogenous_variables, control_variables)

#     n_obs = size(yt, 2)
    
#     # Define variables
#     X̂t = Array{Array{Float64, 1}, 1}(undef, n_obs)
#     P̂t = Array{Array{Float64, 2}, 1}(undef, n_obs)

#     # Initialize variables
#     X̂t[1] = ssm.X_0
#     P̂t[1] = ssm.P_0

#     # Loop over observations
#     for t in 2:n_obs

#         # Get current t_step
#         t_step = ssm.t_start + (t-1)*ssm.dt

#         # Get current matrix A and B
#         A = ssm.A_t(exogenous_variables, ssm.parameters, t_step)
#         B = ssm.B_t(exogenous_variables, ssm.parameters, t_step)
#         H = ssm.H_t(exogenous_variables, ssm.parameters, t_step)
#         Q = ssm.Q_t(exogenous_variables, ssm.parameters, t_step)

#         # Forecast step
#         X̂t[t] = A*X̂t[t-1] + B*control_variables[t-1, :] + ssm.c_t(exogenous_variables, ssm.parameters, t_step)
#         P̂t[t] = transpose(A)*P̂t[t-1]*A + ssm.R_t(exogenous_variables, ssm.parameters, t_step)

#         # Analysis/Correction step 
#         ivar_obs = findall(.!isnan.(yt[:, t]))
#         n = length(ivar_obs)
#         if n > 0
        
#             ŷt = H[ivar_obs, :]*X̂t[t] + ssm.d_t(exogenous_variables, ssm.parameters, t_step)[ivar_obs]
#             innov = yt[ivar_obs, t] - ŷt
        
#             S = H[ivar_obs, :]*P̂t[t]*transpose(H[ivar_obs, :]) + Q[ivar_obs, ivar_obs]
#             K = P̂t[t]*transpose(H[ivar_obs, :])*inv(S)

#             # Update state and covariance
#             X̂t[t] = X̂t[t] + K*innov
#             P̂t[t] = (I - K*H[ivar_obs, :])*P̂t[t]

#         end

#     end

#     return X̂t, P̂t
# end

# function kalman_smoother_old(ssm, yt, exogenous_variables, control_variables, X̂t, P̂t, K, L, S, V)

#     n_obs = size(yt, 2)
    
#     # Define variables
#     smoothed_X̂t = Array{Array{Float64, 1}, 1}(undef, n_obs+1)
#     smoothed_P̂t = Array{Array{Float64, 2}, 1}(undef, n_obs+1)

#     # Initialize variables
#     smoothed_X̂t[n_obs+1] = X̂t[n_obs+1]
#     smoothed_P̂t[n_obs+1] = P̂t[n_obs+1]

#     # Loop over observations
#     r = zeros(1)
#     N = zeros(1, 1)
#     for t in (n_obs):-1:1

#         # Get current t_step
#         t_step = ssm.t_start + (t-1)*ssm.dt

#         # Get current matrix A and B
#         A = ssm.A_t(exogenous_variables, ssm.parameters, t_step)
#         H = ssm.H_t(exogenous_variables, ssm.parameters, t_step)
#         Q = ssm.Q_t(exogenous_variables, ssm.parameters, t_step)

#         # Compute number of observed variables
#         ivar_obs = findall(.!isnan.(yt[:, t]))
#         n = length(ivar_obs)

#         # Backward step
#         S = H[ivar_obs, :]*P̂t[t]*transpose(H[ivar_obs, :]) + Q[ivar_obs, ivar_obs]
#         K = A*P̂t[t]*transpose(H[ivar_obs, :])*inv(S)
#         L = A - K*H[ivar_obs, :]

#         if n > 0
#             innov = yt[ivar_obs, t] - (H[ivar_obs, ivar_obs]*X̂t[t] + ssm.d_t(exogenous_variables, ssm.parameters, t_step)[ivar_obs])
#             r = transpose(H[ivar_obs, :])*inv(S)*innov + transpose(L)*r
#         else
#             r = transpose(L)*r
#         end

#         N = transpose(H[ivar_obs, :])*inv(S)*H[ivar_obs, :] + transpose(L)*N*L


#         # Update state and covariance
#         smoothed_X̂t[t] = X̂t[t] + P̂t[t]*r
#         smoothed_P̂t[t] = P̂t[t] - P̂t[t]*N*P̂t[t]

#     end

#     return smoothed_X̂t, smoothed_P̂t
# end

function kalman_filter_optim(yt, exogenous_variables, control_variables, A_t, B_t, H_t, R_t, Q_t, c_t, dt, X_0, P_0, t_start, parameters)

    n_obs = size(yt, 2)

    # Initialize variables
    X = X_0
    P = P_0
    llk = 0.0
    nb = 0

    # Loop over observations
    for t in 1:n_obs

        # Get current t_step
        t_step = t_start + (t-1)*dt

        # Get current matrix A, B, H and Q
        A = A_t(exogenous_variables, parameters, t_step)
        B = B_t(exogenous_variables, parameters, t_step)
        H = H_t(exogenous_variables, parameters, t_step)
        Q = Q_t(exogenous_variables, parameters, t_step)

        # Check the number of correct observations
        ivar_obs = findall(.!isnan.(yt[:, t]))

        # Compute innovations and Kalman Gain
        v = yt[ivar_obs, t] - (H[ivar_obs, :]*X + d_t(exogenous_variables, parameters, t_step)[ivar_obs])
        S = H[ivar_obs, :]*P*transpose(H[ivar_obs, :]) + Q[ivar_obs, ivar_obs]
        K = A*P*transpose(H[ivar_obs, :])*inv(S)
        L = A - K*H[ivar_obs, :]

        # Compute the log-likelihood
        if length(ivar_obs) > 0
            nb += 1
            llk += - log(2*pi)/2 - (1/2)*(log(det(S)) + transpose(v)*inv(det(S))*v)
        end
        

        # Update state and covariance of time t and make prediction for time t+1
        X = A*X + K*v + B*control_variables[t, :] + c_t(exogenous_variables, parameters, t_step)
        P = A*P*transpose(L) + R_t(exogenous_variables, parameters, t_step)


    end

    llk = llk/nb

    return llk

end

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

ssm = StateSpaceModel(
    A_t,
    B_t,
    H_t,   
    R_t,
    Q_t,
    c_t, 
    d_t,
    [2.39889],
    zeros(1, 1),
    1/1440,
    T_steady_state,
)

function X_in2(t)
    return 6.8924
end

Q_in = sim.params_vec[5]
ssm.parameters = [1333, 200, 0.1, 0.1]
exogenous_variables = [Q_in, X_in2]



# control_variables = hcat([[0.0 for i in 1:100]]...)
# control_variables[80:90, 1] .= 1

# X_hat, Y_hat, P_hat = forecast(ssm, exogenous_variables, control_variables, steps_ahead=100)

# ŷt = hcat(sol_real_complete.u...)[10:10, 1:721]
# control_variables = transpose(hcat(getindex.(sol_real_complete.u, 14)...))[1:721, :]

# for i in 1:721
#     if i%10 != 0
#         ŷt[1, i] = NaN
#     end
# end

# # Add noise
# using Distributions
# ŷt = ŷt .+ rand(Normal(0, 0.2), size(ŷt))

