using Statistics:I
using Plots


# abstract type StateSpaceModel end

mutable struct StateSpaceModel

    # Matrix defining observation and state equation
    A_t::Function
    B_t::Function
    H_t::Function    
    R_t::Function
    Q_t::Function
    c_t::Function
    d_t::Function

    # Initial states and covariance
    X_0::Vector{Float64}
    P_0::Matrix{Float64}

    # Time indexing
    dt::Float64
    t_start::Float64

    # Parameters of the model
    parameters::Vector{Float64}

    function StateSpaceModel(A_t, B_t, H_t, R_t, Q_t, c_t, d_t, X_0, P_0, dt, t_start)

        parameters = []

        return new(A_t, B_t, H_t, R_t, Q_t, c_t, d_t, X_0, P_0, dt, t_start, parameters)
    end
end

# mutable struct NonLinearStateSpaceModel <: StateSpaceModel

#     # Function defining states and observation equation
#     M_t::Function
#     H_t::Function    

#     # Covariances matrix
#     R_t::Function
#     Q_t::Function

#     # Initial states and covariance
#     X_0::Vector{Float64}
#     P_0::Matrix{Float64}

#     # Time indexing
#     dt::Float64
#     t_start::Float64

#     # Parameters of the model
#     parameters::Vector{Float64}

#     function NonLinearStateSpaceModel(M_t, H_t, R_t, Q_t, X_0, P_0, dt, t_start)

#         parameters = []

#         return new(M_t, H_t, R_t, Q_t, X_0, P_0, dt, t_start, parameters)
#     end
# end



function forecast(model::StateSpaceModel, exogenous_variables, control_variables; steps_ahead=1)

    X_hat = Array{Array{Float64, 1}, 1}(undef, steps_ahead+1)
    P_hat = Array{Array{Float64, 2}, 1}(undef, steps_ahead+1)
    Y_hat = Array{Array{Float64, 1}, 1}(undef, steps_ahead+1)

    X_hat[1] = model.X_0
    Y_hat[1] = model.H_t(exogenous_variables, model.parameters, model.t_start)*X_hat[1]
    P_hat[1] = model.P_0
    for step in 2:(steps_ahead+1)

        # Define current t_step
        t_step = model.t_start + (step-1)*model.dt

        # Get current matrix A and B
        A = model.A_t(exogenous_variables, model.parameters, t_step)
        B = model.B_t(exogenous_variables, model.parameters, t_step)
        
        # Update predicted state and covariance
        X_hat[step] = A*X_hat[step-1] + B*control_variables[step-1, :] + model.c_t(exogenous_variables, model.parameters, t_step)
        Y_hat[step] = model.H_t(exogenous_variables, model.parameters, t_step)*X_hat[step] + model.d_t(exogenous_variables, model.parameters, t_step)
        P_hat[step] = transpose(A)*P_hat[step-1]*A + model.R_t(exogenous_variables, model.parameters, t_step)

    end

    return X_hat, Y_hat, P_hat

end


# function forecast(model::NonLinearStateSpaceModel, exogenous_variables, control_variables; steps_ahead=1)

#     X_hat = Array{Array{Float64, 1}, 1}(undef, steps_ahead+1)
#     P_hat = Array{Array{Float64, 2}, 1}(undef, steps_ahead+1)
#     Y_hat = Array{Array{Float64, 1}, 1}(undef, steps_ahead+1)

#     X_hat[1] = model.X_0
#     Y_hat[1] = model.H_t(exogenous_variables, model.parameters, model.t_start)*X_hat[1]
#     P_hat[1] = model.P_0
#     for step in 2:(steps_ahead+1)

#         # Define current t_step
#         t_step = model.t_start + (step-1)*model.dt

#         # Get current matrix A and B
#         A = model.A_t(exogenous_variables, model.parameters, t_step)
#         B = model.B_t(exogenous_variables, model.parameters, t_step)
        
#         # Update predicted state and covariance
#         X_hat[step] = A*X_hat[step-1] + B*control_variables[step-1, :] + model.c_t(exogenous_variables, model.parameters, t_step)
#         Y_hat[step] = model.H_t(exogenous_variables, model.parameters, t_step)*X_hat[step] + model.d_t(exogenous_variables, model.parameters, t_step)
#         P_hat[step] = transpose(A)*P_hat[step-1]*A + model.R_t(exogenous_variables, model.parameters, t_step)

#     end

#     return X_hat, Y_hat, P_hat

# end


function kalman_smoother(ssm, yt, exogenous_variables, control_variables, X̂t, P̂t, L, S, v)

    n_obs = size(yt, 2)
    n_var = size(yt, 1)
    
    # Define variables
    smoothed_X̂t = Array{Array{Float64, 1}, 1}(undef, n_obs+1)
    smoothed_P̂t = Array{Array{Float64, 2}, 1}(undef, n_obs+1)
    N = Array{Array{Float64, 2}, 1}(undef, n_obs+1)

    # Initialize variables
    smoothed_X̂t[n_obs+1] = X̂t[n_obs+1]
    smoothed_P̂t[n_obs+1] = P̂t[n_obs+1]
    r = zeros(n_var)
    N[n_obs+1] = zeros(n_var, n_var)

    # Loop over observations
    for t in (n_obs):-1:1

        # Get current t_step
        t_step = ssm.t_start + (t-1)*ssm.dt

        # Get current matrix A and B
        A = ssm.A_t(exogenous_variables, ssm.parameters, t_step)
        H = ssm.H_t(exogenous_variables, ssm.parameters, t_step)
        Q = ssm.Q_t(exogenous_variables, ssm.parameters, t_step)

        # Compute number of observed variables
        ivar_obs = findall(.!isnan.(yt[:, t]))
        n = length(ivar_obs)

        # Backward step
        r = transpose(H[ivar_obs, :])*inv(S[t])*v[t] + transpose(L[t])*r
        N[t] = transpose(H[ivar_obs, :])*inv(S[t])*H[ivar_obs, :] + transpose(L[t])*N[t+1]*L[t]

        # Update state and covariance
        smoothed_X̂t[t] = X̂t[t] + P̂t[t]*r
        smoothed_P̂t[t] = P̂t[t] - P̂t[t]*N[t]*P̂t[t]

    end

    return smoothed_X̂t, smoothed_P̂t, N
end

function kalman_disturbance_smoother(ssm, yt, exogenous_variables, control_variables, K, L, S, v)

    n_obs = size(yt, 2)
    n_var = size(yt, 1)
    
    # Define variables
    smoothed_ϵt = Array{Array{Float64, 1}, 1}(undef, n_obs)
    smoothed_ηt = Array{Array{Float64, 1}, 1}(undef, n_obs)
    smoothed_var_ϵt = Array{Array{Float64, 2}, 1}(undef, n_obs)
    smoothed_var_ηt = Array{Array{Float64, 2}, 1}(undef, n_obs)

    # Loop over observations
    r = zeros(n_var)
    N = zeros(n_var, n_var)
    for t in (n_obs):-1:1

        # Get current t_step
        t_step = ssm.t_start + (t-1)*ssm.dt

        # Get current matrix A and B
        A = ssm.A_t(exogenous_variables, ssm.parameters, t_step)
        H = ssm.H_t(exogenous_variables, ssm.parameters, t_step)
        Q = ssm.Q_t(exogenous_variables, ssm.parameters, t_step)
        R = ssm.R_t(exogenous_variables, ssm.parameters, t_step)

        # Compute number of observed variables
        ivar_obs = findall(.!isnan.(yt[:, t]))
        n = length(ivar_obs)

        # Update state and covariance
        smoothed_ϵt[t] = Q[ivar_obs, ivar_obs]*(inv(S[t])*v[t] - transpose(K[t])*r)
        smoothed_ηt[t] = R*r
        smoothed_var_ϵt[t] = Q[ivar_obs, ivar_obs] - Q[ivar_obs, ivar_obs]*(inv(S[t]) + transpose(K[t])*N*K[t])*Q[ivar_obs, ivar_obs]
        smoothed_var_ηt[t] = R - R*N*R

        # Backward step
        r = transpose(H[ivar_obs, :])*inv(S[t])*v[t] + transpose(L[t])*r
        N = transpose(H[ivar_obs, :])*inv(S[t])*H[ivar_obs, :] + transpose(L[t])*N*L[t]

    end

    return smoothed_ϵt, smoothed_ηt, smoothed_var_ϵt, smoothed_var_ηt
end

function kalman_pair_smoother(ssm, yt, exogenous_variables, control_variables, N, P̂t, L)

    n_obs = size(yt, 2)
    n_var = size(yt, 1)
    
    # Define variables
    smoothed_pairwise_P̂t = Array{Array{Float64, 2}, 1}(undef, n_obs)

    for t in 1:n_obs
        smoothed_pairwise_P̂t[t] = P̂t[t]*transpose(L[t])*(I - N[t+1]*P̂t[t+1])
    end

    return smoothed_pairwise_P̂t

end


function kalman_filter(ssm, yt, exogenous_variables, control_variables)

    n_obs = size(yt, 2)
    
    # Define variables
    predicted_X̂t = Array{Array{Float64, 1}, 1}(undef, n_obs+1)
    predicted_P̂t = Array{Array{Float64, 2}, 1}(undef, n_obs+1)
    filtered_X̂t = Array{Array{Float64, 1}, 1}(undef, n_obs)
    filtered_P̂t = Array{Array{Float64, 2}, 1}(undef, n_obs)
    K = Array{Array{Float64, 2}, 1}(undef, n_obs)
    M = Array{Array{Float64, 2}, 1}(undef, n_obs)
    L = Array{Array{Float64, 2}, 1}(undef, n_obs)
    S = Array{Array{Float64, 2}, 1}(undef, n_obs)
    v = Array{Array{Float64, 1}, 1}(undef, n_obs)

    # Initialize variables
    predicted_X̂t[1] = ssm.X_0
    predicted_P̂t[1] = ssm.P_0

    # Loop over observations
    for t in 1:n_obs

        # Get current t_step
        t_step = ssm.t_start + (t-1)*ssm.dt

        # Get current matrix A, B, H and Q
        A = ssm.A_t(exogenous_variables, ssm.parameters, t_step)
        B = ssm.B_t(exogenous_variables, ssm.parameters, t_step)
        H = ssm.H_t(exogenous_variables, ssm.parameters, t_step)
        Q = ssm.Q_t(exogenous_variables, ssm.parameters, t_step)

        # Check the number of correct observations
        ivar_obs = findall(.!isnan.(yt[:, t]))

        # Compute innovations and stuff for predicted and filtered states
        v[t] = yt[ivar_obs, t] - (H[ivar_obs, :]*predicted_X̂t[t] + ssm.d_t(exogenous_variables, ssm.parameters, t_step)[ivar_obs])
        S[t] = H[ivar_obs, :]*predicted_P̂t[t]*transpose(H[ivar_obs, :]) + Q[ivar_obs, ivar_obs]
        M[t] = predicted_P̂t[t]*transpose(H[ivar_obs, :])
        inv_S = inv(S[t])

        # Update state and covariance of time t and make prediction for time t+1
        filtered_X̂t[t] = predicted_X̂t[t] + M[t]*inv_S*v[t]
        filtered_P̂t[t] = predicted_P̂t[t] - M[t]*inv_S*transpose(M[t])
        predicted_X̂t[t+1] = A*filtered_X̂t[t] + B*control_variables[t, :] + ssm.c_t(exogenous_variables, ssm.parameters, t_step)
        predicted_P̂t[t+1] = transpose(A)*filtered_P̂t[t]*A + ssm.R_t(exogenous_variables, ssm.parameters, t_step)

        # Compute stuff for Kalman smoother
        K[t] = A*M[t]*inv_S
        L[t] = A - K[t]*H[ivar_obs, :]

    end

    return filtered_X̂t, filtered_P̂t, predicted_X̂t, predicted_P̂t, K, L, S, v

end


function get_coverage_probability(xt, lwb, upb)

    # Compute the coverage probability
    nt = size(lwb, 2)
    nv = size(lwb, 1)
    cp = zeros(Float64, nv)
    for i in 1:nv
        cp[i] = sum((lwb[i, :] .<= xt[i, :]) .& (xt[i, :] .<= upb[i, :])) / nt
    end

    return cp

end


function rmse(x̂, xt)

    return sqrt.(mean((vcat(x̂'...) .- vcat(xt'...)) .^ 2, dims=1))

end


function kalman_filter_optim(yt, ssm, exogenous_variables, control_variables, parameters)

    n_obs = size(yt, 2)

    tol = 10^(-16)
    # Initialize variables
    X = ssm.X_0
    P = ssm.P_0
    llk = 0.0
    nb = 0

    # Loop over observations
    for t in 1:n_obs

        # Get current t_step
        t_step = ssm.t_start + (t-1)*ssm.dt

        # Get current matrix A, B, H and Q
        A = ssm.A_t(exogenous_variables, parameters, t_step)
        B = ssm.B_t(exogenous_variables, parameters, t_step)
        H = ssm.H_t(exogenous_variables, parameters, t_step)
        Q = ssm.Q_t(exogenous_variables, parameters, t_step)

        # Check the number of correct observations
        ivar_obs = findall(.!isnan.(yt[:, t]))

        # Compute innovations and Kalman Gain
        v = yt[ivar_obs, t] - (H[ivar_obs, :]*X + ssm.d_t(exogenous_variables, parameters, t_step)[ivar_obs])
        S = H[ivar_obs, :]*P*transpose(H[ivar_obs, :]) + Q[ivar_obs, ivar_obs]
        K = A*P*transpose(H[ivar_obs, :])*pinv(S)
        L = A - K*H[ivar_obs, :]

        # Compute the log-likelihood
        if length(ivar_obs) > 0
            nb += 1
            llk += - log(2*pi)/2 - (1/2)*(log(det(S) + tol) + transpose(v)*pinv(S)*v)
        end
        

        # Update state and covariance of time t and make prediction for time t+1
        X = A*X + K*v + B*control_variables[t, :] + ssm.c_t(exogenous_variables, parameters, t_step)
        P = A*P*transpose(L) + ssm.R_t(exogenous_variables, parameters, t_step)


    end

    llk = llk/nb

    return llk

end


function E_step(yt, ssm, exogenous_variables, control_variables, parameters)

    # Set parameters of state space model
    ssm.parameters = parameters

    # Kalman filtering pass
    filtered_X̂t, filtered_P̂t, predicted_X̂t, predicted_P̂t, K, L, S, v = kalman_filter(ssm, yt, exogenous_variables, control_variables)

    # Kalman smoothing pass
    smoothed_X̂t, smoothed_P̂t, N = kalman_smoother(ssm, yt, exogenous_variables, control_variables, predicted_X̂t, predicted_P̂t, L, S, v)

    # Kalman autocovariance smoothing pass
    smoothed_pairwise_P̂t = kalman_pair_smoother(ssm, yt, exogenous_variables, control_variables, N, predicted_P̂t, L)

    return smoothed_X̂t, smoothed_P̂t, smoothed_pairwise_P̂t

end

function E_step2(yt, ssm, exogenous_variables, control_variables, parameters)

    # Set parameters of state space model
    ssm.parameters = parameters

    # Kalman filtering pass
    filtered_X̂t, filtered_P̂t, predicted_X̂t, predicted_P̂t, K, L, S, v = kalman_filter(ssm, yt, exogenous_variables, control_variables)

    # Kalman smoothing pass
    smoothed_X̂t, smoothed_P̂t, N = kalman_smoother(ssm, yt, exogenous_variables, control_variables, predicted_X̂t, predicted_P̂t, L, S, v)

    smoothed_ϵt, smoothed_ηt, smoothed_var_ϵt, smoothed_var_ηt = kalman_disturbance_smoother(ssm, yt, exogenous_variables, control_variables, K, L, S, v)

    # Kalman autocovariance smoothing pass
    smoothed_pairwise_P̂t = kalman_pair_smoother(ssm, yt, exogenous_variables, control_variables, N, predicted_P̂t, L)

    return smoothed_X̂t, smoothed_P̂t, smoothed_pairwise_P̂t, smoothed_ϵt, smoothed_ηt, smoothed_var_ϵt, smoothed_var_ηt

end

function M_step(yt, ssm, exogenous_variables, control_variables, parameters, X̂t, P̂t, auto_P̂t)

    n_obs = size(yt, 2)

    # E_i = map(x -> x*transpose(x), X̂t[2:end]) + P̂t[2:end]
    # D_i = map((x, y) -> x*transpose(y), X̂t[2:end], X̂t[1:(end-1)]) + auto_P̂t
    # F_i = map(x -> x*transpose(x), X̂t[1:(end-1)]) + P̂t[1:(end-1)]

    # I_i = map((x, y) -> x*transpose(y), X̂t[2:end], control_variables)
    # J_i = map((x, y) -> x*transpose(y), X̂t[1:(end-1)], control_variables)
    # K_i = map(x -> x*transpose(x), control_variables)

    function Q(parameters, unused)

        t_index = collect(ssm.t_start:ssm.dt:(ssm.t_start+(n_obs-1)*ssm.dt))

        # A_i = map(t -> ssm.A_t(exogenous_variables, parameters, t), t_index)
        # B_i = map(t -> ssm.B_t(exogenous_variables, parameters, t), t_index)
        # c_i = map(t -> ssm.c_t(exogenous_variables, parameters, t), t_index)
        R_i = map(t -> ssm.R_t(exogenous_variables, parameters, t), t_index)

        # H_i = map(t -> ssm.H_t(exogenous_variables, parameters, t), t_index)
        # d_i = map(t -> ssm.d_t(exogenous_variables, parameters, t), t_index)
        Q_i = map(t -> ssm.Q_t(exogenous_variables, parameters, t), t_index)

        L = -(1/2)*( sum(log.(1 .+ det.(R_i))) + sum(log.(1 .+ det.(Q_i))))

        # L = 0
        # for t in 1:n_obs

        #     # Get current t_step
        #     t_step = ssm.t_start + (t-1)*ssm.dt

        #     R_i = ssm.R_t(exogenous_variables, parameters, t_step)
        #     Q_i = ssm.Q_t(exogenous_variables, parameters, t_step)

        #     # Log variance term
        #     L = -(1/2)*( sum(log.(1 .+ det(R_i))) + sum(log.(1 .+ det(Q_i))))
        # end
        
        

        # # Observation equation
        # for t in 1:n_obs

        #     ivar_obs = findall(.!isnan.(yt[:, t]))

        #     ϵ_i = yt[ivar_obs, t] - H_i[t][ivar_obs, :]*X̂t[t] - d_i[t][ivar_obs]
        #     N_i = ϵ_i*transpose(ϵ_i) + H_i[t][ivar_obs, :]*P̂t[t]*transpose(H_i[t][ivar_obs, :])

        #     L = L - (1/2)*sum(N_i*inv(Q_i[t][ivar_obs, ivar_obs])) - log(2*pi)/2
        # end
        

        # # State equation
        # for t in 1:n_obs

        #     M_i = (X̂t[t+1] - A_i[t]*X̂t[t])*transpose(-B_i[t]*control_variables[t, :] - c_i[t])
        #     M_i += (-B_i[t]*control_variables[t, :] - c_i[t])*transpose(X̂t[t+1] - A_i[t]*X̂t[t])
        #     M_i += (-B_i[t]*control_variables[t, :] - c_i[t])*transpose(-B_i[t]*control_variables[t, :] - c_i[t])
        #     M_i += E_i[t] - D_i[t]*transpose(A_i[t]) - A_i[t]*transpose(D_i[t]) - A_i[t]*F_i[t]
            
        #     L = L - (1/2)*sum(M_i*inv(R_i[t])) - log(2*pi)/2
        # end

        return - L/n_obs

    end

    optprob = OptimizationFunction(Q, Optimization.AutoForwardDiff())
    prob = Optimization.OptimizationProblem(optprob, parameters, [], sense=Optimization.MinSense)
    sol = solve(prob, Optim.Newton(), maxiters = 1, progress= true, store_trace=true)

    return sol #Q(parameters, [])

end

function M_step2(yt, ssm, exogenous_variables, control_variables, init_parameters, X̂t, P̂t, auto_P̂t, smoothed_ϵt, smoothed_ηt, smoothed_var_ϵt, smoothed_var_ηt)

    n_obs = size(yt, 2)

    function Q(parameters, unused)

        parameters = [init_parameters[1], init_parameters[2], parameters[1], parameters[2]]

        # t_index = collect(ssm.t_start:ssm.dt:(ssm.t_start+(n_obs-1)*ssm.dt))

        # A_i = map(t -> ssm.A_t(exogenous_variables, parameters, t), t_index)
        # B_i = map(t -> ssm.B_t(exogenous_variables, parameters, t), t_index)
        # c_i = map(t -> ssm.c_t(exogenous_variables, parameters, t), t_index)
        # H_i = map(t -> ssm.H_t(exogenous_variables, parameters, t), t_index)
        # d_i = map(t -> ssm.d_t(exogenous_variables, parameters, t), t_index)

        L = 0
        for t in 1:n_obs

            # Get current t_step
            t_step = ssm.t_start + (t-1)*ssm.dt

            ivar_obs = findall(.!isnan.(yt[:, t]))

            R_i = ssm.R_t(exogenous_variables, parameters, t_step)
            Q_i = ssm.Q_t(exogenous_variables, parameters, t_step)

            # A_i = ssm.A_t(exogenous_variables, parameters, t_step)
            # B_i = ssm.B_t(exogenous_variables, parameters, t_step)
            # c_i = ssm.c_t(exogenous_variables, parameters, t_step)
            # H_i = ssm.H_t(exogenous_variables, parameters, t_step)
            # d_i = ssm.d_t(exogenous_variables, parameters, t_step)

            # ϵ_i = yt[ivar_obs, t] - (H_i[ivar_obs, :]*X̂t[t] + d_i[ivar_obs])
            # V_ϵ_i = H_i[ivar_obs, :]*P̂t[t]*transpose(H_i[ivar_obs, :])
            # η_i = X̂t[t+1] - (A_i*X̂t[t] + B_i*control_variables[t] + c_i)
            if size(ivar_obs, 1) > 0
                L += -(1/2)*(log(2*pi) + sum(log(det(Q_i[ivar_obs, ivar_obs]))) +  tr((smoothed_ϵt[t]*transpose(smoothed_ϵt[t]) + smoothed_var_ϵt[t])*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L += -(1/2)*(log(2*pi) + sum(log(det(R_i))) + tr((smoothed_ηt[t]*transpose(smoothed_ηt[t]) + smoothed_var_ηt[t])*pinv(R_i)) )

        end

        return - L/n_obs

    end

    print("Init loss : ", Q(init_parameters[3:4], []))

    optprob = OptimizationFunction(Q, Optimization.AutoForwardDiff())
    prob = Optimization.OptimizationProblem(optprob, init_parameters[3:4], [])
    sol = solve(prob, Optim.Newton(), maxiters = 10, progress= true, store_trace=true)

    return sol #Q(parameters, [])

end


function M_step3(yt, ssm, exogenous_variables, control_variables, parameters, X̂t, P̂t, auto_P̂t, smoothed_ϵt, smoothed_ηt, smoothed_var_ϵt, smoothed_var_ηt)

    n_obs = size(yt, 2)

    function Q(parameters, fixed_params)

        X̂t = fixed_params[1]
        P̂t = fixed_params[2]
        auto_P̂t = fixed_params[3]
        yt = fixed_params[4]
        control_variables = fixed_params[5]
        exogenous_variables = fixed_params[6]
        ssm = fixed_params[7]

        L = 0
        for t in 1:n_obs

            # Get current t_step
            t_step = ssm.t_start + (t-1)*ssm.dt

            ivar_obs = findall(.!isnan.(yt[:, t]))

            R_i = ssm.R_t(exogenous_variables, parameters, t_step)
            Q_i = ssm.Q_t(exogenous_variables, parameters, t_step)

            A_i = ssm.A_t(exogenous_variables, parameters, t_step)
            B_i = ssm.B_t(exogenous_variables, parameters, t_step)
            c_i = ssm.c_t(exogenous_variables, parameters, t_step)
            H_i = ssm.H_t(exogenous_variables, parameters, t_step)
            d_i = ssm.d_t(exogenous_variables, parameters, t_step)

            ϵ_i = yt[ivar_obs, t] - (H_i[ivar_obs, :]*X̂t[t] + d_i[ivar_obs])
            V_ϵ_i = H_i[ivar_obs, :]*P̂t[t]*transpose(H_i[ivar_obs, :])# + Q_i[ivar_obs, ivar_obs]
            η_i = X̂t[t+1] - (A_i*X̂t[t] + B_i*control_variables[t] + c_i)
            V_η_i = P̂t[t+1] - auto_P̂t[t]*transpose(A_i) - A_i*transpose(auto_P̂t[t]) + A_i*P̂t[t]*transpose(A_i)
            if size(ivar_obs, 1) > 0
                L += -(1/2)*(log(2*pi) + sum(log(det(Q_i[ivar_obs, ivar_obs]))) +  tr((ϵ_i*transpose(ϵ_i) + V_ϵ_i)*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L += -(1/2)*(log(2*pi) + sum(log(det(R_i))) + tr((η_i*transpose(η_i) + V_η_i)*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # print("Init loss : ", Q(init_parameters[3:4], []))

    optprob = OptimizationFunction(Q, Optimization.AutoForwardDiff())
    prob = Optimization.OptimizationProblem(optprob, parameters, [X̂t, P̂t, auto_P̂t, yt, control_variables, exogenous_variables, ssm])
    sol = solve(prob, Optim.LBFGS(), maxiters = 100)

    return sol

end

function M_step4(yt, ssm, exogenous_variables, control_variables, parameters, X̂t, P̂t, auto_P̂t, smoothed_ϵt, smoothed_ηt, smoothed_var_ϵt, smoothed_var_ηt)

    n_obs = size(yt, 2)

    function Q(parameters, unused)

        t_index = collect(ssm.t_start:ssm.dt:(ssm.t_start+(n_obs-1)*ssm.dt))

        ivar_obs = map(t -> findall(.!isnan.(yt[:, t])), collect(1:n_obs))
        nb_var_obs = sum(size.(ivar_obs, 1))

        A_i = map(t -> ssm.A_t(exogenous_variables, parameters, t), t_index)
        B_i = map(t -> ssm.B_t(exogenous_variables, parameters, t), t_index)
        c_i = map(t -> ssm.c_t(exogenous_variables, parameters, t), t_index)
        R_i = map(t -> ssm.R_t(exogenous_variables, parameters, t), t_index)
        H_i = map((t, var_obs) -> ssm.H_t(exogenous_variables, parameters, t)[var_obs, :], t_index, ivar_obs)
        d_i = map((t, var_obs) -> ssm.d_t(exogenous_variables, parameters, t)[var_obs], t_index, ivar_obs)
        Q_i = map((t, var_obs) -> ssm.Q_t(exogenous_variables, parameters, t)[var_obs, var_obs], t_index, ivar_obs)
        y_i = [yt[ivar_obs[i], i] for i in 1:n_obs]
        U_i = [control_variables[i, :] for i in 1:n_obs]

        V_ϵ_i = H_i.*P̂t[1:(end-1)].*transpose.(H_i) + Q_i
        ϵ_i = y_i .- (H_i.*X̂t[1:(end-1)] + d_i)

        η_i = X̂t[2:end] - (A_i.*X̂t[1:(end-1)] + B_i.*U_i + c_i)
        V_η_i = P̂t[2:end] - auto_P̂t.*transpose.(A_i) - A_i.*transpose.(auto_P̂t) + A_i.*P̂t[1:(end-1)].*transpose.(A_i)

        L = -(1/2)*(log(2*pi)*nb_var_obs + sum(log.(det.(Q_i)) +  tr.((ϵ_i.*transpose.(ϵ_i) + V_ϵ_i).*pinv.(Q_i))) ) +
            -(1/2)*(log(2*pi)*n_obs + sum(log.(det.(R_i)) + tr.((η_i.*transpose.(η_i) + V_η_i).*pinv.(R_i))) )


        return - L/n_obs

    end

    # print("Init loss : ", Q(init_parameters[3:4], []))

    optprob = OptimizationFunction(Q, Optimization.AutoForwardDiff())
    # prob = Optimization.OptimizationProblem(optprob, init_parameters[3:4], [])
    prob = Optimization.OptimizationProblem(optprob, parameters, [])
    sol = solve(prob, Optim.Newton(), maxiters = 100, progress= true, store_trace=true)

    return sol #Q(parameters, [])

end


function EM(yt, ssm, exogenous_variables, control_variables, parameters)

    for i in 1:1

        smoothed_X̂t, smoothed_P̂t, smoothed_pairwise_P̂t = E_step(yt, ssm, exogenous_variables, control_variables, parameters)

        sol = M_step(yt, ssm, exogenous_variables, control_variables, parameters, smoothed_X̂t, smoothed_P̂t, smoothed_pairwise_P̂t)
        println(sol)
        parameters = sol.minimizer
    
    end
end

function EM2(yt, ssm, exogenous_variables, control_variables, parameters)

    llk = []
    for i in 1:20

        push!(llk, llh(parameters, ssm))
        println("Likelihood : ", llk[end])

        smoothed_X̂t, smoothed_P̂t, smoothed_pairwise_P̂t, smoothed_ϵt, smoothed_ηt, smoothed_var_ϵt, smoothed_var_ηt = E_step2(yt, ssm, exogenous_variables, control_variables, parameters)

        # println(sqrt(mean(hcat(map(x -> x*transpose(x), smoothed_ϵt)...))))
        # println(sqrt(mean(hcat(map(x -> x*transpose(x), smoothed_ηt)...))))
        # println(sqrt(mean(hcat(map(x -> x*transpose(x), smoothed_var_ϵt)...))))
        # println(sqrt(mean(hcat(map(x -> x*transpose(x), smoothed_var_ηt)...))))

        sol = M_step2(yt, ssm, exogenous_variables, control_variables, parameters, smoothed_X̂t, smoothed_P̂t, smoothed_pairwise_P̂t, smoothed_ϵt, smoothed_ηt, smoothed_var_ϵt, smoothed_var_ηt)
        println(sol)
        parameters = [parameters[1], parameters[2], sol.minimizer[1], sol.minimizer[2]]
    
    end

    return llk
end

function EM3(yt, ssm, exogenous_variables, control_variables, parameters)

    llk = []
    for i in 1:10

        push!(llk, llh(parameters, ssm))
        println("Likelihood : ", llk[end])

        smoothed_X̂t, smoothed_P̂t, smoothed_pairwise_P̂t, smoothed_ϵt, smoothed_ηt, smoothed_var_ϵt, smoothed_var_ηt = E_step2(yt, ssm, exogenous_variables, control_variables, parameters)

        sol = M_step3(yt, ssm, exogenous_variables, control_variables, parameters, smoothed_X̂t, smoothed_P̂t, smoothed_pairwise_P̂t, smoothed_ϵt, smoothed_ηt, smoothed_var_ϵt, smoothed_var_ηt)
        println(sol)
        parameters = sol.minimizer
    
    end

    return llk
end

function EM4(yt, ssm, exogenous_variables, control_variables, parameters)

    llk = []
    for i in 1:10

        push!(llk, llh(parameters, ssm))
        println("Likelihood : ", llk[end])

        smoothed_X̂t, smoothed_P̂t, smoothed_pairwise_P̂t, smoothed_ϵt, smoothed_ηt, smoothed_var_ϵt, smoothed_var_ηt = E_step2(yt, ssm, exogenous_variables, control_variables, parameters)

        # println(sqrt(mean(hcat(map(x -> x*transpose(x), smoothed_ϵt)...))))
        # println(sqrt(mean(hcat(map(x -> x*transpose(x), smoothed_ηt)...))))
        # println(sqrt(mean(hcat(map(x -> x*transpose(x), smoothed_var_ϵt)...))))
        # println(sqrt(mean(hcat(map(x -> x*transpose(x), smoothed_var_ηt)...))))

        sol = M_step4(yt, ssm, exogenous_variables, control_variables, parameters, smoothed_X̂t, smoothed_P̂t, smoothed_pairwise_P̂t, smoothed_ϵt, smoothed_ηt, smoothed_var_ϵt, smoothed_var_ηt)
        println(sol)
        parameters = sol.minimizer#[parameters[1], parameters[2], sol.minimizer[1], sol.minimizer[2]]
    
    end

    return llk
end


function EM5(yt, ssm, exogenous_variables, control_variables, parameters)

    n_obs = size(yt, 2)

    function Q(parameters, fixed_params)

        X̂t = fixed_params[1]
        P̂t = fixed_params[2]
        auto_P̂t = fixed_params[3]
        yt = fixed_params[4]
        control_variables = fixed_params[5]
        exogenous_variables = fixed_params[6]
        ssm = fixed_params[7]

        L = 0
        for t in 1:n_obs

            # Get current t_step
            t_step = ssm.t_start + (t-1)*ssm.dt

            ivar_obs = findall(.!isnan.(yt[:, t]))

            R_i = ssm.R_t(exogenous_variables, parameters, t_step)
            Q_i = ssm.Q_t(exogenous_variables, parameters, t_step)

            A_i = ssm.A_t(exogenous_variables, parameters, t_step)
            B_i = ssm.B_t(exogenous_variables, parameters, t_step)
            c_i = ssm.c_t(exogenous_variables, parameters, t_step)
            H_i = ssm.H_t(exogenous_variables, parameters, t_step)
            d_i = ssm.d_t(exogenous_variables, parameters, t_step)

            ϵ_i = yt[ivar_obs, t] - (H_i[ivar_obs, :]*X̂t[t] + d_i[ivar_obs])
            V_ϵ_i = H_i[ivar_obs, :]*P̂t[t]*transpose(H_i[ivar_obs, :])# + Q_i[ivar_obs, ivar_obs]
            η_i = X̂t[t+1] - (A_i*X̂t[t] + B_i*control_variables[t] + c_i)
            V_η_i = P̂t[t+1] - auto_P̂t[t]*transpose(A_i) - A_i*transpose(auto_P̂t[t]) + A_i*P̂t[t]*transpose(A_i)
            if size(ivar_obs, 1) > 0
                L += -(1/2)*(log(2*pi) + sum(log(det(Q_i[ivar_obs, ivar_obs]))) +  tr((ϵ_i*transpose(ϵ_i) + V_ϵ_i)*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L += -(1/2)*(log(2*pi) + sum(log(det(R_i))) + tr((η_i*transpose(η_i) + V_η_i)*pinv(R_i)) )

        end

        return - L/n_obs

    end

    optprob = OptimizationFunction(Q, Optimization.AutoForwardDiff())

    llk = []
    for i in 1:30

        push!(llk, llh(parameters, ssm))
        println("Likelihood : ", llk[end])

        smoothed_X̂t, smoothed_P̂t, smoothed_pairwise_P̂t = E_step(yt, ssm, exogenous_variables, control_variables, parameters)

        prob = Optimization.OptimizationProblem(optprob, parameters, [smoothed_X̂t, smoothed_P̂t, smoothed_pairwise_P̂t, yt, control_variables, exogenous_variables, ssm])
        sol = solve(prob, Optim.Newton(), maxiters = 5)
        parameters = sol.minimizer
    
    end

    return llk, parameters
end