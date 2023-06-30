using Statistics:I
using Plots

mutable struct StateSpaceModel
    A_t::Function
    B_t::Function
    H_t::Function    
    R_t::Function
    Q_t::Function
    c_t::Function
    d_t::Function
    X_0::Vector{Float64}
    P_0::Matrix{Float64}
    dt::Float64
    t_start::Float64
    parameters::Vector{Float64}

    function StateSpaceModel(A_t, B_t, H_t, R_t, Q_t, c_t, d_t, X_0, P_0, dt, t_start)

        parameters = []

        return new(A_t, B_t, H_t, R_t, Q_t, c_t, d_t, X_0, P_0, dt, t_start, parameters)
    end
end


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


function kalman_smoother(ssm, yt, exogenous_variables, control_variables, X̂t, P̂t, K, L, S, v)

    n_obs = size(yt, 2)
    n_var = size(yt, 1)
    
    # Define variables
    smoothed_X̂t = Array{Array{Float64, 1}, 1}(undef, n_obs+1)
    smoothed_P̂t = Array{Array{Float64, 2}, 1}(undef, n_obs+1)

    # Initialize variables
    smoothed_X̂t[n_obs+1] = X̂t[n_obs+1]
    smoothed_P̂t[n_obs+1] = P̂t[n_obs+1]

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

        # Compute number of observed variables
        ivar_obs = findall(.!isnan.(yt[:, t]))
        n = length(ivar_obs)

        # Backward step
        r = transpose(H[ivar_obs, :])*inv(S[t])*v[t] + transpose(L[t])*r
        N = transpose(H[ivar_obs, :])*inv(S[t])*H[ivar_obs, :] + transpose(L[t])*N*L[t]

        # Update state and covariance
        smoothed_X̂t[t] = X̂t[t] + P̂t[t]*r
        smoothed_P̂t[t] = P̂t[t] - P̂t[t]*N*P̂t[t]

    end

    return smoothed_X̂t, smoothed_P̂t
end


function kalman_filter(ssm, yt, exogenous_variables, control_variables)

    n_obs = size(yt, 2)
    
    # Define variables
    predicted_X̂t = Array{Array{Float64, 1}, 1}(undef, n_obs+1)
    predicted_P̂t = Array{Array{Float64, 2}, 1}(undef, n_obs+1)
    filtered_X̂t = Array{Array{Float64, 1}, 1}(undef, n_obs)
    filtered_P̂t = Array{Array{Float64, 2}, 1}(undef, n_obs)
    K = Array{Array{Float64, 2}, 1}(undef, n_obs)
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

        # Compute innovations and Kalman Gain
        v[t] = yt[ivar_obs, t] - (H[ivar_obs, :]*predicted_X̂t[t] + ssm.d_t(exogenous_variables, ssm.parameters, t_step)[ivar_obs])
        S[t] = H[ivar_obs, :]*predicted_P̂t[t]*transpose(H[ivar_obs, :]) + Q[ivar_obs, ivar_obs]
        K[t] = predicted_P̂t[t]*transpose(H[ivar_obs, :])*inv(S[t])
        L[t] = A - K[t]*H[ivar_obs, :]

        # Update state and covariance of time t and make prediction for time t+1
        filtered_X̂t[t] = predicted_X̂t[t] + K[t]*v[t]
        filtered_P̂t[t] = (I - K[t]*H[ivar_obs, :])*predicted_P̂t[t]
        predicted_X̂t[t+1] = A*filtered_X̂t[t] + B*control_variables[t, :] + ssm.c_t(exogenous_variables, ssm.parameters, t_step)
        predicted_P̂t[t+1] = transpose(A)*filtered_P̂t[t]*A + ssm.R_t(exogenous_variables, ssm.parameters, t_step)

        # Multiply K by A for Kalman smoother
        K[t] = A*K[t]

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
        K = A*P*transpose(H[ivar_obs, :])*inv(S)
        L = A - K*H[ivar_obs, :]

        # Compute the log-likelihood
        if length(ivar_obs) > 0
            nb += 1
            llk += - log(2*pi)/2 - (1/2)*(log(det(S)) + transpose(v)*inv(det(S))*v)
        end
        

        # Update state and covariance of time t and make prediction for time t+1
        X = A*X + K*v + B*control_variables[t, :] + ssm.c_t(exogenous_variables, parameters, t_step)
        P = A*P*transpose(L) + ssm.R_t(exogenous_variables, parameters, t_step)


    end

    llk = llk/nb

    return llk

end
