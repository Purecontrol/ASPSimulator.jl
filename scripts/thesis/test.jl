using ASM1Simulator, DifferentialEquations
using BenchmarkTools, LinearAlgebra, Optim, Optimization



# Fixed values
y_t=y_train
n_obs = size(y_t, 1)

ivar_obs_vec = [findall(.!isnan.(y_t[t, :])) for t in 1:n_obs]
valid_obs_vec = [length(ivar_obs_vec) > 0 for t in 1:n_obs]

model.parameters = opt_params_nmle
filter_output = StateSpaceIdentification.filter(model, y_train, E_train, U_train)
smoother_output = smoother(model, y_train, E_train, U_train, filter_output)

exogenous_variables = E_train
control_variables = U_train


@benchmark Q(model.parameters, smoother_output)


optprob = OptimizationFunction(Q, Optimization.AutoZygote())
prob = Optimization.OptimizationProblem(optprob, parameters, smoother_output)
@benchmark sol = solve(prob, Optim.Newton())



# Q function
function Q(parameters, smoothed_values)

    t=1
    Q_i = model.system.Q_t(exogenous_variables[t, :], parameters)
    R_i = model.system.R_t(exogenous_variables[t, :], parameters)
    H_i = model.system.H_t(exogenous_variables[t, :], parameters)
    d_i = model.system.d_t(exogenous_variables[t, :], parameters)
    B_i = model.system.B_t(exogenous_variables[t, :], parameters)

    L = 0
    @inbounds for t in 1:n_obs

        ivar_obs = ivar_obs_vec[t]

        # R_i = model.system.R_t(exogenous_variables[t, :], parameters)
        A_i = model.system.A_t(exogenous_variables[t, :], parameters)
        # B_i = model.system.B_t(exogenous_variables[t, :], parameters)
        c_i = model.system.c_t(exogenous_variables[t, :], parameters)

        η_i = smoothed_values.smoothed_state[t+1].μ_t - (A_i*smoothed_values.smoothed_state[t].μ_t + B_i*control_variables[t, :] + c_i)
        V_η_i = smoothed_values.smoothed_state[t+1].σ_t - smoothed_values.autocov_state[t]*transpose(A_i) - A_i*transpose(smoothed_values.autocov_state[t]) + A_i*smoothed_values.smoothed_state[t].σ_t*transpose(A_i)
        
        if valid_obs_vec[t]

            # H_i = model.system.H_t(exogenous_variables[t, :], parameters)
            # d_i = model.system.d_t(exogenous_variables[t, :], parameters)
            # Q_i = model.system.Q_t(exogenous_variables[t, :], parameters)

            ϵ_i = y_t[t, ivar_obs] - (H_i[ivar_obs, :]*smoothed_values.smoothed_state[t].μ_t + d_i[ivar_obs])
            V_ϵ_i = H_i[ivar_obs, :]*smoothed_values.smoothed_state[t].σ_t*transpose(H_i[ivar_obs, :])

            L -= (sum(logdet(Q_i[ivar_obs, ivar_obs])) +  tr((ϵ_i*transpose(ϵ_i) + V_ϵ_i)*pinv(Q_i[ivar_obs, ivar_obs])) )
        end
        L -= (sum(logdet(R_i)) + tr((η_i*transpose(η_i) + V_η_i)*pinv(R_i)) )

    end

    return - L/n_obs

end