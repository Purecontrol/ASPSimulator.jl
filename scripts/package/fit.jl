using Optimization, OptimizationOptimJL, OptimizationOptimisers
using SparseArrays


function numerical_MLE(model::ForecastingModel, y_t, exogenous_variables, control_variables)

    function inverse_llk(params_vec, unused)

        return - loglike(model, y_t, exogenous_variables, control_variables; parameters = params_vec)
    
    end

    optprob = OptimizationFunction(inverse_llk, Optimization.AutoForwardDiff())
    prob = Optimization.OptimizationProblem(optprob, model.parameters, [])
    sol = solve(prob, Optim.BFGS(), maxiters = 1000, progress= true, store_trace=true)

    return sol

end


function EM(model::ForecastingModel{GaussianLinearStateSpaceSystem}, y_t, exogenous_variables, control_variables)

    # Fixed values
    n_obs = size(y_t, 1)
    t_start = model.current_state.t
    dt = model.system.dt

    # Q function
    function Q(parameters, smoothed_values)

        L = 0
        for t in 1:n_obs

            # Get current t_step
            t_step = t_start + (t-1)*dt

            ivar_obs = findall(.!isnan.(y_t[t, :]))

            R_i = model.system.R_t(exogenous_variables, parameters, t_step)
            Q_i = model.system.Q_t(exogenous_variables, parameters, t_step)

            A_i = model.system.A_t(exogenous_variables, parameters, t_step)
            B_i = model.system.B_t(exogenous_variables, parameters, t_step)
            c_i = model.system.c_t(exogenous_variables, parameters, t_step)
            H_i = model.system.H_t(exogenous_variables, parameters, t_step)
            d_i = model.system.d_t(exogenous_variables, parameters, t_step)

            ϵ_i = y_t[t, ivar_obs] - (H_i[ivar_obs, :]*smoothed_values.smoothed_state[t].μ_t + d_i[ivar_obs])
            V_ϵ_i = H_i[ivar_obs, :]*smoothed_values.smoothed_state[t].σ_t*transpose(H_i[ivar_obs, :])# + Q_i[ivar_obs, ivar_obs]
            η_i = smoothed_values.smoothed_state[t+1].μ_t - (A_i*smoothed_values.smoothed_state[t].μ_t + B_i*control_variables[t] + c_i)
            V_η_i = smoothed_values.smoothed_state[t+1].σ_t - smoothed_values.autocov_state[t]*transpose(A_i) - A_i*transpose(smoothed_values.autocov_state[t]) + A_i*smoothed_values.smoothed_state[t].σ_t*transpose(A_i)
            if size(ivar_obs, 1) > 0
                L += -(1/2)*(log(2*pi) + sum(log(det(Q_i[ivar_obs, ivar_obs]))) +  tr((ϵ_i*transpose(ϵ_i) + V_ϵ_i)*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L += -(1/2)*(log(2*pi) + sum(log(det(R_i))) + tr((η_i*transpose(η_i) + V_η_i)*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # M function
    optprob = OptimizationFunction(Q, Optimization.AutoForwardDiff())

    llk_array = []
    parameters = model.parameters
    for i in 1:50

        filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters)

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        smoother_ouput = smoother(model, y_t, exogenous_variables, control_variables, filter_output; parameters=parameters)

        prob = Optimization.OptimizationProblem(optprob, parameters, smoother_ouput)
        sol = solve(prob, Optim.Newton(), maxiters = 5)
        parameters = sol.minimizer
    
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters)

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters

end


function EM_EnKS(model::ForecastingModel{GaussianLinearStateSpaceSystem}, y_t, exogenous_variables, control_variables; n_particles=30)

    # Fixed values
    n_obs = size(y_t, 1)
    t_start = model.current_state.t
    dt = model.system.dt

    # Q function
    function Q(parameters, smoothed_values)

        L = 0
        for t in 1:n_obs

            # Get current t_step
            t_step = t_start + (t-1)*dt

            ivar_obs = findall(.!isnan.(y_t[t, :]))

            R_i = model.system.R_t(exogenous_variables, parameters, t_step)
            Q_i = model.system.Q_t(exogenous_variables, parameters, t_step)

            M_i = transition(model.system, smoothed_values.smoothed_state[t].particles_state, exogenous_variables, control_variables[t, :], parameters, t_step)
            H_i = observation(model.system, smoothed_values.smoothed_state[t].particles_state, exogenous_variables, parameters, t_step)[ivar_obs, :]

            ϵ_i = y_t[t, ivar_obs] .- H_i
            Σ = (ϵ_i*ϵ_i') ./ (n_particles - 1)
    
            η_i = smoothed_values.smoothed_state[t+1].particles_state - M_i
            Ω = (η_i*η_i') ./ (n_particles - 1)

            if size(ivar_obs, 1) > 0
                L += -(1/2)*(log(2*pi) + sum(log(det(Q_i[ivar_obs, ivar_obs]))) +  tr(Σ*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L += -(1/2)*(log(2*pi) + sum(log(det(R_i))) + tr(Ω*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # M function
    optprob = OptimizationFunction(Q, Optimization.AutoForwardDiff())

    llk_array = []
    parameters = model.parameters
    for i in 1:50

        filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=EnsembleKalmanFilter(model.current_state, model.system.n_X, model.system.n_Y, n_particles))

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        smoother_ouput = smoother(model, y_t, exogenous_variables, control_variables, filter_output; parameters=parameters, smoother_method=EnsembleKalmanSmoother(model.system.n_X, model.system.n_Y, n_particles))

        prob = Optimization.OptimizationProblem(optprob, parameters, smoother_ouput)
        sol = solve(prob, Optim.Newton(), maxiters = 5)
        parameters = sol.minimizer
    
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters)

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters

end

function speed_EM(model::ForecastingModel{GaussianLinearStateSpaceSystem}, y_t, exogenous_variables, control_variables)

    # Fixed values
    n_obs = size(y_t, 1)
    t_start = model.current_state.t
    dt = model.system.dt

    ivar_obs_vec = [findall(.!isnan.(y_t[t, :])) for t in 1:n_obs]
    
    #
    C = log(2*pi)

    # Q function
    function Q(parameters, smoothed_values)

        smoothed_μ = smoothed_values[1]
        smoothed_σ = smoothed_values[2]
        smoothed_auto_σ = smoothed_values[3]

        L = 0
        for t in 1:n_obs

            # Get current t_step
            t_step = t_start + (t-1)*dt

            ivar_obs = ivar_obs_vec[t]

            R_i = model.system.R_t(exogenous_variables, parameters, t_step)
            Q_i = model.system.Q_t(exogenous_variables, parameters, t_step)

            A_i = model.system.A_t(exogenous_variables, parameters, t_step)
            B_i = model.system.B_t(exogenous_variables, parameters, t_step)
            c_i = model.system.c_t(exogenous_variables, parameters, t_step)
            H_i = model.system.H_t(exogenous_variables, parameters, t_step)
            d_i = model.system.d_t(exogenous_variables, parameters, t_step)

            ϵ_i = y_t[t, ivar_obs] - (H_i[ivar_obs, :]*smoothed_μ[t] + d_i[ivar_obs])
            V_ϵ_i = H_i[ivar_obs, :]*smoothed_σ[t]*transpose(H_i[ivar_obs, :])
            η_i = smoothed_μ[t+1] - (A_i*smoothed_μ[t] + B_i*control_variables[t] + c_i)
            V_η_i = smoothed_σ[t+1] - smoothed_auto_σ[t]*transpose(A_i) - A_i*transpose(smoothed_auto_σ[t]) + A_i*smoothed_σ[t]*transpose(A_i)
            if size(ivar_obs, 1) > 0
                L += -(1/2)*(C + sum(log(det(Q_i[ivar_obs, ivar_obs]))) +  tr((ϵ_i*transpose(ϵ_i) + V_ϵ_i)*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L += -(1/2)*(C + sum(log(det(R_i))) + tr((η_i*transpose(η_i) + V_η_i)*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # M function
    optprob = OptimizationFunction(Q, Optimization.AutoForwardDiff()) #Optimization.AutoForwardDiff()AutoModelingToolkit

    llk_array = []
    parameters = model.parameters
    for i in 1:50

        filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters)

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        smoother_ouput = smoother(model, y_t, exogenous_variables, control_variables, filter_output; parameters=parameters)
        smoothed_μ = [smoother_ouput.smoothed_state[t].μ_t for t in 1:(n_obs+1)]
        smoothed_σ = [smoother_ouput.smoothed_state[t].σ_t for t in 1:(n_obs+1)]
        smoothed_auto_σ = smoother_ouput.autocov_state


        prob = Optimization.OptimizationProblem(optprob, parameters, [smoothed_μ, smoothed_σ, smoothed_auto_σ])
        sol = solve(prob, Optim.Newton(), maxiters = 20)
        parameters = sol.minimizer
    
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters)

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters

end