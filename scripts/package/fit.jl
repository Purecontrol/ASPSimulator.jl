using Optimization, OptimizationOptimJL, OptimizationOptimisers
using SparseArrays
using FiniteDiff


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

    ivar_obs_vec = [findall(.!isnan.(y_t[t, :])) for t in 1:n_obs]
    valid_obs_vec = [length(ivar_obs_vec) > 0 for t in 1:n_obs]

    # Q function
    function Q(parameters, smoothed_values)

        L = 0
        for t in 1:n_obs

            # Get current t_step
            # t_step = t_start + (t-1)*dt

            ivar_obs = ivar_obs_vec[t]

            R_i = model.system.R_t(exogenous_variables[t, :], parameters)
            

            A_i = model.system.A_t(exogenous_variables[t, :], parameters)
            B_i = model.system.B_t(exogenous_variables[t, :], parameters)
            c_i = model.system.c_t(exogenous_variables[t, :], parameters)

            η_i = smoothed_values.smoothed_state[t+1].μ_t - (A_i*smoothed_values.smoothed_state[t].μ_t + B_i*control_variables[t, :] + c_i)
            V_η_i = smoothed_values.smoothed_state[t+1].σ_t - smoothed_values.autocov_state[t]*transpose(A_i) - A_i*transpose(smoothed_values.autocov_state[t]) + A_i*smoothed_values.smoothed_state[t].σ_t*transpose(A_i)
            
            if valid_obs_vec[t]

                H_i = model.system.H_t(exogenous_variables[t, :], parameters)
                d_i = model.system.d_t(exogenous_variables[t, :], parameters)
                Q_i = model.system.Q_t(exogenous_variables[t, :], parameters)

                ϵ_i = y_t[t, ivar_obs] - (H_i[ivar_obs, :]*smoothed_values.smoothed_state[t].μ_t + d_i[ivar_obs])
                V_ϵ_i = H_i[ivar_obs, :]*smoothed_values.smoothed_state[t].σ_t*transpose(H_i[ivar_obs, :])

                L -= (sum(log(det(Q_i[ivar_obs, ivar_obs]))) +  tr((ϵ_i*transpose(ϵ_i) + V_ϵ_i)*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L -= (sum(log(det(R_i))) + tr((η_i*transpose(η_i) + V_η_i)*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # M function
    optprob = OptimizationFunction(Q, Optimization.AutoFiniteDiff())

    llk_array = []
    parameters = model.parameters
    for i in 1:100

        filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters)

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        smoother_ouput = smoother(model, y_t, exogenous_variables, control_variables, filter_output; parameters=parameters)

        prob = Optimization.OptimizationProblem(optprob, parameters, smoother_ouput)
        sol = solve(prob, Optim.LBFGS(), maxiters = 5)
        parameters = sol.minimizer
    
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters)

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters

end


function EM_EnKS(model::ForecastingModel, y_t, exogenous_variables, control_variables; n_particles=30)

    # Fixed values
    n_obs = size(y_t, 1)
    t_start = model.current_state.t
    dt = model.system.dt

    ivar_obs_vec = [findall(.!isnan.(y_t[t, :])) for t in 1:n_obs]
    valid_obs_vec = [length(ivar_obs_vec) > 0 for t in 1:n_obs]

    # Q function
    function Q(parameters, smoothed_values)

        L = 0
        for t in 1:n_obs

            # Get current t_step
            # t_step = t_start + (t-1)*dt

            ivar_obs = ivar_obs_vec[t]

            R_i = model.system.R_t(exogenous_variables[t, :], parameters)

            M_i = transition(model.system, smoothed_values.smoothed_state[t].particles_state, exogenous_variables[t, :], control_variables[t, :], parameters)
    
            η_i = smoothed_values.smoothed_state[t+1].particles_state - M_i
            Ω = (η_i*η_i') ./ (n_particles - 1)

            if valid_obs_vec[t]

                H_i = observation(model.system, smoothed_values.smoothed_state[t].particles_state, exogenous_variables[t, :], parameters)[ivar_obs, :]
                Q_i = model.system.Q_t(exogenous_variables[t, :], parameters)
                ϵ_i = y_t[t, ivar_obs] .- H_i
                Σ = (ϵ_i*ϵ_i') ./ (n_particles - 1)

                L -= (sum(log(det(Q_i[ivar_obs, ivar_obs]))) +  tr(Σ*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L -= (sum(log(det(R_i))) + tr(Ω*pinv(R_i)) )

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
        sol = solve(prob, Optim.LBFGS(), maxiters = 1, show_trace=true, show_every=1)
        parameters = sol.minimizer
        print(sol.minimizer)
    
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=EnsembleKalmanFilter(model.current_state, model.system.n_X, model.system.n_Y, n_particles))

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
    valid_obs_vec = [length(ivar_obs_vec) > 0 for t in 1:n_obs]

    exo_var1 = [exogenous_variables[1](t_start + (t-1)*dt) for t in 1:n_obs]
    exo_var2 = [exogenous_variables[2](t_start + (t-1)*dt) for t in 1:n_obs]
    exo_var = hcat([exo_var1, exo_var2]...)

    # Q function
    function Q(parameters, smoothed_values)

        L = 0.0
        @inbounds for t in 1:n_obs

            # Get current t_step
            # t_step = t_start + (t-1)*dt

            ivar_obs = ivar_obs_vec[t]
            
            A_i = A_bis(exo_var[t, :], parameters)
            B_i = B_bis(exo_var[t, :], parameters)
            c_i = c_bis(exo_var[t, :], parameters)
            R_i = R_bis(exo_var[t, :], parameters)

            η_i = smoothed_values.smoothed_state[t+1].μ_t .- (A_i*smoothed_values.smoothed_state[t].μ_t + B_i*control_variables[t] + c_i)
            V_η_i = smoothed_values.smoothed_state[t+1].σ_t .- smoothed_values.autocov_state[t]*transpose(A_i) .- A_i*transpose(smoothed_values.autocov_state[t]) .+ A_i*smoothed_values.smoothed_state[t].σ_t*transpose(A_i)
            
            if valid_obs_vec[t]

                H_i = H_bis(exo_var[t, :], parameters)
                d_i = d_bis(exo_var[t, :], parameters)
                Q_i = Q_bis(exo_var[t, :], parameters)

                ϵ_i = y_t[t, ivar_obs] - (H_i[ivar_obs, :]*smoothed_values.smoothed_state[t].μ_t + d_i[ivar_obs])
                V_ϵ_i = H_i[ivar_obs, :]*smoothed_values.smoothed_state[t].σ_t*transpose(H_i[ivar_obs, :])

                L += -(1/2)*(sum(log(det(Q_i[ivar_obs, ivar_obs]))) +  tr((ϵ_i*transpose(ϵ_i) + V_ϵ_i)*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L += -(1/2)*(sum(log(det(R_i))) + tr((η_i*transpose(η_i) + V_η_i)*pinv(R_i)) )

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
        sol = solve(prob, Optim.LBFGS(), maxiters = 5)
        parameters = sol.minimizer
    
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters)

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters

end