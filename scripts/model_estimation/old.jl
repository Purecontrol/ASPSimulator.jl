using ASM1Simulator, DifferentialEquations, Statistics, Optimization, OptimizationOptimisers, Plots, OptimizationOptimJL
using StatsPlots, LaTeXStrings, Base.Threads, JLD

### Define the model ###
# Get default parameters
influent_file_path = "/home/victor/Documents/code/asm1-simulator/data/external/influent_files/dryinfluent.ascii"
real_p, X_init = ASM1Simulator.Models.get_default_parameters_asm1(influent_file_path=influent_file_path)
real_p_vec, ~ = ASM1Simulator.Models.get_default_parameters_asm1(get_R=false, influent_file_path=influent_file_path)

# Define the ODE problem
tspan = (0, 10)
tsave_complete = LinRange(0, 10, 10*1440)
tsave = LinRange(9, 10, 1440)
ode_prob = ODEProblem(ASM1Simulator.Models.asm1!, X_init, tspan, real_p)

# Generate the real solution
sol_real_complete = solve(ode_prob, Tsit5(), saveat = tsave_complete, callback=ASM1Simulator.Models.redox_control())
sol_real = solve(ode_prob, Tsit5(), saveat = tsave, callback=ASM1Simulator.Models.external_control(tsave_complete, getindex.(sol_real_complete.u, 14)))

### Estimate the parameters ###
function loss(u, p)

    # Estimate parameters
    p_sim = deepcopy(p)
    p_sim[1] = u[1:14]
    p_sim[2] = ASM1Simulator.Models.get_stoichiometric_matrix(u[15:19])
    p_sim[6] = u[20]
    p_sim[7] = u[21]

    # Estimate initial conditions (remove S_I and X_i because they have no influence with the rest of the model)
    X_init_sim = deepcopy(X_init)
    X_init_sim[2] = u[22]
    X_init_sim[4:7] = u[23:26]
    X_init_sim[11:13] = u[27:29]

    # Simulate the model
    sol = solve(remake(ode_prob, tspan=tspan,p=p_sim, u0=X_init_sim), Tsit5(), saveat = tsave, callback=ASM1Simulator.Models.external_control(tsave_complete, getindex.(sol_real_complete.u, 14)), verbose=false, maxiters=Int(5*1e4))
    
    # prior_loss = mean((u[1:14] .- p[1]).^2) + mean((u[15:19] .- p[2]).^2) + mean((u[20] .- p[6]).^2) + mean((u[21] .- p[7]).^2)
    if sol.retcode == ReturnCode.Success
        # Compute the loss
        O2 = getindex.(sol.u, 8)
        NO = getindex.(sol.u, 9)
        NH = getindex.(sol.u, 10)  

        l2_loss = mean((O2 .- getindex.(sol_real.u, 8)).^2) + mean((NO .- getindex.(sol_real.u, 9)).^2) + mean((NH .- getindex.(sol_real.u, 10)).^2)
    else
        println("Error: ", sol.retcode)
        l2_loss = 1000000
    end
    return l2_loss
end

# Generate an initial solution
p_lower, p_upper, X_init_lower, X_init_upper = ASM1Simulator.Models.get_bounds_parameters_asm1()
bounds = vcat([[p_lower[1][i], p_upper[1][i]] for i in 1:size(real_p_vec[1], 1)], [[p_lower[2][i], p_upper[2][i]] for i in 1:size(real_p_vec[2], 1)], [[p_lower[6], p_upper[6]]], [[p_lower[7], p_upper[7]]], [[X_init_lower[i], X_init_upper[i]] for i in [2, 4, 5, 6, 7, 11, 12, 13]])
lb = [i[1] for i in bounds] ; ub = [i[2] for i in bounds]
p_init = lb + rand(29).*(ub-lb)

#################################################################
################ Solve the optimization problem #################
#################################################################

# Define the bounds on the variables
lb_optim = zeros(29) .+ 0.0001
ub_optim = [10000.0 for i in 1:29]

# Define the optimizer and the options
nb_iter = 10
nb_particules = 10
optimizer = Optim.ParticleSwarm(lower=lb_optim, lower_ini=lb, upper=ub_optim, upper_ini=ub, n_particles=nb_particules)
options = Optim.Options(iterations=nb_iter, extended_trace=true, store_trace=true, show_trace=true, show_every=2)

# Solve the problem
res = optimize(u -> loss(u,real_p_vec), lb_optim, ub_optim, p_init, optimizer, options)

# Get the estimated parameters
p_estimate = deepcopy(real_p)
p_estimate[1] = res.minimizer[1:14]
p_estimate[2] = res.minimizer[15:19]
p_estimate[6] = res.minimizer[20]
p_estimate[7] = res.minimizer[21]
X_init_estimate = deepcopy(X_init)
X_init_estimate[2] = res.minimizer[22]
X_init_estimate[4:7] = res.minimizer[23:26]
X_init_estimate[11:13] = res.minimizer[27:29]

@save "/home/victor/Documents/code/asm1-simulator/data/processed/asm1_estimation/PSO_3.jld" res p_estimate X_init_estimate

# Load the ascii documents in a 2D array
using DelimitedFiles
data = readdlm("/home/victor/Documents/code/asm1-simulator/data/external/influent_files/dryinfluent.ascii")

x = data[: ,1]
y = data[: ,10]
itp = interpolate((x, ), y, Gridded(Linear()))
sitp = scale(itp, x)
x_test = LinRange(minimum(x), 20, 100000)
plot(x_test, itp(x_test))
plot!(x, y)