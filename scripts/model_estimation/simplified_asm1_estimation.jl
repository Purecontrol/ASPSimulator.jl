using ASM1Simulator, DifferentialEquations
using LinearAlgebra:I
using Statistics:mean

# Define parameters for the script
nb_var = 14
index_obs_var = [8, 9, 10]
nb_obs_var = size(index_obs_var, 1)

nb_var_simplified = 14
index_obs_var_simplfied = [2, 3, 4]
nb_obs_var_simplified = size(index_obs_var_simplfied, 1)

H = zeros(14, 14)
H[index_obs_var, index_obs_var] = Matrix(I, nb_obs_var, nb_obs_var)
H = H[index_obs_var, :]

H_simplified = zeros(6, 6)
H_simplified[index_obs_var_simplfied, index_obs_var_simplfied] = Matrix(I, nb_obs_var_simplified, nb_obs_var_simplified)
H_simplified = H_simplified[index_obs_var_simplfied, :]

####################################################################################################################
########################################## REFERENCE DATA FOR OPTIMIZATION #########################################
####################################################################################################################

# Get true init_vector and p_vector
influent_file_path = "/home/victor/Documents/code/asm1-simulator/data/external/influent_files/dryinfluent.ascii"
real_p, X_init = ASM1Simulator.Models.get_default_parameters_asm1(influent_file_path=influent_file_path)
real_p_vec, ~ = ASM1Simulator.Models.get_default_parameters_asm1(get_R=false, influent_file_path=influent_file_path)

# Let the system evolve during 20 days in order to be stable and get new X_init
tspan = (0, 20)
steady_prob = ODEProblem(ASM1Simulator.Models.asm1!, X_init, tspan, real_p)
sol_steady = solve(steady_prob, Tsit5(), callback=ASM1Simulator.Models.redox_control())
X_init_real = sol_steady.u[end]

# Define the real ODE problem
tspan = (0, 5)
tsave_complete = LinRange(0, 5, 5*1440) # 1 value every minute
tsave = LinRange(0, 5, 5*144) # 1 values every 10 minute
ode_prob = ODEProblem(ASM1Simulator.Models.asm1!, X_init_real, tspan, real_p)

# Generate the real solution
sol_real_complete = solve(ode_prob, Tsit5(), saveat = tsave_complete, callback=ASM1Simulator.Models.redox_control())
sol_real = solve(ode_prob, Tsit5(), saveat = tsave, callback=ASM1Simulator.Models.external_control(tsave_complete, getindex.(sol_real_complete.u, 14)))

# Generate real obs vector
xt = hcat(sol_real.u...)
ŷt = H*xt + randn(nb_obs_var, size(xt, 2))*0

####################################################################################################################
########################################### DEFINE THE MODEL TO OPTIMIZE ###########################################
####################################################################################################################

# Get init_vector and p_vector for the simplified model
influent_file_path = "/home/victor/Documents/code/asm1-simulator/data/external/influent_files/dryinfluent.ascii"
simplified_real_p, simplified_X_init = ASM1Simulator.Models.get_default_parameters_simplified_asm1(influent_file_path=influent_file_path)
simplified_real_p_vec, ~ = ASM1Simulator.Models.get_default_parameters_simplified_asm1(get_R=false, influent_file_path=influent_file_path)

# Set the real value of X_init for observed variables
simplified_X_init[2] = X_init_real[8]
simplified_X_init[3] = X_init_real[9]
simplified_X_init[4] = X_init_real[10]
simplified_X_init[6] = X_init_real[14]

# Define the new ODE problem associated with the simplified model
simplified_ode_prob = ODEProblem(ASM1Simulator.Models.simplified_asm1!, simplified_X_init, tspan, simplified_real_p)

function model(opt_params)

    # Copy real vector of parameters
    p_sim = deepcopy(simplified_real_p_vec)
    
    # Copy fixed parameters and update the parameters to optimize
    p_sim[1] = opt_params[1:8]
    p_sim[2] = opt_params[9:13]
    p_sim[3] = ASM1Simulator.Models.get_stoichiometric_matrix_simplified_asm1(opt_params[14:16])
    p_sim[7] = opt_params[17]
    p_sim[8] = opt_params[18]

    # Copy fixed initial conditions and update the initial conditions to optimize
    X_init_sim = deepcopy(simplified_X_init)
    X_init_sim[1] = opt_params[19]
    X_init_sim[5] = opt_params[20]

    # Simulate the model
    x̂ = zeros(6, size(tsave, 1))*NaN
    try
        sol_sim = solve(remake(simplified_ode_prob, tspan=tspan, p=p_sim, u0=X_init_sim), Tsit5(), saveat = tsave, callback=ASM1Simulator.Models.external_control(tsave_complete, getindex.(sol_real_complete.u, 14); index_u = 6), verbose=false, maxiters=Int(5*1e4))
        
        x_sol = hcat(sol_sim.u...)
        if size(x_sol, 2) == size(x̂, 2)
            x̂ = x_sol
        else
            println("Error in ODE solver AHAHAHAHA")
        end

    catch e
        println("Error in ODE solver")
    end

    return H_simplified*x̂

end

####################################################################################################################
########################################### DEFINE THE LOSS TO MINIMIZE ############################################
####################################################################################################################

function L2_loss(pred, real)

    # Compute the loss
    l2_loss = sqrt.(mean((pred .- real) .^ 2))

    if isnan(l2_loss)
        l2_loss = 1e16
    end

    return l2_loss

end

####################################################################################################################
############################################### DEFINE THE OPTIMIZER ###############################################
####################################################################################################################

using Optimization, OptimizationOptimisers, OptimizationOptimJL

# Define the bounds on the variables
nb_opt_params = 20
lb_optim = zeros(nb_opt_params) .+ 0.0001
ub_optim = [10000.0 for i in 1:nb_opt_params]

p_lower, p_upper, X_init_lower, X_init_upper = ASM1Simulator.Models.get_bounds_parameters_simplified_asm1()
bounds = vcat([[p_lower[1][i], p_upper[1][i]] for i in 1:size(simplified_real_p_vec[1], 1)], [[p_lower[2][i], p_upper[2][i]] for i in 1:size(simplified_real_p_vec[2], 1)], [[p_lower[3][i], p_upper[3][i]] for i in 1:size(simplified_real_p_vec[3], 1)], [[p_lower[7], p_upper[7]]], [[p_lower[8], p_upper[8]]], [[X_init_lower[i], X_init_upper[i]] for i in [1, 5]])
lb = [i[1] for i in bounds] ; ub = [i[2] for i in bounds]
p_init = lb + rand(nb_opt_params).*(ub-lb)

# Define the optimizer and the options
nb_iter = 4
nb_particules = 4
optimizer = Optim.ParticleSwarm(lower=lb_optim, lower_ini=lb, upper=ub_optim, upper_ini=ub, n_particles=nb_particules)
options = Optim.Options(iterations=nb_iter, extended_trace=true, store_trace=true, show_trace=true, show_every=2)

####################################################################################################################
############################################## OPTIMIZE THE PARAMETERS #############################################
####################################################################################################################

res = optimize(u -> L2_loss(model(u), ŷt), lb_optim, ub_optim, p_init, optimizer, options)