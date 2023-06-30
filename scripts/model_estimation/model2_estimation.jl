using ASM1Simulator, DifferentialEquations
using LinearAlgebra:I
using Statistics:mean

# Define parameters for the script
nb_var = 14
index_obs_var = [8, 9, 10]
nb_obs_var = size(index_obs_var, 1)

H = zeros(14, 14)
H[index_obs_var, index_obs_var] = Matrix(I, nb_obs_var, nb_obs_var)
H = H[index_obs_var, :]

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