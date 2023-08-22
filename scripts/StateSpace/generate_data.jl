using ASM1Simulator, DifferentialEquations, Distributions
using LinearAlgebra:I
using Statistics:mean

include("env.jl")

# User-defined parameters
T_steady_state = 20 #(in days)
T_training = 0.25 #(in days)
T_testing = 1.5 #(in days)
σ_ϵ = 0.5
dt_obs = 5 #(in minutes)

# Fixed parameters
nb_var = 14
index_obs_var = [8, 10]
nb_obs_var = size(index_obs_var, 1)

####################################################################################################################
########################################## REFERENCE DATA FOR OPTIMIZATION #########################################
####################################################################################################################

# Set up environment
sim = ODECore()

# Let the system evolve during T_steady_state days in order to be stable and get new X_init
res_steady = multi_step!(sim, ASM1Simulator.Models.redox_control(), Day(T_steady_state))
X_init_real = sim.state

# Let the system evolve during T_training + T_testing days to get training and testing data
sol_real  = multi_step!(sim, ASM1Simulator.Models.redox_control(), Hour((T_training + T_testing)*24))

# Define H
H = zeros(nb_var, nb_var)
H[index_obs_var, index_obs_var] = Matrix(I, nb_obs_var, nb_obs_var)
H = H[index_obs_var, :]

# Get training and test set for the state of the system
x_train = hcat(sol_real.u...)[:, 1:Int(T_training*1440)]
x_test = hcat(sol_real.u...)[:, (Int(T_training*1440)+1):end]

# Get training and test set for the observation of the system
y_train = H*x_train + rand(Normal(0, σ_ϵ), (nb_obs_var, size(x_train, 2))) + vcat([reshape([(i-1)%Int(dt_obs) == 0 ? 0 : NaN for i in 1:size(x_train, 2)], (1,size(x_train, 2))) for i in 1:nb_obs_var]...)
y_test = H*x_test + rand(Normal(0, σ_ϵ), (nb_obs_var, size(x_test, 2))) + vcat([reshape([(i-1)%Int(dt_obs) == 0 ? 0 : NaN for i in 1:size(x_test, 2)], (1,size(x_test, 2)))  for i in 1:nb_obs_var]...)

# Get control variables of the system
U_train = transpose(hcat(getindex.(sol_real.u, 14)...))[1:Int(T_training*1440), :]
U_test = transpose(hcat(getindex.(sol_real.u, 14)...))[(Int(T_training*1440)+1):end, :]