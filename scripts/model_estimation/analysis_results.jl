using ASM1Simulator, DifferentialEquations, Statistics, Optimization, OptimizationOptimisers, Plots, OptimizationOptimJL
using StatsPlots, LaTeXStrings, Base.Threads, JLD

### Define the model ###
# Get default parameters
real_p, X_init = ASM1Simulator.Models.get_default_parameters_asm1()
real_p_vec, ~ = ASM1Simulator.Models.get_default_parameters_asm1(get_R=false)

# Define the ODE problem
tspan = (0, 10)
tsave_complete = LinRange(0, 10, 10*1440)
tsave = LinRange(9, 10, 1440)
ode_prob = ODEProblem(ASM1Simulator.Models.asm1!, X_init, tspan, real_p)

# Generate the real solution
sol_real_complete = solve(ode_prob, Tsit5(), reltol = 1e-8, saveat = tsave_complete, callback=ASM1Simulator.Models.redox_control())
sol_real = solve(ode_prob, Rosenbrock23(), reltol = 1e-8, saveat = tsave, callback=ASM1Simulator.Models.external_control(tsave_complete, getindex.(sol_real_complete.u, 14)))

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

data = load("/home/victor/Documents/code/asm1-simulator/data/processed/asm1_estimation/PSO_4.jld")
p_estimate = data["p_estimate"]
X_init_estimate = data["X_init_estimate"]

vec_true = [12.96, 0.4, 0.6480000000000001, 0.2991947305142656, 0.9720000000000001, 0.6735285021183468, 2.5920000000000005, 0.12000000000000002, 0.1243536217555445, 0.020000000000000004, 1.1658452120358755, 0.0020000000000000005, 0.2797894659602351, 0.16200000000000003, 0.009600000000000001, 0.026800000000000004, 0.22992400368149038, 0.05569047970129168, 0.1944, 6.54140178724303, 230.0018829088485, 0.12201200000000001, 21.668697655348033, 1518.8747499502279, 540.010476, 3126.273408, 0.6788255353007405, 12.458772000000002, 0.21685200000000002]
p_estimate = deepcopy(real_p)
p_estimate[1] = vec_true[1:14]
p_estimate[2] = vec_true[15:19]
p_estimate[6] = vec_true[20]
p_estimate[7] = vec_true[21]
X_init_estimate = deepcopy(X_init)
X_init_estimate[2] = vec_true[22]
X_init_estimate[4:7] = vec_true[23:26]
X_init_estimate[11:13] = vec_true[27:29]


p_estimate[2] = ASM1Simulator.Models.get_stoichiometric_matrix(p_estimate[2])

### Plot the results ###
pyplot()
estimate_sol = solve(remake(ode_prob, tspan=tspan,p=p_estimate, u0=X_init_estimate), Rosenbrock23(),reltol = 1e-8, saveat = tsave, callback=ASM1Simulator.Models.external_control(tsave_complete, getindex.(sol_real_complete.u, 14)))
plot(getindex.(sol_real.u, 8), label="Real")
plot!([1.8 for i in 1:1440], label="Estimated")

p_cin = latexstring.(["μ_H", "K_S", "K_{OH}", "K_{NO}", "b_H", "η_g", "η_h", "k_h", "K_X", "μ_A", "K_{NH}", "b_A", "K_{OA}", "k_a"])
p_sto = latexstring.(["Y_A", "Y_H", "f_P", "i_{XB}", "i_{XP}"])
p_oxy = latexstring.(["S_O^{sat}", "K_La"])
groupedbar([real_p_vec[1] p_estimate[1]], label=["Real" "Estimated"], title="Coefficients cinétiques", xticks=(1:14, p_cin))
groupedbar([real_p_vec[2] p_estimate[2]], label=["Real" "Estimated"], title="Coefficients stochiométriques", xticks=(1:5, p_sto))
groupedbar([[real_p_vec[6], real_p_vec[7]]  Float64.(p_estimate[6:7])], label=["Real" "Estimated"], title="Coefficients oxygènes", xticks=(1:2, p_oxy))
groupedbar(log.([X_init  X_init_estimate]), label=["Real" "Estimated"], title="Conditions initiales")