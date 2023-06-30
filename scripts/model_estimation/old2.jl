using ASM1Simulator, DifferentialEquations, Statistics, Optimization, OptimizationOptimisers, Plots, OptimizationOptimJL
using StatsPlots, LaTeXStrings, Base.Threads, JLD

### Define the model ###
# Get default parameters
influent_file_path = "/home/victor/Documents/code/asm1-simulator/data/external/influent_files/dryinfluent.ascii"
real_p, X_init = ASM1Simulator.Models.get_default_parameters_asm1(influent_file_path=influent_file_path)
real_p_vec, ~ = ASM1Simulator.Models.get_default_parameters_asm1(get_R=false, influent_file_path=influent_file_path)

# Parameter of the generation
tspan = (0, 10)
tsave_complete = LinRange(0, 10, 10*1440)
tsave = LinRange(0, 10, 10*1440)

### Generate the true solution
ode_prob_true = ODEProblem(ASM1Simulator.Models.asm1!, X_init, tspan, real_p)
sol_true_complete = solve(ode_prob_true, Tsit5(), reltol = 1e-8, saveat = tsave_complete, callback=ASM1Simulator.Models.redox_control())
sol_true = solve(ode_prob_true, Tsit5(), reltol = 1e-8, saveat = tsave, callback=ASM1Simulator.Models.external_control(tsave_complete, getindex.(sol_true_complete.u, 14)))


### Generate the estimated solution
p_estimate = load("/home/victor/Documents/code/asm1-simulator/data/processed/asm1_estimation/PSO_4.jld", "p_estimate")
X_init_estimate = load("/home/victor/Documents/code/asm1-simulator/data/processed/asm1_estimation/PSO_4.jld", "X_init_estimate")
p_estimate[2] = ASM1Simulator.Models.get_stoichiometric_matrix(p_estimate[2])
p_estimate[5] = real_p[5]
ode_prob_estimated = ODEProblem(ASM1Simulator.Models.asm1!, X_init_estimate, tspan, p_estimate)
sol_estimated = solve(ode_prob_estimated, Tsit5(), reltol = 1e-8, saveat = tsave, callback=ASM1Simulator.Models.external_control(tsave_complete, getindex.(sol_true_complete.u, 14)))

### Plot the results

pyplot()
t1=Plots.plot(tsave, getindex.(sol_true.u, 8), label="True solution", xlabel="Time (h)", ylabel="X (g/L)", title="Oxygen")
plot!(tsave, getindex.(sol_estimated.u, 8), label="Estimated solution")

t2=Plots.plot(tsave, getindex.(sol_true.u, 9), label="True solution", xlabel="Time (h)", ylabel="X (g/L)", title="NO3")
plot!(tsave, getindex.(sol_estimated.u, 9), label="Estimated solution")

t3=Plots.plot(tsave, getindex.(sol_true.u, 10), label="True solution", xlabel="Time (h)", ylabel="X (g/L)", title="NH4")
plot!(tsave, getindex.(sol_estimated.u, 10), label="Estimated solution")

# Put t1, t2, t3 in a grid
Plots.plot(t1, t2, t3, layout=(3,1), size=(1200, 600))