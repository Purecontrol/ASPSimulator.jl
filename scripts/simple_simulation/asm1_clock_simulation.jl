using Plots, LaTeXStrings, DifferentialEquations, Dates
using ASM1Simulator

### Define the model ###
# Get default parameters
influent_file_path = "/home/victor/Documents/code/asm1-simulator/data/external/influent_files/dryinfluent.ascii"
p, X_init = ASM1Simulator.Models.get_default_parameters_asm1(influent_file_path=influent_file_path)

# Define the ODE problem
tspan = (0, 5)
tsave = LinRange(4, 5, 1440)
prob = ODEProblem(ASM1Simulator.Models.asm1!, X_init, tspan, p)

### Solve the ODE ###
# Solve the problem
@time sol = solve(prob, saveat = tsave, callback=ASM1Simulator.Models.clock_control(t_waiting = 120.0), progress=true, reltol = 1e-4, abstol = 1e-4)

# Plot the solution
names = latexstring.(["S_I", "S_S", "X_I", "X_S", "X_{B,H}", "X_{B,A}", "X_P", "S_O", "S_{NO}", "S_{NH}", "S_{ND}", "X_{ND}", "S_{ALK}"])
index_T = [DateTime(2023,1,1,0) + Dates.Second(Int(round((tsave[end] - tsave[1])*(i-1)*24*60*60/size(tsave, 1)))) for i in 1:size(tsave, 1)]
plot(index_T, getindex.(sol.u, 9), label = names[9])
plot!(index_T, getindex.(sol.u, 8) ./2, label=names[8], fill=(0, 0.3, :red))
plot!(index_T, getindex.(sol.u, 10), label = names[10])
DateTick = Dates.format.(index_T[1:100:end], "HH:MM")
plot!(xticks=(index_T[1:100:end],DateTick), xtickfontsize=6)