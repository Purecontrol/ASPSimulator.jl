using Plots, LaTeXStrings, DifferentialEquations, Dates
using ASM1Simulator

### Define the model ###
# Get default parameters
influent_file_path = "/home/victor/Documents/code/asm1-simulator/data/external/influent_files/dryinfluent.ascii"
p, X_init = ASM1Simulator.Models.get_default_parameters_asm1(influent_file_path=influent_file_path)

# Define the ODE problem
tspan = (0, 20)
tsave = LinRange(19, 20, 1000)
prob = ODEProblem(ASM1Simulator.Models.asm1!, X_init, tspan, p)

### Solve the ODE ###
# Solve the problem
@time sol = solve(prob, reltol = 1e-8, saveat = tsave, callback=ASM1Simulator.Models.redox_control())

# Plot the solution
pyplot()
names = latexstring.(["S_I", "S_S", "X_I", "X_S", "X_{B,H}", "X_{B,A}", "X_P", "S_O", "S_{NO}", "S_{NH}", "S_{ND}", "X_{ND}", "S_{ALK}"])
index_T = [DateTime(2023,1,1,0) + Dates.Second(Int(round((tsave[end] - tsave[1])*(i-1)*24*60*60/size(tsave, 1)))) for i in 1:size(tsave, 1)]
plot(index_T, getindex.(sol.u, 9), label = names[9], xtickfontsize=13,ytickfontsize=13 ,xguidefontsize=13,yguidefontsize=13,legendfontsize=18)
plot!(index_T, getindex.(sol.u, 8), label=names[8], fill=(0, 0.3, :red), xtickfontsize=13,ytickfontsize=13 ,xguidefontsize=13,yguidefontsize=13,legendfontsize=18)
plot!(index_T, getindex.(sol.u, 10), label = names[10], xtickfontsize=13,ytickfontsize=13 ,xguidefontsize=13,yguidefontsize=13,legendfontsize=18)
DateTick = Dates.format.(index_T[1:100:end], "HH:MM")
plot!(xticks=(index_T[1:100:end],DateTick), xtickfontsize=13)
plot!(size=(1000,400))
savefig("/home/victor/Documents/code/asm1-simulator/image/redox_example.pdf")
