using Plots, LaTeXStrings, DifferentialEquations, Dates, Statistics
using ASM1Simulator

### Use the true model with redox control to stabilize the system ###
# Get default parameters
influent_file_path = "/home/victor/Documents/code/asm1-simulator/data/external/influent_files/dryinfluent.ascii"
p, X_init = ASM1Simulator.Models.get_default_parameters_asm1(influent_file_path=influent_file_path)

# Define the ODE problem
tspan = (0, 20)
tsave = LinRange(19, 20, 1440)
prob = ODEProblem(ASM1Simulator.Models.asm1!, X_init, tspan, p)

# Solve the problem
@time sol_true = solve(prob, saveat = tsave, reltol = 1e-8, callback=ASM1Simulator.Models.redox_control(), progress=true)

# Plot the solution
names = latexstring.(["S_I", "S_S", "X_I", "X_S", "X_{B,H}", "X_{B,A}", "X_P", "S_O", "S_{NO}", "S_{NH}", "S_{ND}", "X_{ND}", "S_{ALK}"])
index_T = [DateTime(2023,1,1,0) + Dates.Second(Int(round((tsave[end] - tsave[1])*(i-1)*24*60*60/size(tsave, 1)))) for i in 1:size(tsave, 1)]
plot(index_T, getindex.(sol_true.u, 9), label = names[9])
plot!(index_T, getindex.(sol_true.u, 8) ./2, label=names[8], fill=(0, 0.3, :red))
plot!(index_T, getindex.(sol_true.u, 10), label = names[10])
DateTick = Dates.format.(index_T[1:100:end], "HH:MM")
plot!(xticks=(index_T[1:100:end],DateTick), xtickfontsize=6)

### Make prediction for the next day for the true model ###
# Get initial conditions
X_init_pred = sol_true.u[end]

# Define the ODE problem
tspan = (0, 1)
tsave = LinRange(0, 1, 1440)
prob = ODEProblem(ASM1Simulator.Models.asm1!, X_init_pred, tspan, p)

# Solve the problem
@time sol_pred_true = solve(prob, saveat = tsave, reltol = 1e-8, callback=ASM1Simulator.Models.redox_control(), progress=true)

### Make prediction for the next day for the simplified model ###
# Compute the mean value of fixed parameters
X_BH = getindex.(sol_true.u, 5) ; X_BA = getindex.(sol_true.u, 6) ; X_ND = getindex.(sol_true.u, 12) ; X_S = getindex.(sol_true.u, 4); S_S = getindex.(sol_true.u, 2)
fixed_concentration = [X_BH, X_BA, X_ND, X_S, S_S]
X_init_simplified = [X_init_pred[2] + X_init_pred[4], X_init_pred[8], X_init_pred[9], X_init_pred[10], X_init_pred[11], X_init_pred[14]]

# Get default parameters for simplified models
influent_file_path = "/home/victor/Documents/code/asm1-simulator/data/external/influent_files/dryinfluent.ascii"
p_simplified, _ = ASM1Simulator.Models.get_default_parameters_simplified_asm1(influent_file_path=influent_file_path; fixed_concentration = fixed_concentration)

# Define the ODE problem
prob = ODEProblem(ASM1Simulator.Models.simplified_asm1!, X_init_simplified, tspan, p_simplified)

# Solve the problem
@time sol_approx = solve(prob, saveat = tsave, reltol = 1e-8, callback=ASM1Simulator.Models.external_control(tsave, getindex.(sol_pred_true.u, 14)), progress=true)

plot(index_T, getindex.(sol_approx.u, 5), label = "Prediction S_ND")
plot!(index_T, getindex.(sol_pred_true.u, 11), label = "Mesure S_ND")
DateTick = Dates.format.(index_T[1:100:end], "HH:MM")
plot!(xticks=(index_T[1:100:end],DateTick), xtickfontsize=6)

# Plot the solution
names = latexstring.(["S_I", "S_S", "X_I", "X_S", "X_{B,H}", "X_{B,A}", "X_P", "S_O", "S_{NO}", "S_{NH}", "S_{ND}", "X_{ND}", "S_{ALK}"])
index_T = [DateTime(2023,1,1,0) + Dates.Second(Int(round((tsave[end] - tsave[1])*(i-1)*24*60*60/size(tsave, 1)))) for i in 1:size(tsave, 1)]
plot(index_T, getindex.(sol_approx.u, 3), label = "Prediction NO3")
plot!(index_T, getindex.(sol_pred_true.u, 9), label = "Mesure NO3")
DateTick = Dates.format.(index_T[1:100:end], "HH:MM")
plot!(xticks=(index_T[1:100:end],DateTick), xtickfontsize=6)

plot(index_T, getindex.(sol_approx.u, 1), label = "Prediction DCO")
plot!(index_T, getindex.(sol_pred_true.u, 2) .+ getindex.(sol_pred_true.u, 4), label = "Mesure DCO")
DateTick = Dates.format.(index_T[1:100:end], "HH:MM")
plot!(xticks=(index_T[1:100:end],DateTick), xtickfontsize=6)

plot(index_T, getindex.(sol_pred_true.u, 8) ./2, label= "Mesure O2", fill=(0, 0.3, :red))
plot!(index_T, getindex.(sol_approx.u, 2) ./2, label= "Prediction O2", fill=(0, 0.3, :red))
DateTick = Dates.format.(index_T[1:100:end], "HH:MM")
plot!(xticks=(index_T[1:100:end],DateTick), xtickfontsize=6)

plot(index_T, getindex.(sol_approx.u, 4), label = "Prediction NH4")
plot!(index_T, getindex.(sol_pred_true.u, 10), label = "Mesure NH4")
DateTick = Dates.format.(index_T[1:100:end], "HH:MM")
plot!(xticks=(index_T[1:100:end],DateTick), xtickfontsize=6)