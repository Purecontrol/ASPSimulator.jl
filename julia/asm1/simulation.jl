using Plots, LaTeXStrings

include("asm1.jl")
include("control.jl")

### Define the model ###
# Get default parameters
p, X_init = get_default_parameters_asm1()

# Define the ODE problem
tspan = (0, 30)
tsave = LinRange(29, 30, 10000)
prob = ODEProblem(asm1!, X_init, tspan, p)

### Solve the ODE ###
# Solve the problem
@time sol = solve(prob, Tsit5(), reltol = 1e-8, saveat = tsave, callback=clock_control(t_aerating = 120, t_waiting=180))

# Plot the solution
names = latexstring.(["S_I", "S_S", "X_I", "X_S", "X_{B,H}", "X_{B,A}", "X_P", "S_O", "S_{NO}", "S_{NH}", "S_{ND}", "X_{ND}", "S_{ALK}"])
plot(sol.t, getindex.(sol.u, 9), label = names[9])
plot!(sol.t, getindex.(sol.u, 8) ./2, label=names[8], fill=(0, 0.3, :red))
plot!(sol.t, getindex.(sol.u, 10), label = names[10])