using ASM1Simulator, DifferentialEquations, Statistics, Optimization, OptimizationOptimisers, OptimizationOptimJL, Plots
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
sol_real_complete = solve(ode_prob, Rosenbrock23(), reltol = 1e-8, saveat = tsave_complete, callback=ASM1Simulator.Models.redox_control())
sol_real = solve(ode_prob, Rosenbrock23(), reltol = 1e-8, saveat = tsave, callback=ASM1Simulator.Models.external_control(tsave_complete, getindex.(sol_real_complete.u, 14)))

### Estimate the parameters ###
function loss(u, p)
    println(Threads.threadid())
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
    sol = solve(remake(ode_prob, tspan=tspan,p=p_sim, u0=X_init_sim), Rosenbrock23(), reltol = 1e-8, saveat = tsave, callback=ASM1Simulator.Models.external_control(tsave_complete, getindex.(sol_real_complete.u, 14)))
    
    # Compute the loss
    O2 = getindex.(sol.u, 8)
    NO = getindex.(sol.u, 9)
    NH = getindex.(sol.u, 10)  

    loss = mean((O2 .- getindex.(sol_real.u, 8)).^2) + mean((NO .- getindex.(sol_real.u, 9)).^2) + mean((NH .- getindex.(sol_real.u, 10)).^2)

    return loss
end

# Generate an initial solution
bounds = vcat([[i*0.2, i*1.8] for i in vcat(real_p_vec[1:2]...)], [[i*0.2, i*1.8] for i in vcat(real_p_vec[6:7]...)], [[i*0.2, i*1.8] for i in X_init[[2, 4, 5, 6, 7, 11, 12, 13]]])
lb = [i[1] for i in bounds] ; ub = [i[2] for i in bounds]
p_init = lb + rand(29).*(ub-lb)

# See initial solution
p_init_viz = deepcopy(real_p_vec)
p_init_viz[1] = p_init[1:14]
p_init_viz[2] = ASM1Simulator.Models.get_stoichiometric_matrix(p_init[15:19])
p_init_viz[6] = p_init[20]
p_init_viz[7] = p_init[21]
X_init_viz = deepcopy(X_init)
X_init_viz[2] = p_init[22]
X_init_viz[4:7] = p_init[23:26]
X_init_viz[11:13] = p_init[27:29]
# pyplot()
init_sol = solve(remake(ode_prob, tspan=tspan,p=p_init_viz, u0=X_init_viz), Rosenbrock23(),reltol = 1e-8, saveat = tsave, callback=ASM1Simulator.Models.external_control(tsave_complete, getindex.(sol_real_complete.u, 14)))
# plot(getindex.(sol_real.u, 9), label="Real")
# plot!(getindex.(init_sol.u, 9), label="Init solution")

# Solve the optimization problem
lb_optim = lb.*0.2 ; ub_optim = ub.*1.8
prob = OptimizationProblem(loss, p_init, real_p_vec)
sol = solve(prob, Optim.ParticleSwarm(lower=lb_optim, upper=ub_optim, n_particles=10), maxiters=100)

# Get the estimated parameters
p_estimate = deepcopy(real_p)
p_estimate[1] = sol.u[1:14]
p_estimate[2] = sol.u[15:19]
p_estimate[6] = sol.u[20]
p_estimate[7] = sol.u[21]
X_init_estimate = deepcopy(X_init)
X_init_estimate[2] = sol.u[22]
X_init_estimate[4:7] = sol.u[23:26]
X_init_estimate[11:13] = sol.u[27:29]

@save "/home/victor/Documents/code/asm1-simulator/data/processed/asm1_estimation/PSO_1.jld" p_estimate X_init_estimate
data = load("/home/victor/Documents/code/asm1-simulator/data/processed/asm1_estimation/PSO_1.jld")
p_estimate = data["p_estimate"]
X_init_estimate = data["X_init_estimate"]
p_estimate[2] = ASM1Simulator.Models.get_stoichiometric_matrix(p_estimate[2])

### Plot the results ###
pyplot()
estimate_sol = solve(remake(ode_prob, tspan=tspan,p=p_estimate, u0=X_init_estimate), Rosenbrock23(),reltol = 1e-8, saveat = tsave, callback=ASM1Simulator.Models.external_control(tsave_complete, getindex.(sol_real_complete.u, 14)))
plot(getindex.(sol_real.u, 9), label="Real")
plot!(getindex.(estimate_sol.u, 9), label="Estimated")

p_cin = latexstring.(["μ_H", "K_S", "K_{OH}", "K_{NO}", "b_H", "η_g", "η_h", "k_h", "K_X", "μ_A", "K_{NH}", "b_A", "K_{OA}", "k_a"])
p_sto = latexstring.(["Y_A", "Y_H", "f_P", "i_{XB}", "i_{XP}"])
p_oxy = latexstring.(["S_O^{sat}", "K_La"])
groupedbar([real_p_vec[1] p_estimate[1]], label=["Real" "Estimated"], title="Coefficients cinétiques", xticks=(1:14, p_cin))
groupedbar([real_p_vec[2] p_test], label=["Real" "Estimated"], title="Coefficients stochiométriques", xticks=(1:5, p_sto))
groupedbar([[real_p_vec[6], real_p_vec[7]]  Float64.(p_estimate[6:7])], label=["Real" "Estimated"], title="Coefficients oxygènes", xticks=(1:2, p_oxy))
groupedbar([(X_init)  (X_init_estimate)], label=["Real" "Estimated"], title="Conditions initiales")