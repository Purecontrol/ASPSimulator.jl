using Statistics, Plots, JLD, ProgressBars, DifferentialEquations
using ASM1Simulator

### Parameters scripts ###
nb_samples = 10
size_samples = Int((24*60)) # one point per minutes for 24 hours

### Define the model ###
# Get default parameters
p, X_init = ASM1Simulator.Models.get_default_parameters_asm1()

# Define the ODE problem
tspan = (0, 10)
tsave = collect(range(9, stop=10, length=size_samples))
prob = ODEProblem(ASM1Simulator.Models.asm1!, X_init, tspan, p)

### Simulate multiple times the ODE ###
# Define the function to simulate the ODE
generate_samples = function(T)
    new_p, X_init = ASM1Simulator.Models.get_default_parameters_asm1(T=T)
    prob_bis = remake(prob;p=new_p)
    sol = solve(prob_bis, saveat=tsave, reltol = 1e-8, abstol= 1e-8, callback=ASM1Simulator.Models.clock_control())
    return sol
end

# Generate the grid of temperature
T_vec = collect(range(6, stop=25, length=nb_samples))

# Generate the samples
asm1_samples = zeros(nb_samples, 14, size_samples)
for i in tqdm(eachindex(T_vec))
    asm1_samples[i, :, :] = hcat(generate_samples(T_vec[i]).u...)
end

# Save the results
@save "asm1-simulator/data/results/samples_temperature_oxygen.jld" T_vec asm1_samples 

# Plot the results
plot(tsave, asm1_samples[:,8,:]', lab=T_vec')
