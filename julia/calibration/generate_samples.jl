using GlobalSensitivity, Statistics, Plots, QuasiMonteCarlo, JLD

include("../asm1/asm1.jl")
include("../asm1/control.jl")

### Parameters scripts ###
nb_samples = 10
size_samples = 2000

### Define the model ###
# Get default parameters
p, X_init = get_default_parameters_asm1()
p_vec, ~ = get_default_parameters_asm1(get_R=false)

# Define the ODE problem
tspan = (0, 20)
tsave = collect(range(10, stop=20, length=size_samples))
prob = ODEProblem(asm1!, X_init, tspan, p)

### Simulate multiple times the ODE ###
# Define the function to simulate the ODE
p_samples = Array{Vector{Any}, 1}(undef, 0)
generate_samples = function(p)
    p_asm1 = prob.p
    p_asm1[1] = p[1:14]
    p_asm1[2] = get_stoichiometric_matrix(p[15:end])
    push!(p_samples, p_asm1)
    prob_bis = remake(prob;p=p_asm1)
    sol = solve(prob_bis, saveat=tsave, reltol = 1e-8, abstol= 1e-8, callback=clock_control(t_aerating = 120, t_waiting=180))
    return sol
end

# Generate the grid of parameters
bounds = [[i*0.5, i*1.5] for i in vcat(p_vec[1:2]...)]
lb = [i[1] for i in bounds] ; ub = [i[2] for i in bounds]
sampler = SobolSample()
p_grid = QuasiMonteCarlo.sample(nb_samples, lb, ub, sampler)

# Generate the samples
asm1_samples = zeros(nb_samples, 14, size_samples)
for i in 1:nb_samples
    asm1_samples[i, :, :] = hcat(generate_samples(p_grid[:,i]).u...)
end

@save "/home/victor/Documents/code/asm1-simulator/data/samples.jld" p_samples asm1_samples 