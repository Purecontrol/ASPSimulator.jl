using JLD, ASM1Simulator, DifferentialEquations

###################################
### Morris Sample of ASM1 model ###
###################################

### Parameters samples ###
nb_samples = 10
len_design_mat = 5 # nb_parameters 
size_samples = Int((24*60)/1) # 24h in 1min steps

### Define the model ###

# Get default parameters
p, X_init = ASM1Simulator.Models.get_default_parameters_asm1()
p_vec, ~ = ASM1Simulator.Models.get_default_parameters_asm1(get_R=false)

# Define the ODE problem
tspan = (0, 10)
tsave = collect(range(9, stop=10, length=size_samples))
prob = ODEProblem(ASM1Simulator.Models.asm1!, X_init, tspan, p)

### Sample design matrices ###

# Define the bounds of the parameters
bounds = [[i*0.5, i*1.5] for i in vcat(p_vec[1:2]...)]
lb = [i[1] for i in bounds] ; ub = [i[2] for i in bounds]

method = ASM1Simulator.GSA.Morris(num_trajectory=nb_samples, len_design_mat=len_design_mat)
design_matrices = ASM1Simulator.GSA.generate_samples_Morris(method, bounds)

### Sample system ###

# Define the function to simulate the ODE
generate_samples = function(p)
    p_asm1 = prob.p
    p_asm1[1] = p[1:14]
    p_asm1[2] = ASM1Simulator.Models.get_stoichiometric_matrix(p[15:end])
    prob_bis = remake(prob;p=p_asm1)
    sol = solve(prob_bis, AutoTsit5(Rosenbrock23()), saveat=tsave, reltol = 1e-8, abstol= 1e-8, callback=ASM1Simulator.Models.redox_control())
    return sol
end

# Generate the samples
nb_call = nb_samples*len_design_mat
asm1_samples = zeros(nb_call, 14, size_samples)
l = Threads.ReentrantLock()
Threads.@threads for i in 1:nb_call
    sample = hcat(generate_samples(design_matrices[:,i]).u...)
    Threads.lock(l)
    asm1_samples[i, :, :] = sample
    Threads.unlock(l)
    println("Sample $i/$nb_call Done")
    flush(stdout)
end

# Downsample the samples
asm1_samples = asm1_samples[:, :, 1:10:end] # 1 points every 10min

@save "asm1-simulator/data/processed/morris/asm1_morris_samples.jld" method design_matrices asm1_samples 