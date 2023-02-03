using JLD, ASM1Simulator

###########################################
### Sensitivity Analysis for ASM1 model ###
###########################################

# Create function to compute the sensitivity indices for each coefficient
function analysis_Morris(f, method::ASM1Simulator.GSA.Morris, design_matrices, asm1_trajectories)

    all_y = f(asm1_trajectories)
    nb_coef = maximum([size(all_y[j], 2) for j in eachindex(all_y)])
    l = Threads.ReentrantLock()
    list_results = Array{ASM1Simulator.GSA.MorrisResult}(undef, nb_coef)
    Threads.@threads for i in 1:nb_coef
        sample = hcat([size(all_y[j], 2) >=i ? all_y[j][:, i] : zeros(size(all_y[j], 1)) for j in eachindex(all_y)]...)
        Threads.lock(l)
        list_results[i] = ASM1Simulator.GSA.compute_indices_Morris(method, design_matrices, sample')
        Threads.unlock(l)
        println("Coefficients $i/$nb_coef Done")
    end
    return list_results
end

# Load data
data  = load("asm1-simulator/data/processed/morris/asm1_morris_samples.jld")
design_matrices = data["design_matrices"]
method = data["method"]
asm1_samples = data["asm1_samples"]

# Compute the sensitivity indices with the EOF metric
eof_i_morris = analysis_Morris(ASM1Simulator.GSA.EOF_metric, method, design_matrices, asm1_samples)

# Save results
@save "asm1-simulator/data/processed/morris/eof_morris_results.jld" eof_i_morris