using JLD

###########################################
### Sensitivity Analysis for ASM1 model ###
###########################################

include("metrics.jl")
include("tools.jl")

# Load data
data  = load("/home/victor/Documents/code/asm1-simulator/julia/calibration/Morris/data/morris_samples.jld")
design_matrices = data["design_matrices"]
method = data["method"]
asm1_samples = data["asm1_samples"]

# Compute the sensitivity indices with the EOF metric
eof_i_morris = analysis_Morris(EOF_metric, method, design_matrices, asm1_samples)

# Save results
@save "/home/victor/Documents/code/asm1-simulator/julia/calibration/Morris/data/eof_morris_results.jld" eof_i_morris