using ASM1Simulator, Plots, Random, JLD

# Include utils.jl file 
include("../smc/utils.jl")
include("../smc/smc.jl")

# Define parameter physical model
influent_file_path = "asm1-simulator/data/external/influent_files/dryinfluent.ascii"
p_asm1, X_init = ASM1Simulator.Models.get_default_parameters_asm1(influent_file_path=influent_file_path)

# Load estimated parameters
data = load("asm1-simulator/data/processed/asm1_estimation/PSO_4.jld")
p_estimate = data["p_estimate"]
X_init_estimate = data["X_init_estimate"]
p_estimate[2] = ASM1Simulator.Models.get_stoichiometric_matrix(p_estimate[2])
p_estimate[5] = p_asm1[5]

# Define parameter data assimilation
dt_integration = 10/(24*60*60) # 10 secondes
dt_states = 10*6 # 10 minutes
dt_obs = 10*6 # 10 minutes
var_obs = [8, 9, 10] # O2, NO, NH

# Define the list of SNR values to test
SNR_model = [10000.0, 1000.0, 100.0, 10.0, 1.0]
SNR_obs = [10000.0, 1000.0, 100.0, 10.0, 1.0]

index = vcat([[(i,j) for i in 1:size(SNR_model, 1)] for j in 1:size(SNR_obs, 1)]...)

Threads.@threads for ind in index

    i = ind[1]
    j = ind[2]

    print("#### SNR_model: ", SNR_model[i])
    println(" SNR_obs: ", SNR_obs[j], " ####")

    # Define StateSpaceModel structure
    ssm = StateSpaceModel(
        ASM1Simulator.Models.asm1!,
        dt_integration,
        dt_states,
        dt_obs,
        p_asm1,
        var_obs;
        SNR_model = SNR_model[i],
        SNR_obs = SNR_obs[j],
    )

    # Generate data from the SSM model
    T = 1.0
    xt , _, ŷt , U = generate_data(ssm, X_init, T; control=ASM1Simulator.Models.redox_control());

    ssm2 = StateSpaceModel(
        ASM1Simulator.Models.asm1!,
        dt_integration,
        dt_states,
        dt_obs,
        p_estimate,
        var_obs;
        Σ_model = ssm.Σ_model,
        Σ_obs= ssm.Σ_obs,
    )

    # Get the predicted states from the model
    T = 1.0
    _ , x̂t, _ , _ = generate_data(ssm2, X_init_estimate, T; control=U);

    # Run the EnKF algorithm
    da2 = DataAssimilation(ssm2, x̂t)
    x̂, part = PF(da2, ŷt, 100; progress = true, control = U)

    # Compute the RMSE
    rmse = sqrt.(mean((vcat(x̂.u'...) .- vcat(xt.u'...)) .^ 2, dims=1))
    println("RMSE: ", rmse)

    name_file = "asm1-simulator/data/processed/assimilation/PF-FP/SNR_model_$(SNR_model[i])_SNR_obs_$(SNR_obs[j]).jld"
    @save name_file x̂ xt x̂t ŷt part

end