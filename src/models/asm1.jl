using DelimitedFiles: readdlm

"""
     $(TYPEDSIGNATURES)

Return the default parameters and an initial condition for the ASM1 model.
"""
function get_default_parameters_asm1(; T = 15, influent_file_path = nothing,
        variable_inlet_concentration = true, variable_inflow = true, kwargs...)

    # Set kinetics parameters
    μ_H = T_var(T, 4.0, 3)
    K_S = 10.0
    K_OH = 0.2
    K_NO = 0.5
    b_H = T_var(T, 0.3, 0.2)
    η_g = 0.8
    η_h = 0.8
    k_h = T_var(T, 3.0, 2.5)
    K_X = 0.1
    μ_A = T_var(T, 0.5, 0.3)
    K_NH = 1.0
    b_A = T_var(T, 0.05, 0.03)
    K_OA = 0.4
    k_a = T_var(T, 0.05, 0.04)
    kinetic_parameters = (
        μ_H = μ_H, K_S = K_S, K_OH = K_OH, K_NO = K_NO, b_H = b_H, η_g = η_g, η_h = η_h,
        k_h = k_h, K_X = K_X, μ_A = μ_A, K_NH = K_NH, b_A = b_A, K_OA = K_OA, k_a = k_a)

    # Set stoichiometric parameters
    Y_A = 0.24
    Y_H = 0.67
    f_P = 0.08
    i_XB = 0.08
    i_XP = 0.06
    stoichiometric_parameters = (Y_A = Y_A, Y_H = Y_H, f_P = f_P, i_XB = i_XB, i_XP = i_XP)

    # Set other parameters
    V = 1333.0 # volume
    SO_sat = (8 / 10.50237016) * 6791.5 *
             (56.12 * exp(-66.7354 + 87.4755 / ((T + 273.15) / 100.0) +
                  24.4526 * log((T + 273.15) / 100.0)))
    KLa = 200 * (1.024^(T - 15)) # KLa
    other_params = (V = V, SO_sat = SO_sat, KLa = KLa)

    #Set inlet concentrations and inflow
    if influent_file_path ≠ nothing && variable_inlet_concentration
        X_in = get_inlet_concentrations_from_src_files(influent_file_path; kwargs...)
    else
        X_in = [28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699,
            964.8992, 0.0093, 3.9350, 6.8924, 0.9580, 3.8453, 5.4213]
    end

    if influent_file_path ≠ nothing && variable_inflow
        Q_in = get_inflow_from_bsm_files(influent_file_path; kwargs...)
    else
        Q_in = 18061.0
    end
    exogenous_params = (Q_in = Q_in, X_in = X_in)

    # Merge parameters
    p = merge(kinetic_parameters, stoichiometric_parameters, other_params, exogenous_params)

    # Set X_init
    X_init = [28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699, 964.8992,
        0.0093, 3.9350, 6.8924, 0.9580, 3.8453, 5.4213, 1.0]

    return (p, X_init)
end

"""
     $(TYPEDSIGNATURES)

Return the differential equations for the ASM1 model. See ``get_default_parameters_asm1`` for details about p structure.
"""
function asm1!(dX, X, p, t)

    # Compute stoichiometric_matrix
    Y_A = p[15]
    Y_H = p[16]
    f_P = p[17]
    i_XB = p[18]
    i_XP = p[19]
    R = @SMatrix [0 0 0 0 0 0 0 0;
                  -1/Y_H -1/Y_H 0 0 0 0 1 0;
                  0 0 0 0 0 0 0 0;
                  0 0 0 1-f_P 1-f_P 0 -1 0;
                  1 1 0 -1 0 0 0 0;
                  0 0 1 0 -1 0 0 0;
                  0 0 0 f_P f_P 0 0 0;
                  -(1 - Y_H)/Y_H 0 -4.57 / Y_A+1 0 0 0 0 0;
                  0 -(1 - Y_H)/(2.86 * Y_H) 1.0/Y_A 0 0 0 0 0;
                  -i_XB -i_XB -(i_XB + (1.0 / Y_A)) 0 0 1 0 0;
                  0 0 0 0 0 -1 0 1;
                  0 0 0 (i_XB-f_P * i_XP) (i_XB-f_P * i_XP) 0 0 -1;
                  -i_XB/14 (1 - Y_H) / (14 * 2.86 * Y_H)-(i_XB / 14) -(i_XB / 14)+1 / (7 * Y_A) 0 0 1/14 0 0]

    # Compute process rates
    K_OH = p[3]
    saturation_1 = p[1] * (X[2] / (p[2] + X[2]))
    saturation_2 = (X[8] / (K_OH + X[8]))
    saturation_3 = (K_OH / (K_OH + X[8]))
    saturation_4 = (X[9] / (p[4] + X[9]))
    penultimate_term = p[8] * ((X[4] / X[5]) / (p[9] + (X[4] / X[5]))) *
                       (saturation_2 + p[7] * saturation_3 * saturation_4) * X[5]
    process_rates = @SArray [saturation_1 * saturation_2 * X[5], # Aerobic growth of heterotrophs
        saturation_1 * saturation_3 * saturation_4 * p[6] * X[5], # Anoxic growth of heterotrophs
        p[10] * (X[10] / (p[11] + X[10])) * (X[8] / (p[13] + X[8])) * X[6], # Aerobic growth of autotrophs
        p[5] * X[5], # "Decay" of heterotrophs
        p[12] * X[6], # "Decay" of autotrophs
        p[14] * X[11] * X[5], # Ammonification of soluble organic nitrogen
        penultimate_term, # "Hydrolysis" of entrapped organics
        penultimate_term * X[12] / X[4]] # "Hydrolysis" of entrapped organics nitrogen

    # Compute differential equations
    dX[1:13] = (evaluate(p[23], t) / p[20]) * (evaluate.(p[24], t) - X[1:13]) +
               R * process_rates
    dX[14] = 0.0

    # Control input for oxygen
    dX[8] += X[14] * p[22] * (p[21] - X[8])
end
