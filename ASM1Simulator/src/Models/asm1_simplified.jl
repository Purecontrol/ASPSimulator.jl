using Statistics

"""
Return the differential equations for the simplified ASM1 model.

# Argument p
- `p[1]::Array{Float64}[8]` : Kinetic parameters
- `p[2]::Array{Float64}[5]` : Additional parameters
- `p[3]::Array{Float64}[5, 5]` : Stoichiometric matrix
- `p[4]::Float64` : Volume of the reactor
- `p[5]::Float64` : Biomass concentration at the inlet
- `p[6]::Float64` : Flow rate at the inlet
- `p[7]::Float64` : Saturation coefficient of oxygen
- `p[8]::Float64` : KLa

"""
function simplified_asm1!(dX, X, p, t)

    ### Retrieve parameters ###
    # Kinetic parameters
    K_DCO = p[1][1] ; K_OH = p[1][2] ; K_NO = p[1][3] ; η_g = p[1][4] ; η_h = p[1][5] ; K_ND = p[1][6]; K_NH = p[1][7] ; K_OA =  p[1][8]
    #Additional parameters
    θ_1 = p[2][1] ; θ_2 = p[2][2] ; θ_3 = p[2][3] ; θ_4 = p[2][4] ; θ_5 = p[2][5]
    # Matrice with stoichiometric parameters
    R = p[3]
    # Other parameters 
    volume = p[4] ; X_in = p[5] ; Q_in = p[6] ; SO_sat = p[7] ; KLa = p[8]

    # If Q_in is a function of the time t, then evaluate it
    if typeof(Q_in) <: Function
         Q_in = Q_in(t)
    end

    # If X_in is a function of the time t, then evaluate it
    if typeof(X_in) <: Function
         X_in = X_in(t)
    end

    ### Calculate process rates ###
    process_rates = [θ_1*(X[1]/(K_DCO+X[1]))*(X[2]/(K_OH+X[2])), # Aerobic growth of heterotrophs
    θ_1*η_g*(X[1]/(K_DCO+X[1]))*(K_OH/(K_OH+X[2]))*(X[3]/(K_NO+X[3])), # Anoxic growth of heterotrophs
    θ_3*(X[4]/(K_NH+X[4]))*(X[2]/(K_OA+X[2])), # Aerobic growth of autotrophs
    θ_2, # "Decay" of heterotrophs and autotrophs
    θ_4*X[5], # Ammonification of soluble organic nitrogen
    θ_5*((X[1])/(K_ND+X[1]))*((X[2]/(K_OH+X[2]))+η_h*(K_OH/(K_OH+X[2]))*(X[3]/(K_NO+X[3])))] # "Hydrolysis" of entrapped organics nitrogen

    ### Calculate differential equations ###
    # General expression
    dX[1:5] .= (Q_in/volume) * (X_in .- X[1:5]) .+ R * process_rates
    dX[6] = 0.0

    # Control input for oxygen
    dX[2] += X[6] * KLa * (SO_sat - X[2])

end

"""
Return the stoichiometric matrix of the simplified ASM1 model from the parameters given in stoichiometric_parameters.
"""
function get_stoichiometric_matrix_simplified_asm1(stoichiometric_parameters)

    Y_A = stoichiometric_parameters[1] ; Y_H = stoichiometric_parameters[2] ; i_XB = stoichiometric_parameters[3]
    
    R = [ -1/Y_H         -1/Y_H                 0                       1      0     0;
              -(1-Y_H)/Y_H   0                      -4.57/Y_A+1         0      0     0;
              0              -(1-Y_H)/(2.86*Y_H)     1.0/Y_A            0      0     0;
              -i_XB          -i_XB                  -(i_XB+(1.0/Y_A))   0      1     0;
              0              0                      0                   0     -1     1]

    return R

end

"""
Return the default parameters for the simplified ASM1 model.
"""
function get_default_parameters_simplified_asm1(; T = 15, get_R::Bool=true, influent_file_path = nothing, fixed_concentration = nothing)

     ### Define the function that adapts the parameters according to the temperature ###
     function T_var(T, ρ, a)
          return ρ * exp((log2(ρ/a)/5)*(T-15))
     end  

     ### Set vector with default parameters ###
     p = []

     ### Additional_parameters ###
     if fixed_concentration ≠ nothing
          X_BH = fixed_concentration[1]; X_BA = fixed_concentration[2]; X_ND = fixed_concentration[3]; X_S = fixed_concentration[4]; S_S = fixed_concentration[5]
     else
          X_BH = 2238; X_BA = 167; X_ND = 3.10; X_S = 44; S_S = 0.93
     end
     θ_1 = T_var(T, 4.0, 3)*mean(X_BH) #μ_H*X_BH
     θ_2 = (1-0.08)*(T_var(T, 0.3, 0.2)*mean(X_BH) + T_var(T, 0.05, 0.03)*mean(X_BA)) #(1-f_P)*(b_H*X_BH + b_A*X_BA)
     θ_3 = T_var(T, 0.5, 0.3)*mean(X_BA) #μ_A*X_BA
     θ_4 = T_var(T, 0.05, 0.04)*mean(X_BH) #k_a*X_BH
     θ_5 = T_var(T, 3.0, 2.5)*mean(X_BH.*X_ND./X_S)#(X_BH*X_ND*k_h)/X_S
     additional_parameters = [θ_1, θ_2, θ_3, θ_4, θ_5]

     ### Kinetic parameters ###
     K_DCO = 10.0*mean((X_S .+ S_S)./S_S) ; K_OH = 0.2 ; K_NO = 0.5; η_g = 0.8 ; η_h = 0.8; K_ND = 0.1*mean(((X_S .+ S_S)./X_S).*X_BH); K_NH = 1.0; K_OA =  0.4
     kinetic_parameters = [K_DCO, K_OH, K_NO, η_g, η_h, K_ND, K_NH, K_OA]
     push!(p, kinetic_parameters)

     push!(p, additional_parameters)

     ### Stoichiometric parameters ###
     Y_A = 0.24 ; Y_H = 0.67 ; i_XB = 0.08
     if get_R
          R = [ -1/Y_H         -1/Y_H                 0                       1      0     0;
                    -(1-Y_H)/Y_H   0                      -4.57/Y_A+1         0      0     0;
                    0              -(1-Y_H)/(2.86*Y_H)     1.0/Y_A            0      0     0;
                    -i_XB          -i_XB                  -(i_XB+(1.0/Y_A))   0      1     0;
                    0              0                      0                   0     -1     1]
          push!(p, R)
     else
          stoichiometric_parameters = [Y_A, Y_H, i_XB]
          push!(p, stoichiometric_parameters)
     end

     ### Other parameters ###
     push!(p, 1333.0) # volume

     ### Influent concentrations ###
     X_in =  [3.0503 + 63.0433, 0.0093, 3.9350, 6.8924, 0.9580]
     push!(p, X_in) # X_in

     if influent_file_path ≠ nothing
          inflow_generator = readdlm(influent_file_path)
          function Q_in(t) 
               return interpolate((inflow_generator[: ,1], ), inflow_generator[: ,10], Gridded(Linear()))(abs(t) .% maximum(inflow_generator[: ,1]))
          end
     else
          Q_in = 18061.0
     end
     push!(p, Q_in) # Q_in

     ### Control parameters : Redox control ###
     push!(p, (8/10.50237016)*6791.5*(56.12*exp(-66.7354 + 87.4755/((T+273.15)/100.0) + 24.4526*log((T+273.15)/100.0)))) # SO_sat
     push!(p, 200*(1.024^(T-15))) # KLa

     ### X_init ###
     X_init =  [3.0503 + 63.0433, 0.0093, 3.9350, 6.8924, 0.9580, 1.0]

     return (p, X_init)

end

function get_bounds_parameters_simplified_asm1()

     ### Set the lower and upper bounds vectors###
     p_lower = [] ; p_upper = []

     ### Kinetic parameters ###
     K_DCO_lower = 1.0 ; K_OH_lower = 0.1 ; K_NO_lower = 0.25 ; η_g_lower = 0.64 ; η_h_lower = 0.64 ; K_ND_lower = 1.0 ; K_NH_lower = 0.5 ; K_OA_lower =  0.2
     kinetic_parameters_lower = [K_DCO_lower, K_OH_lower, K_NO_lower, η_g_lower, η_h_lower, K_ND_lower, K_NH_lower, K_OA_lower]
     push!(p_lower, kinetic_parameters_lower)

     K_DCO_upper = 1000.0 ; K_OH_upper = 0.3 ; K_NO_upper = 0.75 ; η_g_upper = 0.96 ; η_h_upper = 0.96 ; K_ND_upper = 1000.0 ; K_NH_upper = 1.5; K_OA_upper =  0.6
     kinetic_parameters_upper = [K_DCO_upper, K_OH_upper, K_NO_upper, η_g_upper, η_h_upper, K_ND_upper, K_NH_upper, K_OA_upper]
     push!(p_upper, kinetic_parameters_upper)

     ### Additional parameters ###
     θ_1_lower = 5000.0 ; θ_2_lower = 50.0 ; θ_3_lower = 10.0 ; θ_4_lower = 10.0 ; θ_5_lower = 1.0
     additional_parameters_lower = [θ_1_lower, θ_2_lower, θ_3_lower, θ_4_lower, θ_5_lower]
     push!(p_lower, additional_parameters_lower)

     θ_1_upper = 50000.0 ; θ_2_upper = 10000.0 ; θ_3_upper = 500.0 ; θ_4_upper = 500.0 ; θ_5_upper = 1000.0
     additional_parameters_upper = [θ_1_upper, θ_2_upper, θ_3_upper, θ_4_upper, θ_5_upper]
     push!(p_upper, additional_parameters_upper)

     ### Stoichiometric parameters ###
     Y_A_lower = 0.07 ; Y_H_lower = 0.46 ; i_XB_lower = 0.076
     stoichiometric_parameters_lower = [Y_A_lower, Y_H_lower, i_XB_lower]
     push!(p_lower, stoichiometric_parameters_lower)

     Y_A_upper = 0.28 ; Y_H_upper = 0.69 ; i_XB_upper = 0.084
     stoichiometric_parameters_upper = [Y_A_upper, Y_H_upper, i_XB_upper]
     push!(p_upper, stoichiometric_parameters_upper)

     ### Other parameters ###
     push!(p_lower, 600.0) # volume
     push!(p_upper, 3000.0) # volume

     push!(p_lower, [28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699, 964.8992, 0.0093, 3.9350, 6.8924, 0.9580, 3.8453, 5.4213]) # X_in
     push!(p_upper, [28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699, 964.8992, 0.0093, 3.9350, 6.8924, 0.9580, 3.8453, 5.4213]) # X_in

     push!(p_lower, 18061.0) # Q_in
     push!(p_upper, 18061.0) # Q_in

     ### Control parameters : Redox control ###
     push!(p_lower, 4.36) # SO_sat
     push!(p_upper, 11.46) # SO_sat
     push!(p_lower, 100) # KLa
     push!(p_upper, 400) # KLa

     ### X_init ###
     X_init_lower = [0, 0, 0, 0, 0, 1.0]
     X_init_upper = [100.0, 100.0, 100.0, 100.0, 100.0, 1.0]

     return (p_lower, p_upper, X_init_lower, X_init_upper)
end