using DifferentialEquations

"""
Return the differential equations for ASM1 model.

# Argument p
- `p[1]::Array{Float64}[14]` : Kinetic parameters
- `p[2]::Array{Float64}[13, 13]` : Stoichiometric matrix
- `p[3]::Float64` : Volume of the reactor
- `p[4]::Float64` : Biomass concentration at the inlet
- `p[5]::Float64` : Flow rate at the inlet
- `p[6]::Float64` : Saturation coefficient of oxygen
- `p[7]::Float64` : KLa

"""
function asm1!(dX, X, p, t)

     ### Retrieve parameters ###
     # Kinetic parameters
     μ_H = p[1][1] ; K_S = p[1][2] ; K_OH = p[1][3] ; K_NO = p[1][4] ; b_H = p[1][5] ; η_g = p[1][6] ; η_h = p[1][7] ; k_h = p[1][8] ; K_X = p[1][9] ; μ_A = p[1][10] ; K_NH = p[1][11] ; b_A = p[1][12] ; K_OA =  p[1][13] ; k_a = p[1][14]
     # Matrice with stoichiometric parameters
     R = p[2]
     # Other parameters 
     volume = p[3] ; X_in = p[4] ; Q_in = p[5] ; SO_sat = p[6] ; KLa = p[7]

     ### Calculate process rates ###
     process_rates = [μ_H*(X[2]/(K_S+X[2]))*(X[8]/(K_OH+X[8]))*X[5], # Aerobic growth of heterotrophs
     μ_H*(X[2]/(K_S+X[2]))*(K_OH/(K_OH+X[8]))*(X[9]/(K_NO+X[9]))*η_g*X[5], # Anoxic growth of heterotrophs
     μ_A*(X[10]/(K_NH+X[10]))*(X[8]/(K_OA+X[8]))*X[6], # Aerobic growth of autotrophs
     b_H*X[5], # "Decay" of heterotrophs
     b_A*X[6], # "Decay" of autotrophs
     k_a*X[11]*X[5], # Ammonification of soluble organic nitrogen
     k_h*((X[4]/X[5])/(K_X+(X[4]/X[5])))*((X[8]/(K_OH+X[8]))+η_h*(K_OH/(K_OH+X[8]))*(X[9]/(K_NO+X[9])))*X[5], # "Hydrolysis" of entrapped organics
     (k_h*((X[4]/X[5])/(K_X+(X[4]/X[5])))*((X[8]/(K_OH+X[8]))+η_h*(K_OH/(K_OH+X[8]))*(X[9]/(K_NO+X[9])))*X[5])*X[12]/X[4]] # "Hydrolysis" of entrapped organics nitrogen

     ### Calculate differential equations ###
     # General expression
     dX[1:13] .= (Q_in/volume) * (X_in .- X[1:13]) .+ R * process_rates
     dX[14] = 0.0

     # Control input for oxygen
     dX[8] += X[14] * KLa * (SO_sat - X[8])

end


"""
Return the default parameters for ASM1 model.
"""
function get_default_parameters_asm1(; T = 15, get_R::Bool=true)

     ### Define the function that adapts the parameters according to the temperature ###
     function T_var(T, ρ, a)
          return ρ * exp((log2(ρ/a)/5)*(T-15))
     end  

     ### Set vector with default parameters ###
     p = []

     ### Kinetic parameters ###
     μ_H = T_var(T, 4.0, 3) ; K_S = 10.0 ; K_OH = 0.2 ; K_NO = 0.5 ; b_H = T_var(T, 0.3, 0.2) ; η_g = 0.8 ; η_h = 0.8 ; k_h = T_var(T, 3.0, 2.5) ; K_X = 0.1 ; μ_A = T_var(T, 0.5, 0.3) ; K_NH = 1.0 ; b_A = T_var(T, 0.05, 0.03) ; K_OA =  0.4 ; k_a = T_var(T, 0.05, 0.04)
     kinetic_parameters = [μ_H, K_S, K_OH, K_NO, b_H, η_g, η_h, k_h, K_X, μ_A, K_NH, b_A, K_OA, k_a]
     push!(p, kinetic_parameters)

     ### Stoichiometric parameters ###
     Y_A = 0.24 ; Y_H = 0.67 ; f_P = 0.08 ; i_XB = 0.08 ; i_XP = 0.06
     if get_R
          R = [           0      0     0     0     0     0     0     0;
               -1/Y_H -1/Y_H     0     0     0     0     1     0;
                    0      0     0     0     0     0     0     0;
                    0      0     0 1-f_P 1-f_P     0    -1     0;
                    1      1     0    -1     0     0     0     0;
                    0      0     1     0    -1     0     0     0;
                    0      0     0   f_P   f_P     0     0     0;
          -(1-Y_H)/Y_H      0 -4.57/Y_A+1 0   0     0     0     0;
                    0 -(1-Y_H)/(2.86*Y_H) 1.0/Y_A 0 0 0  0     0;
               -i_XB -i_XB -(i_XB+(1.0/Y_A)) 0 0  1     0     0;
                    0      0     0     0     0    -1     0     1;
                    0      0     0 (i_XB-f_P*i_XP) (i_XB-f_P*i_XP) 0 0 -1;
          -i_XB/14 (1-Y_H)/(14*2.86*Y_H)-(i_XB/14) -(i_XB/14)+1/(7*Y_A) 0 0 1/14 0 0]
          push!(p, R)
     else
          stoichiometric_parameters = [Y_A, Y_H, f_P, i_XB, i_XP]
          push!(p, stoichiometric_parameters)
     end

     ### Other parameters ###
     push!(p, 1333.0) # volume
     push!(p, [28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699, 964.8992, 0.0093, 3.9350, 6.8924, 0.9580, 3.8453, 5.4213]) # X_in
     push!(p, 18061.0) # Q_in

     ### Control parameters : Redox control ###
     push!(p, (8/10.50237016)*6791.5*(56.12*exp(-66.7354 + 87.4755/((T+273.15)/100.0) + 24.4526*log((T+273.15)/100.0)))) # SO_sat
     push!(p, 200*(1.024^(T-15))) # KLa

     ### X_init ###
     X_init =  [28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699, 964.8992, 0.0093, 3.9350, 6.8924, 0.9580, 3.8453, 5.4213, 1.0]

     return (p, X_init)

end

"""
Return the lower and upper bounds for the parameters and the initial condtions of the ASM1 model given in the literature.
"""
function get_bounds_parameters_asm1()

     ### Set the lower and upper bounds vectors###
     p_lower = [] ; p_upper = []

     ### Kinetic parameters ###
     μ_H_lower = 3.0 ; K_S_lower = 5.0 ; K_OH_lower = 0.1 ; K_NO_lower = 0.25 ; b_H_lower = 0.05 ; η_g_lower = 0.64 ; η_h_lower = 0.64 ; k_h_lower = 0.05 ; K_X_lower = 0.015 ; μ_A_lower = 0.34 ; K_NH_lower = 0.5 ; b_A_lower = 0.05 ; K_OA_lower =  0.2 ; k_a_lower = 0.025
     kinetic_parameters_lower = [μ_H_lower, K_S_lower, K_OH_lower, K_NO_lower, b_H_lower, η_g_lower, η_h_lower, k_h_lower, K_X_lower, μ_A_lower, K_NH_lower, b_A_lower, K_OA_lower, k_a_lower]
     push!(p_lower, kinetic_parameters_lower)

     μ_H_upper = 13.2 ; K_S_upper = 15.0 ; K_OH_upper = 0.3 ; K_NO_upper = 0.75 ; b_H_upper = 1.6 ; η_g_upper = 0.96 ; η_h_upper = 0.96 ; k_h_upper = 4.5 ; K_X_upper = 0.045 ; μ_A_upper = 0.65 ; K_NH_upper = 1.5 ; b_A_upper = 0.15 ; K_OA_upper =  0.6 ; k_a_upper = 0.075
     kinetic_parameters_upper = [μ_H_upper, K_S_upper, K_OH_upper, K_NO_upper, b_H_upper, η_g_upper, η_h_upper, k_h_upper, K_X_upper, μ_A_upper, K_NH_upper, b_A_upper, K_OA_upper, k_a_upper]
     push!(p_upper, kinetic_parameters_upper)

     ### Stoichiometric parameters ###
     Y_A_lower = 0.07 ; Y_H_lower = 0.46 ; f_P_lower = 0.076 ; i_XB_lower = 0.076 ; i_XP_lower = 0.057
     stoichiometric_parameters_lower = [Y_A_lower, Y_H_lower, f_P_lower, i_XB_lower, i_XP_lower]
     push!(p_lower, stoichiometric_parameters_lower)

     Y_A_upper = 0.28 ; Y_H_upper = 0.69 ; f_P_upper = 0.084 ; i_XB_upper = 0.084 ; i_XP_upper = 0.063
     stoichiometric_parameters_upper = [Y_A_upper, Y_H_upper, f_P_upper, i_XB_upper, i_XP_upper]
     push!(p_upper, stoichiometric_parameters_upper)

     ### Other parameters ###
     push!(p_lower, 1333.0) # volume
     push!(p_upper, 1333.0) # volume

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
     X_init_lower = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]
     X_init_upper = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1.0]

     return (p_lower, p_upper, X_init_lower, X_init_upper)
end

"""
Return the stoichiometric matrix of the ASM1 model from the parameters given in stoichiometric_parameters.
"""
function get_stoichiometric_matrix(stoichiometric_parameters)

     Y_A = stoichiometric_parameters[1] ; Y_H = stoichiometric_parameters[2] ; f_P = stoichiometric_parameters[3] ; i_XB = stoichiometric_parameters[4] ; i_XP = stoichiometric_parameters[5]
     
     R = [           0      0     0     0     0     0     0     0;
          -1/Y_H -1/Y_H     0     0     0     0     1     0;
               0      0     0     0     0     0     0     0;
               0      0     0 1-f_P 1-f_P     0    -1     0;
               1      1     0    -1     0     0     0     0;
               0      0     1     0    -1     0     0     0;
               0      0     0   f_P   f_P     0     0     0;
          -(1-Y_H)/Y_H      0 -4.57/Y_A+1 0   0     0     0     0;
                    0 -(1-Y_H)/(2.86*Y_H) 1.0/Y_A 0 0 0  0     0;
               -i_XB -i_XB -(i_XB+(1.0/Y_A)) 0 0  1     0     0;
                    0      0     0     0     0    -1     0     1;
                    0      0     0 (i_XB-f_P*i_XP) (i_XB-f_P*i_XP) 0 0 -1;
          -i_XB/14 (1-Y_H)/(14*2.86*Y_H)-(i_XB/14) -(i_XB/14)+1/(7*Y_A) 0 0 1/14 0 0]

     return R

end


