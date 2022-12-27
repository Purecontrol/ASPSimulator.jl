using DifferentialEquations, LaTeXStrings

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
     # Control input for oxygen
     dX[8] += X[14] * KLa * (SO_sat - X[8])

end


"""
Return the default parameters for ASM1 model.
"""
function get_default_parameters_asm1(; get_R::Bool=true)

     ### Set vector with default parameters ###
     p = []

     ### Kinetic parameters ###
     μ_H = 4.0 ; K_S = 10.0 ; K_OH = 0.2 ; K_NO = 0.5 ; b_H = 0.3 ; η_g = 0.8 ; η_h = 0.8 ; k_h = 3.0 ; K_X = 0.1 ; μ_A = 0.5 ; K_NH = 1.0 ; b_A = 0.05 ; K_OA =  0.4 ; k_a = 0.05
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
     push!(p, 8) # SO_sat
     push!(p, 200) # KLa

     ### X_init ###
     X_init =  [28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699, 964.8992, 0.0093, 3.9350, 6.8924, 0.9580, 3.8453, 5.4213, 0.0]

     return (p, X_init)

end