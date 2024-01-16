using Statistics
using StaticArrays

"""
Return the default parameters and an initial condition for the simplified ASM1 model.
"""
function get_default_parameters_simplified_asm1(; T = 15, influent_file_path = nothing, fixed_concentration = nothing, variable_inlet_concentration = true, variable_inflow = true)

     # Set additional parameters defined by simplifications
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
     additional_parameters = (θ_1=θ_1, θ_2=θ_2, θ_3=θ_3, θ_4=θ_4, θ_5=θ_5)


     # Set kinetics parameters
     K_DCO = 10.0*mean((X_S .+ S_S)./S_S) ; K_OH = 0.2 ; K_NO = 0.5; η_g = 0.8 ; η_h = 0.8; K_ND = 0.1*mean(((X_S .+ S_S)./X_S).*X_BH); K_NH = 1.0; K_OA =  0.4
     kinetic_parameters = (K_DCO=K_DCO, K_OH=K_OH, K_NO=K_NO, η_g=η_g, η_h=η_h, K_ND=K_ND, K_NH=K_NH, K_OA=K_OA)

     # Set stoichiometric parameters
     Y_A = 0.24 ; Y_H = 0.67 ; i_XB = 0.08
     stoichiometric_parameters = (Y_A=Y_A, Y_H=Y_H, i_XB=i_XB)

     # Set other parameters
     V = 1333.0 # volume
     SO_sat = (8/10.50237016)*6791.5*(56.12*exp(-66.7354 + 87.4755/((T+273.15)/100.0) + 24.4526*log((T+273.15)/100.0))) 
     KLa = 200*(1.024^(T-15))
     other_params = (V=V, SO_sat=SO_sat, KLa=KLa)

     #Set inlet concentrations and inflow
     if influent_file_path ≠ nothing && variable_inlet_concentration
          X_in = get_inlet_concentrations_from_src_files(influent_file_path, output_index=[[2, 4], 8, 9, 10, 11])
     else
          X_in =  [3.0503 + 63.0433, 0.0093, 3.9350, 6.8924, 0.9580]
     end

     if influent_file_path ≠ nothing && variable_inflow
          Q_in = get_inflow_from_bsm_files(influent_file_path)
     else
          Q_in = 18061.0
     end
     exogenous_params = (Q_in=Q_in, X_in=X_in)

     # Merge parameters
     p = merge(kinetic_parameters, additional_parameters, stoichiometric_parameters, other_params, exogenous_params)

     # Set X_init
     X_init =  [3.0503 + 63.0433, 0.0093, 3.9350, 6.8924, 0.9580, 1.0]

     return (p, X_init)

end


"""
Return the differential equations for the simplified ASM1 model. See ``get_default_parameters_simplified_asm1`` for details about p structure.
"""
function simplified_asm1!(dX, X, p::NamedTuple, t)

     # Compute stoichiometric_matrix
     Y_A = p[14] ; Y_H = p[15] ; i_XB = p[16]
     R = @SMatrix[ -1/Y_H          -1/Y_H                 0                  1      0     0;
                   -(1-Y_H)/Y_H    0                      -4.57/Y_A+1        0      0     0;
                   0               -(1-Y_H)/(2.86*Y_H)    1.0/Y_A            0      0     0;
                   -i_XB           -i_XB                  -(i_XB+(1.0/Y_A))  0      1     0;
                   0               0                      0                  0     -1     1]
 
     # Compute process rates
     K_OH = p[2]
     saturation_oxy_1 = (X[2]/(K_OH+X[2]))
     saturation_dco = p[9]*(X[1]/(p[1]+X[1]))
     saturation_no = (X[3]/(p[3]+X[3]))
     saturation_oxy2_no = saturation_no*K_OH/(K_OH+X[2])
     process_rates = @SArray [saturation_dco*saturation_oxy_1,
                              p[4]*saturation_dco*saturation_oxy2_no, 
                              p[11]*(X[4]/(p[7]+X[4]))*(X[2]/(p[8]+X[2])), 
                              p[10], 
                              p[12]*X[5], 
                              p[13]*((X[1])/(p[6]+X[1]))*(saturation_oxy_1+p[5]*saturation_oxy2_no)]
     
     # Compute differential equations
     dX[1:5] = (evaluate(p[20], t)/p[17]) * (evaluate.(p[21],t) - X[1:5]) + R * process_rates
     dX[6] = 0.0
     dX[2] += X[6] * p[19] * (p[18] - X[2])

end


"""
Return the differential equations for the simplified ASM1 model. Here, it p is an ``Array`` which is useful for parameter optimization.
"""
function simplified_asm1!(dX, X, p::Array, t)

     # Compute stoichiometric_matrix
     Y_A = p[14] ; Y_H = p[15] ; i_XB = p[16]
     R = @SMatrix[ -1/Y_H          -1/Y_H                 0                  1      0     0;
                   -(1-Y_H)/Y_H    0                      -4.57/Y_A+1        0      0     0;
                   0               -(1-Y_H)/(2.86*Y_H)    1.0/Y_A            0      0     0;
                   -i_XB           -i_XB                  -(i_XB+(1.0/Y_A))  0      1     0;
                   0               0                      0                  0     -1     1]
 
     # Compute process rates
     K_OH = p[2]
     saturation_oxy_1 = (X[2]/(K_OH+X[2]))
     saturation_dco = p[9]*(X[1]/(p[1]+X[1]))
     saturation_no = (X[3]/(p[3]+X[3]))
     saturation_oxy2_no = saturation_no*K_OH/(K_OH+X[2])
     process_rates = @SArray [saturation_dco*saturation_oxy_1,
                              p[4]*saturation_dco*saturation_oxy2_no, 
                              p[11]*(X[4]/(p[7]+X[4]))*(X[2]/(p[8]+X[2])), 
                              p[10], 
                              p[12]*X[5], 
                              p[13]*((X[1])/(p[6]+X[1]))*(saturation_oxy_1+p[5]*saturation_oxy2_no)]
     
     # Compute differential equations
     dX[1:5] = (evaluate(p[20], t)/p[17]) * (evaluate.(p[21:25],t) - X[1:5]) + R * process_rates
     dX[6] = 0.0
     dX[2] += X[6] * p[19] * (p[18] - X[2])

end