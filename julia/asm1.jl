using DifferentialEquations, ModelingToolkit, LaTeXStrings

# Define the variables of the system
X_init = [28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699, 964.8992, 0.0093, 3.9350, 6.8924, 0.9580, 3.8453, 5.4213]
@variables t X(t)[1:13] = X_init
X_names = latexstring.(["S_I" "S_S" "X_I" "X_S" "X_{B,H}" "X_{B,A}" "X_P" "S_O" "S_{NO}" "S_{NH}" "S_{ND}" "X_{ND}" "S_{ALK}"])

# Define the parameters of the system
@parameters μ_H = 4.0 K_S = 10.0 K_OH = 0.2 K_NO = 0.5 b_H = 0.3 η_g = 0.8 η_h = 0.8 k_h = 3.0 K_X = 0.1 μ_A = 0.5 K_NH = 1.0 b_A = 0.05 K_OA =  0.4 k_a = 0.05 [description = "Kinetic parameters"]
kinetic_parameters_names = latexstring.(["μ_H" "K_S" "K_{OH}" "K_{NO}" "b_H" "η_g" "η_h" "k_h" "K_X" "μ_A" "K_{NH}" "b_A" "K_{OA}" "k_a"])
@parameters Y_A = 0.24 Y_H = 0.67 f_P = 0.08 i_XB = 0.08 i_XP = 0.06 [description = "Stoichiometric parameters"]
stoichiometric_parameters_names = latexstring.(["Y_A" "Y_H" "f_P" "i_{XB}" "i_{XP}"])
@parameters volume = 1000.0 [description = "Volume of the tank"]
other_parameters_names = latexstring.(["V"])
@parameters X_in[1:13] = X_init Q_in=226.0 [description = "Input flow and concentration"]
input_parameters_names = latexstring.(["S_I^{in}" "S_S^{in}" "X_I^{in}" "X_S^{in}" "X_{B,H}^{in}" "X_{B,A}^{in}" "X_P^{in}" "S_O^{in}" "S_{NO}^{in}" "S_{NH}^{in}" "S_{ND}^{in}" "X_{ND}^{in}" "S_{ALK}^{in}" "Q_{in}"])

# Define our differential: takes the derivative with respect to `t`
D = Differential(t)

# Define the differential equations
R = [0 0 0 0 0 0 0 0; (-1/Y_H) (-1/Y_H) 0 0 0 0 1 0;
     0 0 0 0 0 0 0 0; 0 0 0 (1-f_P) (1-f_P) 0 -1 0;
     1 1 0 -1 0 0 0 0; 0 0 1 0 -1 0 0 0;
     0 0 0 (f_P) (f_P) 0 0 0; -((1-Y_H)/Y_H) 0 (-(4.57/Y_A)+1.0) 0 0 0 0 0;
     0 -((1-Y_H)/(2.86*Y_H)) (1.0/Y_A) 0 0 0 0 0; -i_XB -i_XB -(i_XB+(1.0/Y_A)) 0 0 1 0 0;
     0 0 0 0 0 -1 0 1; 0 0 0 (i_XB-f_P*i_XP) (i_XB-f_P*i_XP) 0 0 -1;
     -i_XB/14.0 ((1.0-Y_H)/(14.0*2.86*Y_H)-(i_XB/14.0)) -((i_XB/14.0)+1.0/(7.0*Y_A)) 0 0 1/14 0 0]

process_rates = [μ_H*(X[2]/(K_S+X[2]))*(X[8]/(K_OH+X[8]))*X[5], # Aerobic growth of heterotrophs
                 μ_H*(X[2]/(K_S+X[2]))*(K_OH/(K_OH+X[8]))*(X[9]/(K_NO+X[9]))*η_g*X[5], # Anoxic growth of heterotrophs
                 μ_A*(X[10]/(K_NH+X[10]))*(X[8]/(K_OA+X[8]))*X[6], # Aerobic growth of autotrophs
                 b_H*X[5], # "Decay" of heterotrophs
                 b_A*X[6], # "Decay" of autotrophs
                 k_a*X[11]*X[5], # Ammonification of soluble organic nitrogen
                 k_h*((X[4]/X[5])/(K_X+(X[4]/X[5])))*((X[8]/(K_OH+X[8]))+η_h*(K_OH/(K_OH+X[8]))*(X[9]/(K_NO+X[9])))*X[5], # "Hydrolysis" of entrapped organics
                 (k_h*((X[4]/X[5])/(K_X+(X[4]/X[5])))*((X[8]/(K_OH+X[8]))+η_h*(K_OH/(K_OH+X[8]))*(X[9]/(K_NO+X[9])))*X[5])*X[12]/X[4]] # "Hydrolysis" of entrapped organics nitrogen

r = R*process_rates 

eqs = [D(X[i]) ~ (Q_in/volume)*(X_in[i] - X[i]) + r[i] for i in 1:7]

oxygen = 350
append!(eqs, [D(X[8]) ~ (Q_in/volume)*(X_in[8] - X[8]) + r[8] + oxygen*((t%1 < 0.3) + (t%1 > 0.4)*(t%1 < 0.5) + (t%1 > 0.8)*(t%1 < 0.9))*(8 - X[8])])

append!(eqs, [D(X[i]) ~ (Q_in/volume)*(X_in[i] - X[i]) + r[i] for i in 9:13])

 
# Bring these pieces together into an ODESystem with independent variable t
@named ASM1_sym = ODESystem(eqs,t)

# Symbolically Simplify the System
ASM1 = structural_simplify(ASM1_sym)