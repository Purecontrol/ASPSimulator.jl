using Catalyst


function M_t(x, exogenous, u, params)

    # params = max.(params, 0.0000001)

    # ode_params = Array{Any, 1}(undef, 8)
    # ode_params[1] = params[1:8]
    # ode_params[2] = params[9:13]
    # ode_params[3] = ASM1Simulator.Models.get_stoichiometric_matrix_simplified_asm1(params[14:16])
    # ode_params[4] = params[17]
    # ode_params[5] = exogenous[1:5]
    # ode_params[6] = Q_in #exogenous[6]
    # ode_params[7] = params[18]
    # ode_params[8] = params[19]

    # ode_params = simplified_params_tot

    # ode_fct! = ASM1Simulator.Models.simplified_asm1!

    ode_problem = ODEProblem(ode_fct!, vcat(x[:, 1], u),  (exogenous[7], exogenous[7] + exogenous[8]), ode_params)

    n_particules = size(x, 2)
    function prob_func(prob, i, repeat)
        remake(ode_problem, u0 = vcat(x[:, i], u))
    end
    monte_prob = EnsembleProblem(ode_problem, prob_func = prob_func)


    sim_results = solve(monte_prob, Euler(), dt = 1/(6*24*60), trajectories = n_particules, saveat=[exogenous[7] + exogenous[8]], maxiters=10e3, sensealg = ForwardSensitivity())

    return hcat([max.(sim_results[i].u[1][1:5], 0.0) for i in 1:n_particules]...)

end

47.749 μs (729 allocations: 22.69 KiB)


function M_t(x, exogenous, u, params)

    # params = max.(params, 0.0000001)

    # ode_params = Array{Any, 1}(undef, 8)
    # ode_params[1] = params[1:8]
    # ode_params[2] = params[9:13]
    # ode_params[3] = ASM1Simulator.Models.get_stoichiometric_matrix_simplified_asm1(params[14:16])
    # ode_params[4] = params[17]
    # ode_params[5] = exogenous[1:5]
    # ode_params[6] = Q_in #exogenous[6]
    # ode_params[7] = params[18]
    # ode_params[8] = params[19]

    # ode_params = simplified_params_tot

    # ode_fct! = ASM1Simulator.Models.simplified_asm1!

    ode_problem = ODEProblem(ode_fct!, vcat(x[:, 1], u),  (exogenous[7], exogenous[7] + exogenous[8]), ode_params, R=ode_params[3])

    sim_results = solve(ode_problem, Euler(), dt = 1/(6*24*60), saveat=[exogenous[7] + exogenous[8]], maxiters=10e3, sensealg = ForwardSensitivity());

end


temp = vcat(simplified_params_tot[vcat(1:2, 4, 6:8)]...)
# params_test = (K_OH = temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7], temp[8], temp[9], temp[10], temp[11], temp[12], temp[13], simplified_params_tot[3], temp[14], simplified_params_tot[5], temp[15], temp[16], temp[17])
params_test = (K_DCO = temp[1],K_OH = temp[2],K_NO = temp[3],η_g = temp[4],η_h = temp[5], K_ND = temp[6], K_NH = temp[7], K_OA = temp[8],θ_1 = temp[9],θ_2 = temp[10],θ_3 = temp[11],θ_4 = temp[12],θ_5 = temp[13], R = simplified_params_tot[3], V = temp[14], X_in = simplified_params_tot[5], Q_in = temp[15], SO_sat = temp[16], KLa = temp[17])
ode_params = params_test

ode_params = vcat(simplified_params...) #
ode_params = SArray{Tuple{25}}(vcat(simplified_params...))
ode_params[23] = Q_in(20.0) 
# ode_params[6] = Q_in(20.0) 
# ode_fct! = ASM1Simulator.Models.simplified_asm1_past! #ASM1Simulator.Models.simplified_asm1!
ode_fct! = ASM1Simulator.Models.simplified_asm1_best!
ode_init = [x_init[1, 1], x_init[2, 1], x_init[3, 1], x_init[4, 1], x_init[5, 1], u]
ode_problem = ODEProblem(ode_fct!, ode_init,  (exogenous[7], exogenous[7] + exogenous[8]), ode_params)

de = modelingtoolkitize(ode_problem)

ode_problem2 = ODEProblem(de, jac=true, sparse=true)

function M2_t(x, exogenous, u, params)

    ode2 = remake(ode_problem, u0 = x,  tspan=(exogenous[7], exogenous[7] + exogenous[8]))

    sim_results = solve(ode2, Tsit5(), saveat=[exogenous[7] + exogenous[8]], maxiters=10e3, sensealg = ForwardSensitivity());

end


@btime M_t(x_init, exogenous_matrix[1, :], U_train[1, :], params);




# @timed M_t(x_init ,exogenous_matrix[1, :], U_train[1, :], params_ter)



x_init_2 = @SArray [x_init[1], x_init[2], x_init[3], x_init[4], x_init[5]]


ode_init = @SArray [x_init[1, 1], x_init[2, 1], x_init[3, 1], x_init[4, 1], x_init[5, 1], u]
ode_problem = ODEProblem(ode_fct!, ode_init,  (exogenous[7], exogenous[7] + exogenous[8]), ode_params)


@btime solve(ode_problem, Euler(), dt = 1/(6*24*60), saveat=[exogenous[7] + exogenous[8]], maxiters=10e3, sensealg = ForwardSensitivity());


# @btime M_t(x_init ,exogenous_matrix[1, :], U_train[1, :], params_ter);



# Separation saturation => 0.0000628 #(tps/10)

# Whithout if => 0.000112496455 #(tps/5)

# SMatrix => 0.0005923

# ModelingToolkit => 7.575055e-6 sans passer par EnsembleProblem a checker !!

using ModelingToolkit

@variables t X_DCO(t) S_O(t) S_NO(t) S_NH(t) S_ND(t) U(t)
@parameters V, Q_in, KLa, SO_sat
@parameters X_DCO_in S_O_in S_NO_in S_NH_in S_ND_in
@parameters K_DCO K_OH K_NO η_g η_h K_ND K_NH K_OA
@parameters θ_1 θ_2 θ_3 θ_4 θ_5
@parameters Y_A Y_H i_XB

D = Differential(t)


R = [ -1/Y_H         -1/Y_H                 0                       1      0     0;
-(1-Y_H)/Y_H   0                      -4.57/Y_A+1         0      0     0;
0              -(1-Y_H)/(2.86*Y_H)     1.0/Y_A            0      0     0;
-i_XB          -i_XB                  -(i_XB+(1.0/Y_A))   0      1     0;
0              0                      0                   0     -1     1]


process_rates = [θ_1*(X_DCO/(K_DCO+X_DCO))*(S_O/(K_OH+S_O)),
θ_1*η_g*(X_DCO/(K_DCO+X_DCO))*(K_OH/(K_OH+S_O))*(S_NO/(K_NO+S_NO)), 
θ_3*(S_NH/(K_NH+S_NH))*(S_O/(K_OA+S_O)), 
θ_2, 
θ_4*S_ND, 
θ_5*((X_DCO)/(K_ND+X_DCO))*((S_O/(K_OH+S_O))+η_h*(K_OH/(K_OH+S_O))*(S_NO/(K_NO+S_NO)))] 


eqs2 = [D(X_DCO) ~ (Q_in/V)*(X_DCO_in - X_DCO) + sum(R[1,:].*process_rates), 
       D(S_O) ~ (Q_in/V)*(S_O_in - S_O) + U * KLa * (SO_sat - S_O) + sum(R[2,:].*process_rates), 
       D(S_NO) ~ (Q_in/V)*(S_NO_in - S_NO) + sum(R[3,:].*process_rates), 
       D(S_NH) ~ (Q_in/V)*(S_NH_in - S_NH) + sum(R[4,:].*process_rates), 
       D(S_ND) ~ (Q_in/V)*(S_ND_in - S_ND) + sum(R[5,:].*process_rates), 
       D(U) ~ 0.0]

@named sys3 = ODESystem(eqs2)

simplified_final = structural_simplify(sys3)



results = @timed for i in 1:1000
    ouh = M2_t(x_init ,exogenous_matrix[1, :], U_train[1, :], simplified_params)
end

# ModelingToolkit => 1.828129



print(results.time/1000)



@timed M2_t(x_init ,exogenous_matrix[1, :], U_train[1, :], simplified_params)

@timed M_t(x_init ,exogenous_matrix[1, :], U_train[1, :], params_ter)

p = simplified_params

K_DCO = p[1][1] ; K_OH = p[1][2] ; K_NO = p[1][3] ; η_g = p[1][4] ; η_h = p[1][5] ; K_ND = p[1][6]; K_NH = p[1][7] ; K_OA =  p[1][8]
#Additional parameters
θ_1 = p[2][1] ; θ_2 = p[2][2] ; θ_3 = p[2][3] ; θ_4 = p[2][4] ; θ_5 = p[2][5]

Y_A = p[3][1]; Y_H = p[3][2] ; i_XB = p[3][3]

volume = p[4] ; X_in = p[5] ; Q_in = p[6](20.0) ; SO_sat = p[7] ; KLa = p[8]

simplified_final.K_DCO = K_DCO
simplified_final.K_OH = K_OH
simplified_final.K_NO = K_NO
simplified_final.η_g = η_g
simplified_final.η_h = η_h
simplified_final.K_ND = K_ND
simplified_final.K_NH = K_NH
simplified_final.K_OA = K_OA

simplified_final.θ_1 = θ_1
simplified_final.θ_2 = θ_2
simplified_final.θ_3 = θ_3
simplified_final.θ_4 = θ_4
simplified_final.θ_5 = θ_5

simplified_final.Y_H = Y_H
simplified_final.Y_A = Y_A
simplified_final.i_XB = i_XB

simplified_final.V = volume
simplified_final.X_DCO_in = X_in[1]
simplified_final.S_O_in = X_in[2]
simplified_final.S_NO_in = X_in[3]
simplified_final.S_NH_in = X_in[4]
simplified_final.S_ND_in = X_in[5]
simplified_final.Q_in = Q_in
simplified_final.SO_sat = SO_sat
simplified_final.KLa = KLa

x= x_init
init_conditions = [X_DCO => x[1], S_O => x[2], S_NO => x[3], S_NH => x[4], S_ND => x[5], U => u]
ode_problem = ODEProblem(simplified_final, init_conditions,  (exogenous[7], exogenous[7] + exogenous[8]), jac = true, sparse = true)

new_model = modelingtoolkitize(ode_problem)

ode_problem_new = ODEProblem(new_model, init_conditions,  (exogenous[7], exogenous[7] + exogenous[8]), jac = true, sparse = true)

n_particules = size(x, 2)
function prob_func(prob, i, repeat)
    remake(ode_problem, u0 = vcat(x[:, i], u))
end
monte_prob = EnsembleProblem(ode_problem, prob_func = prob_func)


function M2_t(x, exogenous, u, p)

    # K_DCO = p[1][1] ; K_OH = p[1][2] ; K_NO = p[1][3] ; η_g = p[1][4] ; η_h = p[1][5] ; K_ND = p[1][6]; K_NH = p[1][7] ; K_OA =  p[1][8]
    # #Additional parameters
    # θ_1 = p[2][1] ; θ_2 = p[2][2] ; θ_3 = p[2][3] ; θ_4 = p[2][4] ; θ_5 = p[2][5]

    # Y_A = p[3][1]; Y_H = p[3][2] ; i_XB = p[3][3]

    # volume = p[4] ; X_in = p[5] ; Q_in = p[6](20.0) ; SO_sat = p[7] ; KLa = p[8]

    # simplified_final.K_DCO = K_DCO
    # simplified_final.K_OH = K_OH
    # simplified_final.K_NO = K_NO
    # simplified_final.η_g = η_g
    # simplified_final.η_h = η_h
    # simplified_final.K_ND = K_ND
    # simplified_final.K_NH = K_NH
    # simplified_final.K_OA = K_OA

    # simplified_final.θ_1 = θ_1
    # simplified_final.θ_2 = θ_2
    # simplified_final.θ_3 = θ_3
    # simplified_final.θ_4 = θ_4
    # simplified_final.θ_5 = θ_5

    # simplified_final.Y_H = Y_H
    # simplified_final.Y_A = Y_A
    # simplified_final.i_XB = i_XB

    # simplified_final.V = volume
    # simplified_final.X_DCO_in = X_in[1]
    # simplified_final.S_O_in = X_in[2]
    # simplified_final.S_NO_in = X_in[3]
    # simplified_final.S_NH_in = X_in[4]
    # simplified_final.S_ND_in = X_in[5]
    # simplified_final.Q_in = Q_in
    # simplified_final.SO_sat = SO_sat
    # simplified_final.KLa = KLa

    # n_particules = size(x, 2)


    sim_results = solve(ode_problem, Euler(), dt = 1/(6*24*60), saveat=[exogenous[7] + exogenous[8]], maxiters=10e3, sensealg = ForwardSensitivity());

    return 0
    # return hcat([max.(sim_results[i].u[1][1:5], 0.0) for i in 1:n_particules]...)

end

@timed M2_t(x_init ,exogenous_matrix[1, :], U_train[1, :], simplified_params)

M_t => 0.0006

M2_t => 0.005