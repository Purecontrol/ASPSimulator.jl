using ASM1Simulator, DifferentialEquations, LinearAlgebra, Statistics, Optimization, OptimizationOptimisers, OptimizationOptimJL, Optim, Plots, JLD

# Generate data
include("generate_data.jl")

choosen_integrator = AutoTsit5(Rosenbrock23())
max_iter_integrator = 10e4

######################################################
#### Link observation model with simplified model ####
######################################################

index_obs_simplified_var = [4]
nb_obs_var = size(index_obs_simplified_var, 1)
link_index_real_system = [[2, 4], [8], [9], [10], [11], [14]] 

H_simplified = zeros(6, 6)
H_simplified[index_obs_simplified_var, index_obs_simplified_var] = Matrix(I, nb_obs_var, nb_obs_var)
H_simplified = H_simplified[index_obs_simplified_var, :]

#####################################################
#### Define the structure of the white box model ####
#####################################################

# Get init_vector and p_vector for the simplified model
# TODO : it can be useful, if we can collect it directly from an instance of ODECore
influent_file_path = joinpath(dirname(pathof(ASM1Simulator)), "../..", "data", "external/influent_files/dryinfluent.ascii")
simplified_real_p, ~ = ASM1Simulator.Models.get_default_parameters_simplified_asm1(influent_file_path=influent_file_path)
simplified_real_p_vec, X_init_simplified = ASM1Simulator.Models.get_default_parameters_simplified_asm1(get_R=false, influent_file_path=influent_file_path)

theorical_opt_params = vcat([simplified_real_p_vec[1], simplified_real_p_vec[2], simplified_real_p_vec[3], simplified_real_p_vec[4], simplified_real_p_vec[7], simplified_real_p_vec[8], X_init_simplified[filter!(e-> e ∉ index_obs_simplified_var ,[1, 2, 3, 4, 5])]]...)

# Set the real value of X_init for observed variables
known_X_init = [sum(x_train[link_index_real_system[i], 1]) for i in index_obs_simplified_var]

# Define the new ODE problem associated with the simplified model
tsave = LinRange(T_steady_state, T_steady_state + T_training, Int(T_training*(1440/dt_obs)))
t_control = LinRange(T_steady_state, T_steady_state + T_training, Int(T_training*(1440)))

simplified_ode_prob = ODEProblem(ASM1Simulator.Models.simplified_asm1!, [0.0 for i in 1:6], (T_steady_state, T_steady_state + T_training), simplified_real_p)

function model(opt_params)

    # Copy real vector of parameters
    p_sim = deepcopy(simplified_real_p_vec)
    
    # Copy fixed parameters and update the parameters to optimize
    p_sim[1] = opt_params[1:8]
    p_sim[2] = opt_params[9:13]
    p_sim[3] = ASM1Simulator.Models.get_stoichiometric_matrix_simplified_asm1(opt_params[14:16])
    p_sim[4] = opt_params[17]
    p_sim[7] = opt_params[18]
    p_sim[8] = opt_params[19]

    # Copy fixed initial conditions and update the initial conditions to optimize
    X_init_sim = [0.0 for i in 1:6]
    id_obs_var = 1
    for i in 1:5
        if i in index_obs_simplified_var
            X_init_sim[i] = known_X_init[id_obs_var]
            id_obs_var = id_obs_var + 1
        else
            X_init_sim[i] = opt_params[19 + i - (id_obs_var - 1)]
        end
    end

    # Fixed known control
    X_init_sim[6] = x_train[14, 1]

    function isoutofdomain(u,p,t)
        return any(u[1:5] .< 0)
    end

    # Simulate the model
    x̂ = zeros(6, size(tsave, 1))*NaN
    try
        sol_sim = solve(remake(simplified_ode_prob, p=p_sim, u0=X_init_sim), choosen_integrator, saveat = tsave, callback=ASM1Simulator.Models.external_control(t_control, vcat(U_train...); index_u = 6), verbose=false, maxiters=Int(max_iter_integrator), progress=true, isoutofdomain=isoutofdomain)
        x_sol = hcat(sol_sim.u...)
        if size(x_sol, 2) == size(x̂, 2)
            x̂ = x_sol
        else
            println("Error in ODE solver")
        end

    catch e
        println(e)
    end

    return H_simplified*x̂

end

####################################################################################################################
############################################## OPTIMIZE THE PARAMETERS #############################################
####################################################################################################################

function L2_loss(opt_params, pred, real)

    # Compute the loss
    l2_loss = sqrt.(mean((pred .- real) .^ 2))

    if isnan(l2_loss)
        l2_loss = 10e5 + 10*mean((opt_params - theorical_opt_params).^2)
    end

    return l2_loss

end

# Define the bounds on the variables
nb_opt_params = 19 + (5 - nb_obs_var)
# lb_optim = zeros(nb_opt_params) .+ 0.0001
# ub_optim = [5000.0 for i in 1:nb_opt_params]

p_lower, p_upper, X_init_lower, X_init_upper = ASM1Simulator.Models.get_bounds_parameters_simplified_asm1()
bounds = vcat([[p_lower[1][i], p_upper[1][i]] for i in 1:size(simplified_real_p_vec[1], 1)], [[p_lower[2][i], p_upper[2][i]] for i in 1:size(simplified_real_p_vec[2], 1)], [[p_lower[3][i], p_upper[3][i]] for i in 1:size(simplified_real_p_vec[3], 1)], [[p_lower[4], p_upper[4]]], [[p_lower[7], p_upper[7]]], [[p_lower[8], p_upper[8]]], [[X_init_lower[i], X_init_upper[i]] for i in 1:5 if i ∉ index_obs_simplified_var])
lb = [i[1] for i in bounds] ; ub = [i[2] for i in bounds]
p_init = lb + rand(nb_opt_params).*(ub-lb)

# Define the optimizer and the options
nb_iter = 1000
nb_particules = 300
optimizer = Optim.ParticleSwarm(lower=lb, lower_ini=lb, upper=ub, upper_ini=ub, n_particles=nb_particules)
options = Optim.Options(iterations=nb_iter, extended_trace=true, store_trace=true, show_trace=true, show_every=2)

y_opt_train = reshape(y_train[.! isnan.(y_train)], (nb_obs_var, Int(T_training*1440/dt_obs)))
res = optimize(u -> L2_loss(u, model(u), y_opt_train), p_init, optimizer, options)

# plot(1:dt_obs:Int(T_training*1440), model(res.minimizer)', label="Solution") # model(res.minimizer)
# plot!((H*x_train)', label="True state")
# scatter!(y_train', label="Observed state")
# plot!(xformatter = x -> Dates.format(DateTime(2023, 1, 1) + Dates.Minute(x), "Jd - Hh"))


opt_params = [1.0, 0.1, 0.25, 0.9599962529003409, 0.64, 5.3354001336455275, 1.5, 0.6, 5000.0, 50.0, 10.0, 10.002464626798378, 1000.0, 0.28, 0.6594666694156143, 0.084, 3000.0, 4.36, 100.00082216236575, 0.0, 0.0, 100.0, 1.1448518693606955]

plot(1:dt_obs:Int(T_training*1440), model(p_init)', label="Solution") # model(res.minimizer)
plot!((H*x_train)', label="True state")
scatter!(y_train', label="Observed state")
plot!(xformatter = x -> Dates.format(DateTime(2023, 1, 1) + Dates.Minute(x), "Jd - Hh"))


# Save the results
try
    cd("asm1-simulator/data/processed/whitebox/")
catch
    mkpath("asm1-simulator/data/processed/whitebox")
    cd("asm1-simulator/data/processed/whitebox/")
end

@save "experiment_1.jld" res x_train y_train

# nb_test = 100

# p_test_tab = [lb_optim + rand(nb_opt_params).*(ub-lb) for i in 1:nb_test]

# choosen_integrator = Rosenbrock23() ## #Tsit5()

# # Cheat => OwrenZen3()

# # Euler() => why not but i have to choose the dt, non-stable*

# @showprogress 1 "Estimating value particles" for i in 1:5

# end






# time_tab = []
# for i in 1:nb_test
#     println("#### Test ", i, " ####")

#     p_test = p_test_tab[i]

#     t_start = time()
#     res = model(p_test)
#     t_end = time()

#     println("Time : ", (t_end - t_start), " s")

#     push!(time_tab, (t_end - t_start))
# end

# mean(time_tab)










    
# plot(1:dt_obs:Int(T_training*1440),(H_simplified*model(opt_params))')
# plot!((H*x_train)')