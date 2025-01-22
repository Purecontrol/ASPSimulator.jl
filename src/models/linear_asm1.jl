using Statistics

"""
     $(TYPEDSIGNATURES)

Return the default parameters and an initial condition for the simplified linear ASM1 model.
"""
function get_default_parameters_linear_asm1(;
        influent_file_path = nothing, fixed_concentration = nothing,
        variable_inlet_concentration = true, variable_inflow = true)
    physical_parameters = (V = 850.7, β = 179)

    #Set inlet concentrations and inflow
    if influent_file_path ≠ nothing && variable_inlet_concentration
        X_in = get_inlet_concentrations_from_src_files(
            influent_file_path, output_index = [10])
    else
        X_in = [6.8924]
    end

    if influent_file_path ≠ nothing && variable_inflow
        Q_in = get_inflow_from_bsm_files(influent_file_path)
    else
        Q_in = 18061.0
    end
    exogenous_params = (Q_in = Q_in, X_in = X_in)

    # Merge parameters
    p = merge(physical_parameters, exogenous_params)

    # Set X_init
    X_init = [6.8924, 1.0]

    return (p, X_init)
end

"""
     $(TYPEDSIGNATURES)

Return the differential equations for the linear ASM1 model. See ``get_default_parameters_linear_asm1`` for details about p structure.
"""
function linear_asm1!(dX, X, p::NamedTuple, t)

    dX[1] = (evaluate(p[3], t) / p[1]) * (evaluate.(p[4], t)[1] - X[1]) - p[2] * X[2]
    dX[2] = 0.0

end

"""
     $(TYPEDSIGNATURES)

Return the differential equations for the linear ASM1 model. Here, it p is an ``Array`` which is useful for parameter optimization.
"""
function linear_asm1!(dX, X, p::Array, t)

    dX[1] = (evaluate(p[3], t) / p[1]) * (evaluate.(p[4], t) - X[1]) - p[2] * X[2]
    dX[2] = 0.0

end
