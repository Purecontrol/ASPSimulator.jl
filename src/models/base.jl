# Constants defining the available systems and chemical species
const available_symbol = [:asm1, :asm1_simplified, :linear_asm1]
const available_chemical_species = [:o2, :no3, :nh4]

# Reference to store the default system
const _default_system = Ref(:asm1)

"""
Get the default system of the package ASPSimulator.jl.
"""
function get_default_system()
    return _default_system[]
end

"""
Check that the system is in `available_symbol`.

    $(TYPEDSIGNATURES)

# Errors
- Throws an error if the system is not in `available_symbol`.
"""
function _check_system(system::Symbol)
    if system ∉ available_symbol
        throw(ArgumentError("No system defined with this symbol. The system must be in $available_symbol."))
    end
end

"""
Set the default system of the package ASPSimulator.jl.

    $(TYPEDSIGNATURES)

# Errors
- Throws an error if the system is not in `available_symbol`.
"""
@inline function set_default_system(system::Symbol)
    _check_system(system)
    _default_system[] = system
end

"""
Returns all the argument to build an ODECore for the desired system with the default parameters.

$(SIGNATURES)

# Arguments
- `system::Symbol`: The system for which to set the ODECore. Defaults to the package's default system.
- `kwargs...`: Optional keyword arguments.

# Returns
- An array of argument to give to the constructor of ODECore specific to the selected system.

# Errors
- Throws an error if the system is not in `available_symbol`.
"""
function set_ode_core(system::Symbol = get_default_system(); kwargs...)
    _check_system(system)

    if system == :asm1
        if !haskey(kwargs, :influent_file_path)
            kwargs = (; kwargs...,
                influent_file_path = dirname(dirname(pathof(ASPSimulator))) *
                                     "/data/external/influent_files/dryinfluent.ascii")
        end
        p, init_x = get_default_parameters_asm1(; kwargs...)

        return [0.0, 1 / 1440, asm1!, init_x, p, [14]]

    elseif system == :asm1_simplified
        if !haskey(kwargs, :influent_file_path)
            kwargs = (; kwargs...,
                influent_file_path = dirname(dirname(pathof(ASPSimulator))) *
                                     "/data/external/influent_files/dryinfluent.ascii")
        end
        p, init_x = get_default_parameters_simplified_asm1(; kwargs...)

        return [0.0, 1 / 1440, simplified_asm1!, init_x, p, [6]]
    elseif system == :linear_asm1
        if !haskey(kwargs, :influent_file_path)
            kwargs = (; kwargs...,
                influent_file_path = dirname(dirname(pathof(ASPSimulator))) *
                                     "/data/external/influent_files/dryinfluent.ascii")
        end
        p, init_x = get_default_parameters_linear_asm1(; kwargs...)

        return [0.0, 10 / 1440, linear_asm1!, init_x, p, [2]]
    end
end

"""
    $(SIGNATURES)

Get the nh4 index into the ODECore for the specified system.
"""
function get_nh4_index(system::Symbol = get_default_system())
    _check_system(system)

    if system == :asm1
        return 10
    elseif system == :asm1_simplified
        return 4
    elseif system == :linear_asm1
        return 1
    end
end

"""
    $(SIGNATURES)

Get the o2 index into the ODECore for the specified system.
"""
function get_o2_index(system::Symbol = get_default_system())
    _check_system(system)

    if system == :asm1
        return 8

    elseif system == :asm1_simplified
        return 2
    elseif system == :linear_asm1
        return @error("No O2 index.")
    end
end

"""
    $(SIGNATURES)

Get the no3 index into the ODECore for the specified system.
"""
function get_no3_index(system::Symbol = get_default_system())
    _check_system(system)

    if system == :asm1
        return 9

    elseif system == :asm1_simplified
        return 3

    elseif system == :linear_asm1
        return @error("No NO3 index.")
    end
end

"""
    $(SIGNATURES)

Get the control index into the ODECore for the specified system.
"""
function get_control_index(system::Symbol = get_default_system())
    _check_system(system)

    if system == :asm1
        return 14

    elseif system == :asm1_simplified
        return 6
    elseif system == :linear_asm1
        return 2
    end
end

"""
    $(SIGNATURES)

Get the number of variables of the ODECore for the specified system.
"""
function get_number_variables(system::Symbol = get_default_system())
    _check_system(system)

    if system == :asm1
        return 14

    elseif system == :asm1_simplified
        return 6
    elseif system == :linear_asm1
        return 2
    end
end

"""
    $(SIGNATURES)

Get the corresponding index into the ODECore for the specified system and specified chemical species.
"""
function get_indexes_from_symbols(
        species::Union{Array{Symbol}, Symbol}, system::Symbol = get_default_system())
    _check_system(system)

    species = vcat([species]...)
    if all([i ∈ available_chemical_species for i in species])

        # Get all indexes
        list_indexes = Vector{Int}()
        for i in species
            if i == :o2
                push!(list_indexes, get_o2_index(system))
            elseif i == :no3
                push!(list_indexes, get_no3_index(system))
            elseif i == :nh4
                push!(list_indexes, get_nh4_index(system))
            end
        end

        return list_indexes
    else
        throw(ArgumentError("One of the species is not contained in $available_chemical_species"))
    end
end

"""
    $(SIGNATURES)

Index ``TSConcentrations`` by symbols of chemical species.
"""
@inline function Base.getindex(
        ts::TSConcentrations, row_ind::Union{Signed, Unsigned, Colon}, col_ind::Union{
            Symbol, Vector{Symbol}})
    return ts[row_ind, ASPSimulator.get_indexes_from_symbols(col_ind)]
end
