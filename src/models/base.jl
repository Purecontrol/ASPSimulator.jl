
const available_symbol = [:asm1, :asm1_simplified]
const available_chemical_species = [:o2, :no3, :nh4]


ASPSimulator.default_sytem = :asm1
"""
Set the default system of the package ASPSimulator.jl.
"""
@inline function set_default_system(system::Symbol)

    if system ∉ available_symbol
        @error "No system defined with this symbol. The system has to be in $available_symbol."
    end

    ASPSimulator.default_sytem = system
end


"""
Define a function that set all the parameters of ODECore for the desired system.
"""
function set_ode_core(system::Symbol=ASPSimulator.default_sytem; kwargs...)

    if system ∉ available_symbol
        @error "No system defined with this symbol. The system has to be in $available_symbol."
    end

    if system == :asm1

        if !haskey(kwargs, :influent_file_path)
            kwargs = (; kwargs..., influent_file_path=dirname(dirname(pathof(ASPSimulator))) * "/data/external/influent_files/dryinfluent.ascii")
        end
        p, init_x  = get_default_parameters_asm1(; kwargs...)

        return [0.0, 1/1440, asm1!, init_x, p, [14]]

    elseif system == :asm1_simplified

        if !haskey(kwargs, :influent_file_path)
            kwargs = (; kwargs..., influent_file_path=dirname(dirname(pathof(ASPSimulator))) * "/data/external/influent_files/dryinfluent.ascii")
        end
        p, init_x = get_default_parameters_simplified_asm1(; kwargs...)

        return [0.0, 1/1440, simplified_asm1!, init_x, p, [6]]

    end
    
end


"""
Get the nh4 index into the ODECore for the specified system.
"""
function get_nh4_index(system::Symbol=ASPSimulator.default_sytem)

    if system ∉ available_symbol
        @error "No system defined with this symbol. The system has to be in $available_symbol."
    end

    if system == :asm1

        return 10

    elseif system == :asm1_simplified

        return 4

    end

end


"""
Get the o2 index into the ODECore for the specified system.
"""
function get_o2_index(system::Symbol=ASPSimulator.default_sytem)

    if system ∉ available_symbol
        @error "No system defined with this symbol.  The system has to be in $available_symbol."
    end

    if system == :asm1

        return 8

    elseif system == :asm1_simplified

        return 2

    end

end


"""
Get the no3 index into the ODECore for the specified system.
"""
function get_no3_index(system::Symbol=ASPSimulator.default_sytem)

    if system ∉ available_symbol
        @error "No system defined with this symbol.  The system has to be in $available_symbol."
    end

    if system == :asm1

        return 9

    elseif system == :asm1_simplified

        return 3

    end

end


"""
Get the control index into the ODECore for the specified system.
"""
function get_control_index(system::Symbol=ASPSimulator.default_sytem)

    if system ∉ available_symbol
        @error "No system defined with this symbol.  The system has to be in $available_symbol."
    end

    if system == :asm1

        return 14

    elseif system == :asm1_simplified

        return 6

    end

end


"""
Get the number of variables of the ODECore for the specified system.
"""
function get_number_variables(system::Symbol=ASPSimulator.default_sytem)

    if system ∉ available_symbol
        @error "No system defined with this symbol.  The system has to be in $available_symbol."
    end

    if system == :asm1

        return 14

    elseif system == :asm1_simplified

        return 6

    end

end


"""
Get the corresponding index into the ODECore for the specified system and specified chemical species.
"""
function get_indexes_from_symbols(species::Union{Array{Symbol}, Symbol}, system::Symbol=ASPSimulator.default_sytem)

    if system ∉ available_symbol
        @error "No system defined with this symbol.  The system has to be in $available_symbol."
    end

    species = vcat([species]...)
    if all([i ∈ available_chemical_species for i in species])

        # Get all indexes
        list_indexes = []
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
        @error "One of the species is not contained in $available_chemical_species"
    end

end


"""
Index arrays by symbols of chemical species
"""
@inline function Base.getindex(arr::Array, row_ind::Symbol, col_ind::Union{Signed, Unsigned, Colon})
    return arr[ASPSimulator.get_indexes_from_symbols(row_ind), col_ind]
end