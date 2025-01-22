import Base: size, getindex, values

CustomTimeType = Union{Dates.TimeType, Real}

"""
A TimeSeries type to represent concentrations evolutions over time. 

$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct TSConcentrations{D, T} <: AbstractMatrix{T}
    "Timestamp of the values"
    timestamp::Vector{D}
    "Values of the concentrations"
    data::Matrix{T}

    function TSConcentrations(x::Matrix{T}) where {T <: Real}
        new{T, T}(zeros(T, size(x, 1)), x)
    end

    function TSConcentrations(
            t::Vector{D}, x::Matrix{T}) where {T <: Real, D <: CustomTimeType}
        if size(t, 1) != size(x, 1)
            throw(DimensionMismatch("The first dimension of the values has to be the same has the size of the vector of timestamp."))
        else
            new{D, T}(t, x)
        end
    end
end

# Overwrite required methods for subtype of AbstractMatrix{T}
@inline size(t::TSConcentrations, d::Int) = size(t.data, d)
@inline size(t::TSConcentrations) = size(t.data)
@inline getindex(t::TSConcentrations, i::Int) = t.data[i]
@inline getindex(t::TSConcentrations, I::Vararg{Int, 2}) = t.data[I...]

"""
    $(TYPEDSIGNATURES)

Get the time indexes of a `TSConcentrations`.
"""
timestamp(ts::TSConcentrations) = getfield(ts, :timestamp)

"""
    $(TYPEDSIGNATURES)

Get the underlying value table of a `TSConcentrations`.
"""
values(ts::TSConcentrations) = getfield(ts, :data)

"""
According to the input, get the number or evaluate the function.
"""
@inline evaluate(v::Function, t) = v(t)
@inline evaluate(v::Number, t) = v

"""
    $(TYPEDSIGNATURES)

Adapt the value of the parameter ρ according to the temperature T and the coefficient a following the Van 't Hoff equation.
"""
function T_var(T, ρ, a)
    return ρ * exp((log2(ρ / a) / 5) * (T - 15))
end

"""
    $(TYPEDSIGNATURES)

Return the inflow specified in a source file coming from the Benchmark Simulation model.
"""
function get_inflow_from_bsm_files(influent_file_path)

    # Read file
    inflow_generator = readdlm(influent_file_path)

    # Set up interpolation function with data provided in the file
    itp = interpolate((inflow_generator[:, 1],), inflow_generator[:, 10], Gridded(Linear()))

    # Define a function giving the inflow
    T_max = maximum(inflow_generator[:, 1])
    function Q_in(t)
        return itp(abs(t) % T_max)
    end

    return Q_in
end

"""
    $(TYPEDSIGNATURES)

Return the inlet concentrations specified in a source file coming from the Benchmark Simulation model.
"""
function get_inlet_concentrations_from_src_files(
        influent_file_path; output_index = collect(1:13))

    # Read file
    inflow_generator = readdlm(influent_file_path)

    # Set up interpolation function with data provided in the file
    # Real usage of the file but don't work whithout using a clarifier
    # list_order = [7, 2, 5, 4, 3, 0.0, 0.0, 0.0, 0.0, 6, 8, 9, 7.0]
    # constant_value = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
    # Our modified usage
    list_order = [28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699,
        964.8992, 0.0093, 3.9350, 6, 0.9580, 3.8453, 5.4213]
    constant_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    divide_by = [1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1]
    list_itp = [(constant_value[i] == 0) ?
                interpolate((inflow_generator[:, 1],),
                    inflow_generator[:, Int(list_order[i])] / divide_by[i],
                    Gridded(Linear())) :
                interpolate((inflow_generator[:, 1],),
                    list_order[i] .* ones(size(inflow_generator, 1)) / divide_by[i],
                    Gridded(Linear())) for i in 1:13]

    # Define a function giving the inlet concentration
    T_max = maximum(inflow_generator[:, 1])
    function X_in(t)
        return [sum([list_itp[j](abs(t) .% T_max) for j in vcat([i]...)])
                for i in output_index]
    end

    return X_in
end
