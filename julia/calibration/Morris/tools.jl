using GlobalSensitivity, Statistics, Plots, QuasiMonteCarlo, JLD, Random

######################################################
### Function from the library GlobalSensitivity.jl ###
######################################################

struct Morris
    p_steps::Array{Int, 1}
    relative_scale::Bool
    num_trajectory::Int
    total_num_trajectory::Int
    len_design_mat::Int
end

function Morris(; p_steps::Array{Int, 1} = Int[], relative_scale::Bool = false,
                num_trajectory::Int = 10,
                total_num_trajectory::Int = 5 * num_trajectory, len_design_mat::Int = 10)
    Morris(p_steps, relative_scale, num_trajectory, total_num_trajectory, len_design_mat)
end

struct MatSpread{T1, T2}
    mat::T1
    spread::T2
end

struct MorrisResult{T1, T2}
    means::T1
    means_star::T1
    variances::T1
    elementary_effects::T2
end

function generate_design_matrix(p_range, p_steps, rng; len_design_mat = 10)
    ps = [range(p_range[i][1], stop = p_range[i][2], length = p_steps[i])
          for i in 1:length(p_range)]
    indices = [rand(rng, 1:i) for i in p_steps]
    all_idxs = Vector{typeof(indices)}(undef, len_design_mat)

    for i in 1:len_design_mat
        j = rand(rng, 1:length(p_range))
        indices[j] += (rand(rng) < 0.5 ? -1 : 1)
        if indices[j] > p_steps[j]
            indices[j] -= 2
        elseif indices[j] < 1.0
            indices[j] += 2
        end
        all_idxs[i] = copy(indices)
    end

    B = Array{Array{Float64}}(undef, len_design_mat)
    for j in 1:len_design_mat
        cur_p = [ps[u][(all_idxs[j][u])] for u in 1:length(p_range)]
        B[j] = cur_p
    end
    reduce(hcat, B)
end

function calculate_spread(matrix)
    spread = 0.0
    for i in 2:size(matrix, 2)
        spread += sqrt(sum(abs2.(matrix[:, i] - matrix[:, i - 1])))
    end
    spread
end

function sample_matrices(p_range, p_steps, rng; num_trajectory = 10,
                         total_num_trajectory = 5 * num_trajectory, len_design_mat = 10)
    matrix_array = []
    if total_num_trajectory < num_trajectory
        error("total_num_trajectory should be greater than num_trajectory preferably atleast 3-4 times higher")
    end
    for i in 1:total_num_trajectory
        mat = generate_design_matrix(p_range, p_steps, rng; len_design_mat = len_design_mat)
        spread = calculate_spread(mat)
        push!(matrix_array, MatSpread(mat, spread))
    end
    sort!(matrix_array, by = x -> x.spread, rev = true)
    matrices = [i.mat for i in matrix_array[1:num_trajectory]]
    reduce(hcat, matrices)
end

###############################################
### Function to make my own Morris analysis ###
###############################################

function generate_samples_Morris(method::Morris, p_range::AbstractVector;
    rng::AbstractRNG = Random.default_rng(), kwargs...)

    @unpack p_steps, relative_scale, num_trajectory, total_num_trajectory, len_design_mat = method
    if !(length(p_steps) == length(p_range))
        for i in 1:(length(p_range) - length(p_steps))
        push!(p_steps, 100)
        end
    end

    design_matrices = sample_matrices(p_range, p_steps, rng;
                                num_trajectory = num_trajectory,
                                total_num_trajectory = total_num_trajectory,
                                len_design_mat = len_design_mat)

    return design_matrices

end


function analysis_Morris(f, method::Morris, design_matrices, asm1_trajectories)

    @unpack p_steps, relative_scale, num_trajectory, total_num_trajectory, len_design_mat = method

    multioutput = false
    all_y = f(asm1_trajectories)
    multioutput = all_y isa AbstractMatrix

    effects = []
    for i in 1:num_trajectory
        y1 = multioutput ? all_y[:, (i - 1) * len_design_mat + 1] :
             all_y[(i - 1) * len_design_mat + 1]
        for j in ((i - 1) * len_design_mat + 1):((i * len_design_mat) - 1)
            y2 = y1
            del = design_matrices[:, j + 1] - design_matrices[:, j]
            change_index = 0
            for k in 1:length(del)
                if abs(del[k]) > 0
                    change_index = k
                    break
                end
            end
            del = sum(del)
            y1 = multioutput ? all_y[:, j + 1] : all_y[j + 1]
            if relative_scale == false
                effect = @. (y1 - y2) / (del)
                elem_effect = typeof(y1) <: Number ? effect : mean(effect, dims = 2)
            else
                if del > 0
                    effect = @. (y1 - y2) / (y2 * del)
                    elem_effect = typeof(y1) <: Number ? effect : mean(effect, dims = 2)
                else
                    effect = @. (y1 - y2) / (y1 * del)
                    elem_effect = typeof(y1) <: Number ? effect : mean(effect, dims = 2)
                end
            end
            if length(effects) >= change_index && change_index > 0
                push!(effects[change_index], elem_effect)
            elseif change_index > 0
                while (length(effects) < change_index - 1)
                    push!(effects, typeof(elem_effect)[])
                end
                push!(effects, [elem_effect])
            end
        end
    end
    means = eltype(effects[1])[]
    means_star = eltype(effects[1])[]
    variances = eltype(effects[1])[]
    for k in effects
        if !isempty(k)
            push!(means, mean(k))
            push!(means_star, mean(x -> abs.(x), k))
            push!(variances, var(k))
        else
            push!(means, zero(effects[1][1]))
            push!(means_star, zero(effects[1][1]))
            push!(variances, zero(effects[1][1]))
        end
    end

    MorrisResult(reduce(hcat, means), reduce(hcat, means_star), reduce(hcat, variances),
                 effects)

end


