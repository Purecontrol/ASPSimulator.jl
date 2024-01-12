using JLD
using JLD2

dt_model = 5
nb_exp = 10
X_in_table = [6.8924, 7.8924, 8.8924, 9.8924]

data_table = []
for x_in_ite in X_in_table
    for num_exp in 1:nb_exp
        filename = "asm1-simulator/data/result_global/"*"$x_in_ite"*"/"*"$num_exp"*".jld"
        push!(data_table, load(filename))
    end
end