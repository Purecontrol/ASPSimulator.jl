nb_exp = 10
X_in_table = [8.8924]

for x_in_ite in X_in_table
    for num_exp in 1:nb_exp
        run(`julia --project=JuliaEnvironment asm1-simulator/scripts/ECC24/test.jl $num_exp $x_in_ite`)
    end
end