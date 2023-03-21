using Plots, JLD, ASM1Simulator, LaTeXStrings

# Load data
data  = load("asm1-simulator/data/processed/morris/ft_morris_results.jld")
eof_i_morris = data["ft_i_morris"]

# Name of the parameters
kinetic_names = latexstring.(["μ_H", "K_S", "K_{OH}", "K_{NO}", "b_H", "η_g", "η_h", "k_h", "K_X", "μ_A", "K_{NH}", "b_A", "K_{OA}", "k_a"])
stochiometric_names = latexstring.(["Y_A", "Y_H", "f_P", "i_{XB}", "i_{XP}"])
oxygen_names = latexstring.(["S_O^{sat}", "K_La"])

# Plot the sensitivity indices
index_species = 10
pyplot()
global_plot = Array{Any}(undef, 4)
for i in 1:4
    global_plot[i] = plot(eof_i_morris[i].means_star[index_species, 1:14], eof_i_morris[i].variances[index_species, 1:14], title="Coefficient $i", seriestype=:scatter, xlabel=(i>2) ? "Mean*" : "", ylabel= (i%2 == 1) ? "Variance" : "", marker = (0.8, :cross, 10), label="Paramètre cinétique", legend=i ==1 ? :topleft : false)
    annotate!(eof_i_morris[i].means_star[index_species, 1:14], eof_i_morris[i].variances[index_species, 1:14].-13, kinetic_names, :top)
    plot!(eof_i_morris[i].means_star[index_species, 15:19], eof_i_morris[i].variances[index_species, 15:19], seriestype=:scatter, xlabel=(i>2) ? "Mean*" : "", ylabel=(i%2 == 1) ? "Variance" : "", marker = (0.8, :hex, 10), label="Paramètre stochiométrique")
    annotate!(eof_i_morris[i].means_star[index_species, 15:19], eof_i_morris[i].variances[index_species, 15:19].+-13, stochiometric_names, :top)
    plot!(eof_i_morris[i].means_star[index_species, 20:21], eof_i_morris[i].variances[index_species, 20:21], seriestype=:scatter, xlabel=(i>2) ? "Mean*" : "", ylabel=(i%2 == 1) ? "Variance" : "", marker = (0.8, :star7, 10), label="Paramètre oxygène")
    annotate!(eof_i_morris[i].means_star[index_species, 20:21], eof_i_morris[i].variances[index_species, 20:21].-13, oxygen_names, :top)
end
plot(global_plot..., size=(1200, 800), layout=(2, 2))
savefig("asm1-simulator/image/ft_morris_nh4.pdf")
