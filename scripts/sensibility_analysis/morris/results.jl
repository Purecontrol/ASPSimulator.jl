using JLD, Plots, LaTeXStrings

include("metrics.jl")
include("tools.jl")

# Load data
eof_data = load("/home/victor/Documents/code/asm1-simulator/julia/calibration/Morris/data/eof_morris_results2.jld")
ft_data  = load("/home/victor/Documents/code/asm1-simulator/julia/calibration/Morris/data/ft_morris_results2.jld")

# Extract results
eof_ind = eof_data["eof_i_morris"]
ft_ind  = ft_data["ft_i_morris"]

# Plot results
X_names = latexstring.(["S_I", "S_S", "X_I", "X_S", "X_{B,H}", "X_{B,A}", "X_P", "S_O", "S_{NO}", "S_{NH}", "S_{ND}", "X_{ND}", "S_{ALK}", "U"])
p_names = latexstring.(["μ_H", "K_S", "K_{OH}", "K_{NO}", "b_H", "η_g", "η_h", "k_h", "K_X", "μ_A", "K_{NH}", "b_A", "K_{OA}", "k_a", "Y_A", "Y_H", "f_P", "i_{XB}", "i_{XP}"])

heatmap(p_names, X_names, log.(eof_ind[1].means_star), yticks=(0.5:13.5, X_names), xticks=(0.5:19.5, p_names))
heatmap(p_names, X_names, (eof_ind[1].means .- a)./(b-a).*2 .-1 , yticks=(0.5:13.5, X_names), xticks=(0.5:19.5, p_names), c=:BuRd, clim=(-1,1))

function show_results(data)

    n = size(data, 1)

    data_plot = data
    data_plot = log10.(abs.(data))
    data_plot[data_plot .< -2] .= -2

    heatmap(p_names, X_names[1:n], data_plot, yticks=(0.5:(n - 0.5), X_names[1:n]), xticks=(0.5:19.5, p_names), c=:bluesreds)

end

function show_results2(data)

    n = size(data, 1)

    data_plot = (data .- map(minimum, eachslice(data, dims=1)))./(map(maximum, eachslice(data, dims=1)) .- map(minimum, eachslice(data, dims=1)))

    heatmap(p_names, X_names[1:n], data_plot, yticks=(0.5:(n - 0.5), X_names[1:n]), xticks=(0.5:19.5, p_names), c=:bluesreds)

end

# i = 3
# show_results(eof_ind[i].means)
# savefig("/home/victor/Documents/code/asm1-simulator/julia/calibration/Morris/data/pdf/eof_mean_$i.pdf")
# show_results(eof_ind[i].means_star)
# savefig("/home/victor/Documents/code/asm1-simulator/julia/calibration/Morris/data/pdf/eof_mean_star_$i.pdf")
# show_results(log.(eof_ind[i].variances))
# savefig("/home/victor/Documents/code/asm1-simulator/julia/calibration/Morris/data/pdf/eof_mean_variance_$i.pdf")

i = 3
show_results2(ft_ind[i].means)
savefig("/home/victor/Documents/code/asm1-simulator/julia/calibration/Morris/data/pdf/ft_mean_$i.pdf")
show_results2(ft_ind[i].means_star)
savefig("/home/victor/Documents/code/asm1-simulator/julia/calibration/Morris/data/pdf/ft_mean_star_$i.pdf")
show_results2(log.(ft_ind[i].variances))
savefig("/home/victor/Documents/code/asm1-simulator/julia/calibration/Morris/data/pdf/ft_mean_variance_$i.pdf")

plot(eof_ind[1].means_star[8, :], log.(eof_ind[1].variances[8, :]), seriestype=:scatter)

