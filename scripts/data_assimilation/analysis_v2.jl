using JLD, ASM1Simulator, Plots, DataFrames, Colors, PlotlyJS, StatsPlots

SNR_model = [10000.0, 1000.0, 100.0, 10.0, 1.0]
SNR_obs = [10000.0, 1000.0, 100.0, 10.0, 1.0]

# Include utils.jl file 
include("smc/utils.jl")

folder = "PF-FP2"

SNR_array = []
x̂_array = Array{TimeSeries, 1}()
xt_array = Array{TimeSeries, 1}()
x̂t_array = Array{TimeSeries, 1}()
ŷt_array = Array{TimeSeries, 1}()
part_array = Array{Array{Array{Float64, 2}, 1}, 1}()

# Load results
for i in eachindex(SNR_model)
    for j in eachindex(SNR_obs)

        try
            data = load("/home/victor/Documents/code/asm1-simulator/data/processed/assimilation/$folder/SNR_model_$(SNR_model[i])_SNR_obs_$(SNR_obs[j]).jld") 
            x̂, xt, x̂t, ŷt, part = data["x̂"], data["xt"], data["x̂t"], data["ŷt"], data["part"]

            push!(SNR_array, (SNR_model[i], SNR_obs[j]))
            push!(x̂_array, x̂)
            push!(xt_array, xt)
            push!(x̂t_array, x̂t)
            push!(ŷt_array, ŷt)
            push!(part_array, part)
        catch
            println("SNR_model_$(SNR_model[i])_SNR_obs_$(SNR_obs[j]).jld not found")
        end

    end
end

# COmpute the RMSE and the Variance for each variable
RMSE = Array{Float64, 2}(undef, length(SNR_array), 14)
VAR = Array{Float64, 3}(undef, length(SNR_array), x̂_array[1].nt,14)
for i in eachindex(SNR_array)
    
    rmse = sqrt.(mean((vcat(x̂_array[i].u'...) .- vcat(xt_array[i].u'...)) .^ 2, dims=1))

    variance = hcat(reshape.(var.(part_array[i], dims=2), (14))...)'

    RMSE[i, :] = rmse

    VAR[i, :, :] = variance

end

SNR_model_array = [i[1] for i in SNR_array]
SNR_obs_array = [i[2] for i in SNR_array]
df = DataFrame(
    SNR_model = log10.(SNR_model_array),
    SNR_obs = log10.(SNR_obs_array),
    RMSE_o2 = RMSE[:, 8],
    RMSE_no3 = RMSE[:, 9],
    RMSE_nh4 = RMSE[:, 10],
)

groupedbar(df.SNR_obs, df.RMSE_o2, group = df.SNR_model, bar_position = :dodge, legendtitle = "SNR model", bar_width=0.7, xlabel = "SNR observation", ylabel = "RMSE", title = "RMSE reconstruction of O2", lw = 0)
groupedbar(df.SNR_model, df.RMSE_o2, group = df.SNR_obs, bar_position = :dodge, legendtitle = "SNR obs", bar_width=0.7, xlabel = "SNR modèle", ylabel = "RMSE", title = "RMSE reconstruction of O2", lw = 0)


groupedbar(df.SNR_obs, df.RMSE_no3, group = df.SNR_model, bar_position = :dodge, legendtitle = "SNR model", bar_width=0.7, xlabel = "SNR observation", ylabel = "RMSE", title = "RMSE reconstruction of NO3", lw = 0)
groupedbar(df.SNR_model, df.RMSE_no3, group = df.SNR_obs, bar_position = :dodge, legendtitle = "SNR obs", bar_width=0.7, xlabel = "SNR modèle", ylabel = "RMSE", title = "RMSE reconstruction of NO3", lw = 0)

groupedbar(df.SNR_obs, df.RMSE_nh4, group = df.SNR_model, bar_position = :dodge, legendtitle = "SNR model", bar_width=0.7, xlabel = "SNR observation", ylabel = "RMSE", title = "RMSE reconstruction of NH4", lw = 0)
groupedbar(df.SNR_model, df.RMSE_nh4, group = df.SNR_obs, bar_position = :dodge, legendtitle = "SNR obs", bar_width=0.7, xlabel = "SNR modèle", ylabel = "RMSE", title = "RMSE reconstruction of NH4", lw = 0)


i=15
variable = 2
# Plot results
Plots.plot(xt_array[i].t, x̂_array[i][variable] - 1.96*sqrt.(VAR[i ,:, variable]), fillrange = x̂_array[i][variable] + 1.96*sqrt.(VAR[i ,:, variable]), fillalpha = 0.35, c = 1, label = "Confidence band", lw=0)
plot!(xt_array[i].t, x̂t_array[i][variable], label="Model")
plot!(xt_array[i].t, x̂_array[i][variable], label="Reconstructed")
plot!(xt_array[i].t, xt_array[i][variable], label="True concentration")
scatter!(ŷt_array[i].t, ŷt_array[i][variable]; markersize = 2, label = "Observed concentration")

