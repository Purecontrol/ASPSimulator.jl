using JLD, ASM1Simulator, Plots, DataFrames, PlotlyJS, Colors

SNR_model = [10000.0, 1000.0, 100.0, 10.0, 1.0]
SNR_obs = [10000.0, 1000.0, 100.0, 10.0, 1.0]

# Include utils.jl file 
include("smc/utils.jl")

RMSE = []
# Load results
for i in eachindex(SNR_model)
    for j in eachindex(SNR_obs)
        rmse=[]
        try
            data = load("/home/victor/Documents/code/asm1-simulator/data/processed/assimilation/EnKF-FP/SNR_model_$(SNR_model[i])_SNR_obs_$(SNR_obs[j]).jld") 
            x̂, xt, x̂t, ŷt = data["x̂"], data["xt"], data["x̂t"], data["ŷt"]

            rmse = sqrt.(mean((vcat(x̂.u'...) .- vcat(xt.u'...)) .^ 2, dims=1))
        catch
            rmse = [NaN for i in 1:14]
        end
        push!(RMSE, rmse)

    end
end

RMSE_mean = [mean(i) for i in RMSE]

df = DataFrame(
    SNR_model = log10.(repeat(SNR_model, inner=length(SNR_obs))),
    SNR_obs = log10.(repeat(SNR_obs, outer=length(SNR_model))),
    RMSE = RMSE_mean,
    color = repeat([:red, :blue, :green, :yellow, :black], inner=length(SNR_obs))
)

# Plot results
PlotlyJS.plot(PlotlyJS.scatter(
    df,
    x=:SNR_model,
    y=:SNR_obs,
    z=:RMSE,
    xlab = "true SNR model",
    camera=(30, 30),
    size=(800, 600),
    # background_color = :transparent,
    type="scatter3d", 
    mode="markers",
    marker=attr(
        size=4,
        color=:color,                # set color to an array/list of desired values   # choose a colorscale
        opacity=0.8
    ),
), 
Layout(margin=attr(l=0, r=0, b=0, t=0), scene = attr(
    xaxis_title="log SNR model",
    yaxis_title="log SNR obs",
    zaxis_title="RMSE")))

