using ASM1Simulator, Plots, Random, JLD

# Define parameter physical model
influent_file_path = "/home/victor/Documents/code/asm1-simulator/data/external/influent_files/dryinfluent.ascii"
p_asm1, X_init = ASM1Simulator.Models.get_default_parameters_asm1(influent_file_path=influent_file_path)

# Define parameter data assimilation
dt_integration = 10/(24*60*60) # 10 secondes
dt_states = 10*6 # 10 minutes
dt_obs = 10*6 # 10 minutes
parameters = p_asm1
var_obs = [8, 9, 10] # O2, NO, NH
sigma2_model = vcat([0 for i in 1:13], [0.0])
sigma2_obs = 0.02

# Define StateSpaceModel structure
ssm = StateSpaceModel(
    ASM1Simulator.Models.asm1!,
    dt_integration,
    dt_states,
    dt_obs,
    parameters,
    var_obs;
    SNR_model = 100.0,
    SNR_obs = 100.0,
)

# Generate data from the SSM model
T = 1.0
xt , x̂t, ŷt , U = generate_data(ssm, X_init, T; control=ASM1Simulator.Models.redox_control());

plotly()
plot(xt.t, vcat(xt.u'...)[:,8], label = "x_t")
plot!(x̂t.t, vcat(x̂t.u'...)[:, 8], label = "x̂_t")
scatter!(ŷt.t, vcat(ŷt.u'...)[:, 8]; markersize = 2, label = "ŷ_t")

da = DataAssimilation(ssm, xt)

x̂, part = EnKF(da, ŷt, 30; progress = true, control = U)

x̂, part = PF(da, ŷt, 100; progress = true, control = U)


function get_coverage_probability(xt, part; prob = 0.95)

    # Get the confidence interval
    lwb, upb = get_confidence_interval(part, prob = prob)

    # Compute the coverage probability
    nt = size(part, 1)
    nv = size(part[1], 1)
    np = size(getindex(part, 1), 2)
    cp = zeros(Float64, nv)
    for i in 1:nv
        cp[i] = sum((lwb[i] .<= xt[i]) .& (xt[i] .<= upb[i])) / nt
    end

    return cp

end

function get_confidence_interval(particules; prob = 0.95)

    # Compute some useful variables
    nt = size(particules, 1)
    nv = size(particules[1], 1)
    np = size(getindex(particules, 1), 2)
    α = 1 - prob
    step = 1/np

    # Compute start and end index for the interval
    i_bottom = Int(α ÷ step)
    i_top = Int(np - i_bottom)

    # Compute the coefficient for the linear interpolation of the interval
    coeff = 1 - (α - step*i_bottom)/step
    
    # Compute the confidence interval
    lwb = TimeSeries(nt, nv)
    upb = TimeSeries(nt, nv)
    for i in 1:nt

        # Sort the particules according to the second dimension
        sort!(particules[i], dims=2)

        # Compute the confidence interval
        lwb.u[i] .= particules[i][:, i_bottom] + coeff*(particules[i][:, i_bottom + 1] - particules[i][:, i_bottom])
        upb.u[i] .= particules[i][:, i_top-1] + coeff*(particules[i][:, i_top] - particules[i][:, i_top-1])

    end

    return lwb, upb

end

function rmse(x̂, xt)

    return sqrt.(mean((vcat(x̂.u'...) .- vcat(xt.u'...)) .^ 2, dims=1))

end



lwb, upb = get_confidence_interval(part)

cp = get_coverage_probability(xt, part; prob = 0.95)

plot(xt.t, lwb[8], fillrange = upb[8], fillalpha = 0.35, c = 1, label = "Confidence band", lw=0)
# plot!(xt.t, x̂t[8], label="Model")
plot!(xt.t, x̂[8], label="Reconstructed")
plot!(xt.t, xt[8], label="True concentration")
scatter!(ŷt.t, ŷt[8]; markersize = 2, label = "Observed concentration")

plot(xt.t, lwb[9], fillrange = upb[9], fillalpha = 0.35, c = 1, label = "Confidence band", lw=0)
# plot!(xt.t, x̂t[9], label="Model")
plot!(xt.t, x̂[9], label="Reconstructed")
plot!(xt.t, xt[9], label="True concentration")
scatter!(ŷt.t, ŷt[9]; markersize = 2, label = "Observed concentration")

plot(xt.t, lwb[10], fillrange = upb[10], fillalpha = 0.35, c = 1, label = "Confidence band", lw=0)
# plot!(xt.t, x̂t[10], label="Model")
plot!(xt.t, x̂[10], label="Reconstructed")
plot!(xt.t, xt[10], label="True concentration")
scatter!(ŷt.t, ŷt[10]; markersize = 2, label = "Observed concentration")

plot(xt.t, lwb[11], fillrange = upb[11], fillalpha = 0.35, c = 1, label = "Confidence band", lw=0)
# plot!(xt.t, x̂t[4], label="Model")
plot!(xt.t, x̂[11], label="Reconstructed")
plot!(xt.t, xt[11], label="True concentration")
scatter!(ŷt.t, ŷt[11]; markersize = 2, label = "Observed concentration")

@save "/home/victor/Documents/code/asm1-simulator/data/processed/assimilation/t.jld" x̂ xt x̂t ŷt