include("utils.jl")
include("time_series.jl")

function EnKF(da::DataAssimilation, yo::TimeSeries, np::Int64; progress = true, control = nothing)

    # dimensions
    nt = yo.nt        # number of observations
    nv = yo.nv        # number of variables (dimensions of problem)

    # initialization
    x̂ = TimeSeries(nt, nv)

    xa = [zeros(Float64, (nv, np)) for i = 1:nt]
    pf = [zeros(Float64, (nv, nv)) for i = 1:nt]
    xf = [zeros(Float64, (nv, np)) for i = 1:nt]
    ef = similar(xf[1])

    if progress
        p = Progress(nt)
    end

    for k = 1:nt

        if progress
            next!(p)
        end

        # update step (compute forecasts)            
        if k == 1
            xf[k] .= max.(da.xb .+ da.B.^(1/2)*rand(Normal(), nv, np), 0)
        else
            xf[k] = max.(da.m(xa[k-1], yo.t[k-1]; control = control) + da.m.Σ_model.^(1/2).*rand(Normal(), nv, np), 10e-9)
        end

        ef .= xf[k] * (Matrix(I, np, np) .- 1 / np)
        pf[k] .= (ef * ef') ./ (np - 1)

        # analysis step (correct forecasts with observations)          
        ivar_obs = findall(.!isnan.(yo.u[k]))
        n = length(ivar_obs)

        if n > 0
            μ = zeros(Float64, n)
            σ = da.R[ivar_obs, ivar_obs]
            yf = max.(da.H[ivar_obs, :] * xf[k] .+ σ.^(1/2)*rand(Normal(), n, np) .+ μ, 10e-9)
            Σ = (da.H[ivar_obs, :] * pf[k]) * da.H[ivar_obs, :]'
            Σ .+= da.R[ivar_obs, ivar_obs]
            invΣ = inv(Σ)
            K = (pf[k] * da.H[ivar_obs, :]') * invΣ
            d = yo.u[k][ivar_obs] .- yf
            xa[k] .= xf[k] .+ K * d
            # compute likelihood
            # innov_ll = mean(yo.u[k][ivar_obs] .- yf, dims = 2)
        else
            xa[k] .= xf[k]
        end

        x̂.u[k] .= vec(sum(xa[k] ./ np, dims = 2))

    end

    x̂, xa

end


function PF(da::DataAssimilation, yo::TimeSeries, np::Int64; progress = true, control = nothing)

    # dimensions
    nt = yo.nt        # number of observations
    nv = yo.nv        # number of variables (dimensions of problem)

    # initialization
    x̂ = TimeSeries(nt, nv)

    # special case for k=1
    k = 1
    m_xa_traj = Array{Float64,2}[]
    xf = max.(da.xb .+ da.B.^(1/2)*rand(Normal(), nv, np), 10e-9)
    ivar_obs = findall(.!isnan.(yo.u[k]))
    nobs = length(ivar_obs)
    weights = zeros(Float64, np)
    indic = zeros(Int64, np)
    part = [zeros(Float64, (nv, np)) for i = 1:nt]

    if nobs > 0

        for ip = 1:np
            μ = vec(da.H[ivar_obs, :] * xf[:, ip])
            σ = Matrix(da.R[ivar_obs, ivar_obs])
            d = MvNormal(μ, σ)
            weights[ip] = pdf(d, yo.u[k][ivar_obs])
        end
        # normalization
        weights ./= sum(weights)
        # resampling
        resample!(indic, weights)
        part[k] .= xf[:, indic]
        weights .= weights[indic] ./ sum(weights[indic])
        x̂.u[k] .= vec(sum(part[k] .* weights', dims = 2))

        # find number of iterations before new observation
        # todo: try the findnext function
        # findnext(.!isnan.(vcat(yo.u'...)), k+1)
        knext = 1
        while knext + k <= nt && all(isnan.(yo.u[k+knext]))
            knext += 1
        end

    else

        weights .= 1.0 / np # weights
        resample!(indic, weights) # resampling

    end

    kcount = 1

    if progress
        p = Progress(nt)
    end

    for k = 2:nt
        if progress
            next!(p)
        end

        # update step (compute forecasts) and add small Gaussian noise
        xf = max.(da.m(part[k-1], yo.t[k-1]; control = control) .+ da.m.Σ_model.^(1/2).*rand(Normal(), nv, np), 10e-9)
        if kcount <= length(m_xa_traj)
            m_xa_traj[kcount] .= xf
        else
            push!(m_xa_traj, xf)
        end
        kcount += 1

        # analysis step (correct forecasts with observations)
        ivar_obs = findall(.!isnan.(yo.u[k]))

        if length(ivar_obs) > 0
            # weights
            σ = Symmetric(da.R[ivar_obs, ivar_obs])
            for ip = 1:np
                μ = vec(da.H[ivar_obs, :] * xf[:, ip])
                d = MvNormal(μ, σ)
                weights[ip] = pdf(d, yo.u[k][ivar_obs])
            end
            # normalization
            weights ./= sum(weights)
            # resampling
            resample!(indic, weights)
            weights .= weights[indic] ./ sum(weights[indic])
            # stock results
            for j = 1:knext
                jm = k - knext + j
                for ip = 1:np
                    part[jm][:, ip] .= m_xa_traj[j][:, indic[ip]]
                end
                x̂.u[jm] .= vec(sum(part[jm] .* weights', dims = 2))
            end
            kcount = 1
            # find number of iterations  before new observation
            knext = 1
            while knext + k <= nt && all(isnan.(yo.u[k+knext]))
                knext += 1
            end
        else
            # stock results
            part[k] .= xf
            x̂.u[k] .= vec(sum(xf .* weights', dims = 2))
        end

    end

    x̂ , part

end