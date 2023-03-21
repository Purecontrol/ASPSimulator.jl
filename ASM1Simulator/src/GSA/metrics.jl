using EmpiricalOrthogonalFunctions, FFTW, LinearAlgebra, Statistics

###################################################
### Sensitivity Analysis Metrics for ASM1 model ###
###################################################

### Define the functions to compute the sensitivity indices ###

# Empirical Orthogonal Function
function EOF_metric(samples; keep_nb_coef = 1, percentage_variance = 0)

    # Compute the EOF for each species
    nb_samples, nb_species, _ = size(samples)
    results = Array{Array{Float64, 2}, 1}(undef, nb_species) 
    for i in 1:nb_species
        
        #Compute the EOF
        eof = EmpiricalOrthogonalFunction(samples[:,i,:]')

        # Determine the number of EOFs to keep
        if percentage_variance > 0
            variance = cumsum(variancefraction(eof))
            n = findfirst(x -> x > percentage_variance, variance)
        else
            n = keep_nb_coef
        end

        # Retrieve the PCs coefficients and the mean
        results[i] = hcat(eof.center', eofs(eof, n=n))
    end

    return results
end

# Fourier Transform
function FT_metric(samples; keep_nb_coef = 1, percentage_variance = 0)

    # Compute the FT for each species
    nb_samples, nb_species, T = size(samples)
    T_center = Int(size(samples, 3) / 2)
    results = Array{Array{Float64, 2}, 1}(undef, nb_species-1) 
    for i in 1:(nb_species-1)

        # Compute the FT
        println("############### Species $i ###############")
        F_init = fft(samples[:, i, :], 2)

        # Determine the number of coefficients to keep
        if percentage_variance > 0
            reconstruction_error = 1
            index = 0
            while reconstruction_error > 1-percentage_variance
                index += 1
                F = fftshift(fft(samples[:, i, :], 2), 2)
                F[:, 1:(T_center-index)] .= 0
                F[:, (T_center+index):end] .= 0
                reconstruction_error = mean(map(norm, eachslice(samples[:, i, :] - real.(ifft(ifftshift(F, 2), 2)), dims=1))./map(norm, eachslice(samples[:, i, :], dims=1)))
                println("Index $(index), Reconstruction error: $(reconstruction_error)")
            end
        else
            index = keep_nb_coef
        end
        results[i] = abs.(F_init[:, (1:index+1)])/T

    end

    return results
end