using EmpiricalOrthogonalFunctions

###################################################
### Sensitivity Analysis Metrics for ASM1 model ###
###################################################

### Define the functions to compute the sensitivity indices ###

# Empirical Orthogonal Function
function EOF_metric(samples)

    # Compute the EOF for each species
    nb_samples, nb_species, _ = size(samples)
    results = Array{Float64, 2}(undef, nb_species, nb_samples)
    for i in 1:nb_species
        
        #Compute the EOF
        eof = EmpiricalOrthogonalFunction(samples[:,i,:]')

        # Determine the number of EOFs to keep
        variance = cumsum(variancefraction(eof))
        n = findfirst(x -> x > 0.99, variance)

        # Retrieve the PCs coefficients and sum them (do I need to take the mean in the analysis ?)
        results[i, :] = sum(eofs(eof, n=n), dims=2)'
    end

    return results
end

# Fourier Transform
function FT_metric(samples)

end

# Wavelet Transform
function WT_metric(samples)

end