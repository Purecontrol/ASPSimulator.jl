using EmpiricalOrthogonalFunctions, FFTW, Plots

# Load data
data  = load("asm1-simulator/data/processed/morris/asm1_morris_samples_postprocess.jld")
asm1_samples = data["asm1_samples2"]

# Compute the EOF for O2, NO3, NH4
eof_oxygen = EmpiricalOrthogonalFunction(asm1_samples[:,8,:]')
eof_no3 = EmpiricalOrthogonalFunction(asm1_samples[:,9,:]')
eof_nh4 = EmpiricalOrthogonalFunction(asm1_samples[:,10,:]')

# Plot the 3 first orthogonal functions of EOF for O2, NO3, NH4
pyplot()
plot(pcs(eof_oxygen)[:,1:3], title="O2", xlabel="Temps (1 pas de temps = 10 minutes)", ylabel="Fonctions", label=["Composante 1" "Composante 2" "Composante 3"], legend_title="EOFs")
plot(pcs(eof_no3)[:,1:3], title="NO3", xlabel="Temps (1 pas de temps = 10 minutes)", ylabel="Fonctions", label=["Composante 1" "Composante 2" "Composante 3"], legend_title="EOFs")
plot(pcs(eof_nh4)[:,1:3], title="NH4", xlabel="Temps (1 pas de temps = 10 minutes)", ylabel="Fonctions", label=["Composante 1" "Composante 2" "Composante 3"], legend_title="EOFs")


# Compute the FFT for O2, NO3, NH4
F_oxygen = fftshift(fft(asm1_samples[:,8,:], 2), 2)
F_no3 = fftshift(fft(asm1_samples[:,9,:], 2), 2)
F_nh4 = fftshift(fft(asm1_samples[:,10,:], 2), 2)

# Plot the FFT of the first timeseries for O2, NO3, NH4
t = collect(range(9, stop=10, length=Int((24*60/10))))
Ts = 1.0/(24*60/10)
freqs = fftfreq(length(t), 1.0/Ts) |> fftshift
bar(freqs, abs.(F_oxygen[1,:]), title = "O2", xlabel = "Fréquence", ylabel = "Amplitude", legend = false)
bar(freqs, abs.(F_no3[1,:]), title = "NO", xlabel = "Fréquence", ylabel = "Amplitude", legend = false)
bar(freqs, abs.(F_nh4[1,:]), title = "NH", xlabel = "Fréquence", ylabel = "Amplitude", legend = false)



