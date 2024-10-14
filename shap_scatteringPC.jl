using FFTW
using Polynomials, SpecialPolynomials
using PyCall
using LinearAlgebra
using SpecialFunctions

psrchive = pyimport("psrchive");
pypolychord = pyimport("pypolychord")
settings = pyimport("pypolychord.settings")

@kwdef  struct Arch
    name::String
    dm::Float64
    period::Float64
    nbin::Int64
    nchan::Int64
    data::Matrix{Float64}
    noise::Vector{Float64}
    freq::Vector{Float64}
    reffreq::Float64
    ϕ::Float64
    nu::Vector{Float64}
    inc_τ::Bool = true
    inc_α::Bool = false
    inc_EFAC::Bool = true
    Nsamp::Int64 # Number of Shapelet amplitudes
    H::Array
    fact::Vector{Float64}
end

function PBF(twopif, τ, ν, α)
    """
    Pulse broadening function in Fourier domain(f, τ, ν, α):
        Return the Fourier pulse broadening function given by Lentati et al.2017:
             - f: Fourier frequencies
             - τ: the scattering time scale
             - ν: the observing frequency
             - α: the scattering index
    """
    twopifnu = twopif.*ν.^α * 10^τ
    return 1 ./(twopifnu.^2 .+ 1) + 1im*(-twopifnu)./(twopifnu.^2 .+ 1)
end

function rotate(fftprofile, ϕ, twopif2)
    """                                                                         
    rotate(fftprofile, ϕ):                                                      
        Return array 'fftprofile' rotated by ϕ to the left.  The           
                rotation is done using the Shift Theorem assuming
                the profile is in freq domain. ϕ is fractional 
                between 0 and 1.  The resulting vector will have           
                the same length as the original.                                    
    """
    #f = collect(0:length(fftprofile)-1)
    return fftprofile .* exp.(1im*twopif2 * ϕ)
end

function shapelet_model(Samp, nbin, nu, β, H, fact)

    pulse_model = zeros(nbin)
    nw = nu * β.^(-1)

    for n in 1:length(H)
        pulse_model += @. Samp[n] *  β^-.5 * fact[n] * H[n].(nw) * exp(-0.5 * nw.*nw)
    end
    return pulse_model
end
    
function load_data(filename, nchan, inc_τ, inc_α, inc_EFAC, Nsamp)
    ar = psrchive.Archive_load(filename)
    ar.tscrunch()
    ar.pscrunch()
    ar.fscrunch_to_nchan(nchan)
    #ar.bscrunch(2)
    ar.remove_baseline()
    dm = ar.get_dispersion_measure()
    nbin = ar.get_nbin()
    reffreq = ar.get_centre_frequency()
    
    # Get the profile data
    data = transpose(ar.get_data()[1,1,:,:])

    # Get Folding period
    integ = ar.get_first_Integration()
    period = integ.get_folding_period()

    # Get channel frequencies
    freqs = ar.get_frequencies()/1000.

    maxs = maximum(data, dims=1)
    data = data ./ maxs
    noise = vec(sqrt.(integ.baseline_stats()[2]) ./ maxs )

    # Estimate the Initial pulse phase
    ar.fscrunch()
    scrunch_prof = ar.get_data()[1,1,1,:]
    ϕ = 0.5 - argmax(scrunch_prof) / nbin
    if ϕ < 0
        ϕ +=1
    end

    # Prepare the Hermite Polynomial
    x = variable(Polynomial{Rational{Int}});
    H=[basis(Hermite, i)(x) for i in 0:Nsamp-1]
    nu = collect(0:nbin-1) / nbin * period .- period/2.

    # Prepare the factorial
    fact = zeros(Nsamp)
    for i in 0:Nsamp-1
        fact[i+1] = (2^i * Float64(factorial(big(i))) * π^.5)^-.5
    end

    arch = Arch(filename, dm, period, nbin, nchan, data, noise, freqs, reffreq, ϕ, nu, inc_τ, inc_α, inc_EFAC, Nsamp, H, fact)
    return arch
end

function prior(cube)
    """
    prior(cube)

    Converts the hyperspace cube to physical units.
    """
    
    ipar = 1
    pcube = zeros(length(cube))
    pcube[ipar] = LogUniformPrior!(cube[ipar], 1e-3, 1e-1); ipar += 1 # Width of the shapelet model
    pcube[ipar] = UniformPrior!(cube[ipar], 0, 1); ipar += 1 # Phase of the pulse at reffreq  

    # Shapelet amplitudes
    for i in 1:ar.Nsamp-1
        pcube[ipar] = UniformPrior!(cube[ipar], -2, 2); ipar+= 1
    end

    # include DM if more than one frequency channel in the model
    if ar.nchan > 1
       pcube[ipar] = GaussianPrior!(cube[ipar], 2100, 50); ipar += 1
    end

    # Scattering time scale at reffreq
    if ar.inc_τ
        pcube[ipar] = UniformPrior!(cube[ipar],-2, 1.); ipar += 1
    end
    
    # Scattering index 
    if ar.nchan > 1 && ar.inc_τ && ar.inc_α
        pcube[ipar] = GaussianPrior!(cube[ipar],-3.8, 0.2); ipar += 1
    end

    # EFAC
    if ar.inc_EFAC 
        pcube[ipar] = LogUniformPrior!(cube[ipar], 0.1,10 ); ipar +=1
    end

    return pcube
end

function UniformPrior!(cube, a, b)
    cube = a + (b-a)*cube
    return cube
end

function LogUniformPrior!(cube, a, b)
    cube = a * (b/a)^cube
    return cube
end

function GaussianPrior!(cube, μ, σ)
    cube = μ + σ*sqrt(2)*erfinv(2*cube-1)
    return cube
end


function likelihood(cube)
    nbin=ar.nbin
    nchan=ar.nchan
    Nsamp=ar.Nsamp
    inc_τ = ar.inc_τ
    inc_α = ar.inc_α
    inc_EFAC = ar.inc_EFAC

    ipar = 1
    β = cube[ipar] ; ipar += 1 # Shapelet width scale factor
    ϕ0 = cube[ipar]; ipar += 1 # Phase of the pulse at ...  

    Samp = ones(1)
    for isamp in 1:Nsamp-1
        append!(Samp, cube[ipar])
        ipar += 1
    end

    # include DM if more than one frequency channel in the model
    if nchan > 1
        DM = cube[ipar]; ipar += 1
    else
        DM = ones(1) * ar.DM
    end

    # Scattering time scale at XXX
    if inc_τ
        τ = cube[ipar]; ipar += 1
    end
    
    # Scattering index 
    if nchan > 1 && inc_τ && inc_α
        α = cube[ipar]; ipar += 1
    else
        α = ones(1) * -3.8
    end

    # EFAC, not implemented yet
    if inc_EFAC 
        EFAC = cube[ipar]; ipar +=1
    else
        EFAC = 1
    end

    # Compute the Noise vector
    Ni = ones(nchan * nbin)
    for i in 1:nchan
        for j in 1:nbin
            Ni[j+(i-1)*nbin] = 1/(ar.noise[i] * EFAC)^2
        end
    end
    Ni = Diagonal(Ni)  # Diagonal matrix of the inverted noise vector, size Nchan * Nbin
    detN = -tr(log(Ni))

    fftmodel_freqs = 2*π * collect(0:nbin/ 2) / ( ar.period) # Could be moved outside of the likelihood
    twopif2 = 2*π * collect(0:nbin/2)

    L = 0 #Likelihood value
    dat = vec(ar.data)
    F = zeros(nchan*nbin , 2*nchan)
    
    model = zeros(nbin)
    rotmodel = Vector{ComplexF64}(undef, div(nbin,2)+1)
    plan = plan_rfft(model)
    plan_i = plan_irfft(rotmodel, nbin; flags=FFTW.PATIENT, timelimit=Inf)

    model = shapelet_model(Samp, nbin, ar.nu, β, ar.H, ar.fact) 
    fftmodel = plan * model
     
    for ichan in 1:nchan

        # Compute dispersive phase delay
        if nchan > 1
            ϕ = modf(DM * 4.14879e-3 * (1/ar.freq[ichan]^2 - 1/ar.reffreq^2) / ar.period)[1]
        else
            ϕ = 0.0
        end

        # Add phase caused by DM delay to the free parameter ϕ0
        dϕ = ϕ0 - ϕ
        while dϕ < 0
            dϕ += 1
        end

        # Rotate in the Fourier domain
        rotmodel = rotate(fftmodel, dϕ, twopif2)

        # Scatter the profile
        if inc_τ
            rotmodel = rotmodel .* PBF(fftmodel_freqs, τ, ar.freq[ichan], α)
        end

        # FFT back the model for each channel
        model = plan_i * rotmodel
        model = model / maximum(model)

        F[(ichan-1)*nbin+1:ichan*nbin, 2*(ichan-1)+1] = model # Amplitude
        F[(ichan-1)*nbin+1:ichan*nbin, 2*(ichan-1)+2] = ones(nbin) # Baseline
    end

    Σ = transpose(F) * Ni * F
    dh = transpose(F) * Ni * dat

    try
        coeff = Σ \ dh
        Q,R = qr(Σ)
        detΣ = tr(log(abs.(Diagonal(R))))
        L = -0.5 * (detN + detΣ + transpose(dat) * Ni * dat - transpose(dh)*coeff)
    catch
        L = -10^20
    end

    return L, 0.

end

function main()
    nchan = 8;
    inc_τ = true;
    inc_α = false;
    inc_EFAC = true;
    Nsamp = 20;

    ar = load_data("/data/scattering/simulation_2100_2.ar", nchan, inc_τ, inc_α, inc_EFAC, Nsamp);
    parnames = ["beta", "phi0"]
    for i in 1:Nsamp-1 # First Shapelet component has a fixed amplitude of 1
       parnames = append!(parnames, ["samp$i"])
    end

    if nchan > 1
        parnames =  append!(parnames, ["DM"])
    end

    if inc_τ
       parnames =  append!(parnames, ["Tau"])
    end

    if nchan > 1 && inc_τ && inc_α
        parnames =  append!(parnames, ["gamma"])
    end

    if inc_EFAC
        parnames =  append!(parnames, ["EFAC"])
    end

    return ar, parnames
end

function dumper()
    println("");
end

ar, paramnames = main()
println(paramnames)

ndim = length(paramnames)
nDerived = 0
s = settings.PolyChordSettings(ndim, nDerived)
s.file_root = "chains"
s.nlive = 200
s.cluster_posteriors = false
s.do_clustering = false
s.write_dead = false
s.write_resume = false
s.read_resume = false
s.num_repeats = ndim * 3
s.synchronous = false

output = pypolychord.run_polychord(likelihood, ndim, nDerived, s, prior, dumper)


# Benchmarking the code
#using BenchmarkTools
#cube = rand(ndim)

#@btime begin
#    for i in 1:1000
#    likelihood(prior(cube))
#    end
#end