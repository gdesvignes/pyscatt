import math,os
import psrchive as p
import scipy, pylab
from scipy.special import factorial
import numpy as np
from pypolychord.priors import GaussianPrior, LogUniformPrior
import json
import numpy as np
import ultranest.plot
import pylab as plt
from multiprocessing import Pool
from ultranest import ReactiveNestedSampler, stepsampler
import corner


class Scattering:
    def __init__(self, filepath, nchan=2, incScatter=True, incScatter_index=False, 
                 Nshapelets=1, have_EFAC=False):
        
        """
        Initialize the Scattering class with given parameters.

        Parameters:
        - filepath (str): Path to the data file (e.g., .ar file).
        - nchan (int): Number of frequency channels to use.
        - incScatter (bool): Include scattering in the model.
        - incScatter_index (bool): Include scattering index as a free parameter.
        - Nshapelets (int): Number of shapelets to model the pulse profile.
        - have_EFAC (bool): Include EFAC (error factor) in the model.
        """

        # Set default priors for parameters
        self.pDM = (1, 500)            # Prior for Dispersion Measure (DM)
        self.pSc = (-6, -1)            # Prior for scattering timescale at 1 GHz (log units)
        self.pSci = (-5, -2)           # Prior for scattering index
        self.pEFAC = (0.5, 10)         # Prior for EFAC
        self.incScatter = incScatter
        self.incScatter_index = incScatter_index
        self.Tau_index = -3.6          # Default fixed value for the scattering index
        self.reffreq = 1               # Reference frequency of 1 GHz
        self.nScatter = 1              # Number of scattering parameters
        self.nfiles = 1                # Number of files (could be extended for multiple files)
        self.labels = []               # List to store parameter labels
        self.have_EFAC = have_EFAC     # Include EFAC in the model if True

        # Load data and set up model
        self.set_data(filepath, nchan)
        self.set_NShapelets(Nshapelets)
        self.set_labels()

    def set_pDM(self, pDM):
        """Set the prior range for Dispersion Measure (DM)."""
        self.pDM = pDM

    def set_pSc(self, pSc):
        """Set the prior range for the scattering timescale."""
        self.pSc = pSc

    def set_pSci(self, pSci):
        """Set the prior range for the scattering index."""
        self.pSci = pSci

    def set_pEFAC(self, pEFAC):
        """Set the prior range for EFAC (error factor)."""
        self.pEFAC = pEFAC

    def set_labels(self):
        """Define parameter labels for the model."""
        self.labels.extend(["width"])  # Shapelet width
        self.labels.extend([f'phi0_{i}' for i in range(self.nfiles)])  # Phase offsets
        self.labels.extend([f'amp_{i}' for i in range(self.nchan)])    # Amplitudes per channel

        # Include DM if more than one frequency channel
        if self.nchan > 1:
            self.labels.extend(["DM"])

        # Scattering timescale at reference frequency
        if self.incScatter:
            self.labels.extend([f'Tau_{i}' for i in range(self.nScatter)])

        # Scattering index
        if self.nchan > 1 and self.incScatter and self.incScatter_index:
            self.labels.extend(["Tau_idx"])

        # Shapelet amplitudes (for n > 1)
        if self.Nmax:
            self.labels.extend([f'Samp_{i}' for i in range(self.Nmax)])

        # Include EFAC if applicable
        if self.have_EFAC:
            self.labels.extend(["EFAC"])

    def get_labels(self):
        """Return the list of parameter labels."""
        return self.labels

    def fft_rotate(self, arr, phase):
        """
        Rotate an array 'arr' by 'phase' to the left using FFT.

        Parameters:
        - arr (numpy.ndarray): Input array to rotate.
        - phase (float): Fractional phase shift between 0 and 1.

        Returns:
        - numpy.ndarray: Rotated array.
        """
        arr = np.asarray(arr)
        freqs = np.arange(arr.size / 2 + 1, dtype=float)
        phasor = np.exp(complex(0.0, 2 * np.pi) * freqs * phase)
        return np.fft.irfft(phasor * np.fft.rfft(arr), arr.size)

    def rotate(self, arr, phase):
        """
        Rotate an array 'arr' by 'phase' to the left in the frequency domain.

        Parameters:
        - arr (numpy.ndarray): Input array in frequency domain.
        - phase (float): Fractional phase shift between 0 and 1.

        Returns:
        - numpy.ndarray: Rotated array.
        """
        freqs = np.arange(arr.size, dtype=float)
        phasor = np.exp(complex(0.0, 2 * np.pi) * freqs * phase)
        return arr * phasor

    def PBF(self, f, tau, nu, alpha):
        """
        Pulse Broadening Function (PBF) to model scattering effects.

        Parameters:
        - f (numpy.ndarray): Fourier frequencies.
        - tau (float): Scattering timescale.
        - nu (float): Observing frequency.
        - alpha (float): Scattering index.

        Returns:
        - numpy.ndarray: PBF values.
        """
        twopifnu = 2 * np.pi * f * nu ** alpha * 10 ** tau
        denominator = twopifnu ** 2 + 1
        return 1 / denominator + 1j * (-twopifnu) / denominator

    def set_NShapelets(self, Nmax):
        """
        Set up the number of shapelet coefficients and precompute Hermite polynomials.

        Parameters:
        - Nmax (int): Total number of shapelets to use.
        """
        self.Nmax = Nmax - 1  # The first shapelet coefficient is set to 1
        self.Scoeff = np.arange(self.Nmax + 1)

        # Precompute Hermite polynomials for efficiency
        self.hermit_ar = [scipy.special.hermite(n) for n in self.Scoeff]

    def Prior(self, cube):
        """
        Transform the unit cube samples into physical parameter values according to the priors.

        Parameters:
        - cube (numpy.ndarray): Unit cube samples.

        Returns:
        - numpy.ndarray: Physical parameter values.
        """
        pcube = np.zeros(cube.shape)
        ipar = 0

        # Shapelet width (log-uniform prior)
        pcube[ipar] = LogUniformPrior(2e-4, 5e-3)(cube[ipar])
        ipar += 1

        # Shapelet profile phase phi0 (Gaussian prior centered at initphase)
        pcube[ipar] = GaussianPrior(self.initphase, 0.01)(cube[ipar])
        ipar += 1

        # Overall scaling factor per channel (uniform prior between 0.8 and 1.0)
        pcube[ipar:ipar + self.nchan] = 0.8 + cube[ipar:ipar + self.nchan] * 0.2
        ipar += self.nchan

        # Include DM if more than one frequency channel
        if self.nchan > 1:
            pcube[ipar] = GaussianPrior(self.pDM[0], self.pDM[1])(cube[ipar])
            ipar += 1

        # Scattering timescale at reference frequency
        if self.incScatter:
            pcube[ipar] = cube[ipar] * (self.pSc[1] - self.pSc[0]) + self.pSc[0]
            ipar += 1

        # Scattering index
        if self.nchan > 1 and self.incScatter and self.incScatter_index:
            pcube[ipar] = GaussianPrior(-3.8, 0.1)(cube[ipar])
            ipar += 1

        # Shapelet amplitudes (for n > 1)
        if self.Nmax:
            pcube[ipar:ipar + self.Nmax] = cube[ipar:ipar + self.Nmax] * 5 - 2.5
            ipar += self.Nmax

        # EFAC
        if self.have_EFAC:
            pcube[ipar] = cube[ipar] * (self.pEFAC[1] - self.pEFAC[0]) + self.pEFAC[0]
            ipar += 1

        return pcube

    def set_data(self, filepath, nchan=4):
        """
        Load and preprocess data from the given file.

        Parameters:
        - filepath (str): Path to the data file.
        - nchan (int): Number of frequency channels to fscrunch to.
        """
        print(f"Loading file: {filepath}")
        self.prof = None
        self.wts = None
        self.period = None
        self.obsfreq = None
        self.prof_stds = None
        self.nbin = None
        self.data = None

        try:
            arch = p.Archive_load(filepath)
        except Exception as e:
            raise Exception(f"Error loading archive: {e}")

        self.nchan = nchan

        # Tscrunch (combine across time) and bscrunch (combine across polarizations)
        print("Tscrunching archive...")
        arch.tscrunch()
        arch.bscrunch(2)

        # Fscrunch (combine frequency channels)
        print(f"Fscrunching archive to {nchan} channels...")
        arch.fscrunch_to_nchan(nchan)
        arch.pscrunch()
        arch.remove_baseline()

        # Get DM and reference frequency
        self.aDM = arch.get_dispersion_measure()
        self.reffreq = arch.get_centre_frequency() / 1000.0  # Convert to GHz

        # Get the first integration
        integ = arch.get_first_Integration()

        # Extract data and weights
        self.data = arch.get_data()[0][0]
        self.wts = arch.get_weights()[0]

        # Normalize the data
        maximums = np.max(self.data, axis=1)
        maximums = np.where(maximums == 0, 1, maximums)
        self.prof = (self.data.T / maximums).T  # Normalize profiles
        self.profT = self.prof.T  # Transpose for later use

        # Get period
        self.period = integ.get_folding_period()

        # Get observing frequencies in GHz
        self.obsfreq = arch.get_frequencies() / 1000.0
        print("Observing frequencies (GHz):", self.obsfreq)

        # Get standard deviations (RMS noise levels)
        stats = np.sqrt((integ.baseline_stats()[1][0]) / maximums)
        stats = np.where(stats == 0, 1.0, stats)
        self.prof_stds = stats

        # Number of bins in the profile
        self.nbin = arch.get_nbin()

        # Create time array centered around zero (-P/2 to +P/2)
        self.nu = np.arange(self.nbin, dtype=float) / self.nbin * self.period - self.period / 2.0

    def plot_profiles(self):
        """Plot the observed and modeled pulse profiles."""
        pylab.xlabel("Pulse phase (bins)")

        # Plot observed profiles
        pylab.subplot(2, 1, 1)
        for ichan in range(self.nchan):
            pylab.plot(self.prof[ichan] + ichan)
        pylab.title("Observed Profiles")

        # Plot modeled profiles
        pylab.subplot(2, 1, 2)
        for ichan in range(self.nchan):
            pylab.plot(self.model[ichan] + ichan)
        pylab.title("Modeled Profiles")
        pylab.tight_layout()
        pylab.show()

    def shapelet_profile(self, Samp, width):
        """
        Generate the pulse profile using shapelets.

        Parameters:
        - Samp (numpy.ndarray): Shapelet amplitudes (excluding the first, which is set to 1).
        - width (float): Width parameter for the shapelets.

        Returns:
        - numpy.ndarray: Generated pulse profile.
        """
        nu_width = self.nu / width

        # Initialize model with the first shapelet (n=0) with amplitude 1
        n = 0
        norm_factor = (2 ** n * factorial(n) * np.pi ** 0.5) ** -0.5
        model = 1 * width ** -0.5 * norm_factor * self.hermit_ar[n](nu_width) * np.exp(-0.5 * nu_width ** 2)

        # Add higher-order shapelets
        for idx, n in enumerate(self.Scoeff[1:], start=1):
            norm_factor = (2 ** n * factorial(n) * np.pi ** 0.5) ** -0.5
            model += Samp[idx - 1] * width ** -0.5 * norm_factor * self.hermit_ar[n](nu_width) * np.exp(-0.5 * nu_width ** 2)

        return model

    def prof_to_ref_freq(self):
        """
        Align profiles to the reference frequency by correcting for dispersion delays.
        """
        prof_tot = np.zeros(self.prof[0].size)

        for ichan in range(self.nchan):
            # Calculate the delay in phase units
            delay = self.aDM * 4.14879e-3 * (1 / self.obsfreq[ichan] ** 2 - 1 / self.reffreq ** 2)
            delay_phase = math.modf(delay / self.period)[0]

            # Rotate profile to align with reference frequency
            prof_rot = self.fft_rotate(self.prof[ichan], delay_phase)
            pylab.plot((prof_rot+ichan))
            prof_tot += prof_rot
        
        pylab.plot((prof_tot+ichan+1))

        # Estimate initial phase for the model
        self.initphase = 0.5 - np.argmax(prof_tot) / self.prof[0].size
        while self.initphase < 0.0:
            self.initphase += 1.0

        print(f"Initial phase estimated: {self.initphase}")
        pylab.show()

    def LogLikelihood(self, cube):
        """
        Calculate the log-likelihood of the model given the data.

        Parameters:
        - cube (numpy.ndarray): Parameter values.

        Returns:
        - float: Log-likelihood value.
        """
        logdet = 0.0
        ipar = 0

        # Extract parameters from cube
        width = cube[ipar]
        ipar += 1

        phi0 = cube[ipar]
        ipar += 1

        Amp = cube[ipar:ipar + self.nchan]
        ipar += self.nchan

        # Dispersion Measure (DM)
        if self.nchan > 1:
            DM = cube[ipar]
            ipar += 1
        else:
            DM = self.pDM[0]  # Use default DM if only one channel

        # Scattering timescale
        if self.incScatter:
            Tau = cube[ipar]
            ipar += 1
        else:
            Tau = self.pSc[0]  # Use default scattering timescale if not included

        # Scattering index
        if self.nchan > 1 and self.incScatter and self.incScatter_index:
            self.Tau_index = cube[ipar]
            ipar += 1

        # Shapelet amplitudes
        if self.Nmax:
            Samp = cube[ipar:ipar + self.Nmax]
            ipar += self.Nmax
        else:
            Samp = np.array([])

        # EFAC (error factor)
        if self.have_EFAC:
            EFAC = cube[ipar]
            ipar += 1
        else:
            EFAC = 1.0

        # Generate the pulse profile using shapelets
        model_profile = self.shapelet_profile(Samp, width)

        # FFT of the model profile
        fftmodel_freqs = np.arange(model_profile.size / 2 + 1, dtype=float) / self.period
        self.fft_model = np.fft.rfft(model_profile)

        self.model = np.zeros((self.nchan, self.nbin))

        # Loop over each frequency channel
        for ichan in range(self.nchan):
            # Skip channels with zero weight
            if not self.wts[ichan]:
                continue

            # Calculate DM delay in phase units
            if self.nchan > 1:
                delay = DM * 4.14879e-3 * (1 / self.obsfreq[ichan] ** 2 - 1 / self.reffreq ** 2)
                delay_phase = math.modf(delay / self.period)[0]
            else:
                delay_phase = 0.0

            # Apply DM delay to phase offset
            phase_off = phi0 - delay_phase
            while phase_off < 0:
                phase_off += 1.0

            # Rotate the model profile
            fft_model_rot = self.rotate(self.fft_model, phase_off)

            # Apply scattering effects if included
            if self.incScatter:
                pbf = self.PBF(fftmodel_freqs, Tau, self.obsfreq[ichan], self.Tau_index)
                fft_model_rot *= pbf

            # Inverse FFT to get the time-domain model profile
            self.model[ichan] = np.fft.irfft(fft_model_rot, model_profile.size)
            self.model[ichan] /= np.max(self.model[ichan])  # Normalize

            # Update log determinant for EFAC
            if self.have_EFAC:
                logdet += np.log((self.prof_stds[ichan] * EFAC) ** 2)

        # Apply amplitude scaling per channel
        self.model = (self.model.T * Amp).T
        self.modelc = self.model.T  # Transpose for comparison with data

        # Calculate chi-squared
        chi_squared = np.sum((self.modelc - self.profT) ** 2 / (self.prof_stds ** 2))

        # Return the log-likelihood
        return -0.5 * chi_squared - 0.5 * logdet
            
class Utils:
    def __init__(self, json_file):
        """
        Initialize the Utils class and load the parameters from the JSON file.
        
        Parameters:
        - json_file (str): Path to the input JSON file.
        """
        self.params = self.load_input_params(json_file)
    
    def load_input_params(self, json_file):
        """
        Load the input parameters from a JSON file.
        
        Parameters:
        - json_file (str): Path to the input JSON file.
        
        Returns:
        - dict: A dictionary of input parameters.
        """
        try:
            with open(json_file, 'r') as f:
                params = json.load(f)
            print(f"Parameters loaded successfully from {json_file}.")
            return params
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None
    
    def get_scattering_parameters(self):
        """
        Extract and return parameters from the JSON input that are necessary for the Scattering class.
        
        Returns:
        - dict: A dictionary containing the necessary parameters for initializing the Scattering class.
        """
        try:
            # Extract required parameters from the loaded JSON
            filepath = self.params['filepath']
            nchan = self.params['nchan']
            incScatter = self.params.get('incScatter', True)
            incScatter_index = self.params.get('incScatter_index', False)
            Nshapelets = self.params.get('Nshapelets', 1)
            have_EFAC = self.params.get('have_EFAC', False)
            
            return {
                'filepath': filepath,
                'nchan': nchan,
                'incScatter': incScatter,
                'incScatter_index': incScatter_index,
                'Nshapelets': Nshapelets,
                'have_EFAC': have_EFAC
            }
        except KeyError as e:
            print(f"Missing required key in JSON input: {e}")
            return None
    
    def get_priors(self):
        """
        Extract and return prior ranges for DM, scattering, and other parameters from the JSON input.
        
        Returns:
        - dict: A dictionary of prior ranges.
        """
        try:
            pDM = self.params['DMr']  # Prior range for Dispersion Measure
            pSc = self.params['Tau_sc']  # Prior range for scattering timescale (log units)
            pSci = self.params.get('Tau_idx', (-5, -2))  # Prior range for scattering index (default range)
            pEFAC = self.params.get('EFAC', (0.5, 10))  # Prior range for EFAC
            
            return {
                'pDM': pDM,
                'pSc': pSc,
                'pSci': pSci,
                'pEFAC': pEFAC
            }
        except KeyError as e:
            print(f"Missing required key in JSON input: {e}")
            return None
    
    # Placeholder for potential plotting function
    def plot_results(self, data, model):
        """
        Placeholder for a plotting function to visualize the results.
        
        Parameters:
        - data (numpy.ndarray): Observed data.
        - model (numpy.ndarray): Modeled data.
        """
        print("Plotting functionality will be added later.")


    def get_ndims_and_paramnames(self):
        """
        Calculate and return the number of dimensions (Ndim) and the list of parameter names (paramnames).
        
        Returns:
        - tuple: (Ndim, paramnames)
        """
        try:
            # Extract parameters from the input
            nchan = self.params['nchan']
            Nshapelets = self.params['Nshapelets']
            incScatter = self.params['incScatter']
            incScatter_index = self.params['incScatter_index']
            have_EFAC = self.params['have_EFAC']
            
            # Initial parameters: Shapelet width and phase
            Ndim = 1 + 1 + nchan + (Nshapelets - 1)  # Sw + Phi + nchan amplitudes + (Nshapelets - 1)
            paramnames = ['Sw', 'Phi'] + [f"A{i}" for i in range(nchan)]
            
            # Add DM if there is more than 1 frequency channel
            if nchan > 1:
                Ndim += 1
                paramnames.append('DM')
            
            # Add scattering parameters if scattering is enabled
            if incScatter:
                Ndim += 1
                paramnames.append('Tau_1')
            
            # Add scattering index if enabled
            if incScatter_index:
                Ndim += 1
                paramnames.append('Tau_idx')
            
            # Add shapelet amplitudes (for n > 1)
            paramnames += [f"Sa{i}" for i in range(Nshapelets - 1)]
            
            # Add EFAC if enabled
            if have_EFAC:
                Ndim += 1
                paramnames.append('EFAC')
            
            print(f"Ndim = {Ndim}\n")
            print(f"Parameter names: {paramnames}")
            
            return Ndim, paramnames
        
        except KeyError as e:
            print(f"Missing required key in JSON input: {e}")
            return None

class Sampler:
    def __init__(self, paramnames, loglikelihood, prior_transform, min_num_live_points=400):
        """
        Initialize the Sampler class for running nested sampling.

        Parameters:
        - paramnames (list): Names of the parameters.
        - loglikelihood (function): Log-likelihood function.
        - prior_transform (function): Function to transform unit cube to parameter space.
        - min_num_live_points (int): Minimum number of live points for nested sampling.
        """
        self.paramnames = paramnames
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.min_num_live_points = min_num_live_points
        self.sampler = None
        self.results = None

    def setup_sampler(self, filepath, Ndim, base_logdir="output"):
        """
        Set up the nested sampler with the provided likelihood and prior functions.

        Parameters:
        - filepath (str): Full path to the input file (used to create the directory name).
        - Ndim (int): Number of dimensions (used to create the directory name).
        - base_logdir (str): Base directory where the log directory will be created. Default is 'output'.

        The final log directory will be <base_logdir>/<filename>_<Ndim>/.
        """
        # Extract the base filename (without extension) from the full filepath
        filename_base = os.path.splitext(os.path.basename(filepath))[0]
        
        # Create the directory name using the base filename and Ndim
        logdir = os.path.join(base_logdir, f"{filename_base}_{Ndim}")
        
        # Ensure the directory exists
        os.makedirs(logdir, exist_ok=True)

        # Set up the sampler with the log directory
        self.sampler = ReactiveNestedSampler(self.paramnames, 
                                            self.loglikelihood, 
                                            self.prior_transform, 
                                            log_dir=logdir,
                                            vectorized=False)  # Specify the log directory

        print(f"Sampler initialized with log directory: {logdir}")
        print(f"Sampler initialized with parameters: {self.paramnames}")


    def run_sampler(self):
        """Run the nested sampling process."""
        if self.sampler is None:
            raise ValueError("Sampler has not been set up. Call setup_sampler() first.")
        
        print(f"Running sampler with {self.min_num_live_points} live points...")
        self.results = self.sampler.run(min_num_live_points=self.min_num_live_points,
                                        viz_callback=False,
                                        show_status=True, #show a simple progress bar
                                        )
        self.sampler.print_results()
        print("Sampling complete.")
    
    def extract_samples(self):
        """Extract the posterior samples from the sampler results."""
        if self.results is None:
            raise ValueError("No results available. Run the sampler first.")
        
        return self.results['samples']
    
    def extract_scattering_params(self):
        """
        Extract the scattering timescale (in milliseconds) and scattering index
        from the posterior samples.
        
        Returns:
        - tuple: (scattering_timescale_ms, scattering_index)
        """
        if self.results is None:
            raise ValueError("No results available. Run the sampler first.")
        
        samples = self.results['samples']
        
        # Find the indices of Tau_1 (scattering timescale) and Tau_idx (scattering index)
        tau_index = self.paramnames.index('Tau_1')
        scattering_index_idx = self.paramnames.index('Tau_idx') if 'Tau_idx' in self.paramnames else None
        
        # Convert scattering timescale from log scale to milliseconds
        tau_samples_ms = 10 ** samples[:, tau_index] * 1000  # Convert to ms
        
        if scattering_index_idx is not None:
            scattering_index_samples = samples[:, scattering_index_idx]
        else:
            scattering_index_samples = None
        
        return tau_samples_ms, scattering_index_samples
    

    def plot_corner_scattering(self, plot_path):
        """Generate a corner plot for scattering timescale, scattering index, 
        and DM using the 'corner' library."""

        if self.results is None:
            raise ValueError("No results available. Run the sampler first.")
        
        # Extract samples
        samples = self.results['samples']

        # Find indices for Tau_1 (scattering timescale), Tau_idx (scattering index), and DM
        try:
            tau_index = self.paramnames.index('Tau_1')
        except ValueError:
            raise ValueError("'Tau_1' (scattering timescale) not found in paramnames.")
        
        scattering_index_idx = self.paramnames.index('Tau_idx') if 'Tau_idx' in self.paramnames else None
        dm_index = self.paramnames.index('DM') if 'DM' in self.paramnames else None

        # Convert Tau_1 from log scale to milliseconds
        tau_samples_ms = 10 ** samples[:, tau_index] * 1000  # Tau_1 is in log scale, convert to ms

        # Prepare data and labels for corner plot
        tau_and_index_samples = [tau_samples_ms]  # Start with scattering timescale
        param_labels = ['Scattering Timescale (ms)']

        # Add scattering index if available
        if scattering_index_idx is not None:
            scattering_idx_samples = samples[:, scattering_index_idx]
            tau_and_index_samples.append(scattering_idx_samples)
            param_labels.append('Scattering Index')

        # Add DM if available
        if dm_index is not None:
            dm_samples = samples[:, dm_index]
            tau_and_index_samples.append(dm_samples)
            param_labels.append('DM')

        # Convert to 2D array and plot using 'corner'
        tau_and_index_samples = np.column_stack(tau_and_index_samples)
        figure = corner.corner(tau_and_index_samples, labels=param_labels)
        if not plot_path == None:
            plt.savefig(plot_path)
            plt.close()
        else:
            plt.show()

    
    def plot_corner_all(self):
        """Generate a corner plot for all parameters."""
        if self.results is None:
            raise ValueError("No results available. Run the sampler first.")
        
        # Plot corner plot for all parameters
        figure = corner.corner(self.results['samples'], labels=self.paramnames)
        plt.show()

    def output_text_summary(self):
        """Print a textual summary of the sampler results including scattering timescale and index,
        with mean, median, and 95% confidence intervals."""
        if self.results is None:
            raise ValueError("No results available. Run the sampler first.")
        
        # Extract scattering timescale (ms) and scattering index
        tau_samples_ms, scattering_index_samples = self.extract_scattering_params()
        
        # Compute mean, median, and 95% confidence intervals for scattering timescale
        tau_mean = np.mean(tau_samples_ms)
        tau_median = np.median(tau_samples_ms)
        tau_lower, tau_upper = np.percentile(tau_samples_ms, [2.5, 97.5])  # 95% confidence interval

        # Output scattering timescale
        print(f"Scattering Timescale (ms):")
        print(f"  Mean: {tau_mean:.3f}")
        print(f"  Median: {tau_median:.3f}")
        print(f"  95% CI: {tau_median:.3f} (+{tau_upper - tau_median:.3f}, -{tau_median - tau_lower:.3f})")

        # If the scattering index is present, compute its statistics
        if scattering_index_samples is not None:
            scattering_idx_mean = np.mean(scattering_index_samples)
            scattering_idx_median = np.median(scattering_index_samples)
            scattering_idx_lower, scattering_idx_upper = np.percentile(scattering_index_samples, [2.5, 97.5])

            # Output scattering index
            print(f"Scattering Index:")
            print(f"  Mean: {scattering_idx_mean:.3f}")
            print(f"  Median: {scattering_idx_median:.3f}")
            print(f"  95% CI: {scattering_idx_median:.3f} (+{scattering_idx_upper - scattering_idx_median:.3f}, -{scattering_idx_median - scattering_idx_lower:.3f})")
            # Return both scattering timescale and index statistics
            return (tau_mean, tau_lower, tau_upper, scattering_idx_mean, 
                    scattering_idx_lower, scattering_idx_upper)
        else:

            print("Scattering index not included in the model.")
            return (tau_mean, tau_lower, tau_upper, None, None, None)


"""
TODO
1. Baseline offsets likely for every channel - add these and marginalise.
2. Account for scattering and pulse period overlaps
3. Is there a better way to measure scattering as a function of time? 
4. Come up a list of pulsars that are not in Lucy's paper that are scattered? 

Advanced TODO:
1. Polarisation fitting and scattering evolution in the Stokes parameters
"""
