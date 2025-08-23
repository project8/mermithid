'''
Generate binned or pseudo unbinned data
Author: T. E. Weiss, C. Claessens, X. Huyan
Date: April 6, 2020
Updated: July 10, 2024

This processsor is employed in Project 8's official Bayesian and frequentist analyses of
tritium beta-decay data.

Specifically, we use this processor in scripts in the termite repository called
fake_data_stan_analysis.py and fake_and_real_data_frequentist_analysis_2021.py for the
Bayesian and frequentist analyses, respectively.

To run this processor, include the following lines in a different python script:
from mermithid.processors.TritiumSpectrum.FakeDataGenerator import FakeDataGenerator
specGen = FakeDataGenerator("specGen")
specGen_config = { ...
    }
#Configuration step
specGen.Configure(specGen_config)
#Generate data
specGen.Run()
results = specGen.results

Where the specGen_config is a dictionary with entries that define the parameters described
below, under InternalConfigure.
'''

from __future__ import absolute_import

import numpy as np
import math
import random
import time
import json
import os

from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.misc.FakeTritiumDataFunctions import *
from mermithid.processors.misc.MultiGasComplexLineShape import MultiGasComplexLineShape
from mermithid.misc import Constants, ComplexLineShapeUtilities, ConversionFunctions
logger = morphologging.getLogger(__name__)


__all__ = []
__all__.append(__name__)


class FakeDataGenerator(BaseProcessor):
    """
    Generates (pseudo) binned tritium data with Project 8 specific features, like lineshape and effciency.
    Can be configured to produce energy and frequency data.
    """

    def InternalConfigure(self, params):
        """
        All parameters have defaults.
        Configurable parameters are:
        - Q [eV]: endpoint energy
        - neutrino_mass [eV]: true neutrino mass
        – Kmin [eV]: low energy cut-off
        – Kmax [eV]: high energy cut-off
        - minf [Hz]: low frequency cut-off
        - maxf [Hz]: high frequency cutoff (optional; if not provided, max frequency is determined from efficiency dict)
        - n_steps: number of energy bins that data will be drawn from
        - B_field: used for energy-frequency conversion
        - sig_trans [eV]: width of thermal broadening
        - other_sig [eV]: width of other broadening
        - channel_runtimes [s]: live time for each channel
        - channel_bounds [Hz]: inner bounds between channels (one less than the number of channels)
        - S: number of signal events
        - poisson_stats (boolean): if True number of total events is random
        - err_from_B [eV]: energy uncertainty originating from B uncertainty
        – gases: list of strings naming gases to be included in complex lineshape model. Options: 'H2', 'He', 'Kr', 'Ar', 'CO'
        - NScatters: lineshape parameter - number of scatters included in lineshape
        - trap_weights: distionary of two lists, labeled 'weights' and 'errors', which respectively include the fractions of counts from each trap and the uncertainties on those fractions
        - scatter_peak_ratio_p: "p" in reconstrudction efficiency curve model: e^(-p*i^(-factor*p+q)), where i is the scatter order
        - scatter_peak_ratio_q: "q" in the same reconstruction efficiency model
        - scatter_peak_ratio_factor: "factor" in the same reconstruction efficiency model
        – scatter_proportion: list of proportion of scatters due to each gas in self.gases (in the same order), in complex lineshape
        - survival_prob: lineshape parameter - probability of electron staying in the trap between two inelastics scatters (it could escape due to elastics scatters or the inelastics scatters, themselves)
        – use_radiation_loss: if True, radiation loss will be included in the complex lineshape; should be set to True except for testing purposes
        – resolution_function: string determinign type of resolution function; options are 'simulated_resolution', 'gaussian_resolution', 'gaussian_lorentzian_composite_resolution'.
        – ratio_gamma_to_sigma: parameter in gaussian_lorentzian_composite_resolution; see the MultiGasComplexLineShape processor
        – gaussian_proportion: also a parameter in gaussian_lorentzian_composite_resolution
        – A_array: parameter for Gaussian resolution with variable width; see the MultiGasComplexLineShape processor
        - sigma_array: also a parameter for Gaussian resolution with variable width
        – fit_recon_eff: if True, determine recon eff parameters by fitting to FTG data, instead of by using parameters given in self.recon_eff_parameters
        – use_combined_four_trap_inst_reso: for simulated resolution, combine resolution distributions provided for four different traps
        - sample_ins_resolution_errors: if True, and if using a simulated instrumental resolution, the count numbers in that distribution will be sampled from uncertainties
        - scattering_sigma [eV]: lineshape parameter - 0-th peak gaussian broadening standard deviation
        - min_energy [eV]: minimum of lineshape energy window for convolution with beta spectrum. Same magnitude is used for the max energy of the window.
        - scale_factor: width scaling for a simulated instrumental resolution
        - ins_res_width_bounds: Bounds (eV) of regions within which the resolution width is approximately constant (and therefore can be parameterized by a single scale_factor). ins_res_width_bounds does not include the outermost bounds - i.e., the Kmin and Kmax. If ins_res_width_bounds is None, then a common scale factor is used for the whole ROI.
        - ins_res_width_factors: Factors that are multiplied by scale_factor in each of the regions defined by ins_res_width_bounds. Note that len(ins_res_width_factors) == len(ins_res_width_bounds) + 1.

        - efficiency_path: path to efficiency vs. frequency (and uncertainties)
        - simplified_scattering_path: path to simplified lineshape parameters
        – path_to_osc_strengths_files: path to oscillator strength files containing energy loss distributions for the gases in self.gases
        – path_to_scatter_spectra_file: path to scatter spectra file, which is generated from the osc strenth files and accounts for cross scattering
        – rad_loss_path: path to file containing data describing radiation loss
        - path_to_detailed_scatter_spectra_dir: path to oscillator and or scatter_spectra_file
        – path_to_ins_resolution_data_txt: path to file containing simulated instrumental resolution data, already combined among the four traps
        - path_to_four_trap_ins_resolution_data_txt: path to files containing simulated instrumental resolution data for each of the four individual traps
        - final_states_file: path to file containing molecular final state binding energies and corresponding probabilities
        - use_lineshape (boolean): determines whether tritium spectrum is smeared by lineshape.
          If False, it will only be smeared with a Gaussian
        - detailed_or_simplified_lineshape: If use lineshape, this string determines which lineshape model is used.
        - apply_efficiency (boolean): determines whether tritium spectrum is multiplied by efficiency
        - return_frequency: data is always generated as energies. If this parameter is true a list of frequencies is added to the dictionary
        - molecular_final_states: if True molecular final states are included in tritium spectrum (using spectrum from Bodine et. al 2015)
        """

        # Read other parameters
        self.Q = reader.read_param(params, 'Q', QT2) #Choose the atomic or molecular tritium endpoint
        self.m = reader.read_param(params, 'neutrino_mass', 0.2) #Neutrino mass (eV)
        self.Kmin = reader.read_param(params, 'Kmin', self.Q-self.m-2300)  #Energy corresponding to lower bound of frequency ROI (eV)
        self.Kmax = reader.read_param(params, 'Kmax', self.Q-self.m+1000)   #Same, for upper bound (eV)
        self.minf = reader.read_param(params, 'minf', 25813125000.0) #Minimum frequency
        self.maxf = reader.read_param(params, 'maxf', None)
        if self.Kmax <= self.Kmin:
            logger.error("Kmax <= Kmin!")
            return False
        self.n_steps = reader.read_param(params, 'n_steps', 1e5)
        if self.n_steps <= 0:
            logger.error("Negative number of steps!")
            return False
        self.B_field = reader.read_param(params, 'B_field', 0.9578186017836624)

        #For a Phase IV gaussian smearing:
        self.sig_trans = reader.read_param(params, 'sig_trans', 0.020856) #Thermal translational Doppler broadening for atomic T (eV)
        self.other_sig = reader.read_param(params, 'other_sig', 0.05) #Kinetic energy broadening from other sources (eV)
        self.broadening = np.sqrt(self.sig_trans**2+self.other_sig**2) #Total energy broadening (eV)

        # Phase II Spectrum parameters
        self.channel_runtimes = reader.read_param(params, 'channel_runtimes', [7185228., 7129663., 7160533.])
        self.channel_bounds = reader.read_param(params, 'channel_bounds', [1.38623121e9+24.5e9, 1.44560621e9+24.5e9])
        self.S = reader.read_param(params, 'S', 3300)
        self.B_1kev = reader.read_param(params, 'B_1keV', 0.1) #Background rate per keV for full runtime
        self.poisson_stats = reader.read_param(params, 'poisson_stats', True)
        self.err_from_B = reader.read_param(params, 'err_from_B', 0.) #In eV, kinetic energy error from f_c --> K conversion


        #Scattering model parameters
        self.gases = reader.read_param(params, 'gases', ['H2', 'He', 'CO'])
        self.NScatters = reader.read_param(params, 'NScatters', 20)
        self.trap_weights = reader.read_param(params, 'trap_weights', {'weights':[0.076,  0.341, 0.381, 0.203], 'errors':[0.003, 0.013, 0.014, 0.02]})
        #self.recon_eff_params = reader.read_param(params, 'recon_eff_params', [0.005569990343215976, 0.351, 0.546])
        self.scatter_peak_ratio_p = reader.read_param(params, 'scatter_peak_ratio_p', 1.)
        self.scatter_peak_ratio_q = reader.read_param(params, 'scatter_peak_ratio_q', 0.6)
        self.scatter_peak_ratio_factor = reader.read_param(params, 'scatter_peak_ratio_factor', 0.5)
        self.scatter_proportion = reader.read_param(params, 'scatter_proportion', [])
        self.survival_prob = reader.read_param(params, 'survival_prob', 1.)
        self.use_radiation_loss = reader.read_param(params, 'use_radiation_loss', True)
        self.resolution_function = reader.read_param(params, 'resolution_function', '')
        self.ratio_gamma_to_sigma = reader.read_param(params, 'ratio_gamma_to_sigma', 0.8)
        self.gaussian_proportion = reader.read_param(params, 'gaussian_proportion', 0.8)
        self.A_array = reader.read_param(params, 'A_array', [0.076, 0.341, 0.381, 0.203])
        self.sigma_array = reader.read_param(params, 'sigma_array', [5.01, 13.33, 15.40, 11.85])
        self.fit_recon_eff = reader.read_param(params, 'fit_recon_eff', False)
        self.use_combined_four_trap_inst_reso = reader.read_param(params, 'use_combined_four_trap_inst_reso', True)
        self.sample_ins_resolution_errors = reader.read_param(params, 'sample_ins_res_errors', True)
        self.scattering_sigma = reader.read_param(params, 'scattering_sigma', 18.6)
        self.min_energy = reader.read_param(params,'min_lineshape_energy', -1000)
        self.scale_factor = reader.read_param(params, 'scale_factor', 1.0)
        self.ins_res_width_bounds = reader.read_param(params, 'ins_res_width_bounds', None) #Default values here need to be corrected
        self.ins_res_width_factors = reader.read_param(params, 'ins_res_width_factors', [1.])
        self.p_factors = reader.read_param(params, 'p_factors', [1.])
        self.q_factors = reader.read_param(params, 'q_factors', [1.])

        #paths
        self.efficiency_path = reader.read_param(params, 'efficiency_path', '')
        self.simplified_scattering_path = reader.read_param(params, 'simplified_scattering_path', '/host/input_data/simplified_scattering_params.txt')
        self.path_to_osc_strengths_files = reader.read_param(params, 'path_to_osc_strengths_files', '/host/')
        self.path_to_scatter_spectra_file = reader.read_param(params, 'path_to_scatter_spectra_file', '/host/')
        self.rad_loss_path = reader.read_param(params, 'rad_loss_path', '')
        self.path_to_ins_resolution_data_txt = reader.read_param(params, 'path_to_ins_resolution_data_txt', '/host/ins_resolution_all.txt')
        self.path_to_four_trap_ins_resolution_data_txt = reader.read_param(params, 'path_to_four_trap_ins_resolution_data_txt', ['/host/analysis_input/complex-lineshape-inputs/T2-1.56e-4/res_cf15.5_trap1.txt', '/host/analysis_input/complex-lineshape-inputs/T2-1.56e-4/res_cf15.5_trap2.txt', '/host/T2-1.56e-4/analysis_input/complex-lineshape-inputs/res_cf15.5_trap3.txt', '/host/analysis_input/complex-lineshape-inputs/T2-1.56e-4/res_cf15.5_trap4.txt'])
        self.final_states_file = reader.read_param(params, 'final_states_file', '')
        self.shake_spectrum_parameters_json_path = reader.read_param(params, 'shake_spectrum_parameters_json_path', 'shake_spectrum_parameters.json')

        #options
        self.use_lineshape = reader.read_param(params, 'use_lineshape', True)
        self.detailed_or_simplified_lineshape = reader.read_param(params, 'detailed_or_simplified_lineshape', 'detailed')
        self.apply_efficiency = reader.read_param(params, 'apply_efficiency', False)
        self.return_frequency = reader.read_param(params, 'return_frequency', True)
        self.molecular_final_states = reader.read_param(params, 'molecular_final_states', False)


        # will be replaced with complex lineshape object if detailed lineshape is used
        self.complexLineShape = None

        # get file content if needed
        # get efficiency dictionary
        if self.apply_efficiency:
            self.efficiency_dict = self.load_efficiency_curve()
            np.random.seed()
        else:
            self.efficiency_dict = None

        # generate data with lineshape
        if self.use_lineshape:
            self.lineshape = self.detailed_or_simplified_lineshape

            # The simplified lineshape model uses Gaussians to model each scatter peak.
            # It is appropriate for a case where the instrumental resolution (detector
            # response of unscattered electrons) is broad, so that the instrumental res
            # width dominates over the asymmetry in the underlying scatter-spectrum shape.
            # For more detail, see the function "simplified_ls" in the script:
            # mermithid/mermithid/misc/FakeTritiumDataFunctions.py
            if self.lineshape == 'simplified':
                self.ls_params = self.load_simp_params(self.scattering_sigma,
                                                        self.survival_prob,
                                                        self.NScatters)
            
            # The detailed lineshape model numerically convolves scatter spectra with a
            # simulated instrumental resolution function to produce scatter peaks. The
            # scatter peak ratios are modeled with a modified exponential and depends on
            # variables p and q (see Eq. 15 of https://arxiv.org/pdf/2303.12055). Gas
            # composition, count ratios in different electron traps, uncertainties on the
            # simulated resolution, and energy loss from cylotron radiation are all included.
            # For the underlying spectrum, either a Dirac Delta function or the krypton
            # spectrum (with shake-up and shake-off) may be used. For more information, see
            # the complexLineShape_config dictionary below, as well as the processor:
            # mermithid/mermithid/processors/misc/MultiGasComplexLineShape 
            elif self.lineshape=='detailed':
                # check path exists
                if 'scatter_spectra_file' in self.path_to_scatter_spectra_file:
                    full_path = self.path_to_scatter_spectra_file
                    self.path_to_scatter_spectra_file, _ = os.path.split(full_path)
                else:
                    full_path = os.path.join(self.path_to_scatter_spectra_file, 'scatter_spectra_file')

                logger.info('Path to scatter_spectra_file: {}'.format(self.path_to_scatter_spectra_file))


                # lineshape params
                if self.resolution_function == 'gaussian_resolution' or self.resolution_function == 'gaussian':
                    self.ls_params = [self.scattering_sigma*2*math.sqrt(2*math.log(2)), self.survival_prob]
                else:
                    self.ls_params = [self.scale_factor, self.survival_prob]

                # Setup and configure lineshape processor
                complexLineShape_config = {
                    'gases': self.gases,
                    'max_scatters': self.NScatters,
                    'trap_weights': self.trap_weights,
                    'fixed_scatter_proportion': True,
                    # When fix_scatter_proportion is True, set the scatter proportion for the gases below
                    'gas_scatter_proportion': self.scatter_proportion,
                    'partially_fixed_scatter_proportion': False,
                    'fixed_survival_probability': True,
                    'survival_prob': self.survival_prob,
                    'use_radiation_loss': self.use_radiation_loss,
                    'sample_ins_res_errors': self.sample_ins_resolution_errors,
		    # Reviewing this code, I (Talia) see that this 'resolution_function' initialization does nothing, here.
		    # In the MultiGasComplexLineShape processor, the 'resolution_function' variable is decides which function
		    # is used when *fitting* Kr data. However, no Kr data is fitted in this FakeDataGenerator, and instead,
		    # 'make_spectrum' functions from the MultiGasComplexLineShape processor are used.
		    # (See mermithid.misc.FakeTritiumDataFunctions, which is imported above.) Which make_spectrum function
		    # is called is *not* decided by line 263 (just below); it's instead decided by line 439 in this file.
                    'resolution_function': self.resolution_function,
                    'scatter_peak_ratio_p': self.scatter_peak_ratio_p,
                    'scatter_peak_ratio_q': self.scatter_peak_ratio_q,
					'factor': self.scatter_peak_ratio_factor,
                    'fit_recon_eff': self.fit_recon_eff,

                    #For analytic resolution functions, only:
                    'ratio_gamma_to_sigma': self.ratio_gamma_to_sigma,
                    'gaussian_proportion': self.gaussian_proportion,
                    'A_array': self.A_array,
                    'sigma_array': self.sigma_array,

                    # This is an important parameter which determines how finely resolved the scatter calculations are. 10000 seems to produce a stable fit with minimal slowdown, for ~4000 fake events. The parameter may need to be increased for larger datasets.
                    'num_points_in_std_array': 10000,#35846,
                    'base_shape': 'dirac',
                    'path_to_osc_strengths_files': self.path_to_osc_strengths_files,
                    'path_to_scatter_spectra_file':self.path_to_scatter_spectra_file,
                    'rad_loss_path': self.rad_loss_path,
                    'path_to_ins_resolution_data_txt': self.path_to_ins_resolution_data_txt,
                    'use_combined_four_trap_inst_reso': self.use_combined_four_trap_inst_reso,
                    'path_to_four_trap_ins_resolution_data_txt': self.path_to_four_trap_ins_resolution_data_txt,
                    'shake_spectrum_parameters_json_path': self.shake_spectrum_parameters_json_path
                }
                logger.info('Setting up complex lineshape object')
                self.complexLineShape = MultiGasComplexLineShape("complexLineShape")
                logger.info('Configuring complex lineshape')
                self.complexLineShape.Configure(complexLineShape_config)
                logger.info('Checking existence of scatter spectra files')
                self.complexLineShape.check_existence_of_scatter_file()
                lineshape_array = self.complexLineShape.std_eV_array()
                self.lineshape_stepize = lineshape_array[1]-lineshape_array[0]
            else:
                logger.error("'detailed_or_simplified' is neither 'detailed' nor 'simplified'")
                return False

        # The detector response can altneratively be a simple gaussian.
        else:
            self.lineshape = 'gaussian'
            self.ls_params = [self.scattering_sigma]
            logger.info('Lineshape is Gaussian')

        # check final states file existence
        if self.molecular_final_states:
            logger.info('Going to use molecular final states from Bodine et al 2015')
            with open(self.final_states_file, 'r') as infile:
                a = json.load(infile)
                index = np.where(np.array(a['Probability'])[:-1]>0)
                self.final_state_array = [np.array(a['Binding energy'])[index], np.array(a['Probability'])[index]]
        else:
            logger.info('Not using molecular final state spectrum')
            self.final_state_array = np.array([[0.], [1.]])

        return True



    def InternalRun(self):

        if self.return_frequency:
            if self.maxf == None:
                ROIbound = [self.minf]
            else:
                ROIbound = [self.minf, self.maxf]
        else:
            ROIbound = [self.Kmin, self.Kmax]

        Kgen = self.generate_unbinned_data(self.Q, self.m,
                                           ROIbound,
                                           self.S, self.B_1kev,
                                           nsteps=self.n_steps,
                                           lineshape=self.lineshape,
                                           params=self.ls_params,
                                           efficiency_dict = self.efficiency_dict,
                                           err_from_B=self.err_from_B,
                                           B_field=self.B_field)


        self.results = Kgen

        if self.return_frequency:
            self.results['F'] = Frequency(Kgen['K'], self.B_field)
        return True



    def load_simp_params(self, sigma, survival_prob, Nscatters=20):
        f = open(self.simplified_scattering_path, "r")
        num = []
        pi = [[] for i in range(4)]
        pi_err = [[] for i in range(4)]
        for line in f.readlines():
            elements = line.strip().split(" ")
            for j in range(9):
                if j==0:
                    num.append(float(elements[j]))
                elif j%2==1:
                    pi[int((j-1)/2.)].append(float(elements[j]))
                else:
                    pi_err[int((j-2)/2.)].append(float(elements[j]))
        f.close()

        #Sample scattering params p0--p3 from normal distributions with stdevs equal to the corresponding uncertainties
        for i,_ in enumerate(pi):
            for j in range(Nscatters):
                pi[i][j] = random.gauss(pi[i][j], pi_err[i][j])
        simp_params=[sigma*2*math.sqrt(2*math.log(2)), survival_prob] + [p[:Nscatters] for p in pi]
        return simp_params


    def load_efficiency_curve(self):
        # Efficiency dictionary
        with open(self.efficiency_path, 'r') as infile:
            eff_dict = json.load(infile)

        return eff_dict


    def generate_unbinned_data(self, Q_mean, mass, ROIbound, S, B_1kev, nsteps=10**4,
                               lineshape='gaussian', params=[0.5], efficiency_dict=None,
                                array_method=True, err_from_B=None, B_field=None):
        """
        Returns list of event kinetic energies (eV).
        The 'nsteps' parameter can be increased to reduce the number of repeated energies in the returned list, improving the reliability of the sampling method employed here.
        The 'lineshape' parameter, specifying the probability distribution that is convolved with the tritium spectrum, can have the following options:
        - 'gaussian'
        - 'simplified': Central gaussian + approximated scattering
        - 'detailed': Central gaussian + detailed scattering
        'params' is a list of the params inputted into the lineshape function. If such a parameter exists, the first entry of the list should be a standard deviation of full width half max that provides the scale of the lineshape width.
        """
        logger.info('Going to generate pseudo-unbinned data with {} lineshape'.format(lineshape))

        if self.return_frequency:
            minf = ROIbound[0]
            if len(ROIbound)==2:
                maxf = ROIbound[1]
            else:
                if efficiency_dict is not None:
                    maxf = max(efficiency_dict['frequencies'])
                else:
                    maxf = max(self.load_efficiency_curve()['frequencies'])
            Kmax, Kmin = Energy(minf, B_field), Energy(maxf, B_field)
        else:
            Kmin, Kmax = ROIbound[0], ROIbound[1]
        B = B_1kev*(Kmax-Kmin)/1000.

        nstdevs = 7 #Number of standard deviations (of size broadening) below Kmin and above Q-m to generate data, for the gaussian case
        FWHM_convert = 2*math.sqrt(2*math.log(2))
        max_energy = -self.min_energy
        min_energy = self.min_energy

        Kmax_eff = Kmax+max_energy #Maximum energy for data is slightly above Kmax>Q-m
        Kmin_eff = Kmin+min_energy #Minimum is slightly below Kmin<Q-m

        if not nsteps > 0:
            raise ValueError('n_steps is not greater zero')
        step_size = (Kmax_eff-Kmin_eff)/float(nsteps)
        logger.info('Stepsize is {} eV'.format(step_size))

        #Options of kinetic energies to be sampled
        self.Koptions = np.arange(Kmin_eff, Kmax_eff, self.lineshape_stepize)

        if efficiency_dict is not None:
            logger.info('Evaluating efficiencies')
            efficiency = random_efficiency_from_interpolation(self.Koptions, efficiency_dict, B_field)
            eff_negative = (efficiency<0.)
            efficiency[eff_negative] = 0. #Whenever this occurs, efficiency_mean=0 and efficiency_error=1
        else:
            efficiency, _ = 1, 0

        #Create array of sampled kinetic energies.
        logger.info('Going to calculate rates')
        time0 = time.time()

        if array_method == True:
            ratesS = convolved_spectral_rate_arrays(self.Koptions, Q_mean,
            mass, Kmin, lineshape, params, self.scatter_peak_ratio_p, self.scatter_peak_ratio_q, self.scatter_proportion, min_energy, max_energy,
            self.complexLineShape, self.final_state_array, self.resolution_function, self.ins_res_width_bounds, self.ins_res_width_factors, self.p_factors, self.q_factors)
        else:
            ratesS = [convolved_spectral_rate(K, Q_mean, mass, Kmin,
                lineshape, params, min_energy, max_energy) for K in
                self.Koptions]

        # multiply rates by efficiency
        #if self.ins_res_width_bounds==None:
        ratesS = ratesS*efficiency
        #else:
        #ratesS = [r*efficiency for r in ratesS]

        time1 = time.time()
        logger.info('... signal rate took {} s'.format(time1-time0))

        # background
        if array_method == True:
            ratesB = convolved_bkgd_rate_arrays(self.Koptions, Kmin, Kmax,
                                                lineshape, params, self.scatter_peak_ratio_p, self.scatter_peak_ratio_q, self.scatter_proportion, min_energy, max_energy,
                                                self.complexLineShape, self.resolution_function)
        else:
            ratesB = [convolved_bkgd_rate(K, Kmin, Kmax, lineshape, params,
            min_energy, max_energy) for K in self.Koptions]

        time2 = time.time()
        logger.info('... background rate took {} s'.format(time2 - time1))

        if err_from_B != None and err_from_B != 0.:
            dE = self.Koptions[1] - self.Koptions[0]
            #n_dE = round(max_energy/dE) #Number of steps for the narrow gaussian for energies > 0
            n_dE = 10*err_from_B/dE

            K_lineshape =  np.arange(-n_dE*dE, n_dE*dE, dE)
            #Generating finely spaced points on a gaussian
            gaussian_rates = gaussian(K_lineshape, [err_from_B, 0])

            ratesB = convolve(ratesB, gaussian_rates, mode='same')
            #if self.ins_res_width_bounds==None:
            ratesS = convolve(ratesS, gaussian_rates, mode='same')

        ratesS[ratesS<0.] = 0.
        rate_sumS = np.sum(ratesS)
        probsS = np.array(ratesS)/rate_sumS

        ratesB[ratesB<0.] = 0.
        rate_sumB = np.sum(ratesB)
        probsB = np.array(ratesB)/rate_sumB

	    #Calculate three different rates variables, for each of the three runtimes
        runtime_ratios = [t/float(self.channel_runtimes[0]) for t in self.channel_runtimes]

        logger.info('Generating data')
        time4 = time.time()

	    #Break up self.Koptions into three different arrays.
	    #Then, sample KE variables for each of the arrays and appropriate elements of self.channel_runtimes, probsS and probsB.
        #Finally, concatenate together the three KE arrays.
        #if self.ins_res_width_bounds==None:
        temp_Koptions, temp_probsS, temp_probsB = self.Koptions, probsS, probsB
        split_Koptions, split_probsS, split_probsB = [], [], []
        print(len(temp_Koptions), len(temp_probsS))
        for i in range(len(self.channel_bounds)):
            split_Koptions.append(temp_Koptions[Frequency(temp_Koptions, self.B_field)<=self.channel_bounds[i]])
            split_probsS.append(temp_probsS[Frequency(temp_Koptions, self.B_field)<=self.channel_bounds[i]])
            split_probsB.append(temp_probsB[Frequency(temp_Koptions, self.B_field)<=self.channel_bounds[i]])
            temp_probsS = temp_probsS[Frequency(temp_Koptions, self.B_field)>self.channel_bounds[i]]
            temp_probsB = temp_probsB[Frequency(temp_Koptions, self.B_field)>self.channel_bounds[i]]
            temp_Koptions = temp_Koptions[Frequency(temp_Koptions, self.B_field)>self.channel_bounds[i]]

        split_Koptions.append(temp_Koptions)
        split_probsS.append(temp_probsS)
        split_probsB.append(temp_probsB)

        rates = []
        for i in range(len(self.channel_runtimes)):
            rates.append((S*runtime_ratios[i]*split_probsS[i] + B*split_probsB[i])/(S*runtime_ratios[i]+B))

        self.Koptions = np.concatenate(split_Koptions)
        rates = np.concatenate(rates)
        self.probs = rates/np.sum(rates)

        if self.poisson_stats:
            KE = np.random.choice(self.Koptions, np.random.poisson(S+B), p = self.probs)
        else:
            KE = np.random.choice(self.Koptions, round(S+B), p = self.probs)

        time5 = time.time()

        logger.info('... took {} s'.format(time5-time4))
        logger.info('Number of values in array that are not unique: {} out of {}'.format(np.size(KE) - len(set(KE)), np.size(KE)))

        if self.return_frequency:
            return {'K':KE, 'maxf':maxf, 'minf':minf}
        else:
            return {'K':KE}
