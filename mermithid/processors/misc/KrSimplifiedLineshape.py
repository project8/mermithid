
"""
Author: C. Claessens
Date: 4/15/2023
Description:
    Processor for fitting with simplified lineshape model


"""

import json
import os
from copy import deepcopy
import numpy as np
from scipy import constants
from scipy.special import erfc
from scipy.signal import convolve
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


from morpho.utilities import morphologging, reader
logger = morphologging.getLogger(__name__)

from mermithid.processors.Fitters import BinnedDataFitter
from mermithid.misc.FakeTritiumDataFunctions import fermi_func, pe, Ee, GF, Vud, Mnuc2
from mermithid.misc import ComplexLineShapeUtilities, Constants
from mermithid.processors.misc.KrComplexLineShape import KrComplexLineShape
#from mermithid.misc.DetectionEfficiencyUtilities import pseudo_integrated_efficiency, integrated_efficiency, power_efficiency





# =============================================================================
# Processor definition
# =============================================================================

class KrSimplifiedLineshape(BinnedDataFitter):


    def InternalConfigure(self, config_dict):

        # ====================================
        # Model configuration
        # ====================================
        self.is_smeared = reader.read_param(config_dict, 'smeared', False) # convolve with resolution
        self.is_scattered = reader.read_param(config_dict, 'scattered', False) # convolve with tail
        self.integrate_bins = reader.read_param(config_dict, 'integrate_bins', True) # integrate spectrum over bin widths
        self.base_shape = reader.read_param(config_dict, 'base_shape', 'shake')
        self.shake_spectrum_parameters_json_path = reader.read_param(config_dict, 'shake_spectrum_parameters_json_path', 'shake_spectrum_parameters.json')
        self.path_to_osc_strengths_files = reader.read_param(config_dict, 'path_to_osc_strengths_files', '/host/')

        logger.info("Base shape is: {}".format(self.base_shape))

        if not os.path.exists(self.shake_spectrum_parameters_json_path) and self.base_shape=='shake':
            raise IOError('Shake spectrum path does not exist')
        if not os.path.exists(self.path_to_osc_strengths_files):
            raise IOError('Path to osc strengths files does not exist')
        
        
        # save plots in 
        self.savepath = reader.read_param(config_dict, 'savepath', '.')

        # detector response options
        self.NScatters = reader.read_param(config_dict, 'NScatters', 20)
        self.resolution_model = reader.read_param(config_dict, 'resolution_model', 'gaussian')
        self.simplified_lineshape_path = reader.read_param(config_dict, 'simplified_lineshape_path', "required")
        self.helium_lineshape_path = reader.read_param(config_dict, 'helium_lineshape_path', "optional")
        self.use_helium_scattering = reader.read_param(config_dict, 'use_helium_scattering', False)
        self.derived_two_gaussian_model = reader.read_param(config_dict, 'derived_two_gaussian_model', True)


        # configure model parameter names
        self.model_parameter_names = reader.read_param(config_dict, 'model_parameter_names',
                                                       ['B', 'mu', 'resolution', 
                                                        'scatter_peak_ratio_p', 'scatter_peak_ratio_q', 
                                                        'h2_fraction',
                                                        'background', 'amplitude'] )
        # initial values and mean of constaints (if constraint) or mean of distribution (if sampled)
        self.model_parameter_means = reader.read_param(config_dict, 'model_parameter_means', [0.9591, 17.8e3, 10, 1, 1, 0.5, 1, 1e3])
        # width for constraints or sample distributions
        self.model_parameter_widths = reader.read_param(config_dict, 'model_parameter_widths', [0.0001, 1, 1, 0.1, 0.1, 0.1, 0.1, 10])
        self.fixed_parameters = reader.read_param(config_dict, 'fixed_parameters', [False, False, False, False, False, False, False, False])
        self.fixed_parameter_dict = reader.read_param(config_dict, 'fixed_parameter_dict', {})
        self.free_in_second_fit = reader.read_param(config_dict, 'free_in_second_fit', [])
        self.fixed_in_second_fit = reader.read_param(config_dict, 'fixed_in_second_fit', [])
        self.limits = reader.read_param(config_dict, 'model_parameter_limits',
                                                    [[None, None],
                                                     [16e3, 19e3],
                                                     [0, None],
                                                     [None, None],
                                                     [None, None],
                                                     [0,1],
                                                     [0, None],
                                                     [0, None]
                                                     ])
        self.error_def = reader.read_param(config_dict,'error_def', 0.5) #0.5 is max ll fit, 1 is chi square

        # check that configuration is consistent
        if (len(self.model_parameter_names) != len(self.model_parameter_means) or len(self.model_parameter_names) != len(self.model_parameter_widths) or len(self.model_parameter_names) != len(self.fixed_parameters)):
            logger.error('Number of parameter names does not match other parameter configurations')
            return False


        # ====================================
        # Parameter configuration
        # ====================================

        self.parameter_samples = {}

        # need to know which parameter is background
        self.background_index = self.model_parameter_names.index('background')
        # and which one is B
        # self.B_index = self.model_parameter_names.index('B')
        # signal counts
        self.amplitude_index = self.model_parameter_names.index('amplitude')


        # resolutions

        self.scale_mean = reader.read_param(config_dict, 'scale_mean', 1)
        self.scale_width = reader.read_param(config_dict, 'scale_width', 0)
        self.width_scaling = self.scale_mean

        if self.is_smeared:
            if 'resolution' in self.model_parameter_names:
                self.res_index = self.model_parameter_names.index('resolution')

            self.res_mean = reader.read_param(config_dict, 'gaussian_resolution_mean', 15.0)
            self.res_width = reader.read_param(config_dict, 'gaussian_resolution_width', 1.0)
            self.res_width_from_maxSNR = reader.read_param(config_dict, 'sigma_std_maxSNR', 0)
            self.res = self.res_mean

            if self.resolution_model != 'gaussian':
                self.two_gaussian_fraction = reader.read_param(config_dict, 'two_gaussian_fraction', 1.)
                self.two_gaussian_mu_1 = reader.read_param(config_dict, 'two_gaussian_mu1', 0)
                self.two_gaussian_mu_2 = reader.read_param(config_dict, 'two_gaussian_mu2', 0)
                if 'two_gaussian_mu_1' in self.model_parameter_names:
                        self.two_gaussian_mu_1_index = self.model_parameter_names.index('two_gaussian_mu_1')
                if 'two_gaussian_mu_2' in self.model_parameter_names:
                        self.two_gaussian_mu_2_index = self.model_parameter_names.index('two_gaussian_mu_2')

                if self.derived_two_gaussian_model:
                    self.two_gaussian_p0 = reader.read_param(config_dict, 'two_gaussian_p0', 1.)
                    self.two_gaussian_p1 = reader.read_param(config_dict, 'two_gaussian_p1', 1.)
                else:
                    if 'two_gaussian_sigma_1' in self.model_parameter_names:
                        self.two_gaussian_sigma_1_index = self.model_parameter_names.index('two_gaussian_sigma_1')
                    if 'two_gaussian_sigma_2' in self.model_parameter_names:
                        self.two_gaussian_sigma_2_index = self.model_parameter_names.index('two_gaussian_sigma_2')


                    self.two_gaussian_sigma_1_mean = reader.read_param(config_dict, 'two_gaussian_sigma_1_mean', 15)
                    self.two_gaussian_sigma_2_mean = reader.read_param(config_dict, 'two_gaussian_sigma_2_mean', 5)
                    self.two_gaussian_sigma_1_width = reader.read_param(config_dict, 'two_gaussian_sigma_1_width', 1)
                    self.two_gaussian_sigma_2_width = reader.read_param(config_dict, 'two_gaussian_sigma_2_width', 1)
                    # initial setting
                    self.two_gaussian_sigma_1 = deepcopy(self.two_gaussian_sigma_1_mean)
                    self.two_gaussian_sigma_2 = deepcopy(self.two_gaussian_sigma_2_mean)




        # scatter peak ratio
        if self.is_scattered:
            if 'scatter_peak_ratio_p' in self.model_parameter_names:
                self.scatter_peak_ratio_p_index = self.model_parameter_names.index('scatter_peak_ratio_p')
            if 'scatter_peak_ratio_q' in self.model_parameter_names:
                self.scatter_peak_ratio_q_index = self.model_parameter_names.index('scatter_peak_ratio_q')
            if 'h2_fraction' in self.model_parameter_names:
                self.h2_fraction_index = self.model_parameter_names.index('h2_fraction')

            self.spr_factor = reader.read_param(config_dict, 'SPR_factor', 0)
            self.scatter_peak_ratio_p_mean = reader.read_param(config_dict, 'scatter_peak_ratio_p_mean', 0.7)
            self.scatter_peak_ratio_p_width = reader.read_param(config_dict, 'scatter_peak_ratio_p_width', 0.1)
            self.scatter_peak_ratio_q_mean = reader.read_param(config_dict, 'scatter_peak_ratio_q_mean', 0.7)
            self.scatter_peak_ratio_q_width = reader.read_param(config_dict, 'scatter_peak_ratio_q_width', 0.1)
            self.h2_fraction_mean = reader.read_param(config_dict, 'h2_fraction_mean', 1.0)
            self.h2_fraction_width = reader.read_param(config_dict, 'h2_fraction_width', 0.0)

            self.scatter_peak_ratio_mean = reader.read_param(config_dict, 'scatter_peak_ratio_mean', 0.5)
            self.scatter_peak_ratio_width = reader.read_param(config_dict, 'scatter_peak_ratio_width', 0.1)

            self.scatter_peak_ratio_p = self.scatter_peak_ratio_p_mean
            self.scatter_peak_ratio_q = self.scatter_peak_ratio_q_mean
            self.h2_fraction = self.h2_fraction_mean



        # identify tritium parameters
        #self.kr_model_indices = reader.read_param(config_dict, 'tritium_model_parameters', [0, 2])
        
        if 'B' in self.model_parameter_names:
            self.B_index = self.model_parameter_names.index('B')
        

        # individual model parameter configurations
        # overwrites model_parameter_means and widths
        self.B_mean = reader.read_param(config_dict, 'B_mean', 1)
        self.B_width = reader.read_param(config_dict, 'B_width', 1e-6)
        self.B = self.B_mean

        # frequency an energy range
        self.min_frequency = reader.read_param(config_dict, 'min_frequency', "required")
        self.max_frequency = reader.read_param(config_dict, 'max_frequency', None)
        self.max_energy = reader.read_param(config_dict, 'resolution_energy_max', 1000)
        

        # path to json with efficiency dictionary
        self.efficiency_file_path = reader.read_param(config_dict, 'efficiency_file_path', '')



        # =================
        # Fit configuration
        # =================

        self.print_level = reader.read_param(config_dict, 'print_level', 0)
        self.use_asimov = False



        # Parameters can be constrained manually or by inlcuding them in the nuisance parameter dictionary
        #self.constrained_parameter_names = reader.read_param(config_dict, 'constrained_parameter_names', [])
        self.constrained_parameters = reader.read_param(config_dict, 'constrained_parameters', [])
        self.constrained_means = np.array(self.model_parameter_means)[self.constrained_parameters]#reader.read_param(config_dict, 'constrained_means', [])
        self.constrained_widths = np.array(self.model_parameter_widths)[self.constrained_parameters]#reader.read_param(config_dict, 'constrained_widths', [])


        self.nuisance_parameters = reader.read_param(config_dict, 'nuisance_parameters', {})

        for i, p in enumerate(self.model_parameter_names):
            if p in self.nuisance_parameters.keys():
                if self.nuisance_parameters[p]:
                    self.constrained_parameters.append(i)
                    self.constrained_means = np.append(self.constrained_means, self.model_parameter_means[i])
                    self.constrained_widths = np.append(self.constrained_widths, self.model_parameter_widths[i])
            if i in self.constrained_parameters:
                self.fixed_parameters[i] = False

        #enforcing correlations
        self.correlated_parameters = []



        # MC uncertainty propagation does not need the fit uncertainties returned by iminuit. uncertainties are instead obtained from the distribution of fit results.
        # But if the uncertainty is instead propagated by adding constraiend nuisance parameters then the fit uncertainties are needed.
        # imnuit can calculated hesse and minos intervals. the former are symmetric. we want the asymetric intrevals-> minos
        # minos_cls is the list of uncertainty level that should be obtained: e.g. [0.68, 0.9]
        self.minos_intervals = reader.read_param(config_dict, 'minos_intervals', False)
        self.minos_cls = reader.read_param(config_dict, 'minos_confidence_levels', [0.683])


        # ==================================
        # lineshape configuration
        # ==================================

        # simplified lineshape parameters
        self.lineshape_p = np.loadtxt(self.simplified_lineshape_path, unpack=True)

        # if true lineshape is plotted during tritium spectrum shape generation (for debugging)
        self.plot_lineshape = False

        # helium lineshape
        if self.use_helium_scattering:
            self.helium_lineshape_p = np.loadtxt(self.helium_lineshape_path, unpack=True)

        # which lineshape should be used?
        self.multi_gas_lineshape = self.simplified_multi_gas_lineshape
        


        # scatter peak ratio
        self.scatter_peak_ratio = reader.read_param(config_dict, 'scatter_peak_ratio', 'modified_exponential')

        if self.scatter_peak_ratio == 'constant':
            self.use_fixed_scatter_peak_ratio = True
        elif self.scatter_peak_ratio == 'modified_exponential':
            self.use_fixed_scatter_peak_ratio = False

        else:
            logger.error("Configuration of scatter_peak_ratio not known. Options are 'constant' and 'modified_exponential")
            raise ValueError("Configuration of scatter_peak_ratio not known. Options are 'constant' and 'efficiency_model")




        # ==================================
        # energies and bins
        # ==================================

        # bin width used for poisson statistics fit
        # energy stepsize is used for integration approximation:
        # counts in a bin are integrated spectrum over bin width.
        # integration is approximated by summing over a number of discrete steps in a bin.
        self.dbins = reader.read_param(config_dict, 'energy_bin_width', 50)
        self.denergy = reader.read_param(config_dict, 'energy_step_size', min([np.round(self.dbins/10, 2), 1]))

        # bin size has to divide energy step size
        self.N_bins = np.round((self.Energy(self.min_frequency)-self.Energy(self.max_frequency))/self.dbins)
        self.N_energy_bins = self.N_bins*np.round(self.dbins/self.denergy)

        # adjust energy stepsize to match bin division
        self.denergy = self.dbins/np.round(self.dbins/self.denergy)


        self.energies = np.arange(self.Energy(self.max_frequency), self.Energy(self.max_frequency)+(self.N_energy_bins)*self.denergy, self.denergy)
        self.bins = np.arange(np.min(self.energies), self.Energy(self.max_frequency)+(self.N_bins)*self.dbins, self.dbins)

        if len(self.energies) > self.N_energy_bins:
            self.energies = self.energies[:-1]
        if len(self.bins) > self.N_bins:
            self.bins = self.bins[:-1]

        self.bin_centers = self.bins[0:-1]+0.5*(self.bins[1]-self.bins[0])
        
        self.frequencies = self.Frequency(self.energies)
        self.freq_bins = self.Frequency(self.bins)
        self.freq_bin_centers = self.Frequency(self.bin_centers)
        
        self.freq_bins = np.sort(self.freq_bins)
        self.freq_bin_centers = np.sort(self.freq_bin_centers)
        self.frequencies = np.sort(self.frequencies)

        self.ReSetBins()
        
        # ==================================
        # Read shake parameters from JSON file
        # ==================================
        
        if self.base_shape == 'shake':
            x_array = self.flip_array(self.std_eV_array())
            self.shakeSpectrumClassInstance = ComplexLineShapeUtilities.ShakeSpectrumClass(self.shake_spectrum_parameters_json_path, x_array)

        # ==================================
        # configure parent BinnedDataFitter
        # ==================================

        # This processor inherits from the BinnedDataFitter that does binned max likelihood fitting
        # overwrite parent model with tritium model used here
        self.model = self.SpectrumWithBackground

        # now configure fit
        self.ConfigureFit()


        return True


    def ConfigureFit(self):
        # configure fit

        # this should not be hard coded
        #neutrino_limits = [-400**2, 400**2]
        #energy_limits = [max([self.endpoint-1000, np.min(self.energies)+np.sqrt(neutrino_limits[1])]), min([self.endpoint+1000, np.max(self.energies)-np.sqrt(np.abs(neutrino_limits[0]))])]
        #if self.print_level == 1:
        #    logger.warning('Neutrino mass limited to: {} - {}'.format(*np.sqrt(np.array(neutrino_limits))))


        #if not self.fit_efficiency_tilt:
        self.parameter_names = self.model_parameter_names 
        self.initial_values = self.model_parameter_means
        self.fixes = self.fixed_parameters 
        self.fixes_dict = self.fixed_parameter_dict

        self.parameter_errors = self.model_parameter_widths

        return True

    # =========================================================================
    # Main fit function
    # =========================================================================
    def std_eV_array(self):
        emin = -1000
        emax = 2000
        array = np.arange(emin,emax,self.denergy)
        return array
    
    def InternalRun(self, sampled_parameters={}, params=[]):
        print("Running")
        
        """plt.figure()
        plt.plot(self.std_eV_array(), self.shakeSpectrumClassInstance.shake_spectrum())
        plt.xlabel("Energy (eV)")
        plt.ylabel("Amplitude (arb. units")
        plt.tight_layout()
        plt.savefig(os.path.join(self.savepath, "shake_spectrum.png"))"""

        # for systematic MC uncertainty propagation: Generate Asimov data to fit with random model
        if self.use_asimov:
            self.hist = self.SpectrumWithBackground(self.bin_centers, *params)
          
        # sample priors that are to be sampled
        if len(sampled_parameters.keys()) > 0:
            self.SamplePriors(sampled_parameters)


            # need to first re-calcualte energy bins with sampled B before getting efficiency
            self.ReSetBins()

            
        # now convert frequency data to energy data and histogram it
        self.ConvertAndHistogram()

        # Call parent fit method using data and model from this instance
        self.results, self.errors = self.fit()

        return True


    # =========================================================================
    # Get random sample from normal, beta, or gamma distribution
    # =========================================================================
    def Gaussian_sample(self, mean, width):
        np.random.seed()
        if isinstance(width, list):
            return np.random.randn(len(width))*width+mean
        else:
            return np.random.randn()*width+mean

    def Beta_sample(self, mean, width):
        np.random.seed()
        a = ((1-mean)/(width**2)-1/mean)*mean**2
        b = (1/mean-1)*a
        return np.random.beta(a, b)

    def Gamma_sample(self, mean, width):
        np.random.seed()
        a = (mean/width)**2
        b = mean/(width**2)
        return np.random.gamma(a, 1/b)


    # def SamplePriors(self, sampled_parameters):

    #     for k in sampled_parameters.keys():
    #         if k in self.nuisance_parameters and self.nuisance_parameters[k] and sampled_parameters[k]:
    #             raise ValueError('{} is nuisance parameter.'.format(k))

    #     for i, p in enumerate(self.model_parameter_names):
    #         if p in sampled_parameters.keys() and sampled_parameters[p] and not self.fixed_parameters[i]:
    #             raise ValueError('{} is a free parameter'.format(p))

    #     logger.info('Sampling: {}'.format([k for k in sampled_parameters.keys() if sampled_parameters[k]]))
    #     self.parameter_samples = {}
    #     sample_values = []
    #     if 'resolution' in sampled_parameters.keys() and sampled_parameters['resolution']:
    #         self.res = self.Gaussian_sample(self.res_mean, self.res_width)
    #         #if self.res <= 30.01/float(2*np.sqrt(2*np.log(2))):
    #         #    logger.warning('Sampled resolution small. Setting to {}'.format(30.01/float(2*np.sqrt(2*np.log(2)))))
    #         #    self.res = 30.01/float(2*np.sqrt(2*np.log(2)))
    #         self.parameter_samples['resolution'] = self.res
    #         sample_values.append(self.res)
    #     if 'h2_fraction' in sampled_parameters.keys() and sampled_parameters['h2_fraction']:
    #         self.h2_fraction = self.Gaussian_sample(self.h2_fraction_mean, self.h2_fraction_width)
    #         if self.h2_fraction > 1: self.h2_fraction=1
    #         elif self.h2_fraction < 0: self.h2_fraction=0
    #         self.parameter_samples['h2_fraction'] = self.h2_fraction
    #         sample_values.append(self.h2_fraction)
    #     if 'two_gaussian_sigma_1' in sampled_parameters.keys() and sampled_parameters['two_gaussian_sigma_1']:
    #         self.two_gaussian_sigma_1 = self.Gaussian_sample(self.two_gaussian_sigma_1_mean, self.two_gaussian_sigma_1_width)
    #         self.parameter_samples['two_gaussian_sigma_1'] = self.two_gaussian_sigma_1
    #         sample_values.append(self.two_gaussian_sigma_1)
    #     if 'two_gaussian_sigma_2' in sampled_parameters.keys() and sampled_parameters['two_gaussian_sigma_2']:
    #         self.two_gaussian_sigma_2 = self.Gaussian_sample(self.two_gaussian_sigma_2_mean, self.two_gaussian_sigma_2_width)
    #         self.parameter_samples['two_gaussian_sigma_2'] = self.two_gaussian_sigma_2
    #         sample_values.append(self.two_gaussian_sigma_2)
    #     if 'scatter_peak_ratio' in sampled_parameters.keys() and sampled_parameters['scatter_peak_ratio']:
    #         self.scatter_peak_ratio_p = self.Beta_sample(self.scatter_peak_ratio_mean, self.scatter_peak_ratio_width)
    #         self.scatter_peak_ratio_q = 1
    #         #self.fix_scatter_ratio_b = True
    #         #self.fix_scatter_ratio_c = True
    #         self.parameter_samples['scatter_peak_ratio'] = self.scatter_peak_ratio_p
    #         sample_values.append(self.scatter_peak_ratio_p)

    #     if self.correlated_p_q and 'scatter_peak_ratio_p' in sampled_parameters.keys() and 'scatter_peak_ratio_q' in sampled_parameters.keys() and sampled_parameters['scatter_peak_ratio_p'] and sampled_parameters['scatter_peak_ratio_q']:
    #         logger.info('Correlated p, q, scale sampling')
    #         correlated_vars = np.random.multivariate_normal([self.scatter_peak_ratio_p_mean, self.scatter_peak_ratio_q_mean], self.p_q_cov_matrix)
    #         self.scatter_peak_ratio_p = correlated_vars[0]
    #         self.scatter_peak_ratio_q = correlated_vars[1]
    #         #self.width_scaling = correlated_vars[2]

    #         #self.fix_scatter_ratio_b = True
    #         self.parameter_samples['scatter_peak_ratio_p'] = self.scatter_peak_ratio_p
    #         sample_values.append(self.scatter_peak_ratio_p)

    #         self.parameter_samples['scatter_peak_ratio_q'] = self.scatter_peak_ratio_q
    #         sample_values.append(self.scatter_peak_ratio_q)
    #         #self.fix_scatter_ratio_c = True

    #     else:

    #         if 'scatter_peak_ratio_p' in sampled_parameters.keys() and sampled_parameters['scatter_peak_ratio_p']:
    #             logger.info('Uncorrelated b, c, scale sampling')
    #             self.scatter_peak_ratio_p = self.Gamma_sample(self.scatter_peak_ratio_p_mean, self.scatter_peak_ratio_p_width)
    #             #self.fix_scatter_ratio_b = True
    #             self.parameter_samples['scatter_peak_ratio_p'] = self.scatter_peak_ratio_p
    #             sample_values.append(self.scatter_peak_ratio_p)
    #         if 'scatter_peak_ratio_q' in sampled_parameters.keys() and sampled_parameters['scatter_peak_ratio_q']:
    #             self.scatter_peak_ratio_q = self.Gamma_sample(self.scatter_peak_ratio_q_mean, self.scatter_peak_ratio_q_width)
    #             self.parameter_samples['scatter_peak_ratio_q'] = self.scatter_peak_ratio_q
    #             sample_values.append(self.scatter_peak_ratio_q)
    #             #self.fix_scatter_ratio_c = True

    #     if 'B' in sampled_parameters.keys() and sampled_parameters['B']:
    #         self.B = self.Gaussian_sample(self.B_mean, self.B_width)
    #         self.parameter_samples['B'] = self.B
    #         sample_values.append(self.B)
    #         logger.info('B field prior: {} +/- {}'.format(self.B_mean, self.B_width))


    #     logger.info('Samples are: {}'.format(sample_values))
    #     #logger.info('Fit parameters: \n{}\nFixed: {}'.format(self.parameter_names, self.fixes))
    #     # set new values in model
    #     self.ConfigureFit()

    #     return self.parameter_samples

    # =========================================================================
    # Frequency - Energy conversion
    # =========================================================================

    def Energy(self, f, mixfreq=0.):
        """
        converts frequency in Hz to energy in eV
        """
        emass = constants.electron_mass/constants.e*constants.c**2
        gamma = (constants.e*self.B)/(2.0*np.pi*constants.electron_mass) * 1./(np.array(f)+mixfreq)

        return (gamma -1.)*emass


    def Frequency(self, E, Theta=None):
        """
        converts energy in eV to frequency in Hz
        """
        if Theta==None:
            Theta=np.pi/2

        emass = constants.electron_mass/constants.e*constants.c**2
        gamma = E/(emass)+1.

        return (constants.e*self.B)/(2.0*np.pi*constants.electron_mass) * 1./gamma



    # =========================================================================
    # Bin and energy array set&get
    # =========================================================================

 
    def ReSetBins(self):
        #self.energies = np.arange(self.Energy(self.max_frequency), self.Energy(self.min_frequency), self.denergy)
        #self.bins = np.arange(np.min(self.energies), np.max(self.energies), self.dbins)

        #self._bin_efficiency, self._bin_efficiency_error = [], []
        # self.energies = np.arange(self.Energy(self.max_frequency), self.Energy(self.max_frequency)+(self.N_energy_bins)*self.denergy, self.denergy)
        # self.bins = np.arange(np.min(self.energies), np.min(self.energies)+(self.N_bins)*self.dbins, self.dbins)
        
        # if len(self._energies) > self.N_energy_bins:
        #     self._energies = self._energies[:-1]
        # if len(self._bins) > self.N_bins:
        #     self._bins = self._bins[:-1]
        
        self.energies = self.Energy(self.frequencies)[::-1]
        self.bins = self.Energy(self.freq_bins)[::-1]
        self.bin_centers = self.Energy(self.freq_bin_centers)[::-1]

        


    # =========================================================================
    # Data generation and histogramming
    # =========================================================================

    def GenerateData(self, params, N):

        x = self.energies[0:-1]+0.5*(self.energies[1]-self.energies[0])
        pdf = np.longdouble(self.model(x, *params))
        pdf[pdf<0]=0.
        np.random.seed()

        pdf = np.float64(np.longdouble(pdf/np.sum(pdf)))

        self.data = np.random.choice(x, np.random.poisson(N), p=pdf/np.sum(pdf))
        self.freq_data = self.Frequency(self.data)

        return self.data, self.freq_data


    def GenerateAsimovData(self, params):

        asimov_data = []
        #x = self.bins #np.arange(min(self.energies), max(self.energies), 25)
        #xc = x[0:-1]+0.5*(x[1]-x[0])
        #pdf = np.round(self.TritiumSpectrumBackground(xc, *params))

        self.use_asimov = True
        #for i, p in enumerate(pdf):
        #    asimov_data.extend(list(itertools.repeat(xc[i], int(p))))

        return np.array(asimov_data), self.Frequency(np.array(asimov_data))



    def ConvertFreqData2EnergyData(self):
        """
        Convert frequencies to energyies with set B
        """
        self.data = self.Energy(self.freq_data)
        return self.data


    # def Histogram(self, weights=None):
    #     """
    #     histogram data using bins
    #     """
    #     h, b = np.histogram(self.data, bins=self.bins, weights=weights)
    #     self.hist = h#float2double(h)
    #     return h


    def ConvertAndHistogram(self, weights=None):
        """
        histogram data using bins
        """
        self.freq_hist, f = np.histogram(self.freq_data, self.freq_bins)
        #self.data = self.ConvertFreqData2EnergyData()
        #self.hist, b = np.histogram(self.data, bins=self.bins, weights=weights)
        self.hist, b = self.freq_hist[::-1], self.Energy(f)[::-1]
        print("Re-converted data")

        # plt.figure()
        # plt.step(self.bin_centers, self.hist)
        # plt.xlabel("Energy (eV)")
        # plt.ylabel("Counts")
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.savepath, "data_histogram.png"))
        #self.hist = float2double(h)

        return self.hist








    # =========================================================================
    # Model functions
    # =========================================================================

    def flip_array(self, array):
        flipped = np.fliplr([array]).flatten()
        return flipped

    def which_model(self, e_spec, *pars):
        if self.base_shape=="shake":
            x_array = self.flip_array(e_spec-Constants.kr_k_line_e())
            self.shakeSpectrumClassInstance.input_std_eV_array = x_array
            return self.shakeSpectrumClassInstance.shake_spectrum()
        else:
            spec = np.zeros(len(e_spec))
            spec[np.where(np.abs(e_spec-Constants.kr_k_line_e())==np.min(np.abs(e_spec-Constants.kr_k_line_e())))] = 1
            return spec
        

    def gauss_resolution_f(self, energy_array, A, sigma, mu):
        f = A*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(((energy_array-mu)/sigma)**2.)/2.)
        return f

    def derived_two_gaussian_resolution(self, energy_array, sigma_s, mu_1, mu_2, A=1):
        sigma_1 = (sigma_s-self.two_gaussian_p0 + self.two_gaussian_fraction * self.two_gaussian_p0)/(self.two_gaussian_fraction + self.two_gaussian_p1 - self.two_gaussian_fraction* self.two_gaussian_p1)
        sigma_2 = self.two_gaussian_p0 + self.two_gaussian_p1 * sigma_1

        lineshape = self.two_gaussian_fraction * self.gauss_resolution_f(energy_array, 1, sigma_1*self.width_scaling, mu_1) + (1 - self.two_gaussian_fraction) * self.gauss_resolution_f(energy_array, 1, sigma_2*self.width_scaling, mu_2)
        return lineshape


    


    def mode_exp_scatter_peak_ratio(self, prob_b, prob_c, j):
        '''
        ratio of successive peaks taking reconstruction efficiency into account
        '''
        c = -self.spr_factor*prob_b + prob_c
        return np.exp(-prob_b*j**c)


    def simplified_ls(self, K, Kcenter, FWHM, prob_b, prob_c=1):
        """
        Simplified lineshape. sum of Gaussians imitating hydrogen only lineshape
        """

        p0, p1, p2, p3 = self.lineshape_p[1], self.lineshape_p[3], self.lineshape_p[5], self.lineshape_p[7]
        sig0 = FWHM/float(2*np.sqrt(2*np.log(2)))
        shape = np.zeros(len(K))#gaussian(K, [sig0, Kcenter])
        norm = 1.

        for i in range(self.NScatters):
            sig = p0[i]+p1[i]*FWHM
            mu = -(p2[i]+p3[i]*np.log(FWHM-30))

            if self.use_fixed_scatter_peak_ratio:
                probi = prob_b**(i+1)
            else:
                probi = self.mode_exp_scatter_peak_ratio(prob_b, prob_c, i+1)

            shape += probi*self.gauss_resolution_f(K, 1, sig, mu+Kcenter)
            norm += probi

        return shape, norm

   

    def simplified_multi_gas_lineshape(self, K, Kcenter, FWHM, prob_b, prob_c=1, h2_fraction=1):
        """
        This uses Gaussians of different mu and sigma for different gases
        """

        if self.plot_lineshape:
            logger.info('Using simplified lineshape. Hydrogen proportion is {}'.format(h2_fraction))

        p0, p1, p2, p3 = self.lineshape_p[1], self.lineshape_p[3], self.lineshape_p[5], self.lineshape_p[7]
        q0, q1, q2, q3 = self.helium_lineshape_p[1], self.helium_lineshape_p[3], self.helium_lineshape_p[5], self.helium_lineshape_p[7]



        sig0 = FWHM/float(2*np.sqrt(2*np.log(2)))
        #if sig0 < 6:
        #    logger.warning('Scatter resolution {} < 6 eV. Setting to 6 eV'.format(sig0))
        #    sig0 = 6
        #shape = self.gauss_resolution_f(K, 1, sig0, Kcenter)
        #shape = np.zeros(len(K))
        norm = 1.

        hydrogen_scattering = np.zeros(len(K))
        helium_scattering = np.zeros(len(K))

        #plt.figure(figsize=(10,10))

        #scatter_peaks = np.array([[self.gauss_resolution_f(K, 1, p2[i]+p3[i]*sig0, -p0[i]+p1[i]*sig0+Kcenter)*self.mode_exp_scatter_peak_ratio(prob_b, prob_c, i+1),
        #                    self.gauss_resolution_f(K, 1, q2[i]+q3[i]*sig0, -q0[i]+q1[i]*sig0+Kcenter)*self.mode_exp_scatter_peak_ratio(prob_b, prob_c, i+1)] for i in range(self.NScatters)])


        for i in range(self.NScatters):

            # hydrogen scattering
            mu = -p0[i]+p1[i]*sig0
            sig = p2[i]+p3[i]*sig0

            if self.use_fixed_scatter_peak_ratio:
                probi = prob_b**(i+1)
            else:
                probi = self.mode_exp_scatter_peak_ratio(prob_b, prob_c, i+1)

            h_scatter_i = probi*self.gauss_resolution_f(K, 1, sig, mu+Kcenter)
            hydrogen_scattering += h_scatter_i
            norm += probi
            #plt.plot(K, h_scatter_i, color='blue', label='hydrogen')

            # helium scattering
            mu_he = -q0[i]+q1[i]*sig0
            sig_he = q2[i]+q3[i]*sig0
            he_scatter_i = probi*self.gauss_resolution_f(K, 1, sig_he, mu_he+Kcenter)
            helium_scattering += he_scatter_i

            #plt.plot(K, he_scatter_i, color='red', label='helium')

        #plt.plot(K, (shape + hydrogen_scattering)/np.max(shape + hydrogen_scattering), color='blue', label='hydrogen')
        #plt.plot(K, (shape + helium_scattering)/np.max(shape + helium_scattering), color='red', label='helium')
        # full lineshape
        #lineshape = self.hydrogen_proportion*hydrogen_scattering + (1-self.hydrogen_proportion)*helium_scattering
        lineshape = h2_fraction*hydrogen_scattering + (1-h2_fraction)*helium_scattering

        #plt.plot(K, lineshape/np.max(lineshape), color='black', label='full')
        #plt.xlim(-200, 200)
        #plt.legend()

        return lineshape, norm

    """def complex_lineshape(self, K, Kcenter, FWHM, prob_b, prob_c=1):
        lineshape_rates = self.complexLineShape.spectrum_func_1(K/1000., FWHM, 0, 1, prob_b)
        plt.plot(K, lineshape_rates/np.max(lineshape_rates), label='complex lineshape', color='purple', linestyle='--')
        return lineshape_rates"""

    def scatter_peaks(self, K, Kcenter, FWHM):

        if self.plot_lineshape:
            logger.info('Using simplified scatter peaks.')

        p0, p1, p2, p3 = self.lineshape_p[1], self.lineshape_p[3], self.lineshape_p[5], self.lineshape_p[7]
        q0, q1, q2, q3 = self.helium_lineshape_p[1], self.helium_lineshape_p[3], self.helium_lineshape_p[5], self.helium_lineshape_p[7]



        sig0 = FWHM/float(2*np.sqrt(2*np.log(2)))
        #if sig0 < 6:
        #    logger.warning('Scatter resolution {} < 6 eV. Setting to 6 eV'.format(sig0))
        #    sig0 = 6
        #shape = self.gauss_resolution_f(K, 1, sig0, Kcenter)
        #shape = np.zeros(len(K))
        norm = 1.
        #hydrogen_scattering = np.array([self.gauss_resolution_f(K, 1, p2[i]+p3[i]*sig0, (-p0[i]+p1[i]*sig0)+Kcenter) for i in range(self.NScatters)])
        #helium_scattering = np.array([self.gauss_resolution_f(K, 1, q2[i]+q3[i]*sig0, (-q0[i]+q1[i]*sig0)+Kcenter) for i in range(self.NScatters)])

        hydrogen_scattering = np.zeros((self.NScatters, len(K)))
        helium_scattering = np.zeros((self.NScatters, len(K)))


        for i in range(self.NScatters):

            # hydrogen scattering
            mu = -p0[i]+p1[i]*sig0
            sig = p2[i]+p3[i]*sig0
            hydrogen_scattering[i] = self.gauss_resolution_f(K, 1, sig, mu+Kcenter)

            # helium scattering
            mu_he = -q0[i]+q1[i]*sig0
            sig_he = q2[i]+q3[i]*sig0
            helium_scattering[i] = self.gauss_resolution_f(K, 1, sig_he, mu_he+Kcenter)

        return hydrogen_scattering, helium_scattering

    def running_mean(self, x, N):
        N_round = round(N)
        cumsum = np.cumsum(x)
        return (cumsum[int(N_round)::int(N_round)] - cumsum[:-int(N_round):int(N_round)]) / float(N_round)



    def Spectrum(self, E=[], *args, error=False):#endpoint=18.6e3, m_nu=0., prob_b=None, prob_c=None, res=None, sig1=None, sig2=None, tilt=0., error=False):
        E = np.array(E)


        # tritium args
        #kr_args = np.array(args)[self.kr_model_indices]
        if 'B' in self.model_parameter_names:
            
            self.B = args[self.B_index]
            self.ReSetBins()
            #self.ConvertAndHistogram()


        if len(E)==0:
            E = self.bin_centers


        ################## E is list or array #################


        # smear spectrum
        if self.is_smeared or self.is_scattered:

            # resolution params
            if 'resolution' not in self.model_parameter_names or 'resolution' in self.parameter_samples.keys():#self.fixed_parameters[self.res_index]:
                res = self.res
                #logger.info('Using self.res')
            else:
                res = args[self.res_index]

            if self.resolution_model != 'gaussian':
                if self.derived_two_gaussian_model:
                    if 'two_gaussian_mean_1' not in self.model_parameter_names or 'two_gaussian_mean_1' in self.parameter_samples.keys():
                        two_gaussian_mu_1 = self.two_gaussian_mu_1
                        two_gaussian_mu_2 = self.two_gaussian_mu_2
                    else:
                        two_gaussian_mu_1 = args[self.two_gaussian_mu_1_index]
                        two_gaussian_mu_2 = args[self.two_gaussian_mu_2_index]

                else:

                    if 'two_gaussian_sigma_1' not in self.model_parameter_names or 'two_gaussian_sigma_1' in self.parameter_samples.keys():
                        sig1 = self.two_gaussian_sigma_1
                        sig2 = self.two_gaussian_sigma_2
                        #logger.info('Using self.two_gaussian_sigma_1')
                    else:
                        sig1 = args[self.two_gaussian_sigma_1_index]
                        sig2 = args[self.two_gaussian_sigma_2_index]


            max_energy = self.max_energy
            dE = np.abs(np.mean(np.diff(self.energies))) #self.energies[1]-self.energies[0]#E[1]-E[0]
            n_dE = round(max_energy/dE)
            #e_add = np.arange(np.min(self.energies)-round(max_energy/dE)*dE, np.min(self.energies), dE)
            e_lineshape = np.arange(-n_dE*dE, n_dE*dE, dE)

            e_spec = np.arange(min(self.energies)-max_energy, max(self.energies)+max_energy, dE)
            #e_spec = self.energies
            #np.r_[e_add, self._energies]

            # energy resolution
            if self.resolution_model != 'two_gaussian':
                lineshape = self.gauss_resolution_f(e_lineshape, 1, res*self.width_scaling, 0)
                #print('not two gaussian')
            elif self.derived_two_gaussian_model:
                lineshape = self.derived_two_gaussian_resolution(e_lineshape, res, two_gaussian_mu_1, two_gaussian_mu_2)
                #print("derived two gaussian")
            else:
                #print("two gaussian")
                lineshape = self.two_gaussian_wide_fraction * self.gauss_resolution_f(e_lineshape, 1, sig1*self.width_scaling, self.two_gaussian_mu_1) + (1 - self.two_gaussian_wide_fraction) * self.gauss_resolution_f(e_lineshape, 1, sig2*self.width_scaling, self.two_gaussian_mu_2)

            # spectrum shape
            spec = self.which_model(e_spec)#endpoint, m_nu)
            
            if self.plot_lineshape:
                plt.figure()
                plt.plot(e_spec, spec, color='red')
                plt.plot(e_lineshape+Constants.kr_k_line_e(), lineshape/np.max(lineshape), color='green')
                plt.xlabel("Energy (eV)")
                plt.ylabel("Amplitude")
                plt.tight_layout()
                #plt.savefig(os.path.join(self.savepath, "shake_spectrum_absolute_energy.png"))
            #spec[np.where(e_spec>args[self.endpoint_index]-np.abs(m_nu)**0.5)]=0.

            if not self.is_scattered:
                # convolve with gauss with spectrum
                K_convolved = convolve(spec, lineshape, mode='same')
                below_Kmin = np.where(e_spec < min(self.energies))
                #np.put(K_convolved, below_Kmin, np.zeros(len(below_Kmin)))
                K_convolved = np.interp(self.energies, e_spec, K_convolved)
                #K_convolved = K_convolved[e_spec>=np.min(self.energies)]



            if self.is_scattered:
                # scatter params
                if 'scatter_peak_ratio_p' not in self.model_parameter_names:
                    prob_b = self.scatter_peak_ratio_p
                    prob_c = self.scatter_peak_ratio_q
                else:
                    prob_b = args[self.scatter_peak_ratio_p_index]
                    prob_c = args[self.scatter_peak_ratio_q_index]

                    if self.fixed_parameters[self.scatter_peak_ratio_p_index]:
                        prob_b = self.scatter_peak_ratio_p
                    if self.fixed_parameters[self.scatter_peak_ratio_q_index]:
                        prob_c = self.scatter_peak_ratio_q

                if 'h2_fraction' in self.model_parameter_names:
                    h2_fraction = args[self.h2_fraction_index]
                else:
                    h2_fraction = self.h2_fraction

                # simplified lineshape
                FWHM = 2.*np.sqrt(2.*np.log(2.))*res *self.width_scaling

                # add tail
                if not self.use_helium_scattering:
                    tail, norm =  self.simplified_ls(e_lineshape, 0, FWHM, prob_b, prob_c)
                    lineshape += tail
                    lineshape = lineshape/norm
                    K_convolved = convolve(spec, lineshape, mode='same')
                    if self.plot_lineshape:
                        logger.info('Using simplified lineshape model')

                else:
                    tail, norm = self.multi_gas_lineshape(e_lineshape, 0, FWHM, prob_b, prob_c, h2_fraction)
                    lineshape += tail
                    lineshape = lineshape/norm
                    K_convolved = convolve(spec, lineshape, mode='same')
                    if self.plot_lineshape:
                        logger.info('Using two gas simplified lineshape model')
                        plt.plot(e_spec, K_convolved/np.max(K_convolved), color='black')

                try:
                    K_convolved = np.interp(self.energies, e_spec, K_convolved)
                    if self.plot_lineshape:
                        plt.plot(self.energies, K_convolved/np.max(K_convolved), color="cyan")
                except Exception as e:
                    print(np.shape(e_spec), np.shape(K_convolved), np.shape(spec))
                    raise e



            #logger.info('Plotting lineshape now: {}'.format(self.plot_lineshape))
            if self.plot_lineshape:
                logger.info('Plotting confirmed')
                #ax.plot(e_gauss, full_lineshape, label='Full lineshape', color='grey')
                # print(FWHM, ratio)
                # ax.set_xlabel('Energy [eV]')
                # ax.set_ylabel('Amplitude')
                # ax.legend(loc='best')
                # plt.xlim(-500, 250)
                # plt.tight_layout()
                # plt.savefig(os.path.join(self.savepath, 'scattering_model.pdf'), dpi=200, transparent=True)

                plt.figure(figsize=(7,5))
                #plt.plot(e_lineshape, self.simplified_ls(e_lineshape, 0, FWHM, ratio), color='red', label='FDG')
                if self.resolution_model != 'two_gaussian':
                    resolution = self.gauss_resolution_f(e_lineshape, 1, res, 0)
                    logger.info('Using Gaussian resolution model')
                elif self.derived_two_gaussian_model:
                    resolution = self.derived_two_gaussian_resolution(e_lineshape, res, two_gaussian_mu_1, two_gaussian_mu_2)
                    logger.info('Using derived two Gaussian resolution model')
                else:
                    resolution = self.two_gaussian_fraction * self.gauss_resolution_f(e_lineshape, 1, sig1*self.width_scaling, self.two_gaussian_mu_1) + (1 - self.two_gaussian_fraction) * self.gauss_resolution_f(e_lineshape, 1, sig2*self.width_scaling, self.two_gaussian_mu_2)
                    logger.info('Using two Gaussian resolution model')


                plt.plot(e_lineshape, resolution/np.max(resolution), label = 'Resolution', color='orange')
                plt.plot(e_lineshape, lineshape/np.max(lineshape), label = 'Full lineshape', color='Darkblue')


                FWHM = 2.*np.sqrt(2.*np.log(2.))*res*self.width_scaling

                #logger.info('Plotting lineshape for FWHM {} and hydrogen proportion {}.'.format(FWHM, self.hydrogen_proportion))
                #simple_ls, simple_norm = self.simplified_ls(e_lineshape, 0, FWHM, prob_b, prob_c)
                #simple_ls = (self.gauss_resolution_f(e_lineshape, 1, res, 0)+simple_ls)/simple_norm
                #plt.plot(e_lineshape, simple_ls/np.nanmax(simple_ls), label='Hydrogen only lineshape', color='red')
                plt.xlabel('Energy [eV]')
                plt.ylabel('Amplitude')
                plt.grid()
                plt.xlim(-500, 250)
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(os.path.join(self.savepath, 'lineshape.pdf'), dpi=200, transparent=True)



        else:
            # base shape
            spec = self.which_model(e_spec)
            K_convolved = spec

        # Fake integration
        if self.integrate_bins:

            dE = self.denergy
            dE_bins = E[1]-E[0]
            N = np.round(dE_bins/dE,4)

            if not (np.abs(N%1) < 1e-5 or np.abs(N%1 - 1)<1e-5):
                logger.error('N', N)
                logger.error('modulo: {}, {}'.format(N%1, N%1-1))
                print('hi')
                raise ValueError('bin sizes have to divide')


            if N > 1:
                #print('Integrate spectrum', len(K_convolved))
                if E[0] -0.5*dE_bins >= self.energies[1]:
                    K_convolved_cut = K_convolved[self.energies>=min(E[0] -0.5*dE_bins)]
                    K_convolved = self.running_mean(K_convolved_cut, N)
                    logger.warning('Cutting spectrum below Kmin:{} > {}'.format(E[0]-0.5*dE_bins, self.energies[0]))
                else:
                    K_convolved = self.running_mean(K_convolved, N)

            else:
                # till here K_convolved was defiend on self._energies
                # now we need it on E
                K_convolved = np.interp(E, self.energies, K_convolved)*(len(E)*1./len(self.energies))


        else:
            #logger.warning('WARNING: tritium spectrum is not integrated over bin widths')
            K_convolved = np.interp(E, self.energies, K_convolved)*(len(E)*1./len(self.energies))


        K_eff=K_convolved
        # finally
        K = K_eff/np.nansum(K_eff)*np.nansum(K_convolved)

        # if error from efficiency on spectrum shape should be returned
        return K



    def SpectrumWithBackground(self, E=[], *args):

        if len(E)==0:
            E = self.bin_centers

        K = self.Spectrum(E, *args)#endpoint, m_nu, prob_b, prob_c, res, sig1, sig2, tilt)
        K_norm = np.sum(self.Spectrum(self.energies, *args))#endpoint, m_nu, prob_b, prob_c, res, sig1, sig2, tilt)*(self._energies[1]-self._energies[0]))




        if isinstance(E, float):
            print('E is float, only returning tritium amplitude')
            return K*(E[1]-E[0])/K_norm
        else:

            b = args[self.background_index]/(max(self.energies)-min(self.energies))
            a = args[self.amplitude_index]#-b

            B = np.ones(np.shape(E))
            B = B*b#/(self._energies[1]-self._energies[0])
            #B= B/np.sum(B)*background


            K = K/K_norm*a
            #K[E>endpoint-np.abs(m_nu**0.5)+de] = np.zeros(len(K[E>endpoint-np.abs(m_nu**2)+de]))
            #K= K/np.sum(K)*a
            #K = K+B



        return K+B


    def normalized_SpectrumWithBackground(self, E=[], *args):#endpoint=18.6e3, background=0, m_nu=0., amplitude=1., prob_b=None, prob_c=None, res=None, sig1=None, sig2=None, tilt=0., error=False):

        t = self.SpectrumWithBackground(E, *args)#endpoint, background, m_nu, amplitude, prob_b, prob_c, res, sig1, sig2, tilt)
        t_norm = np.sum(t)
        return t/t_norm



 



