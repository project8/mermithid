
"""
Author: C. Claessens
Date:8/03/2021
Description:
    Processor for analyzing tritium data with frequentist methods

    contains tritium model(s)
    convolves with energy resolution
    convolves with lineshape
    multiplies with efficiency
    adds background
    uses BinnedDataFitter to fit binned spectrum with iminuit minimizer


"""

import json
import os
from copy import deepcopy
import numpy as np
from scipy import constants
from scipy.special import erfc
import matplotlib.pyplot as plt


from morpho.utilities import morphologging, reader
logger = morphologging.getLogger(__name__)

from mermithid.processors.Fitters import BinnedDataFitter
from mermithid.misc.FakeTritiumDataFunctions import *
from mermithid.processors.misc.KrComplexLineShape import KrComplexLineShape
from mermithid.misc.DetectionEfficiencyUtilities import pseudo_integrated_efficiency, integrated_efficiency, power_efficiency


#electron_mass = constants.electron_mass/constants.e*constants.c**2
#FineStructureConstant = 0.0072973525664

# =============================================================================
# Functions that can be imported from here and facitilate working with the processor
# =============================================================================
#
# They create an instance of the processor defined below and use it for analysis
# GenAndFit tells the processor to generate new fake data and fit it
# DoOneFit give the processor data and tells it to fit it
# GetPDF returns model pdf for some parameters


def GenAndFit(params, counts, fit_config_dict, sampled_priors={},
              i=0, fixed_data = [], error_scaling=1, tilt = None,
              fit_tilt = False, event=None):


    if i%20 == 0:
        logger.info('Sampling: {}'.format(i))


    T = BinnedTritiumMLFitter("TritiumFitter")
    T.InternalConfigure(fit_config_dict)
    #T.error_scaling = error_scaling
    T.integrate_bins = True

    if tilt is not None:# and tilt !=0:
        T.tilted_efficiency = True
        T.tilt = tilt


    # generate random data from best fit parameters
    if len(fixed_data) == 0:
        _, new_data = T.GenerateData(params, counts)

    else:
        _, new_data = T.GenerateAsimovData(params)


    if fit_tilt:
        T.fix_tilt = False

    T.freq_data = new_data
    results, errors = T.SampleConvertAndFit(sampled_priors, params)
    parameter_samples = T.parameter_samples


    if fit_config_dict['minos_intervals']:
        return results, T.minos_errors, parameter_samples
    else:
        return results, errors, parameter_samples


def DoOneFit(data, fit_config_dict, sampled_parameters={}, error_scaling=0,
             tilt=None, fit_tilt = False, data_is_energy=False):


    T = BinnedTritiumMLFitter("TritiumFitter")
    T.InternalConfigure(fit_config_dict)
    T.print_level=1
    #T.error_scaling = error_scaling
    T.integrate_bins = True
    logger.info('Energy stepsize: {}'.format(T.denergy))

    if tilt is not None:# and tilt != 0:
        T.tilted_efficiency = True
        T.tilt = tilt

    if fit_tilt:
        logger.info('Going to fit efficiency tilt')
        T.fix_tilt = False

    if data_is_energy:
        data = T.Frequency(data)
    T.freq_data = data
    results, errors = T.SampleConvertAndFit(sampled_parameters)
    total_counts = results[1]+results[3]

    if fit_config_dict['minos_intervals']:
        return results, T.minos_errors, total_counts
    else:
        return results, errors, total_counts


def GetPDF(fit_config_dict, params, plot=False):
    logger.info('Plotting lineshape: {}'.format(plot))
    logger.info('PDF for params: {}'.format(params))


    T = BinnedTritiumMLFitter("TritiumFitter")
    T.InternalConfigure(fit_config_dict)
    T.plot_lineshape = plot
    T.pass_efficiency_error = True


    pdf = T.TritiumSpectrumBackground(T.energies, *params)
    _, asimov_binned_data = T.GenerateAsimovData(params)
    binned_fit = T.TritiumSpectrumBackground(T.bin_centers, *params)

    return T.energies, pdf, T.bin_centers, binned_fit, asimov_binned_data


# =============================================================================
# Processor definition
# =============================================================================

class BinnedTritiumMLFitter(BinnedDataFitter):


    def InternalConfigure(self, config_dict):

        # ====================================
        # Model configuration
        # ====================================
        self.use_approx_model = reader.read_param(config_dict, 'use_approximate_model', True)
        self.use_toy_model_efficiency = reader.read_param(config_dict, 'use_toy_model_efficiency', False)
        self.is_distorted = reader.read_param(config_dict, 'distorted', False) # multiply spectrum with efficency
        self.is_smeared = reader.read_param(config_dict, 'smeared', True) # convolve with energy resolution
        self.error_scaling = reader.read_param(config_dict, 'error_scaling', 1.) # scales efficiency uncertainty
        self.is_scattered = reader.read_param(config_dict, 'scattered', False) # convolve with lineshape
        self.integrate_bins = reader.read_param(config_dict, 'integrate_bins', True) # integrate spectrum over bin widths
        self.fit_efficiency_tilt = reader.read_param(config_dict, 'fit_efficiency_tilt', False) # efficiency slope is free parameter
        self.fit_nu_mass = reader.read_param(config_dict, 'fit_neutrino_mass', False)
        self.use_relative_livetime_correction = reader.read_param(config_dict, 'use_relative_livetime_correction', False)

        # detector response options
        self.NScatters = reader.read_param(config_dict, 'NScatters', 20)
        self.resolution_model = reader.read_param(config_dict, 'resolution_model', 'gaussian')
        self.lineshape_model = reader.read_param(config_dict, 'lineshape_model', 'simplified')
        self.simplified_lineshape_path = reader.read_param(config_dict, 'simplified_lineshape_path', "required")
        self.helium_lineshape_path = reader.read_param(config_dict, 'helium_lineshape_path', "optional")
        self.hydrogen_proportion = reader.read_param(config_dict, 'hydrogen_proportion', 1)
        self.use_helium_scattering = reader.read_param(config_dict, 'use_helium_scattering', False)
        self.correlated_b_c_scale = reader.read_param(config_dict, 'correlated_b_c_scale', False)


        # configure model parameter names
        self.model_parameter_names = reader.read_param(config_dict, 'model_parameter_names',
                                                       ['endpoint', 'background', 'm_beta_squared', 'Amplitude',
                                                        'scatter_peak_ratio_b', 'scatter_peak_ratio_c',
                                                        'resolution', 'two_gaussian_sigma_1', 'two_gaussian_sigma_2'] )
        # initial values and mean of constaints (if constraint) or mean of distribution (if sampled)
        self.model_parameter_means = reader.read_param(config_dict, 'model_parameter_means', [18.6e3, 0, 0, 5000, 0.8, 1, 15, 10, 10])
        # width for constraints or sample distributions
        self.model_parameter_widths = reader.read_param(config_dict, 'model_parameter_widths', [100, 0.1, 0.1, 500, 0.1, 0, 3, 1, 1])
        self.fixed_parameters = reader.read_param(config_dict, 'fixed_parameters', [False, False, False, False, True, True, True, True, True])
        self.fixed_parameter_dict = reader.read_param(config_dict, 'fixed_parameter_dict', {})
        self.limits = reader.read_param(config_dict, 'model_parameter_limits',
                                                    [[18e3, 20e3],
                                                     [0, None],
                                                     [-300**2, 300**2],
                                                     [100, None],
                                                     [0.1, 1.],
                                                     [0.1, 1.],
                                                     [12, 100],
                                                     [1, 100],
                                                     [1, 100]])


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
        self.amplitude_index = self.model_parameter_names.index('Amplitude')


        # resolutions

        self.scale_mean = reader.read_param(config_dict, 'scale_mean', 1)
        self.scale_width = reader.read_param(config_dict, 'scale_width', 0)
        self.width_scaling = self.scale_mean

        if self.is_smeared:
            self.res_index = self.model_parameter_names.index('resolution')

            self.res_mean = reader.read_param(config_dict, 'gaussian_resolution_mean', 15.0)
            self.res_width = reader.read_param(config_dict, 'gaussian_resolution_width', 1.0)
            self.res = self.res_mean

            if self.resolution_model != 'gaussian':
                self.two_gaussian_sigma_1_index = self.model_parameter_names.index('two_gaussian_sigma_1')
                self.two_gaussian_sigma_2_index = self.model_parameter_names.index('two_gaussian_sigma_2')

                self.two_gaussian_mu_1 = reader.read_param(config_dict, 'two_gaussian_mu1', 0)
                self.two_gaussian_mu_2 = reader.read_param(config_dict, 'two_gaussian_mu2', 0)
                self.two_gaussian_sigma_1_mean = reader.read_param(config_dict, 'two_gaussian_sigma_1_mean', 15)
                self.two_gaussian_sigma_2_mean = reader.read_param(config_dict, 'two_gaussian_sigma_2_mean', 5)
                self.two_gaussian_sigma_1_width = reader.read_param(config_dict, 'two_gaussian_sigma_1_width', 1)
                self.two_gaussian_sigma_2_width = reader.read_param(config_dict, 'two_gaussian_sigma_2_width', 1)
                self.two_gaussian_wide_fraction = reader.read_param(config_dict, 'two_gaussian_wide_fraction', 1.)
                self.two_gaussian_sigma_1 = deepcopy(self.two_gaussian_sigma_1_mean)
                self.two_gaussian_sigma_2 = deepcopy(self.two_gaussian_sigma_2_mean)


        # scatter peak ratio
        if self.is_scattered:
            self.scatter_peak_ratio_b_index = self.model_parameter_names.index('scatter_peak_ratio_b')
            self.scatter_peak_ratio_c_index = self.model_parameter_names.index('scatter_peak_ratio_c')

            self.spr_factor = reader.read_param(config_dict, 'SPR_factor', 0)
            self.scatter_peak_ratio_b_mean = reader.read_param(config_dict, 'scatter_peak_ratio_b_mean', 0.7)
            self.scatter_peak_ratio_b_width = reader.read_param(config_dict, 'scatter_peak_ratio_b_width', 0.1)
            self.scatter_peak_ratio_c_mean = reader.read_param(config_dict, 'scatter_peak_ratio_c_mean', 0.7)
            self.scatter_peak_ratio_c_width = reader.read_param(config_dict, 'scatter_peak_ratio_c_width', 0.1)

            self.scatter_peak_ratio_mean = reader.read_param(config_dict, 'scatter_peak_ratio_mean', 0.5)
            self.scatter_peak_ratio_width = reader.read_param(config_dict, 'scatter_peak_ratio_width', 0.1)

            self.scatter_peak_ratio_b = self.scatter_peak_ratio_b_mean
            self.scatter_peak_ratio_c = self.scatter_peak_ratio_c_mean

            #Adding correlated parameters (will later incorporate into morpho processor)
            self.b_c_corr = reader.read_param(config_dict, 'b_c_corr', 0)
            self.b_scale_corr = reader.read_param(config_dict, 'b_scale_corr', 0)
            self.c_scale_corr = reader.read_param(config_dict, 'c_scale_corr', 0)


            stds = [self.scatter_peak_ratio_b_width, self.scatter_peak_ratio_c_width, self.scale_width]
            self.b_c_scale_cov_matrix =  [[stds[0]**2, self.b_c_corr*stds[0]*stds[1], self.b_scale_corr*stds[0]*stds[2]],
                                          [self.b_c_corr*stds[1]*stds[0], stds[1]**2, self.c_scale_corr*stds[1]*stds[2]],
                                          [self.b_scale_corr*stds[2]*stds[0], self.c_scale_corr*stds[2]*stds[1], stds[2]**2]]




        # identify tritium parameters
        self.tritium_model_indices = reader.read_param(config_dict, 'tritium_model_parameters', [0, 2])
        self.m_beta_index = self.model_parameter_names.index('m_beta_squared')
        self.endpoint_index = self.model_parameter_names.index('endpoint')
        self.fixed_parameters[self.m_beta_index] = not self.fit_nu_mass
        self.endpoint=reader.read_param(config_dict, 'true_endpoint', 18.573e3)
        logger.info('Tritium model parameters: {}'.format(np.array(self.model_parameter_names)[self.tritium_model_indices]))


        # final state spectrum
        self.use_final_states = reader.read_param(config_dict, 'use_final_states', False)
        self.final_state_array = reader.read_param(config_dict, 'final_states_array', [[0], [1]])


        # save plots in (processor can plot lineshape used in tritium model)
        self.savepath = reader.read_param(config_dict, 'savepath', '.')


        # individual model parameter configurations
        # overwrites model_parameter_means and widths
        self.B_mean = reader.read_param(config_dict, 'B_mean', 1)
        self.B_width = reader.read_param(config_dict, 'B_width', 1e-6)
        self.B = self.B_mean

        self.endpoint_mean = reader.read_param(config_dict,'endpoint_mean', self.model_parameter_means[self.endpoint_index])
        self.endpoint_width = reader.read_param(config_dict, 'endpoint_width', self.model_parameter_widths[self.endpoint_index])

        # frequency an energy range
        self.min_frequency = reader.read_param(config_dict, 'min_frequency', "required")
        self.max_frequency = reader.read_param(config_dict, 'max_frequency', None)
        self.max_energy = reader.read_param(config_dict, 'resolution_energy_max', 1200)

        # path to json with efficiency dictionary
        self.efficiency_file_path = reader.read_param(config_dict, 'efficiency_file_path', '')

        # channel livetimes
        self.channel_transition_freqs = np.array(reader.read_param(config_dict, 'channel_transition_freqs', [[0,1.38623121e9+24.5e9],
                                                                                        [1.38623121e9+24.5e9, 1.44560621e9+24.5e9],
                                                                                        [1.44560621e9+24.5e9, 50e9]]))
        self.channel_livetimes = reader.read_param(config_dict, 'channel_livetimes', [7185228, 7129663, 7160533])
        self.channel_relative_livetimes = np.array(self.channel_livetimes)/ np.max(self.channel_livetimes)




        # =================
        # Fit configuration
        # =================

        self.print_level = reader.read_param(config_dict, 'print_level', 0)
        #self.fix_nu_mass = not self.fit_nu_mass
        self.pass_efficiency_error = False
        self.use_asimov = False
        #self.fix_endpoint = not reader.read_param(config_dict, 'fit_endpoint', True)
        #self.fix_background = not reader.read_param(config_dict, 'fit_background', True)
        #self.fix_amplitude = not reader.read_param(config_dict, 'fit_amplitude', True)
        #self.counts_guess = reader.read_param(config_dict, 'counts_guess', 5000)
        #self.mass_guess = reader.read_param(config_dict, 'nu_mass_guess', 0.0)




        # Parameters can be constrained manually or by inlcuding them in the nuisance parameter dictionary
        #self.constrained_parameter_names = reader.read_param(config_dict, 'constrained_parameter_names', [])
        self.constrained_parameters = reader.read_param(config_dict, 'constrained_parameters', [])
        self.constrained_means = list(np.array(self.model_parameter_means)[self.constrained_parameters])#reader.read_param(config_dict, 'constrained_means', [])
        self.constrained_widths = list(np.array(self.model_parameter_widths)[self.constrained_parameters])#reader.read_param(config_dict, 'constrained_widths', [])


        self.nuisance_parameters = reader.read_param(config_dict, 'nuisance_parameters', {})

        for i, p in enumerate(self.model_parameter_names):
            if p in self.nuisance_parameters.keys():
                if self.nuisance_parameters[p]:
                    self.constrained_parameters.append(i)
                    self.constrained_means.append(self.model_parameter_means[i])
                    self.constrained_widths.append(self.model_parameter_widths[i])
            if i in self.constrained_parameters:
                self.fixed_parameters[i] = False


        # MC uncertainty propagation does not need the fit uncertainties returned by iminuit. uncertainties are instead obtained from the distribution of fit results.
        # But if the uncertainty is unstead propagated by adding constraiend nuisance parameters then the fit uncertainties are needed.
        # imnuit can calculated hesse and minos intervals. the former are symmetric. we want the asymetric intrevals-> minos
        # minos_cls is the list of uncertainty level that should be obtained: e.g. [0.68, 0.9]
        self.minos_intervals = reader.read_param(config_dict, 'minos_intervals', False)
        self.minos_cls = reader.read_param(config_dict, 'minos_confidence_levels', [0.683, 0.9])



        # ==================================
        # detection efficiency configuration
        # ==================================

        self.tilted_efficiency = False
        self.tilt = 0.

        if self.use_toy_model_efficiency:
            x = np.linspace(self.min_frequency, self.max_frequency, 100)
            y = power_efficiency(x, plot=False)

            self.power_eff_interp = interp1d(x, y[0], fill_value=0, bounds_error=False)
            self.power_eff_error_interp = interp1d(x, y[1][0], fill_value=1, bounds_error=False)

        elif self.efficiency_file_path != '':
            with open(self.efficiency_file_path, 'r') as infile:
                snr_efficiency_dict = json.load(infile)

                # don't ask ... it's complicated
                snr_efficiency_dict['frequency'] = snr_efficiency_dict['frequencies']
                snr_efficiency_dict['good_fit_index'] = [True]*len((snr_efficiency_dict['frequencies']))
                snr_efficiency_dict['tritium_rates'] = snr_efficiency_dict['eff interp with slope correction']
                snr_efficiency_dict['tritium_rates_error'] = snr_efficiency_dict['error interp with slope correction']
                #snr_efficiency_dict['tritium_rates'] = snr_efficiency_dict['eff interp no energy correction']
                #snr_efficiency_dict['tritium_rates_error'] = snr_efficiency_dict['error interp no energy correction']

                if self.max_frequency == None:
                    self.max_frequency = np.max(snr_efficiency_dict['frequency'])

            self.snr_efficiency_dict = snr_efficiency_dict
        else:
            self.efficiency_file_path = ''

        if self.max_frequency == None:
            raise ValueError('Max frequency undetermined')

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
        if self.lineshape_model == 'simplified':
            self.multi_gas_lineshape = self.simplified_multi_gas_lineshape
        elif self.lineshape_model == 'accurate_simplified':
            self.multi_gas_lineshape = self.more_accurate_simplified_multi_gas_lineshape

        elif self.lineshape_model =='detailed':
            self.detailed_scatter_spectra_path = reader.read_param(config_dict, 'detailed_lineshape_path')

            ## lineshape params
            #self.SimpParams = [self.res*2*np.sqrt(2*np.log(2)), self.scatter_ratio]

            # Setup and configure lineshape processor
            complexLineShape_config = {
                'gases': ["H2","He"],
                'max_scatters': self.NScatters,
                'fix_scatter_proportion': True,
                # When fix_scatter_proportion is True, set the scatter proportion for gas1 below
                'gas1_scatter_proportion': self.hydrogen_proportion,
                # This is an important parameter which determines how finely resolved
                # the scatter calculations are. 10000 seems to produce a stable fit with minimal slowdown, for ~4000 fake events. The parameter may need to
                # be increased for larger datasets.
                'num_points_in_std_array': 10000,
                'B_field': self.B,
                'base_shape': 'dirac',
                'path_to_osc_strengths_files': self.detailed_scatter_spectra_path
            }
            logger.info('Setting up complex lineshape object')
            self.complexLineShape = KrComplexLineShape("complexLineShape")
            logger.info('Configuring complex lineshape')
            self.complexLineShape.Configure(complexLineShape_config)
            logger.info('Checking existence of scatter spectra files')
            self.complexLineShape.check_existence_of_scatter_file()

            self.multi_gas_lineshape = self.complex_lineshape
        else:
            logger.error('Unknown configure lineshape')
            return False

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


        self._energies = np.arange(self.Energy(self.max_frequency), self.Energy(self.max_frequency)+(self.N_energy_bins)*self.denergy, self.denergy)
        self._bins = np.arange(np.min(self.energies), self.Energy(self.max_frequency)+(self.N_bins)*self.dbins, self.dbins)

        if len(self._energies) > self.N_energy_bins:
            self._energies = self._energies[:-1]
        if len(self._bins) > self.N_bins:
            self._bins = self._bins[:-1]


        self.bin_centers = self._bins[0:-1]+0.5*(self._bins[1]-self._bins[0])
        self.freq_bins = self.Frequency(self._bins)
        self.freq_bin_centers = self.Frequency(self.bin_centers)

        # arrays for internal usage
        self._bin_efficiency, self._bin_efficiency_errors = [], []
        self._full_efficiency, self._full_efficiency_errors = [], []
        self._bin_efficiency, self._bin_efficiency_errors = self.Efficiency(self.bin_centers)
        self._full_efficiency, self._full_efficiency_errors = self.Efficiency(self.energies)

        self.ReSetBins()
        # ==================================
        # configure parent BinnedDataFitter
        # ==================================

        # This processor inherits from the BinnedDataFitter that does binned max likelihood fitting
        # overwrite parent model with tritium model used here
        self.model = self.TritiumSpectrumBackground

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
        self.parameter_names = self.model_parameter_names #['Endpoint', 'Background', 'm_beta_squared', 'Amplitude',
        #                        'scatter_peak_ratio_b', 'scatter_peak_ratio_c',
        #                        'res', 'two_gaussia_sigma_1', 'two_gaussian_sigma_2']
        self.initial_values = self.model_parameter_means#[self.endpoint, self.background, self.mass_guess**2, self.counts_guess,
                               #self.scatter_peak_ratio_b, self.scatter_peak_ratio_c,
                               #self.res, self.two_gaussian_sigma_1, self.two_gaussian_sigma_2]

        self.fixes = self.fixed_parameters #[self.fix_endpoint, self.fix_background, self.fix_nu_mass, self.fix_amplitude,
                      #self.fix_scatter_peak_ratio_b, self.fix_scatter_peak_ratio_c,
                      #self.fix_res, self.fix_two_gaussian_sigma_1, self.fix_two_gaussian_sigma_2]

        self.fixes_dict = self.fixed_parameter_dict

        # self.limits = [energy_limits,
        #            [0, None],
        #            neutrino_limits,
        #            [100, None],
        #            [0.1, 1.],
        #            [0.1, 1.],
        #            [30, 100],
        #            [1, 100],
        #            [1, 100]]



        self.parameter_errors = [max([0.1, p]) for p in self.model_parameter_widths]


        # else:
        #     logger.warning('Efficiency tilt will be fitted')
        #     self.tilted_efficiency = True
        #     self.parameter_names = ['Endpoint', 'Background', 'm_beta_squared', 'Amplitude', 'scatter_peak_ratio_b', 'scatter_peak_ratio_c', 'Efficiency tilt']
        #     self.initial_values = [self.endpoint, 1, self.mass_guess**2, self.counts_guess, self.scatter_peak_ratio_b, self.scatter_peak_ratio_c, self.tilt]
        #     self.parameter_errors = [max([0.1, 0.1*p]) for p in self.initial_values]
        #     self.fixes = [self.fix_endpoint, self.fix_background, self.fix_nu_mass, self.fix_amplitude, self.fix_scatter_ratio_b, self.fix_scatter_ratio_c, self.fix_tilt]
        #     self.limits = [energy_limits,
        #                [1e-10, None],
        #                neutrino_limits,
        #                [0, None],
        #                [0.1, 1.],
        #                [0.1, 1.],
        #                [-0.5, 0.5]]


        # if self.plot_lineshape:
        #     logger.info('Parameters: {}'.format(self.parameter_names))
        #     logger.info('Fixed: {}'.format(self.fixes))
        #     logger.info('Initial values: {}'.format(self.initial_values))


        return True

    # =========================================================================
    # Main fit function
    # =========================================================================
    # This is the main function that is called from outside (besides Configure).
    # Maybe I should rename it to InternalRun to match mermithid naming scheme (and make the processor an actual processor)

    def SampleConvertAndFit(self, sampled_parameters={}, params= []):

        # for systematic MC uncertainty propagation: Generate Asimov data to fit with random model
        if self.use_asimov:
            #temp = self.error_scaling
            #self.error_scaling = 0
            #self._bin_efficiency, self._bin_efficiency_error = self.Efficiency(self.bin_centers, pseudo=True)
            self.hist = self.TritiumSpectrumBackground(self.bin_centers, *params)
            #self.error_scaling = temp

        # sample priors that are to be sampled
        if len(sampled_parameters.keys()) > 0:
            self.SamplePriors(sampled_parameters)


            # need to first re-calcualte energy bins with sampled B before getting efficiency
            self.ReSetBins()

            if 'efficiency' in sampled_parameters.keys():
                random_efficiency = True
            else:
                random_efficiency = False

            # re-calculate bin efficiencies, if self.pseudo_eff=True efficiency will be ranomized
            self._bin_efficiency, self._bin_efficiency_error = self.Efficiency(self.bin_centers, pseudo=random_efficiency)


        # now convert frequency data to energy data and histogram it
        if not self.use_asimov:
            self.ConvertAndHistogram()

        # Call parent fit method using data and model from this instance
        return self.fit()


    # =========================================================================
    # Get random sample from normal, beta, or gamma distribution
    # =========================================================================
    def Gaussian_sample(self, mean, width):
        np.random.seed()
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


    def SamplePriors(self, sampled_parameters):

        for k in sampled_parameters.keys():
            if k in self.nuisance_parameters and self.nuisance_parameters[k] and sampled_parameters[k]:
                raise ValueError('{} is nuisance parameter.'.format(k))

        for i, p in enumerate(self.model_parameter_names):
            if p in sampled_parameters.keys() and sampled_parameters[p] and not self.fixed_parameters[i]:
                raise ValueError('{} is a free parameter'.format(p))

        logger.info('Sampling: {}'.format([k for k in sampled_parameters.keys() if sampled_parameters[k]]))
        self.parameter_samples = {}
        sample_values = []
        if 'resolution' in sampled_parameters.keys() and sampled_parameters['resolution']:
            self.res = self.Gaussian_sample(self.res_mean, self.res_width)
            #if self.res <= 30.01/float(2*np.sqrt(2*np.log(2))):
            #    logger.warning('Sampled resolution small. Setting to {}'.format(30.01/float(2*np.sqrt(2*np.log(2)))))
            #    self.res = 30.01/float(2*np.sqrt(2*np.log(2)))
            self.parameter_samples['resolution'] = self.res
            sample_values.append(self.res)
        if 'two_gaussian_sigma_1' in sampled_parameters.keys() and sampled_parameters['two_gaussian_sigma_1']:
            self.two_gaussian_sigma_1 = self.Gaussian_sample(self.two_gaussian_sigma_1_mean, self.two_gaussian_sigma_1_width)
            self.parameter_samples['two_gaussian_sigma_1'] = self.two_gaussian_sigma_1
            sample_values.append(self.two_gaussian_sigma_1)
        if 'two_gaussian_sigma_2' in sampled_parameters.keys() and sampled_parameters['two_gaussian_sigma_2']:
            self.two_gaussian_sigma_2 = self.Gaussian_sample(self.two_gaussian_sigma_2_mean, self.two_gaussian_sigma_2_width)
            self.parameter_samples['two_gaussian_sigma_2'] = self.two_gaussian_sigma_2
            sample_values.append(self.two_gaussian_sigma_2)
        if 'scatter_peak_ratio' in sampled_parameters.keys() and sampled_parameters['scatter_peak_ratio']:
            self.scatter_peak_ratio_b = self.Beta_sample(self.scatter_peak_ratio_mean, self.scatter_peak_ratio_width)
            self.scatter_peak_ratio_c = 1
            self.fix_scatter_ratio_b = True
            self.fix_scatter_ratio_c = True
            self.parameter_samples['scatter_peak_ratio'] = self.scatter_peak_ratio_b
            sample_values.append(self.scatter_peak_ratio_b)

        if self.correlated_b_c_scale and 'scatter_peak_ratio_b' in sampled_parameters.keys() and 'scatter_peak_ratio_c' in sampled_parameters.keys() and sampled_parameters['scatter_peak_ratio_b'] and sampled_parameters['scatter_peak_ratio_c']:
            logger.info('Correlated b, c, scale sampling')
            correlated_vars = np.random.multivariate_normal([self.scatter_peak_ratio_b_mean, self.scatter_peak_ratio_c_mean, self.scale_mean], self.b_c_scale_cov_matrix)
            self.scatter_peak_ratio_b = correlated_vars[0]
            self.scatter_peak_ratio_c = correlated_vars[1]
            self.width_scaling = correlated_vars[2]

            self.fix_scatter_ratio_b = True
            self.parameter_samples['scatter_peak_ratio_b'] = self.scatter_peak_ratio_b
            sample_values.append(self.scatter_peak_ratio_b)

            self.parameter_samples['scatter_peak_ratio_c'] = self.scatter_peak_ratio_c
            sample_values.append(self.scatter_peak_ratio_c)
            self.fix_scatter_ratio_c = True

        else:

            if 'scatter_peak_ratio_b' in sampled_parameters.keys() and sampled_parameters['scatter_peak_ratio_b']:
                logger.info('Uncorrelated b, c, scale sampling')
                self.scatter_peak_ratio_b = self.Gamma_sample(self.scatter_peak_ratio_b_mean, self.scatter_peak_ratio_b_width)
                self.fix_scatter_ratio_b = True
                self.parameter_samples['scatter_peak_ratio_b'] = self.scatter_peak_ratio_b
                sample_values.append(self.scatter_peak_ratio_b)
            if 'scatter_peak_ratio_c' in sampled_parameters.keys() and sampled_parameters['scatter_peak_ratio_c']:
                self.scatter_peak_ratio_c = self.Gamma_sample(self.scatter_peak_ratio_c_mean, self.scatter_peak_ratio_c_width)
                self.parameter_samples['scatter_peak_ratio_c'] = self.scatter_peak_ratio_c
                sample_values.append(self.scatter_peak_ratio_c)
                self.fix_scatter_ratio_c = True

        if 'B' in sampled_parameters.keys() and sampled_parameters['B']:
            self.B = self.Gaussian_sample(self.B_mean, self.B_width)
            self.parameter_samples['B'] = self.B
            sample_values.append(self.B)
        if 'endpoint' in sampled_parameters.keys() and sampled_parameters['endpoint']:
            self.model_parameter_means[self.endpoint_index] = self.Gaussian_sample(self.endpoint_mean, self.endpoint_width)
            self.parameter_samples['endpoint'] = self.endpoint
            self.fix_endpoint = True
            sample_values.append(self.endpoint)

        logger.info('Samples are: {}'.format(sample_values))
        #logger.info('Fit parameters: \n{}\nFixed: {}'.format(self.parameter_names, self.fixes))
        # set new values in model
        self.ConfigureFit()

        return self.parameter_samples


    # =========================================================================
    # Frequency - Energy conversion
    # =========================================================================

    def Energy(self, f, mixfreq=0.):
        """
        converts frequency in Hz to energy in eV
        """
        emass = constants.electron_mass/constants.e*constants.c**2
        gamma = (constants.e*self.B)/(2.0*np.pi*constants.electron_mass) * 1./(f+mixfreq)

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

    @property
    def energies(self):
        return self._energies

    @energies.setter
    def energies(self, some_energies):
        """
        fine grained array.
        delete pre-existing efficiencies when resetting energies.
        """
        self._energies =  some_energies

        self._full_efficiency, self._full_efficiency_errors = [], []
        efficiency = self.Efficiency(self._energies)
        self._full_efficiency = efficiency[0]
        self._full_efficiency_errors = efficiency[1]

        """super(Tritium, self).__init__(a=np.min(self._energies),
                                        b=np.max(self._energies),
                                        xtol=self.xtol, seed=self.seed)"""


    @property
    def bins(self):
        return self._bins#, self._freq_bin_centers

    @bins.setter
    def bins(self, some_bins):
        """
        energy bins.
        delete pre-existing efficiencies when resetting bins.
        """

        # energy bins
        self._bins = some_bins
        if len(self._energies) > self.N_energy_bins:
            self._energies = self._energies[:-1]
        if len(self._bins) > self.N_bins:
            self._bins = self._bins[:-1]

        self.bin_centers = self._bins[0:-1] +0.5*(self._bins[1]-self._bins[0])

        # frequency bins
        self.freq_bins = self.Frequency(self._bins)
        self.freq_bin_centers = self.Frequency(self.bin_centers)

        # bin efficiencies
        self._bin_efficiency, self._bin_efficiency_errors = [], []
        self._bin_efficiency, self._bin_efficiency_errors = self.Efficiency(self.freq_bin_centers, freq=True)


    def ReSetBins(self):
        #self.energies = np.arange(self.Energy(self.max_frequency), self.Energy(self.min_frequency), self.denergy)
        #self.bins = np.arange(np.min(self.energies), np.max(self.energies), self.dbins)

        #self._bin_efficiency, self._bin_efficiency_error = [], []
        self.energies = np.arange(self.Energy(self.max_frequency), self.Energy(self.max_frequency)+(self.N_energy_bins)*self.denergy, self.denergy)
        self.bins = np.arange(np.min(self.energies), np.min(self.energies)+(self.N_bins)*self.dbins, self.dbins)
        self._bin_efficiency, self._bin_efficiency_error = [], []

        if len(self._energies) > self.N_energy_bins:
            self._energies = self._energies[:-1]
        if len(self._bins) > self.N_bins:
            self._bins = self._bins[:-1]

        if self.use_relative_livetime_correction:
            self.channel_energy_edges = self.Energy(self.channel_transition_freqs)
            #logger.info('Channel energy edges: {}'.format(self.channel_energy_edges))



    # =========================================================================
    # Data generation and histogramming
    # =========================================================================

    def GenerateData(self, params, N):

        x = self.energies[0:-1]+0.5*(self.energies[1]-self.energies[0])
        pdf = np.longdouble(self.TritiumSpectrumBackground(x, *params))
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


    def Histogram(self, weights=None):
        """
        histogram data using bins
        """
        h, b = np.histogram(self.data, bins=self.bins, weights=weights)
        self.hist = h#float2double(h)
        return h


    def ConvertAndHistogram(self, weights=None):
        """
        histogram data using bins
        """
        self.data = self.ConvertFreqData2EnergyData()
        self.hist, b = np.histogram(self.data, bins=self.bins, weights=weights)
        #self.hist = float2double(h)
        return self.hist








    # =========================================================================
    # Model functions
    # =========================================================================


    def gauss_resolution_f(self, energy_array, A, sigma, mu):
        f = A*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(((energy_array-mu)/sigma)**2.)/2.)
        return f


    def approximate_shape(self, K, Q, m_nu_squared):#, index):
        shape = np.zeros(len(K))
        nu_mass_shape_squared = (Q - K)**2 -m_nu_squared
        Q_minus_K = Q-K
        index = np.where((nu_mass_shape_squared>0) & (Q_minus_K>0))
        nu_mass_shape = np.sqrt(nu_mass_shape_squared[index])

        shape[index] = (Q_minus_K[index])*nu_mass_shape
        return shape

    def chopped_approximate_spectrum(self, E, Q, m_nu_squared):

        if self.use_final_states:
            N_states = len(self.final_state_array[0])
            Q_states = Q+self.final_state_array[0]-np.max(self.final_state_array[0])
            approximate_e_phase_space = self.ephasespace(E, Q)

            beta_rates_array = [self.approximate_shape(E, Q_states[i], m_nu_squared)#, index[i])
                                * self.final_state_array[1][i]
                                * approximate_e_phase_space for i in range(N_states)]

            spectrum = GF**2.*Vud**2*Mnuc2/(2.*np.pi**3)*np.nansum(beta_rates_array, axis=0)/np.nansum(self.final_state_array[1])

        else:
            approximate_e_phase_space = self.ephasespace(E, Q)
            beta_rates_array = self.approximate_shape(E, Q, m_nu_squared) * approximate_e_phase_space
            spectrum = GF**2.*Vud**2*Mnuc2/(2.*np.pi**3) * beta_rates_array

        channel_a_index = np.where((E<self.channel_energy_edges[0][0]) & (E>self.channel_energy_edges[0][1]))
        channel_b_index = np.where((E<self.channel_energy_edges[1][0]) & (E>self.channel_energy_edges[1][1]))
        channel_c_index = np.where((E<self.channel_energy_edges[2][0]) & (E>self.channel_energy_edges[2][1]))

        spectrum[channel_a_index] = spectrum[channel_a_index]*self.channel_relative_livetimes[0]
        spectrum[channel_b_index] = spectrum[channel_b_index]*self.channel_relative_livetimes[1]
        spectrum[channel_c_index] = spectrum[channel_c_index]*self.channel_relative_livetimes[2]

        return spectrum


    def approximate_spectrum(self, E, Q, m_nu_squared=0):
        """
        model as in mermithid fake data generator:
        https://github.com/project8/mermithid/blob/feature/phase2-analysis/mermithid/processors/TritiumSpectrum/FakeDataGenerator.py

        but the ephasespace is approximate and neutrino parameter is mass squared (some factors neglected)
        """

        if self.use_final_states:
            N_states = len(self.final_state_array[0])
            Q_states = Q+self.final_state_array[0]-np.max(self.final_state_array[0])
            approximate_e_phase_space = self.ephasespace(E, Q)

            #index = [np.where(((Q_states[i]-E)**2-m_nu_squared > 0) & (Q_states[i]-E > 0)) for i in range(N_states)]
            #index = [np.where(E < Q_states[i] -mnu) for i in range(N_states)]
            beta_rates_array = [self.approximate_shape(E, Q_states[i], m_nu_squared)#, index[i])
                                * self.final_state_array[1][i]
                                * approximate_e_phase_space for i in range(N_states)]

            spectrum = GF**2.*Vud**2*Mnuc2/(2.*np.pi**3)*np.nansum(beta_rates_array, axis=0)/np.nansum(self.final_state_array[1])

        else:
            approximate_e_phase_space = self.ephasespace(E, Q)
            #index = np.where(((Q-E)**2-m_nu_squared > 0) & (Q-E > 0))
            beta_rates_array = self.approximate_shape(E, Q, m_nu_squared) * approximate_e_phase_space

            #beta_rates = np.zeros(len(E))

            #index = np.where(E < Q-mnu)
            #K = E[index]

            #nu_mass_shape = ((Q - K)**2 -m_nu_squared)**0.5
            #beta_rates[index] = GF**2.*Vud**2*Mnuc2/(2.*np.pi**3)*self.ephasespace(K, Q)*(Q - K)*nu_mass_shape
            spectrum = GF**2.*Vud**2*Mnuc2/(2.*np.pi**3) * beta_rates_array

        return spectrum

    def ephasespace(self, K, Q):
        #G = rad_corr(K, Q)         #Radiative correction
        #S = screen_corr(K)         #Screening factor
        #I = exchange_corr(K)       #Exchange correction
        #R = recoil_corr(K, Q)      #Recoil effects
        #LC = finite_nuc_corr(K, Q) #Finite nucleus corrections
        #X = coul_corr(K, Q)        #Recoiling Coulomb field correction
        F = fermi_func(K)          #Uncorrected Fermi function
        return pe(K)*Ee(K)*F#*G*S*I*R*LC*X

    def which_model(self, *pars):
        if self.use_relative_livetime_correction:
            return self.chopped_approximate_spectrum(*pars)
        if self.use_approx_model:
            return self.approximate_spectrum(*pars)
        else:
            return self.effective_TritiumSpectrumShape(*pars)

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

    def f_i(self, K, gamma, mu, sigma):
        z = (mu+gamma*sigma**2+K)/(np.sqrt(2)*sigma)
        f = np.exp(gamma*(mu+K+gamma*sigma**2/2.))*erfc(z)
        return f/np.sum(f)

    def more_accurate_simplified_multi_gas_lineshape(self, K, Kcenter, FWHM, prob_b, prob_c=1):
        """
        Still a simplified lineshape but helium is not just a gaussian
        """
        if self.plot_lineshape:
            logger.info('Using more accurate multi gas scattering. Hydrogen proportion is {}'.format(self.hydrogen_proportion))


        p0, p1, p2, p3 = self.lineshape_p[1], self.lineshape_p[3], self.lineshape_p[5], self.lineshape_p[7]
        q0, q1, q2, q3, q4, q5, q6, q7 = self.helium_lineshape_p[1], self.helium_lineshape_p[3],\
                                        self.helium_lineshape_p[5], self.helium_lineshape_p[7],\
                                        self.helium_lineshape_p[9], self.helium_lineshape_p[11],\
                                        self.helium_lineshape_p[13], self.helium_lineshape_p[15]



        # sig0 = FWHM/float(2*np.sqrt(2*np.log(2)))
        #shape0 = self.gauss_resolution_f(K, 1, sig0, Kcenter)
        #shape0 *= 1/np.sum(shape0)
        shape0 = np.zeros(len(K))
        norm = 1.
        norm_h = 1.
        norm_he = 1.

        hydrogen_scattering = np.zeros(len(K))
        helium_scattering = np.zeros(len(K))

        if FWHM < 30:
            FWHM = 30

        #plt.figure(figsize=(10,10))

        for i in range(self.NScatters):

            # hydrogen scattering
            sig = p0[i]+p1[i]*FWHM
            mu = -(p2[i]+p3[i]*np.log(FWHM-30))

            if self.use_fixed_scatter_peak_ratio:
                probi = prob_b**(i+1)
            else:
                probi = self.mode_exp_scatter_peak_ratio(prob_b, prob_c, i+1)

            h_scatter_i = self.gauss_resolution_f(K, 1, sig, mu+Kcenter)
            hydrogen_scattering += probi*h_scatter_i/np.sum(h_scatter_i)
            norm += probi
            #plt.plot(K, h_scatter_i, color='blue', label='hydrogen')

            # helium scattering
            mu_he = (q0[i]+q1[i]*FWHM+q2[i]*FWHM**2)
            sig_he = q3[i]+q4[i]*FWHM
            gamma_he = q5[i]+q6[i]*FWHM+q7[i]*FWHM**2
            he_scatter_i = self.f_i(K, gamma_he, mu_he, sig_he)
            helium_scattering += probi * he_scatter_i

            #plt.plot(K, (he_scatter_i/(np.sum(he_scatter_i)*(K[1]-K[0]))), color='cyan')

        #plt.plot(K, (shape0 + hydrogen_scattering)/np.max(shape0 + hydrogen_scattering), color='blue', label='hydrogen')
        #plt.plot(K, (shape0 + helium_scattering)/np.max(shape0 + helium_scattering), color='red', label='helium')

        # full lineshape
        #norm_h = np.sum(shape0 + hydrogen_scattering)
        #norm_he = np.sum(shape0 + helium_scattering)

        lineshape = (self.hydrogen_proportion*(shape0 + hydrogen_scattering)/norm_h +
                     (1-self.hydrogen_proportion)*(shape0 + helium_scattering)/norm_he)

        #plt.plot(K, lineshape/np.max(lineshape), color='darkgreen', label='full: {} hydrogen'.format(self.hydrogen_proportion))
        #plt.xlim(-200, 200)
        #plt.legend()

        return lineshape, norm

    def simplified_multi_gas_lineshape(self, K, Kcenter, FWHM, prob_b, prob_c=1):
        """
        This uses Gaussians of different mu and sigma for different gases
        """
        if self.plot_lineshape:
            logger.info('Using gaussian multi gas scattering')

        p0, p1, p2, p3 = self.lineshape_p[1], self.lineshape_p[3], self.lineshape_p[5], self.lineshape_p[7]
        q0, q1, q2, q3 = self.helium_lineshape_p[1], self.helium_lineshape_p[3], self.helium_lineshape_p[5], self.helium_lineshape_p[7]



        sig0 = FWHM/float(2*np.sqrt(2*np.log(2)))
        if sig0 < 6:
            logger.warning('Scatter resolution < 6 eV. Setting to 6 eV')
            sig0 = 6
        #shape = self.gauss_resolution_f(K, 1, sig0, Kcenter)
        #shape = np.zeros(len(K))
        norm = 1.

        hydrogen_scattering = np.zeros(len(K))
        helium_scattering = np.zeros(len(K))

        #plt.figure(figsize=(10,10))

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
        lineshape = self.hydrogen_proportion*hydrogen_scattering + (1-self.hydrogen_proportion)*helium_scattering

        #plt.plot(K, lineshape/np.max(lineshape), color='black', label='full')
        #plt.xlim(-200, 200)
        #plt.legend()

        return lineshape, norm

    """def complex_lineshape(self, K, Kcenter, FWHM, prob_b, prob_c=1):
        lineshape_rates = self.complexLineShape.spectrum_func_1(K/1000., FWHM, 0, 1, prob_b)
        plt.plot(K, lineshape_rates/np.max(lineshape_rates), label='complex lineshape', color='purple', linestyle='--')
        return lineshape_rates"""


    def running_mean(self, x, N):
        N_round = round(N)
        cumsum = np.cumsum(x)
        return (cumsum[int(N_round)::int(N_round)] - cumsum[:-int(N_round):int(N_round)]) / float(N_round)


    def TritiumSpectrum(self, E=[], *args, error=False):#endpoint=18.6e3, m_nu=0., prob_b=None, prob_c=None, res=None, sig1=None, sig2=None, tilt=0., error=False):
        E = np.array(E)


        # tritium args
        tritium_args = np.array(args)[self.tritium_model_indices]


        if len(E)==0:
            E = self._bin_centers

        """############### E is float ###############
        if isinstance(E, float):
            if E+m_nu > endpoint:
                K_spec = 0.
            else:

                efficiency = 1.
                if self.is_distorted == True:
                    efficiency = self.Efficiency(E)[0]
                K_spec = self.which_model(E, endpoint, m_nu)*efficiency/self.norm"""


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

                if 'two_gaussian_sigma_1' not in self.model_parameter_names or 'two_gaussian_sigma_1' in self.parameter_samples.keys():
                    sig1 = self.two_gaussian_sigma_1
                    #logger.info('Using self.two_gaussian_sigma_1')
                else:
                    sig1 = args[self.two_gaussian_sigma_1_index]
                if 'two_gaussian_sigma_2' not in self.model_parameter_names or 'two_gaussian_sigma_2' in self.parameter_samples.keys():
                    sig2 = self.two_gaussian_sigma_2
                    #logger.info('Using self.two_gaussian_sigma_2')
                else:
                    sig2 = args[self.two_gaussian_sigma_2_index]


            max_energy = self.max_energy
            dE = self.energies[1]-self.energies[0]#E[1]-E[0]
            n_dE = round(max_energy/dE)
            #e_add = np.arange(np.min(self.energies)-round(max_energy/dE)*dE, np.min(self.energies), dE)
            e_lineshape = np.arange(-n_dE*dE, n_dE*dE, dE)


            e_spec = np.arange(min(self.energies)-max_energy, max(self.energies)+max_energy, dE)
            #np.r_[e_add, self._energies]

            # energy resolution
            if self.resolution_model != 'two_gaussian':
                lineshape = self.gauss_resolution_f(e_lineshape, 1, res*self.width_scaling, 0)
            else:
                lineshape = self.two_gaussian_wide_fraction * self.gauss_resolution_f(e_lineshape, 1, sig1*self.width_scaling, self.two_gaussian_mu_1) + (1 - self.two_gaussian_wide_fraction) * self.gauss_resolution_f(e_lineshape, 1, sig2*self.width_scaling, self.two_gaussian_mu_2)

            # spectrum shape
            spec = self.which_model(e_spec, *tritium_args)#endpoint, m_nu)
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
                prob_b = args[self.scatter_peak_ratio_b_index]
                prob_c = args[self.scatter_peak_ratio_c_index]

                if self.fixed_parameters[self.scatter_peak_ratio_b_index]:
                    prob_b = self.scatter_peak_ratio_b
                if self.fixed_parameters[self.scatter_peak_ratio_c_index]:
                    prob_c = self.scatter_peak_ratio_c


                if prob_b == None:
                    prob_b = self.scatter_peak_ratio_b_mean
                    prob_c = self.scatter_peak_ratio_c_mean


                # simplified lineshape
                FWHM = 2.*np.sqrt(2.*np.log(2.))*res *self.width_scaling

                # get lineshape
                if not self.use_helium_scattering:
                    tail, norm =  self.simplified_ls(e_lineshape, 0, FWHM, prob_b, prob_c)
                    if self.plot_lineshape:
                        logger.info('Using simplified lineshape model')
                else:
                    tail, norm = self.multi_gas_lineshape(e_lineshape, 0, FWHM, prob_b, prob_c)
                    if self.plot_lineshape:
                        logger.info('Using two gas simplified lineshape model')

                lineshape += tail
                lineshape = lineshape/norm

                # lineshape done now convolve
                K_convolved = convolve(spec, lineshape, mode='same')
                below_Kmin = np.where(e_spec < min(self.energies))
                #np.put(K_convolved, below_Kmin, np.zeros(len(below_Kmin)))
                K_convolved = np.interp(self.energies, e_spec, K_convolved)
                #K_convolved = K_convolved[e_spec>=np.min(self.energies)]


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
                    if self.plot_lineshape:
                        logger.info('Using Gaussian resolution model')
                else:
                    resolution = self.two_gaussian_wide_fraction * self.gauss_resolution_f(e_lineshape, 1, sig1*self.width_scaling, self.two_gaussian_mu_1) + (1 - self.two_gaussian_wide_fraction) * self.gauss_resolution_f(e_lineshape, 1, sig2*self.width_scaling, self.two_gaussian_mu_2)
                    if self.plot_lineshape:
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
            spec = self.which_model(e_spec, *tritium_args)
            K_convolved = spec

        # Fake integration
        if self.integrate_bins:

            dE = self.denergy
            dE_bins = E[1]-E[0]
            N = np.round(dE_bins/dE,4)

            if not (np.abs(N%1) < 1e-6 or np.abs(N%1 - 1)<1e-6):
                logger.error('N', N)
                logger.error('modulo', N%1, N%1-1)
                raise ValueError('bin sizes have to divide')


            if N > 1:
                #print('Integrate spectrum', len(K_convolved))
                if E[0] -0.5*dE_bins >= self._energies[1]:
                    K_convolved_cut = K_convolved[self.energies>=min(E[0] -0.5*dE_bins)]
                    K_convolved = self.running_mean(K_convolved_cut, N)
                    logger.warning('Cutting spectrum below Kmin:{} > {}'.format(E[0]-0.5*dE_bins, self._energies[0]))
                else:
                    K_convolved = self.running_mean(K_convolved, N)

            else:
                # till here K_convolved was defiend on self._energies
                # now we need it on E
                K_convolved = np.interp(E, self.energies, K_convolved)*(len(E)*1./len(self._energies))


        else:
            #logger.warning('WARNING: tritium spectrum is not integrated over bin widths')
            K_convolved = np.interp(E, self._energies, K_convolved)*(len(E)*1./len(self._energies))



        if self.is_distorted == True:
            #if self.fit_efficiency_tilt:
            #    self.tilt = tilt
            efficiency, efficiency_errors =  self.Efficiency(E)
        else:
            # multiply efficiency
            efficiency = np.ones(len(E))
            efficiency_errors = [np.zeros(len(E)), np.zeros(len(E))]

        K_eff=K_convolved*efficiency
        # finally
        K = K_eff/np.nansum(K_eff)*np.nansum(K_convolved)

        # if error from efficiency on spectrum shape should be returned
        if error:
            K_error=np.zeros((2, len(E)))
            K_error = K_convolved*(efficiency_errors)/np.nansum(K_eff)*np.nansum(K_convolved)

            return K, K_error
        else:
            return K



    def TritiumSpectrumBackground(self, E=[], *args):

        if len(E)==0:
            E = self.bin_centers

        if self.pass_efficiency_error:
            K, K_error = self.TritiumSpectrum(E, *args, error=True)#endpoint, m_nu, prob_b, prob_c, res, sig1, sig2, tilt)
            K_norm = np.sum(self.TritiumSpectrum(self._energies, *args))#endpoint, m_nu, prob_b, prob_c, res, sig1, sig2, tilt)*(self._energies[1]-self._energies[0]))
        else:

            K = self.TritiumSpectrum(E, *args)#endpoint, m_nu, prob_b, prob_c, res, sig1, sig2, tilt)
            K_norm = np.sum(self.TritiumSpectrum(self._energies, *args))#endpoint, m_nu, prob_b, prob_c, res, sig1, sig2, tilt)*(self._energies[1]-self._energies[0]))




        if isinstance(E, float):
            print('E is float, only returning tritium amplitude')
            return K*(E[1]-E[0])/K_norm
        else:

            b = args[self.background_index]/(max(self._energies)-min(self._energies))
            a = args[self.amplitude_index]#-b

            B = np.ones(np.shape(E))
            B = B*b*(E[1]-E[0])#/(self._energies[1]-self._energies[0])
            #B= B/np.sum(B)*background


            K = (K/K_norm*(E[1]-E[0]))*a
            #K[E>endpoint-np.abs(m_nu**0.5)+de] = np.zeros(len(K[E>endpoint-np.abs(m_nu**2)+de]))
            #K= K/np.sum(K)*a
            #K = K+B



        if self.pass_efficiency_error:
            K_error = K_error*(E[1]-E[0])/K_norm*a
            #K_error = K_error/np.sum(K)*a
            return K+B, K_error
        else: return K+B


    def normalized_TritiumSpectrumBackground(self, E=[], *args):#endpoint=18.6e3, background=0, m_nu=0., amplitude=1., prob_b=None, prob_c=None, res=None, sig1=None, sig2=None, tilt=0., error=False):

        if self.pass_efficiency_error:
            t, t_error = self.TritiumSpectrumBackground(E, *args)#endpoint, background, m_nu, amplitude, prob_b, prob_c, res, sig1, sig2, tilt)
            t_norm = np.sum(t)
            return t/t_norm, t_error/t_norm
        else:
            t = self.TritiumSpectrumBackground(E, *args)#endpoint, background, m_nu, amplitude, prob_b, prob_c, res, sig1, sig2, tilt)
            t_norm = np.sum(t)
            return t/t_norm



    ################# SNR - Efficiency functions #####################


    def Efficiency(self, E, freq=False, pseudo=False):
        """
        get efficiencies for energy or frequency bins

        freq: if true, E is assumed to be frequencies instead of energies
        """

        if not freq:
            f = self.Frequency(E)
        else:
            f = deepcopy(E)
            E = self.Energy(f)


        # if E is float
        if isinstance(f, float):
            #print('Calculating single efficiency value')
            f = np.array([f])
            if not phase_4:
                df = 1e6
            else:
                df = 1
            if pseudo == False:
                print('best efficiency')
                efficiency, errors = integrated_efficiency(f, self.snr_efficiency_dict, df)
                return efficiency, errors
            else:
                print('pseudo efficiency')
                efficiency, errors = pseudo_integrated_efficiency(f, self.snr_efficiency_dict, df, alpha=self.error_scaling)
                return efficiency, errors


        # if E is list or array
        else:

            # if efficiency has already been calculated for this array, skip recalculation to save time
            if len(f) == len(self._bin_efficiency):
                efficiency, errors =  self._bin_efficiency, self._bin_efficiency_errors
            elif len(f) == len(self._full_efficiency):
                efficiency, errors =  self._full_efficiency, self._full_efficiency_errors
            else:
                #logger.info('Calculating efficiency with df = {}...'.format(f[1]-f[0]))

                # toy model efficiency
                if self.use_toy_model_efficiency:
                    efficiency = self.power_eff_interp(f)
                    errors = self.power_eff_error_interp(f)

                # fss efficiency
                else:
                    efficiency, errors = integrated_efficiency(f, self.snr_efficiency_dict)


            if self.tilted_efficiency:# and (pseudo or self.fit_efficiency_tilt):

                #efficiency, errors = det_eff.integrated_efficiency(f, self.snr_efficiency_dict)
                slope = self.tilt/(1e3)

                logger.info('Tilted efficiencies: {}'.format(self.tilt))
                efficiency *= (1.+slope*(E-17.83e3))
                errors *= (1.+slope*(E-17.83e3))

            # if we are doing pseudo experiments
            if pseudo:# and len(f) < len(self.energies):# and len(f) != len(self._bin_efficiency):

                #logger.info('uncorrelated pseudo efficiencies: {}'.format(self.error_scaling))

                pseudo_efficiency = np.random.randn(len(f))*np.mean([errors[0], errors[1]], axis=0)*self.error_scaling
                #pseudo_efficiency[pseudo_efficiency<0]*=errors[0][pseudo_efficiency<0]*self.error_scaling
                #pseudo_efficiency[pseudo_efficiency>=0]*=errors[1][pseudo_efficiency>=0]*self.error_scaling
                pseudo_efficiency += efficiency
                pseudo_efficiency*=np.sum(efficiency)/np.nansum(pseudo_efficiency)
                efficiency = pseudo_efficiency

                if np.min(efficiency) < 0:
                    index = np.where(E>=np.min(E[efficiency<0]))
                    efficiency[index] = 0.


            return efficiency, errors





