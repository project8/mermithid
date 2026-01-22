'''
Fits data to complex lineshape model.
Author: E. Machado, Y.-H. Sun, E. Novitski, T. E. Weiss, X. Huyan
Date: February 9, 2021
Comments added: July 2024 and June-September 2025
    Script re-organized by Talia and Yu-Hao; we finished this process on Sept. 10, 2025.

This processor takes in CRES frequency data in a binned histogram and fits the histogram 
with a complex line shape model that includes scattering with multiple gases.

This processor was used for the official Project 8 Phase II data analysis. In particular,
we used the processor to generate a detector response function for tritium data, then
convolved the response with the underlying beta spectrum using the FakeDataGenerator.
That generator uses the following functions in the MultiGasComplexLineShape processor:
- make_spectrum_simulated_resolution_scaled_fit_scatter_peak_ratio()
- make_spectrum_gaussian_resolution_fit_scatter_peak_ratio()
Please see those functions below for more detail.

This processor was also used for the Phase II Kr analysis. The following function
was used for the Kr fits:
fit_data_simulated_resolution_scaled_fit_scatter_peak_ratio2()


Configurable parameters:

There are two options available for fitting: fix_scatter_proportion = True and False.
gases: array for names of the two gases involved in the scattering process.
max_scatter: max number of scatterings for only single gas scatterings.
max_comprehansive_scatter: max number of scatterings for all cross scatterings.
scatter_proportion: when fix_scatter_proportion is set as true, gives the fixed scatter proportion.
resolution_function: The configuration of the resolution function decides which fit function along with the model for resolution to be used in the fit. To see which fit function is used for a specific setting of the resolution function, see the beginning of the Internal_run().
    The list of available settings for resolution functions are 'simulated_resolution', 'gaussian_resolution', 'gaussian_lorentzian_composite_resolution', 'elevated_gaussian', 'composite_gaussian_scaled', 'simulated_resolution_scaled', 'simulated_resolution_scaled_fit_scatter_peak_ratio', 'gaussian_resolution_fit_scatter_peak_ratio'
num_points_in_std_array: number of points for std_array defining how finely the scatter calculations are.
RF_ROI_MIN: can be found from meta data.
B_field: can be put in hand or found by position of the peak of the frequency histogram.
shake_spectrum_parameters_json_path: path to json file storing shake spectrum parameters.
path_to_osc_strength_files: path to oscillator strength files.

See those code below for more configurable parameters.
'''

from __future__ import absolute_import

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import comb
from scipy import integrate , signal, interpolate
from itertools import product
from math import factorial
from iminuit import Minuit
import os
import time
import sys
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.misc import Constants, ComplexLineShapeUtilities, ConversionFunctions

logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

class MultiGasComplexLineShape(BaseProcessor):

    def InternalConfigure(self, params):
        '''
        Configure
        '''
        # Read other parameters
        self.bins_choice = reader.read_param(params, 'bins_choice', [])
        self.gases = reader.read_param(params, 'gases', ["H2", "Kr", "He", "Ar"])
        # when self.fix_gas_composition and self.fix_width_scale_factor are both True,
        # fit_data_simulated_resolution_scaled_fit_scatter_peak_ratio_with_fixed_gas_composition_and_width_scale_factor is used,
        # Then the first N-1 gas compositions are set below through self.scatter_fractions_for_gases
        # Otherwise, fit_data_simulated_resolution_scaled_fit_scatter_peak_ratio is used
        self.fix_gas_composition = reader.read_param(params, 'fix_gas_composition', False)
        self.fix_width_scale_factor = reader.read_param(params, 'fix_width_scale_factor', False)
        self.scatter_fractions_for_gases = reader.read_param(params, 'scatter_fractions_for_gases', [])
        self.max_scatters = reader.read_param(params, 'max_scatters', 20)
        self.trap_weights = reader.read_param(params, 'trap_weights', {'weights':[0.076,  0.341, 0.381, 0.203], 'errors':[0.003, 0.013, 0.014, 0.02]}) #Weights from Xueying's Sept. 13 slides; errors currently arbitrary
        self.fixed_scatter_proportion = reader.read_param(params, 'fixed_scatter_proportion', True)
        if self.fixed_scatter_proportion == True:
            self.scatter_proportion = reader.read_param(params, 'gas_scatter_proportion', [])
        self.partially_fixed_scatter_proportion = reader.read_param(params, 'partially_fixed_scatter_proportion', True)
        if self.partially_fixed_scatter_proportion == True:
            self.free_gases = reader.read_param(params, 'free_gases', ["H2", "He"])
            self.fixed_gases = reader.read_param(params, 'fixed_gases', ["Ar", "Kr"])
            self.gases = self.free_gases + self.fixed_gases
            self.scatter_proportion_for_fixed_gases = reader.read_param(params, 'scatter_proportion_for_fixed_gases', [0.018, 0.039])
        self.fixed_survival_probability = reader.read_param(params, 'fixed_survival_probability', True)
        if self.fixed_survival_probability == True:
            self.survival_prob = reader.read_param(params, 'survival_prob', 1)
        self.use_radiation_loss = reader.read_param(params, 'use_radiation_loss', True)
        self.sample_ins_resolution_errors = reader.read_param(params, 'sample_ins_res_errors', False)
        # configure the resolution functions: gaussian_resolution, gaussian_resolution_fit_scatter_peak_ratio, gaussian_lorentzian_composite_resolution, elevated_gaussian, composite_gaussian, and simulated_resolution_scaled
        # (see InternalRun for more options)
        # The configuration of the resolution function decides which fit function to be used in the fit. To see which fit function is used for a specific setting of the resolution function, see the beginning of the Internal_run()
        self.resolution_function = reader.read_param(params, 'resolution_function', '')

        # This is used in make_spectrum functions that are used for tritium data generation
        # (See comment at top of file for more detail.)
        # The 'simulated_resolution_scaled_fit_scatter_peak_ratio' function was also used for a Phase II
        # analysis to estimate the maximum SNR by scaling the resoluton width (see Phase II PRC paper).
        # The 'simulated_resolution_scaled_fit_scatter_peak_ratio2' was used for the main Kr analysis.
        if self.resolution_function == 'simulated_resolution_scaled' or 'simulated_resolution_scaled_fit_scatter_peak_ratio' or 'simulated_resolution_scaled_fit_scatter_peak_ratio2' or 'gaussian_resolution_fit_scatter_peak_ratio':
            self.fixed_parameter_names = reader.read_param(params, 'fixed_parameter_names', [])
            self.fixed_parameter_values = reader.read_param(params, 'fixed_parameter_values', [])

        # The resolution functions below do not have a specific known use-case, but they were considered/tested as potential lineshapes.
        # You can find the definitions of these functions toward the bottom of the script, right after
        # a break in the code marked by a comment (with many "########").
        if self.resolution_function == 'gaussian_lorentzian_composite_resolution':
            self.ratio_gamma_to_sigma = reader.read_param(params, 'ratio_gamma_to_sigma', 0.8)
            self.gaussian_proportion = reader.read_param(params, 'gaussian_proportion', 0.8)
        if self.resolution_function == 'elevated_gaussian':
            self.elevation_factor = reader.read_param(params, 'elevation_factor', 20)
        if self.resolution_function == 'composite_gaussian':
            self.A_array = reader.read_param(params, 'A_array', [0.076, 0.341, 0.381, 0.203])
            self.sigma_array = reader.read_param(params, 'sigma_array', [5.01, 13.33, 15.40, 11.85])
        
        # This is an important parameter which determines how finely resolved
        # the scatter calculations are. 10000 seems to produce a stable fit, with minimal slowdown
        self.num_points_in_std_array = reader.read_param(params, 'num_points_in_std_array', 10000)
        self.RF_ROI_MIN = reader.read_param(params, 'RF_ROI_MIN', 25850000000.0)
        # setting the Minuit.errordef parameter
        self.error_inflation_factor = reader.read_param(params, 'error_inflation_factor', 1)
        self.Kr_K_line_eV = reader.read_param(params, 'Kr_K_line_eV', Constants.kr_k_line_e())
        self.base_shape = reader.read_param(params, 'base_shape', 'shake')
        self.shake_spectrum_parameters_json_path = reader.read_param(params, 'shake_spectrum_parameters_json_path', 'shake_spectrum_parameters.json')
        self.path_to_osc_strengths_files = reader.read_param(params, 'path_to_osc_strengths_files', '/host/')
        self.path_to_scatter_spectra_file = reader.read_param(params, 'path_to_scatter_spectra_file', '/host/')
        self.path_to_missing_track_radiation_loss_data_numpy_file = reader.read_param(params, 'rad_loss_path', '/host')
        self.path_to_ins_resolution_data_txt = reader.read_param(params, 'path_to_ins_resolution_data_txt', '/host/res_cf15.5_all.txt')
        self.use_combined_four_trap_inst_reso = reader.read_param(params, 'use_combined_four_trap_inst_reso', False)
        self.path_to_four_trap_ins_resolution_data_txt = reader.read_param(params, 'path_to_four_trap_ins_resolution_data_txt', ['/host/analysis_input/complex-lineshape-inputs/T2-1.56e-4/res_cf15.5_trap1.txt', '/host/analysis_input/complex-lineshape-inputs/T2-1.56e-4/res_cf15.5_trap2.txt', '/host/T2-1.56e-4/analysis_input/complex-lineshape-inputs/res_cf15.5_trap3.txt', '/host/analysis_input/complex-lineshape-inputs/T2-1.56e-4/res_cf15.5_trap4.txt'])
        self.use_quad_trap_eff_interp = reader.read_param(params, 'use_quad_trap_eff_interp', True)
        if self.use_quad_trap_eff_interp == True:
            self.path_to_quad_trap_eff_interp = reader.read_param(params, 'path_to_quad_trap_eff_interp', '/host/quad_interps.npy')
        self.recon_eff_params = reader.read_param(params, 'recon_eff_params', [0.005569990343215976, 0.351, 0.546])
        self.recon_eff_param_a = self.recon_eff_params[0]
        self.recon_eff_param_b = self.recon_eff_params[1]
        self.recon_eff_param_c = self.recon_eff_params[2]
        self.factor = reader.read_param(params, 'factor', [])

        if not os.path.exists(self.shake_spectrum_parameters_json_path) and self.base_shape=='shake':
            raise IOError('Shake spectrum path does not exist')
        if not os.path.exists(self.path_to_osc_strengths_files):
            raise IOError('Path to osc strengths files does not exist')
        # Read shake parameters from JSON file
        if self.base_shape == 'shake':
            self.shakeSpectrumClassInstance = ComplexLineShapeUtilities.ShakeSpectrumClass(self.shake_spectrum_parameters_json_path, self.std_eV_array())
        # read in resolution if simulated
        if 'simulated' in self.resolution_function:
            self.sample_and_interpolate_resolution()
        return True

    def InternalRun(self):

        # number_of_events = len(self.data['StartFrequency'])
        # self.results = number_of_events

        a = self.data['StartFrequency']
        a = np.array(a)[0:-1]

        # fit with shake spectrum
        data_hist_freq, freq_bins= np.histogram(a,bins=self.bins_choice)

        #These are the most useful detector response shapes
        # I.e.: ('gaussian_resolution' or 'gaussian_resolution_fit_scatter_peak_ratio', 'composite_gaussian', 'simulated_resolution', 'simulated_resolution_scaled_fit_scatter_peak_ratio', 'simulated_resolution_scaled_fit_scatter_peak_ratio2')
        if self.resolution_function == 'gaussian_resolution' or 'gaussian_resolution_fit_scatter_peak_ratio':
            self.results = self.fit_data_gaussian_resolution_fit_scatter_peak_ratio(freq_bins, data_hist_freq)
        # While we have not used the composite_gaussian resolution for an official analysis,
        # it could potentially work for a toy model of scatter peaks.
        # (For the Phase II tritium fits, we did use a composite gaussian model of scatter peaks, though the means
        # and standard deviations were calculated as a function of the standard deviation of the instrumental resolution,
        # and we did not use the functions in this processor.)
        elif self.resolution_function == 'composite_gaussian':
            self.results = self.fit_data_composite_gaussian_fixed_scatter_proportion(freq_bins, data_hist_freq)
        elif self.resolution_function == 'simulated_resolution':
            if self.fixed_scatter_proportion == True:
                self.results = self.fit_data_ftc(freq_bins, data_hist_freq)
            else:
                self.results = self.fit_data_ftc_2(freq_bins, data_hist_freq)
        elif self.resolution_function == 'simulated_resolution_scaled_fit_scatter_peak_ratio':
            #In Project 8's Phase II analysis, this function was used for a fit of Kr data
            #performed just before the tritium analysis. This Kr fit served as a double-check
            #of the official Kr analysis (performed with a different function, below).
            self.results = self.fit_data_simulated_resolution_scaled_fit_scatter_peak_ratio(freq_bins, data_hist_freq)
        elif self.resolution_function == 'simulated_resolution_scaled_fit_scatter_peak_ratio2':
            #This is the function used for the main Phase II Kr analyses. 
            #It is similar to the function just above. There may be subtle differences,
            #but the two functions produced compatible results.
            self.results = self.fit_data_simulated_resolution_scaled_fit_scatter_peak_ratio2(freq_bins, data_hist_freq)

        # The resolution functions below do not have a specific known use-case, but they were considered/tested as potential lineshapes.
        # You can find the definitions of these functions toward the bottom of the script, right after
        # a break in the code marked by a comment (with many "########").
        elif self.resolution_function == 'gaussian_lorentzian_composite_resolution':
            self.results = self.fit_data_composite_gaussian_lorentzian_fixed_scatter_proportion(freq_bins, data_hist_freq)
        elif self.resolution_function == 'elevated_gaussian':
            self.results = self.fit_data_elevated_gaussian_fixed_scatter_proportion(freq_bins, data_hist_freq)
        elif self.resolution_function == 'composite_gaussian_scaled':
            self.results = self.fit_data_composite_gaussian_scaled_fixed_scatter_proportion(freq_bins, data_hist_freq)
        return True


    # Establishes a standard energy loss array (SELA) from -1000 eV to 1000 eV
    # with number of points equal to self.num_points_in_std_array. All convolutions
    # will be carried out on this particular discretization
    def std_eV_array(self):
        emin = -1000
        emax = 1000
        array = np.linspace(emin,emax,self.num_points_in_std_array)
        return array

    # A lorentzian line centered at 0 eV, with 2.83 eV width on the SELA
    def std_lorenztian_17keV(self):
        x_array = self.std_eV_array()
        ans = lorentzian(x_array,0,kr_line_width)
        return ans

    #A Dirac delta function
    def std_dirac(self):
        x_array = self.std_eV_array()
        ans = np.zeros(len(x_array))
        min_x = np.min(np.abs(x_array))
        ans[np.abs(x_array)==min_x] = 1.
        logger.warning('Spectrum will be shifted by lineshape by {} eV'.format(min_x))
        if min_x > 0.1:
            logger.warning('Lineshape will shift spectrum by > 0.1 eV')
        if min_x > 1.:
            logger.warning('Lineshape will shift spectrum by > 1 eV')
            raise ValueError('problem with std_eV_array()')
        return ans

    # A gaussian function - this is used by other resolution functions (e.g. composite_gaussian)
    def gaussian(self, x_array, A, sigma, mu):
        f = A*(1./(sigma*np.sqrt(2*np.pi)))*np.exp(-(((x_array-mu)/sigma)**2.)/2.)
        return f

    # A gaussian centered at 0 eV with variable width, on the standard energy loss array (SELA)
    # This is used in the convolve_gaussian function.
    def std_gaussian(self, sigma):
        x_array = self.std_eV_array()
        ans = ComplexLineShapeUtilities.gaussian(x_array,1,sigma,0)
        return ans

    # Sum of an arbitrary number of gaussians, defined by amplitudes self.A_array and standard deviations
    # self.sigma_array. Not automatically normalized.
    # This could be used for a super simple model of scatter peaks.
    def composite_gaussian(self):
        x_array = self.std_eV_array()
        ans = 0
        A_array = self.A_array
        sigma_array = self.sigma_array
        for A, sigma in zip(A_array, sigma_array):
            ans += self.gaussian(x_array, A, sigma, 0)
        return ans

    # normalizes a function, but depends on binning.
    # Only to be used for functions evaluated on the SELA
    def normalize(self, f):
        x_arr = self.std_eV_array()
        f_norm = integrate.simps(f,x=x_arr)
        f_normed = f/f_norm
        return f_normed

    # Function for energy loss from a single scatter of electrons by
    # V.N. Aseev et al. 2000
    # This function does the work of combining fit_func1 and fit_func2 by
    # finding the point where they intersect.
    # Evaluated on the SELA
    def single_scatter_f(self, gas_type):
        energy_loss_array = self.std_eV_array()
        f = 0 * energy_loss_array

        input_filename = self.path_to_osc_strengths_files + gas_type + "OscillatorStrength.txt"
        energy_fOsc = ComplexLineShapeUtilities.read_oscillator_str_file(input_filename)
        fData = interpolate.interp1d(energy_fOsc[0], energy_fOsc[1], kind='linear')
        for i in range(len(energy_loss_array)):
            if energy_loss_array[i] < energy_fOsc[0][0]:
                f[i] = 0
            elif energy_loss_array[i] <= energy_fOsc[0][-1]:
                f[i] = fData(energy_loss_array[i])
            else:
                f[i] = ComplexLineShapeUtilities.aseev_func_tail(energy_loss_array[i], gas_type)

        f_e_loss = ComplexLineShapeUtilities.get_eloss_spec(energy_loss_array, f, self.Kr_K_line_eV)
        f_normed = self.normalize(f_e_loss)
        return f_normed

    # Convolves a function with the single scatter function, on the SELA
    def another_scatter(self, input_spectrum, gas_type):
        single = self.single_scatter_f(gas_type)
        f = signal.convolve(single,input_spectrum,mode='same')
        f_normed = self.normalize(f)
        return f_normed

    def radiation_loss_f(self):
        radiation_loss_data_file_path = self.path_to_missing_track_radiation_loss_data_numpy_file + '/missing_track_radiation_loss.npy'
        data_for_missing_track_radiation_loss = np.load(radiation_loss_data_file_path, allow_pickle = True)
        x_data_for_histogram = data_for_missing_track_radiation_loss.item()['histogram_eV']['x_data']
        energy_loss_array = self.std_eV_array()
        f_radiation_energy_loss = 0 * energy_loss_array
        f_radiation_energy_loss_interp = data_for_missing_track_radiation_loss.item()['histogram_eV']['interp']
        for i in range(len(energy_loss_array)):
            if energy_loss_array[i] >= x_data_for_histogram[0] and energy_loss_array[i] <= x_data_for_histogram[-1]:
                f_radiation_energy_loss[i] = f_radiation_energy_loss_interp(energy_loss_array[i])
            else:
                f_radiation_energy_loss[i] = 0
        return f_radiation_energy_loss

    # Convolves the scatter functions and saves
    # the results to a .npy file.
    def generate_scatter_convolution_file(self):
        t = time.time()
        scatter_spectra_single_gas = {}
        for gas_type in self.gases:
            scatter_spectra_single_gas[gas_type] = {}
            first_scatter = self.single_scatter_f(gas_type)
            if self.use_radiation_loss == True:
                f_radiation_loss = self.radiation_loss_f()
                first_scatter = self.normalize(signal.convolve(first_scatter, f_radiation_loss, mode = 'same'))
            scatter_num_array = range(2, self.max_scatters+1)
            current_scatter = first_scatter
            scatter_spectra_single_gas[gas_type][str(1).zfill(2)] = current_scatter
            # x = std_eV_array() # diagnostic
            for i in scatter_num_array:
                current_scatter = self.another_scatter(current_scatter, gas_type)
                if self.use_radiation_loss == True:
                    f_radiation_loss = self.radiation_loss_f()
                    current_scatter = self.normalize(signal.convolve(current_scatter, f_radiation_loss, mode = 'same'))
                scatter_spectra_single_gas[gas_type][str(i).zfill(2)] = current_scatter
        N = len(self.gases)
        scatter_spectra = {}
        for M in range(1, self.max_scatters + 1):
            gas_scatter_combinations = np.array([np.array(i) for i in product(range(M+1), repeat=N) if sum(i)==M])
            for combination in gas_scatter_combinations:
                mark_first_nonzero_component = 0
                entry_str = ''
                for component, gas_type in zip(combination, self.gases):
                    entry_str += gas_type
                    entry_str += str(component).zfill(2)
                    if component == 0:
                        continue
                    else:
                        if mark_first_nonzero_component == 0:
                            current_full_scatter = scatter_spectra_single_gas[gas_type][str(component).zfill(2)]
                            mark_first_nonzero_component = 1
                        else:
                            scatter_to_add = scatter_spectra_single_gas[gas_type][str(component).zfill(2)]
                            current_full_scatter = self.normalize(signal.convolve(current_full_scatter, scatter_to_add, mode='same'))
                scatter_spectra[entry_str] = current_full_scatter
        np.save(os.path.join(self.path_to_scatter_spectra_file, 'scatter_spectra.npy'), scatter_spectra)
        elapsed = time.time() - t
        logger.info('Files generated in '+str(elapsed)+'s')
        return

    # Checks for the existence of a directory called 'scatter_spectra_file'
    # and checks that this directory contains the scatter spectra files.
    # If not, this function calls generate_scatter_convolution_file.
    # This function also checks to make sure that the scatter file have the correct
    # number of entries and correct number of points in the SELA, and if not, it generates a fresh file.
    # When the variable regenerate is set as True, it generates a fresh file
    def check_existence_of_scatter_file(self, regenerate = True):
        gases = self.gases
        if regenerate == True:
            logger.info('generate fresh scatter file')
            self.generate_scatter_convolution_file()
        else:
            directory = os.listdir(self.path_to_scatter_spectra_file)
            strippeddirs = [s.strip('\n') for s in directory]
            if 'scatter_spectra.npy' not in strippeddirs:
                self.generate_scatter_convolution_file()
            test_file = os.path.join(self.path_to_scatter_spectra_file, 'scatter_spectra.npy')
            test_dict = np.load(test_file, allow_pickle = True)
            N = len(self.gases)
            if len(test_dict.item()) != sum([comb(M + N -1, N -1) for M in range(1, self.max_scatters+1)]):
                logger.info('Number of scatter combinations not matching, generating fresh files')
                self.generate_scatter_convolution_file()
                test_dict = np.load(test_file, allow_pickle = True)
            gas_str = gases[0] + '01'
            for gas in self.gases[1:]:
                gas_str += gas + '00'
            if gas_str not in list(test_dict.item().keys()):
                print('Gas species not matching, generating fresh files')
                self.generate_scatter_convolution_files()
        return

    # Given a function evaluated on the SELA, convolves it with a gaussian
    def convolve_gaussian(self, func_to_convolve, gauss_FWHM_eV):
        sigma = ComplexLineShapeUtilities.gaussian_FWHM_to_sigma(gauss_FWHM_eV)
        resolution_f = self.std_gaussian(sigma)
        ans = signal.convolve(resolution_f, func_to_convolve,mode='same')
        ans_normed = self.normalize(ans)
        return ans_normed

    def convolve_composite_gaussian(self, func_to_convolve):
        resolution_f = self.composite_gaussian()
        ans = signal.convolve(resolution_f, func_to_convolve, mode = 'same')
        ans_normed = self.normalize(ans)
        return ans_normed

    #This is used for convolve_ins_resolution below
    def read_ins_resolution_data(self, path_to_ins_resolution_data_txt):
        ins_resolution_data = np.loadtxt(path_to_ins_resolution_data_txt)
        x_data = ins_resolution_data.T[0]
        y_data = ins_resolution_data.T[1]
        y_err_data = np.zeros(len(y_data))
        y_err_data = ins_resolution_data.T[2]
        x_data = ComplexLineShapeUtilities.flip_array(-1*x_data)
        y_data = ComplexLineShapeUtilities.flip_array(y_data)
        y_err_data = ComplexLineShapeUtilities.flip_array(y_err_data)
        return x_data, y_data, y_err_data

    def convolve_ins_resolution(self, working_spectrum):
        x_data, y_mean_data, y_err_data = self.read_ins_resolution_data(self.path_to_ins_resolution_data_txt)        
        if self.sample_ins_resolution_errors:
            y_data = np.random.normal(y_mean_data)
        else:
            y_data = y_mean_data
        y_data[y_data<0] = 0
        f = interpolate.interp1d(x_data, y_data)
        x_array = self.std_eV_array()
        y_array = np.zeros(len(x_array))
        index_within_range_of_xdata = np.where((x_array >= x_data[0]) & (x_array <= x_data[-1]))
        y_array[index_within_range_of_xdata] = f(x_array[index_within_range_of_xdata])
        convolved_spectrum = signal.convolve(working_spectrum, y_array, mode = 'same')
        normalized_convolved_spectrum = self.normalize(convolved_spectrum)
        return normalized_convolved_spectrum

    # This is used for convolve_ins_resolution_combining_four_trap, below
    def combine_four_trap_resolution_from_txt(self, trap_weights):
        if self.sample_ins_resolution_errors:
            weight_array = np.random.normal(trap_weights['weights'], trap_weights['errors'])
        else:
            weight_array = trap_weights['weights']
        y_data_array = []
        y_err_data_array = []
        for path_to_single_trap_resolution_txt in self.path_to_four_trap_ins_resolution_data_txt:
            x_data, y_data, y_err_data = self.read_ins_resolution_data(path_to_single_trap_resolution_txt)
            y_data_array.append(y_data)
            y_err_data_array.append(y_err_data)
        y_data_combined = weight_array[0]*y_data_array[0] + weight_array[1]*y_data_array[1] + weight_array[2]*y_data_array[2] + weight_array[3]*y_data_array[3]
        y_err_data_combined = np.sqrt((weight_array[0]*y_err_data_array[0])**2 + (weight_array[1]*y_err_data_array[1])**2 + (weight_array[2]*y_err_data_array[2])**2 + (weight_array[3]*y_err_data_array[3])**2)
        return x_data, y_data_combined, y_err_data_combined

    def convolve_ins_resolution_combining_four_trap(self, working_spectrum, weight_array):
        x_data, y_data_combined, y_err_data_combined = self.combine_four_trap_resolution_from_txt(weight_array)
        if self.sample_ins_resolution_errors:
            y_data_combined = np.random.normal(y_data_combined, y_err_data_combined)
        f = interpolate.interp1d(x_data, y_data_combined)
        x_array = self.std_eV_array()
        y_array = np.zeros(len(x_array))
        index_within_range_of_xdata = np.where((x_array >= x_data[0]) & (x_array <= x_data[-1]))
        y_array[index_within_range_of_xdata] = f(x_array[index_within_range_of_xdata])
        convolved_spectrum = signal.convolve(working_spectrum, y_array, mode = 'same')
        normalized_convolved_spectrum = self.normalize(convolved_spectrum)
        return normalized_convolved_spectrum

    def convolve_simulated_resolution_scaled(self, working_spectrum, scale_factor):
        """if self.use_combined_four_trap_inst_reso:
            x_data, y_data, y_err_data = self.combine_four_trap_resolution_from_txt(self.trap_weights)
            logger.info("Combined four instrumental resolution files")
        else:
            x_data, y_data, y_err_data = self.read_ins_resolution_data(self.path_to_ins_resolution_data_txt)
            logger.info("Using ONE simulated instrumental resolution file (not combining four)")
        if self.sample_ins_resolution_errors:
            y_data = np.random.normal(y_data, y_err_data)
            logger.info("Sampling instrumental resolution counts per bin")
        scaled_xdata = x_data*scale_factor
        f = interpolate.interp1d(x_data*scale_factor, y_data)"""

        x_array = self.std_eV_array()
        y_array = np.zeros(len(x_array))

        #index_within_range_of_xdata = np.where((x_array >= scaled_xdata[0]) & (x_array <= scaled_xdata[-1]))
        #y_array[index_within_range_of_xdata] = f(x_array[index_within_range_of_xdata]/scale_factor)
        y_array = self.interpolated_resolution(x_array/scale_factor)
        convolved_spectrum = signal.convolve(working_spectrum, y_array, mode = 'same')
        normalized_convolved_spectrum = self.normalize(convolved_spectrum)
        return normalized_convolved_spectrum

    # Not currently used; we used chi2 functions instead (see below)
    def least_square(self, bin_centers, hist, params):
        # expectation
        expectation = self.spectrum_func_ftc(bin_centers, *params)

        high_count_index = np.where(hist>0)
        #low_count_index = np.where((hist>0) & (hist<=50))
        zero_count_index = np.where(hist==0)

        lsq = ((hist[high_count_index]- expectation[high_count_index])**2/hist[high_count_index]).sum()
        #lsq += ((hist[low_count_index]- expectation[low_count_index])**2/hist[low_count_index]).sum()
        lsq += ((hist[zero_count_index]- expectation[zero_count_index])**2).sum()
        return lsq

    def chi2_Poisson(self, bin_centers, data_hist_freq, params):
        nonzero_bins_index = np.where(data_hist_freq != 0)
        zero_bins_index = np.where(data_hist_freq == 0)
        # expectation
        if self.resolution_function == 'simulated_resolution':
            if self.fixed_scatter_proportion:
                fit_Hz = self.spectrum_func_ftc(bin_centers, *params)
            else:
                fit_Hz = self.spectrum_func_ftc_2(bin_centers, *params)
        chi2 = 2*((fit_Hz - data_hist_freq + data_hist_freq*np.log(data_hist_freq/fit_Hz))[nonzero_bins_index]).sum()
        chi2 += 2*(fit_Hz - data_hist_freq)[zero_bins_index].sum()
        return chi2

    def chi_2_Poisson_composite_gaussian_reso(self, bin_centers, data_hist_freq, eff_array, params):
        nonzero_bins_index = np.where(data_hist_freq != 0)
        zero_bins_index = np.where(data_hist_freq == 0)
        # expectation
        fit_Hz = self.spectrum_func_composite_gaussian_fixed_scatter_proportion(bin_centers, eff_array, *params)
        chi2 = 2*((fit_Hz - data_hist_freq + data_hist_freq*np.log(data_hist_freq/fit_Hz))[nonzero_bins_index]).sum()
        chi2 += 2*(fit_Hz - data_hist_freq)[zero_bins_index].sum()
        return chi2

    def chi_2_Poisson_simulated_resolution_scaled(self, bin_centers, data_hist_freq, eff_array, params):
        # expectation
        fit_Hz = self.spectrum_func_simulated_resolution_scaled_fixed_scatter_proportion(bin_centers, eff_array, *params)
        nonzero_bins_index = np.where((data_hist_freq != 0) & (fit_Hz != 0))
        zero_bins_index = np.where((data_hist_freq == 0) | (fit_Hz == 0))
        chi2 = 2*((fit_Hz - data_hist_freq + data_hist_freq*np.log(data_hist_freq/fit_Hz))[nonzero_bins_index]).sum()
        chi2 += 2*(fit_Hz - data_hist_freq)[zero_bins_index].sum()
        return chi2


    # following the expression in the paper Steve BAKER and Robert D. COUSINS, (1984) CLARIFICATION OF THE USE OF CHI-SQUARE AND LIKELIHOOD FUNCTIONS IN FITS TO HISTOGRAMS
    # This is used in fit_data_ftc and fit_data_ftc_2
    def reduced_chi2_Poisson(self, data_hist_freq, fit_Hz, number_of_parameters):
        nonzero_bins_index = np.where(data_hist_freq != 0)
        zero_bins_index = np.where(data_hist_freq == 0)
        chi2 = 2*((fit_Hz - data_hist_freq + data_hist_freq*np.log(data_hist_freq/fit_Hz))[nonzero_bins_index]).sum()
        chi2 += 2*(fit_Hz - data_hist_freq)[zero_bins_index].sum()
        reduced_chi2 = chi2/(len(data_hist_freq) - number_of_parameters)
        return reduced_chi2


    #There are a few additional chi2 functions below, near the associated make_spectrum functions.


    # using simulated resolution, with multi gas scattering, reconstruction eff, without detection eff, has been used in fake data generator. However, without using detection eff is the right option for tritium fake data generation.
    def make_spectrum_ftc(self, survival_prob, emitted_peak='shake'):
        gases = self.gases
        current_path = self.path_to_scatter_spectra_file
        # check_existence_of_scatter_files()
        #filenames = list_files('scatter_spectra_files')
        p = self.scatter_proportion
        a = self.recon_eff_param_a
        b = self.recon_eff_param_b
        c = self.recon_eff_param_c
        scatter_spectra_file_path = os.path.join(current_path, 'scatter_spectra.npy')
        scatter_spectra = np.load(scatter_spectra_file_path, allow_pickle = True)
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        emitted_peak = self.base_shape
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        elif emitted_peak == 'dirac':
            current_working_spectrum = self.std_dirac()
        if self.use_combined_four_trap_inst_reso:
            current_working_spectrum = self.convolve_ins_resolution_combining_four_trap(current_working_spectrum, self.trap_weights)
        else:
            current_working_spectrum = self.convolve_ins_resolution(current_working_spectrum)
        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += current_working_spectrum
        N = len(self.gases)
        for M in range(1, self.max_scatters + 1):
            relative_reconstruction_eff = np.exp(-b*M**c)
            gas_scatter_combinations = np.array([np.array(i) for i in product(range(M+1), repeat=N) if sum(i)==M])
            for combination in gas_scatter_combinations:
                entry_str = ''
                for component, gas_type in zip(combination, self.gases):
                    entry_str += gas_type
                    entry_str += str(component).zfill(2)
                current_working_spectrum = scatter_spectra.item()[entry_str]
                current_working_spectrum = self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
                coefficient = factorial(sum(combination))
                for component, i in zip(combination, range(len(self.gases))):
                    coefficient = coefficient/factorial(component)*p[i]**component
                for i in range(0, M):
                    coefficient = coefficient*(1-a*np.exp(-b*i**c))
                current_full_spectrum += relative_reconstruction_eff*coefficient*current_working_spectrum*survival_prob**M
        return current_full_spectrum


    # Produces energy spectrum based on make_spectrum_ftc(), where the scatter peak amplitude curve was modeled by the product of a modified exponential function and an exponential function
    # Produces a spectrum in real energy that can now be evaluated off of the SELA.
    def spectrum_func_ftc(self, bins_Hz, *p0):
        B_field = p0[0]
        amplitude = p0[1]
        survival_prob = p0[2]
        x_eV = ConversionFunctions.Energy(bins_Hz, B_field)
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        f = np.zeros(len(x_eV))
        f_intermediate = np.zeros(len(x_eV))

        x_eV_minus_line = self.Kr_K_line_eV - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0], np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_ftc(survival_prob)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx], en_loss_array, full_spectrum)
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])
        return f

    # Perform fit on a frequency histogram with spectrum_fuc_ftc()
    # Call this function to fit a histogram of start frequencies with the model.
    # Note that the data_hist_freq should be the StartFrequencies as given by katydid,
    # which will be from ~0 MHZ to ~100 MHz. You must also pass this function the
    # self.RF_ROI_MIN value from the metadata file of your data.
    # You must also supply a guess for the self.B_field present for the run;
    # 0.959 T is usually sufficient.
    def fit_data_ftc(self, freq_bins, data_hist_freq):
        t = time.time()
        self.check_existence_of_scatter_file()
        bins_Hz = freq_bins + self.RF_ROI_MIN
        bins_Hz = 0.5*(bins_Hz[1:] + bins_Hz[:-1])
        bins_Hz_nonzero , data_hist_nonzero , data_hist_err = ComplexLineShapeUtilities.get_only_nonzero_bins(bins_Hz, data_hist_freq)
        # Initial guesses for curve_fit
        B_field_guess = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[np.argmax(data_hist_freq)])
        amplitude_guess = np.sum(data_hist_freq)/2
        prob_parameter_guess = 0.5
        # Bounds for curve_fit
        B_field_min = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[0])
        B_field_max = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[-1])
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist_freq)*3
        prob_parameter_min = 1e-5
        prob_parameter_max = 1

        p0_guess = [B_field_guess, amplitude_guess, prob_parameter_guess]
        p0_bounds = [(B_field_min,B_field_max), (amplitude_min,amplitude_max), (prob_parameter_min, prob_parameter_max)]
        # Actually do the fitting
        m_binned = Minuit.from_array_func(lambda p: self.chi2_Poisson(bins_Hz, data_hist_freq, p),
                                        start = p0_guess,
                                        limit = p0_bounds,
                                        throw_nan = True
                                        )
        m_binned.migrad()
        params = m_binned.np_values()
        # Name each of the resulting parameters and errors
        ################### Generalize to N Gases ###########################
        B_field_fit = params[0]
        amplitude_fit = params[1]
        survival_prob_fit = params[2]
        total_counts_fit = amplitude_fit

        perr = m_binned.np_errors()
        B_field_fit_err = perr[0]
        amplitude_fit_err = perr[1]
        survival_prob_fit_err = perr[2]
        total_counts_fit_err = amplitude_fit_err

        fit_Hz = self.spectrum_func_ftc(bins_Hz, *params)
        fit_keV = ComplexLineShapeUtilities.flip_array(fit_Hz)
        bins_keV = ConversionFunctions.Energy(bins_Hz, B_field_fit)/1000
        bins_keV = ComplexLineShapeUtilities.flip_array(bins_keV)

        reduced_chi2 = self.reduced_chi2_Poisson(data_hist_freq, fit_Hz, number_of_parameters = 3)
        elapsed = time.time() - t
        output_string = '\n'
        output_string += 'Reduced chi^2 = {:.2e}\n'.format(reduced_chi2)
        output_string += '-----------------\n'
        output_string += 'B field = {:.8e}'.format(B_field_fit)+' +/- '+ '{:.4e} T\n'.format(B_field_fit_err)
        output_string += '-----------------\n'
        output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
        output_string += '-----------------\n'
        output_string += 'Survival probability \n= ' + "{:.8e}".format(survival_prob_fit)\
        +' +/- ' + "{:.6e}".format(survival_prob_fit_err)+'\n'
        output_string += '-----------------\n'
        output_string += 'Fit completed in '+str(round(elapsed,2))+'s'+'\n'
        dictionary_of_fit_results = {
        'output_string': output_string,
        'perr': perr,
        'bins_keV': bins_keV,
        'fit_keV': fit_keV,
        'bins_Hz': bins_Hz,
        'fit_Hz': fit_Hz,
        'B_field_fit': B_field_fit,
        'B_field_fit_err': B_field_fit_err,
        'survival_prob_fit': survival_prob_fit,
        'survival_prob_fit_err': survival_prob_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2
        }
        return dictionary_of_fit_results

    # simulated resolution with scatter_proportion floating, without reconstruction eff curve, without detection eff curve
    def make_spectrum_ftc_2(self, prob_parameter, scatter_proportion, emitted_peak='shake'):
        gases = self.gases
        current_path = self.path_to_scatter_spectra_file
        # check_existence_of_scatter_files()
        #filenames = list_files('scatter_spectra_files')
        p = np.zeros(len(gases))
        p[0:-1] = scatter_proportion
        p[-1] = 1 - sum(scatter_proportion)
        scatter_spectra_file_path = os.path.join(current_path, 'scatter_spectra.npy')
        scatter_spectra = np.load(
        scatter_spectra_file_path, allow_pickle = True
        )
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        emitted_peak = self.base_shape
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        elif emitted_peak == 'dirac':
            current_working_spectrum = self.std_dirac()

        if self.use_combined_four_trap_inst_reso:
            current_working_spectrum = self.convolve_ins_resolution_combining_four_trap(current_working_spectrum, self.trap_weights)
        else:
            current_working_spectrum = self.convolve_ins_resolution(current_working_spectrum)

        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += current_working_spectrum
        N = len(self.gases)
        for M in range(1, self.max_scatters + 1):
            gas_scatter_combinations = np.array([np.array(i) for i in product(range(M+1), repeat=N) if sum(i)==M])
            for combination in gas_scatter_combinations:
                entry_str = ''
                for component, gas_type in zip(combination, self.gases):
                    entry_str += gas_type
                    entry_str += str(component).zfill(2)
                current_working_spectrum = scatter_spectra.item()[entry_str]
                current_working_spectrum = self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
                coefficient = factorial(sum(combination))
                for component, i in zip(combination, range(len(self.gases))):
                    coefficient = coefficient/factorial(component)*p[i]**component
                current_full_spectrum += coefficient*current_working_spectrum*prob_parameter**M
        return current_full_spectrum

    # spectrum_func_ftc2() is constructing an energy spectrum based on the make_spectrum_ftc_2() function.
    # Difference between spectrum_func_ftc() and spectrum_func_ftc_2() is that the modified exponential function is not included in the scatter peak amplitude curve in spectrum_func_ftc_2() [2024-07-12 Fri]
    # The way to take into account of the different possible sequences of the scattering with multiple gas species is outdated. [2024-07-12 Fri]
    def spectrum_func_ftc_2(self, bins_Hz, *p0):
        B_field = p0[0]
        amplitude = p0[1]
        prob_parameter = p0[2]
        N = len(self.gases)
        scatter_proportion = p0[3:2+N]

        x_eV = ConversionFunctions.Energy(bins_Hz, B_field)
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        en_array_rev = ComplexLineShapeUtilities.flip_array(-1*en_loss_array)
        f = np.zeros(len(x_eV))
        f_intermediate = np.zeros(len(x_eV))

        x_eV_minus_line = self.Kr_K_line_eV - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_ftc_2(prob_parameter, scatter_proportion)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx],en_array_rev,full_spectrum)
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])
        return f

    # Fit frequency spectrum with spectrum_func_ftc_2()
    def fit_data_ftc_2(self, freq_bins, data_hist_freq):
        t = time.time()
        self.check_existence_of_scatter_file()
        bins_Hz = freq_bins + self.RF_ROI_MIN
        bins_Hz = 0.5*(bins_Hz[1:] + bins_Hz[:-1])
        bins_Hz_nonzero , data_hist_nonzero , data_hist_err = ComplexLineShapeUtilities.get_only_nonzero_bins(bins_Hz, data_hist_freq)
        # Initial guesses for curve_fit
        B_field_guess = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[np.argmax(data_hist_freq)])
        amplitude_guess = np.sum(data_hist_freq)/2
        prob_parameter_guess = 0.5
        scatter_proportion_guess = 0.5
        # Bounds for curve_fit
        B_field_min = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[0])
        B_field_max = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[-1])
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist_freq)*3
        prob_parameter_min = 1e-5
        prob_parameter_max = 1
        scatter_proportion_min = 1e-5
        scatter_proportion_max = 1
        N = len(self.gases)
        p0_guess = [B_field_guess, amplitude_guess, prob_parameter_guess] + (N-1)*[scatter_proportion_guess]
        p0_bounds = [(B_field_min,B_field_max), (amplitude_min,amplitude_max), (prob_parameter_min, prob_parameter_max)] + (N-1)*[(scatter_proportion_min, scatter_proportion_max)]
        logger.info(p0_guess)
        logger.info(p0_bounds)
        # Actually do the fitting
        m_binned = Minuit.from_array_func(lambda p: self.chi2_Poisson(bins_Hz, data_hist_freq, p),
                                        start = p0_guess,
                                        limit = p0_bounds,
                                        throw_nan = True
                                        )
        m_binned.migrad()
        params = m_binned.np_values()
        # Name each of the resulting parameters and errors
        ################### Generalize to N Gases ###########################
        B_field_fit = params[0]
        amplitude_fit = params[1]
        prob_parameter_fit = params[2]
        scatter_proportion_fit = list(params[3:2+N])+[1- sum(params[3:2+N])]
        total_counts_fit = amplitude_fit

        perr = m_binned.np_errors()
        B_field_fit_err = perr[0]
        amplitude_fit_err = perr[1]
        prob_parameter_fit_err = perr[2]
        scatter_proportion_fit_err = list(perr[3:2+N])+[np.sqrt(sum(perr[3:2+N]**2))]
        total_counts_fit_err = amplitude_fit_err

        fit_Hz = self.spectrum_func_ftc_2(bins_Hz, *params)
        fit_keV = ComplexLineShapeUtilities.flip_array(fit_Hz)
        bins_keV = ConversionFunctions.Energy(bins_Hz, B_field_fit)/1000
        bins_keV = ComplexLineShapeUtilities.flip_array(bins_keV)

        reduced_chi2 = self.reduced_chi2_Poisson(data_hist_freq, fit_Hz, number_of_parameters = 4)
        elapsed = time.time() - t
        output_string = '\n'
        output_string += 'Reduced chi^2 = {:.2e}\n'.format(reduced_chi2)
        output_string += '-----------------\n'
        output_string += 'B field = {:.8e}'.format(B_field_fit)+' +/- '+ '{:.4e} T\n'.format(B_field_fit_err)
        output_string += '-----------------\n'
        output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
        output_string += '-----------------\n'
        output_string += 'Probability parameter \n= ' + "{:.2e}".format(prob_parameter_fit)+' +/- ' + "{:.2e}".format(prob_parameter_fit_err)+'\n'
        output_string += '-----------------\n'
        output_string += ''
        for i in range(len(self.gases)):
            output_string += '{} Scatter proportion \n= '.format(self.gases[i]) + "{:.8e}".format(scatter_proportion_fit[i])\
            +' +/- ' + "{:.2e}".format(scatter_proportion_fit_err[i])+'\n'
            output_string += '-----------------\n'
        output_string += 'Fit completed in '+str(round(elapsed,2))+'s'+'\n'
        dictionary_of_fit_results = {
        'output_string': output_string,
        'perr': perr,
        'bins_keV': bins_keV,
        'fit_keV': fit_keV,
        'bins_Hz': bins_Hz,
        'fit_Hz': fit_Hz,
        'B_field_fit': B_field_fit,
        'B_field_fit_err': B_field_fit_err,
        'survival_prob_fit': prob_parameter_fit,
        'survival_prob_fit_err': prob_parameter_fit_err,
        'scatter_proportion_fit': scatter_proportion_fit,
        'scatter_proportion_fit_err': scatter_proportion_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2
        }
        return dictionary_of_fit_results


    # fitting with superposition of gaussians as resolution function
    def make_spectrum_composite_gaussian_fixed_scatter_proportion(self, survival_prob, emitted_peak='shake'):
        p = self.scatter_proportion
        a = self.recon_eff_param_a
        b = self.recon_eff_param_b
        c = self.recon_eff_param_c
        scatter_spectra_file_path = os.path.join(self.path_to_scatter_spectra_file, 'scatter_spectra.npy')
        scatter_spectra = np.load(scatter_spectra_file_path, allow_pickle = True)
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        current_working_spectrum = self.convolve_composite_gaussian(current_working_spectrum)
        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += zeroth_order_peak
        N = len(self.gases)
        for M in range(1, self.max_scatters + 1):
            relative_reconstruction_eff = np.exp(-b*M**c)
            gas_scatter_combinations = np.array([np.array(i) for i in product(range(M+1), repeat=N) if sum(i)==M])
            for combination in gas_scatter_combinations:
                #print(combination)
                entry_str = ''
                for component, gas_type in zip(combination, self.gases):
                    entry_str += gas_type
                    entry_str += str(component).zfill(2)
                current_working_spectrum = scatter_spectra.item()[entry_str]
                current_working_spectrum = self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
                coefficient = factorial(sum(combination))
                for component, i in zip(combination, range(N)):
                    coefficient = coefficient/factorial(component)*p[i]**component
                for i in range(0, M):
                    coefficient = coefficient*(1-a*np.exp(-b*i**c))
                current_full_spectrum += relative_reconstruction_eff*coefficient*current_working_spectrum*survival_prob**M
        return current_full_spectrum

    def spectrum_func_composite_gaussian_fixed_scatter_proportion(self, bins_Hz, eff_array, *p0):

        B_field = p0[0]
        amplitude = p0[1]
        survival_prob = p0[2]

        x_eV = ConversionFunctions.Energy(bins_Hz, B_field)
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        f = np.zeros(len(x_eV))
        f_intermediate = np.zeros(len(x_eV))

        x_eV_minus_line = self.Kr_K_line_eV - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_composite_gaussian_fixed_scatter_proportion(survival_prob)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx], en_loss_array, full_spectrum)
        f_intermediate = f_intermediate*eff_array
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])

        return f

    def fit_data_composite_gaussian_fixed_scatter_proportion(self, freq_bins, data_hist_freq, print_params=True):
        t = time.time()
        self.check_existence_of_scatter_file()
        bins_Hz = freq_bins + self.RF_ROI_MIN
        bins_Hz = 0.5*(bins_Hz[1:] + bins_Hz[:-1])

        quad_trap_interp = np.load(self.path_to_quad_trap_eff_interp, allow_pickle = True)
        quad_trap_count_rate_interp = quad_trap_interp.item()['count_rate_interp']
        eff_array = quad_trap_count_rate_interp(bins_Hz)
        # Initial guesses for curve_fit
        B_field_guess = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[np.argmax(data_hist_freq)])
        amplitude_guess = np.sum(data_hist_freq)/2
        FWHM_eV_guess = 5
        prob_parameter_guess = 0.5
        scatter_proportion_guess = 0.5
        sigma_guess = 5
        gamma_guess = 3
        gaussian_portion_guess = 0.5
        elevation_factor_guess = 20
        # Bounds for curve_fit
        B_field_min = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[0])
        B_field_max = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[-1])
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist_freq)*3
        FWHM_eV_min = 0
        FWHM_eV_max = ConversionFunctions.Energy(bins_Hz[0], B_field_guess)
        prob_parameter_min = 1e-5
        prob_parameter_max = 1
        scatter_proportion_min = 1e-5
        scatter_proportion_max = 1
        mu_min = -FWHM_eV_max
        mu_max = FWHM_eV_max
        gaussian_portion_min = 1e-5
        gaussian_portion_max = 1
        elevation_factor_min = 0
        elevation_factor_max = 500
        N = len(self.gases)
        p0_guess = [B_field_guess, amplitude_guess, prob_parameter_guess]
        p0_bounds = [(B_field_min,B_field_max), (amplitude_min,amplitude_max), (prob_parameter_min, prob_parameter_max)]
        # Actually do the fitting
        m_binned = Minuit.from_array_func(lambda p: self.chi_2_Poisson_composite_gaussian_reso(bins_Hz, data_hist_freq, eff_array, p),
                                          start = p0_guess,
                                          limit = p0_bounds,
                                          throw_nan = True
                                          )
        m_binned.migrad()
        params = m_binned.np_values()
        B_field_fit = params[0]
        #starting at index 2, grabs every other entry. (which is how scattering probs are filled in for N gases)
        amplitude_fit = params[1]
        survival_prob_fit = params[2]
        total_counts_fit = amplitude_fit

        perr = m_binned.np_errors()
        B_field_fit_err = perr[0]
        amplitude_fit_err = perr[1]
        survival_prob_fit_err = perr[2]
        total_counts_fit_err = amplitude_fit_err

        fit_Hz = self.spectrum_func_composite_gaussian_fixed_scatter_proportion(bins_Hz, eff_array, *params)
        fit_keV = ComplexLineShapeUtilities.flip_array(fit_Hz)
        bins_keV = ConversionFunctions.Energy(bins_Hz, B_field_fit)/1000
        bins_keV = ComplexLineShapeUtilities.flip_array(bins_keV)
        reduced_chi2 = m_binned.fval/(len(fit_Hz)-m_binned.nfit)

        if print_params == True:
            output_string = '\n'
            output_string += 'Reduced chi^2 = {:.2e}\n'.format(reduced_chi2)
            output_string += '-----------------\n'
            output_string += 'B field = {:.8e}'.format(B_field_fit)+' +/- '+ '{:.4e} T\n'.format(B_field_fit_err)
            output_string += '-----------------\n'
            output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
            output_string += '-----------------\n'
            output_string += 'Survival probability = {:.8e}'.format(survival_prob_fit) + ' +/- {:.8e}\n'.format(survival_prob_fit_err)
            output_string += '-----------------\n'
        elapsed = time.time() - t
        output_string += 'Fit completed in '+str(round(elapsed,2))+'s'+'\n'
        dictionary_of_fit_results = {
        'output_string': output_string,
        'perr': perr,
        'bins_keV': bins_keV,
        'fit_keV': fit_keV,
        'bins_Hz': bins_Hz,
        'fit_Hz': fit_Hz,
        'B_field_fit': B_field_fit,
        'B_field_fit_err': B_field_fit_err,
        'survival_prob_fit': survival_prob_fit,
        'survival_prob_fit_err': survival_prob_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2
        }
        return dictionary_of_fit_results
    

    # This is one of the two functions called in the tritium fake data generator, for the
    # final Project 8 Phase II analysis:
    # https://github.com/project8/mermithid/blob/combining_ComplexLineShape_and_FakeDataGenerator/mermithid/misc/FakeTritiumDataFunctions.py#L317
    # This function is *not* used for the main Phase II Kr analyses.
    def make_spectrum_simulated_resolution_scaled_fit_scatter_peak_ratio(self, scale_factor, survival_probability, scatter_peak_ratio_p, scatter_peak_ratio_q, scatter_fraction, emitted_peak='shake'):
        p = np.zeros(len(self.gases))
        p[0:-1] = scatter_fraction
        p[-1] = 1 - sum(scatter_fraction)
        scatter_spectra_file_path = os.path.join(self.path_to_scatter_spectra_file, 'scatter_spectra.npy')
        scatter_spectra = np.load(scatter_spectra_file_path, allow_pickle = True)
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        elif emitted_peak == 'dirac':
            current_working_spectrum = self.std_dirac()
        current_working_spectrum = self.convolve_simulated_resolution_scaled(current_working_spectrum, scale_factor)
        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += zeroth_order_peak
        N = len(self.gases)
        for M in range(1, self.max_scatters + 1):
            scatter_peak_ratio = np.exp(-1.*scatter_peak_ratio_p*M**( -self.factor*scatter_peak_ratio_p + scatter_peak_ratio_q))
            gas_scatter_combinations = np.array([np.array(i) for i in product(range(M+1), repeat=N) if sum(i)==M])
            for combination in gas_scatter_combinations:
                entry_str = ''
                for component, gas_type in zip(combination, self.gases):
                    entry_str += gas_type
                    entry_str += str(component).zfill(2)
                current_working_spectrum = scatter_spectra.item()[entry_str]
                current_working_spectrum = self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
                coefficient = factorial(sum(combination))
                for component, i in zip(combination, range(N)):
                    coefficient = coefficient/factorial(component)*p[i]**component
                current_full_spectrum += coefficient*current_working_spectrum*scatter_peak_ratio*survival_probability**M
        return current_full_spectrum

    def spectrum_func_simulated_resolution_scaled_fit_scatter_peak_ratio(self, bins_Hz, eff_array, *p0):

        B_field = p0[0]
        amplitude = p0[1]
        scale_factor = p0[2]
        survival_probability = p0[3]
        scatter_peak_ratio_p = p0[4]
        scatter_peak_ratio_q = p0[5]
        N = len(self.gases)
        scatter_fraction = p0[6:5+N]

        x_eV = ConversionFunctions.Energy(bins_Hz, B_field)
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        f = np.zeros(len(x_eV))
        f_intermediate = np.zeros(len(x_eV))

        x_eV_minus_line = self.Kr_K_line_eV - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_simulated_resolution_scaled_fit_scatter_peak_ratio(scale_factor, survival_probability, scatter_peak_ratio_p, scatter_peak_ratio_q, scatter_fraction)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx], en_loss_array, full_spectrum)
        f_intermediate = f_intermediate*eff_array
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])

        return f

    def chi_2_simulated_resolution_scaled_fit_scatter_peak_ratio(self, bin_centers, data_hist_freq, eff_array, params):
        # expectation
        fit_Hz = self.spectrum_func_simulated_resolution_scaled_fit_scatter_peak_ratio(bin_centers, eff_array, *params)
        nonzero_bins_index = np.where((data_hist_freq != 0) & (fit_Hz > 0))
        zero_bins_index = np.where((data_hist_freq == 0) | (fit_Hz <= 0))
        chi2 = 2*((fit_Hz - data_hist_freq + data_hist_freq*np.log(data_hist_freq/fit_Hz))[nonzero_bins_index]).sum()
        chi2 += 2*(fit_Hz - data_hist_freq)[zero_bins_index].sum()
        return chi2


    #In Project 8's Phase II analysis, this function was used for a fit of Kr data
    #performed just before the tritium analysis. This Kr fit served as a double-check
    #of the official Kr analysis (performed with a different function, below).
    def fit_data_simulated_resolution_scaled_fit_scatter_peak_ratio(self, freq_bins, data_hist_freq, print_params=True):
        t = time.time()
        self.check_existence_of_scatter_file()
        bins_Hz = freq_bins + self.RF_ROI_MIN
        bins_Hz = 0.5*(bins_Hz[1:] + bins_Hz[:-1])
        if self.use_quad_trap_eff_interp == True:
             quad_trap_interp = np.load(self.path_to_quad_trap_eff_interp, allow_pickle = True)
             quad_trap_count_rate_interp = quad_trap_interp.item()['count_rate_interp']
             eff_array = quad_trap_count_rate_interp(bins_Hz)
        else:
             eff_array = np.ones(len(bins_Hz))
        # Initial guesses for curve_fit
        B_field_guess = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[np.argmax(data_hist_freq)])
        amplitude_guess = np.sum(data_hist_freq)
        FWHM_eV_guess = 5
        survival_probability_guess = 0.5
        scatter_fraction_guess = 0.5
        sigma_guess = 5
        gamma_guess = 3
        gaussian_portion_guess = 0.5
        scale_factor_guess = 1
        scatter_peak_ratio_parameter_p_guess = 0.9
        scatter_peak_ratio_parameter_q_guess = 1.0
        # Bounds for curve_fit
        B_field_min = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[0])
        B_field_max = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[-1])
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist_freq)*3
        FWHM_eV_min = 0
        FWHM_eV_max = ConversionFunctions.Energy(bins_Hz[0], B_field_guess)
        survival_probability_min = 1e-5
        survival_probability_max = 1
        scatter_fraction_min = 1e-5
        scatter_fraction_max = 1
        scale_factor_min = 1e-5
        scale_factor_max = 5
        scatter_peak_ratio_parameter_min = 1e-5
        scatter_peak_ratio_parameter_max = 5
        N = len(self.gases)
        gas_scatter_fraction_parameter_str = []
        for i in range(N-1):
            gas_scatter_fraction_parameter_str += [self.gases[i]+' scatter fraction']
        p0_guess = [B_field_guess, amplitude_guess, scale_factor_guess, survival_probability_guess, scatter_peak_ratio_parameter_p_guess, scatter_peak_ratio_parameter_q_guess]+ (N-1)*[scatter_fraction_guess]
        p0_bounds = [(B_field_min,B_field_max), (amplitude_min, amplitude_max), (scale_factor_min, scale_factor_max), (survival_probability_min, survival_probability_max), (scatter_peak_ratio_parameter_min, scatter_peak_ratio_parameter_max), (scatter_peak_ratio_parameter_min, scatter_peak_ratio_parameter_max)] + (N-1)*[(scatter_fraction_min, scatter_fraction_max)]
        parameter_names = ['B field','amplitude','width scale factor', 'survival probability','scatter peak ratio param b', 'scatter peak ratio param c'] + gas_scatter_fraction_parameter_str
        # Actually do the fitting
        m_binned = Minuit(lambda p: self.chi_2_simulated_resolution_scaled_fit_scatter_peak_ratio(bins_Hz, data_hist_freq, eff_array, p), p0_guess, name = parameter_names)
        m_binned.limits = p0_bounds
        if len(self.fixed_parameter_names)>0:
            for fixed_parameter_name, fixed_parameter_value in zip(self.fixed_parameter_names, self.fixed_parameter_values):
                m_binned.fixed[fixed_parameter_name] = True
                m_binned.values[fixed_parameter_name] = fixed_parameter_value
                m_binned.errors[fixed_parameter_name] = 0
        m_binned.migrad()
        m_binned.hesse()
        params = m_binned.values[0:]
        B_field_fit = params[0]
        #starting at index 2, grabs every other entry. (which is how scattering probs are filled in for N gases)
        amplitude_fit = params[1]
        scale_factor_fit = params[2]
        survival_probability_fit = params[3]
        scatter_peak_ratio_p_fit = params[4]
        scatter_peak_ratio_q_fit = params[5]
        logger.info('\n'+str(m_binned.params))
        scatter_fraction_fit = params[6:5+N]+[1- sum(params[6:5+N])]

        perr = m_binned.errors[0:]
        B_field_fit_err = perr[0]
        amplitude_fit_err = perr[1]
        scale_factor_fit_err = perr[2]
        survival_probability_fit_err = perr[3]
        scatter_peak_ratio_p_fit_err = perr[4]
        scatter_peak_ratio_q_fit_err = perr[5]
        scatter_fraction_fit_err = perr[6:5+N]+[np.sqrt(sum(np.array(perr[6:5+N])**2))]

        fit_Hz = self.spectrum_func_simulated_resolution_scaled_fit_scatter_peak_ratio(bins_Hz, eff_array, *params)
        fit_keV = ComplexLineShapeUtilities.flip_array(fit_Hz)
        bins_keV = ConversionFunctions.Energy(bins_Hz, B_field_fit)/1000
        bins_keV = ComplexLineShapeUtilities.flip_array(bins_keV)
        reduced_chi2 = m_binned.fval/(len(fit_Hz)-m_binned.nfit)
        correlation_matrix = m_binned.covariance.correlation()
        if print_params == True:
            output_string = '\n'
            output_string += 'Reduced chi^2 = {:.2e}\n'.format(reduced_chi2)
            output_string += '-----------------\n'
            output_string += 'B field = {:.8e}'.format(B_field_fit)+' +/- '+ '{:.4e} T\n'.format(B_field_fit_err)
            output_string += '-----------------\n'
            output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
            output_string += '-----------------\n'
            output_string += 'width scaling factor = {:.8e}'.format(scale_factor_fit) + ' +/- {:.8e}\n'.format(scale_factor_fit_err)
            output_string += '-----------------\n'
            output_string += 'survival probability = {:.8e}'.format(survival_probability_fit) + ' +/- {:.8e}\n'.format(survival_probability_fit_err)
            output_string += '-----------------\n'
            output_string += 'scatter_peak_ratio_p = {:.8e}'.format(scatter_peak_ratio_p_fit) + ' +/- {:.8e}\n'.format(scatter_peak_ratio_p_fit_err)
            output_string += '-----------------\n'
            output_string += 'scatter_peak_ratio_q = {:.8e}'.format(scatter_peak_ratio_q_fit) + ' +/- {:.8e}\n'.format(scatter_peak_ratio_q_fit_err)
            output_string += '-----------------\n'
            for i in range(len(self.gases)):
                output_string += '{} scatter fraction \n= '.format(self.gases[i]) + "{:.8e}".format(scatter_fraction_fit[i])\
                +' +/- ' + "{:.8e}".format(scatter_fraction_fit_err[i])+'\n'
                output_string += '-----------------\n'
        elapsed = time.time() - t
        output_string += 'Fit completed in '+str(round(elapsed,2))+'s'+'\n'
        dictionary_of_fit_results = {
        'output_string': output_string,
        'perr': perr,
        'bins_keV': bins_keV,
        'fit_keV': fit_keV,
        'bins_Hz': bins_Hz,
        'fit_Hz': fit_Hz,
        'B_field_fit': B_field_fit,
        'B_field_fit_err': B_field_fit_err,
        'scale_factor_fit': scale_factor_fit,
        'scale_factor_fit_err': scale_factor_fit_err,
        'scatter_peak_ratio_p_fit': scatter_peak_ratio_p_fit,
        'scatter_peak_ratio_p_fit_err': scatter_peak_ratio_p_fit_err,
        'scatter_peak_ratio_q_fit': scatter_peak_ratio_q_fit,
        'scatter_peak_ratio_q_fit_err': scatter_peak_ratio_q_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2,
        'correlation_matrix': np.array(correlation_matrix)
        }
        return dictionary_of_fit_results

    # This is one of the two functions called in the tritium fake data generator, for the
    # final Project 8 Phase II analysis: 
    # https://github.com/project8/mermithid/blob/combining_ComplexLineShape_and_FakeDataGenerator/mermithid/misc/FakeTritiumDataFunctions.py#L321
    def make_spectrum_gaussian_resolution_fit_scatter_peak_ratio(self, gauss_FWHM_eV, survival_probability, scatter_peak_ratio_p, scatter_peak_ratio_q, scatter_fraction, emitted_peak='shake'):
        p = np.zeros(len(self.gases))
        p[0:-1] = scatter_fraction
        p[-1] = 1 - sum(scatter_fraction)
        scatter_spectra_file_path = os.path.join(self.path_to_scatter_spectra_file, 'scatter_spectra.npy')
        scatter_spectra = np.load(scatter_spectra_file_path, allow_pickle = True)
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        elif emitted_peak == 'dirac':
            current_working_spectrum = self.std_dirac()
        current_working_spectrum = self.convolve_gaussian(current_working_spectrum, gauss_FWHM_eV)
        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += zeroth_order_peak
        N = len(self.gases)
        for M in range(1, self.max_scatters + 1):
            scatter_peak_ratio = np.exp(-1.*scatter_peak_ratio_p*M**( -self.factor*scatter_peak_ratio_p + scatter_peak_ratio_q))#np.exp(-1.*scatter_peak_ratio_b*M**scatter_peak_ratio_c)
            gas_scatter_combinations = np.array([np.array(i) for i in product(range(M+1), repeat=N) if sum(i)==M])
            for combination in gas_scatter_combinations:
                #print(combination)
                entry_str = ''
                for component, gas_type in zip(combination, self.gases):
                    entry_str += gas_type
                    entry_str += str(component).zfill(2)
                current_working_spectrum = scatter_spectra.item()[entry_str]
                current_working_spectrum = self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
                coefficient = factorial(sum(combination))
                for component, i in zip(combination, range(N)):
                    coefficient = coefficient/factorial(component)*p[i]**component
                current_full_spectrum += coefficient*current_working_spectrum*scatter_peak_ratio*survival_probability**M
        return current_full_spectrum


    # This function is called by fit_data_gaussian_resolution_fit_scatter_peak_ratio.
    # This function was not used in the final Phase II Kr fits or for tritium data generation
    # (though the Gaussian make_spectrum function is used for tritium data generation).
    # However, it was used in early Phase II Kr fits, when we were developing analysis models
    # and procedures. 
    # We are keeping this function because it is helpful to have a fit function with a 
    # gaussian instrumental resolution, for diagnostics and comparisons.
    def spectrum_func_gaussian_resolution_fit_scatter_peak_ratio(self, bins_Hz, eff_array, *p0):
        B_field = p0[0]
        amplitude = p0[1]
        gauss_FWHM_eV = p0[2]
        survival_probability = p0[3]
        scatter_peak_ratio_p = p0[4]
        scatter_peak_ratio_q = p0[5]
        N = len(self.gases)
        scatter_fraction = p0[6:5+N]

        x_eV = ConversionFunctions.Energy(bins_Hz, B_field)
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        f = np.zeros(len(x_eV))
        f_intermediate = np.zeros(len(x_eV))

        x_eV_minus_line = Constants.kr_k_line_e() - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_gaussian_resolution_fit_scatter_peak_ratio(gauss_FWHM_eV, survival_probability, scatter_peak_ratio_p, scatter_peak_ratio_q, scatter_fraction)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx], en_loss_array, full_spectrum)
        f_intermediate = f_intermediate*eff_array
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])

        return f

    def chi_2_gaussian_resolution_fit_scatter_peak_ratio(self, bin_centers, data_hist_freq, eff_array, params):
        # expectation
        fit_Hz = self.spectrum_func_gaussian_resolution_fit_scatter_peak_ratio(bin_centers, eff_array, *params)
        nonzero_bins_index = np.where((data_hist_freq != 0) & (fit_Hz > 0))
        zero_bins_index = np.where((data_hist_freq == 0) | (fit_Hz <= 0))
        chi2 = 2*((fit_Hz - data_hist_freq + data_hist_freq*np.log(data_hist_freq/fit_Hz))[nonzero_bins_index]).sum()
        chi2 += 2*(fit_Hz - data_hist_freq)[zero_bins_index].sum()
        return chi2


    # This function was not used in the final Phase II Kr fits or for tritium data generation
    # (though the Gaussian make_spectrum function is used for tritium data generation).
    # However, it was used in early Phase II Kr fits, when we were developing analysis models
    # and procedures. 
    # We are keeping this function because it is helpful to have a fit function with a 
    # gaussian instrumental resolution, for diagnostics and comparisons.
    def fit_data_gaussian_resolution_fit_scatter_peak_ratio(self, freq_bins, data_hist_freq, print_params=True):
        t = time.time()
        bins_Hz = freq_bins + self.RF_ROI_MIN
        bins_Hz = 0.5*(bins_Hz[1:] + bins_Hz[:-1])
        if self.use_quad_trap_eff_interp == True:
            quad_trap_interp = np.load(self.path_to_quad_trap_eff_interp, allow_pickle = True)
            quad_trap_count_rate_interp = quad_trap_interp.item()['count_rate_interp']
            eff_array = quad_trap_count_rate_interp(bins_Hz)
        else:
            eff_array = np.ones(len(bins_Hz))
        # Initial guesses for curve_fit
        B_field_guess = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[np.argmax(data_hist_freq)])
        amplitude_guess = np.sum(data_hist_freq)/2
        gauss_FWHM_eV_guess = 1
        survival_probability_guess = 0.5
        scatter_fraction_guess = 0.5
        scale_factor_guess = 0.1
        scatter_peak_ratio_parameter_guess = 0.5
        # Bounds for curve_fit
        B_field_min = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[0])
        B_field_max = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[-1])
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist_freq)*3
        gauss_FWHM_eV_min = 1e-5
        gauss_FWHM_eV_max = ConversionFunctions.Energy(bins_Hz[0], B_field_guess)-ConversionFunctions.Energy(bins_Hz[-1], B_field_guess)
        survival_probability_min = 1e-5
        survival_probability_max = 1
        scatter_fraction_min = 1e-5
        scatter_fraction_max = 1    
        scale_factor_min = 1e-5
        scale_factor_max = 5
        scatter_peak_ratio_parameter_min = 1e-5
        scatter_peak_ratio_parameter_max = 5
        N = len(self.gases)
        gas_scatter_fraction_parameter_str = []
        for i in range(N-1):
            gas_scatter_fraction_parameter_str += [self.gases[i]+' scatter fraction']
        p0_guess = [B_field_guess, amplitude_guess, gauss_FWHM_eV_guess, survival_probability_guess, scatter_peak_ratio_parameter_guess, scatter_peak_ratio_parameter_guess]+ (N-1)*[scatter_fraction_guess]
        p0_bounds = [(B_field_min,B_field_max), (amplitude_min,amplitude_max), (gauss_FWHM_eV_min, gauss_FWHM_eV_max), (survival_probability_min, survival_probability_max), (scatter_peak_ratio_parameter_min, scatter_peak_ratio_parameter_max), (scatter_peak_ratio_parameter_min, scatter_peak_ratio_parameter_max)] + (N-1)*[(scatter_fraction_min, scatter_fraction_max)]
        parameter_names = ['B field','amplitude','gaussian FWHM eV', 'survival probability','scatter peak ratio param b', 'scatter peak ratio param c'] + gas_scatter_fraction_parameter_str
        # Actually do the fitting
        m_binned = Minuit(lambda p: self.chi_2_gaussian_resolution_fit_scatter_peak_ratio(bins_Hz, data_hist_freq, eff_array, p), p0_guess, name = parameter_names)
        m_binned.limits = p0_bounds
        if len(self.fixed_parameter_names)>0:
            for fixed_parameter_name, fixed_parameter_value in zip(self.fixed_parameter_names, self.fixed_parameter_values):
                m_binned.fixed[fixed_parameter_name] = True
                m_binned.values[fixed_parameter_name] = fixed_parameter_value
                m_binned.errors[fixed_parameter_name] = 0
        m_binned.migrad()
        m_binned.hesse()
        params = m_binned.values[0:]
        B_field_fit = params[0]
        #starting at index 2, grabs every other entry. (which is how scattering probs are filled in for N gases)
        amplitude_fit = params[1]
        gauss_FWHM_eV_fit = params[2]
        survival_probability_fit = params[3]
        scatter_peak_ratio_p_fit = params[4]
        scatter_peak_ratio_q_fit = params[5]
        total_counts_fit = amplitude_fit
        logger.info('\n'+str(m_binned.params))
        scatter_fraction_fit = params[6:5+N]+[1- sum(params[6:5+N])]
        perr = m_binned.errors[0:]
        B_field_fit_err = perr[0]
        amplitude_fit_err = perr[1]
        gauss_FWHM_eV_fit_err = perr[2]
        survival_probability_fit_err = perr[3]
        scatter_peak_ratio_p_fit_err = perr[4]
        scatter_peak_ratio_q_fit_err = perr[5]
        total_counts_fit_err = amplitude_fit_err
        scatter_fraction_fit_err = perr[6:5+N]+[np.sqrt(sum(np.array(perr[6:5+N])**2))]
        fit_Hz = self.spectrum_func_gaussian_resolution_fit_scatter_peak_ratio(bins_Hz, eff_array, *params)
        fit_keV = ComplexLineShapeUtilities.flip_array(fit_Hz)
        bins_keV = ConversionFunctions.Energy(bins_Hz, B_field_fit)/1000
        bins_keV = ComplexLineShapeUtilities.flip_array(bins_keV)
        reduced_chi2 = m_binned.fval/(len(fit_Hz)-m_binned.nfit)
        correlation_matrix = m_binned.covariance.correlation()
    
        if print_params == True:
            output_string = '\n'
            output_string += 'Reduced chi^2 = {:.2e}\n'.format(reduced_chi2)
            output_string += '-----------------\n'
            output_string += 'B field = {:.8e}'.format(B_field_fit)+' +/- '+ '{:.4e} T\n'.format(B_field_fit_err)
            output_string += '-----------------\n'
            output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
            output_string += '-----------------\n'
            output_string += 'gaussian FWHM = {:.8e}'.format(gauss_FWHM_eV_fit) + ' +/- {:.8e} eV\n'.format(gauss_FWHM_eV_fit_err)
            output_string += '-----------------\n'
            output_string += 'survival probability = {:.8e}'.format(survival_probability_fit) + ' +/- {:.8e}\n'.format(survival_probability_fit_err)
            output_string += '-----------------\n'
            output_string += 'scatter_peak_ratio_p = {:.8e}'.format(scatter_peak_ratio_p_fit) + ' +/- {:.8e}\n'.format(scatter_peak_ratio_p_fit_err)
            output_string += '-----------------\n'
            output_string += 'scatter_peak_ratio_q = {:.8e}'.format(scatter_peak_ratio_q_fit) + ' +/- {:.8e}\n'.format(scatter_peak_ratio_q_fit_err)
            output_string += '-----------------\n'
            for i in range(len(self.gases)):
                output_string += '{} scatter fraction \n= '.format(self.gases[i]) + "{:.8e}".format(scatter_fraction_fit[i])\
                +' +/- ' + "{:.8e}".format(scatter_fraction_fit_err[i])+'\n'
                output_string += '-----------------\n'
        elapsed = time.time() - t
        output_string += 'Fit completed in '+str(round(elapsed,2))+'s'+'\n'
        dictionary_of_fit_results = {
        'output_string': output_string,
        'perr': perr,
        'bins_keV': bins_keV,
        'fit_keV': fit_keV,
        'bins_Hz': bins_Hz,
        'fit_Hz': fit_Hz,
        'B_field_fit': B_field_fit,
        'B_field_fit_err': B_field_fit_err,
        'gauss_FWHM_eV_fit': gauss_FWHM_eV_fit,
        'gauss_FWHM_eV_fit_err': gauss_FWHM_eV_fit_err,
        'scatter_peak_ratio_p_fit': scatter_peak_ratio_p_fit,
        'scatter_peak_ratio_p_fit_err': scatter_peak_ratio_p_fit_err,
        'scatter_peak_ratio_q_fit': scatter_peak_ratio_q_fit,
        'scatter_peak_ratio_q_fit_err': scatter_peak_ratio_q_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'scatter_fraction_fit': scatter_fraction_fit,
        'scatter_fraction_fit_err': scatter_fraction_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2,
        'correlation_matrix': np.array(correlation_matrix)
        }
        
        return dictionary_of_fit_results
    
    #This function is used for make_spectrum_simulated_resolution_scaled_fit_scatter_peak_ratio2
    #It is not used make_spectrum_simulated_resolution_scaled_fit_scatter_peak_ratio
    #However, the convolutions of scatter peaks in these two make_spectrum functions are equivalent
    #to each other. The convolutions are performed in different ways, which is why only the "ratio2"
    #function uses energy_loss_distribution_one_scatter.
    def energy_loss_distribution_one_scatter(self, scatter_fraction):
        p = np.zeros(len(self.gases))
        p[0:-1] = scatter_fraction
        p[-1] = 1 - sum(scatter_fraction)
        en_array = self.std_eV_array()
        energy_loss_one_scatter = en_array*0
        for i in range(len(self.gases)):
            energy_loss_one_scatter += p[i]*self.single_scatter_f(self.gases[i])
        f_radiation_loss_one_scatter = self.radiation_loss_f()
        energy_loss_one_scatter = self.normalize(signal.convolve(energy_loss_one_scatter, f_radiation_loss_one_scatter, mode = 'same'))
        return energy_loss_one_scatter       

    def make_spectrum_simulated_resolution_scaled_fit_scatter_peak_ratio2(self, scale_factor, survival_probability, scatter_peak_ratio_b, scatter_peak_ratio_c, scatter_fraction, emitted_peak='shake'):
        p = np.zeros(len(self.gases))
        p[0:-1] = scatter_fraction
        p[-1] = 1 - sum(scatter_fraction)
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        elif emitted_peak == 'dirac':
            current_working_spectrum = self.std_dirac()
        shake_spectrum = current_working_spectrum
        current_working_spectrum = self.convolve_simulated_resolution_scaled(current_working_spectrum, scale_factor)
        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += zeroth_order_peak
        energy_loss_distribution_one_scatter = self.energy_loss_distribution_one_scatter(scatter_fraction)
        N = len(self.gases)
        for M in range(1, self.max_scatters + 1):
            scatter_peak_ratio = np.exp(- scatter_peak_ratio_b*M**(-self.factor*scatter_peak_ratio_b + scatter_peak_ratio_c))
            current_working_spectrum = self.normalize(signal.convolve(current_working_spectrum, energy_loss_distribution_one_scatter, mode = 'same'))
            current_full_spectrum += current_working_spectrum*scatter_peak_ratio*survival_probability**M
        return current_full_spectrum

    def spectrum_func_simulated_resolution_scaled_fit_scatter_peak_ratio2(self, bins_Hz, eff_array, *p0):
        #This function parameterizes the scatter peak amplitudes using two variables b and c.
        #These are equivalent to the p and q ued by other functions, which are defined to
        #minimize correlations between the two variables.
        B_field = p0[0]
        amplitude = p0[1]
        scale_factor = p0[2]
        survival_probability = p0[3]
        scatter_peak_ratio_b = p0[4]
        scatter_peak_ratio_c = p0[5]
        N = len(self.gases)
        scatter_fraction = p0[6:5+N]

        x_eV = ConversionFunctions.Energy(bins_Hz, B_field)
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        f = np.zeros(len(x_eV))
        f_intermediate = np.zeros(len(x_eV))

        x_eV_minus_line = self.Kr_K_line_eV - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_simulated_resolution_scaled_fit_scatter_peak_ratio2(scale_factor, survival_probability, scatter_peak_ratio_b, scatter_peak_ratio_c, scatter_fraction)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx], en_loss_array, full_spectrum)
        f_intermediate = f_intermediate*eff_array
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])

        return f

    def chi_2_simulated_resolution_scaled_fit_scatter_peak_ratio2(self, bin_centers, data_hist_freq, eff_array, params):
        # expectation
        fit_Hz = self.spectrum_func_simulated_resolution_scaled_fit_scatter_peak_ratio2(bin_centers, eff_array, *params)
        nonzero_bins_index = np.where((data_hist_freq != 0) & (fit_Hz > 0))
        zero_bins_index = np.where((data_hist_freq == 0) | (fit_Hz <= 0))
        chi2 = 2*((fit_Hz - data_hist_freq + data_hist_freq*np.log(data_hist_freq/fit_Hz))[nonzero_bins_index]).sum()
        chi2 += 2*(fit_Hz - data_hist_freq)[zero_bins_index].sum()
        return chi2

    def fit_data_simulated_resolution_scaled_fit_scatter_peak_ratio2(self, freq_bins, data_hist_freq, print_params=True):
        #This function parameterizes the scatter peak amplitudes using two variables b and c.
        #These are equivalent to the p and q ued by other functions, which are defined to
        #minimize correlations between the two variables.
        
        #In this function, parameters can be configured to be either fixed or fitted,
        #by defined vaiables "fixed_parameter_name, fixed_parameter_value" in the 
        #configuration file.
        t = time.time()
        #self.check_existence_of_scatter_file()
        bins_Hz = freq_bins + self.RF_ROI_MIN
        bins_Hz = 0.5*(bins_Hz[1:] + bins_Hz[:-1])    
        quad_trap_interp = np.load(self.path_to_quad_trap_eff_interp, allow_pickle = True)
        quad_trap_count_rate_interp = quad_trap_interp.item()['count_rate_interp']
        eff_array = quad_trap_count_rate_interp(bins_Hz)
        # Initial guesses for curve_fit
        B_field_guess = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[np.argmax(data_hist_freq)])
        amplitude_guess = np.sum(data_hist_freq)
        FWHM_eV_guess = 5
        survival_probability_guess = 0.5
        scatter_fraction_guess = 0.5
        sigma_guess = 5
        gamma_guess = 3
        gaussian_portion_guess = 0.5
        scale_factor_guess = 0.5
        scatter_peak_ratio_parameter_guess = 0.5
        # Bounds for curve_fit
        B_field_min = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[0])
        B_field_max = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[-1])
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist_freq)*3
        FWHM_eV_min = 0
        FWHM_eV_max = ConversionFunctions.Energy(bins_Hz[0], B_field_guess)
        survival_probability_min = 1e-5
        survival_probability_max = 1
        scatter_fraction_min = 1e-5
        scatter_fraction_max = 1    
        scale_factor_min = 1e-5
        scale_factor_max = 5
        scatter_peak_ratio_parameter_min = 1e-5
        scatter_peak_ratio_parameter_max = 5
        N = len(self.gases)
        gas_scatter_fraction_parameter_str = []
        for i in range(N-1):
            gas_scatter_fraction_parameter_str += [self.gases[i]+' scatter fraction']
        p0_guess = [B_field_guess, amplitude_guess, scale_factor_guess, survival_probability_guess, scatter_peak_ratio_parameter_guess, scatter_peak_ratio_parameter_guess]+ (N-1)*[scatter_fraction_guess]
        p0_bounds = [(B_field_min,B_field_max), (amplitude_min,amplitude_max), (scale_factor_min, scale_factor_max), (survival_probability_min, survival_probability_max), (scatter_peak_ratio_parameter_min, scatter_peak_ratio_parameter_max), (scatter_peak_ratio_parameter_min, scatter_peak_ratio_parameter_max)] + (N-1)*[(scatter_fraction_min, scatter_fraction_max)]
        parameter_names = ['B field','amplitude','width scale factor', 'survival probability','scatter peak ratio param b', 'scatter peak ratio param c'] + gas_scatter_fraction_parameter_str
        # Actually do the fitting
        m_binned = Minuit(lambda p: self.chi_2_simulated_resolution_scaled_fit_scatter_peak_ratio2(bins_Hz, data_hist_freq, eff_array, p), p0_guess, name = parameter_names)
        m_binned.limits = p0_bounds
        m_binned.errordef = self.error_inflation_factor
        logger.info(m_binned.errordef)
        if len(self.fixed_parameter_names)>0:
            for fixed_parameter_name, fixed_parameter_value in zip(self.fixed_parameter_names, self.fixed_parameter_values):
                m_binned.fixed[fixed_parameter_name] = True
                m_binned.values[fixed_parameter_name] = fixed_parameter_value
                m_binned.errors[fixed_parameter_name] = 0
        m_binned.migrad()
        m_binned.hesse()
        params = m_binned.values[0:]
        B_field_fit = params[0]
        amplitude_fit = params[1]
        scale_factor_fit = params[2]
        survival_probability_fit = params[3]
        scatter_peak_ratio_b_fit = params[4]
        scatter_peak_ratio_c_fit = params[5]
        total_counts_fit = amplitude_fit
        logger.info('\n'+str(m_binned.params))
        scatter_fraction_fit = params[6:5+N]+[1- sum(params[6:5+N])]            

        perr = m_binned.errors[0:]
        B_field_fit_err = perr[0]
        amplitude_fit_err = perr[1]
        scale_factor_fit_err = perr[2]
        survival_probability_fit_err = perr[3]
        scatter_peak_ratio_b_fit_err = perr[4]
        scatter_peak_ratio_c_fit_err = perr[5]
        total_counts_fit_err = amplitude_fit_err
        scatter_fraction_fit_err = perr[6:5+N]+[np.sqrt(sum(np.array(perr[6:5+N])**2))]
    
        fit_Hz = self.spectrum_func_simulated_resolution_scaled_fit_scatter_peak_ratio2(bins_Hz, eff_array, *params)
        fit_keV = ComplexLineShapeUtilities.flip_array(fit_Hz)
        bins_keV = ConversionFunctions.Energy(bins_Hz, B_field_fit)/1000
        bins_keV = ComplexLineShapeUtilities.flip_array(bins_keV)
        reduced_chi2 = m_binned.fval/(len(fit_Hz)-m_binned.nfit)
        correlation_matrix = m_binned.covariance.correlation()
    
        if print_params == True:
            output_string = '\n'
            output_string += 'Reduced chi^2 = {:.2e}\n'.format(reduced_chi2)
            output_string += '-----------------\n'
            output_string += 'B field = {:.8e}'.format(B_field_fit)+' +/- '+ '{:.4e} T\n'.format(B_field_fit_err)
            output_string += '-----------------\n'
            output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
            output_string += '-----------------\n'
            output_string += 'width scaling factor = {:.8e}'.format(scale_factor_fit) + ' +/- {:.8e}\n'.format(scale_factor_fit_err)
            output_string += '-----------------\n'
            output_string += 'survival probability = {:.8e}'.format(survival_probability_fit) + ' +/- {:.8e}\n'.format(survival_probability_fit_err)
            output_string += '-----------------\n'
            output_string += 'scatter_peak_ratio_p = {:.8e}'.format(scatter_peak_ratio_b_fit) + ' +/- {:.8e}\n'.format(scatter_peak_ratio_b_fit_err)
            output_string += '-----------------\n'
            output_string += 'scatter_peak_ratio_q = {:.8e}'.format(scatter_peak_ratio_c_fit) + ' +/- {:.8e}\n'.format(scatter_peak_ratio_c_fit_err)
            output_string += '-----------------\n'
            # output_string += 'scale_factor1= {:.8e}'.format(scale_factor1_fit) + ' +/- {:.8e}\n'.format(scale_factor1_fit_err)
#             output_string += '-----------------\n'
            for i in range(len(self.gases)):
                output_string += '{} scatter fraction \n= '.format(self.gases[i]) + "{:.8e}".format(scatter_fraction_fit[i])\
                +' +/- ' + "{:.8e}".format(scatter_fraction_fit_err[i])+'\n'
                output_string += '-----------------\n'
        elapsed = time.time() - t
        output_string += 'Fit completed in '+str(round(elapsed,2))+'s'+'\n'
        dictionary_of_fit_results = {
        'output_string': output_string,
        'perr': perr,
        'bins_keV': bins_keV,
        'fit_keV': fit_keV,
        'bins_Hz': bins_Hz,
        'fit_Hz': fit_Hz,
        'B_field_fit': B_field_fit,
        'B_field_fit_err': B_field_fit_err,
        'scale_factor_fit': scale_factor_fit,
        'scale_factor_fit_err': scale_factor_fit_err,
        'scatter_peak_ratio_p_fit': scatter_peak_ratio_b_fit,
        'scatter_peak_ratio_p_fit_err': scatter_peak_ratio_b_fit_err,
        'scatter_peak_ratio_q_fit': scatter_peak_ratio_c_fit,
        'scatter_peak_ratio_q_fit_err': scatter_peak_ratio_c_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'gases': self.gases,
        'scatter_fraction_fit': scatter_fraction_fit,
        'scatter_fraction_fit_err': scatter_fraction_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2,
        'correlation_matrix': np.array(correlation_matrix)
        }
        return dictionary_of_fit_results
        
    # Previously, we used this function (generate_scatter_peaks) to save scatter peaks
    # to a file and then read them.
    # Now instead we generate scatter peaks while fitting, which is faster.
    # In the current approach, we calculate the average energy loss function over the
    # different gases given their scatter fractions (determined from gas composition), and
    # then convolve the 0th order peak with the averaged energy loss.
    def generate_scatter_peaks(self):
        
        p = np.zeros(len(self.gases))
        scatter_fraction = self.scatter_fractions_for_gases
        p[0:-1] = scatter_fraction
        p[-1] = 1 - sum(scatter_fraction)

        scatter_spectra_file_path = os.path.join(self.path_to_scatter_spectra_file, 'scatter_spectra.npy')
        scatter_spectra = np.load(scatter_spectra_file_path, allow_pickle = True)
        en_array = self.std_eV_array()

        scatter_peaks = np.zeros((self.max_scatters+1, len(en_array)))
        emitted_peak = self.base_shape
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        elif emitted_peak == 'dirac':
            current_working_spectrum = self.std_dirac()
        
        scale_factor = 1
        current_working_spectrum = self.convolve_simulated_resolution_scaled(current_working_spectrum, scale_factor)
        zeroth_order_peak = current_working_spectrum
        scatter_peaks[0] = zeroth_order_peak
        N = len(self.gases)
        for M in range(1, self.max_scatters + 1):
            current_scatter_peak_spectrum = np.zeros(len(en_array))
            gas_scatter_combinations = np.array([np.array(i) for i in product(range(M+1), repeat=N) if sum(i)==M])
            for combination in gas_scatter_combinations:
                entry_str = ''
                for component, gas_type in zip(combination, self.gases):
                    entry_str += gas_type
                    entry_str += str(component).zfill(2)
                current_working_spectrum = scatter_spectra.item()[entry_str]
                current_working_spectrum = self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
                coefficient = factorial(sum(combination))
                for component, i in zip(combination, range(N)):
                    coefficient = coefficient/factorial(component)*p[i]**component
                current_scatter_peak_spectrum += coefficient*current_working_spectrum
            scatter_peaks[M] = current_scatter_peak_spectrum
        return scatter_peaks


    ##############################################################################
    ##############################################################################
    ##############################################################################
    # All functions below here do not have a specific known use-case, but they were considered/tested
    # as potential lineshapes, or were otherwise considered.
    ##############################################################################
    ##############################################################################
    ##############################################################################

    # Do resolution sampling - this function is not currently used; we did the sampling elsewhere.
    def sample_and_interpolate_resolution(self):
        if self.use_combined_four_trap_inst_reso:
            x_data, y_data, y_err_data = self.combine_four_trap_resolution_from_txt(self.trap_weights)
            logger.info("Combined four instrumental resolution files")
        else:
            x_data, y_data, y_err_data = self.read_ins_resolution_data(self.path_to_ins_resolution_data_txt)
            logger.info("Using ONE simulated instrumental resolution file (not combining four)")
        if self.sample_ins_resolution_errors:
            y_data = np.random.normal(y_data, y_err_data)
            logger.info("Sampling instrumental resolution counts per bin")
        self.interpolated_resolution = interpolate.interp1d(x_data, y_data, fill_value=(0,0), bounds_error=False)


    def composite_gaussian_lorentzian(self, sigma):
        x_array = self.std_eV_array()
        w_g = x_array/sigma
        gamma = self.ratio_gamma_to_sigma*sigma
        w_l = x_array/gamma
        lorentzian = 1./(gamma*np.pi)*1./(1+(w_l**2))
        gaussian = 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-0.5*w_g**2)
        p = self.gaussian_proportion
        composite_function = p*gaussian+(1-p)*lorentzian
        return composite_function

    def elevated_gaussian(self, elevation_factor, sigma):
        x_array = self.std_eV_array()
        w_g = x_array/sigma
        gamma = self.ratio_gamma_to_sigma*sigma
        w_l = x_array/gamma
        lorentzian = 1./(gamma*np.pi)*1./(1+(w_l**2))
        gaussian = 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-0.5*w_g**2)
        modified_guassian_function = gaussian*(1 + elevation_factor*lorentzian)
        return modified_guassian_function
    
    # You would use this function if you have a sum of gaussians, and instead
    # of fitting the standard deviations of all gaussians independently, you 
    # fit a single scale factor that scales all standard deviations.
    def composite_gaussian_scaled(self, scale_factor):
        x_array = self.std_eV_array()
        ans = 0
        A_array = self.A_array
        sigma_array = np.array(self.sigma_array)
        sigma_array = sigma_array*scale_factor
        for A, sigma in zip(A_array, sigma_array):
            ans += self.gaussian(x_array, A, sigma, 0)
        return ans

    def convolve_composite_gaussian_lorentzian(self, func_to_convolve, sigma):
        resolution_f = self.composite_gaussian_lorentzian(sigma)
        ans = signal.convolve(resolution_f, func_to_convolve, mode='same')
        ans_normed = self.normalize(ans)
        return ans_normed

    def convolve_elevated_gaussian(self, func_to_convolve, elevation_factor, sigma):
        resolution_f = self.elevated_gaussian(elevation_factor, sigma)
        ans = signal.convolve(resolution_f, func_to_convolve, mode = 'same')
        ans_normed = self.normalize(ans)
        return ans_normed

    
    def convolve_composite_gaussian_scaled(self, func_to_convolve, scale_factor):
        resolution_f = self.composite_gaussian_scaled(scale_factor)
        ans = signal.convolve(resolution_f, func_to_convolve, mode = 'same')
        ans_normed = self.normalize(ans)
        return ans_normed



    def chi_2_Poisson_composite_gaussian_lorentzian_reso(self, bin_centers, data_hist_freq, eff_array, params):
        nonzero_bins_index = np.where(data_hist_freq != 0)
        zero_bins_index = np.where(data_hist_freq == 0)
        # expectation
        if self.fixed_scatter_proportion == True and self.fixed_survival_probability == True:
            fit_Hz = self.spectrum_func_composite_gaussian_lorentzian_fixed_scatter_proportion_and_survival_prob(bin_centers, eff_array, *params)
        elif self.fixed_scatter_proportion == True and self.fixed_survival_probability == False:
            fit_Hz = self.spectrum_func_composite_gaussian_lorentzian_fixed_scatter_proportion(bin_centers, eff_array, *params)
        elif self.fixed_scatter_proportion == False and self.fixed_survival_probability == True and self.partially_fixed_scatter_proportion == False:
            fit_Hz = self.spectrum_func_composite_gaussian_lorentzian_fixed_survival_probability(bin_centers, eff_array, *params)
        elif self.partially_fixed_scatter_proportion == True and self.fixed_survival_probability == True:
            fit_Hz = self.spectrum_func_composite_gaussian_lorentzian_fixed_survival_probability_partially_fixed_scatter_proportion(bin_centers, eff_array, *params)
        chi2 = 2*((fit_Hz - data_hist_freq + data_hist_freq*np.log(data_hist_freq/fit_Hz))[nonzero_bins_index]).sum()
        chi2 += 2*(fit_Hz - data_hist_freq)[zero_bins_index].sum()
        return chi2

    def chi_2_Poisson_elevated_gaussian_reso(self, bin_centers, data_hist_freq, eff_array, params):
        nonzero_bins_index = np.where(data_hist_freq != 0)
        zero_bins_index = np.where(data_hist_freq == 0)
        # expectation
        fit_Hz = self.spectrum_func_elevated_gaussian_fixed_scatter_proportion(bin_centers, eff_array, *params)
        chi2 = 2*((fit_Hz - data_hist_freq + data_hist_freq*np.log(data_hist_freq/fit_Hz))[nonzero_bins_index]).sum()
        chi2 += 2*(fit_Hz - data_hist_freq)[zero_bins_index].sum()
        return chi2

    def chi_2_Poisson_composite_gaussian_scaled_reso(self, bin_centers, data_hist_freq, eff_array, params):
        nonzero_bins_index = np.where(data_hist_freq != 0)
        zero_bins_index = np.where(data_hist_freq == 0)
        # expectation
        fit_Hz = self.spectrum_func_composite_gaussian_scaled_fixed_scatter_proportion(bin_centers, eff_array, *params)
        chi2 = 2*((fit_Hz - data_hist_freq + data_hist_freq*np.log(data_hist_freq/fit_Hz))[nonzero_bins_index]).sum()
        chi2 += 2*(fit_Hz - data_hist_freq)[zero_bins_index].sum()
        return chi2

    #fitting with commposite gaussian lorentzian resolution function and fixed scatter fraction
    def make_spectrum_composite_gaussian_lorentzian_fixed_scatter_proportion(self, survival_prob, sigma, emitted_peak='shake'):
        p = self.scatter_proportion
        a = self.recon_eff_param_a
        b = self.recon_eff_param_b
        c = self.recon_eff_param_c
        scatter_spectra_file_path = os.path.join(self.path_to_scatter_spectra_file, 'scatter_spectra.npy')
        scatter_spectra = np.load(scatter_spectra_file_path, allow_pickle = True)
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        current_working_spectrum = self.convolve_composite_gaussian_lorentzian(current_working_spectrum, sigma)
        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += zeroth_order_peak
        N = len(self.gases)
        for M in range(1, self.max_scatters + 1):
            relative_reconstruction_eff = np.exp(-b*M**c)
            gas_scatter_combinations = np.array([np.array(i) for i in product(range(M+1), repeat=N) if sum(i)==M])
            for combination in gas_scatter_combinations:
                #print(combination)
                entry_str = ''
                for component, gas_type in zip(combination, self.gases):
                    entry_str += gas_type
                    entry_str += str(component).zfill(2)
                current_working_spectrum = scatter_spectra.item()[entry_str]
                current_working_spectrum = self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
                coefficient = factorial(sum(combination))
                for component, i in zip(combination, range(N)):
                    coefficient = coefficient/factorial(component)*p[i]**component
                for i in range(0, M):
                    coefficient = coefficient*(1-a*np.exp(-b*i**c))
                current_full_spectrum += relative_reconstruction_eff*coefficient*current_working_spectrum*survival_prob**M
        return current_full_spectrum

    def spectrum_func_composite_gaussian_lorentzian_fixed_scatter_proportion(self, bins_Hz, eff_array, *p0):

        B_field = p0[0]
        amplitude = p0[1]
        survival_prob = p0[2]
        sigma = p0[3]

        x_eV = ConversionFunctions.Energy(bins_Hz, B_field)
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        f = np.zeros(len(x_eV))
        f_intermediate = np.zeros(len(x_eV))

        x_eV_minus_line = self.Kr_K_line_eV - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_composite_gaussian_lorentzian_fixed_scatter_proportion(survival_prob, sigma)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx], en_loss_array, full_spectrum)
        f_intermediate = f_intermediate*eff_array
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])

        return f

    def fit_data_composite_gaussian_lorentzian_fixed_scatter_proportion(self, freq_bins, data_hist_freq, print_params=True):
        t = time.time()
        self.check_existence_of_scatter_file()
        bins_Hz = freq_bins + self.RF_ROI_MIN
        bins_Hz = 0.5*(bins_Hz[1:] + bins_Hz[:-1])

        quad_trap_interp = np.load(self.path_to_quad_trap_eff_interp, allow_pickle = True)
        quad_trap_count_rate_interp = quad_trap_interp.item()['count_rate_interp']
        eff_array = quad_trap_count_rate_interp(bins_Hz)
        # Initial guesses for curve_fit
        B_field_guess = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[np.argmax(data_hist_freq)])
        amplitude_guess = np.sum(data_hist_freq)/2
        FWHM_eV_guess = 5
        prob_parameter_guess = 0.5
        scatter_proportion_guess = 0.5
        sigma_guess = 5
        gamma_guess = 3
        gaussian_portion_guess = 0.5
        # Bounds for curve_fit
        B_field_min = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[0])
        B_field_max = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[-1])
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist_freq)*3
        FWHM_eV_min = 0
        FWHM_eV_max = ConversionFunctions.Energy(bins_Hz[0], B_field_guess)
        prob_parameter_min = 1e-5
        prob_parameter_max = 1
        scatter_proportion_min = 1e-5
        scatter_proportion_max = 1
        mu_min = -FWHM_eV_max
        mu_max = FWHM_eV_max
        gaussian_portion_min = 1e-5
        gaussian_portion_max = 1
        N = len(self.gases)
        p0_guess = [B_field_guess, amplitude_guess, prob_parameter_guess, sigma_guess]
        p0_bounds = [(B_field_min,B_field_max), (amplitude_min,amplitude_max), (prob_parameter_min, prob_parameter_max), (FWHM_eV_min, FWHM_eV_max)]
        # Actually do the fitting
        m_binned = Minuit.from_array_func(lambda p: self.chi_2_Poisson_composite_gaussian_lorentzian_reso(bins_Hz, data_hist_freq, eff_array, p),
                                          start = p0_guess,
                                          limit = p0_bounds,
                                          throw_nan = True
                                          )
        m_binned.migrad()
        params = m_binned.np_values()
        B_field_fit = params[0]
        #starting at index 2, grabs every other entry. (which is how scattering probs are filled in for N gases)
        amplitude_fit = params[1]
        survival_prob_fit = params[2]
        sigma_fit = params[3]
        total_counts_fit = amplitude_fit

        perr = m_binned.np_errors()
        B_field_fit_err = perr[0]
        amplitude_fit_err = perr[1]
        survival_prob_fit_err = perr[2]
        sigma_fit_err = perr[3]
        total_counts_fit_err = amplitude_fit_err

        fit_Hz = self.spectrum_func_composite_gaussian_lorentzian_fixed_scatter_proportion(bins_Hz, eff_array, *params)
        fit_keV = ComplexLineShapeUtilities.flip_array(fit_Hz)
        bins_keV = ConversionFunctions.Energy(bins_Hz, B_field_fit)/1000
        bins_keV = ComplexLineShapeUtilities.flip_array(bins_keV)
        reduced_chi2 = m_binned.fval/(len(fit_Hz)-m_binned.nfit)

        if print_params == True:
            output_string = '\n'
            output_string += 'Reduced chi^2 = {:.2e}\n'.format(reduced_chi2)
            output_string += '-----------------\n'
            output_string += 'B field = {:.8e}'.format(B_field_fit)+' +/- '+ '{:.4e} T\n'.format(B_field_fit_err)
            output_string += '-----------------\n'
            output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
            output_string += '-----------------\n'
            output_string += 'Survival probability = {:.8e}'.format(survival_prob_fit) + ' +/- {:.8e}\n'.format(survival_prob_fit_err)
            output_string += '-----------------\n'
            output_string += 'sigma = {:.2e}'.format(sigma_fit) + ' +/- {:.4e}\n'.format(sigma_fit_err)
            output_string += '-----------------\n'
        elapsed = time.time() - t
        output_string += 'Fit completed in '+str(round(elapsed,2))+'s'+'\n'
        dictionary_of_fit_results = {
        'output_string': output_string,
        'perr': perr,
        'bins_keV': bins_keV,
        'fit_keV': fit_keV,
        'bins_Hz': bins_Hz,
        'fit_Hz': fit_Hz,
        'B_field_fit': B_field_fit,
        'B_field_fit_err': B_field_fit_err,
        'survival_prob_fit': survival_prob_fit,
        'survival_prob_fit_err': survival_prob_fit_err,
        'sigma_fit': sigma_fit,
        'sigma_fit_err': sigma_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2
        }
        return dictionary_of_fit_results


    # fitting with elevated gaussian resolution function
    def make_spectrum_elevated_gaussian_fixed_scatter_proportion(self, survival_prob, sigma, elevation_factor, emitted_peak='shake'):
        p = self.scatter_proportion
        a = self.recon_eff_param_a
        b = self.recon_eff_param_b
        c = self.recon_eff_param_c
        scatter_spectra_file_path = os.path.join(self.path_to_scatter_spectra_file, 'scatter_spectra.npy')
        scatter_spectra = np.load(scatter_spectra_file_path, allow_pickle = True)
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        current_working_spectrum = self.convolve_elevated_gaussian(current_working_spectrum, elevation_factor, sigma)
        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += zeroth_order_peak
        N = len(self.gases)
        for M in range(1, self.max_scatters + 1):
            relative_reconstruction_eff = np.exp(-b*M**c)
            gas_scatter_combinations = np.array([np.array(i) for i in product(range(M+1), repeat=N) if sum(i)==M])
            for combination in gas_scatter_combinations:
                #print(combination)
                entry_str = ''
                for component, gas_type in zip(combination, self.gases):
                    entry_str += gas_type
                    entry_str += str(component).zfill(2)
                current_working_spectrum = scatter_spectra.item()[entry_str]
                current_working_spectrum = self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
                coefficient = factorial(sum(combination))
                for component, i in zip(combination, range(N)):
                    coefficient = coefficient/factorial(component)*p[i]**component
                for i in range(0, M):
                    coefficient = coefficient*(1-a*np.exp(-b*i**c))
                current_full_spectrum += relative_reconstruction_eff*coefficient*current_working_spectrum*survival_prob**M
        return current_full_spectrum

    def spectrum_func_elevated_gaussian_fixed_scatter_proportion(self, bins_Hz, eff_array, *p0):

        B_field = p0[0]
        amplitude = p0[1]
        survival_prob = p0[2]
        sigma = p0[3]
        elevation_factor = p0[4]

        x_eV = ConversionFunctions.Energy(bins_Hz, B_field)
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        f = np.zeros(len(x_eV))
        f_intermediate = np.zeros(len(x_eV))

        x_eV_minus_line = self.Kr_K_line_eV - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_elevated_gaussian_fixed_scatter_proportion(survival_prob, sigma, elevation_factor)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx], en_loss_array, full_spectrum)
        f_intermediate = f_intermediate*eff_array
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])

        return f

    def fit_data_elevated_gaussian_fixed_scatter_proportion(self, freq_bins, data_hist_freq, print_params=True):
        t = time.time()
        self.check_existence_of_scatter_file()
        bins_Hz = freq_bins + self.RF_ROI_MIN
        bins_Hz = 0.5*(bins_Hz[1:] + bins_Hz[:-1])

        quad_trap_interp = np.load(self.path_to_quad_trap_eff_interp, allow_pickle = True)
        quad_trap_count_rate_interp = quad_trap_interp.item()['count_rate_interp']
        eff_array = quad_trap_count_rate_interp(bins_Hz)
        # Initial guesses for curve_fit
        B_field_guess = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[np.argmax(data_hist_freq)])
        amplitude_guess = np.sum(data_hist_freq)/2
        FWHM_eV_guess = 5
        prob_parameter_guess = 0.5
        scatter_proportion_guess = 0.5
        sigma_guess = 5
        gamma_guess = 3
        gaussian_portion_guess = 0.5
        elevation_factor_guess = 20
        # Bounds for curve_fit
        B_field_min = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[0])
        B_field_max = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[-1])
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist_freq)*3
        FWHM_eV_min = 0
        FWHM_eV_max = ConversionFunctions.Energy(bins_Hz[0], B_field_guess)
        prob_parameter_min = 1e-5
        prob_parameter_max = 1
        scatter_proportion_min = 1e-5
        scatter_proportion_max = 1
        mu_min = -FWHM_eV_max
        mu_max = FWHM_eV_max
        gaussian_portion_min = 1e-5
        gaussian_portion_max = 1
        elevation_factor_min = 0
        elevation_factor_max = 500
        N = len(self.gases)
        p0_guess = [B_field_guess, amplitude_guess, prob_parameter_guess, sigma_guess, elevation_factor_guess]
        p0_bounds = [(B_field_min,B_field_max), (amplitude_min,amplitude_max), (prob_parameter_min, prob_parameter_max), (FWHM_eV_min, FWHM_eV_max), (elevation_factor_min, elevation_factor_max)]
        # Actually do the fitting
        m_binned = Minuit.from_array_func(lambda p: self.chi_2_Poisson_elevated_gaussian_reso(bins_Hz, data_hist_freq, eff_array, p),
                                          start = p0_guess,
                                          limit = p0_bounds,
                                          throw_nan = True
                                          )
        m_binned.migrad()
        params = m_binned.np_values()
        B_field_fit = params[0]
        #starting at index 2, grabs every other entry. (which is how scattering probs are filled in for N gases)
        amplitude_fit = params[1]
        survival_prob_fit = params[2]
        sigma_fit = params[3]
        elevation_factor_fit = params[4]
        total_counts_fit = amplitude_fit

        perr = m_binned.np_errors()
        B_field_fit_err = perr[0]
        amplitude_fit_err = perr[1]
        survival_prob_fit_err = perr[2]
        sigma_fit_err = perr[3]
        elevation_factor_fit_err = perr[4]
        total_counts_fit_err = amplitude_fit_err

        fit_Hz = self.spectrum_func_elevated_gaussian_fixed_scatter_proportion(bins_Hz, eff_array, *params)
        fit_keV = ComplexLineShapeUtilities.flip_array(fit_Hz)
        bins_keV = ConversionFunctions.Energy(bins_Hz, B_field_fit)/1000
        bins_keV = ComplexLineShapeUtilities.flip_array(bins_keV)
        reduced_chi2 = m_binned.fval/(len(fit_Hz)-m_binned.nfit)

        if print_params == True:
            output_string = '\n'
            output_string += 'Reduced chi^2 = {:.2e}\n'.format(reduced_chi2)
            output_string += '-----------------\n'
            output_string += 'B field = {:.8e}'.format(B_field_fit)+' +/- '+ '{:.4e} T\n'.format(B_field_fit_err)
            output_string += '-----------------\n'
            output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
            output_string += '-----------------\n'
            output_string += 'Survival probability = {:.8e}'.format(survival_prob_fit) + ' +/- {:.8e}\n'.format(survival_prob_fit_err)
            output_string += '-----------------\n'
            output_string += 'sigma = {:.2e}'.format(sigma_fit) + ' +/- {:.4e}\n'.format(sigma_fit_err)
            output_string += '-----------------\n'
            output_string += 'elevation factor = {:.2e}'.format(elevation_factor_fit) + ' +/- {:.4e}\n'.format(elevation_factor_fit_err)
            output_string += '-----------------\n'
        elapsed = time.time() - t
        output_string += 'Fit completed in '+str(round(elapsed,2))+'s'+'\n'
        dictionary_of_fit_results = {
        'output_string': output_string,
        'perr': perr,
        'bins_keV': bins_keV,
        'fit_keV': fit_keV,
        'bins_Hz': bins_Hz,
        'fit_Hz': fit_Hz,
        'B_field_fit': B_field_fit,
        'B_field_fit_err': B_field_fit_err,
        'survival_prob_fit': survival_prob_fit,
        'survival_prob_fit_err': survival_prob_fit_err,
        'sigma_fit': sigma_fit,
        'sigma_fit_err': sigma_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2
        }
        return dictionary_of_fit_results
    

    def make_spectrum_composite_gaussian_scaled_fixed_scatter_proportion(self, survival_prob, scale_factor, emitted_peak='shake'):
        p = self.scatter_proportion
        a = self.recon_eff_param_a
        b = self.recon_eff_param_b
        c = self.recon_eff_param_c
        scatter_spectra_file_path = os.path.join(self.path_to_scatter_spectra_file, 'scatter_spectra.npy')
        scatter_spectra = np.load(scatter_spectra_file_path, allow_pickle = True)
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        current_working_spectrum = self.convolve_composite_gaussian_scaled(current_working_spectrum, scale_factor)
        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += zeroth_order_peak
        N = len(self.gases)
        for M in range(1, self.max_scatters + 1):
            relative_reconstruction_eff = np.exp(-b*M**c)
            gas_scatter_combinations = np.array([np.array(i) for i in product(range(M+1), repeat=N) if sum(i)==M])
            for combination in gas_scatter_combinations:
                #print(combination)
                entry_str = ''
                for component, gas_type in zip(combination, self.gases):
                    entry_str += gas_type
                    entry_str += str(component).zfill(2)
                current_working_spectrum = scatter_spectra.item()[entry_str]
                current_working_spectrum = self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
                coefficient = factorial(sum(combination))
                for component, i in zip(combination, range(N)):
                    coefficient = coefficient/factorial(component)*p[i]**component
                for i in range(0, M):
                    coefficient = coefficient*(1-a*np.exp(-b*i**c))
                current_full_spectrum += relative_reconstruction_eff*coefficient*current_working_spectrum*survival_prob**M
        return current_full_spectrum

    def spectrum_func_composite_gaussian_scaled_fixed_scatter_proportion(self, bins_Hz, eff_array, *p0):

        B_field = p0[0]
        amplitude = p0[1]
        survival_prob = p0[2]
        scale_factor = p0[3]

        x_eV = ConversionFunctions.Energy(bins_Hz, B_field)
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        f = np.zeros(len(x_eV))
        f_intermediate = np.zeros(len(x_eV))

        x_eV_minus_line = self.Kr_K_line_eV - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_composite_gaussian_scaled_fixed_scatter_proportion(survival_prob, scale_factor)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx], en_loss_array, full_spectrum)
        f_intermediate = f_intermediate*eff_array
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])

        return f

    def fit_data_composite_gaussian_scaled_fixed_scatter_proportion(self, freq_bins, data_hist_freq, print_params=True):
        t = time.time()
        self.check_existence_of_scatter_file()
        bins_Hz = freq_bins + self.RF_ROI_MIN
        bins_Hz = 0.5*(bins_Hz[1:] + bins_Hz[:-1])

        quad_trap_interp = np.load(self.path_to_quad_trap_eff_interp, allow_pickle = True)
        quad_trap_count_rate_interp = quad_trap_interp.item()['count_rate_interp']
        eff_array = quad_trap_count_rate_interp(bins_Hz)
        # Initial guesses for curve_fit
        B_field_guess = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[np.argmax(data_hist_freq)])
        amplitude_guess = np.sum(data_hist_freq)/2
        FWHM_eV_guess = 5
        prob_parameter_guess = 0.5
        scatter_proportion_guess = 0.5
        sigma_guess = 5
        gamma_guess = 3
        gaussian_portion_guess = 0.5
        scale_factor_guess = 1.
        # Bounds for curve_fit
        B_field_min = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[0])
        B_field_max = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[-1])
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist_freq)*3
        FWHM_eV_min = 0
        FWHM_eV_max = ConversionFunctions.Energy(bins_Hz[0], B_field_guess)
        prob_parameter_min = 1e-5
        prob_parameter_max = 1
        scatter_proportion_min = 1e-5
        scatter_proportion_max = 1
        mu_min = -FWHM_eV_max
        mu_max = FWHM_eV_max
        gaussian_portion_min = 1e-5
        gaussian_portion_max = 1
        scale_factor_min = 1e-5
        scale_factor_max = 500
        N = len(self.gases)
        p0_guess = [B_field_guess, amplitude_guess, prob_parameter_guess, scale_factor_guess]
        p0_bounds = [(B_field_min,B_field_max), (amplitude_min,amplitude_max), (prob_parameter_min, prob_parameter_max), (scale_factor_min, scale_factor_max)]
        # Actually do the fitting
        m_binned = Minuit.from_array_func(lambda p: self.chi_2_Poisson_composite_gaussian_scaled_reso(bins_Hz, data_hist_freq, eff_array, p),
                                          start = p0_guess,
                                          limit = p0_bounds,
                                          throw_nan = True
                                          )
        m_binned.migrad()
        params = m_binned.np_values()
        B_field_fit = params[0]
        #starting at index 2, grabs every other entry. (which is how scattering probs are filled in for N gases)
        amplitude_fit = params[1]
        survival_prob_fit = params[2]
        scale_factor_fit = params[3]
        total_counts_fit = amplitude_fit

        perr = m_binned.np_errors()
        B_field_fit_err = perr[0]
        amplitude_fit_err = perr[1]
        survival_prob_fit_err = perr[2]
        scale_factor_fit_err = perr[3]
        total_counts_fit_err = amplitude_fit_err

        fit_Hz = self.spectrum_func_composite_gaussian_scaled_fixed_scatter_proportion(bins_Hz, eff_array, *params)
        fit_keV = ComplexLineShapeUtilities.flip_array(fit_Hz)
        bins_keV = ConversionFunctions.Energy(bins_Hz, B_field_fit)/1000
        bins_keV = ComplexLineShapeUtilities.flip_array(bins_keV)
        reduced_chi2 = m_binned.fval/(len(fit_Hz)-m_binned.nfit)

        if print_params == True:
            output_string = '\n'
            output_string += 'Reduced chi^2 = {:.2e}\n'.format(reduced_chi2)
            output_string += '-----------------\n'
            output_string += 'B field = {:.8e}'.format(B_field_fit)+' +/- '+ '{:.4e} T\n'.format(B_field_fit_err)
            output_string += '-----------------\n'
            output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
            output_string += '-----------------\n'
            output_string += 'Survival probability = {:.8e}'.format(survival_prob_fit) + ' +/- {:.8e}\n'.format(survival_prob_fit_err)
            output_string += '-----------------\n'
            output_string += 'scale factor = {:.8e}'.format(scale_factor_fit) + ' +/- {:.8e}\n'.format(scale_factor_fit_err)
            output_string += '-----------------\n'
        elapsed = time.time() - t
        output_string += 'Fit completed in '+str(round(elapsed,2))+'s'+'\n'
        dictionary_of_fit_results = {
        'output_string': output_string,
        'perr': perr,
        'bins_keV': bins_keV,
        'fit_keV': fit_keV,
        'bins_Hz': bins_Hz,
        'fit_Hz': fit_Hz,
        'B_field_fit': B_field_fit,
        'B_field_fit_err': B_field_fit_err,
        'survival_prob_fit': survival_prob_fit,
        'survival_prob_fit_err': survival_prob_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2
        }
        return dictionary_of_fit_results
