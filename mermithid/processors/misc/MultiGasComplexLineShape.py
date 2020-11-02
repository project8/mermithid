'''
Fits data to complex lineshape model.
Author: E. Machado, Y.-H. Sun, E. Novitski
Date: 4/8/20

This processor takes in frequency data in binned histogram and fit the histogram with two gas scattering complex line shape model.

Configurable parameters:

There are two options available for fitting: fix_scatter_proportion = True and False.
gases: array for names of the two gases involved in the scattering process.
max_scatter: max number of scatterings for only single gas scatterings.
max_comprehansive_scatter: max number of scatterings for all cross scatterings.
scatter_proportion: when fix_scatter_proportion is set as true, gives the fixed scatter proportion.
num_points_in_std_array: number of points for std_array defining how finely the scatter calculations are.
RF_ROI_MIN: can be found from meta data.
B_field: can be put in hand or found by position of the peak of the frequency histogram.
shake_spectrum_parameters_json_path: path to json file storing shake spectrum parameters.
path_to_osc_strength_files: path to oscillator strength files.
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
        self.gases = reader.read_param(params, 'gases', ["H2", "He"])
        self.max_scatters = reader.read_param(params, 'max_scatters', 20)
        self.fixed_scatter_proportion = reader.read_param(params, 'fixed_scatter_proportion', True)
        if self.fixed_scatter_proportion == True:
            self.scatter_proportion = reader.read_param(params, 'gas_scatter_proportion', [])
        self.use_radiation_loss = reader.read_param(params, 'use_radiation_loss', True)
        # configure the resolution functions: simulated_resolution, gaussian_resolution, gaussian_lorentzian_composite_resolution
        self.resolution_function = reader.read_param(params, 'resolution_function', '')
        # This is an important parameter which determines how finely resolved
        # the scatter calculations are. 10000 seems to produce a stable fit, with minimal slowdown
        self.num_points_in_std_array = reader.read_param(params, 'num_points_in_std_array', 10000)
        self.RF_ROI_MIN = reader.read_param(params, 'RF_ROI_MIN', 25850000000.0)
        self.B_field = reader.read_param(params, 'B_field', 0.957810722501)
        self.shake_spectrum_parameters_json_path = reader.read_param(params, 'shake_spectrum_parameters_json_path', 'shake_spectrum_parameters.json')
        self.path_to_osc_strengths_files = reader.read_param(params, 'path_to_osc_strengths_files', '/host/')
        self.path_to_scatter_spectra_file = reader.read_param(params, 'path_to_scatter_spectra_file', '/host/')
        self.path_to_missing_track_radiation_loss_data_numpy_file = '/host/'
        self.path_to_ins_resolution_data_txt = reader.read_param(params, 'path_to_ins_resolution_data_txt', '/host/ins_resolution_all4.txt')
        self.path_to_quad_trap_eff_interp = reader.read_param(params, 'path_to_quad_trap_eff_interp', '/host/quad_interps.npy')

        if not os.path.exists(self.shake_spectrum_parameters_json_path):
            raise IOError('Shake spectrum path does not exist')
        if not os.path.exists(self.path_to_osc_strengths_files):
            raise IOError('Path to osc strengths files does not exist')
        return True

    def InternalRun(self):

        # Read shake parameters from JSON file
        self.shakeSpectrumClassInstance = ComplexLineShapeUtilities.ShakeSpectrumClass(self.shake_spectrum_parameters_json_path, self.std_eV_array())

        # number_of_events = len(self.data['StartFrequency'])
        # self.results = number_of_events

        a = self.data['StartFrequency']

        # fit with shake spectrum
        data_hist_freq, freq_bins= np.histogram(a,bins=self.bins_choice)
        # histogram = data_hist_freq
#         bins = freq_bins
#         guess = np.where(np.array(histogram) == np.max(histogram))[0][0]
#         kr17kev_in_hz = guess*(bins[1]-bins[0])+bins[0]
        #self.B_field = B(17.8, kr17kev_in_hz + 0)
        if self.fixed_scatter_proportion == True:
            if self.resolution_function == 'simulated_resolution':
                self.results = self.fit_data_ftc(freq_bins, data_hist_freq)
            elif self.resolution_function == 'gaussian_resolution':
                self.results = self.fit_data_1(freq_bins, data_hist_freq)
            elif self.resolution_function == 'gaussian_lorentzian_composite_resolution':
                self.results = self.fit_data_composite_gaussian_lorentzian(freq_bins, data_hist_freq)
        else:
            if self.resolution_function == 'simulated_resolution':
                self.results = self.fit_data_ftc_2(freq_bins, data_hist_freq)
            elif self.resolution_function == 'gaussian_resolution':
                self.results = self.fit_data(freq_bins, data_hist_freq)

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
        
    # A gaussian function
    def gaussian(self, x_array, A, sigma, mu):
        f = A*(1./(sigma*np.sqrt(2*np.pi)))*np.exp(-(((x_array-mu)/sigma)**2.)/2.)
        return f

    # A gaussian centered at 0 eV with variable width, on the SELA
    def std_gaussian(self, sigma):
        x_array = self.std_eV_array()
        ans = ComplexLineShapeUtilities.gaussian(x_array,1,sigma,0)
        return ans
    
    def composite_gaussian(self, A_array, sigma_array):
        x_array = self.std_eV_array()
        ans = 0
        for A, sigma in zip(A_array, sigma_array):
            ans += self.gaussian(x_array, A, sigma, 0)
        return ans

    def asym_triangle(self, x, scale1, scale2, center, exponent=1):
        index_below = np.where(x<center)
        index_above = np.where(x>=center)
        f_below = 1-np.abs((x-center)/scale1)**exponent
        f_above = 1-np.abs((x-center)/scale2)**exponent
        f_below[np.abs(x-center)>=np.abs(scale1)]=0.
        f_above[np.abs(x-center)>=np.abs(scale2)]=0.
        f = np.zeros(len(x))
        f[index_above] = f_above[index_above]
        f[index_below] = f_below[index_below]
        return f

    def smeared_triangle(self, x, center, scale1, scale2, exponent, sigma, amplitude):
        max_energy = 1000
        dx = x[1]-x[0]#E[1]-E[0]
        n_dx = round(max_energy/dx)
        x_smearing = np.arange(-n_dx*dx, n_dx*dx, dx)
        x_triangle = np.arange(min(x)-max_energy, max(x)+max_energy, dx)
        smearing = gaussian(x_smearing, 1, sigma, 0)
        triangle = asym_triangle(x_triangle, scale1, scale2, center, exponent)
        triangle_smeared = signal.convolve(triangle, smearing, mode='same')
        triangle_smeared_norm = triangle_smeared/np.sum(triangle_smeared)*amplitude
        return np.interp(x, x_triangle, triangle_smeared_norm)

    def std_smeared_triangle(self, center, scale1, scale2, exponent, sigma):
        x_array = std_eV_array()
        ans = self.smeared_triangle(x_array, center, scale1, scale2, exponent, sigma, 1)
        return ans

    def composite_gaussian_lorentzian(self, sigma):
        x_array = self.std_eV_array()
        w_g = x_array/sigma
        gamma = 0.8*sigma
        w_l = x_array/gamma 
        lorentzian = 1./(gamma*np.pi)*1./(1+(w_l**2))
        gaussian = 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-0.5*w_g**2)
        p = 0.8
        composite_function = p*gaussian+(1-p)*lorentzian
        return composite_function

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

        f_e_loss = ComplexLineShapeUtilities.get_eloss_spec(energy_loss_array, f, Constants.kr_k_line_e())
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
            f_radiation_loss = self.radiation_loss_f()
            scatter_spectra_single_gas[gas_type] = {}
            first_scatter = self.single_scatter_f(gas_type)
            if self.use_radiation_loss == True:
                first_scatter = self.normalize(signal.convolve(first_scatter, f_radiation_loss, mode = 'same'))
            scatter_num_array = range(2, self.max_scatters+1)
            current_scatter = first_scatter
            scatter_spectra_single_gas[gas_type][str(1).zfill(2)] = current_scatter
            # x = std_eV_array() # diagnostic
            for i in scatter_num_array:
                current_scatter = self.another_scatter(current_scatter, gas_type)
                if self.use_radiation_loss == True:
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
                generate_scatter_convolution_files()
        return

    # Given a function evaluated on the SELA, convolves it with a gaussian
    def convolve_gaussian(self, func_to_convolve,gauss_FWHM_eV):
        sigma = ComplexLineShapeUtilities.gaussian_FWHM_to_sigma(gauss_FWHM_eV)
        resolution_f = self.std_gaussian(sigma)
        ans = signal.convolve(resolution_f,func_to_convolve,mode='same')
        ans_normed = self.normalize(ans)
        return ans_normed
    
    def convolve_composite_gaussian(self, func_to_convolve, A_array, sigma_array):
        resolution_f = self.composite_gaussian(A_array, sigma_array)
        ans = signal.convolve(resolution_f, func_to_convolve, mode='same')
        ans_normed = self.normalize(ans)
        return ans_normed

    def convolve_composite_gaussian_lorentzian(self, func_to_convolve, sigma):
        resolution_f = self.composite_gaussian_lorentzian(sigma)
        ans = signal.convolve(resolution_f, func_to_convolve, mode='same')
        ans_normed = self.normalize(ans)
        return ans_normed

    def read_ins_resolution_data(self, path_to_ins_resolution_data_txt):
        ins_resolution_data = np.loadtxt(path_to_ins_resolution_data_txt)
        x_data = ins_resolution_data.T[0]
        y_data = ins_resolution_data.T[1]
        y_err_data = ins_resolution_data.T[2]
        return x_data, y_data, y_err_data

    def convolve_ins_resolution(self, working_spectrum):
        x_data, y_data, y_err_data = self.read_ins_resolution_data(self.path_to_ins_resolution_data_txt)
        f = interpolate.interp1d(x_data, y_data)
        x_array = self.std_eV_array()
        y_array = np.zeros(len(x_array))
        index_within_range_of_xdata = np.where((x_array >= x_data[0]) & (x_array <= x_data[-1]))
        y_array[index_within_range_of_xdata] = f(x_array[index_within_range_of_xdata])
        convolved_spectrum = signal.convolve(working_spectrum, y_array, mode = 'same')
        normalized_convolved_spectrum = self.normalize(convolved_spectrum)
        return normalized_convolved_spectrum

    def convolve_smeared_triangle(self, func_to_convolve, center, scale1, scale2, exponent, sigma):
        resolution_f = self.std_smeared_triangle(center, scale1, scale2, exponent, sigma)
        ans = signal.convolve(resolution_f, func_to_convolve, mode = 'same')
        ans_normed = normalize(ans)
        return ans_normed

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
        if self.fixed_scatter_proportion:
            if self.fit_ftc:
                fit_Hz = self.spectrum_func_ftc(bin_centers, *params)
            else:
                fit_Hz = self.spectrum_func_1(bin_centers, *params)
        else:
            if self.fit_ftc:
                fit_Hz = self.spectrum_func_ftc_2(bin_centers, *params)
            else:
                fit_Hz = self.spectrum_func(bin_centers, *params)
        chi2 = 2*((fit_Hz - data_hist_freq + data_hist_freq*np.log(data_hist_freq/fit_Hz))[nonzero_bins_index]).sum()
        chi2 += 2*(fit_Hz - data_hist_freq)[zero_bins_index].sum()
        return chi2

    def chi_2_Poisson_composite_gaussian_lorentzian_reso(self, bin_centers, data_hist_freq, eff_array, params):
        nonzero_bins_index = np.where(data_hist_freq != 0)
        zero_bins_index = np.where(data_hist_freq == 0)
        # expectation
        fit_Hz = self.spectrum_func_composite_gaussian_lorentzian(bin_centers, eff_array, *params)
        chi2 = 2*((fit_Hz - data_hist_freq + data_hist_freq*np.log(data_hist_freq/fit_Hz))[nonzero_bins_index]).sum()
        chi2 += 2*(fit_Hz - data_hist_freq)[zero_bins_index].sum()
        return chi2

    def reduced_chi2_Pearson_Neyman_composite(self, data_hist_freq, fit_Hz, number_of_parameters):
        nonzero_bins_index = np.where(data_hist_freq != 0)[0]
        zero_bins_index = np.where(data_hist_freq == 0)[0]
        fit_Hz_nonzero = fit_Hz[nonzero_bins_index]  
        data_Hz_nonzero = data_hist_freq[nonzero_bins_index] 
        fit_Hz_zero = fit_Hz[zero_bins_index]
        data_Hz_zero = data_hist_freq[zero_bins_index]
        chi2 = sum((fit_Hz_nonzero - data_Hz_nonzero)**2/data_Hz_nonzero) + sum((fit_Hz_nonzero - data_Hz_nonzero)**2/fit_Hz_nonzero)
        reduced_chi2 = chi2/(len(data_hist_freq) - number_of_parameters)
        return reduced_chi2

    # following the expression in the paper Steve BAKER and Robert D. COUSINS, (1984) CLARIFICATION OF THE USE OF CHI-SQUARE AND LIKELIHOOD FUNCTIONS IN FITS TO HISTOGRAMS
    def reduced_chi2_Poisson(self, data_hist_freq, fit_Hz, number_of_parameters):
        nonzero_bins_index = np.where(data_hist_freq != 0)
        zero_bins_index = np.where(data_hist_freq == 0)
        chi2 = 2*((fit_Hz - data_hist_freq + data_hist_freq*np.log(data_hist_freq/fit_Hz))[nonzero_bins_index]).sum()
        chi2 += 2*(fit_Hz - data_hist_freq)[zero_bins_index].sum()
        reduced_chi2 = chi2/(len(data_hist_freq) - number_of_parameters)
        return reduced_chi2

    def make_spectrum(self, gauss_FWHM_eV, prob_parameter, scatter_proportion, emitted_peak='shake'):
        gases = self.gases
        max_scatters = self.max_scatters
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
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        current_working_spectrum = self.convolve_gaussian(current_working_spectrum, gauss_FWHM_eV)
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
                    coefficient = coefficient/factorial(component)*p[i]**component*prob_parameter**M
                current_full_spectrum += coefficient*current_working_spectrum
        return current_full_spectrum

    # Produces a spectrum in real energy that can now be evaluated off of the SELA.
    #def spectrum_func(x_keV,FWHM_G_eV,line_pos_keV,scatter_prob,amplitude):
    def spectrum_func(self, bins_Hz, *p0):
        B_field = p0[0]
        FWHM_G_eV = p0[1]
        amplitude = p0[2]
        prob_parameter = p0[3]
        N = len(self.gases)
        scatter_proportion = p0[4:3+N]
        
        x_eV = ConversionFunctions.Energy(bins_Hz, B_field)
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        en_array_rev = ComplexLineShapeUtilities.flip_array(-1*en_loss_array)
        f = np.zeros(len(x_eV))
        f_intermediate = np.zeros(len(x_eV))

        x_eV_minus_line = Constants.kr_k_line_e() - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum(FWHM_G_eV, prob_parameter, scatter_proportion)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx],en_array_rev,full_spectrum)
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])
        return f

    # Call this function to fit a histogram of start frequencies with the model.
    # Note that the data_hist_freq should be the StartFrequencies as given by katydid,
    # which will be from ~0 MHZ to ~100 MHz. You must also pass this function the
    # self.RF_ROI_MIN value from the metadata file of your data.
    # You must also supply a guess for the self.B_field present for the run;
    # 0.959 T is usually sufficient.

    def fit_data(self, freq_bins, data_hist_freq, print_params=True):
        t = time.time()
        self.check_existence_of_scatter_file()
        bins_Hz = freq_bins + self.RF_ROI_MIN
        bins_Hz = 0.5*(bins_Hz[1:] + bins_Hz[:-1])
        bins_Hz_nonzero , data_hist_nonzero , data_hist_err = ComplexLineShapeUtilities.get_only_nonzero_bins(bins_Hz, data_hist_freq)
        # Initial guesses for curve_fit
        FWHM_guess = 5
        B_field_guess = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[np.argmax(data_hist_freq)])
        amplitude_guess = np.sum(data_hist_freq)/2
        prob_parameter_guess = 0.5
        scatter_proportion_guess = 0.5
        # Bounds for curve_fit
        B_field_min = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[0])
        B_field_max = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[-1])
        FWHM_eV_min = 1e-5
        FWHM_eV_max = ConversionFunctions.Energy(bins_Hz[0], B_field_guess) - ConversionFunctions.Energy(bins_Hz[-1], B_field_guess)
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist_freq)*3
        prob_parameter_min = 1e-5
        prob_parameter_max = 1
        scatter_proportion_min = 1e-5
        scatter_proportion_max = 1
        N = len(self.gases)
        p0_guess = [B_field_guess, FWHM_guess, amplitude_guess, prob_parameter_guess] + [scatter_proportion_guess]*(N-1)
        p0_bounds = ([B_field_min, FWHM_eV_min, amplitude_min, prob_parameter_min] + [scatter_proportion_min]*(N-1),  
                    [B_field_max, FWHM_eV_max, amplitude_max, prob_parameter_max] + [scatter_proportion_max]*(N-1) )
        # Actually do the fitting
        params , cov = curve_fit(self.spectrum_func, bins_Hz_nonzero, data_hist_nonzero, sigma=data_hist_err, p0=p0_guess, bounds=p0_bounds)
        # Name each of the resulting parameters and errors
        ################### Generalize to N Gases ###########################
        B_field_fit = params[0]
        FWHM_eV_fit = params[1]
        amplitude_fit = params[2]
        prob_parameter_fit = params[3]
        #starting at index 4, grabs every other entry. (which is how scattering probs are filled in for N gases)
        scatter_proportion_fit = list(params[4:3+N])+[1- sum(params[4:3+N])]
        total_counts_fit = amplitude_fit

        perr = np.sqrt(np.diag(cov))
        B_field_fit_err = perr[0]
        FWHM_eV_fit_err = perr[1]
        amplitude_fit_err = perr[2]
        prob_parameter_fit_err = perr[3]
        scatter_proportion_fit_err = list(perr[4:3+N])+[np.sqrt(sum(perr[4:3+N]**2))]
        total_counts_fit_err = amplitude_fit_err
    
        fit_Hz = self.spectrum_func(bins_Hz, *params)
        fit_keV = ComplexLineShapeUtilities.flip_array(fit_Hz)
        bins_keV = ConversionFunctions.Energy(bins_Hz, B_field_fit)/1000
        bins_keV = ComplexLineShapeUtilities.flip_array(bins_keV)
        
        nonzero_bins_index = np.where(data_hist_freq != 0)[0]
        zero_bins_index = np.where(data_hist_freq == 0)[0]
        fit_Hz_nonzero = fit_Hz[nonzero_bins_index]  
        data_Hz_nonzero = data_hist_freq[nonzero_bins_index] 
        fit_Hz_zero = fit_Hz[zero_bins_index]
        data_Hz_zero = data_hist_freq[zero_bins_index]
        chi2 = sum((fit_Hz_nonzero - data_Hz_nonzero)**2/data_Hz_nonzero) + sum((fit_Hz_nonzero - data_Hz_nonzero)**2/fit_Hz_nonzero)
        reduced_chi2 = chi2/(len(data_hist_freq)-4-len(self.gases)+1)
        elapsed = time.time() - t
        output_string = '\n'
        output_string += 'Reduced chi^2 = {:.2e}\n'.format(reduced_chi2)
        output_string += '-----------------\n'
        output_string += 'B field = {:.8e}'.format(B_field_fit)+' +/- '+ '{:.4e} T\n'.format(B_field_fit_err)
        output_string += '-----------------\n'
        output_string += 'Gaussian FWHM = '+str(round(FWHM_eV_fit,2))+' +/- '+str(round(FWHM_eV_fit_err,2))+' eV\n'
        output_string += '-----------------\n'
        output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
        output_string += '-----------------\n'
        output_string += 'Probability parameter \n= ' + "{:.2e}".format(prob_parameter_fit)\
        +' +/- ' + "{:.2e}".format(prob_parameter_fit_err)+'\n'
        output_string += '-----------------\n'
        for i in range(len(self.gases)):
            output_string += '{} Scatter proportion \n= '.format(self.gases[i]) + "{:.8e}".format(scatter_proportion_fit[i])\
            +' +/- ' + "{:.2e}".format(scatter_proportion_fit_err[i])+'\n'
            output_string += '-----------------\n'
        output_string += 'Fit completed in '+str(round(elapsed,2))+'s'+'\n'
        dictionary_of_fit_results = {
        'output_string': output_string,
        'cov': cov,
        'bins_keV': bins_keV,
        'fit': fit_keV,
        'bins_Hz': bins_Hz,
        'fit_Hz': fit_Hz,
        'B_field_fit': B_field_fit,
        'B_field_fit_err': B_field_fit_err,
        'FWHM_eV_fit': FWHM_eV_fit,
        'FWHM_eV_fit_err': FWHM_eV_fit_err,
        'prob_parameter_fit': prob_parameter_fit,
        'prob_parameter_fit_err': prob_parameter_fit_err,
        'scatter_proportion_fit': scatter_proportion_fit,
        'scatter_proportion_fit_err': scatter_proportion_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2
        }
        return dictionary_of_fit_results

    def make_spectrum_1(self, gauss_FWHM_eV, prob_parameter, emitted_peak='shake'):
        gases = self.gases
        current_path = self.path_to_scatter_spectra_file
        # check_existence_of_scatter_files()
        #filenames = list_files('scatter_spectra_files')
        p = np.zeros(len(gases))
        p = self.scatter_proportion
        scatter_spectra_file_path = os.path.join(current_path, 'scatter_spectra.npy')
        scatter_spectra = np.load(
        scatter_spectra_file_path, allow_pickle = True
        )
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        current_working_spectrum = self.convolve_gaussian(current_working_spectrum, gauss_FWHM_eV)
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
                    coefficient = coefficient/factorial(component)*p[i]**component*prob_parameter**M
                current_full_spectrum += coefficient*current_working_spectrum
        return current_full_spectrum

    def spectrum_func_1(self, bins_Hz, *p0):
        B_field = p0[0]
        FWHM_G_eV = p0[1]
        amplitude = p0[2]
        prob_parameter = p0[3]
        
        x_eV = ConversionFunctions.Energy(bins_Hz, B_field)
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        en_array_rev = ComplexLineShapeUtilities.flip_array(-1*en_loss_array)
        f = np.zeros(len(x_eV))
        f_intermediate = np.zeros(len(x_eV))

        x_eV_minus_line = Constants.kr_k_line_e() - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_1(FWHM_G_eV, prob_parameter,)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx],en_array_rev,full_spectrum)
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])
        return f

    def fit_data_1(self, freq_bins, data_hist_freq):
        t = time.time()
        self.check_existence_of_scatter_file()
        bins_Hz = freq_bins + self.RF_ROI_MIN
        bins_Hz = 0.5*(bins_Hz[1:] + bins_Hz[:-1])
        bins_Hz_nonzero , data_hist_nonzero , data_hist_err = ComplexLineShapeUtilities.get_only_nonzero_bins(bins_Hz, data_hist_freq)
        # Initial guesses for curve_fit
        FWHM_eV_guess = 5
        B_field_guess = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[np.argmax(data_hist_freq)])
        amplitude_guess = np.sum(data_hist_freq)/2
        prob_parameter_guess = 0.5
        # Bounds for curve_fit
        B_field_min = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[0])
        B_field_max = ComplexLineShapeUtilities.central_frequency_to_B_field(bins_Hz[-1])
        FWHM_eV_min = 1e-5
        FWHM_eV_max = ConversionFunctions.Energy(bins_Hz[0], B_field_guess) - ConversionFunctions.Energy(bins_Hz[-1], B_field_guess)
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist_freq)*3
        prob_parameter_min = 1e-5
        prob_parameter_max = 1
        
        # p0_guess = [B_field_guess, FWHM_guess, amplitude_guess, prob_parameter_guess] 
        # p0_bounds = ([B_field_min, FWHM_eV_min, amplitude_min, prob_parameter_min],  
                    # [B_field_max, FWHM_eV_max, amplitude_max, prob_parameter_max])
        # Actually do the fitting
        # params , cov = curve_fit(self.spectrum_func_1, bins_Hz_nonzero, data_hist_nonzero, sigma=data_hist_err, p0=p0_guess, bounds=p0_bounds)
        p0_guess = [B_field_guess, FWHM_eV_guess, amplitude_guess, prob_parameter_guess]
        p0_bounds = [(B_field_min,B_field_max), (FWHM_eV_min, FWHM_eV_max), (amplitude_min,amplitude_max), (prob_parameter_min, prob_parameter_max)]
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
        FWHM_eV_fit = params[1]
        amplitude_fit = params[2]
        prob_parameter_fit = params[3]
        total_counts_fit = amplitude_fit

        perr = m_binned.np_errors()
        B_field_fit_err = perr[0]
        FWHM_eV_fit_err = perr[1]
        amplitude_fit_err = perr[2]
        prob_parameter_fit_err = perr[3]
        total_counts_fit_err = amplitude_fit_err
    
        fit_Hz = self.spectrum_func_1(bins_Hz, *params)
        fit_keV = ComplexLineShapeUtilities.flip_array(fit_Hz)
        bins_keV = ConversionFunctions.Energy(bins_Hz, B_field_fit)/1000
        bins_keV = ComplexLineShapeUtilities.flip_array(bins_keV)

        nonzero_bins_index = np.where(data_hist_freq != 0)[0]
        zero_bins_index = np.where(data_hist_freq == 0)[0]
        fit_Hz_nonzero = fit_Hz[nonzero_bins_index]  
        data_Hz_nonzero = data_hist_freq[nonzero_bins_index] 
        fit_Hz_zero = fit_Hz[zero_bins_index]
        data_Hz_zero = data_hist_freq[zero_bins_index]
        reduced_chi2 = self.reduced_chi2_Poisson(data_hist_freq, fit_Hz, number_of_parameters = 4)
        elapsed = time.time() - t
        output_string = '\n'
        output_string += 'Reduced chi^2 = {:.2e}\n'.format(reduced_chi2)
        output_string += '-----------------\n'
        output_string += 'B field = {:.8e}'.format(B_field_fit)+' +/- '+ '{:.4e} T\n'.format(B_field_fit_err)
        output_string += '-----------------\n'
        output_string += 'Gaussian FWHM = '+str(round(FWHM_eV_fit,2))+' +/- '+str(round(FWHM_eV_fit_err,2))+' eV\n'
        output_string += '-----------------\n'
        output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
        output_string += '-----------------\n'
        output_string += 'Probability parameter \n= ' + "{:.2e}".format(prob_parameter_fit)\
        +' +/- ' + "{:.2e}".format(prob_parameter_fit_err)+'\n'
        output_string += '-----------------\n'
        output_string += 'Fit completed in '+str(round(elapsed,2))+'s'+'\n'
        dictionary_of_fit_results = {
        'output_string': output_string,
        'bins_keV': bins_keV,
        'fit_keV': fit_keV,
        'bins_Hz': bins_Hz,
        'fit_Hz': fit_Hz,
        'B_field_fit': B_field_fit,
        'B_field_fit_err': B_field_fit_err,
        'FWHM_eV_fit': FWHM_eV_fit,
        'FWHM_eV_fit_err': FWHM_eV_fit_err,
        'prob_parameter_fit': prob_parameter_fit,
        'prob_parameter_fit_err': prob_parameter_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2
        }
        return dictionary_of_fit_results

    def make_spectrum_ftc(self, prob_parameter, emitted_peak='shake'):
        gases = self.gases
        current_path = self.path_to_scatter_spectra_file
        # check_existence_of_scatter_files()
        #filenames = list_files('scatter_spectra_files')
        p = np.zeros(len(gases))
        p = self.scatter_proportion
        scatter_spectra_file_path = os.path.join(current_path, 'scatter_spectra.npy')
        scatter_spectra = np.load(
        scatter_spectra_file_path, allow_pickle = True
        )
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
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
                    coefficient = coefficient/factorial(component)*p[i]**component*prob_parameter**M
                current_full_spectrum += coefficient*current_working_spectrum
        return current_full_spectrum

    def spectrum_func_ftc(self, bins_Hz, *p0):
        B_field = p0[0]
        amplitude = p0[1]
        prob_parameter = p0[2]
        
        x_eV = ConversionFunctions.Energy(bins_Hz, B_field)
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        en_array_rev = ComplexLineShapeUtilities.flip_array(-1*en_loss_array)
        f = np.zeros(len(x_eV))
        f_intermediate = np.zeros(len(x_eV))

        x_eV_minus_line = Constants.kr_k_line_e() - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_ftc(prob_parameter)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx],en_array_rev,full_spectrum)
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])
        return f

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
        prob_parameter_fit = params[2]
        total_counts_fit = amplitude_fit

        perr = m_binned.np_errors()
        B_field_fit_err = perr[0]
        amplitude_fit_err = perr[1]
        prob_parameter_fit_err = perr[2]
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
        output_string += 'Probability parameter \n= ' + "{:.2e}".format(prob_parameter_fit)\
        +' +/- ' + "{:.2e}".format(prob_parameter_fit_err)+'\n'
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
        'prob_parameter_fit': prob_parameter_fit,
        'prob_parameter_fit_err': prob_parameter_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2
        }
        return dictionary_of_fit_results


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
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
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
                    coefficient = coefficient/factorial(component)*p[i]**component*prob_parameter**M
                current_full_spectrum += coefficient*current_working_spectrum
        return current_full_spectrum

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

        x_eV_minus_line = Constants.kr_k_line_e() - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_ftc_2(prob_parameter, scatter_proportion)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx],en_array_rev,full_spectrum)
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])
        return f

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
        'prob_parameter_fit': prob_parameter_fit,
        'prob_parameter_fit_err': prob_parameter_fit_err,
        'scatter_proportion_fit': scatter_proportion_fit,
        'scatter_proportion_fit_err': scatter_proportion_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2
        }
        return dictionary_of_fit_results

    def make_spectrum_smeared_triangle(self, prob_parameter, center, scale1, scale2, exponent, sigma, emitted_peak='shake'):
        current_path = get_current_path()
        # check_existence_of_scatter_files()
        #filenames = list_files('scatter_spectra_files')
        p = np.zeros(len(gases))
        p = scatter_proportion
        scatter_spectra = np.load('scatter_spectra_file/scatter_spectra.npy', allow_pickle = True)
        en_array = std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        current_working_spectrum = self.convolve_smeared_triangle(current_working_spectrum, center, scale1, scale2, exponent, sigma)
        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += current_working_spectrum
        N = len(gases)
        for M in range(1, max_scatters + 1):
            gas_scatter_combinations = np.array([np.array(i) for i in product(range(M+1), repeat=N) if sum(i)==M])
            for combination in gas_scatter_combinations:
                #print(combination)
                entry_str = ''
                for component, gas_type in zip(combination, gases):
                    entry_str += gas_type
                    entry_str += str(component).zfill(2)
                current_working_spectrum = scatter_spectra.item()[entry_str]
                current_working_spectrum = self.normalize(sp.signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
                coefficient = factorial(sum(combination))
                for component, i in zip(combination, range(len(gases))):
                    coefficient = coefficient/factorial(component)*p[i]**component*prob_parameter**M
                current_full_spectrum += coefficient*current_working_spectrum

        return current_full_spectrum

    def spectrum_func_smeared_triangle(self, bins_Hz, *p0):
    
        B_field = p0[0]
        amplitude = p0[1]
        prob_parameter = p0[2]
        center = p0[3]
        scale1 = p0[4]
        scale2 = p0[5]
        exponent = p0[6]
        sigma = p0[7]
    
        x_eV = Energy(bins_Hz, B_field)
        en_loss_array = std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        en_array_rev = flip_array(-1*en_loss_array)
        f = np.zeros(len(x_eV))
        f_intermediate = np.zeros(len(x_eV))

        x_eV_minus_line = kr_line*1000 - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]
    
        full_spectrum = self.make_spectrum_smeared_triangle(prob_parameter, center, scale1, scale2, exponent, sigma)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx], en_loss_array, full_spectrum)
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])
        return f

    def fit_data_smeared_triangle(self, RF_ROI_MIN, freq_bins, data_hist_freq, print_params=True):
        t = time.time()
        self.check_existence_of_scatter_files()
        bins_Hz = freq_bins + RF_ROI_MIN
        bins_Hz = 0.5*(bins_Hz[1:] + bins_Hz[:-1])
        bins_Hz_nonzero , data_hist_nonzero , data_hist_err = get_only_nonzero_bins(bins_Hz, data_hist_freq)
        # Initial guesses for curve_fit
        B_field_guess = central_frequency_to_B_field(bins_Hz[np.argmax(data_hist_freq)])
        amplitude_guess = np.sum(data_hist_freq)/2
        prob_parameter_guess = 0.5
        center_guess = 0
        scale_guess = 5
        exponent_guess = 1
        sigma_guess = 3
        # Bounds for curve_fit
        B_field_min = central_frequency_to_B_field(bins_Hz[0])
        B_field_max = central_frequency_to_B_field(bins_Hz[-1])
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist_freq)*3
        prob_parameter_min = 1e-5
        prob_parameter_max = 1
        center_min = -5
        center_max = 5
        width_min = 0
        width_max = Energy(bins_Hz[0], B_field_guess) - Energy(bins_Hz[-1], B_field_guess)
        exponent_min = 0.5
        exponent_max = 2
    
        p0_guess = [B_field_guess, amplitude_guess, prob_parameter_guess, center_guess, scale_guess, scale_guess, exponent_guess, sigma_guess]
        p0_bounds = [(B_field_min,B_field_max), (amplitude_min,amplitude_max), (prob_parameter_min, prob_parameter_max), (center_min, center_max), (width_min, width_max), (width_min, width_max), (exponent_min, exponent_max), (width_min, width_max)]           
        #step_size = [1e-6, 5, 500, 0.1]
        # Actually do the fitting
        print(p0_guess)
        print(p0_bounds)
        m_binned = Minuit.from_array_func(lambda p: chi2_Poisson(bins_Hz, data_hist_freq, p),
                                          start = p0_guess,
                                          limit = p0_bounds,
                                          throw_nan = True
                                          )
        m_binned.migrad()
        params = m_binned.np_values()
        B_field_fit = params[0]
        #starting at index 2, grabs every other entry. (which is how scattering probs are filled in for N gases)
        amplitude_fit = params[1]
        prob_parameter_fit = params[2]
        center_fit = params[3]
        scale1_fit = params[4]
        scale2_fit = params[5]
        exponent_fit = params[6]
        sigma_fit = params[7]
        total_counts_fit = amplitude_fit

        perr = m_binned.np_errors()
        B_field_fit_err = perr[0]
        amplitude_fit_err = perr[1]
        prob_parameter_fit_err = perr[2]
        center_fit_err = perr[3]
        scale1_fit_err = perr[4]
        scale2_fit_err = perr[5]
        exponent_fit_err = perr[6]
        sigma_fit_err = perr[7]
        total_counts_fit_err = amplitude_fit_err
    
        fit_Hz = spectrum_func(bins_Hz,*params)
        fit_keV = flip_array(fit_Hz)
        bins_keV = Energy(bins_Hz, B_field_fit)/1000
        bins_keV = flip_array(bins_keV)
        reduced_chi2 = reduced_chi2_Poisson(data_hist_freq, fit_Hz, number_of_parameters = len(params)) 
        if print_params == True:
            output_string = 'Reduced chi^2 = {:.2e}\n'.format(reduced_chi2)
            output_string += '-----------------\n'
            output_string += 'B field = {:.8e}'.format(B_field_fit)+' +/- '+ '{:.4e} T\n'.format(B_field_fit_err)
            output_string += '-----------------\n'
            output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
            output_string += '-----------------\n'
            output_string += 'Probability parameter \n= ' + "{:.2e}".format(prob_parameter_fit) + ' +/- ' + "{:.2e}".format(prob_parameter_fit_err)+'\n'
            output_string += '-----------------\n'
            output_string += 'Center = {:.4e}'.format(center_fit) + ' +/- {:.4e}'.format(center_fit_err) + '\n'
            output_string += '-----------------\n'
            output_string += 'Scale1 = {:.4e}'.format(scale1_fit) + ' +/- {:.4e}'.format(scale1_fit_err) + '\n'
            output_string += '-----------------\n'
            output_string += 'Scale2 = {:.4e}'.format(scale2_fit) + ' +/- {:.4e}'.format(scale2_fit_err) + '\n'
            output_string += '-----------------\n'
            output_string += 'Exponent = {:.4e}'.format(exponent_fit) + ' +/- {:.4e}'.format(exponent_fit_err) + '\n'
            output_string += '-----------------\n'
            output_string += 'Sigma = {:.4e}'.format(sigma_fit) + ' +/- {:.4e}'.format(sigma_fit_err) + '\n'
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
        'prob_parameter_fit': prob_parameter_fit,
        'prob_parameter_fit_err': prob_parameter_fit_err,
        'center_fit': scatter_proportion_fit,
        'center_fit_err': scatter_proportion_fit_err,
        'scale1_fit': scale1_fit,
        'scale1_fit_err': scale1_fit_err,
        'scale2_fit': scale2_fit,
        'scale2_fit_err': scale2_fit_err,
        'exponent_fit': exponent_fit,
        'exponent_fit_err': exponent_fit_err,
        'sigma_fit': sigma_fit,
        'sigma_fit_err': sigma_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq,
        'reduced_chi2': reduced_chi2
        }
        return dictionary_of_fit_results

    def make_spectrum_composite_gaussian_lorentzian(self, survival_prob, sigma, emitted_peak='shake'):
        p = self.scatter_proportion
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
            relative_reconstruction_eff = np.exp(-0.643*M**0.74)
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
                for i in range(0, M+1):
                    coefficient = coefficient*(1-0.023*np.exp(-0.643*i**0.74))
                current_full_spectrum += relative_reconstruction_eff*coefficient*current_working_spectrum*survival_prob**M
        return current_full_spectrum

    def spectrum_func_composite_gaussian_lorentzian(self, bins_Hz, eff_array, *p0):
    
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

        x_eV_minus_line = Constants.kr_k_line_e() - x_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_eV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_composite_gaussian_lorentzian(survival_prob, sigma)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx], en_loss_array, full_spectrum)
        f_intermediate = f_intermediate*eff_array
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])

        return f

    def fit_data_composite_gaussian_lorentzian(self, freq_bins, data_hist_freq, print_params=True):
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
    
        fit_Hz = self.spectrum_func_composite_gaussian_lorentzian(bins_Hz, eff_array, *params)
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
            output_string += 'Survival probability \n= ' + "{:.2e}".format(survival_prob_fit)\
            +' +/- ' + "{:.2e}".format(survival_prob_fit_err)+'\n'
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
