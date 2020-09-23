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
base_shape: "shake", "lorentzian" or "dirac"
'''

from __future__ import absolute_import

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import comb
from scipy import integrate , signal, interpolate
import os
import time
import sys
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.misc import Constants, ComplexLineShapeUtilities, ConversionFunctions

logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

class KrComplexLineShape(BaseProcessor):

    def InternalConfigure(self, params):
        '''
        Configure
        '''
        # Read other parameters
        self.bins_choice = reader.read_param(params, 'bins_choice', [])
        self.gases = reader.read_param(params, 'gases', ["H2","Kr"])
        self.max_scatters = reader.read_param(params, 'max_scatters', 20)
        self.max_comprehensive_scatters = reader.read_param(params, 'max_comprehensive_scatters', 20)
        self.fix_scatter_proportion = reader.read_param(params, 'fix_scatter_proportion', True)
        if self.fix_scatter_proportion == True:
            self.scatter_proportion = reader.read_param(params, 'gas1_scatter_proportion', 0.8)
            logger.info('Using an H2 scatter proportion of {} with gases {}'.format(self.gases, self.scatter_proportion))
        # This is an important parameter which determines how finely resolved
        # the scatter calculations are. 10000 seems to produce a stable fit, with minimal slowdown
        self.num_points_in_std_array = reader.read_param(params, 'num_points_in_std_array', 10000)
        self.RF_ROI_MIN = reader.read_param(params, 'RF_ROI_MIN', 25850000000.0)
        self.B_field = reader.read_param(params, 'B_field', 0.957810722501)
        self.shake_spectrum_parameters_json_path = reader.read_param(params, 'shake_spectrum_parameters_json_path', 'shake_spectrum_parameters.json')
        self.base_shape = reader.read_param(params, 'base_shape', 'shake')
        self.path_to_osc_strengths_files = reader.read_param(params, 'path_to_osc_strengths_files', '/host/')
        self.path_to_ins_resolution_data_txt = reader.read_param(params, 'path_to_ins_resolution_data_txt', '/host/')

        if self.base_shape=='shake' and not os.path.exists(self.shake_spectrum_parameters_json_path):
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
        histogram = data_hist_freq
        bins = freq_bins
        guess = np.where(np.array(histogram) == np.max(histogram))[0][0]
        kr17kev_in_hz = guess*(bins[1]-bins[0])+bins[0]
        #self.B_field = B(17.8, kr17kev_in_hz + 0)
        if self.fix_scatter_proportion == True:
            logger.info('Scatter proportion fixed')
            self.results = self.fit_data_1(freq_bins, data_hist_freq)
        else:
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
        ans = ComplexLineShapeUtilities.lorentzian(x_array,0,ComplexLineShapeUtilities.kr_17keV_line_width)
        return ans

    #A Dirac delta functin
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


    # A gaussian centered at 0 eV with variable width, on the SELA
    def std_gaussian(self, sigma):
        x_array = self.std_eV_array()
        ans = ComplexLineShapeUtilities.gaussian(x_array,1,sigma,0)
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
        input_filename = os.path.join(self.path_to_osc_strengths_files,  "{}OscillatorStrength.txt".format(gas_type))
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

    # Convolves the scatter functions and saves
    # the results to a .npy file.
    def generate_scatter_convolution_file(self):
        t = time.time()
        scatter_spectra_single_gas = {}
        for gas_type in self.gases:
            scatter_spectra_single_gas[gas_type] = {}
            first_scatter = self.single_scatter_f(gas_type)
            scatter_num_array = range(2, self.max_scatters+1)
            current_scatter = first_scatter
            scatter_spectra_single_gas[gas_type][str(1).zfill(2)] = current_scatter
            for i in scatter_num_array:
                current_scatter = self.another_scatter(current_scatter, gas_type)
                scatter_spectra_single_gas[gas_type][str(i).zfill(2)] = current_scatter
        scatter_spectra = {}
        scatter_spectra['{}_{}'.format(self.gases[0], self.gases[1])] = {}
        for i in range(1, self.max_scatters+1):
            scatter_spectra['{}_{}'.format(self.gases[0], self.gases[1])]['{}_{}'.format(str(i).zfill(2), str(0).zfill(2))]\
            = scatter_spectra_single_gas[self.gases[0]][str(i).zfill(2)]
            scatter_spectra['{}_{}'.format(self.gases[0], self.gases[1])]['{}_{}'.format(str(0).zfill(2), str(i).zfill(2))]\
            = scatter_spectra_single_gas[self.gases[1]][str(i).zfill(2)]
        # i represents total order, while j represents suborder for H2, thus the suborder for Kr is i-j
        for i in range(1, self.max_scatters+1):
            for j in range(1, i):
                H2_scatter = scatter_spectra_single_gas[self.gases[0]][str(j).zfill(2)]
                Kr_scatter = scatter_spectra_single_gas[self.gases[1]][str(i-j).zfill(2)]
                total_scatter = self.normalize(signal.convolve(H2_scatter, Kr_scatter, mode='same'))
                scatter_spectra['{}_{}'.format(self.gases[0], self.gases[1])]['{}_{}'.format(str(j).zfill(2), str(i-j).zfill(2))] = total_scatter
        np.save(
        os.path.join(self.path_to_osc_strengths_files, 'scatter_spectra_file/scatter_spectra.npy'),
        scatter_spectra
        )
        elapsed = time.time() - t
        logger.info('File generated in '+str(elapsed)+'s')
        return

    # Checks for the existence of a directory called 'scatter_spectra_file'
    # and checks that this directory contains the scatter spectra files.
    # If not, this function calls generate_scatter_convolution_file.
    # This function also checks to make sure that the scatter file have the correct
    # number of entries and correct number of points in the SELA, and if not, it generates a fresh file.
    # When the variable regenerate is set as True, it generates a fresh file
    def check_existence_of_scatter_file(self, regenerate = False):
        gases = self.gases
        if regenerate == True:
            logger.info('generate fresh scatter file')
            self.generate_scatter_convolution_file()
        else:
            stuff_in_dir = os.listdir(self.path_to_osc_strengths_files)
            if 'scatter_spectra_file' not in stuff_in_dir:
                logger.info('Scatter spectra folder not found, generating')
                os.mkdir(os.path.join(self.path_to_osc_strengths_files,'scatter_spectra_file'))
                time.sleep(2)
                self.generate_scatter_convolution_file()
            else:
                directory = os.listdir(self.path_to_osc_strengths_files + "/scatter_spectra_file")
                strippeddirs = [s.strip('\n') for s in directory]
                if 'scatter_spectra.npy' not in strippeddirs:
                    self.generate_scatter_convolution_file()
                test_file = os.path.join(self.path_to_osc_strengths_files, 'scatter_spectra_file/scatter_spectra.npy')
                test_dict = np.load(test_file, allow_pickle = True)
                if list(test_dict.item().keys())[0] != '{}_{}'.format(gases[0], gases[1]):
                    logger.info('first entry not matching, generating fresh files')
                    self.generate_scatter_convolution_file()
                elif len(test_dict.item()['{}_{}'.format(gases[0], gases[1])]) \
                            != (1 + self.max_scatters+1)*(self.max_scatters+1)/2 - 1:
                    logger.info('Number of scatters not matching, generating fresh files')
                elif len(test_dict.item()['{}_{}'.format(gases[0], gases[1])]['01_00'])\
                != self.num_points_in_std_array:
                    logger.info('Binning do not match standard array, generating fresh files')
                    self.generate_scatter_convolution_file()
        return

    # Given a function evaluated on the SELA, convolves it with a gaussian
    def convolve_gaussian(self, func_to_convolve,gauss_FWHM_eV):
        sigma = ComplexLineShapeUtilities.gaussian_FWHM_to_sigma(gauss_FWHM_eV)
        resolution_f = self.std_gaussian(sigma)
        ans = signal.convolve(resolution_f,func_to_convolve,mode='same')
        ans_normed = self.normalize(ans)
        return ans_normed
        
    def read_ins_resolution_data(self, path_to_ins_resolution_data_txt):
        ins_resolution_data = np.loadtxt(path_to_ins_resolution_data_txt+'ins_resolution_all4.txt')
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

    def make_spectrum(self, prob_parameter, scatter_proportion, emitted_peak='shake'):
        gases = self.gases
        max_scatters = self.max_scatters
        max_comprehensive_scatters = self.max_comprehensive_scatters
        current_path = self.path_to_osc_strengths_files
        # check_existence_of_scatter_files()
        #filenames = list_files('scatter_spectra_files')
        p = scatter_proportion
        q = 1 - p
        scatter_spectra = np.load(
        current_path + 'scatter_spectra_file/scatter_spectra.npy', allow_pickle = True
        )
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        elif emitted_peak == 'dirac':
            current_working_spectrum = self.std_dirac()
        current_working_spectrum = self.convolve_ins_resolution(current_working_spectrum)
        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += current_working_spectrum
        for n in range(1, max_comprehensive_scatters + 1):
            for r in range(0, n + 1):
                current_working_spectrum = \
                scatter_spectra.item()['{}_{}'.format(gases[0], gases[1])]\
                ['{}_{}'.format(str(r).zfill(2), str(n-r).zfill(2))]
                current_working_spectrum = \
                self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
                current_full_spectrum += current_working_spectrum*comb(n, r)\
                *(prob_parameter*p)**(r)*(prob_parameter*q)**(n-r)

        for n in range(max_comprehensive_scatters + 1, max_scatters + 1):
            current_working_spectrum = \
            scatter_spectra.item()['{}_{}'.format(gases[0], gases[1])]\
            ['{}_{}'.format(str(n).zfill(2), str(0).zfill(2))]
            current_working_spectrum = \
            self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
            current_full_spectrum += current_working_spectrum*(prob_parameter*p)**(n)

            current_working_spectrum = \
            scatter_spectra.item()['{}_{}'.format(gases[0], gases[1])]\
            ['{}_{}'.format(str(0).zfill(2), str(n).zfill(2))]
            current_working_spectrum = \
            self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
            current_full_spectrum += current_working_spectrum*(prob_parameter*q)**(n)

        return current_full_spectrum

    # Produces a spectrum in real energy that can now be evaluated off of the SELA.
    #def spectrum_func(x_keV,FWHM_G_eV,line_pos_keV,scatter_prob,amplitude):
    def spectrum_func(self, x_keV, *p0):
        x_eV = x_keV*1000.
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        en_array_rev = ComplexLineShapeUtilities.flip_array(-1*en_loss_array)
        f = np.zeros(len(x_keV))
        f_intermediate = np.zeros(len(x_keV))

        line_pos_keV = p0[0]
        amplitude = p0[1]
        prob_parameter = p0[2]
        scatter_proportion = p0[3]

        line_pos_eV = line_pos_keV*1000.
        x_eV_minus_line = x_eV - line_pos_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_keV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum(prob_parameter, scatter_proportion)
        full_spectrum_rev = ComplexLineShapeUtilities.flip_array(full_spectrum)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx],en_array_rev,full_spectrum_rev)
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
        bins_keV = ConversionFunctions.Energy(bins_Hz, self.B_field)/1000
        bins_keV = ComplexLineShapeUtilities.flip_array(bins_keV)
        data_hist = ComplexLineShapeUtilities.flip_array(data_hist_freq)
        #data_hist_err = ComplexLineShapeUtilities.get_hist_err_bins(data_hist)
        bins_keV_nonzero , data_hist_nonzero , data_hist_err = ComplexLineShapeUtilities.get_only_nonzero_bins(bins_keV, data_hist)
        # Bounds for curve_fit
        FWHM_eV_min = 1e-5
        FWHM_eV_max = (bins_keV[len(bins_keV)-1] - bins_keV[0])*1000
        line_pos_keV_min = bins_keV[0]
        line_pos_keV_max = bins_keV[len(bins_keV)-1]
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist)*3
        prob_parameter_min = 1e-5
        prob_parameter_max = 1
        scatter_proportion_min = 1e-5
        scatter_proportion_max = 1
        # Initial guesses for curve_fit
        line_pos_guess = bins_keV[np.argmax(data_hist)]
        amplitude_guess = np.sum(data_hist)/2
        prob_parameter_guess = 0.5
        scatter_proportion_guess = 0.9
        p0_guess = [FWHM_guess, line_pos_guess, amplitude_guess, prob_parameter_guess, scatter_proportion_guess]
        p0_bounds = ([FWHM_eV_min, line_pos_keV_min, amplitude_min, prob_parameter_min, scatter_proportion_min],
                    [FWHM_eV_max, line_pos_keV_max, amplitude_max, prob_parameter_max, scatter_proportion_max])
        # Actually do the fitting
        params , cov = curve_fit(self.spectrum_func, bins_keV_nonzero, data_hist_nonzero, sigma=data_hist_err, p0=p0_guess, bounds=p0_bounds)
        # Name each of the resulting parameters and errors
        ################### Generalize to N Gases ###########################
        FWHM_G_eV_fit = params[0]
        line_pos_keV_fit = params[1]
        #starting at index 2, grabs every other entry. (which is how scattering probs are filled in for N gases)
        amplitude_fit = params[2]
        prob_parameter_fit = params[3]
        scatter_proportion_fit = params[4]
        total_counts_fit = amplitude_fit

        perr = np.sqrt(np.diag(cov))
        FWHM_eV_G_fit_err = perr[0]
        line_pos_keV_fit_err = perr[1]
        amplitude_fit_err = perr[2]
        prob_parameter_fit_err = perr[3]
        scatter_proportion_fit_err = perr[4]
        total_counts_fit_err = amplitude_fit_err

        fit = self.spectrum_func(bins_keV,*params)

        line_pos_Hz_fit , line_pos_Hz_fit_err = ComplexLineShapeUtilities.energy_guess_to_frequency(line_pos_keV_fit, line_pos_keV_fit_err, self.B_field)
        B_field_fit , B_field_fit_err = ComplexLineShapeUtilities.central_frequency_to_B_field(line_pos_Hz_fit, line_pos_Hz_fit_err)
        fit_Hz = ComplexLineShapeUtilities.flip_array(fit)
        bins_keV = bins_keV - line_pos_keV_fit + Constants.kr_k_line_e()/1000
        FWHM_eV_fit = FWHM_G_eV_fit
        FWHM_eV_fit_err = FWHM_eV_G_fit_err
        elapsed = time.time() - t
        output_string = '\n'
        output_string += 'B field = {:.6e}'.format(B_field_fit)+' +/- '+ '{:.2e} T\n'.format(B_field_fit_err)
        output_string += '-----------------\n'
        output_string += 'Gaussian FWHM = '+str(round(FWHM_G_eV_fit,2))+' +/- '+str(round(FWHM_eV_G_fit_err,2))+' eV\n'
        output_string += '-----------------\n'
        output_string += 'Line position \n= '+str(round(line_pos_Hz_fit,2))+' +/- '+str(round(line_pos_Hz_fit_err,2))+' Hz\n'
        output_string += '-----------------\n'
        output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
        output_string += '-----------------\n'
        output_string += 'Probability parameter \n= ' + "{:.2e}".format(prob_parameter_fit)\
        +' +/- ' + "{:.2e}".format(prob_parameter_fit_err)+'\n'
        output_string += '-----------------\n'
        output_string += '{} Scatter proportion \n= '.format(self.gases[0]) + "{:.2e}".format(scatter_proportion_fit)\
        +' +/- ' + "{:.2e}".format(scatter_proportion_fit_err)+'\n'
        output_string += '-----------------\n'
        output_string += 'Fit completed in '+str(round(elapsed,2))+'s'+'\n'
        dictionary_of_fit_results = {
        'output_string': output_string,
        'cov': cov,
        'bins_keV': bins_keV,
        'fit': fit,
        'bins_Hz': bins_Hz,
        'fit_Hz': fit_Hz,
        'FWHM_eV_fit': FWHM_eV_fit,
        'FWHM_eV_fit_err': FWHM_eV_fit_err,
        'line_pos_Hz_fit': line_pos_Hz_fit,
        'line_pos_Hz_fit_err': line_pos_Hz_fit_err,
        'B_field_fit': B_field_fit,
        'B_field_fit_err': B_field_fit_err,
        'prob_parameter_fit': prob_parameter_fit,
        'prob_parameter_fit_err': prob_parameter_fit_err,
        'scatter_proportion_fit': scatter_proportion_fit,
        'scatter_proportion_fit_err': scatter_proportion_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq
        }
        return dictionary_of_fit_results

    def make_spectrum_1(self, prob_parameter, emitted_peak='shake'):
        gases = self.gases
        max_scatters = self.max_scatters
        max_comprehensive_scatters = self.max_comprehensive_scatters
        current_path = self.path_to_osc_strengths_files
        # check_existence_of_scatter_files()
        #filenames = list_files('scatter_spectra_files')
        p = self.scatter_proportion
        q = 1 - p
        scatter_spectra = np.load(
        os.path.join(current_path,'scatter_spectra_file/scatter_spectra.npy'), allow_pickle = True
        )
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        elif emitted_peak == 'dirac':
            current_working_spectrum = self.std_dirac()
        current_working_spectrum = self.convolve_ins_resolution(current_working_spectrum)
        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += current_working_spectrum
        for n in range(1, max_comprehensive_scatters + 1):
            for r in range(0, n + 1):
                current_working_spectrum = \
                scatter_spectra.item()['{}_{}'.format(gases[0], gases[1])]\
                ['{}_{}'.format(str(r).zfill(2), str(n-r).zfill(2))]
                current_working_spectrum = \
                self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
                current_full_spectrum += current_working_spectrum*comb(n, r)\
                *(prob_parameter*p)**(r)*(prob_parameter*q)**(n-r)
        for n in range(max_comprehensive_scatters + 1, max_scatters + 1):
            current_working_spectrum = \
            scatter_spectra.item()['{}_{}'.format(gases[0], gases[1])]\
            ['{}_{}'.format(str(n).zfill(2), str(0).zfill(2))]
            current_working_spectrum = \
            self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
            current_full_spectrum += current_working_spectrum*(prob_parameter*p)**(n)

            current_working_spectrum = \
            scatter_spectra.item()['{}_{}'.format(gases[0], gases[1])]\
            ['{}_{}'.format(str(0).zfill(2), str(n).zfill(2))]
            current_working_spectrum = \
            self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
            current_full_spectrum += current_working_spectrum*(prob_parameter*q)**(n)

        return current_full_spectrum

    def spectrum_func_1(self, x_keV, *p0):
        x_eV = x_keV*1000.
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        en_array_rev = ComplexLineShapeUtilities.flip_array(-1*en_loss_array)
        f = np.zeros(len(x_keV))
        f_intermediate = np.zeros(len(x_keV))

        line_pos_keV = p0[0]
        amplitude = p0[1]
        prob_parameter = p0[2]

        line_pos_eV = line_pos_keV*1000.
        x_eV_minus_line = x_eV - line_pos_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_keV)) if i not in zero_idx]

        full_spectrum = self.make_spectrum_1(prob_parameter, emitted_peak=self.base_shape)
        full_spectrum_rev = ComplexLineShapeUtilities.flip_array(full_spectrum)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx],en_array_rev,full_spectrum_rev)
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])
        return f

    def fit_data_1(self, freq_bins, data_hist_freq):
        t = time.time()
        self.check_existence_of_scatter_file()
        bins_Hz = freq_bins + self.RF_ROI_MIN
        bins_Hz = 0.5*(bins_Hz[1:] + bins_Hz[:-1])
        bins_keV = ConversionFunctions.Energy(bins_Hz, self.B_field)/1000
        bins_keV = ComplexLineShapeUtilities.flip_array(bins_keV)
        data_hist = ComplexLineShapeUtilities.flip_array(data_hist_freq)
        #data_hist_err = ComplexLineShapeUtilities.get_hist_err_bins(data_hist)
        bins_keV_nonzero , data_hist_nonzero , data_hist_err = ComplexLineShapeUtilities.get_only_nonzero_bins(bins_keV, data_hist)
        # Bounds for curve_fit
        FWHM_eV_min = 1e-5
        FWHM_eV_max = (bins_keV[len(bins_keV)-1] - bins_keV[0])*1000
        line_pos_keV_min = bins_keV[0]
        line_pos_keV_max = bins_keV[len(bins_keV)-1]
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist)*3
        prob_parameter_min = 1e-5
        prob_parameter_max = 1
        # Initial guesses for curve_fit
        FWHM_guess = 5
        line_pos_guess = bins_keV[np.argmax(data_hist)]
        amplitude_guess = np.sum(data_hist)/2
        prob_parameter_guess = 0.5
        p0_guess = [FWHM_guess, line_pos_guess, amplitude_guess, prob_parameter_guess]
        p0_bounds = ([FWHM_eV_min, line_pos_keV_min, amplitude_min, prob_parameter_min],
                    [FWHM_eV_max, line_pos_keV_max, amplitude_max, prob_parameter_max])
        # Actually do the fitting
        params , cov = curve_fit(self.spectrum_func_1, bins_keV_nonzero, data_hist_nonzero, sigma=data_hist_err, p0=p0_guess, bounds=p0_bounds)
        # Name each of the resulting parameters and errors
        ################### Generalize to N Gases ###########################
        FWHM_G_eV_fit = params[0]
        line_pos_keV_fit = params[1]
        #starting at index 2, grabs every other entry. (which is how scattering probs are filled in for N gases)
        amplitude_fit = params[2]
        prob_parameter_fit = params[3]
        total_counts_fit = amplitude_fit

        perr = np.sqrt(np.diag(cov))
        FWHM_eV_G_fit_err = perr[0]
        line_pos_keV_fit_err = perr[1]
        amplitude_fit_err = perr[2]
        prob_parameter_fit_err = perr[3]
        total_counts_fit_err = amplitude_fit_err

        fit = self.spectrum_func_1(bins_keV,*params)

        line_pos_Hz_fit , line_pos_Hz_fit_err = ComplexLineShapeUtilities.energy_guess_to_frequency(line_pos_keV_fit, line_pos_keV_fit_err, self.B_field)
        B_field_fit , B_field_fit_err = ComplexLineShapeUtilities.central_frequency_to_B_field(line_pos_Hz_fit, line_pos_Hz_fit_err)
        fit_Hz = ComplexLineShapeUtilities.flip_array(fit)
        bins_keV = bins_keV - line_pos_keV_fit + Constants.kr_k_line_e()/1000
        FWHM_eV_fit = FWHM_G_eV_fit
        FWHM_eV_fit_err = FWHM_eV_G_fit_err
        elapsed = time.time() - t
        output_string = '\n'
        output_string += 'B field = {:.6e}'.format(B_field_fit)+' +/- '+ '{:.2e} T\n'.format(B_field_fit_err)
        output_string += '-----------------\n'
        output_string += 'Gaussian FWHM = '+str(round(FWHM_G_eV_fit,2))+' +/- '+str(round(FWHM_eV_G_fit_err,2))+' eV\n'
        output_string += '-----------------\n'
        output_string += 'Line position \n= '+str(round(line_pos_Hz_fit,2))+' +/- '+str(round(line_pos_Hz_fit_err,2))+' Hz\n'
        output_string += '-----------------\n'
        output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
        output_string += '-----------------\n'
        output_string += 'Probability parameter \n= ' + "{:.2e}".format(prob_parameter_fit)\
        +' +/- ' + "{:.2e}".format(prob_parameter_fit_err)+'\n'
        output_string += '-----------------\n'
        output_string += 'Fit completed in '+str(round(elapsed,2))+'s'+'\n'
        dictionary_of_fit_results = {
        'output_string': output_string,
        'cov': cov,
        'bins_keV': bins_keV,
        'fit': fit,
        'bins_Hz': bins_Hz,
        'fit_Hz': fit_Hz,
        'FWHM_eV_fit': FWHM_eV_fit,
        'FWHM_eV_fit_err': FWHM_eV_fit_err,
        'line_pos_Hz_fit': line_pos_Hz_fit,
        'line_pos_Hz_fit_err': line_pos_Hz_fit_err,
        'B_field_fit': B_field_fit,
        'B_field_fit_err': B_field_fit_err,
        'prob_parameter_fit': prob_parameter_fit,
        'prob_parameter_fit_err': prob_parameter_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq
        }
        return dictionary_of_fit_results
