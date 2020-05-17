'''
Fits data to complex lineshape model.
Author: E. Machado, Y.-H. Sun, E. Novitski
Date: 4/8/20
'''

from __future__ import absolute_import

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import comb
from scipy import integrate , signal, interpolate
import os
import time
import sys
import json
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.misc import Constants

logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

# Natural constants
kr_line = Constants.kr_line() # 17.8260 keV
kr_line_width = Constants.kr_line_width() # 2.83 eV
e_charge = Constants.e_charge() # 1.60217662*10**(-19) Coulombs , charge of electron
m_e = Constants.m_e() # 9.10938356*10**(-31) Kilograms , mass of electron
mass_energy_electron = Constants.m_electron()/1000 # 510.9989461 keV

# A lorentzian function
def lorentzian(x_array,x0,FWHM):
    HWHM = FWHM/2.
    func = (1./np.pi)*HWHM/((x_array-x0)**2.+HWHM**2.)
    return func

# A gaussian function
def gaussian(x_array,A,sigma,mu):
    f = A*(1./(sigma*np.sqrt(2*np.pi)))*np.exp(-(((x_array-mu)/sigma)**2.)/2.)
    return f

# Converts a gaussian's FWHM to sigma
def gaussian_FWHM_to_sigma(FWHM):
    sigma = FWHM/(2*np.sqrt(2*np.log(2)))
    return sigma

# Converts a gaussian's sigma to FWHM
def gaussian_sigma_to_FWHM(sigma):
    FWHM = sigma*(2*np.sqrt(2*np.log(2)))
    return FWHM

#returns array with energy loss/ oscillator strength data
def read_oscillator_str_file(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    energyOsc = [[],[]] #2d array of energy losses, oscillator strengths

    for line in lines:
        if line != "" and line[0]!="#":
            raw_data = [float(i) for i in line.split("\t")]
            energyOsc[0].append(raw_data[0])
            energyOsc[1].append(raw_data[1])

    energyOsc = np.array(energyOsc)
    ### take data and sort by energy
    sorted_indices = energyOsc[0].argsort()
    energyOsc[0] = energyOsc[0][sorted_indices]
    energyOsc[1] = energyOsc[1][sorted_indices]
    return energyOsc

# A sub function for the scatter function. Found in
# "Energy loss of 18 keV electrons in gaseous T and quench condensed D films"
# by V.N. Aseev et al. 2000
def aseev_func_tail(energy_loss_array, gas_type):
    if gas_type=="H2":
        A2, omeg2, eps2 = 0.195, 14.13, 10.60
    elif gas_type=="Kr":
        A2, omeg2, eps2 = 0.4019, 22.31, 16.725
    return A2*omeg2**2./(omeg2**2.+4*(energy_loss_array-eps2)**2.)

#convert oscillator strength into energy loss spectrum
def get_eloss_spec(e_loss, oscillator_strength, kr_line): #energies in eV
    kinetic_en = kr_line * 1000
    e_rydberg = 13.605693009 #rydberg energy (eV)
    a0 = 5.291772e-11 #bohr radius
    return np.where(e_loss>0 , 4.*np.pi*a0**2 * e_rydberg / (kinetic_en * e_loss) * oscillator_strength * np.log(4. * kinetic_en * e_loss / (e_rydberg**3.) ), 0)

# Takes only the nonzero bins of a histogram
def get_only_nonzero_bins(bins,hist):
    nonzero_idx = np.where(hist!=0)
    hist_nonzero = hist[nonzero_idx]
    hist_err = np.sqrt(hist_nonzero)
    bins_nonzero = bins[nonzero_idx]
    return bins_nonzero , hist_nonzero , hist_err

# Flips an array left-to-right. Useful for converting between energy and frequency
def flip_array(array):
    flipped = np.fliplr([array]).flatten()
    return flipped

# Given energy in keV and the self.B_field of the trap, returns frequency in Hz
def energy_to_frequency(energy_vec, B_field):
    freq_vec = (e_charge*B_field/((2.*np.pi*m_e)*(1+energy_vec/mass_energy_electron)))
    return freq_vec

# Given frequency in Hz and the self.B_field of the trap, returns energy in keV
def frequency_to_energy(freq_vec,B_field):
    energy_vec = (e_charge*B_field/((2.*np.pi*m_e*freq_vec))-1)*mass_energy_electron
    return energy_vec

# Converts an energy to frequency using a guess for magnetic field. Can handle errors too
def energy_guess_to_frequency(energy_guess,energy_guess_err,B_field_guess):
    frequency = energy_to_frequency(energy_guess,B_field_guess)
    const = e_charge*B_field_guess/(2.*np.pi*m_e)
    frequency_err = const/(1+energy_guess/mass_energy_electron)**2*energy_guess_err/mass_energy_electron
    return frequency , frequency_err

# Given a frequency and error, converts those to B field values assuming the line is the 17.8 keV line
def central_frequency_to_B_field(central_freq,central_freq_err):
    const = (2.*np.pi*m_e)*(1+kr_line/mass_energy_electron)/e_charge
    B_field = const*central_freq
    B_field_err = const*central_freq_err
    return B_field , B_field_err

# given a FWHM for the lorentian component and the FWHM for the gaussian component,
# this function estimates the FWHM of the resulting voigt distribution
def FWHM_voigt(FWHM_L,FWHM_G):
    FWHM_V = 0.5346*FWHM_L+np.sqrt(0.2166*FWHM_L**2+FWHM_G**2)
    return FWHM_V

# Returns the name of the current directory
def get_current_dir():
    current_path = os.path.abspath(os.getcwd())
    stripped_path = [s.strip('\n') for s in current_path]
    stripped_path = stripped_path[0].split('/')
    current_dir = stripped_path[len(stripped_path)-1]
    return current_dir

class ShakeSpectrumClass():
    # The shakeup/shakeoff spectrum for the 17.8 keV line of Kr83m based on Hamish Vadantha paper
    # Overleaf link for the paper https://www.overleaf.com/project/5d56e015162d244b1c283c57
    #######################################################################################
    # Read parameters for shake up shake off spectrum from an excel spread sheet.
    # The parameters are from Hamish and Vedantha shake spectrum paper table IV

    def __init__(self, shake_spectrum_parameters_json_path, input_std_eV_array):

        with open(
        shake_spectrum_parameters_json_path, 'r'
        ) as fp:
            shake_spectrum_parameters = json.load(fp)
        self.A_Intensity = shake_spectrum_parameters['A_Intensity']
        self.B_Binding = shake_spectrum_parameters['B_Binding']
        self.Gamma_Width = shake_spectrum_parameters['Gamma_Width']
        self.E_b_Scale = shake_spectrum_parameters['E_b_Scale']
        self.Ecore = shake_spectrum_parameters['Ecore']
        self.epsilon_shake_spectrum = shake_spectrum_parameters['epsilon_shake_spectrum']
        self.input_std_eV_array = input_std_eV_array

    # nprime in Eq. (6)
    def nprime(self, E_b, W):
        return np.sqrt(E_b/abs(W))

    # Eq.(9) in Hamish Vedantha shake spectrum paper
    def C1s(self, E_b):
        return 98.2/E_b

    # Eq. (6)
    def P_1s_nprime(self, E_b, W):
        n_prime = self.nprime(E_b, W)
        P_1s_nprime = 1
        P_1s_nprime *= self.C1s(E_b)
        P_1s_nprime *= (1-np.exp(-2*np.pi*n_prime))**-1
        P_1s_nprime *= n_prime**8
        P_1s_nprime *= (n_prime**2+1)**-4
        P_1s_nprime *= np.exp(-4*n_prime*np.arctan(1/n_prime))
        return P_1s_nprime

    # Eq. (5) in Hamish Vedantha shake spectrum paper
    # shake up spectrum for the ith state
    def I(self, i, E):
        numerator = self.A_Intensity[i]*self.Gamma_Width[i]
        denominator = 2*np.pi*(self.Gamma_Width[i]**2/4 + (self.Ecore - self.B_Binding[i] - E)**2)
        return numerator/denominator


    # Eq. (11) in Hamish Vedantha shake spectrum paper
    # shake off spectrum for the ith state
    def spectrum1(self, i, E):
        if self.E_b_Scale[i] == 0:
            spectrum = self.I(i, E)
        else:
            factor = np.arctan(2*(self.Ecore - E - self.B_Binding[i]+self.epsilon_shake_spectrum)/self.Gamma_Width[i])/(np.pi)+0.5
            spectrum = self.A_Intensity[i]*factor*self.P_1s_nprime(self.E_b_Scale[i], self.Ecore-E-self.B_Binding[i]+self.epsilon_shake_spectrum)
        return spectrum

    # adding up shake spectrum for all the states
    def full_shake_spectrum(self, E, start_number_of_i, end_number_of_i):
        full_spectrum = 0
        for i in range(start_number_of_i, end_number_of_i, 1):
            full_spectrum += self.spectrum1(i, E)
        return full_spectrum

    # shake spectrum by adding up shake spectrum for all the states up to i=24
    def shake_spectrum(self):
        x_array = self.input_std_eV_array
        x_array = flip_array(x_array)
        shake_spectrum = self.full_shake_spectrum(x_array, 0, 24)
        return shake_spectrum
    ###############################################################################

class ComplexLineShape(BaseProcessor):

    def InternalConfigure(self, params):
        '''
        Configure
        '''
        # Read other parameters
        self.bins_choice = reader.read_param(params, 'bins_choice', [])
        self.gases = reader.read_param(params, 'gases', ["H2","Kr"])
        self.max_scatters = reader.read_param(params, 'max_scatters', 20)
        self.max_comprehensive_scatters = reader.read_param(params, 'max_comprehensive_scatters', 20)
        # This is an important parameter which determines how finely resolved
        # the scatter calculations are. 10000 seems to produce a stable fit, with minimal slowdown
        self.num_points_in_std_array = reader.read_param(params, 'num_points_in_std_array', 10000)
        self.RF_ROI_MIN = reader.read_param(params, 'RF_ROI_MIN', 25850000000.0)
        self.B_field = reader.read_param(params, 'B_field', 0.957810722501)
        self.shake_spectrum_parameters_json_path = reader.read_param(params, 'shake_spectrum_parameters_json_path', 'shake_spectrum_parameters.json')
        self.path_to_osc_strengths_files = reader.read_param(params, 'path_to_osc_strengths_files', '/host/')

    def InternalRun(self):

        # Read shake parameters from JSON file
        self.shakeSpectrumClassInstance = ShakeSpectrumClass(self.shake_spectrum_parameters_json_path, self.std_eV_array())

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

    # A gaussian centered at 0 eV with variable width, on the SELA
    def std_gaussian(self, sigma):
        x_array = self.std_eV_array()
        ans = gaussian(x_array,1,sigma,0)
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
        energy_fOsc = read_oscillator_str_file(input_filename)
        fData = interpolate.interp1d(energy_fOsc[0], energy_fOsc[1], kind='linear')
        for i in range(len(energy_loss_array)):
            if energy_loss_array[i] < energy_fOsc[0][0]:
                f[i] = 0
            elif energy_loss_array[i] <= energy_fOsc[0][-1]:
                f[i] = fData(energy_loss_array[i])
            else:
                f[i] = aseev_func_tail(energy_loss_array[i], gas_type)

        f_e_loss = get_eloss_spec(energy_loss_array, f, kr_line)
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
        self.path_to_osc_strengths_files+'scatter_spectra_file/scatter_spectra.npy', 
        scatter_spectra
        )
        elapsed = time.time() - t
        print('File generated in '+str(elapsed)+'s')
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
            print('generate fresh scatter file')
            self.generate_scatter_convolution_file()
        else: 
            stuff_in_dir = os.listdir(self.path_to_osc_strengths_files)
            if 'scatter_spectra_file' not in stuff_in_dir:
                print('Scatter spectra folder not found, generating')
                os.mkdir(self.path_to_osc_strengths_files+'scatter_spectra_file')
                time.sleep(2)
                self.generate_scatter_convolution_file()
            else:
                directory = os.listdir(self.path_to_osc_strengths_files + "/scatter_spectra_file")
                strippeddirs = [s.strip('\n') for s in directory]
                if 'scatter_spectra.npy' not in strippeddirs:
                    self.generate_scatter_convolution_file()
                test_file = self.path_to_osc_strengths_files+'scatter_spectra_file/scatter_spectra.npy' 
                test_dict = np.load(test_file, allow_pickle = True)
                if list(test_dict.item().keys())[0] != '{}_{}'.format(gases[0], gases[1]):
                    print('first entry not matching, generating fresh files')
                    self.generate_scatter_convolution_file()
                elif len(test_dict.item()['{}_{}'.format(gases[0], gases[1])]) \
                            != (1 + self.max_scatters+1)*(self.max_scatters+1)/2 - 1:
                    print('Number of scatters not matching, generating fresh files')
                elif len(test_dict.item()['{}_{}'.format(gases[0], gases[1])]['01_00'])\
                != self.num_points_in_std_array:
                    print('Binning do not match standard array, generating fresh files')
                    self.generate_scatter_convolution_file()
        return

    # Given a function evaluated on the SELA, convolves it with a gaussian
    def convolve_gaussian(self, func_to_convolve,gauss_FWHM_eV):
        sigma = gaussian_FWHM_to_sigma(gauss_FWHM_eV)
        resolution_f = self.std_gaussian(sigma)
        ans = signal.convolve(resolution_f,func_to_convolve,mode='same')
        ans_normed = self.normalize(ans)
        return ans_normed

    def make_spectrum(self, gauss_FWHM_eV, scale_of_first_order_peak, share_of_gas1_in_first_order_peak, amplitude_decreasing_rate_for_higher_order_peaks, emitted_peak='shake'):
        gases = self.gases
        max_scatters = self.max_scatters
        max_comprehensive_scatters = self.max_comprehensive_scatters
        current_path = self.path_to_osc_strengths_files
        #filenames = list_files('scatter_spectra_files')
        a = share_of_gas1_in_first_order_peak*scale_of_first_order_peak
        b = (1 - share_of_gas1_in_first_order_peak)*scale_of_first_order_peak
        p = amplitude_decreasing_rate_for_higher_order_peaks
        scatter_spectra = np.load(current_path + 'scatter_spectra_file/scatter_spectra.npy', allow_pickle = True)
        en_array = self.std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = self.std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = self.shakeSpectrumClassInstance.shake_spectrum()
        current_working_spectrum = self.convolve_gaussian(current_working_spectrum,gauss_FWHM_eV)
        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += zeroth_order_peak

        current_working_spectrum = \
        scatter_spectra.item()['{}_{}'.format(gases[0], gases[1])]\
        ['{}_{}'.format(str(1).zfill(2), str(0).zfill(2))]
        current_working_spectrum = \
        self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
        first_order_peak1 = current_working_spectrum
        current_full_spectrum += a*first_order_peak1

        current_working_spectrum = \
        scatter_spectra.item()['{}_{}'.format(gases[0], gases[1])]\
        ['{}_{}'.format(str(0).zfill(2), str(1).zfill(2))]
        current_working_spectrum = \
        self.normalize(signal.convolve(zeroth_order_peak, current_working_spectrum, mode='same'))
        first_order_peak2 = current_working_spectrum
        current_full_spectrum += b*first_order_peak2

        for n in range(2, max_comprehensive_scatters + 1):
            for r in range(0, n):
                current_working_spectrum = \
                scatter_spectra.item()['{}_{}'.format(gases[0], gases[1])]\
                ['{}_{}'.format(str(r).zfill(2), str(n-r-1).zfill(2))]

                current_working_spectrum = \
                self.normalize(signal.convolve(first_order_peak1, current_working_spectrum, mode='same'))
                current_full_spectrum += a*current_working_spectrum*comb(n-1, r)\
                *(p)**(n-1)

                current_working_spectrum = \
                self.normalize(signal.convolve(first_order_peak2, current_working_spectrum, mode='same'))
                current_full_spectrum += b*current_working_spectrum*comb(n-1, r)\
                *(p)**(n-1)

        for n in range(max_comprehensive_scatters + 1, max_scatters + 1):
            current_working_spectrum = \
            scatter_spectra.item()['{}_{}'.format(gases[0], gases[1])]\
            ['{}_{}'.format(str(n-1).zfill(2), str(0).zfill(2))]
            current_working_spectrum = \
            self.normalize(signal.convolve(first_order_peak1, current_working_spectrum, mode='same'))
            current_full_spectrum += current_working_spectrum*(p)**(n-1)

            current_working_spectrum = \
            scatter_spectra.item()['{}_{}'.format(gases[0], gases[1])]\
            ['{}_{}'.format(str(0).zfill(2), str(n-1).zfill(2))]
            current_working_spectrum = \
            self.normalize(signal.convolve(first_order_peak2, current_working_spectrum, mode='same'))
            current_full_spectrum += current_working_spectrum*(p)**(n-1)        
        return current_full_spectrum


    # Produces a spectrum in real energy that can now be evaluated off of the SELA.
    #def spectrum_func(x_keV,FWHM_G_eV,line_pos_keV,scatter_prob,amplitude):
    def spectrum_func(self, x_keV, *p0):
        x_eV = x_keV*1000.
        en_loss_array = self.std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        en_array_rev = flip_array(-1*en_loss_array)
        f = np.zeros(len(x_keV))
        f_intermediate = np.zeros(len(x_keV))

        FWHM_G_eV = p0[0]
        line_pos_keV = p0[1]
        amplitude = p0[2]
        scale_of_first_order_peak = p0[3]
        share_of_gas1_in_first_order_peak = p0[4]
        amplitude_decreasing_rate_for_higher_order_peaks = p0[5]
    
        line_pos_eV = line_pos_keV*1000.
        x_eV_minus_line = x_eV - line_pos_eV
        zero_idx = np.r_[np.where(x_eV_minus_line< en_loss_array_min)[0],np.where(x_eV_minus_line>en_loss_array_max)[0]]
        nonzero_idx = [i for i in range(len(x_keV)) if i not in zero_idx]
    
        full_spectrum = self.make_spectrum(FWHM_G_eV, scale_of_first_order_peak, share_of_gas1_in_first_order_peak, amplitude_decreasing_rate_for_higher_order_peaks)
        full_spectrum_rev = flip_array(full_spectrum)
        f_intermediate[nonzero_idx] = np.interp(x_eV_minus_line[nonzero_idx],en_array_rev,full_spectrum_rev)
        f[nonzero_idx] += amplitude*f_intermediate[nonzero_idx]/np.sum(f_intermediate[nonzero_idx])
        return f

    # Call this function to fit a histogram of start frequencies with the model.
    # Note that the data_hist_freq should be the StartFrequencies as given by katydid,
    # which will be from ~0 MHZ to ~90 MHz. You must also pass this function the
    # self.RF_ROI_MIN value from the metadata file of your data.
    # You must also supply a guess for the self.B_field present for the run;
    # 0.959 T is usually sufficient.

    def fit_data(self, freq_bins, data_hist_freq):
        t = time.time()
        self.check_existence_of_scatter_file()
        bins_Hz = freq_bins + self.RF_ROI_MIN
        bins_keV = frequency_to_energy(bins_Hz,self.B_field)
        bins_keV = flip_array(bins_keV)
        data_hist = flip_array(data_hist_freq)
        bins_keV_nonzero , data_hist_nonzero , data_hist_err = get_only_nonzero_bins(bins_keV, data_hist)
        # Bounds for curve_fit
        FWHM_eV_min = 1e-5
        FWHM_eV_max = (bins_keV[len(bins_keV)-1] - bins_keV[0])*1000
        line_pos_keV_min = bins_keV[0]
        line_pos_keV_max = bins_keV[len(bins_keV)-1]
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist)*3
        scale_of_first_order_peak_min = 1e-5
        scale_of_first_order_peak_max = 1
        share_of_gas1_in_first_order_peak_min = 1e-5
        share_of_gas1_in_first_order_peak_max = 1
        amplitude_decreasing_rate_for_higher_order_peaks_min = 1e-5
        amplitude_decreasing_rate_for_higher_order_peaks_max = 1
        # Initial guesses for curve_fit
        FWHM_guess = 5
        line_pos_guess = bins_keV[np.argmax(data_hist)]
        amplitude_guess = np.sum(data_hist)/2
        scale_of_first_order_peak_guess = 0.5
        share_of_gas1_in_first_order_peak_guess = 0.5
        amplitude_decreasing_rate_for_higher_order_peaks_guess = 0.5
        p0_guess = [FWHM_guess, line_pos_guess, amplitude_guess, scale_of_first_order_peak_guess, share_of_gas1_in_first_order_peak_guess, amplitude_decreasing_rate_for_higher_order_peaks_guess] 
        p0_bounds = ([FWHM_eV_min, line_pos_keV_min, amplitude_min, scale_of_first_order_peak_min, share_of_gas1_in_first_order_peak_min, amplitude_decreasing_rate_for_higher_order_peaks_min],  
                    [FWHM_eV_max, line_pos_keV_max, amplitude_max, scale_of_first_order_peak_max, share_of_gas1_in_first_order_peak_max, amplitude_decreasing_rate_for_higher_order_peaks_max])
        # Actually do the fitting
        params , cov = curve_fit(self.spectrum_func,bins_keV_nonzero,data_hist_nonzero,sigma=data_hist_err,p0=p0_guess,bounds=p0_bounds)
        # Name each of the resulting parameters and errors
        FWHM_G_eV_fit = params[0]
        line_pos_keV_fit = params[1]
        amplitude_fit = params[2]
        scale_of_first_order_peak_fit = params[3]
        share_of_gas1_in_first_order_peak_fit = params[4]
        amplitude_decreasing_rate_for_higher_order_peaks_fit = params[5]

        perr = np.sqrt(np.diag(cov))
        FWHM_eV_G_fit_err = perr[0]
        line_pos_keV_fit_err = perr[1]
        amplitude_fit_err = perr[2]
        scale_of_first_order_peak_fit_err = perr[3]
        share_of_gas1_in_first_order_peak_fit_err = perr[4]
        amplitude_decreasing_rate_for_higher_order_peaks_fit_err = perr[5]
    
        fit = self.spectrum_func(bins_keV[0:-1],*params)

        line_pos_Hz_fit , line_pos_Hz_fit_err = energy_guess_to_frequency(line_pos_keV_fit, line_pos_keV_fit_err, self.B_field)
        B_field_fit , B_field_fit_err = central_frequency_to_B_field(line_pos_Hz_fit, line_pos_Hz_fit_err)
        fit_Hz = flip_array(fit)
        bins_keV = bins_keV - line_pos_keV_fit + kr_line
        FWHM_eV_fit = FWHM_G_eV_fit
        FWHM_eV_fit_err = FWHM_eV_G_fit_err
        elapsed = time.time() - t
        output_string = 'Gaussian FWHM = '+str(round(FWHM_G_eV_fit,2))+' +/- '+str(round(FWHM_eV_G_fit_err,2))+' eV\n'
        output_string += '-----------------\n'
        output_string += 'Line position \n= '+str(round(line_pos_Hz_fit,2))+' +/- '+str(round(line_pos_Hz_fit_err,2))+' Hz\n'
        output_string += '-----------------\n'
        output_string += 'Amplitude = {}'.format(round(amplitude_fit,2))+' +/- {}'.format(round(amplitude_fit_err,2)) + '\n'
        output_string += '-----------------\n'
        output_string += 'Scale of first order peak\n= ' + "{:.2e}".format(scale_of_first_order_peak_fit)\
        +' +/- ' + "{:.2e}".format(scale_of_first_order_peak_fit_err)+'\n'
        output_string += '-----------------\n'
        output_string += 'Share of {} in first order peak\n= {:.2e}'.format(self.gases[0], share_of_gas1_in_first_order_peak_fit)\
        + ' +/- ' + '{:.2e}\n'.format(share_of_gas1_in_first_order_peak_fit_err)
        output_string += '-----------------\n'
        output_string += 'Amplitude decreasing rate\n for higher order peaks\n= {:.2e}'.format(amplitude_decreasing_rate_for_higher_order_peaks_fit)\
        + ' +/- ' + "{:.2e}\n".format(amplitude_decreasing_rate_for_higher_order_peaks_fit_err)
        output_string += '-----------------\n'
        elapsed = time.time() - t
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
        'scale_of_first_order_peak_fit': scale_of_first_order_peak_fit,
        'scale_of_first_order_peak_fit_err': scale_of_first_order_peak_fit_err,
        'share_of_gas1_in_first_order_peak_fit': share_of_gas1_in_first_order_peak_fit,
        'share_of_gas1_in_first_order_peak_fit_err': share_of_gas1_in_first_order_peak_fit_err,
        'amplitude_decreasing_rate_for_higher_order_peaks_fit': amplitude_decreasing_rate_for_higher_order_peaks_fit,
        'amplitude_decreasing_rate_for_higher_order_peaks_fit_err': amplitude_decreasing_rate_for_higher_order_peaks_fit_err,
        'amplitude_fit': amplitude_fit,
        'amplitude_fit_err': amplitude_fit_err,
        'data_hist_freq': data_hist_freq
        }
        return dictionary_of_fit_results
