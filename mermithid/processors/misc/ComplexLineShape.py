'''
Bin tritium start frequencies and calculate efficiency for each bin
function.
Author: A. Ziegler, E. Novitski, C. Claessens
Date:3/4/2020

This takes efficiency informations and interpolates between frequency points.
Then, you dump in tritium data.
It assigns an efficiency and efficiency uncertainty upper and lower bounds to each event.
It also bins events and defines an efficiency and efficiency uncertainty upper and
lower bound by integrating the interpolated efficiency over each bin.
'''

from __future__ import absolute_import

import numpy as np
import scipy
from scipy import integrate , signal, interpolate
import os
import time
import pandas as pd
import sys
import json
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.misc import Constants

#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#import scipy as sp
#from scipy import integrate , signal, interpolate
#import os
#import time
#import pandas as pd

logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

# Natural constants
kr_line = 17.8260 # keV
kr_line_width = 2.83 # eV
e_charge = 1.60217662*10**(-19) # Coulombs , charge of electron
m_e = 9.10938356*10**(-31) # Kilograms , mass of electron
mass_energy_electron = 510.9989461 # keV

# Establishes a standard energy loss array (SELA) from -1000 eV to 1000 eV
# with number of points equal to self.num_points_in_std_array. All convolutions
# will be carried out on this particular discretization
def std_eV_array():
    emin = -1000
    emax = 1000
    array = np.linspace(emin,emax,self.num_points_in_std_array)
    return array

# A lorentzian function
def lorentzian(x_array,x0,FWHM):
    HWHM = FWHM/2.
    func = (1./np.pi)*HWHM/((x_array-x0)**2.+HWHM**2.)
    return func

# A lorentzian line centered at 0 eV, with 2.83 eV width on the SELA
def std_lorenztian_17keV():
    x_array = std_eV_array()
    ans = lorentzian(x_array,0,kr_line_width)
    return ans


# A gaussian function
def gaussian(x_array,A,sigma,mu):
    f = A*(1./(sigma*np.sqrt(2*np.pi)))*np.exp(-(((x_array-mu)/sigma)**2.)/2.)
    return f

# A gaussian centered at 0 eV with variable width, on the SELA
def std_gaussian(sigma):
    x_array = std_eV_array()
    ans = gaussian(x_array,1,sigma,0)
    return ans

# Converts a gaussian's FWHM to sigma
def gaussian_FWHM_to_sigma(FWHM):
    sigma = FWHM/(2*np.sqrt(2*np.log(2)))
    return sigma

# Converts a gaussian's sigma to FWHM
def gaussian_sigma_to_FWHM(sigma):
    FWHM = sigma*(2*np.sqrt(2*np.log(2)))
    return FWHM

# normalizes a function, but depends on binning.
# Only to be used for functions evaluated on the SELA
def normalize(f):
    x_arr = std_eV_array()
    f_norm = sp.integrate.simps(f,x=x_arr)
    # if  f_norm < 0.99 or f_norm > 1.01:
    #     print(f_norm)
    f_normed = f/f_norm
    return f_normed

#returns array with energy loss/ oscillator strength data
def read_oscillator_str_file(filename):
    f = open(filename, "r")
    lines = f.readlines()
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

# Function for energy loss from a single scatter of electrons by
# V.N. Aseev et al. 2000
# This function does the work of combining fit_func1 and fit_func2 by
# finding the point where they intersect.
# Evaluated on the SELA
def single_scatter_f(gas_type):
    energy_loss_array = std_eV_array()
    f = 0 * energy_loss_array

    input_filename = "data/" + gas_type + "OscillatorStrength.txt"
    energy_fOsc = read_oscillator_str_file(input_filename)
    fData = interpolate.interp1d(energy_fOsc[0], energy_fOsc[1], kind='linear')
    for i in range(len(energy_loss_array)):
        if energy_loss_array[i] < energy_fOsc[0][0]:
            f[i] = 0
        elif energy_loss_array[i] <= energy_fOsc[0][-1]:
            f[i] = fData(energy_loss_array[i])
        else:
            f[i] = aseev_func_tail(energy_loss_array[i], gas_type)

    f_e_loss = get_eloss_spec(energy_loss_array, f)
    f_normed = normalize(f_e_loss)
    #plt.plot(energy_loss_array, f_e_loss)
    #plt.show()
    return f_normed

# Convolves a function with the single scatter function, on the SELA
def another_scatter(input_spectrum, gas_type):
    single = single_scatter_f(gas_type)
    f = sp.signal.convolve(single,input_spectrum,mode='same')
    f_normed = normalize(f)
    return f_normed

# Convolves the scatter function with itself over and over again and saves
# the results to .npy files.
def generate_scatter_convolution_files(gas_type):
    t = time.time()
    first_scatter = single_scatter_f(gas_type)
    scatter_num_array = range(2,max_scatters+1)
    current_scatter = first_scatter
    np.save('scatter_spectra_files/scatter'+gas_type+"_"+str(1).zfill(2),current_scatter)
    # x = std_eV_array() # diagnostic
    for i in scatter_num_array:
        current_scatter = another_scatter(current_scatter, gas_type)
        np.save('scatter_spectra_files/scatter'+gas_type+"_"+str(i).zfill(2),current_scatter)
        # plt.plot(x,current_scatter) # diagnostic
    # plt.show() # diagnostic
    elapsed = time.time() - t
    print('Files generated in '+str(elapsed)+'s')
    return

# Returns the name of the current path
# def get_current_path():
#     current_path = os.popen("pwd").readlines()
#     stripped_path = [s.strip('\n') for s in current_path]
#     return stripped_path[0]

# Prints a list of the contents of a directory
def list_files(path):
    directory = os.popen("ls "+path).readlines()
    strippeddirs = [s.strip('\n') for s in directory]
    return strippeddirs

# Returns the name of the current directory
# def get_current_dir():
#     current_path = os.popen("pwd").readlines()
#     stripped_path = [s.strip('\n') for s in current_path]
#     stripped_path = stripped_path[0].split('/')
#     current_dir = stripped_path[len(stripped_path)-1]
#     return current_dir

# The shakeup/shakeoff spectrum for the 17.8 keV line of Kr83m based on Hamish Vadantha paper
# Overleaf link for the paper https://www.overleaf.com/project/5d56e015162d244b1c283c57
#######################################################################################
# Read parameters for shake up shake off spectrum from an excel spread sheet.
# The parameters are from Hamish and Vedantha shake spectrum paper table IV
def read_shake_parameters_from_excel_file(path_to_shake_parameters_excel_file):
    Ecore = 0 # set as 17823 to reproduce Fig. 4 in Hamish Vedantha shake spectrum paper
    epsilon = 1e-4 # a small quantity for preventing zero denominator
    df = pd.read_excel(path_to_shake_parameters_excel_file, index_col=0)
    A = []
    B = []
    Gamma = []
    E_b = []
    for i in range(27):
        A.append(df['Unnamed: 1'][i]*100)
        B.append(df['Unnamed: 2'][i])
        Gamma.append(df['Unnamed: 4'][i])
        E_b.append(df['Unnamed: 3'][i])
    return A, B, Gamma, E_b, Ecore, epsilon

# nprime in Eq. (6)
def nprime(E_b, W):
    return np.sqrt(E_b/abs(W))

# Eq.(9) in Hamish Vedantha shake spectrum paper
def C1s(E_b):
    return 98.2/E_b

# Eq. (6)
def P_1s_nprime(E_b, W):
    n_prime = nprime(E_b, W)
    P_1s_nprime = 1
    P_1s_nprime *= C1s(E_b)
    P_1s_nprime *= (1-np.exp(-2*np.pi*n_prime))**-1
    P_1s_nprime *= n_prime**8
    P_1s_nprime *= (n_prime**2+1)**-4
    P_1s_nprime *= np.exp(-4*n_prime*np.arctan(1/n_prime))
    return P_1s_nprime

# Eq. (5) in Hamish Vedantha shake spectrum paper
# shake up spectrum for the ith state
def I(i, E):
    numerator = A_Intensity[i]*Gamma_Width[i]
    denominator = 2*np.pi*(Gamma_Width[i]**2/4 + (Ecore - B_Binding[i] - E)**2)
    return numerator/denominator


# Eq. (11) in Hamish Vedantha shake spectrum paper
# shake off spectrum for the ith state
def spectrum1(i, E):
    if E_b_Scale[i] == 0:
        spectrum = I(i, E)
    else:
        factor = np.arctan(2*(Ecore - E - B_Binding[i]+epsilon_shake_spectrum)/Gamma_Width[i])/(np.pi)+0.5
        spectrum = A_Intensity[i]*factor*P_1s_nprime(E_b_Scale[i], Ecore-E-B_Binding[i]+epsilon_shake_spectrum)
    return spectrum

# adding up shake spectrum for all the states
def full_shake_spectrum(E, start_number_of_i, end_number_of_i):
    full_spectrum = 0
    for i in range(start_number_of_i, end_number_of_i, 1):
        full_spectrum += spectrum1(i, E)
    return full_spectrum

# shake spectrum by adding up shake spectrum for all the states up to i=24
def shake_spectrum():
    x_array = std_eV_array()
    x_array = flip_array(x_array)
    shake_spectrum = full_shake_spectrum(x_array, 0, 24)
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
        # This is an important parameter which determines how finely resolved
        # the scatter calculations are. 10000 seems to produce a stable fit, with minimal slowdown
        self.num_points_in_std_array = reader.read_param(params, 'num_points_in_std_array', 10000)
        self.RF_ROI_MIN = reader.read_param(params, 'RF_ROI_MIN', 25850000000.0)
        self.B_field = reader.read_param(params, 'B_field', 0.957810722501)
        self.path_to_shake_parameters_excel_file = reader.read_param(params, 'path_to_shake_parameters_excel_file', '/host/KrShakeParameters214.xlsx')

    def InternalRun(self):

        # Read shake parameters from Excel file
        A_Intensity, B_Binding, Gamma_Width, E_b_Scale, Ecore, epsilon_shake_spectrum = read_shake_parameters_from_excel_file(self.path_to_shake_parameters_excel_file)

        number_of_events = len(self.data['StartFrequency'])
        self.results = number_of_events

        a = self.data['StartFrequency']

        # fit with shake spectrum
        data_hist_freq, freq_bins= np.histogram(a,bins=self.bins_choice)
        histogram = data_hist_freq
        bins = freq_bins
        guess = np.where(np.array(histogram) == np.max(histogram))[0][0]
        #print(guess)
        kr17kev_in_hz = guess*(bins[1]-bins[0])+bins[0]
        #B_field = B(17.8, kr17kev_in_hz + 0)
        cov, \
        bins_keV , fit , bins_Hz, fit_Hz , \
        FWHM_eV_fit, FWHM_eV_fit_err, \
        line_pos_Hz_fit, line_pos_Hz_fit_err, \
        B_field_fit, B_field_fit_err, \
        scatter_prob_fit, scatter_prob_fit_err, \
        amplitude_fit, amplitude_fit_err = self.fit_data(
            self.RF_ROI_MIN, self.B_field, freq_bins, data_hist_freq
        )

        # fit with shake spectrum
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(15,9))
        plt.step(bins_Hz[0:-1]/1e9,data_hist_freq)
        plt.plot(bins_Hz[0:-1]/1e9,fit_Hz)
        plt.xlabel('frequency GHz')
        plt.title('fit with shake spectrum 2 gas scattering')
        plt.savefig('/host/plots/fit_shake_2_gas_0.png')

        return True



    # Checks for the existence of a directory called 'scatter_spectra_files'
    # and checks that this directory contains the scatter spectra files.
    # If not, this function calls generate_scatter_convolution_files.
    # This function also checks to make sure that the scatter files have the correct
    # number of points in the SELA, and if not, it generates fresh files
    def check_existence_of_scatter_files(self, gas_type):
        current_path = '/host/scatter_spectra_files'
        current_dir = '/scatter_spectra_files'
        stuff_in_dir = list_files(current_path)
        if 'scatter_spectra_files' not in stuff_in_dir and current_dir != 'scatter_spectra_files':
            print('Scatter files not found, generating')
            os.popen("mkdir scatter_spectra_files")
            time.sleep(2)
            generate_scatter_convolution_files(gas_type)
        else:
            directory = os.popen("ls scatter_spectra_files").readlines()
            strippeddirs = [s.strip('\n') for s in directory]
            if len(directory) != len(self.gases) * max_scatters:
                generate_scatter_convolution_files(gas_type)
            test_file = 'scatter_spectra_files/scatter'+gas_type+'_01.npy'
            test_arr = np.load(test_file)
            if len(test_arr) != self.num_points_in_std_array:
                print('Scatter files do not match standard array binning, generating fresh files')
                generate_scatter_convolution_files(gas_type)
        return

    # Given a function evaluated on the SELA, convolves it with a gaussian
    def convolve_gaussian(func_to_convolve,gauss_FWHM_eV):
        sigma = gaussian_FWHM_to_sigma(gauss_FWHM_eV)
        resolution_f = std_gaussian(sigma)
        ans = sp.signal.convolve(resolution_f,func_to_convolve,mode='same')
        ans_normed = normalize(ans)
        return ans_normed

    # Produces a full spectral shape on the SELA, given a gaussian resolution
    # and a scatter probability
    def make_spectrum(gauss_FWHM_eV,scatter_prob,gas_type,emitted_peak='shake'):
        current_path = '/host'
        # check_existence_of_scatter_files()
        #filenames = list_files('scatter_spectra_files')
        scatter_num_array = range(1,max_scatters+1)
        en_array = std_eV_array()
        current_full_spectrum = np.zeros(len(en_array))
        if emitted_peak == 'lorentzian':
            current_working_spectrum = std_lorenztian_17keV()
        elif emitted_peak == 'shake':
            current_working_spectrum = shake_spectrum()
        current_working_spectrum = convolve_gaussian(current_working_spectrum,gauss_FWHM_eV)
        zeroth_order_peak = current_working_spectrum
        current_full_spectrum += current_working_spectrum
        for i in scatter_num_array:
            current_working_spectrum = np.load(current_path+'/scatter_spectra_files/scatter'+gas_type+'_'+str(i).zfill(2)+'.npy')
            current_working_spectrum = normalize(sp.signal.convolve(zeroth_order_peak,current_working_spectrum,mode='same'))
            current_full_spectrum += current_working_spectrum*scatter_prob**scatter_num_array[i-1]
        # plt.plot(en_array,current_full_spectrum) # diagnostic
        # plt.show() # diagnostic
        return current_full_spectrum

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

    # Produces a spectrum in real energy that can now be evaluated off of the SELA.
    #def spectrum_func(x_keV,FWHM_G_eV,line_pos_keV,scatter_prob,amplitude):
    def spectrum_func(x_keV, *p0):
        x_eV = x_keV*1000.
        en_loss_array = std_eV_array()
        en_loss_array_min = en_loss_array[0]
        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
        en_array_rev = flip_array(-1*en_loss_array)
        f = np.zeros(len(x_keV))

        FWHM_G_eV = p0[0]
        line_pos_keV = p0[1]

        line_pos_eV = line_pos_keV*1000.
        x_eV_minus_line = x_eV - line_pos_eV
        zero_idx = np.r_[np.where(x_eV_minus_line<-1*en_loss_array_max)[0],np.where(x_eV_minus_line>-1*en_loss_array_min)[0]]
        nonzero_idx = [i for i in range(len(x_keV)) if i not in zero_idx]

        for gas_index in range(len(self.gases)):
            gas_type = self.gases[gas_index]
            scatter_prob = p0[2*gas_index+2]
            amplitude    = p0[2*gas_index+3]

            full_spectrum = make_spectrum(FWHM_G_eV,scatter_prob, gas_type)
            full_spectrum_rev = flip_array(full_spectrum)
            f[nonzero_idx] += amplitude*np.interp(x_eV_minus_line[nonzero_idx],en_array_rev,full_spectrum_rev)
        return f

    # Given energy in keV and the B_field of the trap, returns frequency in Hz
    def energy_to_frequency(energy_vec, B_field):
        freq_vec = (e_charge*B_field/((2.*np.pi*m_e)*(1+energy_vec/mass_energy_electron)))
        return freq_vec

    # Given frequency in Hz and the B_field of the trap, returns energy in keV
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

    # Call this function to fit a histogram of start frequencies with the model.
    # Note that the data_hist_freq should be the StartFrequencies as given by katydid,
    # which will be from ~0 MHZ to ~90 MHz. You must also pass this function the
    # RF_ROI_MIN value from the metadata file of your data.
    # You must also supply a guess for the B_field present for the run;
    # 0.959 T is usually sufficient.
    # print_params = True will print out the fit parameters. Turn it to False to suppress
    def fit_data(self, RF_ROI_MIN,B_field,freq_bins,data_hist_freq,print_params=True):
        t = time.time()
        for gas in self.gases:
            self.check_existence_of_scatter_files(gas)
        bins_Hz = freq_bins+RF_ROI_MIN
        bins_keV = frequency_to_energy(bins_Hz,B_field)
        bins_keV = flip_array(bins_keV)
        data_hist = flip_array(data_hist_freq)
        bins_keV_nonzero , data_hist_nonzero , data_hist_err = get_only_nonzero_bins(bins_keV,data_hist)
        # Bounds for curve_fit
        FWHM_eV_min = 1e-5
        FWHM_eV_max = (bins_keV[len(bins_keV)-1] - bins_keV[0])*1000
        line_pos_keV_min = bins_keV[0]
        line_pos_keV_max = bins_keV[len(bins_keV)-1]
        scatter_prob_min = 1e-5
        scatter_prob_max = 1
        amplitude_min = 1e-5
        amplitude_max = np.sum(data_hist)*3
        # Initial guesses for curve_fit
        FWHM_guess = 5
        line_pos_guess = bins_keV[np.argmax(data_hist)]
        scatter_prob_guess = 0.5
        amplitude_guess = np.sum(data_hist)/2
        p_guess = [FWHM_guess, line_pos_guess] + [scatter_prob_guess,amplitude_guess] * len(self.gases)
        p_bounds = ([FWHM_eV_min, line_pos_keV_min] + [scatter_prob_min,amplitude_min] * len(selfgases),  [FWHM_eV_max, line_pos_keV_max] + [scatter_prob_max,amplitude_max] * len(self.gases))
        # Actually do the fitting
        params , cov = curve_fit(spectrum_func,bins_keV_nonzero,data_hist_nonzero,sigma=data_hist_err,p0=p_guess,bounds=p_bounds)
        # Name each of the resulting parameters and errors
        ################### Generalize to N Gases ###########################
        FWHM_G_eV_fit = params[0]
        line_pos_keV_fit = params[1]
        #starting at index 2, grabs every other entry. (which is how scattering probs are filled in for N gases)
        scatter_prob_fit = params[2::2]
        amplitude_fit = params[3::2]

        perr = np.sqrt(np.diag(cov))
        FWHM_eV_G_fit_err = perr[0]
        line_pos_keV_fit_err = perr[1]

        scatter_prob_fit_err = perr[2::2]
        amplitude_fit_err = perr[3::2]
        fit = spectrum_func(bins_keV[0:-1],*params)

        line_pos_Hz_fit , line_pos_Hz_fit_err = energy_guess_to_frequency(line_pos_keV_fit,line_pos_keV_fit_err,B_field)
        B_field_fit , B_field_fit_err = central_frequency_to_B_field(line_pos_Hz_fit,line_pos_Hz_fit_err)
        fit_Hz = flip_array(fit)
        bins_keV = bins_keV - line_pos_keV_fit + kr_line

        if print_params == True:
            output_string = 'Gaussian FWHM = '+str(FWHM_G_eV_fit)+' +/- '+str(FWHM_eV_G_fit_err)+' eV\n'
            output_string += 'Line position = '+str(line_pos_Hz_fit)+' +/- '+str(line_pos_Hz_fit_err)+' Hz\n'
            for i in range(len(self.gases)):
                output_string += self.gases[i] + ' Scatter probability = '+str(scatter_prob_fit[i])+' +/- ' + str(scatter_prob_fit_err[i])+'\n'
                output_string += self.gases[i] + ' Amplitude = '+str(amplitude_fit[i])+' +/- '+str(amplitude_fit_err[i]) + '\n'

            print(output_string)

        FWHM_eV_fit = FWHM_G_eV_fit
        FWHM_eV_fit_err = FWHM_eV_G_fit_err
        elapsed = time.time() - t
        print('Fit completed in '+str(elapsed)+'s')
        return cov, bins_keV , fit , bins_Hz, fit_Hz , FWHM_eV_fit, FWHM_eV_fit_err, line_pos_Hz_fit, line_pos_Hz_fit_err, B_field_fit, B_field_fit_err, scatter_prob_fit, scatter_prob_fit_err, amplitude_fit, amplitude_fit_err

    # given a FWHM for the lorentian component and the FWHM for the gaussian component,
    # this function estimates the FWHM of the resulting voigt distribution
    def FWHM_voigt(FWHM_L,FWHM_G):
        FWHM_V = 0.5346*FWHM_L+np.sqrt(0.2166*FWHM_L**2+FWHM_G**2)
        return FWHM_V
