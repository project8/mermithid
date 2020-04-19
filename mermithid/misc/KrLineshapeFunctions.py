'''
Generate binned or pseudo unbinned data
Author: T. Weiss, C. Claessens, Y. Sun
Date:4/6/2020
'''

from __future__ import absolute_import

import numpy as np
import math

from scipy.special import gamma
from scipy import integrate
from scipy import stats
from scipy.optimize import fsolve
from scipy import constants
from scipy.interpolate import interp1d
from scipy.signal import convolve
import random
import os

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

from mermithid.misc.Constants import *
from mermithid.misc.ConversionFunctions import *


"""
functions from detailed lineshape
"""

"""
Adapted from https://github.com/project8/scripts/blob/feature/KrScatterFit/machado/fitting/data_fitting_rebuild.py#L429 - by E. M. Machado

Changes from that version:
- Kr specific functions (shake-up/shake-off) removed
- Spectrum is normalized
"""


# Natural constants
kr_line = kr_k_line_e()*1e-3 #17.8260 # keV
kr_line_width = kr_k_line_width() #2.83 # eV
e_charge = e() #1.60217662*10**(-19) # Coulombs , charge of electron
m_e = m_electron()/e()*c()**2 #9.10938356*10**(-31) # Kilograms , mass of electron
mass_energy_electron = m_electron()*1e-3 #510.9989461 # keV
max_scatters = 20
gases = ["H2"]

# This is an important parameter which determines how finely resolved
# the scatter calculations are. 10000 seems to produce a stable fit, with minimal slowdown
num_points_in_std_array = 10000

# Establishes a standard energy loss array (SELA) from -1000 eV to 1000 eV
# with number of points equal to num_points_in_std_array. All convolutions
# will be carried out on this particular discretization
def std_eV_array():
    emin = -1000
    emax = 1000
    array = np.linspace(emin,emax,num_points_in_std_array)
    return array

# A lorentzian function
def lorentzian(x_array,x0,FWHM):
    HWHM = FWHM/2.
    func = (1./np.pi)*HWHM/((x_array-x0)**2.+HWHM**2.)
    return func

# A lorentzian line centered at 0 eV, with 2.83 eV width on the SELA
def std_lorentzian_17keV():
    x_array = std_eV_array()
    ans = lorentzian(x_array,0,kr_line_width)
    return ans


# A gaussian function
def gaussian(x_array,A,sigma=0,mu=0):
    if isinstance(A, list):
        a = A
        x = x_array
        return 1/((2.*np.pi)**0.5*a[0])*(np.exp(-0.5*((x-a[1])/a[0])**2))
    else:
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
    f_norm = integrate.simps(f,x=x_arr)
    # if  f_norm < 0.99 or f_norm > 1.01:
    #     print(f_norm)
    f_normed = f/f_norm
    return f_normed

#returns array with energy loss/ oscillator strength data
#def read_oscillator_str_file(filename):
#    f = open(filename, "r")
#    lines = f.readlines()
#    energyOsc = [[],[]] #2d array of energy losses, oscillator strengths
#
#    for line in lines:
#        if line != "" and line[0]!="#":
#            raw_data = [float(i) for i in line.split("\t")]
#            energyOsc[0].append(raw_data[0])
#            energyOsc[1].append(raw_data[1])
#
#    energyOsc = np.array(energyOsc)
#    ### take data and sort by energy
#    sorted_indices = energyOsc[0].argsort()
#    energyOsc[0] = energyOsc[0][sorted_indices]
#    energyOsc[1] = energyOsc[1][sorted_indices]
#    return energyOsc

# A sub function for the scatter function. Found in
# "Energy loss of 18 keV electrons in gaseous T and quench condensed D films"
# by V.N. Aseev et al. 2000
#def aseev_func_tail(energy_loss_array, gas_type):
#    if gas_type=="H2":
#        A2, omeg2, eps2 = 0.195, 14.13, 10.60
#    elif gas_type=="Kr":
#        A2, omeg2, eps2 = 0.4019, 22.31, 16.725
#    return A2*omeg2**2./(omeg2**2.+4*(energy_loss_array-eps2)**2.)

#convert oscillator strength into energy loss spectrum
#def get_eloss_spec(e_loss, oscillator_strength, kinetic_en = kr_line * 1000): #energies in eV
#    e_rydberg = 13.605693009 #rydberg energy (eV)
#    a0 = 5.291772e-11 #bohr radius
#    return np.where(e_loss>0 , 4.*np.pi*a0**2 * e_rydberg / (kinetic_en * e_loss) * oscillator_strength * np.log(4. * kinetic_en * e_loss / (e_rydberg**3.) ), 0)

# Function for energy loss from a single scatter of electrons by
# V.N. Aseev et al. 2000
# This function does the work of combining fit_func1 and fit_func2 by
# finding the point where they intersect.
# Evaluated on the SELA
#def single_scatter_f(gas_type):
#    energy_loss_array = std_eV_array()
#    f = 0 * energy_loss_array
#
#    input_filename = "../data/" + gas_type + "OscillatorStrength.txt"
#    energy_fOsc = read_oscillator_str_file(input_filename)
#    fData = interpolate.interp1d(energy_fOsc[0], energy_fOsc[1], kind='linear')
#    for i in range(len(energy_loss_array)):
#        if energy_loss_array[i] < energy_fOsc[0][0]:
#            f[i] = 0
#        elif energy_loss_array[i] <= energy_fOsc[0][-1]:
#            f[i] = fData(energy_loss_array[i])
#        else:
#            f[i] = aseev_func_tail(energy_loss_array[i], gas_type)
#
#    f_e_loss = get_eloss_spec(energy_loss_array, f)
#    f_normed = normalize(f_e_loss)
#    #plt.plot(energy_loss_array, f_e_loss)
#    #plt.show()
#    return f_normed

# Convolves a function with the single scatter function, on the SELA
#def another_scatter(input_spectrum, gas_type):
#    single = single_scatter_f(gas_type)
#    f = convolve(single,input_spectrum,mode='same')
#    f_normed = normalize(f)
#    return f_normed

# Convolves the scatter function with itself over and over again and saves
# the results to .npy files.
#def generate_scatter_convolution_files(gas_type):
#    t = time.time()
#    first_scatter = single_scatter_f(gas_type)
#    scatter_num_array = range(2,max_scatters+1)
#    current_scatter = first_scatter
#    np.save('scatter_spectra_files/scatter'+gas_type+"_"+str(1).zfill(2),current_scatter)
#    # x = std_eV_array() # diagnostic
#    for i in scatter_num_array:
#        current_scatter = another_scatter(current_scatter, gas_type)
#        np.save('scatter_spectra_files/scatter'+gas_type+"_"+str(i).zfill(2),current_scatter)
#    # plt.plot(x,current_scatter) # diagnostic
#    # plt.show() # diagnostic
#    elapsed = time.time() - t
#    logger.info('Files generated in '+str(elapsed)+'s')
#    return

# Returns the name of the current path
def get_current_path():
    path = os.path.abspath(os.getcwd())
    return path

# Prints a list of the contents of a directory
def list_files(path):
    directory = os.popen("ls "+path).readlines()
    strippeddirs = [s.strip('\n') for s in directory]
    return strippeddirs

# Returns the name of the current directory
def get_current_dir():
    current_path = os.popen("pwd").readlines()
    stripped_path = [s.strip('\n') for s in current_path]
    stripped_path = stripped_path[0].split('/')
    current_dir = stripped_path[len(stripped_path)-1]
    return current_dir

# Checks for the existence of a directory called 'scatter_spectra_files'
# and checks that this directory contains the scatter spectra files.
# If not, this function calls generate_scatter_convolution_files.
# This function also checks to make sure that the scatter files have the correct
# number of points in the SELA, and if not, it generates fresh files
#def check_existence_of_scatter_files(gas_type):
#    current_path = get_current_path()
#    current_dir = get_current_dir()
#    stuff_in_dir = list_files(current_path)
#    if 'scatter_spectra_files' not in stuff_in_dir and current_dir != 'scatter_spectra_files':
#        logger.warning('Scatter files not found, generating')
#        os.popen("mkdir scatter_spectra_files")
#        time.sleep(2)
#        generate_scatter_convolution_files(gas_type)
#    else:
#        directory = os.popen("ls scatter_spectra_files").readlines()
#        strippeddirs = [s.strip('\n') for s in directory]
#        if len(directory) != len(gases) * max_scatters:
#            generate_scatter_convolution_files(gas_type)
#        test_file = 'scatter_spectra_files/scatter'+gas_type+'_01.npy'
#        test_arr = np.load(test_file)
#        if len(test_arr) != num_points_in_std_array:
#            logger.warning('Scatter files do not match standard array binning, generating fresh files')
#            generate_scatter_convolution_files(gas_type)
#    return

# Given a function evaluated on the SELA, convolves it with a gaussian
def convolve_gaussian(func_to_convolve,gauss_FWHM_eV):
    sigma = gaussian_FWHM_to_sigma(gauss_FWHM_eV)
    resolution_f = std_gaussian(sigma)
    ans = convolve(resolution_f,func_to_convolve,mode='same')
    ans_normed = normalize(ans)
    return ans_normed

# Produces a full spectral shape on the SELA, given a gaussian resolution
# and a scatter probability
def make_spectrum(gauss_FWHM_eV,scatter_prob,gas_type, emitted_peak='lorentzian'):
    current_path = get_current_path()
    # check_existence_of_scatter_files()
    #filenames = list_files('scatter_spectra_files')
    scatter_num_array = range(1,max_scatters+1)
    en_array = std_eV_array()
    current_full_spectrum = np.zeros(len(en_array))
    if emitted_peak == 'lorentzian':
        current_working_spectrum = std_lorentzian_17keV() #Normalized
    elif emitted_peak == 'shake':
        current_working_spectrum = shake_spectrum()
    current_working_spectrum = convolve_gaussian(current_working_spectrum,gauss_FWHM_eV) #Still normalized
    zeroth_order_peak = current_working_spectrum
    current_full_spectrum += current_working_spectrum #I believe, still normalized
    norm = 1
    for i in scatter_num_array:
        current_working_spectrum = np.load(os.path.join(current_path, 'scatter_spectra_files/scatter'+gas_type+'_'+str(i).zfill(2)+'.npy'))
        current_working_spectrum = normalize(convolve(zeroth_order_peak,current_working_spectrum,mode='same'))
        current_full_spectrum += current_working_spectrum*scatter_prob**scatter_num_array[i-1]
        norm += scatter_prob**scatter_num_array[i-1]
    # plt.plot(en_array,current_full_spectrum) # diagnostic
    # plt.show() # diagnostic
    return current_full_spectrum/norm

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


def spectrum_func(x_keV, *p0):
    logger.info('Using detailed lineshape')
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

    for gas_index in range(len(gases)):
        gas_type = gases[gas_index]
        scatter_prob = p0[2*gas_index+2]
        amplitude    = p0[2*gas_index+3]

        full_spectrum = make_spectrum(FWHM_G_eV,scatter_prob, gas_type)
        full_spectrum_rev = flip_array(full_spectrum)
        f[nonzero_idx] += amplitude*np.interp(x_eV_minus_line[nonzero_idx],en_array_rev,full_spectrum_rev)
    return f

