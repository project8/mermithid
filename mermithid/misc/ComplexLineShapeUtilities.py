import numpy as np
import json
from mermithid.misc import Constants, ConversionFunctions

# Natural constants
kr_17keV_line = Constants.kr_k_line_e()/1000 # 17.8260 keV
kr_17keV_line_width = Constants.kr_k_line_width() # 2.83 eV
e_charge = Constants.e() # 1.60217662*10**(-19) Coulombs , charge of electron
m_e = Constants.m_electron()*Constants.e()/((Constants.c())**2) # 9.10938356*10**(-31) Kilograms , mass of electron
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
    elif gas_type=="He":
        A2, omeg2, eps2 = 0.1187, 33.40, 10.43
    elif gas_type=="Ar":
        A2, omeg2, eps2 = 0.3344, 21.91, 21.14
    return A2*omeg2**2./(omeg2**2.+4*(energy_loss_array-eps2)**2.)

#convert oscillator strength into energy loss spectrum
def get_eloss_spec(e_loss, oscillator_strength, kr_17keV_line): #energies in eV
    kinetic_en = kr_17keV_line
    e_rydberg = 13.605693009 #rydberg energy (eV)
    a0 = 5.291772e-11 #bohr radius
    argument_of_log = np.where(e_loss > 0, 4. * kinetic_en * e_rydberg / (e_loss**2.) , 1e-5)
    return np.where(e_loss>0 , 1./(e_loss) * oscillator_strength*np.log(argument_of_log), 0)

# Takes only the nonzero bins of a histogram
def get_only_nonzero_bins(bins,hist):
    nonzero_idx = np.where(hist!=0)
    hist_nonzero = hist[nonzero_idx]
    hist_err = np.sqrt(hist_nonzero)
    bins_nonzero = bins[nonzero_idx]
    return bins_nonzero , hist_nonzero , hist_err

# Takes bins of a histogram and return the errs
def get_hist_err_bins(hist):
    hist_err = np.sqrt(hist)
    zero_idx = np.where(hist == 0)
    hist_err[zero_idx] = 1e-5
    return hist_err

# Flips an array left-to-right. Useful for converting between energy and frequency
def flip_array(array):
    flipped = np.fliplr([array]).flatten()
    return flipped

## Given energy in keV and the self.B_field of the trap, returns frequency in Hz
#def energy_to_frequency(energy_vec, B_field):
#    freq_vec = (e_charge*B_field/((2.*np.pi*m_e)*(1+energy_vec/mass_energy_electron)))
#    return freq_vec
#
## Given frequency in Hz and the self.B_field of the trap, returns energy in keV
#def frequency_to_energy(freq_vec,B_field):
#    energy_vec = (e_charge*B_field/((2.*np.pi*m_e*freq_vec))-1)*mass_energy_electron
#    return energy_vec

# Converts an energy to frequency using a guess for magnetic field. Can handle errors too
def energy_guess_to_frequency(energy_guess, energy_guess_err, B_field_guess):
    frequency = ConversionFunctions.Frequency(energy_guess*1000, B_field_guess)
    const = e_charge*B_field_guess/(2.*np.pi*m_e)
    frequency_err = const/(1+energy_guess/mass_energy_electron)**2*energy_guess_err/mass_energy_electron
    return frequency , frequency_err

# Given a frequency and error, converts those to B field values assuming the line is the 17.8 keV line
def central_frequency_to_B_field(central_freq,central_freq_err):
    const = (2.*np.pi*m_e)*(1+kr_17keV_line/mass_energy_electron)/e_charge
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