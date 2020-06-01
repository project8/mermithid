import numpy as np
import json

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

    # Flips an array left-to-right.
    def flip_array(self, array):
        flipped = np.fliplr([array]).flatten()
        return flipped

    # shake spectrum by adding up shake spectrum for all the states up to i=24
    def shake_spectrum(self):
        x_array = self.input_std_eV_array
        x_array = self.flip_array(x_array)
        shake_spectrum = self.full_shake_spectrum(x_array, 0, 24)
        return shake_spectrum
    ###############################################################################