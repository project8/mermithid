'''
Generate binned or pseudo unbinned data
Author: T. Weiss, C. Claessens
Date:4/6/2020
'''

from __future__ import absolute_import

import numpy as np
import math
import scipy as sp
from scipy.special import gamma
from scipy import integrate
from scipy import stats
from scipy.optimize import fsolve
from scipy import constants
from scipy.interpolate import interp1d
from scipy.signal import convolve
import random
import time
import json
import os

from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.misc import Constants
from mermithid.misc.FakeTritiumDataFunctions import *

logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

#class DetailedLineshape():
#    def __init__(self):
#        print('Using detailed lineshape')
#
#    def spectrum_func(self, x_keV, *p0):
#        x_eV = x_keV*1000.
#        en_loss_array = std_eV_array()
#        en_loss_array_min = en_loss_array[0]
#        en_loss_array_max = en_loss_array[len(en_loss_array)-1]
#        en_array_rev = flip_array(-1*en_loss_array)
#        f = np.zeros(len(x_keV))
#
#        FWHM_G_eV = p0[0]
#        line_pos_keV = p0[1]
#
#        line_pos_eV = line_pos_keV*1000.
#        x_eV_minus_line = x_eV - line_pos_eV
#        zero_idx = np.r_[np.where(x_eV_minus_line<-1*en_loss_array_max)[0],np.where(x_eV_minus_line>-1*en_loss_array_min)[0]]
#        nonzero_idx = [i for i in range(len(x_keV)) if i not in zero_idx]
#
#        for gas_index in range(len(gases)):
#            gas_type = gases[gas_index]
#            scatter_prob = p0[2*gas_index+2]
#            amplitude    = p0[2*gas_index+3]
#
#            full_spectrum = make_spectrum(FWHM_G_eV,scatter_prob, gas_type)
#            full_spectrum_rev = flip_array(full_spectrum)
#            f[nonzero_idx] += amplitude*np.interp(x_eV_minus_line[nonzero_idx],en_array_rev,full_spectrum_rev)
#        return f


class FakeDataGenerator(BaseProcessor):

    def InternalConfigure(self, params):
        '''
        Configure
        '''

        # Read other parameters
        self.Q = reader.read_param(params, 'Q', QT2) #Choose the atomic or molecular tritium endpoint
        self.m = reader.read_param(params, 'neutrino_mass', 0.0085) #Neutrino mass (eV)
        self.Kmin = reader.read_param(params, 'Kmin', self.Q-self.m-2300)  #Energy corresponding to lower bound of frequency ROI (eV)
        self.Kmax = reader.read_param(params, 'Kmax', self.Q-self.m+1000)   #Same, for upper bound (eV)
        self.bin_region_widths = reader.read_param(params, 'bin_region_width', [1., 9.]) #List of widths (eV) of regions where bin sizes do not change, from high to low energy
        self.nbins_per_region = reader.read_param(params, 'nbins_per_region', [300., 9.]) #List of number of bins in each of those regions, from high to low energy

        self.Nparticles = reader.read_param(params, 'Nparticles', 10**(19)) #Density*volume*efficiency
        self.runtime = reader.read_param(params, 'runtime', 31556952.) #In seconds
        self.n_steps = reader.read_param(params, 'n_steps', 1000)
        self.B_field = reader.read_param(params, 'B_field', 0.9578186017836624)
        #For Phase IV:
        #A_s = find_signal_activity(Nparticles, m, Q, Kmin) #Signal activity: events/s in ROI
        #S = A_s*runtime #Signal poisson rate
        self.S = reader.read_param(params, 'S', 2500)
        self.A_b = reader.read_param(params, 'A_b', 10**(-12)) #Flat background activity: events/s/eV
        self.B =self.A_b*self.runtime*(self.Kmax-self.Kmin) #Background poisson rate
        self.fb = self.B/(self.S+self.B) #Background fraction
        self.err_from_B = reader.read_param(params, 'err_from_B', 0.5) #In eV, kinetic energy error from f_c --> K conversion

        #For a Phase IV gaussian smearing:
        self.sig_trans = reader.read_param(params, 'sig_trans', 0.020856) #Thermal translational Doppler broadening for atomic T (eV)
        self.other_sig = reader.read_param(params, 'other_sig', 0.05) #Kinetic energy broadening from other sources (eV)
        self.broadening = np.sqrt(self.sig_trans**2+self.other_sig**2) #Total energy broadening (eV)



        #Simplified scattering model parameters
        self.scattering_prob = reader.read_param(params, 'scattering_prob', 0.77)
        self.scattering_sigma = reader.read_param(params, 'scattering_sigma', 18.6)
        self.NScatters = reader.read_param(params, 'NScatters', 20)


        #paths
        self.simplified_scattering_path = reader.read_param(params, 'simplified_scattering_path', '/host/input_data/simplified_scattering_params.txt')
        self.efficiency_path = reader.read_param(params, 'efficiency_path', '/host/input_data/combined_energy_corrected_eff_at_quad_trap_frequencies.json')


        #options
        self.use_lineshape = reader.read_param(params, 'use_lineshape', True)
        self.detailed_or_simplified_lineshape = reader.read_param(params, 'detailed_or_simplified_lineshape', 'simplified')
        self.apply_efficiency = reader.read_param(params, 'apply_efficiency', False)
        self.return_frequency = reader.read_param(params, 'return_frequency', True)
        return True



    def InternalRun(self):

        # get efficiency dictionary
        if self.apply_efficiency:
            efficiency_dict = self.load_efficiency_curve()
        else:
            efficiency_dict = None

        # generate data with lineshape
        if self.use_lineshape:
            lineshape = self.detailed_or_simplified_lineshape
            SimpParams = self.load_simp_params(self.scattering_sigma,
                                               self.scattering_prob,
                                               self.NScatters)

        else:
            lineshape = 'gaussian'
            SimpParams = [self.broadening]
            logger.info('Lineshape is Gaussian')

        Kgen = self.generate_unbinned_data(self.Q, self.m,
                                           self.Kmin, self.Kmax,
                                           self.S, self.B,
                                           nsteps=self.n_steps,
                                           lineshape=lineshape,
                                           params=SimpParams,
                                           efficiency_dict = efficiency_dict,
                                           err_from_B=self.err_from_B,
                                           B_field=self.B_field)


        self.results = {'K': Kgen}

        if self.return_frequency:
            self.results['F'] = Frequency(Kgen, self.B_field)



        return True



    def load_simp_params(self, sigma, scattering_prob, Nscatters=20):
        f = open(self.simplified_scattering_path, "r")
        num = []
        pi = [[] for i in range(4)]
        pi_err = [[] for i in range(4)]
        for line in f.readlines():
            elements = line.strip().split(" ")
            for j in range(9):
                if j==0:
                    num.append(float(elements[j]))
                elif j%2==1:
                    pi[int((j-1)/2.)].append(float(elements[j]))
                else:
                    pi_err[int((j-2)/2.)].append(float(elements[j]))
        f.close()
        #Sample scattering params p0--p3 from normal distributions with stdevs equal to the corresponding uncertainties
        for i in range(len(pi)):
            for j in range(Nscatters):
                pi[i][j] = random.gauss(pi[i][j], pi_err[i][j])
        simp_params=[sigma*2*math.sqrt(2*math.log(2)), scattering_prob] + [p[:Nscatters] for p in pi]
        return simp_params

    def load_efficiency_curve(self):
        # Efficiency dictionary
        with open(self.efficiency_path, 'r') as infile:
            eff_dict = json.load(infile)

        return eff_dict

    def generate_unbinned_data(self, Q_mean, mass, Kmin, Kmax, S, B, nsteps=10**4, lineshape=
                               'gaussian', params=[0.5], efficiency_dict=None, array_method=True, err_from_B=None, B_field=None):
        """
            Returns list of event kinetic energies (eV).
            The 'nsteps' parameter can be increased to reduce the number of repeated energies in the returned list, improving the reliability of the sampling method employed here.
            The 'lineshape' parameter, specifying the probability distribution that is convolved with the tritium spectrum, can have the following options:
            - 'gaussian'
            - 'simplified': Central gaussian + approximated scattering
            - 'detailed': Central gaussian + detailed scattering
            'params' is a list of the params inputted into the lineshape function. The first entry of the list should be a standard deviation of full width half max that provides the scale of the lineshape width.
        """
        logger.info('Going to generate pseudo-unbinned data with {} lineshape'.format(lineshape))

        nstdevs = 7 #Number of standard deviations (of size broadening) below Kmin and above Q-m to generate data, for the gaussian case
        FWHM_convert = 2*math.sqrt(2*math.log(2))
        if lineshape=='gaussian':
            max_energy = nstdevs*params[0]
            min_energy = -1000
        elif lineshape=='simplified_scattering' or lineshape=='simplified' or lineshape=='detailed_scattering' or lineshape=='detailed':
            max_energy = nstdevs/FWHM_convert*params[0]
            min_energy = -1000

        Kmax_eff = Kmax+max_energy #Maximum energy for data is slightly above Kmax>Q-m
        Kmin_eff = Kmin+min_energy #Minimum is slightly below Kmin<Q-m

        step_size = (Kmax_eff-Kmin_eff)/float(nsteps)
        logger.info('Stepsize is {} eV'.format(step_size))

        #Options of kinetic energies to be sampled
        Koptions = np.arange(Kmin_eff, Kmax_eff, step_size)

        if efficiency_dict is not None:
            logger.info('Evaluating efficiencies')
            efficiency, efficiency_error = efficiency_from_interpolation(Koptions, efficiency_dict, B_field)
        else:
            efficiency, efficiency_error = 1, 0

        #Create array of sampled kinetic energies.
        logger.info('Going to calculate rates')
        time0 = time.time()

        if array_method == True:
            ratesS = convolved_spectral_rate_arrays(Koptions, Q_mean, mass, Kmin, lineshape, params, min_energy, max_energy)
        else:
            ratesS = [convolved_spectral_rate(K, Q_mean, mass, Kmin, lineshape, params, min_energy, max_energy) for K in Koptions]

        # multiply rates by efficiency
        ratesS = ratesS*efficiency

        time1 = time.time()
        print('... signal rate took {} s'.format(time1-time0))

        # background
        if array_method == True:
            ratesB = convolved_bkgd_rate_arrays(Koptions, Kmin, Kmax, lineshape, params, min_energy, max_energy)
        else:
            ratesB = [convolved_bkgd_rate(K, Kmin, Kmax, lineshape, params, min_energy, max_energy) for K in Koptions]

        time2 = time.time()
        print('... background rate took {} s'.format(time2 - time1))

        if err_from_B != None:
            dE = Koptions[1] - Koptions[0]
            #n_dE = round(max_energy/dE) #Number of steps for the narrow gaussian for energies > 0
            n_dE = 10*err_from_B/dE

            K_lineshape =  np.arange(-n_dE*dE, n_dE*dE, dE)
            #Generating finely spaced points on a gaussian
            gaussian_rates = gaussian(K_lineshape, [err_from_B, 0])

            ratesS = convolve(ratesS, gaussian_rates, mode='same')
            ratesB = convolve(ratesB, gaussian_rates, mode='same')


        ratesS[ratesS<0.] = 0.
        ratesB[ratesB<0.] = 0.
        rate_sumS, rate_sumB = np.sum(ratesS), np.sum(ratesB)
        probsS = np.array(ratesS)/rate_sumS
        probsB = np.array(ratesB)/rate_sumB
        probs = (S*probsS + B*probsB)/(S+B)


        logger.info('Generating data')
        time4 = time.time()
        #KE = np.random.choice(Koptions, np.random.poisson(S+B), p = probs)
        KE = np.random.choice(Koptions, round(S+B), p = probs)
        time5 = time.time()
        print('... took {} s'.format(time5-time4))
        print('Number of values in array that are not unique:', np.size(KE) - len(set(KE)), 'out of', np.size(KE))

        return KE


