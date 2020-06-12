'''
Generate binned or pseudo unbinned data
Author: T. Weiss, C. Claessens
Date:4/6/2020
'''

from __future__ import absolute_import

import numpy as np
import math
import random
import time
import json
import os

from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.misc.FakeTritiumDataFunctions import *
from mermithid.processors.misc.KrComplexLineShape import KrComplexLineShape
logger = morphologging.getLogger(__name__)


__all__ = []
__all__.append(__name__)


class FakeDataGenerator(BaseProcessor):
    """
    Generates (pseudo) binned tritium data with Project 8 specific features, like lineshape and effciency.
    Can be confifured to produce energy and frequency data.
    """

    def InternalConfigure(self, params):
        """
        All parameters have defaults.
        Configurable parameters are:
        - Q [eV]: endpoint energy
        - neutrino_mass [eV]: true neutrino mass
        - minf [Hz]: low frequency cut-off (high cutoff is determined from efficiency dict)
        – Kmin [eV]: low energy cut-off
        – Kmax [eV]: high energy cut-off
        - n_steps: number of energy bins that data will be drawn from
        - B_field: used for energy-frequency conversion
        - sig_trans [eV]: width of thermal broadening
        - other_sig [eV]: width of other broadening
        - runtime [s]: used to calculate number of background events
        - S: number of signal events
        - A_b [1/eV/s]: background rate
        - poisson_stats (boolean): if True number of total events is random
        - err_from_B [eV]: energy uncertainty originating from B uncertainty
        - survival_prob: lineshape parameter - ratio of n+t/nth peak
        - scattering_sigma [eV]: lineshape parameter - 0-th peak gaussian broadening standard deviation
        - NScatters: lineshape parameter - number of scatters included in lineshape
        - scatter_proportion: fraction of hydrogen in complex lineshape
        - simplified_scattering_path: path to simplified lineshape parameters
        - path_to_detailed_scatter_spectra_dir: path to oscillator and or scatter_spectra_file
        - efficiency_path: path to efficiency vs. frequency (and uncertainties)
        - use_lineshape (boolean): determines whether tritium spectrum is smeared by lineshape.
          If False, it will only be smeared with a Gaussian
        - detailed_or_simplified_lineshape: If use lineshape, this string determines which lineshape model is used.
        - apply_efficiency (boolean): determines whether tritium spectrum is multiplied by efficiency
        - return_frequency: data is always generated as energies. If this parameter is true a list of frequencies is added to the dictionary
        """

        # Read other parameters
        self.Q = reader.read_param(params, 'Q', QT2) #Choose the atomic or molecular tritium endpoint
        self.m = reader.read_param(params, 'neutrino_mass', 0.2) #Neutrino mass (eV)
        self.Kmin = reader.read_param(params, 'Kmin', self.Q-self.m-2300)  #Energy corresponding to lower bound of frequency ROI (eV)
        self.Kmax = reader.read_param(params, 'Kmax', self.Q-self.m+1000)   #Same, for upper bound (eV)
        self.minf = reader.read_param(params, 'minf', 25.8e+9) #Minimum frequency
        if self.Kmax <= self.Kmin:
            logger.error("Kmax <= Kmin!")
            return False
        self.n_steps = reader.read_param(params, 'n_steps', 1e5)
        if self.n_steps <= 0:
            logger.error("Negative number of steps!")
            return False
        self.B_field = reader.read_param(params, 'B_field', 0.9578186017836624)

        #For a Phase IV gaussian smearing:
        self.sig_trans = reader.read_param(params, 'sig_trans', 0.020856) #Thermal translational Doppler broadening for atomic T (eV)
        self.other_sig = reader.read_param(params, 'other_sig', 0.05) #Kinetic energy broadening from other sources (eV)
        self.broadening = np.sqrt(self.sig_trans**2+self.other_sig**2) #Total energy broadening (eV)

        # Phase II Spectrum parameters
        self.runtime = reader.read_param(params, 'runtime', 6.57e6) #In seconds. Default time is ~2.5 months.
        self.S = reader.read_param(params, 'S', 3300)
        self.B_1kev = reader.read_param(params, 'B_1keV', 0.1) #Background rate per keV for full runtime
        self.A_b = reader.read_param(params, 'A_b', self.B_1kev/float(self.runtime)/1000.) #Flat background activity: events/s/eV
        self.B =self.A_b*self.runtime*(self.Kmax-self.Kmin) #Background poisson rate
        self.poisson_stats = reader.read_param(params, 'poisson_stats', True)
        self.err_from_B = reader.read_param(params, 'err_from_B', 0.) #In eV, kinetic energy error from f_c --> K conversion


        #Scattering model parameters
        self.survival_prob = reader.read_param(params, 'survival_prob', 0.77)
        self.scattering_sigma = reader.read_param(params, 'scattering_sigma', 18.6)
        self.NScatters = reader.read_param(params, 'NScatters', 20)
        self.scatter_proportion = reader.read_param(params, 'scatter_proportion', 0.8)

        #paths
        self.simplified_scattering_path = reader.read_param(params, 'simplified_scattering_path', '/host/input_data/simplified_scattering_params.txt')
        self.detailed_scatter_spectra_path = reader.read_param(params, 'path_to_detailed_scatter_spectra_dir', '.')
        self.efficiency_path = reader.read_param(params, 'efficiency_path', '')

        #options
        self.use_lineshape = reader.read_param(params, 'use_lineshape', True)
        self.detailed_or_simplified_lineshape = reader.read_param(params, 'detailed_or_simplified_lineshape', 'detailed')
        self.apply_efficiency = reader.read_param(params, 'apply_efficiency', False)
        self.return_frequency = reader.read_param(params, 'return_frequency', True)

        # nwill be replaced with complex lineshape object if detailed lineshape is used
        self.complexLineShape = None

        # get file content if needed
        # get efficiency dictionary
        if self.apply_efficiency:
            self.efficiency_dict = self.load_efficiency_curve()
            np.random.seed()
        else:
            self.efficiency_dict = None

        # generate data with lineshape
        if self.use_lineshape:
            self.lineshape = self.detailed_or_simplified_lineshape
            if self.lineshape == 'simplified':
                self.SimpParams = self.load_simp_params(self.scattering_sigma,
                                                        self.survival_prob,
                                                        self.NScatters)
            elif self.lineshape=='detailed':
                # check path exists
                if 'scatter_spectra_file' in self.detailed_scatter_spectra_path:
                    full_path = self.detailed_scatter_spectra_path
                    self.detailed_scatter_spectra_path, _ = os.path.split(full_path)
                else:
                    full_path = os.path.join(self.detailed_scatter_spectra_path, 'scatter_spectra_file')

                logger.info('Path to scatter_spectra_file: {}'.format(self.detailed_scatter_spectra_path))

                if not os.path.exists(full_path):
                    raise IOError('{} does not exist'.format(full_path))

                # lineshape params
                self.SimpParams = [self.scattering_sigma*2*math.sqrt(2*math.log(2)), self.survival_prob]

                # complex lineshape object
                # Setup and configure lineshape processor
                complexLineShape_config = {
                    'gases': ["H2","He"],
                    'max_scatters': self.NScatters,
                    'fix_scatter_proportion': True,
                    # When fix_scatter_proportion is True, set the scatter proportion for gas1 below
                    'gas1_scatter_proportion': self.scatter_proportion,
                    # This is an important parameter which determines how finely resolved
                    # the scatter calculations are. 10000 seems to produce a stable fit, with minimal slowdown
                    'num_points_in_std_array': 10000,
                    'B_field': self.B_field,
                    'base_shape': 'lorentzian', # needs to be replaced by dirac
                    'path_to_osc_strengths_files': self.detailed_scatter_spectra_path
                }

                # create complex lineshape object and configure
                self.complexLineShape = KrComplexLineShape("complexLineShape")
                logger.info('Configuring complex lineshape')
                self.complexLineShape.Configure(complexLineShape_config)
            else:
                raise ValueError("'detailed_or_simplified' is neither 'detailed' nor 'simplified'")

        else:
            self.lineshape = 'gaussian'
            self.SimpParams = [self.broadening]
            logger.info('Lineshape is Gaussian')
        return True



    def InternalRun(self):

        if self.return_frequency:
            ROIbound = [self.minf]
        else:
            ROIbound = [self.Kmin, self.Kmax]

        Kgen = self.generate_unbinned_data(self.Q, self.m,
                                           ROIbound,
                                           self.S, self.B_1kev,
                                           nsteps=self.n_steps,
                                           lineshape=self.lineshape,
                                           params=self.SimpParams,
                                           efficiency_dict = self.efficiency_dict,
                                           err_from_B=self.err_from_B,
                                           B_field=self.B_field)


        self.results = Kgen

        if self.return_frequency:
            self.results['F'] = Frequency(Kgen['K'], self.B_field)
        return True



    def load_simp_params(self, sigma, survival_prob, Nscatters=20):
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
        for i,_ in enumerate(pi):
            for j in range(Nscatters):
                pi[i][j] = random.gauss(pi[i][j], pi_err[i][j])
        simp_params=[sigma*2*math.sqrt(2*math.log(2)), survival_prob] + [p[:Nscatters] for p in pi]
        return simp_params


    def load_efficiency_curve(self):
        # Efficiency dictionary
        with open(self.efficiency_path, 'r') as infile:
            eff_dict = json.load(infile)

        return eff_dict


    def generate_unbinned_data(self, Q_mean, mass, ROIbound, S, B_1kev, nsteps=10**4,
                               lineshape='gaussian', params=[0.5], efficiency_dict=None,
                                array_method=True, err_from_B=None, B_field=None):
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

        if self.return_frequency:
            minf = ROIbound[0]
            if efficiency_dict is not None:
                maxf = max(efficiency_dict['frequencies'])
            else:
                maxf = max(self.load_efficiency_curve()['frequencies'])
            Kmax, Kmin = Energy(minf, B_field), Energy(maxf, B_field)
        else:
            Kmin, Kmax = ROIbound[0], ROIbound[1]
        B = B_1kev*(Kmax-Kmin)/1000.

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

        if not nsteps > 0:
            raise ValueError('n_steps is not greater zero')
        step_size = (Kmax_eff-Kmin_eff)/float(nsteps)
        logger.info('Stepsize is {} eV'.format(step_size))

        #Options of kinetic energies to be sampled
        self.Koptions = np.arange(Kmin_eff, Kmax_eff, step_size)

        if efficiency_dict is not None:
            logger.info('Evaluating efficiencies')
            efficiency_mean, efficiency_error = efficiency_from_interpolation(self.Koptions, efficiency_dict, B_field)
            logger.info("Sampling efficiencies given means and uncertainties")
            efficiency = np.random.normal(efficiency_mean, efficiency_error)
            eff_negative = (efficiency<0.)
            efficiency[eff_negative] = 0. #Whenever this occurs, efficiency_mean=0 and efficiency_error=1
        else:
            efficiency, _ = 1, 0

        #Create array of sampled kinetic energies.
        logger.info('Going to calculate rates')
        time0 = time.time()

        if array_method == True:
            ratesS = convolved_spectral_rate_arrays(self.Koptions, Q_mean, mass, Kmin, lineshape, params, min_energy, max_energy, self.complexLineShape)
        else:
            ratesS = [convolved_spectral_rate(K, Q_mean, mass, Kmin, lineshape, params, min_energy, max_energy) for K in self.Koptions]

        # multiply rates by efficiency
        ratesS = ratesS*efficiency

        time1 = time.time()
        logger.info('... signal rate took {} s'.format(time1-time0))

        # background
        if array_method == True:
            ratesB = convolved_bkgd_rate_arrays(self.Koptions, Kmin, Kmax, lineshape, params, min_energy, max_energy, self.complexLineShape)
        else:
            ratesB = [convolved_bkgd_rate(K, Kmin, Kmax, lineshape, params, min_energy, max_energy) for K in self.Koptions]

        time2 = time.time()
        logger.info('... background rate took {} s'.format(time2 - time1))

        if err_from_B != None and err_from_B != 0.:
            dE = self.Koptions[1] - self.Koptions[0]
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
        self.probs = (S*probsS + B*probsB)/(S+B)

        logger.info('Generating data')
        time4 = time.time()

        if self.poisson_stats:
            KE = np.random.choice(self.Koptions, np.random.poisson(S+B), p = self.probs)
        else:
            KE = np.random.choice(self.Koptions, round(S+B), p = self.probs)
        time5 = time.time()

        logger.info('... took {} s'.format(time5-time4))
        logger.info('Number of values in array that are not unique: {} out of {}'.format(np.size(KE) - len(set(KE)), np.size(KE)))

        if self.return_frequency:
            return {'K':KE, 'maxf':maxf, 'minf':minf}
        else:
            return {'K':KE}
