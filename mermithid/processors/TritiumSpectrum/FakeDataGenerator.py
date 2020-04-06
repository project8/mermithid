'''
Generate binned or pseudo unbinned data
Author: T. Weiss, C. Claessens
Date:4/6/2020
'''

from __future__ import absolute_import

import numpy as np
import math
import numpy as np
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

from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.misc import Constants

logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)


class FakeDataGenerator(BaseProcessor):

    def InternalConfigure(self, params):
        '''
        Configure
        '''
        #Physical constants
        self.me = 510999. #eV
        self.alpha = 1/137.036
        self.c = 299792458. #m/s
        self.hbar = 6.58212*10**(-16) #eV*s
        self.gv=1. #Vector coupling constant
        self.lambdat = 1.2724 # +/- 0.0023, from PDG (2018): http://pdg.lbl.gov/2018/listings/rpp2018-list-n.pdf
        self.ga=self.gv*(-self.lambdat) #Axial vector coupling constant
        self.Mnuc2 = self.gv**2 + 3*self.ga**2 #Nuclear matrix element
        self.GF =  1.1663787*10**(-23) #Gf/(hc)^3, in eV^(-2)
        self.Vud = 0.97425 #CKM element

        #Beta decay-specific physical constants
        self.QT = 18563.251 #For atomic tritium (eV), from Bodine et al. (2015)
        self.QT2 =  18573.24 #For molecular tritium (eV), Bodine et al. (2015)
        self.Rn =  2.8840*10**(-3) #Helium-3 nuclear radius in units of me, from Kleesiek et al. (2018): https://arxiv.org/pdf/1806.00369.pdf
        self.M = 5497.885 #Helium-3 mass in units of me, Kleesiek et al. (2018)
        self.atomic_num = 2. #For helium-3
        self.g =(1-(self.atomic_num*self.alpha)**2)**0.5 #Constant to be used in screening factor and Fermi function calculations
        self.V0 = 76. #Nuclear screening potential of orbital electron cloud of the daughter atom, from Kleesiek et al. (2018)
        self.mu = 5.107 #Difference between magnetic moments of helion and triton, for recoil effects correction

        # Read other parameters
        self.Q = reader.read_param(params, 'Q', self.QT2) #Choose the atomic or molecular tritium endpoint
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
        self.detailed_scattering_path = reader.read_param(params, 'detailed_scattering_path', '/host/input_data/detailed_scattering')
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
            SimpParams = self.load_simp_params(self.scattering_sigma, self.scattering_prob, self.NScatters)

        else:
            lineshape = 'gaussian'
            SimpParams = [self.broadening]

        Kgen = self.generate_unbinned_data(self.Q, self.m, self.Kmin, self.Kmax, self.S, self.B, nsteps=self.n_steps, lineshape=lineshape, params=SimpParams, efficiency_dict = efficiency_dict, err_from_B=self.err_from_B, B_field=self.B_field)


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
        print('\nGoing to generate pseudo-unbinned data with {} lineshape'.format(lineshape))

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
        print('Stepsize is {} eV'.format(step_size))

        #Options of kinetic energies to be sampled
        Koptions = np.arange(Kmin_eff, Kmax_eff, step_size)

        if efficiency_dict is not None:
            print('Evaluating efficiencies')
            efficiency, efficiency_error = efficiency_from_interpolation(Koptions, efficiency_dict, B_field)
        else:
            efficiency, efficiency_error = 1, 0

        #Create array of sampled kinetic energies.
        print('\nGoing to calculate rates')
        time0 = time.time()
        if array_method == True:
            ratesS = convolved_spectral_rate_arrays(Koptions, Q_mean, mass, Kmin, lineshape, params, min_energy, max_energy)
        else:
            ratesS = [convolved_spectral_rate(K, Q_mean, mass, Kmin, lineshape, params, min_energy, max_energy) for K in Koptions]
        ratesS = ratesS*efficiency
        time1 = time.time()
        print('... signal rate took {} s'.format(time1-time0))
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

        rate_sumS, rate_sumB = np.sum(ratesS), np.sum(ratesB)
        probsS = np.array(ratesS)/rate_sumS
        probsB = np.array(ratesB)/rate_sumB
        probs = (S*probsS + B*probsB)/(S+B)

        print(np.min(probsS), np.min(probsB), np.min(probs), rate_sumS, rate_sumB)

        print('\nGenerating data')
        time4 = time.time()
        #KE = np.random.choice(Koptions, np.random.poisson(S+B), p = probs)
        KE = np.random.choice(Koptions, round(S+B), p = probs)
        time5 = time.time()
        print('... took {} s'.format(time5-time4))
        print('Number of values in array that are not unique:', np.size(KE) - len(set(KE)), 'out of', np.size(KE))

        return KE


#Physical constants
me = 510999. #eV
alpha = 1/137.036
c = 299792458. #m/s
hbar = 6.58212*10**(-16) #eV*s
gv=1. #Vector coupling constant
lambdat = 1.2724 # +/- 0.0023, from PDG (2018): http://pdg.lbl.gov/2018/listings/rpp2018-list-n.pdf
ga=gv*(-lambdat) #Axial vector coupling constant
Mnuc2 = gv**2 + 3*ga**2 #Nuclear matrix element
GF =  1.1663787*10**(-23) #Gf/(hc)^3, in eV^(-2)
Vud = 0.97425 #CKM element

#Beta decay-specific physical constants
QT = 18563.251 #For atomic tritium (eV), from Bodine et al. (2015)
QT2 =  18573.24 #For molecular tritium (eV), Bodine et al. (2015)
Rn =  2.8840*10**(-3) #Helium-3 nuclear radius in units of me, from Kleesiek et al. (2018): https://arxiv.org/pdf/1806.00369.pdf
M = 5497.885 #Helium-3 mass in units of me, Kleesiek et al. (2018)
atomic_num = 2. #For helium-3
g =(1-(atomic_num*alpha)**2)**0.5 #Constant to be used in screening factor and Fermi function calculations
V0 = 76. #Nuclear screening potential of orbital electron cloud of the daughter atom, from Kleesiek et al. (2018)
mu = 5.107 #Difference between magnetic moments of helion and triton, for recoil effects correction

"""
Functions to calculate kinematic electron properties
"""
def Ee(K):
    return K+me

def pe(K):
    return np.sqrt(K**2 + 2*K*me)

def beta(K):
    return pe(K)/Ee(K)

"""
Corrections to the simple spectrum
"""
#Radiative corretion to the Fermi function (from interactions with real and virtual photons)


def rad_corr(K, Q):
    t = 1./beta(K)*np.arctanh(beta(K))-1
    G = (Q-K)**(2*alpha*t/np.pi)*(1.+2*alpha/np.pi*(t*(np.log(2)-3./2.+(Q-K)/Ee(K))+0.25*(t+1.)*(2.*(1+beta(K)**2)+2*np.log(1-beta(K))+(Q-K)**2/(6*Ee(K)**2))-2+beta(K)/2.-17./36.*beta(K)**2+5./6.*beta(K)**3))
    return G

#Correction for screening by the Coulomb field of the daughter nucleus
def screen_corr(K):
    eta = alpha*atomic_num/beta(K)
    Escreen = Ee(K)-V0
    pscreen = np.sqrt(Escreen**2-me**2)
    etascreen = alpha*atomic_num*Escreen/pscreen
    S = Escreen/Ee(K)*(pscreen/pe(K))**(-1+2*g)*np.exp(np.pi*(etascreen-eta))*(np.absolute(gamma(g+etascreen*1j)))**2/(np.absolute(gamma(g+eta*1j)))**2
    return S

#Correction for exchange with the orbital 1s electron
def exchange_corr(K):
    tau = -2.*alpha/pe(K)
    a = np.exp(2.*tau*np.arctan(-2./tau))*(tau**2/(1+tau**2/4))**2
    I = 1 + 729./256.*a**2 + 27./16.*a
    return I

#Recoil effects (weak magnetism, V-A interference)
def recoil_corr(K, Q):
    A = 2.*(5.*lambdat**2 + lambdat*mu + 1)/M
    B = 2.*lambdat*(mu + lambdat)/M
    C = 1. + 3*lambdat**2 - ((Q+me)/me)*B
    R = 1 + (A*Ee(K)/me - B/(Ee(K)/me))/C
    return R


#Correction for scaling of Coulomb field within daughter nucleus (effect of finite nuclear size) (L)
#Correction for interference of electron/neutrino wavefuction with nucleonic wave function in the nuclear volume (C)
def finite_nuc_corr(K, Q):
    L = 1 + 13./60.*(alpha*atomic_num)**2 - Ee(K)/me*Rn*alpha*atomic_num*(41.-26.*g)/(15.*(2.*g-1.)) - (alpha*atomic_num*Rn*g/(30.*Ee(K)/me))*((17.-2*g)/(2.*g-1.))
    C0 = -233./630.*(alpha*atomic_num)**2 - 1./5.*((Q+me)/me)**2*Rn**2 + 2./35.*(Q+me)/me*Rn*alpha*atomic_num
    C1 = -21./35.*Rn*alpha*atomic_num + 4./9.*(Q+me)/me*Rn**2
    C2 = -4./9.*Rn**2
    C = 1. + C0 + C1*Ee(K)/me + C2*(Ee(K)/me)**2
    return L*C


#Correction for recoiling charge distribution of emitted electron
def coul_corr(K, Q):
    X = 1 - np.pi*alpha*atomic_num/(M*pe(K))*(1.+(1-lambdat**2)/(1+3*lambdat**2)*(Q-K)/(3*Ee(K)))
    return X


#Uncorrected Fermi function for a relativistic electron
def fermi_func(K):
    eta = alpha*atomic_num/beta(K)
    F = 4/(2*pe(K)*Rn)**(2*(1-g))*(np.absolute(gamma(g+eta*1j)))**2/((gamma(2*g+1))**2)*np.exp(np.pi*eta)
    return F

"""
Electron phase space definition
"""
def ephasespace(K, Q):
    G = rad_corr(K, Q)         #Radiative correction
    S = screen_corr(K)         #Screening factor
    I = exchange_corr(K)       #Exchange correction
    R = recoil_corr(K, Q)      #Recoil effects
    LC = finite_nuc_corr(K, Q) #Finite nucleus corrections
    X = coul_corr(K, Q)        #Recoiling Coulomb field correction
    F = fermi_func(K)          #Uncorrected Fermi function
    return pe(K)*Ee(K)*F*G*S*I*R*LC*X

"""
Tritium beta spectrum definition
"""
#Unsmeared beta spectrum
def spectral_rate(K, Q, mnu):
    if K < Q-mnu:
        return GF**2.*Vud**2*Mnuc2/(2.*np.pi**3)*ephasespace(K, Q)*(Q - K)*np.sqrt((Q - K)**2 - (mnu)**2)
    else:
        return 0.

#Beta spectrum with a lower energy bound Kmin
def spectral_rate_in_window(K, Q, mnu, Kmin):
    if Q-mnu > K > Kmin:
        return GF**2.*Vud**2*Mnuc2/(2.*np.pi**3)*ephasespace(K, Q)*(Q - K)*np.sqrt((Q - K)**2 - (mnu)**2)
    else:
        return 0.

#Beta spectrum without a lower energy bound
def spectral_rate(K, Q, mnu):
    if Q-mnu > K > 0:
        return GF**2.*Vud**2*Mnuc2/(2.*np.pi**3)*ephasespace(K, Q)*(Q - K)*np.sqrt((Q - K)**2 - (mnu)**2)
    else:
            return 0.
    #np.heaviside(Q-mnu-K, 0.5)*np.heaviside(K-V0, 0.5)


#Flat background with lower and upper bounds Kmin and Kmax
def bkgd_rate_in_window(K, Kmin, Kmax):
    if Kmax > K > Kmin:
        return 1.
    else:
        return 0.


#Flat background
def bkgd_rate():
    return 1.

#Lineshape option: In this case, simply a Gaussian
def gaussian(x,a):
    return 1/((2.*np.pi)**0.5*a[0])*(np.exp(-0.5*((x-a[1])/a[0])**2))


#Normalized simplified linesahape with scattering
def simplified_ls(K, Kcenter, FWHM, prob, p0, p1, p2, p3):
    sig0 = FWHM/float(2*np.sqrt(2*np.log(2)))
    shape = gaussian(K, [sig0, Kcenter])
    norm = 1.
    for i in range(len(p0)):
        sig = p0[i]+p1[i]*FWHM
        mu = -(p2[i]+p3[i]*np.log(FWHM-30))
        probi = prob**(i+1)
        shape += probi*gaussian(K, [sig, mu+Kcenter])
        norm += probi
    return shape/norm


#Integrand of the signal (spectrum) gaussian convolution
def _S_integrand_gaussian(Kp, K, Q, mnu, broadening):
    return spectral_rate(Kp, Q, mnu)*gaussian(K-Kp, [broadening, 0])

#Integrand of the background gaussian convolution
def _B_integrand_gaussian(Kp, K, broadening):
    return bkgd_rate()*gaussian(K-Kp, [broadening, 0])


#Integrand of the spectrum convolution with a simplified lineshape
def _S_integrand_simp(Kp, K, Q, mnu, ls_params):
    return spectral_rate(Kp, Q, mnu)*simplified_ls(K-Kp, 0, ls_params[0], ls_params[1], ls_params[2], ls_params[3], ls_params[4], ls_params[5])

#Integrand of the background convolution with a simplified lineshape
def _B_integrand_simp(Kp, K, ls_params):
    return bkgd_rate()*simplified_ls(K-Kp, 0, ls_params[0], ls_params[1], ls_params[2], ls_params[3], ls_params[4], ls_params[5])



#Convolution of signal and lineshape by integrating directly
#No detailed lineshape option
def convolved_spectral_rate(K, Q, mnu, Kmin,
                            lineshape, ls_params, min_energy, max_energy):
    #Integral bounds chosen to approximately minimize computation error
    pts = []
    lbound = K+min_energy
    ubound = K+max_energy
    if lbound<Q-mnu<ubound:
        pts.append(Q-mnu)
    if K>Kmin:
        if lineshape=='gaussian':
            return integrate.quad(_S_integrand_gaussian, lbound, ubound, args=(K, Q, mnu, ls_params[0]), epsrel=1e-2, epsabs=0, limit=150, points=pts)[0]
        elif lineshape=='simplified_scattering' or lineshape=='simplified':
            return integrate.quad(_S_integrand_simp, lbound, ubound, args=(K, Q, mnu, ls_params), epsrel=1e-2, epsabs=0, limit=150, points=pts)[0]
    else:
        return 0.


#Convolution of background and lineshape by integrating directly
#No detailed lineshape option
#Currently, the whole background is smeared. In reality, only background from electons should be smeared; false events from noise fluctuations produce an adititional unsmeared background.
def convolved_bkgd_rate(K, Kmin, Kmax, lineshape, ls_params, min_energy, max_energy):
    #Integral bounds chosen to approximately minimize computation error
    pts = []
    lbound = K+min_energy
    ubound = K+max_energy
    if K>Kmin:
        if lineshape=='gaussian':
            return integrate.quad(_B_integrand_gaussian, lbound, ubound, args=(K, ls_params[0]), epsrel=1e-3, epsabs=0, limit=150, points=pts)[0]
        elif lineshape=='simplified_scattering' or lineshape=='simplified':
            return integrate.quad(_B_integrand_simp, lbound, ubound, args=(K, ls_params), epsrel=1e-3, epsabs=0, limit=150, points=pts)[0]
    else:
        return 0.


#Convolution of signal and lineshape using scipy.signal.convolve
def convolved_spectral_rate_arrays(K, Q, mnu, Kmin,
                                   lineshape, ls_params, min_energy, max_energy):
    """K is an array-like object
    """
    print('Going to use scipy convolve')
    energy_half_range = max(max_energy, abs(min_energy))

    dE = K[1] - K[0]
    n_dE_pos = round(energy_half_range/dE) #Number of steps for the lineshape for energies > 0
    n_dE_neg = round(energy_half_range/dE) #Same, for energies < 0
    K_lineshape =  np.arange(-n_dE_neg*dE, n_dE_pos*dE, dE)

    #Generating finely spaced points on the lineshape
    if lineshape=='gaussian':
        print('broadening', ls_params[0])
        lineshape_rates = gaussian(K_lineshape, [ls_params[0], 0])
    elif lineshape=='simplified_scattering' or lineshape=='simplified':
        lineshape_rates = simplified_ls(K_lineshape, 0, ls_params[0], ls_params[1], ls_params[2], ls_params[3], ls_params[4], ls_params[5])
    elif lineshape=='detailed_scattering' or lineshape=='detailed':
        lineshape_rates = detailed.spectrum_func(K_lineshape/1000., ls_params[0], 0, ls_params[1], 1)

    beta_rates = np.zeros(len(K))
    for i in range(len(K)):
        beta_rates[i] = spectral_rate(K[i], Q, mnu)

    #Convolving
    convolved = convolve(beta_rates, lineshape_rates, mode='same')
    below_Kmin = np.where(K < Kmin)
    np.put(convolved, below_Kmin, np.zeros(len(below_Kmin)))
    return convolved



#Convolution of background and lineshape using scipy.signal.convolve
def convolved_bkgd_rate_arrays(K, Kmin, Kmax, lineshape, ls_params, min_energy, max_energy):
    """K is an array-like object
    """
    energy_half_range = max(max_energy, abs(min_energy))

    dE = K[1] - K[0]
    n_dE_pos = round(energy_half_range/dE) #Number of steps for the lineshape for energies > 0
    n_dE_neg = round(energy_half_range/dE) #Same, for energies < 0
    K_lineshape =  np.arange(-n_dE_neg*dE, n_dE_pos*dE, dE)

    #Generating finely spaced points on the lineshape
    if lineshape=='gaussian':
        lineshape_rates = gaussian(K_lineshape, [ls_params[0], 0])
    elif lineshape=='simplified_scattering' or lineshape=='simplified':
        lineshape_rates = simplified_ls(K_lineshape, 0, ls_params[0], ls_params[1], ls_params[2], ls_params[3], ls_params[4], ls_params[5])
    elif lineshape=='detailed_scattering' or lineshape=='detailed':
        lineshape_rates = detailed.spectrum_func(K_lineshape/1000., ls_params[0], 0, ls_params[1], 1)

    bkgd_rates = np.full(len(K), bkgd_rate())
    if len(K) < len(K_lineshape):
        raise Exception("lineshape array is longer than Koptions")

    #Convolving
    convolved = convolve(bkgd_rates, lineshape_rates, mode='same')
    below_Kmin = np.where(K < Kmin)
    np.put(convolved, below_Kmin, np.zeros(len(below_Kmin)))
    return convolved



"""
Functions to calculate number of events to generate
"""
#Fraction of events near the endpoint
#Currently, this only holds for the last 13.6 eV of the spectrum
def frac_near_endpt(Kmin, Q, mass, atom_or_mol='atom'):
    A = integrate.quad(spectral_rate, Kmin, Q-mass, args=(Q,mass))
    B = integrate.quad(spectral_rate, V0, Q-mass, args=(Q,mass)) #Minimum at V0 because electrons with energy below screening barrier do not escape
    f = (A[0])/(B[0])
    if atom_or_mol=='atom':
        return 0.7006*f
    elif atom_or_mol=='mol' or atom_or_mol=='molecule':
        return 0.57412*f
    else:
        print("Choose 'atom' or 'mol'.")


#Convert [number of particles]=(density*volume*efficiency) to a signal activity A_s, measured in events/second.
def find_signal_activity(Nparticles, m, Q, Kmin, atom_or_mol='atom', nTperMolecule=2):
    br = frac_near_endpt(Kmin, Q, m, atom_or_mol)
    Thalflife = 3.8789*10**8
    A_s = Nparticles*np.log(2)/(Thalflife)*br
    if atom_or_mol=='atom':
        return A_s
    elif atom_or_mol=='mol' or atom_or_mol=='molecule':
        #Multiply molecular activity by the average number of tritium atoms per molecule.
        #For example: A gas of half HT and half T2 would have nTperMolecule==1.5.
        return nTperMolecule*A_s

"""
Function to calculate efficiency
"""

def Frequency(E, B):
    emass = constants.electron_mass/constants.e*constants.c**2
    gamma = E/(emass)+1
    return (constants.e*B)/(2.0*np.pi*constants.electron_mass) * 1/gamma

def efficiency_from_interpolation(x, efficiency_dict, B=0.9578186017836624):
    f = Frequency(x, B)

    interp_efficiency = interp1d(efficiency_dict['frequencies'], efficiency_dict['eff interp with slope correction'], fill_value='0', bounds_error=False)
    interp_error = interp1d(efficiency_dict['frequencies'], np.mean(efficiency_dict['error interp with slope correction'], axis=0), fill_value=1, bounds_error=False)
    return interp_efficiency(f), interp_error(f)