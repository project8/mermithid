'''
Generate binned or pseudo unbinned data
Author: T. Weiss, C. Claessens, Y. Sun
Date:4/6/2020
'''

from __future__ import absolute_import

import numpy as np
import json

from scipy.special import gamma
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.signal import convolve

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

from mermithid.misc.Constants import *
from mermithid.misc.ConversionFunctions import *

import matplotlib.pyplot as plt

"""
Constants and functions used by processors/TritiumSpectrum/FakeDataGenerator.py
"""

"""
Physical constants
"""

me = m_electron() #510999. #eV
alpha = fine_structure_constant() #1/137.036
c = c() #299792458. #m/s
hbar = hbar() #6.58212*10**(-16) #eV*s
gv = gv() # 1. #Vector coupling constant
lambdat = lambdat() # 1.2724 # +/- 0.0023, from PDG (2018): http://pdg.lbl.gov/2018/listings/rpp2018-list-n.pdf
ga = ga() #gv*(-lambdat) #Axial vector coupling constant
Mnuc2 = Mnuc2() #gv**2 + 3*ga**2 #Nuclear matrix element
GF =  GF() #1.1663787*10**(-23) #Gf/(hc)^3, in eV^(-2)
Vud = Vud() #0.97425 #CKM element

#Beta decay-specific physical constants
QT = QT() #18563.251 #For atomic tritium (eV), from Bodine et al. (2015)
QT2 = QT2() # 18573.24 #For molecular tritium (eV), Bodine et al. (2015)
Rn =  Rn() #2.8840*10**(-3) #Helium-3 nuclear radius in units of me, from Kleesiek et al. (2018): https://arxiv.org/pdf/1806.00369.pdf
M = M_3He_in_me() #5497.885 #Helium-3 mass in units of me, Kleesiek et al. (2018)
atomic_num = atomic_num() #2. #For helium-3
g =(1-(atomic_num*alpha)**2)**0.5 #Constant to be used in screening factor and Fermi function calculations
V0 = V0() #76. #Nuclear screening potential of orbital electron cloud of the daughter atom, from Kleesiek et al. (2018)
mu = mu_diff_hel_trit() #5.107 #Difference between magnetic moments of helion and triton, for recoil effects correction


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

#Beta spectrum with a lower energy bound Kmin
def spectral_rate_in_window(K, Q, mnu, Kmin):
    if Q-mnu > K > Kmin:
        return GF**2.*Vud**2*Mnuc2/(2.*np.pi**3)*ephasespace(K, Q)*(Q - K)*np.sqrt((Q - K)**2 - (mnu)**2)
    else:
        return 0.


def beta_rates(K, Q, mnu, index):
    beta_rates = np.zeros(len(K))
    nu_mass_shape = ((Q - K[index])**2 -mnu**2)**0.5
    beta_rates[index] = GF**2.*Vud**2*Mnuc2/(2.*np.pi**3)*ephasespace(K[index], Q)*(Q - K[index])*nu_mass_shape
    return beta_rates



# Unsmeared eta spectrum without a lower energy bound
def spectral_rate(K, Q, mnu, final_state_array):

    if isinstance(K, list) or isinstance(K, np.ndarray):
        N_states = len(final_state_array[0])
        beta_rates_array = np.zeros([N_states, len(K)])

        Q_states = Q+final_state_array[0]-np.max(final_state_array[0])

        index = [np.where(K < Q_states[i]-mnu) for i in range(N_states)]

        beta_rates_array = [beta_rates(K, Q_states[i], mnu, index[i])*final_state_array[1][i] for i in range(N_states)]
        to_return = np.nansum(beta_rates_array, axis=0)/np.nansum(final_state_array[1])

        return to_return

    else:

        return_value = 0.

        for i, e_binding in enumerate(final_state_array[0]):
            # binding energies are negative
            Q_state = Q+e_binding
            if Q_state-mnu > K > 0:
                return_value += final_state_array[1][i] *(GF**2.*Vud**2*Mnuc2/(2.*np.pi**3)*ephasespace(K, Q_state)*(Q_state - K)*np.sqrt((Q_state - K)**2 - (mnu)**2))

        return return_value/np.sum(final_state_array[1])


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


#Normalized simplified lineshape with scattering
def simplified_ls(K, Kcenter, FWHM, prob, p0, p1, p2, p3):
    sig0 = FWHM/float(2*np.sqrt(2*np.log(2)))
    shape = gaussian(K, [sig0, Kcenter])
    norm = 1.
    for i,_ in enumerate(p0):
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
                                   lineshape, ls_params, scatter_peak_ratio_p, scatter_peak_ratio_q, scatter_fraction, min_energy, max_energy,
                                   complexLineShape, final_state_array, resolution_function, ins_res_width_bounds, ins_res_width_factors, p_factors, q_factors):
    """K is an array-like object
    """
    logger.info('Using scipy convolve')
    logger.info('Lineshape is {} with {}'.format(lineshape, resolution_function))
    energy_half_range = max(max_energy, abs(min_energy))

    #logger.info('Using {} frequency regions. Mean and std of p are {} and {}. For q its {} and {}'.format(len(ins_res_width_bounds)-1,
    #                                                                                                        np.mean(p_factors), np.std(p_factors),
    #                                                                                                        np.mean(q_factors), np.std(q_factors)))

    if ins_res_width_bounds != None:
        Kbounds = [np.min(K)] + ins_res_width_bounds + [np.max(K)]
    else:
        Kbounds = [np.min(K), np.max(K)]

    K_segments = []
    for i in range(len(Kbounds)-1):
        K_segments.append(K[np.logical_and(Kbounds[i]<=K, K<=Kbounds[i+1])])

    dE = K[1] - K[0]
    n_dE_pos = round(energy_half_range/dE) #Number of steps for the lineshape for energies > 0
    n_dE_neg = round(energy_half_range/dE) #Same, for energies < 0
    K_lineshape =  np.arange(-n_dE_neg*dE, n_dE_pos*dE, dE)

    #Generating finely spaced points on the lineshape
    if lineshape=='gaussian':
        logger.info('broadening: {}'.format(ls_params[0]))
        lineshape_rates = gaussian(K_lineshape, [ls_params[0], 0])
    elif lineshape=='simplified_scattering' or lineshape=='simplified':
        lineshape_rates = simplified_ls(K_lineshape, 0, ls_params[0], ls_params[1], ls_params[2], ls_params[3], ls_params[4], ls_params[5])
    elif lineshape=='detailed_scattering' or lineshape=='detailed':
        if resolution_function == 'simulated_resolution' or resolution_function == 'simulated':
            lineshape_rates = []
            scale_factors = [ls_params[0]*f for f in ins_res_width_factors]
            for i in range(len(scale_factors)):
                lineshape_rates.append(np.flipud(complexLineShape.make_spectrum_simulated_resolution_scaled_fit_scatter_peak_ratio(scale_factors[i], ls_params[1], scatter_peak_ratio_p*p_factors[i], scatter_peak_ratio_q*q_factors[i], scatter_fraction, emitted_peak='dirac')))
        elif resolution_function == 'gaussian_resolution' or resolution_function == 'gaussian':
            logger.warn("Scatter peak ratio function for lineshape with Gaussian resolution may not be up-to-date!")
            gaussian_widths = [ls_params[0]*f for f in ins_res_width_factors]
            lineshape_rates = [np.flipud(complexLineShape.make_spectrum_gaussian_resolution_fit_scatter_peak_ratio(gaussian_widths[i], ls_params[1], scatter_peak_ratio_p*p_factors[i], scatter_peak_ratio_q*q_factors[i], scatter_fraction, emitted_peak='dirac')) for i in range(len(gaussian_widths))]
        else:
            logger.warn('{} is not a resolution function that has been implemented in the FakeDataGenerator'.format(resolution_function))

    below_Kmin = np.where(K < Kmin)

    #Convolving
    if (lineshape=='detailed_scattering' or lineshape=='detailed'):# and (resolution_function == 'simulated_resolution' or resolution_function == 'simulated'):
        convolved_segments = []
        beta_rates = spectral_rate(K, Q, mnu, final_state_array)
        plt.figure(figsize=(7,5))
        for j in range(len(lineshape_rates)):
            plt.plot(lineshape_rates[j])
            #beta_rates = spectral_rate(K_segments[j], Q, mnu, final_state_array)
            plt.plot(lineshape_rates[j])
            convolved_j = convolve(beta_rates, lineshape_rates[j], mode='same')
            np.put(convolved_j, below_Kmin, np.zeros(len(below_Kmin)))
            #Only including the part of convolved_j that corresponds to the right values of K
            convolved_segments.append(convolved_j[np.logical_and(Kbounds[j]<=K, K<=Kbounds[j+1])])
            #convolved.append(convolved_j)
        convolved = np.concatenate(convolved_segments, axis=None)
        plt.savefig('varied_lineshapes.png', dpi=200)
    """elif resolution_function=='gaussian':
        lineshape_rates = np.flipud(lineshape_rates)
        beta_rates = spectral_rate(K, Q, mnu, final_state_array)
        convolved = convolve(beta_rates, lineshape_rates, mode='same')
        np.put(convolved, below_Kmin, np.zeros(len(below_Kmin)))"""

    if (lineshape=='gaussian' or lineshape=='simplified_scattering' or lineshape=='simplified'):
        beta_rates = spectral_rate(K, Q, mnu, final_state_array)
        convolved = convolve(beta_rates, lineshape_rates, mode='same')
        np.put(convolved, below_Kmin, np.zeros(len(below_Kmin)))

    return convolved



#Convolution of background and lineshape using scipy.signal.convolve
def convolved_bkgd_rate_arrays(K, Kmin, Kmax, lineshape, ls_params, scatter_peak_ratio_p, scatter_peak_ratio_q, scatter_fraction, min_energy, max_energy, complexLineShape, resolution_function):
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
        if resolution_function == 'simulated_resolution' or resolution_function == 'simulated':
            lineshape_rates = complexLineShape.make_spectrum_simulated_resolution_scaled_fit_scatter_peak_ratio(ls_params[0], ls_params[1], scatter_peak_ratio_p, scatter_peak_ratio_p, scatter_fraction, emitted_peak='dirac')
        elif resolution_function == 'gaussian_resolution' or resolution_function == 'gaussian':
            lineshape_rates = complexLineShape.make_spectrum_gaussian_resolution_fit_scatter_peak_ratio(ls_params[0], ls_params[1], scatter_peak_ratio_p, scatter_peak_ratio_q, scatter_fraction, emitted_peak='dirac')
        else:
            logger.warn('{} is not a resolution function that has been implemented in the FakeDataGenerator'.format(resolution_function))
        lineshape_rates = np.flipud(lineshape_rates)

    bkgd_rates = np.full(len(K), bkgd_rate())
    if len(K) < len(K_lineshape):
        raise Exception("lineshape array is longer than Koptions")

    #Convolving
    convolved = convolve(bkgd_rates, lineshape_rates, mode='same')
    below_Kmin = np.where(K < Kmin)
    np.put(convolved, below_Kmin, np.zeros(len(below_Kmin)))

    return convolved



#Fraction of events near the endpoint
def frac_near_endpt(Kmin, Q, mass, final_state_array, atom_or_mol='mol', range='wide'):
    """
    Options for range:
        - 'narrow': Only extends ~18 eV (or less) below the endpoint, so that all decays are to the ground state
        - 'wide': Wide enough that the probability of decay to a 3He electronic energy level that would shift Q below the ROI is very low
    """
    A = integrate.quad(spectral_rate, Kmin, Q-mass, args=(Q, mass, final_state_array))
    B = integrate.quad(spectral_rate, V0, Q-mass, args=(Q, mass, final_state_array)) #Minimum at V0 because electrons with energy below screening barrier do not escape
    f = (A[0])/(B[0])
    if range=='narrow':
        if atom_or_mol=='atom':
            return 0.7006*f
        elif atom_or_mol=='mol' or atom_or_mol=='molecule':
            return 0.57412*f
        else:
            logger.warn("Choose 'atom' or 'mol'.")
    elif range=='wide':
        return f
    else:
        logger.warn("Choose range 'narrow' or 'wide'")


#Convert [number of particles]=(density*volume*efficiency) to a signal activity A_s, measured in events/second.
def find_signal_activity(Nparticles, m, Q, Kmin, atom_or_mol='atom', nTperMolecule=2):
    """
    Functions to calculate number of events to generate
    """
    br = frac_near_endpt(Kmin, Q, m, final_state_array, atom_or_mol)
    Thalflife = 3.8789*10**8
    A_s = Nparticles*np.log(2)/(Thalflife)*br
    if atom_or_mol=='atom':
        return A_s
    elif atom_or_mol=='mol' or atom_or_mol=='molecule':
        #Multiply molecular activity by the average number of tritium atoms per molecule.
        #For example: A gas of half HT and half T2 would have nTperMolecule==1.5.
        return nTperMolecule*A_s



def efficiency_from_interpolation(x, efficiency_dict, B=0.9578186017836624):
    """
    Function to calculate efficiency
    """
    logger.info('Interpolating efficiencies')
    f = Frequency(x, B)

    interp_efficiency = interp1d(efficiency_dict['frequencies'], efficiency_dict['eff interp with slope correction'], fill_value='0', bounds_error=False)
    interp_error = interp1d(efficiency_dict['frequencies'], np.mean(efficiency_dict['error interp with slope correction'], axis=0), fill_value=1, bounds_error=False)
    return interp_efficiency(f), interp_error(f)



def random_efficiency_from_interpolation(x, efficiency_dict, B=0.9578186017836624):
    """
    Function to calculate efficiency
    """
    logger.info('Sampling efficiencies before interpolation')
    f = Frequency(x, B)

    efficiency_mean = efficiency_dict['eff interp with slope correction']
    efficiency_error = np.mean(efficiency_dict['error interp with slope correction'], axis=0)
    random_efficiencies = np.random.normal(efficiency_mean, efficiency_error)
    random_efficiencies[random_efficiencies<0] = 0.
    interp_efficiency = interp1d(efficiency_dict['frequencies'], random_efficiencies, fill_value='0', bounds_error=False)

    return interp_efficiency(f)