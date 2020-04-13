
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
import os

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


##Lineshape option: In this case, simply a Gaussian
#def gaussian(x,a):
#    return 1/((2.*np.pi)**0.5*a[0])*(np.exp(-0.5*((x-a[1])/a[0])**2))


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
        detailed = DetailedLineshape()
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
        detailed = DetailedLineshape()
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


##############################################
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
kr_line = 17.8260 # keV
kr_line_width = 2.83 # eV
e_charge = 1.60217662*10**(-19) # Coulombs , charge of electron
m_e = 9.10938356*10**(-31) # Kilograms , mass of electron
mass_energy_electron = 510.9989461 # keV
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
def get_eloss_spec(e_loss, oscillator_strength, kinetic_en = kr_line * 1000): #energies in eV
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

    input_filename = "../data/" + gas_type + "OscillatorStrength.txt"
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
def check_existence_of_scatter_files(gas_type):
    current_path = get_current_path()
    current_dir = get_current_dir()
    stuff_in_dir = list_files(current_path)
    if 'scatter_spectra_files' not in stuff_in_dir and current_dir != 'scatter_spectra_files':
        print('Scatter files not found, generating')
        os.popen("mkdir scatter_spectra_files")
        time.sleep(2)
        generate_scatter_convolution_files(gas_type)
    else:
        directory = os.popen("ls scatter_spectra_files").readlines()
        strippeddirs = [s.strip('\n') for s in directory]
        if len(directory) != len(gases) * max_scatters:
            generate_scatter_convolution_files(gas_type)
        test_file = 'scatter_spectra_files/scatter'+gas_type+'_01.npy'
        test_arr = np.load(test_file)
        if len(test_arr) != num_points_in_std_array:
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
        current_working_spectrum = normalize(sp.signal.convolve(zeroth_order_peak,current_working_spectrum,mode='same'))
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


class DetailedLineshape():
    def __init__(self):
        print('Using detailed lineshape')

    def spectrum_func(self, x_keV, *p0):
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

