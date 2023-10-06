'''
Some constants useful for various things...
'''

def m_electron(): return  510998.910			# Electron mass in eV
def e(): return 1.6021766208e-19			# Electron charge in C
def hbar(): return 6.582119514e-16               # Reduced Planck's constant in eV*s
def c() : return  299792458.			   	   	 	# Speed of light in m/s
def hbarc(): return hbar() * c()						# hbar * c
def omega_c(): return 1.758820088e+11		     # Angular gyromagnetic ratio in rad Hz/Tesla
def freq_c(): return omega_c()/(2. * ROOT.TMath.Pi())			 # Gyromagnetic ratio in Hz/Tesla
def k_boltzmann(): return  8.61733238e-5		         	# Boltzmann's constant in eV/Kelvin
def unit_mass() : return 931.494061e6			 	# Unit mass in eV
def seconds_per_year(): return 365.25 * 86400.

def fine_structure_constant(): return 0.0072973525664   # fine structure constant (no unit)

#  Tritium-specific constants

def tritium_rate_per_eV(): return 2.0e-13				 # fraction of rate in last 1 eV
def tritium_atomic_mass(): return  3.016 * unit_mass()   	 # Atomic tritium mass in eV
def tritium_halflife(): return 12.32 * seconds_per_year()  # Halflife of tritium (in seconds)
def tritium_lifetime(): return tritium_halflife()/0.69314718056  # Lifetime of tritium (in seconds)
def tritium_endpoint(): return 18.6E+3 # Tritium endpoint value


def gv(): return 1. #Vector coupling constant
def lambdat(): return 1.2724 # +/- 0.0023, from PDG (2018): http://pdg.lbl.gov/2018/listings/rpp2018-list-n.pdf
def ga(): return gv()*(-lambdat()) #Axial vector coupling constant
def Mnuc2(): return gv()**2 + 3*ga()**2 #Nuclear matrix element
def GF(): return 1.1663787*10**(-23) #Gf/(hc)^3, in eV^(-2)
def Vud(): return 0.97425 #CKM element

#Beta decay-specific physical constants
def QT(): return 18563.251 #For atomic tritium (eV), from Bodine et al. (2015)
def QT2(): return 18574.01 #For molecular tritium (eV), Bodine et al. (2015) and Meyers et al. Discussion: https://projecteight.slack.com/archives/CG5TY2UE7/p1649963449399179 
def Rn(): return 2.8840*10**(-3) #Helium-3 nuclear radius in units of me, from Kleesiek et al. (2018): https://arxiv.org/pdf/1806.00369.pdf
def M_3He_in_me(): return 5497.885 #Helium-3 mass in units of me, Kleesiek et al. (2018)
def atomic_num(): return 2. #For helium-3
def V0(): return 76. #Nuclear screening potential of orbital electron cloud of the daughter atom, from Kleesiek et al. (2018)
def mu_diff_hel_trit(): return 5.107 #Difference between magnetic moments of helion and triton, for recoil effects correction

# Kr specific constants
# Based on Katrin recent paper https://iopscience.iop.org/article/10.1088/1361-6471/ab8480
def kr_k_line_e(): return 17.8260*1e3
def kr_k_line_width(): return 2.774
