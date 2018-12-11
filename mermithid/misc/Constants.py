'''
Some constants useful for various things...
'''

def m_electron(): return  510998.910			# Electron mass in eV
def e(): return 1.6021766208e-19                # Electron charge
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