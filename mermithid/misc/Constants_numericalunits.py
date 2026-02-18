'''
Some constants useful for various things...
The constants here use the numericalunits package. For constants not using this package import form Constants.py
'''

import numpy as np

from numericalunits import e, me, c0, eps0, mu0, kB, hbar, Rgas, NA, hPlanck    # Constants
from numericalunits import meV, eV, keV, MeV, nJ, J, mm, cm, m                  # Energy, Length
from numericalunits import nT, uT, mT, T, mK, K, F, W                           # Magnetic field, Temp, Power
from numericalunits import hour, year, day, s, ms, ns, Hz, kHz, MHz, GHz        # Time, Frequency
from numericalunits import kg, g, amu, mol                                      # Mass
from numericalunits import C, A, mA, uA, V, mV, nV, kV                          # Charge, Amps, Voltage
from numericalunits import Pa, bar, mbar, atm, torr, mtorr, L, mL               # Pressure, Volume

T0 = -273.15*K
tritium_livetime = 5.605e8*s
tritium_mass_atomic = 3.016* amu *c0**2
tritium_electron_crosssection_atomic = 9.e-23*m**2 #Inelastic cross-section Hamish extrapolated to 18.6keV using Shah et al. (1987): https://iopscience.iop.org/article/10.1088/0022-3700/20/14/022. Full Bethe formula must be exactly right (down to magnetic and QED corrections) for the hydrogen atom.
#tritium_electron_crosssection_atomic = 1.32e-22*m**2 #Inelastic cross-section + Elastic cross-section for T-e
tritium_tritium_crosssection_atomic = 4.40e-16*m**2 # T-T cross-section at T=0
tritium_endpoint_atomic = 18563.251*eV
last_1ev_fraction_atomic = 2.067914e-13/eV**3

tritium_mass_molecular = 6.032099 * amu *c0**2
tritium_electron_crosssection_molecular = 3.67*1e-22*m**2 #[Inelastic from Aseev (2000) for T2] + [Elastic from Liu (1987) for H2, extrapolated by Elise to 18.6keV]
tritium_endpoint_molecular = 18574.01*eV
last_1ev_fraction_molecular = 1.67364e-13/eV**3

ground_state_width = 0.436 * eV
ground_state_width_uncertainty = 0.001*0.436*eV

gyro_mag_ratio_proton = 42.577*MHz/T

# Atomic Calculator
# C008 - [m^3] Volume of 1mol of ideal gas at 1atm. Higher pressure --> smaller volume; higher temp --> lower volume
molar_volume = 2.24*e-2*m**3
# C009 - [m^2] H-He Cross-section at low temp but > 5K (Berlinsky)
H_He_crosssection = 2e-19*m**2
# C010 - [m^2] H-He Cross-section at low temps (Berlinsky)
H_He_crosssection_low_temp = 3e-20*m**2
# C012 - [m^2] Hard Spheres cross-section (289 pm kinetic diameter
H_H2_crosssection = 2.62e-19*m**2
# C019 - [eV/amu] Conversion
eV_amu = 931494100*eV/amu
# C020 - [kg/amu] Conversion
kg_amu = 1.66e-27*kg/amu
# C021 - Boltzmann Constant [eV/K] = kB * (1 eV / 1.61e-19 J)
kB_eV = 8.617e-5*eV/K
# C027 - [eV/T] Bohr Magneton
bohr_magneton = 5.776e-5*eV/T
# C028 - 1 Ci = 3.7e10 Bq
Ci_Bq = 3.7e10
# C031 - Ground state branch (atomic)
ground_state_branch_atomic = 0.702
# C032 - Ground state branch (molecular)
ground_state_branch_molecular = 0.570
# C036 - [eV] Recoil energy of tritium atom (Bodine)
atomic_tritium_recoil_energy = 3.409*eV
# C038 - [eV] Binding energy of tritium molecule (Bodine)
molecular_tritium_binding_energy = 4.59*eV
# C039 - [eV] Molecular final-state g.s. manifold standard deviation
molecular_final_state_manifold = 0.436*eV
# C040 - Constant for saturated T2 vapor: A,  Souers et al.
T2_vapor_A = 5.84605
# C041 - Constant for saturated T2 vapor: B
T2_vapor_B = -160.7
# C042 - Constant for saturated T2 vapor: B'
T2_vapor_B_prime = 2.3235
# C043 - [m/s^2] gravitational constant
gravity = 9.80 *m/s**2
# C048 - Multiplier for Lagendijk G^d rates: Ben Jones, Morgan Elliott CM presentation 10/24
LGd_rates = 50
# C049 - per beta decay: see: https://www.overleaf.com/2817746228snnghrnzfthk
molecules_desorbed_wall_beta = 1000
# C050 - eV to J conversion
eV_J = 1.605e-19 *J/eV

# units that do not show up in numericalunits
# missing pre-factors
fW = W*1e-15

# unitless units, relative fractions
pc = 0.01
ppm = 1e-6
ppb = 1e-9
ppt = 1e-12
ppq = 1e-15

# radian and degree which are also not really units
rad = 1
deg = np.pi/180
