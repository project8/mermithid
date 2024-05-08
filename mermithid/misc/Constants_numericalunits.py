'''
Some constants useful for various things...
The constants here use the numericalunits package. For constants not using this package import form Constants.py
'''

import numpy as np

from numericalunits import e, me, c0, eps0, kB, hbar
from numericalunits import meV, eV, keV, MeV, cm, m, ns, s, Hz, kHz, MHz, GHz, amu, nJ
from numericalunits import nT, uT, mT, T, mK, K,  C, F, g, W
from numericalunits import hour, year, day, s, ms
from numericalunits import mu0, NA, kB, hbar, me, c0, e, eps0, hPlanck


T0 = -273.15*K

tritium_livetime = 5.605e8*s
tritium_mass_atomic = 3.016* amu *c0**2
tritium_electron_crosssection_atomic = 9.e-23*m**2 #Hamish extrapolated to 18.6keV using Shah et al. (1987): https://iopscience.iop.org/article/10.1088/0022-3700/20/14/022
tritium_endpoint_atomic = 18563.251*eV
last_1ev_fraction_atomic = 2.067914e-13/eV**3

tritium_mass_molecular = 6.032099 * amu *c0**2
tritium_electron_crosssection_molecular = 3.67*1e-22*m**2 #[Inelastic from Aseev (2000) for T2] + [Elastic from Liu (1987) for H2, extrapolated by Elise to 18.6keV]
tritium_endpoint_molecular = 18574.01*eV
last_1ev_fraction_molecular = 1.67364e-13/eV**3

ground_state_width = 0.436 * eV
ground_state_width_uncertainty = 0.001*0.436*eV

gyro_mag_ratio_proton = 42.577*MHz/T

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