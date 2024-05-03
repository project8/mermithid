'''
Miscellaneous functions for CRES conversions
Author: C. Claessens
Date: 05/02/2024
'''

from __future__ import absolute_import

import numpy as np

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

from mermithid.misc.Constants_numericalunits import *


def gamma(kin_energy):
    return kin_energy/(me*c0**2) + 1

def beta(kin_energy):
    # electron speed at kin_energy
    return np.sqrt(kin_energy**2+2*kin_energy*me*c0**2)/(kin_energy+me*c0**2)

def frequency(kin_energy, magnetic_field):
    # cyclotron frequency
    return e/(2*np.pi*me)/gamma(kin_energy)*magnetic_field

def wavelength(kin_energy, magnetic_field):
    return c0/frequency(kin_energy, magnetic_field)

def kin_energy(freq, magnetic_field):
    return (e*c0**2/(2*np.pi*freq)*magnetic_field - me*c0**2)

def rad_power(kin_energy, pitch, magnetic_field):
    # electron radiation power
    f = frequency(kin_energy, magnetic_field)
    b = beta(kin_energy)
    Pe = 2*np.pi*(e*f*b*np.sin(pitch/rad))**2/(3*eps0*c0*(1-b**2))
    return Pe

def track_length(rho, kin_energy=None, molecular=True):
    if kin_energy is None:
        kin_energy = tritium_endpoint_molecular if molecular else tritium_endpoint_atomic
    crosssect = tritium_electron_crosssection_molecular if molecular else tritium_electron_crosssection_atomic
    return 1 / (rho * crosssect * beta(kin_energy) * c0)

def sin2theta_sq_to_Ue4_sq(sin2theta_sq):
    return 0.5*(1-np.sqrt(1-sin2theta_sq**2))

def Ue4_sq_to_sin2theta_sq(Ue4_sq):
    return 4*Ue4_sq*(1-Ue4_sq)