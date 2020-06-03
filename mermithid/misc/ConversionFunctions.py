'''
Miscellaneous functions for CRES conversions
Author: C. Claessens
Date:4/19/2020
'''

from __future__ import absolute_import

import numpy as np

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

from mermithid.misc.Constants import *

def Frequency(E, B):
    """
    Conversion function from energy to frequency
    """
    emass = m_electron()*e()/(c()**2)#constants.electron_mass/constants.e*constants.c**2
    gamma = E/(m_electron())+1
    return (e()*B)/(2.0*np.pi*emass) * 1/gamma


def Energy(F, B=1):
    """
    Conversion function from frequency to energy
    """
    emass_kg = m_electron()*e()/(c()**2)
    if isinstance(F, list):
        gamma = [(e()*B)/(2.0*np.pi*emass_kg) * 1/(f) for f in F]
        return [(g-1)*m_electron() for g in gamma]
    else:
        gamma = (e()*B)/(2.0*np.pi*emass_kg) * 1/(F)
        return (gamma-1)*m_electron()

# Given energy in keV and the B_field of the trap, returns frequency in Hz
def energy_to_frequency(energy_vec, B_field):
    freq_vec = (e()*B_field/((2.*np.pi*m_e())*(1+energy_vec/(m_electron()/1000))))
    return freq_vec

# Given frequency in Hz and the B_field of the trap, returns energy in keV
def frequency_to_energy(freq_vec,B_field):
    energy_vec = (e()*B_field/((2.*np.pi*m_e()*freq_vec))-1)*m_electron()/1000
    return energy_vec