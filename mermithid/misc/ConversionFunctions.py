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
    emass_kg = Constants.m_electron()*Constants.e()/(Constants.c()**2)
    if isinstance(F, list):
        gamma = [(Constants.e()*B)/(2.0*TMath.Pi()*emass_kg) * 1/(f) for f in F]
        return [(g -1)*Constants.m_electron() for g in gamma]
    else:
        gamma = (Constants.e()*B)/(2.0*TMath.Pi()*emass_kg) * 1/(F)
        return (gamma -1)*Constants.m_electron()