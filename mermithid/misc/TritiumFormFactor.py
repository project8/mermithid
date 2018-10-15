from .Constants import fine_structure_constant, m_electron
try:
    from ROOT import TMath
except ImportError:
    pass

def DiracFermiFunction(E, Z, m):
    beta = TMath.Sqrt(2 * E / m)
    eta = fine_structure_constant() * (Z + 1) / beta
    return 2 * TMath.Pi() * eta / (1 - TMath.Exp(-2 * TMath.Pi() * eta))

def RFactor(E, Z):
    # Form factor to use to define the spectrum shape
    # Use this for defining the Kurie plot
    return DiracFermiFunction(E, Z, m_electron()) * TMath.Sqrt(2 * m_electron() * E) * (E + m_electron());

def TritiumSpectrumShape(E, E0, Z, mbeta):
    return RFactor(E, Z) * (E0 - E) * TMath.Sqrt(TMath.Power(E0 - E, 2) - TMath.Power(mbeta, 2));
