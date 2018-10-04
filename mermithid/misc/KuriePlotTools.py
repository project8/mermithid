'''                                                                                                                                     
Generate a Kurie binning from energy data
Author: M. Guigue
Date: Sept 28 2018
'''

from __future__ import absolute_import

from ROOT import TMath

from morpho.utilities import morphologging
from mermithid.misc import TritiumFormFactor

logger = morphologging.getLogger(__name__)
__all__ = []
__all__.append(__name__)

def KuriePlotBinning(data, xRange = [0,-1], nBins=100):
    '''
    For a given list of data points, returns two lists:
    - one containing the bins contents
    - one containing the bins errors 
    '''
    xmin = xRange[0]
    xmax = xRange[1]
    if xmin>=xmax:
        xmin = min(data)
        xmax = max(data)
    
    lower_bound = [xmin + (xmax-xmin)*i/nBins for i in range(nBins)]
    upper_bound = [xmin + (xmax-xmin)*(i+1)/nBins for i in range(nBins)]
    central_value = [xmin + (xmax-xmin)*(i+0.5)/nBins for i in range(nBins)]
    binned_data = [0]*nBins
    error_bins = [0]*nBins
    # Data binning
    for value in data:
        for i in range(nBins):
            if lower_bound[i] <= value <= upper_bound[i]:
                binned_data[i] = binned_data[i] + 1
                continue
    for i in range(nBins):
        error_bins[i] = 0.5/TMath.Sqrt(TritiumFormFactor.RFactor(central_value[i],1))
        binned_data[i] = TMath.Sqrt(binned_data[i]/TritiumFormFactor.RFactor(central_value[i],1))
    return central_value, binned_data, error_bins

def KurieFunction(x, par):
    '''
    Kurie function: A(Q-E)+B
    par[0]: A
    par[1]: Q
    par[2]: B
    '''
    value = 0
    if x < par[1][0]:
        value = par[0]*(par[1]-x)
    return value + par[2]

def UnormalizedSquarePoisson(y,l):
    '''
    Return the probability density function for Y=sqrt(X)
    when X follows a Poisson distribution with param lambda
    '''
    return 2*y*TMath.Exp(-l)*TMath.Pow(l,y*y)/TMath.tgamma(y*y)
