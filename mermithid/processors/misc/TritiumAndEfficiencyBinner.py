'''
Bin tritium start frequencies and calculate efficiency for each bin
function.
Author: A. Ziegler, E. Novitski, C. Claessens
Date:1/20/2020
'''

from __future__ import absolute_import

import numpy as np
import sys
from ROOT import TF1, TMath, TH1F
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from morpho.processors.plots import RootCanvas, RootHistogram
from mermithid.misc import Constants

logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

def Energy(f, B=None, Theta=None):
    #print(type(F))
    if B==None:
        B = 0.95777194923080811
    emass_kg = Constants.m_electron()*Constants.e()/(Constants.c()**2)
    if isinstance(f, list):
        gamma = [(Constants.e()*B)/(2.0*TMath.Pi()*emass_kg) * 1/(F) for F in f]
        return [(g -1)*Constants.m_electron() for g in gamma]
    else:
        gamma = (Constants.e()*B)/(2.0*TMath.Pi()*emass_kg) * 1/(f)
        return (gamma -1)*Constants.m_electron()

class TritiumAndEfficiencyBinner(BaseProcessor):

    def InternalConfigure(self, params):
        '''
        Configure
        '''
        self.rootcanvas = RootCanvas(params, optStat=0)

        # Read other parameters
        self.namedata = reader.read_param(params, 'variables', "required")
        self.N = reader.read_param(params, 'N', 'N')
        self.eff_eqn = reader.read_param(params, 'efficiency', '1')
        self.mode = reader.read_param(params, 'mode', 'unbinned')
        self.n_bins_x = reader.read_param(params, 'n_bins_x', 100)
        self.range = reader.read_param(params, 'range', [0., -1.])
        self.asInteger = reader.read_param(params, 'asInteger', False)
        self.energy_or_frequency = reader.read_param(params, 'energy_or_frequency', 'energy')
        self.total_counts = 0
        self.total_weighted_counts = 0
        self.eff_norm = 0

        # initialize the histogram to store the corrected data
        if self.energy_or_frequency == 'energy':
            print(sys.getrefcount(self.corrected_data))
            self.output_bin_variable='KE'
            self.bins = np.linspace(self.range[0], self.range[1], self.n_bins_x)
        elif self.energy_or_frequency == 'frequency':
            self.output_bin_variable='F'
            self.bins = np.linspace(self.range[0], self.range[1], self.n_bins_x)

        return True

    def InternalRun(self):
        print('namedata:',self.namedata)

        N,b = np.histogram(self.data[self.namedata], self.bins)
        self.bin_centers = self.bins[0:-1]+0.5*(self.bins[1]-self.bins[0])

        self.bin_efficiencies = np.zeros(self.n_bins_x-1)
        self.bin_efficiency_errors = np.zeros(self.n_bins_x-1)

        for i in range(self.n_bins_x-1):

            self.bin_efficiencies[i], self.bin_efficiency_errors[i] = self.EfficiencyAssignment(self.bin_centers[i])

        self.bin_efficiencies_normed = self.bin_efficiencies/np.sum(self.bin_efficiencies)
        self.bin_efficiency_errors_normed = self.bin_efficiency_errors/np.sum(self.bin_efficiencies)

        # put corrected data in a dictionary form if requested
        temp_dictionary = {self.output_bin_variable: [], 'N': [], 'bin_efficiencies': [], 'bin_efficiency_errors': []}
        temp_dictionary['N'] = N
        temp_dictionary[self.output_bin_variable] = self.bin_centers
        temp_dictionary['bin_efficiencies'] = self.bin_efficiencies_normed
        temp_dictionary['bin_efficiency_errors'] = self.bin_efficiency_errors_normed
        self.results = temp_dictionary
        #print(self.corrected_data.keys())

        return True

    def EfficiencyAssignment(self, f):
        efficiency = 0.9
        efficiency_error = 0.05
        return efficiency, efficiency_error
