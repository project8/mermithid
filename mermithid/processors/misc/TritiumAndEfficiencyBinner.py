'''
Bin tritium start frequencies and calculate efficiency for each bin
function.
Author: A. Ziegler, E. Novitski, C. Claessens
Date:2/17/2020
'''

from __future__ import absolute_import

import numpy as np
import sys
import json
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
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

        # Read other parameters
        self.namedata = reader.read_param(params, 'variables', "required")
        self.N = reader.read_param(params, 'N', 'N')
        self.eff_eqn = reader.read_param(params, 'efficiency', '1')
        self.mode = reader.read_param(params, 'mode', 'unbinned')
        self.n_bins_x = reader.read_param(params, 'n_bins_x', 100)
        self.range = reader.read_param(params, 'range', [0., -1.])
        self.asInteger = reader.read_param(params, 'asInteger', False)
        self.energy_or_frequency = reader.read_param(params, 'energy_or_frequency', 'energy')
        self.efficiency_filepath = reader.read_param(params, 'efficiency_filepath', '/host/combined_energy_corrected_eff_at_quad_trap_frequencies.json')

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

        self.bin_efficiencies, self.bin_efficiency_errors = self.EfficiencyAssignment(self.bin_centers)

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
        with open(self.efficiency_filepath, 'r') as infile:
            a = json.load(infile)
        print(a.keys)
        # keys: ['frequencies_slope_cut', 'eff interp no slope correction', 'error interp no slope correction', 'frequencies', 'eff interp with slope correction', 'error interp with slope correction']
        fss_frequencies = a['frequencies']
        fss_efficiencies = a['eff interp with slope correction']
        fss_efficiency_errors = a['error interp with slope correction']
        bin_center_efficiencies = np.interp(f, fss_frequencies, fss_efficiencies, left=0, right=0)
        bin_center_efficiency_lower_errors = np.interp(f, fss_frequencies, fss_efficiency_errors[0], left=1, right=1)
        bin_center_efficiency_upper_errors = np.interp(f, fss_frequencies, fss_efficiency_errors[1], left=1, right=1)
        # to do: integrate over bin width instead of just taking value at bin bin_centers
        # to do: figure out how to deal with the situation where
        #   frequencies have been slope/power corrected
        return bin_center_efficiencies, [bin_center_efficiency_lower_errors, bin_center_efficiency_upper_errors]
