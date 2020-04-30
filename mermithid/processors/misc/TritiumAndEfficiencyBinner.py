'''
Bin tritium start frequencies and calculate efficiency for each bin
function.
Author: A. Ziegler, E. Novitski, C. Claessens
Date:3/4/2020

This takes efficiency informations and interpolates between frequency points.
Then, you dump in tritium data.
It assigns an efficiency and efficiency uncertainty upper and lower bounds to each event.
It also bins events and defines an efficiency and efficiency uncertainty upper and
lower bound by integrating the interpolated efficiency over each bin.
'''

from __future__ import absolute_import

import numpy as np
import scipy
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
        self.bins = reader.read_param(params, 'bins', [])
        self.asInteger = reader.read_param(params, 'asInteger', False)
        self.energy_or_frequency = reader.read_param(params, 'energy_or_frequency', 'energy')
        self.efficiency_filepath = reader.read_param(params, 'efficiency_filepath', 'combined_energy_corrected_eff_at_quad_trap_frequencies.json')
        self.fss_bins = reader.read_param(params, "fss_bins", False)
        # If self.fss_bins is True, self.bins is ignored and overwritten

        # initialize the histogram to store the corrected data
        if self.energy_or_frequency == 'energy':
            print(sys.getrefcount(self.corrected_data))
            self.output_bin_variable='KE'
        elif self.energy_or_frequency == 'frequency':
            self.output_bin_variable='F'

        if self.fss_bins == True:
            a = self.GetEfficiencyFileContent()
            self.bin_centers = a['frequencies']
            self.bins = np.array(self.bin_centers) - (self.bin_centers[1]-self.bin_centers[0])/2
            self.bins = np.append(self.bins, [self.bin_centers[-1]+(self.bin_centers[1]-self.bin_centers[0])/2])
        else:
            self.bin_centers = self.bins[0:-1]+0.5*(self.bins[1]-self.bins[0])
        return True

    def InternalRun(self):
        logger.info('namedata: {}'.format(self.namedata))

        N,b = np.histogram(self.data[self.namedata], self.bins)

        # Efficiencies for bins
        # If we want to use fss bins, we want False to be passed to EfficiencyAssignment
        # because we just want to use the efficiency info directly from the file.
        self.bin_efficiencies, self.bin_efficiency_errors = self.EfficiencyAssignment(self.bin_centers, self.bins, not self.fss_bins)
        self.bin_efficiencies_normed = self.bin_efficiencies/np.sum(self.bin_efficiencies)
        self.bin_efficiency_errors_normed = self.bin_efficiency_errors/np.sum(self.bin_efficiencies)

        # Efficiencies for events, but with binned uncertainties (?)
        self.event_efficiencies, self.event_efficiency_errors = self.EfficiencyAssignment(self.data[self.namedata], None, False)
        self.event_efficiencies_normed = self.event_efficiencies/np.sum(self.event_efficiencies)
        self.event_efficiency_errors_normed = self.event_efficiency_errors/np.sum(self.event_efficiencies)

        # Put it all in a dictionary
        temp_dictionary = {self.output_bin_variable: [], 'N': [], 'bin_efficiencies': [], 'bin_efficiency_errors': [], 'event_efficiencies': [], 'event_efficiency_errors': []}
        temp_dictionary['N'] = N
        temp_dictionary[self.output_bin_variable] = self.bin_centers
        temp_dictionary['bin_efficiencies'] = self.bin_efficiencies_normed
        temp_dictionary['bin_efficiency_errors'] = self.bin_efficiency_errors_normed
        temp_dictionary['event_efficiencies'] = self.event_efficiencies_normed
        temp_dictionary['event_efficiency_errors'] = self.event_efficiency_errors_normed
        self.results = temp_dictionary

        return True

    def EfficiencyAssignment(self, f_bin_centers, f_bins = None, integrate_bin_width = False):
        a = self.GetEfficiencyFileContent()
        fss_frequencies = a['frequencies']
        fss_efficiencies = a['eff interp with slope correction']
        fss_efficiency_errors = a['error interp with slope correction']
        efficiency_interpolation = scipy.interpolate.interp1d(fss_frequencies, fss_efficiencies, bounds_error=False, fill_value=0)
        efficiency_error_interpolation_lower = scipy.interpolate.interp1d(fss_frequencies, fss_efficiency_errors[0], bounds_error=False, fill_value=1)
        efficiency_error_interpolation_upper = scipy.interpolate.interp1d(fss_frequencies, fss_efficiency_errors[1], bounds_error=False, fill_value=1)
        if integrate_bin_width == False:
            # FOI means frequency of interest
            logger.info('Not integrating efficiencies')
            FOI_efficiencies = efficiency_interpolation(f_bin_centers)
            FOI_efficiency_lower_errors = efficiency_error_interpolation_lower(f_bin_centers)
            FOI_efficiency_upper_errors = efficiency_error_interpolation_upper(f_bin_centers)
        else:
            logger.info('Integrating efficiencies')
            number_of_bins = len(f_bins)-1
            FOI_efficiencies = np.zeros(number_of_bins)
            FOI_efficiency_lower_errors = np.zeros(number_of_bins)
            FOI_efficiency_upper_errors = np.zeros(number_of_bins)
            for i in range(number_of_bins):
                FOI_efficiencies[i] = scipy.integrate.quad(efficiency_interpolation, f_bins[i], f_bins[i+1])[0]/(f_bins[i+1]-f_bins[i])
                FOI_efficiency_lower_errors[i] = scipy.integrate.quad(efficiency_error_interpolation_lower, f_bins[i], f_bins[i+1])[0]/(f_bins[i+1]-f_bins[i])
                FOI_efficiency_upper_errors[i] = scipy.integrate.quad(efficiency_error_interpolation_upper, f_bins[i], f_bins[i+1])[0]/(f_bins[i+1]-f_bins[i])
        # to do: figure out how to deal with the situation where frequencies have been slope/power corrected
        return FOI_efficiencies, [FOI_efficiency_lower_errors, FOI_efficiency_upper_errors]

    def GetEfficiencyFileContent(self):
        with open(self.efficiency_filepath, 'r') as infile:
            a = json.load(infile)
            if "frequencies" not in a.keys():
                logger.error("Missing Frequencies key")
                return False
        return a
