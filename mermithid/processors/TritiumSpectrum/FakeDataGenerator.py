'''
Generate binned or pseudo unbinned data
Author: T. Weiss, C. Claessens
Date:4/6/2020
'''

from __future__ import absolute_import

import numpy as np
#import scipy
#import sys
#import json
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.misc import Constants

logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)


class FakeDataGenerator(BaseProcessor):

    def InternalConfigure(self, params):
        '''
        Configure
        '''
        #Physical constants
        self.me = 510999. #eV
        self.alpha = 1/137.036
        self.c = 299792458. #m/s
        self.hbar = 6.58212*10**(-16) #eV*s
        self.gv=1. #Vector coupling constant
        self.lambdat = 1.2724 # +/- 0.0023, from PDG (2018): http://pdg.lbl.gov/2018/listings/rpp2018-list-n.pdf
        self.ga=self.gv*(-self.lambdat) #Axial vector coupling constant
        self.Mnuc2 = self.gv**2 + 3*self.ga**2 #Nuclear matrix element
        self.GF =  1.1663787*10**(-23) #Gf/(hc)^3, in eV^(-2)
        self.Vud = 0.97425 #CKM element

        #Beta decay-specific physical constants
        self.QT = 18563.251 #For atomic tritium (eV), from Bodine et al. (2015)
        self.QT2 =  18573.24 #For molecular tritium (eV), Bodine et al. (2015)
        self.Rn =  2.8840*10**(-3) #Helium-3 nuclear radius in units of me, from Kleesiek et al. (2018): https://arxiv.org/pdf/1806.00369.pdf
        self.M = 5497.885 #Helium-3 mass in units of me, Kleesiek et al. (2018)
        self.atomic_num = 2. #For helium-3
        self.g =(1-(self.atomic_num*self.alpha)**2)**0.5 #Constant to be used in screening factor and Fermi function calculations
        self.V0 = 76. #Nuclear screening potential of orbital electron cloud of the daughter atom, from Kleesiek et al. (2018)
        self.mu = 5.107 #Difference between magnetic moments of helion and triton, for recoil effects correction

        # Read other parameters
        self.namedata = reader.read_param(params, 'variables', "required")

        self.Q = reader.read_param(params, 'Q', self.QT2) #Choose the atomic or molecular tritium endpoint
        self.m = reader.read_param(params, 'neutrino_mass', 0.0085) #Neutrino mass (eV)
        self.Kmin = reader.read_param(params, 'Kmin', self.Q-self.m-2300)  #Energy corresponding to lower bound of frequency ROI (eV)
        self.Kmax = reader.read_param(params, 'Kmax', self.Q-self.m+1000)   #Same, for upper bound (eV)
        self.bin_region_widths = reader.read_param(params, 'bin_region_width', [1., 9.]) #List of widths (eV) of regions where bin sizes do not change, from high to low energy
        self.nbins_per_region = reader.read_param(params, 'nbins_per_region', [300., 9.]) #List of number of bins in each of those regions, from high to low energy

        self.Nparticles = reader.read_param(params, 'Nparticles', 10**(19)) #Density*volume*efficiency
        self.runtime = reader.read_param(params, 'runtime', 31556952.) #In seconds
        #For Phase IV:
        #A_s = find_signal_activity(Nparticles, m, Q, Kmin) #Signal activity: events/s in ROI
        #S = A_s*runtime #Signal poisson rate
        self.S = reader.read_param(params, 'S', 2500)
        self.A_b = reader.read_param(params, 'A_b', 10**(-12)) #Flat background activity: events/s/eV
        self.B =self.A_b*self.runtime*(self.Kmax-self.Kmin) #Background poisson rate
        self.fb = self.B/(self.S+self.B) #Background fraction
        self.err_from_B = reader.read_param(params, 'err_from_B', 0.5) #In eV, kinetic energy error from f_c --> K conversion

        #For a Phase IV gaussian smearing:
        self.sig_trans = reader.read_param(params, 'sig_trans', 0.020856) #Thermal translational Doppler broadening for atomic T (eV)
        self.other_sig = reader.read_param(params, 'other_sig', 0.05) #Kinetic energy broadening from other sources (eV)
        self.broadening = np.sqrt(self.sig_trans**2+self.other_sig**2) #Total energy broadening (eV)


        #Simplified scattering model parameters
        self.scattering_prob = reader.read_param(params, 'scattering_prob', 0.77)
        self.scattering_sigma = reader.read_param(params, 'scattering_sigma', 18.6)
        return True

    def InternalRun(self):
        print('namedata:',self.namedata)

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