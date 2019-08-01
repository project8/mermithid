'''
Processor for converting energies to frequencies.
'''

from __future__ import absolute_import

try:
    from ROOT import TMath
except ImportError:
    pass

from morpho.processors import BaseProcessor
from morpho.utilities import morphologging, reader
logger=morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)

class Energy2FrequencyConversionProcessor(BaseProcessor):
    '''
    Convert frequency data to energy data given a magnetic field
    '''

    def get_frequency(self, energy_kev, B_tesla): # returns frequency in GHz
        freq_c = self.omega_c/(2.0*TMath.Pi())
        x = 1/(1 + (energy_kev*1000)/self.m_electron)
        f = freq_c * B_tesla * x
        return f/1000000000

    def InternalConfigure(self, params):
        """
        Args:
            energy_data: An list of energies to be converted to
                frequencies, in kev.
            B: Magnetic field used to convert frequency to energy in T
            m_electron: Electron mass in eV (Default=510998.910)
            omega_c: (Default=1.758820088e+11)
        Input:
            energies: list of frequencies to be converted
        Results:
            frequencies: list of the energies converted from frequencies in Hz
        """
        self.params = params
        self.B = reader.read_param(params, "B", "required")
        self.m_electron = reader.read_param(params, "m_electron", 510998.910)
        self.omega_c = reader.read_param(params, "omega_c", 1.758820088e+11)
        self.frequencies = list()
        self.energies = list()
        return True

    def InternalRun(self):
        """
        Convert the frequencies to energies

        Returns: Success of the execution
        """
        for energy in self.energies:
            self.frequencies.append(self.get_frequency(energy, self.B))
        return True
