'''
Processor for converting frequencies into energies
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

class FrequencyEnergyConversionProcessor(BaseProcessor):
    '''
    Convert frequency data to energy data given a magnetic field
    '''

    def get_kinetic_energy(self, frequency_Hz, B_tesla):
        freq_c = self.omega_c/2.0/TMath.Pi()
        gamma = freq_c / frequency_Hz * B_tesla;
        return (gamma -1.) * self.m_electron;

    def InternalConfigure(self, params):
        """
        Args:
            frequency_data: An list of frequencies to be converted to
                energies, in Hz
            B: Magnetic field used to convert frequency to energy in T
            m_electron: Electron mass in eV (Default=510998.910)
            omega_c: (Default=1.758820088e+11)
        Input:
            frequencies: list of frequencies to be converted
        Results:
            energies: list of the energies converted from frequencies in Hz
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
        for freq in self.frequencies:
            self.energies.append(self.get_kinetic_energy(freq, self.B))
        return True
