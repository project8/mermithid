'''
Bin tritium start frequencies or energies.

Author: T. Weiss, derived from TritiumAndEfficiencyBinner by A. Ziegler, E. Novitski, C. Claessens

This processor takes tritium data (frequency or energy) and bins it.
Unlike TritiumAndEfficiencyBinner, it does not read, interpolate, or
output any efficiency information.

(Really, this could work for a non-tritium spectrum. Should we rename the processor?)
'''

from __future__ import absolute_import

import numpy as np

from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor

logger = morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)


class TritiumBinner(BaseProcessor):
    '''
    Processor that takes in tritium data and outputs a binned histogram
    of event counts.  No efficiency information is used or produced.

    Args:
        energy_or_frequency (str): Whether the input data are energies or
            frequencies.  Accepted values: 'energy', 'frequency'.
            Default: 'energy'.
        variables (str): Name of the column in *data* that holds the
            energy / frequency values (e.g. 'KE' or 'F').
            **Required.**
        bins (array-like): Bin edges to use for the histogram.  Must be
            provided unless *fss_bins* is True.
        asInteger (bool): If True, cast bin counts to integers.
            Default: False.

    Inputs:
        data (dict): Dictionary containing the unbinned tritium data.

    Output:
        results (dict): Dictionary with keys

            * 'KE' or 'F' — bin centres (energy or frequency),
            * 'N'         — event counts per bin.
    '''

    def InternalConfigure(self, params):
        '''Configure the processor from *params*.'''

        # Required: name of the data column to histogram
        self.namedata = reader.read_param(params, 'variables', 'required')

        # Optional parameters
        self.energy_or_frequency = reader.read_param(
            params, 'energy_or_frequency', 'energy')
        self.bins    = reader.read_param(params, 'bins', [])
        self.asInteger = reader.read_param(params, 'asInteger', False)

        # Determine the output bin-centre variable name
        if self.energy_or_frequency == 'energy':
            self.output_bin_variable = 'KE'
        elif self.energy_or_frequency == 'frequency':
            self.output_bin_variable = 'F'
        else:
            logger.error(
                "energy_or_frequency must be 'energy' or 'frequency', "
                "got '{}'".format(self.energy_or_frequency))
            return False

        # Validate that bins were supplied
        if len(self.bins) < 2:
            logger.error(
                "'bins' must contain at least two edges; "
                "got {}".format(self.bins))
            return False

        self.bins = np.asarray(self.bins)
        self.bin_centers = self.bins[:-1] + 0.5 * (self.bins[1] - self.bins[0])

        return True

    def InternalRun(self):
        '''Bin the data and store results.'''

        logger.info('namedata: {}'.format(self.namedata))

        N, _ = np.histogram(self.data[self.namedata], self.bins)

        if self.asInteger:
            N = N.astype(int)

        self.results = {
            self.output_bin_variable: self.bin_centers,
            'N': N,
        }

        return True
