"""
Sample energy from a monoenergetic peak.
Author: S. M. Lee
First Date: September 02, 2025
Last Update: September 02, 2025
"""

from __future__ import absolute_import
from typing import List

import numpy as np

from morpho.utilities import morphologging

from .EnergySampler import EnergySampler

logger = morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)


class Monoenergetic(EnergySampler):
    """
    Sample energy from a monoenergetic peak.
    """

    def __init__(
        self,
        name,
        peak_energy: float = 18573.24 + 0.02,  # (eV)
        peak_rate: float = 1,  # (1/s)
        **kwargs,
    ):
        super(Monoenergetic, self).__init__(name, **kwargs)
        logger.debug("Creating Monoenergetic <{}>".format(self._samplerName))

        self.peak_energy = peak_energy  # (eV)
        self.peak_rate = peak_rate  # (1/s)

    def CalculateRate(self):
        """
        Calculate the monoenergetic event rate for the binned sampling.

        Results:
            True if successful.
        """
        logger.debug(f"Calculating monoenergetic event rate at {self.peak_energy} eV")

        # calculate the energy spectrum. shape=(ke_bins,)
        ke_spectrum = np.zeros(self._ke_bins, dtype="float64")  # (1/s)
        peak_i = np.digitize(self.peak_energy, self._edge["ke"]) - 1
        if 0 <= peak_i < self._ke_bins:
            ke_spectrum[peak_i] = self.peak_rate  # (1/s)
        else:
            logger.warning(f"Peak energy {self.peak_energy} is out of bounds.")

        # add to self._rate (1/s). shape=(ke_bins,)
        self._rate += ke_spectrum

        return True

    def UnbinnedSample(self, runtimes: List[float]) -> bool:
        """
        Sample from the monoenergetic peak in unbinned mode.

        Parameters:
            runtimes: The list of runtimes in seconds.
        Results:
            True if successful.
        """
        logger.debug("Unbinned sampling for <{}>".format(self.name))
        for runtime in runtimes:
            expected_counts = self.peak_rate * runtime  # (counts)
            if expected_counts < 0:
                logger.error("Negative expected counts found in <{}>".format(self.name))
                return False
            counts = np.random.poisson(expected_counts)  # shape=()

            samples = np.full(counts, self.peak_energy, dtype="float64")

            self._sample_energy.append(samples)

        return True
