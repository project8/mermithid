"""
Sample energy from a flat spectrum.
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
__all__.append(__name__)  # type: ignore


class Flat(EnergySampler):
    """
    Sample energy from a flat spectrum.
    """

    def __init__(
        self,
        name,
        flat_rate: float = 1e-5,  # (1/s/eV)
        **kwargs,
    ):
        super(Flat, self).__init__(name, **kwargs)
        logger.debug("Creating Flat <{}>".format(self._samplerName))

        self.flat_rate: float = flat_rate  # (1/s/eV)

    def CalculateRate(self) -> bool:
        """
        Calculate the event rate for the binned sampling.

        Results:
            True if successful.
        """
        logger.debug(f"Calculating flat event rate with {self.flat_rate:.5f} (1/s/eV)")

        # calculate the energy spectrum. shape=(ke_bins,)
        ke_spectrum = np.full(self._ke_bins, self.flat_rate, dtype="float64")  # (1/s/eV)
        ke_spectrum *= self._edge[1:] - self._edge[:-1]  # (1/s)

        # add to self._rate (1/s). shape=(ke_bins,)
        self._rate += ke_spectrum

        return True

    def UnbinnedSample(self, runtimes: List[float]) -> bool:
        """
        Sample energy from the flat spectrum in unbinned mode.

        Parameters:
            runtimes: The list of runtimes in seconds.
        Results:
            True if successful.
        """
        logger.debug("Unbinned sampling for <{}>".format(self.name))
        for runtime in runtimes:
            expected_counts = (
                self.flat_rate * (self._edge[-1] - self._edge[0]) * runtime
            )  # (counts)
            if expected_counts < 0:
                logger.error("Negative expected counts found in <{}>".format(self.name))
                return False
            counts = np.random.poisson(expected_counts)  # (counts,)
            samples = np.random.uniform(self._edge[0], self._edge[-1], counts)

            self._sample_energy.append(samples)

        return True
