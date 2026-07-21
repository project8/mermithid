"""
Sample energy from a user-provided histogram.
Author: S. M. Lee
First Date: July 14, 2026
Last Update: July 20, 2026
"""

from __future__ import absolute_import
from typing import List, Union

import numpy as np

from morpho.utilities import morphologging

from .EnergySampler import EnergySampler

logger = morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)


class Numerical(EnergySampler):
    """
    Sample kinetic energy from user-provided binned weights.

    Parameters:
        name: The name of the energy sampler.
        path: The path to the numpy file containing the numerical energy `rates`
            and `ke_edges`. `rates` should be a 1D array of the length of N in
            the unit of 1/s, and `ke_edges` should be a 1D array of the length
            of N+1 in the unit of eV.
    """

    def __init__(
        self,
        name: str,
        path: str,
        **kwargs,
    ):
        super(Numerical, self).__init__(name, **kwargs)
        logger.debug("Creating Numerical <{}>".format(self._samplerName))

        try:
            data = np.load(path)
            rate = data["rate"]
            edges = data["ke_edges"]
            data.close()
            logger.info(f"{self.name}: Loaded energy rate and ke_edges from {path}")
        except Exception as e:
            raise RuntimeError(
                f"{self.name}: Failed to load energy rate and ke_edges from {path}: {e}"
            )
        
        self._numerical_rate = np.asarray(rate, dtype="float64")  # (numerical_ke_bins,) (1/s)
        self._numerical_edges = np.asarray(edges, dtype="float64")  # (numerical_ke_bins + 1,) (eV)

        if self._numerical_rate.ndim != 1:
            logger.error("Energy histogram must be 1-dimensional.")
            raise ValueError("Energy histogram must be 1-dimensional.")
        if self._numerical_edges.ndim != 1:
            logger.error("Energy histogram edges must be 1-dimensional.")
            raise ValueError("Energy histogram edges must be 1-dimensional.")
        if self._numerical_edges.shape[0] != self._numerical_rate.shape[0] + 1:
            logger.error("Energy histogram edges must have one more element than the histogram.")
            raise ValueError("Energy histogram edges must have one more element than the histogram.")
        if np.any(self._numerical_edges[1:] < self._numerical_edges[:-1]):
            logger.error("Energy histogram edges must be in ascending order.")
            raise ValueError("Energy histogram edges must be in ascending order.")
        if np.any(self._numerical_rate < 0):
            logger.error("Energy histogram contains negative values.")
            raise ValueError("Energy histogram contains negative values.")

    def CalculateRate(self) -> bool:
        """
        Calculate per-bin rates for binned sampling.
        This method splits the numerical energy rates to the global energy bins.
        """
        # split the numerical rates to the global edges
        numerical_ke_min = self._numerical_edges[:-1]  # (numerical_ke_bins,)
        numerical_ke_max = self._numerical_edges[1:]  # (numerical_ke_bins,)
        bin_widths = numerical_ke_max - numerical_ke_min  # (numerical_ke_bins,)
        
        ke_min = self._edges[:-1, np.newaxis]  # (ke_bins, 1)
        ke_max = self._edges[1:, np.newaxis]  # (ke_bins, 1)
        
        overlap_ke = np.maximum(  # (ke_bins, numerical_ke_bins)
            0,
            np.minimum(ke_max, numerical_ke_max)  # (ke_bins, numerical_ke_bins)
            - np.maximum(ke_min, numerical_ke_min)  # (ke_bins, numerical_ke_bins)
        )
        split_matrix = overlap_ke / bin_widths  # (ke_bins, numerical_ke_bins)
        split_rate = np.sum(self._numerical_rate * split_matrix, axis=1)  # (ke_bins,)

        self._rate += split_rate
        return True

    def UnbinnedSample(self, runtimes: List[float]) -> bool:
        """
        Sample energies from the histogram using inverse transform sampling.

        Parameters:
            runtimes: The list of runtimes in seconds.
        Results:
            True if successful.
        """
        cdf = np.cumsum(self._numerical_rate)
        cdf = np.insert(cdf, 0, 0.0)  # (numerical_ke_bins + 1,)
        cdf /= cdf[-1]
        cdf[-1] = 1.0

        for runtime in runtimes:
            if runtime < 0:
                logger.error("Runtime must be non-negative for <{}>".format(self.name))
                return False

            expected_counts = self._numerical_rate.sum() * runtime
            counts = np.random.poisson(expected_counts)
            if counts == 0:
                self._sample_energy.append(np.zeros(0, dtype="float64"))
                continue

            # Randomly sample from the linear interpolation of the CDF
            # TODO: a better interpolation method?
            u = np.random.uniform(0.0, 1.0, counts)
            samples = np.interp(u, cdf, self._numerical_edges)  # (counts,)
            self._sample_energy.append(samples)

        return True
