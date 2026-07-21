"""
Energy sampler base class.
Author: S. M. Lee
First Date: September 04, 2025
Last Update: July 20, 2026
"""

from __future__ import absolute_import
import abc
import six
from typing import List, Optional

import numpy as np

from morpho.utilities import morphologging

logger = morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)


@six.add_metaclass(abc.ABCMeta)
class EnergySampler:
    """
    The energy sampling class. The child class must implement `CalculateRate()`
    and `UnbinnedSample()` methods.

    Parameters:
        name: The name of the instance
    Results:
        The sampled energies in eV are accessible via the `result` property,
        stored in `self._sample_energy`.
    """

    def __init__(
        self,
        name,
        binned_mode: bool = False,
        ke_edges: Optional[np.ndarray] = None,  # (eV)
        ke_min: float = 18573.24 - 2300,  # (eV)
        ke_max: float = 18573.24 + 1000,  # (eV)
        ke_bins: int = 100,
        **kwargs,
    ):
        # Class information
        self._samplerName = name
        logger.debug("Creating Spectrum <{}>".format(self._samplerName))

        self.binned_mode = binned_mode
        self._sample_energy: List[np.ndarray] = []  # (eV)

        # ROI properties
        if ke_edges is None:
            ke_edges = np.linspace(ke_min, ke_max, ke_bins + 1)
        if not ke_edges.ndim == 1:
            logger.error("ke_edges must be 1-dimensional.")
            raise ValueError("ke_edges must be 1-dimensional.")
        if np.any(np.diff(ke_edges) <= 0):
            logger.error("ke_edges must be in ascending order.")
            raise ValueError("ke_edges must be in ascending order.")

        self._edges: np.ndarray = np.asarray(ke_edges)
        self._ke_bins: int = len(self._edges) - 1
        self._rate: np.ndarray = np.zeros(self._ke_bins, dtype="float64")

    @property
    def name(self):
        return self._samplerName

    @property
    def rate(self) -> np.ndarray:
        return self._rate

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    @property
    def result(self) -> List[np.ndarray]:
        """
        Returns the list of sampled energy arrays.
        Each array is 1-dimensional and of dtype float64, corresponds to each
        runtime.
        """
        return self._sample_energy

    def Sample(self, runtimes: List[float]) -> bool:
        """
        Sample energy based on the binned rate or unbinned model and given runtimes.

        Parameters:
            runtimes: The list of runtimes in seconds.
        Returns:
            True if successful.
        """
        logger.info("Sampling energy for <{}>...".format(self.name))
        if self.binned_mode:
            if not self.BinnedSample(runtimes):
                logger.error("Error while <{}> sampling energy".format(self.name))
                return False
        else:
            if not self.UnbinnedSample(runtimes):
                logger.error("Error while <{}> sampling energy".format(self.name))
                return False

        logger.info("Done sampling energy for <{}>".format(self.name))
        return True

    @abc.abstractmethod
    def UnbinnedSample(self, runtimes: List[float]) -> bool:
        """
        Method called by `Sample()` to sample the energy from the unbinned model.
        Must be overridden by child class.

        Parameters:
            runtimes: The list of runtimes in seconds.
        Results:
            True if successful.
        """
        return False

    def BinnedSample(self, runtimes: List[float]) -> bool:
        """
        Method called by `Sample()` to sample the energy from the binned rate.

        Parameters:
            runtimes: The list of runtimes in seconds.
        Results:
            True if successful.
        """
        logger.debug("Binned sampling for <{}>".format(self.name))
        if not self.CalculateRate():
            logger.error("Error while <{}> calculating rate".format(self.name))
            return False

        for runtime in runtimes:
            expected_counts = self._rate * runtime  # (counts)
            if any(expected_counts < 0):
                logger.error("Negative expected counts found in <{}>".format(self.name))
                return False
            counts = np.random.poisson(expected_counts)  # shape=(ke_bins,)

            samples = np.zeros(counts.sum(), dtype="float64")
            idx = 0

            # spread samples in each bin
            for bin_idx, count in enumerate(counts):
                if count > 0:
                    samples[idx : idx + count] = np.random.uniform(
                        self._edges[bin_idx], self._edges[bin_idx + 1], count
                    )
                    idx += count

            self._sample_energy.append(samples)

        return True

    @abc.abstractmethod
    def CalculateRate(self) -> bool:
        """
        Method called by `BinnedSample()` to calculate the rate.
        Must be overridden by child class.

        Results:
            True if successful.
        """
        return False
