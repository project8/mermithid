"""
Sample energy from a user-provided histogram.
Author: S. M. Lee
First Date: July 14, 2026
Last Update: September 09, 2026
"""

from __future__ import absolute_import
from typing import List, Optional, Union

import numpy as np
from typing import Optional, Union, List

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
        path: The path to the numpy file containing the numerical energy `rate`
            and `ke_edges`. `rate` should be a 1D array of the length of N in
            the unit of 1/s, and `ke_edges` should be a 1D array of the length
            of N+1 in the unit of eV. If path is provided, the `rate` and
            `ke_edges` will be loaded from the file.
        rate: The 1D array of the rate in the unit of 1/s. If path is provided,
            this will be ignored.
        ke_edges: The 1D array of the energy edges in the unit of eV. If path
            is provided, this will be ignored.
        ke_min: The minimum kinetic energy in the unit of eV. If ke_edges is
            provided, this will be ignored.
        ke_max: The maximum kinetic energy in the unit of eV. If ke_edges is
            provided, this will be ignored.
        ke_bins: The number of kinetic energy bins. If ke_edges is provided,
            this will be ignored.
    """

    def __init__(
        self,
        name: str,
        path: Optional[str] = None,
        binned_mode: bool = False,
        rate: Optional[np.ndarray] = None,  # (1/s)
        ke_edges: Optional[np.ndarray] = None,  # (eV)
        **kwargs,
    ):
        super(Numerical, self).__init__(name, binned_mode=binned_mode, **kwargs)
        logger.debug("Creating Numerical <{}>".format(self._samplerName))

        _given_rate: Optional[np.ndarray] = None
        _given_edges: Optional[np.ndarray] = None
        print(path)
        if path is not None:
            try:
                data = np.load(path)
                _given_rate = np.asarray(data["rate"], dtype="float64")
                _given_edges = np.asarray(data["ke_edges"], dtype="float64")
                data.close()
                logger.info(f"{self.name}: Loaded energy rate and ke_edges from {path}")
            except Exception as e:
                raise RuntimeError(
                    f"{self.name}: Failed to load energy rate and ke_edges from {path}: {e}"
                )
        elif rate is not None:
            if ke_edges is None:
                raise ValueError("If 'rate' is provided, 'ke_edges' must also be provided.")
            _given_rate = np.asarray(rate, dtype="float64")
            _given_edges = np.asarray(ke_edges, dtype="float64")
        else:
            raise ValueError("Either 'path' or 'rate' must be provided.")

        if not isinstance(_given_rate, np.ndarray):
            logger.error("Energy histogram is not a numpy array.")
            raise ValueError("Energy histogram is not a numpy array.")
        if not isinstance(_given_edges, np.ndarray):
            logger.error("Energy histogram edges are not a numpy array.")
            raise ValueError("Energy histogram edges are not a numpy array.")

        if _given_rate.ndim != 1:
            logger.error("Energy histogram must be 1-dimensional.")
            raise ValueError("Energy histogram must be 1-dimensional.")
        if _given_edges.ndim != 1:
            logger.error("Energy histogram edges must be 1-dimensional.")
            raise ValueError("Energy histogram edges must be 1-dimensional.")
        if _given_edges.shape[0] != _given_rate.shape[0] + 1:
            logger.error("Energy histogram edges must have one more element than the histogram.")
            raise ValueError("Energy histogram edges must have one more element than the histogram.")
        if np.any(_given_edges[1:] < _given_edges[:-1]):
            logger.error("Energy histogram edges must be in ascending order.")
            raise ValueError("Energy histogram edges must be in ascending order.")
        if np.any(_given_rate < 0):
            logger.error("Energy histogram contains negative values.")
            raise ValueError("Energy histogram contains negative values.")

        self._edges = _given_edges
        self._ke_bins = len(self._edges) - 1
        self._rate = _given_rate

    def CalculateRate(self) -> bool:
        """
        Calculate per-bin rate for binned sampling.
        This `Numerical` class assumes that the user has provided a numerical
        energy histogram and edges. So nothing happens here.
        """
        return True

    def UnbinnedSample(self, runtimes: List[float]) -> bool:
        """
        Sample energies from the histogram using inverse transform sampling.

        Parameters:
            runtimes: The list of runtimes in seconds.
        Results:
            True if successful.
        """
        cdf = np.cumsum(self._rate)
        cdf = np.insert(cdf, 0, 0.0)  # (ke_bins + 1,)
        cdf /= cdf[-1]
        cdf[-1] = 1.0

        for runtime in runtimes:
            if runtime < 0:
                logger.error("Runtime must be non-negative for <{}>".format(self.name))
                return False

            expected_counts = self._rate.sum() * runtime
            counts = np.random.poisson(expected_counts)
            if counts == 0:
                self._sample_energy.append(np.zeros(0, dtype="float64"))
                continue

            # Randomly sample from the linear interpolation of the CDF
            # TODO: a better interpolation method?
            u = np.random.uniform(0.0, 1.0, counts)
            samples = np.interp(u, cdf, self._edges)  # (counts,)
            self._sample_energy.append(samples)

        return True
