"""
Spatial sampler base class.
Author: S. M. Lee
First Date: September 15, 2025
Last Update: September 15, 2025
"""

from __future__ import absolute_import
import abc
import six
from typing import Dict, List, Optional

import numpy as np

from morpho.utilities import morphologging

logger = morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)  # type: ignore


@six.add_metaclass(abc.ABCMeta)
class SpatialSampler:
    """
    The spatial sampling class. The child class must implement `UnbinnedSample()`
    methods.

    Parameters:
        name: The name of the instance
    Results:
        The sampled positions (r, theta, phi) in meters, radians, radians units
        are accessible via the `result` property, stored in `self._sample_position`.
    """

    def __init__(
        self,
        name,
        binned_mode: bool = False,
        r_edges: Optional[np.ndarray] = None,  # (m)
        r_min: float = 0.0,  # (m)
        r_max: float = 0.1,  # (m)
        r_bins: int = 10,
        theta_edges: Optional[np.ndarray] = None,  # (rad)
        theta_min: float = 0.0,  # (rad)
        theta_max: float = np.pi,  # (rad)
        theta_bins: int = 36,
        phi_edges: Optional[np.ndarray] = None,  # (rad)
        phi_min: float = 0.0,  # (rad)
        phi_max: float = 2 * np.pi,  # (rad)
        phi_bins: int = 36,
        **kwargs,
    ):
        # Class information
        self._samplerName = name
        logger.debug("Creating Spectrum <{}>".format(self._samplerName))

        self.binned_mode = binned_mode
        self._sample_position: List[np.ndarray] = (
            list()
        )  # (m, rad, rad) shape: [(entries, 3), ...]

        # ROI properties
        if r_edges is None:
            r_edges = np.linspace(r_min, r_max, r_bins + 1)
        if not r_edges.ndim == 1:
            logger.error("r_edges must be 1-dimensional.")
            raise ValueError("r_edges must be 1-dimensional.")
        if np.any(np.diff(r_edges) <= 0):
            logger.error("r_edges must be in ascending order.")
            raise ValueError("r_edges must be in ascending order.")

        if theta_edges is None:
            theta_edges = np.linspace(theta_min, theta_max, theta_bins + 1)
        if not theta_edges.ndim == 1:
            logger.error("theta_edges must be 1-dimensional.")
            raise ValueError("theta_edges must be 1-dimensional.")
        if np.any(np.diff(theta_edges) <= 0):
            logger.error("theta_edges must be in ascending order.")
            raise ValueError("theta_edges must be in ascending order.")

        if phi_edges is None:
            phi_edges = np.linspace(phi_min, phi_max, phi_bins + 1)
        if not phi_edges.ndim == 1:
            logger.error("phi_edges must be 1-dimensional.")
            raise ValueError("phi_edges must be 1-dimensional.")
        if np.any(np.diff(phi_edges) <= 0):
            logger.error("phi_edges must be in ascending order.")
            raise ValueError("phi_edges must be in ascending order.")

        self._r_edge: np.ndarray = np.asarray(r_edges)
        self._theta_edge: np.ndarray = np.asarray(theta_edges)
        self._phi_edge: np.ndarray = np.asarray(phi_edges)
        self._r_bins: int = len(self._r_edge) - 1
        self._theta_bins: int = len(self._theta_edge) - 1
        self._phi_bins: int = len(self._phi_edge) - 1

    @property
    def name(self):
        return self._samplerName

    @property
    def r_edges(self) -> Optional[np.ndarray]:
        if self.binned_mode:
            return None
        return self._r_edge

    @property
    def theta_edges(self) -> Optional[np.ndarray]:
        if self.binned_mode:
            return None
        return self._theta_edge

    @property
    def phi_edges(self) -> Optional[np.ndarray]:
        if self.binned_mode:
            return None
        return self._phi_edge

    @property
    def result(self) -> List[np.ndarray]:
        """
        Returns the list of sampled position arrays.
        Each array is 2-dimensional and of dtype float64, corresponds to each
        runtime, and has shape (N, 3) where N is the number of samples for that
        runtime, and the 3 columns correspond to (r, theta, phi).
        """
        return self._sample_position

    def Sample(
        self,
        ke: List[np.ndarray],  # (eV) shape: [(entries,), ...]
    ) -> bool:
        """
        Sample position based on the binned rate or unbinned model and given
        kinetic energies.

        Returns:
            True if successful.
        """
        logger.info("Sampling position for <{}>...".format(self.name))

        self.ke = ke
        self._sample_position = [
            np.zeros((k.shape[0], 3), dtype="float64") for k in self.ke
        ]  # (m, rad, rad) shape: [(entries, 3), ...]

        if self.binned_mode:
            raise NotImplementedError("Binned mode is not implemented yet.")
            # if not self.BinnedSample():
            #     logger.error("Error while <{}> sampling position".format(self.name))
            #     return False
        else:
            if not self.UnbinnedSampleR():
                logger.error("Error while <{}> sampling r".format(self.name))
                return False
            if not self.UnbinnedSampleTheta():
                logger.error("Error while <{}> sampling theta".format(self.name))
                return False
            if not self.UnbinnedSamplePhi():
                logger.error("Error while <{}> sampling phi".format(self.name))
                return False

        logger.info("Done sampling position for <{}>".format(self.name))
        return True

    @abc.abstractmethod
    def UnbinnedSampleR(self) -> bool:
        """
        Method called by `Sample()` to sample the radius from the unbinned model.
        The marginal distribution p(r|E) should be used.
        Must be overridden by child class.

        Results:
            True if successful.
        """
        raise NotImplementedError("UnbinnedSampleR method is not implemented.")

    @abc.abstractmethod
    def UnbinnedSampleTheta(self) -> bool:
        """
        Method called by `Sample()` to sample the polar angle from the unbinned model.
        The marginal distribution p(theta|r, E) should be used.
        Must be overridden by child class.

        Results:
            True if successful.
        """
        raise NotImplementedError("UnbinnedSampleTheta method is not implemented.")

    @abc.abstractmethod
    def UnbinnedSamplePhi(self) -> bool:
        """
        Method called by `Sample()` to sample the azimuthal angle from the unbinned model.
        The marginal distribution p(phi|theta, r, E) should be used.
        Must be overridden by child class.

        Results:
            True if successful.
        """
        raise NotImplementedError("UnbinnedSamplePhi method is not implemented.")
