"""
Sample position from a uniform sphere.
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

from .SpatialSampler import SpatialSampler

logger = morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)  # type: ignore


class UniformSphere(SpatialSampler):
    """
    Sample position from a uniform sphere.
    """

    def __init__(
        self,
        name,
        sphere_radius: float = 0.1,  # (m)
        **kwargs,
    ):
        super(UniformSphere, self).__init__(name, **kwargs)
        logger.debug("Creating UniformSphere <{}>".format(self._samplerName))

        self.sphere_radius = sphere_radius  # (m)

    def UnbinnedSampleR(self) -> bool:
        """
        Sample radius from a uniform sphere in unbinned mode.

        PDF: p(r|E) dr = 4 pi r^2 dr / (4/3 pi R^3) = 3 r^2 dr / R^3
        CDF: C(r|E) = (r/R)^3
        Inverse CDF: r = R * (u)^(1/3), where u is a uniform random number in [0, 1).

        Results:
            True if successful.
        """
        logger.debug("Sampling radius from a uniform sphere")

        for i, ke in enumerate(self.ke):
            entries = ke.shape[0]
            u = np.random.uniform(0, 1, entries)
            r = self.sphere_radius * (u ** (1.0 / 3.0))
            self._sample_position[i][:, 0] = r

        return True

    def UnbinnedSampleTheta(self) -> bool:
        """
        Sample theta from a uniform sphere in unbinned mode.

        PDF: p(theta|r, E) dtheta = 2 pi sin(theta) dtheta / (4 pi) = sin(theta) dtheta / 2
        CDF: C(theta|r, E) = (1 - cos(theta)) / 2
        Inverse CDF: theta = arccos(1 - 2u), where u is a uniform random number in [0, 1).

        Results:
            True if successful.
        """
        logger.debug("Sampling theta from a uniform sphere")

        for i, ke in enumerate(self.ke):
            entries = ke.shape[0]
            u = np.random.uniform(0, 1, entries)
            theta = np.arccos(1 - 2 * u)  # (rad)
            self._sample_position[i][:, 1] = theta

        return True

    def UnbinnedSamplePhi(self) -> bool:
        """
        Sample phi from a uniform sphere in unbinned mode.

        PDF: p(phi|theta, r, E) dphi = dphi / (2 pi)
        CDF: C(phi|theta, r, E) = phi / (2 pi)
        Inverse CDF: phi = 2 pi u, where u is a uniform random number in [0, 1).

        Results:
            True if successful.
        """
        logger.debug("Sampling phi from a uniform sphere")

        for i, ke in enumerate(self.ke):
            entries = ke.shape[0]
            phi = np.random.uniform(0, 2 * np.pi, entries)  # (rad)
            self._sample_position[i][:, 2] = phi

        return True
