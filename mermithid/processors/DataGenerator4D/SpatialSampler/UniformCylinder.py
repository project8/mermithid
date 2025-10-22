"""
Sample theta (pitch angle) from a spherically symmetric distribution and the
positions r, phi from a uniform cylinder.
Author: S. M. Lee
First Date: September 15, 2025
Last Update: October 22, 2025
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


class UniformCylinder(SpatialSampler):
    """
    Sample theta (pitch angle) from a spherically symmetric distribution and the
    positions r, phi from a uniform cylinder.

    Parameters:
        name: The name of the instance
        radius: The radius of the cylinder (m)

    Results:
        The sampled pitch angle (theta) and positions (r, phi) in rad, m, rad
        units are accessible via the `result` property, stored in
        `self._sample_position`.
    """

    def __init__(
        self,
        name,
        radius: float = 0.1,  # (m)
        **kwargs,
    ):
        super(UniformCylinder, self).__init__(name, **kwargs)
        logger.debug("Creating UniformCylinder <{}>".format(self._samplerName))

        self.radius = radius  # (m)

    def UnbinnedSampleTheta(self) -> bool:
        """
        Sample theta (pitch angle) from a spherically symmetric distribution in unbinned mode.

        PDF: p(theta|E) dtheta = 2 pi sin(theta) dtheta / (4 pi) = sin(theta) dtheta / 2
        CDF: C(theta|E) = (1 - cos(theta)) / 2
        Inverse CDF: theta = arccos(1 - 2u), where u is a uniform random number in [0, 1).

        Results:
            True if successful.
        """
        logger.debug("Sampling theta from UniformCylinder")

        for i, ke in enumerate(self.ke):
            entries = ke.shape[0]
            u = np.random.uniform(0, 1, entries)
            theta = np.arccos(1 - 2 * u)  # (rad)
            self._sample_position[i][:, 0] = theta

        return True

    def UnbinnedSampleR(self) -> bool:
        """
        Sample radius from a uniform cylinder in unbinned mode.

        PDF: p(r|E, theta) dr = 2 pi r dr / (pi R^2) = 2 r dr / R^2
        CDF: C(r|E, theta) = (r/R)^2
        Inverse CDF: r = R * (u)^(1/2), where u is a uniform random number in [0, 1).

        Results:
            True if successful.
        """
        logger.debug("Sampling radius from UniformCylinder")

        for i, ke in enumerate(self.ke):
            entries = ke.shape[0]
            u = np.random.uniform(0, 1, entries)
            r = self.radius * (u ** (1.0 / 2.0))
            self._sample_position[i][:, 1] = r

        return True

    def UnbinnedSamplePhi(self) -> bool:
        """
        Sample phi from a uniform cylinder in unbinned mode.

        PDF: p(phi|E, theta, r) dphi = dphi / (2 pi)
        CDF: C(phi|E, theta, r) = phi / (2 pi)
        Inverse CDF: phi = 2 pi u, where u is a uniform random number in [0, 1).

        Results:
            True if successful.
        """
        logger.debug("Sampling phi from UniformCylinder")

        for i, ke in enumerate(self.ke):
            entries = ke.shape[0]
            phi = np.random.uniform(0, 2 * np.pi, entries)  # (rad)
            self._sample_position[i][:, 2] = phi

        return True
