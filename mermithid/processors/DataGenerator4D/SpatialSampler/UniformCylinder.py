"""
Sample theta (pitch angle) from a spherically symmetric distribution and the
positions r, phi from a uniform cylinder.
Author: S. M. Lee
First Date: September 15, 2025
Last Update: November 26, 2025
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

    def BuildThetaRPDF(self) -> bool:
        """
        Build the joint PDF of theta and r for the uniform cylinder. It assumes
        independence from energy.

        PDF: p(theta, r) dr dtheta = p(r, theta) p(theta) dr dtheta
                                   = (2r/R^2) * (sin(theta)/2) dr dtheta
                                   = r sin(theta) / R^2 dr dtheta

        Results:
            True if successful.
        """
        logger.debug("Building r-theta PDF for UniformCylinder")

        # Create meshgrid for r and theta
        # theta_mesh[i, j] = theta_bin_centers[j], r_mesh[i, j] = r_bin_centers[i]
        theta_mesh, r_mesh = np.meshgrid(
            self._theta_centers, self._r_centers
        )  # (bins_r, bins_theta)

        # Calculate the joint PDF
        pdf = (r_mesh * np.sin(theta_mesh)) / (self.radius**2)  # (bins_r, bins_theta)
        pdf[r_mesh > self.radius] = 0.0  # zero outside the cylinder

        # Apply trapping efficiency from the cavity field if available
        if self.cavity_field is not None:
            efficiency = self.cavity_field.GetEfficiency(
                r=r_mesh, theta=theta_mesh
            )  # (bins_r, bins_theta)
            pdf *= efficiency  # (bins_r, bins_theta)

            # TEST: log the efficiency
            for r, theta, eff in zip(r_mesh[0], theta_mesh[0], efficiency[0]):
                logger.debug(
                    "r={:.4f} m, theta={:.4f}rad (s2={:.2f}), efficiency = {:.4f}".format(
                        r, theta, np.sin(theta)**2, eff
                    )
                )

        # Normalize the PDF
        dr = np.diff(self._r_edge)  # (bins_r,)
        dtheta = np.diff(self._theta_edge)  # (bins_theta,)
        normalization = np.sum(pdf * dr[:, np.newaxis] * dtheta[np.newaxis, :])
        self.r_theta_pdf = pdf / normalization  # (bins_r, bins_theta)

        logger.debug("Done building r-theta PDF for UniformCylinder")
        return True

    def BinnedSampleThetaR(self) -> bool:
        """
        Sample theta and r from the binned joint PDF p(theta, r).

        Results:
            True if successful.
        """
        logger.debug("Sampling theta and r from UniformCylinder")

        if self.r_theta_pdf is None:
            logger.error("r-theta PDF is not built yet for <{}>".format(self.name))
            return False

        # Flatten the PDF after normalization and create a cumulative distribution function (CDF)
        dr = np.diff(self._r_edge)  # (bins_r,)
        dtheta = np.diff(self._theta_edge)  # (bins_theta,)
        flat_pdf = (
            self.r_theta_pdf * dr[:, np.newaxis] * dtheta[np.newaxis, :]
        ).flatten()  # (bins_r * bins_theta,)
        cdf = np.cumsum(flat_pdf)
        cdf /= cdf[-1]  # ensure normalization of CDF

        # for each runtime, sample from the CDF
        for i, ke in enumerate(self.ke):
            entries = ke.shape[0]
            u = np.random.uniform(0, 1, entries)  # (entries,)

            # Find the bin indices using inverse transform sampling
            bin_indices = np.searchsorted(cdf, u)  # (entries,)
            bin_indices = np.clip(bin_indices, 0, len(flat_pdf) - 1)

            # Convert flat indices back to 2D indices
            r_bin_indices = bin_indices // self._theta_bins  # (entries,)
            theta_bin_indices = bin_indices % self._theta_bins  # (entries,)

            # Sample r and theta from the bin centers
            r_samples = self._r_centers[r_bin_indices]  # (entries,)
            theta_samples = self._theta_centers[theta_bin_indices]  # (entries,)

            self._sample_geometry[i][:, 0] = theta_samples  # (rad)
            self._sample_geometry[i][:, 1] = r_samples  # (m)

        return True

    def BinnedSamplePhi(self) -> bool:
        """
        Sample phi from the binned uniform distribution p(phi|theta, r).

        Results:
            True if successful.
        """
        logger.debug("Sampling phi from UniformCylinder")

        for i, ke in enumerate(self.ke):
            entries = ke.shape[0]
            phi = np.random.uniform(0, 2 * np.pi, entries)  # (rad)
            self._sample_geometry[i][:, 2] = phi

        return True

    def UnbinnedSampleTheta(self) -> bool:
        """
        Sample theta (pitch angle) from a spherically symmetric distribution in unbinned mode.

        PDF: p(theta|E) dtheta = 2 pi sin(theta) dtheta / (4 pi) = sin(theta) / 2 dtheta
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
            self._sample_geometry[i][:, 0] = theta

        return True

    def UnbinnedSampleR(self) -> bool:
        """
        Sample radius from a uniform cylinder in unbinned mode.

        PDF: p(r|E, theta) dr = 2 pi r dr / (pi R^2) = 2 r / R^2 dr
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
            self._sample_geometry[i][:, 1] = r

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
            self._sample_geometry[i][:, 2] = phi

        return True
