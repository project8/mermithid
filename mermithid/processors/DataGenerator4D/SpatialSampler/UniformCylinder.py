"""
Spatial variable sampler for the uniform cylinder geometry.
Author: S. M. Lee
First Date: September 15, 2025
Last Update: September 09, 2026
"""

from __future__ import absolute_import
import abc
import six

import numpy as np

from morpho.utilities import morphologging

from .SpatialSampler import SpatialSampler

logger = morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)


class UniformCylinder(SpatialSampler):
    """
    Sample theta_start (pitch angle when the electron is produced) from a
    spherically symmetric distribution and the positions r_start, phi_start from
    a uniform cylinder. Convert theta_start to theta_center (pitch angle at trap
    center) using the cavity field if available.

    Parameters:
        name: The name of the instance
        radius: The radius of the cylinder (m)

    Results:
        The sampled pitch angle at the trap center (theta_center), positions
        (r_start, phi_start), and pitch angle at production (theta_start) in
        rad, m, rad, and rad units are accessible via the `result` property,
        stored in `self._sample_geometry`.
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
        Build the joint PDF of theta_start and r_start for the uniform cylinder.
        It assumes independence from energy.

        PDF: p(theta_start, r_start) dr_start dtheta_start
            = p(r_start, theta_start) p(theta_start) dr_start dtheta_start
            = (2r_start/R^2) * (sin(theta_start)/2) dr_start dtheta_start
            = r_start sin(theta_start) / R^2 dr_start dtheta_start

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
            efficiency, z_thr = self.cavity_field.GetEfficiency(
                r=r_mesh, theta=theta_mesh, return_z_thr=True
            )  # (bins_r, bins_theta), (bins_r, bins_theta, 2)
            self.z_thr = z_thr  # (bins_r, bins_theta, 2)

            total_before = np.sum(pdf)
            pdf *= efficiency  # (bins_r, bins_theta)
            total_after = np.sum(pdf)
            total_efficiency = total_after / total_before if total_before > 0 else 0.0

            msg = "Trapping efficiency is applied to r-theta PDF."
            msg += " Total efficiency: {:.6f}".format(total_efficiency)
            logger.debug(msg)

        # Normalize the PDF
        dr = np.diff(self._r_edges)  # (bins_r,)
        dtheta = np.diff(self._theta_edges)  # (bins_theta,)
        normalization = np.sum(pdf * dr[:, np.newaxis] * dtheta[np.newaxis, :])
        self.r_theta_pdf = pdf / normalization  # (bins_r, bins_theta)

        logger.debug("Done building r-theta PDF for UniformCylinder")
        return True

    def BinnedSampleThetaR(self) -> bool:
        """
        Sample theta_start and r_start from the binned joint PDF p(theta_start, r_start).
        Convert theta_start to theta_center using the cavity field if available.
        The conversion formula is:
        $\\theta_{center} = \\arcsin\\left(\\sin(\\theta_{start}) \\sqrt{\\frac{B_{center}}{B_{start}}}\\right)$

        Results:
            True if successful.
        """
        logger.debug("Sampling theta_center and r_start from UniformCylinder")

        if self.r_theta_pdf is None:
            logger.error("r-theta PDF is not built yet for <{}>".format(self.name))
            return False

        # Flatten the PDF after normalization and create a cumulative distribution function (CDF)
        dr = np.diff(self._r_edges)  # (bins_r,)
        dtheta = np.diff(self._theta_edges)  # (bins_theta,)
        flat_mass = (
            self.r_theta_pdf * dr[:, np.newaxis] * dtheta[np.newaxis, :]
        ).flatten()  # (bins_r * bins_theta,)
        cdf = np.cumsum(flat_mass)
        cdf /= cdf[-1]  # ensure normalization of CDF

        # for each runtime, sample from the CDF
        # TODO: continuous sampling?
        for i, ke in enumerate(self.ke):
            entries = ke.shape[0]
            if entries == 0:
                self._sample_r_bin_indices.append(np.zeros(0, dtype=int))
                self._sample_theta_bin_indices.append(np.zeros(0, dtype=int))
                continue

            u = np.random.uniform(0, 1, entries)  # (entries,)

            # Find the bin indices using inverse transform sampling
            bin_indices = np.searchsorted(cdf, u)  # (entries,)
            bin_indices = np.clip(bin_indices, 0, len(flat_mass) - 1)

            # Convert flat indices back to 2D indices
            r_bin_indices = bin_indices // self._theta_bins  # (entries,)
            theta_bin_indices = bin_indices % self._theta_bins  # (entries,)

            # # Sample r and theta from the bin centers
            # r_samples = self._r_centers[r_bin_indices]  # (entries,)
            # theta_samples = self._theta_centers[theta_bin_indices]  # (entries,)

            # Sample r and theta from the bin
            r_samples = np.random.uniform(
                self._r_edges[r_bin_indices], self._r_edges[r_bin_indices + 1]
            )  # (entries,)
            theta_samples = np.random.uniform(
                self._theta_edges[theta_bin_indices], self._theta_edges[theta_bin_indices + 1]
            )  # (entries,)

            self._sample_theta_start[i] = theta_samples  # (rad)
            self._sample_r_start[i] = r_samples  # (m)

            # Transform into theta_center
            # theta_center = arcsin(sin(theta_start) sqrt(B_center / B_start))$
            if self.z_thr is not None:
                z_thr_samples = self.z_thr[
                    r_bin_indices, theta_bin_indices
                ]  # (entries, 2)
                z_lower = z_thr_samples[:, 0]  # (entries,)
                z_upper = z_thr_samples[:, 1]  # (entries,)

                # randomly select z_start between z_lower and z_upper
                z_start_samples = np.random.uniform(z_lower, z_upper)  # (entries,)
                B_start_samples = self.cavity_field.GetFieldValue(
                    r=r_samples, z=z_start_samples
                )  # (entries,)
                B_center_samples = self.cavity_field.GetFieldValue(
                    r=r_samples, z=np.zeros_like(z_start_samples)
                )  # (entries,)

                theta_center_samples = np.arcsin(
                    np.sin(theta_samples) * np.sqrt(B_center_samples / B_start_samples)
                )  # (entries,)
                lower_hemisphere = theta_samples > (np.pi / 2)
                theta_center_samples[lower_hemisphere] = (
                    np.pi - theta_center_samples[lower_hemisphere]
                )

                self._sample_theta_center[i] = theta_center_samples  # (rad)
            else:
                msg = "z_thr is not available. CavityField may not be set."
                msg += " Assigning theta_center as theta_start."
                logger.warning(msg)
                self._sample_theta_center[i] = theta_samples  # (rad)

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
            phi = np.random.uniform(self._phi_edges[0], self._phi_edges[-1], entries)  # (rad)
            self._sample_phi_start[i] = phi

        return True

    # TODO: implement unbinned sampling methods if needed
    # def UnbinnedSampleTheta(self) -> bool:
    #     """
    #     Sample theta (pitch angle) from a spherically symmetric distribution in unbinned mode.

    #     PDF: p(theta|E) dtheta = 2 pi sin(theta) dtheta / (4 pi) = sin(theta) / 2 dtheta
    #     CDF: C(theta|E) = (1 - cos(theta)) / 2
    #     Inverse CDF: theta = arccos(1 - 2u), where u is a uniform random number in [0, 1).

    #     Results:
    #         True if successful.
    #     """
    #     logger.debug("Sampling theta from UniformCylinder")

    #     for i, ke in enumerate(self.ke):
    #         entries = ke.shape[0]
    #         u = np.random.uniform(0, 1, entries)
    #         theta = np.arccos(1 - 2 * u)  # (rad)
    #         self._sample_theta_start[i] = theta

    #         # TODO: transform into theta_center
    #         self._sample_theta_center[i] = theta  # (rad)

    #     return True

    # def UnbinnedSampleR(self) -> bool:
    #     """
    #     Sample radius from a uniform cylinder in unbinned mode.

    #     PDF: p(r|E, theta) dr = 2 pi r dr / (pi R^2) = 2 r / R^2 dr
    #     CDF: C(r|E, theta) = (r/R)^2
    #     Inverse CDF: r = R * (u)^(1/2), where u is a uniform random number in [0, 1).

    #     Results:
    #         True if successful.
    #     """
    #     logger.debug("Sampling radius from UniformCylinder")

    #     for i, ke in enumerate(self.ke):
    #         entries = ke.shape[0]
    #         u = np.random.uniform(0, 1, entries)
    #         r = self.radius * (u ** (1.0 / 2.0))
    #         self._sample_r_start[i] = r

    #     return True

    # def UnbinnedSamplePhi(self) -> bool:
    #     """
    #     Sample phi from a uniform cylinder in unbinned mode.

    #     PDF: p(phi|E, theta, r) dphi = dphi / (2 pi)
    #     CDF: C(phi|E, theta, r) = phi / (2 pi)
    #     Inverse CDF: phi = 2 pi u, where u is a uniform random number in [0, 1).

    #     Results:
    #         True if successful.
    #     """
    #     logger.debug("Sampling phi from UniformCylinder")

    #     for i, ke in enumerate(self.ke):
    #         entries = ke.shape[0]
    #         phi = np.random.uniform(0, 2 * np.pi, entries)  # (rad)
    #         self._sample_phi_start[i] = phi

    #     return True
