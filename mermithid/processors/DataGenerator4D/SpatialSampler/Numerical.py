"""
Sample spatial variables from a user-provided histogram.
Author: S. M. Lee
First Date: July 14, 2026
Last Update: September 09, 2026
"""

from __future__ import absolute_import
from typing import List, Optional

import numpy as np

from morpho.utilities import morphologging

from .SpatialSampler import SpatialSampler

logger = morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)


class Numerical(SpatialSampler):
    """
    Sample (r_start, theta_center, phi_start) from a user-provided histogram.

    Parameters:
        path: The path to the numpy file containing the spatial PDF histogram.
        The file contains `pdf`, the 3D spatial PDF histogram with shape
        (r_bins, theta_bins, phi_bins), and `r_edges`, `theta_edges`, `phi_edges`,
        the bin edges for each dimension with shape (r_bins + 1,), (theta_bins + 1,),
        and (phi_bins + 1,).
        apply_trapping_efficiency: If True and cavity field is configured, apply
            cavity trapping-efficiency weighting to (r, theta) bins before sampling.
    
    Results:
        The sampled pitch angle at the trap center (theta_center), positions
        (r_start, phi_start), and pitch angle at production (theta_start) in
        rad, m, rad, and rad units are accessible via the `result` property,
        stored in `self._sample_geometry`.
    """

    def __init__(
        self,
        name: str,
        path: str,
        apply_trapping_efficiency: bool = False,
        pdf: Optional[np.ndarray] = None,
        r_edges: Optional[np.ndarray] = None,
        theta_edges: Optional[np.ndarray] = None,
        phi_edges: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super(Numerical, self).__init__(name, **kwargs)
        logger.debug("Creating Numerical spatial sampler <{}>".format(self._samplerName))

        if path is not None:
            try:
                with np.load(path) as data:
                    _given_pdf = data["pdf"]
                    _given_r_edges = data["r_edges"]
                    _given_theta_edges = data["theta_edges"]
                    _given_phi_edges = data["phi_edges"]
                logger.info(f"{self.name}: Loaded spatial PDF histogram from {path}")
            except Exception as e:
                raise RuntimeError(
                    f"{self.name}: Failed to load spatial PDF histogram from {path}: {e}"
                )
        elif pdf is not None:
            if r_edges is None or theta_edges is None or phi_edges is None:
                raise ValueError("If 'pdf' is provided, 'r_edges', 'theta_edges', and 'phi_edges' must also be provided.")
            _given_pdf = np.asarray(pdf, dtype="float64")
            _given_r_edges = np.asarray(r_edges, dtype="float64")
            _given_theta_edges = np.asarray(theta_edges, dtype="float64")
            _given_phi_edges = np.asarray(phi_edges, dtype="float64")
        else:
            raise ValueError("Either 'path' or 'pdf' must be provided.")
        
        self._r_edges = np.asarray(_given_r_edges, dtype="float64")
        self._theta_edges = np.asarray(_given_theta_edges, dtype="float64")
        self._phi_edges = np.asarray(_given_phi_edges, dtype="float64")

        self._r_centers = 0.5 * (self._r_edges[1:] + self._r_edges[:-1])
        self._theta_centers = 0.5 * (self._theta_edges[1:] + self._theta_edges[:-1])
        self._phi_centers = 0.5 * (self._phi_edges[1:] + self._phi_edges[:-1])
        self._theta_bins = self._theta_edges.size - 1
        self._r_bins = self._r_edges.size - 1
        self._phi_bins = self._phi_edges.size - 1

        self._numerical_pdf = np.asarray(_given_pdf, dtype="float64")  # (r_bins, theta_bins, phi_bins)

        if self._numerical_pdf.ndim != 3:
            logger.error("Spatial PDF histogram must be 3-dimensional.")
            raise ValueError("Spatial PDF histogram must be 3-dimensional.")
        if self._numerical_pdf.shape != (
            self._r_edges.size - 1,
            self._theta_edges.size - 1,
            self._phi_edges.size - 1,
        ):
            logger.error("Spatial PDF edges must have one more element than the PDF's dimensions.")
            raise ValueError("Spatial PDF edges must have one more element than the PDF's dimensions.")
        if np.any(self._numerical_pdf < 0):
            logger.error("Spatial PDF histogram contains negative values.")
            raise ValueError("Spatial PDF histogram contains negative values.")
        
        self.apply_trapping_efficiency = apply_trapping_efficiency

        self._weighted_hist: Optional[np.ndarray] = None
        self._sample_r_bin_indices: Optional[List[np.ndarray]] = None
        self._sample_theta_bin_indices: Optional[List[np.ndarray]] = None

    def BuildThetaRPDF(self) -> bool:
        """
        Build r-theta PDF by marginalizing over phi from user-provided histogram.
        """
        hist = np.array(self._numerical_pdf, copy=True)  # (r_bins, theta_bins, phi_bins)

        # TODO: make this trapping efficiency application as a common method of SpatialSampler
        if self.apply_trapping_efficiency:
            if self.cavity_field is None:
                logger.warning("Cavity field is empty. Trapping efficiency will not be applied.")
            
            if self.cavity_field is not None:
                theta_mesh, r_mesh = np.meshgrid(self._theta_centers, self._r_centers)  # (r_bins, theta_bins)
                efficiency, z_thr = self.cavity_field.GetEfficiency(
                    r=r_mesh, theta=theta_mesh, return_z_thr=True
                )  # (r_bins, theta_bins), (r_bins, theta_bins, 2)
                self.z_thr = z_thr  # (r_bins, theta_bins, 2)
                hist *= efficiency[:, :, np.newaxis]  # (r_bins, theta_bins, phi_bins)
        
        dr = np.diff(self._r_edges)[:, np.newaxis, np.newaxis]  # (r_bins, 1, 1)
        dtheta = np.diff(self._theta_edges)[np.newaxis, :, np.newaxis]  # (1, theta_bins, 1)
        dphi = np.diff(self._phi_edges)[np.newaxis, np.newaxis, :]  # (1, 1, phi_bins)
        normalization = np.sum(hist * dr * dtheta * dphi)
        if normalization <= 0:
            logger.error("Spatial histogram normalization is zero after weighting.")
            return False

        self._weighted_hist = hist / normalization  # (r_bins, theta_bins, phi_bins)
        self.r_theta_pdf = np.sum(self._weighted_hist * dphi, axis=2)  # (r_bins, theta_bins)

        logger.debug("Done building r-theta PDF for Numerical")
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
        if self._weighted_hist is None:
            logger.error("Weighted spatial histogram is not built yet for <{}>".format(self.name))
            return False

        # project to (r, theta) plane
        dphi = np.diff(self._phi_edges)[np.newaxis, np.newaxis, :]  # (1, 1, phi_bins)
        r_theta_mass = np.sum(self._weighted_hist * dphi, axis=2)  # (r_bins, theta_bins)

        # flatten the mass to a 1D array
        dr = np.diff(self._r_edges)[:, np.newaxis]  # (r_bins, 1)
        dtheta = np.diff(self._theta_edges)[np.newaxis, :]  # (1, theta_bins)
        flat_mass = (r_theta_mass * dr * dtheta).ravel()  # (r_bins * theta_bins)

        cdf = np.cumsum(flat_mass)
        cdf /= cdf[-1]  # ensure normalization of CDF

        self._sample_r_bin_indices = []
        self._sample_theta_bin_indices = []

        # for each runtime, sample from the CDF
        for i, ke in enumerate(self.ke):
            entries = ke.shape[0]
            if entries == 0:
                self._sample_r_bin_indices.append(np.zeros(0, dtype=int))
                self._sample_theta_bin_indices.append(np.zeros(0, dtype=int))
                continue

            u = np.random.uniform(0.0, 1.0, entries)  # (entries,)

            # Find the bin indices using inverse transform sampling
            bin_indices = np.searchsorted(cdf, u)  # (entries,)
            bin_indices = np.clip(bin_indices, 0, len(flat_mass) - 1)

            # Convert flat indices back to 2D indices
            r_bin_indices = bin_indices // self._theta_bins  # (entries,)
            theta_bin_indices = bin_indices % self._theta_bins  # (entries,)
            self._sample_r_bin_indices.append(r_bin_indices)
            self._sample_theta_bin_indices.append(theta_bin_indices)

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
        Sample phi_start conditional on sampled (r, theta) bins.
        
        Results:
            True if successful.
        """
        logger.debug("Sampling phi from Numerical")

        if self._weighted_hist is None:
            logger.error("Weighted spatial histogram is not built yet for <{}>".format(self.name))
            return False
        if self._sample_r_bin_indices is None or self._sample_theta_bin_indices is None:
            logger.error("Theta-r samples are not available for <{}>".format(self.name))
            return False

        dphi = np.diff(self._phi_edges)  # (phi_bins,)
        phi_masses = self._weighted_hist * dphi[np.newaxis, np.newaxis, :]  # (r_bins, theta_bins, phi_bins)
        phi_cdfs = np.cumsum(phi_masses, axis=2)  # (r_bins, theta_bins, phi_bins)
        phi_cdfs = np.insert(phi_cdfs, 0, 0.0, axis=2)  # (r_bins, theta_bins, phi_bins + 1)
        phi_cdfs /= phi_cdfs[:, :, -1][:, :, np.newaxis]  # normalize CDFs to 1

        for i, (r_bin_indices, theta_bin_indices) in enumerate(
            zip(self._sample_r_bin_indices, self._sample_theta_bin_indices)
        ):
            entries = r_bin_indices.shape[0]
            us = np.random.uniform(0.0, 1.0, entries)  # (entries,)

            for entry, (u, r_i, theta_i) in enumerate(
                zip(us, r_bin_indices, theta_bin_indices)
            ):
                phi_cdf = phi_cdfs[r_i, theta_i]  # (phi_bins + 1,)
                phi_sample = np.interp(u, phi_cdf, self._phi_edges)  # (entries,)
                self._sample_phi_start[i][entry] = phi_sample

        return True
