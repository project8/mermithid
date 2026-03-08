"""
EnergyError class for DataGenerator4D.
Handles loading and applying 5D energy error probability maps.

The energy error represents the difference between observed and true kinetic
energy due to detector resolution, electronics noise, and signal processing
uncertainties.

Author: S. M. Lee
Date: March 6, 2026
"""

from __future__ import absolute_import

from typing import Dict, Optional, Union, List, Tuple
import numpy as np

from morpho.utilities import morphologging

logger = morphologging.getLogger(__name__)


class EnergyError:
    """
    Manages energy error simulation for the DataGenerator4D.
    
    The energy error is defined by a 5D probability map over:
    - ke_error: observed - true kinetic energy (eV)
    - ke_start: true kinetic energy (eV)
    - theta_center: pitch angle at center (rad)
    - r_start: starting radial position (m)
    - phi_start: starting azimuthal angle (rad)
    
    For each event, this class:
    1. Finds the nearest bin in 4D phase space (ke_start, theta_center, r_start, phi_start)
       TODO: PDF interpolation across 4D space is not implemented yet.
    2. Selects the 1D ke_error probability distribution from that bin
    3. Samples ke_error continuously using the cumulative distribution function (CDF)
    """

    def __init__(
        self,
        name: str,
        energy_error_map_path: Optional[str] = None,
        ke_error_edges: Optional[np.ndarray] = None,
        ke_start_edges: Optional[np.ndarray] = None,
        theta_center_edges: Optional[np.ndarray] = None,
        r_edges: Optional[np.ndarray] = None,
        phi_edges: Optional[np.ndarray] = None,
        energy_error_map: Optional[np.ndarray] = None,
    ):
        """
        Initialize the EnergyError.

        Parameters:
            name: Name of the energy error instance
            energy_error_map_path: Path to .npy file containing energy error map
            ke_error_edges: Bin edges for energy error (eV), shape (n_ke_error_bins + 1,)
            ke_start_edges: Bin edges for true kinetic energy (eV), shape (n_ke_start_bins + 1,)
            theta_center_edges: Bin edges for theta_center (rad), shape (n_theta_center_bins + 1,)
            r_edges: Bin edges for radial position (m), shape (n_r_bins + 1,)
            phi_edges: Bin edges for azimuthal angle (rad), shape (n_phi_bins + 1,)
            energy_error_map: 5D probability array shaped (ke_error, ke_start, theta_center, r, phi)
        """
        self.name = name
        self.energy_error_map_path = energy_error_map_path
        
        # If path is provided, load from file
        if energy_error_map_path is not None:
            self._load_from_file(energy_error_map_path)
        else:
            # Use provided arrays
            self.ke_error_edges = (
                np.asarray(ke_error_edges) if ke_error_edges is not None else None
            )
            self.ke_start_edges = (
                np.asarray(ke_start_edges) if ke_start_edges is not None else None
            )
            self.theta_center_edges = (
                np.asarray(theta_center_edges)
                if theta_center_edges is not None
                else None
            )
            self.r_edges = np.asarray(r_edges) if r_edges is not None else None
            self.phi_edges = np.asarray(phi_edges) if phi_edges is not None else None
            self.energy_error_map = (
                np.asarray(energy_error_map) if energy_error_map is not None else None
            )

        # Validate that we have all necessary data
        if self.energy_error_map is None:
            raise ValueError(f"{name}: No energy error map provided")
        
        if any(
            edges is None
            for edges in [
                self.ke_error_edges,
                self.ke_start_edges,
                self.theta_center_edges,
                self.r_edges,
                self.phi_edges,
            ]
        ):
            raise ValueError(f"{name}: Missing bin edges")

        # Create bin centers for interpolation
        self.ke_error_centers = (self.ke_error_edges[:-1] + self.ke_error_edges[1:]) / 2
        self.ke_start_centers = (self.ke_start_edges[:-1] + self.ke_start_edges[1:]) / 2
        self.theta_center_centers = (
            self.theta_center_edges[:-1] + self.theta_center_edges[1:]
        ) / 2
        self.r_centers = (self.r_edges[:-1] + self.r_edges[1:]) / 2
        self.phi_centers = (self.phi_edges[:-1] + self.phi_edges[1:]) / 2

        # Validate energy_error_map shape
        expected_shape = (
            len(self.ke_error_centers),
            len(self.ke_start_centers),
            len(self.theta_center_centers),
            len(self.r_centers),
            len(self.phi_centers),
        )
        if self.energy_error_map.shape != expected_shape:
            raise ValueError(
                f"{name}: Energy error map shape {self.energy_error_map.shape} does not match "
                f"expected shape {expected_shape}"
            )

        # Verify normalization (distributions along axis 0 should sum to ~1)
        sums = self.energy_error_map.sum(axis=0)
        if not np.allclose(sums, 1.0, atol=1e-3):
            logger.warning(
                f"{name}: Energy error map normalization check failed. "
                f"Sum range: [{sums.min():.6f}, {sums.max():.6f}]. "
                f"Distributions should sum to 1.0 along ke_error axis."
            )

        logger.info(
            f"{name}: Initialized with energy error map shape {self.energy_error_map.shape}"
        )
        logger.info(
            f"{name}: ke_error range: [{self.ke_error_edges[0]:.2f}, {self.ke_error_edges[-1]:.2f}] eV"
        )
        logger.info(
            f"{name}: ke_start range: [{self.ke_start_edges[0]:.2f}, {self.ke_start_edges[-1]:.2f}] eV"
        )

    def _load_from_file(self, path: str):
        """Load energy error map from a .npy file."""
        try:
            data = np.load(path, allow_pickle=True).item()
            self.energy_error_map = data["energy_error_map"]
            self.ke_error_edges = data["ke_error_edges"]
            self.ke_start_edges = data["ke_start_edges"]
            self.theta_center_edges = data["theta_center_edges"]
            self.r_edges = data["r_edges"]
            self.phi_edges = data["phi_edges"]
            logger.info(f"{self.name}: Loaded energy error map from {path}")
        except Exception as e:
            raise RuntimeError(
                f"{self.name}: Failed to load energy error map from {path}: {e}"
            )

    def _find_nearest_bin(
        self,
        ke_start: np.ndarray,
        theta_center: np.ndarray,
        r_start: np.ndarray,
        phi_start: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find nearest bin indices for given event parameters in 4D phase space.
        Uses nearest-neighbor lookup without interpolation.

        Parameters:
            ke_start: True kinetic energy values (eV), shape (N,)
            theta_center: Pitch angle at center (rad), shape (N,)
            r_start: Starting radial position (m), shape (N,)
            phi_start: Starting azimuthal angle (rad), shape (N,)

        Returns:
            Tuple of (ke_start_idx, theta_center_idx, r_idx, phi_idx), each shape (N,)
            Bin indices for nearest neighbor in 4D phase space
        """
        n_events = len(ke_start)
        
        # Find nearest bin for each dimension
        ke_start_idx = np.searchsorted(self.ke_start_centers, ke_start)
        ke_start_idx = np.clip(ke_start_idx, 0, len(self.ke_start_centers) - 1)
        
        theta_center_idx = np.searchsorted(self.theta_center_centers, theta_center)
        theta_center_idx = np.clip(theta_center_idx, 0, len(self.theta_center_centers) - 1)
        
        r_idx = np.searchsorted(self.r_centers, r_start)
        r_idx = np.clip(r_idx, 0, len(self.r_centers) - 1)
        
        phi_idx = np.searchsorted(self.phi_centers, phi_start)
        phi_idx = np.clip(phi_idx, 0, len(self.phi_centers) - 1)
        
        return ke_start_idx, theta_center_idx, r_idx, phi_idx

    def _build_cdf(self, pdf: np.ndarray) -> np.ndarray:
        """
        Build cumulative distribution function from a normalized 1D probability distribution.

        Parameters:
            pdf: 1D probability distribution normalized to sum to 1.0, shape (n_bins,)

        Returns:
            CDF array where cdf[i] = sum(pdf[0:i+1]), shape (n_bins,)
        """
        cdf = np.cumsum(pdf)
        # Ensure CDF is monotonically increasing and ends at 1.0
        cdf = np.clip(cdf, 0, 1.0)
        if cdf[-1] < 1.0:
            cdf = cdf / cdf[-1]
        return cdf

    def _sample_from_cdf(
        self,
        cdf: np.ndarray,
        ke_error_edges: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        """
        Sample a single ke_error value continuously from a 1D PDF using its CDF.

        Parameters:
            cdf: Cumulative distribution function for pdf, shape (n_bins,)
            ke_error_edges: Bin edges for linear interpolation, shape (n_bins+1,)
            rng: Random number generator

        Returns:
            Sampled ke_error value (float)
        """
        # Sample uniform random number in [0, 1)
        u = rng.random()
        
        # Find the bin where CDF crosses u
        bin_idx = np.searchsorted(cdf, u, side='right')
        bin_idx = np.clip(bin_idx, 0, len(cdf) - 1)
        
        # Linear interpolation within the bin for continuous sampling
        # If u is between cdf[bin_idx-1] and cdf[bin_idx], interpolate within bin bin_idx
        if bin_idx > 0:
            cdf_low = cdf[bin_idx - 1]
            cdf_high = cdf[bin_idx]
        else:
            cdf_low = 0.0
            cdf_high = cdf[bin_idx] if cdf[bin_idx] > 0 else 1.0
        
        # Fraction within the bin
        if cdf_high > cdf_low:
            frac = (u - cdf_low) / (cdf_high - cdf_low)
        else:
            frac = 0.5  # Fallback if bin has zero width
        
        frac = np.clip(frac, 0.0, 1.0)
        
        # Interpolate between bin edges
        ke_error = ke_error_edges[bin_idx] + frac * (ke_error_edges[bin_idx + 1] - ke_error_edges[bin_idx])
        
        return ke_error

    def sample_energy_error(
        self,
        ke_start: np.ndarray,
        theta_center: np.ndarray,
        r_start: np.ndarray,
        phi_start: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample energy errors for given event parameters using nearest-neighbor bin lookup
        and continuous CDF sampling.

        This is the main method for applying energy errors in DataGenerator4D.
        For each event, it:
        1. Finds the nearest bin in 4D phase space (nearest-neighbor, no interpolation)
        2. Selects the 1D ke_error PDF from that bin
        3. Samples ke_error continuously from the CDF of that PDF

        Parameters:
            ke_start: True kinetic energy values (eV), shape (N,)
            theta_center: Pitch angle at center (rad), shape (N,)
            r_start: Starting radial position (m), shape (N,)
            phi_start: Starting azimuthal angle (rad), shape (N,)
            rng: Random number generator (default: np.random.default_rng())

        Returns:
            Sampled energy errors (eV), shape (N,)
            These should be added to ke_start to get observed energies:
            ke_observed = ke_start + ke_error
        """
        if rng is None:
            rng = np.random.default_rng()

        n_events = len(ke_start)
        
        # Find nearest bins in 4D phase space
        ke_start_idx, theta_center_idx, r_idx, phi_idx = self._find_nearest_bin(
            ke_start, theta_center, r_start, phi_start
        )
        
        # Sample energy errors using nearest-neighbor PDFs and continuous CDF
        ke_errors = np.zeros(n_events)
        for i in range(n_events):
            # Get the 1D PDF from the nearest bin in 4D space
            pdf = self.energy_error_map[:, ke_start_idx[i], theta_center_idx[i], r_idx[i], phi_idx[i]]
            
            # Build CDF for this PDF
            cdf = self._build_cdf(pdf)
            
            # Sample from CDF continuously
            ke_errors[i] = self._sample_from_cdf(cdf, self.ke_error_edges, rng)

        return ke_errors
