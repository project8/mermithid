"""
DetectionEfficiency class for DataGenerator4D.
Handles loading and applying 4D detection efficiency map.

Author: S. M. Lee
First Date: March 7, 2025
Last Update: July 20, 2026
"""

from __future__ import absolute_import

from typing import Dict, Optional, Union, List
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from morpho.utilities import morphologging

logger = morphologging.getLogger(__name__)


class DetectionEfficiency:
    """
    Manages detection efficiency for the DataGenerator4D.
    
    The detection efficiency is a 4D map defined over:
    - ke: kinetic energy
    - theta_center: pitch angle at center
    - r_start: starting radial position
    - phi_start: starting azimuthal angle
    
    The efficiency map is loaded from a file and used to compute
    detection probabilities for sampled events.
    """

    def __init__(
        self,
        name: str,
        efficiency_map_path: Optional[str] = None,
        ke_edges: Optional[np.ndarray] = None,
        theta_center_edges: Optional[np.ndarray] = None,
        r_edges: Optional[np.ndarray] = None,
        phi_edges: Optional[np.ndarray] = None,
        efficiency: Optional[np.ndarray] = None,
        nan_fill_value: Optional[float] = None,
    ):
        """
        Initialize the DetectionEfficiency.

        Parameters:
            name: Name of the efficiency instance
            efficiency_map_path: Path to .npy file containing efficiency map
            ke_edges: Bin edges for kinetic energy (eV)
            theta_center_edges: Bin edges for theta_center (rad)
            r_edges: Bin edges for radial position (m)
            phi_edges: Bin edges for azimuthal angle (rad)
            efficiency: 4D efficiency array (ke, theta_center, r, phi)
            nan_fill_value: Value to fill for NaN entries in efficiency map. If
            None, NaNs will raise an error.
        """
        self.name = name
        self.efficiency_map_path = efficiency_map_path
        self.nan_fill_value = nan_fill_value
        
        # If path is provided, load from file
        if efficiency_map_path is not None:
            self._load_from_file(efficiency_map_path)
        else:
            # Use provided arrays
            self.ke_edges = np.asarray(ke_edges) if ke_edges is not None else None
            self.theta_center_edges = (
                np.asarray(theta_center_edges)
                if theta_center_edges is not None
                else None
            )
            self.r_edges = np.asarray(r_edges) if r_edges is not None else None
            self.phi_edges = np.asarray(phi_edges) if phi_edges is not None else None
            self.efficiency = np.asarray(efficiency) if efficiency is not None else None

        # Validate that we have all necessary data
        if self.efficiency is None:
            raise ValueError(f"{name}: No efficiency data provided")
        
        if any(
            edges is None
            for edges in [
                self.ke_edges,
                self.theta_center_edges,
                self.r_edges,
                self.phi_edges,
            ]
        ):
            raise ValueError(f"{name}: Missing bin edges")

        # Create bin centers for interpolation
        self.ke_centers = (self.ke_edges[:-1] + self.ke_edges[1:]) / 2
        self.theta_center_centers = (
            self.theta_center_edges[:-1] + self.theta_center_edges[1:]
        ) / 2
        self.r_centers = (self.r_edges[:-1] + self.r_edges[1:]) / 2
        self.phi_centers = (self.phi_edges[:-1] + self.phi_edges[1:]) / 2

        # Validate efficiency shape
        expected_shape = (
            len(self.ke_centers),
            len(self.theta_center_centers),
            len(self.r_centers),
            len(self.phi_centers),
        )
        if self.efficiency.shape != expected_shape:
            raise ValueError(
                f"{name}: Efficiency shape {self.efficiency.shape} does not match "
                f"expected shape {expected_shape}"
            )

        # Fill NaN values if a fill value is provided
        if self.nan_fill_value is not None:
            self.efficiency = np.where(np.isnan(self.efficiency), self.nan_fill_value, self.efficiency)
        else:
            if np.isnan(self.efficiency).any():
                raise ValueError(f"{name}: Efficiency map contains NaN values and no fill value was provided")

        # Using interpolator
        # FIXME: it returns 0 for points outside the grid, made of centers.
        # self.efficiency_fcn = RegularGridInterpolator(
        #     (
        #         self.ke_centers,
        #         self.theta_center_centers,
        #         self.r_centers,
        #         self.phi_centers,
        #     ),
        #     self.efficiency,
        #     bounds_error=False,
        #     fill_value=0.0,
        # )

        def _grid_value_getter(points):
            """
            Find the bin and return the corresponding efficiency value for each
            point, without any interpolation. Points outside the bin edges will
            return 0 efficiency.
            """
            # Unpack points
            ke, theta_c, r, phi = points.T  # 4 * (N,)

            # Find bin indices
            ke_idx = np.searchsorted(self.ke_edges, ke, side="right") - 1  # (N,)
            theta_c_idx = np.searchsorted(self.theta_center_edges, theta_c, side="right") - 1  # (N,)
            r_idx = np.searchsorted(self.r_edges, r, side="right") - 1  # (N,)
            phi_idx = np.searchsorted(self.phi_edges, phi, side="right") - 1  # (N,)

            mask_valid = (
                (ke_idx >= 0) & (ke_idx < len(self.ke_centers))
                & (theta_c_idx >= 0) & (theta_c_idx < len(self.theta_center_centers))
                & (r_idx >= 0) & (r_idx < len(self.r_centers))
                & (phi_idx >= 0) & (phi_idx < len(self.phi_centers))
            )
            efficiency_values = np.zeros(len(points))
            efficiency_values[mask_valid] = self.efficiency[
                ke_idx[mask_valid],
                theta_c_idx[mask_valid],
                r_idx[mask_valid],
                phi_idx[mask_valid],
            ]
            
            return efficiency_values

        self.efficiency_fcn = _grid_value_getter

        logger.info(
            f"{name}: Initialized with efficiency map shape {self.efficiency.shape}"
        )
        logger.info(
            f"{name}: Efficiency range: [{self.efficiency.min():.3f}, "
            f"{self.efficiency.max():.3f}], mean: {self.efficiency.mean():.3f}"
        )

    def _load_from_file(self, path: str):
        """Load efficiency map from a .npy file."""
        try:
            data = np.load(path, allow_pickle=True).item()
            self.efficiency = data["efficiency"]
            self.ke_edges = data["ke_edges"]
            self.theta_center_edges = data["theta_center_edges"]
            self.r_edges = data["r_edges"]
            self.phi_edges = data["phi_edges"]
            logger.info(f"{self.name}: Loaded efficiency map from {path}")
        except Exception as e:
            raise RuntimeError(
                f"{self.name}: Failed to load efficiency map from {path}: {e}"
            )

    def get_efficiency(
        self,
        ke: np.ndarray,
        theta_center: np.ndarray,
        r_start: np.ndarray,
        phi_start: np.ndarray,
    ) -> np.ndarray:
        """
        Get detection efficiency for given event parameters.

        Parameters:
            ke: Kinetic energy values (eV), shape (N,)
            theta_center: Pitch angle at center (rad), shape (N,)
            r_start: Starting radial position (m), shape (N,)
            phi_start: Starting azimuthal angle (rad), shape (N,)

        Returns:
            Efficiency values, shape (N,)
        """
        # Stack coordinates for interpolation
        points = np.column_stack(
            [ke, theta_center, r_start, phi_start]
        )  # shape (N, 4)

        efficiency = self.efficiency_fcn(points)
        efficiency = np.clip(efficiency, 0, 1)

        return efficiency

    def apply_efficiency(
        self,
        ke: np.ndarray,
        theta_center: np.ndarray,
        r_start: np.ndarray,
        phi_start: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Apply detection efficiency by randomly accepting/rejecting events.

        Parameters:
            ke: Kinetic energy values (eV), shape (N,)
            theta_center: Pitch angle at center (rad), shape (N,)
            r_start: Starting radial position (m), shape (N,)
            phi_start: Starting azimuthal angle (rad), shape (N,)
            rng: Random number generator (default: np.random.default_rng())

        Returns:
            Boolean mask indicating which events are detected, shape (N,)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Get efficiency for each event
        efficiency = self.get_efficiency(
            ke, theta_center, r_start, phi_start
        )

        # Generate random numbers and compare with efficiency
        random_values = rng.random(len(efficiency))
        detected = random_values < efficiency

        return detected
