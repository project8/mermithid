"""
DetectionEfficiency class for DataGenerator4D.
Handles loading and applying 4D detection efficiency map.

Author: S. M. Lee
Date: March 7, 2026
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
        """
        self.name = name
        self.efficiency_map_path = efficiency_map_path
        
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

        # Create interpolator
        self.interpolator = RegularGridInterpolator(
            (
                self.ke_centers,
                self.theta_center_centers,
                self.r_centers,
                self.phi_centers,
            ),
            self.efficiency,
            bounds_error=False,
            fill_value=0.0,
        )

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
        )

        # Interpolate efficiency
        efficiency = self.interpolator(points)

        # Ensure efficiency is in [0, 1]
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
