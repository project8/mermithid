"""
The cavity field implementation that reads from a field map file or a provided
array.

Author: S. M. Lee
First Date: November 21, 2025
Last Update: December 01, 2025
"""

from __future__ import absolute_import
import abc
import six
from typing import Dict, List, Optional, Union

import os
import numpy as np

# TODO: scipy is old. Newer version has more interpolation options.
# TODO: and Rbf is not intuitive. Find an alternative, such as linear interpolation.
from scipy.interpolate import Rbf, LinearNDInterpolator

from morpho.utilities import morphologging
from .CavityField import CavityField

logger = morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)  # type: ignore


class Numeric(CavityField):
    """
    The cavity field implementation that reads from a field map file or a provided
    array.

    Parameters:
        name: The name of the instance
    """

    def __init__(
        self,
        name,
        path: Optional[str] = None,
        r_edges: Optional[np.ndarray] = None,  # (m)
        z_edges: Optional[np.ndarray] = None,  # (rad)
        Bz_map: Optional[np.ndarray] = None,  # (T)
        **kwargs,
    ):
        """
        Initialize the Numerical cavity field. If a path is provided, it reads the
        field map from the file and ignore the provided arrays.

        Parameters:
            name: The name of the instance
            path: Optional path to the field map file
            r_edges: Optional array of radial edges (m)
            z_edges: Optional array of z edges (m)
            Bz_map: Optional magnetic field map array (T)

        Results:
            None
        """
        super(Numeric, self).__init__(name, **kwargs)
        logger.debug("Creating CavityField.Numeric <{}>".format(self._fieldName))

        # Initialize
        self._r_edges: Optional[np.ndarray] = r_edges  # (m) (bins_r + 1,)
        self._z_edges: Optional[np.ndarray] = z_edges  # (m) (bins_z + 1,)

        # _Bz_map[i, j]: Bz at z at _z_edges[i:i+1] and r at _r_edges[j:j+1]
        self._Bz_map: Optional[np.ndarray] = Bz_map  # (T) (bins_z, bins_r)

        # required preprocessed data
        self._Bz_interp: Optional[Union[Rbf, LinearNDInterpolator]] = None  # Callable interpolator (r, z) -> Bz
        self._Bz_max_binned: Optional[np.ndarray] = None  # (T) (bins_r,)
        self._Bz_max_interp: Optional[Union[Rbf, LinearNDInterpolator]] = None  # Callable interpolator r -> Bz_max

        self.Bz_thr_exceed_count = 0  # count of Bz_thr > Bz_max occurrences

        # Load from file if path is provided
        if path is not None:
            # TODO: if the provided path is only the file name, search in default directories
            if not self.ReadFieldMap(path):
                logger.error(
                    "Failed to initialize Numerical cavity field from path <{}>.".format(
                        path
                    )
                )
                return

        elif r_edges is None or z_edges is None or Bz_map is None:
            logger.error(
                "Numerical cavity field requires either a path or all of r_edges, "
                "z_edges, and Bz_map to be provided."
            )
            return

        # Preprocess the field map for efficient evaluation
        if not self.PreprocessFieldMap():
            logger.error(
                "Failed to preprocess the Numerical cavity field map for <{}>.".format(
                    self._fieldName
                )
            )
            return

        return

    def ReadFieldMap(self, path: str) -> bool:
        """
        Read the CCA magnetic field map from a file.

        Results:
            True if successful.
        """
        # TODO: Implement support for different file formats if needed
        try:
            with np.load(path) as data:
                # TODO: Coordinate with the data format
                self._r_edges = data["r_edges"]  # (m)
                self._z_edges = data["z_edges"]  # (rad)
                self._Bz_map = data["Bz_map"]  # (T)

                if (
                    type(self._r_edges) is not np.ndarray
                    or type(self._z_edges) is not np.ndarray
                    or type(self._Bz_map) is not np.ndarray
                ):
                    msg = "Field map arrays must be numpy arrays."
                    raise ValueError(msg)

                if (
                    self._r_edges.ndim != 1
                    or self._z_edges.ndim != 1
                    or self._Bz_map.ndim != 2
                ):
                    msg = "Field map arrays have incorrect dimensions."
                    raise ValueError(msg)

                if self._Bz_map.shape != (
                    self._z_edges.size - 1,
                    self._r_edges.size - 1,
                ):
                    msg = "Bz_map shape does not match r_edges and z_edges."
                    raise ValueError(msg)

            logger.info("Numerical field map loaded from <{}>.".format(path))
            return True

        except Exception as e:
            logger.error("Failed to read field map from <{}>: {}".format(path, e))
            return False

    def PreprocessFieldMap(
        self, interpolator_kwargs: dict = {"function": "linear", "smooth": 0}
    ) -> bool:
        """
        Preprocess the field map for efficient evaluation. Methods to declare:
            - $B_{z, max}(r)$: which is maximum $B_{z}$ for given r
            - $z_{thr}^{upper/lower}(r, \\theta)$: which gives z position where
              $B_{z}(r, z_{thr})=B_{z, max}(r) sin^2 \\theta$.
            - $z_{max}^{upper/lower}(r)$: which gives maximum $B_{z}(z)$ for given r

        Results:
            True if successful.
        """
        try:
            # Check if field map is loaded
            if self._r_edges is None or self._z_edges is None or self._Bz_map is None:
                logger.error("Field map is not loaded; cannot preprocess.")
                raise RuntimeError("Field map is not loaded.")

            # B_z interpolator
            r_centers = 0.5 * (self._r_edges[:-1] + self._r_edges[1:])  # (m) (bins_r,)
            z_centers = 0.5 * (self._z_edges[:-1] + self._z_edges[1:])  # (m) (bins_z,)
            r_mesh, z_mesh = np.meshgrid(r_centers, z_centers)  # (bins_z, bins_r)

            points = np.vstack([r_mesh.flatten(), z_mesh.flatten()]).T  # (bins_z * bins_r, 2)

            interpolator_kwargs = {"fill_value": self._Bz_map.min(), "rescale": False}

            self._Bz_interp = LinearNDInterpolator(
                points,
                self._Bz_map.flatten(),
                **interpolator_kwargs,
            )

            # B_z_max (r)
            # TODO: does it have to be min(B_max_upper, B_max_lower)?
            # TODO: It is not being used currently.
            self._Bz_max_binned = np.max(self._Bz_map, axis=0)  # (T) (bins_r,)
            self._Bz_max_interp = Rbf(
                r_centers,
                self._Bz_max_binned,
                **interpolator_kwargs,
            )

            logger.info("Numerical field map preprocessing completed.")
            return True

        except Exception as e:
            logger.error(f"Failed to preprocess field map: {e}")
            return False

    def GetZArgmax(
        self,
        r: Union[float, np.ndarray],  # (m)
        z_center: float = 0.0,  # (m)
    ) -> np.ndarray:
        """
        Calculate the z position(s) where Bz(r, z_argmax) = Bz_max(r).

        Parameters:
            r: (N,) shaped array or (N, M) mesh of radial position(s) (m)
            z_center: The center z position (m)

        Results:
            (N, 2) array or (N, M, 2) mesh of the z position(s) (m). The last
            dimension corresponds to (z_max_lower, z_max_upper).
        """
        # check if preprocessed data is available
        if (
            self._Bz_max_interp is None
            or self._Bz_interp is None
            or self._z_edges is None
        ):
            msg = "Preprocessed field map data is not available."
            logger.error(msg)
            raise RuntimeError(msg)

        if ~np.any(self._z_edges > z_center) or ~np.any(self._z_edges < z_center):
            msg = "z_edges must span both sides of z_center."
            logger.error(msg)
            raise ValueError(msg)

        r = np.asarray(r)

        # Find z_max by searching in z direction
        # TODO: optimize this part if needed. Multithreading? Vectorization?
        z_max_shape = r.shape + (2,)  # (N, 2) or (N, M, 2)
        z_max = np.full(z_max_shape, np.nan)  # (N, 2) or (N, M, 2)

        r_flat = r.flatten()  # (N,) or (N*M,)
        z_max_flat = z_max.reshape(-1, 2)  # (N,) or (N*M, 2)
        for i, r_i in enumerate(r_flat):
            # Search in z direction
            Bz_values = self._Bz_interp(
                r_i * np.ones_like(self._z_edges), self._z_edges
            )  # (bins_z + 1,)

            # for z >= z_center
            upper_mask = self._z_edges >= z_center
            z_upper = self._z_edges[upper_mask]
            Bz_upper = Bz_values[upper_mask]

            z_upper_max = z_upper[np.argmax(Bz_upper)]

            # for z < z_center
            lower_mask = self._z_edges < z_center
            z_lower = self._z_edges[lower_mask]
            Bz_lower = Bz_values[lower_mask]

            z_lower_max = z_lower[np.argmax(Bz_lower)]

            z_max_flat[i, 0] = z_lower_max
            z_max_flat[i, 1] = z_upper_max

        z_max = z_max_flat.reshape(z_max_shape)  # (N, 2) or (N, M, 2)
        return z_max

    def GetZThreshold(
        self,
        r: Union[float, np.ndarray],  # (m)
        theta: Union[float, np.ndarray],  # (rad)
        z_center: float = 0.0,  # (m)
    ) -> np.ndarray:
        """
        Calculate the z position(s) where Bz(r, z_thr) = Bz_max(r) * sin^2(theta).

        Parameters:
            r: (N,) shaped array or (N, M) mesh of radial position(s) (m)
            theta: (N,) shaped array or (N, M) mesh of pitch angle(s) (radians)
            z_center: The center z position (m)

        Results:
            (N, 2) array or (N, M, 2) mesh of the z position(s) (m). The last
            dimension corresponds to (z_thr_lower, z_thr_upper).
        """
        # check if preprocessed data is available
        if (
            self._Bz_max_interp is None
            or self._Bz_interp is None
            or self._z_edges is None
        ):
            msg = "Preprocessed field map data is not available."
            logger.error(msg)
            raise RuntimeError(msg)

        if ~np.any(self._z_edges > z_center) or ~np.any(self._z_edges < z_center):
            msg = "z_edges must span both sides of z_center."
            logger.error(msg)
            raise ValueError(msg)

        r = np.asarray(r)
        theta = np.asarray(theta)
        if r.shape != theta.shape:
            msg = "r and theta must have the same shape."
            logger.error(msg)
            raise ValueError(msg)

        # Evaluate Bz_max and Bz_thr
        # TODO: not using self._Bz_max_interp
        # Bz_max = self._Bz_max_interp(r.flatten()).reshape(r.shape)  # (N,) or (N, M)
        # Bz_thr = Bz_max * np.sin(theta) ** 2  # (N,) or (N, M)

        # Find z_thr by searching in z direction
        # TODO: optimize this part if needed. Multithreading? Vectorization?
        z_thr_shape = r.shape + (2,)  # (N, 2) or (N, M, 2)
        z_thr = np.full(z_thr_shape, np.nan)  # (N, 2) or (N, M, 2)

        r_flat = r.flatten()  # (N,) or (N*M,)
        theta_flat = theta.flatten()  # (N,) or (N*M,)
        # Bz_thr_flat = Bz_thr.flatten()
        z_thr_flat = z_thr.reshape(-1, 2)  # (N,) or (N*M, 2)
        # for i, (r_i, Bz_thr_i) in enumerate(zip(r_flat, Bz_thr_flat)):
        for i, (r_i, theta_i) in enumerate(zip(r_flat, theta_flat)):
            # Search in z direction
            Bz_values = self._Bz_interp(
                r_i * np.ones_like(self._z_edges), self._z_edges
            )  # (bins_z + 1,)
            Bz_thr_i = Bz_values.max() * np.sin(theta_i) ** 2  # scalar

            # Bz_thr is too low
            if Bz_thr_i < Bz_values.min() or Bz_thr_i < self._Bz_interp(r_i, z_center):
                z_thr_flat[i, 0] = z_center
                z_thr_flat[i, 1] = z_center
                continue

            # for z >= z_center
            upper_mask = self._z_edges >= z_center
            z_upper = self._z_edges[upper_mask]
            Bz_upper = Bz_values[upper_mask]

            # find the first z where Bz - Bz_thr changes sign
            above_indices = np.where(Bz_upper >= Bz_thr_i)[0]
            if above_indices.size > 0:
                first_above = above_indices[0]
                if first_above == 0:
                    z_thr_upper = z_upper[0]
                else:
                    # Linear interpolation
                    # TODO: use a better interpolation method if needed
                    z1 = z_upper[first_above - 1]
                    z2 = z_upper[first_above]
                    B1 = Bz_upper[first_above - 1]
                    B2 = Bz_upper[first_above]
                    z_thr_upper = z1 + (Bz_thr_i - B1) * (z2 - z1) / (B2 - B1)
                z_thr_flat[i, 1] = z_thr_upper
            else:
                self.Bz_thr_exceed_count += 1
                z_thr_flat[i, 1] = np.nan  # no crossing found

                if self.Bz_thr_exceed_count == 1:
                    logger.warning(
                        f"Bz_thr can not exceed Bz_max. thr={Bz_thr_i:.2f}, max={Bz_values.max():.2f}"
                    )

            # for z < z_center
            lower_mask = self._z_edges < z_center
            z_lower = self._z_edges[lower_mask]
            Bz_lower = Bz_values[lower_mask]

            # find the last z where Bz - Bz_thr changes sign
            below_indices = np.where(Bz_lower >= Bz_thr_i)[0]
            if below_indices.size > 0:
                last_below = below_indices[-1]
                if last_below == z_lower.size - 1:
                    z_thr_lower = z_lower[-1]
                else:
                    # Linear interpolation
                    # TODO: use a better interpolation method if needed
                    z1 = z_lower[last_below]
                    z2 = z_lower[last_below + 1]
                    B1 = Bz_lower[last_below]
                    B2 = Bz_lower[last_below + 1]
                    z_thr_lower = z1 + (Bz_thr_i - B1) * (z2 - z1) / (B2 - B1)
                z_thr_flat[i, 0] = z_thr_lower
            else:
                self.Bz_thr_exceed_count += 1
                z_thr_flat[i, 0] = np.nan  # no crossing found

                if self.Bz_thr_exceed_count == 1:
                    logger.warning(
                        f"Bz_thr can not exceed Bz_max. thr={Bz_thr_i:.2f}, max={Bz_values.max():.2f}"
                    )

        z_thr = z_thr_flat.reshape(z_thr_shape)  # (N, 2) or (N, M, 2)
        return z_thr

    def GetEfficiency(
        self,
        r: Union[float, np.ndarray],  # (m)
        theta: Union[float, np.ndarray],  # (rad)
        z_center: float = 0.0,  # (m)
    ) -> np.ndarray:
        """
        Calculate the trapping efficiency for given (r, theta) samples.

        Parameters:
            r: (N,) shaped array or (N, M) mesh of radial position(s) (m)
            theta: (N,) shaped array or (N, M) mesh of pitch angle(s) (radians)
            z_center: The center z position (m)

        Results:
            (N,) array or (N, M) mesh of the trapping efficiency value(s).
        """
        # check inputs
        r = np.asarray(r)
        theta = np.asarray(theta)
        input_shape = r.shape

        if r.shape != theta.shape:
            msg = "r and theta must have the same shape."
            logger.error(msg)
            raise ValueError(msg)

        r = r.flatten()  # (N,) or (N*M,)
        theta = theta.flatten()  # (N,) or (N*M,)

        # Calculate z_argmax and z_thr
        msg = "Calculating z_argmax for efficiency evaluation."
        logger.debug(msg)
        z_argmax = self.GetZArgmax(r, z_center)  # (N, 2) or (N*M, 2)

        msg = "Calculating z_thr for efficiency evaluation."
        logger.debug(msg)
        z_thr = self.GetZThreshold(r, theta, z_center)  # (N, 2) or (N*M, 2)

        # Calculate efficiency
        msg = "Calculating trapping efficiency."
        logger.debug(msg)

        efficiency = np.zeros(r.shape)  # (N,) or (N*M,)
        valid_mask = ~np.isnan(z_argmax).any(axis=1) & ~np.isnan(z_thr).any(axis=1)
        efficiency[valid_mask] = (z_thr[valid_mask, 1] - z_thr[valid_mask, 0]) / (
            z_argmax[valid_mask, 1] - z_argmax[valid_mask, 0]
        )  # (N,) or (N*M,)

        efficiency[np.isnan(efficiency)] = 0.0
        efficiency = np.clip(efficiency, 0.0, 1.0)

        efficiency = efficiency.reshape(input_shape)
        return efficiency
