"""
The spatial variables sampling class.

Author: S. M. Lee
First Date: September 15, 2025
Last Update: July 20, 2026
"""

from __future__ import absolute_import
import abc
import six
from typing import Dict, List, Optional

import numpy as np

from . import CavityField

from morpho.utilities import morphologging

logger = morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)


@six.add_metaclass(abc.ABCMeta)
class SpatialSampler:
    """
    The spatial variables sampling class. The goal is to sample three spatial
    variables: pitch angle at the trap center (theta_center), cylindrical radial
    position at start (r_start), and azimuthal angle at start (phi_start). This
    will also sample the pitch angle at start (theta_start) and convert it to
    theta_center using the cavity field if available.

    Parameters:
        name: The name of the instance
    Results:
        The sampled pitch angle at the trap center (theta_center) and positions
        at start (r_start, phi_start) in rad, m, rad units are accessible via
        the `result` property, stored in `self._sample_position`.
    """

    def __init__(
        self,
        name,
        binned_mode: bool = False,
        theta_edges: Optional[np.ndarray] = None,  # (rad)
        theta_min: float = 0.0,  # (rad)
        theta_max: float = np.pi,  # (rad)
        theta_bins: int = 36,
        r_edges: Optional[np.ndarray] = None,  # (m)
        r_min: float = 0.0,  # (m)
        r_max: float = 0.1,  # (m)
        r_bins: int = 10,
        phi_edges: Optional[np.ndarray] = None,  # (rad)
        phi_min: float = 0.0,  # (rad)
        phi_max: float = 2 * np.pi,  # (rad)
        phi_bins: int = 36,
        cavity_field_option: str = "none",
        cavity_field_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        # Class information
        self._samplerName = name
        logger.debug("Creating Spectrum <{}>".format(self._samplerName))

        self.binned_mode = binned_mode

        self._sample_r_start: List[np.ndarray] = list()  # (m)
        self._sample_theta_start: List[np.ndarray] = list()  # (rad)
        self._sample_theta_center: List[np.ndarray] = list()  # (rad)
        self._sample_phi_start: List[np.ndarray] = list()  # (rad)
        self._sample_z_start: List[np.ndarray] = list()  # (m)

        self._sample_geometry: List[Dict[str, np.ndarray]] = (
            list()
        )  # shape: [{"theta_center": (N,), "r_start": (N,), "phi_start": (N,)}, ...]

        # ROI properties
        if theta_edges is None:
            theta_edges = np.linspace(theta_min, theta_max, theta_bins + 1)
        if not theta_edges.ndim == 1:
            logger.error("theta_edges must be 1-dimensional.")
            raise ValueError("theta_edges must be 1-dimensional.")
        if np.any(np.diff(theta_edges) <= 0):
            logger.error("theta_edges must be in ascending order.")
            raise ValueError("theta_edges must be in ascending order.")

        if r_edges is None:
            r_edges = np.linspace(r_min, r_max, r_bins + 1)
        if not r_edges.ndim == 1:
            logger.error("r_edges must be 1-dimensional.")
            raise ValueError("r_edges must be 1-dimensional.")
        if np.any(np.diff(r_edges) <= 0):
            logger.error("r_edges must be in ascending order.")
            raise ValueError("r_edges must be in ascending order.")

        if phi_edges is None:
            phi_edges = np.linspace(phi_min, phi_max, phi_bins + 1)
        if not phi_edges.ndim == 1:
            logger.error("phi_edges must be 1-dimensional.")
            raise ValueError("phi_edges must be 1-dimensional.")
        if np.any(np.diff(phi_edges) <= 0):
            logger.error("phi_edges must be in ascending order.")
            raise ValueError("phi_edges must be in ascending order.")

        self._theta_edges: np.ndarray = np.asarray(theta_edges)
        self._r_edges: np.ndarray = np.asarray(r_edges)
        self._phi_edges: np.ndarray = np.asarray(phi_edges)
        self._theta_centers: np.ndarray = 0.5 * (
            self._theta_edges[:-1] + self._theta_edges[1:]
        )  # (bins_theta,)
        self._r_centers: np.ndarray = 0.5 * (
            self._r_edges[:-1] + self._r_edges[1:]
        )  # (bins_r,)
        self._phi_centers: np.ndarray = 0.5 * (
            self._phi_edges[:-1] + self._phi_edges[1:]
        )  # (bins_phi,)
        self._theta_bins: int = len(self._theta_centers)
        self._r_bins: int = len(self._r_centers)
        self._phi_bins: int = len(self._phi_centers)

        # PDF for binned mode. [i, j] element corresponds to r bin i and theta bin j.
        self.r_theta_pdf: Optional[np.ndarray] = None  # (bins_r, bins_theta)
        self.z_thr: Optional[np.ndarray] = None  # (bins_r, bins_theta, 2)

        # set the cavity_field object for trapping efficiency and theta conversion
        self.cavity_field: Optional[CavityField.CavityField] = None
        if cavity_field_option == "numeric" and cavity_field_kwargs is not None:
            self.cavity_field = CavityField.Numeric(
                name="CavityField_{}".format(self._samplerName),
                **cavity_field_kwargs,
            )
        elif cavity_field_option == "analytic":
            logger.error("Analytic cavity field option is not implemented yet.")
            raise NotImplementedError(
                "Analytic cavity field option is not implemented yet."
            )
        elif cavity_field_option == "none":
            self.cavity_field = None
        else:
            logger.error("Invalid cavity_field_option: {}".format(cavity_field_option))
            raise ValueError(
                "Invalid cavity_field_option: {}".format(cavity_field_option)
            )

    @property
    def name(self):
        return self._samplerName

    @property
    def theta_edges(self) -> Optional[np.ndarray]:
        if self.binned_mode:
            return None
        return self._theta_edges

    @property
    def r_edges(self) -> Optional[np.ndarray]:
        if self.binned_mode:
            return None
        return self._r_edges

    @property
    def phi_edges(self) -> Optional[np.ndarray]:
        if self.binned_mode:
            return None
        return self._phi_edges

    @property
    def cavityField(self) -> Optional[CavityField.CavityField]:
        """
        Returns the cavity field object.
        """
        return self.cavity_field

    @property
    def result(self) -> List[Dict[str, np.ndarray]]:
        """
        Returns the list of sampled position arrays.
        Each dictionary corresponds to each runtime and contains four keys:
        "theta_center", "r_start", "phi_start", and "theta_start", each with
        shape (N,) where N is the number of samples for that runtime.
        """
        return self._sample_geometry

    def Sample(
        self,
        ke: List[np.ndarray],  # (eV) shape: [(entries,), ...]
    ) -> bool:
        """
        Sample position variables based on the binned or unbinned model and
        given kinetic energies.

        Returns:
            True if successful.
        """
        logger.info("Sampling position for <{}>...".format(self.name))

        self.ke = ke
        self._sample_r_start = [
            np.zeros((k.shape[0],), dtype="float64") for k in self.ke
        ]  # (m) shape: [(entries,)]
        self._sample_theta_start = [
            np.zeros((k.shape[0],), dtype="float64") for k in self.ke
        ]  # (rad) shape: [(entries,)]
        self._sample_theta_center = [
            np.zeros((k.shape[0],), dtype="float64") for k in self.ke
        ]  # (rad) shape: [(entries,)]
        self._sample_phi_start = [
            np.zeros((k.shape[0],), dtype="float64") for k in self.ke
        ]  # (rad) shape: [(entries,)]
        self._sample_z_start = [
            np.zeros((k.shape[0],), dtype="float64") for k in self.ke
        ]  # (m) shape: [(entries,)]

        if self.binned_mode:
            if not self.BuildThetaRPDF():
                logger.error("Error while <{}> building theta-r PDF".format(self.name))
                return False
            if not self.BinnedSampleThetaR():
                logger.error("Error while <{}> sampling theta and r".format(self.name))
                return False
            if not self.BinnedSamplePhi():
                logger.error("Error while <{}> sampling phi".format(self.name))
                return False
        else:
            raise NotImplementedError("Unbinned sampling is not implemented yet.")
            # TODO: implement unbinned sampling
            # if not self.UnbinnedSampleTheta():
            #     logger.error("Error while <{}> sampling theta".format(self.name))
            #     return False
            # if not self.UnbinnedSampleR():
            #     logger.error("Error while <{}> sampling r".format(self.name))
            #     return False
            # if not self.UnbinnedSamplePhi():
            #     logger.error("Error while <{}> sampling phi".format(self.name))
            #     return False

        self._sample_geometry = list()
        for i in range(len(self.ke)):
            sample_dict = dict()
            sample_dict["theta_center"] = self._sample_theta_center[i]
            sample_dict["r_start"] = self._sample_r_start[i]
            sample_dict["phi_start"] = self._sample_phi_start[i]
            sample_dict["theta_start"] = self._sample_theta_start[i]
            self._sample_geometry.append(sample_dict)

        logger.info("Done sampling position for <{}>".format(self.name))
        return True

    @abc.abstractmethod
    def BuildThetaRPDF(self) -> bool:
        """
        Build the joint PDF of theta_start and r_start. It assumes independence
        from energy.

        PDF: p(theta_start, r_start) dr_start dtheta_start
            = p(r_start, theta_start) p(theta_start) dr_start dtheta_start

        Results:
            True if successful.
        """
        raise NotImplementedError("BuildThetaRPDF method is not implemented.")

    @abc.abstractmethod
    def BinnedSampleThetaR(self) -> bool:
        """
        Sample theta_start and r_start from the binned joint PDF p(theta, r).
        Convert theta_start to theta_center using the cavity field if available.

        Results:
            True if successful.
        """
        raise NotImplementedError("BinnedSampleThetaR method is not implemented.")

    @abc.abstractmethod
    def BinnedSamplePhi(self) -> bool:
        """
        Sample phi_start from the binned pdf p(phi_start|theta_start, r_start).

        Results:
            True if successful.
        """
        raise NotImplementedError("BinnedSamplePhi method is not implemented.")

    # @abc.abstractmethod
    # def UnbinnedSampleTheta(self) -> bool:
    #     """
    #     Method called by `Sample()` to sample the polar angle from the unbinned model.
    #     The marginal distribution p(theta_start|E) should be used.
    #     Must be overridden by child class.

    #     Results:
    #         True if successful.
    #     """
    #     raise NotImplementedError("UnbinnedSampleTheta method is not implemented.")

    # @abc.abstractmethod
    # def UnbinnedSampleR(self) -> bool:
    #     """
    #     Method called by `Sample()` to sample the radius from the unbinned model.
    #     The marginal distribution p(r_start|E, theta_start) should be used.
    #     Must be overridden by child class.

    #     Results:
    #         True if successful.
    #     """
    #     raise NotImplementedError("UnbinnedSampleR method is not implemented.")

    # @abc.abstractmethod
    # def UnbinnedSamplePhi(self) -> bool:
    #     """
    #     Method called by `Sample()` to sample the azimuthal angle from the unbinned model.
    #     The marginal distribution p(phi_start|E, theta_start, r_start) should be used.
    #     Must be overridden by child class.

    #     Results:
    #         True if successful.
    #     """
    #     raise NotImplementedError("UnbinnedSamplePhi method is not implemented.")
