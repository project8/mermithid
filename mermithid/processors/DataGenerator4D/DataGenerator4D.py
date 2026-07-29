"""
The data generator class for CCA.
Author: S. M. Lee
First Date: August 25, 2025
Last Update: July 20, 2026
"""

from __future__ import absolute_import

import json
import os
from collections import OrderedDict as od
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, OrderedDict

import numpy as np

from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor

from mermithid.misc.FakeTritiumDataFunctions import *

from . import EnergySampler
from . import SpatialSampler
from . import DetectionEfficiency
from . import EnergyError

logger = morphologging.getLogger(__name__)


__all__ = []
__all__.append(__name__)


class DataGenerator4D(BaseProcessor):
    """
    Generate pseudo electrons for CCA. It samples 4D data
    (ke_observed, theta_center, r_start, phi_start) from a given source and background
    model. This samples ke_start and theta_start together, which are internally
    used for the trapping_efficiency/energy_error application.

    Parameters:
        name: The name of the instance
        The other configurations are set by `InternalConfigure()` via the
        `super().Configure()` method.
    Results:
        `super().Run()` runs the 4D-data generation.
        The sampled 4D data are stored in `self.results`.
    """

    @staticmethod
    def _read_first_param(
        params: Dict[str, Any], keys: Sequence[str], default: Any = None
    ) -> Any:
        """
        Read the first available key from a list of aliases.
        """
        for key in keys:
            if key in params:
                return params[key]
        return default

    @staticmethod
    def _read_spatial_numerical_data(path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load histogram data either from a numpy file.
        Supported file extensions: .npy, .npz

        Returns:
            A dictionary of the loaded data, or None if the file could not be loaded.
        """
        if path is not None:
            if not os.path.exists(path):
                logger.error(f"Histogram file does not exist: {path}")
                return None

            ext = os.path.splitext(path)[1].lower()
            try:
                if ext == ".npy" or ext == ".npz":
                    with np.load(path) as file:
                        pdf = file["pdf"]
                        r_edges = file["r_edges"]
                        theta_edges = file["theta_edges"]
                        phi_edges = file["phi_edges"]
                    return {"pdf": pdf, "r_edges": r_edges, "theta_edges": theta_edges, "phi_edges": phi_edges}
                else:
                    logger.error(f"Unsupported histogram file extension: {ext}")
                    return None
            except (OSError, ValueError, json.JSONDecodeError) as error:
                logger.error(f"Failed to load histogram from {path}: {error}")
                return None

    def InternalConfigure(self, params):
        """
        Configure the `DataGenerator4D` instance.

        TODO: explain parameters
        """
        # Choose the source
        source_type_menu = ["mono", "numerical"]  # TODO: ["e-gun", "Kr"]
        self.source_types: List[str] = reader.read_param(
            params, "source_types", ["mono"]
        )
        for t in self.source_types:
            if not t in source_type_menu:
                logger.error(
                    f"DataGenerator {self._procName}: invalid source {t}."
                    + f" Available: {', '.join(source_type_menu)}."
                )
                return False

        # Choose the background
        bkgd_type_menu = ["flat", "numerical"]
        self.bkgd_types: List[str] = reader.read_param(params, "bkgd_types", [])
        for t in self.bkgd_types:
            if not t in bkgd_type_menu:
                logger.error(
                    f"DataGenerator {self._procName}: invalid background {t}."
                    + f" Available: {', '.join(bkgd_type_menu)}."
                )
                return False

        # Choose the spatial model
        spatial_model_menu = ["uniform_cylinder", "numerical"]  # TODO: other models
        self.spatial_model: str = reader.read_param(
            params, "spatial_model", "uniform_cylinder"
        )
        if not self.spatial_model in spatial_model_menu:
            logger.error(
                f"DataGenerator {self._procName}: invalid spatial model {self.spatial_model}."
                + f" Available: {', '.join(spatial_model_menu)}."
            )
            return False

        # Operational metadata
        self.runtimes: List[float] = reader.read_param(
            params, "channel_runtimes", [3600.0]
        )  # (s)

        # ROI configurations
        self.ke_min: float = reader.read_param(
            params, "ke_min", 18573.24 - 2300
        )  # (eV)
        self.ke_max: float = reader.read_param(
            params, "ke_max", 18573.24 + 1000
        )  # (eV)
        self.ke_bins: int = reader.read_param(params, "ke_bins", 100)
        self.ke_edges: Optional[Union[np.ndarray, List[float]]] = reader.read_param(
            params, "ke_edges", None
        )  # (eV)
        self.theta_bins: int = reader.read_param(params, "theta_bins", 3600)
        self.theta_edges: Optional[Union[np.ndarray, List[float]]] = reader.read_param(
            params, "theta_edges", None
        )  # (rad)
        self.r_max: float = reader.read_param(params, "r_max", 0.01)  # (m)
        self.r_bins: int = reader.read_param(params, "r_bins", 1400)
        self.r_edges: Optional[Union[np.ndarray, List[float]]] = reader.read_param(
            params, "r_edges", None
        )  # (m)
        self.phi_bins: int = reader.read_param(params, "phi_bins", 360)
        self.phi_edges: Optional[Union[np.ndarray, List[float]]] = reader.read_param(
            params, "phi_edges", None
        )  # (rad)

        # Monoenergetic source configurations
        self.mono_energy: float = reader.read_param(
            params, "mono_energy", 18573.24 + 0.02
        )  # (eV)
        self.mono_rate: float = reader.read_param(params, "mono_rate", 1)  # (1/s)
        self.mono_binned_mode: bool = reader.read_param(
            params, "mono_binned_mode", False
        )  # (bool)

        # Flat background configurations
        self.bkgd_flat_rate: float = reader.read_param(
            params, "bkgd_flat_rate", 1e-5
        )  # (1/s/eV/m^3)
        self.bkgd_flat_binned_mode: bool = reader.read_param(
            params, "bkgd_flat_binned_mode", False
        )  # (bool)

        # TODO: e-gun source configurations

        # TODO: Kr source configurations

        # Numerical source/background configurations
        self.source_numerical_rate_path: Optional[str] = self._read_first_param(
            params,
            ["source_numerical_ke_rate_path", "source_numerical_rate_path"],
            None,
        )
        self.source_numerical_binned_mode: bool = reader.read_param(
            params, "source_numerical_binned_mode", False
        )

        self.bkgd_numerical_rate_path: Optional[str] = self._read_first_param(
            params,
            ["bkgd_numerical_ke_rate_path", "bkgd_numerical_rate_path"],
            None,
        )
        self.bkgd_numerical_binned_mode: bool = reader.read_param(
            params, "bkgd_numerical_binned_mode", False
        )

        # Spatial model configurations
        self.spatial_binned_mode: bool = reader.read_param(
            params, "spatial_binned_mode", False
        )  # (bool)

        # UniformCylinder spatial model configurations
        self.uniform_cylinder_radius: float = reader.read_param(
            params, "uniform_cylinder_radius", 0.01
        )  # (m)

        # Numerical spatial model configurations
        self.spatial_numerical_hist_path: Optional[str] = self._read_first_param(
            params, ["spatial_numerical_hist_path", "numerical_spatial_hist_path"], None
        )
        self.spatial_numerical_apply_trapping_efficiency: bool = reader.read_param(
            params, "spatial_numerical_apply_trapping_efficiency", False
        )

        # Cavity field configurations
        cavity_field_option_menu = ["none", "numeric"]  # TODO: analytic
        self.cavity_field_option: str = reader.read_param(
            params, "cavity_field_option", "none"
        )
        if not self.cavity_field_option in cavity_field_option_menu:
            logger.error(
                f"DataGenerator {self._procName}: invalid cavity field model {self.cavity_field_option}."
                + f" Available: {', '.join(cavity_field_option_menu)}."
            )
            return False

        self.cavity_field_path: Optional[str] = reader.read_param(
            params, "cavity_field_map_path", None
        )  # (str)
        self.cavity_field_r_edges: Optional[Union[np.ndarray, List[float]]] = (
            reader.read_param(params, "cavity_field_r_edges", None)
        )  # (m)
        self.cavity_field_theta_edges: Optional[Union[np.ndarray, List[float]]] = (
            reader.read_param(params, "cavity_field_theta_edges", None)
        )  # (rad)
        self.cavity_field_B_map: Optional[Union[np.ndarray, List[float]]] = (
            reader.read_param(params, "cavity_field_B_map", None)
        )  # (T)

        self.cavity_field_kwargs: Dict = {
            "r_edges": self.cavity_field_r_edges,
            "z_edges": self.cavity_field_theta_edges,
            "B_map": self.cavity_field_B_map,
            "path": self.cavity_field_path,
        }

        # Detection efficiency configurations
        self.detection_efficiency_enabled: bool = reader.read_param(
            params, "detection_efficiency_enabled", False
        )  # (bool)
        self.detection_efficiency_path: Optional[str] = reader.read_param(
            params, "detection_efficiency_path", None
        )  # (str)
        self.detection_efficiency_nan_fill_value: Optional[float] = reader.read_param(
            params, "detection_efficiency_nan_fill_value", None
        )  # (float)

        # Energy error configurations
        self.energy_error_enabled: bool = reader.read_param(
            params, "energy_error_enabled", False
        )  # (bool)
        self.energy_error_map_path: Optional[str] = reader.read_param(
            params, "energy_error_map_path", None
        )  # (str)
        self.energy_error_nan_fill_value: Optional[float] = reader.read_param(
            params, "energy_error_nan_fill_value", None
        )  # (float)

        # Instantiate the samplers
        self._edges: Dict[str, np.ndarray] = dict()
        if self.ke_edges is None:
            self.ke_edges = np.linspace(self.ke_min, self.ke_max, self.ke_bins + 1)
        self._edges["ke_edges"] = np.asarray(self.ke_edges)
        if self.theta_edges is None:
            self.theta_edges = np.linspace(0, np.pi, self.theta_bins + 1)
        self._edges["theta_edges"] = np.asarray(self.theta_edges)
        if self.r_edges is None:
            self.r_edges = np.linspace(0, self.r_max, self.r_bins + 1)
        self._edges["r_edges"] = np.asarray(self.r_edges)
        if self.phi_edges is None:
            self.phi_edges = np.linspace(0, 2 * np.pi, self.phi_bins + 1)
        self._edges["phi_edges"] = np.asarray(self.phi_edges)

        # overwrite the default edges values when the spatial model is numerical
        if self.spatial_model == "numerical":
            _spatial_numerical_data = self._read_spatial_numerical_data(self.spatial_numerical_hist_path)
            if _spatial_numerical_data is None:
                logger.error("Failed to read spatial numerical data")
                return False
            
            self._edges["theta_edges"] = _spatial_numerical_data["theta_edges"]
            self._edges["r_edges"] = _spatial_numerical_data["r_edges"]
            self._edges["phi_edges"] = _spatial_numerical_data["phi_edges"]

        # energy samplers
        self._energy_samplers: OrderedDict[str, EnergySampler.EnergySampler] = od()

        for source_type in self.source_types:
            if source_type == "mono":
                sampler = EnergySampler.Monoenergetic(
                    name=self._procName + "_mono",
                    peak_energy=self.mono_energy,
                    peak_rate=self.mono_rate,
                    binned_mode=self.mono_binned_mode,
                    **self._edges,
                )
                self._energy_samplers[source_type] = sampler
            elif source_type == "numerical":
                sampler = EnergySampler.Numerical(
                    name=self._procName + "_source_numerical",
                    path=self.source_numerical_rate_path,
                    binned_mode=self.source_numerical_binned_mode,
                    **self._edges,
                )
                self._energy_samplers[source_type] = sampler
            # TODO: elif self.source_type == "Kr":
            # TODO: elif self.source_type == "e-gun":
            else:
                logger.error(f"Unknown source type: {source_type}")
                return False

        for bkgd_type in self.bkgd_types:
            if bkgd_type == "flat":
                sampler = EnergySampler.Flat(
                    name=self._procName + "_flat",
                    flat_rate=self.bkgd_flat_rate,
                    binned_mode=self.bkgd_flat_binned_mode,
                    **self._edges,
                )
                self._energy_samplers[bkgd_type] = sampler
            elif bkgd_type == "numerical":
                sampler = EnergySampler.Numerical(
                    name=self._procName + "_bkgd_numerical",
                    path=self.bkgd_numerical_rate_path,
                    binned_mode=self.bkgd_numerical_binned_mode,
                    **self._edges,
                )
                self._energy_samplers[bkgd_type] = sampler
            # TODO: elif self.bkgd_type == "slope":
            else:
                logger.error(f"Unknown background type: {bkgd_type}")
                return False

        if len(self._energy_samplers) == 0:
            logger.error("No energy sampler is defined.")
            return False

        # spatial sampler
        if self.spatial_model == "uniform_cylinder":
            self._spatial_sampler = SpatialSampler.UniformCylinder(
                name=self._procName + "_uniform_cylinder",
                radius=self.uniform_cylinder_radius,
                binned_mode=self.spatial_binned_mode,
                cavity_field_option=self.cavity_field_option,
                cavity_field_kwargs=self.cavity_field_kwargs,
                **self._edges,
            )
        elif self.spatial_model == "numerical":
            self._spatial_sampler = SpatialSampler.Numerical(
                name=self._procName + "_spatial_numerical",
                binned_mode=self.spatial_binned_mode,
                path=self.spatial_numerical_hist_path,
                apply_trapping_efficiency=self.spatial_numerical_apply_trapping_efficiency,
                cavity_field_option=self.cavity_field_option,
                cavity_field_kwargs=self.cavity_field_kwargs,
                **self._edges,
            )
        # TODO: elif self.spatial_model == other models:
        else:
            logger.error(f"Unknown spatial model: {self.spatial_model}")
            return False

        # detection efficiency
        self._detection_efficiency: Optional[DetectionEfficiency.DetectionEfficiency] = None
        if self.detection_efficiency_enabled:
            if self.detection_efficiency_path is None:
                logger.error("Detection efficiency enabled but no path provided.")
                return False
            
            self._detection_efficiency = DetectionEfficiency.DetectionEfficiency(
                name=self._procName + "_detection_efficiency",
                efficiency_map_path=self.detection_efficiency_path,
                nan_fill_value=self.detection_efficiency_nan_fill_value,
            )

        # energy error
        self._energy_error: Optional[EnergyError.EnergyError] = None
        if self.energy_error_enabled:
            if self.energy_error_map_path is None:
                logger.error("Energy error enabled but no map path provided.")
                return False
            
            self._energy_error = EnergyError.EnergyError(
                name=self._procName + "_energy_error",
                energy_error_map_path=self.energy_error_map_path,
                nan_fill_value=self.energy_error_nan_fill_value,
            )

        # placeholder for the InternalRun result
        self.results: List[Dict[str, np.ndarray]] = [{} for _ in self.runtimes]

        return True

    @property
    def edges(self) -> Dict[str, np.ndarray]:
        """
        Returns the dictionary of edges for each dimension.
        Keys: "ke_edges", "r_edges", "theta_edges", "phi_edges"
        """
        return self._edges

    @property
    def energy_samplers(self) -> OrderedDict[str, EnergySampler.EnergySampler]:
        """
        Returns the ordered dictionary of energy samplers.
        Keys are the sampler types.
        """
        return self._energy_samplers

    def InternalRun(self):
        """
        Run `self.generate_data()` method.
        """
        self.results = self.generate_data()

        return True

    def generate_data(
        self,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate 4-dimensional data (E, theta_center, r_start, phi_start) from
        binned/unbinned source and background models.
        First, sample energy from the source and background models.
        Then, sample theta_center, r_start, phi_start from the spatial model.
        The trapping efficiency and the conversion of theta_start -> theta_center
        are performed internally if a cavity field is set.
        TODO: Lastly, apply the detector responses: energy resolution and efficiency.

        Returns:
            A dictionary containing the sampled 4D data arrays for each runtime.
            Keys: "sampler_id", "ke_start", "ke_observed", "theta_center", "r_start", "phi_start", "theta_start"
            - "ke_start": Initial kinetic energy before detector smearing
            - "ke_observed": Kinetic energy after applying detector energy error (smearing)
            Each value is a list of np.ndarray, one for each runtime.
        """
        info_msg = f"{self._procName} is generating pseudo-data:"
        info_msg += " source from " + ", ".join(self.source_types)
        info_msg += " & background from " + "+".join(self.bkgd_types)
        logger.info(info_msg)

        # sample energy
        fake_data: List[Dict[str, np.ndarray]] = [
            {
                "sampler_id": np.zeros(0, dtype=int),
                "ke_start": np.zeros(0, dtype=float),
                "ke_observed": np.zeros(0, dtype=float),
                "theta_center": np.zeros(0, dtype=float),
                "r_start": np.zeros(0, dtype=float),
                "phi_start": np.zeros(0, dtype=float),
                "theta_start": np.zeros(0, dtype=float),
            }
            for _ in self.runtimes
        ]
        for sampler_id, sampler in enumerate(self._energy_samplers.values()):
            if not sampler.Sample(self.runtimes):
                return fake_data

            ke_samples = sampler.result  # List[np.ndarray], (eV)
            n_samples = [len(arr) for arr in ke_samples]

            for i in range(len(self.runtimes)):
                fake_data[i]["sampler_id"] = np.concatenate(
                    (fake_data[i]["sampler_id"], np.full(n_samples[i], sampler_id))
                )
                fake_data[i]["ke_start"] = np.concatenate((fake_data[i]["ke_start"], ke_samples[i]))
                fake_data[i]["ke_observed"] = np.concatenate((fake_data[i]["ke_observed"], ke_samples[i]))

        # sample geometry
        if not self._spatial_sampler.Sample([fd["ke_start"] for fd in fake_data]):
            return fake_data

        geometry_samples = (
            self._spatial_sampler.result
        )  # List[Dict[str, np.ndarray]], {"theta_center": (N,), "r_start": (N,), "phi_start": (N,), "theta_start": (N,)}
        for i in range(len(self.runtimes)):
            fake_data[i]["theta_center"] = geometry_samples[i]["theta_center"]
            fake_data[i]["r_start"] = geometry_samples[i]["r_start"]
            fake_data[i]["phi_start"] = geometry_samples[i]["phi_start"]
            fake_data[i]["theta_start"] = geometry_samples[i]["theta_start"]

        # Apply detection efficiency
        if self._detection_efficiency is not None:
            logger.info("Applying detection efficiency")
            for i in range(len(self.runtimes)):
                n_before = len(fake_data[i]["ke_start"])
                
                # Apply efficiency and get boolean mask of detected events
                detected = self._detection_efficiency.apply_efficiency(
                    fake_data[i]["ke_start"],
                    fake_data[i]["theta_center"],
                    fake_data[i]["r_start"],
                    fake_data[i]["phi_start"],
                )
                
                # Filter all arrays to keep only detected events
                fake_data[i]["sampler_id"] = fake_data[i]["sampler_id"][detected]
                fake_data[i]["ke_start"] = fake_data[i]["ke_start"][detected]
                fake_data[i]["ke_observed"] = fake_data[i]["ke_observed"][detected]
                fake_data[i]["theta_center"] = fake_data[i]["theta_center"][detected]
                fake_data[i]["r_start"] = fake_data[i]["r_start"][detected]
                fake_data[i]["phi_start"] = fake_data[i]["phi_start"][detected]
                fake_data[i]["theta_start"] = fake_data[i]["theta_start"][detected]
                
                n_after = len(fake_data[i]["ke_start"])
                efficiency_measured = n_after / n_before if n_before > 0 else 0
                logger.info(
                    f"    Runtime {i}: {n_before} -> {n_after} events "
                    f"(measured efficiency: {efficiency_measured:.3f})"
                )

        # Apply energy error
        if self._energy_error is not None:
            logger.info("Applying energy error")
            for i in range(len(self.runtimes)):
                n_events = len(fake_data[i]["ke_start"])
                if n_events == 0:
                    logger.info(f"    Runtime {i}: No events to apply energy error")
                    continue
                
                # Sample energy errors for each event
                ke_errors = self._energy_error.sample_energy_error(
                    fake_data[i]["ke_start"],
                    fake_data[i]["theta_center"],
                    fake_data[i]["r_start"],
                    fake_data[i]["phi_start"],
                )
                
                # Create observed kinetic energy by adding error to initial kinetic energy
                fake_data[i]["ke_observed"] = fake_data[i]["ke_start"] + ke_errors
                
                # Log statistics
                logger.info(
                    f"    Runtime {i}: Applied energy error to {n_events} events. "
                    f"Mean error: {ke_errors.mean():.3f} eV, "
                    f"Std: {ke_errors.std():.3f} eV, "
                    f"Range: [{ke_errors.min():.3f}, {ke_errors.max():.3f}] eV"
                )

        logger.info("Generated 4-dimensional data")
        for i, runtime in enumerate(self.runtimes):
            n_events = len(fake_data[i]["ke_observed"])
            logger.info(f"    Runtime {i} ({runtime:.1f} s): {n_events} events")

        return fake_data
