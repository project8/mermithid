"""
The data generator class for CCA.
Author: S. M. Lee
First Date: August 25, 2025
Last Update: January 19, 2026
"""

from __future__ import absolute_import

from collections import OrderedDict as od
from typing import Dict, List, Optional, Union, OrderedDict

from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor

from mermithid.misc.FakeTritiumDataFunctions import *

from . import EnergySampler
from . import SpatialSampler

logger = morphologging.getLogger(__name__)


__all__ = []
__all__.append(__name__)


class DataGenerator4D(BaseProcessor):
    """
    Generate pseudo electrons for CCA. It samples 4D data
    (E, theta_center, r_start, phi_start) from a given source and background
    model. This will sample theta_start together, which is internally used for
    the detector response simulation.

    Parameters:
        name: The name of the instance
        The other configurations are set by `InternalConfigure()` via the
        `super().Configure()` method.
    Results:
        `super().Run()` runs the 4D-data generation.
        The sampled 4D data are stored in `self.results`.
    """

    def InternalConfigure(self, params):
        """
        Configure the `DataGenerator4D` instance.

        TODO: explain parameters
        """
        # Choose the source
        source_type_menu = ["mono"]  # TODO: ["e-gun", "Kr"]
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
        bkgd_type_menu = ["flat"]
        self.bkgd_types: List[str] = reader.read_param(params, "bkgd_types", [])
        for t in self.bkgd_types:
            if not t in bkgd_type_menu:
                logger.error(
                    f"DataGenerator {self._procName}: invalid background {t}."
                    + f" Available: {', '.join(bkgd_type_menu)}."
                )
                return False

        # Choose the spatial model
        spatial_model_menu = ["uniform_cylinder"]  # TODO: other models
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

        # Spatial model configurations
        self.spatial_binned_mode: bool = reader.read_param(
            params, "spatial_binned_mode", False
        )  # (bool)

        # UniformCylinder spatial model configurations
        self.uniform_cylinder_radius: float = reader.read_param(
            params, "uniform_cylinder_radius", 0.01
        )  # (m)

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

        # Instantiate the samplers
        self._edge: Dict[str, np.ndarray] = dict()
        if self.ke_edges is None:
            self.ke_edges = np.linspace(self.ke_min, self.ke_max, self.ke_bins + 1)
        self._edge["ke_edges"] = np.asarray(self.ke_edges)
        if self.theta_edges is None:
            self.theta_edges = np.linspace(0, np.pi, self.theta_bins + 1)
        self._edge["theta_edges"] = np.asarray(self.theta_edges)
        if self.r_edges is None:
            self.r_edges = np.linspace(0, self.r_max, self.r_bins + 1)
        self._edge["r_edges"] = np.asarray(self.r_edges)
        if self.phi_edges is None:
            self.phi_edges = np.linspace(0, 2 * np.pi, self.phi_bins + 1)
        self._edge["phi_edges"] = np.asarray(self.phi_edges)

        # energy samplers
        self._energy_samplers: OrderedDict[str, EnergySampler.EnergySampler] = od()

        for source_type in self.source_types:
            if source_type == "mono":
                sampler = EnergySampler.Monoenergetic(
                    name=self._procName + "_mono",
                    peak_energy=self.mono_energy,
                    peak_rate=self.mono_rate,
                    binned_mode=self.mono_binned_mode,
                    **self._edge,
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
                    **self._edge,
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
                **self._edge,
            )
        # TODO: elif self.spatial_model == other models:
        else:
            logger.error(f"Unknown spatial model: {self.spatial_model}")
            return False

        # placeholder for the InternalRun result
        self.results: List[Dict[str, np.ndarray]] = [{} for _ in self.runtimes]

        return True

    @property
    def edge(self) -> Dict[str, np.ndarray]:
        """
        Returns the dictionary of edges for each dimension.
        Keys: "ke_edges", "r_edges", "theta_edges", "phi_edges"
        """
        return self._edge

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
            Keys: "sampler_id", "ke", "theta_center", "r_start", "phi_start", "theta_start"
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
                "ke": np.zeros(0, dtype=float),
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
                fake_data[i]["ke"] = np.concatenate((fake_data[i]["ke"], ke_samples[i]))

        # sample geometry
        if not self._spatial_sampler.Sample([fd["ke"] for fd in fake_data]):
            return fake_data

        geometry_samples = (
            self._spatial_sampler.result
        )  # List[Dict[str, np.ndarray]], {"theta_center": (N,), "r_start": (N,), "phi_start": (N,), "theta_start": (N,)}
        for i in range(len(self.runtimes)):
            fake_data[i]["theta_center"] = geometry_samples[i]["theta_center"]
            fake_data[i]["r_start"] = geometry_samples[i]["r_start"]
            fake_data[i]["phi_start"] = geometry_samples[i]["phi_start"]
            fake_data[i]["theta_start"] = geometry_samples[i]["theta_start"]

        # TODO: apply the detector response
        # test_detector = Detector4D.Detector4D(name="test_detector")
        # test_detector.SetResolution(
        #     Detector4D.ResolutionOnEnergy.ConstantGaussian(
        #         name="test_resolution",
        #         resolution=100,  # (eV)
        #     )
        # )

        logger.info("Generated 4-dimensional data")
        for i, runtime in enumerate(self.runtimes):
            n_events = len(fake_data[i]["ke"])
            logger.info(f"    Runtime {i} ({runtime:.1f} s): {n_events} events")

        return fake_data
