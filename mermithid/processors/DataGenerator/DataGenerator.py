"""
The data generator class
Author: S. M. Lee
First Date: August 25, 2025
Last Update: August 25, 2025
"""

from __future__ import absolute_import

from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor

from mermithid.misc.FakeTritiumDataFunctions import *

logger = morphologging.getLogger(__name__)


__all__ = []
__all__.append(__name__)


class DataGenerator(BaseProcessor):
    """
    Generate (pseudo) (un)binned electron data
    """

    def InternalConfigure(self, params):
        self.Q = reader.read_param(
            params, "Q", QT2
        )  # Choose the atomic or molecular tritium endpoint
        self.m = reader.read_param(params, "neutrino_mass", 0.2)  # Neutrino mass (eV)
        self.ke_min = reader.read_param(
            params, "ke_min", self.Q - self.m - 2300
        )  # Energy corresponding to lower bound of frequency ROI (eV)
        self.ke_max = reader.read_param(
            params, "ke_max", self.Q - self.m + 1000
        )  # Same, for upper bound (eV)
        self.r_max = reader.read_param(params, "r_max", 0.01)  # maximum radius (m)

        self.results = {}

    def InternalRun(self):
        fake_data = self.test_generate_unbinned_data()

        for key, value in fake_data.items():
            self.results[key] = value

        return True

    def test_generate_unbinned_data(
        self,
        particles=10**2,
    ):
        """
        Generate unbinned data for a given number of particles from a uniform distribution.
        """
        logger.info(
            f"DataGenerator {self._procName} is generating {particles:d} pseudo-unbinned electrons."
        )

        # An example E, r, theta, phi distribution
        ke = np.random.uniform(self.ke_min, self.ke_max, particles)  # eV
        r = np.random.uniform(0, self.r_max, particles)  # m
        theta = np.random.uniform(0, np.pi, particles)  # rad
        phi = np.random.uniform(0, 2 * np.pi, particles)  # rad

        fake_data = {
            "ke": ke,
            "r": r,
            "theta": theta,
            "phi": phi,
        }

        return fake_data
