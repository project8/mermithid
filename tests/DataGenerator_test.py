"""
To test mermithid.processors.DataGenerator.DataGenerator.
Author: S. M. Lee
First Date: August 26, 2025
Last Date: August 26, 2025
"""

import unittest

from morpho.utilities import morphologging

logger = morphologging.getLogger(__name__)

import matplotlib.pyplot as plt


class DataGeneratorTest(unittest.TestCase):

    def test_data_generation(self):
        from mermithid.processors.DataGenerator import DataGenerator

        specGen_config = {
            "Q": 18573.0,  # eV
            "m": 0.1,  # eV
            "ke_min": 18000,  # eV
            "ke_max": 19000,  # eV
            "r_max": 0.1,  # m
        }

        specGen = DataGenerator("specGen")
        specGen.Configure(specGen_config)
        specGen.Run()
        results = specGen.results

        # plot histograms
        plt.figure(figsize=(8, 8))

        plt.subplot(221)
        plt.hist(results["ke"], bins=50)
        plt.xlabel("Kinetic Energy ke [eV]")
        plt.ylabel("N")

        plt.subplot(222)
        plt.hist(results["r"], bins=50)
        plt.xlabel("Radius r [m]")
        plt.ylabel("N")

        plt.subplot(223)
        plt.hist(results["theta"], bins=50)
        plt.xlabel("Theta theta [rad]")
        plt.ylabel("N")

        plt.subplot(224)
        plt.hist(results["phi"], bins=50)
        plt.xlabel("Phi phi [rad]")
        plt.ylabel("N")

        plt.tight_layout()
        plt.savefig("DataGenerator_test.png", dpi=300)


if __name__ == "__main__":
    unittest.main()
