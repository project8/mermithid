"""
To test mermithid.processors.DataGenerator4D.DataGenerator4D.
Author: S. M. Lee
First Date: August 26, 2025
Last Date: August 26, 2025
"""

import unittest

from morpho.utilities import morphologging

logger = morphologging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np


class DataGenerator4DTest(unittest.TestCase):

    def test_data_generation(self):
        from mermithid.processors.DataGenerator4D import DataGenerator4D

        specGen_config = {
            # ROI configurations
            "ke_min": 18000,  # eV
            "ke_max": 19000,  # eV
            "ke_bins": 100,
            "r_max": 0.1,  # m
            "r_bins": 10,
            "theta_bins": 36,
            "phi_bins": 36,
            # Monoenergetic rate configurations
            "source_types": ["mono"],
            "mono_energy": 18300,  # eV
            "mono_rate": 1,  # (1/s)
            "mono_binned_mode": False,
            # Background configurations
            "bkgd_types": ["flat"],
            "bkgd_flat_rate": 1e-2,  # (1/s/eV)
            "bkgd_flat_binned_mode": False,
            # Spatial model configurations
            "spatial_binned_mode": False,
            "spatial_model": "uniform_sphere",
            "uniform_sphere_radius": 0.08,  # (m)
            # Operational metadata
            "channel_runtimes": [600.0, 1200.0],  # s
        }

        specGen = DataGenerator4D("specGen")
        specGen.Configure(specGen_config)
        specGen.Run()
        results = specGen.results

        for key in results[0].keys():
            msg = f"{key:10s}: "
            for i in range(len(results)):
                msg += f"{results[i][key].shape} "

            logger.info(msg)

        # plot histograms
        fig, axs = plt.subplots(
            2, 4, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1, 1, 0.3]}
        )

        for i, runtime in enumerate(specGen.runtimes):
            axs[0, 0].hist(
                results[i]["ke"],
                bins=50,
                range=((specGen_config["ke_min"], specGen_config["ke_max"])),
                alpha=0.5,
                label=f"Runtime {i} ({runtime:.1f} s; N={len(results[i]['ke'])})",
            )

        axs[0, 0].set_xlabel("Kinetic Energy [eV]")
        axs[0, 0].set_xlim(
            specGen_config["ke_min"], specGen_config["ke_max"]
        )
        axs[0, 0].set_ylabel("N")
        axs[0, 0].legend()

        for i, runtime in enumerate(specGen.runtimes):
            axs[0, 1].hist(
                results[i]["r"],
                bins=50,
                range=(0, specGen_config["r_max"]),
                alpha=0.5,
            )

        axs[0, 1].set_xlabel("Radius [m]")
        axs[0, 1].set_xlim(0, specGen_config["r_max"])
        axs[0, 1].set_ylabel("N")

        for i, runtime in enumerate(specGen.runtimes):
            axs[1, 0].hist(
                results[i]["theta"],
                bins=50,
                range=(0, np.pi),
                alpha=0.5,
            )

        axs[1, 0].set_xlabel("Theta [rad]")
        axs[1, 0].set_xlim(0, np.pi)
        axs[1, 0].set_ylabel("N")

        for i, runtime in enumerate(specGen.runtimes):
            axs[1, 1].hist(
                results[i]["phi"],
                bins=50,
                range=(0, 2 * np.pi),
                alpha=0.5,
            )

        axs[1, 1].set_xlabel("Phi [rad]")
        axs[1, 1].set_xlim(0, 2 * np.pi)
        axs[1, 1].set_ylabel("N")

        # draw 3D scatter plot on axs[2, 0] and hide axs[2, 1]
        axs[0, 2].axis("off")
        axs[1, 2].axis("off")
        axs[0, 3].axis("off")
        axs[1, 3].axis("off")
        ax_3d = [
            fig.add_subplot(2, 4, 3, projection="3d"),
            fig.add_subplot(2, 4, 7, projection="3d"),
        ]

        ax_cbar = [axs[0, 3], axs[1, 3]]
        for i, runtime in enumerate(specGen.runtimes):
            r = results[i]["r"]
            theta = results[i]["theta"]
            phi = results[i]["phi"]
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            ke = results[i]["ke"]
            mappable = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=plt.Normalize(vmin=ke.min(), vmax=ke.max()),  # type: ignore
            )
            mappable.set_array([])

            cbar = plt.colorbar(mappable, ax=ax_cbar[i])
            cbar.set_label("Kinetic Energy ke [eV]")

            ax_3d[i].scatter(x, y, z, s=1, alpha=0.1, c=mappable.to_rgba(ke))  # type: ignore
            ax_3d[i].set_xlim([-specGen_config["r_max"], specGen_config["r_max"]])
            ax_3d[i].set_ylim([-specGen_config["r_max"], specGen_config["r_max"]])
            ax_3d[i].set_zlim([-specGen_config["r_max"], specGen_config["r_max"]])
            ax_3d[i].set_xlabel("X [m]")
            ax_3d[i].set_ylabel("Y [m]")
            ax_3d[i].set_zlabel("Z [m]")
            ax_3d[i].set_title(
                f"Runtime {i} ({runtime:.1f} s; N={len(results[i]['ke'])})"
            )

        plt.tight_layout()
        plt.savefig("DataGenerator4D_test.png", dpi=300)


if __name__ == "__main__":
    unittest.main()
