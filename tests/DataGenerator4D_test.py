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
            "theta_bins": 36,
            "r_max": 0.1,  # m
            "r_bins": 10,
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
            "spatial_model": "uniform_cylinder",
            "uniform_cylinder_radius": 0.08,  # (m)
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
        ax_ke = axs[0, 0]
        ax_theta = axs[0, 1]
        ax_r = axs[1, 0]
        ax_phi = axs[1, 1]

        for i, runtime in enumerate(specGen.runtimes):
            ax_ke.hist(
                results[i]["ke"],
                bins=50,
                range=((specGen_config["ke_min"], specGen_config["ke_max"])),
                alpha=0.5,
                label=f"Runtime {i} ({runtime:.1f} s; N={len(results[i]['ke'])})",
            )

        ax_ke.set_xlabel("Kinetic Energy [eV]")
        ax_ke.set_xlim(
            specGen_config["ke_min"], specGen_config["ke_max"]
        )
        ax_ke.set_ylabel("N")
        ax_ke.legend()

        for i, runtime in enumerate(specGen.runtimes):
            ax_theta.hist(
                results[i]["theta"],
                bins=50,
                range=(0, np.pi),
                alpha=0.5,
            )

        ax_theta.set_xlabel(r"Pitch angle $\theta$ [rad]")
        ax_theta.set_xlim(0, np.pi)
        ax_theta.set_xticks(np.linspace(0, np.pi, 5))
        ax_theta.set_xticklabels(
            [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
        )
        ax_theta.set_ylabel("N")

        for i, runtime in enumerate(specGen.runtimes):
            ax_r.hist(
                results[i]["r"],
                bins=50,
                range=(0, specGen_config["r_max"]),
                alpha=0.5,
            )

        ax_r.set_xlabel(r"Radius $r$ [m]")
        ax_r.set_xlim(0, specGen_config["r_max"])
        ax_r.set_ylabel("N")

        for i, runtime in enumerate(specGen.runtimes):
            ax_phi.hist(
                results[i]["phi"],
                bins=50,
                range=(0, 2 * np.pi),
                alpha=0.5,
            )

        ax_phi.set_xlabel(r"Azimuthal angle $\phi$ [rad]")
        ax_phi.set_xlim(0, 2 * np.pi)
        ax_phi.set_xticks(np.linspace(0, 2 * np.pi, 5))
        ax_phi.set_xticklabels(
            ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
        )
        ax_phi.set_ylabel("N")

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
            n = len(results[i]['ke'])

            r = results[i]["r"][:n//10]
            phi = results[i]["phi"][:n//10]
            theta = results[i]["theta"][:n//10]
            ke = results[i]["ke"][:n//10]

            x = r * np.cos(phi)
            y = r * np.sin(phi)
            # z is sampled uniformly from -r to r
            z = np.random.uniform(-r, r, size=r.shape)

            mappable = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=plt.Normalize(vmin=ke.min(), vmax=ke.max()),  # type: ignore
            )
            mappable.set_array([])

            cbar = plt.colorbar(mappable, ax=ax_cbar[i])
            cbar.set_label("Kinetic Energy ke [eV]")

            # draw 3d arrows.
            scale = 0.3 * specGen_config["r_max"] / np.max(ke)
            pitch_phi = np.random.uniform(0, 2 * np.pi, size=ke.shape)
            u = scale * ke * np.sin(theta) * np.cos(pitch_phi)  # scale u
            v = scale * ke * np.sin(theta) * np.sin(pitch_phi)  # scale v
            w = scale * ke * np.cos(theta)  # scale w

            ax_3d[i].quiver(x, y, z, u, v, w, color=mappable.to_rgba(ke), alpha=0.1)

            ax_3d[i].set_xlim([-1.3 * specGen_config["r_max"], 1.3 * specGen_config["r_max"]])
            ax_3d[i].set_ylim([-1.3 * specGen_config["r_max"], 1.3 * specGen_config["r_max"]])
            ax_3d[i].set_zlim([-1.3 * specGen_config["r_max"], 1.3 * specGen_config["r_max"]])
            ax_3d[i].set_xlabel("X [m]")
            ax_3d[i].set_ylabel("Y [m]")
            ax_3d[i].set_zlabel("Z (random) [m]")
            ax_3d[i].set_title(
                f"Runtime {i} ({runtime:.1f} s; N={len(results[i]['ke'])})"
                + "\n(10% of samples)"
            )

        plt.tight_layout()
        plt.savefig("DataGenerator4D_test.png", dpi=300)


if __name__ == "__main__":
    unittest.main()
