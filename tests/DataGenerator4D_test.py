"""
To test mermithid.processors.DataGenerator4D.DataGenerator4D.
Author: S. M. Lee
First Date: August 26, 2025
Last Date: November 26, 2025
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
            "theta_bins": 144,
            "r_max": 0.06,  # m
            "r_bins": 30,
            "phi_bins": 360,
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
            "spatial_binned_mode": True,
            "spatial_model": "uniform_cylinder",
            "uniform_cylinder_radius": 0.06,  # (m)
            # Cavity field configurations
            "cavity_field_option": "numeric",
            "cavity_field_map_path": "cavity_field_map_CCA_Trap_V45.npz",
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

        # visualize results
        draw_config = specGen_config.copy()
        draw_config["theta_min"] = 0
        draw_config["theta_max"] = np.pi
        draw_config["phi_min"] = 0
        draw_config["phi_max"] = 2 * np.pi
        draw_config["r_min"] = 0

        fig = plt.figure(figsize=(12, 8))
        fig_joint, fig_3d = fig.subfigures(1, 2, width_ratios=[2, 1])

        axs_joint = fig_joint.subplots(4, 4)

        gs_3d = fig_3d.add_gridspec(20, 1)
        axs_3d = [
            fig_3d.add_subplot(gs_3d[3:10, 0], projection="3d"),
            fig_3d.add_subplot(gs_3d[13:20, 0], projection="3d"),
        ]
        ax_cbar = fig_3d.add_subplot(gs_3d[0, 0])

        # Joint plots
        colors = ["red", "blue"]
        cmaps = [plt.get_cmap("Reds"), plt.get_cmap("Blues")]

        var_names = ["ke", "theta", "r", "phi"]
        var_labels = [
            "Kinetic Energy [eV]",
            r"Pitch angle $\theta$ [rad]",
            r"Radius $r$ [m]",
            r"Azimuthal angle $\phi$ [rad]",
        ]
        var_ticks = [
            None,
            np.linspace(0, np.pi, 5),
            None,
            np.linspace(0, 2 * np.pi, 5),
        ]
        var_ticklabels = [
            None,
            [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"],
            None,
            [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
        ]
        for var_x in range(4):
            for var_y in range(4):
                ax = axs_joint[var_y, var_x]
                if var_x == var_y:
                    for i, runtime in enumerate(draw_config["channel_runtimes"]):
                        ax.hist(
                            results[i][var_names[var_x]],
                            bins=draw_config[f"{var_names[var_x]}_bins"],
                            range=(
                                draw_config[f"{var_names[var_x]}_min"],
                                draw_config[f"{var_names[var_x]}_max"],
                            ),
                            color=colors[i],
                            alpha=0.5,
                        )
                    ax.set_xlim(
                        draw_config[f"{var_names[var_x]}_min"],
                        draw_config[f"{var_names[var_x]}_max"],
                    )
                    ax.set_ylim(0, None)
                elif var_x < var_y:
                    for i, runtime in enumerate(draw_config["channel_runtimes"]):
                        ax.scatter(
                            results[i][var_names[var_x]],
                            results[i][var_names[var_y]],
                            color=colors[i],
                            s=1,
                            alpha=0.05,
                        )
                    ax.set_xlim(
                        draw_config[f"{var_names[var_x]}_min"],
                        draw_config[f"{var_names[var_x]}_max"],
                    )
                    ax.set_ylim(
                        draw_config[f"{var_names[var_y]}_min"],
                        draw_config[f"{var_names[var_y]}_max"],
                    )
                else:
                    for i, runtime in enumerate(draw_config["channel_runtimes"]):
                        ax.hist2d(
                            results[i][var_names[var_x]],
                            results[i][var_names[var_y]],
                            bins=(
                                draw_config[f"{var_names[var_x]}_bins"],
                                draw_config[f"{var_names[var_y]}_bins"],
                            ),
                            range=(
                                (
                                    draw_config[f"{var_names[var_x]}_min"],
                                    draw_config[f"{var_names[var_x]}_max"],
                                ),
                                (
                                    draw_config[f"{var_names[var_y]}_min"],
                                    draw_config[f"{var_names[var_y]}_max"],
                                ),
                            ),
                            cmap=cmaps[i],
                            alpha=0.5,
                        )
                    ax.set_xlim(
                        draw_config[f"{var_names[var_x]}_min"],
                        draw_config[f"{var_names[var_x]}_max"],
                    )
                    ax.set_ylim(
                        draw_config[f"{var_names[var_y]}_min"],
                        draw_config[f"{var_names[var_y]}_max"],
                    )

        for var_x in range(4):
            for var_y in range(4):
                ax = axs_joint[var_y, var_x]
                if var_x == 0:
                    ax.set_ylabel(var_labels[var_y])
                    if var_ticks[var_y] is not None:
                        ax.set_yticks(var_ticks[var_y])
                        ax.set_yticklabels(var_ticklabels[var_y])
                elif var_x == 3:
                    # draw y-axis labels on the right side for the last column
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(var_labels[var_y])

                    ax.yaxis.tick_right()
                    if var_ticks[var_y] is not None and var_y != 3:
                        ax.set_yticks(var_ticks[var_y])
                        ax.set_yticklabels(var_ticklabels[var_y])
                else:
                    ax.set_yticklabels([])

                if var_y == 3:
                    ax.set_xlabel(var_labels[var_x])
                    if var_ticks[var_x] is not None:
                        ax.set_xticks(var_ticks[var_x])
                        ax.set_xticklabels(var_ticklabels[var_x])
                elif var_y == 0:
                    # draw x-axis labels on the top side for the first row
                    ax.xaxis.set_label_position("top")
                    ax.set_xlabel(var_labels[var_x])

                    ax.xaxis.tick_top()
                    if var_ticks[var_x] is not None:
                        ax.set_xticks(var_ticks[var_x])
                        ax.set_xticklabels(var_ticklabels[var_x])
                else:
                    ax.set_xticklabels([])

        axs_joint[0, 0].set_ylabel("Counts")
        axs_joint[3, 3].set_ylabel("Counts")

        # 3D plots
        mappable = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=draw_config["ke_min"], vmax=draw_config["ke_max"]),  # type: ignore
        )
        for i, ax in enumerate(axs_3d):
            n = len(results[i]['ke'])
            runtime = draw_config["channel_runtimes"][i]

            r = results[i]["r"][:n//10]
            phi = results[i]["phi"][:n//10]
            theta = results[i]["theta"][:n//10]
            ke = results[i]["ke"][:n//10]

            x = r * np.cos(phi)
            y = r * np.sin(phi)
            # z is along the cavity axis, randomly sampled
            z = np.random.uniform(
                -draw_config["uniform_cylinder_radius"],
                draw_config["uniform_cylinder_radius"],
                len(r),
            )

            cbar = fig_3d.colorbar(mappable, cax=ax_cbar, orientation="horizontal")
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.xaxis.tick_top()
            cbar.set_label("Kinetic Energy [eV]")

            # draw 3d arrows
            scale = 0.3 * draw_config["r_max"] / np.max(ke)
            pitch_phi = np.random.uniform(0, 2 * np.pi, len(r))
            u = scale * ke * np.sin(theta) * np.cos(pitch_phi)  # scale u
            v = scale * ke * np.sin(theta) * np.sin(pitch_phi)  # scale v
            w = scale * ke * np.cos(theta)  # scale w

            axs_3d[i].quiver(x, y, z, u, v, w, color=mappable.to_rgba(ke), alpha=0.1)

            axs_3d[i].set_xlim([-1.3 * specGen_config["r_max"], 1.3 * specGen_config["r_max"]])
            axs_3d[i].set_ylim([-1.3 * specGen_config["r_max"], 1.3 * specGen_config["r_max"]])
            axs_3d[i].set_zlim([-1.3 * specGen_config["r_max"], 1.3 * specGen_config["r_max"]])
            axs_3d[i].set_xlabel("X [m]")
            axs_3d[i].set_ylabel("Y [m]")
            axs_3d[i].set_zlabel("Z (random) [m]")
            axs_3d[i].set_title(
                f"Runtime {i} ({runtime:.1f} s; N={len(results[i]['ke'])})"
                + "\n(10% of samples)"
            )

        plt.savefig("DataGenerator4D_test.png", dpi=300)


if __name__ == "__main__":
    unittest.main()
