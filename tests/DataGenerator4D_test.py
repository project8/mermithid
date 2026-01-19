"""
To test mermithid.processors.DataGenerator4D.DataGenerator4D.
It will take ~5 min to run.
Author: S. M. Lee
First Date: August 26, 2025
Last Date: January 19, 2026
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
            "theta_bins": 7200,
            "r_max": 0.007,  # m
            "r_bins": 300,
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
            "channel_runtimes": [6000.0, 12000.0],  # s
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

        var_items = list()
        var_items.append(
            {
                "name": "ke",
                "equation": r"$E_{k}$",
                "min": draw_config["ke_min"],
                "max": draw_config["ke_max"],
                "bins": draw_config["ke_bins"],
                "label": "Kinetic Energy [eV]",
                "ticks": np.linspace(draw_config["ke_min"], draw_config["ke_max"], 3),
                "ticklabels": [
                    f"{int(x):d}"
                    for x in np.linspace(
                        draw_config["ke_min"], draw_config["ke_max"], 3
                    )
                ],
            }
        )
        var_items.append(
            {
                "name": "theta_center",
                "equation": r"$\theta_{\mathrm{c}}$",
                "min": 0.48 * np.pi,
                "max": 0.52 * np.pi,
                "bins": int(
                    draw_config["theta_bins"]
                    * (0.52 * np.pi - 0.48 * np.pi)
                    / (2 * np.pi)
                ),
                "label": r"Pitch angle $\theta_{\mathrm{center}}$ [rad]",
                "ticks": np.linspace(0.48 * np.pi, 0.52 * np.pi, 5),
                "ticklabels": [
                    r"$0.48 \pi$",
                    r"$0.49 \pi$",
                    r"$0.5 \pi$",
                    r"$0.51 \pi$",
                    r"$0.52 \pi$",
                ],
            }
        )
        var_items.append(
            {
                "name": "r_start",
                "equation": r"$r_{\mathrm{s}}$",
                "min": 0,
                "max": draw_config["r_max"],
                "bins": draw_config["r_bins"],
                "label": r"Radius $r_{\mathrm{start}}$ [m]",
                "ticks": [0, 0.0035, 0.007],
                "ticklabels": ["0", "0.0035", "0.007"],
            }
        )
        var_items.append(
            {
                "name": "phi_start",
                "equation": r"$\phi_{\mathrm{s}}$",
                "min": 0,
                "max": 2 * np.pi,
                "bins": draw_config["phi_bins"],
                "label": r"Azimuthal angle $\phi_{\mathrm{start}}$ [rad]",
                "ticks": np.linspace(0, 2 * np.pi, 5),
                "ticklabels": [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
            }
        )

        fig = plt.figure(figsize=(12, 8))
        fig_joint, fig_theta = fig.subfigures(1, 2, width_ratios=[2, 1])

        axs_joint = fig_joint.subplots(4, 4)

        gs_theta = fig_theta.add_gridspec(20, 20)
        ax_theta_hist: plt.Axes = fig_theta.add_subplot(gs_theta[0:9, 2:])
        ax_theta_transform: plt.Axes = fig_theta.add_subplot(gs_theta[11:20, 2:])

        # Joint plots
        colors = ["red", "blue"]
        cmaps = [plt.get_cmap("Reds"), plt.get_cmap("Blues")]

        for i_x, var_x in enumerate(var_items):
            for i_y, var_y in enumerate(var_items):
                ax: plt.Axes = axs_joint[i_y, i_x]
                if i_x == i_y:
                    for i, runtime in enumerate(draw_config["channel_runtimes"]):
                        ax.hist(
                            results[i][var_x["name"]],
                            bins=var_x["bins"],
                            range=(var_x["min"], var_x["max"]),
                            color=colors[i],
                            alpha=0.5,
                        )
                    ax.set_xlim(var_x["min"], var_x["max"])
                    ax.set_ylim(0, None)
                elif i_x < i_y:
                    for i, runtime in enumerate(draw_config["channel_runtimes"]):
                        ax.scatter(
                            results[i][var_x["name"]],
                            results[i][var_y["name"]],
                            color=colors[i],
                            s=1,
                            alpha=0.01,
                        )
                    ax.set_xlim(var_x["min"], var_x["max"])
                    ax.set_ylim(var_y["min"], var_y["max"])
                else:
                    for i, runtime in enumerate(draw_config["channel_runtimes"]):
                        ax.hist2d(
                            results[i][var_x["name"]],
                            results[i][var_y["name"]],
                            bins=(
                                var_x["bins"],
                                var_y["bins"],
                            ),
                            range=(
                                (var_x["min"], var_x["max"]),
                                (var_y["min"], var_y["max"]),
                            ),
                            cmap=cmaps[i],
                            alpha=0.5,
                        )
                    ax.set_xlim(var_x["min"], var_x["max"])
                    ax.set_ylim(var_y["min"], var_y["max"])

        for i_x, var_x in enumerate(var_items):
            for i_y, var_y in enumerate(var_items):
                ax: plt.Axes = axs_joint[i_y, i_x]

                # in-axis title
                title = ""
                if i_x == i_y:
                    title = var_x["equation"] + " Histogram"
                elif i_x < i_y:
                    title = var_y["equation"] + "-" + var_x["equation"] + " Scatter"
                else:
                    title = var_y["equation"] + "-" + var_x["equation"] + " 2D Hist"

                ax.text(
                    0.05,
                    0.95,
                    title,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    color="gray",
                    fontstyle="italic",
                    alpha=0.7,
                )

                # x-axis labels
                if i_y == 3:
                    ax.set_xlabel(var_x["label"])
                    if var_x["ticks"] is not None:
                        ax.set_xticks(var_x["ticks"])
                        ax.set_xticklabels(var_x["ticklabels"], rotation=30)
                elif i_y == 0:
                    # draw x-axis labels on the top side for the first row
                    ax.xaxis.set_label_position("top")
                    ax.xaxis.set_ticks_position("top")
                    ax.set_xlabel(var_x["label"])
                    if var_x["ticks"] is not None:
                        ax.set_xticks(var_x["ticks"])
                        ax.set_xticklabels(var_x["ticklabels"], rotation=30)
                else:
                    if var_x["ticks"] is not None:
                        ax.set_xticks(var_x["ticks"])
                    ax.set_xticklabels([])

                # y-axis labels
                if i_x == 0:
                    if i_y != i_x:
                        ax.set_ylabel(var_y["label"])
                        if var_y["ticks"] is not None:
                            ax.set_yticks(var_y["ticks"])
                            ax.set_yticklabels(var_y["ticklabels"])
                    else:
                        ax.set_ylabel("Counts")
                elif i_x == 3:
                    # draw y-axis labels on the right side for the last column
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.set_ticks_position("right")
                    if i_y != i_x:
                        ax.set_ylabel(var_y["label"])
                        if var_y["ticks"] is not None:
                            ax.set_yticks(var_y["ticks"])
                            ax.set_yticklabels(var_y["ticklabels"])
                    else:
                        ax.set_ylabel("Counts")
                elif i_x == i_y:
                    ax.set_ylabel("")
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                else:
                    if var_y["ticks"] is not None:
                        ax.set_yticks(var_y["ticks"])
                    ax.set_yticklabels([])

                ax.xaxis.set_ticks_position("both")
                if i_x != i_y:
                    ax.yaxis.set_ticks_position("both")
                ax.tick_params(direction="in", which="both")

        # Trapped theta plots
        theta_start_all = np.concatenate(
            [results[i]["theta_start"] for i in range(len(results))]
        )
        theta_center_all = np.concatenate(
            [results[i]["theta_center"] for i in range(len(results))]
        )

        theta_config = var_items[1]  # theta_center config

        # theta histograms
        theta_start_hist = ax_theta_hist.hist(
            theta_start_all,
            bins=theta_config["bins"],
            range=(theta_config["min"], theta_config["max"]),
            histtype="step",
            color="blue",
            alpha=0.5,
            label=r"$\theta_{\mathrm{start}}$",
        )
        ax_theta_hist.hist(
            theta_center_all,
            bins=theta_config["bins"],
            range=(theta_config["min"], theta_config["max"]),
            histtype="step",
            color="green",
            alpha=0.5,
            label=r"$\theta_{\mathrm{center}}$",
        )

        analytic_theta = np.linspace(theta_config["min"], theta_config["max"], 1000)
        analytic_y = np.sin(analytic_theta)
        analytic_y *= theta_start_hist[0].max() / analytic_y.max()  # normalize
        ax_theta_hist.plot(
            analytic_theta,
            analytic_y,
            color="black",
            linestyle="--",
            label=r"$\sin(\theta)$",
        )

        ax_theta_hist.set_xlabel("Trapped Pitch Angle " + r"$\theta$" + " [rad]")
        ax_theta_hist.set_xticks(theta_config["ticks"])
        ax_theta_hist.set_xticklabels(theta_config["ticklabels"])
        ax_theta_hist.set_xlim(theta_config["min"], theta_config["max"])
        ax_theta_hist.set_ylabel("Counts")

        ticks_degree_min = np.degrees(theta_config["min"])
        ticks_degree_min = int(np.ceil(ticks_degree_min))
        ticks_degree_max = np.degrees(theta_config["max"])
        ticks_degree_max = int(np.floor(ticks_degree_max))

        ticks_degree = np.arange(ticks_degree_min, ticks_degree_max + 1, dtype=int)
        ticks_degree_labels = [f"{d:d}" + r"$^\circ$" for d in ticks_degree]
        ticks_degree_positions = np.radians(ticks_degree)

        ax_theta_hist_twiny = ax_theta_hist.twiny()
        ax_theta_hist_twiny.set_xticks(ticks_degree_positions)
        ax_theta_hist_twiny.set_xticklabels(ticks_degree_labels)
        ax_theta_hist_twiny.set_xlim(theta_config["min"], theta_config["max"])

        ax_theta_hist.legend()

        # theta transformation scatter
        ax_theta_transform.scatter(
            theta_start_all,
            theta_center_all,
            color="purple",
            s=1,
            alpha=0.01,
        )

        ax_theta_transform.set_xlabel(
            r"Trapped $\theta_{\mathrm{start}}$ [rad]"
        )
        ax_theta_transform.set_xticks(theta_config["ticks"])
        ax_theta_transform.set_xticklabels(theta_config["ticklabels"])
        ax_theta_transform.set_xlim(theta_config["min"], theta_config["max"])

        ax_theta_transform.set_ylabel(
            r"Trapped $\theta_{\mathrm{center}}$ [rad]"
        )
        ax_theta_transform.set_yticks(theta_config["ticks"])
        ax_theta_transform.set_yticklabels(theta_config["ticklabels"])
        ax_theta_transform.set_ylim(theta_config["min"], theta_config["max"])

        ax_theta_transform_twiny = ax_theta_transform.twiny()
        ax_theta_transform_twiny.set_xticks(ticks_degree_positions)
        ax_theta_transform_twiny.set_xticklabels(ticks_degree_labels)
        ax_theta_transform_twiny.set_xlim(theta_config["min"], theta_config["max"])

        ax_theta_transform_twinx = ax_theta_transform_twiny.twinx()
        ax_theta_transform_twinx.set_yticks(ticks_degree_positions)
        ax_theta_transform_twinx.set_yticklabels(ticks_degree_labels)
        ax_theta_transform_twinx.set_ylim(theta_config["min"], theta_config["max"])

        plt.savefig("DataGenerator4D_test.png", dpi=300)


if __name__ == "__main__":
    unittest.main()
