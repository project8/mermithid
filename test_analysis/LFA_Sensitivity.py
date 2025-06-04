"""
Script to make Sensitivty plots for cavity experiments
Author: C. Claessens, T. Weiss
Date: October 6, 2023
"""


from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

from mermithid.processors.Sensitivity import CavitySensitivityCurveProcessor

import numpy as np

# Configuration for CCA Sensitivity vs. density plot
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/config_LFA_Experiment_1GHz.cfg", #Config_LFA_Experiment_1GHz.cfg", 
    #"config_file_path": "/host_repos/sensitivity_branches/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment.cfg",
    "plot_path": "./lfa_sensitivity_vs_density_curve.pdf",
    # optional
    "figsize": (7.0,5), #Was (7, 6)
    "track_length_axis": True,
    "molecular_axis": False,
    "atomic_axis": False,
    "density_axis": True,
    "optimize_main_density": True,
    "cavity": True,
    #"y_limits": [2e-2, 20],
    "y_limits": [2e-1, 20],
    "density_range": [1e13,1e20],
    #"efficiency_range": [0.0001, 1],
    #"density_range": [1e8, 1e12],
    "configure_sigma_theta_r": True,
    "sigmae_theta_r": np.linspace(0.1, 0.5, 10), #in eV, energy broadening from theta and r reconstruction
    "main_curve_upper_label": r"$\sigma^B_\mathrm{corr} = 0.5\,\mathrm{eV}$", #r"Molecular"+"\n"+"Reaching target",
    "main_curve_lower_label": r"$\sigma^B_\mathrm{corr} = 0.1\,\mathrm{eV}$", #r"Molecular"+"\n"+"Reaching target",
    #"comparison_curve_label": [#"Molecular, reaching target", 
    #                            "Atomic, alternative scenario 1"], 
    #                            #"Atomic, reaching target"], 
    #                            # #["Molecular"+"\n"+"Conservative", "Atomic"+"\n"+"Conservative", r"Atomic"+"\n"+"Reaching target"],
    #"goals": {"Pilot T goal (0.1 eV)": 0.1, "Phase IV (0.04 eV)": 0.04},
    "comparison_curve": False,
    "comparison_curve_colors": ["red"],
    #"comparison_config_file_path": [#"/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment.cfg", 
    #                                "/termite/sensitivity_config_files/Config_atomic_find_factor_22_Experiment_conservative.cfg"], #, 
    #                                #"/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg"],
    "comparison_label_y_position": [2, 0.105, 0.046], #[2, 0.105, 0.046],
    "comparison_label_x_position": [4.5e15, 7e14, 7e14], #[4.5e15, 2.2e16, 1e15],
    #"sigmae_theta_r": 0.159,
    "lower_label_y_position": 0.4,
    "upper_label_y_position": 4,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 1.2e14, #0.0002
    "plot_key_parameters": True
    }
#sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
#sens_curve.Configure(sens_config_dict)
#sens_curve.Run()


sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_LFA_Experiment_max_BNL_diam.cfg", #"/termite/sensitivity_config_files/Config_LFA_Experiment.cfg",
    "plot_path": "./lfa_with_BNL_constraints_sensitivity_vs_exposure_curve_PhaseIV.pdf",
    "exposure_axis": True,
    # optional
    "figsize": (10,6),
    "fontsize": 15,
    "legend_location": "upper right",
    "track_length_axis": False,
    "molecular_axis": False,
    "atomic_axis": False,
    "density_axis": False,
    "cavity": True,
    "add_1year_1cav_point_to_last_ref": False,
    "y_limits": [10e-3, 500],
    "density_range": [1e12,1e19],
    "exposure_range": [1e-11, 1e4],
    #"main_curve_upper_label": r"LFA (atomic T)$\,-\,$1 GHz",
    "main_curve_upper_label":  r"Phase III scenario: 1 GHz, L/D=9",
    #"main_curve_lower_label": r"$\sigma^\bar{B}_\mathrm{reco} = 0.16\,\mathrm{eV}$",
    "goals": {"Phase III (0.4 eV)": (0.4**2/1.64), "Phase IV (0.04 eV)": (0.04**2/1.64)},
    "comparison_curve": False,
    "comparison_config_file_path": ["/termite/sensitivity_config_files/Config_LFA_Experiment_max_BNL_diam.cfg",
                                    "/termite/sensitivity_config_files/Config_LFA_Experiment_max_BNL_diam.cfg"],
                                   # "/termite/sensitivity_config_files/Config_atomic_find_factor_22_Experiment_conservative.cfg"], 
    #"comparison_config_file_path": ["/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment_conservative.cfg", 
    #                                "/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment_conservative.cfg", 
    #                                "/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg"],
    #"comparison_curve_label": [r"Molecular, conservative", "Atomic, conservative", "Atomic, reaching PIV target"],
    "comparison_curve_label": [r"Phase III scenario: 500 MHz, L/D=5", r"Phase III scenario: 560 MHz, L/D=8"], #r"Phase IV scenario: 150 MHz, L/D=5"],
    #"comparison_curve_colors": ["blue", "darkred", "red"],
    "comparison_curve_colors": ["blue", "darkviolet"],
    "optimize_main_density": False,
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 0.015, #1.5e15, #0.02, #1e14,
    "goals_x_position": 0.2e-10, #2e12 #0.0002
    "goals_y_rel_position": 0.4,
    "add_PhaseII": False
    #"PhaseII_config_path": "/termite/sensitivity_config_files/Config_PhaseII_Experiment.cfg"
    }
sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
sens_curve.Configure(sens_config_dict)
sens_curve.Run()






