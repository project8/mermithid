"""
Script to make Sensitivty plots for cavity experiments
Author: C. Claessens, T. Weiss
Date: October 6, 2023
"""


from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

from mermithid.processors.Sensitivity import CavitySensitivityCurveProcessor

import numpy as np

# Configuration for Sensitivity vs. exposure plot
# Phase III and IV at 325 MHz
# Phase II point and curve for comparison

sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment_conservative.cfg",
    "plot_path": "./sensitivity_vs_exposure_curve.pdf",
    # optional
    "figsize": (10,6),
    "fontsize": 15,
    "legend_location": "upper right",
    "track_length_axis": False,
    "molecular_axis": False,
    "atomic_axis": False,
    "density_axis": False,
    "cavity": True,
    "add_PhaseII": True,
    "add_1year_1cav_point_to_last_ref": True,
    "PhaseII_config_path": "/termite/sensitivity_config_files/Config_PhaseII_Experiment.cfg",
    "y_limits": [10e-3, 500],
    "density_range": [1e12,1e19],
    "exposure_range": [1e-11, 1e4],
    "main_curve_upper_label": r"Molecular, conservative",
    "main_curve_lower_label": r"$\sigma^\bar{B}_\mathrm{reco} = 0.07\,\mathrm{eV}$",
    "goals": {"Phase III (0.2 eV)": (0.2**2/np.sqrt(1.64)), "Phase IV (0.04 eV)": (0.04**2/np.sqrt(1.64))},
    "comparison_curve": True,
    "comparison_config_file_path": ["/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment.cfg", 
                                    "/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment_conservative.cfg", 
                                    "/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg"],
    "comparison_curve_label": [r"Molecular, reaching PIII target", "Atomic, conservative", "Atomic, reaching PIV target"],
    "comparison_curve_colors": ["blue", "darkred", "red"],
    #"B_inhomogeneity": np.linspace(0.1, 2.1, 10)*1e-6,
    #"B_inhom_uncertainty": 0.01,
    "sigmae_theta_r": 0.159, #in eV, energy broadening from theta and r reconstruction
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 0.015, #1.5e15, #0.02, #1e14,
    "goals_x_position": 0.2e-10, #2e12 #0.0002
    "goals_y_rel_position": 0.4
    }
#sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
#sens_curve.Configure(sens_config_dict)
#sens_curve.Run()


# Configuration for Sensitivity vs. frequency plot
sens_config_dict2 = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment_conservative.cfg",
    "plot_path": "./sensitivity_vs_frequency2.pdf",
    # optional
    "figsize": (9,6),
    "fontsize": 15,
    "legend_location": "upper left",
    "track_length_axis": False,
    "molecular_axis": False,
    "atomic_axis": False,
    "density_axis": False,
    "frequency_axis": True,
    "magnetic_field_axis": True,
    "cavity": True,
    "add_PhaseII": False,
    "add_1year_1cav_point_to_last_ref": False,
    "PhaseII_config_path": "/termite/sensitivity_config_files/Config_PhaseII_Experiment.cfg",
    "y_limits": [10e-3, 10],
    "frequency_range": [1e7, 20e9],
    #"efficiency_range": [0.0001, 1],
    "density_range": [1e7, 1e20],
    "main_curve_upper_label": r"Molecular, conservative",
    "main_curve_lower_label": r"$\sigma^\bar{B}_\mathrm{reco} = 0.07\,\mathrm{eV}$",
    "goals": {"Phase IV (0.04 eV)": (0.04**2/np.sqrt(1.64))},
    "comparison_curve": True,
    "comparison_config_file_path": ["/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment.cfg", 
                                    "/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment_conservative.cfg", 
                                    "/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg"],
    "comparison_curve_label": [r"Molecular, reaching PIII target", "Atomic, conservative", "Atomic, reaching PIV target"],
    "comparison_curve_colors": ["blue", "darkred", "red"],
    #"config_file_path": "/host_repos/sensitivity_branches/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg",
    #"B_inhomogeneity": np.linspace(0.1, 2.1, 10)*1e-6,
    #"B_inhom_uncertainty": 0.01,
    "sigmae_theta_r": 0.159, #in eV, energy broadening from theta and r reconstruction
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 1e8, #1.5e15, #0.02, #1e14,
    "goals_x_position": 1e9, #2e12 #0.0002
    "goals_y_rel_position": 0.4,
    "plot_key_parameters": True
    }
#sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
#sens_curve.Configure(sens_config_dict2)
#sens_curve.Run()

# Configuration for Sensitivity vs. density plot for different B sigmas
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment.cfg",
    "plot_path": "./sensitivity_vs_density_curve_with_sigmaBcorrs.pdf",
    # optional
    "track_length_axis": True,
    "molecular_axis": True,
    "atomic_axis": True,
    "density_axis": True,
    "cavity": True,
    "y_limits": [2e-2, 4],
    "density_range": [1e12,1e19],
    "efficiency_range": [0.0001, 1],
    #"density_range": [1e8, 1e12],
    "main_curve_upper_label": r"Molecular"+"\n"+"2 years"+"\n"+r"$\sigma^B_\mathrm{corr} = 1\,\mathrm{eV}$",
    "main_curve_lower_label": r"$\sigma^B_\mathrm{corr} = 0.16\,\mathrm{eV}$",
    "comparison_curve_label": [r"Atomic"+"\n"+r"10 $\times$ 3 years"+"\n"+r"$\sigma^B_\mathrm{corr} = 0.16\,\mathrm{eV}$"],
    "goals": {"Phase III (0.2 eV)": 0.2, "Phase IV (0.04 eV)": 0.04},
    "comparison_curve": True,
    "comparison_config_file_path": ["/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg"],
    #"comparison_config_file_path": "/host_repos/sensitivity_branches/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg",
    #"B_inhomogeneity": np.linspace(0.1, 2.1, 10)*1e-6,
    #"B_inhom_uncertainty": 0.01,
    "sigmae_theta_r": np.linspace(0.16, 1., 10), #in eV, energy broadening from theta and r reconstruction
    "comparison_label_y_position": [0.044],
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 1.5e15, #0.02, #1e14,
    "goals_x_position": 2e12, #0.0002
    "plot_key_parameters": True
    }
#sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
#sens_curve.Configure(sens_config_dict)
#sens_curve.Run()


# Configuration for Sensitivity vs. density plot
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg",
    "plot_path": "./sensitivity_vs_density_curve.pdf",
    # optional
    "figsize": (7.0,6),
    "track_length_axis": True,
    "molecular_axis": False,
    "atomic_axis": True,
    "density_axis": True,
    "cavity": True,
    "y_limits": [2e-2, 4],
    "density_range": [1e12,3e18],
    "efficiency_range": [0.0001, 1],
    #"density_range": [1e8, 1e12],
    "main_curve_upper_label": r"Atomic, reaching target", #r"Molecular"+"\n"+"Reaching target",
    "comparison_curve_label": [#"Molecular, reaching target", 
                                "Atomic, alternative scenario 1"], 
                                #"Atomic, reaching target"], 
                                # #["Molecular"+"\n"+"Conservative", "Atomic"+"\n"+"Conservative", r"Atomic"+"\n"+"Reaching target"],
    "goals": {"Pilot T goal (0.1 eV)": 0.1, "Phase IV (0.04 eV)": 0.04},
    "comparison_curve": True,
    "comparison_curve_colors": ["red"],
    "comparison_config_file_path": [#"/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment.cfg", 
                                    "/termite/sensitivity_config_files/Config_atomic_find_factor_22_Experiment_conservative.cfg"], #, 
                                    #"/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg"],
    "comparison_label_y_position": [2, 0.105, 0.046], #[2, 0.105, 0.046],
    "comparison_label_x_position": [4.5e15, 7e14, 7e14], #[4.5e15, 2.2e16, 1e15],
    #"sigmae_theta_r": 0.159,
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": True
    }
sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
sens_curve.Configure(sens_config_dict)
sens_curve.Run()


# Configuration for Sensitivity vs. density plot for best possible molecular scenario
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment_best_case.cfg",
    "plot_path": "./sensitivity_vs_density_T2_best_case_curve.pdf",
    # optional
    "figsize": (6.7,6),
    "track_length_axis": False,
    "molecular_axis": True,
    "atomic_axis": False,
    "density_axis": True,
    "cavity": True,
    "y_limits": [2e-2, 4],
    "density_range": [1e12,3e18],
    "efficiency_range": [0.0001, 1],
    "main_curve_upper_label": r"Molecular, best-case scenario",
    "goals": {"Phase III (0.2 eV)": 0.2, "Phase IV (0.04 eV)": 0.04},
    "comparison_curve": False,
    "sigmae_theta_r": 0.159,
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 4e14, 
    "goals_x_position": 1.2e12, 
    "plot_key_parameters": True
    }
#sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
#sens_curve.Configure(sens_config_dict)
#sens_curve.Run()

# Configuration for Sensitivity vs. densty plot for molecular best case and Phase III atomic scenarios
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg",
    "plot_path": "./sensitivity_vs_density_T2_best_case_curve_comparison.pdf",
    # optional
    "figsize": (6.7,6),
    "legend_location": "upper left",
    "track_length_axis": True,
    "molecular_axis": True,
    "atomic_axis": True,
    "density_axis": True,
    "cavity": True,
    "y_limits": [3e-2, 0.5],
    "density_range": [1e14, 7e17],#[1e12,3e18],
    "efficiency_range": [0.0001, 1],
    "main_curve_upper_label": r"Atomic, $\Delta B_{r, \phi, t}=0$, rate 'boosted' $\times 2$",
    "goals": {"Phase IV (0.04 eV)": 0.04},
    "comparison_curve": True,
    "comparison_config_file_path": ["/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment_best_case.cfg"],
    "comparison_curve_label": [r"Molecular, same conditions"], 
    "comparison_label_y_position": [2, 0.105, 0.046],
    "comparison_label_x_position": [4.5e15, 7e14, 7e14],
    "sigmae_theta_r": 0.0,
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 4e14, 
    "goals_x_position": 1.2e14, 
    "goals_y_rel_position": 1.1,
    "plot_key_parameters": True
    }
#sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
#sens_curve.Configure(sens_config_dict)
#sens_curve.Run()

# Configuration for Sensitivity vs. density plot with 0 positional field uncertainty
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg",
    "plot_path": "./sensitivity_vs_density_T2_no-DeltaB-r-phi-t.pdf",
    # optional
    "figsize": (7.5,5),
    "track_length_axis": False,
    "molecular_axis": False,
    "atomic_axis": True,
    "density_axis": True,
    "cavity": True,
    "y_limits": [3e-2, 0.5], #[2e-2, 4],
    "density_range": [1e14, 7e17], #[1e12,3e18],
    "efficiency_range": [0.0001, 1],
    "main_curve_upper_label": r" ", #r"Atomic, $\Delta B_{r, \phi, t}=0$, rate 'boosted' $\times 2$",
    "goals": {"Phase IV (0.04 eV)": 0.04},
    "comparison_curve": False,
    "sigmae_theta_r": 0.0,
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 4e14, 
    "goals_x_position": 1.2e14, 
    "goals_y_rel_position": 1.1,
    #"legend_location": "upper left",
    "plot_key_parameters": True
    }
#sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
#sens_curve.Configure(sens_config_dict)
#sens_curve.Run()

# Configuration for CCA Sensitivity vs. density plot
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_CCA_Experiment.cfg",
    "plot_path": "./cca_sensitivity_vs_density_curve.pdf",
    # optional
    "figsize": (7.0,6),
    "track_length_axis": True,
    "molecular_axis": True,
    "atomic_axis": False,
    "density_axis": True,
    "cavity": True,
    "y_limits": [2e-2, 20],
    "density_range": [1e13,1e21],
    "efficiency_range": [0.0001, 1],
    #"density_range": [1e8, 1e12],
    "main_curve_upper_label": r"CCA", #r"Molecular"+"\n"+"Reaching target",
    "comparison_curve_label": [#"Molecular, reaching target", 
                                "Atomic, alternative scenario 1"], 
                                #"Atomic, reaching target"], 
                                # #["Molecular"+"\n"+"Conservative", "Atomic"+"\n"+"Conservative", r"Atomic"+"\n"+"Reaching target"],
    "goals": {"Pilot T goal (0.1 eV)": 0.1, "Phase IV (0.04 eV)": 0.04},
    "comparison_curve": False,
    "comparison_curve_colors": ["red"],
    "comparison_config_file_path": [#"/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment.cfg", 
                                    "/termite/sensitivity_config_files/Config_atomic_find_factor_22_Experiment_conservative.cfg"], #, 
                                    #"/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg"],
    "comparison_label_y_position": [2, 0.105, 0.046], #[2, 0.105, 0.046],
    "comparison_label_x_position": [4.5e15, 7e14, 7e14], #[4.5e15, 2.2e16, 1e15],
    #"sigmae_theta_r": 0.159,
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": True
    }
#sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
#sens_curve.Configure(sens_config_dict)
#sens_curve.Run()








