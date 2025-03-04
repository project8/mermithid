"""
Script to make sensitivity plots for cavity experiments
Author: C. Claessens, T. E. Weiss
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
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_150MHz_minpitch_87deg.cfg", #Config_PhaseIII_325MHz_Experiment_conservative.cfg",
    "plot_path": "./sensitivity_vs_exposure_curve_PhaseIV_87deg_min_pitch.pdf",
    # optional
    "figsize": (10,6),
    "fontsize": 15,
    "legend_location": "upper right",
    "track_length_axis": False,
    "molecular_axis": False,
    "atomic_axis": False,
    "density_axis": False,
    "exposure_axis": True,
    "cavity": True,
    "add_PhaseII": True,
    "add_1year_1cav_point_to_last_ref": True,
    "PhaseII_config_path": "/termite/sensitivity_config_files/Config_PhaseII_Experiment.cfg",
    "y_limits": [10e-3, 500],
    "density_range": [0, 1e19],
    "exposure_range": [1e-11, 1e4],
    "main_curve_upper_label": r"Phase IV scenario", #"Molecular, conservative",
    #"main_curve_lower_label": r"$\sigma^\bar{B}_\mathrm{reco} = 0.07\,\mathrm{eV}$",
    "goals": {"Phase IV (0.04 eV)": 0.04},
    "comparison_curve": False,
    "comparison_config_file_path": ["/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment.cfg", 
                                    "/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment_conservative.cfg", 
                                    "/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg"
                                    ],
    "comparison_curve_label": [r"Molecular, reaching PIII target", "Atomic, conservative", "Atomic, reaching PIV target"],
    "comparison_curve_colors": ["blue", "darkred", "red"],
    #"B_inhomogeneity": np.linspace(0.1, 2.1, 10)*1e-6,
    #"B_inhom_uncertainty": 0.01,
    #"sigmae_theta_r": 0.159, #in eV, energy broadening from theta and r reconstruction
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
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_150MHz_minpitch_87deg.cfg",
    "plot_path": "./sensitivity_vs_frequency_Oct-22-2024.pdf",
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
    "goals": {"Phase IV (0.04 eV)": (0.04**2/1.64)},
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
    "plot_key_parameters": False
    }
#sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
#sens_curve.Configure(sens_config_dict2)
#sens_curve.Run()

# Configuration for Sensitivity vs. density plot for different B sigmas
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment_conservative.cfg",
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
    "comparison_config_file_path": ["/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment_conservative.cfg"],
    #"comparison_config_file_path": "/host_repos/sensitivity_branches/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg",
    #"B_inhomogeneity": np.linspace(0.1, 2.1, 10)*1e-6,
    #"B_inhom_uncertainty": 0.01,
    "configure_sigma_theta_r": True,
    "sigmae_theta_r": np.linspace(0.16, 1., 10), #in eV, energy broadening from theta and r reconstruction
    "comparison_label_y_position": [0.044],
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 1.5e15, #0.02, #1e14,
    "goals_x_position": 2e12, #0.0002
    "plot_key_parameters": False
    }
#sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
#sens_curve.Configure(sens_config_dict)
#sens_curve.Run()


# Configuration for Sensitivity vs. density plot
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_LFA_Experiment_max_BNL_diam.cfg", #"/termite/sensitivity_config_files/Config_LFA_Experiment_1GHz.cfg", #Config_atomic_325MHz_Experiment_conservative.cfg",
    "plot_path": "./LFA_and_PhaseIV_sensitivity_vs_density_target_March-3-2025.pdf",
    # optional
    "figsize": (7.5,6.4), 
    "fontsize": 15,
    "track_length_axis": False,
    "legend_location": "upper left",
    "legend_bbox_to_anchor": (0.155,-0.01,1.12,0.885), #(0.17,-0.01,1.1,0.87),
    "molecular_axis": False,
    "atomic_axis": True,
    "density_axis": True,
    "track_length_axis": True,
    "effs_for_sampled_radii": True,
    "y_limits": [2e-2, 4],
    "density_range": [3e14,3e18], #5e13
    "det_thresh_range": [5, 115],
    "main_curve_upper_label":  r"LFA (Phase III): 560 MHz", #Phase III scenario: 1 GHz",
    "goals": {"LFA (0.45 eV)": 0.45, "Phase IV (0.04 eV)": 0.04}, #"Pilot T goal (0.1 eV)": 0.1,
    "comparison_curve": True,
    "main_curve_color": "blue",
    "comparison_curve_colors": ["red", "black"],
    "comparison_config_file_path": ["/termite/sensitivity_config_files/Config_atomic_150MHz_minpitch_87deg.cfg", "/termite/sensitivity_config_files/Config_PIVmodule1_150MHz_minpitch_87deg.cfg"], 
    "comparison_curve_label": [r"Phase IV scenario: 150 MHz", r"One Phase IV cavity"],
    "comparison_label_y_position": [2, 0.105, 0.046], #[2, 0.105, 0.046],
    "comparison_label_x_position": [4.5e15, 7e14, 7e14], #[4.5e15, 2.2e16, 1e15],
    #"sigmae_theta_r": 0.159,
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 6e14, #3.3e14, #5.5e13, 
    "plot_key_parameters": True,
    "goals_y_rel_position": 0.79 #0.755
    }
#sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
#sens_curve.Configure(sens_config_dict)
#sens_curve.Run()


sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_LFA_Experiment_max_BNL_diam.cfg", #"/termite/sensitivity_config_files/Config_LFA_Experiment.cfg",
    "plot_path": "./lfa_and_PhaseIV_sensitivity_vs_ncav-eff-time_curve_target_March-3-2025.pdf", #ncav-eff-time
    "exposure_axis": True,
    # optional
    "figsize": (8.3, 6.3), #(10,6),
    "fontsize": 15,
    "legend_location": "upper right",
    "legend_bbox_to_anchor": (-0.,0,0.86,0.955),
    "track_length_axis": False,
    "molecular_axis": False,
    "atomic_axis": False,
    "exposure_axis": False,
    "density_axis": False,
    "ncav_eff_time_axis": True,
    "ncavities_livetime_axis": False,
    "livetime_axis": False,
    "cavity": True,
    "add_PhaseII": False,
    "PhaseII_config_path": "/termite/sensitivity_config_files/Config_PhaseII_Experiment.cfg",
    "add_1year_1cav_point_to_last_ref": True,
    "y_limits": [2.5e-2, 1],
    #"density_range": [1e12,1e19],
    "year_range": [0.1,10**3],
    "main_curve_upper_label":  r"LFA (Phase III): 560 MHz, V = 1.7 m$^3$",
    "goals": {"LFA (0.45 eV)": 0.45, "Phase IV (0.04 eV)": 0.04},
    "comparison_curve": True,
    "comparison_config_file_path": ["/termite/sensitivity_config_files/Config_atomic_150MHz_minpitch_87deg.cfg"], #"/termite/sensitivity_config_files/Config_PIVmodule1_150MHz_minpitch_87deg.cfg"],
    "comparison_curve_label": [r"Phase IV scenario: 150 MHz, V = 94 m$^3 \times 10$"], #, "One Phase IV cavity"],
    "main_curve_color": "blue",
    "comparison_curve_colors": ["red", "black"], 
    "optimize_main_density": False,
    "optimize_comparison_density": False,
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 0.115, 
    "goals_x_position": 4e-2, #<-- Number for ncav*eff*time   #0.11, <-- Number for ncavities*livetime
    "goals_y_rel_position": 0.86, #0.84, <-- Number for ncav*eff*time   #0.81, <-- Number for ncavities*livetime
    }
sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
sens_curve.Configure(sens_config_dict)
sens_curve.Run()



sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_LFA_Experiment_1GHz.cfg", #"/termite/sensitivity_config_files/Config_LFA_Experiment.cfg",
    "plot_path": "./lfa_and_PhaseIV_sensitivity_vs_exposure_curve_Oct-22-2024.pdf",
    "exposure_axis": True,
    # optional
    "figsize": (10,6),
    "fontsize": 15,
    "legend_location": "upper right",
    "track_length_axis": False,
    "molecular_axis": False,
    "atomic_axis": False,
    "exposure_axis": True,
    "density_axis": False,
    "cavity": True,
    "add_PhaseII": True,
    "PhaseII_config_path": "/termite/sensitivity_config_files/Config_PhaseII_Experiment.cfg",
    "add_1year_1cav_point_to_last_ref": False,
    "y_limits": [10e-3, 500],
    #"density_range": [1e12,1e19],
    "exposure_range": [1e-11, 1e4],
    #"main_curve_upper_label": r"LFA (atomic T)$\,-\,$1 GHz",
    "main_curve_upper_label":  r"Phase III scenario: 1 GHz",
    #"main_curve_lower_label": r"$\sigma^\bar{B}_\mathrm{reco} = 0.16\,\mathrm{eV}$",
    "goals": {"Phase IV (0.04 eV)": (0.04)},
    "comparison_curve": True,
    "comparison_config_file_path": ["/termite/sensitivity_config_files/Config_LFA_Experiment_max_BNL_diam.cfg",
                                    "/termite/sensitivity_config_files/Config_atomic_150MHz_minpitch_87deg.cfg"], 
    "comparison_curve_label": [r"Phase III scenario: 560 MHz", r"Phase IV scenario: 150 MHz"], 
    "main_curve_color": "blue",
    "comparison_curve_colors": ["red", "black"],
    "optimize_main_density": False,
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 0.015, #1.5e15, #0.02, #1e14,
    "goals_x_position": 0.2e-10, #2e12 #0.0002
    "goals_y_rel_position": 0.4,
    }
#sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
#sens_curve.Configure(sens_config_dict)
#sens_curve.Run()


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
    "plot_key_parameters": False
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
    "plot_key_parameters": False
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
    "config_file_path": "/termite/sensitivity_config_files/Config_CCA_pitchAngUncertVerification.cfg",
    "plot_path": "./cca_sensitivity_vs_density_curve_new_config.pdf",
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







