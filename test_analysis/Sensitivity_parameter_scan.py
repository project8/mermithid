"""
Script to make Sensitivty plots for cavity experiments
Author: C. Claessens, T. Weiss
Date: October 6, 2023
"""


from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

from mermithid.processors.Sensitivity import SensitivityParameterScanProcessor

import numpy as np
from numericalunits import eV
deg = np.pi/180




# Configuration for Sensitivity vs. density plot
# Currently comparing conservative atomic vs. scenario that reaches target without statistics boost
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_find_factor_22_Experiment_conservative.cfg",
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
    #"density_range": [1e8, 1e12],
    "goals": {"Pilot T goal (0.1 eV)": 0.1, "Phase IV (0.04 eV)": 0.04},
    "scan_parameter_name": "MagneticField.sigmae_r",
    "scan_parameter_range": [0.1, 5],
    "scan_parameter_steps": 3,
    "scan_parameter_unit": eV,
    "plot_sensitivity_scan_on_log_scale": True,
    #"sigmae_theta_r": 0.159,
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False
    }
#sens_curve = SensitivityParameterScanProcessor("sensitivity_curve_processor")
#sens_curve.Configure(sens_config_dict)
#sens_curve.Run()


# Configuration for Sensitivity vs. density plot
# Currently comparing conservative atomic vs. scenario that reaches target without statistics boost
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_find_factor_22_Experiment_conservative.cfg",
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
    #"density_range": [1e8, 1e12],
    "goals": {"Pilot T goal (0.1 eV)": 0.1, "Phase IV (0.04 eV)": 0.04},
    "scan_parameter_name": "FrequencyExtraction.minimum_angle_in_bandwidth",
    "scan_parameter_range": [83, 89],
    "scan_parameter_steps": 3,
    "scan_parameter_unit": deg,
    "plot_sensitivity_scan_on_log_scale": True,
    #"sigmae_theta_r": 0.159,
    "lower_label_y_position": 0.17,
    "upper_label_y_position": 0.7,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False
    }
sens_curve = SensitivityParameterScanProcessor("sensitivity_curve_processor")
sens_curve.Configure(sens_config_dict)
sens_curve.Run()











