"""
Script to make Sensitivty plots for cavity experiments
Author: C. Claessens, T. Weiss
Date: October 6, 2023
"""


from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

from mermithid.processors.Sensitivity import SensitivityParameterScanProcessor

import numpy as np

# import all the scanned parameter units
from numericalunits import eV, K, mK, T # whatever you need
deg = np.pi/180




# Configuration for magnetic homogeneity scan
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_find_factor_22_Experiment_conservative.cfg",
    "plot_path": ".",
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
    "goals": {"Phase IV (0.04 eV)": 0.04},
    "scan_parameter_name": "MagneticField.sigmae_r",
    "scan_parameter_range": [0.1, 5],
    "scan_parameter_steps": 4,
    "scan_parameter_unit": eV,
    "plot_sensitivity_scan_on_log_scale": False,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False
    }
#sens_scan = SensitivityParameterScanProcessor("sensitivity_curve_processor")
#sens_scan.Configure(sens_config_dict)
#sens_scan.Run()


# Configuration for minimum pitch angle scan
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_150MHz_minpitch_87deg.cfg",
    "plot_path": ".",
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
    "goals": {"Phase IV (0.04 eV)": 0.04},
    "scan_parameter_name": "FrequencyExtraction.minimum_angle_in_bandwidth",
    "scan_parameter_range": [83, 89],
    "scan_parameter_steps": 10,
    "scan_parameter_unit": deg,
    "plot_sensitivity_scan_on_log_scale": False,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False
    }
#sens_scan = SensitivityParameterScanProcessor("sensitivity_curve_processor")
#sens_scan.Configure(sens_config_dict)
#sens_scan.Run()


# Configuration for L over D scan
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_150MHz_minpitch_87deg.cfg",
    "plot_path": ".",
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
    "goals": {"Phase IV (0.04 eV)": 0.04},
    "scan_parameter_name": "Experiment.l_over_d",
    "scan_parameter_range": [5, 8],
    "scan_parameter_steps": 3,
    "scan_parameter_unit": 1,
    "plot_sensitivity_scan_on_log_scale": False,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False
    }
#sens_scan = SensitivityParameterScanProcessor("sensitivity_curve_processor")
#sens_scan.Configure(sens_config_dict)
#sens_scan.Run()

# Configuration for efficiency scan
sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_150MHz_minpitch_87deg.cfg",
    "plot_path": ".",
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
    "goals": {"Phase IV (0.04 eV)": 0.04},
    "scan_parameter_name": "Experiment.livetime",
    "scan_parameter_range": [0.1*3e7, 20*3e7],
    "scan_parameter_steps": 10,
    "scan_parameter_unit": 1,
    "plot_sensitivity_scan_on_log_scale": False,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False
    }
sens_scan = SensitivityParameterScanProcessor("sensitivity_curve_processor")
sens_scan.Configure(sens_config_dict)
sens_scan.Run()

print(sens_scan.results)









