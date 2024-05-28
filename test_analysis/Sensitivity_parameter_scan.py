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
from numericalunits import e, me, c0, eps0, kB, hbar
from numericalunits import meV, eV, keV, MeV, cm, m, mm
from numericalunits import nT, uT, mT, T, mK, K,  C, F, g, W
from numericalunits import hour, year, day, ms, ns, s, Hz, kHz, MHz, GHz
deg = np.pi/180




# Configuration for Sensitivity vs. density plot
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
    #"scan_parameter_name": "MagneticField.sigmae_r",
    #"scan_parameter_range": [0.1, 5],
    #"scan_parameter_steps": 4,
    #"scan_parameter_unit": eV,
    "scan_parameter_name": "Experiment.l_over_d",
    "scan_parameter_range": [5, 15],
    "scan_parameter_steps": 5,
    "scan_parameter_unit": 1,
    "plot_sensitivity_scan_on_log_scale": False,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False
    }
#sens_scan = SensitivityParameterScanProcessor("sensitivity_curve_processor")
#sens_scan.Configure(sens_config_dict)
#sens_scan.Run()


# Configuration for Sensitivity vs. density plot
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
    #"scan_parameter_name": "FrequencyExtraction.minimum_angle_in_bandwidth",
    #"scan_parameter_range": [83, 89],
    #"scan_parameter_steps": 10,
    #"scan_parameter_unit": deg,
    "scan_parameter_name": "Experiment.l_over_d",
    "scan_parameter_range": [5, 15],
    "scan_parameter_steps": 5,
    "scan_parameter_unit": 1,
    "plot_sensitivity_scan_on_log_scale": False,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False
    }
#sens_scan = SensitivityParameterScanProcessor("sensitivity_curve_processor")
#sens_scan.Configure(sens_config_dict)
#sens_scan.Run()


# Configuration for Sensitivity vs. density plot
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
    #"scan_parameter_name": "Experiment.l_over_d",
    #"scan_parameter_range": [5, 8],
    #"scan_parameter_steps": 3,
    #"scan_parameter_unit": 1,
    "scan_parameter_name": "Experiment.n_cavities",
    "scan_parameter_range": [4, 16],
    "scan_parameter_steps": 7, # This is the one that currently has text output
    "scan_parameter_unit": 1,
    "plot_sensitivity_scan_on_log_scale": False,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False
    }
sens_scan = SensitivityParameterScanProcessor("sensitivity_curve_processor")
sens_scan.Configure(sens_config_dict)
sens_scan.Run()

print(sens_scan.results)









