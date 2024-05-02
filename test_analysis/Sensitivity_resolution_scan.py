"""
Script to make Sensitivty plots for cavity experiments
Author: C. Claessens, T. Weiss
Date: October 6, 2023
"""


from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

from mermithid.processors.Sensitivity import SensitivityParameterScanProcessor

import numpy as np
import matplotlib.pyplot as plt


# import all the scanned parameter units
from numericalunits import eV, K, mK, T # whatever you need
deg = np.pi/180


NScenarios = 15

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
    "goals": {"Phase IV target (0.04 eV)": 0.04},
    "scan_parameter_name": "MagneticField.sigmae_r",
    "scan_parameter_range": [0.01, 20],
    "scan_parameter_steps": NScenarios,
    "scan_parameter_unit": eV,
    "save_plots": False,
    "plot_sensitivity_scan_on_log_scale": True,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False,
    "plot_on_total_sigma_axis": True, 
    }
sens_scan = SensitivityParameterScanProcessor("sensitivity_curve_processor")
sens_scan.Configure(sens_config_dict)
sens_scan.Run()

#print(sens_scan.results)

sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_PhaseIV_single_cavity.cfg",
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
    "goals": {"Phase IV target (0.04 eV)": 0.04},
    "scan_parameter_name": "MagneticField.sigmae_r",
    "scan_parameter_range": [0.01, 20],
    "scan_parameter_steps": NScenarios,
    "scan_parameter_unit": eV,
    "save_plots": False,
    "plot_sensitivity_scan_on_log_scale": True,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False,
    "plot_on_total_sigma_axis": True, 
    }
sens_scan_1 = SensitivityParameterScanProcessor("sensitivity_curve_processor")
sens_scan_1.Configure(sens_config_dict)
sens_scan_1.Run()

#print(sens_scan.results)

sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_LFA_Experiment_1GHz.cfg",
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
    "goals": {"Phase IV target (0.04 eV)": 0.04},
    "scan_parameter_name": "MagneticField.sigmae_r",
    "scan_parameter_range": [0.01, 20],
    "scan_parameter_steps": NScenarios,
    "scan_parameter_unit": eV,
    "save_plots": False,
    "plot_sensitivity_scan_on_log_scale": True,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False,
    "plot_on_total_sigma_axis": True, 
    }
sens_scan_lfa = SensitivityParameterScanProcessor("sensitivity_curve_processor")
sens_scan_lfa.Configure(sens_config_dict)
sens_scan_lfa.Run()


sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_find_factor_22_Experiment_delta_0.05.cfg",
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
    "goals": {"Phase IV target (0.04 eV)": 0.04},
    "scan_parameter_name": "MagneticField.sigmae_r",
    "scan_parameter_range": [0.01, 20],
    "scan_parameter_steps": NScenarios,
    "scan_parameter_unit": eV,
    "save_plots": False,
    "plot_sensitivity_scan_on_log_scale": True,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False,
    "plot_on_total_sigma_axis": True, 
    }
sens_scan_5percent = SensitivityParameterScanProcessor("sensitivity_curve_processor")
sens_scan_5percent.Configure(sens_config_dict)
sens_scan_5percent.Run()

#print(sens_scan.results)

sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_atomic_PhaseIV_single_cavity_delta_0.05.cfg",
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
    "goals": {"Phase IV target (0.04 eV)": 0.04},
    "scan_parameter_name": "MagneticField.sigmae_r",
    "scan_parameter_range": [0.01, 20],
    "scan_parameter_steps": NScenarios,
    "scan_parameter_unit": eV,
    "save_plots": False,
    "plot_sensitivity_scan_on_log_scale": True,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False,
    "plot_on_total_sigma_axis": True, 
    }
sens_scan_1_5percent = SensitivityParameterScanProcessor("sensitivity_curve_processor")
sens_scan_1_5percent.Configure(sens_config_dict)
sens_scan_1_5percent.Run()

#print(sens_scan.results)

sens_config_dict = {
    # required
    "config_file_path": "/termite/sensitivity_config_files/Config_LFA_Experiment_1GHz_delta_0.05.cfg",
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
    "goals": {"Phase IV target (0.04 eV)": 0.04},
    "scan_parameter_name": "MagneticField.sigmae_r",
    "scan_parameter_range": [0.01, 20],
    "scan_parameter_steps": NScenarios,
    "scan_parameter_unit": eV,
    "save_plots": False,
    "plot_sensitivity_scan_on_log_scale": True,
    "label_x_position": 4e14, #4e14, #0.02, #1e14,
    "goals_x_position": 1.2e12, #0.0002
    "plot_key_parameters": False,
    "plot_on_total_sigma_axis": True, 
    }
sens_scan_lfa_5percent = SensitivityParameterScanProcessor("sensitivity_curve_processor")
sens_scan_lfa_5percent.Configure(sens_config_dict)
sens_scan_lfa_5percent.Run()

plt.figure()
plt.plot(sens_scan.results["total_energy_resolutions_eV"]*2.355, sens_scan.results["optimum_limits_eV"],
         label="Phase IV             1%", color="black", linestyle="-")
plt.plot(sens_scan_1.results["total_energy_resolutions_eV"]*2.355, sens_scan_1.results["optimum_limits_eV"], 
         label="Phase III final      1%", color="orangered", linestyle="-")
plt.plot(sens_scan_lfa.results["total_energy_resolutions_eV"]*2.355, sens_scan_lfa.results["optimum_limits_eV"], 
         label="Phase III mid-scale  1%", color="dodgerblue", linestyle="-")

plt.plot(sens_scan_5percent.results["total_energy_resolutions_eV"]*2.355, sens_scan_5percent.results["optimum_limits_eV"],
         label="Phase IV             5%", color="black", linestyle="--")
plt.plot(sens_scan_1_5percent.results["total_energy_resolutions_eV"]*2.355, sens_scan_1_5percent.results["optimum_limits_eV"], 
         label="Phase III final      5%", color="orangered", linestyle="--")
plt.plot(sens_scan_lfa_5percent.results["total_energy_resolutions_eV"]*2.355, sens_scan_lfa_5percent.results["optimum_limits_eV"], 
         label="Phase III mid-scale  5%", color="dodgerblue", linestyle="--")

plt.legend(frameon=False, prop={'family': 'DejaVu Sans Mono'})

plt.axhline(0.04, color="black", linestyle=":", label="Phase IV target (0.04 eV)", alpha=0.75, zorder=0)
plt.axhline(0.1, color="orangered", linestyle=":", label="Phase IV single cavity target (0.1 eV)", alpha=0.75, zorder=1)
plt.axhline(0.4, color="dodgerblue", linestyle=":", label="LFA target (0.4 eV)", alpha=0.75, zorder=2)

#plt.text(0.9, 0.04*0.75, "Phase IV target (0.04 eV)", color="dimgrey", horizontalalignment='right')
#plt.text(1.3, 0.1*0.57, "Phase IV single cavity \ntarget (0.1 eV)", color="orangered", horizontalalignment='right')
#plt.text(2, 0.4*0.75, "LFA target (0.4 eV)", color="steelblue", horizontalalignment='right')

plt.xlabel("Energy resolution FWHM (eV)")
plt.ylabel(r"90% CL m$_\beta$ (eV)")

plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
#plt.savefig("sensitivity_comparison_in_resolution_scan.pdf")
plt.savefig("sensitivity_comparison_in_resolution_scan_uncertainty_0.01_and_0.05.pdf")








