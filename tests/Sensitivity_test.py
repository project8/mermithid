"""
Script to test the Sensitivty processors
Author: C. Claessens
Date: December 12, 2021
"""

import unittest

from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

import numpy as np

class SensitivityTest(unittest.TestCase):

    def test_SensitivityCurveProcessor(self):
        from mermithid.processors.Sensitivity import CavitySensitivityCurveProcessor


        sens_config_dict = {
            # required
            "config_file_path": "/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment.cfg",
            #"config_file_path": "/host_repos/sensitivity_branches/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment.cfg",
            "plot_path": "./sensitivity_vs_efficiency_curve.pdf",
            # optional
            "track_length_axis": False,
            "molecular_axis": False,
            "atomic_axis": False,
            "density_axis": False,
            "cavity": True,
            "y_limits": [2e-2, 4],
            "density_range": [1e12,1e18],
            "efficiency_range": [0.0001, 1],
            #"density_range": [1e8, 1e12],
            "main_curve_upper_label": r"Molecular"+"\n"+"2 years"+"\n"+r"$\sigma^\bar{B}_\mathrm{reco} = 0.07\,\mathrm{eV}$",
            "main_curve_lower_label": r"$\sigma^\bar{B}_\mathrm{reco} = 1\,\mathrm{eV}$",
            "comparison_curve_label": r"Atomic"+"\n"+r"10 $\times$ 3 years"+"\n"+r"$\sigma^\bar{B}_\mathrm{reco} = 0.07\,\mathrm{eV}$",
            "goals": {"Phase III (0.1 eV)": 0.1, "Phase IV (0.04 eV)": 0.04},
            "comparison_curve": True,
            "comparison_config_file_path": "/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg",
            #"config_file_path": "/host_repos/sensitivity_branches/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg",
            #"B_inhomogeneity": np.linspace(0.1, 2.1, 10)*1e-6,
            #"B_inhom_uncertainty": 0.01,
            "sigmaE_theta_r": np.linspace(0.07, 0.15, 10), #in eV, energy broadening from theta and r reconstruction
            "lower_label_y_position": 0.17,
            "upper_label_y_position": 0.7,
            "label_x_position": 0.015, #1.5e15, #0.02, #1e14,
            "goals_x_position": 0.0002 #2e12 #0.0002
            }
        #sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
        #sens_curve.Configure(sens_config_dict)
        #sens_curve.Run()
        
        sens_config_dict = {
            # required
            "config_file_path": "/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment.cfg",
            #"config_file_path": "/host_repos/sensitivity_branches/termite/sensitivity_config_files/Config_PhaseIII_325MHz_Experiment.cfg",
            "plot_path": "./sensitivity_vs_density_curve.pdf",
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
            "main_curve_upper_label": r"Molecular"+"\n"+"2 years"+"\n"+r"$\sigma^\bar{B}_\mathrm{reco} = 0.07\,\mathrm{eV}$",
            "main_curve_lower_label": r"$\sigma^\bar{B}_\mathrm{reco} = 1\,\mathrm{eV}$",
            "comparison_curve_label": r"Atomic"+"\n"+r"10 $\times$ 3 years"+"\n"+r"$\sigma^\bar{B}_\mathrm{reco} = 0.07\,\mathrm{eV}$",
            "goals": {"Phase III (0.2 eV)": 0.2, "Phase IV (0.04 eV)": 0.04},
            "comparison_curve": True,
            "comparison_config_file_path": "/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg",
            #"comparison_config_file_path": "/host_repos/sensitivity_branches/termite/sensitivity_config_files/Config_atomic_325MHz_Experiment.cfg",
            #"B_inhomogeneity": np.linspace(0.1, 2.1, 10)*1e-6,
            #"B_inhom_uncertainty": 0.01,
            "sigmae_theta_r": np.linspace(0.07, 1., 10), #in eV, energy broadening from theta and r reconstruction
            "lower_label_y_position": 0.17,
            "upper_label_y_position": 0.7,
            "label_x_position": 1.5e15, #0.02, #1e14,
            "goals_x_position": 2e12, #0.0002
            "plot_key_parameters": True
            }
        sens_curve = CavitySensitivityCurveProcessor("sensitivity_curve_processor")
        sens_curve.Configure(sens_config_dict)
        sens_curve.Run()

    def test_SensitivityProcessor(self):
        from mermithid.processors.Sensitivity import AnalyticSensitivityEstimation


        sens_config_dict = {
            # required
            "config_file_path": "/termite/sensitivity_config_files/Config_PhaseIII_1GHz_Experiment.cfg"
            #"config_file_path": "/host_repos/sensitivity_branches/termite/sensitivity_config_files/Config_PhaseIII_1GHz_Experiment.cfg"
            }
        # sens = AnalyticSensitivityEstimation("sensitivity_processor")
        # sens.Configure(sens_config_dict)
        # sens.Run()

        # sens.print_statistics()
        # sens.print_systematics()

        # results = sens.results
        # logger.info(results)

    def test_ConstantSensitivityCurvesProcessor(self):
        from mermithid.processors.Sensitivity import ConstantSensitivityParameterPlots


        sens_config_dict = {
            # required
            "config_file_path": "/termite/sensitivity_config_files/Config_PhaseIII_1GHz_Experiment.cfg",
            #"config_file_path": "//host_repos/sensitivity_branches/termite/sensitivity_config_files/Config_PhaseIII_1GHz_Experiment.cfg",
            "sensitivity_target": [0.4**2/np.sqrt(1.64)]#, 0.7**2/np.sqrt(1.64), 1**2/np.sqrt(1.64)]
            }
        #sens = ConstantSensitivityParameterPlots("sensitivity_processor")
        #sens.Configure(sens_config_dict)
        #sens.Run()

        #sens.print_statistics()
        #sens.print_systematics()





if __name__ == '__main__':

    args = parser.parse_args(False)


    logger = morphologging.getLogger('morpho',
                                     level=args.verbosity,
                                     stderr_lb=args.stderr_verbosity,
                                     propagate=False)
    logger = morphologging.getLogger(__name__,
                                     level=args.verbosity,
                                     stderr_lb=args.stderr_verbosity,
                                     propagate=False)

    unittest.main()