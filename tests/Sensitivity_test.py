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
        from mermithid.processors.Sensitivity import SensitivityCurveProcessor


        sens_config_dict = {
            # required
            #"config_file_path": "/host_termite/sensitivity_config_files/Config_PhaseIII_1GHz_Experiment.cfg",
            "config_file_path": "/home/chrischtel/repos/another_termite/termite/sensitivity_config_files/Config_PhaseIII_1GHz_Experiment.cfg",
            "plot_path": "./sensitivity_curve.pdf",
            # optional
            "track_length_axis": True,
            "molecular_axis": False,
            "atomic_axis": True,
            "y_limits": [1e-1, 1e2],
            "density_range": [1e12,1e21],
            "main_curve_upper_label": r"Atomic CRES at 0.04 T"+"\n"+r"$V_\mathrm{eff} = 0.01\, \mathrm{m}^3$"+"\n"+r"$\sigma_B = 2\,\mathrm{ppm}$",
            "main_curve_lower_label": r"$\sigma_B = 0.15\,\mathrm{ppm}$",
            "goals": {"Phase III (0.4 eV)": 0.4},# "Phase IV (40 meV)": 0.04},
            "comparison_curve": False,
            "comparison_config_file_path": "/host_scripts/rreimann/data/Sensitivity/Config_PhaseIV_atomic_V_eff_5m3.cfg",
            "B_inhomogeneity": np.arange(0.15, 2.05, 0.15)*1e-6,
            "B_inhom_uncertainty": 0.05,
            "lower_label_y_position": 0.5,
            "upper_label_y_position": 5,
            "label_x_position": 1e14
            }
        sens_curve = SensitivityCurveProcessor("sensitivity_curve_processor")
        sens_curve.Configure(sens_config_dict)
        sens_curve.Run()

    def test_SensitivityProcessor(self):
        from mermithid.processors.Sensitivity import AnalyticSensitivityEstimation


        sens_config_dict = {
            # required
            #"config_file_path": "/host_termite/sensitivity_config_files/Config_PhaseIII_1GHz_Experiment.cfg"
            "config_file_path": "/home/chrischtel/repos/another_termite/termite/sensitivity_config_files/Config_PhaseIII_1GHz_Experiment.cfg"
            }
        sens = AnalyticSensitivityEstimation("sensitivity_processor")
        sens.Configure(sens_config_dict)
        sens.Run()

        sens.print_statistics()
        sens.print_systematics()

        results = sens.results
        logger.info(results)

"""    def test_ConstantSensitivityCurvesProcessor(self):
        from mermithid.processors.Sensitivity import ConstantSensitivityParameterPlots


        sens_config_dict = {
            # required
            #"config_file_path": "/host_termite/sensitivity_config_files/Config_PhaseIII_1GHz_Experiment.cfg"
            "config_file_path": "/home/chrischtel/repos/another_termite/termite/sensitivity_config_files/Config_PhaseIII_1GHz_Experiment.cfg"
            }
        sens = ConstantSensitivityParameterPlots("sensitivity_processor")
        sens.Configure(sens_config_dict)
        sens.Run()

        sens.print_statistics()
        sens.print_systematics()"""





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
