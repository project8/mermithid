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
            "config_file_path": "/host_termite/sensitivity_config_files/Config_PhaseIII_FSCD_molecular_V_eff_2cm3.cfg",
            "plot_path": "./sensitivity_curve.pdf",
            # optional
            "track_length_axis": True,
            "molecular_axis": True,
            "atomic_axis": False,
            "y_limits": [1e-2, 1e2],
            "main_curve_upper_label": r"molecular"+"\n"+r"$V_\mathrm{eff} = 2\, \mathrm{cm}^3$"+"\n"+r"$\sigma_B = 7\,\mathrm{ppm}$",
            "goals": {"Phase III (2 eV)": 2, "Phase IV (40 meV)": 0.04},
            "comparison_curve": False,
            "comparison_config_file_path": "/host_scripts/rreimann/data/Sensitivity/Config_PhaseIV_atomic_V_eff_5m3.cfg",
            "B": np.arange(1, 8)*1e-6

            }
        sens_curve = SensitivityCurveProcessor("sensitivity_curve_processor")
        sens_curve.Configure(sens_config_dict)
        sens_curve.Run()

    def test_SensitivityProcessor(self):
        from mermithid.processors.Sensitivity import AnalyticSensitivityEstimation


        sens_config_dict = {
            # required
            "config_file_path": "/host_termite/sensitivity_config_files/Config_PhaseIII_FSCD_molecular_V_eff_2cm3.cfg"
            }
        sens = AnalyticSensitivityEstimation("sensitivity_processor")
        sens.Configure(sens_config_dict)
        sens.Run()

        sens.print_statistics()
        sens.print_systematics()

        results = sens.results
        logger.info(results)



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
