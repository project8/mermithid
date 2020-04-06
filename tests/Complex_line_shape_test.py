'''
This scripts aims at testing Tritium specific processors.
Author: M. Guigue, C. Claessens, A. Ziegler, E. Novitski
Date: 3/4/20
'''

import numpy as np
import unittest
import matplotlib.pyplot as plt

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

class ComplexLineShapeTests(unittest.TestCase):

    def test_complex_lineshape(self):
        from mermithid.processors.IO import IOCicadaProcessor
        from mermithid.processors.misc.ComplexLineShape import ComplexLineShape
        from mermithid.misc.Constants import seconds_per_year, tritium_endpoint

        reader_config = {
            "action": "read",
            "filename": "/host/ShallowTrap8603-8669.root",
            "object_type": "TMultiTrackEventData",
            "object_name": "multiTrackEvents:Event",
            "use_katydid": False,
            "variables": ['StartTimeInAcq','StartFrequency']
        }
        complexLineShape_config = {
            "energy_or_frequency": 'frequency',
            "variables": "F",
            "title": "corrected_spectrum",
            "efficiency": "-265.03357206889626 + 6.693200670990694e-07*(x-24.5e9) + -5.795611253664308e-16*(x-24.5e9)^2 + 1.5928835520798478e-25*(x-24.5e9)^3 + 2.892234977030861e-35*(x-24.5e9)^4 + -1.566210147698845e-44*(x-24.5e9)^5",
            'bins': np.linspace(24.5e9+1300e6, 24.5e9+1550e6, 15),
        }

        b = IOCicadaProcessor("reader")
        complexLineShape = ComplexLineShape("complexLineShape")

        b.Configure(reader_config)
        complexLineShape.Configure(complexLineShape_config)

        b.Run()
        data = b.data
        logger.info("Data extracted = {}".format(data.keys()))
        for key in data.keys():
            logger.info("{} -> size = {}".format(key,len(data[key])))

        complexLineShape.data = data

        complexLineShape.Run()

        results = complexLineShape.results
        print(results)

if __name__ == '__main__':
    unittest.main()
