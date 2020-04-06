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
            "variables": "F",
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
