'''
This scripts aims at testing Tritium specific processors.
Author: M. Guigue
Date: Apr 1 2018
'''

import unittest

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)


class FakeDataGenerationTest(unittest.TestCase):

    def test_data_generation(self):
        from mermithid.processors.TritiumSpectrum.FakeDataGenerator import FakeDataGenerator
        from mermithid.misc.Constants import seconds_per_year, tritium_endpoint

        specGen_config = {
            "variables": 'F'
        }

        specGen = FakeDataGenerator("specGen")

        specGen.Configure(specGen_config)

        #specGen.Run()
        #result = specGen.results

        #print(result.keys())


if __name__ == '__main__':
    unittest.main()
