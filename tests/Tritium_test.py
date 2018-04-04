'''
This scripts aims at testing Tritium specific processors.
Author: M. Guigue
Date: Apr 1 2018
'''

import unittest

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

class TritiumTests(unittest.TestCase):
    
    def test_KuriePlot(self):
        from mermithid.processors.TritiumSpectrum import TritiumSpectrumGenerator
        from mermithid.misc.Constants import seconds_per_year, tritium_endpoint

        specGen_config = {
            "volume": 1, # [m3]
            "density": 1e18, # [1/m3]
            "experiment duration": 1*seconds_per_year(), # [s]
            "neutrino mass" :0, # [eV]
            "energy window": [tritium_endpoint()-1e3,tritium_endpoint()+1e3], # [KEmin,KEmax]
            "background": 1e-6, # [counts/eV/s]
            "energy resolution": 0# [eV]
        }
        specGen = TritiumSpectrumGenerator.TritiumSpectrumGenerator("specGen")
        specGen.Configure(specGen_config)
        specGen.Run()

if __name__ == '__main__':
    unittest.main()