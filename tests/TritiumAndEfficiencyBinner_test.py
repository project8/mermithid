'''
This scripts aims at testing Tritium specific processors.
Author: M. Guigue, C. Claessens, A. Ziegler
Date: Aug 1 2018
'''

import unittest

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

class TritiumTests(unittest.TestCase):

    def test_Corrected_spectrum(self):
        from mermithid.processors.TritiumSpectrum import DistortedTritiumSpectrumLikelihoodSampler
        from mermithid.processors.misc.TritiumAndEfficiencyBinner import TritiumAndEfficiencyBinner
        from mermithid.misc.Constants import seconds_per_year, tritium_endpoint
        #import importlib.machinery
        #modulename = importlib.machinery.SourceFileLoader('modulename','/Users/ziegler/docker_share/builds/mermithid/mermithid/processors/TritiumSpectrum/DistortedTritiumSpectrumLikelihoodSampler.py').load_module()
        #from modulename import DistortedTritiumSpectrumLikelihoodSampler


        specGen_config = {
            "volume": 7e-6*1e-2, # [m3]
            "density": 3e17, # [1/m3]
            "duration": 1.*seconds_per_year()/12., # [s]
            "neutrino_mass" :0, # [eV]
            "energy_window": [tritium_endpoint()-1e3,tritium_endpoint()+1e3], # [KEmin,KEmax]
            "frequency_window": [-100e6, +100e6], #[Fmin, Fmax]
            "energy_or_frequency": "frequency",
            # "energy_window": [0.,tritium_endpoint()+1e3], # [KEmin,KEmax]
            "background": 1e-6, # [counts/eV/s]
            "energy_resolution": 5,# [eV]
            "frequency_resolution": 2e6,# [Hz]
            "mode": "generate",
            "varName": "F",
            "iter": 10000,
            "interestParams": "F",
            "fixedParams": {"m_nu": 0},
            "options": {"snr_efficiency": True, "channel_efficiency":False, "smearing": False},
            "snr_efficiency_coefficients": [-265.03357206889626, 6.693200670990694e-07, -5.795611253664308e-16, 1.5928835520798478e-25, 2.892234977030861e-35, -1.566210147698845e-44],
            "channel_central_frequency": 1400e6,
            "mixing_frequency": 24.5e9
        }
        tritiumAndEfficiencyBinner_config = {
            "variables": "F",
            "range": [24.5e9+1300e6, 24.5e9+1550e6],
            "n_bins_x": 100,
            "title": "corrected_spectrum",
            "efficiency": "-265.03357206889626 + 6.693200670990694e-07*(x-24.5e9) + -5.795611253664308e-16*(x-24.5e9)^2 + 1.5928835520798478e-25*(x-24.5e9)^3 + 2.892234977030861e-35*(x-24.5e9)^4 + -1.566210147698845e-44*(x-24.5e9)^5",
            "mode": "unbinned",
            "histogram_or_dictionary": "dictionary"

        }


        specGen = DistortedTritiumSpectrumLikelihoodSampler("specGen")
        tritiumAndEfficiencyBinner = TritiumAndEfficiencyBinner("tritiumAndEfficiencyBinner")

        specGen.Configure(specGen_config)
        tritiumAndEfficiencyBinner.Configure(tritiumAndEfficiencyBinner_config)


        #specGen.definePdf()
        specGen.Run()
        results = specGen.data

        tritiumAndEfficiencyBinner.data = results
        tritiumAndEfficiencyBinner.Run()
        #print(effCorr.corrected_data)


if __name__ == '__main__':
    unittest.main()
