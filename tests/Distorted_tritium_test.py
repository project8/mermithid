'''
This scripts aims at testing Tritium specific processors.
Author: C. Claessens
Date: Jul 31 2019
'''

import unittest

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

class TritiumTests(unittest.TestCase):

    def test_KuriePlot(self):
        from mermithid.processors.TritiumSpectrum import TritiumSpectrumLikelihoodSampler, KuriePlotFitter
        from morpho.processors.plots import Histogram
        from mermithid.misc.Constants import seconds_per_year, tritium_endpoint
        import importlib.machinery
        modulename = importlib.machinery.SourceFileLoader('modulename','/host-mermithid/mermithid/processors/TritiumSpectrum/TritiumSpectrumLikelihoodSampler.py').load_module()
        from modulename import TritiumSpectrumLikelihoodSampler

        specGen_config = {
            "volume": 7e-6*1e-2, # [m3]
            "density": 3e17, # [1/m3]
            "duration": 1.*seconds_per_year()/12.*1, # [s]
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
            "interestParams": ["F"],
            "fixedParams": {"m_nu": 0},
            "options": {"snr_efficiency": True, "channel_efficiency":False, "smearing": True},
            "snr_efficiency_coefficients": [-265.03357206889626, 6.693200670990694e-07, -5.795611253664308e-16, 1.5928835520798478e-25, 2.892234977030861e-35, -1.566210147698845e-44], #[-451719.97479592788, 5.2434404146607557e-05, -2.0285859980859651e-15, 2.6157820559434323e-26],
            "channel_central_frequency": 1400e6,
            "mixing_frequency": 24.5e9
        }
        histo_plot = {
            "variables": ["F"],
            "n_bins_x": 100,
            "title": "spectrum",
            "range": [1300e6, 1550e6]
        }


        specGen = TritiumSpectrumLikelihoodSampler("specGen")
        histo = Histogram("histo")

        specGen.Configure(specGen_config)
        histo.Configure(histo_plot)

        specGen.Run()
        result = specGen.data
        print('Number of events: {}'.format(len(result["F"])))
        #result_E = {"E": specGen.Energy(result["F"])}
        #result_E["is_sample"] = result["is_sample"]
        #print(result_E["E"][0:100])

        result_mixed = result
        result_mixed["F"] = [f-24.5e9 for f in result_mixed["F"]]
        histo.data = result_mixed
        histo.Run()



if __name__ == '__main__':
    unittest.main()
