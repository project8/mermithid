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
        from mermithid.processors.TritiumSpectrum import TritiumSpectrumLikelihoodSampler, KuriePlotFitter
        from mermithid.processors.plots import Histogram
        from mermithid.misc.Constants import seconds_per_year, tritium_endpoint
        import importlib.machinery
        modulename = importlib.machinery.SourceFileLoader('modulename','/host-mermithid/mermithid/processors/TritiumSpectrum/TritiumSpectrumLikelihoodSampler.py').load_module()
        from modulename import TritiumSpectrumLikelihoodSampler

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
            "interestParams": ["F"],
            "fixedParams": {"m_nu": 0},
            "options": {"snr_efficiency": True, "channel_efficiency":False, "smearing": False},
            "snr_efficiency_coefficients": [-451719.97479592788, 5.2434404146607557e-05, -2.0285859980859651e-15, 2.6157820559434323e-26],
            #"channel_efficiency_coefficients": [24587.645303008387, 7645.8567999493698, 24507.145055859062, -11581.288750763715, 0.98587787287591955],
            "channel_central_frequency": 1400e6,
            "mixing_frequency": 24.5e9
        }
        histo_plot = {
            "variables": ["F", "Fb", "Fa"],
            "n_bins_x": 100,
            "title": "spectrum",
            "range": [24.5e9+1330e6, 24.5e9+1560e6]
        }
        kurie_plot = {
            "variables": "F",
            "n_bins_x": 1000,
            "title": "kurie_plot"
        }

        specGen = TritiumSpectrumLikelihoodSampler("specGen")
        histo = Histogram("histo")

        specGen.Configure(specGen_config)
        histo.Configure(histo_plot)


        #specGen.definePdf()
        specGen.Run()
        result = specGen.data

        histo.data = result
        histo.Run()


if __name__ == '__main__':
    unittest.main()
