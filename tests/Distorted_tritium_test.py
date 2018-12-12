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
        from mermithid.processors.TritiumSpectrum import TritiumSpectrumLikelihoodSampler, KuriePlotFitter
        from morpho.processors.plots import Histogram
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
            "frequency_window": [1320e6+24.5e9, 1550e6+24.5e9], #[Fmin, Fmax]
            "energy_or_frequency": "frequency",
            # "energy_window": [0.,tritium_endpoint()+1e3], # [KEmin,KEmax]
            "background": 1e-6, # [counts/eV/s]
            "energy_resolution": 5,# [eV]
            "mode": "generate",
            "varName": "F",
            "iter": 10000,
            "interestParams": ["F"],
            "fixedParams": {"m_nu": 0},
            "options": {"snr_efficiency": True, "channel_efficiency":True},
            "snr_efficiency_coefficients": [-451719.97479592788, 5.2434404146607557e-05, -2.0285859980859651e-15, 2.6157820559434323e-26],
            "channel_efficiency_coefficients": [24587.645303008387, 7645.8567999493698, 24507.145055859062, -11581.288750763715, 0.98587787287591955],
            "channel_central_frequency": 1378.125e6
        }
        histo_plot = {
            "variables": "F",
            "n_bins_x": 100,
            "title": "spectrum"
        }
        kurie_plot = {
            "variables": "F",
            "n_bins_x": 1000,
            "title": "kurie_plot"
        }

        specGen = TritiumSpectrumLikelihoodSampler("specGen")
        histo = Histogram("histo")
        kurieHisto = KuriePlotFitter("kurieHisto")

        specGen.Configure(specGen_config)
        histo.Configure(histo_plot)
        kurieHisto.Configure(kurie_plot)


        #specGen.definePdf()
        specGen.Run()
        result = specGen.data
        histo.data = result
        kurieHisto.data = result
        histo.Run()
        kurieHisto.Run()

if __name__ == '__main__':
    unittest.main()
