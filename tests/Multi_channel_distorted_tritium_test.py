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
            "duration": 1.*seconds_per_year()/12., # [s]
            "neutrino_mass" :0, # [eV]
            "energy_window": [tritium_endpoint()-1e3,tritium_endpoint()+1e3], # [KEmin,KEmax]
            "frequency_window": [-44e6, +40e6], #[Fmin, Fmax]
            "energy_or_frequency": "frequency",
            # "energy_window": [0.,tritium_endpoint()+1e3], # [KEmin,KEmax]
            "background": 1e-6, # [counts/eV/s]
            "energy_resolution": 5,# [eV]
            "frequency_resolution": 2e6,# [Hz]
            "mode": "generate",
            "varName": "F",
            "iter": 150,
            "interestParams": ["F"],
            "fixedParams": {"m_nu": 0},
            "options": {"snr_efficiency": True, "channel_efficiency":True, "smearing": False},
            "snr_efficiency_coefficients": [-451719.97479592788, 5.2434404146607557e-05, -2.0285859980859651e-15, 2.6157820559434323e-26],
            "channel_efficiency_coefficients": [24587.645303008387, 7645.8567999493698, 24507.145055859062, -11581.288750763715, 0.98587787287591955],
            "channel_central_frequency": 1378.125e6,
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

        specGen_a = TritiumSpectrumLikelihoodSampler("specGen_a")
        specGen_b = TritiumSpectrumLikelihoodSampler("specGen_b")
        specGen_c = TritiumSpectrumLikelihoodSampler("specGen_c")
        histo = Histogram("histo")
        kurieHisto = KuriePlotFitter("kurieHisto")

        specGen_a.Configure(specGen_config)

        specGen_config_b = specGen_config
        specGen_config_b["channel_central_frequency"] = 1437.5e6
        specGen_config_b["iter"] = 1400
        specGen_config_b["varName"] = "F"
        specGen_config_b["interestParams"] = ["F"]
        specGen_b.Configure(specGen_config_b)

        specGen_config_c = specGen_config
        specGen_config_c["channel_central_frequency"] = 1509.375e6
        specGen_config_c["iter"] = 2320
        specGen_c.Configure(specGen_config_c)
        histo.Configure(histo_plot)
        kurieHisto.Configure(kurie_plot)


        #specGen.definePdf()
        specGen_a.Run()
        specGen_b.Run()
        specGen_c.Run()
        result_a = specGen_a.data
        result_b = specGen_b.data
        result_c = specGen_c.data
        #print(result_a["Fa"])
        #print(result_b["Fb"])
        histo.data = {"F": result_c["F"], "Fb": result_b["F"], "Fa": result_a["F"]}
        kurieHisto.data = result_a
        histo.Run()
        kurieHisto.Run()

if __name__ == '__main__':
    unittest.main()
