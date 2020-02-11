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
        from morpho.processors.plots import Histogram
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
        effCorr_config = {
            "variables": "F",
            "range": [24.5e9+1300e6, 24.5e9+1550e6],
            "n_bins_x": 100,
            "title": "corrected_spectrum",
            "efficiency": "-265.03357206889626 + 6.693200670990694e-07*(x-24.5e9) + -5.795611253664308e-16*(x-24.5e9)^2 + 1.5928835520798478e-25*(x-24.5e9)^3 + 2.892234977030861e-35*(x-24.5e9)^4 + -1.566210147698845e-44*(x-24.5e9)^5",
            "mode": "binned",
            "histogram_or_dictionary": "dictionary"

        }
        histo_plot = {
            "variables": "F",
            "n_bins_x": 100,
            "title": "spectrum",
            "range": [24.5e9+1300e6, 24.5e9+1550e6]
        }


        specGen = DistortedTritiumSpectrumLikelihoodSampler("specGen")
        effCorr = TritiumAndEfficiencyBinner("effCorr")
        histo = Histogram("histo")

        specGen.Configure(specGen_config)
        effCorr.Configure(effCorr_config)
        histo.Configure(histo_plot)


        #specGen.definePdf()
        specGen.Run()
        results = specGen.data

        histo.data = results
        histo.Run()

        """

        2/11/20: the below note is out of date, from when EfficiencyCorrector
        rather than TritiumAndEfficiencyBinner was used. Being replumbed now.

        Not proud of how I create binned data here. But it works =). Currently
        the EfficiencyCorrector needs to know the bin centers and bin occupancy
        to work. EfficiencyCorrector is based on Histogram and creates a
        histogram along with returning the corrected data.
        """

        results.update({'bin_centers': [], 'N': []})

        for i in range(histo_plot['n_bins_x']):
            results['bin_centers'].append(histo.histo.histo.GetBinCenter(i+1))
            results['N'].append(histo.histo.histo.GetBinContent(i+1))

        #print(results.keys())
        effCorr.data = results
        effCorr.Run()
        #print(effCorr.corrected_data)


if __name__ == '__main__':
    unittest.main()
