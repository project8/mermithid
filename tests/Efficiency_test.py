'''
This script tests the Divide method of the modified Histogram and
RootHistogram classes from Morpho. It requires that those modified classes
be available to import if they are not already in the version of morpho
being used.
Author: M. Guigue, A. Ziegler
Date: Aug 1 2019
'''

import unittest

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

class EfficiencyTest(unittest.TestCase):



    def test_EffCorr(self):
        from mermithid.processors.TritiumSpectrum import TritiumSpectrumGenerator
        from mermithid.processors.plots.Histogram import Histogram
        from mermithid.misc.Constants import seconds_per_year, tritium_endpoint
        #from Energy2FrequencyConversionProcessor import Energy2FrequencyConversionProcessor

        specGen_config = {
            "volume": 7e-6*1e-2, # [m3]
            "density": 3e17, # [1/m3]
            "duration": 1.*seconds_per_year()/12., # [s]
            "neutrino_mass" :200, # [eV]
            "energy_window": [tritium_endpoint()-1e3,tritium_endpoint()+1e3], # [KEmin,KEmax]
            # "energy_window": [0.,tritium_endpoint()+1e3], # [KEmin,KEmax]
            "background": 1e-6, # [counts/eV/s]
            "energy_resolution": 5# [eV]
        }

        histo_plot = {
            "variables": "KE",
            "n_bins_x": 100,
            "title": "spectrum",
            #"range": [tritium_endpoint()-1e3,tritium_endpoint()+1e3]
            "range": [26.9585,27.0605] # Range in GHz
        }

        corr_histo_plot = {
            "variables": "KE",
            "n_bins_x": 100,
            "title": "corrected_spectrum",
            #"range": [tritium_endpoint()-1e3,tritium_endpoint()+1e3],
            "range": [26.9585,27.0605],
            "efficiency": "0.05*cos(6.28*(x/18600))+1"# Equation that divides
                                                      # the spectrum.
        }

        energy2freq_config = {
            "B": 1
        }

        specGen = TritiumSpectrumGenerator("specGen")
        histo = Histogram("histo")
        corr_histo = Histogram("corr_histo")
        #energy2freq = Energy2FrequencyConversionProcessor("energy2freq")

        specGen.Configure(specGen_config)
        histo.Configure(histo_plot)
        corr_histo.Configure(corr_histo_plot)
        #energy2freq.Configure(energy2freq_config)

        specGen.Run()
        results = specGen.results

        #energy2freq.energies = results
        #energy2freq.Run()


        histo.data = results
        corr_histo.data = results
        histo.Run()
        corr_histo.Run()

if __name__ == '__main__':
    unittest.main()
