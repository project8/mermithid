'''
This script tests the EfficiencyCorrectorProcessor.
Author: M. Guigue, A. Ziegler
Date: Aug 1 2019
'''

import unittest

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

class GaussianDivideTest(unittest.TestCase):

    def test_EffCorr(self):
        from morpho.processors.sampling import GaussianSamplingProcessor
        from plots.Histogram import Histogram # New version of Histogram
        #from morpho.processors.plots import Histogram Old verion of Histogram
        from mermithid.misc.Constants import seconds_per_year, tritium_endpoint

        gauss_config = {
            "iter": 1000000, # Num
            "mean": 0,
            "width": 1
        }

        histo_plot = {
            "variables": "x",
            "n_bins_x": 100,
            "title": "gaussian",
            "range": [-4,4]
        }

        corr_histo_plot = {
            "variables": "x",
            "n_bins_x": 100,
            "title": "divided_gaussian",
            "range": [-4,4],
            "efficiency": "-pow(x,2)+1"
        }

        gauss = GaussianSamplingProcessor("gauss")
        histo = Histogram("histo")
        corr_histo = Histogram("corr_histo")

        gauss.Configure(gauss_config)
        histo.Configure(histo_plot)
        corr_histo.Configure(corr_histo_plot)

        gauss.Run()
        results = gauss.results


        histo.data = results
        corr_histo.data = results
        histo.Run()
        corr_histo.Run()

if __name__ == '__main__':
    unittest.main()
