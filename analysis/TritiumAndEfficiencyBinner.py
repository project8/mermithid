'''
This script takes efficiency output from the detection efficiency analysis, as
well as tritium data, and outputs binned or unbinned tritium data with
efficiencies and efficiency errors.
Author: C. Claessens, E. Novitski
Date: 3/9/20
'''

import numpy as np
import unittest
import matplotlib.pyplot as plt

from morpho.utilities import morphologging
logger = morphologging.getLogger(__name__)

class TritiumBinningTests(unittest.TestCase):

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
            "background": 0,#1e-6, # [counts/eV/s]
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
            "energy_or_frequency": 'frequency',
            "variables": "F",
#            "title": "corrected_spectrum",
#            "efficiency": "-265.03357206889626 + 6.693200670990694e-07*(x-24.5e9) + -5.795611253664308e-16*(x-24.5e9)^2 + 1.5928835520798478e-25*(x-24.5e9)^3 + 2.892234977030861e-35*(x-24.5e9)^4 + -1.566210147698845e-44*(x-24.5e9)^5",
            'bins': np.linspace(24.5e9+1300e6, 24.5e9+1550e6, 25),
            'fss_bins': True, # If fss_bins is True, bins is ignored and overwritten
            'efficiency_filepath': 'combined_energy_corrected_eff_at_quad_trap_frequencies.json'
        }


        specGen = DistortedTritiumSpectrumLikelihoodSampler("specGen")
        tritiumAndEfficiencyBinner = TritiumAndEfficiencyBinner("tritiumAndEfficiencyBinner")

        specGen.Configure(specGen_config)
        tritiumAndEfficiencyBinner.Configure(tritiumAndEfficiencyBinner_config)


        specGen.Run()
        data = specGen.data

        tritiumAndEfficiencyBinner.data = data
        tritiumAndEfficiencyBinner.Run()
        results = tritiumAndEfficiencyBinner.results

        plt.figure()
        plt.subplot(1,2,1)
        plt.errorbar(results['F'], results['N'], yerr = np.sqrt(results['N']), drawstyle = 'steps-mid')
        plt.subplot(1,2,2)
        plt.errorbar(results['F'], results['bin_efficiencies'], yerr = results['bin_efficiency_errors'], drawstyle = 'steps-mid')
        plt.savefig('TritiumAndEfficiencyBinnerOutputPlot.png')

        plt.figure()
        plt.errorbar(data['F'], results['event_efficiencies'], yerr = results['event_efficiency_errors'], fmt = 'none')
        plt.savefig('TritiumAndEfficiencyBinnerOutputPlotByEvent.png')

if __name__ == '__main__':
    unittest.main()
