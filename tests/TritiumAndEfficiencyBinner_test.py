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

from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

class TritiumBinningTests(unittest.TestCase):

    def test_Corrected_spectrum(self):
        from mermithid.processors.TritiumSpectrum import DistortedTritiumSpectrumLikelihoodSampler
        from mermithid.processors.misc.TritiumAndEfficiencyBinner import TritiumAndEfficiencyBinner
        from mermithid.misc.Constants import seconds_per_year, tritium_endpoint


        specGen_config = {
            "volume": 7e-6*1e-2, # [m3]
            "density": 3e17, # [1/m3]
            "duration": 1.*seconds_per_year()/12., # [s]
            "neutrino_mass" :0, # [eV]
            "energy_window": [tritium_endpoint()-1e3,tritium_endpoint()+1e3], # [KEmin,KEmax]
            "frequency_window": [-100e6, +100e6], #[Fmin, Fmax]
            "energy_or_frequency": "frequency",
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
            "energy_or_frequency": 'frequency', #Currently only set up to use frequency
            "variables": "F",
            'bins': np.arange(24.5e9+1300e6, 24.5e9+1490e6, 5e6),
            'fss_bins': False, # If fss_bins is True, bins is ignored and overwritten
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

        # run again with fss bins
        tritiumAndEfficiencyBinner_config['fss_bins'] = True
        tritiumAndEfficiencyBinner.Configure(tritiumAndEfficiencyBinner_config)
        tritiumAndEfficiencyBinner.Run()
        results2 = tritiumAndEfficiencyBinner.results


        plt.figure()

        eff1 = np.array(results['bin_efficiencies'])
        eff2 = np.array(results2['bin_efficiencies'])
        plt.plot(np.array(results['F'])-24.5e9, eff1/np.mean(eff1[eff1>0]), label='Piecewise integrated', marker='.')
        plt.plot(np.array(results2['F'])-24.5e9, eff2/np.mean(eff2[eff2>0]), label='FSS bin center', marker='.')
        plt.xlabel('Frequency - 24.5 GHz [Hz]')
        plt.ylabel('Efficiency')
        plt.legend()
        plt.tight_layout()
        plt.savefig('TritiumAndEfficiencyBinnerOutputIntegrationComparison.png')

        plt.figure()
        plt.subplot(1,2,1)
        plt.errorbar(results['F'], results['N'], yerr = np.sqrt(results['N']), drawstyle = 'steps-mid')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Counts')
        plt.subplot(1,2,2)
        plt.errorbar(results['F'], results['bin_efficiencies'], yerr = results['bin_efficiency_errors'], drawstyle = 'steps-mid')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Bin efficiency')
        plt.tight_layout()
        plt.savefig('TritiumAndEfficiencyBinnerOutputPlot.png')

        plt.figure()
        plt.errorbar(data['F'], results['event_efficiencies'], yerr = results['event_efficiency_errors'], fmt = 'none')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Event efficiency')
        plt.tight_layout()
        plt.savefig('TritiumAndEfficiencyBinnerOutputPlotByEvent.png')

if __name__ == '__main__':

    args = parser.parse_args(False)


    logger = morphologging.getLogger('morpho',
                                     level=args.verbosity,
                                     stderr_lb=args.stderr_verbosity,
                                     propagate=False)
    logger = morphologging.getLogger(__name__,
                                     level=args.verbosity,
                                     stderr_lb=args.stderr_verbosity,
                                     propagate=False)


    unittest.main()
