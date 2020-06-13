"""
Script to test the miscalleneous processors
Author: J. Johnston
Date: April 24, 2018

Converts frequencies corresponding to kinetic energy
"""

import unittest

from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np

class MiscTest(unittest.TestCase):

    def test_FreqConversionTest(self):
        from mermithid.processors.misc import FrequencyEnergyConversionProcessor

        freq_data = [27.9925*10**9, 27.0094*10**9,
                     26.4195*10**9, 26.4169*10**9,
                     26.3460*10**9, 26.3457*10**9]
        logger.info("Will convert the following frequencies: %s"%freq_data)
        logger.debug("At 1 T, these correspond to kinetic energies (in keV) of " +
              "[0, 18.6, 30.424, 30.477, 31.934, 31.942]")

        freq_energy_dict = {
            "B": 1
        }

        freq_proc = FrequencyEnergyConversionProcessor("freq_energy_processor")
        freq_proc.Configure(freq_energy_dict)
        freq_proc.frequencies = freq_data
        freq_proc.Run()

        logger.info("Resulting energies: %s"%freq_proc.energies)


    def test_BinnedDataFitter(self):
        from mermithid.processors.misc import BinnedDataFitter

        logger.info('iMinuit fit test')
        config_dict = {
            'variables': 'K',
            'bins': np.linspace(-3, 3, 60),
            'parameter_names': ['A', 'mu', 'sigma'],
            'initial_values': [100, 0, 1],
            'limits': [[0, None], [None, None], [0, None]],
            'constrained_parameter_indices': [1],
            'constrained_parameter_means': [0],
            'constrained_parameter_widths': [0.1]

            }

        random_data = {'K': np.random.randn(1000)*0.5+1}
        negll_fitter = BinnedDataFitter('iminuit_processor')
        negll_fitter.Configure(config_dict)
        negll_fitter.data = random_data
        negll_fitter.Run()

        results = negll_fitter.results

        for k in results.keys():
            logger.info('{}: {}'.format(k, results[k]))

        x = negll_fitter.bin_centers
        hist = negll_fitter.hist
        result_list = results['param_values']
        error_list = results['param_errors']

        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(10,10))
        plt.subplot(211)
        plt.errorbar(x, hist, np.sqrt(hist), drawstyle='steps-mid', label='Binned data')
        plt.plot(x, negll_fitter.PDF(x, *result_list), label='fit')
        plt.xlabel(negll_fitter.namedata)
        plt.ylabel('Counts')

        plt.subplot(212)
        plt.scatter(x, (hist-negll_fitter.PDF(x, *result_list))/np.sqrt(hist), label='Pearson residuals')
        plt.xlabel(negll_fitter.namedata)
        plt.ylabel('Residuals')


        plt.savefig('iminit_fit.png')




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