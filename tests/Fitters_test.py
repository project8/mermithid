"""
Script to test fit processors
Author: C. Claessens
Date: August 16, 2020

"""

import unittest

from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np

class FittersTest(unittest.TestCase):


    def test_BinnedDataFitter(self):
        from mermithid.processors.Fitters import BinnedDataFitter

        logger.info('iMinuit fit test')
        config_dict = {
            'variables': 'K',
            'bins': np.linspace(-3, 3, 100),
            'parameter_names': ['A', 'mu', 'sigma'],
            'initial_values': [100, 0, 1],
            'limits': [[0, None], [None, None], [0, None]],
            'constrained_parameter_indices': [],
            'constrained_parameter_means': [0.5],
            'constrained_parameter_widths': [1]

            }

        random_data = {'K': np.random.randn(10000)*0.5+1}
        negll_fitter = BinnedDataFitter('iminuit_processor')
        negll_fitter.Configure(config_dict)
        negll_fitter.data = random_data
        negll_fitter.Run()

        results = negll_fitter.results

        for k in results.keys():
            logger.info('{}: {}'.format(k, results[k]))


        result_list = results['param_values']
        error_list = results['param_errors']
        x = negll_fitter.bin_centers
        hist = negll_fitter.hist
        hist_fit = negll_fitter.PDF(x, *result_list)
        residuals = (hist-hist_fit)/np.sqrt(hist_fit)

        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(10,10))
        plt.subplot(211)
        plt.errorbar(x, hist, np.sqrt(hist), drawstyle='steps-mid', label='Binned data')
        plt.plot(x, negll_fitter.PDF(x, *result_list), label='Fit')
        plt.xlabel(negll_fitter.namedata)
        plt.ylabel('Counts')
        plt.legend()

        plt.subplot(212)
        plt.scatter(x, residuals, label='Pearson residuals')
        plt.axhline(np.nanmean(residuals))
        plt.xlabel(negll_fitter.namedata)
        plt.ylabel('Residuals')
        plt.legend()


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