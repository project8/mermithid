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

    def test_iminuit(self):
        logger.info('iMinuit test')
        from iminuit import Minuit

        def f(x, y, z):
            return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

        def g(params):
            return f(*params)

        m = Minuit(f)

        m.migrad()  # run optimiser
        print(m.values)  # {'x': 2,'y': 3,'z': 4}

        m.hesse()   # run covariance estimator
        print(m.errors)  # {'x': 1,'y': 1,'z': 1}

        # repeat using from_array_func
        m2 = Minuit(g, [0, 0, 0], name=['a1', 'b1', 'c1'])
        m2.errors = [0.1, 0.1, 0.1]
        m2.errordef = 1
        m2.print_level = 0
        m2.migrad()
        print(m2.values)

        logger.info('iMinuit test done')

    def test_BinnedDataFitter(self):
        from mermithid.processors.Fitters import BinnedDataFitter

        logger.info('iMinuit fit test')
        config_dict = {
            'variables': 'N',
            'bins': np.linspace(-3, 3, 100),
            'parameter_names': ['A', 'mu', 'sigma'],
            'initial_values': [100, 0, 1],
            'limits': [[0, None], [None, None], [0, None]],
            'constrained_parameter_indices': [],
            'constrained_parameter_means': [0.5],
            'constrained_parameter_widths': [1],
            'binned_data': True,
            'print_level': 0

            }

        random_data = {'K': np.random.randn(10000)*0.5+1}

        # histogramming could be done by processor
        binned_data, _ = np.histogram(random_data['K'], config_dict['bins'])
        binned_data_dict = {'N': binned_data}

        # setup processor
        negll_fitter = BinnedDataFitter('iminuit_processor')
        negll_fitter.Configure(config_dict)

        # define new model and overwrite processor's model
        def gaussian(x, A, mu, sigma):
            """
            This is the same function that is implemented as default model
            """
            f = A*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(((x-mu)/sigma)**2.)/2.)
            return f

        negll_fitter.model = gaussian

        # hand over data and run
        negll_fitter.data = binned_data_dict #random_data
        negll_fitter.Run()

        # collect fit results
        results = negll_fitter.results
        result_list = results['param_values']
        error_list = results['param_errors']

        for k in results.keys():
            logger.info('{}: {}'.format(k, results[k]))



        # get bins histogram and fit curve from processor for plotting
        x = negll_fitter.bin_centers
        hist = negll_fitter.hist
        hist_fit = negll_fitter.model(x, *result_list)

        # calculate normalized residuals
        residuals = (hist-hist_fit)/np.sqrt(hist_fit)

        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(10,10))
        plt.subplot(211)
        plt.errorbar(x, hist, np.sqrt(hist), drawstyle='steps-mid', label='Binned data')
        plt.plot(x, negll_fitter.model(x, *result_list), label='Fit')
        plt.xlabel(negll_fitter.namedata)
        plt.ylabel('Counts')
        plt.legend()

        plt.subplot(212)
        plt.scatter(x, residuals, label='Pearson residuals')
        plt.axhline(np.nanmean(residuals))
        plt.xlabel(negll_fitter.namedata)
        plt.ylabel('Residuals')
        plt.legend()


        plt.savefig('iminuit_fit.png')




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