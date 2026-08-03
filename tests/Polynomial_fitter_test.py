"""
Script to test fit processors
Author: C. Claessens
Date: January 24, 2021

"""

import unittest

from morpho.utilities import morphologging, parser
logger = morphologging.getLogger(__name__)


class FittersTest(unittest.TestCase):

    def test_BinnedDataFitter(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from mermithid.processors.Fitters import LeastSquaresPolynomialFitter

        logger.info('Polynomial fit test')

        # generate data
        a_mean = 1
        a_error = 0.1
        b_mean = 1
        b_error = 0.1

        random_x_values = np.random.uniform(0, 10, 10)
        random_y_values = np.random.normal(a_mean, a_error, 10) + random_x_values * (np.random.normal(b_mean, b_error, 10))
        y_uncertainty = 0.1*random_y_values
        random_data = {'K': np.random.randn(10000)*0.5+1}


        # config dict
        config_dict = {
            'order': 1,
            'x_key': 'x',
            'y_key': 'y'
            }


        # setup processor

        # before running give data to processor

        # run processor

        # collect results


        # plot
        plt.figure()
        plt.errorbar(random_x_values, random_y_values, y_uncertainty, fmt='o', label='data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best')
        plt.tight_layout
        plt.savefig('polynomial_fitter_test.png')




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