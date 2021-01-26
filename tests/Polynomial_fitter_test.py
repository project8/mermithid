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
        from mermithid.processors.TritiumSpectrum import FakeDataGenerator

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
            'order': 2,
            'x_key': 'x',
            'y_key': 'y',
            'error_key': 'y_error'
            }


        # setup processor
        poly_fitter = LeastSquaresPolynomialFitter('poly_fitter')
        poly_fitter.Configure(config_dict)

        # before running give data to processor
        poly_fitter.data = {'x': random_x_values, 'y':random_y_values, 'y_error': y_uncertainty}

        # run processor
        poly_fitter.Run()

        # collect results
        results = poly_fitter.results
        logger.info('Fit results:')
        logger.info(results)


        # plot
        plt.figure()
        plt.errorbar(random_x_values, random_y_values, y_uncertainty, fmt='o', label='data')
        plt.plot(results['sorted_x'], results['best_fit'], label='Fit result')
        plt.title('Fitted coefficients: {}'.format(np.round(results['coefficients'],3)))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best')
        plt.tight_layout
        plt.savefig('polynomial_fitter_test.png')


        #######################################################################
        # repeat using FakeDataGenerator
        specGen_config = {
            "apply_efficiency": False,
            "efficiency_path": "./combined_energy_corrected_eff_at_quad_trap_frequencies.json",
            "simplified_lineshape_path": None,
            "path_to_detailed_scatter_spectra_dir": '/host',
            "detailed_or_simplified_lineshape": "detailed", #"simplified" or "detailed"
            "use_lineshape": False, # if False only gaussian smearing is applied
            "return_frequency": True,
            "scattering_sigma": 18.6, # only used if use_lineshape = True
            "survival_prob": 0.77, # only used if use_lineshape = True
            "scatter_proportion": 0.8, # only used if use_lineshape = True and lineshape = detailed
            "B_field": 0.9578186017836624,
            "S": 4500, # number of tritium events
            "n_steps": 10000, # stepsize for pseudo continuous data is: (Kmax_eff-Kmin_eff)/nsteps
            "A_b": 1e-10, # background rate 1/eV/s
            "poisson_stats": True,
            "molecular_final_states": False,
            "final_states_file": "../mermithid/misc/saenz_mfs.json"
        }

        specGen = FakeDataGenerator("specGen")

        specGen.Configure(specGen_config)

        specGen.Run()
        results = specGen.results

        # plot histograms of generated data
        Kgen = results['K']
        Fgen = results['F']

        plt.figure()
        n, b, p = plt.hist(Kgen, bins=20)
        plt.xlabel('Kgen')
        plt.ylabel('N')
        plt.savefig('tritium_energies_histogram.png')

        # re-configure poly_fitter
        # config dict
        config_dict = {
            'order': 2,
            'x_key': 'K',
            'y_key': 'N',
            'error_key': 'sqrt_N'
            }
        poly_fitter.Configure(config_dict)

        bin_centers = b[0:-1]+0.5*(b[1]-b[0])
        poly_fitter.data = {'K': bin_centers, 'N': n, 'sqrt_N': np.sqrt(n)}

        # run and add fit result to previous plot
        poly_fitter.Run()
        tritium_fit_results = poly_fitter.results

        plt.plot(tritium_fit_results['sorted_x'], tritium_fit_results['best_fit'], label='Fit result')
        plt.legend()
        plt.tight_layout()
        plt.savefig('tritium_polynomial_fit.png')




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