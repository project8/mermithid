'''
Author: C. Claessens
Date:6/12/2020
Description: Configure with arbitrary histogram and pdf.
Minimizes poisson loglikelihood using iMinuit.

'''

from __future__ import absolute_import

import numpy as np
import scipy
import sys
import json
from iminuit import Minuit
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor

logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

class BinnedDataFitter(BaseProcessor):
    '''
    Processor that
    Args:
        variables
        parameter_names
        initial_values
        limits
        fixed
        bins
        print_level
        constrained_parameter_indices
        constrained_parameter_means
        constrained_parameter_widths

    Inputs:
        data:
    Output:
        result: dictionary containing
    '''
    def InternalConfigure(self, params):
        '''
        Configure
        '''

        # Read other parameters
        self.namedata = reader.read_param(params, 'variables', "required")
        self.parameter_names = reader.read_param(params, 'parameter_names', ['A', 'mu', 'sigma'])
        self.initial_values = reader.read_param(params, 'initial_values', [1, 0, 1])
        self.limits = reader.read_param(params, 'limits', [[None, None], [None, None], [None, None]])
        self.fixes = reader.read_param(params, 'fixed', [False, False, False])
        self.bins = reader.read_param(params, 'bins', np.linspace(-2, 2, 100))
        self.print_level = reader.read_param(params, 'print_level', 1)

        self.constrained_parameters = reader.read_param(params, 'constrained_parameter_indices', [])
        self.constrained_means = reader.read_param(params, 'constrained_parameter_means', [])
        self.constrained_widths = reader.read_param(params, 'constrained_parameter_widths', [])

        # derived configurations
        self.bin_centers = self.bins[0:-1]+0.5*(self.bins[1]-self.bins[0])
        self.parameter_errors = [max([0.1, 0.1*p]) for p in self.initial_values]

        return True

    def InternalRun(self):
        logger.info('namedata: {}'.format(self.namedata))

        self.hist, _ = np.histogram(self.data[self.namedata], self.bins)

        result_array, error_array = self.fit()

        # save results
        self.results = {}
        self.results['param_values'] = result_array
        self.results['param_errors'] = error_array
        for i, k in enumerate(self.parameter_names):
            self.results[k] = {'value': result_array[i], 'error': error_array[i]}

        return True



    def PDF(self, x, A, mu, sigma):
        """
        Overwrite by whatever PDF
        """
        f = A*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(((x-mu)/sigma)**2.)/2.)
        return f



    def fit(self):
        # Now minimize neg log likelihood using iMinuit
        if self.print_level == 1:
            logger.info('This is the plan:')
            logger.info('Fitting data consisting of {} elements'.format(np.sum(self.hist)))
            logger.info('Fit parameters: {}'.format(self.parameter_names))
            logger.info('Initial values: {}'.format(self.initial_values))
            logger.info('Initial error: {}'.format(self.parameter_errors))
            logger.info('Fixed in fit: {}'.format(self.fixes))
            logger.info('Constrained parameters: {}'.format([self.parameter_names[i] for i in self.constrained_parameters]))

        m_binned = Minuit.from_array_func(self.negPoissonLogLikelihood,
                                      self.initial_values,
                                      error=self.parameter_errors,
                                      errordef = 0.5, limit = self.limits,
                                      name=self.parameter_names,
                                      fix=self.fixes,
                                      print_level=self.print_level,
                                      throw_nan=True
                                      )


        # minimze
        m_binned.migrad(resume=False)
        self.param_states = m_binned.get_param_states()
        self.m_binned = m_binned

        # results
        result_array = m_binned.np_values()
        error_array = m_binned.np_errors()

        if self.print_level == 1:
            logger.info('Fit results: {}'.format(result_array))
            logger.info('Errors: {}'.format(error_array))
        return result_array, error_array


    def PoissonLogLikelihood(self, params):

        # binned data
        hist = self.hist

        # expectation
        pdf_return = self.PDF(self.bin_centers, *params)
        if np.shape(pdf_return)[0] == 2:
            expectation, expectation_error = pdf_return
        else:
            expectation = pdf_return

        # exclude bins where expectation is <= zero or nan
        index = np.where(expectation>0)

        # poisson log likelihoood
        ll = (hist[index]*np.log(expectation[index]) - expectation[index]).sum()

        # extended ll: poisson total number of events
        N = np.nansum(expectation)
        extended_ll = -N+np.sum(hist)*np.log(N)+ll
        return extended_ll


    def negPoissonLogLikelihood(self, params):

        # neg log likelihood
        neg_ll = - self.PoissonLogLikelihood(params)

        # constrained parameters
        if len(self.constrained_parameters) > 0:
            for i in range(len(self.constrained_parameters)):
                i_param = self.constrained_parameters[i]
                neg_ll += 0.5 * ((params[i_param] - self.constrained_means[i])/ self.constrained_widths[i])**2

        return neg_ll
