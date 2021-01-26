'''
Author: C. Claessens
Date:1/24/2021
Description:
    Currently empty processor

'''

from __future__ import absolute_import

import numpy as np
from scipy.optimize import curve_fit

from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor


logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

class LeastSquaresPolynomialFitter(BaseProcessor):
    '''
    Processor that ...
    Args:

    Inputs:
        data:

    Output:
        result: dictionary containing fit results and uncertainties
    '''
    def InternalConfigure(self, params):
        '''
        Args:
            x_key (required): key for data x coordinates
            y_key (required): key for data y coordinates
            error_key (optional): key for data y uncertainties. default '' is assumed to mean no uncertainties are provided.
            order (optional): polynomial order to be fitted. default is 1.
            initial_guess (optional): list with initial guesses. length has to match order. defaults are 1.
        '''

        self.x_key = reader.read_param(params, 'x_key', 'required')
        self.y_key = reader.read_param(params, 'y_key', 'required')
        self.error_key = reader.read_param(params, 'error_key', '')
        self.order = reader.read_param(params, 'order', 1)
        self.initial_guess = reader.read_param(params, 'initial_guess', np.ones(self.order+1))

        # do checks
        if len(self.initial_guess) != self.order+1:
            logger.error('Length of inital guess list does not match poylnomial order')
            return False

        # derived configurations
        if self.error_key != '':
            self.fit_with_uncertainties = True
        else:
            self.fit_with_uncertainties = False

        return True


    def InternalRun(self):
        '''
        Fits polynomial to data
        '''
        x = self.data[self.x_key]
        y = self.data[self.y_key]

        if self.fit_with_uncertainties:
            y_error = self.data[self.error_key]
            absolute_sigma = True
        else:
            y_error = None
            absolute_sigma = False

        # now fit
        pars, cov = curve_fit(self.fit_function, x, y, sigma=y_error,
                              p0=self.initial_guess, absolute_sigma=absolute_sigma)

        sorted_x = np.sort(x)
        self.results = {'sorted_x': sorted_x, 'best_fit': self.fit_function(sorted_x, *pars),
                        'coefficients': pars, 'covariance_matrix': cov}

        return True

    def fit_function(self, x, *p):
        '''
        Parameters
        ----------
        x : numpy array
        p : list or array
            coefficients

        Returns
        -------
        fit_result
        '''

        fit_result = np.zeros(np.shape(x))
        for i in range(len(p)):
            fit_result += p[i]*np.array(x)**i
        return fit_result


