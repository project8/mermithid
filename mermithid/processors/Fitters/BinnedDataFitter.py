'''
Author: C. Claessens
Date:6/12/2020
Description:
    Minimizes poisson loglikelihood using iMinuit.
    Configure with arbitrary histogram and model.
    The model should consist of a pdf multiplied with a free amplitude parameter representing the total number of events.

'''

from __future__ import absolute_import

import numpy as np
#import scipy
from scipy.optimize import NonlinearConstraint
#import sys
#import json
from iminuit import Minuit
from iminuit.cost import LeastSquares
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from scipy.stats import chisquare
from scipy.special import factorial

logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

class BinnedDataFitter(BaseProcessor):
    '''
    Processor that
    Args:
        variables: dictionary key under which data is stored
        parameter_names: names of model parameters
        initial_values: list of parameter initial values
        limits: parameter limits given as [lower, upper] for each parameter
        fixed: boolean list of same length as parameters. Only un-fixed parameters will be fitted
        bins: bins for data histogram
        binned_data: if True data is assumed to already be histogrammed
        print_level: if 1 fit plan and result summaries are printed
        constrained_parameter_indices: list of indices indicating which parameters are contraint. Contstraints will be Gaussian.
        constrained_parameter_means: Mean of Gaussian constraint
        constrained_parameter_widths: Standard deviation of Gaussian constrained

    Inputs:
        data:
    Output:
        result: dictionary containing fit results and uncertainties
    '''
    def InternalConfigure(self, params):
        '''
        Configure
        '''

        # Read other parameters
        self.namedata = reader.read_param(params, 'variables', "required")
        self.parameter_names = reader.read_param(params, 'parameter_names', ['A', 'mu', 'sigma'])
        self.initial_values = reader.read_param(params, 'initial_values', [1]*len(self.parameter_names))
        self.limits = reader.read_param(params, 'limits', [[None, None]]*len(self.parameter_names))
        self.fixes = reader.read_param(params, 'fixed', [False]*len(self.parameter_names))
        self.fixes_dict = reader.read_param(params, 'fixed_parameter_dict', {})
        self.bins = reader.read_param(params, 'bins', np.linspace(-2, 2, 100))
        self.binned_data = reader.read_param(params, 'binned_data', False)
        self.print_level = reader.read_param(params, 'print_level', 1)
        self.constrained_parameters = reader.read_param(params, 'constrained_parameter_indices', [])
        self.constrained_means = np.array(reader.read_param(params, 'constrained_parameter_means', []))
        self.constrained_widths = np.array(reader.read_param(params, 'constrained_parameter_widths', []))
        self.correlated_parameters = reader.read_param(params, 'correlated_parameter_indices', [])
        self.cov_matrix = np.array(reader.read_param(params, 'covariance_matrix', []))
        self.minos_cls = reader.read_param(params, 'minos_confidence_level_list', [])
        self.minos_intervals = reader.read_param(params,'find_minos_intervals', False)
        self.free_in_second_fit = reader.read_param(params, 'free_in_second_fit', [])
        self.fixed_in_second_fit = reader.read_param(params, 'fixed_in_second_fit', [])
        self.error_def = reader.read_param(params,'error_def', 1)

        # derived configurations
        self.bin_centers = self.bins[0:-1]+0.5*(self.bins[1]-self.bins[0])
        self.parameter_errors = [max([0.1, 0.1*p]) for p in self.initial_values]

        return True

    def InternalRun(self):
        logger.info('namedata: {}'.format(self.namedata))

        if self.binned_data:
            logger.info('Data is already binned. Getting binned data')
            self.hist = self.data[self.namedata]
            if len(self.hist) != len(self.bin_centers):
                logger.error('Number of bins and histogram entries do not match')
                raise ValueError('Number of bins and histogram entries do not match')
        else:
            logger.info('Data is unbinned. Will histogram before fitting.')
            self.hist, _ = np.histogram(self.data[self.namedata], self.bins)

        logger.info('Total counts: {}'.format(np.sum(self.hist)))

        result_array, error_array = self.fit()

        # save results
        self.results = {}
        self.results['param_values'] = result_array
        self.results['param_errors'] = error_array
        self.results['correlation_matrix'] = np.array(self.m_binned.covariance.correlation())
        for i, k in enumerate(self.parameter_names):
            self.results[k] = {'value': result_array[i], 'error': error_array[i], 'likelihood': self.likelihood}

        return True



    def model(self, x, A, mu, sigma):
        """
        Overwrite by whatever Model
        """
        f = A*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(((x-mu)/sigma)**2.)/2.)
        return f


    def setup_minuit(self):

        if self.error_def == 0.5:
            self.m_binned = Minuit(self.negPoissonLogLikelihood,
                                        self.initial_values,
                                        name=self.parameter_names,
                                        )

            self.m_binned.errordef = 0.5
            logger.info('Doing MLL analysis')
        else:
            #least_squares = LeastSquares(self.bin_centers, self.hist, np.sqrt(self.hist), self.model)
            self.m_binned = Minuit(self.leastSquares,
                                    self.initial_values,
                                    name=self.parameter_names)
            self.m_binned.errordef = 1.0
            logger.info('Doing chi square analysis')

        self.m_binned.errors = self.parameter_errors
        self.m_binned.throw_nan = False
        self.m_binned.strategy = 1
        self.m_binned.print_level = self.print_level

        self.constraints = []

        for i, name in enumerate(self.parameter_names):
            if name in self.fixes_dict.keys():
                self.m_binned.fixed[name]= self.fixes_dict[name]
                #logger.info('Fixing {}'.format(name))
            else:
                self.m_binned.fixed[name] = self.fixes[i]
            self.m_binned.limits[name] = self.limits[i]
            #if not all(l is None for l in self.limits[i]):
            #    self.constraints.append(NonlinearConstraint(lambda x: x, self.limits[i][0], self.limits[i][1]))


    def fit(self):
        # Now minimize neg log likelihood using iMinuit
        self.setup_minuit()


        if self.print_level > 0:
            logger.info('This is the plan:')
            logger.info('\tFitting data consisting of {} elements'.format(np.sum(self.hist)))
            logger.info('\tFit parameters: {}'.format(self.parameter_names))
            logger.info('\tInitial values: {}'.format(self.initial_values))
            logger.info('\tInitial error: {}'.format(self.parameter_errors))
            logger.info('\tLimits: {}'.format(self.limits))
            logger.info('\tFixed in fit: {}'.format(self.fixes))
            logger.info('\tConstrained parameters: {}'.format([self.parameter_names[i] for i in self.constrained_parameters]))
            logger.info('\tConstraint means: {}'.format(self.constrained_means))
            logger.info('\tConstraint widths: {}'.format(self.constrained_widths))
            logger.info('\tCorrelated parameters: {}'.format([self.parameter_names[i] for i in self.correlated_parameters]))
            logger.info('\tError def : {}'.format(self.m_binned.errordef))
            logger.info('\tMinos uncertainties: {}'.format(self.minos_cls))


        # minimze
        self.m_binned.simplex().migrad() #scipy(constraints=self.constraints)
        if len(self.free_in_second_fit) > 0:
            for p in self.free_in_second_fit:
                self.m_binned.fixed[p] = False
            logger.info('{} now free.'.format(self.free_in_second_fit))
        if len(self.fixed_in_second_fit) > 0:
            for p in self.fixed_in_second_fit:
                self.m_binned.fixed[p] = True
                #self.m_binned.limits[p] = [None, None]
            logger.info('{} now fixed'.format(self.fixed_in_second_fit))
        if len(self.free_in_second_fit) > 0 or len(self.fixed_in_second_fit) > 0:
            logger.info('Minimize again')
            self.m_binned.migrad()
        self.m_binned.hesse()
        #m_binned.minos()
        #self.param_states = m_binned.get_param_states()
        if self.print_level:
            logger.info(self.m_binned.params)
            logger.info(self.m_binned.values)
            logger.info(self.m_binned.errors)


        # results
        result_array = np.array(self.m_binned.values)
        error_array = np.array(self.m_binned.errors)
        self.likelihood = self.PoissonLogLikelihood(result_array)


        if self.minos_intervals:
            self.minos_errors = {}
            for mcl in self.minos_cls:
                logger.info('Getting minos errors for CL = {}'.format(mcl))
                try:
                    self.m_binned.minos(cl=mcl)

                    self.minos_errors[mcl] = {}
                    if self.print_level:
                        logger.info(self.m_binned.params)
                        logger.info(self.m_binned.merrors)
                    for k in self.m_binned.merrors.keys():
                        self.minos_errors[mcl][k] = {'interval': [self.m_binned.merrors[k].lower, self.m_binned.merrors[k].upper],
                                                    'number': self.m_binned.merrors[k].number,
                                                    'name': self.m_binned.merrors[k].name,
                                                    'is_valid': self.m_binned.merrors[k].is_valid and self.m_binned.valid}

                except RuntimeError as e:
                    print(self.m_binned.params)
                    print(self.m_binned.merrors)
                    logger.error('Minos failed. Returning Hesse instead')
                    self.minos_errors[mcl] = {}
                    for i in range(len(self.parameter_names)):
                        self.minos_errors[mcl][self.parameter_names[i]] = {'interval': [-error_array[i], error_array[i]],
                                                                            'number': i,
                                                                            'name': self.parameter_names[i],
                                                                            'is_valid': self.m_binned.valid}
                    #raise e

        else:
            self.hesse_errors = {0.683: {}}
            for i in range(len(self.parameter_names)):
                self.hesse_errors[0.683][self.parameter_names[i]] = {'interval': [-error_array[i], error_array[i]],
                                                                            'number': i,
                                                                            'name': self.parameter_names[i],
                                                                            'is_valid': self.m_binned.valid}

        if self.print_level == 1:
            logger.info(self.m_binned.fmin)
            #logger.info('Correlation matrix: {}'.format(self.m_binned.covariance.correlation()))

        return result_array, error_array


    def PoissonLogLikelihood(self, params):


        # expectation
        model_return = self.model(self.bin_centers, *params)
        if np.shape(model_return)[0] == 2:
            expectation, expectation_error = model_return
        else:
            expectation = model_return

        if np.min(expectation) < 0:
            logger.error('Expectation contains negative numbers: Minimum {} -> {}.'.format(np.argmin(expectation), np.min(expectation)))
            logger.error('FYI, the parameters are: {}'.format(params))
            logger.info('Expectation is: {}'.format(expectation))
            import matplotlib.pyplot as plt
            plt.figure()
            plt.step(self.bin_centers, self.hist)
            plt.plot(self.bin_centers, expectation)

            #raise ValueError('Expecation below zero')

            logger.error("try again")
            # expectation
            model_return = self.model(self.bin_centers, *params)
            if np.shape(model_return)[0] == 2:
                expectation, expectation_error = model_return
            else:
                expectation = model_return
            plt.plot(self.bin_centers, expectation,linestyle='--')
            plt.savefig('negative_expectation.png', dpi=200)

        # exclude bins where expectation is <= zero or nan
        index = np.where(expectation>0)#np.where(expectation>0)#np.finfo(0.0).resolution)
        if "background" in self.parameter_names and self.m_binned.fixed["background"]:
            endpoint_parameter = self.parameter_names.index('endpoint')
            index = np.where((self.bin_centers+0.5*(self.bin_centers[1]-self.bin_centers[0])<=params[endpoint_parameter]) &
                            (self.hist>=5))
            #index = np.arange(np.min(np.where(self.hist>1)[0]), np.max(np.where(self.hist>1)[0])+1)
            #print(self.hist[index])
            #print(self.bin_centers[index])

        # poisson log likelihoood
        log_factorial = np.array([np.sum(np.log(np.arange(1, n+1))) for n in self.hist[index]])
        ll = (self.hist[index]*np.log(expectation[index]) - expectation[index]-log_factorial).sum()

        # extended ll: poisson total number of events
        N = np.nansum(expectation)
        log_factorial = np.sum(np.log(np.arange(1, np.sum(self.hist)+1)))
        extended_ll = -N+np.sum(self.hist)*np.log(N)+ll-log_factorial
        return extended_ll


    def negPoissonLogLikelihood(self, params):

        # neg log likelihood
        neg_ll = - self.PoissonLogLikelihood(params)

        # constrained parameters
        if len(self.constrained_parameters) > 0:
            for i, param in enumerate(self.constrained_parameters):
                # only do uncorrelated parameters
                if not param in self.correlated_parameters:
                    neg_ll += 0.5 * ((params[param] - self.constrained_means[i])/ self.constrained_widths[i])**2 + 0.5*np.log(2*np.pi) + np.log(self.constrained_widths[i])

        constrained_indices = np.in1d(self.constrained_parameters, self.correlated_parameters).nonzero()[0]
        if len(constrained_indices) > 0:
            dim = len(self.correlated_parameters)
            constrained_indices = np.in1d(self.constrained_parameters, self.correlated_parameters).nonzero()[0]
            param_indices = self.correlated_parameters
            neg_ll += 0.5*(np.log(np.linalg.det(self.cov_matrix)) + \
                    np.dot(np.transpose(np.subtract(params[param_indices], np.array(self.constrained_means)[constrained_indices])), \
                        np.dot(np.linalg.inv(self.cov_matrix), np.subtract(params[param_indices], np.array(self.constrained_means)[constrained_indices]))) + \
                            dim*np.log(2*np.pi))

        return neg_ll


    def leastSquares(self, params):

        # expectation
        model_return = self.model(self.bin_centers, *params)
        if np.shape(model_return)[0] == 2:
            expectation, expectation_error = model_return
        else:
            expectation = model_return

        nonzero_bins_index = np.where(self.hist>0)#np.where(expectation>0)#np.finfo(0.0).resolution)
        zero_bins_index = np.where(self.hist==0)
        index = np.where(expectation>0)
        if "background" in self.parameter_names and self.m_binned.fixed["background"]:
        #    endpoint_parameter = self.parameter_names.index('endpoint')
            index = np.where((expectation>1) & (self.hist>1))
        total_chisquare, _ = chisquare(self.hist[index], expectation[index])
        #chi2 = np.nansum((self.hist[nonzero_bins_index]-expectation[nonzero_bins_index])**2/expectation[nonzero_bins_index])
        #chi2+= np.nansum((self.hist[zero_bins_index]-expectation[zero_bins_index])**2)
        #chi2 = 2*((expectation - self.hist + self.hist*np.log(self.hist/expectation))[nonzero_bins_index]).sum()
        #chi2 += 2*(expectation - self.hist)[zero_bins_index].sum()
        return total_chisquare #LeastSquares(self.bin_centers, self.hist, np.sqrt(self.hist), self.model)