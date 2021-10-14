'''
Author: C. Claessens
Date:8/2/2021
Description:


'''

from __future__ import absolute_import

import numpy as np
from copy import deepcopy

from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor

logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

class MCUncertaintyPropagation(BaseProcessor):
    '''
    Processor that
    Args:

    Inputs:
        data:
    Output:
        result: dictionary containing fit results and uncertainties
    '''
    def InternalConfigure(self, params):
        '''
        Configure
        '''

        self.model = reader.read_param(params, 'model', "required")
        self.fit = reader.read_param(params, 'fit_function', "required")
        self.gen_and_fit = reader.read_param(params, 'gen_and_fit_function', "required")
        self.fit_config_dict = deepcopy(reader.read_param(params, 'fit_config_dict', "required"))
        self.fit_options = reader.read_param(params, 'fit_options', "optional")
        self.sample_parameters = reader.read_param(params, 'sample_parameters', [])
        self.stat_sys_combined = reader.read_param(params, 'stat_sys_combined', [True, True, True])
        self.N = reader.read_param(params, 'N_samples', 50)

        return True

    def InternalRun(self):

        self.results = {}

        for k in self.fit_options.keys():
            self.fit_config_dict[k] = self.fit_options[k]

        self.InitialFit()
        self.ParameterSampling()


        #self.results['best_fit'] = list(self.fitted_params)

        return True

    def InitialFit(self):
        '''
        Fit data with best model

        Returns
        -------
        None.

        '''

        fit_successful = False
        counter = 0
        #while (not fit_successful) and counter < 15:
        #    counter += 1
        #    try:

        self.fitted_params, self.fitted_params_errors, self.Counts = self.fit(self.data,
                                                               self.fit_config_dict)
        fit_successful = True
        #    except Exception as e:
        #       print(e)
        #       logger.error('Repeating fit')
        #       continue


        logger.info('Best fit: {}'.format(self.fitted_params))
        x, pdf, bins, fitted_model, asimov_data = self.model(self.fit_config_dict,
                                                                     params=self.fitted_params)
        return True

    def ParameterSampling(self):

        start_j = 0

        # propagate errors by sampling events and prior
        return_dict_keys = ['stat', 'sys', 'combined']
        parameter_sampling = [{}, self.sample_parameters, self.sample_parameters]


        for k_i, k in enumerate(return_dict_keys):
            if self.stat_sys_combined[k_i]:

                if k == 'sys':
                    fixed_data = ['asimov']
                elif k == 'combined':
                    fixed_data = []
                else:
                    fixed_data = []


                Nparams = len(self.fitted_params)
                offsets = np.zeros((Nparams, self.N))
                sigmas = np.zeros((Nparams, self.N))
                fit_results = np.zeros((Nparams, self.N))
                parameter_samples = []


                # sequential sampling
                all_fit_returns = []
                for i in range(start_j, self.N):
                    fit_successful = False
                    counter = 0
                    while (not fit_successful) and counter < 5:
                        counter += 1
                        try:
                            all_fit_returns.append(self.gen_and_fit(self.fitted_params, self.Counts,
                                                             self.fit_config_dict,
                                                             parameter_sampling[k_i],
                                                             i, fixed_data))
                            fit_successful = True
                        except Exception as e:
                            print(e)
                            logger.error('Repeating fit')
                            continue



                ###################################
                # sort fit results
                ###################################
                #all_fit_returns = np.array(all_fit_returns)
                for ii in range(len(all_fit_returns)):
                    j = ii + start_j
                    results = all_fit_returns[ii][0]
                    errors = all_fit_returns[ii][1]
                    parameter_samples.append(all_fit_returns[ii][2])


                    for i in range(len(self.fitted_params)):
                        offsets[i][j] = results[i]-self.fitted_params[i]
                        sigmas[i][j] = errors[i]
                        fit_results[i][j] = results[i]


                parameter_samples_transpose = {}
                for p_key in self.sample_parameters.keys():
                    parameter_samples_transpose[p_key] = []
                    if len(parameter_samples) > 0 and p_key in parameter_samples[0].keys():
                        for p_i in parameter_samples:
                            parameter_samples_transpose[p_key].append(p_i[p_key])


                self.results[k] = {'offsets': [list(o) for o in offsets],
                                   'sigmas': [list(s) for s in sigmas],
                                   'results': [list(fr) for fr in fit_results],
                                   'parameter_samples': parameter_samples_transpose,
                                   'best_fit': list(self.fitted_params),
                                   'best_fit_errors': list(self.fitted_params_errors)
                                 }



