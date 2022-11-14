'''
Calculate analytic sensitivity
function.
Author: C. Claessens
Date:12/16/2021

More description
'''

from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import time
import warnings
warnings.filterwarnings("ignore")

# Numericalunits is a package to handle units and some natural constants
# natural constants

from numericalunits import meV, eV, m, T


# morpho imports
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.processors.Sensitivity import AnalyticSensitivityEstimation
from mermithid.misc.SensitivityFormulas import Sensitivity


logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

class ConstantSensitivityParameterPlots(AnalyticSensitivityEstimation):
    '''
    Description
    Args:

    Inputs:

    Output:

    '''
    # first do the BaseProcessor __init__
    def __init__(self, name, *args, **kwargs):
        BaseProcessor.__init__(self, name, *args, **kwargs)

    def InternalConfigure(self, params):
        '''
        Configure
        '''
        # file paths
        self.config_file_path = reader.read_param(params, 'config_file_path', "required")
        self.sensitivity_target = reader.read_param(params, 'sensitivity_target', [0.4**2/np.sqrt(1.64), 0.7**2/np.sqrt(1.64), 1**2/np.sqrt(1.64)] )
        self.initialInhomogeneity = reader.read_param(params, 'initial_Inhomogeneity', 1e-8)

        self.veff_range = reader.read_param(params, 'veff_range', [0.001, 1])
        self.density_range = reader.read_param(params, 'density_range', [5.e15,1.e19])
        self.BError_range = reader.read_param(params, 'BError_range', [1e-8,1e-3])

        # Configuration is done in __init__ of SensitivityClass
        Sensitivity.__init__(self, self.config_file_path)

        self.rhos = np.linspace(self.density_range[0], self.density_range[1], 1000)/(m**3)
        self.veffs = np.logspace(np.log10(self.veff_range[0]), np.log10(self.veff_range[1]), 25)*m**3
        self.Berrors = np.logspace(np.log10(self.BError_range[0]), np.log10(self.BError_range[1]), 2000)

        freq_res = np.empty(np.shape(self.rhos))
        freq_res_delta = np.empty(np.shape(self.rhos))

        for i in range(len(self.rhos)):
            self.Experiment.number_density = self.rhos[i]
            freq_res[i], freq_res_delta[i] = self.syst_frequency_extraction()


        """plt.figure()
        plt.subplot(121)
        plt.plot(self.rhos*m**3, freq_res/eV)
        plt.xlabel('Density (/m³')
        plt.ylabel('Energy resolution from frequency uncertainty (eV)')
        plt.xscale('log')

        plt.subplot(122)
        plt.plot(self.rhos*m**3, freq_res_delta/eV)
        plt.xlabel('Density (/m³)')
        plt.ylabel('Uncertainty on energy resolution from frequency uncertainty (eV)')
        plt.xscale('log')
        plt.tight_layout()

        plt.savefig('Frequency_resolution_vs_rho.pdf')
        plt.show()"""

        return True



    def InternalRun(self):



        #self.results = {'CL90_limit': self.CL90()/eV, 'm_beta_squared_uncertainty': self.sensitivity()/(eV**2)}
        #print(self.results)

        self.needed_Bs = np.empty((len(self.sensitivity_target), len(self.veffs)))
        self.needed_res = np.empty((len(self.sensitivity_target), len(self.veffs)))
        self.needed_res_sigma = np.empty((len(self.sensitivity_target), len(self.veffs)))
        self.needed_freq_res = np.empty((len(self.sensitivity_target), len(self.veffs)))
        self.needed_freq_res_sigma = np.empty((len(self.sensitivity_target), len(self.veffs)))
        self.rho_opts = np.empty((len(self.sensitivity_target), len(self.veffs)))
        self.CLs = np.empty((len(self.sensitivity_target), len(self.veffs)))
        index = []

        for j in range(len(self.sensitivity_target)):
            for i, veff in enumerate(self.veffs):
                logger.info('\nVeff = {} m³'.format(veff/(m**3)))
                self.Experiment.v_eff = veff
                self.MagneticField.inhomogenarity = self.initialInhomogeneity
                self.rho_opts[j, i]=self.FindOptimumPressure()
                self.BHomogeneityAndResNeeded(self.sensitivity_target[j])
                n = 0
                drho =  self.density_range[1]/(m**3)*1.5

                while np.abs(drho)> self.rho_opts[j, i] * 0.001 and n<10:
                    logger.info('Iteration: {}'.format(n))
                    n+=1

                    new_rho = self.FindOptimumPressure()
                    drho = self.rho_opts[j, i]-new_rho
                    self.rho_opts[j, i] = new_rho
                    self.needed_Bs[j, i], self.needed_res[j, i], self.needed_res_sigma[j, i], self.needed_freq_res[j, i], self.needed_freq_res_sigma[j, i] = self.BHomogeneityAndResNeeded(self.sensitivity_target[j])

                self.CLs[j, i] = self.CL90()



            index.append(np.where((self.CLs[j]/eV<np.sqrt(1.1*self.sensitivity_target[j]*np.sqrt(1.64))) & (self.needed_Bs[j]>self.Berrors[1])))
            logger.info('Achieved 90CL limits: {}'.format(self.CLs[j]/eV))
            time.sleep(1)


        plt.figure(figsize=(10, 5))

        plt.subplot(121)

        for j in range(len(self.sensitivity_target)):
            plt.plot(self.veffs[index[j]]/(m**3), self.needed_Bs[j][index[j]], label='90% CL = {} eV'.format(np.round(np.sqrt(np.sqrt(1.64)*self.sensitivity_target[j]),1)))
            #plt.scatter(self.veffs/(m**3), self.needed_Bs, c=self.CLs/eV, marker='.')
        #plt.colorbar()
        plt.xlabel('Effective Volume (m³)')
        plt.ylabel('Required field homogeneity')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        #plt.savefig('B_vs_veff.pdf')

        plt.subplot(122)
        for j in range(len(self.sensitivity_target)):
            plt.plot(self.veffs[index[j]]/(m**3), self.rho_opts[j][index[j]]*m**3, label='90% CL = {} eV'.format(np.round(np.sqrt(np.sqrt(1.64)*self.sensitivity_target[j]),1)))
        plt.xlabel('Effective Volume (m³)')
        plt.ylabel('Optimum number density (1/m³)')
        plt.xscale('log')
        #plt.yscale('log')
        plt.legend()
        plt.tight_layout()

        plt.savefig('B_rho_vs_veff.pdf')
        plt.savefig('B_rho_vs_veff.png', dpi=300)

        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        for j in range(len(self.sensitivity_target)):
            plt.plot(self.veffs[index[j]]/(m**3), self.needed_res[j][index[j]]/eV, label='90% CL = {} eV'.format(np.round(np.sqrt(np.sqrt(1.64)*self.sensitivity_target[j]),1)))
        plt.xlabel('Effective Volume (m³)')
        plt.ylabel('Total energy resolution (eV)')
        plt.xscale('log')
        #plt.yscale('log')
        plt.legend()
        plt.tight_layout()

        plt.subplot(122)
        for j in range(len(self.sensitivity_target)):
            plt.plot(self.veffs[index[j]]/(m**3), self.needed_res_sigma[j][index[j]]/eV, label='90% CL = {} eV'.format(np.round(np.sqrt(np.sqrt(1.64)*self.sensitivity_target[j]),1)))
        plt.xlabel('Effective Volume (m³)')
        plt.ylabel('Uncertainty on total energy resolution (eV)')
        plt.xscale('log')
        #plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig('res_vs_veff.pdf')
        plt.savefig('res_vs_veff.png', dpi=300)

        plt.show()
        return True

    def SensVSrho(self, rho):
        self.Experiment.number_density = rho
        return self.CL90(Experiment={"number_density": rho})

    def DistanceToTargetVSBError(self, BError, sensitivity_target):
        sig = self.BToKeErr(self.MagneticField.nominal_field*BError, self.MagneticField.nominal_field)
        self.MagneticField.usefixedvalue = True
        self.MagneticField.default_systematic_smearing = sig
        self.MagneticField.default_systematic_uncertainty = 0.05*sig
        #print(self.sensitivity()/(eV**2), self.sensitivity_target)
        return np.abs(self.sensitivity()/(eV**2)-sensitivity_target)

    def FindOptimumPressure(self):
        limit = [self.SensVSrho(rho)/eV for rho in self.rhos]
        opt_rho_index = np.argmin(limit)

        rho_opt = self.rhos[opt_rho_index]
        if rho_opt == self.rhos[0] or rho_opt == self.rhos[-1]:
            raise ValueError('Optimum rho {} is on edge or range'.format(rho_opt*m**3))

        result = minimize(self.SensVSrho, rho_opt, method='Nelder-Mead')
        if result.success:
            logger.info('\tReplacing numerical value by actual optimiuation result.')
            rho_opt = result.x[0]

        limit = self.CL90(Experiment={"number_density": rho_opt})/eV
        logger.info('\tOptimum density: {}'.format(rho_opt*m**3))
        return rho_opt

    def FindMaxAllowedBerror(self, sensitivity_target):
        distance_to_target = [self.DistanceToTargetVSBError(Berror, sensitivity_target) for Berror in self.Berrors]
        optBerror_index = np.argmin(distance_to_target)
        Berror_opt = self.Berrors[optBerror_index]



        result = minimize(self.DistanceToTargetVSBError, Berror_opt, args=(sensitivity_target), method='Nelder-Mead')
        if result.success and result.x[0]>0:
            Berror_opt = result.x[0]

        sig = self.BToKeErr(self.MagneticField.nominal_field*Berror_opt, self.MagneticField.nominal_field)
        self.MagneticField.usefixedvalue = True
        self.MagneticField.default_systematic_smearing = sig
        self.MagneticField.default_systematic_uncertainty = 0.05*sig

        return Berror_opt


    def BHomogeneityAndResNeeded(self, sensitivity_target):
        neededBres = self.FindMaxAllowedBerror(sensitivity_target)
        labels, sigmas, deltas = self.get_systematics()
        # #b_sigma_tmp = (sigma_sys/4)**2
        # needed_total_sigma = 0
        # for i,l in enumerate(labels):
        #     #print('\t', l, sigmas[i]/eV, deltas[i])
        #     if l != 'Magnetic Field':
        #         #b_sigma_tmp -= (sigmas[i]*deltas[i])**2
        #         needed_total_sigma += sigmas[i]**2
        #     else:
        #         sigma_b_old = sigmas[i]
        #         delta_b_old = deltas[i]

        # needed_sigma_b = sigma_b_old #np.sqrt(b_sigma_tmp)/delta_b_old
        needed_sigma_b = sigmas[np.where(labels=='Magnetic Field')]
        needed_b_error = self.KeToBerr(needed_sigma_b, self.MagneticField.nominal_field)
        # needed_total_sigma = np.sqrt(needed_sigma_b**2 + needed_total_sigma)
        needed_total_sigma = np.sqrt(np.sum(sigmas**2))
        needed_total_delta = np.sqrt(np.sum(deltas**2))

        needed_freq_sigma = sigmas[np.where(labels=='Start Frequency Resolution')]
        needed_freq_delta = deltas[np.where(labels=='Start Frequency Resolution')]
        return needed_b_error/self.MagneticField.nominal_field, needed_total_sigma, needed_total_delta, needed_freq_sigma, needed_freq_delta



    # # Sensitivity formulas:
    # # These are the functions that have so far been in the SensitivityClass but would not be used by a fake data experiment
    # # I did not include DeltaEWidth here becuase I consider it more general information about an experiment that could be used by the fake data studies too

    # def StatSens(self):
    #     """Pure statistic sensitivity assuming Poisson count experiment in a single bin"""
    #     sig_rate = self.SignalRate()
    #     DeltaE = self.DeltaEWidth()
    #     sens = 2/(3*sig_rate*self.Experiment.LiveTime)*np.sqrt(sig_rate*self.Experiment.LiveTime*DeltaE
    #                                                               +self.BackgroundRate()*self.Experiment.LiveTime/DeltaE)
    #     return sens

    # def SystSens(self):
    #     """Pure systematic componenet to sensitivity"""
    #     labels, sigmas, deltas = self.get_systematics()
    #     sens = 4*np.sqrt(np.sum((sigmas*deltas)**2))
    #     return sens

    # def sensitivity(self, **kwargs):
    #     """Combined statisical and systematic uncertainty.
    #     Using kwargs settings in namespaces can be changed.
    #     Example how to change number density which lives in namespace Experiment:
    #         self.sensitivity(Experiment={"number_density": rho})
    #     """
    #     for sect, options in kwargs.items():
    #         for opt, val in options.items():
    #             self.__dict__[sect].__dict__[opt] = val

    #     StatSens = self.StatSens()
    #     SystSens = self.SystSens()

    #     # Standard deviation on a measurement of m_beta**2
    #     sigma_m_beta_2 =  np.sqrt(StatSens**2 + SystSens**2)
    #     return sigma_m_beta_2

    # def CL90(self, **kwargs):
    #     """ Gives 90% CL upper limit on neutrino mass."""
    #     # 90% of gaussian are contained in +-1.64 sigma region
    #     return np.sqrt(np.sqrt(1.64)*self.sensitivity(**kwargs))

    # def sterial_m2_limit(self, Ue4_sq):
    #     return np.sqrt(np.sqrt(1.64)*np.sqrt((self.StatSens()/Ue4_sq)**2 + self.SystSens()**2))


    # # print functions
    # def print_statistics(self):
    #     print("Statistic", " "*18, "%.2f"%(np.sqrt(self.StatSens())/meV), "meV")

    # def print_systematics(self):
    #     labels, sigmas, deltas = self.get_systematics()

    #     print()
    #     for label, sigma, delta in zip(labels, sigmas, deltas):
    #         print(label, " "*(np.max([len(l) for l in labels])-len(label)),  "%8.2f"%(sigma/meV), "+/-", "%8.2f"%(delta/meV), "meV")