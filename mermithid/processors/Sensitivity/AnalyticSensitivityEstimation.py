'''
Calculate analytic sensitivity
function.
Author: C. Claessens
Date:12/16/2021

More description
'''

from __future__ import absolute_import


import numpy as np


# Numericalunits is a package to handle units and some natural constants
# natural constants

from numericalunits import meV, eV


# morpho imports
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from mermithid.misc.SensitivityFormulas import Sensitivity


logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

class AnalyticSensitivityEstimation(BaseProcessor, Sensitivity):
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

        # setup sensitivities requires to run Sensitivity input class __init__
        Sensitivity.__init__(self, self.config_file_path)

        return True



    def InternalRun(self):

        self.results = {'CL90_limit': self.CL90()/eV, 'm_beta_squared_uncertainty': self.sensitivity()/(eV**2)}

        return True


    # Sensitivity formulas

    def StatSens(self):
        """Pure statistic sensitivity assuming Poisson count experiment in a single bin"""
        sig_rate = self.SignalRate()
        DeltaE = self.DeltaEWidth()
        sens = 2/(3*sig_rate*self.Experiment.LiveTime)*np.sqrt(sig_rate*self.Experiment.LiveTime*DeltaE
                                                                  +self.BackgroundRate()*self.Experiment.LiveTime/DeltaE)
        return sens

    def SystSens(self):
        """Pure systematic componenet to sensitivity"""
        labels, sigmas, deltas = self.get_systematics()
        sens = 4*np.sqrt(np.sum((sigmas*deltas)**2))
        return sens

    def sensitivity(self, **kwargs):
        """Combined statisical and systematic uncertainty.
        Using kwargs settings in namespaces can be changed.
        Example how to change number density which lives in namespace Experiment:
            self.sensitivity(Experiment={"number_density": rho})
        """
        for sect, options in kwargs.items():
            for opt, val in options.items():
                self.__dict__[sect].__dict__[opt] = val

        StatSens = self.StatSens()
        SystSens = self.SystSens()

        # Standard deviation on a measurement of m_beta**2
        sigma_m_beta_2 =  np.sqrt(StatSens**2 + SystSens**2)
        return sigma_m_beta_2

    def CL90(self, **kwargs):
        """ Gives 90% CL upper limit on neutrino mass."""
        # 90% of gaussian are contained in +-1.64 sigma region
        return np.sqrt(np.sqrt(1.64)*self.sensitivity(**kwargs))

    def sterial_m2_limit(self, Ue4_sq):
        return np.sqrt(np.sqrt(1.64)*np.sqrt((self.StatSens()/Ue4_sq)**2 + self.SystSens()**2))


    # print functions
    def print_statistics(self):
        print("Statistic", " "*18, "%.2f"%(np.sqrt(self.StatSens())/meV), "meV")

    def print_systematics(self):
        labels, sigmas, deltas = self.get_systematics()

        print()
        for label, sigma, delta in zip(labels, sigmas, deltas):
            print(label, " "*(np.max([len(l) for l in labels])-len(label)),  "%8.2f"%(sigma/meV), "+/-", "%8.2f"%(delta/meV), "meV")