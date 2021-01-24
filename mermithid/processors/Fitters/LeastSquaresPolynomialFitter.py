'''
Author: C. Claessens
Date:1/24/2021
Description:
    Currently empty processor

'''

from __future__ import absolute_import

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


        return True


    def InternalRun(self):
        '''
        Description
        '''

        self.results = {}

        return True


