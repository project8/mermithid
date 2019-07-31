'''
Processor for shifting frequencies
Author: M. Guigue
Date: 10/11/18
'''

from __future__ import absolute_import

try:
    import numpy as np
except ImportError:
    pass

from morpho.processors import BaseProcessor
from morpho.utilities import morphologging, reader
logger=morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)

class FrequencyShifter(BaseProcessor):

    def InternalConfigure(self, params):
        self.frequencies = list()
        self.results = list()
        self.frequency_shift = reader.read_param(params, "frequency_shift", 0)
        return True

    def InternalRun(self):
        if len(self.frequencies) == 0:
            logger.error("No frequencies given")
            self.results = {}
            return True
        if self.frequency_shift == 0:
            looger.warning("Zero frequency shift!")
            self.results = self.frequencies
            return True
        for frequency in self.frequencies:
            self.results.append(frequency+self.frequency_shift)
        return True 
        

