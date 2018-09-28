'''                                                                                                                                     
Generate a Kurie plot from energy data
Author: M. Guigue
Date: Mar 30 2018
'''

from __future__ import absolute_import

import json
import os


from morpho.utilities import morphologging, reader, plots
from morpho.processors import BaseProcessor
from morpho.processors.plots import RootCanvas, RootHistogram
from mermithid.misc import TritiumFormFactor, KuriePlotBinning
logger=morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)

class KuriePlotGeneratorProcessor(BaseProcessor):
    '''                                                                                                                                
    Describe.
    '''

    def InternalConfigure(self, params):
        '''
        Configure
        '''
        # Initialize Canvas
        self.rootcanvas = RootCanvas(params,optStat=0)
        self.histo = RootHistogram(params,optStat=0)

        # Read other parameters
        self.namedata = reader.read_param(params,'data',"required")
        return True

    def InternalRun(self):
        from ROOT import TMath, TH1F
        data = self.data.get(self.namedata)
        kurieList, errorList = KuriePlotBinning.KuriePlotBinning(data, xRange=[self.histo.x_min,self.histo.x_max],nBins=self.histo.histo.GetNbinsX())
        self.histo.SetBinsContent(kurieList[i])
        self.histo.SetBinsError(errorList[i])
        self.rootcanvas.cd()
        self.histo.Draw("hist")
        self.rootcanvas.Save()
        return True
    

