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
from mermithid.misc import TritiumFormFactor, KuriePlotTools
logger = morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)

class KuriePlotFitter(BaseProcessor):
    '''                                                                                                                                
    Describe.
    '''

    def InternalConfigure(self, params):
        '''
        Configure
        '''
        # Initialize Canvas
        self.rootcanvas = RootCanvas(params, optStat=0)
        self.histo = RootHistogram(params, optStat=0)
        logger.debug(params)
        # Read other parameters
        self.namedata = reader.read_param(params, 'variables', "required")
        return True

    def _FitKuriePlot(self, centralList, kurieList, errorList):
        logger.debug("Fitter private method")
        from ROOT import TGraphErrors, TF1
        kurieGraph = TGraphErrors()
        for i, KE in enumerate(centralList):
            kurieGraph.SetPoint(i, KE, kurieList[i])
            kurieGraph.SetPointError(i, 0, errorList[i])
        fitFunction = TF1("kurieFit", "[0]*([1]-x)*(x<[1]) + [2]", centralList[0], centralList[-1], 3)
        fitFunction.SetParameters(kurieList[0]/(18600-centralList[0]), 18600, kurieList[-1])
        kurieGraph.Fit(fitFunction, 'MLER') 

    def InternalRun(self):
        from ROOT import TMath, TH1F
        data = self.data.get(self.namedata)
        centralValueList, kurieList, errorList = KuriePlotTools.KuriePlotBinning(data, xRange=[self.histo.x_min, self.histo.x_max], nBins=self.histo.histo.GetNbinsX())
        logger.debug("Setting values and errors")
        self.histo.SetBinsContent(kurieList)
        self.histo.SetBinsError(errorList)
        logger.debug("Draw kurie histo")
        self.rootcanvas.cd()
        self.histo.Draw("hist")
        self.rootcanvas.Save()
        self._FitKuriePlot(centralValueList, kurieList, errorList)
        return True
    

