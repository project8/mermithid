'''                                                                                                                                     
Generate a Kurie plot from energy data
'''

from __future__ import absolute_import

import json
import os


from morpho.utilities import morphologging, reader, plots
from morpho.processors import BaseProcessor
from morpho.processors.plots import RootCanvas, RootHistogram
from mermithid.misc import TritiumFormFactor
logger=morphologging.getLogger(__name__)

__all__ = []
__all__.append(__name__)

class KuriePlotGeneratorProcessor(BaseProcessor):
    '''                                                                                                                                
    Describe.
    '''

    def _Configure(self, params):
        '''
        Configure
        '''
        # Initialize Canvas
        self.rootcanvas = RootCanvas(params,optStat=0)
        self.histo = RootHistogram(params,optStat=0)

        # Read other parameters
        self.namedata = reader.read_param(params,'data',"required")

    def _Run(self):
        from ROOT import TMath, TH1F
        data = self.data.get(self.namedata)
        if self.histo.x_min>self.histo.x_max:
            xMin = min(data)
            xMax = max(data)
            self.histo.x_min, self.histo.x_max = xMin,xMax
            self.histo._createHisto()
        histo1 = TH1F(self.histo.title+"_data", self.histo.title+"_data", self.histo.n_bins_x, self.histo.x_min, self.histo.x_max)
        kurieList = []
        for value in data:
            histo1.Fill(value)
        for iBin in range(histo1.GetNbinsX()):
            kurieList.append(TMath.Sqrt(histo1.GetBinContent(iBin) / TritiumFormFactor.RFactor(histo1.GetBinCenter(iBin), 1)))
        self.histo.SetBinsContent(kurieList)
        self.rootcanvas.cd()
        self.histo.Draw("hist")
        self.rootcanvas.Save()
    

