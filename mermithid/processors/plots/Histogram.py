'''
Plot a corrected histogram by dividing the histogram by a defined efficiency
function.
Author: M. Guigue, A. Ziegler
Date:8/1/2019
'''

from __future__ import absolute_import

from ROOT import TF1
from morpho.utilities import morphologging, reader
from morpho.processors import BaseProcessor
from morpho.processors.plots import RootCanvas
from .RootHistogram import RootHistogram
logger = morphologging.getLogger(__name__)



__all__ = []
__all__.append(__name__)

class Histogram(BaseProcessor):

    def InternalConfigure(self, params):
        '''
        Configure
        '''
        # Initialize Canvas
        # try:
        #     self.rootcanvas = RootCanvas.RootCanvas(params, optStat=0)
        # except:
        self.rootcanvas = RootCanvas(params, optStat=0)


        # Read other parameters
        self.namedata = reader.read_param(params, 'variables', "required")
        self.efficiency = reader.read_param(params, 'efficiency', '1')
        self.multipleHistos = False
        if isinstance(self.namedata, list):
            self.multipleHistos = True
        if self.multipleHistos:
            self.histos = []
            for var in self.namedata:
                aParamsDict = params
                aParamsDict.update({"variables": str(var)})
                self.histos.append(RootHistogram(params, optStat=0))
        else:
            self.histo = RootHistogram(params, optStat=0)
            self.range = reader.read_param(params, 'range', [0., -1.])
            self.eff_func = TF1("eff_func",self.efficiency,self.range[0],self.range[1])
        return True

    def InternalRun(self):
        self.rootcanvas.cd()
        if self.multipleHistos:
            for i, (var, histo) in enumerate(zip(self.namedata, self.histos)):
                histo.Fill(self.data.get(var))
                if i ==0:
                    histo.Draw("hist")
                    histo.SetLineColor(i, len(self.histos))
                else:
                    histo.Draw("sameHist")
                    histo.SetLineColor(i, len(self.histos))
        else:
            self.histo.Fill(self.data.get(self.namedata))
            self.histo.Divide(self.eff_func)
            self.histo.Draw("hist")
        self.rootcanvas.Save()
        return True
