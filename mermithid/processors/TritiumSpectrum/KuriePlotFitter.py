'''                                                                                                                                     
Generate a Kurie plot from energy data
Author: M. Guigue
Date: Mar 30 2018
'''

from __future__ import absolute_import

import json
import os

import ROOT

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
        self.fitFunction = TF1("kurieFit", "[0]*([1]-x)*(x<[1]) + [2]", centralList[0], centralList[-1], 3)
        self.fitFunction.SetParameters(kurieList[0]/(18600-centralList[0]), 18600, kurieList[-1])
        rootResults = kurieGraph.Fit(self.fitFunction, 'MERS')
        resultsDict = {
            "amplitude": rootResults.Parameter(0),
            "err_amplitude": rootResults.ParError(0),
            "endpoint": rootResults.Parameter(1),
            "err_endpoint": rootResults.ParError(1),
            "background": rootResults.Parameter(2),
            "err_background": rootResults.ParError(2),
            "chi2": rootResults.Chi2(),
            "ndf": rootResults.Ndf(),
            "p-value": rootResults.Prob()
        } 
        return resultsDict

    def _PoissonFitKuriePlot(self, centralList, akurieList):
        '''
        Fit Kurie plot assuming that each bin follows a new pdf 
        f(y,lambda)=2/sqrt(2pi)*exp(y^2(1-ln(y^2/lambda^2)-lambda)
        with y=sqrt(x) where x follows Poisson(lambda) 
        with lambda = gamma^2 
        '''
        kurieList = [val*1e6 for val in akurieList]
        KE = ROOT.RooRealVar("KE","KE", min(centralList), max(centralList))
        print(min(centralList),max(centralList))
        #y = ROOT.RooRealVar("y","y",min(kurieList),max(kurieList))
        y = ROOT.RooRealVar("y","y",0,100)
        amplitude = ROOT.RooRealVar("amp","amp",0.012,0, 1)
        endpoint = ROOT.RooRealVar("endpoint","endpoint",18600, 16600, 19600)
        bkg = ROOT.RooRealVar("bkg","bkg",0.2,0,10)
        kurieShape = ROOT.RooFormulaVar("gamma", "amp*(endpoint-KE)*(endpoint>KE)+bkg",ROOT.RooArgList(amplitude,endpoint,bkg,KE))
        

        res = ROOT.RooFormulaVar("res","gamma-y",ROOT.RooArgList(y,kurieShape))
        pdf = ROOT.RooGenericPdf("pdf","pdf","exp(-gamma*gamma)*pow(gamma*gamma,y*y)/tgamma(y*y)",ROOT.RooArgList(y,kurieShape)) 
        null = ROOT.RooRealVar("null","null",0)
        one = ROOT.RooRealVar("one","one",1)
        #pdf = ROOT.RooGaussian("pdf","pdf", res, null,one)
        
        data = ROOT.RooDataSet("kuriePlot", "kuriePlot", ROOT.RooArgSet(KE, y))
        for ke, val in zip(centralList, kurieList):
            #if val>0:
                KE.setVal(ke)
                y.setVal(val)
                data.add(ROOT.RooArgSet(KE, y))
       
        data.Print() 
        #result = pdf.fitTo(data, ROOT.RooFit.Save(), ROOT.RooFit.NumCPU(3))
        #result = kurieShape.chi2FitTo(data,ROOT.RooFit.YVar(y), ROOT.RooFit.Save())
        #result.Print()
        #logger.debug("Covariance matrix:")
        #result.covarianceMatrix().Print()

        frame1 = KE.frame()
        frame1.SetMinimum(0.)
        frame1.SetMaximum(max(kurieList))
        data.plotOnXY(frame1,ROOT.RooFit.YVar(y))
        kurieShape.plotOn(frame1)
        c = ROOT.TCanvas("c","c",600,400)
        frame1.Draw()
        c.SaveAs("test.pdf")

        frame2 = y.frame()
        #pdf = ROOT.RooPoisson("pdf","pdf",y,one)
        #pdf = ROOT.RooGenericPdf("pdf","pdf","exp(-one)*pow(one,y)/tgamma(y)",ROOT.RooArgList(y,one)) 
        pdf.plotOn(frame2)
        frame2.Draw()
        c.SaveAs("test2.pdf")

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
        self.result = self._FitKuriePlot(centralValueList, kurieList, errorList)
        #self._PoissonFitKuriePlot(centralValueList, kurieList)
        self.fitFunction.Draw("sameL")
        self.rootcanvas.Save()
        return True
    

