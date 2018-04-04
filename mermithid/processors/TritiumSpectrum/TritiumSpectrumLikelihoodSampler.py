import PhylloxeraPy
PhylloxeraPy.loadLibraries(False)
import ROOT

from morpho.utilities import morphologging, reader
logger = morphologging.getLogger(__name__)

from morpho.processors.sampling import RooFitLikelihoodSampler
import mermithid.misc.Constants

class TritiumSpectrumLikelihoodSampler(RooFitLikelihoodSampler):

    def definePdf(self,wspace):
        '''
        Defines the Pdf that RooFit will sample and add it to the workspace.
        Users should edit this function.
        '''
        logger.debug("Defining pdf")
        var = wspace.var(self.varName)

        # Variables required by this model
        m_nu = ROOT.RooRealVar("m_nu","m_nu",0.,-300.,300.) 
        endpoint = ROOT.RooRealVar("endpoint", "endpoint", Constants.tritium_endpoint(),
                                                           Constants.tritium_endpoint()-10.,
                                                           Constants.tritium_endpoint()+10.)
        meanSmearing = ROOT.RooRealVar("meanSmearing","meanSmearing",0.)
        widthSmearing = ROOT.RooRealVar("widthSmearing","widthSmearing",1.)
        NEvents = ROOT.RooRealVar("NEvents","NEvents",3e5,1e3,5e5)
        NBkgd = ROOT.RooRealVar("NBkgd","NBkgd",1.3409e+03,5e1,2e4)
        # NEvents = ROOT.RooRealVar("NEvents","NEvents",1.3212e+04)
        # NBkgd = ROOT.RooRealVar("NBkgd","NBkgd",5.3409e+03)
        b = ROOT.RooRealVar("background","background",0.000001,-1,1)
        # muWidthSmearing = ROOT.RooRealVar("muWidthSmearing","widthSmearing",1.)
        # sigmaWidthSmearing = ROOT.RooRealVar("sigmaWidthSmearing","sigmaWidthSmearing",1.)


        # Spectrum pdf
        spectrum = ROOT.RealTritiumSpectrum("spectrum","spectrum",var,endpoint,m_nu)

        pdffactory = ROOT.PdfFactory("myPdfFactory")

  
        # Background addition
        background = ROOT.RooUniform("background","background",ROOT.RooArgSet(var))

        # PDF of the model:
        # this should have "pdf" as name 
        # pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)("pdf",var,spectrum,NEvents,NBkgd)
        if "smearing" in self.options and self.options["smearing"]:
                    # Define PdfFactory to add background and smearing

            # Spectrum smearing
            gauss = ROOT.RooGaussian("gauss","gauss",var,meanSmearing,widthSmearing)
            # priorSmearing = ROOT.RooGaussian("priorSmearing","priorSmearing",widthSmearing,ROOT.RooFit.RooConst(1.),ROOT.RooFit.RooConst(1.))
            smearedspectrum = pdffactory.GetSmearedPdf(ROOT.RealTritiumSpectrum)("smearedspectrum", 2, var, spectrum, meanSmearing, widthSmearing,100000)
        
            pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)("pdf",var,smearedspectrum,NEvents,NBkgd)
        else:
            pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)("pdf",var,spectrum,NEvents,NBkgd)
            
        # pdf = ROOT.RooProdPdf("pdf","pdf",total_spectrum,priorSmearing)


        # pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)("pdf",var,smearedspectrum,NEvents,NBkgd)
        
        # can = ROOT.TCanvas("can","can",600,400)
        # frame = var.frame()
        # totalSpectrum.plotOn(frame)
        # frame.Draw()
        # can.SaveAs("plots/model.pdf") 

        # Save pdf: this will save all required variables and functions
        getattr(wspace,'import')(pdf)
        return wspace
