import PhylloxeraPy
PhylloxeraPy.loadLibraries(True)
import ROOT

from morpho.utilities import morphologging, reader
logger = morphologging.getLogger(__name__)

from morpho.processors.sampling import RooFitLikelihoodSampler
from mermithid.misc import Constants


class KryptonSpectrumLikelihoodSampler(RooFitLikelihoodSampler):

    def InternalConfigure(self, config_dict):
        super().InternalConfigure(config_dict)
        self.KrMeanValueInit = reader.read_param(config_dict,"init_mean",50E+6)
        return True

    def definePdf(self, wspace):
        '''
        Defines the Pdf that RooFit will sample and add it to the workspace.
        Users should edit this function.
        '''
        logger.debug("Defining pdf")
        # Data variable
        var = wspace.var(self.varName)

        proba = 0.2
        meanJump = 1E+6

        # Variables required by this model
        Kr_mean = ROOT.RooRealVar("Kr_mean", "Kr_mean", self.KrMeanValueInit, self.KrMeanValueInit-10E+6, self.KrMeanValueInit+10E+6)
        Kr_hwhm = ROOT.RooRealVar("Kr_hwhm", "Kr_hwhm", 0.2E+6)#, 0.01E+6, 10E+6)
        # 12.6 eV -> 617133 Hz
        # mean1 = ROOT.RooRealVar("mean1", "mean1", 617133)  # -> 12.6 eV
        # mean2 = ROOT.RooRealVar("mean2", "mean2", 700397)
        mean1 = ROOT.RooRealVar("mean1", "mean1", meanJump)  # -> 12.6 eV
        mean2 = ROOT.RooRealVar("mean2", "mean2", meanJump+1e+5)
        width1 = ROOT.RooRealVar("width1", "width1", 90610)
        width2 = ROOT.RooRealVar("width2", "width2", 612235)
        amplitude1 = ROOT.RooRealVar("amplitude1", "amplitude1", 10)
        amplitude2 = ROOT.RooRealVar("amplitude2", "amplitude2", 45)#,10,100)

        One = ROOT.RooRealVar("one", "one", 1)
        Proba = ROOT.RooRealVar("proba", "proba", proba)
        ProbaSquare = ROOT.RooRealVar("proba2", "proba2", ROOT.TMath.Power(proba, 2))
        ProbaThree = ROOT.RooRealVar("proba3", "proba3", ROOT.TMath.Power(proba, 3))
        ProbaFour = ROOT.RooRealVar("proba4", "proba4", ROOT.TMath.Power(proba, 4))
        # widthSmearing = ROOT.RooRealVar("widthSmearing", "widthSmearing", 1.)
        # NEvents = ROOT.RooRealVar("NEvents", "NEvents", 3e5, 1e3, 5e5)
        # NBkgd = ROOT.RooRealVar("NBkgd", "NBkgd", 1.3409e+03, 5e1, 2e4)
        # NEvents = ROOT.RooRealVar("NEvents","NEvents",1.3212e+04)
        # NBkgd = ROOT.RooRealVar("NBkgd","NBkgd",5.3409e+03)
        # b = ROOT.RooRealVar("background", "background", 0.000001, -1, 1)
        # muWidthSmearing = ROOT.RooRealVar("muWidthSmearing","widthSmearing",1.)
        # sigmaWidthSmearing = ROOT.RooRealVar("sigmaWidthSmearing","sigmaWidthSmearing",1.)

        # Spectrum pdf
        # spectrum = ROOT.RealTritiumSpectrum(
            # "spectrum", "spectrum", var, endpoint, m_nu)

        pdffactory = ROOT.PdfFactory("myPdfFactory")

        energyLossPdf = ROOT.EnergyLossPdf("energyLoss", "Energy Loss pdf", var, mean1, mean2, width1, width2, amplitude1, amplitude2);
        KrLineshape = ROOT.KryptonLine("KrLineShape", "Krypton lineshape", var, Kr_mean, Kr_hwhm)
        distortedKrLineshape1 = pdffactory.GetMultiConvolPdf(ROOT.KryptonLine,ROOT.EnergyLossPdf)("distortedKrLineshape1", var, KrLineshape, energyLossPdf, 1, 1000000)
        distortedKrLineshape2 = pdffactory.GetMultiConvolPdf(ROOT.KryptonLine,ROOT.EnergyLossPdf)("distortedKrLineshape2", var, KrLineshape, energyLossPdf, 2, 1000000)
        distortedKrLineshape3 = pdffactory.GetMultiConvolPdf(ROOT.KryptonLine,ROOT.EnergyLossPdf)("distortedKrLineshape3", var, KrLineshape, energyLossPdf, 3, 1000000)
        distortedKrLineshape4 = pdffactory.GetMultiConvolPdf(ROOT.KryptonLine,ROOT.EnergyLossPdf)("distortedKrLineshape4", var, KrLineshape, energyLossPdf, 4, 1000000)

        pdf = ROOT.RooAddPdf("pdf", "pdf", ROOT.RooArgList(KrLineshape, distortedKrLineshape1, distortedKrLineshape2, distortedKrLineshape3, distortedKrLineshape4), ROOT.RooArgList(One, Proba, ProbaSquare, ProbaThree, ProbaFour))
    
        can = ROOT.TCanvas("can","can",600,400)
        frame = var.frame()
        # dataset.plotOn(frame)
        pdf.plotOn(frame)
        frame.Draw()
        can.SaveAs("plots/pdf.pdf")

    # Background addition
        # background = ROOT.RooUniform(
            # "background", "background", ROOT.RooArgSet(var))

        # PDF of the model:
        # this should have "pdf" as name
        # pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)("pdf",var,spectrum,NEvents,NBkgd)
        # if "smearing" in self.options and self.options["smearing"]:
        #             # Define PdfFactory to add background and smearing

        #     # Spectrum smearing
        #     gauss = ROOT.RooGaussian(
        #         "gauss", "gauss", var, meanSmearing, widthSmearing)
        #     # priorSmearing = ROOT.RooGaussian("priorSmearing","priorSmearing",widthSmearing,ROOT.RooFit.RooConst(1.),ROOT.RooFit.RooConst(1.))
        #     smearedspectrum = pdffactory.GetSmearedPdf(ROOT.RealTritiumSpectrum)(
        #         "smearedspectrum", 2, var, spectrum, meanSmearing, widthSmearing, 100000)

        #     pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)(
        #         "pdf", var, smearedspectrum, NEvents, NBkgd)
        # else:
        #     pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)(
        #         "pdf", var, spectrum, NEvents, NBkgd)

        # pdf = ROOT.RooProdPdf("pdf","pdf",total_spectrum,priorSmearing)

        # pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)("pdf",var,smearedspectrum,NEvents,NBkgd)

        # can = ROOT.TCanvas("can","can",600,400)
        # frame = var.frame()
        # totalSpectrum.plotOn(frame)
        # frame.Draw()
        # can.SaveAs("plots/model.pdf")

        # Save pdf: this will save all required variables and functions
        getattr(wspace, 'import')(pdf)
        return wspace
