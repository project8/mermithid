'''
Fit Tritium spectrum
Author: M. Guigue
Date: Mar 30 2018
'''
try:
    import ROOT
except ImportError:
    pass

from morpho.utilities import morphologging, reader
logger = morphologging.getLogger(__name__)

try:
    import PhylloxeraPy
    PhylloxeraPy.loadLibraries(True)
except ImportError:
    logger.warning("Cannot import PhylloxeraPy")
    pass

from morpho.processors.sampling import RooFitInterfaceProcessor
from mermithid.misc import Constants

class TritiumSpectrumLikelihoodSampler(RooFitInterfaceProcessor):

    def InternalConfigure(self,config_dict):
        '''
        Args:
            null_neutrino_mass (bool): set the neutrino mass to zero during fit
        '''
        super().InternalConfigure(config_dict)
        self.null_m_nu = reader.read_param(config_dict,"null_neutrino_mass",False)


        self.KEmin, self.KEmax = reader.read_param(config_dict, "energy_window", [Constants.tritium_endpoint()-1e3, Constants.tritium_endpoint()+1e3])
        self.Fmin, self.Fmax = reader.read_param(config_dict, "frequency_window", [self.Frequency(Constants.tritium_endpoint()+1e3), self.Frequency(Constants.tritium_endpoint()-1e3)])
        self.energy_resolution = reader.read_param(config_dict, "energy_resolution", 0)
        self.frequency_resolution = reader.read_param(config_dict, "frequency_resolution", 0)
        self.energy_or_frequency = reader.read_param(config_dict, "energy_or_frequency", "frequency")
        self.B = reader.read_param(config_dict, "B-field-strength", 0.95777194923080811)
        return True


    def definePdf(self,wspace):
        '''
        Defines the Pdf that RooFit will sample and add it to the workspace.
        Users should edit this function.
        '''
        print('hello')
        logger.debug("Defining pdf")
        #var = wspace.var(self.varName)
        if self.energy_or_frequency == "energy":
            self.increase_range = 10*self.energy_resolution
            var = ROOT.RooRealVar(self.varName, self.varName, self.KEmin -
                                 self.increase_range, self.KEmax+self.increase_range)
        elif self.energy_or_frequency == "frequency":
            self.increase_range = 10*self.frequency_resolution
            var = ROOT.RooRealVar(self.varName, self.varName, self.Fmin -
                                 self.increase_range, self.Fmax+self.increase_range)
            print(self.Fmin, self.Fmax)
        else:
            raise Exception("no valid energy_or_frequency")

        print('var', var)
        print('var defined')

        # Variables required by this model
        if self.null_m_nu:
            m_nu = ROOT.RooRealVar("m_nu","m_nu",0.)
        else:
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
        print('start with real tritium spectrum')
        if self.energy_or_frequency == "energy":
            spectrum = ROOT.RealTritiumSpectrum("spectrum","spectrum",var,endpoint,m_nu)
        else:
            B = ROOT.RooRealVar("B", "B", self.B)
            p0 = ROOT.RooRealVar("p0", "p0", 1.)
            p1 = ROOT.RooRealVar("p1", "p1", 0.)
            eff_coeff = ROOT.RooArgList(p0, p1, var)
            eff_coeff.Print()
            efficiency = ROOT.RooFormulaVar("efficiency", "efficiency", "@0 + @1*TMath::Power(@2,1)", eff_coeff)
            efficiency.Print()
            print(efficiency.evaluate())

            spectrum = ROOT.RealTritiumFrequencySpectrum("spectrum","spectrum",var, B, endpoint,m_nu)
            spectrum.SetEfficiencyCoefficients(p0, p1, var)
            print("Generated frequency spectrum")


        pdffactory = ROOT.PdfFactory("myPdfFactory")


        # Background addition
        background = ROOT.RooUniform("background","background",ROOT.RooArgSet(var))

        # PDF of the model:
        # this should have "pdf" as name
        # pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)("pdf",var,spectrum,NEvents,NBkgd)
        if "smearing" in self.options and self.options["smearing"]:
            # Define PdfFactory to add background and smearing
            print("Smearing spectrum")

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

    """
    def _Generator(self):
        '''
        Generate the data by sampling the pdf defined in the workspace
        '''
        print("my own generator")
        wspace = ROOT.RooWorkspace()
        wspace = self.definePdf(wspace)
        logger.debug("Workspace content:")
        wspace.Print()
        wspace = self._FixParams(wspace)
        print("get pdf from wspace")
        pdf = wspace.pdf("pdf")
        print("getting params of interest", self.paramOfInterestNames)
        paramOfInterest = self._getArgSet(wspace, self.paramOfInterestNames)
        print("generate data")
        data = pdf.generate(paramOfInterest, self.iter)
        data.Print()

        self.data = {}
        for name in self.paramOfInterestNames:
            self.data.update({name: []})

        for i in range(0, data.numEntries()):
            for item in self.data:
                self.data[item].append(
                    data.get(i).getRealValue(item))
        self.data.update({"is_sample": [1]*(self.iter)})
        return True

    def _getArgSet(self, wspace, listNames):
        argSet = ROOT.RooArgSet()
        for name in listNames:
            print(name)
            argSet.add(wspace.var(name))
        return argSet
        """
    def Frequency(self, E, B=None, Theta=None):
        print("Calculate frequency from energy")
        if Theta==None:
            Theta=3.141592653589793/2
        e = Constants.e()
        c = Constants.c()
        if B == None:
            B = 0.95777194923080811
        emass = Constants.m_electron()
        emass_kg = Constants.m_electron() * Constants.e()/Constants.c()**2
        print(E, emass)
        gamma = E/(emass)+1
        #return e*B*c**2/(E*e+E0*e)*(1+1/np.tan(Theta)**2/2)/(2*m.pi)
        return (Constants.e()*B)/(2.0*3.141592653589793*emass_kg) * 1/gamma