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
from ROOT import TMath



class TritiumSpectrumLikelihoodSampler(RooFitInterfaceProcessor):

    def InternalConfigure(self,config_dict):
        '''
        Args:
            null_neutrino_mass (bool): set the neutrino mass to zero during fit
        '''
        super().InternalConfigure(config_dict)
        self.null_m_nu = reader.read_param(config_dict,"null_neutrino_mass",False)


        self.KEmin, self.KEmax = reader.read_param(config_dict, "energy_window", [Constants.tritium_endpoint()-1e3, Constants.tritium_endpoint()+1e3])
        self.Fwindow = reader.read_param(config_dict, "frequency_window", [-50e6, 50e6])
        self.energy_resolution = reader.read_param(config_dict, "energy_resolution", 0)
        self.frequency_resolution = reader.read_param(config_dict, "frequency_resolution", 0)
        self.energy_or_frequency = reader.read_param(config_dict, "energy_or_frequency", "frequency")
        self.B = reader.read_param(config_dict, "B-field-strength", 0.95777194923080811)
        self.snr_eff_coeff = reader.read_param(config_dict, "snr_efficiency_coefficients", [0.1, 0, 0, 0])
        self.channel_eff_coeff = reader.read_param(config_dict, "channel_efficiency_coefficients", [24587.645303008387, 7645.8567999493698, 24507.145055859062, -11581.288750763715, 0.98587787287591955])
        self.channel_cf = reader.read_param(config_dict, "channel_central_frequency", 1000e6)
        self.mix_frequency = reader.read_param(config_dict, "mixing_frequency", 24.5e9)
        self.Fmin, self.Fmax = [self.mix_frequency + self.channel_cf + self.Fwindow[0] , self.mix_frequency + self.channel_cf + self.Fwindow[1]]
        return True


    def definePdf(self,wspace):
        '''
        Defines the Pdf that RooFit will sample and add it to the workspace.
        Users should edit this function.
        '''
        logger.debug("Defining pdf")
        if self.energy_or_frequency == "energy":
            self.resolution = self.energy_resolution
            self.increase_range = 10*self.energy_resolution
            var = ROOT.RooRealVar(self.varName, self.varName, self.KEmin -
                                 self.increase_range, self.KEmax+self.increase_range)
        elif self.energy_or_frequency == "frequency":
            self.resolution = self.frequency_resolution
            self.increase_range = 10*self.frequency_resolution
            var = ROOT.RooRealVar(self.varName, self.varName, self.Fmin -
                                 self.increase_range, self.Fmax+self.increase_range)
            logger.debug("min frequency: {}Hz, max frequency: {}Hz".format(self.Fmin, self.Fmax))
        else:
            raise Exception("no valid energy_or_frequency")


        # Variables required by this model
        if self.null_m_nu:
            m_nu = ROOT.RooRealVar("m_nu","m_nu",0.)
        else:
            m_nu = ROOT.RooRealVar("m_nu","m_nu",0.,-300.,300.)
        endpoint = ROOT.RooRealVar("endpoint", "endpoint", Constants.tritium_endpoint(),
                                                           Constants.tritium_endpoint()-10.,
                                                           Constants.tritium_endpoint()+10.)
        meanSmearing = ROOT.RooRealVar("meanSmearing","meanSmearing",0.)
        widthSmearing = ROOT.RooRealVar("widthSmearing","widthSmearing",self.resolution)
        NEvents = ROOT.RooRealVar("NEvents","NEvents",3e5,1e3,5e5)
        NBkgd = ROOT.RooRealVar("NBkgd","NBkgd",1.3409e+03,5e1,2e4)
        background = ROOT.RooRealVar("background","background",0.000001,-1,1)



        # Spectrum pdf
        if self.energy_or_frequency == "energy":
            shapePdf = ROOT.RealTritiumSpectrum("shapePdf","shapePdf",var,endpoint,m_nu)
            logger.debug("Defined energy spectrum")
        else:
            B = ROOT.RooRealVar("B", "B", self.B)
            shapePdf = ROOT.RealTritiumFrequencySpectrum("shapePdf","shapePdf",var, B, endpoint,m_nu)
            #spectrum.SetEfficiencyCoefficients(eff_coeff)
            logger.debug("Defined frequency spectrum")



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
            smearedspectrum = pdffactory.GetSmearedPdf(ROOT.RealTritiumFrequencySpectrum)("smearedspectrum", 2, var, shapePdf, meanSmearing, widthSmearing,100000)
            pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)("pdf",var,smearedspectrum,NEvents,NBkgd)


        if "snr_efficiency" in self.options and self.options["snr_efficiency"]:
            logger.info("Appyling SNR efficiency")

            # Spectrum distortion
            p0 = ROOT.RooRealVar("p0", "p0", self.snr_eff_coeff[0])
            p1 = ROOT.RooRealVar("p1", "p1", self.snr_eff_coeff[1])
            p2 = ROOT.RooRealVar("p2", "p2", self.snr_eff_coeff[2])
            p3 = ROOT.RooRealVar("p3", "p3", self.snr_eff_coeff[3])
            eff_coeff = ROOT.RooArgList(var, p0, p1, p2, p3)

            effFunc = ROOT.RooFormulaVar("efficiency", "efficiency", "p0 + p1*TMath::Power(@0,1) + p2*TMath::Power(@0,2) + p3*TMath::Power(@0,3)", eff_coeff)

            if "smearing" in self.options and self.options["smearing"]:
                distortedSpectrum = ROOT.RooEffProd("distortedSpectrum", "distortedSpectrum", smearedspectrum, effFunc)
            else:
                distortedSpectrum = ROOT.RooEffProd("distortedSpectrum", "distortedSpectrum", shapePdf, effFunc)


            if "channel_efficiency" in self.options and self.options["channel_efficiency"]:
                logger.info("Applying channel efficiency")
                c0 = ROOT.RooRealVar("c0", "c0", self.channel_eff_coeff[0])
                c1 = ROOT.RooRealVar("c1", "c1", self.channel_eff_coeff[1])
                c2 = ROOT.RooRealVar("c2", "c2", self.channel_eff_coeff[2])
                c3 = ROOT.RooRealVar("c3", "c3", self.channel_eff_coeff[3])
                c4 = ROOT.RooRealVar("c4", "c4", self.channel_eff_coeff[4])
                cf = ROOT.RooRealVar("cf", "cf", self.channel_cf*1e-6-50)
                channel_eff_coeff = ROOT.RooArgList(var, c0, c1, c2, c3, c4, cf)

                # filter = args[4] * np.sqrt(1/(1 + 1*(x/args[0])**args[1]))* np.sqrt(1/(1 + 1*(x/args[2])**args[3]))
                # x is Frequency in channel in MHz: (@0*TMath::Power(10, -6)-cf)
                channelEffFunc = ROOT.RooFormulaVar("channel_efficiency", "channel_efficiency", "c4 * TMath::Sqrt(1/(1 + 1*TMath::Power((@0*TMath::Power(10, -6)-cf)/c0, c1)))* TMath::Sqrt(1/(1 + TMath::Power((@0*TMath::Power(10, -6)-cf)/c2, c3)))", channel_eff_coeff)
                superDistortedSpectrum = ROOT.RooEffProd("superDistortedSpectrum", "superDistortedSpectrum", distortedSpectrum, channelEffFunc)
                pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)("pdf",var,superDistortedSpectrum,NEvents,NBkgd)

            else:
                pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)("pdf",var,distortedSpectrum,NEvents,NBkgd)
        else:
            pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)("pdf",var,shapePdf,NEvents,NBkgd)



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
    def Frequency(self, E, B=None):
        print("Calculate frequency from energy")
        if B == None:
            B = self.B
        emass = Constants.m_electron()
        emass_kg = Constants.m_electron() * Constants.e()/Constants.c()**2
        gamma = E/(emass)+1
        #return e*B*c**2/(E*e+E0*e)*(1+1/np.tan(Theta)**2/2)/(2*m.pi)
        return (Constants.e()*B)/(2.0*3.141592653589793*emass_kg) * 1/gamma

    def Energy(self, F, B=None, Theta=None):
        #print(type(F))
        if B==None:
            B = self.B
        emass_kg = Constants.m_electron()*Constants.e()/(Constants.c()**2)
        if isinstance(F, list):
            gamma = [(Constants.e()*B)/(2.0*TMath.Pi()*emass_kg) * 1/(f) for f in F]
            return [(g -1)*Constants.m_electron() for g in gamma]
        else:
            gamma = (Constants.e()*B)/(2.0*TMath.Pi()*emass_kg) * 1/(f)
            return (gamma -1)*Constants.m_electron()
        #shallow trap 0.959012745568
        #deep trap 0.95777194923080811
