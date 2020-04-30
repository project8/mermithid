'''
Fit Tritium spectrum
Author: C. Claessens
Date: Aug 07 2019
'''
try:
    import ROOT
    from ROOT import TMath
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




class DistortedTritiumSpectrumLikelihoodSampler(RooFitInterfaceProcessor):
    """
    Use to generate data following the tritium spectrum shape multiplied by a polynomial efficiency
    """

    def InternalConfigure(self,config_dict):
        '''
        Args:
            - null_neutrino_mass (bool): set the neutrino mass to zero during fit
            - background: background rate in 1/eV/s
            - event_rate: tritium events in 1/s
            - duration:  duration of data taking in s
            - options: Dicitonary (for example: {"snr_efficiency": True, "channel_efficiency":False, "smearing": False})
                determines which efficiencies are multiplied and whether spectrum is smeared
            - energy_or_frequency: determines whether generated data is in energy or frequency domain
            - KEmin: minimum energy in eV. Only used if domain is energy.
            - KEmax: maximum energy in eV. Only used if domain is energy.
            - frequency_window: frequency range around central frequency in Hz. Only used if domain in frequency.
            - energy_resolution: width of Gaussian that will be used to smeare spectrum. Only used if domain is energy.
            - frequency_resolution: width of Gaussian that will be used to smeare spectrum. Only used if domain is frequency.
            - B_field_strength: used to translate energies to frequencies. Required for efficiency and frequency domain data.
            - snr_efficiency_coefficiency: polynomial coefficients for global (channel independent) efficiency.
            - channel_efficiency: coefficiency for filter function used to calcualte channel efficiency.
                        function is: "c4 * TMath::Sqrt(1/(1 + 1*TMath::Power((@0*TMath::Power(10, -6)-cf)/c0, c1)))* TMath::Sqrt(1/(1 + TMath::Power((@0*TMath::Power(10, -6)-cf)/c2, c3)))
                        cf is in MHz
            - channel_central_frequency: central frequency needed to calculate channel efficiency. Only used if channel efficiency is applied.
            - mixing_frequency: frequency to substract from global frequencies.

        '''
        super().InternalConfigure(config_dict)
        self.null_m_nu = reader.read_param(config_dict,"null_neutrino_mass",False)
        self.background = reader.read_param(config_dict, "background", 1e-9)
        self.event_rate = reader.read_param(config_dict, "event_rate", 1.)
        self.duration = reader.read_param(config_dict, "duration", 24*3600)
        self.energy_or_frequency = reader.read_param(config_dict, "energy_or_frequency", "frequency")
        self.KEmin = reader.read_param(config_dict, "KEmin", 18.6e3-1e3)
        self.KEmax = reader.read_param(config_dict, "KEmax", 18.6e3+1e3)
        self.Fwindow = reader.read_param(config_dict, "frequency_window", [-50e6, 50e6])
        self.energy_resolution = reader.read_param(config_dict, "energy_resolution", 0)
        self.frequency_resolution = reader.read_param(config_dict, "frequency_resolution", 0)
        self.B = reader.read_param(config_dict, "B_field_strength", 0.95777194923080811)
        self.snr_eff_coeff = reader.read_param(config_dict, "snr_efficiency_coefficients", [0.1, 0, 0, 0, 0, 0])
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
            raise ValueError("no valid energy_or_frequency")

        self.N_signal_events = self.event_rate*self.duration
        self.N_background_events = self.background*self.duration*(self.KEmax-self.KEmin)


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
        NEvents = ROOT.RooRealVar("NEvents","NEvents",self.N_signal_events,1,5e6)
        NBkgd = ROOT.RooRealVar("NBkgd","NBkgd",self.N_background_events,0,5e6)



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
            mif_freq = ROOT.RooRealVar("mf", "mf", self.mix_frequency)
            p0 = ROOT.RooRealVar("p0", "p0", self.snr_eff_coeff[0])
            p1 = ROOT.RooRealVar("p1", "p1", self.snr_eff_coeff[1])
            p2 = ROOT.RooRealVar("p2", "p2", self.snr_eff_coeff[2])
            p3 = ROOT.RooRealVar("p3", "p3", self.snr_eff_coeff[3])
            p4 = ROOT.RooRealVar("p4", "p4", self.snr_eff_coeff[4])
            p5 = ROOT.RooRealVar("p5", "p5", self.snr_eff_coeff[5])
            eff_coeff = ROOT.RooArgList(var, mif_freq, p0, p1, p2, p3, p4, p5)

            effFunc = ROOT.RooFormulaVar("efficiency", "efficiency", "p0 + p1*TMath::Power(@0-mf,1) + p2*TMath::Power(@0-mf,2) + p3*TMath::Power(@0-mf,3) + p4*TMath::Power(@0-mf,4) + p5*TMath::Power(@0-mf,5)", eff_coeff)

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



        # Save pdf: this will save all required variables and functions

        getattr(wspace,'import')(pdf)
        return wspace

