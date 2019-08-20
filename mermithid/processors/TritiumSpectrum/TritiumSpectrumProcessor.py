from mermithid.misc import Constants
from morpho.processors.sampling import RooFitInterfaceProcessor
from morpho.utilities import morphologging, reader

try:
    import ROOT
except ImportError:
    pass

'''
Fit Tritium spectrum processor
Author: M. Guigue
Date: Mar 30 2018
'''
logger = morphologging.getLogger(__name__)

try:
    import PhylloxeraPy
    PhylloxeraPy.loadLibraries(True)
except ImportError:
    logger.warning("Cannot import PhylloxeraPy")
    pass


class TritiumSpectrumProcessor(RooFitInterfaceProcessor):

    def InternalConfigure(self, config_dict):
        '''
        Args:
            null_neutrino_mass (bool): set the neutrino mass to zero during fit
        '''
        super().InternalConfigure(config_dict)
        self.fixed_m_nu = reader.read_param(config_dict, "fixed_m_nu", False)
        self.neutrino_mass = reader.read_param(
            config_dict, "neutrino_mass", 0.)
        self.energy_resolution = reader.read_param(
            config_dict, "energy_resolution", 0.)
        # self.n_events = reader.read_param(config_dict, "n_events", 1200)
        # self.n_bkgd = reader.read_param(config_dict, "n_kbg", 100)
        self.background = reader.read_param(config_dict, "background", 1e-6)
        self.volume = reader.read_param(config_dict, "volume", 1e-6)
        self.density = reader.read_param(config_dict, "density", 1e18)
        self.duration = reader.read_param(config_dict, "duration", 1e18)
        self.KE_min, self.KE_max = reader.read_param(
            config_dict, "paramRange", "required")["KE"]
        self.numberDecays = reader.read_param(config_dict, "number_decays", -1)
        self.poisson_fluctuations = reader.read_param(
            config_dict, "poisson_fluctuations", False)
        return True

    def _GetNEvents_Window(self, KE, spectrum):
        '''
        Calculate the number of decays events generated in the energy window
        '''
        KE = ROOT.RooRealVar("KE_tmp", "KE_tmp", 0,
                             Constants.tritium_endpoint()*2)
        KE.setRange("FullRange", 0, Constants.tritium_endpoint()*2)
        KE.setRange("Window", self.KE_min-self.increase_range,
                    self.KE_max+self.increase_range)

        m_nu = ROOT.RooRealVar("m_nu_tmp", "m_nu_tmp",
                               self.neutrino_mass, -200, 200)
        endpoint = ROOT.RooRealVar("endpoint_tmp", "endpoint_tmp", Constants.tritium_endpoint(),
                                   Constants.tritium_endpoint()-10.,
                                   Constants.tritium_endpoint()+10.)
        # Define the standard tritium spectrum
        spectrum = ROOT.RealTritiumSpectrum(
            "spectrum_tmp", "spectrum_tmp", KE, endpoint, m_nu)

        # Define PdfFactory to add background and smearing
        pdffactory = ROOT.PdfFactory("myPdfFactory_tmp")
        if self.doSmearing:
            meanSmearing = ROOT.RooRealVar(
                "meanSmearing_tmp", "meanSmearing_tmp", 0)
            widthSmearing = ROOT.RooRealVar(
                "widthSmearing_tmp", "widthSmearing_tmp", self.energy_resolution)
            spectrum = pdffactory.GetSmearedPdf(ROOT.RealTritiumSpectrum)(
                "smearedspectrum_tmp", 2, KE, spectrum, meanSmearing, widthSmearing, 1000000)
        fullSpectrumIntegral = spectrum.createIntegral(
            ROOT.RooArgSet(KE), ROOT.RooFit.Range("FullRange"))
        windowSpectrumIntegral = spectrum.createIntegral(
            ROOT.RooArgSet(KE), ROOT.RooFit.Range("Window"))
        ratio = windowSpectrumIntegral.getVal()/fullSpectrumIntegral.getVal()
        logger.debug("Fraction in window [{},{}]: {}".format(
            self.KE_min-self.increase_range, self.KE_max+self.increase_range, ratio))
        return ratio

    def _GetNBackground_Window(self):
        '''
        Calculate the number of background events generated in the energy window
        '''
        return self.background * (self.KE_max - self.KE_min + 2*self.increase_range) * self.duration

    def definePdf(self, wspace):
        '''
        Defines the Pdf that RooFit will sample and add it to the workspace.
        Users should edit this function.
        '''
        logger.debug("Defining pdf")
        if hasattr(self, "energy_resolution") and self.energy_resolution > 0.:
            logger.debug("Will use a smeared spectrum with {} eV energy res.".format(
                self.energy_resolution))
            self.increase_range = 30*self.energy_resolution
            self.doSmearing = True
        else:
            logger.debug("Will use a normal spectrum")
            self.increase_range = 0
            self.doSmearing = False

        # We have to define the range here: the setRange() methods are only useful for the calculating integrals
        # The energy window is increased to accommodate the convolution
        KE = ROOT.RooRealVar("KE", "KE", self.KE_min -
                             self.increase_range, self.KE_max+self.increase_range)
        KE.setBins(1000, "cache")

        KE.setBins(50, "cache")
        m_nu = ROOT.RooRealVar("m_nu", "m_nu", self.neutrino_mass, -1000, 1000)
        endpoint = ROOT.RooRealVar("endpoint", "endpoint", Constants.tritium_endpoint(),
                                   self.KE_min,
                                   self.KE_max)
        # Define the standard tritium spectrum
        spectrum = ROOT.RealTritiumSpectrum(
            "spectrum", "spectrum", KE, endpoint, m_nu)

        # Define PdfFactory to add background and smearing
        pdffactory = ROOT.PdfFactory("myPdfFactory")
        # Smearing of the spectrum
        if self.doSmearing:
            meanSmearing = ROOT.RooRealVar("meanSmearing", "meanSmearing", 0.)
            widthSmearing = ROOT.RooRealVar(
                "widthSmearing", "widthSmearing", self.energy_resolution, 0., 10*self.energy_resolution)
            smearedspectrum = pdffactory.GetSmearedPdf(ROOT.RealTritiumSpectrum)(
                "smearedspectrum", 2, KE, spectrum, meanSmearing, widthSmearing, 100000)

        # Calculate number of events and background
        if self.numberDecays <= 0:
            number_atoms = self.volume*self.density
            total_number_decays = number_atoms*self.duration/Constants.tritium_lifetime()
            if self.doSmearing:
                number_decays_window = total_number_decays * \
                    self._GetNEvents_Window(KE, smearedspectrum)
            else:
                number_decays_window = total_number_decays * \
                    self._GetNEvents_Window(KE, spectrum)
        else:
            number_decays_window = self.numberDecays
        logger.debug("Number decays in window: {}".format(
            number_decays_window))
        self.number_bkgd_window = self._GetNBackground_Window()
        logger.debug("Number bkgd in window: {}".format(
            self.number_bkgd_window))

        # Calculate the number of events to generate:
        # - If the poisson_fluctuations is True, it uses a Poisson process to get the number of events to generate
        #   (the mean being the number of events in the window)
        # - Else use the value calculated
        if self.poisson_fluctuations:
            ran = ROOT.TRandom3()
            self.number_decays_window_to_generate = int(
                ran.Poisson(number_decays_window))
            self.number_bkgd_window_to_generate = int(
                ran.Poisson(self.number_bkgd_window))
        else:
            self.number_decays_window_to_generate = int(number_decays_window)
            self.number_bkgd_window_to_generate = int(self.number_bkgd_window)

        NEvents = ROOT.RooRealVar(
            "NEvents", "NEvents", self.number_decays_window_to_generate, 0., 10*self.number_decays_window_to_generate)
        NBkgd = ROOT.RooRealVar(
            "NBkgd", "NBkgd", self.number_bkgd_window_to_generate, 0., 10*self.number_bkgd_window_to_generate)
        if self.doSmearing:
            pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)(
                "pdf", KE, smearedspectrum, NEvents, NBkgd)
        else:
            pdf = pdffactory.AddBackground(ROOT.RooAbsPdf)(
                "pdf", KE, spectrum, NEvents, NBkgd)

        getattr(wspace, 'import')(KE)
        getattr(wspace, 'import')(pdf)
        return wspace
