'''
Generate Tritium fake spectrum
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

from morpho.processors import BaseProcessor
from mermithid.misc import Constants


increase_range = 10.  # energy increase required for the convolution product to work


class TritiumSpectrumGenerator(BaseProcessor):
    '''
    Generate a smeared tritium spectrum.
    It accounts for the background to estimate the number of events to generate
    based on the exposure time.
    '''

    def InternalConfigure(self, config_dict={}):
        '''
        Required class attributes:
        - volume [m3]
        - density [1/m3]
        - experiment duration [s]
        - neutrino mass [eV]
        - energy window [KEmin,KEmax]
        - background [counts/eV/s]
        - energy resolution [eV]
        '''

        self.KEmin, self.KEmax = reader.read_param(config_dict, "energy_window", [
                                                   Constants.tritium_endpoint()-1e3, Constants.tritium_endpoint()+1e3])
        self.volume = reader.read_param(config_dict, "volume", 1e-6)
        self.density = reader.read_param(config_dict, "density", 1e18)
        self.duration = reader.read_param(config_dict, "duration", 1e18)
        self.neutrinomass = reader.read_param(
            config_dict, "neutrino_mass", 1e18)
        self.background = reader.read_param(config_dict, "background", 1e-6)
        self.poisson_fluctuations = reader.read_param(
            config_dict, "poisson_fluctuations", False)
        self.energy_resolution = reader.read_param(
            config_dict, "energy_resolution", 0)
        self.numberDecays = reader.read_param(config_dict, "number_decays", -1)
        return True

    def _GetNEvents_Window(self, KE, spectrum):
        '''
        Calculate the number of decays events generated in the energy window
        '''
        KE = ROOT.RooRealVar("KE_tmp", "KE_tmp", 0,
                             Constants.tritium_endpoint()*2)
        KE.setRange("FullRange", 0, Constants.tritium_endpoint()*2)
        KE.setRange("Window", self.KEmin-self.increase_range,
                    self.KEmax+self.increase_range)

        m_nu = ROOT.RooRealVar("m_nu_tmp", "m_nu_tmp",
                               self.neutrinomass, -200, 200)
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
            smearedspectrum = pdffactory.GetSmearedPdf[ROOT.RealTritiumSpectrum](
                "smearedspectrum_tmp", 2, KE, spectrum, meanSmearing, widthSmearing, 1000000)
        fullSpectrumIntegral = spectrum.createIntegral(
            ROOT.RooArgSet(KE), ROOT.RooFit.Range("FullRange"))
        windowSpectrumIntegral = spectrum.createIntegral(
            ROOT.RooArgSet(KE), ROOT.RooFit.Range("Window"))
        ratio = windowSpectrumIntegral.getVal()/fullSpectrumIntegral.getVal()
        logger.debug("Fraction in window [{},{}]: {}".format(
            self.KEmin-self.increase_range, self.KEmax+self.increase_range, ratio))
        return ratio

    def _GetNBackground_Window(self):
        '''
        Calculate the number of background events generated in the energy window
        '''
        return self.background * (self.KEmax - self.KEmin + 2*self.increase_range) * self.duration

    def _PrepareWorkspace(self):
        # for key in config_dict:
        #     setattr(self, key, config_dict[key])
        if hasattr(self, "energy_resolution") and self.energy_resolution > 0.:
            logger.debug("Will use a smeared spectrum with {} eV energy res.".format(
                self.energy_resolution))
            self.increase_range = 10*self.energy_resolution
            self.doSmearing = True
        else:
            logger.debug("Will use a normal spectrum")
            self.increase_range = 0
            self.doSmearing = False

        # We have to define the range here: the setRange() methods are only useful for the calculating integrals
        # The energy window is increased to accommodate the convolution
        KE = ROOT.RooRealVar("KE", "KE", self.KEmin -
                             self.increase_range, self.KEmax+self.increase_range)
        # logger.debug(self.KEmin-self.increase_range,self.KEmax+self.increase_range)
        # KE.setRange("FullRange",0,Constants.tritium_endpoint()*2)
        # KE.setRange("Window",self.KEmin-10,self.KEmax+10)
        m_nu = ROOT.RooRealVar("m_nu", "m_nu", self.neutrinomass, -20, 20)
        endpoint = ROOT.RooRealVar("endpoint", "endpoint", Constants.tritium_endpoint(),
                                   Constants.tritium_endpoint()-10.,
                                   Constants.tritium_endpoint()+10.)
        b = ROOT.RooRealVar("background", "background", -
                            10*self.background, 10*self.background)

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
            # KE.setBins(100000, "cache")
            smearedspectrum = pdffactory.GetSmearedPdf[ROOT.RealTritiumSpectrum](
                "smearedspectrum", 2, KE, spectrum, meanSmearing, widthSmearing, 100000)

        # Background
        background = ROOT.RooUniform(
            "background", "background", ROOT.RooArgSet(KE))

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
            totalSpectrum = pdffactory.AddBackground[ROOT.RooAbsPdf](
                "totalSpectrum", KE, smearedspectrum, NEvents, NBkgd)
        else:
            totalSpectrum = pdffactory.AddBackground[ROOT.RooAbsPdf](
                "totalSpectrum", KE, spectrum, NEvents, NBkgd)

        # Save things in a Workspace
        self.workspace = ROOT.RooWorkspace()
        getattr(self.workspace, 'import')(totalSpectrum)
        getattr(self.workspace, 'import')(background)
        self.workspace.Print()

    def InternalRun(self):
        self._PrepareWorkspace()
        self.results = self._GenerateData()
        return True

    def _GenerateData(self):

        logger.debug("Generate data")
        KE = self.workspace.var("KE")
        KE.setRange("window", self.KEmin, self.KEmax)
        background = self.workspace.pdf("background")
        totalSpectrum = self.workspace.pdf("totalSpectrum")
        totalEvents = self.number_decays_window_to_generate + \
            self.number_bkgd_window_to_generate
        dataLarge = totalSpectrum.generate(ROOT.RooArgSet(
            KE), totalEvents, ROOT.RooFit.Range("window"))
        data = dataLarge.reduce(ROOT.RooFit.CutRange("window"))
        dataList = []
        for i in range(data.numEntries()):
            dataList.append(data.get(i).getRealValue("KE"))
        # Have to delete the workspace to precent some nasty errors at the end...
        del self.workspace
        return {"KE": dataList}
