'''
Class calculating neutrino mass sensitivities based on analytic formulas from CDR.
Author: R. Reimann, C. Claessens
Date:11/17/2020

The statistical method and formulars are described in 
CDR (CRES design report, Section 1.3) https://www.overleaf.com/project/5b9314afc673d862fa923d53.
'''
import numpy as np
import configparser
from numpy import pi

# Numericalunits is a package to handle units and some natural constants
# natural constants
from numericalunits import e, me, c0, eps0, kB, hbar
from numericalunits import meV, eV, keV, MeV, cm, m, ns, s, Hz, kHz, MHz, GHz, amu
from numericalunits import nT, uT, mT, T, mK, K,  C, F, g, W
from numericalunits import hour, year, day
from numericalunits import mu0, NA, kB, hbar, me, c0, e, eps0, hPlanck

T0 = -273.15*K

tritium_livetime = 5.605e8*s
tritium_mass_atomic = 3.016* amu *c0**2
tritium_electron_crosssection_atomic = 1.1e-22*m**2
tritium_endpoint_atomic = 18563.251*eV
last_1ev_fraction_atomic = 2.067914e-13/eV**3

tritium_mass_molecular = 6.032099 * amu *c0**2
tritium_electron_crosssection_molecular = 3.487*1e-22*m**2
tritium_endpoint_molecular = 18573.24*eV
last_1ev_fraction_molecular = 1.67364e-13/eV**3

ground_state_width = 0.436 * eV
ground_state_width_uncertainty = 0.01*0.436*eV

gyro_mag_ratio_proton = 42.577*MHz/T

# units that do not show up in numericalunits
# missing pre-factors
fW = W*1e-15

# unitless units, relative fractions
pc = 0.01
ppm = 1e-6
ppb = 1e-9
ppt = 1e-12
ppq = 1e-15

# radian and degree which are also not really units
rad = 1
deg = np.pi/180


try:
    from morpho.utilities import morphologging
    logger = morphologging.getLogger(__name__)
except:
    print("Run without morpho!")

class NameSpace(object):
    def __init__(self, iteritems):
        if type(iteritems) == dict:
            iteritems = iteritems.items()
        for k, v in iteritems:
            setattr(self, k.lower(), v)
    def __getattribute__(self, item):
        return object.__getattribute__(self, item.lower())

##############################################################################
# CRES functions
def gamma(kin_energy):
    return kin_energy/(me*c0**2) + 1

def beta(kin_energy):
    # electron speed at kin_energy
    return np.sqrt(kin_energy**2+2*kin_energy*me*c0**2)/(kin_energy+me*c0**2)

def frequency(kin_energy, magnetic_field):
    # cyclotron frequency
    return e/(2*np.pi*me)/gamma(kin_energy)*magnetic_field

def kin_energy(freq, magnetic_field):
    return (e*c0**2/(2*np.pi*freq)*magnetic_field - me*c0**2)

def rad_power(kin_energy, pitch, magnetic_field):
    # electron radiation power
    f = frequency(kin_energy, magnetic_field)
    b = beta(kin_energy)
    Pe = 2*np.pi*(e*f*b*np.sin(pitch/rad))**2/(3*eps0*c0*(1-b**2))
    return Pe

def track_length(rho, kin_energy=None, molecular=True):
    if kin_energy is None:
        kin_energy = tritium_endpoint_molecular if molecular else tritium_endpoint_atomic
    crosssect = tritium_electron_crosssection_molecular if molecular else tritium_electron_crosssection_atomic
    return 1 / (rho * crosssect * beta(kin_energy) * c0)



###############################################################################
class Sensitivity(object):
    """
    Documentation:
        * Phase IV sensitivity document: https://www.overleaf.com/project/5de3e02edd267500011b8cc4
        * Talias sensitivity script: https://3.basecamp.com/3700981/buckets/3107037/documents/2388170839
        * Nicks CRLB for frequency resolution: https://3.basecamp.com/3700981/buckets/3107037/uploads/2009854398
        * Molecular contamination in atomic tritium: https://3.basecamp.com/3700981/buckets/3107037/documents/3151077016
    """
    def __init__(self, config_path):
        self.cfg = configparser.ConfigParser()
        with open(config_path, 'r') as configfile:
            self.cfg.read_file(configfile)

        # display configuration
        try:
            logger.info("Config file content:")
            for sect in self.cfg.sections():
               logger.info('    Section: {}'.format(sect))
               for k,v in self.cfg.items(sect):
                  logger.info('        {} = {}'.format(k,v))
        except:
            pass

        self.Experiment = NameSpace({opt: eval(self.cfg.get('Experiment', opt)) for opt in self.cfg.options('Experiment')})

        self.tau_tritium = tritium_livetime
        if self.Experiment.atomic:
            self.T_mass = tritium_mass_atomic
            self.Te_crosssection = tritium_electron_crosssection_atomic
            self.T_endpoint = tritium_endpoint_atomic
            self.last_1ev_fraction = last_1ev_fraction_atomic
        else:
            self.T_mass = tritium_mass_molecular
            self.Te_crosssection = tritium_electron_crosssection_molecular
            self.T_endpoint = tritium_endpoint_molecular
            self.last_1ev_fraction = last_1ev_fraction_molecular


        self.DopplerBroadening = NameSpace({opt: eval(self.cfg.get('DopplerBroadening', opt)) for opt in self.cfg.options('DopplerBroadening')})
        self.FrequencyExtraction = NameSpace({opt: eval(self.cfg.get('FrequencyExtraction', opt)) for opt in self.cfg.options('FrequencyExtraction')})
        self.MagneticField = NameSpace({opt: eval(self.cfg.get('MagneticField', opt)) for opt in self.cfg.options('MagneticField')})
        self.MissingTracks = NameSpace({opt: eval(self.cfg.get('MissingTracks', opt)) for opt in self.cfg.options('MissingTracks')})
        self.PlasmaEffects = NameSpace({opt: eval(self.cfg.get('PlasmaEffects', opt)) for opt in self.cfg.options('PlasmaEffects')})

    # SENSITIVITY
    def SignalRate(self):
        """signal events in the energy interval before the endpoint, scale with DeltaE**3"""
        signal_rate = self.Experiment.number_density*self.Experiment.v_eff*self.last_1ev_fraction/self.tau_livetime
        if not self.Experiment.atomic:
            signal_rate *= 2
        return signal_rate

    def DeltaEWidth(self):
        """optimal energy bin width"""
        labels, sigmas, deltas = self.get_systematics()
        return np.sqrt(self.Experiment.background_rate_per_eV/self.SignalRate()
                              + 8*np.log(2)*(np.sum(sigmas**2)))

    def StatSens(self):
        """Pure statistic sensitivity assuming Poisson count experiment in a single bin"""
        sig_rate = self.SignalRate()
        DeltaE = self.DeltaEWidth()
        sens = 4/(6*sig_rate*self.Experiment.LiveTime)*np.sqrt(sig_rate*self.Experiment.LiveTime*DeltaE
                                                                  +self.Experiment.background_rate_per_eV*self.Experiment.LiveTime/DeltaE)
        return sens

    def SystSens(self):
        """Pure systematic componenet to sensitivity"""
        labels, sigmas, deltas = self.get_systematics()
        sens = 4*np.sqrt(np.sum((sigmas*deltas)**2))
        return sens

    def sensitivity(self, **kwargs):
        """Combined statisical and systematic uncertainty. 
        Using kwargs settings in namespaces can be changed.
        Example how to change number density which lives in namespace Experiment:
            self.sensitivity(Experiment={"number_density": rho})
        """
        for sect, options in kwargs.items():
            for opt, val in options.items():
                self.__dict__[sect].__dict__[opt] = val

        StatSens = self.StatSens()
        SystSens = self.SystSens()

        # Standard deviation on a measurement of m_beta**2 assuming a mass of zero
        sigma_m_beta_2 =  np.sqrt(StatSens**2 + SystSens**2)
        return sigma_m_beta_2

    def CL90(self, **kwargs):
        """ Gives 90% CL upper limit on neutrino mass."""
        # 90% of gaussian are contained in +-1.64 sigma region
        return np.sqrt(np.sqrt(1.64)*self.sensitivity(**kwargs))

    # PHYSICS Functions
    def BToKeErr(self, BErr, B):
         return e*BErr/(2*np.pi*frequency(self.T_endpoint, B)/c0**2)

    def track_length(self, rho):
        return track_length(rho, self.T_endpoint, not self.Experiment.atomic)

    # SYSTEMATICS

    def get_systematics(self):
        # Different types of uncertainty contributions
        sigma_trans, delta_sigma_trans = self.syst_doppler_broadening()
        sigma_f, delta_sigma_f = self.syst_frequency_extraction()
        sigma_B, delta_sigma_B = self.syst_magnetic_field()
        sigma_Miss, delta_sigma_Miss = self.syst_missing_tracks()
        sigma_Plasma, delta_sigma_Plasma = self.syst_plasma_effects()

        labels = ["Termal Doppler Broadening", "Start Frequency Resolution", "Magnetic Field", "Missing Tracks", "Plasma Effects"]
        sigmas = [sigma_trans, sigma_f, sigma_B, sigma_Miss, sigma_Plasma]
        deltas = [delta_sigma_trans, delta_sigma_f, delta_sigma_B, delta_sigma_Miss, delta_sigma_Plasma]

        if not self.Experiment.atomic:
            labels.append("Molecular final state")
            sigmas.append(ground_state_width)
            deltas.append(ground_state_width_uncertainty)

        return np.array(labels), np.array(sigmas), np.array(deltas)

    def print_statistics(self):
        print("Statistic", " "*18, "%.2f"%(np.sqrt(self.StatSens())/meV), "meV")

    def print_systematics(self):
        labels, sigmas, deltas = self.get_systematics()

        print()
        for label, sigma, delta in zip(labels, sigmas, deltas):
            print(label, " "*(np.max([len(l) for l in labels])-len(label)),  "%8.2f"%(sigma/meV), "+/-", "%8.2f"%(delta/meV), "meV")

    def syst_doppler_broadening(self):
        # estimated standard deviation of Doppler broadening distribution from
        # translational motion of tritium atoms / molecules
        # Predicted uncertainty on standard deviation, based on expected precision
        # of temperature knowledge
        if self.DopplerBroadening.UseFixedValue:
            sigma = self.DopplerBroadening.Default_Systematic_Smearing
            delta = self.DopplerBroadening.Default_Systematic_Uncertainty
            return sigma, delta

        # termal doppler broardening
        gasTemp = self.DopplerBroadening.gas_temperature
        mass_T = self.T_mass
        endpoint = self.T_endpoint

        # these factors are mainly neglidible in the recoil equation below
        E_rec = 3.409 * eV # maximal value # same for molecular tritium?
        mbeta = 0*eV # term neglidible
        betanu = 1 # neutrinos are fast
        # electron-neutrino correlation term: 1 + 0.105(6)*betae*cosThetanu
        # => expectation value of cosThetanu = 0.014
        cosThetaenu = 0.014

        Ke = endpoint
        betae = np.sqrt(Ke**2+2*Ke*me*c0**2)/(Ke+me*c0**2) ## electron speed at energy Ke
        Emax = endpoint + me*c0**2
        Ee = endpoint + me*c0**2
        p_rec = np.sqrt( Emax**2-me**2*c0**4 + (Emax - Ee - E_rec)**2 - mbeta**2 + 2*Ee*(Emax - Ee - E_rec)*betae*betanu*cosThetaenu )
        sigma_trans = np.sqrt(p_rec**2/(2*mass_T)*2*kB*gasTemp)

        delta_trans = np.sqrt(p_rec**2/(2*mass_T)*kB/gasTemp*self.DopplerBroadening.gas_temperature_uncertainty**2)
        return sigma_trans, delta_trans


    def syst_frequency_extraction(self):
        # cite{https://3.basecamp.com/3700981/buckets/3107037/uploads/2009854398} (Section 1.2, p 7-9)
        # Are we double counting the antenna collection efficiency? We use it here. Does it also impact the effective volume, v_eff ?
        # Are we double counting the effect of magnetic field uncertainty here?
        # Is 'sigma_f_Bfield' the same as 'rRecoErr', 'delta_rRecoErr', 'rRecoPhiErr', 'delta_rRecoPhiErr'?

        if self.FrequencyExtraction.UseFixedValue:
            sigma = self.FrequencyExtraction.Default_Systematic_Smearing
            delta = self.FrequencyExtraction.Default_Systematic_Uncertainty
            return sigma, delta

        # Cramer-Rao lower bound / how much worse are we than the lower bound
        ScalingFactorCRLB = self.FrequencyExtraction.CRLB_scaling_factor
        ts = self.FrequencyExtraction.track_timestep
        # "This is apparent in the case of resonant patch antennas and cavities, in which the time scale of the signal onset is set by the Q-factor of the resonant structure."
        # You can get it from the finite impulse response of the antennas from HFSS
        Gdot = self.FrequencyExtraction.track_onset_rate

        fEndpoint = frequency(self.T_endpoint, self.MagneticField.nominal_field)
        betae = beta(self.T_endpoint)
        Pe = rad_power(self.T_endpoint, self.FrequencyExtraction.pitch_angle, self.MagneticField.nominal_field)
        alpha_approx = fEndpoint * 2 * np.pi * Pe/me/c0**2 # track slope
        # quantum limited noise
        sigNoise = np.sqrt((2*pi*fEndpoint*hbar*self.FrequencyExtraction.amplifier_noise_scaling+kB*self.FrequencyExtraction.antenna_noise_temperature)/ts) # noise level
        Amplitude = np.sqrt(self.FrequencyExtraction.epsilon_collection*Pe)
        Nsteps = 1 / (self.Experiment.number_density * self.Te_crosssection*betae*c0*ts) # Number of timesteps of length ts

        # sigma_f from Cramer-Rao lower bound in Hz
        sigma_f_CRLB = (ScalingFactorCRLB /(2*np.pi) * sigNoise/Amplitude * np.sqrt(alpha_approx**2/(2*Gdot)
                    + 96.*Nsteps/(ts**2*(Nsteps**4-5*Nsteps**2+4))))
        # uncertainty in alpha
        delta_alpha = 6*sigNoise/(Amplitude*ts**2) * np.sqrt(10/(Nsteps*(Nsteps**4-5*Nsteps**2+4)))
        # uncetainty in sigma_f in Hz due to uncertainty in alpha
        delta_sigma_f_CRLB = delta_alpha * alpha_approx *sigNoise**2/(8*np.pi**2*Amplitude**2*Gdot*sigma_f_CRLB*ScalingFactorCRLB**2)

        # sigma_f from Cramer-Rao lower bound in eV
        sigma_K_f_CRLB =  e*self.MagneticField.nominal_field/(2*np.pi*fEndpoint**2)*sigma_f_CRLB*c0**2
        delta_sigma_K_f_CRLB = e*self.MagneticField.nominal_field/(2*np.pi*fEndpoint**2)*delta_sigma_f_CRLB*c0**2

        # combined sigma_f in eV
        sigma_f = np.sqrt(sigma_K_f_CRLB**2 + self.FrequencyExtraction.magnetic_field_smearing**2)
        delta_sigma_f = np.sqrt((delta_sigma_K_f_CRLB**2 + self.FrequencyExtraction.magnetic_field_uncertainty**2)/2)

        return sigma_f, delta_sigma_f

    def syst_magnetic_field(self):
        if self.MagneticField.UseFixedValue:
            sigma = self.MagneticField.Default_Systematic_Smearing
            delta = self.MagneticField.Default_Systematic_Uncertainty
            return sigma, delta

        B = self.MagneticField.nominal_field
        if self.MagneticField.useinhomogenarity:
            inhomogenarity = self.MagneticField.inhomogenarity
            sigma = self.BToKeErr(inhomogenarity*B, B)
            return sigma, 0.05*sigma

        BMapErr = self.MagneticField.probe_repeatability  # Probe Repeatability
        delta_BMapErr = self.MagneticField.probe_resolution # Probe resolution

        BFlatErr = self.MagneticField.BFlatErr # averaging over the flat part of the field
        delta_BFlatErr = self.MagneticField.relative_uncertainty_BFlatErr*BFlatErr # UPDATE ?

        Delta_t_since_calib = self.MagneticField.time_since_calibration
        shiftBdot = self.MagneticField.shift_Bdot
        smearBdot = self.MagneticField.smear_Bdot
        delta_shiftBdot = self.MagneticField.uncertainty_shift_Bdot
        delta_smearBdot = self.MagneticField.uncertainty_smearBdot
        BdotErr = Delta_t_since_calib * np.sqrt(shiftBdot**2 + smearBdot**2)
        delta_BdotErr = Delta_t_since_calib**2/BdotErr * np.sqrt(shiftBdot**2 * delta_shiftBdot**2 + smearBdot**2 * delta_smearBdot**2)

        rRecoErr = self.MagneticField.rRecoErr
        delta_rRecoErr = self.MagneticField.relative_Uncertainty_rRecoErr * rRecoErr

        rRecoPhiErr = self.MagneticField.rRecoPhiErr
        delta_rRecoPhiErr = self.MagneticField.relative_uncertainty_rRecoPhiErr * rRecoPhiErr

        rProbeErr = self.MagneticField.rProbeErr
        delta_rProbeErr = self.MagneticField.relative_uncertainty_rProbeErr * rProbeErr

        rProbePhiErr = self.MagneticField.rProbePhiErr
        delta_rProbePhiErr = self.MagneticField.relative_uncertainty_rProbePhiErr * rProbePhiErr

        Berr = np.sqrt(BMapErr**2 +
                       BFlatErr**2 +
                       BdotErr**2 +
                       rRecoErr**2 +
                       rRecoPhiErr**2 +
                       rProbeErr**2 +
                       rProbePhiErr**2)

        delta_Berr = 1/Berr * np.sqrt(BMapErr**2 * delta_BMapErr**2 +
                                      BFlatErr**2 * delta_BFlatErr**2 +
                                      BdotErr**2 * delta_BdotErr**2 +
                                      rRecoErr**2 * delta_rRecoErr**2 +
                                      rRecoPhiErr**2 * delta_rRecoPhiErr**2 +
                                      rProbeErr**2 * delta_rProbeErr**2 +
                                      rProbePhiErr**2 * delta_rProbePhiErr**2)

        return self.BToKeErr(Berr, B), self.BToKeErr(delta_Berr, B)

    def syst_missing_tracks(self):
        if self.MissingTracks.UseFixedValue:
            sigma = self.MissingTracks.Default_Systematic_Smearing
            delta = self.MissingTracks.Default_Systematic_Uncertainty
            return sigma, delta

    def syst_plasma_effects(self):
        if self.PlasmaEffects.UseFixedValue:
            sigma = self.PlasmaEffects.Default_Systematic_Smearing
            delta = self.PlasmaEffects.Default_Systematic_Uncertainty
            return sigma, delta
        else:
            raise NotImplementedError()
