'''
Class calculating neutrino mass sensitivities based on analytic formulas from CDR.
Author: R. Reimann, C. Claessens
Date:11/17/2020

The statistical method and formulas are described in
CDR (CRES design report, Section 1.3) https://www.overleaf.com/project/5b9314afc673d862fa923d53.
'''
import numpy as np
import configparser

from mermithid.misc.Constants_numericalunits import *
from mermithid.misc.CRESFunctions_numericalunits import *

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
        
        # seetings fro molecular or atomic tritium
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
            
        # effective volume if configured
        if hasattr(self.Experiment, "v_eff"):
            self.effective_volume = self.Experiment.v_eff # v_eff can be configured in Experiment section


        self.DopplerBroadening = NameSpace({opt: eval(self.cfg.get('DopplerBroadening', opt)) for opt in self.cfg.options('DopplerBroadening')})
        self.FrequencyExtraction = NameSpace({opt: eval(self.cfg.get('FrequencyExtraction', opt)) for opt in self.cfg.options('FrequencyExtraction')})
        self.MagneticField = NameSpace({opt: eval(self.cfg.get('MagneticField', opt)) for opt in self.cfg.options('MagneticField')})
        self.MissingTracks = NameSpace({opt: eval(self.cfg.get('MissingTracks', opt)) for opt in self.cfg.options('MissingTracks')})
        self.PlasmaEffects = NameSpace({opt: eval(self.cfg.get('PlasmaEffects', opt)) for opt in self.cfg.options('PlasmaEffects')})
        
        if self.cfg.has_section('FinalStates'):
            self.FinalStates = NameSpace({opt: eval(self.cfg.get('FinalStates', opt)) for opt in self.cfg.options('FinalStates')})
        
        if not self.Experiment.atomic and not self.cfg.has_section('FinalStates'):
            logger.warning(f"No configuration of ground state width uncertainty. Using default value {ground_state_width_uncertainty/eV} eV")
                

    # SENSITIVITY
    def SignalRate(self):
        """signal events in the energy interval before the endpoint, scale with DeltaE**3"""
        signal_rate = self.Experiment.number_density*self.effective_volume*self.last_1ev_fraction/self.tau_tritium
        if not self.Experiment.atomic:
            if hasattr(self.Experiment, 'gas_fractions'):
                avg_n_T_atoms = self.AvgNumTAtomsPerParticle_MolecularExperiment(self.Experiment.gas_fractions, self.Experiment.H2_type_gas_fractions)
                signal_rate *= avg_n_T_atoms
            else:
                signal_rate *= 2
        if hasattr(self.Experiment, 'active_gas_fraction'):
            signal_rate *= self.Experiment.active_gas_fraction
        return signal_rate

    def BackgroundRate(self):
        """background rate, can be calculated from multiple components.
        Currently, RF noise and cosmic ray backgrounds are included.
        Assumes that background rate is constant over considered energy / frequency range."""
        self.cosmic_ray_background = self.Experiment.cosmic_ray_bkgd_per_tritium_particle*self.Experiment.number_density*self.effective_volume
        self.background_rate = self.Experiment.RF_background_rate_per_eV + self.cosmic_ray_background
        return self.background_rate

    def SignalEvents(self):
        """Number of signal events."""
        return self.SignalRate()*self.Experiment.LiveTime*self.DeltaEWidth()**3

    def BackgroundEvents(self):
        """Number of background events."""
        return self.BackgroundRate()*self.Experiment.LiveTime*self.DeltaEWidth()

    def DeltaEWidth(self):
        """optimal energy bin width"""
        labels, sigmas, deltas = self.get_systematics()
        return np.sqrt(self.BackgroundRate()/self.SignalRate()
                              + 8*np.log(2)*(np.sum(sigmas**2)))

    def StatSens(self):
        """Pure statistic sensitivity assuming Poisson count experiment in a single bin"""
        sig_rate = self.SignalRate()
        DeltaE = self.DeltaEWidth()
        sens = 2/(3*sig_rate*self.Experiment.LiveTime)*np.sqrt(sig_rate*self.Experiment.LiveTime*DeltaE
                                                                  +self.BackgroundRate()*self.Experiment.LiveTime/DeltaE)
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
        #return np.sqrt(np.sqrt(1.64)*self.sensitivity(**kwargs))
        return np.sqrt(1.64*self.sensitivity(**kwargs))

    def sterial_m2_limit(self, Ue4_sq):
        return np.sqrt(1.64*np.sqrt((self.StatSens()/Ue4_sq)**2 + self.SystSens()**2))

    # PHYSICS Functions

    def AvgNumTAtomsPerParticle_MolecularExperiment(self, gas_fractions, H2_type_gas_fractions):
        """
        Given gas composition info (H2 vs. other gases, and how much of each H2-type isotopolog), returns an average number of tritium atoms per gas particle.

        Inputs:
        - gas_fractions: dict of composition fractions of each gas (different from scatter fractions!); all H2 isotopologs are combined under key 'H2'
        - H2_type_gas_fractions: dict with fraction of each isotopolog, out of total amount of H2
        """
        H2_iso_avg_num = 0
        for (key, val) in H2_type_gas_fractions.items():
            if key=='T2':
              H2_iso_avg_num += 2*val
            elif key=='HT' or key=='DT':
              H2_iso_avg_num += val
            elif key=='H2' or key=='HD' or key=='D2':
              pass
        return gas_fractions['H2']*H2_iso_avg_num

    def BToKeErr(self, BErr, B):
         return e*BErr/(2*np.pi*frequency(self.T_endpoint, B)/c0**2)

    def KeToBerr(self, KeErr, B):
        return KeErr/e*(2*np.pi*frequency(self.T_endpoint, B)/c0**2)

    def track_length(self, rho):
        return track_length(rho, self.T_endpoint, not self.Experiment.atomic)

    # SYSTEMATICS

    def get_systematics(self):
        """ Returns list of energy broadenings (sigmas) and
        uncertainties on these energy broadenings (deltas)
        for all considered systematics. We need to make sure
        that we do not include effects twice or miss any
        important effect.

        Returns:
             * list of labels
             * list of energy broadenings
             * list of energy broadening uncertainties
        """

        # Different types of uncertainty contributions
        sigma_trans, delta_sigma_trans = self.syst_doppler_broadening()
        sigma_f, delta_sigma_f = self.syst_frequency_extraction()
        sigma_B, delta_sigma_B = self.syst_magnetic_field()
        sigma_Miss, delta_sigma_Miss = self.syst_missing_tracks()
        sigma_Plasma, delta_sigma_Plasma = self.syst_plasma_effects()

        labels = ["Thermal Doppler Broadening", "Start Frequency Resolution", "Magnetic Field", "Missing Tracks", "Plasma Effects"]
        sigmas = [sigma_trans, sigma_f, sigma_B, sigma_Miss, sigma_Plasma]
        deltas = [delta_sigma_trans, delta_sigma_f, delta_sigma_B, delta_sigma_Miss, delta_sigma_Plasma]

        if not self.Experiment.atomic:
            labels.append("Molecular final state")
            sigmas.append(ground_state_width)
            if self.cfg.has_section('FinalStates') and hasattr(self.FinalStates, "ground_state_width_uncertainty_fraction"):
                deltas.append(self.FinalStates.ground_state_width_uncertainty_fraction*ground_state_width)
            else: 
                deltas.append(ground_state_width_uncertainty)

        return np.array(labels), np.array(sigmas), np.array(deltas)

    def print_statistics(self):
        print("Contribution to sigma_(m_beta^2)", " "*18, "%.2f"%(self.StatSens()/meV**2), "meV^2 ->", "%.2f"%(np.sqrt(self.StatSens())/meV), "meV")
        print("Statistical mass limit", " "*18, "%.2f"%(np.sqrt(1.64*self.StatSens())/meV), "meV")

    def print_systematics(self):
        labels, sigmas, deltas = self.get_systematics()

        print()
        sigma_squared = 0
        for label, sigma, delta in zip(labels, sigmas, deltas):
            print(label, " "*(np.max([len(l) for l in labels])-len(label)),  "%8.2f"%(sigma/meV), "+/-", "%8.2f"%(delta/meV), "meV")
            sigma_squared += sigma**2
        sigma_total = np.sqrt(sigma_squared)
        print("Total sigma", " "*(np.max([len(l) for l in labels])-len("Total sigma")), "%8.2f"%(sigma_total/meV),)
        try:
            print("(Contribution from axial variation: ", "%8.2f"%(self.sigma_K_reconstruction/meV)," meV)")
        except AttributeError:
            pass
        print("Contribution to sigma_(m_beta^2)", " "*18, "%.2f"%(self.SystSens()/meV**2), "meV^2 ->", "%.2f"%(np.sqrt(self.SystSens())/meV), "meV")
        print("Systematic mass limit", " "*18, "%.2f"%(np.sqrt(1.64*self.SystSens())/meV), "meV")
        logger.info("f_c uncertainty: {} Hz".format(self.sigma_f_c_CRLB/Hz))
        return np.sqrt(1.64*self.SystSens())/meV, np.sqrt(np.sum(sigmas**2))/meV

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

        if self.Experiment.atomic == True:
            delta_trans = np.sqrt(p_rec**2/(2*mass_T)*kB/gasTemp*self.DopplerBroadening.gas_temperature_uncertainty**2)
        else:
            delta_trans = sigma_trans*self.DopplerBroadening.fraction_uncertainty_on_doppler_broadening
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
        sigNoise = np.sqrt((2*np.pi*fEndpoint*hbar*self.FrequencyExtraction.amplifier_noise_scaling+kB*self.FrequencyExtraction.antenna_noise_temperature)/ts) # noise level
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

        # the magnetic_field_smearing and uncertainty added here consider the following effect:
        # thinking in terms of a phase II track, there is some smearing of the track / width of the track which influences the frequency extraction
        # this does not account for any effect comming from converting frequency to energy
        # the reason behind the track width / smearing is the change in B field that the electron sees within one axial oscillation.
        # Depending on the trap shape this smearing may be different.

        return sigma_f, delta_sigma_f

    def syst_magnetic_field(self):

        # magnetic field uncertainties can be decomposed in several part
        # * true magnetic field inhomogeneity
        #   (would be there also without a trap)
        # * magnetic field calibration has uncertainties
        #   (would be there also without a trap)
        # * position / pitch angle reconstruction has uncertainties
        #   (this can even be the degenerancy we see for harmonic traps)
        #   (depends on trap shape)

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

        # position uncertainty is linear in wavelength
        # position uncertainty is nearly constant w.r.t. radial position
        # based on https://3.basecamp.com/3700981/buckets/3107037/uploads/3442593126
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
        # this systematic should describe the energy broadening due to the line shape.
        # Line shape is caused because you miss the first n tracks but then detect the n+1
        # track and you assume that this is the start frequency.
        # This depends on the gas composition, density and cross-section.
        if self.MissingTracks.UseFixedValue:
            sigma = self.MissingTracks.Default_Systematic_Smearing
            delta = self.MissingTracks.Default_Systematic_Uncertainty
            return sigma, delta
        else:
            raise NotImplementedError("Missing track systematic is not implemented.")

    def syst_plasma_effects(self):
        if self.PlasmaEffects.UseFixedValue:
            sigma = self.PlasmaEffects.Default_Systematic_Smearing
            delta = self.PlasmaEffects.Default_Systematic_Uncertainty
            return sigma, delta
        else:
            raise NotImplementedError("Plasma effect sysstematic is not implemented.")
