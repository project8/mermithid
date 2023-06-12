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
from numericalunits import meV, eV, keV, MeV, cm, m, ns, s, Hz, kHz, MHz, GHz, amu, nJ
from numericalunits import nT, uT, mT, T, mK, K,  C, F, g, W
from numericalunits import hour, year, day, s, ms
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
ground_state_width_uncertainty = 0.0*0.436*eV

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

def wavelength(kin_energy, magnetic_field):
    return c0/frequency(kin_energy, magnetic_field)

def kin_energy(freq, magnetic_field):
    return (e*c0**2/(2*np.pi*freq)*magnetic_field - me*c0**2)

"""def rad_power(kin_energy, pitch, magnetic_field):
    # electron radiation power
    f = frequency(kin_energy, magnetic_field)
    b = beta(kin_energy)
    Pe = 2*np.pi*(e*f*b*np.sin(pitch/rad))**2/(3*eps0*c0*(1-b**2))
    return Pe"""

def rad_power(kin_energy, pitch, magnetic_field):
    Pe = 2*(e**2*magnetic_field*np.sin(pitch))**2*(gamma(kin_energy)**2-1)/(12*eps0*c0*np.pi*me**2)
    return Pe

def track_length(rho, kin_energy=None, molecular=True):
    if kin_energy is None:
        kin_energy = tritium_endpoint_molecular if molecular else tritium_endpoint_atomic
    crosssect = tritium_electron_crosssection_molecular if molecular else tritium_electron_crosssection_atomic
    return 1 / (rho * crosssect * beta(kin_energy) * c0)

def sin2theta_sq_to_Ue4_sq(sin2theta_sq):
    return 0.5*(1-np.sqrt(1-sin2theta_sq**2))

def Ue4_sq_to_sin2theta_sq(Ue4_sq):
    return 4*Ue4_sq*(1-Ue4_sq)

# Wouters functinos
def db_to_pwr_ratio(q_db):
    return 10**(q_db/10)

def axial_frequency(length, kin_energy, max_pitch_angle=86):
    pitch_max = max_pitch_angle/180*np.pi
    return (beta(kin_energy)*c0*np.cos(pitch_max)) / (2*length)

def mean_field_frequency_variation(cyclotron_frequency, length_diameter_ratio, max_pitch_angle=86):
    # Because of the differenct electron trajectories in the trap,
    # An electron will see a slightly different magnetic field
    # depending on its position in the trap, especially the pitch angle.
    # This is a rough estimate of the mean field variation, inspired by calcualtion performed by Rene.
    y = (90-max_pitch_angle)/4
    return 0.002*y**2*cyclotron_frequency*(10/length_diameter_ratio)

# Noise power entering the amplifier, inclding the transmitted noise from the cavity and the reflected noise from the circulator.
# Insertion loss is included.
def Pn_dut_entrance(t_cavity,
                    t_amplifier,
                    att_line_db,
                    att_cir_db,
                    coupling,
                    freq,
                    bandwidth,
                    loaded_Q):
    att_cir = db_to_pwr_ratio(att_cir_db)
    att_line = db_to_pwr_ratio(att_line_db)
    assert( (np.all(att_cir<=1)) & (np.all(att_line<=1)) )

    # Calculate the noise at the cavity
    Pn_cav = Pn_cavity(t_cavity, coupling, loaded_Q, bandwidth, freq)
    Pn_circulator = t_effective(t_amplifier, freq)*kB*bandwidth
    Pn_circulator_after_reflection = Pn_reflected(Pn_f(Pn_circulator,t_amplifier, t_cavity,att_line, bandwidth), coupling, loaded_Q, bandwidth, freq)
    # Propagate the noise over the line towards the circulator
    Pn_entrance = Pn_f(Pn_circulator_after_reflection+Pn_cav,t_cavity,t_amplifier,att_line,bandwidth)
    # Apply the effect of the circulator
    return Pn_f(Pn_entrance,t_amplifier,t_amplifier,att_cir,bandwidth)

# Noise power genrated in the cavity integrated over the bandwidth that couples into the readout line.
def Pn_cavity(t_cavity, coupling, loaded_Q, bandwidth, freq):
    return kB*t_effective(t_cavity, freq)*4*coupling/(1+coupling)**2*freq/loaded_Q*np.arctan(loaded_Q*bandwidth/freq)

# Noise power reflecting of the cavity
def Pn_reflected(Pn_incident, coupling, loaded_Q, bandwidth, freq):
    
    reflection_coefficient = 1-freq/loaded_Q/bandwidth*np.arctan(loaded_Q*bandwidth/freq)*4*coupling/(1+coupling)**2
    return Pn_incident*reflection_coefficient

# Power at the end of a lossy line with temperature gradient
def Pn_f(Pn_i,t_i,t_f,a,bandwidth): # eq 10
    if hasattr(a, "__len__") or a!=1:
        return Pn_i+ kB*bandwidth*(t_f-t_i)*(1+ (1-a)/np.log(a))+ (t_i*kB*bandwidth-Pn_i)*(1-a)
    else:
        return Pn_i*np.ones_like(t_f)
    
# Effective temperature taking the quantum photon population into account.
def t_effective(t_physical, cyclotron_frequency):
    quantum = 2*np.pi*hbar*cyclotron_frequency/kB
    #for numerical stability
    if np.all(quantum/t_physical < 1e-2):
        return t_physical
    else:
       return quantum*(1/2+1/(np.exp(quantum/t_physical)-1))

###############################################################################
class CavitySensitivity(object):
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
        self.Efficiency = NameSpace({opt: eval(self.cfg.get('Efficiency', opt)) for opt in self.cfg.options('Efficiency')})
        
        self.CavityRadius()
        self.CavityVolume()
        self.EffectiveVolume()
        self.CavityPower()

    # CAVITY
    def CavityRadius(self):
        axial_mode_index = 1
        self.cavity_radius = c0/(2*np.pi*frequency(self.T_endpoint, self.MagneticField.nominal_field))*np.sqrt(3.8317**2+axial_mode_index**2*np.pi**2/(4*self.Experiment.L_over_D**2))
        return self.cavity_radius
    
    def CavityVolume(self):
        #radius = 0.5*wavelength(self.T_endpoint, self.MagneticField.nominal_field)
        self.total_volume = 2*self.cavity_radius*self.Experiment.L_over_D*np.pi*(self.cavity_radius)**2
        
        logger.info("Frequency: {} MHz".format(round(frequency(self.T_endpoint, self.MagneticField.nominal_field)/MHz, 3)))
        logger.info("Wavelength: {} cm".format(round(wavelength(self.T_endpoint, self.MagneticField.nominal_field)/cm, 3)))
        logger.info("Radius: {} cm".format(round(self.cavity_radius/cm, 3)))
        
        return self.total_volume
    
    def EffectiveVolume(self):
        if self.Efficiency.usefixedvalue:
            self.effective_volume = self.total_volume * self.Efficiency.total_efficiency
        else:
            # trapping efficiecny is currently configured. replace with box trap calculation
            self.effective_volume = self.total_volume*self.Efficiency.radial_efficiency*self.Efficiency.trapping_efficiency
            
        return self.effective_volume
        
    def CavityPower(self):
        # from Hamish's atomic calculator
        Jprime_0 = 3.8317
        
        self.signal_power = self.FrequencyExtraction.mode_coupling_efficiency * self.CavityLoadedQ() * self.FrequencyExtraction.hanneke_factor * self.T_endpoint/eV * e/C * Jprime_0**2 / (self.total_volume/m**3 * frequency(self.T_endpoint, self.MagneticField.nominal_field)*s)*W
        
        return self.signal_power
    
    
    def CavityLoadedQ(self):
        # Using Wouter's calculation:
        # Total required bandwidth is the sum of the endpoint region and the axial frequency. 
        # I will assume the bandwidth is dominated by the sidebands and not by the energy ROI
        endpoint_frequency = frequency(self.T_endpoint, self.MagneticField.nominal_field)
        required_bw_axialfrequency = axial_frequency(self.Experiment.L_over_D*self.CavityRadius()*2, 
                                                     self.T_endpoint, 
                                                     self.FrequencyExtraction.minimum_angle_in_bandwidth/deg)
        
        required_bw_meanfield = mean_field_frequency_variation(endpoint_frequency, 
                                                               self.Experiment.L_over_D,
                                                               self.FrequencyExtraction.minimum_angle_in_bandwidth/deg)
        
        required_bw = np.add(required_bw_axialfrequency,required_bw_meanfield) # Broadcasting
    
        # Cavity coupling
        self.loaded_q = endpoint_frequency/required_bw # FWHM
        return self.loaded_q
    
    # SENSITIVITY
    def SignalRate(self):
        """signal events in the energy interval before the endpoint, scale with DeltaE**3"""
        self.EffectiveVolume()
        signal_rate = self.Experiment.number_density*self.effective_volume*self.last_1ev_fraction/self.tau_tritium
        if not self.Experiment.atomic:
            if hasattr(self.Experiment, 'gas_fractions'):
                avg_n_T_atoms = self.AvgNumTAtomsPerParticle_MolecularExperiment(self.Experiment.gas_fractions, self.Experiment.H2_type_gas_fractions)
                signal_rate *= avg_n_T_atoms
            else:
                signal_rate *= 2
        return signal_rate

    def BackgroundRate(self):
        """background rate, can be calculated from multiple components.
        Assumes that background rate is constant over considered energy / frequency range."""
        return self.Experiment.background_rate_per_eV

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
        return np.sqrt(np.sqrt(1.64)*self.sensitivity(**kwargs))

    def sterial_m2_limit(self, Ue4_sq):
        return np.sqrt(np.sqrt(1.64)*np.sqrt((self.StatSens()/Ue4_sq)**2 + self.SystSens()**2))

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
            deltas.append(ground_state_width_uncertainty)

        return np.array(labels), np.array(sigmas), np.array(deltas)

    def print_statistics(self):
        print("Statistical", " "*18, "%.2f"%(np.sqrt(self.StatSens())/meV), "meV")

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

    def calculate_tau_snr(self, time_window):
        
        endpoint_frequency = frequency(self.T_endpoint, self.MagneticField.nominal_field)
    
        # Cavity coupling
        self.CavityLoadedQ()
        coupling = self.FrequencyExtraction.unloaded_q/self.loaded_q-1
    
        # Attenuation frequency dependence at hoc method
        #att_cir_db = -0.3
        #att_line_db = -0.05
        att_cir_db_freq = self.FrequencyExtraction.att_cir_db*(1+endpoint_frequency/(10*GHz))
        att_line_db_freq = self.FrequencyExtraction.att_line_db*(1+endpoint_frequency/(10*GHz))
        
        # Noise power for bandwidth set by density/track length
        fft_bandwidth = 3/time_window #(delta f) is the frequency bandwidth of interest. We have a main carrier and 2 axial side bands, so 3*(FFT bin width)
        tn_fft = Pn_dut_entrance(self.FrequencyExtraction.cavity_temperature,
                                 self.FrequencyExtraction.amplifier_temperature,
                                 att_line_db_freq,att_cir_db_freq,
                                 coupling,
                                 endpoint_frequency,
                                 fft_bandwidth,self.loaded_q)/kB/fft_bandwidth
    
        # Noise temperature of amplifier
        tn_amplifier = endpoint_frequency*hbar*2*np.pi/kB/self.FrequencyExtraction.quantum_amp_efficiency
        tn_system_fft = tn_amplifier+tn_fft
        
        # Pe = rad_power(self.T_endpoint, self.FrequencyExtraction.pitch_angle, self.MagneticField.nominal_field)
        # logger.info("Power: {}".format(Pe/W))
        Pe = self.signal_power
        
        P_signal_received = Pe*db_to_pwr_ratio(att_cir_db_freq+att_line_db_freq)
        tau_snr = kB*tn_system_fft/P_signal_received
        
        # end of Wouter's calculation
        return tau_snr
        

    def syst_frequency_extraction(self):
        # cite{https://3.basecamp.com/3700981/buckets/3107037/uploads/2009854398} (Section 1.2, p 7-9)
        # Are we double counting the antenna collection efficiency? We use it here. Does it also impact the effective volume, v_eff ?
        
        if self.FrequencyExtraction.UseFixedValue:
            sigma = self.FrequencyExtraction.Default_Systematic_Smearing
            delta = self.FrequencyExtraction.Default_Systematic_Uncertainty
            return sigma, delta

        """ # Cramer-Rao lower bound / how much worse are we than the lower bound
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
                    + 96.*Nsteps/(ts**2*(Nsteps**4-5*Nsteps**2+4))))"""
        
       
        endpoint_frequency = frequency(self.T_endpoint, self.MagneticField.nominal_field)
        # using Pe and alpha (aka slope) from above
        Pe = self.CavityPower()/self.FrequencyExtraction.mode_coupling_efficiency 
        self.larmor_power = rad_power(self.T_endpoint, self.FrequencyExtraction.pitch_angle, self.MagneticField.nominal_field) # currently not used
        
        self.slope = endpoint_frequency * 2 * np.pi * Pe/me/c0**2 # track slope
        time_window = track_length(self.Experiment.number_density, self.T_endpoint, molecular=(not self.Experiment.atomic))
        
        time_window_slope_zero = abs(frequency(self.T_endpoint, self.MagneticField.nominal_field)-frequency(self.T_endpoint+10*meV, self.MagneticField.nominal_field))/self.slope
        
        tau_snr_full_length = self.calculate_tau_snr(time_window)
        tau_snr_part_length = self.calculate_tau_snr(time_window_slope_zero)
       
        
        
        # use different crlb based on slope
        delta_E_slope = abs(kin_energy(endpoint_frequency, self.MagneticField.nominal_field)-kin_energy(endpoint_frequency+self.slope*time_window, self.MagneticField.nominal_field))
        logger.info("slope is {} Hz/ms".format(self.slope/Hz*ms))
        # logger.info("slope corresponds to {} meV / ms".format(delta_E_slope/meV))
        if time_window_slope_zero >= time_window:
            #logger.info("slope is approximately 0".format(self.slope/meV*ms))
            CRLB_constant = np.sqrt(12)
            ratio_window_to_length = 1
            #sigma_f_CRLB = gamma(self.T_endpoint)*self.T_endpoint/((gamma(self.T_endpoint)-1)*2*np.pi*endpoint_frequency)*np.sqrt(CRLB_constant*tau_snr*0.3/(time_window**3*ratio_window_to_length**2.3))
            sigma_f_CRLB = np.sqrt((CRLB_constant*tau_snr_full_length/time_window**3))/(2*np.pi)*self.FrequencyExtraction.CRLB_scaling_factor
            self.slope_is_zero=True
        else:
            CRLB_constant = np.sqrt(12)
            sigma_CRLB_slope_zero = np.sqrt((CRLB_constant*tau_snr_part_length/time_window_slope_zero**3))/(2*np.pi)*self.FrequencyExtraction.CRLB_scaling_factor
            
            sigma_f_CRLB_slope_fitted = np.sqrt((20*(self.slope*tau_snr_full_length)**2 + 180*tau_snr_full_length/time_window**3)/(2*np.pi)**2)*self.FrequencyExtraction.CRLB_scaling_factor
        
            sigma_f_CRLB = np.min([sigma_CRLB_slope_zero, sigma_f_CRLB_slope_fitted])
            
            # logger.info("CRLB options are: {} , {}".format(sigma_CRLB_slope_zero/Hz, sigma_f_CRLB_slope_fitted/Hz))
            self.slope_is_zero = False
            self.crlb_decision=np.argmin([sigma_CRLB_slope_zero, sigma_f_CRLB_slope_fitted])
        
        """# uncertainty in alpha
        delta_alpha = 6*sigNoise/(Amplitude*ts**2) * np.sqrt(10/(Nsteps*(Nsteps**4-5*Nsteps**2+4)))
        # uncetainty in sigma_f in Hz due to uncertainty in alpha
        delta_sigma_f_CRLB = delta_alpha * alpha_approx *sigNoise**2/(8*np.pi**2*Amplitude**2*Gdot*sigma_f_CRLB*ScalingFactorCRLB**2)"""

        # sigma_f from Cramer-Rao lower bound in eV
        sigma_K_f_CRLB =  e*self.MagneticField.nominal_field/(2*np.pi*endpoint_frequency**2)*sigma_f_CRLB*c0**2
        # delta_sigma_K_f_CRLB = e*self.MagneticField.nominal_field/(2*np.pi*endpoint_frequency**2)*delta_sigma_f_CRLB*c0**2

        # combined sigma_f in eV
        sigma_f = np.sqrt(sigma_K_f_CRLB**2 + self.FrequencyExtraction.magnetic_field_smearing**2)
        # delta_sigma_f = np.sqrt((delta_sigma_K_f_CRLB**2 + self.FrequencyExtraction.magnetic_field_uncertainty**2)/2)
        if self.FrequencyExtraction.usefixeduncertainty:
            return sigma_f, self.FrequencyExtraction.fixed_relativ_uncertainty*sigma_f
        else:
            raise NotImplementedError("Unvertainty on CRLB for cavity noise calculation is not implemented.")

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
