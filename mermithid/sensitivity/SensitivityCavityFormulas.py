'''
Class calculating neutrino mass sensitivities based on analytic formulas from CDR.
Author: R. Reimann, C. Claessens, T. Weiss, W. Van De Pontseele
Date:06/07/2023

The statistical method and formulas are described in
CDR (CRES design report, Section 1.3) https://www.overleaf.com/project/5b9314afc673d862fa923d53.
'''
import numpy as np

from mermithid.misc.Constants_numericalunits import *
from mermithid.misc.CRESFunctions_numericalunits import *
from mermithid.cavity.HannekeFunctions import *
from mermithid.sensitivity.SensitivityFormulas import *



try:
    from morpho.utilities import morphologging
    logger = morphologging.getLogger(__name__)
except:
    print("Run without morpho!")



# Wouters functinos
def db_to_pwr_ratio(q_db):
    return 10**(q_db/10)

def axial_motion(magnetic_field, pitch, trap_length, minimum_trapped_pitch, kin_energy, flat_fraction=0.5, trajectory = None):
    # returns the axial motion frequency and a trajectory of point along the axial motion 
    # also return the average magnetic field seen by the electron
    # from z=0 to z=cavity_length/2 with npoints set by the trajectory variable
    # See LUCKEY write-up for a little more on Talia's "flat fraction" trap model
    pitch = pitch/180*np.pi
    minimum_trapped_pitch = minimum_trapped_pitch/180*np.pi

    # Axial motion:
    z_w = trap_length/2
    speed = beta(kin_energy)*c0
    transverse_speed = speed*np.cos(pitch)
    tan_min = np.tan(minimum_trapped_pitch)
    # Axial frequency
    time_flat = z_w*flat_fraction/transverse_speed
    time_harmonic = np.pi*z_w*(1-flat_fraction)*tan_min/(2*speed*np.sin(pitch))
    axial_frequency = 1/4/(time_flat+time_harmonic)

    #Average magnetic field:
    magnetic_field_avg_harm = magnetic_field/2*(1+1/np.sin(pitch)**2) 
    magnetic_field_avg = (magnetic_field_avg_harm*time_harmonic + magnetic_field*time_flat)/(time_harmonic+time_flat)

    # Trajectory:
    if trajectory is None:
       z_t = None
    else:
        omega_harm = speed*np.sin(pitch)/z_w/tan_min/(1-flat_fraction)
        time = np.linspace(0, time_flat+time_harmonic, trajectory)
        z_t = np.heaviside(time_flat-time, 0.5)*time*transverse_speed +\
              np.heaviside(time-time_flat, 0.5)*(z_w*flat_fraction + z_w*(1-flat_fraction)*tan_min/np.tan(pitch)*np.sin(omega_harm*(time-time_flat)))
  
    return axial_frequency, magnetic_field_avg, z_t

def magnetic_field_flat_harmonic(z, magnetic_field, trap_length, minimum_trapped_pitch, flat_fraction=0.5):
    z_w = trap_length/2
    a = z_w*(1-flat_fraction)*np.tan(minimum_trapped_pitch)
    return magnetic_field*(1+np.heaviside(np.abs(z)-z_w*flat_fraction, 0.5)*(np.abs(z)-z_w*flat_fraction)**2/a**2)


def axial_frequency_box(length, kin_energy, max_pitch_angle=86):
    pitch_max = max_pitch_angle/180*np.pi
    return (beta(kin_energy)*c0*np.cos(pitch_max)) / (2*length)

def mean_field_frequency_variation(cyclotron_frequency, length_diameter_ratio, max_pitch_angle=86):
    # Because of the different electron trajectories in the trap,
    # An electron will see a slightly different magnetic field
    # depending on its position in the trap, especially the pitch angle.
    # This is a rough estimate of the mean field variation, inspired by calcualtion performed by Rene.
    #y = (90-max_pitch_angle)/4
    phi_rad = (90-max_pitch_angle)/180*np.pi
    return 0.16*phi_rad**2*cyclotron_frequency*(10/length_diameter_ratio)
    #return 0.002*y**2*cyclotron_frequency*(10/length_diameter_ratio)

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
class CavitySensitivity(Sensitivity):
    """
    Documentation:
        * Phase IV sensitivity document: https://www.overleaf.com/project/5de3e02edd267500011b8cc4
        * Talia's sensitivity script: https://3.basecamp.com/3700981/buckets/3107037/documents/2388170839
        * Nicks CRLB for frequency resolution: https://3.basecamp.com/3700981/buckets/3107037/uploads/2009854398
        * Molecular contamination in atomic tritium: https://3.basecamp.com/3700981/buckets/3107037/documents/3151077016
    """
    def __init__(self, config_path):
        Sensitivity.__init__(self, config_path)
        self.Efficiency = NameSpace({opt: eval(self.cfg.get('Efficiency', opt)) for opt in self.cfg.options('Efficiency')})
        
        if self.Experiment.trap_L_over_D == 0:
            self.Experiment.trap_L_over_D = self.Experiment.L_over_D

        self.CRLB_constant = 12
        #self.CRLB_constant = 90
        if hasattr(self.FrequencyExtraction, "crlb_constant"):
            self.CRLB_constant = self.FrequencyExtraction.crlb_constant
            logger.info("Using configured CRLB constant")       
 
        self.CavityRadius()
        self.CavityVolume()
        self.EffectiveVolume()
        self.PitchDependentTrappingEfficiency()
        self.CavityPower()

    # CAVITY
    def CavityRadius(self):
        axial_mode_index = 1
        self.cavity_radius = c0/(2*np.pi*frequency(self.T_endpoint, self.MagneticField.nominal_field))*np.sqrt(3.8317**2+axial_mode_index**2*np.pi**2/(4*self.Experiment.L_over_D**2))
        return self.cavity_radius
    
    def CavityVolume(self):
        #radius = 0.5*wavelength(self.T_endpoint, self.MagneticField.nominal_field)
        self.total_cavity_volume = 2*self.cavity_radius*self.Experiment.L_over_D*np.pi*(self.cavity_radius)**2*self.Experiment.n_cavities
        
        logger.info("Frequency: {} MHz".format(round(frequency(self.T_endpoint, self.MagneticField.nominal_field)/MHz, 3)))
        logger.info("Wavelength: {} cm".format(round(wavelength(self.T_endpoint, self.MagneticField.nominal_field)/cm, 3)))
        logger.info("Cavity radius: {} cm".format(round(self.cavity_radius/cm, 3)))
        logger.info("Cavity length: {} cm".format(round(2*self.cavity_radius*self.Experiment.L_over_D/cm, 3)))
        logger.info("Total cavity volume {} m^3".format(round(self.total_cavity_volume/m**3)))\
        
        return self.total_cavity_volume
    

    # ELECTRON TRAP
    def TrapVolume(self):
        # Total volume of the electron traps in all cavities
        self.total_trap_volume = 2*self.cavity_radius*self.Experiment.trap_L_over_D*np.pi*(self.cavity_radius)**2*self.Experiment.n_cavities
    
        logger.info("Trap radius: {} cm".format(round(self.cavity_radius/cm, 3)))
        logger.info("Trap length: {} cm".format(round(2*self.cavity_radius*self.Experiment.trap_L_over_D/cm, 3)))
        logger.info("Total trap volume {} m^3 ()".format(round(self.total_trap_volume/m**3)))
        
        return self.total_trap_volume


    
    def EffectiveVolume(self):
        if self.Efficiency.usefixedvalue:
            self.effective_volume = self.total_trap_volume * self.Efficiency.fixed_efficiency
        else:
            # radial and detection efficiency are configured in the config file
            #logger.info("Radial efficiency: {}".format(self.Efficiency.radial_efficiency))
            #logger.info("Detection efficiency: {}".format(self.Efficiency.detection_efficiency))
            #logger.info("Pitch angle efficiency: {}".format(self.PitchDependentTrappingEfficiency()))
            #logger.info("SRI factor: {}".format(self.Experiment.sri_factor))
            
            self.effective_volume = self.total_trap_volume*self.Efficiency.radial_efficiency*self.Efficiency.detection_efficiency*self.PitchDependentTrappingEfficiency()   
        #logger.info("Total efficiency: {}".format(self.effective_volume/self.total_volume))        
        self.effective_volume*=self.Experiment.sri_factor
        
        # for parent SignalRate function
        # self.Experiment.v_eff = self.effective_volume
        
        return self.effective_volume
        
    def PitchDependentTrappingEfficiency(self):
        self.pitch_angle_efficiency = np.cos(self.FrequencyExtraction.minimum_angle_in_bandwidth)
        return self.pitch_angle_efficiency

    def CavityPower(self):
        # from Hamish's atomic calculator
        #Jprime_0 = 3.8317
        max_ax_freq, mean_field, z_t = axial_motion(self.MagneticField.nominal_field,
                                                  self.FrequencyExtraction.minimum_angle_in_bandwidth/deg,
                                                  self.Experiment.trap_L_over_D*self.CavityRadius()*2,
                                                  self.FrequencyExtraction.minimum_angle_in_bandwidth/deg, 
                                                  self.T_endpoint, flat_fraction=self.MagneticField.trap_flat_fraction, trajectory = 1000)

        #self.signal_power = self.FrequencyExtraction.mode_coupling_efficiency * self.CavityLoadedQ() * self.FrequencyExtraction.hanneke_factor * self.T_endpoint/eV * e/C * Jprime_0**2 / (2*np.pi**2*self.Experiment.L_over_D*2*self.cavity_radius**3/m**3 * frequency(self.T_endpoint, self.MagneticField.nominal_field)*s)*W
        self.signal_power = np.mean(larmor_orbit_averaged_hanneke_power(np.random.triangular(0, self.cavity_radius, self.cavity_radius, size=2000),
                                                                            z_t, self.CavityLoadedQ(), 
                                                                            2*self.Experiment.L_over_D*self.cavity_radius, 
                                                                            self.cavity_radius, 
                                                                            frequency(self.T_endpoint, self.MagneticField.nominal_field)))
        return self.signal_power
    
    
    def CavityLoadedQ(self):
        # Using Wouter's calculation:
        # Total required bandwidth is the sum of the endpoint region and the axial frequency. 
        # I will assume the bandwidth is dominated by the sidebands and not by the energy ROI
        
        #self.loaded_q =1/(0.22800*((90-self.FrequencyExtraction.minimum_angle_in_bandwidth)*np.pi/180)**2+2**2*0.01076**2/(4*0.22800))

        endpoint_frequency = frequency(self.T_endpoint, self.MagneticField.nominal_field)
        #required_bw_axialfrequency = axial_frequency(self.Experiment.L_over_D*self.CavityRadius()*2, 
        #                                             self.T_endpoint, 
        #                                             self.FrequencyExtraction.minimum_angle_in_bandwidth/deg)
        max_ax_freq, mean_field, _ = axial_motion(self.MagneticField.nominal_field,
                                                  self.FrequencyExtraction.minimum_angle_in_bandwidth/deg,
                                                  self.Experiment.trap_L_over_D*self.CavityRadius()*2,
                                                  self.FrequencyExtraction.minimum_angle_in_bandwidth/deg, 
                                                  self.T_endpoint, flat_fraction=self.MagneticField.trap_flat_fraction)
        required_bw_axialfrequency = max_ax_freq
        self.required_bw_axialfrequency = required_bw_axialfrequency
        required_bw_meanfield = required_bw_meanfield = np.abs(frequency(self.T_endpoint, mean_field) - endpoint_frequency)
        required_bw = np.add(required_bw_axialfrequency,required_bw_meanfield) # Broadcasting
        self.required_bw = required_bw
    
        # Cavity coupling
        self.loaded_q = endpoint_frequency/required_bw # FWHM
        return self.loaded_q
    
    # SENSITIVITY
    # see parent class in SensitivityFormulas.py
 

    # SYSTEMATICS
    # Generic systematics are implemented in the parent class in SensitivityFormulas.py

    def calculate_tau_snr(self, time_window, sideband_power_fraction=1):
        
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
        self.fft_bandwidth = fft_bandwidth
        tn_fft = Pn_dut_entrance(self.FrequencyExtraction.cavity_temperature,
                                 self.FrequencyExtraction.amplifier_temperature,
                                 att_line_db_freq,att_cir_db_freq,
                                 coupling,
                                 endpoint_frequency,
                                 fft_bandwidth,self.loaded_q)/kB/fft_bandwidth
    
        # Noise temperature of amplifier
        tn_amplifier = endpoint_frequency*hbar*2*np.pi/kB/self.FrequencyExtraction.quantum_amp_efficiency
        tn_system_fft = tn_amplifier+tn_fft
        self.noise_temp = tn_system_fft
        
        # Pe = rad_power(self.T_endpoint, self.FrequencyExtraction.pitch_angle, self.MagneticField.nominal_field)
        # logger.info("Power: {}".format(Pe/W))
        Pe = self.signal_power * sideband_power_fraction
        
        P_signal_received = Pe*db_to_pwr_ratio(att_cir_db_freq+att_line_db_freq)
        self.received_power = P_signal_received
        tau_snr = kB*tn_system_fft/P_signal_received
        self.noise_energy = kB*tn_system_fft

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
        Pe = self.signal_power #/self.FrequencyExtraction.mode_coupling_efficiency 
        self.larmor_power = rad_power(self.T_endpoint, self.FrequencyExtraction.pitch_angle, self.MagneticField.nominal_field) # currently not used
        
        self.slope = endpoint_frequency * 2 * np.pi * Pe/me/c0**2 # track slope
        self.time_window = track_length(self.Experiment.number_density, self.T_endpoint, molecular=(not self.Experiment.atomic))
        
        self.time_window_slope_zero = abs(frequency(self.T_endpoint, self.MagneticField.nominal_field)-frequency(self.T_endpoint+20*meV, self.MagneticField.nominal_field))/self.slope
        
        tau_snr_full_length = self.calculate_tau_snr(self.time_window)
        tau_snr_part_length = self.calculate_tau_snr(self.time_window_slope_zero)
        
        
        # use different crlb based on slope
        # delta_E_slope = abs(kin_energy(endpoint_frequency, self.MagneticField.nominal_field)-kin_energy(endpoint_frequency+self.slope*ms, self.MagneticField.nominal_field))
        # logger.info("slope is {} Hz/ms".format(self.slope/Hz*ms))
        # logger.info("slope corresponds to {} meV / ms".format(delta_E_slope/meV))
        # if True: #self.time_window_slope_zero >= self.time_window:
        # logger.info("slope is approximately 0: {} meV".format(delta_E_slope/meV))
        sigma_f_CRLB = np.sqrt((self.CRLB_constant*tau_snr_full_length/self.time_window**3)/(2*np.pi)**2)*self.FrequencyExtraction.CRLB_scaling_factor
        self.best_time_window = self.time_window

        # non constant slope
        self.sigma_f_CRLB_slope_fitted = np.sqrt((20*(self.slope*tau_snr_full_length)**2 + self.CRLB_constant*tau_snr_full_length/self.time_window**3)/(2*np.pi)**2)*self.FrequencyExtraction.CRLB_scaling_factor
        if self.CRLB_constant > 10: sigma_f_CRLB = self.sigma_f_CRLB_slope_fitted
        self.sigma_f_c_CRLB = sigma_f_CRLB
        """
        CRLB_constant = 6
        sigma_CRLB_slope_zero = np.sqrt((CRLB_constant*tau_snr_part_length/self.time_window_slope_zero**3)/(2*np.pi)**2)*self.FrequencyExtraction.CRLB_scaling_factor
        
        
    
        sigma_f_CRLB = np.min([sigma_CRLB_slope_zero, sigma_f_CRLB_slope_fitted])
        
        # logger.info("CRLB options are: {} , {}".format(sigma_CRLB_slope_zero/Hz, sigma_f_CRLB_slope_fitted/Hz))
        self.best_time_window=[self.time_window_slope_zero, self.time_window][np.argmin([sigma_CRLB_slope_zero, sigma_f_CRLB_slope_fitted])]"""
        
        """# uncertainty in alpha
        delta_alpha = 6*sigNoise/(Amplitude*ts**2) * np.sqrt(10/(Nsteps*(Nsteps**4-5*Nsteps**2+4)))
        # uncetainty in sigma_f in Hz due to uncertainty in alpha
        delta_sigma_f_CRLB = delta_alpha * alpha_approx *sigNoise**2/(8*np.pi**2*Amplitude**2*Gdot*sigma_f_CRLB*ScalingFactorCRLB**2)"""

        # sigma_f from Cramer-Rao lower bound in eV
        self.sigma_K_f_CRLB =  e*self.MagneticField.nominal_field/(2*np.pi*endpoint_frequency**2)*sigma_f_CRLB*c0**2
        # delta_sigma_K_f_CRLB = e*self.MagneticField.nominal_field/(2*np.pi*endpoint_frequency**2)*delta_sigma_f_CRLB*c0**2
        
        # sigma_f from pitch angle reconstruction
        if self.FrequencyExtraction.crlb_on_sidebands:
            tau_snr_full_length_sideband = self.calculate_tau_snr(self.time_window, self.FrequencyExtraction.sideband_power_fraction)
            sigma_f_sideband_crlb = np.sqrt((self.CRLB_constant*tau_snr_full_length_sideband/self.time_window**3)/(2*np.pi)**2)*self.FrequencyExtraction.CRLB_scaling_factor
            
            # calculate uncertainty of energy correction for pitch angle
            var_f0_reconstruction = (sigma_f_sideband_crlb**2+sigma_f_CRLB**2)/self.FrequencyExtraction.sideband_order**2 
            max_ax_freq, mean_field, _ = axial_motion(self.MagneticField.nominal_field, 
                                                      self.FrequencyExtraction.minimum_angle_in_bandwidth/deg, 
                                                      self.Experiment.trap_L_over_D*self.CavityRadius()*2, 
                                                      self.FrequencyExtraction.minimum_angle_in_bandwidth/deg, 
                                                      self.T_endpoint, 
                                                      flat_fraction=self.MagneticField.trap_flat_fraction)
            #max_ax_freq = axial_frequency(self.Experiment.L_over_D*self.CavityRadius()*2, 
            #                              self.T_endpoint, 
            #                              self.FrequencyExtraction.minimum_angle_in_bandwidth/deg)
            # 0.16 is the trap quadratic term. 3.8317 is the first 0 in J'0
            var_f0_reconstruction *= (8 * 0.16 * (3.8317*self.Experiment.L_over_D / (np.pi * beta(self.T_endpoint)))**2*max_ax_freq/endpoint_frequency)**2*(1/3.0)
            sigma_f0_reconstruction = np.sqrt(var_f0_reconstruction)
            self.sigma_K_reconstruction = e*self.MagneticField.nominal_field/(2*np.pi*endpoint_frequency**2)*sigma_f0_reconstruction*c0**2
            
            self.sigma_K_f_CRLB = np.sqrt(self.sigma_K_f_CRLB**2 + self.sigma_K_reconstruction**2)


        # combined sigma_f in eV
        sigma_f = np.sqrt(self.sigma_K_f_CRLB**2 + self.FrequencyExtraction.magnetic_field_smearing**2)
        # delta_sigma_f = np.sqrt((delta_sigma_K_f_CRLB**2 + self.FrequencyExtraction.magnetic_field_uncertainty**2)/2)
        if self.FrequencyExtraction.usefixeduncertainty:
            return sigma_f, self.FrequencyExtraction.fixed_relativ_uncertainty*sigma_f
        else:
            raise NotImplementedError("Unvertainty on CRLB for cavity noise calculation is not implemented.")

    def syst_magnetic_field(self):
        """
        Magnetic field uncertanty is in principle generic but its impact on efficiency depends on reconstruction and therefore on detector technology.
        """
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
        if self.MagneticField.useinhomogeneity:
            frac_uncertainty = self.MagneticField.fraction_uncertainty_on_field_broadening
            sigma_meanB = self.MagneticField.sigma_meanb
            sigmaE_meanB = self.BToKeErr(sigma_meanB*B, B)
            sigmaE_r = self.MagneticField.sigmae_r
            sigmaE_theta = self.MagneticField.sigmae_theta
            sigmaE_phi = self.MagneticField.sigmae_theta
            sigma = np.sqrt(sigmaE_meanB**2 + sigmaE_r**2 + sigmaE_theta**2 + sigmaE_phi**2)
            return sigma, frac_uncertainty*sigma
        else:
            return 0, 0
        
    
    # PRINTS
    def print_SNRs(self, rho=None):
        logger.info("SNR parameters:")
        if rho != None:
            logger.warning("Deprecation warning: This function does not modify the number density in the Experiment namespace. Values printed are for pre-set number density.")
        
        track_duration = self.time_window 
        tau_snr = self.calculate_tau_snr(track_duration, sideband_power_fraction=1)
        
        
        eV_bandwidth = np.abs(frequency(self.T_endpoint, self.MagneticField.nominal_field) - frequency(self.T_endpoint + 1*eV, self.MagneticField.nominal_field))
        SNR_1eV = 1/eV_bandwidth/tau_snr
        SNR_track_duration = track_duration/tau_snr
        SNR_1ms = 0.001*s/tau_snr
        
        logger.info("Number density: {} m^-3".format(self.Experiment.number_density*m**3))
        logger.info("Track duration: {}ms".format(track_duration/ms))
        logger.info("tau_SNR: {}s".format(tau_snr/s))
        logger.info("Sampling duration for 1eV: {}ms".format(1/eV_bandwidth/ms))
        
        logger.info("Received power: {}W".format(self.received_power/W))
        logger.info("Noise temperature: {}K".format(self.noise_temp/K))
        logger.info("Noise power in 1eV: {}W".format(self.noise_energy*eV_bandwidth/W))
        logger.info("SNR for 1eV bandwidth: {}".format(SNR_1eV))
        logger.info("SNR 1 eV from temperatures:{}".format(self.received_power/(self.noise_energy*eV_bandwidth)))
        logger.info("SNR for track duration: {}".format(SNR_track_duration))
        logger.info("SNR for 1 ms: {}".format(SNR_1ms))
        
        
        logger.info("Optimum energy window: {} eV".format(self.DeltaEWidth()/eV))
        
        logger.info("CRLB if slope is nonzero and needs to be fitted: {} Hz".format(self.sigma_f_CRLB_slope_fitted/Hz))
        logger.info("CRLB constant: {}".format(self.CRLB_constant))
        
        return self.noise_temp, SNR_1eV, track_duration
    
    
    def print_Efficiencies(self):
        
        if not self.Efficiency.usefixedvalue:
            # radial and detection efficiency are configured in the config file
            logger.info("Radial efficiency: {}".format(self.Efficiency.radial_efficiency))
            logger.info("Detection efficiency: {}".format(self.Efficiency.detection_efficiency))
            logger.info("Pitch angle efficiency: {}".format(self.PitchDependentTrappingEfficiency()))
            logger.info("SRI factor: {}".format(self.Experiment.sri_factor))
            
        logger.info("Effective volume: {} mm^3".format(round(self.effective_volume/mm**3, 3)))
        logger.info("Total efficiency: {}".format(self.effective_volume/self.total_volume))  
