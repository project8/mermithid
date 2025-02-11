'''
Class calculating neutrino mass sensitivities based on analytic formulas from CDR.
Author: R. Reimann, C. Claessens, T. E. Weiss, W. Van De Pontseele
Date: 06/07/2023
Updated: December 2024

The statistical method and formulas are described in
CDR (CRES design report, Section 1.3) https://www.overleaf.com/project/5b9314afc673d862fa923d53.
'''
import numpy as np
from scipy.stats import ncx2, chi2
from scipy.special import roots_laguerre

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
    
    #pitch = pitch/180*np.pi
    #minimum_trapped_pitch = minimum_trapped_pitch/180*np.pi

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


def axial_frequency_box(length, kin_energy, max_pitch_angle=86*np.pi/180):
    #pitch_max = max_pitch_angle/180*np.pi
    return (beta(kin_energy)*c0*np.cos(max_pitch_angle)) / (2*length)

def mean_field_frequency_variation(cyclotron_frequency, length_diameter_ratio, max_pitch_angle=86*np.pi/180, q=0.16):
    # Because of the different electron trajectories in the trap,
    # An electron will see a slightly different magnetic field
    # depending on its position in the trap, especially the pitch angle.
    # This is a rough estimate of the mean field variation, inspired by calculation performed by Rene.
    #y = (90-max_pitch_angle)/4
    phi_rad = (np.pi/2-max_pitch_angle)
    return q*phi_rad**2*cyclotron_frequency*(10/length_diameter_ratio)
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

# Trapping efficiency from axial field variation.
def trapping_efficiency(z_range, bg_magnetic_field, min_pitch_angle, trap_flat_fraction = 0.5):

    """
    Calculate the trapping efficiency for a given trap length and flat fraction.

    The trapping efficiency is computed using the formula:
        epsilon(z) = sqrt(1 - B(z)/B_max(z))
    where B(z) is the magnetic field at position z, and B_max(z) is the maximum magnetic field along the z axis.

    Parameters
    ----------
    z_range : float
        The axial range (in z-direction, from trap center) over which electron trapping happens.
    bg_magnetic_field : float
        The background magnetic field.
    min_pitch_angle : float
        Minimum pitch angle to be trapped.
    trap_flat_fraction : float, optional
        Flat fraction of the trap. Default is 0.5.

    Returns
    -------
    mean_efficiency : float
        The mean trapping efficiency across the trap z-range.

    Notes
    -----
    The magnetic field profile is computed using the `magnetic_field_flat_harmonic` function, currently it only produces z-profile of the trap without radial variation. 
    No radial field variation was assumed for this calculation.
    The mean trapping efficiency is averaged over the region where the trapping field exists.
    """

    zs = np.linspace(-z_range, z_range, 500)

    profiles = []
    #Collect z profile of the magnetic field
    for z in zs:
        profiles.append(magnetic_field_flat_harmonic(z, bg_magnetic_field, z_range*2, min_pitch_angle, trap_flat_fraction))
    
    #Calculate maximum trapping field along z (Bz_max)
    maximum_Bz = max(profiles)

    #Calculate mean trapping efficiency using mean of epsilon(z) = sqrt(1-B(z)/B_max(z)) at z = 0
    mean_efficiency = np.mean(np.array([np.sqrt(1-b_at_z/maximum_Bz) for b_at_z in profiles]))
    
    return mean_efficiency



###############################################################################cd
class CavitySensitivity(Sensitivity):
    """
    Documentation:
        * Phase IV sensitivity document: https://www.overleaf.com/project/5de3e02edd267500011b8cc4
        * Talia's sensitivity script: https://3.basecamp.com/3700981/buckets/3107037/documents/2388170839
        * Nick's CRLB for frequency resolution: https://3.basecamp.com/3700981/buckets/3107037/uploads/2009854398
        * Molecular contamination in atomic tritium: https://3.basecamp.com/3700981/buckets/3107037/documents/3151077016
    """
    def __init__(self, config_path):
        Sensitivity.__init__(self, config_path)
        
        ###
        #Initialization related to the effective volume:
        ###
        self.Jprime_0 = 3.8317
        self.cavity_freq = frequency(self.T_endpoint, self.MagneticField.nominal_field)
        self.CavityRadius()
        
        #Get trap length from cavity length if not specified
        if not hasattr(self.Experiment, 'trap_length'):
            self.Experiment.trap_length = 0.8 * 2 * self.cavity_radius * self.Experiment.cavity_L_over_D
            logger.info("Calc'd trap length: {} m".format(round(self.Experiment.trap_length/m, 3), 2))

        self.Efficiency = NameSpace({opt: eval(self.cfg.get('Efficiency', opt)) for opt in self.cfg.options('Efficiency')})
        self.CavityVolume()
        self.CavityPower()

        #Calculate position dependent trapping efficiency
        self.pos_dependent_trapping_efficiency = trapping_efficiency( z_range = self.Experiment.trap_length /2,
                                                                    bg_magnetic_field = self.MagneticField.nominal_field, 
                                                                    min_pitch_angle = self.FrequencyExtraction.minimum_angle_in_bandwidth, 
                                                                    trap_flat_fraction = self.MagneticField.trap_flat_fraction
                                                                    )          
        
        #We may decide to remove the "Threshold" section and move the threshold-related parameters to the "Efficiency" section.
        if not self.Efficiency.usefixedvalue:
            self.Threshold = NameSpace({opt: eval(self.cfg.get('Threshold', opt)) for opt in self.cfg.options('Threshold')})
   
        #Cyclotron radius is sometimes used in the effective volume calculation
        self.cyc_rad = cyclotron_radius(self.cavity_freq, self.T_endpoint) 

        #Assigning the background constant if it's not in the config file
        if hasattr(self.Experiment, "bkgd_constant"):
            self.bkgd_constant = self.Experiment.bkgd_constant
            logger.info("Using background rate constant of {}/eV/s".format(self.bkgd_constant))
        else:
            self.bkgd_constant = 1
            logger.info("Using background rate constant of 1/eV/s") 
        
        #Calculate the effective volume and print out related quantities
        self.EffectiveVolume()
        logger.info("Trap radius: {} cm".format(round(self.cavity_radius/cm, 3), 2))
        logger.info("Total trap volume: {} m^3".format(round(self.total_trap_volume/m**3), 3))
        logger.info("Cyclotron radius: {}m".format(self.cyc_rad/m))
        if self.use_cyc_rad:
            logger.info("Using cyclotron radius as unusable distance from wall, for radial efficiency calculation")

        ####
        #Initialization related to the energy resolution:
        ####
        #No longer using this CRLB_constant. If this change sticks, will remove it.
        self.CRLB_constant = 6
        if hasattr(self.FrequencyExtraction, "crlb_constant"):
            self.CRLB_constant = self.FrequencyExtraction.crlb_constant
            logger.info("Using configured CRLB constant")      
        
        #Number of steps in pitch angle between min_pitch and pi/2 for the frequency noise uncertainty calculation
        self.pitch_steps = 100
        if hasattr(self.FrequencyExtraction, "pitch_steps"):
            self.pitch_steps = self.FrequencyExtraction.pitch_steps
            logger.info("Using configured pitch_steps value")  

        #Just calculated for comparison
        self.larmor_power = rad_power(self.T_endpoint, np.pi/2, self.MagneticField.nominal_field) # currently not used
        
        if self.Threshold.use_detection_threshold:
            logger.info("Overriding any detection eff and RF background in the config file; calculating these from the detection_threshold.")
        else:  
            logger.info("Using the detection eff and RF background rate from the config file.")


    # CAVITY
    def CavityRadius(self):
        axial_mode_index = 1
        self.cavity_radius = c0/(2*np.pi*self.cavity_freq)*np.sqrt(self.Jprime_0**2+axial_mode_index**2*np.pi**2/(4*self.Experiment.cavity_L_over_D**2))
        return self.cavity_radius
    
    def CavityVolume(self):
        #radius = 0.5*wavelength(self.T_endpoint, self.MagneticField.nominal_field)
        self.total_cavity_volume = 2*self.cavity_radius*self.Experiment.cavity_L_over_D*np.pi*(self.cavity_radius)**2*self.Experiment.n_cavities
        
        logger.info("Frequency: {} MHz".format(round(self.cavity_freq/MHz, 3)))
        logger.info("Wavelength: {} cm".format(round(wavelength(self.T_endpoint, self.MagneticField.nominal_field)/cm, 3)))
        logger.info("Cavity radius: {} cm".format(round(self.cavity_radius/cm, 3)))
        logger.info("Cavity length: {} cm".format(round(2*self.cavity_radius*self.Experiment.cavity_L_over_D/cm, 3)))
        logger.info("Total cavity volume: {} m^3".format(round(self.total_cavity_volume/m**3, 3)))\
        
        return self.total_cavity_volume
    

    # ELECTRON TRAP
    def TrapVolume(self):
        # Total volume of the electron traps in all cavities
        self.total_trap_volume = self.Experiment.trap_length*np.pi*(self.cavity_radius)**2*self.Experiment.n_cavities
        return self.total_trap_volume


    
    def EffectiveVolume(self):
        self.total_trap_volume = self.TrapVolume()

        if self.Efficiency.usefixedvalue:
            self.effective_volume = self.total_trap_volume * self.Efficiency.fixed_efficiency
            self.use_cyc_rad = False
        else:
            #Detection efficiency
            if self.Threshold.use_detection_threshold:
                #Calculating the detection efficiency given the SNR of data and the threshold.
                #If the config file contains a detection effciency or RF background rate, they are overridden.
                self.assign_background_rate_from_threshold()
                self.assign_detection_efficiency_from_threshold()
            else:
                #Using the inputted detection efficiency and RF background rate from the config file.
                self.detection_efficiency = self.Efficiency.detection_efficiency
                self.RF_background_rate_per_eV = self.Experiment.RF_background_rate_per_eV    


            #Radial efficiency
            if self.Efficiency.unusable_dist_from_wall >= self.cyc_rad:
                self.radial_efficiency = (self.cavity_radius - self.Efficiency.unusable_dist_from_wall)**2/self.cavity_radius**2
                self.use_cyc_rad = False
            else:
                self.radial_efficiency = (self.cavity_radius - self.cyc_rad)**2/self.cavity_radius**2
                self.use_cyc_rad = True
            
            #Efficiency from a cut during analysis on the axial frequency
            self.fa_cut_efficiency = trapping_efficiency(z_range = self.Experiment.trap_length /2,
                                                                    bg_magnetic_field = self.MagneticField.nominal_field, 
                                                                    min_pitch_angle = self.Efficiency.min_pitch_used_in_analysis, 
                                                                    trap_flat_fraction = self.MagneticField.trap_flat_fraction
                                                                    )/self.pos_dependent_trapping_efficiency 
            
            #The effective volume includes the three efficiency factors above, as well as the trapping efficiency
            self.effective_volume = self.total_trap_volume*self.radial_efficiency*self.detection_efficiency*self.fa_cut_efficiency*self.pos_dependent_trapping_efficiency   
            
        # The "signal rate improvement" factor can be toggled to test the increase in statistics required to reach some sensitivity
        self.effective_volume*=self.Experiment.sri_factor
        return self.effective_volume
        

    def BoxTrappingEfficiency(self):
        self.box_trapping_efficiency = np.cos(self.FrequencyExtraction.minimum_angle_in_bandwidth)
        return self.box_trapping_efficiency

    def TrapLength(self):
        self.Experiment.trap_length = 0.8 * 2 * self.cavity_radius * self.Experiment.cavity_L_over_D
        logger.info("Calc'd trap length: {} m".format(round(self.Experiment.trap_length/m, 3), 2))

    def CavityPower(self):
        #Jprime_0 = 3.8317
        max_ax_freq, mean_field, z_t = axial_motion(self.MagneticField.nominal_field,
                                                  self.FrequencyExtraction.minimum_angle_in_bandwidth,
                                                  self.Experiment.trap_length,
                                                  self.FrequencyExtraction.minimum_angle_in_bandwidth, 
                                                  self.T_endpoint, flat_fraction=self.MagneticField.trap_flat_fraction, trajectory = 1000) #1000

        #The np.random.triangluar function weights the radii, accounting for the fact that there are more electrons at large radii than small ones
        power_vs_r_with_zeros = np.mean(larmor_orbit_averaged_hanneke_power(np.random.triangular(0, self.cavity_radius, self.cavity_radius, size=50),
                                                                            z_t, self.CavityLoadedQ(), 
                                                                            2*self.Experiment.cavity_L_over_D*self.cavity_radius, 
                                                                            self.cavity_radius, 
                                                                            self.cavity_freq), axis=1)
        #Remove zeros, since these represent electrons that hit the cavity wall and are not detected
        self.signal_power_vs_r = power_vs_r_with_zeros[power_vs_r_with_zeros != 0]

        self.signal_power = np.mean(self.signal_power_vs_r)
        return self.signal_power
    

    def CavityLoadedQ(self):
        # Using Wouter's calculation:
        # Total required bandwidth is the sum of the endpoint region and the axial frequency. 
        # I will assume the bandwidth is dominated by the sidebands and not by the energy ROI
        
        #self.loaded_q =1/(0.22800*((90-self.FrequencyExtraction.minimum_angle_in_bandwidth)*np.pi/180)**2+2**2*0.01076**2/(4*0.22800))

        endpoint_frequency = self.cavity_freq
        #required_bw_axialfrequency = axial_frequency(self.Experiment.cavity_L_over_D*self.CavityRadius()*2, 
        #                                             self.T_endpoint, 
        #                                             self.FrequencyExtraction.minimum_angle_in_bandwidth/deg)
        max_ax_freq, mean_field, _ = axial_motion(self.MagneticField.nominal_field,
                                                  self.FrequencyExtraction.minimum_angle_in_bandwidth,
                                                  self.Experiment.trap_length,
                                                  self.FrequencyExtraction.minimum_angle_in_bandwidth, 
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

    def calculate_tau_snr(self, time_window, power_fraction=1, tau_snr_array_for_radii=False):
        """
        power_fraction may be used as a carrier or a sideband power fraction,
        relative to the power of a 90 degree carrier.
        """
        endpoint_frequency = self.cavity_freq
    
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
        if tau_snr_array_for_radii:
            Pe = self.signal_power_vs_r * power_fraction
        else:
            Pe = self.signal_power * power_fraction
        
        P_signal_received = Pe*db_to_pwr_ratio(att_cir_db_freq+att_line_db_freq)
        self.received_power = P_signal_received
        tau_snr = kB*tn_system_fft/P_signal_received
        self.noise_energy = kB*tn_system_fft

        # end of Wouter's calculation
        return tau_snr
        
    """
    def print_SNRs(self, rho_opt):
        tau_snr = self.calculate_tau_snr(self.time_window, sideband_power_fraction=1)
        logger.info("tau_SNR: {}s".format(tau_snr/s))
        eV_bandwidth = np.abs(frequency(self.T_endpoint, self.MagneticField.nominal_field) - frequency(self.T_endpoint + 1*eV, self.MagneticField.nominal_field))
        SNR_1eV = 1/eV_bandwidth/tau_snr
        track_duration = track_length(rho_opt, self.T_endpoint, molecular=(not self.Experiment.atomic))
        SNR_track_duration = track_duration/tau_snr
        SNR_1ms = 0.001*s/tau_snr
        logger.info("SNR for 1eV bandwidth: {}".format(SNR_1eV))
        logger.info("SNR 1 eV from temperatures:{}".format(self.received_power/(self.noise_energy*eV_bandwidth)))
        logger.info("Track duration: {}ms".format(track_duration/ms))
        logger.info("Sampling duration for 1eV: {}ms".format(1/eV_bandwidth/ms))
        logger.info("SNR for track duration: {}".format(SNR_track_duration))
        logger.info("SNR for 1 ms: {}".format(SNR_1ms))
        logger.info("Received power: {}W".format(self.received_power/W))
        logger.info("Noise power in 1eV: {}W".format(self.noise_energy*eV_bandwidth/W))
        logger.info("Noise temperature: {}K".format(self.noise_temp/K))
        logger.info("Opimtum energy window: {} eV".format(self.DeltaEWidth()/eV))
    """

    def frequency_variance_from_CRLB(self, tau_SNR):
        self.eta = self.slope*self.time_window/(4*self.cavity_freq*np.pi)
        if self.eta < 1e-6:
            # This is for the case where the track is flat (almost no slope), and where we
            # treat it as a pure sinusoid (don't fit the slope when extracting the frequency).
            # Applies for a complex signal.
            return self.FrequencyExtraction.CRLB_scaling_factor*(6*tau_SNR/self.time_window**3)/(2*np.pi)**2 
        else:
            # Non-zero, fitted slope. Still assumes that alpha*T/2 << omega_c.
            # The first term relies on the relation delta_t_start = sqrt(20)*tau_snr. This is from Equation 6.40 of Nick's thesis,
            # derived in Appendix A and verified with an MC study.
            # Using a factor of 23 instead of 20, from re-calculating Nick's integrals (though this derivation is approximate).
            # Nick's derivation uses an expression for P_fa - we need to check if it's consistent with what Rene uses now.
            return self.FrequencyExtraction.CRLB_scaling_factor*(23*(self.slope*tau_SNR)**2 + 96*tau_SNR/self.time_window**3)/(2*np.pi)**2

    
    def syst_frequency_extraction(self):
        # cite{https://3.basecamp.com/3700981/buckets/3107037/uploads/2009854398} (Section 1.2, p 7-9)
        # Are we double counting the antenna collection efficiency? We use it here. Does it also impact the effective volume, v_eff ?
        
        if self.FrequencyExtraction.UseFixedValue:
            sigma = self.FrequencyExtraction.Default_Systematic_Smearing
            delta = self.FrequencyExtraction.Default_Systematic_Uncertainty
            return sigma, delta
        
       
        endpoint_frequency = self.cavity_freq
        # using Pe and alpha (aka slope) from above
        Pe = self.signal_power #/self.FrequencyExtraction.mode_coupling_efficiency
        
        self.slope = endpoint_frequency * 2 * np.pi * Pe/me/c0**2 # track slope
        self.time_window = track_length(self.Experiment.number_density, self.T_endpoint, molecular=(not self.Experiment.atomic))
        
        self.time_window_slope_zero = abs(self.cavity_freq-frequency(self.T_endpoint+20*meV, self.MagneticField.nominal_field))/self.slope
        
        tau_snr_full_length = self.calculate_tau_snr(self.time_window, self.FrequencyExtraction.carrier_power_fraction)
        tau_snr_part_length = self.calculate_tau_snr(self.time_window_slope_zero, self.FrequencyExtraction.carrier_power_fraction)
        
        #Calculate the frequency variance from the CRLB
        self.var_f_c_CRLB = self.frequency_variance_from_CRLB(tau_snr_full_length)
        self.best_time_window = self.time_window

        # sigma_f from pitch angle reconstruction
        if self.FrequencyExtraction.crlb_on_sidebands:
            #Calculate noise contribution to uncertainty, including energy correction for pitch angle.
            #This comes from section 6.1.9 of the CDR.

            tau_snr_full_length_sideband = self.calculate_tau_snr(self.time_window, self.FrequencyExtraction.sideband_power_fraction)
            # (sigmaf_lsb)^2:
            var_f_sideband_crlb = self.frequency_variance_from_CRLB(tau_snr_full_length_sideband)
            #var_f_sideband_crlb = self.FrequencyExtraction.CRLB_scaling_factor*(self.CRLB_constant*tau_snr_full_length_sideband/self.time_window**3)/(2*np.pi)**2

            m = self.FrequencyExtraction.sideband_order #For convenience

            #Define phi_max, corresponding to the minimum pitch angle
            phi_max = np.pi/2 - self.FrequencyExtraction.minimum_angle_in_bandwidth
            phis = np.linspace(0, phi_max, self.pitch_steps)

            #Define the trap parameter p based on the relation between the trap length and the cavity mode
            #This p is for a box trap
            self.p_box = np.pi*beta(self.T_endpoint)*self.cavity_radius/self.Jprime_0/self.Experiment.trap_length

            #Now find p for the actual trap that we have
            #Using the average p across the pitch angle range
            ax_freq_array, mean_field_array, z_t = axial_motion(self.MagneticField.nominal_field,
                                    np.pi/2-phis, self.Experiment.trap_length,
                                    self.FrequencyExtraction.minimum_angle_in_bandwidth, 
                                    self.T_endpoint, flat_fraction=self.MagneticField.trap_flat_fraction)
            fc0_endpoint = self.cavity_freq
            p_array = ax_freq_array/fc0_endpoint/phis
            self.p = np.mean(p_array[1:]) #Cut out theta=pi/2 (ill defined there)

            #Now calculating q for the trap that we have
            #Using the q for the minimum trapped pitch angle
            fc_endpoint_min_theta = frequency(self.T_endpoint, mean_field_array[self.pitch_steps-1])
            self.q = (fc_endpoint_min_theta/fc0_endpoint - 1)/(phis[self.pitch_steps-1])**2

            #Derivative of f_c0 (frequency corrected to B-field at bottom of the trap) with respect to f_c
            dfc0_dfc_array = 0.5*(1 - (1 - 4*self.q*phis/m/self.p + self.q*phis**2)/(1 - self.q*phis**2))

            #Derivative of f_c0 with respect to f_lsb (lower sideband frequency)
            dfc0_dlsb_array = 0.5 - 2*self.q*phis/m/self.p/(1 - self.q*phis**2)

            #Noise variance term from the carrier frequency uncertainty
            var_noise_from_fc_array = dfc0_dfc_array**2*self.var_f_c_CRLB

            #Noise variance term from the lower sideband frequency uncertainty
            var_noise_from_flsb_array = dfc0_dlsb_array**2*var_f_sideband_crlb

            #Total uncertainty for each pitch angle
            var_f_noise_array = var_noise_from_fc_array + var_noise_from_flsb_array

            #Next, we average over sigma_noise values.
            #This is a quadrature sum average,
            #reflecting that the detector response function could be constructed by sampling
            #from many normal distributions with different standard deviations (sigma_noise_array),
            #then finding the standard deviation of the full group of sampled values.
            self.sigma_f_noise = np.sqrt(np.sum(var_f_noise_array)/self.pitch_steps)

        else:
            self.sigma_f_noise = np.sqrt(self.var_f_c_CRLB)

        #Convert uncertainty from frequency to energy
        self.sigma_K_noise = e*self.MagneticField.nominal_field/(2*np.pi*endpoint_frequency**2)*self.sigma_f_noise*c0**2

        # combined sigma_f in eV
        sigma_f = np.sqrt(self.sigma_K_noise**2 + self.FrequencyExtraction.magnetic_field_smearing**2)
        # delta_sigma_f = np.sqrt((delta_sigma_K_f_CRLB**2 + self.FrequencyExtraction.magnetic_field_uncertainty**2)/2)
        if self.FrequencyExtraction.usefixeduncertainty:
            return sigma_f, self.FrequencyExtraction.fixed_relativ_uncertainty*sigma_f
        else:
            raise NotImplementedError("Uncertainty on CRLB for cavity noise calculation is not implemented.")

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

    def det_efficiency_track_duration(self):
        """
        Detection efficiency implemented based on René's slides, with faster and stable implementation using Gauss-Laguerre quadrature:
        https://3.basecamp.com/3700981/buckets/3107037/documents/8013439062
        Gauss-Laguerre Quadrature: https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature

        Returns:
        detection_efficiency : float
            SNR and threshold dependent detection efficieny.
                   
        Notes
        -----
        Also check the antenna paper for more details. 
        Especially the section on the signal detection with matched filtering.
        """
        # Calculate the mean track duration
        #FIX: Only do the lines below ones for a given density; don't repeat for each threshold being scanned ...
        mean_track_duration = track_length(self.Experiment.number_density, self.T_endpoint, molecular=(not self.Experiment.atomic))
        tau_snr_ex_total = self.calculate_tau_snr(mean_track_duration, self.FrequencyExtraction.carrier_power_fraction + self.FrequencyExtraction.sideband_power_fraction, tau_snr_array_for_radii=self.Efficiency.calculate_det_eff_for_sampled_radii)
        if isinstance(tau_snr_ex_total, float):
            tau_snr_ex_total = [tau_snr_ex_total]

        # Roots and weights for the Laguerre polynomial
        x, w = roots_laguerre(100) #n=100 is the number of quadrature points
        
        # Scale the track duration to match the form of Gauss-Laguerre quadrature
        scaled_x = x * mean_track_duration

        # Evaluate the non-central chi-squared dist values at the scaled quadrature points
        sf_values = np.array([ncx2(df=2, nc=2 * scaled_x / tau_snr).sf(self.Threshold.detection_threshold) for tau_snr in tau_snr_ex_total])

        # Calculate and return the integration result from weighted sum
        eff_for_each_r = np.sum(w * sf_values, axis=1)

        #Average efficiencies over the sampled electron radii. Weighting for radial distribution is accounted for in sampling, earlier.
        avg_efficiency = np.mean(eff_for_each_r) 
        return avg_efficiency

    def assign_detection_efficiency_from_threshold(self):
        self.detection_efficiency = self.det_efficiency_track_duration()
        return self.detection_efficiency

    def rf_background_rate_cavity(self):
        # Detection efficiency implemented based on René's slides
        # https://3.basecamp.com/3700981/buckets/3107037/documents/8013439062
        # Also check the antenna paper for more details, especially the section
        # on the signal detection with matched filtering.
        # The background constant will need to be determined from Monte Carlo simulations.
        return chi2(df=2).sf(self.Threshold.detection_threshold)*self.bkgd_constant/(eV*s)

    def assign_background_rate_from_threshold(self):
        self.RF_background_rate_per_eV = self.rf_background_rate_cavity()
        return self.RF_background_rate_per_eV

    
        
    
    # PRINTS
    def print_SNRs(self, rho=None):
        #logger.warning("Deprecation warning: This function does not modify the number density in the Experiment namespace. Values printed are for pre-set number density.")
        
        logger.info("**SNR parameters**:")
        if rho == None:
            track_duration = self.time_window
            logger.info("SNR-related parameters are printed for pre-set number density.")
        else:
            track_duration = track_length(rho, self.T_endpoint, molecular=(not self.Experiment.atomic))
        
        tau_snr_90deg = self.calculate_tau_snr(track_duration, power_fraction=1)
        #For an example carrier:
        tau_snr_ex_carrier = self.calculate_tau_snr(track_duration, self.FrequencyExtraction.carrier_power_fraction)
        
        
        eV_bandwidth = np.abs(self.cavity_freq - frequency(self.T_endpoint + 1*eV, self.MagneticField.nominal_field))
        SNR_1eV_90deg = 1/eV_bandwidth/tau_snr_90deg
        SNR_track_duration_90deg = track_duration/tau_snr_90deg
        SNR_1ms_90deg = 0.001*s/tau_snr_90deg

        SNR_1eV_ex_carrier = 1/eV_bandwidth/tau_snr_ex_carrier
        SNR_track_duration_ex_carrier = track_duration/tau_snr_ex_carrier
        SNR_1ms_ex_carrier = 0.001*s/tau_snr_ex_carrier
        
        logger.info("Number density: {} m^-3".format(self.Experiment.number_density*m**3))
        logger.info("Track duration: {}ms".format(track_duration/ms))
        logger.info("tau_SNR for 90° carrier: {}s".format(tau_snr_90deg/s))
        logger.info("tau_SNR for carrier used in calculation (see config file): {}s".format(tau_snr_ex_carrier/s))
        logger.info("Sampling duration for 1eV: {}ms".format(1/eV_bandwidth/ms))
        
        logger.info("Received power for 90° carrier: {}W".format(self.received_power/W))
        logger.info("Noise temperature: {}K".format(self.noise_temp/K))
        logger.info("Noise power in 1eV: {}W".format(self.noise_energy*eV_bandwidth/W))
        logger.info("SNRs of carriers (90°, used in calc) for 1eV bandwidth: {}, {}".format(SNR_1eV_90deg, SNR_1eV_ex_carrier))
        #logger.info("SNR 1 eV from temperatures:{}".format(self.received_power/(self.noise_energy*eV_bandwidth)))
        logger.info("SNRs of carriers (90°, used in calc) for track duration: {}, {}".format(SNR_track_duration_90deg, SNR_track_duration_ex_carrier))
        logger.info("SNR of carriers (90°, used in calc) for 1 ms: {}, {}".format(SNR_1ms_90deg, SNR_1ms_ex_carrier))
        
        
        logger.info("Optimum energy window: {} eV".format(self.DeltaEWidth()/eV))
        
        #logger.info("CRLB if slope is nonzero and needs to be fitted: {} Hz".format(np.sqrt(self.var_f_CRLB_slope_fitted)/Hz))
        #logger.info("CRLB constant: {}".format(self.CRLB_constant))
        logger.info("**Done printing SNR parameters.**")
        
        return self.noise_temp, SNR_1eV_90deg, track_duration
    
    
    def print_Efficiencies(self):
        
        logger.info("Effective volume: {} mm^3".format(round(self.effective_volume/mm**3, 3)))
        logger.info("Total efficiency: {}".format(self.effective_volume/self.total_trap_volume))  
    
        if not self.Efficiency.usefixedvalue:
            # radial and detection efficiency are configured in the config file
            logger.info("Radial efficiency: {}".format(self.radial_efficiency))
            logger.info("Detection efficiency: {}".format(self.detection_efficiency))
            #logger.info("Detection efficiency integration error: {}".format(self.abs_err))
            logger.info("Trapping efficiency: {}".format(self.pos_dependent_trapping_efficiency))
            logger.info("Efficiency from axial frequency cut: {}".format(self.fa_cut_efficiency))
            logger.info("SRI factor: {}".format(self.Experiment.sri_factor))




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


"""fc_endpoint_array = frequency(self.T_endpoint, mean_field_array)
self.q_array = 1/phis**2*(fc_endpoint_array/fc0_endpoint - 1)
self.q = np.mean(self.q_array[1:])"""
