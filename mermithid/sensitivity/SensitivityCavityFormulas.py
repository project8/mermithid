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
import matplotlib.pyplot as plt  
from scipy.optimize import nnls

from mermithid.misc.Constants_numericalunits import *
from mermithid.misc.CRESFunctions_numericalunits import *
from mermithid.cavity.HannekeFunctions import *
from mermithid.sensitivity.SensitivityFormulas import *



try:
    from morpho.utilities import morphologging
    logger = morphologging.getLogger(__name__)
except:
    print("Run without morpho!")


from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class SignalMode:
    name:            str
    axial_mode_index: int
    q_unloaded:      float
    q_loaded:        float = 0.0
    q_externals:     dict  = field(default_factory=dict)
    power_fractions: Optional[Dict[str, object]] = None
    signal_power:        float  = 0.0      # emitted power [W] (Phase 7)
    signal_power_vs_r:   object = None      # per-radius array (Phase 7)

@dataclass
class OutputPort:
    """One physical extraction port and its RF noise chain."""
    name:                   str
    z_position:             float
    amplifier_temperature:  float
    att_line_db:            float
    att_cir_db:             float
    quantum_amp_efficiency: float
    

# Wouters functinos
def db_to_pwr_ratio(q_db):
    return 10**(q_db/10)

def axial_motion(magnetic_field, pitch, trap_length, minimum_trapped_pitch, kin_energy, flat_fraction=0.5, trajectory = None):
    # returns the axial motion frequency and a trajectory of point along the axial motion 
    # also return the average magnetic field seen by the electron
    # from z=0 to z=cavity_length/2 with npoints set by the trajectory variable
    # See LUCKEY write-up for a little more on Talia's "flat fraction" trap model

    # Input parameters:
    # pitch and minimum_trapped_pitch are in radians

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

def effective_coupling_multiport(beta_i, beta_total):
    """Equivalent single-port coupling reproducing the multiport transmission.

    The Pn_* helpers above express coupling through the single-port factor
    4c/(1+c)^2, which is only valid when beta is the cavity's ONLY coupling.
    For port i of a multiport cavity, the fraction of incident power that is
    actually lost (dissipated in the cavity walls) is

        T_i = 4*beta_i/(1 + beta_total)^2 ,

    which follows from the resonator S-matrix
        S_ii = (2*beta_i - 1 - beta_total)/(1 + beta_total),
        S_ij = 2*sqrt(beta_i*beta_j)/(1 + beta_total):
    summing sum_j |S_ij|^2 gives exactly 1 - 4*beta_i/(1+beta_total)^2, i.e.
    the extra power absorbed at a port loaded by the other ports is exactly
    restored by cross-port leakage from those ports. This holds when all ports
    carry the same circulator/amplifier noise temperature (the caller warns
    otherwise).

    Inverting 4c/(1+c)^2 = T_i on the over-coupled branch gives the coupling to
    hand the helpers. Identity for beta_i == beta_total, so a single-port
    configuration is numerically unchanged.
    """
    if beta_i <= 0:
        return 0.0
    T = min(4.0*beta_i/(1.0 + beta_total)**2, 1.0)
    return ((2.0 - T) + 2.0*np.sqrt(max(0.0, 1.0 - T)))/T

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


# Calculate threshold z to trap electrons born at some pitch angle theta_start
# Electrons are trapped if they start at z values less than this threshold
# (Considering one axial side of the trap)
def max_z_to_trap_vs_theta_start(theta_start, trap_length, minimum_trapped_pitch, flat_fraction=0.5):
    z_w = trap_length/2
    sec_min = 1/np.cos(minimum_trapped_pitch)
    sin2_min = np.sin(minimum_trapped_pitch)**2
    sin2_start = np.sin(theta_start)**2
    return z_w*flat_fraction + z_w*(1-flat_fraction)*sec_min*np.sqrt(sin2_start - sin2_min)

def dist_of_theta_start_after_trapping(theta_start, trap_length, minimum_trapped_pitch, flat_fraction=0.5):
    # Distribution of theta_start for electrons born uniformly along z
    # Multiplied by sin(theta_start), to account for the birth pitch angles - is this
    # correct? Is another normalization needed after multiplying by sin(theta_start)?
    z_threshold = max_z_to_trap_vs_theta_start(theta_start, trap_length, minimum_trapped_pitch, flat_fraction)
    return z_threshold/(trap_length/2)*np.sin(theta_start)

def theta_bottom_from_theta_start(theta_start, B_min, B_start):
    return np.arcsin(np.sin(theta_start)*np.sqrt(B_min/B_start))

def dist_of_theta_bottom_after_trapping(B_min, theta_start_array, trap_length, minimum_trapped_pitch, flat_fraction=0.5, n_z_start=1000, n_theta_bottom=10):
    z_start_array = np.linspace(0, trap_length/2, n_z_start)
    B_start_array = magnetic_field_flat_harmonic(z_start_array, B_min, trap_length, minimum_trapped_pitch, flat_fraction)
    theta_bottoms = []
    for theta_start in theta_start_array:
        theta_bottoms.append(theta_bottom_from_theta_start(theta_start, B_min, B_start_array))
    theta_bottoms = np.array(theta_bottoms)
    theta_bottoms_bin_centers = np.linspace(minimum_trapped_pitch, np.pi/2, n_theta_bottom)
    bin_size = (np.pi/2 - minimum_trapped_pitch)/n_theta_bottom
    prob_theta_bottom = np.zeros(len(theta_bottoms_bin_centers))
    for i in range(len(theta_bottoms)):
        for j in range(len(theta_bottoms[0])):
            for k in range(len(theta_bottoms_bin_centers)):
                if (theta_bottoms[i][j] >= theta_bottoms_bin_centers[k]-bin_size/2) and (theta_bottoms[i][j] < theta_bottoms_bin_centers[k]+bin_size/2):
                    prob_theta_bottom[k] += dist_of_theta_start_after_trapping(theta_start_array[i], trap_length, minimum_trapped_pitch, flat_fraction)
    normalization = np.sum(prob_theta_bottom)
    prob_theta_bottom = prob_theta_bottom/normalization #Is this the correct approach?
    return theta_bottoms_bin_centers, prob_theta_bottom

"""
figure = plt.figure()
theta_start_array = np.linspace(87*deg, np.pi/2, 1000)
prob_theta_start = dist_of_theta_start_after_trapping(theta_start_array, 4.05*m, 87*deg, flat_fraction=0.75)
plt.scatter(theta_start_array/deg, prob_theta_start)
plt.xlabel("Starting pitch angle $\\theta_{start}$ ($\degree$)", fontsize=14)
plt.ylabel("Probability (arb. units)", fontsize=14)
plt.savefig("test_theta_start_dist.png", dpi=300)
plt.show()
"""


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



###############################################################################
class CavitySensitivity(Sensitivity):
    """
    Documentation:
        * Phase IV sensitivity document: https://www.overleaf.com/project/5de3e02edd267500011b8cc4
        * Talia's sensitivity script: https://3.basecamp.com/3700981/buckets/3107037/documents/2388170839
        * Nick's CRLB for frequency resolution: https://3.basecamp.com/3700981/buckets/3107037/uploads/2009854398
        * Molecular contamination in atomic tritium: https://3.basecamp.com/3700981/buckets/3107037/documents/3151077016
    """
    def __init__(self, config_path, verbose=True):
        Sensitivity.__init__(self, config_path, verbose=verbose)

        # Calc non-config parameters outside of init function:
        ## Allows re-calcing params if config values changed later, e.g. param scans
        self.CalcDefaults(overwrite=False)

    # Add any additional initialization to this function, NOT __INIT__!!
    def CalcDefaults(self, overwrite=False): 
        ###
        #Initialization related to the effective volume:
        ###
        self.Jprime_0 = 3.8317
        self.cavity_freq = frequency(self.T_endpoint, self.MagneticField.nominal_field)
        self.CavityRadius()
        self.cavity_length = 2 * self.cavity_radius * self.Experiment.cavity_L_over_D
        
        #Get trap length from cavity length if not specified
        if ((not hasattr(self.Experiment, 'trap_length')) or overwrite):
            self.Experiment.trap_length = 0.8 * 2 * self.cavity_radius * self.Experiment.cavity_L_over_D
            logger.info("Calc'd trap length: {} m".format(round(self.Experiment.trap_length/m, 3), 2))

        #Geometry sanity: electrons cannot be trapped outside the resonator.
        if self.Experiment.trap_length > self.cavity_length:
            logger.warning("Configured trap_length ({:.2f} m) exceeds the cavity length "
                           "({:.2f} m): the trap extends beyond the resonator. Check "
                           "trap_length and cavity_L_over_D.".format(
                               self.Experiment.trap_length/m, self.cavity_length/m))

        self.Efficiency = NameSpace({opt: eval(self.cfg.get('Efficiency', opt)) for opt in self.cfg.options('Efficiency')})
        self.CavityVolume()

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
        
        # Need to get power fractions before calculating effective volume (given impact on detection efficiency)
        # Power fractions are relative to the power of a 90° carrier electron
        # If average_power_fractions==True, use carrier and sideband pitch power fractions averaged over the usable pitch angle range.
        # If average_power_fractions==False, instead read in a csv file with power fractions vs. pitch angle from simulations.
        # This file should have three columns: pitch angle (in degrees), carrier power, sideband power.
        if hasattr(self.FrequencyExtraction, "use_average_power_fractions"):
            if self.FrequencyExtraction.use_average_power_fractions:
                logger.info("Using average carrier and sideband power fractions")
            else:
                logger.info("Using carrier and sideband (power fractions vs. pitch angle) from file")
                # Read powers from the file and then scale them by power of maximum
                # pitch angle to get power fractions. (Reading extracted into
                # ReadPowerFractionsFile so per-mode files load through identical code.)
                self.theta_array, self.carrier_power_fraction_array, self.sideband_power_fraction_array = \
                    self.ReadPowerFractionsFile(self.FrequencyExtraction.powers_vs_theta_file)

                # Calculating distribution of pitch angles at the bottom of the trap, after trapping
                theta_start_array = np.linspace(self.FrequencyExtraction.minimum_angle_in_bandwidth, np.pi/2, self.Efficiency.n_theta_start_for_trapped_pitch_dist)
                self.theta_bottoms_bin_centers, self.prob_theta_bottom = dist_of_theta_bottom_after_trapping(self.MagneticField.nominal_field, theta_start_array, self.Experiment.trap_length, self.FrequencyExtraction.minimum_angle_in_bandwidth, flat_fraction=self.MagneticField.trap_flat_fraction, n_z_start=self.Efficiency.n_z_for_trapped_pitch_dist, n_theta_bottom=self.Efficiency.n_theta_bottom_for_trapped_pitch_dist)
                
                # Determining which probability corresponds to each pitch angle in self.theta_array
                self.prob_theta_array = np.interp(self.theta_array, self.theta_bottoms_bin_centers, self.prob_theta_bottom)
        
                # Plotting pitch angle distribution
                figure = plt.figure()
                plt.scatter(self.theta_bottoms_bin_centers/deg, self.prob_theta_bottom, s=3, label="Binned distribution", color='red')
                plt.scatter(self.theta_array/deg, self.prob_theta_array, s=2, marker='v', label="Interpolated to $\\theta_{bottom}$ values in simulations", color='blue')
                plt.xlabel("Pitch angle at bottom of trap $\\theta_{bottom}$ ($\degree$)", fontsize=14)
                plt.ylabel("Probability density", fontsize=14)
                plt.legend(fontsize=12, loc='lower center')
                plt.tight_layout()
                plt.savefig("theta_bottom_dist_interpolated_{}.png".format(self.Experiment.exp_label), dpi=300)
                
        #Set up cavity signal modes and readout ports configurations
        self.SetupModesAndPorts()
        self.SolveExternalQ()
        #CavityPower computes the signal power for ALL modes on one shared radius sample
        self.CavityPower()

        #Calculate the effective volume and print out related quantities
        self.EffectiveVolume()
        logger.info("Trap radius: {} cm".format(round(self.cavity_radius/cm, 3), 2))
        logger.info("Total trap volume: {} m^3".format(self.total_trap_volume/m**3))
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
        
        # Number of steps in pitch angle between min_pitch and pi/2 for the frequency noise uncertainty calculation
        self.pitch_steps = 100
        if hasattr(self.FrequencyExtraction, "pitch_steps"):
            self.pitch_steps = self.FrequencyExtraction.pitch_steps
            logger.info("Using configured pitch_steps value")  

        #Just calculated for comparison
        self.larmor_power = rad_power(self.T_endpoint, np.pi/2, self.MagneticField.nominal_field) # currently not used

        # Determining whether to use a fixed detection efficiency or calculate it from the detection threshold
        if not self.Efficiency.usefixedvalue:
            if self.Threshold.use_detection_threshold:
                logger.info("Overriding any detection eff and RF background in the config file; calculating these from the detection_threshold.")
        else:  
            logger.info("Using the detection eff and RF background rate from the config file.")

    def SetupModesAndPorts(self):
        """Build self.modes and self.ports from config. Defaults to a single
        TE011 mode + single center port (reproducing the single-mode setup).
        Multimode opt-in via FrequencyExtraction.axial_mode_indices and
        port_z_fractions; per-port/per-mode keys broadcast from the existing
        scalar config values when omitted."""
        fe = self.FrequencyExtraction

        def _broadcast(name, n, default):
            """Return an n-long list from a config key: list (must match n),
            scalar (repeated), or absent (default repeated)."""
            val = getattr(fe, name, None)
            if val is None:
                return [default]*n
            if hasattr(val, "__len__") and not isinstance(val, str):
                if len(val) != n:
                    raise ValueError("{}: expected {} entries, got {}".format(name, n, len(val)))
                return list(val)
            return [val]*n

        # --- Modes ---
        axial_indices = getattr(fe, "axial_mode_indices", [1])
        if not hasattr(axial_indices, "__len__"):
            axial_indices = [int(axial_indices)]
        axial_indices = [int(p) for p in axial_indices]
        unloaded_qs = _broadcast("mode_unloaded_qs", len(axial_indices), fe.unloaded_q)

        # Per-mode power-fraction files: broadcast from powers_vs_theta_file when
        # omitted; an entry of None means the mode carries no independent signal
        # (it still contributes noise and interference). Each file must be
        # normalized to ITS OWN mode's 90-deg carrier (see ReadPowerFractionsFile).
        primary_file = getattr(fe, "powers_vs_theta_file", None)
        frac_files = _broadcast("mode_powers_vs_theta_files", len(axial_indices), primary_file)

        self.modes = []
        for k, p in enumerate(axial_indices):
            if k == 0:
                # Primary mode: reference the arrays CalcDefaults already built
                # (bit-identity with the single-mode path).
                fractions = None
                if hasattr(self, "carrier_power_fraction_array"):
                    fractions = {"carrier":  self.carrier_power_fraction_array,
                                 "sideband": self.sideband_power_fraction_array}
            elif frac_files[k] is None:
                fractions = None
            elif frac_files[k] == primary_file and hasattr(self, "carrier_power_fraction_array"):
                # Same file as the primary: reuse its arrays (no re-read, identical values).
                fractions = {"carrier":  self.carrier_power_fraction_array,
                             "sideband": self.sideband_power_fraction_array}
            else:
                th, car, sb = self.ReadPowerFractionsFile(frac_files[k])
                if hasattr(self, "theta_array") and not np.array_equal(th, self.theta_array):
                    logger.warning("TE01{}: interpolating power fractions onto the "
                                   "primary theta grid.".format(p))
                    car = np.interp(self.theta_array, th, car)
                    sb  = np.interp(self.theta_array, th, sb)
                fractions = {"carrier": car, "sideband": sb}
            self.modes.append(SignalMode(
                name="TE01{}".format(p), axial_mode_index=p,
                q_unloaded=unloaded_qs[k],
                power_fractions=fractions))

        # --- Ports ---
        z_fractions = getattr(fe, "port_z_fractions", None)
        if z_fractions is None:
            z_fractions = [0.5]                       # single center port (default)
        elif not hasattr(z_fractions, "__len__"):
            z_fractions = [float(z_fractions)]
        n_p = len(z_fractions)
        amp_temps = _broadcast("port_amplifier_temperatures", n_p, fe.amplifier_temperature)
        att_lines = _broadcast("port_att_line_db",            n_p, fe.att_line_db)
        att_cirs  = _broadcast("port_att_cir_db",             n_p, fe.att_cir_db)
        quantum_effs = _broadcast("port_quantum_amp_efficiencies", n_p, fe.quantum_amp_efficiency)

        self.ports = []
        for i in range(n_p):
            self.ports.append(OutputPort(
                name="Port_{}".format(i+1),
                z_position=self.cavity_length*float(z_fractions[i]),
                amplifier_temperature=amp_temps[i],
                att_line_db=att_lines[i], att_cir_db=att_cirs[i],
                quantum_amp_efficiency=quantum_effs[i]))

        # --- validation ---
        for port in self.ports:
            if port.att_line_db > 0 or port.att_cir_db > 0:
                raise ValueError("Port '{}': attenuations must be negative dB losses.".format(port.name))
        for mode in self.modes:
            for port in self.ports:
                if abs(np.sin(mode.axial_mode_index*np.pi*port.z_position/self.cavity_length)) < 1e-12:
                    logger.warning("Mode TE01{} has zero field at port '{}'; it cannot couple there.".format(
                        mode.axial_mode_index, port.name))

    def ReadPowerFractionsFile(self, filepath):
        """Read a power-fractions CSV (pitch angle [deg], carrier power, sideband power;
        one header line) and return (theta_array [rad], carrier_fraction, sideband_fraction),
        normalized to the carrier power at the maximum pitch angle in the file.

        CONVENTION: each mode's file must be normalized to THAT mode's own 90-degree
        carrier power. The absolute inter-mode suppression comes from the generalized
        Hanneke power (CavityPower); normalizing a higher mode's file to TE011
        instead would double-count the mode's Lorentzian suppression."""
        theta_array, carrier_power_array, sideband_power_array = [], [], []
        power_file = open(filepath, 'r')
        for i in power_file.readlines()[1:]: # Skip header line
            line = i.strip()
            theta_array.append(float(line.split(",")[0])) # In degrees
            carrier_power_array.append(float(line.split(",")[1]))
            sideband_power_array.append(float(line.split(",")[2]))
        power_file.close()
        theta_array = np.array(theta_array)*deg # The "*deg" multiplies by np.pi/180
        carrier_power_array = np.array(carrier_power_array)
        sideband_power_array = np.array(sideband_power_array)
        # The calculation below assumes that the file contains a pitch angle very close to 90 degrees:
        max_theta_index = np.argmax(theta_array)
        carrier_fraction = carrier_power_array / carrier_power_array[max_theta_index]
        sideband_fraction = sideband_power_array / carrier_power_array[max_theta_index]
        return theta_array, carrier_fraction, sideband_fraction

    # CAVITY
    def CavityRadius(self):
        axial_mode_index = 1
        self.cavity_radius = c0/(2*np.pi*self.cavity_freq)*np.sqrt(self.Jprime_0**2+axial_mode_index**2*np.pi**2/(4*self.Experiment.cavity_L_over_D**2))
        return self.cavity_radius
    
    def CavityModeFrequency(self, axial_mode_index=1):
        """Resonant frequency of the TE_01l cylindrical cavity mode, l = axial_mode_index.
        l=1 (TE011) is the fundamental that sets the cavity geometry, so
        CavityModeFrequency(1) == self.cavity_freq by construction (round-trip).
        Uses self.Jprime_0 (3.8317) so the single-mode geometry is unchanged."""
        k_r = self.Jprime_0 / self.cavity_radius
        k_z = axial_mode_index * np.pi / self.cavity_length
        return c0 / (2 * np.pi) * np.sqrt(k_r**2 + k_z**2)
    
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
            self.RF_background_rate_per_eV = self.Experiment.RF_background_rate_per_eV    
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
        """Signal power for every mode in self.modes, evaluated on ONE shared
        radius sample with a single wall-clipping mask, so all per-mode
        signal_power_vs_r arrays are index-aligned (required for the per-radius
        tau combination). self.signal_power(_vs_r) alias the primary mode's
        values, so there is a single source of truth and no stale copies.

        The primary mode passes mode_frequency=None: the cavity is tuned so
        TE011 sits on the cyclotron frequency, which also avoids the ~1 ulp
        difference CavityModeFrequency(1) carries."""
        max_ax_freq, mean_field, z_t = axial_motion(self.MagneticField.nominal_field,
                                                  self.FrequencyExtraction.minimum_angle_in_bandwidth,
                                                  self.Experiment.trap_length,
                                                  self.FrequencyExtraction.minimum_angle_in_bandwidth, 
                                                  self.T_endpoint, flat_fraction=self.MagneticField.trap_flat_fraction, trajectory = 1000) #1000

        #The np.random.triangluar function weights the radii, accounting for the fact that there are more electrons at large radii than small ones
        r_sample_size = 50
        if((not self.Efficiency.calculate_det_eff_for_sampled_radii) or (self.Efficiency.usefixedvalue)): r_sample_size = 1000
        radii_sample = np.random.triangular(0, self.cavity_radius, self.cavity_radius, size=r_sample_size)

        mask = None
        for k, mode in enumerate(self.modes):
            f_mode = None if k == 0 else self.CavityModeFrequency(mode.axial_mode_index)
            power_vs_r_with_zeros = np.mean(larmor_orbit_averaged_hanneke_power(radii_sample,
                                                                               z_t, mode.q_loaded,
                                                                               self.cavity_length,
                                                                               self.cavity_radius,
                                                                               self.cavity_freq,
                                                                               mode_frequency=f_mode,
                                                                               axial_mode_index=mode.axial_mode_index), axis=1)
            if mask is None:
                #Remove zeros, since these represent electrons that hit the cavity wall and are not detected.
                #The primary mode defines the mask; wall clipping is geometry-only, so it is mode independent.
                mask = power_vs_r_with_zeros != 0
            mode.signal_power_vs_r = power_vs_r_with_zeros[mask]
            if len(mode.signal_power_vs_r) == 0 or not np.any(mode.signal_power_vs_r):
                logger.warning("CavityPower: TE01{} contributes no signal on the "
                               "sampled radii.".format(mode.axial_mode_index))
                mode.signal_power = 0.0
            else:
                mode.signal_power = np.mean(mode.signal_power_vs_r)

        self._radii_sample = radii_sample
        self._radii_nonzero_mask = mask
        self.signal_power_vs_r = self.modes[0].signal_power_vs_r
        self.signal_power = self.modes[0].signal_power
        return self.signal_power

    #New functions for multi-mode
    def CavityLoadedQ(self, f_mode=None, tuning_pitch=None):
        # Defaults reproduce the single-mode (TE011) result exactly.
        store = (f_mode is None and tuning_pitch is None)
        if f_mode is None:
            f_mode = self.cavity_freq
        if tuning_pitch is None:
            tuning_pitch = self.FrequencyExtraction.minimum_angle_in_bandwidth
        max_ax_freq, mean_field, _ = axial_motion(self.MagneticField.nominal_field,
                                                  tuning_pitch, self.Experiment.trap_length,
                                                  tuning_pitch, self.T_endpoint,
                                                  flat_fraction=self.MagneticField.trap_flat_fraction)
        required_bw_axialfrequency = max_ax_freq * self.FrequencyExtraction.sideband_order
        required_bw_meanfield = np.abs(frequency(self.T_endpoint, mean_field) - f_mode)
        required_bw = np.add(required_bw_axialfrequency, required_bw_meanfield)
        loaded_q = f_mode / required_bw
        # Optional user override of the PRIMARY mode's loaded Q (config).
        # required_bw_* stay as the physical bandwidth requirement.
        user_qls = getattr(self.FrequencyExtraction, "mode_loaded_qs", None)
        if store and user_qls is not None:
            user_q = user_qls[0] if hasattr(user_qls, "__len__") else user_qls
            if user_q is not None:
                if user_q > loaded_q:
                    logger.warning("Configured loaded Q {:.0f} exceeds the bandwidth-required "
                                   "{:.0f}: mode is narrower than signal + sidebands; this "
                                   "signal loss is NOT modeled.".format(user_q, loaded_q))
                loaded_q = user_q
        if store:   # single-mode path: keep the existing side effects
            self.required_bw_axialfrequency = required_bw_axialfrequency
            self.required_bw = required_bw
            self.loaded_q = loaded_q
        return loaded_q
    
    def SolveExternalQ(self, target_qls=None, target_weights=None, ortho_weight=1e10, ortho_tol=1e-6):
        """Solve per-port external Qs so each mode hits its target loaded Q while
        keeping modes orthogonal. Single mode/port reduces to coupling = Q0/Ql - 1."""
        n_modes = len(self.modes)
        L = self.cavity_length
        min_pitch = self.FrequencyExtraction.minimum_angle_in_bandwidth
        # Bandwidth-required loaded Q per mode: always computed, both as the
        # default target and as the reference for the achieved-Q check below.
        computed_qls = [self.CavityLoadedQ(f_mode=self.CavityModeFrequency(m.axial_mode_index),
                                           tuning_pitch=min_pitch) for m in self.modes]
        if target_qls is None:
            computed = computed_qls
            user_qls = getattr(self.FrequencyExtraction, "mode_loaded_qs", None)
            if user_qls is None:
                target_qls = computed
            else:
                if not hasattr(user_qls, "__len__"):
                    user_qls = [user_qls]*n_modes
                if len(user_qls) != n_modes:
                    raise ValueError("mode_loaded_qs: expected {} entries, got {}".format(n_modes, len(user_qls)))
                target_qls = [u if u is not None else c for u, c in zip(user_qls, computed)]
                for a, (u, c) in enumerate(zip(user_qls, computed)):
                    if u is not None and u > c:
                        logger.warning("TE01{}: configured loaded Q {:.0f} exceeds bandwidth-required "
                                       "{:.0f}; signal loss not modeled.".format(
                                           self.modes[a].axial_mode_index, u, c))
        system_A, system_b = [], []
        if target_weights is None:
            # Primary mode's bandwidth target takes priority; higher modes are
            # bonus channels and absorb the geometric compromise.
            target_weights = [100.0] + [1.0]*(n_modes-1)
        for alpha, mode in enumerate(self.modes):
            required_inv_q_ext = max(0.0, 1.0/target_qls[alpha] - 1.0/mode.q_unloaded)
            system_A.append([target_weights[alpha]*np.sin(mode.axial_mode_index*np.pi*port.z_position/L)**2
                             for port in self.ports])
            system_b.append(target_weights[alpha]*required_inv_q_ext)
        for alpha in range(n_modes):
            for beta in range(alpha+1, n_modes):
                system_A.append([np.sin(self.modes[alpha].axial_mode_index*np.pi*port.z_position/L)
                                 * np.sin(self.modes[beta].axial_mode_index*np.pi*port.z_position/L)
                                 * ortho_weight for port in self.ports])
                system_b.append(0.0)
        # Prefer UNIFORM port couplings among degenerate optima: the tau_SNR
        # combination (CRLB doc Eq. snrsum) assumes independent recovered
        # per-mode noise, which holds for uniform antennas on the DST grid;
        # a vertex solution (all coupling on one port) makes every mode share
        # one amplifier (rho -> 1) and the 1/tau summation over-counts.
        n_ports_solver = len(self.ports)
        positive_b = [v for v in system_b[:n_modes] if v > 0]
        if n_ports_solver > 1 and positive_b:
            reg = 1e-3*max(positive_b)
            for i in range(n_ports_solver - 1):
                row = [0.0]*n_ports_solver
                row[i] = reg
                row[i+1] = -reg
                system_A.append(row)
                system_b.append(0.0)
        A_mat, b_vec = np.array(system_A), np.array(system_b)
        x_opt, x_res = nnls(A_mat, b_vec)
        if n_modes > 1:
            ortho_error = np.max(np.abs(A_mat[n_modes:] @ x_opt)) / ortho_weight
            if ortho_error > ortho_tol:
                logger.warning("Fixed z-positions cause mode hybridization "
                               "(max error {:.2e})".format(ortho_error))
        for alpha, mode in enumerate(self.modes):
            total_inv_q_ext = 0.0
            for i, port in enumerate(self.ports):
                field_sq = np.sin(mode.axial_mode_index*np.pi*port.z_position/L)**2
                port_inv_q = x_opt[i]*field_sq
                mode.q_externals[port.name] = (1.0/port_inv_q if port_inv_q > 1e-10 else np.inf)
                total_inv_q_ext += port_inv_q
            mode.q_loaded = (1.0/(1.0/mode.q_unloaded + total_inv_q_ext)
                             if total_inv_q_ext > 0 else mode.q_unloaded)
        # A mode whose ACHIEVED loaded Q exceeds its bandwidth-required value is
        # narrower than its own signal + axial sidebands; that signal loss is not
        # modelled, so its contribution to the tau_SNR combination is optimistic.
        # (The configured-Q override warns separately in CavityLoadedQ; this
        # catches the case where the port geometry, not the user, causes it.)
        if not getattr(self, "_solver_bw_warned", False):
            for alpha, mode in enumerate(self.modes):
                required_ql = computed_qls[alpha]
                if mode.q_loaded > required_ql*(1.0 + 1e-6):
                    self._solver_bw_warned = True
                    logger.warning("TE01{}: solved loaded Q {:.0f} exceeds the bandwidth-required "
                                   "{:.0f} (+{:.0%}); the mode is narrower than its signal and the "
                                   "resulting capture loss is NOT modelled, so its contribution to "
                                   "the tau_SNR combination is optimistic.".format(
                                       mode.axial_mode_index, mode.q_loaded, required_ql,
                                       mode.q_loaded/required_ql - 1.0))
        return x_opt, x_res
        
    def BuildInterferenceMatrix(self):
        """Inter-mode interference matrix R^2_{alpha,beta} ('SNR of Multimode
        Signal Readout', Sec. 3):  R^2 = 1_N - R_d^dagger R_d, where
        R_d[i,alpha] = sign(sin(p*pi*z_i/L)) * sqrt(Q_alpha / Q_ext_{alpha,i}).
        Diagonal = fraction of each mode's power in unobserved channels
        (single mode/port: R^2 = 1 - W_i); off-diagonals = mode cross-talk.
        Diagnostic only -- not folded into tau_SNR (the omitted correction is
        |dtau/tau| ~ f_RF * R^2_aa; see notebook Sec. 8)."""
        n_modes = len(self.modes)
        n_ports = len(self.ports)
        L = self.cavity_length

        R_d = np.zeros((n_ports, n_modes), dtype=complex)
        for i, port in enumerate(self.ports):
            for a, mode in enumerate(self.modes):
                q_ext = mode.q_externals.get(port.name, np.inf)
                if not np.isfinite(q_ext) or q_ext <= 0:
                    continue
                field = np.sin(mode.axial_mode_index*np.pi*port.z_position/L)
                sign  = np.sign(field) if field != 0 else 1.0
                R_d[i, a] = sign*np.sqrt(mode.q_loaded/q_ext)

        R2 = np.eye(n_modes, dtype=complex) - R_d.conj().T @ R_d
        max_crosstalk = (np.max(np.abs(R2 - np.diag(np.diag(R2))))
                         if n_modes > 1 else 0.0)
        if max_crosstalk > 1e-3:
            logger.warning("Interference matrix: mode cross-talk |R2_ab| up to "
                           "{:.2e}; zero-forcing isolation may be degraded.".format(max_crosstalk))
        return R2
           
    # SENSITIVITY
    # see parent class in SensitivityFormulas.py
 

    # SYSTEMATICS
    # Generic systematics are implemented in the parent class in SensitivityFormulas.py

    def calculate_tau_snr(self, time_window, power_fraction=1, tau_snr_array_for_radii=False,
                          components=None):
        """Multimode tau_SNR per Rick's multi-mode SNR document.

        components: optional tuple of power_fractions keys, e.g. ("carrier",) or
        ("carrier", "sideband"). When given, each mode's signal is
        mode power * sum of ITS OWN power_fractions[component] arrays (per-mode
        pitch dependence); a mode with power_fractions=None contributes no signal
        (but still contributes noise and interference). When None (legacy), the
        power_fraction argument is applied to the primary mode as before and
        higher modes fall back to their scalar mean power."""
        self.CavityLoadedQ()
        fft_bandwidth = 3/time_window
        self.fft_bandwidth = fft_bandwidth
        L = self.cavity_length
        n_modes = len(self.modes)

        if components is not None and self.modes[0].power_fractions is None:
            raise ValueError("calculate_tau_snr(components=...) requires power fractions "
                             "from file (use_average_power_fractions must be False).")

        # Per-mode signal powers for the interference ratios. With components these
        # are per-theta arrays (mode power x that mode's own fractions); ratios use
        # the radius-averaged scalar mode powers as the absolute scale.
        scalar_powers = [m.signal_power for m in self.modes]
        if components is not None:
            mode_powers = [scalar_powers[a]*sum(m.power_fractions[c] for c in components)
                           if m.power_fractions is not None else 0.0
                           for a, m in enumerate(self.modes)]
        else:
            mode_powers = scalar_powers

        # --- interference factors (document main result), diagonal-normalized ---
        R2 = self.BuildInterferenceMatrix()
        # per-mode cavity-generated noise P_N,alpha (cavity paper expression;
        # document: 'same as a single port at that temperature with the loaded Q')
        PN_modes = np.empty(n_modes)
        inv_qext_tot_list = np.empty(n_modes)
        for a, mode in enumerate(self.modes):
            f_a = self.CavityModeFrequency(mode.axial_mode_index)
            inv_qext_tot = sum(1.0/q for q in mode.q_externals.values()
                               if np.isfinite(q) and q > 0)
            inv_qext_tot_list[a] = inv_qext_tot
            coupling_tot = mode.q_unloaded*inv_qext_tot
            PN_modes[a] = Pn_cavity(self.FrequencyExtraction.cavity_temperature,
                                    coupling_tot, mode.q_loaded, fft_bandwidth, f_a)

        def _F(X):
            """Diagonal-normalized interference factors; entries of X may be
            scalars or per-theta arrays (elementwise). Exactly 1 for one mode."""
            if n_modes == 1:
                return [1.0]
            F = []
            for a in range(n_modes):
                d = abs(1.0 - R2[a, a])**2
                Xa = np.asarray(X[a], dtype=float)
                if d < 1e-12 or np.all(Xa <= 0):
                    F.append(1.0)
                    continue
                s = np.ones_like(Xa, dtype=complex)
                for b in range(n_modes):
                    Xb = np.asarray(X[b], dtype=float)
                    safe_den = np.where(Xa > 0, Xa, 1.0)
                    ratio = np.where((Xa > 0) & (Xb > 0), np.sqrt(Xb/safe_den), 0.0)
                    s = s - R2[a, b]*ratio
                Fa = np.abs(s)**2/d
                F.append(np.where(Xa > 0, Fa, 1.0))
            return F
        F_sig = _F(mode_powers)
        F_N   = _F(list(PN_modes))
        if any(np.any(np.abs(np.asarray(f) - 1) > 0.5) for f in F_sig) or \
           any(np.any(np.abs(np.asarray(f) - 1) > 0.5) for f in F_N):
            logger.warning("Interference factors far from 1: strong port hybridization; "
                           "the independent-mode treatment (per-mode tau + F factors) "
                           "is outside its validated regime.")

        inv_tau_total = None
        v_list = []
        # The multiport coupling correction (effective_coupling_multiport) relies
        # on cross-port leakage cancelling the extra absorption, which requires
        # equal circulator/amplifier noise at every port.
        if len(self.ports) > 1 and not getattr(self, "_amptemp_warned", False):
            amp_temps = set(float(p.amplifier_temperature) for p in self.ports)
            if len(amp_temps) > 1:
                self._amptemp_warned = True
                logger.warning("Ports have differing amplifier temperatures; the multiport "
                               "coupling correction assumes equal circulator noise at all "
                               "ports (cross-port leakage cancellation).")
        for m_idx, mode in enumerate(self.modes):
            f_mode = self.CavityModeFrequency(mode.axial_mode_index)
            q_l    = mode.q_loaded
            # All modes share one radius sample and mask (CavityPower), so every
            # mode's signal_power_vs_r is index-aligned and usable directly.
            base = mode.signal_power_vs_r if tau_snr_array_for_radii else mode.signal_power
            if components is not None:
                if mode.power_fractions is None:
                    continue        # mode carries no independent signal
                Pe = base*sum(mode.power_fractions[c] for c in components)
            else:
                Pe = base*power_fraction

            n_ports = len(self.ports)
            Pn_total_list = np.empty(n_ports)
            Pn_cav_list   = np.empty(n_ports)
            signs         = np.empty(n_ports)
            weight_factor = np.empty(n_ports)
            for i, port in enumerate(self.ports):
                q_ext = mode.q_externals.get(port.name, np.inf)
                if np.isfinite(q_ext):
                    coupling = mode.q_unloaded/q_ext
                    W_i      = q_l/q_ext
                else:
                    coupling = 0.0; W_i = 0.0
                # Multiport correction: the Pn_* helpers assume the port is the
                # cavity's only coupling. Convert this port's beta_i to the
                # equivalent single-port coupling that reproduces the true
                # multiport loss 4*beta_i/(1+beta_total)^2. Identity when there
                # is one port, so the single-mode path is unchanged.
                beta_total = mode.q_unloaded*inv_qext_tot_list[m_idx]
                coupling_eff = effective_coupling_multiport(coupling, beta_total)
                field = np.sin(mode.axial_mode_index*np.pi*port.z_position/L)
                signs[i] = np.sign(field) if field != 0 else 1.0
                att_line_db_freq = port.att_line_db*(1+f_mode/(10*GHz))
                att_cir_db_freq  = port.att_cir_db*(1+f_mode/(10*GHz))
                att_tot = db_to_pwr_ratio(att_line_db_freq+att_cir_db_freq)
                weight_factor[i] = np.sqrt(W_i*att_tot)
                Pn_at_amp = Pn_dut_entrance(self.FrequencyExtraction.cavity_temperature,
                                            port.amplifier_temperature,
                                            att_line_db_freq, att_cir_db_freq,
                                            coupling_eff, f_mode, fft_bandwidth, q_l)
                tn_amp = f_mode*hbar*2*np.pi/kB/port.quantum_amp_efficiency
                Pn_at_amp += kB*tn_amp*fft_bandwidth
                # Chain/RF noise split: subtract the cavity noise as EMBEDDED in
                # Pn_dut_entrance (same effective coupling) to isolate the
                # RF-only part, which is per-port and not mixed (SNR doc Sec. 3).
                Pn_cav_embedded = Pn_cavity(self.FrequencyExtraction.cavity_temperature,
                                            coupling_eff, q_l, fft_bandwidth, f_mode)*att_tot
                Pn_rf_i = Pn_at_amp - Pn_cav_embedded
                # Cavity thermal noise: the TOTAL delivered power is set by the
                # total coupling (document: 'the same as a single port at that
                # temperature with the associated loaded Q'); each port carries
                # its share (1/q_ext,i)/(sum 1/q_ext). Evaluating the single-port
                # formula at each per-port coupling and summing would create
                # thermal power from nowhere (transmission 4b/(1+b)^2 grows as
                # coupling is subdivided).
                share_i = ((1.0/q_ext)/inv_qext_tot_list[m_idx]
                           if (np.isfinite(q_ext) and inv_qext_tot_list[m_idx] > 0) else 0.0)
                Pn_cav_list[i]   = PN_modes[m_idx]*share_i*att_tot*F_N[m_idx]
                Pn_total_list[i] = Pn_rf_i + Pn_cav_list[i]

            Sigma = np.diag(Pn_total_list).astype(float)
            for i in range(n_ports):
                for j in range(n_ports):
                    if i != j:
                        Sigma[i, j] = signs[i]*signs[j]*np.sqrt(Pn_cav_list[i]*Pn_cav_list[j])

            # --- max-SNR (optimal) port combination for this mode ---
            # Per-port signal amplitude v_i = sign * sqrt(W_i * att_i)  (the R_d
            # entries of the multimode-readout document). SNR_opt quadratic form:
            v = signs*weight_factor
            v_list.append(v)
            qform = v @ np.linalg.solve(Sigma, v)          # = v^T Sigma^-1 v  [1/W]
            tau_mode = 1.0/(Pe*F_sig[m_idx]*qform*fft_bandwidth)

            with np.errstate(divide='ignore', invalid='ignore'):
                inv = np.where(tau_mode > 0, 1.0/tau_mode, 0.0)
            inv_tau_total = inv if inv_tau_total is None else inv_tau_total + inv

            if m_idx == 0:
                # diagnostic bookkeeping; reduces to the single-port values exactly
                self.received_power = Pe*np.sum(v**2)
                self.noise_temp     = np.sum(v**2)/qform/(kB*fft_bandwidth)
                self.noise_energy   = kB*self.noise_temp

        # Validity check for the 1/tau summation (CRLB doc Eq. snrsum): the
        # combination assumes independent recovered per-mode noise, i.e. near-
        # orthogonal per-mode port-amplitude vectors. Report the cross-mode
        # correlation when it is materially violated (doc, off-diag appendix).
        if len(v_list) > 1:
            Vm = np.array(v_list)
            Gm = Vm @ Vm.T
            norms = np.sqrt(np.clip(np.diag(Gm), 1e-300, None))
            rho = Gm/np.outer(norms, norms) - np.eye(len(v_list))
            max_rho = float(np.max(np.abs(rho)))
            if max_rho > 0.1 and not getattr(self, "_rho_warned", False):
                self._rho_warned = True
                logger.warning("Cross-mode recovered-noise correlation up to {:.2f}: "
                               "the 1/tau summation over-counts correlated mode "
                               "contributions (CRLB doc, off-diagonal appendix). "
                               "Check SolveExternalQ coupling distribution. "
                               "(Warning shown once per instance.)".format(max_rho))

        with np.errstate(divide='ignore', invalid='ignore'):
            tau_snr = np.where(inv_tau_total > 0, 1.0/inv_tau_total, np.inf)
        return float(tau_snr) if np.ndim(tau_snr) == 0 else tau_snr
                
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
            # Non-zero, fitted slope. 
            # Doesn't assume that alpha*T/2 << omega_c, since it includes the 5*eta/(1-eta) term in Eq. 25 of Joe's write-up: https://3.basecamp.com/3700981/buckets/3107037/documents/6331876030.
            # CODE IMPLEMENTATION NEEDS TO BE DOUBLE-CHECKED BY CONSIDERING AN EXPERIMENT WITH LARGE-ISH ETA.
            # The first term relies on the relation delta_t_start = sqrt(20)*tau_snr. This is from Equation 6.40 of Nick's thesis,
            # derived in Appendix A and verified with an MC study.
            # Using a factor of 23 instead of 20, from re-calculating Nick's integrals (though this derivation is approximate).
            # Nick's derivation uses an expression for P_fa assuming the phase is known. 
            # The phase won't be known, but it's more difficult to determine the unknown-phase expression.
            # Working on that.
            return self.FrequencyExtraction.CRLB_scaling_factor*(23*(self.slope*tau_SNR)**2 + tau_SNR/self.time_window**3*(96 - 6*5*self.eta/(1+self.eta)))/(2*np.pi)**2

    
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
        
        if self.FrequencyExtraction.use_average_power_fractions:
            tau_snr_full_length = self.calculate_tau_snr(self.time_window, self.FrequencyExtraction.carrier_power_fraction)
        else:
            tau_snr_full_length = self.calculate_tau_snr(self.time_window, components=("carrier",))
            tau_snr_full_length = tau_snr_full_length[:len(self.theta_array)-1] #Cut out theta=pi/2, since sideband power is 0 there, resulting in infinite tau_snr.

        #Calculate the frequency variance from the CRLB
        self.var_f_c_CRLB = self.frequency_variance_from_CRLB(tau_snr_full_length)
        self.best_time_window = self.time_window

        # sigma_f from pitch angle reconstruction
        if self.FrequencyExtraction.crlb_on_sidebands:
            #Calculate noise contribution to uncertainty, including energy correction for pitch angle.
            #This comes from section 6.1.9 of the CDR.

            if self.FrequencyExtraction.use_average_power_fractions:
                tau_snr_full_length_sideband = self.calculate_tau_snr(self.time_window, self.FrequencyExtraction.sideband_power_fraction)
            else:
                tau_snr_full_length_sideband = self.calculate_tau_snr(self.time_window, components=("sideband",))
                tau_snr_full_length_sideband = tau_snr_full_length_sideband[:len(self.theta_array)-1] #Cut out theta=pi/2, since sideband power is 0 there, resulting in infinite tau_snr.
            
            # (sigmaf_lsb)^2:
            var_f_sideband_crlb = self.frequency_variance_from_CRLB(tau_snr_full_length_sideband)
            m = self.FrequencyExtraction.sideband_order #For convenience

            # Defining array of pitch angle complements (pi/2 - theta) used when calculating
            # the parameters describing the track shape (p and q)
            thetas_for_p_and_q_calc = np.linspace(self.FrequencyExtraction.minimum_angle_in_bandwidth, 90*deg, self.pitch_steps)
            pitch_comps_for_p_and_q_calc = np.pi/2 - thetas_for_p_and_q_calc
            
            # Defining array of pitch angle complement values over which we calculate the
            # resolution contribution from noise.
            if self.FrequencyExtraction.use_average_power_fractions:
                pitch_comps = pitch_comps_for_p_and_q_calc
            else:
                pitch_comps = np.pi/2 - self.theta_array[:len(self.theta_array)-1] #Cut out theta=pi/2 since sideband power is 0 there, resulting in infinite tau_snr.

            #Define the trap parameter p based on the relation between the trap length and the cavity mode
            #This p is for a box trap
            self.p_box = np.pi*beta(self.T_endpoint)*self.cavity_radius/self.Jprime_0/self.Experiment.trap_length

            #Now find p for the actual trap that we have
            #Using the average p across the pitch angle range
            ax_freq_array, mean_field_array, z_t = axial_motion(self.MagneticField.nominal_field,
                                    thetas_for_p_and_q_calc, self.Experiment.trap_length,
                                    self.FrequencyExtraction.minimum_angle_in_bandwidth, 
                                    self.T_endpoint, flat_fraction=self.MagneticField.trap_flat_fraction)
            fc0_endpoint = self.cavity_freq
            p_array = ax_freq_array/fc0_endpoint/pitch_comps_for_p_and_q_calc #An array
            if self.FrequencyExtraction.use_average_power_fractions:
                p_array = p_array[:1] #Cut out theta=pi/2 (ill defined there)
            self.p = np.mean(p_array)

            # Now calculating q for the trap that we have
            # Using the q for the minimum trapped pitch angle
            fc_endpoint_min_theta = frequency(self.T_endpoint, mean_field_array[0])
            self.q = (fc_endpoint_min_theta/fc0_endpoint - 1)/(pitch_comps_for_p_and_q_calc[0])**2

            # Derivative of f_c0 (frequency corrected to B-field at bottom of the trap) with respect to f_c
            dfc0_dfc_array = 0.5*(1 - (1 - 4*self.q*pitch_comps/m/self.p + self.q*pitch_comps**2)/(1 - self.q*pitch_comps**2))

            # Derivative of f_c0 with respect to f_lsb (lower sideband frequency)
            dfc0_dlsb_array = 0.5 - 2*self.q*pitch_comps/m/self.p/(1 - self.q*pitch_comps**2)

            # Noise variance term from the carrier frequency uncertainty
            var_noise_from_fc_array = dfc0_dfc_array**2*self.var_f_c_CRLB

            # Noise variance term from the lower sideband frequency uncertainty
            var_noise_from_flsb_array = dfc0_dlsb_array**2*var_f_sideband_crlb

            # Total uncertainty for each pitch angle
            var_f_noise_array = var_noise_from_fc_array + var_noise_from_flsb_array

            # Next, we average over sigma_noise values.
            # This is a quadrature sum average weighted by the pitch angle distribution,
            # reflecting that the detector response function could be constructed by sampling
            # from many normal distributions with different standard deviations (sigma_noise_array),
            # then finding the standard deviation of the full group of sampled values.
            # IS THE BELOW CORRECT?
            prob_theta_array_without_pi_over_2 = self.prob_theta_array[:len(self.theta_array)-1] #Cut out theta=pi/2 since sideband power is 0 there, resulting in infinite tau_snr.
            self.sigma_f_noise = np.sqrt(np.sum(var_f_noise_array*prob_theta_array_without_pi_over_2)/np.sum(self.prob_theta_array))

        else:
            self.sigma_f_noise = np.sqrt(self.var_f_c_CRLB)

        # Convert uncertainty from frequency to energy
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
            sigmaE_phi = self.MagneticField.sigmae_phi
            sigma = np.sqrt(sigmaE_meanB**2 + sigmaE_r**2 + sigmaE_theta**2 + sigmaE_phi**2)
            return sigma, frac_uncertainty*sigma
        else:
            return 0, 0

    def det_efficiency_track_duration(self):
        """
        Detection efficiency implemented based on René's slides, with faster and stable implementation using Gauss-Laguerre quadrature (G-L method):
        https://3.basecamp.com/3700981/buckets/3107037/documents/8013439062
        Gauss-Laguerre Quadrature: https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature

        The following changes were made to the original integral to fit the G-L method:
        Original integrand: ∫[0 to \inf] ncx2(df=2, nc=t/τ).sf(thres) * (1/μ) * exp(-t/μ) dt
        
        Where. t = track_duration, μ (\mu) = mean_track_duration, τ (\tau) = tau_snr_ex_carrier, thres = detection_threshold

        We do the change of variable, x = t / μ. So, t = x μ, or, dt = μ dx

        Substituting into the original integral: 
        ∫[0 to \inf] ncx2(df=2, nc=xμ/τ).sf(thres) * (1/μ) * exp(-x) μ dx
        The μ's cancel out, and the integral takes the form:
        ∫[0 to \inf] f(x) * exp(-x) dx
        where, f(x) = ncx2(df=2, nc=xμ/τ).sf(thres)
        
        Parameters: None
        
        Returns: avg_efficiency (float): SNR and threshold dependent detection efficieny.
                   
        Notes: Also check the antenna paper for more details. Especially the section on the signal detection with matched filtering.
        """
        # Calculate the mean track duration
        # TO-DO: Only do the lines below ones for a given density; don't repeat for each threshold being scanned ...
        mean_track_duration = track_length(self.Experiment.number_density, self.T_endpoint, molecular=(not self.Experiment.atomic))
        if self.FrequencyExtraction.use_average_power_fractions:
            tau_snr_ex_total = self.calculate_tau_snr(mean_track_duration, self.FrequencyExtraction.carrier_power_fraction + self.FrequencyExtraction.sideband_power_fraction, tau_snr_array_for_radii=self.Efficiency.calculate_det_eff_for_sampled_radii)
        else:
            tau_snr_ex_total = self.calculate_tau_snr(mean_track_duration, components=("carrier", "sideband"), tau_snr_array_for_radii=self.Efficiency.calculate_det_eff_for_sampled_radii)
        if isinstance(tau_snr_ex_total, float):
            tau_snr_ex_total = [tau_snr_ex_total]

        # Roots and weights for the Laguerre polynomial
        x, w = roots_laguerre(100) #n=100 is the number of quadrature points
        
        # Scale the track duration to match the form of Gauss-Laguerre quadrature
        scaled_x = x * mean_track_duration # scaled_x = xμ

        # Evaluate the non-central chi-squared dist values at the scaled quadrature points
        sf_values = np.array([ncx2(df=2, nc=2 * scaled_x / tau_snr).sf(self.Threshold.detection_threshold) for tau_snr in tau_snr_ex_total])

        # Calculate and return the integration result from weighted sum
        eff_for_each_r_and_theta = np.sum(w * sf_values, axis=1)

        # Average efficiencies over the sampled electron radii and pitch angles.
        # Calculation below accounts for trapped pitch angle distribution (self.prob_theta_array).
        # Weighting for radial distribution is accounted for in sampling, earlier.
        if self.FrequencyExtraction.use_average_power_fractions:
            avg_efficiency = np.mean(eff_for_each_r_and_theta)
        else:
            if not self.Efficiency.calculate_det_eff_for_sampled_radii:
                avg_efficiency = np.sum(self.prob_theta_array * eff_for_each_r_and_theta)/sum(self.prob_theta_array)
            else:
                #Sum over radii with equal weights, and sum over pitch angles with probability weights
                #I'm not sure if I get the axes right, below.
                avg_efficiency = np.sum(eff_for_each_r_and_theta, axis=0)/len(self.signal_power_vs_r)
                avg_efficiency = np.sum(self.prob_theta_array * avg_efficiency)/sum(self.prob_theta_array)
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
        if self.FrequencyExtraction.use_average_power_fractions:
            tau_snr_ex_carrier = self.calculate_tau_snr(track_duration, self.FrequencyExtraction.carrier_power_fraction)
        else:
            tau_snr_ex_carrier = np.mean(self.calculate_tau_snr(track_duration, components=("carrier",)))

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
        logger.info("SNRs of carriers (90°, used in calc) for track duration at optimum density: {}, {}".format(SNR_track_duration_90deg, SNR_track_duration_ex_carrier))
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
