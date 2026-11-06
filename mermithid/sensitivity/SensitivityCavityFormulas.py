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
        # File format: header 'theta_deg, n0, n1, n2, ...', one column per axial
        # harmonic (see ReadPowerFractionsFile for the parity selection rule).
        if hasattr(self.FrequencyExtraction, "use_average_power_fractions"):
            if self.FrequencyExtraction.use_average_power_fractions:
                logger.info("Using average carrier and sideband power fractions")
            else:
                logger.info("Using carrier and sideband (power fractions vs. pitch angle) from file")
                # Read powers from the file and then scale them by power of maximum
                # pitch angle to get power fractions. (Reading extracted into
                # ReadPowerFractionsFile so per-mode files load through identical code.)
                sideband_order = getattr(self.FrequencyExtraction, "sideband_order", 2)
                self.theta_array, harmonic_fractions = \
                    self.ReadPowerFractionsFile(self.FrequencyExtraction.powers_vs_theta_file)
                self.harmonic_power_fraction_arrays = harmonic_fractions
                # The primary mode (assumed TE011) sets the reference power that
                # every other mode's file is divided by.
                self.te011_reference_power = self._last_read_reference_power
                # "carrier" and "sideband" are aliases for harmonics 0 and sideband_order
                # (single-mode path and older callers).
                if 0 in harmonic_fractions:
                    self.carrier_power_fraction_array = harmonic_fractions[0]
                if sideband_order in harmonic_fractions:
                    self.sideband_power_fraction_array = harmonic_fractions[sideband_order]
                elif self.FrequencyExtraction.crlb_on_sidebands:
                    # The sideband harmonic can be missing from the primary file because
                    # axial parity forbids it for the primary mode (odd index: only even
                    # harmonics), e.g. TE011 carrier + TE012 first sideband with
                    # sideband_order = 1. Raise only if the primary's parity allows it
                    # (data missing) or no configured mode can supply it.
                    axial_indices_for_guard = getattr(self.FrequencyExtraction, "axial_mode_indices", [1])
                    if not hasattr(axial_indices_for_guard, "__len__"):
                        axial_indices_for_guard = [int(axial_indices_for_guard)]
                    axial_indices_for_guard = [int(p) for p in axial_indices_for_guard]
                    p0 = axial_indices_for_guard[0]
                    primary_parity_allows = (sideband_order % 2) == (0 if p0 % 2 else 1)
                    other_mode_can_supply = any(
                        (sideband_order % 2) == (0 if p % 2 else 1)
                        for p in axial_indices_for_guard[1:])
                    if primary_parity_allows or not other_mode_can_supply:
                        raise ValueError(
                            "{}: power-fractions file supplies harmonics {} but "
                            "sideband_order = {} is not among them, and no "
                            "configured mode (axial_mode_indices = {}) can supply "
                            "harmonic n{} by axial parity either (odd axial index: "
                            "only even harmonics; even axial index: only odd "
                            "harmonics). The sideband would silently contribute "
                            "zero for every mode. For an odd axial index only EVEN "
                            "harmonics are nonzero, so the sideband column for "
                            "sideband_order={} must be headed 'n{}'.".format(
                                self.FrequencyExtraction.powers_vs_theta_file,
                                sorted(harmonic_fractions), sideband_order,
                                axial_indices_for_guard, sideband_order,
                                sideband_order, sideband_order))

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
        self.SetCouplings()
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

        # Per-mode power-fraction files (default: the primary file). None means the mode
        # has no independent signal but still adds noise and interference. All files are
        # normalised to the primary mode's reference power (see ReadPowerFractionsFile).
        primary_file = getattr(fe, "powers_vs_theta_file", None)
        frac_files = _broadcast("mode_powers_vs_theta_files", len(axial_indices), primary_file)

        self.modes = []
        for k, p in enumerate(axial_indices):
            if k == 0:
                # Primary mode: reference the dict CalcDefaults already built
                fractions = getattr(self, "harmonic_power_fraction_arrays", None)
            elif frac_files[k] is None:
                fractions = None
            elif frac_files[k] == primary_file and hasattr(self, "harmonic_power_fraction_arrays"):
                # Same file as the primary: reuse its dict (no re-read, identical values).
                fractions = self.harmonic_power_fraction_arrays
            else:
                if not hasattr(self, "te011_reference_power"):
                    raise ValueError(
                        "TE01{}: mode_powers_vs_theta_files points at a distinct "
                        "file, which needs the primary mode's reference power, "
                        "but no such reference exists. Set "
                        "use_average_power_fractions = False and provide "
                        "powers_vs_theta_file for the primary mode.".format(p))
                th, fractions = self.ReadPowerFractionsFile(
                    frac_files[k], reference_power=self.te011_reference_power)
                if hasattr(self, "theta_array") and not np.array_equal(th, self.theta_array):
                    logger.warning("TE01{}: interpolating power fractions onto the "
                                   "primary theta grid.".format(p))
                    fractions = {n: np.interp(self.theta_array, th, arr)
                                for n, arr in fractions.items()}
            # Parity sanity check: warn if a file supplies power at a harmonic
            # forbidden by the trap's axial symmetry for this mode's index p
            # (odd p: only even n may be nonzero; even p: only odd n).
            if fractions is not None:
                bad_parity = [n for n, arr in fractions.items()
                             if (n % 2) != (0 if p % 2 else 1) and np.any(np.abs(arr) > 1e-12)]
                if bad_parity:
                    logger.warning("TE01{}: power file has nonzero power at harmonic(s) {} "
                                   "forbidden by axial symmetry for this mode index. Check "
                                   "the file.".format(p, bad_parity))
            self.modes.append(SignalMode(
                name="TE01{}".format(p), axial_mode_index=p,
                q_unloaded=unloaded_qs[k],
                power_fractions=fractions))

        # Check for unconfigured loaded Q values when using powers vs pitch angle data
        if len(self.modes) > 1 and \
                any(m.power_fractions is not None for m in self.modes) and \
                getattr(fe, "mode_loaded_qs", None) is None:
            raise ValueError(
                "{} modes with file-based power fractions are configured "
                "(mode_powers_vs_theta_files / powers_vs_theta_file) but mode_loaded_qs "
                "is not set. The per-mode power fractions already carry the cavity's own "
                "resonant response at whatever loaded Q the simulation was run with; "
                "leaving mode_loaded_qs unset lets SetCouplings derive a DIFFERENT Q_L "
                "per mode from the bandwidth requirement, which makes the noise/bandwidth "
                "calculation inconsistent with the data it is reading (observed: up to "
                "2.4x overstatement of higher-mode power). Set mode_loaded_qs explicitly "
                "to the Q_L the simulation assumed.".format(len(self.modes)))

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

        # --- every mode must couple to at least one port ---
        z_arr = np.array([port.z_position for port in self.ports])
        V_check = np.array([[np.sin(mode.axial_mode_index*np.pi*z/self.cavity_length)
                            for z in z_arr] for mode in self.modes])

        # a mode with zero field at every port cannot be read out
        for a, mode in enumerate(self.modes):
            if np.all(np.abs(V_check[a]) < 1e-12):
                logger.warning(
                    "TE01{} has zero field at EVERY configured port (g_q = 0): "
                    "this mode cannot be read out with the current port "
                    "positions, regardless of coupling.".format(mode.axial_mode_index))


    def ReadPowerFractionsFile(self, filepath, reference_power=None):
        """Read a power-fractions CSV keyed by axial-motion harmonic index and
        return (theta_array [rad], {harmonic_index: fraction_array, ...}).
        """
        theta_array = []
        rows = []
        with open(filepath, 'r') as power_file:
            lines = power_file.readlines()
        header = [h.strip() for h in lines[0].strip().split(",")]
        if len(header) < 2 or not all(h.lower().startswith("n") for h in header[1:]):
            raise ValueError(
                "{}: expected a header 'theta_deg, n0, n1, ...' with one column "
                "per axial-motion harmonic; got {}".format(filepath, header))
        harmonics = [int(h[1:]) for h in header[1:]]
        for i in lines[1:]:
            line = i.strip()
            if not line:
                continue
            parts = line.split(",")
            theta_array.append(float(parts[0]))
            rows.append([float(v) for v in parts[1:]])
        theta_array = np.array(theta_array)*deg
        rows = np.array(rows)                      # shape (n_theta, n_harmonics)

        if reference_power is not None:
            # Non-primary mode: use the shared primary-mode reference (this file may be
            # all zero at 90 degrees, e.g. an even-p mode).
            ref_val = reference_power
        else:
            # Primary mode: reference = total power at the highest pitch angle in the file
            # (the 90-degree carrier for an odd-p mode).
            totals = rows.sum(axis=1)
            ref_index = None
            for idx in np.argsort(theta_array)[::-1]:
                if totals[idx] > 0:
                    ref_index = idx
                    break
            if ref_index is None:
                raise ValueError("{}: every harmonic column is zero at every "
                                 "pitch angle; cannot normalise.".format(filepath))
            if not np.isclose(theta_array[ref_index], np.max(theta_array)):
                logger.warning("{}: this is the PRIMARY mode's file, used as the "
                               "reference for every other mode, but it has no "
                               "power at the maximum pitch angle. Falling back to "
                               "theta = {:.3f} deg; check this is intended."
                               .format(filepath, theta_array[ref_index]/deg))
            ref_val = totals[ref_index]

        self._last_read_reference_power = ref_val    # exposed for the caller to reuse
        fractions = {n: rows[:, k]/ref_val for k, n in enumerate(harmonics)}
        return theta_array, fractions

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
        """Signal power of every mode, on one shared radius sample so the per-mode
        signal_power_vs_r arrays are index-aligned. self.signal_power(_vs_r) is the
        primary mode's. The primary mode passes mode_frequency=None because the
        cavity is tuned to put TE011 on the cyclotron frequency."""
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

    def SetCouplings(self):
        """Set the external and loaded Q of every mode from the port couplings. Each port has one coupling strength
        (config port_strengths), 1/Q_ext,qk = strength_k sin^2(p_q pi z_k / L); the loaded Q follows."""
        cavity_length = self.cavity_length
        port_strengths = getattr(self.FrequencyExtraction, "port_strengths", None)
        if port_strengths is None:
            # no port_strengths in the config: equal strengths giving the primary mode its target loaded Q
            # (mode_loaded_qs[0] if set, else the bandwidth-required value); single mode/port: Q0/Ql - 1
            primary = self.modes[0]
            primary_sw_sq = sum(np.sin(primary.axial_mode_index*np.pi*port.z_position/cavity_length)**2
                                for port in self.ports)
            port_strengths = (1.0/self.CavityLoadedQ() - 1.0/primary.q_unloaded)/primary_sw_sq*np.ones(len(self.ports))
        port_strengths = np.asarray(port_strengths, float)
        for mode in self.modes:
            standing_wave_sq = [np.sin(mode.axial_mode_index*np.pi*port.z_position/cavity_length)**2
                                for port in self.ports]
            # port at a node of the mode does not couple (Q_ext = inf)
            mode.q_externals = {port.name: (1.0/(strength*sw_sq) if strength*sw_sq > 1e-12 else np.inf)
                                for port, strength, sw_sq in zip(self.ports, port_strengths, standing_wave_sq)}
            mode.q_loaded = 1.0/(1.0/mode.q_unloaded
                                 + sum(1.0/q_ext for q_ext in mode.q_externals.values() if np.isfinite(q_ext)))


    def BuildInterferenceMatrix(self):
        """Mode interference matrix R^2 = 1 - R_d^dagger R_d ('SNR of Multimode Signal
        Readout', Sec. 3), with R_d[i,alpha] = sign(sin(p*pi*z_i/L)) * sqrt(Q_alpha/Q_ext).
        Diagonal: fraction of a mode's power not seen by the ports. Off-diagonal: mode
        cross-talk. Diagnostic only: calculate_tau_snr uses the full coupling matrix."""
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

    def _resolve_components(self, components):
        """Translate a components tuple into harmonic indices. Integers pass
        through; the strings "carrier" and "sideband" are kept as aliases for
        0 and FrequencyExtraction.sideband_order, for callers not yet updated
        to the harmonic-index API."""
        sideband_order = getattr(self.FrequencyExtraction, "sideband_order", 2)
        alias = {"carrier": 0, "sideband": sideband_order}
        return tuple(alias.get(c, c) for c in components)

    def _mode_component_sum(self, mode, components):
        """Sum of a mode's power fractions over the requested harmonics. A harmonic
        absent from the file (forbidden by axial parity) contributes 0. No detuning
        correction is applied: the file already includes the simulation's Q_L."""
        if mode.power_fractions is None:
            return 0.0
        total = 0.0
        for c in self._resolve_components(components):
            total = total + mode.power_fractions.get(c, 0.0)
        return total

    def calculate_tau_snr(self, time_window, power_fraction=1, tau_snr_array_for_radii=False, components=None):
        """Multimode tau_SNR for any number of modes and ports. The signal is a set of spectral lines;
        each line has a frequency and a power in each mode. For one line,
        information = a^dagger (kB N)^-1 a, with a = port signal vector and N = port noise covariance,
        both at the line frequency (mode interference is in the off-diagonals of R^dagger R, inverted
        exactly in port_response). Lines are at different frequencies, so their information adds.
        Lines: components given (file fractions) -> one per harmonic h, at f_carrier(theta) - h f_axial(theta)
        for every pitch angle; otherwise (average fractions) -> one per mode, at that mode's resonance."""
        self.CavityLoadedQ()
        self.fft_bandwidth = 3/time_window
        if components is not None and self.modes[0].power_fractions is None:
            raise ValueError("calculate_tau_snr(components=...) requires power fractions "
                             "from file (use_average_power_fractions must be False).")
        cavity_length, modes, ports = self.cavity_length, self.modes, self.ports
        fe = self.FrequencyExtraction
        n_modes, n_ports = len(modes), len(ports)
        cavity_temp = fe.cavity_temperature

        # mode and port couplings (do not depend on frequency)
        loaded_q = np.array([m.q_loaded for m in modes])
        unloaded_q = np.array([m.q_unloaded for m in modes])
        mode_omega = 2*np.pi*np.array([self.CavityModeFrequency(m.axial_mode_index) for m in modes])
        field_sign = np.array([[np.sign(np.sin(m.axial_mode_index*np.pi*p.z_position/cavity_length))
                                for m in modes] for p in ports])
        external_q = np.array([[m.q_externals[p.name] for m in modes] for p in ports])
        port_coupling = field_sign*np.sqrt(loaded_q[None, :]/external_q)   # port x mode
        wall_coupling = np.diag(np.sqrt(loaded_q/unloaded_q))              # wall loss of each mode
        coupling = np.vstack([wall_coupling, port_coupling])
        coupling_sq = coupling.T @ coupling

        # spectral lines: (frequency, power received by each mode)
        lines = []
        if components is None:
            # average fractions: each mode's own power is the line sitting on that mode
            for q, mode in enumerate(modes):
                power = (mode.signal_power_vs_r if tau_snr_array_for_radii else mode.signal_power)*power_fraction
                lines.append((mode_omega[q]/(2*np.pi), [power if i == q else 0.0 for i in range(n_modes)]))
        else:
            # file fractions: the lines move with the pitch angle theta at the bottom of the trap
            theta = fe.minimum_angle_in_bandwidth if tau_snr_array_for_radii else self.theta_array
            axial_frequency, mean_field, _ = axial_motion(self.MagneticField.nominal_field, theta,
                                                          self.Experiment.trap_length, fe.minimum_angle_in_bandwidth,
                                                          self.T_endpoint, flat_fraction=self.MagneticField.trap_flat_fraction)
            carrier_frequency = frequency(self.T_endpoint, mean_field)
            signal_power = modes[0].signal_power_vs_r if tau_snr_array_for_radii else modes[0].signal_power
            for harmonic in self._resolve_components(components):
                lines.append((carrier_frequency - harmonic*axial_frequency,
                              [signal_power*self._mode_component_sum(m, (harmonic,)) for m in modes]))

        information = 0.0
        scalar_output = True
        for line_index, (line_frequency, mode_powers) in enumerate(lines):
            scalar_output = scalar_output and all(np.ndim(x) == 0 for x in [line_frequency] + mode_powers)
            n_samples = max(np.size(x) for x in [line_frequency] + mode_powers)
            line_frequency = np.broadcast_to(np.atleast_1d(line_frequency), (n_samples,)).astype(float)
            mode_powers = np.stack([np.broadcast_to(np.atleast_1d(x), (n_samples,)).astype(float) for x in mode_powers], axis=1)
            omega = 2*np.pi*line_frequency

            # response of the ports to the modes at the line frequency; detuning in loaded half-linewidths
            detuning = loaded_q[None, :]*(omega[:, None]**2 - mode_omega[None, :]**2
                                          + mode_omega[None, :]**2/(4*loaded_q[None, :]**2))/(2*mode_omega[None, :]**2)
            propagator = np.linalg.inv(coupling_sq[None] - 2j*detuning[:, :, None]*np.eye(n_modes)[None])
            port_response = port_coupling[None] @ propagator                       # sample x port x mode

            # readout chain of each port: line and circulator attenuation, incident and added noise
            incident_temp = np.empty((n_samples, n_ports))
            added_temp = np.empty((n_samples, n_ports))
            gains = np.empty((n_samples, n_ports))
            # lossy line with linear temperature gradient temp_start -> temp_end, input temperature temp_in
            def line(temp_in, temp_start, temp_end, a):
                a_safe = np.where(a == 1, 0.5, a)
                return np.where(a == 1, temp_in, a_safe*temp_in + (1-a_safe)*temp_start
                                + (temp_end-temp_start)*(1 + (1-a_safe)/np.log(a_safe)))
            for k, port in enumerate(ports):
                amplifier_temp = port.amplifier_temperature
                line_transmission = db_to_pwr_ratio(port.att_line_db*(1 + line_frequency/(10*GHz)))
                circulator_transmission = db_to_pwr_ratio(port.att_cir_db*(1 + line_frequency/(10*GHz)))
                gains[:, k] = line_transmission*circulator_transmission
                incident_temp[:, k] = line(t_effective(amplifier_temp, line_frequency), amplifier_temp, cavity_temp, line_transmission)
                added_temp[:, k] = (circulator_transmission*line(0.0, cavity_temp, amplifier_temp, line_transmission)
                                    + (1-circulator_transmission)*amplifier_temp
                                    + 2*np.pi*hbar*line_frequency/(kB*port.quantum_amp_efficiency))/gains[:, k]

            # noise at the ports: reflected incident noise + cavity wall noise + added noise
            port_to_port = -np.eye(n_ports)[None] + 2*port_response @ port_coupling.T[None]
            wall_to_port = 2*port_response @ wall_coupling.T[None]
            wall_temp = np.broadcast_to(t_effective(cavity_temp, line_frequency), (n_samples,))
            noise_cov = ((port_to_port*incident_temp[:, None, :]) @ port_to_port.conj().transpose(0, 2, 1)
                         + wall_temp[:, None, None]*(wall_to_port @ wall_to_port.conj().transpose(0, 2, 1))
                         + added_temp[:, :, None]*np.eye(n_ports)[None])
            noise_inverse = np.linalg.inv(kB*noise_cov)

            # port signal; drive amplitude per mode such that |H_q|^2 A_q^2 = P_q
            drive_amplitude = np.sqrt(mode_powers)*np.abs(1 - 2j*detuning)
            port_signal = np.einsum('nkq,nq->nk', port_response, drive_amplitude)
            information = information + np.real(np.einsum('nk,nkl,nl->n', port_signal.conj(), noise_inverse, port_signal))

            if line_index == 0:   # diagnostics for print_SNRs (primary mode alone, referred to the amplifier inputs)
                primary_signal = port_response[:, :, 0]*drive_amplitude[:, 0:1]
                primary_information = np.real(np.einsum('nk,nkl,nl->n', primary_signal.conj(), noise_inverse, primary_signal))
                received_power = np.sum(gains*np.abs(primary_signal)**2, axis=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    noise_temp = received_power/(kB*primary_information)
                self.received_power = received_power[0] if scalar_output else received_power
                self.noise_temp = noise_temp[0] if scalar_output else noise_temp
                self.noise_energy = kB*self.noise_temp

        with np.errstate(divide='ignore'):
            tau_snr = np.where(information > 0, 1.0/information, np.inf)
        return float(tau_snr[0]) if scalar_output else tau_snr


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
                p_array = p_array[:1]
            else:
                p_array = p_array[:-1]
            self.p = np.mean(p_array)

            # Now calculating q for the trap that we have
            # Using the q for the minimum trapped pitch angle
            fc_endpoint_min_theta = frequency(self.T_endpoint, mean_field_array[0])
            self.q = (fc_endpoint_min_theta/fc0_endpoint - 1)/(pitch_comps_for_p_and_q_calc[0])**2

            # Derivative of f_c0 (frequency corrected to B-field at bottom of the trap) with respect to f_c
            dfc0_dfc_array = (1 - 2*self.q*pitch_comps/m/self.p)/(1 - self.q*pitch_comps**2)

            # Derivative of f_c0 with respect to f_lsb (lower sideband frequency)
            dfc0_dlsb_array = 2*self.q*pitch_comps/m/self.p/(1 - self.q*pitch_comps**2)

            # Noise variance term from the carrier frequency uncertainty
            var_noise_from_fc_array = dfc0_dfc_array**2*self.var_f_c_CRLB

            # Noise variance term from the lower sideband frequency uncertainty
            var_noise_from_flsb_array = dfc0_dlsb_array**2*var_f_sideband_crlb

            # Total uncertainty for each pitch angle
            var_f_noise_array = var_noise_from_fc_array + var_noise_from_flsb_array

            # Weighted average of the per-pitch-angle noise variance, weighted by the trapped-pitch-angle distribution. 
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
