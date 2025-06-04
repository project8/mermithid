import numpy as np
from scipy import special
from numericalunits import T, kB, hbar, e, me, eV, c0, eps0, m, Hz, MHz

# Physics constants
endpoint = 18563.251*eV # 30472.604*eV # Krypton
bessel_derivative_zero = special.jnp_zeros(0, 1)[0]

def magneticfield_from_frequency(cyclotron_frequency, kin_energy=endpoint):
    # magnetic field for a given cyclotron frequency
    return 2*np.pi*me*cyclotron_frequency/e*gamma(kin_energy)

def gamma(kin_energy):
    return kin_energy/(me*c0**2) + 1

def beta_factor(kin_energy=endpoint):
    # electron speed at kin_energy
    return np.sqrt(kin_energy**2+2*kin_energy*me*c0**2)/(kin_energy+me*c0**2)

def larmor_radius(magnetic_field, kin_energy=endpoint, pitch=np.pi/2):
    transverse_speed = beta_factor(kin_energy)*c0*np.sin(pitch)
    # Larmor radius, exact, can also be apporximated by beta*Xprime01*cavity_radius.
    return gamma(kin_energy)*me*transverse_speed/(e*magnetic_field)

# Hanneke factor for TE011 mode, for an electron at fixed location (r_position, z_position).
# Note: We don't "halve the power," given this from Rick: https://3.basecamp.com/3700981/buckets/3107037/uploads/8101664058. 
def hanneke_factor_TE011(r_position, z_position, loaded_Q, l_cav, r_cav, cyclotron_frequency, mode_frequency):
    global bessel_derivative_zero
    # Calculate the lambda_mnp_squared factor
    mode_p = 1 # TE011 mode
    z_L = l_cav/2
    # Calculate the lambda_mnp_squared factor
    classical_electron_radius_c_squared = e**2 / (4 * np.pi * eps0 * me)
    lambda_bessel_part = (1 / (special.jvp(0, bessel_derivative_zero, n=2) * special.jv(0, bessel_derivative_zero)))
    constant_factor = - 2 * lambda_bessel_part * classical_electron_radius_c_squared
    #constant_factor_unitless = constant_factor/m**3/Hz**2
    #print("Hanneke prefactor [units /m**3/Hz**2]", constant_factor_unitless)
    
    lambda_mnp_squared = constant_factor / (z_L * r_cav**2) # This should be in units of Hz^2
    angular_part = special.jvp(0, bessel_derivative_zero * r_position/r_cav)**2
    axial_part = np.sin(mode_p * np.pi/2 * (z_position/z_L + 1))**2

    # Efficienctly deal with 2D scans.
    if hasattr(axial_part, "__len__") and hasattr(angular_part, "__len__"):
        lambda_mnp_squared *= np.outer(angular_part,axial_part)
    else:
        lambda_mnp_squared *= (angular_part*axial_part)
    # Calculate the damping factor
    delta = 2*loaded_Q*( cyclotron_frequency/mode_frequency-1)
    return lambda_mnp_squared, delta

# Calculate the radiated power for an electron at fixed location (r_position, z_position) with energy tranverse_kinetic_energy.
def hanneke_radiated_power(r_position, z_position, loaded_Q, l_cav, r_cav, cyclotron_frequency, tranverse_kinetic_energy, mode_frequency=None):
    if mode_frequency is None:
        # Assume that the center of the mode and the cyclotron frequency are identical
        mode_frequency = cyclotron_frequency

    lambda_mnp_squared, delta = hanneke_factor_TE011(r_position, z_position, loaded_Q, l_cav, r_cav, cyclotron_frequency, mode_frequency=mode_frequency)
    return tranverse_kinetic_energy*(2*loaded_Q/(1+delta**2))*lambda_mnp_squared/(mode_frequency*np.pi*2)

# Calculate the radiated power for an electron at fixed location (r_position, z_position) with energy tranverse_kinetic_energy.
# Averaging over the larmor power included.
def larmor_orbit_averaged_hanneke_power(r_position, z_position, loaded_Q, l_cav, r_cav, cyclotron_frequency, 
                                        kinetic_energy=endpoint, pitch=np.pi/2, mode_frequency=None, n_points=100):
    if mode_frequency is None:
        # Assume that the center of the mode and the cyclotron frequency are identical
        mode_frequency = cyclotron_frequency
    tranverse_kinetic_energy = kinetic_energy*np.sin(pitch)**2

    magnetic_field = magneticfield_from_frequency(cyclotron_frequency, kinetic_energy)
    electron_orbit_r = larmor_radius(magnetic_field, kin_energy=kinetic_energy, pitch=pitch)
    
    random_angles = (np.linspace(0,1,n_points)+np.random.rand())*2*np.pi # equally spaced ponts on a circle with random offset
    x_random = electron_orbit_r*np.cos(random_angles)
    y_random = electron_orbit_r*np.sin(random_angles)

    r_scalar = False
    if not hasattr(r_position, "__len__"):
        r_scalar = True
        r_position = np.array([r_position])
    if hasattr(z_position, "__len__"):
        hanneke_powers = np.empty((len(r_position),len(z_position)))
    else:
        hanneke_powers = np.empty(len(r_position))

    for i,r_pos_center in enumerate(r_position):
        r_pos_orbit = np.sqrt((x_random+r_pos_center)**2 + y_random**2)
        # Check for points outside the cavity
        if np.any(np.abs(r_pos_orbit) > r_cav):
            if hasattr(z_position, "__len__"):
                hanneke_powers[i] = np.zeros(len(z_position))
            else:
                hanneke_powers[i] = 0
        else:
            hanneke_power = hanneke_radiated_power(r_pos_orbit, z_position, loaded_Q, l_cav, r_cav, cyclotron_frequency, tranverse_kinetic_energy, mode_frequency=mode_frequency)
            hanneke_powers[i] = np.mean(hanneke_power.T, axis=-1)
    if r_scalar:
        hanneke_powers = hanneke_powers[0]

    return hanneke_powers
"""
def larmor_orbit_averaged_hanneke_power(r_position, z_position, loaded_Q, l_cav, r_cav, cyclotron_frequency, 
                                        kinetic_energy=endpoint, pitch=np.pi/2, mode_frequency=None, n_points=100):
    if mode_frequency is None:
        # Assume that the center of the mode and the cyclotron frequency are identical
        mode_frequency = cyclotron_frequency
    tranverse_kinetic_energy = kinetic_energy*np.sin(pitch)**2

    magnetic_field = magneticfield_from_frequency(cyclotron_frequency, kinetic_energy)
    electron_orbit_r = larmor_radius(magnetic_field, kin_energy=kinetic_energy, pitch=pitch)
    
    random_angles = (np.linspace(0,1,n_points)+np.random.rand())*2*np.pi # equally spaced ponts on a circle with random offset
    x_random = electron_orbit_r*np.cos(random_angles)
    y_random = electron_orbit_r*np.sin(random_angles)
    if hasattr(z_position, "__len__"):
        hanneke_powers = np.empty((len(r_position),len(z_position)))
    else:
        hanneke_powers = np.empty(len(r_position))

    for i,r_pos_center in enumerate(r_position):
        r_pos_orbit = np.sqrt((x_random+r_pos_center)**2 + y_random**2)
        # Check for points outside the cavity
        if np.any(np.abs(r_pos_orbit) > r_cav):
            if hasattr(z_position, "__len__"):
                hanneke_powers[i] = np.zeros(len(z_position))
            else:
                hanneke_powers[i] = 0
        else:
            hanneke_power = hanneke_radiated_power(r_pos_orbit, z_position, loaded_Q, l_cav, r_cav, cyclotron_frequency, tranverse_kinetic_energy, mode_frequency=mode_frequency)
            hanneke_powers[i] = np.mean(hanneke_power.T, axis=-1)
    return hanneke_powers
"""
    
# Calculate the average radiated power for an electron with radius r_position in a box trap:
def larmor_orbit_averaged_hanneke_power_box(r_position, loaded_Q, l_cav, r_cav, cyclotron_frequency,
                                            kinetic_energy=endpoint, pitch=np.pi/2, mode_frequency=None, n_points=100):
    
    return larmor_orbit_averaged_hanneke_power(r_position, 0, loaded_Q, l_cav, r_cav, cyclotron_frequency, 
                                               kinetic_energy=kinetic_energy, pitch=pitch, mode_frequency=mode_frequency, n_points=n_points)/2
