import numpy as np
from scipy.special import roots_laguerre
from mermithid.misc.Constants_numericalunits import *
'''
Ben Jones calculation disagrees with evaporation losses (specifically out the top > 5 mK, agrees otherwise) in atomic calc and must be integrated: https://3.basecamp.com/3700981/buckets/3107037/uploads/8902667883

'''

# Jins functions - Atomic Calculator
# Python decides whether a name is local or global at compile time (not runtime). The name must already exist in the module’s global namespace by the time the function runs, or Python has nowhere to bind it.

# C157 - [m] Fiducial distance from the cavity wall between max(ioffe_bite, larmor_radius)
# Ioffe field at wall is now the quadrature of the central field and Ioffe field by itself, everywhere it is used.
def calculate_ioffe_bite(nominal_field, magnetic_inhomogenity, ioffe_field, ioffe_multipolarity, cavity_radius):
    return ((2 * nominal_field**2 * magnetic_inhomogenity / (2 * nominal_field**2 * magnetic_inhomogenity + ioffe_field**2))**(1/(ioffe_multipolarity-2)) * cavity_radius * -1) + cavity_radius

# C271 - [m/s] Atomic thermal speed of 3T or 3He (C270)
def calculate_average_velocity(temp):
    average_velocity = c0 * np.sqrt((8 * 0.025*eV * temp / np.absolute(T0)) / (np.pi * tritium_mass_atomic))
    #average_velocity = c0 * np.sqrt((8 * kB  * temp) / (np.pi * tritium_mass_atomic * eV_J))
    return average_velocity

# C334 - [m^2] Cylinder wall; no endcaps included (2*pi*R*L)
def calculate_trap_wall_area(cavity_radius, cavity_length):
    # (Cavity length is cavity top plate - top of cone / 2)
    trap_wall_area = 2 * np.pi * cavity_radius * cavity_length
    return trap_wall_area

# C201 - [m^-1] Add the gravity-temperature scale parameter “b” for magnetogravitational trap. For pure magnetic trap, make gravity  weaker by ratio of L/D.
def calculate_gravity_temperature_scale(trapped_gas_temp, pure_magnetic_flag, cavity_L_over_D):
    gravity_temperature_scale = ((tritium_mass_atomic/c0**2 * gravity / (kB * trapped_gas_temp)) * np.absolute(pure_magnetic_flag * (1 + 1 / cavity_L_over_D) - 1))
    return gravity_temperature_scale

# C202 - [m^-2] Add the surface density scale parameter “a/b” for magnetogravitational trap
def calculate_surface_density_scale(design_density, gravity_temperature_scale, top_plate_cavity):
    surface_density_scale = design_density * top_plate_cavity / (1 - np.exp(-gravity_temperature_scale * top_plate_cavity))
    return surface_density_scale

# C204 - [atom] Inventory in physical volume
def calculate_inventory(design_density, volume):
    return design_density * volume

# C203 - [m^-3] Mean Trap Density within coils. Density in the electron-trapping region as a function of the density in the full volume. Differ depending on whether the experiment is horizontal or vertical. (mean density)
# Full volume (everywhere where atoms are trapped) includes some length of the cavity past the ends of the trap, as well as the volume inside the Ioffe cones.
# In a horizontal experiment, the average atom density in the electron trapping region would be about the same as in the full volume, because the electron trapping region and the non-electron trapping region have the same range/distribution of heights in the them.
# In a vertical experiment, the atom density will be different in the full volume compared with the electron-trapping volume, since atoms pool toward the bottom Ioffe cone.
def calculate_mean_trap_density(design_density, top_plate_cavity, trapped_gas_temp, pure_magnetic_flag, cavity_L_over_D, trap_coil_1, trap_coil_2):
    gravity_temp_scale = calculate_gravity_temperature_scale(trapped_gas_temp, pure_magnetic_flag, cavity_L_over_D)
    surface_density_scale = calculate_surface_density_scale(design_density, gravity_temp_scale, top_plate_cavity)
    mean_trap_density = design_density * top_plate_cavity * (np.exp(-gravity_temp_scale * trap_coil_1) - np.exp(-gravity_temp_scale * trap_coil_2)) / ((1 - np.exp(-gravity_temp_scale * top_plate_cavity)) * (trap_coil_2 - trap_coil_1))
    return mean_trap_density

# C208 - [decays/s] Radioactivity in trap per cavity
def calculate_activity_in_trap(number_density, cavity_radius, trap_coil_1, trap_coil_2):
    # C206 - [atom] Use exponential density vertical gradient, calculate mean density between trap coils. Calculate atoms and activity between the trap coils.
    atoms_between_trap_coils = number_density * np.pi * cavity_radius**2 * np.absolute(trap_coil_2 - trap_coil_1)
    return atoms_between_trap_coils / tritium_livetime

# Background T2 in atomic trap:
def calculate_T2_background_atomic_trap(cavity_radius, cavity_length, cavity_wall_temp, max_ratio_T2_T, number_density):
    # C306 - [mbar] Vapor Pressure of T2 w/ constants for saturated T2 vapor from Souers et al.
    T2_vapor_pressure = mbar * np.exp(T2_vapor_A + (T2_vapor_B*K/cavity_wall_temp) + T2_vapor_B_prime * np.log(cavity_wall_temp/K)) / 0.76
    # C307 - [m^-3] Density of saturated vapor
    sat_vapor_density = NA * (T2_vapor_pressure/mbar) * np.absolute(T0/cavity_wall_temp) / (1000 * molar_volume)
    # C308 -  [m^-3] = s^-1 * sqrt(kg/eV) / m^2 T2 density form desorption at end of a cycle
    T2_density_desorp = 4 * molecules_desorbed_wall_beta * (wall_activity * Ci_Bq) * np.sqrt(molecules_desorbed_wall_beta * (tritium_mass_atomic / c0**2) / (2 * atomic_tritium_recoil_energy * eV_J )) / calculate_trap_wall_area(cavity_radius, cavity_length)
    # C309 [m^-3] Total T2 density
    T2_total_density = sat_vapor_density + T2_density_desorp
    # C310 - T2/T number ratio. Activity ratio is 1.64 times bigger
    T2_T_ratio = T2_total_density / number_density
    #return T2_vapor_pressure, sat_vapor_density,  T2_density_desorp, T2_total_density, T2_T_ratio
    return T2_total_density, T2_T_ratio

# Aperture Heat Leak:
def calculate_aperture_heat_leak(trapped_gas_temp, design_density, volume):
    # C250 - [m/s] Average velocity of trapped gas
    trapped_gas_velocity = calculate_average_velocity(trapped_gas_temp)
    # C251 - [atom/s] Inward atom current required through aperture to balance losses (j_aperture)
    current_aperture_leak = area_atom_loading_aperture * design_density * trapped_gas_velocity / 4
    # C252 - [s] Time constant for loss through aperture
    time_constant_aperture = calculate_inventory(design_density, volume) / current_aperture_leak
    return time_constant_aperture, current_aperture_leak

# Radioactivity Heat Leak:
# Radioactivity loss is just a single number from Ben Clark’s thesis, needs to be done better. Dependent on a cross-section that varies with temp.
def calculate_rad_heat_leak(cavity_radius, number_density, trap_coil_1, trap_coil_2, design_density, volume, net_efficiency):
    # C256 - [s] Input from Ben Clark's thesis with a mirror ratio of 0.5
    half_life = 11 * 24 * 3600 * s
    trap_activity = calculate_activity_in_trap(number_density, cavity_radius, trap_coil_1, trap_coil_2)
    # C257 - [atom/s] Atom current required to balance loss due to radioactivity
    current_rad_leak = trap_activity * np.log(2) * tritium_livetime / (net_efficiency * half_life)
    # C258 - [s] Time constant for loss due to radioactivty (tau_rad)
    time_constant_rad = calculate_inventory(design_density, volume) / current_rad_leak
    return time_constant_rad, current_rad_leak

# T2 Desorption from the wall: T2 desorption from walls as a background and a trap-heating loss.
# Burst of molecules is emitted from the wall with each decay and can knock out atoms from the trap. Description in CDR 4.5.5. New parameter to enter is the number of molecules (choose 1000 for now).
def calculate_T2_desorption_from_wall(cavity_radius, cavity_length, design_density, volume):
    # C261 - [s] Mean lifetime of atom in trap from desorption
    time_constant_desorp =  calculate_trap_wall_area(cavity_radius, cavity_length) / (2 * molecules_desorbed_wall_beta * (wall_activity * Ci_Bq) * H_H2_crosssection)
    # C262 - [atom/s] Atom current required to keep up with desorption losses
    current_desorp = calculate_inventory(design_density, volume) / time_constant_desorp
    return time_constant_desorp, current_desorp

# He Heat Leak: Notable difference comes from pumping speed, physical volume, and design density
def calculate_He_heat_leak(pumping_speed_theoretical, pumping_cavity_termination, turbopump_speed, cavity_temperature, design_density, volume):
    He_velocity = calculate_average_velocity(cavity_temperature)
    # C266 - [atom/s] Not including flow for He heat leak itself
    He_production_rate = wall_activity * Ci_Bq
    # C269 - [m^-3] Number density of Helium
    He_density = He_production_rate * ((1/pumping_speed_theoretical) + (1/pumping_cavity_termination) + (1/turbopump_speed))
    # C272 - [s] Time constant for loss due to He-3 heat leak
    time_constant_He = 1 / (He_density * He_velocity * H_He_crosssection)
    # C273 - Fraction of total gas that is He
    He_fraction = He_density / design_density
    # C274 - [s^-1] Atom current required to keep up with He-3 heat leak
    current_He_leak = calculate_inventory(design_density, volume) / time_constant_He
    return time_constant_He, current_He_leak

# Dipolar loss rate: Calculation with z-dependent density, cylinder & cone. Polynomial fit of nominal field.
def calculate_dipolar_loss(nominal_field, cavity_radius, design_density, volume, trapped_gas_temp, cavity_L_over_D, top_cone, top_plate_cavity, pure_magnetic_flag):
    gravity_temperature_scale = calculate_gravity_temperature_scale(trapped_gas_temp, pure_magnetic_flag, cavity_L_over_D)
    surface_density_scale = calculate_surface_density_scale(design_density, gravity_temperature_scale, top_plate_cavity)
    # C298 - [m^3/s] Polynomial fit for Dipolar spin-flip rate (G_dd). Depends on field (Lagendijk et al). Polynomial-log fit used now.
    dipolar_spin_flip_rate = (60.106 + 13.812 * np.log(nominal_field) - 4.7867 * np.log(nominal_field)**2 - 2.3192 * np.log(nominal_field)**3 \
                             - 0.32663 * np.log(nominal_field)**4 - 0.015775 * np.log(nominal_field)**5) * 1e-22 * LGd_rates * m**3/s
    # C299 - [atom/s] Flow to keep up with dipolar losses
    current_dipolar = dipolar_spin_flip_rate * np.pi * cavity_radius**2 * surface_density_scale**2 * (gravity_temperature_scale * (np.exp(-gravity_temperature_scale * top_cone) \
                      - np.exp(-gravity_temperature_scale * top_plate_cavity)) / 2 + (1 - np.exp(-2 * gravity_temperature_scale * top_cone) * (2 * gravity_temperature_scale**2  \
                      * top_cone**2 + 2 * gravity_temperature_scale * top_cone + 1)) / (4 * top_cone**2 * gravity_temperature_scale))
    # C300 - [s] Time constant for loss due to dipolar spin-flip loss
    time_constant_dipolar = calculate_inventory(design_density, volume) / current_dipolar
    return time_constant_dipolar, current_dipolar

# Evaporation loss rate: Does not take into account density of states with height. Magnetic potential limits evaporation.
# Cone and cylinder now separate because cone is weaker owing to azimuthal modulation of Ioffe field.  Each now has its own density multiplier.  The weaker cone field is handled in a separate Igor calculation Coneangle.pxp outside this SS and entered as a loss rate multiplier in C111.
def calculate_evaporation_loss(cavity_L_over_D, pure_magnetic_flag, trapped_gas_temp, cavity_radius, ioffe_field, nominal_field, cavity_length, top_cone, top_plate_cavity, design_density, relative_loss_rate_cone_wall, volume):
    gravity_temperature_scale = calculate_gravity_temperature_scale(trapped_gas_temp, pure_magnetic_flag, cavity_L_over_D)
    surface_density_scale = calculate_surface_density_scale(design_density, gravity_temperature_scale, top_plate_cavity)

    # C283 - [m] Mean free path at the base of cavity
    mfp_cavity_base = 1 / (gravity_temperature_scale * surface_density_scale * tritium_tritium_crosssection_atomic)
    # C284 - Pure Magnetic eta
    eta = (bohr_magneton / eV_J) * (np.sqrt(ioffe_field**2 + nominal_field**2) - nominal_field) / (trapped_gas_temp * 0.025*eV / np.absolute(T0))
    # C285 - Magnetogravitational eta
    eta_grav = (tritium_mass_atomic/c0**2) * kg_amu * gravity * cavity_length / (kB * trapped_gas_temp)
    # C286 - [s^-1] Evaporation out the top; set to 1/1000 for pure magnetic trap
    evaporation_top = (np.log(1 + np.log(2) * np.exp(gravity_temperature_scale * top_plate_cavity) / (surface_density_scale * tritium_tritium_crosssection_atomic)) \
    * surface_density_scale * gravity_temperature_scale * np.exp(-gravity_temperature_scale * top_plate_cavity) * np.sqrt(np.pi * kB * trapped_gas_temp / (2 * tritium_mass_atomic / c0**2)) * cavity_radius**2) \
    * np.absolute(pure_magnetic_flag * (1 + 0.001) -1)
    # C287 - [s] Time constant for evaporation out of the top
    time_constant_evap_top = calculate_inventory(design_density, volume) / evaporation_top
    # C288 - [s^-1] Evaporation to the sides
    evaporation_sides = (2 * np.pi * cavity_radius * cavity_length * design_density * np.sqrt(kB * trapped_gas_temp / (2 * np.pi * tritium_mass_atomic / c0**2)) \
    * np.exp(-eta) * top_plate_cavity * (np.exp(-gravity_temperature_scale * top_cone) - np.exp(-gravity_temperature_scale * top_plate_cavity)) \
    / ((1 - np.exp(-gravity_temperature_scale * top_plate_cavity)) * (top_plate_cavity - top_cone)))
    # C290 - [s^-1] Evaporation in cone; for pure magnetic, double it to account for both ends
    evaporation_cone = (relative_loss_rate_cone_wall * np.pi * cavity_radius * np.sqrt(cavity_radius**2 + top_cone**2) * design_density \
    * np.sqrt(kB * trapped_gas_temp / (2 * np.pi * tritium_mass_atomic/c0**2)) * np.exp(-eta) * top_plate_cavity * (1 - np.exp(-gravity_temperature_scale * top_cone)) \
    / ((1 - np.exp(-gravity_temperature_scale * top_plate_cavity)) * top_cone)) * np.absolute(pure_magnetic_flag + 1)
    # C292 - [s] Time constant for evaporation loss of cone and sides
    time_constant_evap_cone_sides = calculate_inventory(design_density, volume) / (evaporation_cone + evaporation_sides)
    # C295 - [atom/s] Flow to keep up with evaporation losses aka Total evaporation_sum
    current_evaporation = evaporation_top + evaporation_sides + evaporation_cone
    # C294 - [s] Time constant for total evaporation loss
    time_constant_total_evaporation = calculate_inventory(design_density, volume) / current_evaporation
    return time_constant_total_evaporation, current_evaporation, evaporation_top, evaporation_sides, evaporation_cone

# T2 Heat Leak:
def calculate_T2_heat_leak(aperture_current, rad_current, He_current, evap_current, dipolar_current, T2_total_density, cavity_wall_temp, design_density, volume):
    # C277 - [atom/s] T2 production rate (all atoms entering, except for T2 heat leak itself)
    T2_production_rate = aperture_current + rad_current + He_current + evap_current + dipolar_current
    # C279 - [s] Time constant for loss due to T2 heat leak
    time_constant_T2 = 1 / (T2_total_density * H_H2_crosssection * calculate_average_velocity(cavity_wall_temp) / np.sqrt(2))
    # C280 - [atom/s]
    current_T2_leak = calculate_inventory(design_density, volume) / time_constant_T2
    return time_constant_T2, current_T2_leak

# C303 - [s] Total lifetime of trap:
def calculate_trap_lifetime(time_rad, time_desorp, time_He, time_T2, time_evap, time_dipolar):
    return 1 / ((1/time_rad) + (1/time_desorp) + (1/time_He) + (1/time_T2) + (1/time_evap) + (1/time_dipolar))

# Atom Supply into Trap:
# The inputted atom current determines the (average) density in the full volume, but the density for mass sensitivity is average atom density within the electron trap. Conversion is needed between these two densities
def calculate_trap_atom_supply(rad_current, evap_current, dipolar_current, He_current, T2_current, design_density, volume):
    # The ‘atomic current required’ includes the c-state atoms as well as d-state, although those are lost almost immediately to spin exchange.
    # C107 - C state flag for hyperfine states for total gas into trap.  Trap is always d-state only. No c-states (1), include c-states (2)
    c_states_flag = 1
    # C318 - [atom/s] Total current is atom current (d state only) + He heat leak + T2 heat leak
    total_atom_current = rad_current + evap_current + dipolar_current + He_current + T2_current
    # C319 - [s] Total time constant for atoms to remain in the trap
    total_time_constant_trap = calculate_inventory(design_density, volume) / total_atom_current
    # C320 - [atom/s] Atom current with c-states and d-states in calculation
    total_atom_current_states = total_atom_current * c_states_flag
    return total_time_constant_trap, total_atom_current_states
'''
There are two possible methods for pumping away the tritium, all of which is eventually in molecular form. One is to keep the tritium in circulation by using turbopumps and avoiding temperatures below 10K in the trap region, and the second is to cool parts of the trap region outside the magnetic wall to < 3K in order to cryopump the tritium.
The mechanical pumping method is impractical because of the pumping speed required, a fraction of a billion L/s, where an achievable upper limit is 4 orders of magnitude smaller.
The speed requirement is driven by the combination of the input atomic current and the need to maintain the molecular fraction below 10−4.
The cryopumping method ties up very large amounts of tritium, tens to hundreds of kCi, by the end of a day, and the need to warm up and recycle that tritium on such a short time scale would lead to low statistical precision and instabilities.
Still need turbos to handle He-3 and to do pumpout during recycling, but initial hope that we could run turbos only and keep the T2 pressure low enough was not realized. Must recycle.
'''
# Turbopump Calculations: Need to convert tritium mass amu to eV : 1 amu ~ 931 MeV/c^2 and kB = 8.6E-05 eV/K. Multiply by sqrt(2) if atomic; # Useful for Atomic and Helium-3
# C233/C236 - [m^3/s] (molecular/atomic) theoretical pumping speed. The mean speed of the cylinder connected to a perfect pump (Dushman). Uniform source, outputs mean density in cavity. Cannot exceed (obstruction not included).
def turbopump_speed_limit(cavity_radius, cavity_temperature, cavity_L_over_D, atomic_flag):
    return (np.pi * cavity_radius**2 * c0 * np.sqrt(kB_eV * cavity_temperature  / (4 * np.pi * tritium_mass_atomic)) / (0.5 + cavity_L_over_D / 8)) * (1 + atomic_flag * (np.sqrt(2) - 1))
# C235/C238 - [m^3/s] (molecular/atomic) Assumed ambient room air (28-29 amu) temperature (293 K,  and a pumping speed of 0.5 L/s
def cavity_termination_speed(turbopumping_speed_air, cavity_top_plate_temp, atomic_flag):
    return (turbopumping_speed_air * np.sqrt(cavity_top_plate_temp * 28 * amu * c0**2 / (293 * K * 2 * tritium_mass_atomic))) * (1 + atomic_flag * (np.sqrt(2) - 1))
# C239 - [m^3/s] Turbopump in series speed (2.5 m^3/s * 2)
def turbopump_speed(number_turbos, turbopumping_speed_gas):
    return number_turbos * turbopumping_speed_gas
# C327 - [m^-3] Molecular density allowed by molecular/atomic assuming total atom density in all of physical volume
def molecular_density_allowed(design_density, max_ratio_T2_T):
    return design_density * max_ratio_T2_T * ground_state_branch_atomic / 2
# The Pumping speed required (molecular) in C328 is kind of a dimensional calculation, not really a speed. As a result, the cryopumping speed limit is probably not real. Ideal cryopumping is the impingement rate: it all sticks. The actual molecular density is determined by the vapor pressure (Souers formula).
# C328 - [m^3/s] - Pumping speed required to keep molecular pressure from exceeding reference limit
def turbopump_speed_required(atom_current, molecular_density_limit):
    return atom_current / 2 / molecular_density_limit
# C331 - If above 1000, critical
def ratio_turbopump_speed(pumping_speed_required, pumping_speed_limit, cavity_termination_speed, turbopump_speed):
    return pumping_speed_required * ((1 / pumping_speed_limit) + (1 / cavity_termination_speed) + (1 / turbopump_speed))

# Cryopumping Calculations:
def calculate_cryopump_speed(cavity_wall_temp, cavity_radius, cavity_length, pumping_speed_required, atom_current, design_density, max_ratio_T2_T):
    # C335 - [m/s] Cryopumping speed per area for mass-3 atoms at wall temperature stated above. sqrt(k  T/(2 pi M)); see https://www.synsysco.com/wp-content/uploads/2016/02/Basics-of-Cryopumping-Booklet.pdf.  Using M=3 because most gas is spin-flipped atoms.
    cryopump_speed_per_area = np.sqrt(kB_eV * cavity_wall_temp / (2 * np.pi * tritium_mass_atomic)) * c0
    # C336  - [m^3/s] With given cylindrical dimensions at wall temperature stated above for mass-3 atoms
    trap_interior_surface_speed = cryopump_speed_per_area * calculate_trap_wall_area(cavity_radius, cavity_length)
    # C338 - Ratio of required molecular speed to cryopumping speed. Ideal cryopumping is the impingement rate (all sticks).
    ratio_cryopump_speed = pumping_speed_required / trap_interior_surface_speed
    # C339 - [W] Heat delivered to the cold surface
    recombination_heat_load = atom_current * molecular_tritium_binding_energy * eV_J / 2
    # C340 - [Ci/day] Accumulated activity per day
    accumulated_activity_day = atom_current * lambda_tritium * (86400*s/day) / Ci_Bq
    # C342 - [day] Time between recycling cryosurface. For cryopumping atomic experiment; determines inventory on walls
    time_recycling_cryosurface = wall_activity / accumulated_activity_day
    # C344 - [m^3/s] Cryopumping speed for molecules
    cryopump_speed = np.sqrt(kB_eV * cavity_wall_temp / (4 * np.pi * tritium_mass_atomic)) * c0 * calculate_trap_wall_area(cavity_radius, cavity_length)
    # C345 - [molecules/s] max allowed injection rate of molecules
    max_molecules_injection = cryopump_speed * molecular_density_allowed(design_density, max_ratio_T2_T)
    return ratio_cryopump_speed, cryopump_speed, time_recycling_cryosurface

# Injection Line Calculations: Polynomial fit of injection field
# Density in the injection line, mfp in injection line, dipolar loss in injection line.  There are 2 new entries for this in the choices – beam temperature and beamline field.
def calculate_injection_line(atom_current, cavity_radius, design_density, trapped_gas_temp, nominal_field, injection_gas_temp, injection_field):
    # C349 - [m/s] Vertical flow speed in trap at bottom and in the beamline
    vertical_injection_speed = atom_current / (np.pi * cavity_radius**2 * design_density)
    # C350 - [J/atom] Energy in trapped gas and in beam gas
    trapped_gas_energy = (1.5 * kB * trapped_gas_temp) + (0.5 * (tritium_mass_atomic/c0**2) * vertical_injection_speed**2) + (bohr_magneton * nominal_field)
    # C351 - [m^-3] Density in the injection line
    injection_density = atom_current * np.sqrt((tritium_mass_atomic/c0**2) / (2 * trapped_gas_energy - 3 * kB * injection_gas_temp - 2 * bohr_magneton * injection_field)) / area_atom_loading_aperture
    # C352 - [m] Mean free path in beamline
    mfp_beamline = 1 / (injection_density * tritium_tritium_crosssection_atomic)
    # C353 - [m/s] Vertical flow speed through aperture
    vertical_aperture_speed = design_density * np.pi * cavity_radius**2 * vertical_injection_speed / (area_atom_loading_aperture * injection_density)
    # C354 - [J/atom] Energy in injected gas
    injection_gas_energy = (1.5 * kB * injection_gas_temp) + (0.5 * (tritium_mass_atomic/c0**2) * vertical_aperture_speed**2) + (bohr_magneton * injection_field)
    # C355 - [m^3/s] 5th degree polynomial fit for injection line dipolar loss rate
    injection_dipolar_loss_rate = (60.106 + 13.812 * np.log(injection_field/T) - 4.7867 * np.log(injection_field/T)**2 - 2.3192 * np.log(injection_field/T)**3 \
                             - 0.32663 * np.log(injection_field/T)**4 - 0.015775 * np.log(injection_field/T)**5) * 1e-22 * LGd_rates * m**3/s
    # C356 - [m^-1] Beamline dipolar loss
    beamline_dipolar = injection_dipolar_loss_rate * injection_density / vertical_aperture_speed
    return vertical_injection_speed, trapped_gas_energy, injection_density, vertical_aperture_speed, injection_gas_energy


def calculate_activity_last_1eV_spectrum(self, atomic_flag, number_density, cavity_radius, trap_coil_1, trap_coil_2, total_efficiency):
    if atomic_flag:
        trap_activity = calculate_activity_in_trap(number_density, cavity_radius, trap_coil_1, trap_coil_2)
        # C209 - [decay/s] Activity in last 100 eV of spectrum * net efficiency
        # Note last_1eV_fraction_atomic is the fraction of events in the last 1eV
        activity_last_100eV_efficiency = trap_activity * last_1ev_fraction_atomic * total_efficiency * 1000000 / ground_state_branch_atomic
        # C211 - [decay/s]
        last_1eV_atomic = activity_last_100eV_efficiency * ground_state_branch_atomic / 1000000
        return last_1eV_atomic
    else:
        print("Yeah I didn't do that yet - Jin")
        return None
