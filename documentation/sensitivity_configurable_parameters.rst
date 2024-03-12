------------------
Configuring mermithid sensitivity calculation
------------------

The sensitivity calculation is configured using a configuration file. The configuration files for Project 8 live in the termite repository: https://github.com/project8/termite/tree/feature/sensitivity_config_files/sensitivity_config_files

Our main goal is the calcualtion of sensitivity in a cavity experiment. The configurations below are to be used for https://github.com/project8/mermithid/blob/feature/sensitivity_curve/mermithid/misc/SensitivityCavityFormulas.py


Structure of a config file
--------------------------

Configuration files have several sections:


* Experiment
* Efficiency
* FrequencyExtraction
* DopplerBroadening
* MagneticField
* FinalStates
* MissingTracks
* PlasmaEffects

Each section has a number of parameters that are used for the calcualtion of the sensitivty contribution of the respective section.


Parameters
----------

Below is a list of the parameters with a short description of what role they play in the calculation.

**Experiment**

* ``L_over_D``: The ratio of the length of the cavity to the diameter of the cavity. Together with the frequency of the cavity, this parameter determines the length and volume of a single cavity. It also impacts the cavity Q-factor which is determined by the bandwidth needed to observe the axial frequency of the trapped electrons.
* ``livetime``: Run duration of the experiment. Together with the total volume, efficiency, and the gas density, this determines the statistical power of the experiment. 
* ``n_cavities``: Number of identical cavities in the experimennt. This parameter multiplies the single cavity volume to give the total experimental volume.
* ``background_rate_per_ev``: Background rate per electronvolt for the entire experiment. It is not automultiplied by the number of channels (cavities) in the experiment.
* ``number_density``: The gas number density together with the total volume, livetime, and the efficiency determines the statistical power of the experiment. Gas density also determines the track length and therefore the frequency resolution. The sensitivity curve processor can optimize this parameter to maximize the sensitivity. In that case this number is overwritten in the calculation. 
* ``sri_factor``: The statistical rate increase factor articifially increases the number of observed events. It is highly recommended to set it to 1.
* ``atomic``: If true the calculation is done for atomic tritium. If false moecular tritium is assumed. This affects the number of decays per gas molecule/atom (2 for molecular 1 for atomic), the track length in a given gas density (via electron scattering cross section), and the width of the final ground state.


**Efficiency**

* ``usefixedvalue``: So far we only have a fixed efficiency implemented. Work is in progress of moving the efficiency calcualtion into the sensivity calculation.
* ``fixed_efficiency``: For example, set to roughly 2% for a 88deg minimum trapped pitch angle, assuming 100% detection efficiency of the trapped angles.

Not yet used (work in progress):
* ``radial_efficiency``: Typically set to 0.67 from a calcualtion done for a 325MHz cavity with Halbach bite and radial cut on power of > 0.5 * maximum power.
* ``trapping_efficiency``: Not yet in use.

**FrequencyExtraction**

We use the CRLB for calculating the frequency resolution. The CRLB is calculated from the signal to noise ratio (SNR) and track length. The sensitivity calculations therefore include a calculation of the noise and signal power.

* ``usefixedvalue``: If true all parameters below are ignored but ``default_systematic_smearing`` and ``default_systematic_uncertainty``.
* ``default_systematic_smearing``: Used if ``usefixedvalue`` is true. Units must be eV.
* ``default_systematic_uncertainty``: Used if ``usefixedvalue`` is true. Units must be eV
* ``usefixeduncertainty``: If true, the uncertainty on the frequency extraction is fixed to the value of ``fixed_relativ_uncertainty``. False will result in an error because no calculation is currently implemented.
* ``fixed_relativ_uncertainty``: If ``usefixeduncertainty`` is true, this relative value is used as the uncertainty on the frequency extraction.
* ``crlb_on_sidebands``: If true, the total resolution from frequency extraction includes the correction from axial field variation based on the CRLB resolution of sidebands.
* ``sideband_power_fraction``: Fraction of power in the sidebands. This is used to calculate the CRLB resolution of the sidebands.
* ``sideband_order``: The order of the sidebands used to calculate the axial field correction. 2 gives better precision but would require lower Q which in turn affects SNR. However, the sideband order is currently not included in the Q calculation.
* ``amplifier_temperature``: Used to calculate noise temperature.
* ``quantum_amp_efficiency``: Used to calculate noise temperature.
* ``att_cir_db``: Used to calculate noise temperature.
* ``att_line_db``: Used to calculate noise temperature.
* ``cavity_temperature``: Used to calculate noise temperature. Has to be compatible with the gas species (molecular tritium freezes below 30K).
* ``unloaded_q``: The unloaded Q of hte cavity. From the axial frequency of the minimum trapped angle (depending on ``L_over_D`` and the cavity frequency) and the bandwidth needed to observe it, the loaded Q-factor is calculated. The unloaded Q-factor is used to calculate the coupling and therefore impacts the SNR.
* ``minimum_angle_in_bandwidth``: Minimum pitch angle of which the axial frequency is contained in the bandwidth. Impacts the loaded Q and therefore the SNR. In the future it will be linked to the efficiency above.
* ``crlb_scaling_factor``: Arbitrary factor to scale the resolution.
* ``magnetic_field_smearing``: A magnetic field smearing in eV can be added here. 

**DopplerBroadening**

* ``usefixedvalue``: If True ``default_systematic_smearing`` and ``default_systematic_uncertainty`` are used.
* ``default_systematic_smearing``: Default systematic broadening for this category. Units must be eV.
* ``default_systematic_uncertainty``: Default systematic uncertainty for this category. Units must be eV.
* ``gas_temperature``: Temperature of the source gas. This should only be different from the cavity temperature if the gas is not in thermal equilibrium with the cavity. The gas temperature is used to calculate the Doppler broadening.
* ``gas_temperature_uncertainty``: Absolute uncertainty of the gas temperature.
* ``fraction_uncertainty_on_doppler_broadening``: Fractional uncertainty on the Doppler broadening.


**MagneticField**

* ``usefixedvalue``: If True ``default_systematic_smearing`` and ``default_systematic_uncertainty`` are used.
* ``default_systematic_smearing``: Default systematic broadening for this category. Units must be eV.
* ``default_systematic_uncertainty``: Default systematic uncertainty for this category. Units must be eV.
* ``nominal_field``: Determines the CRES and cavity TE011 mode frequency. The cavity dimensions are derived from this and ``L_over_D``
* ``useinhomogeneity``: True
* ``fraction_uncertainty_on_field_broadening``: Fractional uncertainty on field inhomogeneity. Applies to all parameters below
* ``sigma_meanb``: Fixed input in eV. Magnetic field instability (which is not fully corrected using live calibration) and unknown wiggles in the z-field profile, relative to a smooth trap shape.
* ``sigmae_r``: Fixed input in eV. Energy broadening from radial field inhomogeneity that remains after radial reconstruction. Accounts for both the uncertainty on each electron's radius and the uncertainty on the radial field profile.
* ``sigmae_theta``: Fixed input in eV. Energy broadening remaining after theta reconstruction, from electrons with lower pitch angles exploring high fields. Accounts for both the uncertainty on theta and uncertainties on the trap depth/boxiness.
* ``sigmae_phi``: Fixed input in eV. Energy broadening from phi field inhomogeneity that remains after phi reconstruction.

**FinalStates**

* ``ground_state_width_uncertainty_fraction``: Uncertainty on the ground state width. Recommended to use 0.001.


The sections below have so far not been used and are assumed to be negligible.

**MissingTracks**

* ``usefixedvalue``: If True ``default_systematic_smearing`` and ``default_systematic_uncertainty`` are used.
* ``default_systematic_smearing``: Default systematic broadening for this category. Units must be eV.
* ``default_systematic_uncertainty``: Default systematic uncertainty for this category. Units must be eV.

**PlasmaEffects**

* ``usefixedvalue``: If True ``default_systematic_smearing`` and ``default_systematic_uncertainty`` are used.
* ``default_systematic_smearing``: Default systematic broadening for this category. Units must be eV.
* ``default_systematic_uncertainty``: Default systematic uncertainty for this category. Units must be eV.


