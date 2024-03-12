------------------
Configuring mermithid sensitivity calculation
------------------

The sensitivity calculation is configured using a configuration file. The configuration files for Project 8 live in the termite repository: https://github.com/project8/termite/tree/feature/sensitivity_config_files/sensitivity_config_files


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

* `L_over_D`: The ratio of the length of the cavity to the diameter of the cavity. Together with the frequency of the cavity, this parameter determines the length and volume of a single cavity. It also impacts the cavity Q-factor which is determined by the bandwidth needed to observe the axial frequency of the trapped electrons.
* `livetime`: Run duration of the experiment. Together with the total volume, efficiency, and the gas density, this determines the statistical power of the experiment. 
* `n_cavities`: Number of identical cavities in the experimennt. This parameter multiplies the single cavity volume to give the total experimental volume.