------------------
Neutrino Mass Sensitivity Calculation
------------------


Scripts used in sensitivity calculations
----------------------------------

The mermithid sensitivity calculation is written in python and a python script can be used to configure the calculation and make sensitivity plots.
Mermithid has several processors to make this a lot esaier. For cavity sensitivity calculations that is the `CavitySensitivityCurveProcessor`_.

.. _CavitySensitivityCurveProcessor: https://github.com/project8/mermithid/blob/feature/sensitivity_curve/mermithid/processors/Sensitivity/CavitySensitivityCurveProcessor.py

Mermithid processors are all designed to be used in the same way:
1. Define a dictionary with the processors configurable parameters
2. Instantiate a processor and pass the configuration dictionary to it's ``Configure()`` method.
3. Call the processors ``Run()`` method to make it perform it's task.

In `mermithid/tests`_ the ``Sensitivity_test.py`` script contains several examples of how to perform sensitivity calculations using mermithid processors. In ``test_SensitivityCurveProcessor`` the ``CavitySensitivityCurveProcessor`` is used to calculate and plot the sensitivity of a cavity experiment to the neutrino mass as a function of gas density.
Other working examples for creating sensivitiy plots vs. frequency or exposure can be found in `mermithid/test_analyses/Cavity_Sensitivity_analysis.py`_.

.. _mermithid/tests: https://github.com/project8/mermithid/blob/feature/sensitivity_curve/mermithid/tests
.. _mermithid/test_analyses/Cavity_Sensitivity_analysis.py: https://github.com/project8/mermithid/blob/feature/sensitivity_curve/test_analysis/Cavity_Sensitivity_analysis.py


Analytic sensitivity formula
----------------------------------


Systematic uncertainty contributions
----------------------------------

The following contributions to energy broadening of the beta spectrum are included:

1. ``sigma_trans``: Translational Doppler broadening due to thermal motion of tritium atoms or molecules.
2. ``sigma_f``: Energy broadening due to mis-reconstruction of the start frequencies of events from the range of electron pitch angles (after axial frequency corrections), and from the process of fitting chirp tracks that have finite length and SNR to find where the tracks start.
3. ``sigma_B``: Energy broadening due to radial, azimuthal, and temporal varation of the magnetic field. This is the remaining broadening after any event-by-event corrections for the field have been performed.
4. ``sigma_Miss``: Energy broadening due to the detection of events after one or more tracks have been missed.
5. ``sigma_Plasma``: Energy broadening due to plasma effects of the charged particles in the source volume of an atomic tritium experiment.

In the list above, each variable that starts with ``sigma`` is a standard deviation of a distribution of energies. The measured spectrum is the convolution of the underlying beta spectrum with such distributions of energies. Asymmetries in these distributions are not accounted for in this approximate model.

Each ``sigma`` has an associated ``delta``, which is the uncertainty on ``sigma`` from calibration, theory, and/or simulation (see the previous section). For example, ``sigma_trans`` has an associated variable ``delta_sigma_trans``.

In the script ``mermithid/mermithid/misc/SensitivityCavityFormulas.py``, lists of these energy broadening standard deviations (sigmas) and uncertainties on them (deltas) are returned by the ``get_systematics`` method of the ``CavitySensitivity`` class.

Contributions 4 and 5 are simply inputted in the sensitivity configuration file; we do not yet have a way to calculate these in mermithid. Contributions 1, 2, and 3 are calculated in mermithid, as described below.


Translational Doppler broadening (``sigma_trans``)
============================
The translational Doppler broadening is 


Track start frequency determination and pitch angle correction (``sigma_f``)
============================


Radial, azimuthal, and temporal field broadening (``sigma_B``)
============================

