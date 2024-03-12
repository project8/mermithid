------------------
Neutrino Mass Sensitivity Calculation
------------------


Scripts used in sensitivity calculations
----------------------------------


Analytic sensitivity formula
----------------------------------


Systematic uncertainty contributions
----------------------------------

The following contributions to energy broadening of the beta spectrum are included:

1. ``sigma_trans``: Translational Doppler broadening due to thermal motion of tritium atoms or molecules;
2. ``sigma_f``: Uncertainty on the start frequency of the event, after an axial frequency correction to account for the pitch angle.
3. ``sigma_B``: Energy broadening due to radial, azimuthal, and temporal varation of the magnetic field. This is the remaining broadening after any event-by-event corrections for the field have been performed.
4. ``sigma_Miss``:
5. ``sigma_Plasma``: 

In the list above, each variable that starts with ``sigma`` is a standard deviation of a distribution of energies. The measured spectrum is the convolution of the underlying beta spectrum with such distributions of energies. Each ``sigma`` has an associated ``delta``, which is the uncertainty on ``sigma`` from calibration, theory, and/or simulation (see :ref:`Analytic sensitivity formula`). For example, ``sigma_trans`` has an associated variable ``delta_sigma_trans``.

In the script ``mermithid/mermithid/misc/SensitivityCavityFormulas.py``, lists of these energy broadening standard deviations (sigmas) and uncertainties on them (deltas) are returned by the ``get_systematics`` method of the ``CavitySensitivity`` class.

Contributions 4 and 5 are simply inputted in the sensitivity configuration file; we do not yet have a way to calculate these contribution in mermithid. Contributions 1, 2, and 3 are described in more detail, below.

