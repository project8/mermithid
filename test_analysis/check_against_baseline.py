'''
make_baseline.py  --  Phase 0 golden-baseline driver for the multimode port.

Run this ONCE against the *current, unedited* single-mode
SensitivityCavityFormulas.py (on branch feature/multimode_sensitivity, before
any multimode edits) to freeze the single-mode numbers into baseline_old.json.

Every later version of the engine is then checked against this file, so you
never need to keep the old code runnable.

It mirrors how CavitySensitivityCurveProcessor produces the plotted curve:
  * builds CavitySensitivity(config_path)
  * the plotted quantity is CL90() in eV
  * detection threshold is optimized by scanning CL90(Threshold={...})
  * number density is optimized by scanning CL90(Experiment={"number_density": rho})

Usage
-----
    python make_baseline.py /path/to/your_config.cfg
    python make_baseline.py /path/to/your_config.cfg --out baseline_old.json
    python make_baseline.py /path/to/your_config.cfg --no-optimize
    python make_baseline.py /path/to/your_config.cfg \
        --density-range 1e14 1e21 --n-density 80 \
        --thresh-range 1 160 --n-thresh 40

Notes
-----
  * All numericalunits-scaled quantities are divided by their unit before
    being written, so the JSON holds plain floats in known, named units
    (e.g. "cavity_freq_Hz", "CL90_eV"). This is what makes two runs comparable.
  * Every capture is wrapped so a single missing attribute can't abort the
    dump; anything that fails is recorded under "_capture_errors" with the
    exception text, and we refine from there.
'''
from __future__ import absolute_import

import argparse
import json
import sys
import traceback

import numpy as np

# numericalunits scale factors (same import style as the processor)
from numericalunits import eV, GHz, Hz, W, K, s, m

from mermithid.sensitivity.SensitivityCavityFormulas import CavitySensitivity


def _safe(store, errors, key, fn):
    """Evaluate fn() and store float(result) under key; record failures."""
    try:
        val = fn()
        if isinstance(val, (list, tuple, np.ndarray)):
            val = np.asarray(val, dtype=float).tolist()
        else:
            val = float(val)
        store[key] = val
    except Exception as exc:  # noqa: BLE001  - we want the dump to survive
        store[key] = None
        errors[key] = "{}: {}".format(type(exc).__name__, exc)


def build_baseline(config_path, optimize=True,
                   density_range=(1e14, 1e21), n_density=80,
                   thresh_range=(1.0, 160.0), n_thresh=40,
                   seed=42):
    out = {}
    errors = {}
    out["_config_file"] = config_path
    out["_seed"] = seed

    # Fix the RNG before constructing: CavityPower draws electron radii with
    # np.random.triangular, and that randomness propagates into signal_power ->
    # detection_efficiency -> effective_volume -> CL90. Seeding makes the frozen
    # baseline reproducible so the gate can compare at tight tolerance. The check
    # script seeds identically before the same call sequence.
    np.random.seed(seed)

    s_main = CavitySensitivity(config_path, verbose=False)

    # ---- Plotted quantity at the configured density/threshold --------------
    # Call CL90() first so the full chain populates the lower-level attributes,
    # THEN snapshot those attributes (before any override scan changes state).
    _safe(out, errors, "CL90_configured_eV",
          lambda: s_main.CL90() / eV)

    # ---- Lower-level gate quantities (the per-module comparison targets) ----
    # These are the values each phase's single-mode reduction must reproduce.
    _safe(out, errors, "cavity_freq_Hz",      lambda: s_main.cavity_freq / Hz)
    _safe(out, errors, "cavity_freq_GHz",     lambda: s_main.cavity_freq / GHz)
    _safe(out, errors, "cavity_radius_m",     lambda: s_main.cavity_radius / m)
    _safe(out, errors, "loaded_q",            lambda: s_main.CavityLoadedQ())
    _safe(out, errors, "required_bw_Hz",      lambda: s_main.required_bw / Hz)
    _safe(out, errors, "signal_power_W",      lambda: s_main.signal_power / W)
    _safe(out, errors, "effective_volume_m3", lambda: s_main.effective_volume / m**3)
    _safe(out, errors, "total_trap_volume_m3", lambda: s_main.total_trap_volume / m**3)
    _safe(out, errors, "pos_dep_trapping_eff", lambda: s_main.pos_dependent_trapping_efficiency)

    # Call syst_frequency_extraction ONCE and snapshot the entire CRLB chain
    # from that single coherent engine state, at the configured operating point.
    # (Capturing these piecemeal later risks reading state left by intervening
    #  parent-class recomputes, e.g. StatSens/SystSens/DeltaEWidth.)
    crlb = {}
    crlb_err = {}
    try:
        sigma_f, _delta = s_main.syst_frequency_extraction()
        crlb["syst_freq_extraction_sigma_eV"] = float(sigma_f / eV)
    except Exception as exc:  # noqa: BLE001
        crlb["syst_freq_extraction_sigma_eV"] = None
        crlb_err["syst_freq_extraction_sigma_eV"] = "{}: {}".format(type(exc).__name__, exc)

    # All of the following reflect the SAME call above.
    _safe(crlb, crlb_err, "sigma_K_noise_eV",   lambda: s_main.sigma_K_noise / eV)
    _safe(crlb, crlb_err, "sigma_f_noise_Hz",   lambda: s_main.sigma_f_noise / Hz)
    _safe(crlb, crlb_err, "var_f_c_CRLB_Hz2",   lambda: s_main.var_f_c_CRLB / Hz**2)
    _safe(crlb, crlb_err, "time_window_s",      lambda: s_main.time_window / s)
    _safe(crlb, crlb_err, "best_time_window_s", lambda: s_main.best_time_window / s)
    _safe(crlb, crlb_err, "slope_Hz_per_s",     lambda: s_main.slope / (Hz / s))
    _safe(crlb, crlb_err, "eta",                lambda: s_main.eta)
    _safe(crlb, crlb_err, "CRLB_constant",      lambda: s_main.CRLB_constant)
    # p/q live only in the sideband/pitch-dependent branch and can be NaN by
    # construction (division by a zero pitch-complement at theta=pi/2). The
    # NaN-aware gate treats NaN==NaN as "unchanged".
    _safe(crlb, crlb_err, "crlb_p", lambda: getattr(s_main, "p"))
    _safe(crlb, crlb_err, "crlb_q", lambda: getattr(s_main, "q"))
    out.update(crlb)
    errors.update(crlb_err)

    # tau_snr for the 90-degree carrier at the configured time window
    _safe(out, errors, "tau_snr_90_s",
          lambda: float(s_main.calculate_tau_snr(s_main.time_window, 1)) / s)
    _safe(out, errors, "noise_temp_K",
          lambda: float(np.atleast_1d(s_main.noise_temp)[0]) / K)
    _safe(out, errors, "received_power_W",
          lambda: float(np.atleast_1d(s_main.received_power).ravel()[0]) / W)

    # Sensitivity decomposition (eV^2 on m_beta^2) and energy window
    _safe(out, errors, "StatSens_eV2", lambda: s_main.StatSens() / eV**2)
    _safe(out, errors, "SystSens_eV2", lambda: s_main.SystSens() / eV**2)
    _safe(out, errors, "DeltaEWidth_eV", lambda: s_main.DeltaEWidth() / eV)
    _safe(out, errors, "detection_efficiency", lambda: s_main.detection_efficiency)
    _safe(out, errors, "RF_background_rate_per_eV_s",
          lambda: s_main.RF_background_rate_per_eV * eV * s)


    # ===================================================================
    # Intermediate gate quantities -- the per-phase reduction targets.
    # Added so a regression fails loudly at the responsible phase instead
    # of only showing up as a small shift in CL90 at the very end.
    # ===================================================================

    # -- Phase 2: geometry / frequency --
    _safe(out, errors, "cavity_volume_m3", lambda: s_main.CavityVolume() / m**3)
    _safe(out, errors, "cyc_rad_m",        lambda: s_main.cyc_rad / m)
    _safe(out, errors, "required_bw_axialfrequency_Hz",
          lambda: s_main.required_bw_axialfrequency / Hz)
    # Per-mode TE_01l resonant frequencies (added in Phase 2). CavityModeFrequency(1)
    # must equal cavity_freq exactly; p=2,3 are the multimode channels.
    _safe(out, errors, "cavity_length_m", lambda: s_main.cavity_length / m)
    _safe(out, errors, "mode_freq_p1_Hz", lambda: s_main.CavityModeFrequency(1) / Hz)
    _safe(out, errors, "mode_freq_p2_Hz", lambda: s_main.CavityModeFrequency(2) / Hz)
    _safe(out, errors, "mode_freq_p3_Hz", lambda: s_main.CavityModeFrequency(3) / Hz)
    _safe(out, errors, "mode_freq_p1_minus_cavity_freq_rel",
          lambda: float(abs(s_main.CavityModeFrequency(1) - s_main.cavity_freq)
                        / s_main.cavity_freq))

    # -- Phase 3: mode/port data structures (default 1 TE011 + 1 port) --
    # These lock in that a nominal cfg builds exactly one TE011 mode and one
    # port, with the port carrying the existing single-chain config values.
    _safe(out, errors, "n_modes", lambda: len(s_main.modes))
    _safe(out, errors, "n_ports", lambda: len(s_main.ports))
    _safe(out, errors, "mode0_axial_index", lambda: s_main.modes[0].axial_mode_index)
    _safe(out, errors, "mode0_q_unloaded",  lambda: s_main.modes[0].q_unloaded)
    _safe(out, errors, "port0_z_position_m", lambda: s_main.ports[0].z_position / m)
    _safe(out, errors, "port0_amp_temp_K",
          lambda: s_main.ports[0].amplifier_temperature / K)
    _safe(out, errors, "port0_att_line_db", lambda: s_main.ports[0].att_line_db)
    _safe(out, errors, "port0_att_cir_db",  lambda: s_main.ports[0].att_cir_db)
    _safe(out, errors, "port0_quantum_amp_efficiency",
          lambda: s_main.ports[0].quantum_amp_efficiency)

    # -- Phase 4: loaded Q + port coupling (the single-port reduction target) --
    # coupling is a local in the engine; recompute from the primitives so the
    # port solver can be checked against "coupling = unloaded_q/loaded_q - 1".
    _safe(out, errors, "unloaded_q", lambda: s_main.FrequencyExtraction.unloaded_q)
    _safe(out, errors, "coupling_simple",
          lambda: s_main.FrequencyExtraction.unloaded_q / s_main.loaded_q - 1.0)
    # Per-mode Q from SolveExternalQ. For the single TE011 mode these MUST reduce
    # to the inline single-mode quantities (the two *_rel metrics ~ 0).
    _safe(out, errors, "mode0_q_loaded", lambda: s_main.modes[0].q_loaded)
    _safe(out, errors, "mode0_q_ext_port0",
          lambda: list(s_main.modes[0].q_externals.values())[0])
    _safe(out, errors, "mode0_q_loaded_minus_loaded_q_rel",
          lambda: float(abs(s_main.modes[0].q_loaded - s_main.loaded_q) / s_main.loaded_q))
    _safe(out, errors, "mode0_coupling_minus_inline_rel",
          lambda: float(abs(
              s_main.modes[0].q_unloaded / list(s_main.modes[0].q_externals.values())[0]
              - (s_main.FrequencyExtraction.unloaded_q / s_main.loaded_q - 1.0))
              / (s_main.FrequencyExtraction.unloaded_q / s_main.loaded_q - 1.0)))

    # -- Phase 6: SNR / noise denominator pieces --
    _safe(out, errors, "noise_energy_eV",  lambda: s_main.noise_energy / eV)
    _safe(out, errors, "fft_bandwidth_Hz", lambda: s_main.fft_bandwidth / Hz)

    # -- Phase 7 CRLB chain (var_f_c_CRLB, sigma_f_noise, best_time_window,
    #    eta, CRLB_constant, crlb_p, crlb_q) is captured above as a coherent
    #    snapshot right after syst_frequency_extraction(); not repeated here. --

    # -- Phases 5-7: deterministic per-theta distributions (full arrays) --
    # These are the actual inputs to the SNR and CRLB averages; capturing them
    # lets those gates compare the whole distribution, not just a scalar.
    _safe(out, errors, "theta_array_rad",
          lambda: np.asarray(s_main.theta_array, float))
    _safe(out, errors, "carrier_power_fraction_array",
          lambda: np.asarray(s_main.carrier_power_fraction_array, float))
    _safe(out, errors, "sideband_power_fraction_array",
          lambda: np.asarray(s_main.sideband_power_fraction_array, float))
    _safe(out, errors, "prob_theta_array",
          lambda: np.asarray(s_main.prob_theta_array, float))
    # signal_power_vs_r uses seeded RNG radii; capture summary stats (mean/std)
    # rather than the raw draw so it stays reproducible and compact.
    _safe(out, errors, "signal_power_vs_r_mean_W",
          lambda: float(np.mean(s_main.signal_power_vs_r)) / W)
    _safe(out, errors, "signal_power_vs_r_std_W",
          lambda: float(np.std(s_main.signal_power_vs_r)) / W)

    # ---- Optimized plotted point (matches the processor's curve minimum) ----
    if optimize:
        try:
            rhos = np.logspace(np.log10(density_range[0]),
                               np.log10(density_range[1]), n_density) / m**3
            thresholds = np.linspace(thresh_range[0], thresh_range[1], n_thresh)

            # Threshold optimization at the configured density
            thr_limits = [s_main.CL90(Threshold={"detection_threshold": th})
                          for th in thresholds]
            thr_idx = int(np.argmin(thr_limits))
            out["thresh_opt_configured_density"] = float(thresholds[thr_idx])
            out["CL90_threshopt_configured_density_eV"] = float(thr_limits[thr_idx] / eV)

            # Joint density + threshold optimization (the plotted operating point)
            best = []
            best_thr = []
            for rho in rhos:
                tl = []
                for th in thresholds:
                    s_main.Threshold.detection_threshold = th
                    tl.append(s_main.CL90(Experiment={"number_density": rho}))
                k = int(np.argmin(tl))
                best.append(tl[k])
                best_thr.append(float(thresholds[k]))
            opt_idx = int(np.argmin(best))
            out["rho_opt_per_m3"] = float(rhos[opt_idx] * m**3)
            out["thresh_opt_at_rho_opt"] = best_thr[opt_idx]
            out["CL90_optimized_eV"] = float(best[opt_idx] / eV)
        except Exception as exc:  # noqa: BLE001
            errors["optimization"] = "{}: {}".format(type(exc).__name__, exc)
            errors["optimization_traceback"] = traceback.format_exc()

    if errors:
        out["_capture_errors"] = errors
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Freeze single-mode baseline numbers.")
    p.add_argument("config_path", help="Path to the mermithid config (.cfg) file.")
    p.add_argument("--out", default="baseline_old.json",
                   help="Output JSON path (default: baseline_old.json).")
    p.add_argument("--no-optimize", action="store_true",
                   help="Skip the density/threshold optimization scan.")
    p.add_argument("--density-range", nargs=2, type=float, default=(1e14, 1e21),
                   metavar=("LO", "HI"))
    p.add_argument("--n-density", type=int, default=80)
    p.add_argument("--thresh-range", nargs=2, type=float, default=(1.0, 160.0),
                   metavar=("LO", "HI"))
    p.add_argument("--n-thresh", type=int, default=40)
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducible radius sampling (default: 42).")
    args = p.parse_args(argv)

    baseline = build_baseline(
        args.config_path,
        optimize=not args.no_optimize,
        density_range=tuple(args.density_range), n_density=args.n_density,
        thresh_range=tuple(args.thresh_range), n_thresh=args.n_thresh,
        seed=args.seed,
    )

    with open(args.out, "w") as fh:
        json.dump(baseline, fh, indent=2)

    print("Wrote {} keys to {}".format(len(baseline), args.out))
    if "_capture_errors" in baseline:
        print("WARNING: some captures failed (see _capture_errors in the JSON):")
        for k, v in baseline["_capture_errors"].items():
            if not k.endswith("_traceback"):
                print("  - {}: {}".format(k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
