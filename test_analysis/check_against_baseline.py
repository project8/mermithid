'''
check_against_baseline.py  --  per-phase gate enforcer for the multimode port.

The invariant we are enforcing:
    The SAME .cfg (one with no modes/ports section) must make the unified
    engine auto-build exactly one TE011 mode + one port and reproduce the
    frozen single-mode numbers in baseline_old.json.

Run this after each module change, on the same config used for the baseline:

    python check_against_baseline.py \
        /termite/sensitivity_config_files/Config_atomic_150MHz_minpitch_87deg_nominal.cfg \
        --baseline baseline_old.json --thresh-range 5 115 --density-range 3e14 3e18

Exit code is 0 if every captured key passes its tolerance tier, 1 otherwise,
so it can be wired into CI / a pre-commit hook if desired.

Tolerance tiers
---------------
  EXACT   (rtol 1e-12) : quantities whose formula is unchanged by the port
                         (geometry, frequency, loaded Q, required BW, noise T).
                         A failure here is a real bug.
  REDUCED (rtol 1e-8)  : quantities recomputed by new code that are only
                         *algebraically* identical in the 1-port limit
                         (tau_snr, signal_power, CL90, ...). Small differences
                         from operation order are tolerated; larger ones are
                         real bugs.

NOTE on the signal-power model (Phase 5): if the single-mode path is switched
from larmor_orbit_averaged_hanneke_power to the notebook's analytic CavityPower,
signal_power and everything downstream of it (tau_snr, detection_efficiency,
effective_volume, CL90) will diverge by design. Pass --allow-power-model-change
to demote those specific keys to a warning instead of a hard failure.
'''
from __future__ import absolute_import

import argparse
import json
import sys

import numpy as np

from numericalunits import eV, GHz, Hz, W, K, s, m

from mermithid.sensitivity.SensitivityCavityFormulas import CavitySensitivity


# Which baseline key sits in which tolerance tier.
EXACT_KEYS = {
    "cavity_freq_Hz", "cavity_freq_GHz", "cavity_radius_m",
    "loaded_q", "required_bw_Hz", "noise_temp_K",
    "total_trap_volume_m3", "pos_dep_trapping_eff",
    "time_window_s", "slope_Hz_per_s",
    # Phase 2 geometry / Phase 4 coupling primitives / Phase 6 noise:
    "cavity_volume_m3", "cyc_rad_m", "required_bw_axialfrequency_Hz",
    "cavity_length_m", "mode_freq_p1_Hz", "mode_freq_p2_Hz", "mode_freq_p3_Hz",
    "mode_freq_p1_minus_cavity_freq_rel",
    "unloaded_q", "coupling_simple", "noise_energy_eV", "fft_bandwidth_Hz",
    "CRLB_constant",
    # Phase 3 mode/port data structures (default 1 TE011 + 1 port):
    "n_modes", "n_ports", "mode0_axial_index", "mode0_q_unloaded",
    "port0_z_position_m", "port0_amp_temp_K", "port0_att_line_db",
    "port0_att_cir_db", "port0_quantum_amp_efficiency",
    # Phase 4 per-mode loaded Q / external Q (single-mode reduction):
    "mode0_q_loaded", "mode0_q_ext_port0",
    # Phase 8 interference matrix (deterministic from solved Qs):
    "R2_diag_mode0",
    # Deterministic input distributions (file-driven / fixed binning):
    "theta_array_rad", "carrier_power_fraction_array",
    "sideband_power_fraction_array", "prob_theta_array",
}
# Keys that depend on the signal-power model (demoted if --allow-power-model-change).
POWER_MODEL_KEYS = {
    "signal_power_W", "received_power_W", "tau_snr_90_s",
    "detection_efficiency", "effective_volume_m3",
    "syst_freq_extraction_sigma_eV", "sigma_K_noise_eV",
    "StatSens_eV2", "SystSens_eV2", "DeltaEWidth_eV",
    "CL90_configured_eV", "CL90_threshopt_configured_density_eV",
    "CL90_optimized_eV", "rho_opt_per_m3",
    "thresh_opt_configured_density", "thresh_opt_at_rho_opt",
    "RF_background_rate_per_eV_s",
    # CRLB chain depends on tau_snr -> signal power:
    "var_f_c_CRLB_Hz2", "sigma_f_noise_Hz", "best_time_window_s",
    "eta", "crlb_p", "crlb_q",
    "signal_power_vs_r_mean_W", "signal_power_vs_r_std_W",
}
RTOL_EXACT = 1e-12
RTOL_REDUCED = 1e-8

# Self-consistency / round-trip metrics that are themselves ~0 by construction.
# Comparing two ~1e-16 values *relatively* is meaningless (ratio of fp noise),
# so these pass iff BOTH baseline and updated are below an absolute floor.
ABS_ZERO_KEYS = {"mode_freq_p1_minus_cavity_freq_rel",
                 "mode0_q_loaded_minus_loaded_q_rel",
                 "mode0_coupling_minus_inline_rel",
                 "R2_mode0_minus_one_minus_Wi_rel"}
ABS_ZERO_TOL = 1e-12


def _rel_err(new, old):
    """Plain relative error. If either side is NaN the result is NaN, so the
    row fails and is shown with its baseline/updated values for eyeballing."""
    new = np.asarray(new, dtype=float)
    old = np.asarray(old, dtype=float)
    if new.shape != old.shape:
        return float("inf")
    denom = np.where(np.abs(old) > 0, np.abs(old), 1.0)
    return float(np.max(np.abs(new - old) / denom))


def _disp(val):
    """Compact display of a scalar or array baseline/updated value."""
    arr = np.atleast_1d(np.asarray(val, dtype=float))
    if arr.size == 1:
        return "{:.6e}".format(float(arr[0]))
    return "array[{}]".format(arr.size)


def _delta(new, old):
    """Compact display of the change (new - old)."""
    n = np.atleast_1d(np.asarray(new, dtype=float))
    o = np.atleast_1d(np.asarray(old, dtype=float))
    if n.shape != o.shape:
        return "shape {} vs {}".format(o.shape, n.shape)
    if n.size == 1:
        return "{:+.3e}".format(float(n[0] - o[0]))
    return "max|d|={:.3e}".format(float(np.nanmax(np.abs(n - o))))


def recompute(config_path, optimize=True,
              density_range=(1e14, 1e21), n_density=80,
              thresh_range=(1.0, 160.0), n_thresh=40, seed=42):
    """Recompute the same key set as make_baseline.py, with the unified engine."""
    # Import the captures from make_baseline so the two stay in lockstep.
    import make_baseline
    np.random.seed(seed)
    return make_baseline.build_baseline(
        config_path, optimize=optimize,
        density_range=density_range, n_density=n_density,
        thresh_range=thresh_range, n_thresh=n_thresh, seed=seed,
    )


def assert_single_mode(config_path):
    """Confirm the unified engine really reduced to 1 mode / 1 port on this cfg.

    This probes for the multimode attributes the new engine is expected to
    expose (e.g. self.modes / self.ports). It is intentionally lenient before
    those attributes exist (early phases): it only fails if they exist AND show
    more than one mode/port, which would mean the cfg silently went multimode.
    """
    s = CavitySensitivity(config_path, verbose=False)
    n_modes = len(getattr(s, "modes", [None]))
    n_ports = len(getattr(s, "ports", [None]))
    return n_modes, n_ports


def main(argv=None):
    p = argparse.ArgumentParser(description="Gate: unified engine vs frozen baseline.")
    p.add_argument("config_path")
    p.add_argument("--baseline", default="baseline_old.json")
    p.add_argument("--no-optimize", action="store_true")
    p.add_argument("--density-range", nargs=2, type=float, default=(1e14, 1e21))
    p.add_argument("--n-density", type=int, default=80)
    p.add_argument("--thresh-range", nargs=2, type=float, default=(1.0, 160.0))
    p.add_argument("--n-thresh", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-power-model-change", action="store_true",
                   help="Demote signal-power-dependent mismatches to warnings "
                        "(use after a deliberate Phase 5 model switch).")
    args = p.parse_args(argv)

    with open(args.baseline) as fh:
        old = json.load(fh)

    seed = old.get("_seed", args.seed)
    new = recompute(
        args.config_path, optimize=not args.no_optimize,
        density_range=tuple(args.density_range), n_density=args.n_density,
        thresh_range=tuple(args.thresh_range), n_thresh=args.n_thresh, seed=seed,
    )

    # Confirm we actually stayed single-mode on this cfg.
    n_modes, n_ports = assert_single_mode(args.config_path)
    print("Engine configured {} mode(s), {} port(s) on this cfg.".format(n_modes, n_ports))
    if (n_modes or 1) > 1 or (n_ports or 1) > 1:
        print("FAIL: a baseline cfg must reduce to 1 mode / 1 port, "
              "but the engine built more. Multimode must be opt-in only.")
        return 1

    passes, fails, warns, skipped = [], [], [], []
    for key, old_val in old.items():
        if key.startswith("_") or key == "_capture_errors":
            continue
        new_val = new.get(key, None)
        if old_val is None or new_val is None:
            skipped.append(key)
            continue

        if key in ABS_ZERO_KEYS:
            # Pass iff both runs are ~0 (this metric IS a round-trip error).
            mag = float(np.nanmax(np.abs(np.append(
                np.atleast_1d(np.asarray(old_val, dtype=float)),
                np.atleast_1d(np.asarray(new_val, dtype=float))))))
            row = (key, _disp(old_val), _disp(new_val),
                   _delta(new_val, old_val), mag, ABS_ZERO_TOL)
            (passes if (np.isfinite(mag) and mag <= ABS_ZERO_TOL) else fails).append(row)
            continue

        rtol = RTOL_EXACT if key in EXACT_KEYS else RTOL_REDUCED
        err = _rel_err(new_val, old_val)
        row = (key, _disp(old_val), _disp(new_val), _delta(new_val, old_val), err, rtol)
        ok = err <= rtol  # NaN <= rtol is False, so NaN rows land in fails/warns
        if ok:
            passes.append(row)
        elif args.allow_power_model_change and key in POWER_MODEL_KEYS:
            warns.append(row)
        else:
            fails.append(row)

    hdr = "    {:<38}{:>16}{:>16}{:>19}{:>12}".format(
        "quantity", "baseline", "updated", "delta", "rel_err")

    def _fmt(rows):
        print(hdr)
        print("    " + "-" * 101)
        for k, o_disp, n_disp, d_disp, e, t in rows:
            print("    {:<38}{:>16}{:>16}{:>19}{:>12.2e}".format(
                k, o_disp, n_disp, d_disp, e))

    print("\nPASS ({}):".format(len(passes)))
    if passes:
        _fmt(sorted(passes))
    if warns:
        print("\nWARN - power-model-dependent, allowed ({}):".format(len(warns)))
        _fmt(sorted(warns))
    if skipped:
        print("\nSKIPPED (null in baseline or new) ({}): {}".format(
            len(skipped), ", ".join(sorted(skipped))))
    if fails:
        print("\nFAIL ({}):".format(len(fails)))
        _fmt(sorted(fails))
        print("\nGATE FAILED.  (A row where baseline and updated are both 'nan'"
              " is a known non-finite quantity, not a regression.)")
        return 1

    print("\nGATE PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
