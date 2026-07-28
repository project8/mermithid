'''
check_multimode.py -- Phase 9 engine-side multimode validation.

Run on a MULTIMODE config (axial_mode_indices = 1,2,3 + port_z_fractions):

    python check_multimode.py /termite/sensitivity_config_files/Config_atomic_150MHz_minpitch_87deg_multimode.cfg

Part A checks internal invariants that need no external reference.
Part B dumps the numbers to compare against the notebook (the second golden
reference): per-mode frequencies, loaded Qs, external Qs, signal powers,
R^2 matrix, interference factors, and combined vs single-mode tau/CL90.
'''
from __future__ import absolute_import
import sys
import json

import numpy as np
from numericalunits import eV, Hz, GHz, W, s, m, K

from mermithid.sensitivity.SensitivityCavityFormulas import CavitySensitivity


def main(config_path, out_json="multimode_engine_reference.json"):
    np.random.seed(42)
    sens = CavitySensitivity(config_path, verbose=False)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print("  [{}] {} {}".format(status, name, detail))

    n_modes, n_ports = len(sens.modes), len(sens.ports)
    print("Engine built {} modes, {} ports from {}".format(n_modes, n_ports, config_path))

    print("\n=== Part A: internal invariants ===")
    check("multimode active", n_modes > 1 and n_ports > 1,
          "({} modes / {} ports)".format(n_modes, n_ports))

    freqs = [sens.CavityModeFrequency(mm.axial_mode_index) for mm in sens.modes]
    check("mode frequencies strictly increasing with p",
          all(freqs[i] < freqs[i+1] for i in range(len(freqs)-1)),
          "({} GHz)".format([round(f/GHz, 6) for f in freqs]))
    check("primary mode on cyclotron frequency",
          abs(freqs[0] - sens.cavity_freq)/sens.cavity_freq < 1e-12)

    R2 = sens.BuildInterferenceMatrix()
    offdiag = float(np.max(np.abs(R2 - np.diag(np.diag(R2))))) if n_modes > 1 else 0.0
    check("R2 Hermitian", bool(np.allclose(R2, R2.conj().T)))
    check("R2 off-diagonal (cross-talk) small", offdiag < 1e-6,
          "(max |R2_ab| = {:.2e})".format(offdiag))
    diag = np.real(np.diag(R2))
    check("R2 diagonal in [0,1] (unobserved-power fraction)",
          bool(np.all((diag >= 0) & (diag <= 1))), "({})".format(np.round(diag, 5)))

    # per-mode achieved loaded Q vs target (SolveExternalQ may not reach all targets)
    for a, mode in enumerate(sens.modes):
        tgt = sens.CavityLoadedQ(f_mode=freqs[a],
                                 tuning_pitch=sens.FrequencyExtraction.minimum_angle_in_bandwidth)
        print("  [INFO] TE01{}: target Q_L = {:.1f}, achieved = {:.1f} ({:+.1%})".format(
            mode.axial_mode_index, tgt, mode.q_loaded, (mode.q_loaded - tgt)/tgt))

    powers = np.array([sens.signal_power] + [mm.signal_power for mm in sens.modes[1:]])
    check("higher modes suppressed (P decreases with p)",
          bool(np.all(np.diff(powers) < 0)),
          "({} W)".format(["{:.3e}".format(p/W) for p in powers]))

    tw = sens.time_window if hasattr(sens, "time_window") else 5e-3*s
    tau_comb = float(sens.calculate_tau_snr(tw, 1))
    modes_all = sens.modes
    sens.modes = modes_all[:1]
    tau_m0 = float(sens.calculate_tau_snr(tw, 1))
    sens.modes = modes_all
    check("combined tau <= primary-mode tau", tau_comb <= tau_m0*(1+1e-12),
          "(combined {:.4e} vs mode0 {:.4e}, gain {:.2%})".format(
              tau_comb/s, tau_m0/s, 1 - tau_comb/tau_m0))

    print("\n=== Part B: reference dump for notebook comparison ===")
    ref = {
        "config": config_path,
        "n_modes": n_modes, "n_ports": n_ports,
        "mode_freqs_Hz": [f/Hz for f in freqs],
        "q_loaded": [mm.q_loaded for mm in sens.modes],
        "q_ext": [{p: (q if np.isfinite(q) else None) for p, q in mm.q_externals.items()}
                  for mm in sens.modes],
        "signal_powers_W": [p/W for p in powers],
        "R2_real": np.real(R2).tolist(),
        "R2_imag": np.imag(R2).tolist(),
        "tau_combined_s": tau_comb/s,
        "tau_mode0_only_s": tau_m0/s,
        "CL90_eV": float(sens.CL90()/eV),
    }
    json.dump(ref, open(out_json, "w"), indent=2)
    print("  wrote {}".format(out_json))
    for k, v in ref.items():
        if k not in ("R2_real", "R2_imag", "q_ext"):
            print("  {:22s} {}".format(k, v))

    print("\n{}".format("MULTIMODE INVARIANTS PASSED" if ok else "SOME INVARIANTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    cfg = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "multimode_engine_reference.json"
    sys.exit(main(cfg, out))
