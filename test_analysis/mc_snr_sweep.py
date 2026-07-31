'''
mc_snr_sweep.py -- Monte Carlo validation across a range of SNR, with a plot.

mc_validate.py checks the predicted variance at the nominal operating point.
This sweeps the signal power over several decades and asks a different
question: WHERE is the CRLB actually achievable?

Frequency estimation has a threshold effect. Well above threshold the estimator
tracks the bound (ratio ~1). Below it, the periodogram peak lands on a noise
spike, the estimate becomes uniform over the band, and the variance jumps by
orders of magnitude. The CRLB is only a meaningful sensitivity prediction above
that threshold, so this plot shows how much margin the operating point has.

Usage
-----
    python mc_snr_sweep.py <config.cfg> [--trials 200] [--nsamp 4096]
                           [--scale-min 0.003] [--scale-max 30] [--n-scale 14]
                           [--stage {1,2,3}] [--out mc_snr_sweep.png]

The sweep multiplies the emitted power Pe by a scale factor; scale = 1 is the
engine's nominal operating point and is marked on the plot.
'''
from __future__ import absolute_import

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numericalunits import Hz, s, W

from mermithid.sensitivity.SensitivityCavityFormulas import CavitySensitivity

from mc_validate import run_stage


def to_plain(mdl):
    out = dict(mdl)
    out["Sigma"] = np.asarray(mdl["Sigma"], dtype=float)/W
    out["Pn_rf"] = np.asarray(mdl["Pn_rf"], dtype=float)/W
    out["Pn_cav"] = np.asarray(mdl["Pn_cav"], dtype=float)/W
    out["Pe"] = float(np.mean(np.atleast_1d(mdl["Pe"])))/W
    out["F_sig"] = float(np.mean(np.atleast_1d(mdl["F_sig"])))
    out["fft_bandwidth"] = float(mdl["fft_bandwidth"]/Hz)
    out["v"] = np.asarray(mdl["v"], dtype=float)
    return out


def predicted_tau(models, m_idx, p_idx):
    """Engine's tau for this mode/port subset: 1/tau = sum_modes Pe*F*v^T S^-1 v*bw."""
    inv_tau = 0.0
    for a in m_idx:
        mdl = models[a]
        S = mdl["Sigma"][np.ix_(p_idx, p_idx)]
        v = mdl["v"][p_idx]
        inv_tau += mdl["Pe"]*mdl["F_sig"]*(v @ np.linalg.solve(S, v))*mdl["fft_bandwidth"]
    return 1.0/inv_tau


def main(argv=None):
    p = argparse.ArgumentParser(description="MC validation across SNR, with plot.")
    p.add_argument("config_path")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--nsamp", type=int, default=4096)
    p.add_argument("--scale-min", type=float, default=0.003)
    p.add_argument("--scale-max", type=float, default=30.0)
    p.add_argument("--n-scale", type=int, default=14)
    p.add_argument("--stage", type=int, default=None, choices=[1, 2, 3],
                   help="Run only one stage (default: all applicable).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--components", default="full",
                   choices=["full", "carrier", "sideband", "carrier+sideband"],
                   help="Which signal populates the readout model. 'full' uses "
                        "power_fraction=1 (a 90-deg carrier electron, matching "
                        "mc_validate.py); the others use the per-theta power "
                        "fractions for that component, as syst_frequency_extraction "
                        "does. This choice sets the nominal SNR, so state it when "
                        "quoting results.")
    p.add_argument("--out", default="mc_snr_sweep.png")
    args = p.parse_args(argv)

    np.random.seed(42)
    sens = CavitySensitivity(args.config_path, verbose=False)
    sens.syst_frequency_extraction()
    T_s = float(sens.time_window/s)
    # Populate the readout models EXPLICITLY, so the operating point is defined
    # by this script rather than by whichever internal call happened to run last.
    if args.components == "full":
        sens.calculate_tau_snr(sens.time_window, 1)
    else:
        comps = tuple(args.components.split("+"))
        sens.calculate_tau_snr(sens.time_window, components=comps)
    models = [to_plain(m) for m in sens._last_readout_models]
    n_modes, n_ports = len(models), len(models[0]["v"])

    fs = args.nsamp/T_s
    f0 = 0.2*fs
    rng = np.random.default_rng(args.seed)

    stages = [("single mode, single port", [0], [0]),
              ("single mode, all ports", [0], list(range(n_ports)))]
    if n_modes > 1:
        stages.append(("all modes, all ports", list(range(n_modes)), list(range(n_ports))))
    if args.stage is not None:
        stages = [stages[args.stage-1]]

    scales = np.logspace(np.log10(args.scale_min), np.log10(args.scale_max), args.n_scale)
    print("Config: {}".format(args.config_path))
    print("{} mode(s), {} port(s); T = {:.4e} s; {} trials/point".format(
        n_modes, n_ports, T_s, args.trials))
    print("signal populating the readout model: --components {}\n".format(args.components))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    results = {}

    for label, m_idx, p_idx in stages:
        pred_list, emp_list, snr_list = [], [], []
        print("{:<28}{:>12}{:>14}{:>14}{:>9}".format(
            label, "int.SNR", "predicted", "empirical", "ratio"))
        for sc in scales:
            scaled = []
            for mdl in models:
                m2 = dict(mdl)
                m2["Pe"] = mdl["Pe"]*sc
                scaled.append(m2)
            tau = predicted_tau(scaled, m_idx, p_idx)
            pred = (6.0*tau/T_s**3)/(2*np.pi)**2
            emp, _ = run_stage(rng, scaled, m_idx, p_idx, f0, T_s,
                               args.nsamp, args.trials)
            int_snr = args.nsamp/(tau*fs)
            pred_list.append(pred); emp_list.append(emp); snr_list.append(int_snr)
            print("{:<28}{:>12.4g}{:>14.4e}{:>14.4e}{:>9.3f}".format(
                "  scale={:.3g}".format(sc), int_snr, pred, emp, emp/pred))
        print()
        results[label] = (np.array(snr_list), np.array(pred_list), np.array(emp_list))

        ax1.loglog(snr_list, emp_list, "o-", ms=4, label="MC: " + label)
        ax1.loglog(snr_list, pred_list, "--", lw=1, alpha=.7,
                   label="CRLB: " + label)
        ax2.semilogx(snr_list, np.array(emp_list)/np.array(pred_list), "o-", ms=4,
                     label=label)

    # nominal operating point (scale = 1) for the widest stage
    label0, m0, p0 = stages[-1]
    tau_nom = predicted_tau(models, m0, p0)
    snr_nom = args.nsamp/(tau_nom*fs)
    for ax in (ax1, ax2):
        ax.axvline(snr_nom, color="k", ls=":", lw=1.2)
        ax.set_xlabel("integrated SNR")
    ax1.annotate("nominal", xy=(snr_nom, ax1.get_ylim()[1]), xytext=(4, -12),
                 textcoords="offset points", fontsize=9, rotation=90, va="top")
    ax1.set_ylabel(r"var($\hat{f}$)  [Hz$^2$]")
    ax1.set_title("Empirical variance vs CRLB")
    ax1.legend(fontsize=7); ax1.grid(alpha=.3, which="both")
    ax2.axhline(1.0, color="k", lw=1)
    ax2.set_ylabel("empirical / CRLB")
    ax2.set_title("Ratio (1 = bound achieved; >>1 = below threshold)")
    ax2.set_yscale("log"); ax2.legend(fontsize=8); ax2.grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print("wrote {}".format(args.out))
    print("nominal integrated SNR = {:.4g}  (--components {})".format(snr_nom, args.components))
    if snr_nom < 50:
        print("NOTE: the nominal point is at or below the estimator threshold seen in\n"
              "      this sweep, so the CRLB is not achievable there by a periodogram\n"
              "      estimator. Worth checking against the carrier / full-power case.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
