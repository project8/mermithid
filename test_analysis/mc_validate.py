'''
mc_validate.py -- Monte Carlo validation of the multimode readout chain.

Everything else we run (the baseline gate, check_multimode) tests INTERNAL
consistency. This script tests whether the predicted frequency variance is
actually achievable: it synthesises per-port time series from the engine's own
signal vector and noise covariance, runs a real frequency estimator, and
compares the empirical variance with the engine's prediction.

Three stages of increasing scope:

  Stage 1  single mode, single port
           -> validates the tau_SNR -> CRLB link and the estimator itself.
  Stage 2  single mode, all ports
           -> validates Sigma (correlated cavity noise + independent RF noise)
              and the max-SNR combiner w = Sigma^-1 v.
  Stage 3  all modes, all ports, signals summed COHERENTLY at f_c
           -> validates the 1/tau = sum_q 1/tau_q mode combination.
              NOTE: all modes carry the SAME cyclotron frequency; they are
              separable only through their orthogonal spatial patterns. This
              stage also exposes the relative mode phases, which the engine
              does not model (it uses |power| only) -- see --phase-mode.

Usage
-----
    python mc_validate.py <config.cfg> [--trials 400] [--nsamp 4096]
                          [--phase-mode {locked,random}] [--seed 1]

Reads the per-mode readout model that calculate_tau_snr stashed on the engine
(self._last_readout_models), so the simulation can never drift from the
implementation it is meant to validate.

What this does NOT validate: the inputs themselves (cavity Q, coupling values,
absolute Hanneke power). Those need EM simulation or measurement.
'''
from __future__ import absolute_import

import argparse
import sys

import numpy as np
from numericalunits import Hz, s, W

from mermithid.sensitivity.SensitivityCavityFormulas import CavitySensitivity


# ----------------------------------------------------------------------------
# Core simulation
# ----------------------------------------------------------------------------
def estimate_frequency(y, fs):
    """ML-ish frequency estimate: periodogram peak + quadratic interpolation."""
    n = len(y)
    nfft = 1 << int(np.ceil(np.log2(n)) + 4)
    P = np.abs(np.fft.fft(y, nfft))**2
    k = int(np.argmax(P))
    a, b, c = P[(k-1) % nfft], P[k], P[(k+1) % nfft]
    denom = (a - 2*b + c)
    delta = 0.5*(a - c)/denom if denom != 0 else 0.0
    f = ((k + delta) % nfft)*fs/nfft
    return f


def synthesise_noise(rng, models, n_samples, fs, bandwidth, single_mode_index=None):
    """Per-port noise time series.

    Physically there is ONE noise vector per port:
        n_i(t) = sum_a sign_{a,i} sqrt(Pn_cav_{a,i}) c_a(t)   [cavity, correlated
                                                               across ports,
                                                               independent between
                                                               modes]
                 + sqrt(Pn_rf_i) w_i(t)                       [RF, independent]
    The engine's per-mode Sigma includes only that mode's cavity noise; including
    the other modes' cavity noise here is deliberate, so Stage 3 tests that
    omission too.
    """
    n_ports = len(models[0]["v"])
    # PSD -> per-sample variance for complex circular noise
    scale = fs/bandwidth
    noise = np.zeros((n_ports, n_samples), dtype=complex)

    use = models if single_mode_index is None else [models[single_mode_index]]
    for mdl in use:
        c = (rng.normal(size=n_samples) + 1j*rng.normal(size=n_samples))/np.sqrt(2)
        signs = np.sign(mdl["v"])
        signs[signs == 0] = 1.0
        for i in range(n_ports):
            noise[i] += signs[i]*np.sqrt(max(mdl["Pn_cav"][i], 0.0)*scale)*c

    # RF noise: one draw per port, shared by all modes (it is one physical chain)
    rf = models[0]["Pn_rf"]
    for i in range(n_ports):
        w = (rng.normal(size=n_samples) + 1j*rng.normal(size=n_samples))/np.sqrt(2)
        noise[i] += np.sqrt(max(rf[i], 0.0)*scale)*w
    return noise


def run_stage(rng, models, mode_indices, port_indices, f0, T, n_samples,
              n_trials, phase_mode="locked", power_scale=1.0):
    """Simulate, combine, estimate; return empirical variance of f_hat [Hz^2]."""
    fs = n_samples/T
    t = np.arange(n_samples)/fs
    bandwidth = models[0]["fft_bandwidth"]

    sub = [models[a] for a in mode_indices]
    # signal amplitude per (mode, port); scalar Pe only (per-theta arrays are
    # averaged, since the estimator sees one electron at a time)
    amps = []
    for mdl in sub:
        amps.append(np.sqrt(power_scale*mdl["Pe"]*mdl["F_sig"])*mdl["v"][port_indices])
    amps = np.array(amps)

    # combiner from the PRIMARY mode of this stage, using the engine's Sigma
    Sig = sub[0]["Sigma"][np.ix_(port_indices, port_indices)]
    w = np.linalg.solve(Sig, sub[0]["v"][port_indices])

    ests = np.empty(n_trials)
    for k in range(n_trials):
        noise = synthesise_noise(rng, models, n_samples, fs, bandwidth,
                                 single_mode_index=(mode_indices[0]
                                                    if len(mode_indices) == 1 else None))
        x = noise[port_indices, :].copy()
        for a, mdl in enumerate(sub):
            if phase_mode == "random":
                phi = rng.uniform(0, 2*np.pi)
            else:
                phi = 0.0            # locked: all modes in phase (engine implicitly)
            carrier = np.exp(2j*np.pi*f0*t + 1j*phi)
            for j, _ in enumerate(port_indices):
                x[j] += amps[a][j]*carrier
        y = w @ x
        ests[k] = estimate_frequency(y, fs)
    return float(np.var(ests)), ests


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Monte Carlo validation of the readout chain.")
    p.add_argument("config_path")
    p.add_argument("--trials", type=int, default=400)
    p.add_argument("--nsamp", type=int, default=4096)
    p.add_argument("--phase-mode", choices=["locked", "random"], default="locked")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sweep", action="store_true",
                   help="Sweep the signal power and plot empirical vs predicted variance.")
    p.add_argument("--sweep-range", nargs=2, type=float, default=(-2.0, 2.0),
                   metavar=("LO", "HI"), help="log10 power-scale range (default -2 2).")
    p.add_argument("--sweep-points", type=int, default=13)
    p.add_argument("--plot", default="mc_validate_sweep.png")
    args = p.parse_args(argv)

    np.random.seed(42)
    sens = CavitySensitivity(args.config_path, verbose=False)

    # Populate the readout models and the engine's own prediction.
    sigma_f_eV, _ = sens.syst_frequency_extraction()
    T = sens.time_window
    tau = float(np.mean(np.atleast_1d(sens.calculate_tau_snr(T, 1))))
    models = getattr(sens, "_last_readout_models", None)
    if not models:
        print("ERROR: engine did not expose _last_readout_models "
              "(is the calculate_tau_snr stash applied?)")
        return 1

    # The engine's quantities carry numericalunits scale factors. Convert ONCE
    # to plain SI floats here so the whole simulation is unit-consistent:
    #   v, F_sig      dimensionless
    #   Sigma, Pn, Pe -> watts
    #   fft_bandwidth -> hertz
    # (Mixing numericalunits values with plain sample rates silently rescales
    #  the SNR, which shows up as a wildly inflated empirical variance.)
    def _plain(mdl):
        out = dict(mdl)
        out["Sigma"] = np.asarray(mdl["Sigma"], dtype=float)/W
        out["Pn_rf"] = np.asarray(mdl["Pn_rf"], dtype=float)/W
        out["Pn_cav"] = np.asarray(mdl["Pn_cav"], dtype=float)/W
        out["Pe"] = np.mean(np.atleast_1d(mdl["Pe"]))/W
        out["F_sig"] = float(np.mean(np.atleast_1d(mdl["F_sig"])))
        out["fft_bandwidth"] = float(mdl["fft_bandwidth"]/Hz)
        out["v"] = np.asarray(mdl["v"], dtype=float)
        return out
    models = [_plain(m) for m in models]
    T_s = float(T/s)

    n_modes = len(models)
    n_ports = len(models[0]["v"])
    print("Config: {}".format(args.config_path))
    print("{} mode(s), {} port(s); T = {:.4e} s, trials = {}, samples = {}".format(
        n_modes, n_ports, T_s, args.trials, args.nsamp))

    rng = np.random.default_rng(args.seed)
    fs = args.nsamp/T_s
    f0 = 0.2*fs                     # arbitrary in-band test frequency [Hz]

    def predicted_var(tau_eff):
        """Engine's CRLB for a flat track: 6*tau/T^3/(2pi)^2 [Hz^2]."""
        return (6.0*tau_eff/T_s**3)/(2*np.pi)**2

    stages = [("1: single mode, single port", [0], [0]),
              ("2: single mode, all ports",   [0], list(range(n_ports)))]
    if n_modes > 1:
        stages.append(("3: all modes, all ports (coherent)",
                       list(range(n_modes)), list(range(n_ports))))

    print("{:<38}{:>14}{:>14}{:>10}".format("stage", "predicted", "empirical", "ratio"))
    print("-"*76)
    ok = True
    for label, m_idx, p_idx in stages:
        sub = [models[a] for a in m_idx]
        Sig = sub[0]["Sigma"][np.ix_(p_idx, p_idx)]
        v = sub[0]["v"][p_idx]
        qform = v @ np.linalg.solve(Sig, v)
        Pe = sub[0]["Pe"]
        inv_tau = Pe*sub[0]["F_sig"]*qform*sub[0]["fft_bandwidth"]
        if len(m_idx) > 1:      # add the other modes' information (engine's rule)
            for mdl in sub[1:]:
                Sg = mdl["Sigma"][np.ix_(p_idx, p_idx)]
                vv = mdl["v"][p_idx]
                inv_tau += (mdl["Pe"]*mdl["F_sig"]
                            * (vv @ np.linalg.solve(Sg, vv))*mdl["fft_bandwidth"])
        tau_eff = 1.0/inv_tau
        pred = predicted_var(tau_eff)

        emp, _ = run_stage(rng, models, m_idx, p_idx, f0, T_s, args.nsamp,
                           args.trials, phase_mode=args.phase_mode)
        per_sample_snr = 1.0/(tau_eff*fs)
        ratio = emp/pred
        flag = "" if 0.7 < ratio < 1.6 else "   <-- CHECK"
        if flag:
            ok = False
        print("{:<38}{:>14.4e}{:>14.4e}{:>10.3f}{}".format(label, pred, emp, ratio, flag))
        print("{:<38}{:>14}{:>14.3g}".format("   (per-sample SNR, integrated SNR)", "",
                                              per_sample_snr*args.nsamp))

    print("\nA practical estimator sits slightly ABOVE the bound, so ratios of")
    print("1.0-1.3 are expected. Ratios well below 1 mean the prediction is")
    print("pessimistic; well above 1 means the model is optimistic.")
    if n_modes > 1:
        print("\nStage 3 with --phase-mode random probes the relative mode phases,")
        print("which the engine does not model (it combines |power| only).")

    if args.sweep:
        run_sweep(args, models, T_s, fs, f0, stages, predicted_var, rng)
    return 0 if ok else 1


def run_sweep(args, models, T_s, fs, f0, stages, predicted_var, rng):
    """Sweep the signal power and compare empirical vs predicted variance.

    The CRLB scales as 1/SNR; a periodogram estimator tracks it down to a
    threshold SNR, below which peak-picking produces outliers and the variance
    departs sharply upward. That threshold is the practically useful output of
    this sweep: it is the SNR below which the predicted sigma_f is NOT
    achievable, which bears directly on the detection threshold used in the
    sensitivity calculation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scales = np.logspace(args.sweep_range[0], args.sweep_range[1], args.sweep_points)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    colours = ["C0", "C1", "C2"]

    print("\nSweeping signal power ({} points, {} trials each)...".format(
        len(scales), args.trials))
    for si, (label, m_idx, p_idx) in enumerate(stages):
        sub = [models[a] for a in m_idx]
        Sig = sub[0]["Sigma"][np.ix_(p_idx, p_idx)]
        v = sub[0]["v"][p_idx]
        inv_tau_1 = sub[0]["Pe"]*sub[0]["F_sig"]*(v @ np.linalg.solve(Sig, v))*sub[0]["fft_bandwidth"]
        for mdl in sub[1:]:
            Sg = mdl["Sigma"][np.ix_(p_idx, p_idx)]
            vv = mdl["v"][p_idx]
            inv_tau_1 += mdl["Pe"]*mdl["F_sig"]*(vv @ np.linalg.solve(Sg, vv))*mdl["fft_bandwidth"]

        preds, emps, snrs = [], [], []
        for sc in scales:
            tau_eff = 1.0/(inv_tau_1*sc)      # tau ~ 1/signal power
            preds.append(predicted_var(tau_eff))
            snrs.append(args.nsamp/(tau_eff*fs))
            emp, _ = run_stage(rng, models, m_idx, p_idx, f0, T_s, args.nsamp,
                               args.trials, phase_mode=args.phase_mode, power_scale=sc)
            emps.append(emp)
        preds, emps, snrs = np.array(preds), np.array(emps), np.array(snrs)

        short = label.split(":")[1].strip()
        ax1.loglog(snrs, preds, "-", color=colours[si % 3], label="{} (CRLB)".format(short))
        ax1.loglog(snrs, emps, "o", color=colours[si % 3], ms=4, label="{} (MC)".format(short))
        ax2.semilogx(snrs, emps/preds, "o-", color=colours[si % 3], ms=4, label=short)
        print("  {:<34} done".format(short))

    ax1.set_ylabel(r"var($\hat{f}$)  [Hz$^2$]")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3, which="both")
    ax1.set_title("Monte Carlo vs predicted CRLB ({} trials/point)".format(args.trials))
    ax2.axhline(1.0, color="k", lw=0.8, ls="--")
    ax2.set_ylabel("MC / CRLB")
    ax2.set_xlabel("integrated SNR")
    ax2.set_ylim(0, 5)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=140)
    print("wrote {}".format(args.plot))


if __name__ == "__main__":
    sys.exit(main())
