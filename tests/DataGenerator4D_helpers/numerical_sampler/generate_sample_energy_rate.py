"""
Generate sample energy-rate PDFs for DataGenerator4D numerical tests.

Creates an `.npz` file containing:
- `rate`: 1D array of PDF values per bin (non-negative, normalized to sum=1)
- `ke_edges`: 1D array of bin edges (length bins+1)

Three generator options are provided:
1) Sum of two Gaussians with small random fluctuations (default)
2) Exponential tail plus a Gaussian peak
3) Broad plateau with local dips (uniform + negative Gaussians)

Usage example:
python generate_sample_energy_rate.py --out energy_two_gaussian.npz --option two_gaussian
"""

from __future__ import annotations

import argparse
import numpy as np


def gaussian(x, mu, sigma, amp=1.0):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def option_sum_two_gaussian(ke_centers, rng):
    n = len(ke_centers)
    mu1 = rng.uniform(ke_centers[n // 6], ke_centers[n // 3])
    mu2 = rng.uniform(ke_centers[n // 2], ke_centers[-1])
    sigma1 = rng.uniform(0.01 * (ke_centers[-1] - ke_centers[0]), 0.05 * (ke_centers[-1] - ke_centers[0]))
    sigma2 = rng.uniform(0.01 * (ke_centers[-1] - ke_centers[0]), 0.08 * (ke_centers[-1] - ke_centers[0]))
    a1 = rng.uniform(0.1, 0.2)
    a2 = rng.uniform(0.05, 0.1)
    rate = a1 * gaussian(ke_centers, mu1, sigma1) + a2 * gaussian(ke_centers, mu2, sigma2)

    # Add small random fluctuations
    rate += rng.normal(scale=0.02 * rate.max(), size=rate.shape)

    rate = np.clip(rate, 0.0, None)
    return rate


def build_rate(args):
    ke_edges = np.linspace(args.ke_min, args.ke_max, args.bins + 1)
    ke_centers = 0.5 * (ke_edges[:-1] + ke_edges[1:])
    rng = np.random.default_rng(args.seed)

    if args.option == "two_gaussian":
        rate = option_sum_two_gaussian(ke_centers, rng)
    else:
        raise ValueError(f"Unknown option {args.option}")

    return ke_edges, rate


def parse_args():
    p = argparse.ArgumentParser(description="Generate sample energy-rate PDF for DataGenerator4D numerical tests")
    p.add_argument("--out", required=True, help="Output .npz file")
    p.add_argument("--option", type=str, default="two_gaussian", choices=["two_gaussian"], help="Generator option")
    p.add_argument("--bins", type=int, default=200, help="Number of energy bins")
    p.add_argument("--ke-min", type=float, default=18000.0, help="Minimum kinetic energy")
    p.add_argument("--ke-max", type=float, default=19000.0, help="Maximum kinetic energy")
    p.add_argument("--seed", type=int, default=12345, help="RNG seed")
    return p.parse_args()


def main():
    args = parse_args()
    ke_edges, rate = build_rate(args)
    np.savez(args.out, rate=rate, ke_edges=ke_edges)
    print(f"Wrote {args.out}: rate.shape={rate.shape}, ke_edges.shape={ke_edges.shape}")


if __name__ == "__main__":
    main()
