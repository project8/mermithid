"""
Generate sample spatial (r, theta, phi) PDFs for DataGenerator4D numerical tests.

Creates an `.npz` file containing:
- `pdf`: 3D array shaped (n_theta_bins, n_r_bins, n_phi_bins)
- `r_edges`: 1D array of radial bin edges (length n_r_bins+1)
- `theta_edges`: 1D array of angular bin edges (length n_theta_bins+1)
- `phi_edges`: 1D array of azimuthal bin edges (length n_phi_bins+1)

Three generator options are provided:
1) Flat rectangular distribution. Cylindrical radius, r, and azimuthal angle, phi,
    follow the uniform distribution in a user-set rectangle, while the pitch angle,
    theta, is uniform in [0, pi].

Usage example:
python generate_sample_spatial_distribution.py --out spatial_box.npz --option box
"""

from __future__ import annotations

import argparse
import numpy as np


def cartesian_to_r_phi(x, y):
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    # map phi to [0, 2pi)
    phi = np.mod(phi, 2 * np.pi)
    return r, phi


def generate_box_histogram(r_edges, theta_edges, phi_edges, rng, x0, y0, x1, y1, samples=2000000):
    # Uniform theta in [0, pi]
    theta_samples = rng.uniform(0.0, np.pi, size=samples)
    
    # Uniform in x,y in [x0, x1] and [y0, y1]
    x = rng.uniform(x0, x1, size=samples)
    y = rng.uniform(y0, y1, size=samples)
    r, phi = cartesian_to_r_phi(x, y)

    # Histogram in (theta, r, phi)
    hist, _ = np.histogramdd(
        np.column_stack((r, theta_samples, phi)),
        bins=(r_edges, theta_edges, phi_edges)
    )

    return hist


def build_pdf(args):
    r_edges = np.linspace(0.0, args.r_max, args.r_bins + 1)
    theta_edges = np.linspace(0.0, np.pi, args.theta_bins + 1)
    phi_edges = np.linspace(0.0, 2 * np.pi, args.phi_bins + 1)
    rng = np.random.default_rng(args.seed)

    if args.option == "box":
        hist = generate_box_histogram(r_edges, theta_edges, phi_edges, rng, args.box_center_x0, args.box_center_y0, args.box_center_x1, args.box_center_y1, samples=args.samples)
        pdf = hist.astype(float)
    else:
        raise ValueError(f"Unknown option {args.option}")

    # Ensure non-negative and normalize to sum=1
    pdf = np.clip(pdf, 0.0, None)
    
    dr = np.diff(r_edges)[:, np.newaxis, np.newaxis]  # (r_bins, 1, 1)
    dtheta = np.diff(theta_edges)[np.newaxis, :, np.newaxis]  # (1, theta_bins, 1)
    dphi = np.diff(phi_edges)[np.newaxis, np.newaxis, :]  # (1, 1, phi_bins)

    normalization = np.sum(pdf * dr * dtheta * dphi)
    if normalization <= 0:
        raise RuntimeError("Generated spatial PDF has non-positive sum")
    pdf = pdf / normalization
    return r_edges, theta_edges, phi_edges, pdf


def parse_args():
    p = argparse.ArgumentParser(description="Generate sample spatial (r, theta, phi) PDF for DataGenerator4D numerical tests")
    p.add_argument("--out", required=True, help="Output .npz file")
    p.add_argument("--option", type=str, default="box", choices=["box"], help="Generator option")
    p.add_argument("--theta-bins", type=int, default=180, help="Number of angular bins")
    p.add_argument("--r-max", type=float, default=0.007, help="Maximum radius (m)")
    p.add_argument("--r-bins", type=int, default=140, help="Number of radial bins")
    p.add_argument("--phi-bins", type=int, default=360, help="Number of azimuthal bins")
    p.add_argument("--samples", type=int, default=2000000, help="Number of random samples to generate")
    p.add_argument("--box-center-x0", type=float, default=-0.001, help="x0 of box for box option (m)")
    p.add_argument("--box-center-y0", type=float, default=-0.001, help="y0 of box for box option (m)")
    p.add_argument("--box-center-x1", type=float, default=+0.001, help="x1 of box for box option (m)")
    p.add_argument("--box-center-y1", type=float, default=+0.001, help="y1 of box for box option (m)")
    p.add_argument("--seed", type=int, default=12345, help="RNG seed")
    return p.parse_args()


def main():
    args = parse_args()
    r_edges, theta_edges, phi_edges, pdf = build_pdf(args)
    np.savez(args.out, pdf=pdf, theta_edges=theta_edges, r_edges=r_edges, phi_edges=phi_edges)
    print(f"Wrote {args.out}: pdf.shape={pdf.shape}, r_edges.shape={r_edges.shape}, theta_edges.shape={theta_edges.shape}, phi_edges.shape={phi_edges.shape}")


if __name__ == "__main__":
    main()
