"""
Script to generate a sample detection efficiency map file.
This creates a 4D efficiency map based on:
- ke (kinetic energy)
- theta_center (pitch angle at center)
- r_start (starting radius)
- phi_start (starting azimuthal angle)

Usage examples:
  python generate_sample_efficiency_map.py
  python generate_sample_efficiency_map.py --ke-bins 50 --theta-center-bins 12 --r-bins 50 --phi-bins 12 --output sample_efficiency_map_small.npy > output_small.txt
  python generate_sample_efficiency_map.py --model gaussian_r_theta --ke-bins 100
  python generate_sample_efficiency_map.py --output custom_efficiency.npy

Author: S. M. Lee
Date: March 7, 2026
"""

import numpy as np
import argparse


def generate_efficiency_map(
    ke_bins=50,
    theta_center_bins=36,
    r_bins=50,
    phi_bins=36,
    ke_range=(18000, 19000),
    theta_range=(0, np.pi),
    r_range=(0, 0.01),
    phi_range=(0, 2 * np.pi),
    efficiency_model="radial_falloff",
    output_path="sample_efficiency_map.npy",
):
    """
    Generate a 4D detection efficiency map.

    Parameters:
        ke_bins: Number of bins for kinetic energy
        theta_center_bins: Number of bins for theta_center
        r_bins: Number of bins for radial position
        phi_bins: Number of bins for azimuthal angle
        ke_range: (min, max) for kinetic energy in eV
        theta_range: (min, max) for pitch angles in radians
        r_range: (min, max) for radial position in meters
        phi_range: (min, max) for azimuthal angle in radians
        efficiency_model: Type of efficiency model to generate
        output_path: Path to save the efficiency map
    """
    # Create bin edges
    ke_edges = np.linspace(ke_range[0], ke_range[1], ke_bins + 1)
    theta_center_edges = np.linspace(theta_range[0], theta_range[1], theta_center_bins + 1)
    r_edges = np.linspace(r_range[0], r_range[1], r_bins + 1)
    phi_edges = np.linspace(phi_range[0], phi_range[1], phi_bins + 1)

    # Create bin centers for evaluation
    ke_centers = (ke_edges[:-1] + ke_edges[1:]) / 2
    theta_center_centers = (theta_center_edges[:-1] + theta_center_edges[1:]) / 2
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2

    # Create meshgrid for efficiency calculation
    KE, THETA_C, R, PHI = np.meshgrid(
        ke_centers,
        theta_center_centers,
        r_centers,
        phi_centers,
        indexing="ij",
    )

    # Generate efficiency values based on the model
    if efficiency_model == "uniform":
        # Uniform efficiency
        efficiency = np.ones_like(KE) * 0.95

    elif efficiency_model == "gaussian_r_theta":
        # Gaussian in radius and peaked around theta=pi/2
        r_peak = (r_range[0] + r_range[1]) / 2
        r_sigma = (r_range[1] - r_range[0]) / 6
        theta_peak = np.pi / 2
        theta_sigma = np.pi / 6

        r_eff = np.exp(-((R - r_peak) ** 2) / (2 * r_sigma**2))
        theta_c_eff = np.exp(-((THETA_C - theta_peak) ** 2) / (2 * theta_sigma**2))

        efficiency = 0.5 + 0.45 * r_eff * theta_c_eff

    elif efficiency_model == "radial_falloff":
        # Efficiency decreases with radius
        r_max = r_range[1]
        efficiency = 0.95 * (1 - (R / r_max) ** 2) + 0.05

    elif efficiency_model == "theta_dependent":
        # Efficiency peaked around pi/2 for theta_center
        theta_c_eff = np.sin(THETA_C)
        efficiency = 0.5 + 0.5 * theta_c_eff

    else:
        raise ValueError(f"Unknown efficiency model: {efficiency_model}")

    # Ensure efficiency is in [0, 1]
    efficiency = np.clip(efficiency, 0, 1)

    # Save the efficiency map and bin edges
    np.save(
        output_path,
        {
            "efficiency": efficiency,
            "ke_edges": ke_edges,
            "theta_center_edges": theta_center_edges,
            "r_edges": r_edges,
            "phi_edges": phi_edges,
        },
    )

    print(f"Generated efficiency map with shape: {efficiency.shape}")
    print(f"Efficiency range: [{efficiency.min():.3f}, {efficiency.max():.3f}]")
    print(f"Mean efficiency: {efficiency.mean():.3f}")
    print(f"Saved to: {output_path}")

    return efficiency, ke_edges, theta_center_edges, r_edges, phi_edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a sample detection efficiency map file"
    )
    parser.add_argument(
        "--ke-bins", type=int, default=50, help="Number of kinetic energy bins"
    )
    parser.add_argument(
        "--theta-center-bins",
        type=int,
        default=180,
        help="Number of theta_center bins",
    )
    parser.add_argument("--r-bins", type=int, default=50, help="Number of radius bins")
    parser.add_argument(
        "--phi-bins", type=int, default=36, help="Number of azimuthal angle bins"
    )
    parser.add_argument(
        "--ke-min", type=float, default=18000, help="Minimum kinetic energy (eV)"
    )
    parser.add_argument(
        "--ke-max", type=float, default=19000, help="Maximum kinetic energy (eV)"
    )
    parser.add_argument(
        "--r-max", type=float, default=0.01, help="Maximum radius (m)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="radial_falloff",
        choices=["uniform", "gaussian_r_theta", "radial_falloff", "theta_dependent"],
        help="Efficiency model to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_efficiency_map.npy",
        help="Output file path",
    )

    args = parser.parse_args()
    
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    generate_efficiency_map(
        ke_bins=args.ke_bins,
        theta_center_bins=args.theta_center_bins,
        r_bins=args.r_bins,
        phi_bins=args.phi_bins,
        ke_range=(args.ke_min, args.ke_max),
        r_range=(0, args.r_max),
        efficiency_model=args.model,
        output_path=args.output,
    )
