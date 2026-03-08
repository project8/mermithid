"""
Generate 5D energy error probability maps for DataGenerator4D simulator.

Energy error models the difference between observed and true kinetic energy due to:
detector resolution, electronics noise, and signal processing uncertainties.

For each point in 4D phase space (ke_start, theta_center, r, phi), generates a
probability distribution over ke_error values.

Error models:
  - gaussian_uniform: Constant Gaussian resolution (sigma = sigma_base)
  - gaussian_r_dependent (default): Resolution degrades with radius
  - gaussian_ke_dependent: Statistical energy-dependent resolution
  - gaussian_combined: Both radius and energy dependence

Default parameters generate 5D map with:
  Energy errors: [-20, +20] eV (400 bins)
  True energy: 18000-19000 eV (20 bins)
  Pitch angle: [0, π] rad (18 bins)
  Radius: [0, 0.01] m (20 bins)
  Azimuth: [0, 2π] rad (18 bins)
  Base resolution sigma: 5 eV

Output: sample_energy_error_map.npy containing energy_error_map and bin edges.

Usage examples:
  python generate_sample_energy_error_map.py
  python generate_sample_energy_error_map.py --ke-error-bins 100 --ke-start-bins 100 --theta-center-bins 6 --r-bins 10 --phi-bins 6 --output sample_energy_error_map_small.npy
  python generate_sample_energy_error_map.py --model gaussian_combined --sigma-base 1.5

Author: S. M. Lee
Date: March 7, 2026
"""

import numpy as np
import argparse


def generate_energy_error_map(
    error_model="gaussian_r_dependent",
    ke_error_range=(-20, 20),  # (eV)
    ke_error_bins=400,
    ke_start_range=(18000, 19000),  # (eV)
    ke_start_bins=20,
    theta_range=(0, np.pi),  # (rad)
    theta_center_bins=18,
    r_range=(0, 0.01),  # (m)
    r_bins=20,
    phi_range=(0, 2 * np.pi),  # (rad)
    phi_bins=18,
    sigma_base=5.0,
    output_path="sample_energy_error_map.npy",
):
    """
    Generate a 5D energy error probability map.

    Parameters:
        error_model: Type of error model to generate
        ke_error_range: (min, max) for energy error in eV
        ke_error_bins: Number of bins for energy error (eV)
        ke_start_range: (min, max) for kinetic energy in eV
        ke_start_bins: Number of bins for true kinetic energy
        theta_range: (min, max) for pitch angles in radians
        theta_center_bins: Number of bins for theta_center
        r_range: (min, max) for radial position in meters
        r_bins: Number of bins for radial position
        phi_range: (min, max) for azimuthal angle in radians
        phi_bins: Number of bins for azimuthal angle
        sigma_base: Base resolution parameter (eV) for Gaussian models
        output_path: Path to save the energy error map
    """
    # Create bin edges
    ke_error_edges = np.linspace(
        ke_error_range[0], ke_error_range[1], ke_error_bins + 1
    )
    ke_start_edges = np.linspace(
        ke_start_range[0], ke_start_range[1], ke_start_bins + 1
    )
    theta_center_edges = np.linspace(
        theta_range[0], theta_range[1], theta_center_bins + 1
    )
    r_edges = np.linspace(r_range[0], r_range[1], r_bins + 1)
    phi_edges = np.linspace(phi_range[0], phi_range[1], phi_bins + 1)

    # Create bin centers for evaluation
    ke_error_centers = (ke_error_edges[:-1] + ke_error_edges[1:]) / 2
    ke_start_centers = (ke_start_edges[:-1] + ke_start_edges[1:]) / 2
    theta_center_centers = (theta_center_edges[:-1] + theta_center_edges[1:]) / 2
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2

    # Create meshgrid for all 5 dimensions
    # Note: ke_error is first dimension (axis=0) for normalization
    KE_ERROR, KE_START, THETA_C, R, PHI = np.meshgrid(
        ke_error_centers,
        ke_start_centers,
        theta_center_centers,
        r_centers,
        phi_centers,
        indexing="ij",
    )

    # Generate energy error probability distributions based on the model
    if error_model == "gaussian_uniform":
        # Uniform Gaussian with constant sigma
        sigma = sigma_base
        energy_error_map = np.exp(-(KE_ERROR**2) / (2 * sigma**2))

    elif error_model == "gaussian_r_dependent":
        # Gaussian where sigma increases with radius
        # Models worse resolution at larger radii
        r_max = r_range[1]
        alpha = 1.0  # Scaling factor for radial dependence
        sigma = sigma_base * (1 + alpha * R / r_max)
        energy_error_map = np.exp(-(KE_ERROR**2) / (2 * sigma**2))

    elif error_model == "gaussian_ke_dependent":
        # Gaussian where sigma scales with sqrt(energy)
        # Models statistical energy-dependent resolution
        ke_ref = 18600.0  # Reference energy (eV)
        sigma = sigma_base * np.sqrt(KE_START / ke_ref)
        energy_error_map = np.exp(-(KE_ERROR**2) / (2 * sigma**2))

    elif error_model == "gaussian_combined":
        # Combined r and ke dependence
        r_max = r_range[1]
        ke_ref = 18600.0
        alpha = 0.3
        sigma = sigma_base * np.sqrt(KE_START / ke_ref) * (1 + alpha * R / r_max)
        energy_error_map = np.exp(-(KE_ERROR**2) / (2 * sigma**2))

    else:
        raise ValueError(f"Unknown error model: {error_model}")

    # Normalize probability distributions along ke_error axis (axis=0)
    # Each distribution over ke_error should sum to 1
    norm = energy_error_map.sum(axis=0, keepdims=True)
    # Avoid division by zero
    norm = np.where(norm > 0, norm, 1.0)
    energy_error_map = energy_error_map / norm

    # Verify normalization
    sums = energy_error_map.sum(axis=0)
    assert np.allclose(
        sums, 1.0
    ), f"Normalization failed: sum range [{sums.min():.6f}, {sums.max():.6f}]"

    # Save the energy error map and bin edges
    np.save(
        output_path,
        {
            "energy_error_map": energy_error_map,
            "ke_error_edges": ke_error_edges,
            "ke_start_edges": ke_start_edges,
            "theta_center_edges": theta_center_edges,
            "r_edges": r_edges,
            "phi_edges": phi_edges,
        },
    )

    print(f"Generated energy error map with shape: {energy_error_map.shape}")
    print(
        f"  ke_error bins: {ke_error_bins} [{ke_error_range[0]}, {ke_error_range[1]}] eV"
    )
    print(
        f"  ke_start bins: {ke_start_bins} [{ke_start_range[0]}, {ke_start_range[1]}] eV"
    )
    print(
        f"  theta_center bins: {theta_center_bins} [{theta_range[0]:.3f}, {theta_range[1]:.3f}] rad"
    )
    print(f"  r bins: {r_bins} [{r_range[0]:.4f}, {r_range[1]:.4f}] m")
    print(
        f"  phi bins: {phi_bins} [{phi_range[0]:.3f}, {phi_range[1]:.3f}] rad"
    )
    print(f"Normalization check: sum along axis 0 in [{sums.min():.6f}, {sums.max():.6f}]")
    print(f"Error model: {error_model}")
    print(f"Base sigma: {sigma_base} eV")
    print(f"Saved to: {output_path}")

    return (
        energy_error_map,
        ke_error_edges,
        ke_start_edges,
        theta_center_edges,
        r_edges,
        phi_edges,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a sample energy error map file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gaussian_r_dependent",
        choices=[
            "gaussian_uniform",
            "gaussian_r_dependent",
            "gaussian_ke_dependent",
            "gaussian_combined",
        ],
        help="Error model to use",
    )
    parser.add_argument(
        "--ke-error-min",
        type=float,
        default=-20.0,
        help="Minimum energy error (eV)",
    )
    parser.add_argument(
        "--ke-error-max",
        type=float,
        default=20.0,
        help="Maximum energy error (eV)",
    )
    parser.add_argument(
        "--ke-error-bins",
        type=int,
        default=400,
        help="Number of energy error bins",
    )
    parser.add_argument(
        "--ke-start-min",
        type=float,
        default=18000,
        help="Minimum kinetic energy (eV)",
    )
    parser.add_argument(
        "--ke-start-max",
        type=float,
        default=19000,
        help="Maximum kinetic energy (eV)",
    )
    parser.add_argument(
        "--ke-start-bins",
        type=int,
        default=20,
        help="Number of kinetic energy bins",
    )
    parser.add_argument(
        "--theta-center-bins",
        type=int,
        default=18,
        help="Number of theta_center bins",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=0.01,
        help="Maximum radius (m)",
    )
    parser.add_argument(
        "--r-bins",
        type=int,
        default=20,
        help="Number of radius bins",
    )
    parser.add_argument(
        "--phi-bins",
        type=int,
        default=18,
        help="Number of azimuthal angle bins",
    )
    parser.add_argument(
        "--sigma-base",
        type=float,
        default=5.0,
        help="Base resolution parameter (eV) for Gaussian models",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_energy_error_map.npy",
        help="Output file path",
    )

    args = parser.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    generate_energy_error_map(
        error_model=args.model,
        ke_error_range=(args.ke_error_min, args.ke_error_max),
        ke_error_bins=args.ke_error_bins,
        ke_start_range=(args.ke_start_min, args.ke_start_max),
        ke_start_bins=args.ke_start_bins,
        theta_center_bins=args.theta_center_bins,
        r_range=(0, args.r_max),
        r_bins=args.r_bins,
        phi_bins=args.phi_bins,
        sigma_base=args.sigma_base,
        output_path=args.output,
    )
