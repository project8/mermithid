"""
This script generates sample cavity field maps for testing the DataGenerator4D
class. It creates two types of magnetic field traps: an ideal box trap and an
ideal harmonic trap. The generated field maps are saved as .npz files, which
can be used in the DataGenerator4D tests.

Usage examples:
  python generate_sample_field_map.py

Author: S. M. Lee
Date: January 19, 2026
"""

import numpy as np

# An ideal box magnetic field trap
# config and coordinates
B_min = 0.950  # (T)
B_wall = 1.0  # (T)
r_domain = 7e-2  # (m)
z_domain = 0.06  # (m)
z_wall = 0.05  # (m)
z_wall_width = 0.002  # (m)

r_edges = np.linspace(0, r_domain, 701)
z_edges = np.linspace(-z_domain, z_domain, 1201)
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

r_mesh, z_mesh = np.meshgrid(r_centers, z_centers)

# create field map
B_map = B_min * np.ones_like(r_mesh)  # (T)
B_map[z_mesh < -z_wall] = B_wall
B_map[z_mesh > z_wall] = B_wall
B_map[z_mesh < -z_wall - z_wall_width] = B_min
B_map[z_mesh > z_wall + z_wall_width] = B_min

# save it to npz file
np.savez("box.npz", r_edges=r_edges, z_edges=z_edges, B_map=B_map)
print("Saved box.npz")

# effective volume: 2 \pi R^{2} z_{w} \sqrt{1 - \dfrac{B_{\min}}{B_{w}}}
# total efficiency: \sqrt{1 - \dfrac{B_{\min}}{B_{w}}}$
total_efficiency = np.sqrt(1 - B_min / B_wall)
print(f"Total expected efficiency for this box trap should be: {total_efficiency:.4f}")

# An ideal harmonic magnetic field trap
# $B(z)=B_{\min} (1+(z/a)^2)$ for $z \in (-z_w, z_w)$, otherwise $B_{\min}$
# where $a=z_{w} \sqrt{\dfrac{B_{\min}}{B_{w} - B_{\min}}}$.

# config and coordinates
B_min = 0.950  # (T)
B_wall = 1.0  # (T)
r_domain = 7e-2  # (m)
z_domain = 0.06  # (m)
z_wall = 0.05  # (m)

r_edges = np.linspace(0, r_domain, 701)
z_edges = np.linspace(-z_domain, z_domain, 1201)
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

r_mesh, z_mesh = np.meshgrid(r_centers, z_centers)

# create field map
a = z_wall * np.sqrt(B_min / (B_wall - B_min))
B_map = B_min * (1 + (z_mesh / a) ** 2)  # (T)
B_map[z_mesh < -z_wall] = B_min
B_map[z_mesh > z_wall] = B_min

# save it to npz file
np.savez("harmonic.npz", r_edges=r_edges, z_edges=z_edges, B_map=B_map)
print("Saved harmonic.npz")

# effective volume: \dfrac{\pi^2 R^2}{2} z_{w} \sqrt{1 - \dfrac{B_{\min}}{B_{w}}}
# total efficiency: \dfrac{\pi}{4} \sqrt{1 - \dfrac{B_{\min}}{B_{w}}}$
total_efficiency = np.pi / 4 * np.sqrt(1 - B_min / B_wall)
print(f"Total expected efficiency for harmonic trap should be: {total_efficiency:.4f}")
