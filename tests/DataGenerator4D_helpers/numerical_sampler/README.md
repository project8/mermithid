# Numerical sampler helper generators

- This folder contains helper scripts to create sample numerical PDFs used by DataGenerator4D numerical sampler tests.

## Files produced

- `generate_sample_energy_rate.py` -> output `.npz` with keys `pdf`, `ke_edges`.
- `generate_sample_spatial_distribution.py` -> output `.npz` with keys `pdf`, `theta_edges`, `r_edges`, and `phi_edges`.

## Recommended usage examples

```python
python generate_sample_energy_rate.py --out energy_two_gaussian.npz --option two_gaussian --ke-min 18500 --ke-max 18900
python generate_sample_spatial_distribution.py --out spatial_box.npz --option box --box-center-x0 -0.002 --box-center-y0 -0.001 --box-center-x1 0.002 --box-center-y1 0.003
```

- These files are intentionally lightweight and depend only on NumPy. They produce binned PDFs normalized to sum=1 so they can be scaled by the test harness or `DataGenerator4D` as needed.
