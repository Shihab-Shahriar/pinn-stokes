# Stokesian Dynamics operator diagnostics (benchmarks/sd_diagnostics.py; SD checkout 6b9117d)

## Two spheres: SD vs exact (velocity of sphere 1 in units of F / 6 pi mu a)

| r | forcing | MFS | SD | SD far field | SD - MFS | far field - MFS |
|---|---|---|---|---|---|---|
| 2.1 | same F along line | 1.536333 | 1.536334 | 1.543479 | +6.9e-07 | +7.1e-03 |
| 2.1 | same F perpendicular | 1.391752 | 1.391738 | 1.401995 | -1.4e-05 | +1.0e-02 |
| 2.1 | opposite F along line | 0.135385 | 0.134893 | 0.289855 | -4.9e-04 | +1.5e-01 |
| 2.1 | torque on sphere 1 | 0.878985 | 0.878824 | 0.956035 | -1.6e-04 | +7.7e-02 |
| 2.5 | same F along line | 1.486071 | 1.486071 | 1.491611 | +4.8e-08 | +5.5e-03 |
| 2.5 | same F perpendicular | 1.326380 | 1.326380 | 1.329725 | +3.5e-08 | +3.3e-03 |
| 2.5 | opposite F along line | 0.360696 | 0.360696 | 0.397814 | -7.7e-08 | +3.7e-02 |
| 2.5 | torque on sphere 1 | 0.974187 | 0.974187 | 0.984546 | +2.7e-07 | +1.0e-02 |
| 3.0 | same F along line | 1.432040 | 1.432040 | 1.435407 | +8.1e-08 | +3.4e-03 |
| 3.0 | same F perpendicular | 1.266802 | 1.266802 | 1.268000 | +3.2e-08 | +1.2e-03 |
| 3.0 | opposite F along line | 0.490520 | 0.490520 | 0.500820 | +6.7e-08 | +1.0e-02 |
| 3.0 | torque on sphere 1 | 0.992894 | 0.992894 | 0.994838 | +2.1e-08 | +1.9e-03 |
| 3.9 | same F along line | 1.354501 | 1.354501 | 1.355639 | +7.8e-08 | +1.1e-03 |
| 3.9 | same F perpendicular | 1.200407 | 1.200407 | 1.200675 | +3.2e-08 | +2.7e-04 |
| 3.9 | opposite F along line | 0.616378 | 0.616378 | 0.618272 | +7.1e-08 | +1.9e-03 |
| 3.9 | torque on sphere 1 | 0.998724 | 0.998724 | 0.998933 | +9.8e-09 | +2.1e-04 |
| 4.4 | same F along line | 1.320533 | 1.321162 | 1.321162 | +6.3e-04 | +6.3e-04 |
| 4.4 | same F perpendicular | 1.176167 | 1.176301 | 1.176301 | +1.3e-04 | +1.3e-04 |
| 4.4 | opposite F along line | 0.661057 | 0.661954 | 0.661954 | +9.0e-04 | +9.0e-04 |
| 4.4 | torque on sphere 1 | 0.999406 | 0.999483 | 0.999483 | +7.7e-05 | +7.7e-05 |

## Compact clusters: relative error vs MFS (%), SD vs its far field alone

| cluster | forcing | SD PRMSE | SD lin | SD ang | far-field PRMSE | far-field lin | far-field ang |
|---|---|---|---|---|---|---|---|
| triangle, side 2.2 | random wrench | 1.17 | 1.17 | 1.18 | 6.57 | 5.66 | 8.04 |
| triangle, side 2.2 | gravity | 0.86 | 0.86 | 0.60 | 1.16 | 0.53 | 8.48 |
| triangle, side 2.5 | random wrench | 0.50 | 0.51 | 0.45 | 2.26 | 2.24 | 2.35 |
| triangle, side 2.5 | gravity | 0.44 | 0.44 | 0.51 | 0.28 | 0.20 | 1.65 |
| triangle, side 3.0 | random wrench | 0.22 | 0.25 | 0.14 | 0.94 | 1.15 | 0.31 |
| triangle, side 3.0 | gravity | 0.15 | 0.15 | 0.16 | 0.08 | 0.08 | 0.18 |
| cube of 8, spacing 2.3 | random wrench | 2.96 | 3.33 | 1.68 | 6.53 | 6.20 | 7.32 |
| cube of 8, spacing 2.3 | gravity | 3.39 | 3.38 | 5.11 | 0.64 | 0.56 | 4.17 |
| cube of 8, spacing 3.0 | random wrench | 0.73 | 0.81 | 0.27 | 0.66 | 0.70 | 0.47 |
| cube of 8, spacing 3.0 | gravity | 0.82 | 0.81 | 1.25 | 0.11 | 0.11 | 0.12 |

## Lubrication cutoff sweep on `uniform_N200_phi0.1_seed4423_grav.npz` (N = 200, phi = 0.1, gravity)

cutoff_factor = r* / (a1 + a2): pairs closer than r* receive R2B,exact - R2B,inf; 1.05 admits none of this configuration's pairs (min gap 0.1), 2 is SD's default.

| cutoff_factor | pairs corrected (per sphere) | translational PRMSE | fluctuation PRMSE | rotational | mean settling speed / U_stokes (MFS) |
|---|---|---|---|---|---|
| 1.05 | 0.00 | 0.26 | 1.77 | 1.79 | 20.01 (19.98) |
| 1.25 | 0.92 | 5.02 | 6.97 | 6.26 | 18.98 (19.98) |
| 1.5 | 2.09 | 8.20 | 11.56 | 9.99 | 18.35 (19.98) |
| 2.0 | 5.78 | 11.79 | 16.18 | 14.27 | 17.64 (19.98) |
| 2.25 | 8.21 | 12.73 | 17.51 | 15.41 | 17.45 (19.98) |

## numba kernels vs SD's pure-Python path (60-sphere subset of the phi = 0.1 configuration)

max |difference| 3.55e-15, relative L2 4.91e-16 (process wall incl. imports: numba 5 s, pure Python 8 s)

