# HIGNN baseline vs NeMO on the Fig 3 / Fig 4 accuracy protocols (gravity forcing, torque-free)

*2026-09-07. Code: `src/hignn_ops.py` (adapter), `benchmarks/paper_accuracy_v2.py` (ops `HIGNN_2b`, `HIGNN_full`, exp `fig4g`),
`figures/fig_hignn_compare.py` (figures + `artifacts/hignn_comparison_tables.md`), `benchmarks/hignn_cpp_check.py` +
`slurm/hignn_cpp_check.sbatch` (their C++ engine, cross-check). HIGNN checkout: github.com/Pan-Group-UW-Madison/hignn @ 37830e9
(2025-12-18), referenced through `HIGNN_ROOT` (AGPL-3, nothing copied into this repo).*

## 1. Bottom line

On the torque-free gravity version of the Fig 3 / Fig 4 protocol, **NeMO (moments + learned diagonal) is 1.5–2.3× more
accurate than the full HIGNN model and 1.9–2.6× more accurate than HIGNN's production two-body engine at N = 200**
(translational PRMSE 0.66 / 1.08 / 1.73 / 2.43 % vs 1.01 / 1.93 / 3.74 / 5.47 % vs 1.23 / 2.20 / 4.22 / 6.24 % at
φ = 0.025 / 0.05 / 0.1 / 0.15; on the velocity fluctuations the ratio is 1.3–1.6×). Two further facts frame that number:

- **HIGNN's two-body engine is slightly *less* accurate than plain RPY** at every φ and every N ≤ 300 (e.g. 6.24 vs 5.97 % at
  φ = 0.15, N = 200), and **the full HIGNN model (their 3-body + self corrections added) lands on top of NeMO's
  two-body-only operator** (1.01 vs 0.95, 1.93 vs 1.82, 3.74 vs 3.61, 5.47 vs 5.41 %) — behind the paper's published
  n-body operator (b1) and behind the 3-body summation variant at every volume fraction.
- **The advantage is a near-field advantage and it shrinks with N under gravity.** Beyond N ≈ 3000 every operator converges
  to the same error (≈ 3.3 % at φ = 0.1, ≈ 16.5 % on the fluctuations) because what remains is the collective, far-field
  (back-flow) part of the settling velocity, which every pairwise far field — NeMO's RPY beyond 8 radii and HIGNN's
  RPY-like kernel alike — gets wrong in the same way. At N = 10 000 the NeMO / HIGNN ratio is 1.03–1.07. Their 3-body and
  self terms make no difference at all there (HIGNN full = HIGNN 2-body to 0.02 points).

Their kernel was evaluated exactly (dense, no H-matrix truncation) with their own cutoff (5.0, verified optimal), so
these are HIGNN's best-case numbers; angular velocities, which NeMO predicts to 6.6–19.5 % under gravity at N = 200,
have no HIGNN counterpart at all.

## 2. What HIGNN actually computes (verified in their code)

HIGNN (Ma, Ye & Pan, CMAME 400 (2022) 115496) and its hierarchical-matrix successor H-HIGNN (2025) are the "ML mobility"
baseline the paper compares against on speed (Fig 13). Reading the toolkit they ship establishes what the model *is*:

- **Translational only, torque-free.** The operator maps forces (N,3) to velocities (N,3). There are no torques and no
  angular velocities anywhere in the C++ or Python code (`TimeIntegrator.hpp` has the angular write-back commented out;
  `UpdateCoord` documents `nDim = 6` as "not implemented yet"). The training data are Stokesian-Dynamics 2- and 3-sphere
  solves with the torques set to zero (`python/data generation/input_setups.py`), i.e. freely rotating spheres, so the
  learned 3x3 blocks are the torque-free-condensed translational mobility.
- **Their C++/Kokkos H-matrix engine (`HignnModel.dot`) is a pure pairwise two-body sum.** `u_i = F_i + sum_{j != i}
  M2(x_j - x_i) F_j` over all pairs, with `M2(x) = (I + MLP(x)) / |x|` and the MLP = `nn/two_body_unbounded.pkl`
  (3->128->512->9, tanh; `python/convert.py::Net`). No RPY, no analytic far field, no cutoff: the H-matrix (ACA, default
  tolerance 0.05) only compresses the far blocks of that same kernel. `LoadThreeBodyModel` is an empty stub.
- **The 3-body and self corrections exist only in their Python path** (`python/HIGNN/model_structure.py::HIGNN_mdoel`,
  driven by `python/gravity_field.py`): chain triples (source ~ mediator ~ target, target != source) within centre
  distance `eps3 = 5.0`, `M3 = net3([x_m - x_s | x_t - x_m]) / (|x_m - x_s| |x_t - x_m|)`, and a per-particle
  correction `Mself = netself(x_n - x_t) / |x_n - x_t|^2` summed over the same neighbours. Weights
  `python/Saved_Model/Unbounded_try1/HIGNN_nn_{2body,3body,self}.pkl` (md5 68139a4c / bc424e04 / 6cdcdfc1); the 2-body
  one is bit-identical to `nn/two_body_unbounded.pkl`, so "engine 2-body + Python 3-body/self" is one consistent model
  (the other pickles, `nn/three_body.pkl` and `Saved_Model/Train1`, are an incompatible family: 101 % error when mixed).
- **Units:** radius a = 1, viscosity mu = 1, self mobility = identity, i.e. velocities in units of F/(6 pi mu a); we divide
  by 6 pi. Valid separation range 2 <= r <= ~600 a (their file name `UB_max600`); the kernel loses positive-definiteness
  below r = 2, which never occurs here (surface gap >= 0.1 a).

Two variants are therefore evaluated: **HIGNN 2-body** (what their production engine computes, evaluated densely, i.e.
without the additional ACA error) and **HIGNN full** (2-body + 3-body + self, the model of the CMAME paper).

**3-body cutoff.** `eps3 = 5.0` is an inference-time choice from their scripts, not stored with the weights. Sweep on
single N = 200 gravity configurations (translational PRMSE %, seed in parentheses):

| φ (seed) | HIGNN 2-body | full, eps3 = 3 | eps3 = 4 | **eps3 = 5** | eps3 = 6 | eps3 = 8 |
|---|---|---|---|---|---|---|
| 0.025 (1423) | 1.345 | 1.213 | 1.159 | **1.125** | 1.131 | 1.528 |
| 0.05 (2423) | 2.106 | 1.940 | 1.877 | **1.826** | 1.860 | 2.757 |
| 0.10 (4423) | 4.287 | 4.051 | 3.948 | **3.816** | 3.909 | 6.567 |
| 0.15 (6423) | 5.996 | 5.712 | 5.533 | **5.203** | 5.317 | 9.811 |

Their 5.0 is the optimum everywhere (a wider cutoff feeds the 3-body net triples outside its training support), so it is
what the `HIGNN_full` rows use; the comparison does not under-configure their model.

## 3. Protocol

- **Why gravity / torque-free.** The paper's Fig 3/4 truths carry random unit forces *and torques* and score all six
  velocity components. HIGNN can neither take a torque nor return an angular velocity, so those truths cannot be reused
  fairly. The repository already holds an MFS truth set on the *same configurations and seeds as Fig 4* with uniform
  gravity F = (0, 0, -9.81), T = 0 (`tmp/nbody_moments_truth/*_grav.npz`, widebvh Broms MFS, Xfine clouds, no far-field
  truncation): 756 configurations, N = 20 ... 10 000 at φ ∈ {0.025, 0.05, 0.1, 0.15} (10 seeds for N <= 500, 5 at
  1000/1500, 3 at 2000 ... 10 000) plus N = 20 000 / 30 000 at φ = 0.1 (2 seeds). No new truths were generated (user
  decision), so the "Fig 3-style" view is the N = 200 and N = 300 slices (4 volume fractions, φ = 0.2 unavailable) and
  the "Fig 4-style" view is the full N grid. Gravity is also exactly HIGNN's intended use case (sedimentation).
- **Configurations** are the paper's: RSA of unit spheres in a cube sized for φ, minimum surface gap 0.1, particle 0 at
  the origin, `default_rng(seed)`; identical positions for every operator (read from the truth files).
- **Metrics (translational only, since HIGNN has no angular output):**
  - `prmse_lin` = 100 ||U_pred − U_true||_F / ||U_true||_F (the paper's PRMSE restricted to the translational block);
  - `prmse_fluct` = the same on the fluctuations U − mean(U): under uniform gravity the collective settling velocity
    dominates ||U|| and is easy for every operator (it is a far-field quantity), so this is the closer analogue of the
    random-forcing PRMSE and isolates the near-field / disorder part;
  - `err_mean_pct` = error of the mean (collective) velocity; `max_rel_lin` = worst particle.
  - Every cell is the mean over the seeds (population std for the bands), as in the paper's harness.
- **Operators.** RPY (analytic self + Rotne–Prager–Yamakawa, all pairs), NeMO 2-body, NeMO 3-body summations (our own
  triplet-sum variant, architecturally the closest to HIGNN's 3-body term), NeMO n-body (b1, the published Fig 3/4
  operator), NeMO (moments pc8c pair model + learned diagonal, the current stack; GPU op over the whole N range, CPU op
  at N <= 300, parity 1e-3 points), HIGNN 2-body, HIGNN full. All NeMO/RPY operators receive T = 0 and are scored on
  their translational output only.
- **How HIGNN was evaluated.** `src/hignn_ops.py` re-implements their forward pass in plain torch from their pickled
  MLPs: all-pairs dense 2-body sum (float32 kernel, float64 accumulation, as their engine), neighbour lists and chain
  triples exactly as `NeighborLists::BuildThreeBodyInfo`, `index_add_` in place of `torch_scatter`. Dense evaluation is
  HIGNN's best case (their production `dot` adds the ACA error on top). Validation: `tests/test_hignn_ops.py` (isolated
  sphere, two-sphere block values `M_zz = 0.47403`, `M_xx = 0.26801` at r = 3, symmetry/parity/rotation of the learned
  block, brute-force edge enumeration, transcription of `HIGNN_mdoel.forward` to 1e-5, gravity spot values) and the
  C++-engine cross-check of §5.

## 4. Results

All numbers: mean over the seeds (10 for N ≤ 500, 5 at N = 1000/1500, 3 above), ± population std where shown; full
tables in `artifacts/hignn_comparison_tables.md`, per-row data in `data/paper_accuracy_v2.csv` (exp `fig4g`).

### 4.1 Fig-3 style: error vs volume fraction at fixed N (translational PRMSE, %)

N = 200:

| operator | φ = 0.025 | 0.05 | 0.1 | 0.15 |
|---|---|---|---|---|
| **NeMO (moments + learned diagonal)** | **0.66 ± 0.05** | **1.08 ± 0.07** | **1.73 ± 0.06** | **2.43 ± 0.13** |
| NeMO 3-body summations | 0.86 ± 0.06 | 1.57 ± 0.07 | 2.78 ± 0.09 | 3.81 ± 0.13 |
| NeMO n-body (paper, b1) | 0.85 ± 0.06 | 1.55 ± 0.08 | 3.15 ± 0.14 | 4.92 ± 0.16 |
| NeMO 2-body | 0.95 ± 0.07 | 1.82 ± 0.08 | 3.61 ± 0.15 | 5.41 ± 0.14 |
| HIGNN full (2-body + 3-body + self) | 1.01 ± 0.07 | 1.93 ± 0.08 | 3.74 ± 0.14 | 5.47 ± 0.15 |
| RPY | 1.13 ± 0.07 | 2.07 ± 0.09 | 4.02 ± 0.16 | 5.97 ± 0.15 |
| HIGNN 2-body (their engine's kernel) | 1.23 ± 0.08 | 2.20 ± 0.09 | 4.22 ± 0.16 | 6.24 ± 0.15 |

N = 300:

| operator | φ = 0.025 | 0.05 | 0.1 | 0.15 |
|---|---|---|---|---|
| **NeMO (moments + learned diagonal)** | **0.70 ± 0.04** | **1.23 ± 0.05** | **2.05 ± 0.10** | **2.91 ± 0.07** |
| NeMO 3-body summations | 0.87 ± 0.05 | 1.63 ± 0.06 | 2.94 ± 0.11 | 4.10 ± 0.09 |
| NeMO n-body (paper, b1) | 0.86 ± 0.05 | 1.62 ± 0.06 | 3.20 ± 0.13 | 4.99 ± 0.10 |
| NeMO 2-body | 0.94 ± 0.06 | 1.83 ± 0.06 | 3.56 ± 0.13 | 5.38 ± 0.10 |
| HIGNN full | 0.98 ± 0.05 | 1.91 ± 0.06 | 3.69 ± 0.13 | 5.47 ± 0.10 |
| RPY | 1.07 ± 0.06 | 2.02 ± 0.07 | 3.87 ± 0.14 | 5.81 ± 0.10 |
| HIGNN 2-body | 1.13 ± 0.06 | 2.11 ± 0.07 | 4.03 ± 0.14 | 6.03 ± 0.10 |

Ratios HIGNN / NeMO (translational PRMSE): full 1.54 / 1.78 / 2.17 / 2.25 at N = 200 and 1.40 / 1.55 / 1.80 / 1.88 at
N = 300; 2-body engine 1.86 / 2.03 / 2.45 / 2.57 at N = 200 (φ = 0.025 / 0.05 / 0.1 / 0.15).

Figure: `figures/fig_hignn_compare_phi.{pdf,png}` (and `_fluct`).

### 4.2 Fluctuation PRMSE (collective settling velocity removed), N = 200

| operator | φ = 0.025 | 0.05 | 0.1 | 0.15 |
|---|---|---|---|---|
| **NeMO (moments + learned diagonal)** | **3.93 ± 0.22** | **7.06 ± 0.40** | **13.1 ± 0.6** | **20.2 ± 1.3** |
| NeMO 3-body summations | 4.55 | 8.77 | 17.6 | 26.9 |
| NeMO n-body (paper, b1) | 4.52 | 8.79 | 19.3 | 31.8 |
| NeMO 2-body | 4.87 | 9.57 | 20.4 | 33.0 |
| HIGNN full | 5.03 ± 0.28 | 9.83 ± 0.44 | 20.7 ± 0.6 | 33.0 ± 1.3 |
| RPY | 5.36 | 10.1 | 21.2 | 34.2 |
| HIGNN 2-body | 5.80 ± 0.33 | 10.5 ± 0.4 | 21.7 ± 0.6 | 34.8 ± 1.2 |

Ratios HIGNN full / NeMO: 1.28 / 1.39 / 1.58 / 1.63 (N = 200), 1.19 / 1.26 / 1.38 / 1.43 (N = 300). The ordering of the
seven operators is the same as for the total error, i.e. the comparison is not an artefact of the collective velocity.

### 4.3 Fig-4 style: error vs N

Translational PRMSE (%), seed means (full grid in the tables file; figure `figures/fig_hignn_compare_N.{pdf,png}`):

| φ | operator | N = 20 | 50 | 100 | 200 | 300 | 1000 | 3000 | 10 000 | 30 000 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.025 | NeMO | 0.43 | 0.50 | 0.57 | 0.66 | 0.70 | 0.78 | 0.79 | 0.79 | – |
| | HIGNN full | 1.21 | 1.10 | 1.07 | 1.01 | 0.98 | 0.91 | 0.86 | 0.84 | – |
| | HIGNN 2-body | 2.24 | 1.63 | 1.38 | 1.23 | 1.13 | 0.97 | 0.89 | 0.85 | – |
| 0.05 | NeMO | 0.68 | 0.75 | 0.78 | 1.08 | 1.23 | 1.45 | 1.55 | 1.60 | – |
| | HIGNN full | 1.72 | 1.94 | 1.91 | 1.93 | 1.91 | 1.78 | 1.72 | 1.69 | – |
| | HIGNN 2-body | 3.26 | 2.70 | 2.34 | 2.20 | 2.11 | 1.87 | 1.76 | 1.71 | – |
| 0.1 | NeMO | 0.66 | 1.02 | 1.23 | 1.73 | 2.05 | 2.80 | 3.09 | 3.21 | 3.27 |
| | HIGNN full | 1.92 | 3.48 | 3.61 | 3.74 | 3.69 | 3.66 | 3.51 | 3.42 | 3.38 |
| | HIGNN 2-body | 5.33 | 4.92 | 4.47 | 4.22 | 4.03 | 3.79 | 3.57 | 3.44 | 3.40 |
| 0.15 | NeMO | 0.87 | 1.22 | 1.48 | 2.43 | 2.91 | 4.08 | 4.60 | 4.90 | – |
| | HIGNN full | 2.14 | 4.15 | 4.98 | 5.47 | 5.47 | 5.44 | 5.29 | 5.23 | – |
| | HIGNN 2-body | 7.37 | 6.69 | 6.39 | 6.24 | 6.03 | 5.65 | 5.38 | 5.27 | – |

Ratio HIGNN full / NeMO averaged over the shared N grid: 1.66 / 1.87 / 2.12 / 2.34 (φ = 0.025 / 0.05 / 0.1 / 0.15);
at N = 10 000 it is 1.06 / 1.06 / 1.07 / 1.07.

The two families move in opposite directions with N. NeMO's error grows with N (0.66 → 0.79 %, 1.73 → 3.21 %) while
HIGNN's shrinks (1.23 → 0.85 %, 4.22 → 3.44 %), and both, together with RPY, meet at the same value. The decomposition
explains why:

| φ | error of the mean (collective) velocity, % | N = 20 | 200 | 1000 | 10 000 |
|---|---|---|---|---|---|
| 0.1 | NeMO | 0.31 | 0.64 | 1.84 | 2.39 |
| | HIGNN full | 1.14 | 2.77 | 2.77 | 2.64 |
| | HIGNN 2-body | 4.12 | 3.30 | 2.93 | 2.67 |
| 0.15 | NeMO | 0.36 | 0.84 | 2.66 | 3.64 |
| | HIGNN full | 0.87 | 4.03 | 4.12 | 4.03 |
| | HIGNN 2-body | 5.85 | 4.88 | 4.37 | 4.08 |

The mean settling velocity of a large cloud is a far-field, many-body quantity (back-flow / hydrodynamic screening).
Every operator here uses a pairwise-additive far field — NeMO switches to RPY beyond 8 radii, HIGNN's kernel agrees
with RPY to ~0.3 % beyond r ≈ 4 — so all of them carry the same bias in the collective velocity once the cloud is large,
and that bias (2.4–2.6 % at φ = 0.1, 3.6–4.1 % at φ = 0.15) is most of the total error at N ≥ 3000. The fluctuation error
converges the same way (≈ 16.5 % at φ = 0.1 for every operator at N ≥ 10 000): at that size the fluctuations are the
large-scale convective motion of the drop, again far-field-dominated. NeMO's near-field corrections (pair moments within
8 radii, learned diagonal) are what buy the 1.5–2.6× at N ≤ 300 and cannot move this far-field term; HIGNN's 3-body and
self terms (cutoff 5) are worth ~0.5 points at N ≤ 300 and nothing at large N.

### 4.4 Worst particle and angular velocities

Max per-particle translational relative error at N = 200 (mean over seeds): NeMO 1.5 / 2.0 / 3.1 / 4.2 %, HIGNN full
2.1 / 3.5 / 6.7 / 9.7 %, HIGNN 2-body 3.1 / 4.3 / 7.5 / 10.7 %, RPY 2.6 / 3.9 / 7.1 / 10.2 %.

Angular velocities under gravity (pure disorder-induced rotation), relative L2 at N = 200: NeMO 6.6 / 9.4 / 14.4 / 19.5 %,
NeMO 2-body 7.8 / 12.1 / 21.6 / 32.4 %, RPY 8.5 / 12.7 / 22.2 / 33.1 %. HIGNN predicts none.

### 4.5 Cost note (not a speed benchmark)

On the RTX 4060, the dense HIGNN evaluation takes 0.01 / 0.27 / 0.8 / 11 / 65 s per apply at N = 200 / 1000 / 3000 /
10 000 / 30 000 (O(N²) pairs; their H-matrix engine exists to avoid exactly this), the NeMO GPU operator with its chunked
dense RPY far field 0.04 / 0.16 / 1.1 / 10 / 105 s. Neither number is the production far field (widebvh treecode for NeMO,
H-matrix for HIGNN); the paper's Fig 13 comparison stands for speed.

## 5. Cross-check against their C++ engine

Their engine was built from the same checkout inside the authors' CPU image (`panlabuwmadison/hignn:cpu` via
Singularity on the MSU HPCC, `python3 python/init.py --rebuild`, the CI recipe; `slurm/hignn_cpp_check.sbatch`; the
container needs `--cleanenv`, otherwise its OpenMPI tries SLURM's PMIx and aborts in `hignn.Init()`), and
`benchmarks/hignn_cpp_check.py --run` fed it our gravity configurations (float32 positions and forces, their
`simulate.py` block size and large-N pool settings). `dense_dot` is the exact O(N²) sum of their TorchScript kernel;
`dot` is the production H-matrix path (ACA far field) at the three (epsilon, max_iter) settings their scripts use.
Relative L2 against the pure-torch 2-body operator of `src/hignn_ops.py` (HIGNN units) and translational PRMSE against
the MFS truth (`data/hignn_cpp_check.csv`):

| configuration | engine path | rel-L2 vs `src/hignn_ops.py` | PRMSE vs truth (%) | wall (16 CPU threads) |
|---|---|---|---|---|
| N = 200, φ = 0.1, seed 4423 | pure torch (this repo) | – | 4.287 | 0.2 s (GPU) |
| | `dense_dot` | 2.8e-8 | 4.287 | 1.9 s |
| | `dot`, ε = 0.05 / 0.1 / 0.01 | 1.7e-4 (all three) | 4.289 | 12 s |
| N = 200, φ = 0.1, seed 4424 | `dense_dot` / `dot` | 2.8e-8 / 1.3e-4 | 4.077 / 4.081 | 1.9 s / 12 s |
| N = 200, φ = 0.025, seed 1423 | `dense_dot` / `dot` | 3.3e-8 / 1.6e-4 | 1.346 / 1.344 | 1.9 s / 12 s |
| N = 1000, φ = 0.05, seed 2723 | `dense_dot` | 2.6e-8 | 1.890 | 51 s |
| | `dot`, ε = 0.05 / 0.1 / 0.01 | 4.0e-4 / 5.7e-4 / 1.9e-4 | 1.887 / 1.890 / 1.887 | 39 s |
| N = 3000, φ = 0.1, seed 5123 | `dense_dot` | 3.0e-8 | 3.508 | 494 s |
| | `dot`, ε = 0.05 / 0.1 / 0.01 | 1.5e-3 / 2.8e-3 / 4.0e-4 | 3.530 / 3.525 / 3.507 | 180–194 s |
| N = 10 000, φ = 0.1, seed 5423 | pure torch (this repo) | – | 3.439 | 6.7 s (GPU) |
| | `dot`, ε = 0.05 (default) | 1.6e-3 | 3.437 | 849 s |

- `dense_dot` reproduces the pure-torch evaluation to float32 round-off (≤ 3.3e-8): the `HIGNN_2b` rows in the CSV are
  what their engine computes.
- Their production `dot` deviates from the dense sum by 1.3e-4 (N = 200, where the two-leaf tree has no far blocks and
  the whole difference is the `use_symmetry` reuse of M_ijᵀ for the (j, i) pair) to 1.5–2.8e-3 at N = 3000 with their
  default/gravity-run tolerances (ε = 0.05 / 0.1). The effect on accuracy against the truth is ≤ 0.02 PRMSE points, i.e.
  the dense numbers reported here are their engine's numbers, marginally flattering it.
- At N = 10 000 their engine **segfaulted inside `HignnModel::FarDot`** (OpenMP, ACA far field) on the first attempt with
  the default pool sizes; with their own large-N settings (`set_mat_pool_size_factor(200)`,
  `set_max_far_dot_work_node_size(10000)`, `set_max_relative_coord(1000000)`, as in `simulate.py`) it runs, and at their
  default tolerance it again agrees with the dense pure-torch evaluation to 1.6e-3 (PRMSE 3.437 vs 3.439 %). Their engine
  on 16 CPU cores is not a speed reference (849 s per apply at N = 10 000; their production runs are multi-GPU).

## 6. Caveats

- HIGNN is scored only on what it predicts: translational velocities under torque-free forcing. NeMO's angular
  velocities (`prmse_ang`, 6–20 % relative under gravity, where rotation is a pure disorder effect) have no HIGNN
  counterpart. `rel_rmse`, `rmse`, `mae`, `max_rel_rmse` in the CSV include the angular block and must not be read for the
  HIGNN rows (angular predicted as 0 → `prmse_ang` = 100 %).
- HIGNN's kernel is evaluated densely and exactly; their H-matrix engine is *less* accurate than the numbers here by its
  ACA tolerance (quantified in §5).
- The 3-body/self terms are their Python path, not their accelerated engine; the H-HIGNN large-scale runs use the
  2-body engine plus that Python term (`gravity_field.py`), which is exactly the `HIGNN_full` composition.
- The learned 2-body block is symmetric but only approximately parity-even (|M(x) − M(−x)| up to 3e-3 at r ≥ 2.1) and
  rotation-equivariant to ~1 %; their engine's `use_symmetry` flag reuses M_ij^T for the (j, i) pair, which is part of
  the engine-vs-dense difference in §5.
- Gravity forcing makes every relative error 2.5–3x smaller than the paper's random-forcing PRMSE (the collective
  velocity dominates the normalisation); the fluctuation metric restores the near-field sensitivity, and ratios between
  operators, not absolute levels, are the transferable message.
- φ = 0.2 and the intermediate volume fractions are not in the gravity truth set; a torque-free variant of the exact
  Fig 3 protocol (random unit forces, T = 0, N = 200/300, 8 φ) needs ~1 GPU-hour of new MFS truths and would slot in
  as another forcing of the same harness.

## 7. Reproduction

```sh
# once: HIGNN_ROOT points at the Pan-group checkout (default /home/shihab/throwaway/hignn)
python -m pytest tests/test_hignn_ops.py -q
python src/hignn_ops.py --case 200 0.1 4423 --variant full --eps3-sweep 3 4 5 6 8     # cutoff table above
# rows (all local; the GPU is used by the HIGNN ops if present)
TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig4g --phis 0.025 0.05 0.1 0.15 \
    --ops HIGNN_2b HIGNN_full --gpu-ops --part --skip-done
TORCH_COMPILE_DISABLE=1 python benchmarks/paper_accuracy_v2.py --exp fig4g --phis 0.025 0.05 0.1 0.15 \
    --N 20 30 40 50 60 70 80 90 100 120 140 160 180 200 300 \
    --ops M_rpy M_2b M_3b M_nbody_b1 M_mom_v2_kinf_rc8_pc8c_diag --workers 6 --part --skip-done
TORCH_COMPILE_DISABLE=1 bash docker/run_local.sh python benchmarks/paper_accuracy_v2.py --exp fig4g \
    --phis 0.025 0.05 0.1 0.15 --ops M_mom_gpu_pc8c_diag --gpu-ops --part           # headline op, new metric columns
python benchmarks/paper_accuracy_v2.py --merge --summary
python figures/fig_hignn_compare.py                                                 # figures + tables
# their engine (cluster, Singularity, CPU image): sbatch slurm/hignn_cpp_check.sbatch, then
python benchmarks/hignn_cpp_check.py --compare --out tmp/hignn_cpp_check.npz
```
