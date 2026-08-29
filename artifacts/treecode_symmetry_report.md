# Is the NeMO grand mobility still symmetric with the treecode far field?

> **Superseded in two places — read this first.**
>
> 1. **The far field is now widebvh, not `WarpFMM`.** Everything below measures `WarpFMM`.
>    The current numbers are in `artifacts/widebvh_far_field_report.md` §5: the production
>    asymmetry is **4e-04**, not 1.5e-02, and it is flat in N.
> 2. **Every measurement here above ~400k particles is contaminated.** `src/gpu_nbody_mob.py`
>    rebuilt the n-body neighbour table per pair-chunk, so once the pair count passed
>    `pair_chunk_size = 8_000_000` each pair lost its source's neighbour context,
>    directionally. That inflated the N = 1,000,000 row of §4 and it is *not* the treecode
>    growth this report attributes it to. Fixed in `_per_particle_topk`; see
>    `widebvh_far_field_report.md` §5 for the four measurements that pin it down. The
>    N = 150 / 10,000 / 100,000 rows are single-chunk and stand.
>
> §1, §2, §3 and §6 are unaffected and remain the reference for *why* a target-centric
> treecode is asymmetric at all, and for the translation-only far field, which is still open.

**Short answer: no — but the near field is, and the asymmetry is not a bug on top of the
treecode approximation, it *is* the treecode approximation.**

At production `theta = 0.3` the grand mobility is **1.5 % asymmetric** in relative Frobenius
norm at N = 1 M. Turning the treecode off (dense analytic RPY far field) drops that to
**4 × 10⁻⁹** — float32 round-off. So every bit of the asymmetry comes from the tree.

A second, larger finding fell out of the same measurement: the treecode far field is
**translation-only**, and the rotation–translation coupling it discards carries **10 % of the
far-field operator's Frobenius norm** — about 6× the symmetry defect, and systematic rather
than noise-like.

Everything below is reproduced by `benchmarks/symmetry_treecode.py`.

---

## 1. The near field is symmetric — premise confirmed

`Mob_Nbody_Torch` with `far_field_2b=None` (self + two-body NN + n-body NN, nothing else),
assembled column-by-column into a 900 × 900 matrix:

| quantity | value |
| --- | --- |
| ‖M − Mᵀ‖_F / ‖M‖_F | 4.85 × 10⁻⁹ |
| max\|M − Mᵀ\| | 1.40 × 10⁻⁹ |

Symmetric to float32 precision. Worth noting because `src/model_archs.py` warns that the
two-body kernel `M_t` is deliberately *not* symmetric — that turns out to be exactly the
condition that makes the assembled grand matrix symmetric (`RT = c(r)·L3(d̂)` with
`L3(−d̂) = −L3(d̂) = L3(d̂)ᵀ`), so the directed-edge scatter in `PairVelKernel` is fine.

## 2. The far field is not — and the cause is provable

Far-field block assembled in isolation via `get_far_field_vel`, N = 800, compared against a
dense float64 RPY reference over all pairs with r ≥ 6:

| θ | rel. asymmetry | truncation error | ratio |
| --- | --- | --- | --- |
| 0.000 | 0 (exact) | 8.16 × 10⁻⁸ | — |
| 0.100 | 0 (exact) | 8.16 × 10⁻⁸ | — |
| 0.125 | 0 (exact) | 8.16 × 10⁻⁸ | — |
| 0.150 | 2.42 × 10⁻⁵ | 1.71 × 10⁻⁵ | 1.414 |
| 0.175 | 2.52 × 10⁻⁴ | 1.81 × 10⁻⁴ | 1.392 |
| 0.200 | 9.77 × 10⁻⁴ | 7.06 × 10⁻⁴ | 1.383 |
| 0.250 | 4.17 × 10⁻³ | 3.09 × 10⁻³ | 1.352 |
| **0.300** | **9.03 × 10⁻³** | **6.76 × 10⁻³** | **1.337** |
| 0.400 | 2.12 × 10⁻² | 1.59 × 10⁻² | 1.333 |
| 0.500 | 3.70 × 10⁻² | 2.79 × 10⁻² | 1.329 |
| 0.700 | 7.38 × 10⁻² | 5.54 × 10⁻² | 1.333 |

Three things this establishes:

**The multipole acceptance is the cause.** Below θ ≈ 0.15 no node passes the MAC, traversal
degenerates to all-direct pairs, and the asymmetry is *identically zero* — not small, zero.
`rpy_far_velocity_pair3x3` builds a tensor that is symmetric and even in **r**, and with a unit
force on a single particle there is no summation to reorder, so (i,j) and (j,i) come out
bitwise equal. Asymmetry switches on exactly when the tree starts lumping sources.

**The near/far partition is clean.** At θ = 0 the assembled block matches the dense float64
reference to 8 × 10⁻⁸. The `closest_dist_sq < 36.0` guard in
`warp/native/bvh.h:640` and the `dot(rvec,rvec) < near_cutoff2` skip in `treecode.py:178`
really do produce the exact `r ≥ 6` partition, with no double counting and no gaps.

**The asymmetry is the truncation error, not an extra defect.** The ratio sits at √2 = 1.414 at
onset and drifts only to ~1.33. If the treecode error `E = M − M_exact` were uncorrelated with
its own transpose you would get exactly √2; a value at or just below √2 means essentially none
of `E` is a symmetric common-mode. In other words there is no separate "asymmetry problem" to
fix — shrink the truncation error and the asymmetry shrinks with it, one-for-one.

## 3. Full grand mobility, treecode vs. no treecode

N = 150, θ = 0.3, 900 × 900 assembled both ways:

| | rel. asymmetry | max\|M − Mᵀ\| | neg. eigenvalues | min eigenvalue |
| --- | --- | --- | --- | --- |
| treecode far field | 1.19 × 10⁻³ | 2.42 × 10⁻⁴ | 33 | −9.565 × 10⁻³ |
| dense analytic RPY far field | 3.77 × 10⁻⁹ | 1.16 × 10⁻⁹ | 33 | −9.428 × 10⁻³ |

**The treecode is the sole source of asymmetry** — five orders of magnitude between the two rows.

**The treecode is *not* the source of the negative eigenvalues.** Both operators have exactly 33,
with near-identical minima. The grand mobility is already non-SPD without any tree; that comes
from the NN near field and is a pre-existing property, not something this analysis introduces.
(The `far_field_2b=None` operator in §1 shows 110 negative eigenvalues, but that is an artifact
of hard-truncating the far field to zero, which cannot preserve positive-definiteness — it is
not an operator anyone runs.)

## 4. At production scale

Hutchinson probes — `E[(uᵀMv − vᵀMu)²] = ‖M − Mᵀ‖_F²` for Rademacher u, v — so no assembly is
needed. Validated first against the dense N = 150 result (probe 1.35 × 10⁻³ ± 2.1 × 10⁻⁴ vs.
dense 1.19 × 10⁻³, agreement within one standard error). θ = 0.3, K = 24 probes:

| N | rel. asymmetry |
| --- | --- |
| 150 | 1.19 × 10⁻³ (dense) |
| 10 000 | 1.10 × 10⁻² ± 1.7 × 10⁻³ |
| 100 000 | 1.25 × 10⁻² ± 1.8 × 10⁻³ |
| 1 000 000 | 1.53 × 10⁻² ± 2.5 × 10⁻³ |

It grows with N and then saturates around 1.5 % — deeper trees mean more of the far field is
served by lumped nodes, but that fraction plateaus.

## 5. What it costs to fix by lowering theta

N = 100 000, full `WarpFMM.apply` step timed on the H200:

| θ | rel. asymmetry | far field | full step |
| --- | --- | --- | --- |
| 0.05 | 4.66 × 10⁻⁵ | 236.3 ms | 309.2 ms |
| 0.10 | 1.11 × 10⁻³ | 203.6 ms | 274.7 ms |
| **0.20** | **4.10 × 10⁻³** | **49.8 ms** | **120.9 ms** |
| 0.30 (current) | 1.39 × 10⁻² | 18.7 ms | 88.8 ms |
| 0.40 | 2.66 × 10⁻² | 8.9 ms | 79.2 ms |
| 0.50 | 4.49 × 10⁻² | 5.2 ms | 75.3 ms |

The near field is ~70 ms of every step regardless of θ, which makes the far field cheap to buy
down. **θ = 0.3 → 0.2 cuts asymmetry 3.4× for 1.36× the step time** — and cuts the far-field
truncation error by the same 3.4×, since §2 showed they move together.

## 6. The bigger finding: the far field is translation-only

`WarpFMM.apply` slices `forces[:, :3]` (`treecode.py:417`) and writes back into
`total_vel[:, :3]` (`:431`). Torques never enter the far field and the RT/TR/RR blocks are
exactly zero beyond r = 6. That is symmetric — a zero block is its own transpose — but it is a
real physics truncation, and it is larger than the symmetry defect.

Dense full-6×6 far-field operator (N = 150, validated against
`benchmarks/bench_rpy.py:two_body_rpy_batch` to 5 × 10⁻⁸):

| block | Frobenius norm | kept? |
| --- | --- | --- |
| TT (translation ← force) | 1.4149 | yes |
| RT (translation ← torque) | 0.1002 | **no** |
| TR (rotation ← force) | 0.1002 | **no** |
| RR (rotation ← torque) | 0.0115 | **no** |

**10.0 % of the far-field operator's Frobenius norm is discarded.** That accounts almost exactly
for the 5.8 % gap between the treecode grand matrix and the dense-RPY grand matrix measured in
§3 (0.10 × 1.4149 / 2.449 = 5.8 %).

The damage is concentrated in angular velocity, and specifically in the **TR block — rotation
driven by neighbours' forces**, not by their torques. For a sedimentation-style loading (uniform
gravity + random torques):

| config | ‖Ω_far‖ / ‖Ω_near‖ | ‖Ω_far‖ from forces | ‖Ω_far‖ from torques | U error from dropped coupling |
| --- | --- | --- | --- | --- |
| N = 800, φ ≈ 0.1 | 1.17 | 2.012 | 0.029 | 3.25 × 10⁻³ |
| N = 10 000, φ ≈ 0.1 | 1.74 | 11.33 | 0.099 | 8.01 × 10⁻⁴ |

The discarded far-field angular velocity is **larger than the entire near-field angular
velocity**, and 99 % of it comes from the force→rotation coupling. Translational velocity is
barely affected (~10⁻³). So: linear dynamics are fine; **particle rotation rates in a
treecode run are not trustworthy.**

---

## Fix options

**1. Lower theta — cheapest real lever, recommended now.** θ = 0.2 buys 3.4× on both asymmetry
and far-field accuracy for 36 % more wall time (§5). Nothing to implement.

**2. Symmetrized apply, `u = ½(Mf + Mᵀf)`.** `Mᵀf` needs no assembly: re-run the *same*
traversal and scatter with atomics — for each source `j` in an accepted node of target `i`, add
`M_eff[i,j]ᵀ f_i` into `v_j`. Identical traversal ⇒ identical pair set ⇒ symmetric by
construction, exactly, at any θ. Cost ≈ 2× the far field plus atomic contention; at θ = 0.3 that
is +19 ms on an 89 ms step, cheaper than dropping to θ = 0.2. This is the only option that gives
*exact* symmetry without touching the vendored Warp BVH. Note it symmetrizes but does not make
the result more accurate — the truncation error stays.

**3. Dual-tree traversal with a symmetric node–node MAC + M2L.** The principled fix: makes the
partition reciprocal and, with matched truncation order on both sides, close to self-adjoint.
Large rewrite of `warp/native/bvh.{h,cu}`. Worth noting even textbook FMM with M2L is not
*exactly* symmetric, so this buys accuracy more than it buys symmetry.

**4. Accept it.** Defensible for deterministic dynamics: 1.5 % asymmetry sits below the
treecode's own truncation error budget, and it is dwarfed by the TT-only truncation in §6. It is
**not** defensible for Brownian / fluctuating hydrodynamics — there you need a symmetric PSD M
to form `M^{1/2}` (Cholesky or Lanczos) and to satisfy fluctuation–dissipation, and this operator
is neither symmetric nor PSD.

**Recommendation.** If the far field's rotational physics matters at all, §6 is the thing to fix
first — it is 6× larger than the symmetry defect, systematic, and entirely fixable by carrying
torques through the multipole (the dipole moment already in `bvh.cu` is the right object to
extend). For symmetry specifically, option 2 is exact and affordable; option 1 is free today.

---

## Incidental findings (flagged, not fixed)

- **`near_field_cutoff` is a live footgun.** It is a `WarpFMM` constructor argument
  (`treecode.py:211`) but the matching gate is hardcoded as `36.0` in `warp/native/bvh.h:640`.
  Any value ≠ 6.0 silently double-counts pairs (< 6) or drops them (> 6). Recommend
  `assert near_field_cutoff == 6.0`.
- **Viscosity is ignored inside `WarpFMM.apply`.** The far-field kernels are called with
  `a=1.0, mu=1.0` hardcoded (`treecode.py:172,181`) and `_gpu_near_field_pass` is invoked with
  `viscosity=1.0` (`:426`), so the `vis_arr` argument does nothing. Silent wrong answer for
  µ ≠ 1.
- **`src/grpy.py` has two bugs.** `tr_block = -muRT_flat.T` (`:124`) makes the flat `(6N,6N)`
  output non-symmetric and disagrees with its own `blockmatrix=True` branch (`:114`); and
  `_build_cross` (`:136`) has the opposite rotlet sign from the PyGRPY reference in
  `src/grpy_tensors.py:14`. No module imports it today, which is why it has gone unnoticed.
  The production paths (`benchmarks/bench_rpy.py`, `src/grpy_tensors.py`) are both correct and
  symmetric.

## Reproducing

```bash
source ~/warp_env.sh
export TORCH_COMPILE_DISABLE=1

python benchmarks/symmetry_treecode.py near                 # §1
python benchmarks/symmetry_treecode.py theta --n-far 800    # §2
python benchmarks/symmetry_treecode.py grand --n 150        # §3
python benchmarks/symmetry_treecode.py probe --K 24         # §4
python benchmarks/symmetry_treecode.py thetascale --K 16    # §5
python benchmarks/symmetry_treecode.py ttonly               # §6
python benchmarks/symmetry_treecode.py all                  # everything -> artifacts/*.json
```

Self-checks the script asserts before reporting anything: the dense float64 reference must be
exactly symmetric; the θ = 0 far field must match it to float32; and the Hutchinson estimator
must reproduce the dense answer at N = 150. All three pass.
