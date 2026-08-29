# Replacing NeMO's far field with the widebvh treecode

**Result: 37x more accurate and 1.1-4.6x faster, at the same near field.**

NeMO's far field was `WarpFMM` (`src/treecode.py`), a Barnes-Hut treecode carrying a
monopole plus first-order dipole expansion on a patched fork of NVIDIA Warp. It is now
`WidebvhFMM` (`src/treecode_widebvh.py`), a thin `ctypes` wrapper over the widebvh
BaryStokes treecode -- degree-7 barycentric-Lagrange interpolation at Chebyshev proxy
points, on a cuBQL LBVH.

No widebvh source is copied into this repo. The engine gained two features for this
integration and both default to their previous behaviour, so every existing widebvh
benchmark is bit-identical (verified, §4).

> **Since this report was written, the other expansion in widebvh — the analytic Cartesian
> Taylor policy — was wired up and measured head-to-head.** It loses: 1.09x slower end to end
> at 50k rising to 1.25x at 750k, and 1.24x on the 1M two-drop, because matching this
> report's accuracy needs a 2.4x tighter `mac` and the near-pair set that follows costs more
> than its (genuinely cheaper) M2P saves. Everything below stands. Two findings from that
> work do bear on this one: the engine's automatic bucket cell edge does **not** scale with N
> (~1024 cells at any size), which leaves even BaryStokes ~5% off its optimum at 1M; and
> `WidebvhFMM` was leaking `TC_HILBERT_Q` between instances in one process. See
> `artifacts/cartesian_far_field_report.md`.

---

## 1. Choosing the operating point

The premise: NeMO's learned near field carries **~7.5% PRMSE** at 10% volume fraction, so
far-field accuracy far below that is wasted work. widebvh's published operating points sit
at 1e-8. The question is how much of that we can spend.

Measured against a dense **float64 RPY translation-translation sum over r >= 6**
(`benchmarks/mac_calibration.py`, whose reference agrees with the independently-written
dense reference in `benchmarks/symmetry_treecode.py` to **5.9e-16**), at 2048 evenly
spaced sample targets, on four particle distributions x two loadings:

| distribution | N | |
|---|---|---|
| 3k sedimenting drop | 3,071 | R=40, phi=0.048 |
| uniform suspension | 100,000 | phi=0.1, the Figure-11/12 regime |
| two-drop sedimentation t=0 | 1,047,968 | the H-HIGNN benchmark, initial state |
| two-drop sedimentation t=100 | 1,047,968 | same, after the drops elongate (most clustered) |

Loadings are uniform gravity and random unit forces. **Random is the binding case**
everywhere -- gravity is benign because the far field is dominated by a coherent
long-range sum. All numbers below are worst-case over all eight (distribution, loading)
combinations.

`rel_total` is the metric that matters: the far-field error as a fraction of the total
reported velocity, i.e. what actually propagates into PRMSE.

### widebvh, maxLeaf 1024

| PDEG | mac | worst rel_total | far-field ms (1M, t=100) |
|---|---|---|---|
| 7 | 0.60 | 2.9e-05 | 193.7 |
| 7 | 0.70 | 1.1e-04 | 141.3 |
| 7 | 0.75 | 2.1e-04 | 122.2 |
| **7** | **0.80** | **3.7e-04** | **107.6** |
| 7 | 0.85 | 6.7e-04 | 95.1 |
| 7 | 0.90 | 1.2e-03 | 86.0 |
| 5 | 0.70 | 9.1e-04 | 109.3 |
| 3 | 0.40 | 7.7e-04 | 309.6 |

### WarpFMM, the incumbent

| theta | worst rel_total | far-field ms (1M, t=100) |
|---|---|---|
| 0.30 (production) | 1.4e-02 | 492.7 |
| 0.28 | 1.1e-02 | 581.4 |
| 0.20 | 5.2e-03 | 1291.7 |

### The pick: **PDEG 7, mac 0.80, maxLeaf 1024**

Chosen as the cheapest configuration whose worst-case `rel_total` stays under **1e-3** --
a factor ~200 below the near field's own 7.5%, so it contributes nothing measurable in
quadrature -- with margin to spare rather than sitting on the line. Against the theta=0.3
it replaces that is **37x more accurate and 4.6x faster** on the 1M two-drop.

maxLeaf is flat from 512 to 2048 at this point (within 5%); 1024 is widebvh's own tuned
value for uniform-like clouds. mac 0.85 is available if the extra 12% matters, at 6.7e-4.

### Lower PDEG is a false economy here — worth reporting

The obvious idea, given how loose NeMO's target is, is to drop the polynomial degree:
PDEG 3 uses 64 proxy points per M2P instead of 512. It loses, and not marginally. At
matched accuracy:

| target rel_total | best config | far-field ms (1M, t=100) |
|---|---|---|
| ~7e-04 | PDEG 7, mac 0.85 | **95.1** |
| ~7e-04 | PDEG 3, mac 0.40 | 309.6 |
| ~1e-04 | PDEG 7, mac 0.70 | **141.3** |
| ~1e-04 | PDEG 5, mac 0.50 | 222.0 |

A lower degree needs a much tighter `mac` for the same error, and the tighter `mac`
explodes the direct near-pair count -- **62.4M pairs at PDEG 3/mac 0.4 versus 10.5M at
PDEG 7/mac 0.8** on the 1M t=100 cloud. The P2P blow-up costs far more than the cheaper
M2P saves. PDEG 7 dominates the Pareto front at every accuracy level tested, which is a
non-obvious result: the usual expectation is that a low-accuracy regime wants a low-order
expansion.

### Where widebvh does not win

At **N = 3,071** the far field is fixed-cost dominated (~5.7 ms regardless of `mac`, versus
WarpFMM's 3.7 ms), because the grid-hilbert bucketizer's auto cell edge is
`q = cbrt(max(1024, N/maxLeaf))` and the `max(1024, .)` floor gives the 3k drop only ~6
particles per bucket. The accuracy is still 4-16x better. If small-N throughput ever
matters, `TC_HILBERT_Q` is exposed on the constructor for exactly this.

Raw data: `data/widebvh_mac_calibration.csv`.

---

## 2. End-to-end: Figure 12

One application of the full NeMO grand mobility (analytic self + two-body NN + n-body NN
inside r=6, treecode beyond), uniform suspension at 10% volume fraction, H200, 6 warmup +
6 timed applications with the published trim. `torch.compile` enabled.

| N | published | WarpFMM re-measured | **widebvh** | total speedup | far field: warp -> widebvh | M updates/s |
|---|---|---|---|---|---|---|
| 50k | 21.30 | 22.03 | **22.50** | 0.98x | 10.75 -> 11.14 ms (0.97x) | 2.22 |
| 100k | 36.37 | 38.25 | **34.91** | 1.10x | 18.54 -> 14.75 ms (1.26x) | 2.86 |
| 200k | 81.75 | 85.14 | **60.45** | 1.41x | 47.36 -> 22.50 ms (2.10x) | 3.31 |
| 500k | 166.42 | 176.34 | **139.02** | 1.27x | 83.34 -> 46.37 ms (1.80x) | 3.60 |
| 750k | 266.96 | 283.36 | **206.27** | 1.37x | 145.00 -> 67.89 ms (2.14x) | 3.64 |

The WarpFMM column re-measures the incumbent on this node with the same protocol; it lands
within a few percent of the published figure, so the comparison is like for like. Both
columns were taken **after** the n-body chunking fix in §5, which is what makes them
comparable at 500k and 750k -- the earlier version of this table had both backends carrying
the bug.

Throughput rises from the paper's 2.4-2.7M particle updates/sec to **2.2-3.6M**, i.e. one
mobility application on a 3.6M-particle suspension in about a second. The far field itself
is up to 2.1x faster; the end-to-end gain is smaller because the near field (unchanged) is
the other half of the step -- Amdahl, not a measurement artifact. At 50k the far field is
fixed-cost dominated and the swap is a wash, as §1 predicted.

Raw data: `data/fig12_scaling_h200.csv`. Figure: `figures/grand_M_scaling_test_h200.pdf`
(rendered by `figures/grand_M_perf.py`, which now reads the CSV instead of hardcoded
lists). `figures/grand_M_far_field_ab.pdf` is a companion showing both backends with the
near/far split.

## 2b. The 1M two-drop sedimentation

`benchmarks/two_suspensions_1M.py`, 1,047,968 particles, 50 explicit-Euler steps at
dt = 0.01 after 5 warmup steps, H200:

| | per step | far field | far-field share |
|---|---|---|---|
| paper (WarpFMM theta=0.3) | 480 ms | 283 ms | 61% |
| WarpFMM re-measured | 461 ms | 284 ms | 62% |
| **widebvh mac 0.8** | **280 ms** | **104 ms** | **37%** |

**1.65x faster per step, 2.7x faster far field.** The 50-step run drops from 23.1 s to
14.0 s. The re-measured WarpFMM row again lands on the published number, so the comparison
is sound. Both rows are post-fix (§5). Raw data:
`data/widebvh_perf_nemo_distros.csv`.

---

## 3. What changed in widebvh

All in `/mnt/ffs24/home/khanmd/programs/widebvh`; nothing copied here.

### 3a. `Config::nearCutoff` — the near/far partition

NeMO's neighbour list owns every pair with `r < 6`, so the far field must cover exactly the
complement. widebvh had no such notion. Two rules, both required:

* **Node acceptance** (`nodeOutsideNearRadius`, `treecode.cuh`): a node is accepted only if
  its bounding sphere lies entirely outside `rc`, `r - hd >= rc`, written sqrt-free.
  `NodeMAC` carries only `{cx, cy, cz, halfDiag2}`, so this is the bounding *sphere*, which
  is conservative relative to an AABB test -- it descends slightly more often near the
  cutoff, never less.
* **Pair evaluation** (`stokes::p2p`, both overloads): skip `r^2 < rc^2`.

The value lives in `__constant__` memory rather than in kernel arguments. That is
deliberate: the MAC predicate appears at four sites and is consumed by both a count pass
and a pair-write pass which must make **bit-identical** decisions or the emitted pair list
desyncs from its counts. One symbol they all read cannot drift.

**Rejected alternative** — "sum all pairs with widebvh, then subtract the `r<6` pairs using
NeMO's existing neighbour list." It needs no widebvh changes, and it is wrong at any useful
`mac`: a near pair can be swallowed by an accepted multipole, and subtracting the exact near
kernel then leaves the multipole's error sitting on an `O(1/r)` term while the answer being
computed is `O(1/60)`. It only holds as `mac -> 0`, the opposite of where we operate.

### 3b. `WIDEBVH_RPY_A` — the RPY kernel

NeMO's far field is the Rotne-Prager-Yamakawa mobility of unit spheres, not the bare
Stokeslet. In the factored form widebvh already uses, with `c = 2a^2/3`, that is 4 extra
fp64 ops: `u_i += A f_i + (B q) R_i`, `A = 1/r + c/r^3`, `B = 1/r - 3c/r^3`.

The catch, found by reading rather than assuming: **`BaryStokes::m2p` and `m2pWarp`
hand-inline the Stokeslet** instead of calling `stokes::p2p`. Changing `p2p` alone would
have left the far field on the plain Stokeslet with no visible symptom -- the difference is
only ~4e-6 of the far field on these clouds. The physics now shares
`stokes::accumStokesFactored` across four of the five sites, with `m2p` (expanded `ir3`
form) and the fp64 direct-sum reference carrying matching branches.

Compile-time, not runtime: `WIDEBVH_RPY_A=0` is the default and reproduces the previous
code expression for expression.

### 3c. `WIDEBVH_PDEG`, `TC_QUIET`, `libwidebvh_nemo*.so`

Degree is now a build option so the PDEG 3/5/7 Pareto above could be measured at all.
`TC_QUIET=1` suppresses the per-apply `[treecode]` banners, which a per-timestep caller
cannot tolerate. `src/nemo_capi.cu` is the C ABI.

---

## 4. Verification

| check | result |
|---|---|
| ABI: `ctypes` load + 2M-particle apply inside the torch process, including under a 133 GB torch allocation | pass |
| widebvh unchanged with `nearCutoff=0`, `RPY_A=0`: `relL2err` on `two_ball_t50` at mac 0.5/0.66 x {split-warpspec, split-warpspec-atomic, direct-warpspec} | **bit-identical** (2.854166e-08 / 5.029489e-07) |
| **Partition test**: `nearCutoff=6` with `mac=0.02` (no node can be accepted) vs the fp64 direct `r>=6` sum | **1.12e-12** — no double counting, no gap at rc |
| RPY M2P vs the fp64 RPY direct sum, same test | **1.12e-12** |
| `WidebvhFMM.get_far_field_vel` vs the fp64 Python reference at `mac=0.05` | 1.1e-08 (the fp32-position floor) |
| fp64 reference vs `symmetry_treecode.exact_far_tt` | 5.9e-16 |
| full grand mobility, WarpFMM vs WidebvhFMM at N=50k | differs by 1.33e-02 translational, 4.4e-08 rotational — i.e. exactly WarpFMM's own truncation error, and the rotational block (zero in both) is untouched |

Reproduce:

```bash
source ~/warp_env.sh
export TORCH_COMPILE_DISABLE=1                       # accuracy runs only
python benchmarks/mac_calibration.py --selftest
python benchmarks/mac_calibration.py --case all --pdegs 3,5,7 \
       --loadings gravity,random --csv data/widebvh_mac_calibration.csv

python benchmarks/symmetry_treecode.py probe --K 24   # section 5 asymmetry table
python benchmarks/symmetry_treecode.py theta --n-far 800   # mac sweep + the mac=0.02 control

unset TORCH_COMPILE_DISABLE                          # performance runs
python benchmarks/figure12_grand_M.py --backend widebvh
python benchmarks/figure12_grand_M.py --backend warp
python benchmarks/two_suspensions_1M.py
python figures/grand_M_perf.py
```

`NEMO_FAR_FIELD=warp` selects the old backend in any of these; `NEMO_MAC` overrides the
operating point. The `theta` and `thetascale` modes previously ignored their sweep argument
under widebvh and emitted a column of identical rows -- they now sweep whichever knob the
selected backend actually uses.

widebvh side (from its own checkout, `source ./env.sh`):

```bash
TC_NEAR_CUTOFF=6.0 TC_PATH=split-warpspec-atomic \
  ./build-nemo/two_ball 0.02 1024 10 distros/two_ball_t50.bin 2000   # partition test
TC_NEAR_CUTOFF=6.0 ./build-nemo/two_ball_rpy 0.02 1024 10 distros/two_ball_t50.bin 2000
```

---

## 5. Effect on grand-mobility symmetry

`artifacts/treecode_symmetry_report.md` established that the treecode is the sole source of
asymmetry in NeMO's grand mobility, and that the asymmetry **is** the truncation error --
they move one-for-one. Shrinking the truncation error should therefore shrink the asymmetry
by the same factor. Re-running that diagnostic against the new far field
(`NEMO_FAR_FIELD=widebvh python benchmarks/symmetry_treecode.py grand|probe`):

| | WarpFMM theta=0.3 | **widebvh mac 0.8** | |
|---|---|---|---|
| N=150, dense assembly | 1.19e-03 | **1.03e-06** | 1160x |
| N=10,000, Hutchinson | 1.10e-02 | **3.23e-04** | 34x |
| N=100,000, Hutchinson | 1.25e-02 | **4.12e-04** | 30x |
| N=300,000, Hutchinson | -- | **4.00e-04** | |
| N=500,000, Hutchinson | -- | **4.46e-04** | |
| N=750,000, Hutchinson | -- | **4.30e-04** | |
| N=1,000,000, Hutchinson | 1.53e-02 | **4.61e-04** | 33x |

Flat in N, as it should be: the acceptance criterion is scale-free, so the truncation
error it produces is too. The dense-RPY control is unchanged at 3.85e-09, and the 33
negative eigenvalues (min -9.56e-03) are identical with and without the tree -- they come
from the NN near field and are untouched by any of this, exactly as the original report
concluded.

The relationship established in `treecode_symmetry_report.md` §2 holds for this backend
too: at N=200, mac 0.8 gives rel_asym 6.67e-06 against a truncation error of 4.72e-06, a
ratio of **1.414 = sqrt(2)** -- i.e. the error is uncorrelated with its own transpose and
none of it is a symmetric common mode. Shrink the truncation and the asymmetry follows.

### The 1M row was a near-field bug, not the far field

An earlier version of this table reported **3.08e-03** at N=1M and flagged it as
unexplained, since it disagreed with the directly measured far-field truncation by ~13x.
It was not the far field at all. `src/gpu_nbody_mob.py` chunks the n-body pair loop at
`pair_chunk_size = 8_000_000` and rebuilt the per-particle neighbour table **per chunk**,
from that chunk's targets only. The table is indexed by *both* endpoints of every pair, and
edges are grouped by target, so once a second chunk existed every pair whose source lay
outside the current chunk's target range silently lost half its neighbour context -- and
lost it directionally, `(t,s)` keeping t's neighbours while `(s,t)` kept s's.

Four independent measurements, all in `data/symmetry_widebvh.csv`:

| evidence | result |
|---|---|
| Hold N=100k fixed, force 1 / 2 / 3 / 6 chunks | 4.12e-04 / 3.66e-03 / 4.21e-03 / 4.73e-03 |
| ... against the predicted `sqrt(1 - 1/C)` scaling | 5.15 / 5.14 / 5.20e-03 — the mechanism, quantitatively |
| N-scan across the 8M-pair boundary (~385k particles) | 3.80e-04, 4.00e-04 (1 chunk) then **1.20e-03** at 400k (2 chunks) |
| N=1M with chunking disabled (`pair_chunk_size=32M`) | 3.08e-03 -> **4.61e-04** |
| Near field alone, no treecode anywhere, N=1M | **1.22e-02** — the far field was never implicated |

It was never only a symmetry defect. The velocities themselves were wrong: at N=1M the
buggy path differs from the correct answer by **1.75e-02** in relative L2. Everything at
N >~ 400k was affected, which is Figure 12's 500k and 750k points and the 1M two-drop.

The fix hoists the per-particle table out of the chunk loop (`_per_particle_topk`), so it is
built once over the complete edge list. Verified:

| check | result |
|---|---|
| N=100k, 1 chunk, pre-fix vs post-fix | 9.10e-08 — identical to the 9.02e-08 run-to-run `index_add_` nondeterminism floor |
| same at N=150 / 800 / 10k (the accuracy-benchmark sizes) | 1.27e-07 / 1.43e-07 / 9.14e-08, all at their own floors |
| N=1M, post-fix, 3 chunks vs 1 chunk (**chunk-invariance**) | 2.68e-07 — the property the bug destroyed |
| N=1M, post-fix 3 chunks vs pre-fix 1 chunk | 2.68e-07 — the fix reproduces the correct answer |
| near field alone at N=1M | 1.22e-02 -> **8.85e-06** |

Since the fix is exact below the chunking threshold, no accuracy result at the sizes those
benchmarks use can move; the equality checks above are the proof, which is cheaper and
stronger than regenerating MFS ground truth.

**A torch.compile trap found while fixing it, worth knowing about.** The obvious fix --
build the table at the top of `get_nbody_velocity` -- is 4-13% *faster* on the
fixed-configuration benchmarks and a disaster in a dynamics run. It puts a graph break
ahead of the chunk loop, after which dynamo guards the loop bound on the *exact* pair
count; that count drifts every timestep, so the run burns through `config.recompile_limit`
(8) and falls back to eager permanently. The 1M two-drop went from 283 to **681 ms/step**
with no error, only a `[__recompiles]` line buried in the log. Marking the dimension dynamic
instead makes `range()` unrepresentable and raises a hard error. The shipped structure keeps
the loop in Python and compiles one chunk at a time, with the pair dimension marked dynamic:
one graph serves every chunk size, and nothing silently degrades.

### Effect on performance

| | pre-fix | post-fix |
|---|---|---|
| 1M two-drop, per step | 283 ms | **280 ms** |
| Figure 12, 50k / 100k / 200k / 500k / 750k (ms) | 21.86 / 33.61 / 60.46 / 142.06 / 208.28 | 22.50 / 34.91 / 60.45 / 139.02 / 206.27 |
| peak VRAM at 750k | 8.21 GB | **7.93 GB** |

Within a few percent either way, with slightly lower peak VRAM. The correctness is the
point; it is not paid for in throughput. Figure 12's headline numbers in §2 are the
post-fix ones.

### What it costs to be more symmetric

With the near field clean, asymmetry is once again purely the far-field truncation, so the
only lever is `mac` -- and it buys accuracy and symmetry together, one for one. Measured
(`data/symmetry_widebvh.csv`, K=16 probes):

| mac | rel_asym @100k | rel_asym @1M | far-field @1M | est. production step @1M |
|---|---|---|---|---|
| 0.90 | 1.09e-03 | 1.40e-03 | 74.8 ms | ~0.91x |
| **0.80** (current) | 3.93e-04 | 3.92e-04 | 91.1 ms | 1.00x |
| 0.70 | 1.01e-04 | 8.71e-05 | 118.6 ms | ~1.10x |
| 0.60 | 2.24e-05 | 2.17e-05 | 162.8 ms | ~1.26x |

**mac 0.8 -> 0.6 buys 18x on asymmetry and ~13x on far-field truncation for about 1.26x
step time.** The old lever was worth 3.4x for 1.36x (`treecode_symmetry_report.md` §5), so
this one is roughly an order of magnitude better per unit of wall clock.

The step-time column is scaled from the production far-field share (far field is ~37% of
the 1M two-drop step), *not* read off the probe runs directly: those run with
`TORCH_COMPILE_DISABLE=1`, where the near field is ~3x slower and the far field's share is
correspondingly understated.

**Recommendation: stay at mac 0.8.** 4e-04 asymmetry sits ~200x below the near field's own
~7.5% PRMSE and contributes nothing in quadrature, and the far field is no longer the
accuracy-limiting term in this operator by a wide margin. mac 0.7 is the one to reach for
if a future application wants more headroom -- 4.5x tighter for ~10% step time. Buying
symmetry beyond that is not useful work: it would not make the operator SPD, because the
33 negative eigenvalues come from the NN near field and are present with the tree, without
the tree, and at every mac.

## 6. Notes and limits

* The far field remains **translation-only** in both backends. The RT/TR/RR blocks are still
  exactly zero beyond r=6, so the finding in `artifacts/treecode_symmetry_report.md` §6 --
  that the discarded far-field angular velocity exceeds the entire near-field angular
  velocity -- is unchanged by this work. The `fmm_vs_baseline_rel` gap of 5.8% against a
  dense 6x6 RPY far field is likewise unchanged. It is the larger physics gap and remains
  open.
* `WarpFMM` is still in the tree and still importable; `NEMO_FAR_FIELD=warp` selects it in
  `benchmarks/performance_grand_M.py`, `benchmarks/figure12_grand_M.py`,
  `benchmarks/two_suspensions_1M.py` and `benchmarks/symmetry_treecode.py`.
* **Unrelated bug found and fixed along the way.** `PYTHONPATH` contains an older
  `/mnt/home/khanmd/pinn-stokes` checkout. Scripts run as `python benchmarks/foo.py` get the
  script's own directory as `sys.path[0]`, not the cwd, so a `sys.path.append(repo_root)`
  lands *after* that stale checkout and `import src.*` silently resolves to the wrong tree.
  `two_suspensions_1M.py`, `symmetry_treecode.py` and `large_scale_dynamics.py` all did
  this; they now use `sys.path.insert(0, ...)`. This was masking the new module and would
  have silently mixed old and new `src` code in any run of those three.
* The `.so` is loaded with `RTLD_LOCAL` and built with hidden visibility. This is
  load-bearing, not hygiene: the degree variants export identical symbol names, and with
  default visibility whichever loads first interposes on the others -- the first calibration
  run reported PDEG 3, 5 and 7 as producing *identical* errors, which is what exposed it.
