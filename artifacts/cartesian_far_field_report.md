# The Cartesian Taylor far field: an A/B against the production BaryStokes treecode

**Verdict up front.** The Cartesian Taylor expansion reaches NeMO's accuracy target, and its
M2P really is cheaper than the barycentric one — but it needs a 2.4x tighter `mac` to get
there, and the near-pair set that follows costs more than the cheaper M2P saves. At matched
accuracy, with both policies tuned at their own best bucket granularity, it is **1.09x slower
end to end at N=50k rising to 1.25x at 750k, and 1.24x on the 1M two-drop** (1.3x–1.8x on the
far field alone). Production stays BaryStokes at `mac 0.8`. The Cartesian path ships as
`NEMO_FAR_FIELD=widebvh-cart`.

Two things found on the way are worth more than the verdict:

1. widebvh's Cartesian policy had **no RPY term** — its m2p was the bare Stokeslet while the
   P2P beside it was RPY. Any NeMO library built from it would have carried a mixed kernel
   with a 6.8e-3 error floor. Section 1 adds it.
2. The engine's automatic bucket cell edge is `q = cbrt(max(1024, n/maxLeaf))` — about 1024
   cells however large `n` is. That was tuned in bary's sparse near-pair regime; in the
   Cartesian one it costs **286 ms of P2P out of a 346 ms far field at N=1M**, which is
   most of the naive 3.8x gap. Section 4.

---

## 1. The Cartesian policy needed an RPY term

`src/bary_stokes.cuh` branches on `stokes::RPY_ON` inside its m2p. `src/cartesian_stokes.cuh`
never mentioned RPY. NeMO's far field is the Rotne–Prager–Yamakawa mobility of unit spheres
(`WIDEBVH_RPY_A=1.0`), not the bare Stokeslet, so a Cartesian NeMO library built as-is would
have evaluated **RPY in the P2P and Stokeslet in the M2P** — the two halves of the same sum
disagreeing about the kernel.

Measured size of that mismatch, dense fp64 over r >= 6 at phi = 0.1:

| loading | rel. difference, Stokeslet vs RPY far field |
|---|---|
| random (incoherent) | **6.8e-03** |
| gravity (coherent) | 1.4e-04 |

The random figure is the one that matters: it is 18x the operating point being calibrated to,
and it is a floor no `mac` can lower.

### The identity

The correction is a second derivative of the same Laplace kernel the expansion is already
built on, so it costs no new machinery — only two more derivative orders. With `G` the Oseen
tensor and `c = 2a^2/3`:

```
d_i d_j (1/r) = 3 R_i R_j / r^5 - d_ij / r^3
RPY_ij        = G_ij + c (d_ij/r^3 - 3 R_i R_j / r^5) = G_ij - c d_i d_j (1/r)
```

so `d^alpha RPY_ij = d^alpha G_ij - c * T_{alpha + e_i + e_j}`, valid for r >= 2a — which the
near-field cutoff at r = 6 guarantees many times over. Moments and the upward pass are
untouched: RPY changes the contraction, not the moments.

Implementation (`src/cartesian_stokes.cuh`): `DERIV_EXTRA` becomes 2 when `RPY_ON`, so the
derivative table reaches `ORDER+2`; `m2p` gains one term under `if constexpr`. The scaling
factor is load-bearing — moments are node-scaled and the recurrence runs at `R/scale`, so the
new term needs **`invS^2`** on top of the single `1/scale` every other term telescopes to.
Non-RPY builds are unchanged, expression for expression.

### Gate

`two_ball_cart_rpy` (new CMake target: `TWO_BALL_CARTESIAN=1` **and** `WIDEBVH_RPY_A=1.0`)
validates the M2P against a direct sum that switches kernel with it, so `relL2err` is pure
multipole truncation. On `two_ball_t0` (N=1,047,968), `TC_NEAR_CUTOFF=6`:

| order | mac 0.3 | mac 0.5 | mac 0.7 | rate |
|---|---|---|---|---|
| 1 | 2.61e-03 | 7.94e-03 | 1.24e-02 | mac^1.8 |
| 2 | 1.79e-04 | 9.91e-04 | 3.07e-03 | mac^3.4 |
| 3 | 2.24e-05 | 2.25e-04 | 8.34e-04 | mac^4.3 |
| 4 | **2.74e-06** | 5.85e-05 | 3.72e-04 | mac^5.8 |

Monotone in both knobs, no plateau, and order 4 at mac 0.3 sits ~2000x below the size of the
RPY correction itself. The Stokeslet build (`two_ball_cart`) agrees to 5 significant figures
at every point, so the added term costs nothing in convergence.

**That gate is necessary but not sufficient**, and it is worth saying why: `two_ball` only
supports coherent gravity loading, where the RPY correction is 48x weaker (table above). It
cannot tell a correct RPY term from a missing one. The decisive check is a separate
line-for-line reimplementation of the contraction (node-scaled moments, the same recurrence,
the same slot coefficients) against a dense RPY sum over one node:

| mac | order | RPY term on | RPY term off |
|---|---|---|---|
| 0.3 | 2 | 1.01e-02 | 6.56e-02 |
| 0.3 | 3 | 5.67e-03 | 7.19e-02 |
| 0.3 | 4 | **1.36e-03** | 7.15e-02 |

With the term on the error falls with order. With it off it **plateaus at 7.15e-02, which is
exactly the RPY-vs-Stokeslet gap at that node (7.02e-02)** — it converges to the Stokeslet
answer instead. Sign, multi-index, derivative order and the `invS^2` are all confirmed.

---

## 2. Building it into NeMO

- `src/nemo_capi.cu`: `Treecode<MP>` with `MP` selected by `NEMO_CARTESIAN`; new
  `wbnemo_policy()` and `wbnemo_max_order()`; `wbnemo_create` gains an `order` argument
  (0 = policy default, clamped to the policy's `MAX_ORDER` — the engine's own bound is an
  `assert`, compiled out in Release, so passing bary's default order 6 to a policy that caps
  at 4 would have been a silent wrong answer). ABI 2 -> 3.
- `CMakeLists.txt`: the per-variant body is factored into `add_widebvh_nemo_target` so the
  Cartesian library provably shares the hidden-visibility and `-Bsymbolic` treatment. Not
  cosmetic: the variants export identical symbol names, and an A/B run loads two of them into
  one process — without it, whichever loads first interposes and the comparison silently comes
  out bary-vs-bary. `WidebvhFMM` asserts `wbnemo_policy()` after every `dlopen`, which is the
  check that both mechanisms held. Verified: all three libraries loaded together report
  distinct identities.
- `src/treecode_widebvh.py`: `policy` and `order` arguments, `_LIB_CACHE` keyed on both.
- `NEMO_FAR_FIELD=widebvh-cart` selects it in every benchmark, via
  `performance_grand_M.build_far_field`.

### A leak fixed while sweeping

`WidebvhFMM.__init__` set `TC_HILBERT_Q` only when `hilbert_q` was given. `os.environ`
persists for the process, so a default-constructed instance **inherited the previous
solver's cell edge**. Invisible in a single run; it silently corrupted the baseline row of
the first granularity sweep here. It is now popped explicitly ("auto" has to be the absence
of the variable — the engine rejects a non-positive value outright).

---

## 3. Calibration: matching today's error

Protocol is the one that chose `DEFAULT_MAC` — dense fp64 RPY reference over r >= 6, match on
the **worst** case, which is random loading. Target is bary p7 mac 0.8: `rel_far` 3.853e-04
on uniform100k/random (re-measured in the same session as the Cartesian rows: 3.853e-04,
14.62 ms — the published value reproduces exactly).

Order sweep on uniform100k/random, and the `mac` each order needs to hit the target:

| order | mac for 3.85e-4 | far ms | note |
|---|---|---|---|
| 2 | ~0.155 | ~66 | |
| 3 | ~0.24 | ~27 | |
| **4** | **0.33** | **18.7** | shipped |

Order 4 wins outright, and it is free: the LB traversal kernel compiles to **128 registers
with ~zero spill at orders 2, 3 and 4 alike** (`-Xptxas -v`), so lower orders only buy a
tighter `mac` for the same money. The register-pressure risk this work was scoped around
does not exist.

`max_leaf` is inert at this operating point — 128 through 2048 give bit-identical error and
pair counts, because the grid-hilbert bucketizer derives its own cell edge (see section 4).

### Accuracy across the full case set, `order 4, mac 0.33`

| case | N | loading | cart rel_far | bary rel_far | cart rel_total | bary rel_total |
|---|---|---|---|---|---|---|
| drop3k | 3,071 | random | **1.09e-03** | 4.08e-04 | 9.51e-04 | 3.57e-04 |
| drop3k | 3,071 | gravity | 1.63e-05 | 2.22e-06 | 1.61e-05 | 2.20e-06 |
| uniform100k | 100,000 | random | 3.95e-04 | 3.85e-04 | 3.81e-04 | 3.71e-04 |
| uniform100k | 100,000 | gravity | 1.50e-06 | 8.73e-07 | 1.50e-06 | 8.72e-07 |
| twoball0 | 1,047,968 | random | 2.50e-04 | 3.06e-04 | 2.48e-04 | 3.04e-04 |
| twoball100 | 1,047,968 | random | 1.89e-04 | 2.89e-04 | 1.88e-04 | 2.87e-04 |

Matched on uniform100k by construction, and slightly *better* at 1M. The exception is
**drop3k, where the Cartesian error is 2.7x worse** (9.5e-4 vs 3.6e-4 of total velocity) —
that small, dense, strongly non-uniform drop is the case the tighter `mac` fails to cover.
Matching there too would require `mac` ~0.27 and would make the cost gap worse. Both remain
far below the near field's ~7.5% PRMSE, so neither is visible end to end, but the comparison
below is at equal-on-uniform, not equal-everywhere.

---

## 4. Bucket granularity: most of the apparent cost gap

The first numbers had the Cartesian far field **3.8x slower at N=1M**. Most of that was not
the expansion — it was a tuning constant.

The engine's grid-hilbert bucketizer sets its own cell edge from
`q = cbrt(max(1024, n/maxLeaf))` cells per axis, which **floors at ~1024 cells however large
`n` is**. (This is also why `max_leaf` looked inert in section 3: below `n/1024` it never
binds.) At N=1M that is ~800 particles per bucket. Bary, whose near-pair set is small, barely
notices. The Cartesian policy at its 2.4x tighter `mac` carries ~11x the near pairs, and pays
for that granularity in P2P:

`twoball0`, N=1,047,968, random loading, `order 4, mac 0.33`:

| q (cells/axis) | buckets | near pairs | traverse ms | **P2P ms** | far ms |
|---|---|---|---|---|---|
| auto (~10) | 1,309 | 83.4M | 47.8 | **285.5** | 345.8 |
| 14 | 1,795 | 68.4M | 50.5 | 212.8 | 277.6 |
| 18 | 2,265 | 49.8M | 56.8 | 118.3 | 189.0 |
| 20 | 2,982 | 49.0M | 62.7 | 90.6 | 167.1 |
| **26** | **6,101** | **43.8M** | 75.1 | **45.2** | **135.2** |
| 36 | 15,362 | 41.2M | 101.8 | 28.3 | 146.7 |

P2P falls 6x from auto to q=26; traverse rises, and the sum has a clear minimum. Bary's own
optimum on the same case is q=16 (91.5 ms, from 95.9 on auto) — a 5% gain, versus 2.6x.

Measured optima are ~76 particles/bucket at uniform100k, ~183 at uniform750k, ~172 at the 1M
two-drop. `cart_hilbert_q()` targets 150 and falls back to the engine's auto rule wherever
auto is already at least that fine, so behaviour below ~150k is unchanged. It assumes a
box-filling cloud; the two-drop occupies ~35% of its bounding box, so `two_suspensions_1M.py`
passes the swept value directly.

Everything in section 6 uses each policy at its own best granularity. Reporting the 3.8x
would have been measuring the tuning constant, not the expansion.

Measured with `benchmarks/bucket_granularity.py`.

---

## 5. Symmetry

Asymmetry is far-field truncation — `treecode_symmetry_report.md` established that for the
Warp treecode and the widebvh report confirmed it for BaryStokes. **It holds for the
Cartesian expansion too, and exactly**: over the whole `mac` sweep on the isolated far-field
block, `rel_asym / trunc_err` = 1.408–1.413 against sqrt(2) = 1.4142.

| mac | rel_asym | trunc_err | ratio |
|---|---|---|---|
| 0.02 (control) | 1.43e-08 | 4.81e-08 | — |
| 0.15 | 3.14e-05 | 2.22e-05 | 1.412 |
| 0.25 | 2.98e-04 | 2.11e-04 | 1.413 |
| 0.30 | 6.20e-04 | 4.39e-04 | 1.413 |
| 0.40 | 1.79e-03 | 1.27e-03 | 1.409 |
| 0.80 | 1.11e-02 | 7.88e-03 | 1.413 |

The `mac=0.02` control still reproduces the dense reference to 4.8e-08, so the near/far split
is the clean `r >= 6` partition under this policy as well.

Hutchinson probes on the full grand operator (K=24; the estimator validates against the dense
assembly at N=150, ratio 0.933):

| N | cart (order 4, mac 0.33) | bary (mac 0.8) |
|---|---|---|
| 150 (dense) | **5.30e-04** | **1.03e-06** |
| 10,000 | 6.16e-04 | 3.23e-04 |
| 100,000 | 4.03e-04 | 4.12e-04 |
| 1,000,000 | 4.39e-04 | 4.61e-04 |

**On the uniform clouds at scale the two are symmetry-matched** — within 3% at 100k and 5% at
1M. The N=150 row is the same story as drop3k in section 3, sharper: on a small dense
configuration bary p7 is 500x more symmetric. Bary converges as `mac^9.1` and cart as
`mac^5.8`, so on a shallow tree with only a few half-diagonals of far-field range bary has
enormous margin at `mac 0.8` while cart at 0.33 does not. Matching on uniform does not match
everywhere, and the direction of the mismatch is always the same.

Neither is SPD: the operator carries 33 negative eigenvalues from the learned near field,
identical under both policies and present with no tree at all.

---

## 6. Head-to-head

### Far field alone, each policy at its own best bucket granularity

Random loading; `rel_far` matched at uniform100k by construction.

| case | N | bary p7 mac 0.8 | cart p4 mac 0.33 | ratio |
|---|---|---|---|---|
| uniform100k | 100,000 | 14.55 ms (auto q) | 18.61 ms (auto q) | 1.28x |
| uniform750k | 750,000 | 67.95 ms (auto q) | 119.35 ms (q=18) | 1.76x |
| twoball0 | 1,047,968 | 91.47 ms (q=16) | 135.15 ms (q=26) | 1.48x |

The mechanism, from the same rows: **the Cartesian M2P is genuinely cheaper and its near
field genuinely is not.** At twoball0 the traversal (which carries the M2P) costs 47.8 ms for
cart against 58.4 ms for bary on auto q — a real 18% win, exactly the hypothesis this study
was worth testing. But cart needs 43.8M near pairs to bary's 4.1M, and the P2P that follows
costs 45 ms against 12. The cheaper expansion does not pay for the tighter `mac` it needs.

### End to end (Figure 12, uniform phi=0.1, torch.compile enabled, H200)

| N | bary total | cart total | ratio | bary far | cart far | far ratio |
|---|---|---|---|---|---|---|
| 50,000 | 22.50 ms | 24.58 ms | 1.09x | 11.14 | 13.24 | 1.19x |
| 100,000 | 34.91 ms | 39.09 ms | 1.12x | 14.75 | 19.01 | 1.29x |
| 200,000 | 60.45 ms | 70.21 ms | 1.16x | 22.50 | 32.32 | 1.44x |
| 500,000 | 139.02 ms | 167.64 ms | 1.21x | 46.37 | 77.59 | 1.67x |
| 750,000 | 206.27 ms | 257.71 ms | 1.25x | 67.89 | 118.99 | 1.75x |

Peak VRAM is a wash (7.90 vs 7.93 GB at 750k). The Cartesian node moments are ~14x smaller
(436 B against ~6 KB), but at these bucket counts node storage is a few MB either way, so the
memory argument for the Cartesian policy does not survive contact with the numbers.

The end-to-end penalty is much smaller than the far-field penalty because the far field is a
minority of the step — at 100k the learned near field is ~70 ms of an ~85 ms step, so a 4 ms
far-field difference is ~5%. The gap widens with N because the far field's share does.

### The production path is unchanged

None of this moves a published number, and that is checked rather than assumed. The ABI bump
rebuilt all four libraries, and `WidebvhFMM`'s bary path was re-measured three independent
ways against values from before this work: `rel_far` 3.853e-04 / 14.62 ms on uniform100k
(calibration), `rel_asym` 1.026e-06 on the dense N=150 grand mobility, and 279.9 ms/step on
the 1M two-drop against the recorded 280. All three reproduce. `benchmarks/accuracy_grand_M.py`
was therefore not re-run: its PRMSE numbers are a property of the bary path, which is
bit-for-bit what it was. The Cartesian path's accuracy is characterised in section 3 against
the exact fp64 reference, which is a tighter statement than an MFS comparison anyway.

### 1M two-drop dynamics (50 steps, both re-measured in one session)

| | bary p7 mac 0.8 | cart p4 mac 0.33 (q=26) |
|---|---|---|
| total | **279.9 ms/step** (sd 4.8) | **348.0 ms/step** (sd 6.6) |
| far field | 102.3 ms | 169.3 ms |
| near field | 177.6 ms | 178.6 ms |
| peak allocated | 10165.6 MB | 10167.0 MB |
| wall clock, 50 steps | 14.0 s | 17.4 s |

1.24x per step. The near field agrees to 0.6%, which is the control this comparison needs:
the entire difference is the far field, nothing else moved. (Bary reproduces last session's
280 ms/step exactly.)

**The trajectories do not diverge.** Applying both operators to the same 1M configuration
under the same random 6-vector loading, the full grand mobility agrees to **6.2e-04** relative
(translation only 6.2e-04, worst component 8.8e-04) — the quadrature sum of two independent
~3e-04 far-field truncations, and ~120x below the near field's own ~7.5% PRMSE.

---

## 7. Verdict

**Production stays BaryStokes p7 at mac 0.8.** The Cartesian Taylor expansion is slower at
every size measured — 1.09x end to end at 50k rising to 1.25x at 750k, 1.24x on the 1M
two-drop — with no compensating win: peak VRAM is a wash, symmetry is matched on uniform
clouds and worse on dense ones, and accuracy on the drop3k case is 2.7x worse at the `mac`
that matches uniform100k.

The hypothesis worth testing was real and it half-held: **the Cartesian M2P genuinely is
cheaper** (18% less traversal time at 1M despite far more accepted work). It simply cannot
pay for the near-pair set that its 2.4x tighter `mac` drags along — the same mechanism that
sank bary degrees 3 and 5, now confirmed with a completely different expansion. That is a
property of the accuracy-per-`mac` of low-order expansions in this near/far split, not of any
one implementation, and it is the reusable conclusion here.

The Cartesian path ships as `NEMO_FAR_FIELD=widebvh-cart`, the way `NEMO_FAR_FIELD=warp` was
kept. Two pieces of it are worth keeping regardless of the verdict:

- the **RPY term in `cartesian_stokes.cuh`**, which makes the policy usable with NeMO's
  kernel at all, and which every future Cartesian experiment needs;
- the finding that the engine's **automatic bucket cell edge does not scale with N**
  (`~1024 cells` at any size). Bary is only 5% from its optimum at 1M so it never showed, but
  it is worth 2.6x in a pair-dense regime, and it will bite anything else that lands there.

### If it were to be revisited

The cost is P2P, not the expansion, so the lever is the near/far split rather than the
multipole order: a larger `nearCutoff` would shrink the accepted-node region the tighter
`mac` inflates. That trades against NeMO's near field, which is a learned model with a fixed
r=6 training range, so it is not a far-field change at all — which is exactly why it is out
of scope here.

---

## Reproduce

```bash
# widebvh: the RPY term and its gate
cd /mnt/ffs24/home/khanmd/programs/widebvh && source ./env.sh
cmake -S . -B build-nemo -G Ninja -DCMAKE_BUILD_TYPE=Release -DWIDEBVH_NEMO_PDEG="3;5;7"
cmake --build build-nemo -j --target two_ball_cart two_ball_cart_rpy \
      widebvh_nemo widebvh_nemo_p5 widebvh_nemo_p3 widebvh_nemo_cart
TC_QUIET=1 TC_NEAR_CUTOFF=6 TC_ORDER=4 \
    ./build-nemo/two_ball_cart_rpy 0.3 1024 10.0 distros/two_ball_t0.bin 2048

# NeMO: calibration
cd /mnt/ffs24/home/khanmd/throwaway/pinn-stokes && source ~/warp_env.sh
export TORCH_COMPILE_DISABLE=1
python benchmarks/mac_calibration.py --selftest
python benchmarks/mac_calibration.py --policy cart --orders 2,3,4 \
       --macs 0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6 --case uniform100k \
       --loadings gravity,random --thetas "" --csv data/cartesian_mac_calibration.csv
python benchmarks/mac_calibration.py --policy cart --orders 4 --macs 0.33 --case all \
       --loadings gravity,random --thetas "" --csv data/cartesian_mac_calibration.csv

# NeMO: symmetry (theta = the far-field-only mac sweep; thetascale = cost curve)
NEMO_FAR_FIELD=widebvh-cart python benchmarks/symmetry_treecode.py grand --n 150
NEMO_FAR_FIELD=widebvh-cart python benchmarks/symmetry_treecode.py theta
NEMO_FAR_FIELD=widebvh-cart python benchmarks/symmetry_treecode.py probe --K 24
NEMO_FAR_FIELD=widebvh-cart python benchmarks/symmetry_treecode.py thetascale

# NeMO: end-to-end (performance runs: torch.compile stays ON)
unset TORCH_COMPILE_DISABLE
python benchmarks/figure12_grand_M.py --backend widebvh-cart
python figures/grand_M_perf.py                 # three-backend A/B figure
NEMO_FAR_FIELD=widebvh-cart python benchmarks/two_suspensions_1M.py
python benchmarks/two_suspensions_1M.py        # bary, same session, for the pair
```

Plus the two drivers written for this study:

```bash
# section 4: bucket granularity, both policies, -> data/cartesian_bucket_granularity.csv
TORCH_COMPILE_DISABLE=1 python benchmarks/bucket_granularity.py uniform100k uniform750k twoball0

# section 1: the RPY m2p identity, in isolation, CPU only, no GPU or widebvh needed
python benchmarks/cart_rpy_m2p_check.py
```

Data: `data/cartesian_mac_calibration.csv`, `data/cartesian_bucket_granularity.csv`,
`data/fig12_scaling_h200.csv` (rows keyed by `backend`), `data/symmetry_widebvh.csv`
(Cartesian rows carry `far=widebvh-cart`). Figures: `figures/grand_M_far_field_ab.{png,pdf}`.
