# Figure 13: what `mac` can and cannot buy, and why the A4500 run is slow

Answers two questions about the 1M two-drop sedimentation benchmark
(nemo.pdf §3.4 / Fig. 13, `benchmarks/two_suspensions_1M.py`):

1. Which `mac` still reproduces the sedimentation patterns?
2. Why does the run take so long on the RTX A4500, and does `mac` fix it?

> **CORRECTION, after real A4500 data.** The §5 prediction below ("~1.2-1.7 s/step,
> nothing is wrong") is **wrong**. A measured A4500 step is **7.13 s**: far field
> **6139 ms**, near field 991 ms. The near field scales as predicted (5.8x the H200);
> the far field is **57x**, i.e. ~10x worse again. The cause is the fp64 P2P, which
> this report under-weighted: sm_86 has 2 fp64 lanes per SM against 128 fp32, so each
> fp64 instruction serializes ~32:1 per warp -- an issue-slot cost, not a FLOP cost.
> Modelling H200 `traverse + 32*p2p`, scaled by 5.8, predicts 6680 ms against the
> measured 6139 ms (9%).
>
> **The fix is `max_leaf`, not `mac`.** At mac 0.8, leaf 64 cuts p2p from 33.7 to
> 6.5 ms at *identical* accuracy (rel_far gravity 8.25e-06 vs 8.49e-06), predicting
> ~1845 ms of far field -- a **3.6x** far-field and **~2.6x** step improvement. See §7.

**Short answers.** Any `mac` up to the expansion's saturation point reproduces the
figure — the parameter is 50-200x away from being the accuracy bottleneck. And no,
`mac` does not fix the A4500: the entire far field is 38% of the step, `mac`
saturates at 1.0, and the hardest far-field tuning available buys **7.6% of the
step, measured end to end**.

All measurements on the H200 (sm_90) on this cluster, git working tree at
`ac78eb4` + the changes listed at the bottom.

---

## 1. `mac` saturates at 1.0

New rows in `data/widebvh_mac_calibration.csv`, PDEG 7 / maxLeaf 1024, on the
Figure-13 cloud at t=100 (`twoball100`, N=1,047,968):

| mac | rel_total, gravity | rel_total, random | far ms | near pairs |
|---|---|---|---|---|
| 0.8 (production) | 8.49e-06 | 2.89e-04 | 107.5 | 10,510,649 |
| 0.9 | 3.63e-05 | 1.20e-03 | 86.2 | 8,022,957 |
| 0.95 | 7.17e-05 | 2.61e-03 | 81.4 | 7,891,233 |
| 1.0 | 1.28e-04 | 7.24e-03 | 80.6 | 7,885,989 |
| **1.1** | **1.28e-04** | **7.24e-03** | **80.5** | **7,885,989** |

`mac` 1.0 and 1.1 agree to the last decimal digit on both error metrics and emit
an identical pair count. **Above 1.0 the tree has no further nodes it can accept**,
so both the error and the cost stop moving. The far field cannot be driven below
~80 ms by this knob on this cloud, at any value.

This also bounds the useful range from the other side: the whole span from
production to saturation is 107.5 -> 80.6 ms, i.e. **27 ms**, against a 280 ms step.

## 2. Which `mac` reproduces the figure

Three anchors, all on this cloud, all gravity loading (the uniform -z force the
figure actually applies):

- **The operator that produced the published figure** is `WarpFMM` theta=0.3:
  rel_total 4.76e-04 gravity / 4.82e-03 random, 492.8 ms. Production `mac` 0.8 is
  **56x tighter**; `mac` 0.9 is still **13x tighter**; even saturated `mac` 1.0 is
  3.7x tighter on gravity.
- **The network's own floor** is ~7.5% PRMSE. `mac` 0.8 sits ~200x below it
  (`src/treecode_widebvh.py:127-134`); `mac` 0.9 ~60x below.
- **The dynamics**, measured directly in §4 below.

So the binding constraint on `mac` is the expansion's convergence, not the figure.
`mac` **0.9** is the pick: it captures 79% of the total available saving while
staying 13x tighter than the operator that made the published picture. `mac` 0.95
and 1.0 give up 2-6x accuracy for 5 more ms.

## 3. The accuracy-neutral knobs matter more than `mac`

`max_leaf` and `hilbert_q` move the traverse/P2P split at essentially constant
error. Sweep on `twoball100` (scratch, 27 configurations):

| mac | max_leaf | hilbert_q | far ms | traverse | p2p | rel_far gravity | rel_far random |
|---|---|---|---|---|---|---|---|
| 0.8 | 1024 | auto | 107.6 | 59.8 | 33.7 | 8.49e-06 | 2.89e-04 |
| 0.8 | 1024 | 20 | **101.0** | 59.8 | 25.7 | 7.32e-06 | 3.06e-04 |
| 0.8 | 512 | 20 | 101.1 | 69.4 | 15.8 | 7.03e-06 | 3.07e-04 |
| 0.9 | 1024 | auto | 86.1 | 46.7 | 25.5 | 3.63e-05 | 1.21e-03 |
| **0.9** | **512** | **20** | **81.8** | 53.3 | 13.3 | 2.97e-05 | 1.22e-03 |
| 1.0 | 512 | 20 | 76.7 | 48.3 | 13.3 | 1.04e-04 | 7.32e-03 |

Two things worth carrying forward:

- **`hilbert_q=20` is free at production `mac`**: 107.6 -> 101.0 ms, 6% off the far
  field with the error unchanged in the third digit. `bucket_granularity.py` already
  found q=16 best on `twoball0`; 20 is better on the later, more spread cloud.
- **`max_leaf` is the knob to reach for on a weak-FP64 card.** At leaf 256 the P2P
  is 11 ms and traversal 90 ms; at 2048 it is 57 ms and 50 ms. The treecode's P2P
  inner loop is entirely fp64 geometry (`widebvh/src/stokes_kernel.cuh:151`, called
  from `treecode.cuh:867`, with a `cub::WarpReduce<double>`), and GA102 runs fp64 at
  1/64 of fp32 — 0.43 TFLOPS against the H200's 34, a 78x gap where fp32 is only
  2.4x. Accuracy does not move across the whole leaf range, so this is free to tune
  per card.

## 4. The dynamics agree: measured, 100 steps

Two seeded 100-step runs of the actual figure configuration, one process each,
identical initial cloud (verified bit-identical at t=0):

- **A**, production: `mac` 0.8, leaf 1024, q auto
- **B**, loose: `mac` 0.9, leaf 512, q 20

| | wall / step | far field | near field |
|---|---|---|---|
| A, production | 282.09 ms | 104.84 ms | 175.39 ms |
| B, loose | 260.71 ms | 83.66 ms | 175.32 ms |
| delta | **-7.6%** | **-20.2%** | **-0.04%** |

The near field is the control and it is exact to 0.04%, so the entire difference is
the far field, as intended.

**Trajectory divergence over the full run**, drop radius 175, bulk settling
distance 2641.9:

| t | RMS displacement | max displacement | / drop radius | / settled |
|---|---|---|---|---|
| 0.2 | 1.46e-03 | 1.47e-02 | 8.4e-05 | 2.7e-05 |
| 0.6 | 7.55e-03 | 3.29e-02 | 1.9e-04 | 2.0e-05 |
| 1.0 | 3.04e-02 | 1.36e-01 | 7.8e-04 | 5.2e-05 |

Worst case at the end of the run is **0.136 of one particle radius**. Bulk
statistics: z mean -2416.886 vs -2416.871, z std 310.801 vs 310.794, bounding box
agreeing to four decimals.

**Rendered comparison** (`figures/drop_1m.py`, the six published panels, 35,000
points per frame): 99.86% of pixels bit-identical, max channel delta 3/255, and
**zero pixels differ by more than 4/255**. The two figures are indistinguishable.

## 5. Why the A4500 is slow

It is not `mac`, and it is not a fault.

- **`mac` can only reach 38% of the step.** On the H200 the step is 280 ms: near
  field 175 ms (62%), far field 105 ms (38%). The near field — Warp hash grid plus
  the self / 2-body / n-body networks over ~21.5M pairs — is untouched by any
  far-field parameter and is the larger half. §4 measured the ceiling: **7.6%**.
- **The A4500 is simply ~4-6x slower here.** The paper's own A4500 figure is
  2.6 s/step with `WarpFMM`, against 0.465 s/step (50-step) / 0.626 s/step
  (150-step) for `WarpFMM` on the H200. Hardware: 640 GB/s vs 4.8 TB/s memory
  bandwidth (7.5x); 56 SMs at 1.65 GHz vs 132 at ~1.8 GHz. Expect **~1.2-1.7 s/step**
  with widebvh, i.e. ~2-3 minutes of stepping for the figure's 100 steps.
- **If it is much slower than that, look elsewhere.** In order of likelihood:
  first-run `torch.compile` + Warp codegen for an sm_86 the image never saw at build
  time (minutes, and the 5 warm-up steps do not cover it); the silent
  `torch.compile` eager fallback documented in `CLAUDE.md` (`config.recompile_limit`
  -> permanent eager, ~2.4x, no error); the snapshot path writing 11 VTK frames of
  ~75 MB of ASCII; and `NEMO_DEVICE_MEM=1` if set, which forks `nvidia-smi` once per
  step.
- **VRAM is not it.** `vram_scaling_report.md` gives ~8.4M particles on an A4500;
  this case peaks at 4.07 GiB. Do not set `TC_PAIR_BUDGET_GB`.

**Diagnosis recipe.** The `[MobFMM]` stdout keys already carry the split, so one
run separates the causes: near and far both ~4-6x the H200 means nothing is wrong;
both inflated uniformly past that means the compile fell back; far alone inflated
means the fp64 P2P / traversal balance, and `--max-leaf 256,512,1024` is the fix.

## 6. Recommendations

- **§3.4 A4500 timing run (measurement #2):** keep production `mac` 0.8, leaf 1024,
  q auto, `benchmark_mode` (50 steps). Changing any of them makes the number
  incomparable with the H200 column and with the published 2.6 s baseline it
  supersedes. The run being slow is the result, not an obstacle to it.
- **Regenerating the Fig-13 snapshots:** `--mac 0.9 --max-leaf 512 --hilbert-q 20`
  is verified equivalent (§4) and 7.6% faster. But note `paper_update_plan.md` §1.3
  says the images are fine and unchanged, and the existing VTKs and PDF are valid.
- **§2.5 (the MAC sentence, p. 21):** if it is worth a clause, `mac` 0.8 is not a
  tight setting chosen for accuracy — it is 56x tighter than the operator it
  replaced and ~200x below the network's floor, and the reason to keep it is that
  loosening it buys under 8% of the step.
- **Not done:** `data/widebvh_mac_calibration.csv` still has no `gpu`/`git_sha`
  column, unlike the `fig11`/`fig12` CSVs. Adding one is a schema change across six
  consumers, so the new rows carry the same provenance gap as the existing 216.
  Every row in that file is H200.

## Changes made

| file | change |
|---|---|
| `benchmarks/two_suspensions_1M.py` | argparse (`--mac`, `--max-leaf`, `--hilbert-q`, `--t-final`, `--snapshots`, `--seed`, `--out-dir`); `np.random.seed` so two runs share a cloud; `.npy` dump beside each VTK. Defaults reproduce the previous behaviour — verified at 282.7 ms/step / 14.14 s over 50 steps against `widebvh_perf_nemo_distros.csv`'s 287.4 ms / 14.37 s. |
| `benchmarks/far_field_drift.py` | same `--mac` / `--max-leaf` / `--hilbert-q` flags, for the A4500 per-step diagnosis |
| `figures/drop_1m.py` | `--data-dir` / `--out`, so two runs can be rendered and compared |
| `data/widebvh_mac_calibration.csv` | +16 rows: `mac` 0.9/0.95/1.0/1.1 on `twoball0` and `twoball100`, both loadings |


---

## 7. A4500: the `max_leaf` result (added after measurement)

`max_leaf` sets the particles per leaf bucket. Large leaves mean fewer, larger P2P
batches -- optimal when fp64 is half-rate (H200), pathological when it is 1/64
(GA102). Shrinking it moves work into traversal, which is fp32.

Sweep on `twoball100`, mac 0.8, H200 measured, A4500 column modelled as
`(traverse + 32*p2p) * 5.8`:

| max_leaf | H200 far ms | traverse | p2p | A4500 predicted | rel_far gravity | rel_far random |
|---|---|---|---|---|---|---|
| 1024 (default) | 107.6 | 59.9 | 33.7 | ~6600 ms | 8.49e-06 | 2.89e-04 |
| 512 | 108.2 | 74.3 | 18.6 | ~3890 ms | 9.24e-06 | 3.02e-04 |
| 256 | 118.8 | 90.6 | 11.1 | ~2580 ms | 5.99e-06 | 3.03e-04 |
| **64** | 134.8 | 108.6 | **6.5** | **~1845 ms** | 8.25e-06 | 2.90e-04 |
| 32 | 154.7 | 126.8 | 6.2 | ~1890 ms | 4.80e-06 | 2.79e-04 |

**Accuracy does not move across the column** -- this is free. The H200 far field gets
*worse* below 1024, which is precisely why 1024 is the tuned default and why it is the
wrong default for a consumer/pro Ampere card.

`mac` is a distraction here: at leaf 64, mac 0.8 -> 0.9 gives 1845 -> 1711 ms (7%) for
4x the error. Keep mac 0.8.

    python benchmarks/two_suspensions_1M.py --max-leaf 64

Confirm on the card itself with `--max-leaf 64`, `128`, `256`: the 32x factor is
calibrated on a single measured point, and traversal may scale worse than 5.8x on a
GPU with less L2.

**Implication for the paper.** The §3.4 A4500 timing should be taken at the
per-card-tuned `max_leaf`, not the H200 default, and the tuning noted -- otherwise the
H-HIGNN headline is quoting a far field running ~3.6x slower than it needs to.
