# Updating `nemo.pdf` for the widebvh far field

Scope: what the far-field replacement (`artifacts/widebvh_far_field_report.md`), the
Cartesian A/B (`artifacts/cartesian_far_field_report.md`) and the two consumer-GPU docker runs
(`artifacts/5090_docker.md`, `artifacts/A4500_docker.md` — RTX 5090 and RTX A4500, both with the
widebvh **fp32 level 3** far field) force to change in the paper, and the smallest set of
writing edits that leaves it correct. Line numbers refer to the margin numbering in the current
`nemo.pdf`; where a `.tex` line is cited it is `sections/03-experiments.tex`.

---

## 0. What does *not* change — check this first

**Figures 1–5 and the accuracy sections (§3.1) are untouched.** `benchmarks/accuracy_grand_M.py`
constructs its operators with `far_field_2b='rpy'` (`benchmarks/accuracy_grand_M.py:520`,
`:661`), which in `gpu_mob_2b.NNMobTorch` is the **direct O(N²) RPY sum**, not the treecode
(`src/gpu_mob_2b.py:103`). No treecode has ever entered a PRMSE number in this paper. The
far-field swap cannot move Figures 1, 2, 3, 4 or 5, and the n-body chunking fix is exact below
the chunk threshold, so it cannot either (verified by the equality checks in
`widebvh_far_field_report.md` §5: 1.3e-07 at N=150/800/10k, at the `index_add_` nondeterminism
floor).

**Figures 6, 7, 9** (validation: line of spheres, equilateral triangle, 3 falling spheres) are
small-N and unaffected.

**Algorithm 1** is still correct as written. Line 7 ("Build LBVH tree and far-field cluster
moments") holds — widebvh builds a cuBQL LBVH — and line 22's `TreecodeRPY` is now *more*
accurate as a name than it was, since the M2P evaluates RPY too (see §2 below).

**No erratum is needed for the n-body chunking bug.** It affected accuracy at N ≳ 400k only,
and the paper reports no accuracy number above N = 200. Its performance effect is within a few
percent (`widebvh_far_field_report.md` §5).

**The fp32 far-field work does not touch any H200 number.** widebvh's fp32 fast path is
selected per level (`fp32_level` 0..3, one `.so` each), and **level 0 — the production setting
on the H200 — is SASS-identical to the pre-fp32 engine** (byte-diffed; see
`artifacts/consumer_gpu_far_field_report.md`). Every H200 timing, every accuracy figure and
every symmetry number in this document is therefore unaffected. Only the two consumer-GPU
columns (RTX 5090 in Fig. 12, RTX A4500 in §3.4) use level 3, and §2.9 below is about saying so
in the paper.

**Update 2026-08-22:** the Fig. 12 H200 column was re-measured at **fp32 level 2** to 2M
(§1.2) and the capacity run (§2.8) is also level 2. If those are used, the "H200 = fp64
throughout" disclosure in §2.9 becomes "H200: fp64 upward pass, fp32 M2P/P2P (level 2)" for
Fig. 12 and §3.4's capacity sentence; Fig. 11, Fig. 13 and the drift run stay fp64.

---

## 1. Figures

### 1.0 Figure 10 — now two figures — **DONE, measured**

§3.3.1 becomes a subsection on the performance of individual components. Figure 10 becomes
**two independent figures**, to be set side by side in LaTeX (`subfigure`/`minipage`) rather
than emitted as one 1×2 grid — so the two can be sized, captioned and referenced separately.
Neither carries an "(a)"/"(b)" prefix in its title (LaTeX supplies those) and neither states the
treecode parameters in the title (`bary`, p=7, mac 0.8 belong in the caption):

| file | content |
|---|---|
| `figures/component_pair_kernel.{png,pdf}` | RPY vs the learned `m_t^(2)` against batch size |
| `figures/component_far_field.{png,pdf}` | far-field treecode against N |

Both are rendered by `figures/component_performance.py` from `data/fig10_pair_kernels.csv` +
`data/fig10_far_field.csv`, produced by `benchmarks/figure10_components.py --panel both`. The
published figure had **no plotting code anywhere in the repo or its history** — it was
transcribed by hand from stdout — so it is reproducible for the first time.

**The far-field figure is new**: the treecode vs N (**50k…4M**, φ=0.1, random loading),
throughput on the left axis and, on a log right axis, two quantities that are **flat in N** — the
far-field rel. L2 error against a sampled fp64 direct sum (2.27e-04…5.05e-04, a **2.2× spread over
an 80× size range**, no trend) and the grand operator's `rel_asym` (3.80e-04…4.61e-04, a **1.2×
spread**, the flatter of the two, drawn with its Hutchinson standard error as bars). Both sit
~200× below the near field's own 7.5% PRMSE, drawn as a grey reference line, so *flat* and
*negligible* are visible at a glance. This is the scale-free-truncation claim, now shown rather
than asserted.

The two error curves **nearly coincide** — that is not redundancy, it is the √2 law of §3.1
showing up in the data (asymmetry = √2 × truncation error), so they are kept visually separable:
open square/solid for rel. L2, open triangle/dashed for asymmetry.

Throughput rises 4.8 → 11.1 M updates/s, peaks at 750k, then **declines gently to 9.9 M/s at 4M**
(10.5 at 2M). Do not describe it as monotone: it saturates and rolls off, which is what an
O(N log N) traversal with a growing tree should do. Far-field wall clock is 191 ms at 2M and
403 ms at 4M.

**How the asymmetry is computed** (needed for the caption and for §3.1 below): `rel_asym` is
‖M − Mᵀ‖_F / ‖M‖_F of the assembled grand mobility, estimated **without ever forming the
6N × 6N operator** by a Hutchinson-style probe (`benchmarks/symmetry_treecode.py:hutchinson`,
K = 24 Rademacher ±1 vectors z ∈ {−1,+1}^(N×6), seed 2) that forms all K(K−1)/2 probe pairs and
uses the identities E[(uᵀMv)²] = ‖M‖_F² and E[(uᵀMv − vᵀMu)²] = ‖M − Mᵀ‖_F², so the cost is K
mobility applications rather than 6N. The estimator is validated against a dense assembly at
N = 150 (ratio 0.933), and its own standard error, ~13% of the value, is drawn as the error bars —
comparable to the 1.2× spread across N, which is what makes "flat" a statement about the data
rather than about the eye.

N = 2M and 4M had no configuration on disk; they were generated with
`cluster.uniform_cluster_generation_large(0.1, N, seed=0)` (65 s and 131 s) into
`tmp/uniform_large_0.1_{2000000,4000000}.csv`. The pre-existing files up to 1M record no seed.

N = 5k and 10k **were measured and remain in the CSV, but are excluded from the figure**: with
`max_leaf`=1024 a cloud that small is only a handful of buckets, so the far field is pinned by its
~7 ms fixed build cost (7.2 ms at both 5k and 10k, i.e. flat in N) and few nodes pass the MAC.
Those points measure tree construction rather than the asymptotic regime the figure is about, and
they compress the throughput curve against the axis. The cut is one argument (`min_n` in
`_load_far_field`), so the small sizes are easy to show if a referee asks for them.

**The pair-kernel figure — one number moved and it changes the text.** Re-measured on this H200
with the same code paths (the learned curve is now labelled `m_t^(2)`, matching the paper's
notation, rather than `f_t`):

| | published | re-measured | |
|---|---|---|---|
| RPY @ 2^22 | 1023 M/s | **1214 M/s** | 1.19× |
| `m_t^{2)}` @ 2^22 | 495 M/s | **496.5 M/s** | 1.00× |

`m_t^{2)}` reproduces exactly; RPY is 19% faster than the published figure (same script, same flags —
most likely a different node or driver). **The ratio therefore moves from ~2× to 2.43×**, so
§3.3.1 lines 653–657 need updating: *"495 million pair evaluations per second, corresponding to
roughly a 2× slowdown"* becomes ~2.4×. The surrounding argument is unaffected — it is still a
small runtime gap against ~120× the arithmetic — but the number is quoted explicitly.

Two protocol notes recorded in the CSV rather than silently resolved:
- `benchmark.py` ships 1 warmup / 3 timed iterations, which puts `torch.compile`'s compilation
  *inside* the timed window; that is what produced the published low-batch `f_t` points. The CSV
  carries both that (`protocol=published`, which reproduces 495 M/s) and a 10/50 protocol
  (`protocol=matched`, plotted). They agree to 1% at 2^22 and diverge only at small batch.
- The paper says *"torch.compile with autotuning set to max"*, but neither `bench_rpy.py` nor
  `benchmark.py` passes `mode="max-autotune"`. Not changed — it would break "same code paths" —
  but the caption's claim does not match the code and needs a decision.

### 1.1 Figure 11 — runtime breakdown (p. 23) — **DONE, measured**

**First, a provenance correction to this section as it was originally written.** It said the
figure comes from the hardcoded `times_ms_h200` in `figures/grand_M_perf.py:118`. It does not.
That function draws a **pie chart** at a single size, is never called from `__main__`, and
appears nowhere in `nemo.pdf`. Figure 11 is the **two-panel** figure — (a) three operator
curves, (b) a four-segment stacked bar — drawn by
`figures/plot_runtime_breakdown.py:plot_runtime_summary` from a **1319-line hand-transcribed
`RAW_DATA` dict**, itself transcribed from five archived stdout captures in
`figures/runtime_breakdown/*_h200.txt`, and saved to an absolute path in a *different
checkout*. Like Figure 10, it was not reproducible from the repo.

It is now. `benchmarks/figure11_breakdown.py` measures it into
`data/fig11_breakdown_h200.csv`; `figures/grand_M_perf.py:runtime_breakdown()` renders
`figures/runtime_summary_two_panel.{png,pdf}` from that CSV. `--from-logs` re-parses the
archived captures, so the published Warp column sits in the same CSV — and doubles as a self
test: it rebuilds the published run's own wall-clock summary to within **1.9%** at all 15
points. `plot_runtime_breakdown.py` is kept, marked superseded, for provenance.

**Measured, H200, uniform φ=0.1.** This table is *published vs new* — the left of each arrow
is `warp-published`, i.e. what the paper reported. For "what the far-field swap bought", use
the `warp`-at-HEAD control in §1.1b instead; the two differ because the near field also moved.

| N | far, published → widebvh | total, published → widebvh | far share | n-body overhead |
|---|---|---|---|---|
| 10k | 7.33 → 7.67 | 11.58 → 11.68 | 63.3% → 65.6% | 1.21× → 1.23× |
| 50k | 10.99 → 10.96 | 21.62 → 21.91 | 50.8% → 50.0% | 1.53× → 1.54× |
| 100k | 18.52 → **14.70** | 36.92 → **34.41** | 50.1% → 42.7% | 1.60× → 1.78× |
| 200k | 47.42 → **22.65** | 82.04 → **60.15** | 57.8% → 37.7% | 1.49× → 2.02× |
| 1M | 224.79 → **91.70** | 389.05 → **275.10** | 57.8% → **33.3%** | 1.52× → **2.24×** |

Below 50k the swap is a wash — the far field is fixed-cost dominated there and widebvh is
marginally *slower* — exactly as `widebvh_far_field_report.md` §1 found for Figure 12.

**Three conventions changed; all three are recorded in the script and matter for anyone
diffing against the old figure.**

1. *Panel (a) now plots the operator's own end-to-end timer, not the sum of panel (b)'s four
   segments.* Those segments do not tile the step: ~0.6 ms of tensor staging and the final
   far+near add sit outside every named timer (5% of the step at 10k, 0.3% at 1M). It is in
   the CSV as `unaccounted_ms`.
2. *The neighbour-search segment no longer double-counts.* The published figure summed
   `Warp::HashGrid took` and `[MobFMM] near-field construction`, which are the same interval
   timed two ways (25.07 vs 25.195 ms at 1M). Only the CUDA-event value is used now, for both
   backends. Curiosity worth knowing: that double count (~0.55 ms) almost exactly cancelled
   the unaccounted staging time in (1), which is why the old panel (a) tracked the wall clock
   despite summing four segments that do not add up to it.
3. *Reduction is a median over the six timed applies, not a mean of the last three.* The
   far-field timer is genuinely noisy — `src/treecode.py:321` says so in a FIXME — and the
   mean has no outlier protection. Both reductions reproduce the published run equally well
   (1.78% vs 1.89% worst case), so this costs nothing and buys robustness.

**Two measurement traps found the hard way; both produce plausible-looking wrong numbers with
no error, and both are now prevented by running every (size, operator) in its own process.**

- *Across sizes*: sweeping all five in one process trips `config.recompile_limit` and falls
  back to eager permanently. It inflated the 1M self+2-body segment from 28.2 to **116.8 ms**
  — 4×, at the last size only, so it reads as a scaling result rather than an artifact. Same
  failure mode as `widebvh_far_field_report.md` §5.
- *Across operators*: the third operator built in a process intermittently reports a ~10× far
  field (**108 ms against 11 at 50k**; 85.6 against 7.6 at 10k). Building two solvers first
  leaves the caching allocator fragmented, and the resulting `cudaMalloc`/`cudaFree` syncs
  land inside the far field's own CUDA-event bracket. `op.close()` and `empty_cache()` do not
  clear it. The published run never hit this — `WarpFMM` allocates differently.

#### 1.1a Text changes Figure 11 forces (§3.3.2, lines 670–681)

Ordered by severity. Two of these are corrections to what §1.1 originally claimed.

**(i) Lines 678–681 are *reversed*, not "softened" — this section previously got that wrong.**
Currently: *"the far-field solver depends on the full suspension and therefore grows with N,
with the slight superlinear trend expected from the O(N log N) complexity of our LBVH/treecode
traversal. Consequently, at fixed φ, a larger fraction of the total runtime is spent in M_ff
as N increases."* Under Warp the far-field share was flat-to-rising and stayed around half the
step (50.8% → 57.8% over 50k…1M as published; 49.5% → 55.2% for the control at HEAD), which
supported it. Under widebvh it **falls monotonically, 65.6% → 33.3%**.

The right rewrite keeps the complexity argument and drops only its conclusion, because the
mechanism is not that O(N log N) stopped being true — it is that it does not yet dominate.
`data/fig10_far_field.csv` resolves this cleanly: the widebvh far field carries a fixed LBVH
build cost that is nearly flat in N (4.7 ms at 10k, 7.0 ms at 1M) and is essentially the
*entire* far field at the small end. Over 10k…1M that constant amortizes faster than the
traversal grows, so the share falls. **The superlinearity is real and shows up just past the
figure's range** — the same CSV gives 2.10× and 2.11× per doubling at 1M → 2M → 4M. So: at
fixed φ the near field is O(N) and the far field O(N log N), but over the sizes shown the far
field's fixed setup cost dominates that difference; its share falls and flattens, with the
crossover beyond 1M.

**(ii) Lines 673–674 — a required change this document did not previously list at all.**
*"The addition of the n-body term adds around 50% overhead in large scale simulations
(>100k)."* That was a stable ~1.5–1.7× under Warp at every size, and still is for the control
at HEAD. Under widebvh it is **1.78× at 100k, 2.02× at 200k, 2.24× at 1M** — the n-body
correction more than *doubles* the runtime at 1M, and the overhead now **grows with N** instead
of being constant. Worth one clause on why, because it is emphatically not a regression: §1.1b
shows the n-body segment costs the same under both backends to the second decimal (151.65 vs
151.64 ms at 1M). What changed is the operator it is measured *against* — the two-body operator is
far-field dominated, so it got cheaper while the correction did not. This is the flip side of
(i) and belongs next to it.

**(iii) Lines 670–673 survive, with a bigger number.** *"single-digit percentage slowdown
compared to just RPY"* still holds — NeMO-2b vs RPY moves from **1.7–3.4%** (published;
0.7–4.2% for the control at HEAD) to **1.1–6.0%** under widebvh, worst at 100k and 4.3% at 1M.
Same cause as (ii): the shared far-field term that both operators pay shrank, so the fixed gap
between them is a larger fraction of a smaller total. No rewrite needed — but do not restate
it as "~2%", and note the claim is now closer to its "single-digit" ceiling than it was.

**(iv) Line 681 survives.** *"Neighbor search remains a relatively small part of the total
runtime throughout"* — 4.9% at 10k falling monotonically to 1.0% at 1M, so "throughout" is if
anything understated.

**(v) Panel (b) is now the cleanest single illustration of the modularity claim** (§3.3
below): with the near field held fixed, one segment shrinks 2.5× at 1M and nothing else moves.
See the control in §1.1b.

#### 1.1b The control — and the confound it removes

`widebvh` and `warp-published` differ in **two** things, not one: the far field, and the
near-field work done since. Their `self + 2-body` segments agree to 1–4% (identical code), but
the **n-body segment is 10–15% more expensive at HEAD** than in the archived logs (131.8 →
151.6 ms at 1M) — that is the pair-chunking commit `ac78eb4`, not the far field. The archived
column alone therefore cannot support "only the far field moved", and would also overstate the
end-to-end speedup.

So a **`warp` run at HEAD** was added: same code, same node, same protocol, only the far-field
solver differs. It is the clean A/B, and it is about as clean as this kind of comparison gets —
at 1M the two backends' near-field segments agree to the second decimal:

| N=1M segment | warp @ HEAD | widebvh | |
|---|---|---|---|
| far field | 226.48 | **91.70** | **2.47× faster** |
| self + 2-body | 28.07 | 28.08 | identical |
| n-body correction | 151.65 | 151.64 | identical |
| neighbour search | 2.61 | 2.65 | identical |
| **total** | **410.09** | **275.10** | **1.49× faster** |

Far-field speedup by size: 1.00× at 10k, 0.98× at 50k, 1.27× at 100k, 2.09× at 200k, 2.47× at
1M — the swap pays off only once the far field stops being fixed-cost dominated, consistent
with `widebvh_far_field_report.md` §1.

**This control is what makes §1.1a(ii) safe to assert.** Under `warp` at HEAD the n-body
overhead stays at the published ~1.5–1.7× at every size (1.23, 1.58, 1.67, 1.52, 1.59);
under widebvh it climbs to 2.24× at 1M. Since the n-body segment itself is identical between
the two, the rise is caused by the far field getting cheaper and nothing else — it is not code
drift, and not a regression in the n-body kernel.

Three backend keys are deliberately kept separate in the CSV: `warp-published` (what the paper
reported), `warp` (the A/B control at HEAD), `widebvh` (production). Cite `warp` for any
"what did the swap buy" claim and `warp-published` only for "what the paper said".

#### 1.1c Independent cross-checks

Figure 11 was measured by a new script, so it is worth noting it agrees with two measurements
made earlier by different code on different protocols:

- **vs Figure 12** (`data/fig12_scaling_h200.csv`, wall-clock timing, trimmed mean, one
  process per size): at the one shared size, 200k, Fig 12 gives 60.45 ms total / 22.50 ms far
  and Fig 11 gives 60.15 / 22.65 — **0.5% and 0.7%** apart.
- **vs Figure 10's far-field sweep** (`data/fig10_far_field.csv`, far field only, dedicated
  repeat loop, `TORCH_COMPILE_DISABLE=1`): 7.22/10.47/14.39/22.10/90.92 there against
  7.67/10.96/14.70/22.65/91.70 here — **+0.9% to +6.2%**, the gap shrinking with N, which is
  what a per-call fixed overhead that Fig 10's tighter loop excludes should look like.
- **vs the published run**: the `--from-logs` parser rebuilds the archived wall-clock summary
  at all 15 points to within **1.9%**.

### 1.2 Figure 12 — end-to-end scaling (p. 24) — **H200 column re-measured 2026-08-22 to 2M (fp32 L2); 5090 column re-measured 2026-08-23 to 2M (fp32 L3)**

**New H200 column** (`data/fig12_scaling_h200_f32l2.csv`, report
`artifacts/fig12_h200_f32l2_report.md`): current tree, widebvh **fp32 level 2**, one process
per size above 1.5M, RunPod H200. Throughput saturates at **~3.97M updates/s from 750k and is
flat to 2M** (252 ms at 1M, 504 ms at 2M; far field a steady 25%). fp64 on the same node: 280 ms
at 1M / 570 ms at 2M — level 2 buys 1.5× on the far field, 11–13% on the step. True process
VRAM (nvidia-smi, exclusive GPU): 6.2 GB at 1M, **9.0 GB at 2M**. For §2.6 quote "**4.0M
updates/s, flat from 750k to 2M**" and "2M particles in half a second". Caption must say the
H200 column is level 2 (§0 update). The single-process sweep tripped `recompile_limit` at the
9th size (1.75M read 3.5× slow) — the §1.1 trap again; the CSV carries the isolated re-runs.

*Original notes (pre-re-run) follow.*

H200 numbers are already measured and in `data/fig12_scaling_h200.csv` (widebvh, warp and
widebvh-cart at all five sizes, `mac` 0.8 / p7 / `max_leaf` 1024, git `ac78eb4`). The RTX 5090
column is now measured too (below), and the published two-GPU figure is regenerated by
`figures/grand_M_perf.py:scaling_test_h200_vs_5090` → `figures/gpu_scaling_h200_vs_5090.{pdf,png}`,
copied to the paper as `figs/gpu_scaling_h200_vs_5090_08_22.pdf` (the `_05_19` file is kept
beside it; `03-experiments.tex:379` now points at the new one). **Regenerated again 2026-08-23
with the final 5090 column** (all ten sizes paired, both cards to 2M — see below), so the
`_08_22` copy in the paper repo is stale and needs re-copying.

**Validated against the process-isolation trap found in §1.1** — worth recording, because
`benchmarks/figure12_grand_M.py:155` sweeps all five sizes in *one* process, which is the
pattern that silently degraded the Fig 11 run. It is the milder case (one operator per size,
so 5 compilations rather than 15) and it did **not** degrade: 500k and 750k re-measured in
isolated processes give 138.36 / 205.28 ms against the CSV's 139.02 / 206.27 — **0.5% apart,
with the far field matching to 0.2%**, and in the opposite direction from a compile fallback.
The 200k point independently cross-checks against Fig 11 to 0.5%. No re-run needed.

Figure 11 also extends this curve to **1M: 275.10 ms = 3.635M updates/s** — the same rate as
750k (3.636M), so the throughput has *flattened* by 1M rather than still climbing. Prefer that
framing over "rises to 3.6M", and 1M is the rounder number to quote.

| N | published H200 (from the figure) | new widebvh | throughput, old → new |
|---|---|---|---|
| 50k | ~21 ms | 22.50 | 2.4M → 2.22M |
| 100k | ~36 ms | 34.91 | 2.7M → 2.86M |
| 200k | ~81 ms | 60.45 | 2.5M → 3.31M |
| 500k | ~187 ms | 139.02 | 2.7M → 3.60M |
| 750k | ~278 ms | 206.27 | 2.4M → 3.64M |

**The character of the curve changes**, which is the part that needs writing attention rather
than just a number swap. The paper currently claims *flat* throughput (2.4–2.7M "across the
full range", line 685) and explains it by near-linear scaling in N. The new throughput
**rises** with N, 2.2M → 3.6M, because the far field was the superlinear term and it just got
cheaper. At 50k the new number is very slightly *worse* than published (the far field is
fixed-cost dominated there and the swap is a wash — `widebvh_far_field_report.md` §1). Report
it honestly as a range with a trend.

#### The RTX 5090 column — **re-measured 2026-08-23, machine-written, 50k…2M** (`artifacts/fig12_5090_runpod_report.md`)

**Supersedes the 2026-08-22 old-box measurement** (`artifacts/5090_docker.md`, 50k…750k,
hand-transcribed; its table no longer appears here — the transcript and the old CSV rows are
archived). Re-run on a
RunPod RTX 5090 (32 GB, driver 590.48.01) from the same image stack (pinn-stokes @
`ac78eb4-dirty`, widebvh `03efcdb`), same protocol as the H200 f32l2 column — **one process per
size** (the §1.1 trap), `NEMO_FAR_FP32_LEVEL=3`, `mac` 0.8 / p7 / `max_leaf` 1024, the 1M–2M
configurations generated with the documented `seed=0` recipe — and **extended to the full
50k…2M grid**, so the figure now has both cards at every size. `data/fig12_scaling_5090.csv`
is machine-written (full precision, `std_ms`/`min_ms`/`git_sha` populated); the old
hand-transcribed rows are archived in `artifacts/logs/fig12_5090_runpod/` beside all run logs
and the driver script. The new pod is faster than the old box at every overlapping size
(0.71×…0.96×, converging with N): the old box's small-N rows were GeForce **idle-clock**
artifacts, not hardware — this pod idles at 180 MHz SM, clock-locking is not permitted, and
the 50k row needs `--warmup 400` to hold boost clocks through the timed window (report §
"Protocol"; sizes ≥100k self-warm).

| N | H200 total / far (fp32 L2) | RTX 5090 total / far (fp32 L3) | 5090/H200 | throughput H200 → 5090 |
|---|---|---|---|---|
| 50k | 21.77 / 10.65 | 23.42 / 10.99 | **1.08×** | 2.30M → 2.13M |
| 100k | 32.97 / 12.59 | 35.36 / 12.35 | 1.07× | 3.03M → 2.83M |
| 200k | 56.97 / 18.18 | 60.79 / 17.15 | 1.07× | 3.51M → 3.29M |
| 500k | 128.59 / 33.25 | 137.32 / 29.08 | 1.07× | 3.89M → 3.64M |
| 750k | 189.36 / 47.10 | 201.99 / 39.04 | 1.07× | 3.96M → 3.71M |
| 1M | 252.03 / 61.54 | 269.04 / 50.32 | 1.07× | 3.97M → 3.72M |
| 1.25M | 315.83 / 77.45 | 338.87 / 64.32 | 1.07× | 3.96M → 3.69M |
| 1.5M | 379.51 / 93.40 | 406.63 / 78.82 | 1.07× | 3.95M → 3.69M |
| 1.75M | 440.90 / 109.06 | 476.38 / 88.71 | 1.08× | 3.97M → 3.67M |
| 2M | 503.58 / 123.75 | 542.63 / 101.22 | **1.08×** | 3.97M → 3.69M |

(ms; peak torch-allocated VRAM on the 5090: 1.01 GB at 50k → 4.17 GB at 2M — the same 4.17 GB
the H200 reports there, same allocation pattern.)

**The story is no longer "the gap closes with N" — it is "there is no gap to speak of."** The
earlier table paired the fp64 H200 column with the old-box 5090 and read 1.46× → 1.02×; with
both cards at their operating points (H200 level 2, 5090 level 3) the ratio is a **flat
1.07–1.08× across the entire 50k…2M range**, and the 5090's throughput plateaus at ~3.7M
updates/s from 750k just as the H200's does at ~3.97M — **2M particles in 0.54 s on a consumer
card**. From 100k up the 5090's level-3 far field is outright *faster* than the H200's level-2
one (0.98× shrinking to 0.82× at 2M: 101 vs 124 ms); the entire remaining gap — and then some —
is the near field (441 vs 380 ms at 2M, 1.16×), which is TF32 tensor-core work on both cards.
The old narrative's small-N caveat (the 5090 paying disproportionately for the fixed-cost far
field) is gone with the clock artifact: at 50k the two cards' far fields are 10.99 vs 10.65 ms.
This is §2.9's argument in its cleanest form — with the far field in fp32, every stage of the
step is single-precision work, and a flagship consumer card lands within 8% of an H200 at
every size. Each card is still shown at its own operating point; the comparison should not be
read as "same kernels".

**Styling.** Same as the published figure (colours, hatching, markers, twin axes, 9.1 × 4.7 in)
with one deliberate change: the legend moved from centre-left to upper-left with 1.3× headroom
on the throughput axis, because the new throughput curves rise from the bottom-left corner and
sat under the old legend. **The caption must state both operating points** — H200 fp32 level 2
(fp32 M2P/P2P, fp64 upward pass), RTX 5090 fp32 level 3, per the §0 update and §2.9.

**The three provenance items from the first measurement, updated:**

1. ~~Hand-transcribed CSV / retrieve `/persistent/results/fig12_5090.csv`~~ — **resolved by
   the re-measurement**: the CSV is machine-written with `std_ms`/`min_ms`/`git_sha`. The old
   rows are archived, not merged (different 5090 system; mixing boxes in one column was the
   other reason to re-measure). The 5090 half of measurement #8 is moot.
2. `benchmarks/figure12_grand_M.py:51` `FIELDS` still has **no `fp32_level` column** and no
   `--fp32-level` flag. Mitigated but not fixed: level 3 is now evidenced on disk — the
   archived driver script in `artifacts/logs/fig12_5090_runpod/` exports
   `NEMO_FAR_FP32_LEVEL=3` and the report records it — rather than living only in run notes.
   Adding the column remains the clean fix.
3. ~~The H200 rows predate the two-body pair chunking — re-measure at the current tree~~ —
   **done 2026-08-22** (measurement #7): both columns are now the same tree (`ac78eb4-dirty`),
   the same protocol, and machine-written.

### 1.3 Figure 13 / §3.4 sedimentation (p. 25) — **images fine, timings stale**

The snapshots are qualitative and unchanged. All the numbers in the paragraph at lines 709–714
are stale — see §3.3 below.


---

## 2. Text that is now **wrong** and must change

These are correctness fixes, not enhancements. Ordered by severity.

### 2.1 §2.4.3 Treecodes, lines 388–396 — the expansion description (p. 12)

Currently: *"we use a second-order Taylor expansion, retaining monopole, dipole, and quadrupole
moments … We keep the expansion order low so that each accepted-cluster evaluation remains
cheap enough for a simple target-wise traversal kernel."*

The production far field is a **degree-7 barycentric-Lagrange interpolation at Chebyshev proxy
points** on a cuBQL LBVH. Every clause above is now false, and the last one is not merely stale
but **backwards** — see §2.3.

*Rewrite: one paragraph.* Keep the LBVH-vs-octree justification (lines 382–387) verbatim, it
is unchanged and still good. Replace only the expansion description.

### 2.2 §2.4.3, lines 394–396 and Eq. (21) — the mixed kernel (p. 12)

Currently: *"accepted far-field clusters are evaluated using a lower-order Stokeslet-based M2P
approximation, similar in spirit to Kernel-Aggregated FMM"*, and Eq. (21) writes the accepted
term as a Stokeslet M2P operator `T_M2P` beside a direct `G_RPY`.

**The M2P is now RPY** (`WIDEBVH_RPY_A`; `widebvh_far_field_report.md` §3b). Delete the mixed-kernel
sentence and make Eq. (21) use one kernel throughout.

Worth one clause noting this is a *strict improvement*: the Stokeslet/RPY mismatch is a floor
no MAC can lower — measured at **6.8e-3** of the far field under incoherent (random) loading
(`cartesian_far_field_report.md` §1), i.e. 18× the current operating point. That single clause
justifies the change and pre-empts a referee asking why the kernels used to differ.

### 2.3 §2.4.2, lines 370–372 — the low-order argument (p. 11)

Currently: *"Finally, the target accuracy is moderate … This pushes the fast-summation method
toward low-order approximations, where there is less arithmetic work per accepted interaction
to amortize traversal, scheduling, and memory-access overheads."*

**This is the one substantive claim in the paper that the new results reverse.** Low order
loses, and not marginally: matched at `rel_total` ≈ 7e-4, PDEG 7 costs 95 ms against PDEG 3's
310 ms at 1M, because the tighter `mac` a low order needs explodes the near-pair count (10.5M →
62.4M). Confirmed independently with a completely different expansion — the Cartesian Taylor
policy at order 4 needs a 2.4× tighter `mac` and lands 1.24× slower end to end
(`cartesian_far_field_report.md` §7).

Rewrite: just remove that sentence/argument.

### 2.4 §2.4.2, lines 364–367 — the barycentric FMM citation (p. 11)

Currently: *"in the sedimentation benchmark reported in Sec. 3.4, a GPU-accelerated barycentric
FMM implementation [33] was about an order of magnitude slower than the fast learned near-field
stage."*

The production far field is **now a barycentric-Lagrange treecode**, so as written this reads
as an argument against the method finally adopted. Reword to 1. emphasize that implementtaion is a bit old or stale. 2. we're only using the same moments as them, nothing else of FMM pipeline. (do it using fewest words possible, dont want to dwell on it too much)


### 2.5 §3.2.3, lines 616–617 — the MAC value (p. 21)

Currently: *"With a MAC parameter of 0.3, the far-field component preserves the large-scale
hydrodynamic structure…"*

Now `mac = 0.8`. **Do not simply swap the number** — `theta` (WarpFMM) and `mac` (widebvh) are
different acceptance criteria and the paper will look sloppy if 0.3 becomes 0.8 with no
explanation. If §2.4.3 is rewritten as in §2.1, the parameter is defined there and this line
just cites it.

### 2.6 §3.3.2, lines 683–687 — throughput (p. 24)

*"roughly 2.4 to 2.7 million particle updates per second across the full range"* → 2.2M at 50k
rising to 3.6M at 750k. *"in a suspension with about 2.7 × 10⁶ particles, one application of
NeMO operator takes about a second"* → 3.6 × 10⁶. See §1.2 for the change in the curve's
character.

Figure 11 now extends this to 1M on the same clouds and agrees: **275.10 ms**, i.e. 3.64M
updates/s — the same rate Figure 12 reaches at 750k, so the throughput curve has flattened by
1M rather than still climbing. Worth using, since 1M is a rounder number to quote than 750k
and it is now measured rather than extrapolated.

The RTX 5090 sentence in the same paragraph gets the same treatment (re-measured 2026-08-23,
§1.2): **2.13M at 50k rising to 3.72M at 1M and flat to 2M** — 2M particles in 0.54 s — i.e. a
range with a trend on both cards, and the two cards a **flat 1.07–1.08× apart at every size**,
not the closing gap the old column suggested. The rest of that paragraph — the hardware
explanation — is §2.9.

### 2.7 §3.4, lines 709–714 — sedimentation timings (p. 25) — **DONE, measured per step**

`benchmarks/far_field_drift.py` → `data/far_field_drift_1M.csv`: 150 timed steps at
N = 1,047,968, both backends, seeded so they see an identical cloud, one process each.
Averaged over the full 150 steps the paper quotes:

| | published | warp, re-measured | **widebvh** |
|---|---|---|---|
| per step | 0.48 s | 0.626 s | **0.280 s** |
| far field M_ff | 283 ms (61%) | 452 ms (72%) | **107 ms (38%)** |
| near field M_nf | 182 ms (39%) | 173 ms (28%) | **172 ms (61%)** |
| first 150 steps | 74 s | 93.9 s | **42.1 s** |

The near field is the control and it is a very tight one: **172.43 vs 172.57 ms** mean over
150 steps, 0.08% apart. Every other difference in the table is the far field.

**The headline speedup is 2.23×, not the 1.62× the existing 50-step aggregate implies** — and
the reason is measurement #4, below.

#### The t-dependence — the sentence at 711–713 survives, but only because of the swap

*"as the simulation progresses, the particle distribution becomes more non-uniform, which leads
to a slight increase in the far-field runtime due to the treecode traversal."* The cloud does
become non-uniform as claimed (σ_z 238 → 511, bounding volume ×5.9 over the 150 steps). What
the two backends do about it could hardly be more different:

| far field, mean of 10 steps | steps 0–9 | 95–104 | 140–149 | drift |
|---|---|---|---|---|
| widebvh | 98.94 | 108.31 | 115.19 | **+16.4%** |
| warp | 256.28 | 490.56 | 969.05 | **+278%** (3.78×) |

So the sentence is *accurate for widebvh* — 16% over 150 steps is genuinely "slight" — and was
a severe understatement for the backend it was written about. Keep the sentence, and consider
saying what "slight" now means, because it is a real robustness result rather than a caveat:
the near field is identical in both columns (−11.8% vs −11.6%, tracking the near-pair count as
the drops spread, −11.4%), so the degradation is purely the traversal.

Two consequences worth writing:

- **The widebvh total is flat, not falling or rising**: 285.7 ms over steps 0–9 → 279.9 over
  140–149, −2.0%. The far field grows 16% while the near field falls 12% as pairs thin out, and
  the two nearly cancel. That is a better sentence than either component alone.
- **The published "74 s for the first 150 steps" does not reproduce.** A measured warp run takes
  **93.9 s**. 150 × the early-step rate (452.8 ms over steps 0–49) gives 68 s, so 74 s looks like
  an early rate extrapolated across the run rather than a run that was timed to completion. Do
  not carry it forward; 42.1 s is measured end to end.

**Why the 50-step aggregate could not have shown this** (and why measurement #4 was correctly
flagged as still open): over steps 0–49 the two backends sum to 14.0 s and 22.6 s, which
reproduces `data/widebvh_perf_nemo_distros.csv`'s 14.37 / 23.26 s to within 3% — the divergence
is almost entirely after step 50. Averaging over the wrong window hid a 3.8× effect.

**Bearing on §2.8 — resolved by re-running.** The A4500 H-HIGNN number (2.6 s/step) was
measured with warp, whose per-step cost on this case nearly quadruples over 150 steps, and the
question "what window was 2.6 s averaged over" turned out to be unanswerable: **no log, CSV or
note of that measurement exists anywhere in the repo** (`grep -rli a4500` finds only prose). It
is not used as a baseline for anything below; the A4500 has simply been re-measured (§2.8).

**One more number to fix while in this paragraph: the particle count.** The paper says the two
drops total **1,046,610** particles (line 425, and the caption); every log of this case — H200,
A4500, 4060 — says **1,047,968** (`benchmarks/two_suspensions_1M.py`, seed 0). Use the code's
number; the 1,358-particle gap is presumably an earlier generator seed.

### 2.8 §3.4, lines 715–726 — the H-HIGNN comparison — **DONE, measured on the A4500**

The 11× headline is *"2.6 s per step on a single RTX A4500"* against H-HIGNN's 28.8 s on four
GPUs. That 2.6 s was measured with `WarpFMM`, and §2.7 explains why it cannot be audited. The
case has now been re-run on an RTX A4500 in the `nemo:2.0` image (`artifacts/A4500_docker.md`):
`benchmarks/two_suspensions_1M.py --fp32-level 3 --log-csv`, N = 1,047,968, **50 timed steps
after 5 warmup**, dt = 0.01, `mac` 0.8 / p7 / `max_leaf` 1024 — i.e. production everything
except the far-field precision, which is the consumer-GPU operating point of §2.9.

| | published (warp: fp32 kernels, fp64 accumulation) | **widebvh fp32 L3, measured** | |
|---|---|---|---|
| per step, wall incl. Euler update | 2.6 s | **1.12 s** (min 1.07, max 1.17) | **2.3× faster** |
| far field M_ff | — | **215 ms** (19%) | |
| near field M_nf | — | **901 ms** (81%): self+2b 263, n-body 628, nsearch 8.5 | |
| total GPU | — | 1117 ms | |
| 50 steps | 132 s | **55.9 s** | |
| peak VRAM (torch-allocated) | "fits comfortably" | **2.58 GB** of 20 GB | |
| vs H-HIGNN, 28.8 s/step on **four** A4500s | 11× | **25.8× → "~26×"**, on 1/4 the hardware | |

(28.8 / 1.118 = 25.8; per 50 steps, H-HIGNN's ~24 min against 55.9 s gives the same 25.8×.)

**It beat the estimate, and the breakdown says why.** This document previously guessed 1.5–1.7
s/step by assuming the far field is the same fraction of the A4500 step as of the H200's
(38%). It is **19%**: level 3 takes the A4500 far field from 6139 ms (fp64, measured earlier —
`artifacts/fig13_mac_report.md:9-14`) to 215 ms, 28.6×, so the treecode is no longer what the
step costs on this card. What remains is the learned near field — 901 ms, **5.2× the H200's 172
ms** on the same cloud — which is TF32 tensor-core work and scales with the card's tensor
throughput, not its fp64 rate. Worth one clause in the paper, because it is the honest reading
of "what a consumer GPU is bottlenecked by" and it points at the right future-work item
(cheaper n-body correction), not at the far field.

**Window.** Both figures are 50-step means, so the comparison is like-for-like in window as
well as in problem. The warp-era concern — that an early window understates a drifting far
field — does not apply: the widebvh far field drifts +16% over 150 steps (§2.7), so a 150-step
figure would be ~1.15 s/step, not materially different. Quote 50 steps, which is what H-HIGNN
reports.

**Precision.** Say once that the A4500 run uses the single-precision far-field kernels (§2.9).
Before writing that it is like-for-like, **check what precision H-HIGNN \cite{hignn25} reports
running at** — it is a PyTorch/GNN code so fp32 is likely, but it is not recorded here. If it
does, the precision caveat is a disclosure, not a concession; either way the accuracy cost is
quantified in §2.9 (0.035 radii over 100 steps with the production tree, 0.045 with leaf 512).

**VRAM.** "Fits comfortably on a single GPU" can now carry a number: 2.58 GB torch-allocated
of the A4500's 20 GB. The true process footprint is ~0.7–1.8 GiB higher because the treecode's
`cudaMalloc`s bypass the caching allocator (`CLAUDE.md`; `NEMO_DEVICE_MEM` was off for this
timing run), so write "**under 5 GB**" unless a precise figure is wanted, in which case one
untimed run with `NEMO_DEVICE_MEM=1` gives it.

**Suggested replacement for line 440** (`03-experiments.tex:440`): *By comparison, our
implementation requires 1.12 s per step on a single RTX A4500 (single-precision far field, see
§3.3.2), or 56 s for 50 steps, in under 5 GB of memory. NeMO thus delivers a roughly 26× wall-clock
speedup while using a quarter of the hardware.*

#### Add the single-GPU capacity here — it is a second, independent axis of the same claim

**Superseded 2026-08-22** (`artifacts/max_particles_h200_f32l2_report.md`,
`benchmarks/max_particles.py`, fp32 L2): **65,450,827 particles (403³) is now a sustainable
steady state at 18.75 s per mobility application** on one H200, with `torch.cuda.empty_cache()`
before each step and `TC_PAIR_BUDGET_GB=11`. 404³ still OOMs (31 GiB int64 transient in
`_per_particle_topk`), so fp32 does not move the ceiling — it was never the engine. The old
"52.3M repeatable" was the caching allocator's reserved pool creeping step over step until
widebvh's fresh `cudaMalloc` failed; 52.3M itself dies on step 4. `empty_cache` costs ~2%, not
the 2.1× quoted below — that was P2P tiling from the default pair budget (+8 s/step above
~54M). **Replace 52.3M / 17.0 s / 52× with 65.45M / 18.75 s / 65× everywhere below**, and the
suggested sentence becomes: *NeMO evaluates the grand mobility for 65.4M particles — 65× as
many — on one H200 in 18.8 s.* The "engine reuses its tree" future-work line is no longer
needed; allocator hygiene suffices. The original analysis is kept below for provenance.

H-HIGNN's 28.8 s is for **1M particles on four GPUs**. The comparison as written is purely
about *speed at a fixed size*, which leaves the stronger point unstated: after the VRAM work
(`artifacts/vram_scaling_report.md`), **one H200 simulates 52,313,624 particles at φ=0.1 — 52×
the size H-HIGNN runs on four GPUs — evaluating the grand mobility in 17.0 s.**

Suggested framing for §3.4: the speedup is one axis, capacity is the other. Being able to put a
52M-particle suspension on a *single* card is the claim that a multi-GPU baseline at 1M cannot
answer, and it costs one sentence.

**Quote the repeatable capacity, not the single-shot one.** Measuring the warm apply turned
up a distinction that matters for how this is worded: the largest configuration that fits
*once* is not the largest one you can actually *simulate*.

| | N | mobility application | note |
|---|---|---|---|
| single cold `apply()` | 65,450,827 (403³) | 47.7 s (includes `torch.compile`) | fits once; **cannot do a second step** |
| **repeatable, steady state** | **52,313,624** (374³) | **17.0 s** | ← **the number for §3.4** |
| repeatable + `empty_cache()` per step | 64,000,000 (400³) | 35.5 s | +22% capacity for 2.1× the time |

Warm timings are extremely stable (52,313,624: min 16.971, max 16.982 over two independent
runs), and scale near-linearly: 0.275 s at 1M, 14.87 s at 46.7M, 16.98 s at 52.3M.

**Suggested §3.4 sentence:** H-HIGNN needs **four** GPUs for 1M particles at 28.8 s/step;
NeMO evaluates the grand mobility for **52.3M particles — 52× as many — on one H200 in 17.0
s**. Capacity and speed in the same clause, against a multi-GPU baseline at 1/52 the size.

**Why single-shot ≠ repeatable.** The widebvh engine `cudaMalloc`s its whole working set fresh
on every call (`reuse_tree=0`, `freeBuild()` at the end of `Treecode::apply`), while PyTorch's
caching allocator never returns freed blocks to the driver. Past ~52M the cache holds enough
that the engine's second-call allocation fails — the error surfaces as
`widebvh error (rc=1): std::bad_alloc: cudaErrorMemoryAllocation`, *not* a torch OOM. Calling
`torch.cuda.empty_cache()` each step buys the capacity back at 2.1× the step time. Making the
engine reuse its tree across steps would likely get both, and is worth a line in §4 future work.

**Two caveats.** (a) The 52.3M boundary is **stochastic near the edge** — 52,734,375 OOM'd on
one attempt and passed on another at 17.1 s. 52.3M passed twice; quote that, not 52.7M.
(b) Do not extrapolate the capacity in the text: scaling past ~100M particles hits a **silent**
int32 wrap in the edge-list prefix sum (`src/hashgrid_neighbors.py:160`), not an OOM. 52.3M
sits at 52% of that ceiling, so the claim as stated is safe.

### 2.9 §3.3.2, lines 688–697, and §3.2.3 line 328 — the consumer-GPU precision story (pp. 21, 24)

**This claim is reversed by the new measurements, not merely stale.** Lines 688–697
(`03-experiments.tex:386`) argue that the RTX 5090 stays close to the H200 *despite* the H200's
"about 21× higher" FP64 throughput, concluding that *"the measured runtime is governed less by
memory bandwidth or FP64 arithmetic alone and more by a mix of FP32/TF32 computation and
latency-sensitive stages"*. The 5090 column it now sits beside was measured with the fp64
far-field kernels **swapped out** (fp32 level 3), and the reason is exactly the one the paragraph
denies: on GeForce-class silicon (fp64 at 1/64 the fp32 rate) the fp64 far field *is* the
bottleneck. Measured, same engine, same parameters:

| card, N ≈ 1.05M two-drop | far field, fp64 | far field, fp32 L3 | step |
|---|---|---|---|
| RTX 4060 laptop | 8.05 s | **0.35 s** (23×) | 10.75 → 2.52 s |
| RTX A4500 | 6.14 s | **0.215 s** (28.6×) | 7.13 → 1.12 s |

(`artifacts/consumer_gpu_far_field_report.md`, `artifacts/fig13_mac_report.md:9-14`,
`artifacts/A4500_docker.md`.) The 5090 keeps up **because** its far field runs in fp32 and the
near field was TF32/FP32 already (line 328) — so the whole step is single-precision work, where
the 5090's nominal FP32 rate actually exceeds the H200's. That is the sentence the paragraph
should make. The re-measured Fig. 12 columns (§1.2, 2026-08-23) make it quantitative at every
size: with each card at its operating point the 5090 is a **flat 1.07–1.08× off the H200 from
50k to 2M**, its level-3 far field is *faster* than the H200's level-2 one from 100k up (101 vs
124 ms at 2M), and the entire remaining gap is the TF32 near field (441 vs 380 ms at 2M, 1.16×).

*Rewrite, same length.* Keep the consumer-GPU conclusion (cheaper, widely available, close to
the H200) and the bandwidth / FP32 / TF32 TFLOP/s numbers, which now *support* the argument
rather than puzzle it. Drop the "despite … FP64" mechanism and the "governed less by FP64"
sentence. State the operating point once: *the consumer-GPU results use the treecode's
single-precision kernels (fp32 M2P, P2P and upward pass); the H200 runs the double-precision
kernels throughout* — with the §0 caveat that the re-measured Fig. 12 H200 column is level 2,
so for that figure the H200 side reads "fp32 M2P/P2P, fp64 upward pass". The closing remark
about module-level task parallelism on the H200 can stay.

**Disclosure — three places, one clause each.** The paper nowhere says the far field has an
fp32 mode; grepping all of `nemo.tex` and `sections/*.tex` for fp32/FP64/single/double precision
finds only lines 328, 345 and 386.

1. **Fig. 12 caption** (`03-experiments.tex:380`): state **both** operating points — the H200
   column uses the level-2 kernels (fp32 M2P/P2P, fp64 upward pass), the RTX 5090 the level-3
   (fully single-precision) kernels; see §3.3.2 and the §0 update.
2. **§3.4 A4500 sentence** (`03-experiments.tex:440`): same clause; wording in §2.8.
3. **§3.2.3 line 328** — *"Velocity accumulation is performed in double precision"* is true
   on the H200 and **false under level 3**, where the M2P, P2P and upward pass accumulate in
   fp32. Qualify it: "… in double precision on the H200; on the consumer GPUs of §3.3.2 and §3.4
   the far-field treecode runs in single precision."

**And the accuracy cost, once, so the referee does not have to ask.** Level 3 changes the
far field's relative L2 error from 1.13e-6 to 1.14e-6 under gravity loading and leaves it at
3.09e-4 under random loading (two-drop, N = 1.05M, vs an exact fp64 sum; the interaction lists
are identical); over a 100-step sedimentation run the fp32 and fp64 trajectories differ by at
most **0.035 particle radii** with the production tree (0.045 with leaf 512, where the tree
itself differs; 4060 A/B, `artifacts/consumer_gpu_far_field_report.md`).
The relative errors sit ~200× below the near field's own 7.5% PRMSE, which is the same comparison §3.1
already makes for the truncation error. One sentence next to the disclosure in §3.3.2 covers
all three sites.

---

## 3. Additions worth making — ranked, all small

### 3.1 Symmetry — **recommended, ~1 paragraph + 4-row table**

The paper claims symmetry by construction (§2.3.1 "Encoding Symmetry", and §4 line 785) and
never quantifies how much the far field breaks it. It now can, and favourably: the treecode is
the **sole** source of asymmetry (the learned blocks are symmetric by construction and verified
so), the asymmetry **is** the truncation error scaled by exactly √2 — measured ratio 1.408–1.414
across every backend and every operating point — and it dropped 33× at 1M, from 1.53e-02 to
4.61e-04.

That last number is the point: **4e-4 asymmetry sits ~200× below the near field's own ~7.5%
PRMSE**, so the "symmetric by construction" claim is now true to within a rounding error of the
model's own accuracy, which it was not before. Data: `data/symmetry_widebvh.csv`, and
`data/fig10_far_field.csv` for the N-sweep.

**The N-sweep is drawn in the far-field figure (§1.0)**, so this section can point at it rather
than re-tabulate: `rel_asym` is flat across the sweep — 3.80e-04 at 200k to 4.61e-04 at 1M, a
1.2× spread over 50k…4M, an 80× range — so the treecode's contribution to asymmetry does not grow
with problem size. Worth stating explicitly that it holds **across the n-body pair-chunking
boundary** (1 chunk at 300k, 2 at 400k, 11 at 4M), which is an independent check on the
`_per_particle_topk` invariance fix.

State the method in one sentence wherever this lands — a referee will otherwise ask how a
6N × 6N symmetry norm was evaluated at N = 10⁶: `rel_asym = ‖M − Mᵀ‖_F / ‖M‖_F` is estimated by
Hutchinson probing with K = 24 Rademacher vectors, never assembling the operator, at a cost of K
mobility applications (details and validation in §1.0 above).

Best home: end of §3.1.2 (Mobility Operator), or as a short paragraph in §3.5 if you want to
keep §3.1 as-is. Add the caveat already established: this does **not** make the operator SPD —
the 33 negative eigenvalues come from the learned near field and are present with the tree,
without the tree, and at every `mac`. That is consistent with §4 line 785 and worth saying
explicitly, since it closes off an obvious referee question.

### 3.2 The near/far cutoff as a first-class design constraint — **1 sentence in §2.4.2**

Standard fast solvers have no notion of a near cutoff; NeMO's far field must cover exactly the
complement of the learned model's `r < 6` neighbour list, which required adding node-level and
pair-level exclusion to the engine (`widebvh_far_field_report.md` §3a). Worth one sentence
because it is the reason an off-the-shelf FMM cannot simply be dropped in, which strengthens
§2.4.2's whole argument.

The rejected alternative — sum everything, then subtract the near pairs — is worth a
**footnote**: it is the first thing a reader will propose, and it is wrong at any useful `mac`
(subtracting the exact near kernel leaves the multipole's error on an O(1/r) term while the
answer is O(1/60)).

### 3.3 §3.5 Discussion, lines 757–761 — modularity, now with a second data point

Currently: *"when we struggled to obtain good performance from existing long-range solvers on
GPUs, it was easier to swap out FMM for a bespoke GPU-resident treecode."*

The far-field solver has now been **replaced a second time**, with no change to any learned
component and no retraining, for 37× accuracy and 2.7× speed on the 1M benchmark. That is the
modularity claim demonstrated rather than asserted. *Extend the existing sentence, ~2 clauses —
no new paragraph.*

Figure 11(b) is now the picture of this, and it can be cited rather than described: across the
swap the learned segments are unchanged and only the far-field segment moves (2.5× shorter at
1M). Use the **`warp`-at-HEAD** rows of `data/fig11_breakdown_h200.csv` for that comparison,
not `warp-published` — see §1.1b for why the archived column confounds the far-field swap with
intervening near-field work.

`data/far_field_drift_1M.csv` is the same argument in its strongest form and is worth one
clause here: over 150 sedimentation steps the near field is invariant to the swap to **0.08%**
(172.43 vs 172.57 ms) while the far field goes from degrading 3.8× to degrading 16%. A
component was replaced under a *changing* input distribution and nothing else moved.

### 3.3a Robustness to distribution change — **new, 2–3 sentences, high value**

The paper argues for the far field on accuracy and speed at fixed φ. §2.7 adds a third axis it
never claims: **stability as the suspension deforms.** Warp's per-step cost nearly quadruples
over a 150-step drop; widebvh's rises 16% and its *total* step cost is flat to 2% because the
thinning near field cancels it. For a dynamics code that is arguably the more important
property — it is the difference between a run whose cost you can budget and one that does not
finish in the time you planned. This costs nothing to add: the numbers are already in §2.7 and
the table there is small enough to inline.

The consumer-GPU runs extend the same picture to two more cards: on the A4500 the far field is
19% of a 1.12 s step (§2.8) and on the 5090 it is *faster* than even the H200's level-2 far
field from 100k up (§1.2), while the near field stays the operator it was — so "the far field
is cheap and the near field is invariant to it" holds on an H200, an A4500 and a 5090, not on
one datacentre GPU.

### 3.4 §4, lines 799–812 — future work

The far field is still **translation-only**: the RT/TR/RR blocks are exactly zero beyond r = 6
in both backends, and the discarded far-field angular velocity exceeds the entire near-field
angular velocity (`artifacts/treecode_symmetry_report.md` §6). This gap is unchanged by the
present work and is now, by some distance, the **largest remaining far-field physics error** —
truncation is down at 4e-4 while this is a 5.8% effect. Worth one sentence in future work; it is
also a defensible answer to "why not push `mac` lower".

---

## 4. Measurements still needed

| # | measurement | blocks | cost |
|---|---|---|---|
| 1 | ~~Fig. 11 breakdown, widebvh, 5 sizes incl. 1M, H200~~ | — | **DONE** → `data/fig11_breakdown_h200.csv` (widebvh + warp at HEAD + warp-published), figure regenerated |
| 2 | ~~1M two-drop on RTX A4500, widebvh~~ | — | **DONE** → `artifacts/A4500_docker.md` (fp32 level 3, 50 steps; **1.12 s/step**, ~26× vs H-HIGNN); see §2.8. The per-step CSV `a4500_two_drop_fp32.csv` was not retrieved (#8) |
| 3 | ~~Fig. 12 RTX 5090 column, widebvh~~ | — | **DONE, re-measured 2026-08-23** → machine-written `data/fig12_scaling_5090.csv` (RunPod 5090, **50k–2M**, fp32 L3, one process per size; `artifacts/fig12_5090_runpod_report.md`; old-box rows archived in `artifacts/logs/fig12_5090_runpod/`); figure regenerated with both cards at all 10 sizes. See §1.2 |
| 4 | ~~far-field cost at t=0 vs t=100~~ | — | **DONE** → `data/far_field_drift_1M.csv` (150 steps × 2 backends, `benchmarks/far_field_drift.py`); see §2.7 |
| 5 | ~~max particles on one H200~~ | §3.4 H-HIGNN capacity claim | **DONE, revised 2026-08-22** → **65,450,827 sustainable** (fp32 L2, `empty_cache` + `TC_PAIR_BUDGET_GB=11`); `artifacts/max_particles_h200_f32l2_report.md`, §2.8 |
| 6 | ~~warm (steady-state) apply~~ | §3.4 step time | **DONE, revised** → **18.75 s at 65,450,827**; 14.6 s at 52.3M (fp32 L2) |
| 7 | ~~Fig. 12 H200 column at the current tree~~ | — | **DONE 2026-08-22** → `data/fig12_scaling_h200_f32l2.csv`, 50k–2M at fp32 L2 (+ fp64 at 1M/2M); §1.2. Still open: `fp32_level` column/flag in `figure12_grand_M.py` (`grand_M_perf.py` reads the f32l2 CSV and the figure was re-rendered 2026-08-23 with the final 5090 column) |
| 8 | retrieve the A4500 container CSV | provenance only | `docker cp`/volume read of `/persistent/results/a4500_two_drop_fp32.csv` (per-step rows, and the `fp32_level` header line that *proves* level 3) into `data/`. The `fig12_5090.csv` half is **moot** — superseded by the 2026-08-23 re-measurement (#3) |

**On #4 — it was not on disk, and running it changed an answer.**
`data/widebvh_perf_nemo_distros.csv` has exactly **two rows**, one per backend, each a 50-step
aggregate — no t split, and the 50-step window turns out to be shorter than the effect. The new
per-step run shows the far field drifting +16% (widebvh) against +278% (warp) over 150 steps,
which both rescues the sentence at lines 711–713 and raises the §3.4 speedup from 1.62× to
2.23×. Full write-up in §2.7.

**Everything hardware-blocked is done, #7 included.** #8 has shrunk to one file copy (the
A4500 per-step CSV). For the record: the 5090 column's level 3 is now evidenced on disk — the
archived driver script and report in `artifacts/logs/fig12_5090_runpod/` — while the A4500's
remains run-notes only until #8, and neither is stamped inside a CSV until the `fp32_level`
column exists.

Everything else needed is already on disk: `data/far_field_drift_1M.csv`,
`data/fig11_breakdown_h200.csv`, `data/fig12_scaling_h200.csv`,
`data/fig12_scaling_h200_f32l2.csv`, `data/fig12_scaling_5090.csv`,
`artifacts/fig12_5090_runpod_report.md`, `artifacts/5090_docker.md`,
`artifacts/A4500_docker.md`, `data/widebvh_mac_calibration.csv`,
`data/symmetry_widebvh.csv`, `data/widebvh_perf_nemo_distros.csv`.

---

## 5. Suggested order

1. ~~**Measurements 1–7**~~ — all **done** (#7 on 2026-08-22; #3 re-measured to 2M with a
   machine-written CSV on 2026-08-23). Only **#8** remains, now just the A4500 per-step CSV;
   it does not block writing.
2. **§2.4.2 + §2.4.3 rewrite** (items 2.1–2.4). This is one editing pass over two pages, and
   every other far-field text change depends on the parameter (`mac`) being defined there.
3. **Number swaps and rewrites**: 2.5, 2.6, 2.7, **2.8** (now a number, ~26×), **2.9** (the
   reversed FP64 argument — a rewrite, plus the three one-clause precision disclosures), plus
   **1.1a (i)–(iii)** — note (i) and (ii) are rewrites, not swaps.
4. **Figure regeneration** — **done 2026-08-23**: `python figures/grand_M_perf.py` re-rendered
   the two-GPU Fig. 12 with both columns final (all ten sizes paired, both cards to 2M). Still
   to do in the paper repo: copy `figures/gpu_scaling_h200_vs_5090.pdf` over the stale `_08_22`
   file (or add an `_08_23` name and repoint `03-experiments.tex:379`), and state both fp32
   levels in the caption (§2.9 disclosure 1).
5. **Additions** 3.1–3.4, in that order of value. All are ≤ 1 paragraph each.

Total new prose if all of §3 is taken: roughly one page, most of it replacing text that has to
be touched anyway. The mandatory part (§2 alone) is about half a page of replacement, no net
growth.
