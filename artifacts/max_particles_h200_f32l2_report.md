# Maximum particle count on one H200, widebvh far field at fp32 level 2

Measured 2026-08-22 on a RunPod H200 (143,771 MiB = 140.4 GiB device, 139.80 GiB
visible to torch; exclusive), `sskhan39/nemo:2.0` image (pinn-stokes @ ac78eb4-dirty,
widebvh @ 03efcdb, torch 2.8.0+cu128). Script: `benchmarks/max_particles.py`, which
reproduces the protocol of `artifacts/vram_scaling_report.md` section 5 (phi = 0.1
cubic lattice, a = 3.472931, +/-5% jitter built on the GPU, uniform gravity, the
production `WidebvhFMM` + `Mob_Nbody_Torch` stack, `expandable_segments:True`, one
N per process, cold apply then warm applies on the same positions). Far field
`NEMO_FAR_FP32_LEVEL=2`, pdeg 7, mac 0.8, leaf 1024. Raw logs in
`artifacts/logs/max_particles_h200_f32l2/`, probe table in
`data/max_particles_h200_f32l2.csv`.

## Headline

| | fp64 (previous report) | fp32 L2 (this run) |
|---|---|---|
| single `apply()` completes | 65,450,827 (403^3) | **65,450,827 (403^3)** -- 404^3 OOMs, same as before |
| sustainable steady state, production defaults | 52,313,624 for ~3 steps | 52,313,624 for **3 steps, dies on the 4th** |
| sustainable steady state, `empty_cache()` per step + `TC_PAIR_BUDGET_GB=11` | 64,000,000 at 35.5 s | **65,450,827 at 18.75 s/step** (3/3), 64,964,808 at 18.56 s, 64,000,000 at 18.32 s |
| sustainable, `set_per_process_memory_fraction(0.92)`, no `empty_cache` | -- | 52,313,624 at 14.6 s/step, 8/8 |

**fp32 in the far field does not move the particle ceiling at all.** The ceiling is
PyTorch's near-field working set (2249 B/particle allocated, flat from 5M to 65M) and,
at the very top, one 31.4 GiB int64 index transient in `_per_particle_topk`
(`src/gpu_nbody_mob.py:181`, the advanced-indexing scatter over 1.4e9 pairs) -- the
treecode's own footprint is a few GB either way. What fp32 changes is the step time:
52.3M went from 17.0 s (fp64) to 14.35 s, and the far field at 65.45M is 5.0 s.

What *did* move the usable number is understanding why the old "repeatable" limit
was 52M when a single shot managed 65M. The answer is the caching allocator, not
the engine, and it can be fixed from outside.

## Probes (one process each, fp32 L2)

| N | n_side | settings | cold | warm steps | step s | torch alloc / reserved GiB | process peak GB | outcome |
|---:|---:|---|---|---|---:|---|---:|---|
| 52,313,624 | 374 | defaults | ok | 3/3 | 14.35 | 110.8 / 131.0 | 138.3 | ok (matches old "repeatable") |
| 52,313,624 | 374 | defaults, 6 warm | ok | **3/6** | 14.35 | 110.8 / 134.2 | 142.0 | widebvh bad_alloc on step 4 |
| 54,872,000 | 380 | defaults (tiled) | ok | 0/4 | -- | 115.0 / 129.1 | 142.6 | bad_alloc on step 1 |
| 57,066,625 | 385 | defaults (tiled) | ok | 0/4 | -- | 119.6 / 134.0 | 139.3 | bad_alloc on step 1 |
| 59,319,000 | 390 | defaults (tiled) | ok | 2/3 | 24.2 | 125.6 / 128.6 | 142.8 | bad_alloc on step 3 |
| 59,319,000 | 390 | `TC_PAIR_BUDGET_GB=10` | ok | 2/4 | 16.6 | 125.6 / 131.7 | 140.4 | bad_alloc on step 3 |
| 65,450,827 | 403 | defaults (tiled) | ok | 0/3 | -- | 137.1 / 137.2 | 142.8 | bad_alloc on step 1 |
| 65,939,264 | 404 | single shot | **fail** | -- | -- | 121.8 / 123.7 | 127.7 | torch OOM, 31.44 GiB in `_per_particle_topk` |
| 66,923,416 | 406 | single shot | **fail** | -- | -- | 123.6 / 125.3 | 138.8 | torch OOM, 31.91 GiB, same site |
| 52,313,624 | 374 | `mem_fraction=0.92` | ok | **8/8** | 14.6 | 110.8 / 118.1 | 135.9 | ok, reserved held at 113-121 GB |
| 52,313,624 | 374 | `empty_cache` | ok | 4/4 | 14.67 | 109.6 / 122.8 | 127.7 | ok |
| 64,000,000 | 400 | `empty_cache`, budget 10 | ok | 3/3 | 18.32 | 134.1 / 134.2 | 139.6 | ok |
| 64,964,808 | 402 | `empty_cache`, budget 11 | ok | 3/3 | 18.56 | 136.1 / 136.2 | 141.7 | ok |
| **65,450,827** | **403** | **`empty_cache`, budget 11** | ok | **3/3** | **18.75** | 137.1 / 137.2 | **142.8** | **ok -- the single-shot max, sustained** |
| 65,450,827 | 403 | `empty_cache`, default budget | ok | 3/3 | 27.1 | 137.1 / 137.2 | 142.8 | ok but tiled P2P: +8.3 s/step |

Every completed apply had finite velocities and a transverse/axial net-velocity ratio
of 1e-8..1e-9 (uniform gravity -> no net transverse motion). 21.48 pairs/particle
throughout (1.406e9 pairs at 403^3, the same 65% of the int32 pair ceiling the previous
report flagged).

## What sets the "repeatable" limit, and the two fixes

With identical inputs every step, `torch.cuda.max_memory_reserved` still grows step
over step -- at 52.3M: 125.8 -> 127.0 -> 128.2 -> 134.2 GB over four applies -- while
allocated stays at 110.8 GiB. The widebvh engine `cudaMalloc`s its whole working set
fresh each call (`reuse_tree=0`) and needs ~5-11 GB of *driver-level* free memory; the
caching allocator never returns blocks to the driver, so once its hoard crosses
~134 GB the engine's allocation fails with `std::bad_alloc` and the run dies, on
whichever step that happens to be. That is why the old figure was 52M "for two
repeats" and why 380 and 385 died on step 1 while 390 survived two: the edge is
stochastic in the allocator's fragmentation, not a function of N.

Two fixes, both outside the operators:

1. **Cap the allocator: `torch.cuda.set_per_process_memory_fraction(f, 0)`.** When
   the cap binds, torch frees cached blocks and retries instead of growing. At 52.3M
   with f = 0.92 (128.6 GB) reserved settled at 113-121 GB and eight warm steps ran
   at 14.6 s (vs 14.35 uncapped) -- a 2% cost for a run that no longer dies. Needs
   `f * 139.8 GiB > allocated + a few GB`, so it cannot help at 403^3 (allocated
   137.1 GiB leaves no room for an 11 GB engine budget); it is the mid-range tool.
2. **`torch.cuda.empty_cache()` before each apply.** Reserved is then pinned at
   ~allocated (137.1 / 137.2 GiB at 403^3) and the engine always finds its room.
   Cost at 52.3M is 14.67 vs 14.35 s (+2%), at 64M the step is 18.32 s of which
   ~0.25 s is re-acquiring blocks. **This is not the 2.1x the previous report
   measured.** That 35.5 s at 64M was with the default pair budget, where the
   engine tiles P2P over two target tiles; the tiling, not the `empty_cache`,
   was the cost -- see the last two rows above (27.1 vs 18.75 s at 403^3 with the
   only difference being `TC_PAIR_BUDGET_GB`).

## The pair budget is a speed cliff at ~52M

`_default_pair_budget_gb()` gives the H200 8.4 GB = 225,164,820 pairs per tile. The
near-pair count at mac 0.8 is ~4.15 per particle, so the engine starts tiling at
**N ~ 54M** (225M pairs; 52.3M carries 219.6M and is the last un-tiled cubic size).
Tiled P2P costs 8-9 s per step at these sizes (far field 12.2 s vs 4.6 s at 59.3M,
13.3 s vs 5.0 s at 65.45M). Set `TC_PAIR_BUDGET_GB` to `40 * 4.15 * N / 1e9`
rounded up -- 10 for 60M, 11 for 65.45M -- and the far field stays at ~76 ms per
million particles all the way up. The budget is engine memory, so it trades against
the allocator: at 403^3 the 11 GB budget plus 137.1 GiB of torch leaves 0.9 GB on the
card, which is why `empty_cache` is mandatory there.

## Steady-state step time at the top

At 65,450,827 particles, fp32 L2, budget 11, `empty_cache`: **18.75 s per mobility
application** = 5.0 s far field + 13.5 s near field (two-body + n-body NN, 1.406e9
pairs) + 0.25 s allocator. Linear from 52.3M (14.4-14.6 s) through 64M (18.32 s);
the previous superlinearity was the tiling.

## Recommended operating point for a max-size H200 run

```
NEMO_FAR_FP32_LEVEL=2  TC_PAIR_BUDGET_GB=11  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N <= 65,450,827 (403^3 at phi=0.1); torch.cuda.empty_cache() before every apply()
```

For anything up to ~55M the allocator cap (`set_per_process_memory_fraction(0.92, 0)`)
is the cheaper fix and no `empty_cache` is needed. A dynamics run changes the pair
count every step, which makes fragmentation worse, not better -- `empty_cache` per
step is the safe default for anything above ~50M. The next wall is unchanged from the
previous report: the int32 pair ceiling at N ~ 100M.
