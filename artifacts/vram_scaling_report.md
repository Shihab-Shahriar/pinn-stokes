# NeMO's VRAM footprint: what it was, what set it, and how far it now scales

**Result: the 1M two-drop went from 15.4 GiB to 4.07 GiB for +1% step time, and one
H200 now simulates 52,313,624 particles at phi=0.1 at 17.0 s per mobility application
(65,450,827 if you only need a single evaluation).**

Two things were wrong before this work. The repo could not *see* its own footprint —
every `peak_vram_gb` column and the `[MobFMM] peak GPU memory` print report PyTorch's
caching allocator, which is blind to the widebvh engine's raw `cudaMalloc` — and the
one path that dominated peak memory, the two-body NN pair evaluation, was the only
pair path that had never been chunked. Fixing the second required fixing the first,
because the instrument said the two-body and n-body operators had identical peaks
(`data/fig11_breakdown_h200.csv`, N=1M: 10.0349 vs 10.0348 GiB) when in fact one was
setting a floor the other could not cross.

All measurements on an NVIDIA H200 (139.8 GiB usable, sm_90), `git` working tree at
the time of writing, `torch 2.8.0+cu128`, `warp 1.12.0.dev0`.

---

## 1. Measuring it at all

`torch.cuda.max_memory_allocated/reserved` miss everything the treecode allocates
directly: the cuBQL BVH, the grid buckets, and the P2P pair list are `cudaMalloc` /
`thrust::device_vector`, outside the caching allocator. At N=1M that is 0.7-1.8 GiB
invisible.

Two methods were tried. **A `mem_get_info` free-memory delta does not work on a
shared GPU** — when another tenant frees memory mid-run the delta reads low, and it
produced three impossible results (a "total" *below* the same process's own
`reserved`) before the method was changed. **Per-PID accounting via
`nvidia-smi --query-compute-apps` is the reliable source** and is what every number
below uses.

This is now in-tree: `_process_device_memory_mb` (`src/treecode.py`), gated behind
`NEMO_DEVICE_MEM=1` and off by default, because it forks `nvidia-smi` once per step —
~200 ms, enough to dominate a 280 ms step (the 50-step two-drop reports 25.2 s with it
on, 14.2 s off). `benchmarks/figure11_breakdown.py` parses the extra field into a
`peak_total_gb` column; the existing `peak_alloc_mb` regex is unchanged, so old logs
still parse.

| N=1,047,968, two-drop | before | after |
|---|---|---|
| torch allocated | 10.51 GiB | **2.51 GiB** |
| torch reserved | 13.63 GiB | **3.40 GiB** |
| **process total** | **15.41 GiB** | **4.07 GiB** |
| 50-step wall | 14.0 s | 14.2-14.5 s |

---

## 2. What set the peak: the unchunked two-body path

`ac78eb4` chunked the n-body correction at `pair_chunk_size = 8M`. But
`NNMobTorch.apply` evaluated the **entire** ~21.2M-pair edge list in a single call
(`src/gpu_mob_2b.py`), and every intermediate in that call is sized `(num_pairs, ...)`:

| tensor | bytes/pair | at 21.2M pairs |
|---|---|---|
| MLP hidden activation `(P,64)` fp32 | 256 | 5.4 GB each |
| `K_s`, `K_t` `(P,6,6)` fp32 | 288 | 6.1 GB |
| `M_s/mu`, `M_t/mu` (fresh copies) | 288 | 6.1 GB |

Inductor fuses much of this, and the surviving live set was 10.34 GiB. That is a
floor: with it in place, `pair_chunk_size` had nothing to bind on. Dropping it 8M ->
2M alone moved total VRAM by **0.04 GiB**, which is why the knob looked inert.

The fix is `NNMobTorch._two_body_velocity`, mirroring
`Mob_Nbody_Torch.get_nbody_velocity`. Both invariants documented in `CLAUDE.md` apply
and are honoured: the loop stays in Python outside the compiled region (its bound is
the pair count, which drifts every timestep — inside `torch.compile` that is a
recompile per step until `recompile_limit` trips and dynamo falls back to eager
permanently, ~2.4x slower with no error), and `mark_dynamic` on the pair dimension
keeps the short trailing chunk on the same graph. `rel` is built per chunk so the
`(num_pairs, 3)` gather never materialises at full size.

Verified: **zero recompiles** across a full 50-step run under `TORCH_LOGS=recompiles`.

### The two chunk sizes are not independent

Neither knob does much alone; the two-body chunk unblocks the n-body one. Swept at
N=1.05M / 21.2M pairs, one configuration per process:

**Two-body chunk** (`pair_chunk_size` fixed at 2M) — overhead is ~0.17 ms per chunk,
and peak saturates once it drops below the n-body chunk's floor:

| tb chunk | chunks | self+2body | peak allocated |
|---|---|---|---|
| off | 1 | 27.95 ms | 10.34 GiB |
| 8M | 3 | 28.25 ms | 4.26 GiB |
| **4M (default)** | **6** | **29.16 ms** | **2.51 GiB** |
| 2M | 11 | 29.65 ms | 2.51 GiB |
| 1M | 22 | 31.62 ms | 2.51 GiB |
| 500k | 43 | 35.04 ms | 2.51 GiB |

**N-body chunk** (`two_body_chunk_size` fixed at 4M):

| nb chunk | n-body | peak allocated |
|---|---|---|
| 8M (old default) | 154.26 ms | 8.37 GiB |
| 4M | 154.88 ms | 4.46 GiB |
| **2M (default)** | **156.00 ms** | **2.51 GiB** |
| 1M | 158.24 ms | 2.40 GiB |

`DEFAULT_TWO_BODY_CHUNK = 4M` sits exactly at the knee — everything below pays
per-chunk overhead for memory the n-body chunk already holds. **Keep it at or below
~5M**, or the two-body path becomes the binding term again (8M does). `pair_chunk_size
= 2M` is where its own trade stops paying: 5.86 GiB for 1.7 ms, against 0.11 GiB for
another 2.2 ms below it.

Total cost of both: ~2.9 ms of a 288 ms step, **~+1%**.

---

## 3. `TC_PAIR_BUDGET_GB` was misunderstood

`docker/README.md` previously stated that `TC_PAIR_BUDGET_GB=6` was "not optional" on a
20 GB card and predicted an OOM without it. Both halves were wrong, and the reasoning
is easy to repeat, so it is recorded here.

The 12 GB default was never eagerly allocated. The engine takes

```cpp
size_t pairCapacity = std::min(nTarget_ * (size_t)AVG_NEAR_LEAVES, pairCap_);
```

(`widebvh/src/treecode.cuh:3965`, `AVG_NEAR_LEAVES = 128`) as two
`thrust::device_vector<int>`. At N=1.05M the first term is 134M pairs = 1.07 GB and
**binds**; `pairCap_` at a 6 GB budget is 161M pairs, so `=6` changes nothing.
Measured saving: **0.14 GiB**. `=1` puts `pairCap_` at 26.8M — below the 134M term —
and saves **0.88 GiB at no time cost**. The budget only starts to bite below ~5 GB,
and the actual emitted pair list is only 7.3M at mac 0.8
(`data/widebvh_mac_calibration.csv`), so `AVG_NEAR_LEAVES` over-provisions ~18x.

The predicted OOM also rested on the undercounting instrument: the quoted 11.05 GB
allocated / 13.94 GB reserved omitted the engine, and the true 15.4 GiB would in fact
have fitted in 20 GB.

`WidebvhFMM` now sizes the budget from the device — `_default_pair_budget_gb()`,
~6% of total VRAM clamped to [1, 12] GB, so 1.2 GB on a 20 GB card and 8.4 GB on an
H200. It is resolved once and cached on first use, not at import, because querying the
device would otherwise initialise a CUDA context as a side effect of importing the
module. `TC_PAIR_BUDGET_GB` still overrides.

---

## 4. Correctness of the chunking

Chunking partitions the **pair list**, not space and not particles. Neighbour
identification completes in the hash grid before any chunk exists, and every chunk
receives the full `positions` and `force` arrays — only the index list is sliced. So
`sum(all pairs) = sum(chunk 1) + sum(chunk 2) + ...` exactly, and the residual is
floating-point summation order.

| test | result |
|---|---|
| new defaults vs old behaviour, N=1M | relative L2 **2.2e-09** |
| net velocity, transverse/axial ratio | **bit-identical** |
| full stack vs MFS, N=300, unchunked | 13.088996 % |
| ... chunked at production defaults | 13.088995 % |
| ... chunked at 17/13 pairs (hundreds of boundaries) | 13.089004 % |
| recompiles over 50 steps | **0** |

The residual *grows* as chunks shrink (2.2e-09 at 2M-pair chunks, 5.0e-06 at 13-pair
chunks) and stays at the 1e-9..1e-6 level — the signature of summation order, not of
lost data.

**The test has power.** Deliberately dropping one chunk of 13 pairs out of ~4600 —
0.3% of the interactions — produces **1.25e-02** relative error, 2500x the residual
that correct chunking reports. A dropped-pair bug is not something these numbers could
hide.

The failure mode that *is* real for chunking lives in the n-body path: its correction
needs the K-nearest-neighbour table of **both** endpoints of each pair, so a per-chunk
table silently truncates and breaks symmetry (it cost 1.75e-2 at N=1M once already).
`_per_particle_topk` is built once over the complete edge list and passed into the
loop; that is what makes shrinking `pair_chunk_size` safe. The two-body path has no
such table — a pair's contribution depends only on its own two particles.

---

## 5. How far it scales: 52.3M simulated, 65.5M in a single shot

Uniform suspension at **phi = 0.100000**, cubic lattice, spacing
`a = ((4/3)pi / phi)^(1/3) = 3.472931`, +/-5% jitter, one `apply()` through the
production stack (`WidebvhFMM` + `Mob_Nbody_Torch`). Lattice built directly on the GPU
(0.05 s at 65M; a float64 `np.meshgrid` would cost several GB of host RAM). One N per
process; peak taken over the whole process **including** `torch.compile` autotune
scratch, since a fresh process must survive its first call.

| n_side | N | pairs | torch alloc | process total | alloc B/particle | apply |
|---|---|---|---|---|---|---|
| 100 | 1,000,000 | 21.2M | 2.69 GiB | 3.57 GiB | 2889 | — |
| 170 | 4,913,000 | 105M | 10.28 GiB | 12.29 GiB | 2247 | 14.5 s |
| 250 | 15,625,000 | 335M | 32.72 GiB | 37.67 GiB | 2248 | 19.1 s |
| 320 | 32,768,000 | 703M | 68.64 GiB | 78.35 GiB | 2249 | 23.8 s |
| 348 | 42,144,192 | 905M | 88.29 GiB | 90.02 GiB | 2249 | 31.5 s |
| 400 | 64,000,000 | 1.375e9 | 134.09 GiB | 136.34 GiB | 2250 | 47.2 s |
| 402 | 64,964,808 | 1.396e9 | 136.11 GiB | 138.38 GiB | 2250 | 47.6 s |
| **403** | **65,450,827** | **1.406e9** | **137.13 GiB** | **139.42 GiB** | **2250** | **47.4 s** |
| 404 | 65,939,264 | — | — | **OOM** | — | — |

The 403/404 boundary was reproduced twice each. At 403 the run holds **139.4 of the
139.8 GiB card — 99.7%**; `expandable_segments:True` (already set in
`benchmarks/two_suspensions_1M.py`) is what makes utilisation that high.

### The scaling law is on *allocated*, not on the process total

**`torch.cuda.max_memory_allocated` is 2249 bytes/particle, flat to 0.1% from 5M to
65M.** That is the real invariant. Two other terms sit on top of it and behave
differently:

| term | behaviour |
|---|---|
| allocated | **2249 B/particle**, exact above ~5M (2889 at 1M — fixed overhead still visible) |
| non-torch (engine + CUDA context) | ~0.85 GiB + ~24 B/particle (0.88 GiB at 1M, 2.30 GiB at 65M) |
| reserved − allocated | **opportunistic**: 1.2-9.1 GiB when the card is roomy, 0.09-0.11 GiB under pressure |

That last row is the trap. The mid-range rows above were measured on a *shared* node
with ~90 GiB free, where the caching allocator over-reserved by up to 9.1 GiB; the
near-ceiling rows could not, so their reserved collapses to within 0.1 GiB of
allocated. **The slack is not a requirement** — reading a per-particle cost off a
process total measured with headroom will overstate the footprint by up to 12%.

Under memory pressure — the regime that decides the limit — the total is
`allocated + non-torch`, i.e. **~2273 B/particle plus ~0.85 GiB**:

```
VRAM_GiB  ~=  0.85 + 2.117 * N/1e6          N_max  ~=  0.47e6 * (VRAM_GiB - 0.85)
```

Checks against the measured edge: 139.8 GiB -> 65.6M (measured 65.45M pass, 65.94M
OOM). Applied to a 20 GB A4500 at ~18.7 GiB usable it gives **~8.4M particles**, and it
reproduces the 1M two-drop's 4.07 GiB to within the fixed term.

### Single-shot capacity is not simulation capacity

The 65,450,827 above is the largest configuration that completes **one** `apply()`. It cannot
do a second one. Measuring warm applies (same positions, so the compiled graph is reused):

| | N | mobility application |
|---|---|---|
| single cold `apply()` | 65,450,827 | 47.7 s (includes `torch.compile`) |
| **repeatable, steady state** | **52,313,624** | **17.0 s** |
| repeatable + `torch.cuda.empty_cache()` per step | 64,000,000 | 35.5 s |

Warm times are near-linear in N and very stable — 0.275 s at 1M, 14.87 s at 46.7M, 16.34 s at
50.7M, 16.98 s at 52.3M, with a spread of ~10 ms across repeats. (The mild superlinearity is
consistent with the hash grid's hardcoded `grid_dim=(128,128,128)`: the cell hash is a modulo,
so cells alias as the box grows. Both kernels re-test the radius, so the answer stays correct —
it just costs rejected candidates.)

**The failure above ~52M is not a torch OOM.** It is
`widebvh error (rc=1): std::bad_alloc: cudaErrorMemoryAllocation`. The engine `cudaMalloc`s its
entire working set fresh on every call (`reuse_tree=0`, `freeBuild()` at the end of
`Treecode::apply`), while PyTorch's caching allocator never hands freed blocks back to the
driver. At 52.3M torch holds 131.0 GiB reserved of 139.8, leaving ~8.8 GiB — enough. Past that
it isn't, and the engine, not torch, is the one that fails.

`empty_cache()` before each step buys the capacity back (64M works) at 2.1× the step time,
because torch must then re-acquire ~135 GiB of blocks from the driver every step. Teaching the
engine to reuse its tree across steps would plausibly get both.

The boundary is **stochastic right at the edge**: 52,734,375 OOM'd on one attempt and passed on
another (17.1 s). 52,313,624 passed twice. Quote the latter.

### Sanity checks at every size

Velocities finite; transverse/axial ratio of the net velocity 6.8e-09 at the largest N
(uniform gravity should produce no net transverse motion); **21.48 pairs per
particle** throughout; max **26** neighbours per particle, against the 128 of
`MAX_PARTICLE_NEIGHBORS` and the 255 of the `uint8` `no_of_nn` — both have ~5-10x
headroom and neither is an N limit, only a density limit.

> **The lattice sits on a knife edge.** In lattice units the 6.0 cutoff is at 1.7276.
> Shells: (100) at 1.000 (6 neighbours), (110) at 1.414 (12), and (111) at **1.7320 —
> 0.26% outside**. An unjittered lattice therefore gives exactly 18 neighbours and
> excludes the body diagonals by a hair; the +/-5% jitter flips a good fraction back in,
> giving the measured 21.48. A 0.3% change in `a` moves the pair count by ~40%, so
> re-measure rather than assume if the lattice constant is ever changed.

---

## 6. Where the bytes go, and the next wall

At 2249 B/particle allocated the dominant terms are per-particle tables and per-pair transients
in `_per_particle_topk`, not the (now chunked) network evaluation:

| item | bytes/particle |
|---|---|
| `neighbors` (N,129) int32 | 516 |
| `neighbor_dists` (N,129) fp32 | 516 |
| int64 chain (`arange`, `prefix[t_idx]`, `local_pos`, `where`) | ~500-660 |
| `pos[s_idx]`, `pos[t_idx]` (P,3) fp32 transients | ~494 |
| edge list, `edge_dist`, top-K outputs, velocity buffers | ~410 |
| widebvh engine (pair list, buckets, fp64 geometry) — *not torch* | ~140 |

These sum to ~2650, above the measured 2249: the list is a per-term budget, not a
simultaneously-live set. The transients are freed and their blocks reused within the
call, and Inductor fuses several away. The last row is outside PyTorch's allocator
entirely and belongs to the non-torch term, not to the 2249.

**The next wall after memory is silent, and it is close.** At 65.45M the run carries
1.406e9 pairs — **65% of the int32 ceiling of 2,147,483,647**. That ceiling lives in

- `torch.cumsum(nn_count_uint8, dim=0, dtype=torch.int32)`
  (`src/hashgrid_neighbors.py:160`), the prefix-offset table for the whole edge list,
  and
- Warp's `array_t.shape`, which is `ctypes.c_int32` — `wp.from_torch` on the edge
  buffer writes `num_pairs` into it and **wraps without raising**.

At 21.48 pairs/particle that is **N ~= 100M**. Past it the prefix goes negative and
`fill_edge_indexes_kernel` writes at negative offsets, with Warp's bounds assert
compiled out in release: an out-of-bounds device write, no error.

Today memory protects you with ~1.5x margin. **Anyone who buys memory to go bigger
must change that `cumsum` to int64 first.** The obvious lever is
`MAX_PARTICLE_NEIGHBORS` — its two `(N,129)` tables are 1032 of the 2249 bytes/particle,
so cutting it to 32 would free ~768 B/particle and reach ~90M, which lands *inside* the
corruption zone. Note also that the fill order in `_per_particle_topk` is edge-arrival
order, not sorted, so truncating that table can drop genuine near neighbours — it is
not a free knob.

Two other ceilings, both further out: `cub::DeviceRadixSort::SortPairs(..., (int)n, ...)`
in widebvh's `grid_buckets.cu` casts the **particle** count to `int` (N < 2.147e9), and
widebvh's own pair budget triggers its tiled P2P path (correct, just slower, and it
announces itself on **stderr**) once near-pairs exceed `pairCap_`.

---

## 7. Known inefficiency, not yet fixed

`PairVelKernel.forward` allocates a full `(N,6)` `zeros_like(force)` per chunk and
`_two_body_velocity` accumulates it with `+=`. That is one N-sized buffer per chunk:
1.46 GiB and ~3 buffer traversals each at N=65M with 352 chunks. `get_nbody_velocity`
does it correctly, with a single accumulator and `index_add_`. Making the two-body path
match would remove ~1 GiB of live memory and roughly 1.5 TB of memory traffic at the
largest sizes — worth perhaps 1% on both axes, which is why it was left alone rather
than changed under a measurement campaign.

---

## 8. Reproducing

```bash
source ~/warp_env.sh

# 1M two-drop, with the true footprint in the log
NEMO_DEVICE_MEM=1 python benchmarks/two_suspensions_1M.py     # 4168 MB process total
python benchmarks/two_suspensions_1M.py                       # 14.2 s / 50 steps, no probe overhead

# accuracy: chunking must not change the answer
TORCH_COMPILE_DISABLE=1 python benchmarks/accuracy_grand_M.py

# no silent eager fallback
TORCH_LOGS=recompiles python benchmarks/two_suspensions_1M.py 2>&1 | grep -c Recompiling   # 0
```

Measure **one configuration per process** — peak memory, like timing, does not survive
a sweep inside a single process. Use per-PID accounting, not a `mem_get_info` delta, on
any shared card.
