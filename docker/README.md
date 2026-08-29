# NeMO in a container — runbook

One image, built once, that runs the two measurements still outstanding in
`artifacts/paper_update_plan.md` §4 on GPUs this cluster does not have:

| # | measurement | GPU | entry point |
|---|---|---|---|
| 3 | Figure 12, end-to-end scaling | RTX 5090 (sm_120) | `benchmarks/figure12_grand_M.py` |
| 2 | Figure 13 / §3.4, 1M two-drop sedimentation | RTX A4500 (sm_86) | `benchmarks/two_suspensions_1M.py` |

Scope is those two, widebvh only. Baked data is the five Figure-12 configurations
(85.8 MB) plus the models — nothing else.

Steps 1–8 below are in order. **Do not skip step 6**: it is the gate that proves
the image reproduces known numbers before you spend time on hardware you cannot
re-check.

---

## Step 0 — what you need, and where

| | |
|---|---|
| **On the cluster** (here) | nothing extra; `source ~/warp_env.sh` |
| **On a build machine** | Docker + sudo, ~40 GB free disk, network. No GPU needed to build — but see the RAM note in step 3: the default parallelism assumes a machine with roughly 1 GB of RAM per core, and OOMs the host otherwise. |
| **A Docker Hub account** | you will `docker login` in step 5 |
| **On the 5090 / A4500 hosts** | Docker + NVIDIA Container Toolkit (`--gpus all` must work), driver **≥ 570** for the 5090, ≥ 525 elsewhere. Check with `nvidia-smi` first. |

Pick your image name once and reuse it:

```bash
export DOCKERHUB_USER=<your-dockerhub-username>
export IMG=$DOCKERHUB_USER/nemo:2.0
```

---

## Step 1 — pack the build context (on the cluster)

```bash
cd /mnt/ffs24/home/khanmd/throwaway/pinn-stokes
source ~/warp_env.sh
bash docker/pack_context.sh
```

**Expect:** `wrote .../nemo-ctx.tar.gz (42M)`, with `widebvh` at ~2.7 MB and
`pinn-stokes` at ~84 MB in the size listing.

The script refuses to run if any critical untracked file is missing (the widebvh
C ABI, the `.wt` models, the five configs), so a clean exit means the context is
complete. It also aborts if the widebvh copy exceeds 50 MB, which means the
whitelist picked up a build or data directory.

> Add `--with-warp-fork` **only** if you also want the old treecode for an A/B.
> It is not needed for either measurement — see "Warp" at the bottom.

## Step 2 — copy it to the build machine

```bash
scp nemo-ctx.tar.gz you@docker-box:
```

## Step 3 — build the image

```bash
# on the build machine
mkdir nemo-ctx && tar -C nemo-ctx -xzf nemo-ctx.tar.gz && cd nemo-ctx
cat MANIFEST.txt                        # records which working trees went in

export DOCKERHUB_USER=<your-dockerhub-username>
export IMG=$DOCKERHUB_USER/nemo:2.0

# set NINJA_JOBS/NVCC_THREADS to suit the machine -- see below, this is not
# optional on a laptop-class host
docker build \
  --build-arg NEMO_GIT_SHA="$(cat GIT_SHA)" \
  --build-arg CUDA_ARCHS="86-real;89-real;90-real;120-real;120-virtual" \
  --build-arg NVCC_THREADS=1 \
  --build-arg NINJA_JOBS=4 \
  --build-arg WIDEBVH_FP32_LEVELS="1;2;3" \
  -t "$IMG" . 2>&1 | tee build.log
```

`WIDEBVH_FP32_LEVELS="1;2;3"` is what makes this the 2.0 image: it adds the
fp32 far-field libraries (`libwidebvh_nemo_f32l{1,2,3}.so`) next to the fp64
production one, which stays the default -- see step 7c for why and how to select
them. 1.0 was the same build without that argument (fp64 only).

**Expect:** ending with the `ls -la` of `libwidebvh_nemo.so`,
`libwidebvh_nemo_cart.so`, `libwidebvh_nemo_f32l{1,2,3}.so` and
`libcuBQL_cuda_float3.so`. Measured for the 2.0 build on a 24-core / 15 GB
laptop with exactly the arguments above: 22.5 min for the widebvh step (five
libraries x five architectures), ~5 min for the python layers, image 17.4 GB. Result is ~17 GB on
disk (~5–6 GB pushed). Keep the log — if the build dies, the failing step number
is the whole diagnosis, and re-running without it wastes the cached layers.
Note `| tee` makes `$?` tee's status, not the build's; check the log tail.

BuildKit cancels the build if the `docker build` *client* goes away, so for a
20-minute run on a machine you might also be working on, detach it:

```bash
setsid nohup docker build ... -t "$IMG" . > build.log 2>&1 < /dev/null &
tail -f build.log
```

### Sizing the build to the machine

**The Dockerfile default is `ninja -j$(nproc)` with `nvcc -t4`, and the two
multiply.** Ninja launches one `nvcc` per ready TU — six heavy ones
(`nemo_capi.cu`, `grid_buckets.cu`, `common.cu` × the `widebvh_nemo` and
`widebvh_nemo_cart` targets) — and each forks four `cicc` processes. On
`nemo_capi.cu` at `PDEG=7` those are multi-GB apiece. On a 24-core / 15 GB
laptop the default is ~24 concurrent `cicc` and reliably OOMs the **host**,
taking the buildkit session with it — which is why the command above sets them.
Only omit them on a machine with roughly 1 GB of RAM per core to spare.

- `NINJA_JOBS` × `NVCC_THREADS` is the knob that matters — it bounds the peak
  `cicc` count. Empty `NINJA_JOBS` means `$(nproc)`, the historical default.
- `CUDA_ARCHS` is the other cost driver. nvcc compiles its `-gencode` targets
  *sequentially*, so the list length is close to a linear multiplier — a
  single-architecture build of the three widebvh TUs measured 6m26s at `-j16`.
  The five above cover every GPU this runbook touches: sm_86 (A4500, step 7b),
  sm_90 (H200, step 6), sm_120 (5090, step 7a), plus `120-virtual` PTX for a
  future architecture. **Add the sm of your build machine too** if you intend to
  run step 4 there — sm_120 PTX cannot JIT down to an older card, so a smoke test
  on, say, an Ada laptop needs `89-real` in the list. Dropping the Dockerfile
  default's `75-real;80-real` costs nothing here; keep them only if the image
  needs to run on Turing or A100.

**Measured**, 24-core / 15 GB host, five architectures at `-j3`/`-t1`: the
widebvh layer took **710s (~12 min)**, total build ~20 min, no memory pressure.
The old "~25–35 min" figure assumed the 7-architecture list on a large machine.

## Step 4 — smoke test

```bash
docker run --rm --gpus all "$IMG"
```

**Expect** three stages and `all stages passed`:

```
=== 1/3  widebvh C ABI + GPU (no torch) ===
    abi=4  policy=0 (0=bary)  pdeg=7  max_order=7  RPY_A=1.0  fp32_level=0
    wbnemo_smoke(...) -> checksum -2.450400935e+07   (H200 reference -2.450400935e+07, PDEG 7 / fp32_level 0)
=== 2/3  Warp ===
    warp 1.12.0   FMM-patched build: False        <- expected, see "Warp" below
=== 3/3  WidebvhFMM end to end, N=50,000 ===
    |U|  mean 0.294435  max 0.796436              <- H200 reference values
```

Exit status is 0 only if all three pass. A checksum differing in the last digits
on another GPU is floating point; a sign flip or an order of magnitude is a fault.

Then repeat for the fp32 libraries, which the 2.0 image also ships:

```bash
for l in 1 2 3; do
  docker run --rm --gpus all -e NEMO_SMOKE_LIB=libwidebvh_nemo_f32l$l.so "$IMG"
done
```

Each reports `fp32_level=$l` and a checksum within 1e-4 relative of the fp64
reference (measured on an RTX 4060: -2.450400915e+07 for levels 1 and 2,
-2.450400896e+07 for level 3, i.e. 8e-9 and 1.6e-8 relative).

## Step 5 — push to Docker Hub

```bash
docker login
docker push "$IMG"
```

~5–6 GB uploaded.

## Step 6 — validate on the H200 **before** touching the other GPUs

Back on the cluster. Docker Hub is reachable from the compute nodes and
`singularity-ce 4.1.2` is installed, so this needs no root.

```bash
mkdir -p results
singularity exec --nv --containall -B "$PWD/results:/results" \
  docker://$DOCKERHUB_USER/nemo:2.0 \
  python /workspace/pinn-stokes/benchmarks/figure12_grand_M.py \
    --backend widebvh --csv /results/fig12_h200_container.csv
```

Compare against `data/fig12_scaling_h200.csv` (widebvh, git `ac78eb4`):

| N | expected total ms |
|---|---|
| 50k | 22.50 |
| 100k | 34.91 |
| 200k | 60.45 |
| 500k | 139.02 |
| 750k | 206.27 |

Process-isolated re-measurements of the last two gave 138.36 / 205.28, so
**within ~2% is the pass criterion.** A systematic offset means the image's
torch/CUDA/architecture combination is not the one that produced the paper's
numbers — resolve that before the 5090 runs, or the new column will not be
comparable with the rest of the figure.

`--containall` is required. Without it Singularity mounts the host `$HOME` and
`$PWD`, re-introducing the `PYTHONPATH` shadowing (`/mnt/home/khanmd/pinn-stokes`,
an older checkout) that the image exists to eliminate.

## Step 7a — RTX 5090: Figure 12 (measurement #3)

```bash
mkdir -p results
docker run --rm --gpus all -v "$PWD/results:/results" "$IMG" \
  python benchmarks/figure12_grand_M.py --backend widebvh \
    --csv /results/fig12_5090.csv
```

Five sizes, ~10 min. Peak VRAM at 750k was 7.9 GB on the H200, so a 32 GB 5090 is
comfortable.

> **Write to a separate CSV, as above.** `figure12_grand_M.py` rewrites *all* rows
> matching `--backend`, keyed on the backend alone and not on the GPU — pointing
> `--csv` at `data/fig12_scaling_h200.csv` would delete the H200 widebvh rows.
> Merge deliberately in step 8.

## Step 7b — RTX A4500: Figure 13 / §3.4 (measurement #2)

```bash
mkdir -p results
docker run --rm --gpus all -v "$PWD/results:/results" -e TC_PAIR_BUDGET_GB=6 \
  "$IMG" \
  bash -c "python benchmarks/two_suspensions_1M.py 2>&1 | tee /results/a4500_two_drop.log"
```

**`TC_PAIR_BUDGET_GB=6` is not optional here.** The 1M two-drop peaks at 11.05 GB
allocated / 13.94 GB reserved, and `WidebvhFMM`'s P2P pair-list budget defaults to
12 GB — fine on a 140 GB H200, an OOM on a 20 GB A4500. Drop to 4 if 6 still
fails.

Worth adding on the same card, since it needs no extra baked data and gives the
per-step curve rather than a single average — §2.7 notes the published 2.6 s/step
may have been an early-window figure:

```bash
docker run --rm --gpus all -v "$PWD/results:/results" -e TC_PAIR_BUDGET_GB=6 \
  "$IMG" \
  python benchmarks/far_field_drift.py --backend widebvh --steps 150 \
    --csv /results/far_field_drift_a4500.csv
```

## Step 7c — any GeForce-class card (measured: RTX 4060 laptop, 8 GB): use the fp32 far field

On a card whose fp64 rate is 1/64 of fp32 (GeForce Ada/Ampere) the fp64 far-field
kernels are the whole step: 8.7 s of a 10.75 s step at N=1M on an RTX 4060 laptop
(the A4500's 6.1 s of 7.1 s in step 7b is the same effect). widebvh now has an
opt-in fp32 fast path (`WIDEBVH_FP32_LEVEL`, one .so per level; details and every
measurement in `artifacts/consumer_gpu_far_field_report.md`). The 2.0 image
(step 3, `WIDEBVH_FP32_LEVELS="1;2;3"`) ships all three levels next to the fp64
production library, which stays the default; select a level at run time:

```bash
docker run --rm --gpus all -v "$PWD/results:/results" "$IMG" \
  bash -c "python benchmarks/two_suspensions_1M.py --fp32-level 3 2>&1 | tee /results/two_drop_fp32.log"
```

`--fp32-level 3` (everything else as production: PDEG 7, mac 0.8, leaf 1024) is the
operating point the sweep picked: far field 8.68 s -> 0.39 s per step, step 10.75 s ->
2.52 s (4.3x), no measurable change in far-field error (rel_total 3.1e-4 random
loading) or symmetry; peak process VRAM 3.7 GB; `--max-leaf 512` measures the same. `NEMO_FAR_FP32_LEVEL=3` does the same for scripts without
the flag. Levels 1 and 2 are in the image only for the ablation
(`--fp32-level 1` = fp32 M2P, `2` = + fp32 P2P); `benchmarks/mac_calibration.py
--fp32-levels 0,3` reproduces the accuracy comparison on any card. Each level
costs one more compile of `nemo_capi.cu` per architecture -- keep `CUDA_ARCHS`
short when you set it. `WIDEBVH_EXTRA_PDEG="5"` adds the
PDEG-5 variants for degree sweeps (they were not worth it: see the report).

**Iterating on this repo or widebvh on a local box** without rebuilding the image:
`bash docker/run_local.sh <cmd>` bind-mounts the working tree over
`/workspace/pinn-stokes`, `~/envs/nemo-ctx/widebvh` over `/opt/widebvh-src`
(`WIDEBVH_BUILD_DIR=/opt/widebvh-src/build-<tag>`, build it inside the container
with the image's nvcc for the local sm) and a persistent JIT cache under
`~/envs/nemo-cache`. That is how everything in the report was measured.

## Step 8 — bring the results back

```bash
scp you@5090-box:results/fig12_5090.csv  data/
scp you@a4500-box:results/a4500_two_drop.log data/
```

Both CSVs carry `gpu` and `git_sha` columns, so provenance survives the trip —
`NEMO_GIT_SHA` is baked into the image and used in place of `git rev-parse`, which
has no repository to consult inside the container.

Then update `artifacts/paper_update_plan.md` §4 (mark #2 and #3 done), §1.2 (the
5090 column) and §2.8 (the H-HIGNN headline).

---

## If something fails

| symptom | cause / fix |
|---|---|
| `pack_context.sh`: `MISSING .../nemo_capi.cu` | the widebvh working tree moved; set `WIDEBVH_SRC=/path/to/widebvh` |
| `pack_context.sh`: `widebvh context is N MB` | the whitelist matched a build/data dir; check what grew in `~/programs/widebvh` |
| build: CMake cannot find OpenMP or pkg-config | apt list in the Dockerfile was trimmed; both are `REQUIRED` unconditionally in widebvh's `CMakeLists.txt` (lines 229, 279) |
| build: PETSc / OpenBLAS warnings | expected and harmless — it only skips the MFS executables, not the `widebvh_nemo*` libraries |
| build: dies in `[8/13] RUN cmake -S /opt/widebvh` with `Canceled`, `context canceled`, or a killed terminal, and `journalctl -k` shows `oom-killer` | the **host** ran out of RAM, not the container. Lower `--build-arg NINJA_JOBS` (3, then 2) and `--build-arg NVCC_THREADS=1`; see "Sizing the build" in step 3. Check `dockerd` logs for `failed to read oom_kill event ... span="[ 8/13] ..."` to confirm. Layers before step 8 stay cached, so a retry restarts at the compile |
| build: killed with no OOM in the logs, `no space left on device` | ~40 GB free is the real requirement; `docker system df` then prune |
| smoke stage 3 or a benchmark: `ModuleNotFoundError` | a package on the import chain is missing from the Dockerfile's pip list. Probe the whole chain in one shot rather than rebuilding per module: `docker run --rm --entrypoint python "$IMG" -c "import sys,types; sys.modules['<missing>']=types.ModuleType('x'); import benchmarks.performance_grand_M"` |
| smoke stage 1: `no /opt/widebvh/build-nemo/libwidebvh_nemo.so` | the widebvh build stage did not produce a library; re-read the build log |
| smoke stage 3: assertion about the patched Warp fork | something reached `WarpFMM.get_far_field_vel`; you asked for `--backend warp` in an image built without the fork |
| run: OOM on the A4500 | lower `TC_PAIR_BUDGET_GB` (6 → 4) |
| run: first invocation is very slow | torch.compile and Warp JIT for an architecture the build machine never saw. Not an error — both entry points warm up (6 and 5 iterations) before timing |
| Singularity: import errors or wrong module picked up | you omitted `--containall` |

---

## Background

### Why a packing script instead of `git clone`

Neither of the two things the image needs can be cloned:

- **widebvh** (`github.com/Shihab-Shahriar/treecode`) has 9 modified tracked files,
  and `src/nemo_capi.cu` — the entire `wbnemo_*` C ABI that
  `src/treecode_widebvh.py` dlopens — is **untracked**. `src/cpu/` is untracked
  too and is referenced unconditionally by `add_executable(two_ball_cpu ...)`, so
  a clone fails at CMake *configure* time. `cuBQL/` is a gitlink with no
  `.gitmodules`, so `--recursive` leaves it empty.
- **this repo** has `src/treecode_widebvh.py`, both production `.wt` models,
  `requirements.txt` and 8 `benchmarks/figure*.py` untracked, plus ~43 modified
  tracked files.

The script copies the working trees and refuses to run if any critical file is
absent.

### Warp: required, but stock

Warp is not optional even without the old treecode — the **near-field neighbour
search is Warp's hash grid** (`src/hashgrid_neighbors.py`, stock API only),
running every step inside a near field that is 61% of the widebvh step time.

The **patched fork** is a different matter. Its additions (`wp.bvh_mp_query`,
`wp.multipole_query_next`, `wp.bvh_primitive_id`, `wp.Bvh(..., "lbvh",
leaf_size=)`) appear at exactly four places, all inside
`WarpFMM.get_far_field_vel` (`src/treecode.py`), which `WidebvhFMM` overrides —
and Warp codegens a module's kernels lazily on first launch, so on the widebvh
path they are never resolved. The image installs **stock `warp-lang==1.12.0`**,
the PyPI release the fork (`1.12.0.dev0`) is a dev build of, whose wheel is built
with CUDA 12.8 and so already carries sm_120.

Consequence: `--backend warp` / `NEMO_FAR_FIELD=warp` are unavailable and say so
with a clear assertion rather than a Warp codegen error. To get them back:
`pack_context.sh --with-warp-fork` and `docker build --build-arg WITH_WARP_FORK=1`.

Note `hasattr(wp, "bvh_mp_query")` is **False even on the fork** — Warp's
`add_builtin` registers into a function table and the `wp.`-qualified names are
only type stubs. `src/treecode.warp_fmm_patch_state()` queries the table instead.

### What was verified before this was written

On the H200, on this cluster:

- the smoke test passes all three stages, exit 0;
- the trimmed widebvh tree **configures** with the exact Dockerfile flags,
  including the 7-architecture list (the default; step 3 now recommends a
  5-architecture subset, which builds and links the same way);
- widebvh **builds and links for sm_120** — `cuobjdump --list-elf` confirms
  sm_120 SASS in the result, against sm_80/sm_90 in the library on the cluster
  today. The 5090 path is tested, not assumed;
- `nvidia/cuda:12.8.1-devel-ubuntu24.04` exists on Docker Hub;
- every pip pin resolves, and torch 2.8.0+cu128 pulls triton 3.4.0,
  cublas 12.8.4.1, cudnn 9.10.2.21 and nccl 2.27.3 — an exact match to the
  environment that produced the paper's H200 numbers.

### What the first real build changed

The above was written before the image had ever been built end to end. Building
it (24-core / 15 GB host, RTX 4060, sm_89) turned up one thing the analysis had
missed:

- **`seaborn` was absent from the pip list and is on the critical import chain.**
  `benchmarks/figure12_grand_M.py` → `benchmarks/performance_grand_M.py` →
  `src/mob_op_2b_combined.py` → `src/analysis_utils.py:3  import seaborn as sns`.
  It is a *dead* import — nothing in `analysis_utils.py` calls `sns.*` — but it
  gates smoke stage 3 and, more to the point, **measurement #3 on the 5090**.
  Step 7b was unaffected: `two_suspensions_1M.py` does not touch that chain.
  Fixed by pinning `seaborn==0.13.2` in the same `pip install` as the other
  pins, so the resolver honours them; `numpy`, `scipy`, `pandas`, `matplotlib`,
  `torch` and `triton` were re-checked afterwards and are unchanged.

  The Dockerfile's claim that the production import chain is "only numpy,
  pandas, scipy, matplotlib, torch, triton and warp" describes what `src/`
  *uses*, not what it *imports*. Treat any further `ModuleNotFoundError` the
  same way — probe the whole chain with a stub before rebuilding.

Verified on that build, and consistent with the H200 references:

- smoke stage 1 checksum `-2.450400935e+07` — an **exact** match, so the native
  build and the C ABI are byte-for-byte right;
- smoke stage 3 `|U|` mean 0.294469 / max 0.796676 against the H200's 0.294435 /
  0.796436 — agreement to ~1e-4 relative, which is fp32 accumulation ordering on
  a different GPU, well inside the "sign flip or order of magnitude is a fault"
  criterion;
- `cuobjdump --list-elf libwidebvh_nemo.so` → `sm_86 sm_89 sm_90 sm_120`, so the
  5090 and A4500 both get native SASS rather than a JIT.

The one thing still **not** verified is the step-6 H200 comparison against
`data/fig12_scaling_h200.csv`. That remains the gate before the 5090 runs.

### Other notes

- `WIDEBVH_BUILD_DIR=/opt/widebvh/build-nemo` is the only path knob
  `src/treecode_widebvh.py` needs; the image sets it.
- Caches (inductor, Triton, Warp, matplotlib) live under `/workspace/.cache`, are
  world-writable, and fall back to a temp dir if that path is read-only, so
  `--user` and Singularity both work.
- Nothing in `src/` can be imported without a GPU — `src/treecode.py` calls
  `wp.get_stream("cuda:0")` at module scope. That is why the Dockerfile runs no
  import check at build time, and why smoke stage 1 is subprocessed: running
  `wbnemo_smoke` before torch initialises CUDA in the same process crashes the
  later stage.
- `p3`/`p5` widebvh variants are not built; they serve degree sweeps neither
  experiment runs. Add them to the `cmake --build --target` line if needed.
