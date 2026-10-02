# NeMO — neural mobility operators for Stokes-flow particle suspensions

NeMO replaces the boundary solve of a rigid-sphere Stokes suspension with learned mobility
operators: an analytic self term, a two-body neural kernel inside a switch distance, a
moments-based n-body correction (rotation-invariant band moments of each pair's
neighbourhood → coefficients on analytic tensor bases, reciprocal and O(3)-equivariant by
construction; `moments_for_nbody.md`) and a learned per-particle diagonal, with the RPY far
field beyond the switch distance summed by the widebvh GPU treecode (vendored in
`extern/widebvh`). Truth for training and evaluation comes from Method of Fundamental
Solutions (MFS) solvers in `src/`.

`CLAUDE.md` is the detailed map of the code (operators, models, conventions, pitfalls);
`reproduction.md` lists, figure by figure, the command behind every paper figure.

## Running it

Everything runs from the repo root with the root on `PYTHONPATH`:

```sh
git clone https://github.com/Shihab-Shahriar/pinn-stokes && cd pinn-stokes
export PYTHONPATH=$PWD
```

Python ≥ 3.11. `requirements.txt` pins the versions of the Docker image, which ran the
paper's measurements.

### CPU and Triton parts

The CPU operators, the accuracy harness (`benchmarks/paper_accuracy_v2.py`), training
(`experiments/train_*_v2.py`) and pytest are plain Python; the MFS truth solvers
(`BatchedMFS`, Triton) also need an NVIDIA GPU, but neither Warp nor widebvh:

```sh
pip install -r requirements.txt
export TORCH_COMPILE_DISABLE=1           # for accuracy work; leave compile on for timing
python benchmarks/paper_accuracy_v2.py --help
```

### GPU operators

`Mob_Nbody_Moments_Torch` (the near field), `WidebvhFMM` (the far field), the performance
benchmarks and the 1M-particle dynamics need Warp (`warp-lang`, in `requirements.txt`) and the
widebvh libraries. widebvh is built from `extern/widebvh` **once per GPU architecture, on that
GPU**: `extern/widebvh/build_nemo.sh` detects the GPU and writes
`extern/widebvh/build-sm<compute capability>/`, and `WidebvhFMM` loads the build for the GPU it
runs on, so a checkout shared by several kinds of GPU needs one run on each. The source builds
for any GPU from Turing (sm_75) on.

**Natively.** Besides the Python packages this needs an NVIDIA driver new enough for torch's
CUDA build (≥ 570 for the pinned torch 2.8 / CUDA 12.8 wheel), a CUDA toolkit for `nvcc`
(12.x or 13.x: checked with 12.9 on an A100 and an H200, 13.1 on an RTX 4060), CMake
≥ 3.18 and Ninja (`pip install cmake ninja` if the system's are missing or old), a C++17
compiler with OpenMP that `nvcc` accepts, and pkg-config.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
bash extern/widebvh/build_nemo.sh                 # ~3 min -> extern/widebvh/build-sm<CC>/
python docker/smoke_test.py                       # widebvh C ABI, Warp, one 50k-particle apply
python benchmarks/two_suspensions_1M.py --near-op moments-v3   # 1M-sphere sedimentation, 50 steps
```

If loading a library fails with `libcublas.so.<N>: cannot open shared object file`, put the
toolkit's `lib64` on `LD_LIBRARY_PATH`: the widebvh libraries link the cuBLAS of the toolkit
that built them (a CUDA 12 build is also satisfied by the copy pip installs with torch).

**In Docker** (Docker + the NVIDIA container toolkit; the image `sskhan39/nemo:2.0` has CUDA
12.8, torch 2.8, Warp 1.12, CMake and Ninja). `docker/run_local.sh` mounts this checkout into
the image, so the build lands in the same `extern/widebvh/build-sm<CC>/` and also works natively:

```sh
bash docker/run_local.sh bash extern/widebvh/build_nemo.sh
bash docker/run_local.sh python docker/smoke_test.py
bash docker/run_local.sh python benchmarks/two_suspensions_1M.py --near-op moments-v3
```

**From Python** — the production operator, moments-v3 near field + widebvh far field, both
switching at 8 radii (all inputs float32 CUDA tensors; spheres take the identity quaternion,
scalar last):

```python
import torch
from src.gpu_nbody_moments import Mob_Nbody_Moments_Torch
from src.treecode_widebvh import WidebvhFMM

near = Mob_Nbody_Moments_Torch(
    shape="sphere",
    self_nn_path="data/models/self_interaction_model.pt",
    two_nn_path="data/models/combined_2body.wt",
    moments_nn_path="experiments/nbody_moments_v3_nb8lin_tr2_kinf_rc8_pc8c.wt",
    diag_nn_path="experiments/nbody_diag_v2_pc8c.wt",
    near_field_2b="nn", far_field_2b=None, switch_dist=8.0)
op = WidebvhFMM(near_field_operator=near, near_field_cutoff=8.0)

pos = ...                                         # (N, 3) sphere centres, radius 1
N = pos.shape[0]
quat = torch.zeros(N, 4, device="cuda"); quat[:, 3] = 1
force = ...                                       # (N, 6) [Fx, Fy, Fz, Tx, Ty, Tz]
vel = op.apply(pos, quat, force, torch.ones(N, device="cuda"))   # (N, 6) [U, Omega]; last arg viscosity
```

Performance numbers belong on a datacenter GPU (the paper's are H200, `slurm/h200_v3/`). Pass
`--near-op moments-v3` to the performance scripts: their default is the published baseline
operator. See `extern/widebvh/README.md` and `docker/README.md` for more.

## Data that is not in git

Everything below is regenerated by the repo's own scripts:

| what | where | how |
|---|---|---|
| performance configurations (uniform φ = 0.1 boxes, 5k–4M spheres) | `tmp/uniform_large_0.1_<N>.csv` | generated on first use (`benchmarks/cluster.py:ensure_uniform_large`, ~1 min per million) |
| MFS truth cache of the accuracy protocols | `tmp/nbody_moments_truth/` (~210 MB) | `python benchmarks/paper_accuracy_v2.py --exp ... --phis ... --truth-only` on a GPU (~8 min per N = 200 configuration on an RTX 4060); `slurm/paper_truth.sbatch` |
| training dataset v2 (full MFS grand mobility matrices) | `data/multibody_v2/` (4.5 GB) | `src/create_dataset_multibody_v2.py --plan default` (~4.5 H200 GPU-hours, `slurm/gen_multibody_v2.sbatch`) plus `--family chain`, then `experiments/build_nbody_v2_cache.py` |

The trained models are in git (`data/models/*.pt` + `.json` sidecars, `experiments/*.wt`).
The HIGNN and Stokesian Dynamics baselines are third-party code, read from checkouts at
`HIGNN_ROOT` / `SD_ROOT` (see `src/hignn_ops.py`, `src/sd_ops.py`).

## Development log

### Jan 14

+ Optimize two_suspens_1M.py. Baseline: for t=0.5 i.e. 50 timesteps
    + t=.5 => 1.05/1.04. t=.25=>1.004/1.002. Total: 75.22s/75.23
    + After making `positions` tensors contiguousL 72.75s. t=.25=>.96
