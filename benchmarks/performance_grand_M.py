"""Performance benchmark for grand mobility operators.

This script mirrors the configuration used in ``accuracy_grand_M.py`` but focuses
solely on wall-clock performance of the ``apply`` method across several grand
mobility operators:

* ``NNMob``      – CPU-oriented implementation.
* ``NNMobTorch`` – GPU-friendly variant backed by TorchScript.
* ``NNMob`` with ``rpy_only`` fallback – analytic Rotne–Prager–Yamakawa mode.

We load a single reference configuration (``data/n100.csv``), generate a
deterministic synthetic wrench for each particle, and time repeated calls to
``apply`` with warm-up runs to amortize JIT/cache effects.  Results are printed
to the console in a compact table.
"""

from __future__ import annotations

import math
import gc
import sys
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
import torch._inductor.config as inductor_config
try:
    from torch._inductor.exc import InductorError as TorchInductorError
except Exception:  # pragma: no cover - optional dependency detail
    TorchInductorError = None
try:
    from torch._dynamo.exc import TorchDynamoError as TorchDynamoError
except Exception:  # pragma: no cover - optional dependency detail
    TorchDynamoError = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gpu_mob_2b import NNMobTorch
from src.mob_op_2b_combined import NNMob
from src.gpu_nbody_mob import Mob_Nbody_Torch
from src.treecode import WarpFMM

# Setting this env variable below totally messed up timing results
# torch.compile has so far been an absolute headache for benchmarking
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
inductor_config.triton.cudagraph_skip_dynamic_graphs = True
inductor_config.triton.cudagraphs = False

_DYNAMO_EXC: tuple[type[BaseException], ...] = ()
if TorchInductorError is not None:
    _DYNAMO_EXC += (TorchInductorError,)
if TorchDynamoError is not None:
    _DYNAMO_EXC += (TorchDynamoError,)

torch.set_grad_enabled(False)

DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
TMP_DIR = ROOT / "tmp"


@dataclass
class BenchmarkConfig:
    """Benchmark configuration parameters."""

    viscosity: float = 1.0
    warmup_runs: int = 6
    timed_runs: int = 6

    def total_runs(self) -> int:
        return self.warmup_runs + self.timed_runs


@dataclass
class BenchmarkResult:
    """Container capturing timing statistics for one operator."""

    label: str
    device: str
    timings: np.ndarray

    @property
    def mean_ms(self) -> float:
        return float(self.timings.mean() * 1_000)

    @property
    def std_ms(self) -> float:
        if self.timings.size < 2:
            return float("nan")
        return float(self.timings.std(ddof=1) * 1_000)

    @property
    def min_ms(self) -> float:
        return float(self.timings.min() * 1_000)

    @property
    def max_ms(self) -> float:
        return float(self.timings.max() * 1_000)

    @property
    def throughput_hz(self) -> float:
        mean_seconds = self.timings.mean()
        if mean_seconds == 0:
            return math.inf
        return float(1.0 / mean_seconds)


def load_configuration(csv_path: Path) -> np.ndarray:
    """Load particle configuration (positions + quaternion) from CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {csv_path}")

    df = pd.read_csv(csv_path, float_precision="high")
    required_cols = ["x", "y", "z"]
    assert all(col in df.columns for col in required_cols), (
        f"Missing required columns: {[col for col in required_cols if col not in df.columns]}"
    )

    positions = df[required_cols].to_numpy(dtype=np.float64, copy=True)

    quat_cols = ["q_x", "q_y", "q_z", "q_w"]
    if all(col in df.columns for col in quat_cols):
        orientations = df[quat_cols].to_numpy(dtype=np.float64, copy=True)
    else:
        orientations = np.zeros((positions.shape[0], 4), dtype=np.float64)
        orientations[:, 3] = 1.0

    config = np.concatenate([positions, orientations], axis=1)
    return np.ascontiguousarray(config)


def build_forces(
    num_particles: int,
    seed: int = 42,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Generate deterministic random forces/torques for benchmarking."""

    rng = np.random.default_rng(seed)
    forces = rng.standard_normal(size=(num_particles, 6)).astype(dtype, copy=False)
    return np.ascontiguousarray(forces)


def build_lattice_config(num_particles: int, spacing: float = 3.0) -> np.ndarray:
    """Build a cubic lattice configuration with identity orientations."""

    if num_particles <= 0:
        raise ValueError("num_particles must be positive")

    n_side = int(math.ceil(num_particles ** (1.0 / 3.0)))
    offsets = (n_side - 1) / 2.0

    xs = (np.arange(n_side) - offsets) * spacing
    ys = (np.arange(n_side) - offsets) * spacing
    zs = (np.arange(n_side) - offsets) * spacing

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    positions = grid[:num_particles].astype(np.float32, copy=False)

    orientations = np.zeros((num_particles, 4), dtype=np.float32)
    orientations[:, 3] = 1.0

    config = np.concatenate([positions, orientations], axis=1)
    return np.ascontiguousarray(config)


def resolve_device(operator) -> torch.device:
    """Best-effort resolution of the torch device backing an operator."""

    device_attr = getattr(operator, "device", None)
    if isinstance(device_attr, torch.device):
        return device_attr
    if isinstance(device_attr, str):
        return torch.device(device_attr)
    return torch.device("cpu")
        


def benchmark_apply(
    operator,
    label: str,
    config: np.ndarray,
    forces: np.ndarray,
    bench_cfg: BenchmarkConfig,
) -> BenchmarkResult:
    """Measure wall-clock timings for repeated ``apply`` invocations."""
    
    device = torch.device("cuda")
    torch.cuda.synchronize(device)
    
    n_particles = config.shape[0]
    
    # Extract positions (first 3 columns) and orientations (last 4 columns)
    positions = torch.as_tensor(config[:, :3], dtype=torch.float32, device=device)
    orientations = torch.as_tensor(config[:, 3:], dtype=torch.float32, device=device)
    forces_t = torch.as_tensor(forces, dtype=torch.float32, device=device)
    vis_arr = torch.full((n_particles,), bench_cfg.viscosity, dtype=torch.float32, device=device)
    
    # Far-field solvers (WarpFMM, WidebvhFMM) take (positions, orientations,
    # forces, vis_arr); bare near-field operators take a viscosity scalar instead
    # of the per-particle array. Duck-typed so any far-field backend works.
    is_fmm = hasattr(operator, "get_far_field_vel")

    def run_apply():
        if is_fmm:
            return operator.apply(positions, orientations, forces_t, vis_arr)
        return operator.apply(positions, orientations, forces_t,
                              bench_cfg.viscosity)

    # Warm-up executes without timing to stabilize caches/JIT on both CPU & GPU.
    for _ in range(bench_cfg.warmup_runs):
        run_apply()
        torch.cuda.synchronize(device)
    print()

    timings: List[float] = []
    for _ in range(bench_cfg.timed_runs):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        run_apply()
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        timings.append(end - start)
        time.sleep(0.1)  # brief pause to avoid GPU overheating issues


    print(f"total time for {bench_cfg.timed_runs} runs:", sum(timings)*1000, "ms", timings)
    
    timings = np.array(sorted(timings), dtype=np.float64)
    timings = timings[1:-2]  # discard anomalies, specially the last timings to avoid recompilation effects

    
    device = resolve_device(operator)
    return BenchmarkResult(label=label, device=device.type, timings=timings)


def format_results(results: Iterable[BenchmarkResult], bench_cfg: BenchmarkConfig, batch_size: int) -> str:
    """Render benchmark outcomes as a formatted table string."""

    header = f"Benchmark: N={batch_size}, warmup={bench_cfg.warmup_runs}, runs={bench_cfg.timed_runs}"
    lines = [header, "-" * len(header)]
    lines.append(f"{'Operator':<32}{'Device':<8}{'Mean (ms)':>12}{'Std (ms)':>10}")
    lines.append("-" * len(lines[-1]))

    for res in results:
        lines.append(
            f"{res.label:<32}{res.device:<8}"
            f"{res.mean_ms:>12.2f}{res.std_ms:>10.2f}"
        )

    return "\n".join(lines)


def build_operators(shape: str) -> List[tuple[str, object]]:
    """Near-field-only operators, i.e. no far field at all.

    Useful for isolating the near-field cost; the full grand mobility is
    build_fmm_operators() below. Note these carry an O(N^2) dense far field when
    far_field="rpy", so keep them for small N only.
    """
    return [
        ("NNMob_GPU_Nbody", build_near_field("nbody", shape)),
        ("NNMobTorch_rpy", build_near_field("2b_rpy", shape)),
        ("NNMobTorch", build_near_field("2b_nn", shape)),
    ]


# Model weights. These are `.wt` state dicts, not TorchScript `.pt` -- both
# NNMobTorch and Mob_Nbody_Torch assert on the extension.
SELF_MODEL = MODELS_DIR / "self_interaction_model.pt"
TWO_BODY_MODEL = MODELS_DIR / "combined_2body.wt"
NBODY_MODEL = MODELS_DIR / "nbody_cross_tmp.wt"

# Far-field defaults. `theta` belongs to WarpFMM (Warp BVH, monopole+dipole),
# `mac` to WidebvhFMM (widebvh BaryStokes); they are different acceptance
# criteria and the values do not transfer between them.
FMM_THETA = 0.3
FMM_LEAF_SIZE = 16
FMM_BLOCK_DIM = 256

# Which far field the module-level entry points use. Override with
# NEMO_FAR_FIELD=warp to re-measure the published baseline.
FAR_FIELD_BACKEND = os.environ.get("NEMO_FAR_FIELD", "widebvh")


def build_far_field(near_field_operator, backend: str, near_field_cutoff: float,
                    **kwargs):
    """Wrap a near-field operator in the requested far-field solver.

    backend="warp"        -> src.treecode.WarpFMM        (the published baseline)
    backend="widebvh"     -> WidebvhFMM, BaryStokes       (production)
    backend="widebvh-cart"-> WidebvhFMM, CartesianStokes  (the A/B expansion)

    The two widebvh backends are the same engine and the same near-field
    complement; only the multipole expansion differs, so they need different
    `mac` values for the same accuracy (see each policy's DEFAULT_* constants).
    """
    if backend == "warp":
        return WarpFMM(
            near_field_operator=near_field_operator,
            theta=kwargs.pop("theta", FMM_THETA),
            leaf_size=FMM_LEAF_SIZE,
            near_field_cutoff=near_field_cutoff,
            device="cuda",
            block_dim=FMM_BLOCK_DIM,
        )
    if backend in ("widebvh", "widebvh-cart"):
        from src.treecode_widebvh import (
            WidebvhFMM, DEFAULT_CART_MAC, DEFAULT_CART_ORDER)
        if backend == "widebvh-cart":
            kwargs.setdefault("policy", "cart")
            kwargs.setdefault("mac", DEFAULT_CART_MAC)
            kwargs.setdefault("order", DEFAULT_CART_ORDER)
        return WidebvhFMM(
            near_field_operator=near_field_operator,
            near_field_cutoff=near_field_cutoff,
            **kwargs,
        )
    raise ValueError(f"unknown far-field backend {backend!r}")


def build_near_field(kind: str, shape: str, near_field_cutoff: float = 6.0):
    """Near-field operator only, no far field (the treecode owns r >= cutoff).

    kind="2b_rpy" -> analytic RPY pair kernel inside the cutoff
    kind="2b_nn"  -> learned two-body kernel
    kind="nbody"  -> learned two-body + many-body correction (the full NeMO)
    """
    assert SELF_MODEL.exists(), f"Missing model: {SELF_MODEL}"
    assert TWO_BODY_MODEL.exists(), f"Missing model: {TWO_BODY_MODEL}"

    if kind in ("2b_rpy", "2b_nn"):
        return NNMobTorch(
            shape=shape,
            self_nn_path=str(SELF_MODEL),
            two_nn_path=str(TWO_BODY_MODEL),
            near_field="rpy" if kind == "2b_rpy" else "nn",
            far_field=None,
            switch_dist=near_field_cutoff,
        )
    if kind == "nbody":
        assert NBODY_MODEL.exists(), f"Missing model: {NBODY_MODEL}"
        return Mob_Nbody_Torch(
            shape=shape,
            self_nn_path=str(SELF_MODEL),
            two_nn_path=str(TWO_BODY_MODEL),
            nbody_nn_path=str(NBODY_MODEL),
            near_field_2b="nn",
            far_field_2b=None,
            near_far_switch=near_field_cutoff,
        )
    raise ValueError(f"unknown near-field kind {kind!r}")


def build_fmm_operators(shape: str, near_field_cutoff: float = 6.0,
                        backend: str = "warp", kinds=("2b_rpy", "2b_nn", "nbody"),
                        **kwargs) -> List[tuple[str, object]]:
    """Instantiate the benchmark's mobility operators.

    Each is a far-field solver wrapping a near-field operator that has its own
    far field disabled, so the treecode owns r >= cutoff exclusively.
    """
    labels = {"2b_rpy": "FMM_2body_RPY", "2b_nn": "FMM_2body_NN",
              "nbody": "FMM_Nbody_NN"}
    return [
        (labels[k],
         build_far_field(build_near_field(k, shape, near_field_cutoff),
                         backend, near_field_cutoff, **kwargs))
        for k in kinds
    ]


def max_sim_size(start_n: int = 8_000_000) -> int:
    """
    Estimate maximum simulation size fitting in GPU memory.
    
    Starting from N=1M, double until out-of-memory, then do a coarse binary search.

    Particles are placed in a cubic lattice with spacing 3.0 radii (to avoid near-field issues).
    Only the FMM N-body operator is considered.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, cannot estimate max sim size.")
        return 0

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory
    print(f"GPU: {props.name}")
    print(f"Total GPU memory: {total_mem / (1024**3):.2f} GB")

    near_field_cutoff = 6.0
    fmm_nbody = build_far_field(
        build_near_field("nbody", "sphere", near_field_cutoff),
        FAR_FIELD_BACKEND, near_field_cutoff)

    min_step = 100_000
    spacing = 3.0

    def _try_size(n_particles: int) -> tuple[bool, float]:
        positions = orientations = forces_t = vis_arr = None
        config = forces = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            config = build_lattice_config(n_particles, spacing=spacing)
            forces = build_forces(n_particles, seed=2024, dtype=np.float32)
            positions = torch.as_tensor(config[:, :3], dtype=torch.float32, device=device)
            orientations = torch.as_tensor(config[:, 3:], dtype=torch.float32, device=device)
            forces_t = torch.as_tensor(forces, dtype=torch.float32, device=device)
            vis_arr = torch.full((n_particles,), 1.0, dtype=torch.float32, device=device)

            torch.cuda.synchronize(device)
            fmm_nbody.apply(positions, orientations, forces_t, vis_arr)
            torch.cuda.synchronize(device)

            peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            return True, peak_gb
        except (RuntimeError, AssertionError) + _DYNAMO_EXC as exc:
            msg = str(exc).lower()
            if (
                "out of memory" in msg
                or "cudagraph" in msg
                or "cuda graph" in msg
                or "inductor" in msg
                or "dynamo" in msg
                or "triton" in msg
                or "xblock" in msg
            ):
                print(f"Trial N={n_particles:,} failed: {type(exc).__name__}: {exc}")
                return False, 0.0
            raise
        finally:
            del positions, orientations, forces_t, vis_arr, config, forces
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

    warmup_n = min(10_000, start_n)
    print(f"Warming up kernels with N={warmup_n:,} ...")
    warmup_ok, _ = _try_size(warmup_n)
    if not warmup_ok:
        print("Warmup run failed; aborting.")
        return 0

    print(f"Searching max N with step >= {min_step:,} ...")
    low = 0
    high = None
    _, peak = _try_size(start_n)
    low = start_n
    print(f"N={start_n:,} fits (peak {peak:.2f} GB)")
    n = start_n
    for _ in range(32):
        n *= 2
        fit, peak = _try_size(n)
        if fit:
            low = n
            print(f"N={n:,} fits (peak {peak:.2f} GB)")
        else:
            high = n
            print(f"N={n:,} OOM")
            break

    if high is None:
        max_fit = low
        print(f"No OOM encountered up to N={max_fit:,}.")
    else:
        while high - low > min_step:
            step = (high - low) // 2
            if step < min_step:
                break
            candidate = low + step
            fit, peak = _try_size(candidate)
            if fit:
                low = candidate
                print(f"N={candidate:,} fits (peak {peak:.2f} GB)")
            else:
                high = candidate
                print(f"N={candidate:,} OOM")
        max_fit = low

    if max_fit <= 0:
        return 0

    print(f"Max N (coarse, step >= {min_step:,}): {max_fit:,}")

    # Apply safety margin for benchmark (90% of max to account for torch.compile overhead)
    benchmark_n = int(max_fit * 0.9)
    benchmark_n = (benchmark_n // 100_000) * 100_000  # Round down to nearest 100k
    print(f"Using N={benchmark_n:,} for benchmark (90% safety margin)")

    # Full cleanup before benchmark
    del fmm_nbody
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    torch.cuda.synchronize(device)

    # Recreate operator fresh for benchmark
    fmm_nbody_bench = build_far_field(
        build_near_field("nbody", "sphere", near_field_cutoff),
        FAR_FIELD_BACKEND, near_field_cutoff)

    bench_cfg = BenchmarkConfig(warmup_runs=2, timed_runs=6)
    config = build_lattice_config(benchmark_n, spacing=spacing)
    forces = build_forces(benchmark_n, seed=2024, dtype=np.float32)
    result = benchmark_apply(fmm_nbody_bench, "FMM_Nbody_NN", config, forces, bench_cfg)
    print(format_results([result], bench_cfg, batch_size=benchmark_n))
    return max_fit


def main(n) -> None:
    filename = "uniform_large_0.1_{}.csv".format(n)
    print(f"Running performance benchmark on configuration: {filename}")
    bench_cfg = BenchmarkConfig()
    #csv_path = DATA_DIR / "n100.csv"
    csv_path = TMP_DIR / filename

    config = load_configuration(csv_path)
    forces = build_forces(config.shape[0], seed=2024)

    operators = build_fmm_operators(shape="sphere")

    results: List[BenchmarkResult] = []
    for label, operator in operators:
        result = benchmark_apply(operator, label, config, forces, bench_cfg)
        results.append(result)
        print(f"Completed benchmark for operator: {label}")
        # time.sleep(2)  # BUGXFIX: GPU underclocked during sleep
        print("Cooled GPU down a bit\n\n\n")

    print(format_results(results, bench_cfg, batch_size=config.shape[0]))


if __name__ == "__main__":
    #n = int(sys.argv[1]) 
    # main(n)
    
    max_sim_size()

    if torch.cuda.is_available():
        max_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\nTotal Peak VRAM Usage: {max_vram_gb:.2f} GB")
