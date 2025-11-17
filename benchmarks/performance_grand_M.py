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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gpu_mob_2b import NNMobTorch
from src.mob_op_2b_combined import NNMob
from src.gpu_nbody_mob import Mob_Nbody_Torch


torch.set_grad_enabled(False)

DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
TMP_DIR = ROOT / "tmp"


@dataclass
class BenchmarkConfig:
    """Benchmark configuration parameters."""

    viscosity: float = 1.0
    warmup_runs: int = 4
    timed_runs: int = 5

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
    expected_cols = ["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]
    config = df[expected_cols].to_numpy(dtype=np.float64, copy=True)
    return np.ascontiguousarray(config)


def build_forces(num_particles: int, seed: int = 42) -> np.ndarray:
    """Generate deterministic random forces/torques for benchmarking."""

    rng = np.random.default_rng(seed)
    forces = rng.standard_normal(size=(num_particles, 6)).astype(np.float64, copy=False)
    return np.ascontiguousarray(forces)


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
    config = torch.as_tensor(config, dtype=torch.float32, device=device)
    forces = torch.as_tensor(forces, dtype=torch.float32, device=device)

    # Warm-up executes without timing to stabilize caches/JIT on both CPU & GPU.
    for _ in range(bench_cfg.warmup_runs):
        operator.apply(config, forces, bench_cfg.viscosity)
        torch.cuda.synchronize(device)

    timings: List[float] = []
    for _ in range(bench_cfg.timed_runs):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        operator.apply(config, forces, bench_cfg.viscosity)
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        timings.append(end - start)

    device = resolve_device(operator)
    return BenchmarkResult(label=label, device=device.type, timings=np.array(timings, dtype=np.float64))


def format_results(results: Iterable[BenchmarkResult], bench_cfg: BenchmarkConfig, batch_size: int) -> str:
    """Render benchmark outcomes as a formatted table string."""

    header = f"Benchmark: N={batch_size}, warmup={bench_cfg.warmup_runs}, runs={bench_cfg.timed_runs}"
    lines = [header, "-" * len(header)]
    lines.append(f"{'Operator':<32}{'Device':<8}{'Mean (ms)':>12}{'Std (ms)':>10}{'Min (ms)':>10}{'Max (ms)':>10}{'Hz':>12}")
    lines.append("-" * len(lines[-1]))

    for res in results:
        lines.append(
            f"{res.label:<32}{res.device:<8}"
            f"{res.mean_ms:>12.2f}{res.std_ms:>10.2f}{res.min_ms:>10.2f}{res.max_ms:>10.2f}{res.throughput_hz:>12.1f}"
        )

    return "\n".join(lines)


def build_operators(shape: str) -> List[tuple[str, object]]:
    """Instantiate the operators we want to benchmark."""

    self_path = MODELS_DIR / "self_interaction_model.pt"
    two_body_path = MODELS_DIR / "two_body_combined_model.pt"

    if not self_path.exists() or not two_body_path.exists():
        missing = [path for path in (self_path, two_body_path) if not path.exists()]
        raise FileNotFoundError(f"Missing TorchScript models: {missing}")

    operators = [
        (
            "NNMobTorch",
            NNMobTorch(
                shape=shape,
                self_nn_path=str(self_path),
                two_nn_path=str(two_body_path),
                nn_only=False,
                rpy_only=False,
                switch_dist=6.0,
            ),
        ),
        # (
        #     "NNMobTorch_rpy",
        #     NNMobTorch(
        #         shape=shape,
        #         self_nn_path=str(self_path),
        #         two_nn_path=str(two_body_path),
        #         nn_only=False,
        #         rpy_only=True,
        #         switch_dist=6.0,
        #     ),
        # ),
        (
            "NNMob_GPU_Nbody",
            Mob_Nbody_Torch(
                shape=shape,
                self_nn_path=str(self_path),
                two_nn_path=str(two_body_path),
                nbody_nn_path="data/models/nbody_pinn_b1.pt",
                nn_only=False,
                rpy_only=False,
                switch_dist=6.0,
            ),
        ),
    ]

    return operators


def main() -> None:
    filename = "uniform_sphere_0.1_1600.csv"
    print(f"Running performance benchmark on configuration: {filename}")
    bench_cfg = BenchmarkConfig()
    #csv_path = DATA_DIR / "n100.csv"
    csv_path = TMP_DIR / filename

    config = load_configuration(csv_path)
    forces = build_forces(config.shape[0], seed=2024)

    operators = build_operators(shape="sphere")

    results: List[BenchmarkResult] = []
    for label, operator in operators:
        result = benchmark_apply(operator, label, config, forces, bench_cfg)
        results.append(result)
        print(f"Completed benchmark for operator: {label}")

    print(format_results(results, bench_cfg, batch_size=config.shape[0]))


if __name__ == "__main__":
    main()
    if torch.cuda.is_available():
        max_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\nTotal Peak VRAM Usage: {max_vram_gb:.2f} GB")
