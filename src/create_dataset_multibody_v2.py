#!/usr/bin/env python3
"""n-body training data v2: full grand mobility matrices of realistic sphere configurations.

For every configuration (P unit spheres, viscosity 1) the batched MFS solver (src/mfs_batched.py)
computes the full grand mobility matrix M (6P x 6P), [U;Omega]_t = sum_s M_ts [F;T]_s, from all 6P unit
force/torque right-hand sides at once.  Any (target, source, neighbourhood) training row can be extracted
later (src/nbody_features.load_multibody_v2 / pair_rows_from_M), including random-force resampling.

Configuration families (parameters sampled per shard; all with a fixed minimum surface gap, like the
generators used in practice):
  uniform  random sequential addition in a box sized from the volume fraction phi
           (benchmarks/cluster.py::uniform_sphere_cluster semantics: gap >= 0.1, particle 0 at the origin)
  grown    cluster growth: each new sphere at exactly gap delta from a random existing sphere and at
           least 2 + delta from all others (benchmarks/cluster.py::grow_cluster semantics)
  lattice  primitive cubic lattice with spacing a = ((4 pi / 3) / phi)^(1/3), the P points nearest the
           origin (a small sedimenting-drop patch), jittered by +-j a per coordinate; gap >= 0.05
  chain    quasi-1D line of spheres: per-gap centre spacing iid uniform in [2.1, 8], transverse
           Gaussian jitter of absolute scale j (j = 0 is the exactly collinear manifold, which the
           box families never sample and where the pc8 pair model had an unconstrained error cliff
           -- see reproduction.md, Figure 6)

Storage: data/multibody_v2/{family}/{tag}/shard_{k:04d}.npz with positions (n,P,3) f64, M (n,6P,6P)
(fp32 by default), per-config seed / iterations / residuals / symmetry error / wall time and metadata;
one manifest CSV per worker; failures.csv.  Resumable (existing shards are skipped), deterministic seeds
(seed = f(seed0, family, params, P, shard, index)), time-budgeted, multi-worker (disjoint plan items).

    python src/create_dataset_multibody_v2.py --plan default --acc fine --time-budget 28800 \\
        --num-workers 4 --worker 0 --backend torch64 --mem-budget-gb 40      # (per GPU on the H200)
    python src/create_dataset_multibody_v2.py --family grown --delta 0.1 --P 8 --n-configs 16 --backend triton32
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(ROOT)]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

GAP_MIN = 0.05            # global minimum surface-to-surface gap (as in every existing dataset)
FAMILY_ID = {"uniform": 1, "grown": 2, "lattice": 3, "chain": 4}
SHARD_SIZE = {8: 1024, 12: 768, 16: 512, 24: 384, 32: 256, 48: 128, 64: 64}   # configs per shard by P

DEFAULT_PLAN = {
    # family: (parameter grid, P list, weight per (params, P) item -> configs per round relative to the shard)
    "uniform": {"phi": [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25], "P": [16, 32, 48, 64]},
    "grown": {"delta": [0.05, 0.1, 0.2, 0.3, 0.5, 1.0], "P": [8, 12, 16, 24, 32]},
    "lattice": {"phi": [0.05, 0.1, 0.15], "jitter": [0.05, 0.1, 0.2], "P": [32, 64]},
}
# rounds interleave the families so that the mix by configuration count is ~60 / 30 / 10 %
FAMILY_ROUND_WEIGHT = {"uniform": 1.0, "grown": 0.5, "lattice": 0.25, "chain": 1.0}
CHAIN_JITTER = [0.0, 0.02, 0.05, 0.1, 0.3]  # transverse sigma (radii); 0 = exactly collinear
CHAIN_P = [8, 12, 16]


# ----------------------------------------------------------------------------- configuration families
def _min_gap(pos: np.ndarray) -> float:
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min() - 2.0)


def gen_uniform(P: int, phi: float, rng: np.random.Generator, gap: float = 0.1, max_attempts: int = 20000):
    """Random sequential addition in a centred box, particle 0 at the origin (uniform_sphere_cluster)."""
    L = (P * (4.0 / 3.0) * np.pi / phi) ** (1.0 / 3.0)
    half = 0.5 * L
    assert half - 1.0 > 0, "box too small"
    pos = np.zeros((P, 3))
    for i in range(1, P):
        for _ in range(max_attempts):
            x = rng.uniform(-half + 1.0, half - 1.0, 3)
            if np.all(np.linalg.norm(pos[:i] - x, axis=1) >= 2.0 + gap):
                pos[i] = x
                break
        else:
            return None
    return pos


def gen_grown(P: int, delta: float, rng: np.random.Generator, max_attempts: int = 10000):
    """Cluster growth at exactly gap delta from a random existing sphere (grow_cluster semantics)."""
    d = 2.0 + delta
    centers = [np.zeros(3)]
    for _ in range(1, P):
        placed = False
        for _ in range(max_attempts):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            c = centers[int(rng.integers(len(centers)))] + d * u
            if all(np.linalg.norm(c - x) >= d - 1e-12 for x in centers):
                centers.append(c)
                placed = True
                break
        if not placed:
            return None
    return np.array(centers)


def gen_lattice(P: int, phi: float, jitter: float, rng: np.random.Generator, max_attempts: int = 200):
    """Jittered primitive cubic lattice patch (sedimenting-drop structure), the P sites nearest the origin."""
    a = ((4.0 / 3.0) * np.pi / phi) ** (1.0 / 3.0)
    n = int(np.ceil((P ** (1.0 / 3.0)) / 2)) + 2
    grid = np.array(list(itertools.product(range(-n, n + 1), repeat=3)), dtype=np.float64) * a
    order = np.argsort(np.linalg.norm(grid, axis=1), kind="stable")
    base = grid[order[:P]]
    for _ in range(max_attempts):
        pos = base + rng.uniform(-jitter * a, jitter * a, size=base.shape)
        pos = pos - pos[0]                     # keep a particle at the origin (convention)
        if _min_gap(pos) >= GAP_MIN:
            return pos
    return None


def gen_chain(P: int, jitter: float, rng: np.random.Generator):
    """Quasi-1D chain along x: gaps iid uniform in [2.1, 8] (so adjacent pairs span the whole corrected
    range and next-nearest sums straddle the d = 8 cutoff), transverse Gaussian jitter of scale ``jitter``.
    Transverse jitter cannot shrink any centre distance below its axial gap, so gap >= 0.1 by construction."""
    pos = np.zeros((P, 3))
    pos[1:, 0] = np.cumsum(rng.uniform(2.1, 8.0, size=P - 1))
    if jitter > 0:
        pos[:, 1:] = rng.normal(scale=jitter, size=(P, 2))
        pos -= pos[0]                          # keep particle 0 at the origin (convention)
    return pos


def make_config(family: str, params: dict, P: int, rng: np.random.Generator):
    if family == "uniform":
        return gen_uniform(P, params["phi"], rng)
    if family == "grown":
        return gen_grown(P, params["delta"], rng)
    if family == "lattice":
        return gen_lattice(P, params["phi"], params["jitter"], rng)
    if family == "chain":
        return gen_chain(P, params["jitter"], rng)
    raise ValueError(family)


def config_seed(seed0: int, family: str, params: dict, P: int, shard: int, idx: int) -> int:
    key = [seed0, FAMILY_ID[family], P, shard, idx] + [int(round(v * 1e6)) for _, v in sorted(params.items())]
    return int(np.random.SeedSequence(key).generate_state(1, dtype=np.uint64)[0] % (2 ** 63 - 1))


def item_tag(params: dict, P: int, acc: str) -> str:
    return "_".join(f"{k}{v:g}" for k, v in sorted(params.items())) + f"_P{P}_{acc}"


# ----------------------------------------------------------------------------- plan
def build_plan(args) -> list:
    """List of items {family, params, P}; the default plan expands DEFAULT_PLAN."""
    items = []
    if args.plan == "default":
        fams = DEFAULT_PLAN
    else:
        assert args.family, "--family required without --plan default"
        fams = {args.family: {}}
        if args.family == "uniform":
            fams["uniform"] = {"phi": args.phi or DEFAULT_PLAN["uniform"]["phi"], "P": args.P or DEFAULT_PLAN["uniform"]["P"]}
        elif args.family == "grown":
            fams["grown"] = {"delta": args.delta or DEFAULT_PLAN["grown"]["delta"], "P": args.P or DEFAULT_PLAN["grown"]["P"]}
        elif args.family == "chain":
            fams["chain"] = {"jitter": args.jitter or CHAIN_JITTER, "P": args.P or CHAIN_P}
        else:
            fams["lattice"] = {"phi": args.phi or DEFAULT_PLAN["lattice"]["phi"],
                               "jitter": args.jitter or DEFAULT_PLAN["lattice"]["jitter"], "P": args.P or DEFAULT_PLAN["lattice"]["P"]}
    for family, spec in fams.items():
        keys = [k for k in spec if k != "P"]
        for combo in itertools.product(*[spec[k] for k in keys]):
            params = dict(zip(keys, combo))
            for P in spec["P"]:
                items.append({"family": family, "params": params, "P": int(P)})
    return items


# ----------------------------------------------------------------------------- io
def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def shard_path(out: Path, item: dict, acc: str, shard: int) -> Path:
    return out / item["family"] / item_tag(item["params"], item["P"], acc) / f"shard_{shard:04d}.npz"


def write_shard(path: Path, positions, Ms, seeds, iters, n_matvec, residual, dv, symm, wall, meta: dict,
                store_dtype: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:                 # file handle: numpy must not append ".npz" to the temp name
        np.savez(fh, positions=np.asarray(positions, dtype=np.float64),
                 M=np.asarray(Ms, dtype=np.float32 if store_dtype == "float32" else np.float64),
                 seed=np.asarray(seeds, dtype=np.int64), iters=np.asarray(iters, dtype=np.int32),
                 n_matvec=np.asarray(n_matvec, dtype=np.int32), residual=np.asarray(residual, dtype=np.float32),
                 dv=np.asarray(dv, dtype=np.float32), symm_err=np.asarray(symm, dtype=np.float32),
                 wall=np.asarray(wall, dtype=np.float32), meta=json.dumps(meta))
    os.replace(tmp, path)


def append_csv(path: Path, row: dict):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", choices=["default", "custom"], default="custom")
    ap.add_argument("--family", choices=list(FAMILY_ID))
    ap.add_argument("--phi", type=float, nargs="*")
    ap.add_argument("--delta", type=float, nargs="*")
    ap.add_argument("--jitter", type=float, nargs="*")
    ap.add_argument("--P", type=int, nargs="*")
    ap.add_argument("--n-configs", type=int, default=None, help="configs per item per round (default: shard size by P)")
    ap.add_argument("--acc", choices=["fine", "Xfine"], default="fine")
    ap.add_argument("--backend", choices=["torch64", "triton32"], default="torch64")
    ap.add_argument("--tol-v", type=float, default=None, help="velocity-change tolerance (default per backend)")
    ap.add_argument("--out", type=Path, default=Path("data/multibody_v2"))
    ap.add_argument("--seed0", type=int, default=20260830)
    ap.add_argument("--time-budget", type=float, default=None, help="seconds; stop starting new shards after this")
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--store-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--mem-budget-gb", type=float, default=4.0)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import torch
    from src.mfs_batched import BatchedMFS, MFSConvergenceError, SOLVER_VERSION

    t_start = time.time()
    items = build_plan(args)
    items = [it for i, it in enumerate(items) if i % args.num_workers == args.worker]
    if not items:
        print("no plan items for this worker")
        return
    solver = BatchedMFS(acc=args.acc, device=args.device, backend=args.backend, tol_v=args.tol_v,
                        mem_budget_gb=args.mem_budget_gb, verbose=args.verbose)
    manifest = args.out / f"manifest_worker{args.worker}.csv"
    failures = args.out / f"failures_worker{args.worker}.csv"
    meta_common = {"acc": args.acc, "backend": args.backend, "tol": solver.tol, "tol_v": solver.tol_v,
                   "method": "gmres", "solver_version": SOLVER_VERSION, "git": git_hash(), "seed0": args.seed0,
                   "store_dtype": args.store_dtype, "gap_min": GAP_MIN}
    print(f"[gen] worker {args.worker}/{args.num_workers}: {len(items)} plan items, backend={args.backend} "
          f"acc={args.acc} tol_v={solver.tol_v:g} out={args.out}", flush=True)

    round_idx = 0
    n_written = 0
    max_rounds = args.max_rounds
    if max_rounds is None and args.time_budget is None:
        max_rounds = 1                       # no budget given: a single pass over the plan
    while True:
        started_any = False
        for item in items:
            family, params, P = item["family"], item["params"], item["P"]
            # family interleaving: uniform every round, grown every 2nd, lattice every 4th (approx. 60/30/10)
            period = int(round(1.0 / FAMILY_ROUND_WEIGHT[family]))
            if round_idx % period != 0:
                continue
            shard = round_idx // period
            path = shard_path(args.out, item, args.acc, shard)
            if path.exists():
                continue
            if args.time_budget is not None and time.time() - t_start > args.time_budget:
                print(f"[gen] time budget exhausted after {n_written} shards", flush=True)
                return
            started_any = True
            n_cfg = args.n_configs or SHARD_SIZE.get(P, 256)
            # ---- configurations
            cfgs, seeds = [], []
            for idx in range(n_cfg):
                seed = config_seed(args.seed0, family, params, P, shard, idx)
                pos = make_config(family, params, P, np.random.default_rng(seed))
                if pos is None or _min_gap(pos) < GAP_MIN - 1e-12:
                    append_csv(failures, {"family": family, "params": json.dumps(params), "P": P, "shard": shard,
                                          "idx": idx, "seed": seed, "reason": "generator"})
                    continue
                cfgs.append(pos)
                seeds.append(seed)
            if not cfgs:
                continue
            # ---- solve (batched on the GPU, chunked by the memory budget inside the solver)
            t0 = time.time()
            try:
                Ms, infos = solver.solve_mobility_matrix_batch(cfgs, raise_on_fail=False)
            except MFSConvergenceError as e:  # pragma: no cover (raise_on_fail=False)
                print(f"[gen] batch failed: {e}", flush=True)
                continue
            keep = [i for i, inf in enumerate(infos) if inf.converged]
            for i, inf in enumerate(infos):
                if not inf.converged:
                    append_csv(failures, {"family": family, "params": json.dumps(params), "P": P, "shard": shard,
                                          "idx": i, "seed": seeds[i], "reason": f"noconv res={inf.max_rel_residual:.2e} dv={inf.max_rel_dv:.2e}"})
            if not keep:
                continue
            wall = (time.time() - t0) / len(cfgs)
            meta = dict(meta_common, family=family, params=params, P=P, shard=shard, n=len(keep),
                        created=time.strftime("%Y-%m-%d %H:%M:%S"))
            write_shard(path, [cfgs[i] for i in keep], [Ms[i].cpu().numpy() for i in keep], [seeds[i] for i in keep],
                        [infos[i].iters for i in keep], [infos[i].n_matvec for i in keep],
                        [infos[i].max_rel_residual for i in keep], [infos[i].max_rel_dv for i in keep],
                        [infos[i].symm_err for i in keep], [wall] * len(keep), meta, args.store_dtype)
            append_csv(manifest, {"path": str(path.relative_to(args.out)), "family": family, "params": json.dumps(params),
                                  "P": P, "acc": args.acc, "n": len(keep), "n_failed": len(cfgs) - len(keep),
                                  "seed_first": seeds[0], "mean_iters": float(np.mean([infos[i].iters for i in keep])),
                                  "mean_wall_s": wall, "max_residual": float(max(infos[i].max_rel_residual for i in keep)),
                                  "max_dv": float(max(infos[i].max_rel_dv for i in keep)),
                                  "max_symm_err": float(max(infos[i].symm_err for i in keep)),
                                  "elapsed_s": time.time() - t_start})
            n_written += 1
            print(f"[gen] {path.relative_to(args.out)}: n={len(keep)} P={P} {wall:.2f} s/config "
                  f"iters~{np.mean([infos[i].iters for i in keep]):.0f} symm<={max(infos[i].symm_err for i in keep):.1e} "
                  f"[{(time.time() - t_start) / 60:.1f} min]", flush=True)
        round_idx += 1
        if max_rounds is not None and round_idx >= max_rounds:
            break
    print(f"[gen] done: {n_written} shards in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
