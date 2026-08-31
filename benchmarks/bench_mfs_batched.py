#!/usr/bin/env python3
"""Performance of the batched MFS solver (src/mfs_batched.py): operator throughput, matvec cost, and
seconds per configuration for the full grand mobility matrix, by backend / accuracy / P.

    python benchmarks/bench_mfs_batched.py                       # default sweep -> data/mfs_batched_perf.csv
    python benchmarks/bench_mfs_batched.py --backends torch64 --P 16 32 64 --acc fine
    python benchmarks/bench_mfs_batched.py --baseline            # also time the reference single-RHS Triton solver

Run on the machine that will generate the data (the H200 has fp64 at half the fp32 rate; this laptop's
GeForce has 1/64, so torch64 numbers here are ~50x pessimistic).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(ROOT)]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.mfs_batched import BatchedMFS  # noqa: E402
from src.create_dataset_multibody_v2 import gen_uniform  # noqa: E402


def time_op(fn, n=3):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["triton32", "torch64"])
    ap.add_argument("--acc", nargs="+", default=["fine"])
    ap.add_argument("--P", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--phi", type=float, default=0.15)
    ap.add_argument("--n-sys", type=int, default=1, help="configurations batched per solve (small P)")
    ap.add_argument("--mem-budget-gb", type=float, default=4.0)
    ap.add_argument("--csv", type=Path, default=Path("data/mfs_batched_perf.csv"))
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()

    rows = []
    for acc in args.acc:
        for backend in args.backends:
            solver = BatchedMFS(acc=acc, backend=backend, mem_budget_gb=args.mem_budget_gb)
            for P in args.P:
                R = 6 * P
                pos = gen_uniform(P, args.phi, np.random.default_rng(P))
                ctx = solver._make_ctx([pos] * args.n_sys, R)
                S = torch.randn((3 * solver.M, ctx.LD), dtype=solver.wdtype, device=solver.device)
                t_kernel = time_op(lambda: solver.oseen_offdiag(S, ctx))
                W = solver.oseen_offdiag(S, ctx)
                t_gemm = time_op(lambda: solver.apply_Kf(W))
                flops = args.n_sys * P * (P - 1) * 3 * solver.N * 3 * solver.M * R * 2
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                if args.n_sys == 1:
                    Mm, info = solver.solve_mobility_matrix(pos, raise_on_fail=False)
                    infos = [info]
                else:
                    Ms, infos = solver.solve_mobility_matrix_batch([pos] * args.n_sys, raise_on_fail=False)
                torch.cuda.synchronize()
                t_cfg = (time.time() - t0) / args.n_sys
                mem = torch.cuda.max_memory_allocated() / 2 ** 30
                row = {"acc": acc, "backend": backend, "P": P, "R": R, "n_sys": args.n_sys,
                       "kernel_ms": t_kernel * 1e3, "kernel_tflops": flops / t_kernel / 1e12,
                       "gemm_ms": t_gemm * 1e3, "matvec_ms": (t_kernel + t_gemm) * 1e3,
                       "iters": infos[0].iters, "n_matvec": infos[0].n_matvec, "converged": infos[0].converged,
                       "max_rel_dv": infos[0].max_rel_dv, "symm_err": infos[0].symm_err,
                       "s_per_config": t_cfg, "configs_per_hour": 3600.0 / t_cfg, "peak_mem_gib": mem,
                       "gpu": torch.cuda.get_device_name(0)}
                rows.append(row)
                print(f"{acc:>5} {backend:>8} P={P:3d} R={R:3d} x{args.n_sys}: op {row['kernel_ms']:8.1f} ms "
                      f"({row['kernel_tflops']:5.2f} TFLOPS) gemm {row['gemm_ms']:7.1f} ms | grand M: {infos[0].iters:2d} it, "
                      f"{t_cfg:7.2f} s/config, {row['configs_per_hour']:7.0f}/h, dv {infos[0].max_rel_dv:.1e}, "
                      f"peak {mem:.2f} GiB", flush=True)
    df = pd.DataFrame(rows)
    if args.csv.exists():
        df = pd.concat([pd.read_csv(args.csv), df], ignore_index=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False, float_format="%.5g")
    print(f"-> {args.csv}")

    if args.baseline:
        from src.triton_mfs import MobMFSTriton
        df = pd.read_csv("tmp/testcase_uniform_0.15_40.csv")
        config = df[["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].values
        forces = df[["f_x", "f_y", "f_z", "t_x", "t_y", "t_z"]].values
        mob = MobMFSTriton("sphere", "fine")
        t0 = time.time()
        mob.apply(config, forces, 1.0)
        torch.cuda.synchronize()
        print(f"baseline MobMFSTriton fine, P=40, 1 RHS: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
