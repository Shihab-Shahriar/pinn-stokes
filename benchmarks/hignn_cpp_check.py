#!/usr/bin/env python3
"""Cross-check of HIGNN's own C++/Kokkos engine against src/hignn_ops.py (the pure-torch 2-body operator).

--run      inside their container, from the hignn checkout with python/hignn.so built (python3 python/init.py
           --rebuild): for each truth npz, their engine's dense_dot (exact O(N^2) sum of the TorchScript kernel)
           and dot (H-matrix with ACA far field) at the epsilon / max_iter settings their scripts use; saves the
           velocities (HIGNN units, float32 as the engine writes them) and wall times. Imports only numpy + hignn.
--compare  in this repo's env: rel-L2 of each engine result vs HignnMob("2b") and translational PRMSE vs truth;
           writes data/hignn_cpp_check.csv.

  # cluster, inside `singularity exec ... hignn_cpu.sif` with the checkout at /workspace and truths at /truth:
  python3 hignn_cpp_check.py --run --hignn-root /workspace --out /workspace/cpp_check.npz /truth/uniform_N200_phi0.1_seed4423_grav.npz ...
  # laptop:
  python benchmarks/hignn_cpp_check.py --compare --out tmp/cpp_check.npz --truth-dir tmp/nbody_moments_truth
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# name, epsilon, max_iter  (None = dense_dot).  Defaults per HignnModel; the other two per their driver scripts.
SETTINGS = [("dense", None, None), ("aca_eps0.05_it100", 0.05, 100),
            ("aca_eps0.1_it15", 0.1, 15), ("aca_eps0.01_it50", 0.01, 50)]


def run(args):
    root = Path(args.hignn_root).resolve()
    os.chdir(root)                                   # load_two_body_model('nn/two_body_unbounded') is relative
    sys.path.insert(0, str(root / "python"))
    import hignn  # noqa: E402  (python/hignn.so)
    hignn.Init()
    out = {"files": np.array([str(Path(f).name) for f in args.files])}
    for f in args.files:
        d = np.load(f)
        X = np.ascontiguousarray(d["config"][:, :3], dtype=np.float32)
        Fv = np.ascontiguousarray(d["forces"][:, :3], dtype=np.float32)
        N = X.shape[0]
        stem = Path(f).stem
        m = hignn.HignnModel(X, 100 if N > 100 else N // 2)      # simulate.py's block-size rule
        m.load_two_body_model("nn/two_body_unbounded")
        m.set_post_check_flag(False)
        m.set_use_symmetry_flag(True)
        m.set_mat_pool_size_factor(200)                           # simulate.py's large-N settings
        m.set_max_far_dot_work_node_size(10000)
        m.set_max_relative_coord(1000000)
        t0 = time.time()
        m.update_coord(X)
        out[f"{stem}__update_wall"] = time.time() - t0
        for name, eps, it in SETTINGS:
            if name == "dense" and N > args.dense_max_n:
                continue
            u = np.zeros((N, 3), dtype=np.float32)                # MUST be float32: a float64 array is silently copied
            t0 = time.time()
            if name == "dense":
                m.dense_dot(u, Fv)
            else:
                m.set_epsilon(eps)
                m.set_max_iter(it)
                m.dot(u, Fv)
            wall = time.time() - t0
            out[f"{stem}__{name}"] = u.copy()
            out[f"{stem}__{name}__wall"] = wall
            print(f"{stem} {name:18s} N={N:6d} wall={wall:8.2f}s  mean|u|={np.linalg.norm(u, axis=1).mean():.6f}", flush=True)
            np.savez(args.out, **out)                             # crash-safe: their engine can segfault at large N
        del m
    np.savez(args.out, **out)
    print(f"-> {args.out}")
    hignn.Finalize()


def compare(args):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.hignn_ops import SIX_PI, HignnMob  # noqa: E402
    zs = [np.load(f, allow_pickle=True) for f in [args.out] + args.files]
    z = {k: v for zz in zs for k, v in zz.items() if k != "files"}
    z["files"] = np.concatenate([zz["files"] for zz in zs])
    op = HignnMob(variant="2b")
    rows = []
    for name in z["files"]:
        if not any(str(k).startswith(Path(str(name)).stem + "__") for k in z):
            print(f"[skip] no engine rows for {name}")
            continue
        stem = Path(str(name)).stem
        d = np.load(Path(args.truth_dir) / str(name))
        V = d["velocity"][:, :3]
        ref = op.apply(d["config"], d["forces"])[:, :3] * SIX_PI          # HIGNN units, float64
        base = 100 * np.linalg.norm(ref / SIX_PI - V) / np.linalg.norm(V)
        rows.append(dict(case=stem, N=len(V), engine="torch_dense(src/hignn_ops.py)", rel_l2_vs_torch=0.0,
                         prmse_lin=base, wall_s=op.last.get("wall_s", np.nan)))
        for sname, eps, it in SETTINGS:
            k = f"{stem}__{sname}"
            if k not in z:
                continue
            u = z[k].astype(np.float64)
            rows.append(dict(case=stem, N=len(V), engine=sname, epsilon=eps, max_iter=it,
                             rel_l2_vs_torch=float(np.linalg.norm(u - ref) / np.linalg.norm(ref)),
                             prmse_lin=100 * np.linalg.norm(u / SIX_PI - V) / np.linalg.norm(V),
                             wall_s=float(z[f"{k}__wall"])))
    import pandas as pd
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3e}" if abs(x) < 1e-2 and x != 0 else f"{x:.4f}"))
    out_csv = root / "data" / "hignn_cpp_check.csv"
    df.to_csv(out_csv, index=False)
    print(f"-> {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="truth npz files (--run); extra result npz files (--compare)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--hignn-root", default="/workspace")
    ap.add_argument("--out", default="cpp_check.npz")
    ap.add_argument("--truth-dir", default="tmp/nbody_moments_truth")
    ap.add_argument("--dense-max-n", type=int, default=3000, help="skip dense_dot above this N")
    args = ap.parse_args()
    assert args.run != args.compare, "pick exactly one of --run / --compare"
    (run if args.run else compare)(args)


if __name__ == "__main__":
    main()
