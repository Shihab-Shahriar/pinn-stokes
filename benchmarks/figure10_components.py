"""Figure 10: performance of NeMO's individual components.

Two panels, two producers, one file:

  (a) --panel left
      Two-body pair-kernel throughput -- the analytic RPY kernel versus the
      learned cross kernel f_t -- against batch size. This reproduces the
      published Figure 10, which had NO plotting code anywhere in the repo or
      its history: the numbers were transcribed by hand from the stdout of
      benchmarks/bench_rpy.py and benchmarks/benchmark.py. This script drives
      those same two `measure_throughput` functions, so the panel becomes
      reproducible without changing what it measures.

  (b) --panel right
      The far-field treecode (WidebvhFMM, bary PDEG 7, mac 0.8) against
      particle count N: throughput, the far-field relative L2 error versus a
      sampled fp64 direct sum, and the grand operator's relative asymmetry.
      The claim being made is that the last two are FLAT in N -- the multipole
      acceptance criterion is scale-free, so the far field contributes a fixed,
      negligible share of the error budget at every problem size.

Why this cannot be one process
------------------------------
The panels need opposite torch.compile settings, and `TORCH_COMPILE_DISABLE` is
consumed when torch._dynamo is first imported -- it cannot be toggled later.

  (a) needs compile ENABLED: compiled-kernel throughput is the entire
      measurement. If TORCH_COMPILE_DISABLE is inherited from the shell (and
      CLAUDE.md tells you to export it for accuracy work, so it lingers),
      torch.compile silently becomes a no-op and the "compile" rows come out
      equal to the "eager" rows with no error.
  (b) runs with TORCH_COMPILE_DISABLE=1, the repo convention for accuracy runs.
      That costs it nothing: the quantity it times is the far field, a ctypes
      call into libwidebvh_nemo.so which torch.compile cannot touch. Only the
      near field slows down, and the near field is not timed here.

`--panel both` therefore re-execs this file twice with the right environment.

Usage
-----
    source ~/warp_env.sh
    cd <repo root>              # model paths in symmetry_treecode are cwd-relative
    python benchmarks/figure10_components.py --panel both
    python figures/component_performance.py

    # smoke test the expensive panel first
    python benchmarks/figure10_components.py --panel right --sizes 5000,10000,100000
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# insert(0), never append: PYTHONPATH carries an older /mnt/home/khanmd/pinn-stokes
# checkout that contains both `src/` and `benchmarks/`, so an appended path leaves
# `import benchmarks.bench_rpy` resolving to the wrong tree. Same fix as
# two_suspensions_1M.py / symmetry_treecode.py / large_scale_dynamics.py.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAIR_CSV = ROOT / "data" / "fig10_pair_kernels.csv"
FAR_CSV = ROOT / "data" / "fig10_far_field.csv"

# The verbatim list from bench_rpy.bench() and benchmark.bench(). Swept ascending:
# bench_rpy caches one dynamic=True graph in _COMPILED_TWO_BODY across all sizes,
# so only the first size pays compilation -- pay it at the cheapest one.
PAIR_BATCHES = (16384, 24064, 32768, 65536, 2**17, 2**18, 2**19, 2**20, 2**21, 2**22)

# 2M and 4M were generated for this figure with
#   cluster.uniform_cluster_generation_large(0.1, N, seed=0)
# (the pre-existing files up to 1M carry no recorded seed). 5k and 10k stay in
# the sweep -- they are measured and kept in the CSV -- but the plotter drops
# them: below ~50k the far field is pinned by its fixed ~7 ms build cost.
FAR_SIZES = (5_000, 10_000, 50_000, 100_000, 200_000, 300_000,
             400_000, 500_000, 750_000, 1_000_000, 2_000_000, 4_000_000)

PAIR_FIELDS = ("kernel", "batch", "mode", "protocol", "n_warmup", "n_iter",
               "throughput_per_sec", "dtype", "matmul_precision",
               "peak_vram_gb", "gpu", "torch_version", "git_sha")

FAR_FIELDS = ("backend", "n", "mac", "pdeg", "max_leaf", "near_cutoff", "loading",
              "far_ms", "far_ms_std", "far_updates_per_sec", "far_repeats",
              "near_pairs", "num_nodes", "num_source_buckets",
              "traverse_ms", "p2p_ms", "build_ms", "upward_ms",
              "rel_far", "rel_total", "ref_samples", "ref_cutoff", "ref_s",
              "rel_asym", "rel_asym_stderr", "fro_asym", "fro_M",
              "hutch_K", "hutch_seed", "hutch_wall_s",
              "nbody_num_pairs", "pair_chunk_size", "nbody_chunks",
              "peak_vram_gb", "compile_disabled", "gpu", "git_sha")

# Published rel_asym for the full grand operator at mac 0.8, post-chunking-fix,
# from data/symmetry_widebvh.csv (phase `postfix_full`, K=24, seed=2). Used only
# to print a regression table -- those rows are hand-assembled and untracked, so
# they are a cross-check, not an input.
KNOWN_REL_ASYM = {10_000: 3.2261e-04, 100_000: 4.1242e-04, 300_000: 3.9972e-04,
                  500_000: 4.4618e-04, 750_000: 4.2977e-04, 1_000_000: 4.6103e-04}

# data/widebvh_mac_calibration.csv, uniform100k / random / bary p7 / mac 0.8.
KNOWN_REL_FAR = {100_000: 3.853e-04}


def git_sha() -> str:
    """Copied from figure12_grand_M.py rather than imported.

    Importing benchmarks.performance_grand_M for it would be a trap: that module
    sets inductor_config.triton.cudagraphs=False and torch.set_grad_enabled(False)
    at import time, process-globally, which would move the left panel's numbers.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_rows(path: Path, fields, rows, key_fn):
    """Rewrite only the rows this run measured; keep everything else.

    Deviation from figure12_grand_M.py, which rewrites a whole backend: the key
    here includes N, so `--sizes 1000000` can top up a single row without
    discarding the other nine of a long sweep.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    measured = {key_fn(r) for r in rows}
    kept = []
    if path.exists():
        with path.open() as fh:
            kept = [r for r in csv.DictReader(fh) if key_fn(r) not in measured]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
        w.writerows(rows)


# ----------------------------------------------------------------------------
# panel (a): two-body pair kernels
# ----------------------------------------------------------------------------

def run_left(args) -> None:
    for var in ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE"):
        assert not os.environ.get(var), (
            f"{var}={os.environ[var]!r} is set. This is a PERFORMANCE run and "
            f"torch.compile would silently become a no-op, making the 'compile' "
            f"rows identical to the 'eager' ones. Run `unset {var}` first.")

    import torch
    from benchmarks import bench_rpy

    dev = torch.device("cuda")
    sha, gpu, tver = git_sha(), torch.cuda.get_device_name(0), torch.__version__
    batches = [b for b in PAIR_BATCHES if b <= args.max_batch]
    rows = []

    # RPY first, and benchmarks.benchmark imported only afterwards: that module
    # sets torch.set_float32_matmul_precision('high') at import, process-wide.
    # bench_rpy has no matmul/einsum/bmm (checked), so TF32 cannot reach it --
    # but measuring RPY before the global is touched keeps that a fact rather
    # than an assumption, and matches what `python benchmarks/bench_rpy.py` does.
    print("\n########## panel (a): RPY pair kernel ##########", flush=True)
    for mode in ("eager", "compile"):
        for batch in batches:
            torch.cuda.reset_peak_memory_stats()
            t = bench_rpy.measure_throughput(
                batch, n_warmup=args.warmup, n_iter=args.iters,
                dtype=torch.float32, use_compile=(mode == "compile"))
            rows.append(dict(
                kernel="rpy", batch=batch, mode=mode, protocol="matched",
                n_warmup=args.warmup, n_iter=args.iters,
                throughput_per_sec=t, dtype="float32",
                matmul_precision=torch.get_float32_matmul_precision(),
                peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3,
                gpu=gpu, torch_version=tver, git_sha=sha))
            print(f"  rpy/{mode:<7} batch={batch:>8,} -> {t/1e6:8.2f} M eval/s",
                  flush=True)

    # The published protocol for RPY is (10, 50), i.e. already `matched`; only
    # f_t differs. Re-timing RPY under the same numbers would duplicate rows.

    print("\n########## panel (a): learned f_t kernel ##########", flush=True)
    from benchmarks import benchmark as nn_bench  # noqa: E402  (see comment above)

    base_model = nn_bench.ScNetwork(nn_bench.input_dim).to(dev)
    models = {"eager": base_model,
              "compile": torch.compile(base_model, fullgraph=False)}

    # (1, 3) is what benchmark.bench() ships and is presumably what produced the
    # published 495 M/s; it puts compilation inside the timed window and samples
    # a sub-10 ms kernel three times. (10, 50) is the honest protocol. Record
    # both: raising f_t's iteration count can only flatter the LEARNED kernel, so
    # the change cannot be read as favouring the RPY baseline.
    protocols = {"published": (1, 3), "matched": (args.warmup, args.iters)}

    for mode, model in models.items():
        for proto, (nw, ni) in protocols.items():
            for batch in batches:
                torch.cuda.reset_peak_memory_stats()
                # benchmark.measure_throughput returns M samples/s; the CSV is
                # raw samples/s so both kernels share one unit.
                t = nn_bench.measure_throughput(model, dev, batch,
                                                n_warmup=nw, n_iter=ni) * 1e6
                rows.append(dict(
                    kernel="ft", batch=batch, mode=mode, protocol=proto,
                    n_warmup=nw, n_iter=ni,
                    throughput_per_sec=t, dtype="float32",
                    matmul_precision=torch.get_float32_matmul_precision(),
                    peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3,
                    gpu=gpu, torch_version=tver, git_sha=sha))
                print(f"  ft/{mode:<7}/{proto:<9} batch={batch:>8,} -> "
                      f"{t/1e6:8.2f} M samples/s", flush=True)

    write_rows(PAIR_CSV, PAIR_FIELDS, rows,
               lambda r: (str(r["kernel"]), str(r["mode"]), str(r["protocol"]),
                          int(r["batch"])))
    print(f"\nwrote {len(rows)} rows -> {PAIR_CSV}")

    top = max(batches)
    for kern, proto, published in (("rpy", "matched", 1023.0),
                                   ("ft", "published", 495.0)):
        got = [r for r in rows if r["kernel"] == kern and r["mode"] == "compile"
               and r["protocol"] == proto and r["batch"] == top]
        if got:
            v = got[0]["throughput_per_sec"] / 1e6
            print(f"  check {kern}@{top:,} ({proto}): {v:.1f} M/s  "
                  f"vs published {published:.0f} M/s  ({v/published:.2f}x)")


# ----------------------------------------------------------------------------
# panel (b): far-field treecode vs N
# ----------------------------------------------------------------------------

def run_right(args) -> None:
    # Must precede `import torch`: torch._dynamo reads this when it is imported
    # and the setting cannot be changed afterwards.
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    import numpy as np
    import torch

    assert os.environ.get("TORCH_COMPILE_DISABLE") == "1"
    assert (ROOT / "data" / "models" / "combined_2body.wt").exists()
    # symmetry_treecode resolves SELF_NN/TWO_NN/NBODY_NN against the CWD.
    assert (Path.cwd() / "data" / "models" / "combined_2body.wt").exists(), (
        f"run from the repo root (cwd is {Path.cwd()}):\n"
        f"  cd {ROOT} && python benchmarks/figure10_components.py --panel right")

    from benchmarks import mac_calibration as mc
    from benchmarks import symmetry_treecode as sym
    from src.treecode_widebvh import WidebvhFMM

    dev = torch.device("cuda")
    sha, gpu = git_sha(), torch.cuda.get_device_name(0)
    sizes = args.sizes or list(FAR_SIZES)
    rows = []

    for n in sizes:
        cfg = ROOT / "tmp" / f"uniform_large_0.1_{n}.csv"
        assert cfg.exists(), f"missing config {cfg}"
        print(f"\n########## panel (b): N={n:,} ##########", flush=True)

        # load_config uses pandas; mac_calibration.load_csv_positions is
        # np.loadtxt and takes 60-90 s on the 54 MB N=1M file.
        pos, orient = sym.load_config(str(cfg))
        assert pos.shape[0] == n, f"{n} requested, CSV has {pos.shape[0]}"
        pos_t, orient_t, vis = sym.to_gpu(pos, orient)

        f_np = mc.make_force(n, args.loading)
        f3 = torch.as_tensor(f_np, dtype=torch.float32, device=dev).contiguous()

        near_op = sym.make_nbody(far_field=None)
        # Constructed directly, not via sym.make_solver: that helper reads `mac`
        # from a module constant frozen at import from $NEMO_MAC, which would let
        # a stale env var silently override the value recorded in the CSV.
        solver = WidebvhFMM(near_field_operator=near_op, mac=args.mac,
                            near_field_cutoff=mc.CUTOFF, policy="bary",
                            pdeg=args.pdeg, max_leaf=args.max_leaf, device="cuda")
        torch.cuda.reset_peak_memory_stats()

        # -- phase A: far-field throughput ---------------------------------
        # First, on the cleanest GPU: no fp64 reference buffers and no probe
        # storage resident yet.
        with mc.quiet():
            for _ in range(3):
                solver.get_far_field_vel(pos_t, f3)
            samples_ms = []
            for _ in range(args.far_repeats):
                solver.get_far_field_vel(pos_t, f3)
                samples_ms.append(float(solver.last_stats["far_ms"]))
            st = dict(solver.last_stats)

        trimmed = sorted(samples_ms)[1:-1] if len(samples_ms) > 2 else samples_ms
        far_ms = float(np.mean(trimmed))
        far_std = float(np.std(trimmed))
        updates = n / (far_ms * 1e-3)
        print(f"  far field   {far_ms:8.3f} ms (+-{far_std:.3f})  "
              f"{updates/1e6:6.2f} M updates/s", flush=True)

        # -- phase B: far-field accuracy vs a sampled fp64 direct sum -------
        s = min(args.ref_samples, n)
        sample = torch.linspace(0, n - 1, s, device=dev).long()
        pos64 = pos_t.double()
        f64 = torch.as_tensor(f_np, dtype=torch.float64, device=dev).contiguous()

        t0 = time.perf_counter()
        ref = mc.far_ref_tt(pos64, f64, sample, cutoff=mc.CUTOFF)
        ref_s = time.perf_counter() - t0

        with mc.quiet():
            # Indexed and cloned in one expression: get_far_field_vel returns a
            # VIEW on a buffer the next call overwrites.
            u = solver.get_far_field_vel(pos_t, f3)[sample].clone()

        err = float(torch.linalg.norm(u - ref))
        rel_far = err / float(torch.linalg.norm(ref))
        # Denominator is the analytic self term plus the far field -- the same
        # convention as mac_calibration.run_case, so these are directly
        # comparable to data/widebvh_mac_calibration.csv.
        self_u = f64[sample] / (6.0 * math.pi * mc.RADIUS)
        rel_total = err / float(torch.linalg.norm(self_u + ref))
        print(f"  rel_far     {rel_far:.4e}   rel_total {rel_total:.4e}   "
              f"(ref {ref_s:.1f} s, {s} targets)", flush=True)

        del pos64, f64, ref, u, self_u
        torch.cuda.empty_cache()

        # -- phase C: symmetry of the FULL grand operator -------------------
        num_pairs, n_chunks = float("nan"), float("nan")
        try:
            with mc.quiet():
                edges = solver.get_edge_indexes(pos_t, mc.CUTOFF)
            num_pairs = int(edges[0].numel())
            n_chunks = math.ceil(num_pairs / near_op.pair_chunk_size)
        except Exception as exc:          # provenance only, never fatal
            print(f"  (edge count unavailable: {type(exc).__name__}: {exc})")

        # .clone() is mandatory: apply() returns the near-field buffer it
        # mutates in place (src/treecode.py:430).
        apply_fn = lambda f: solver.apply(pos_t, orient_t, f, vis).clone()
        with mc.quiet():
            apply_fn(torch.zeros(n, 6, device=dev))   # warm hash grid + topk

        t0 = time.perf_counter()
        hs = sym.hutchinson(apply_fn, n, K=args.K, seed=args.seed)
        hutch_s = time.perf_counter() - t0
        print(f"  rel_asym    {hs['rel_asym']:.4e} +- {hs['rel_asym_stderr']:.1e}  "
              f"(K={args.K}, {hutch_s:.1f} s)", flush=True)

        rows.append(dict(
            backend="widebvh", n=n, mac=args.mac, pdeg=args.pdeg,
            max_leaf=args.max_leaf, near_cutoff=mc.CUTOFF, loading=args.loading,
            far_ms=far_ms, far_ms_std=far_std, far_updates_per_sec=updates,
            far_repeats=args.far_repeats,
            near_pairs=int(st.get("near_pairs", 0)),
            num_nodes=int(st.get("num_nodes", 0)),
            num_source_buckets=int(st.get("num_source_buckets", 0)),
            traverse_ms=st.get("traverse_ms", ""), p2p_ms=st.get("p2p_ms", ""),
            build_ms=st.get("build_bvh_ms", 0.0) + st.get("bucket_ms", 0.0),
            upward_ms=st.get("upward_ms", ""),
            rel_far=rel_far, rel_total=rel_total, ref_samples=s,
            ref_cutoff=mc.CUTOFF, ref_s=ref_s,
            rel_asym=hs["rel_asym"], rel_asym_stderr=hs["rel_asym_stderr"],
            fro_asym=hs["fro_asym"], fro_M=hs["fro_M"],
            hutch_K=args.K, hutch_seed=args.seed, hutch_wall_s=hutch_s,
            nbody_num_pairs=num_pairs, pair_chunk_size=near_op.pair_chunk_size,
            nbody_chunks=n_chunks,
            peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3,
            compile_disabled=1, gpu=gpu, git_sha=sha))

        # Flush every N: a crash at 1M must not discard nine good rows.
        write_rows(FAR_CSV, FAR_FIELDS, rows,
                   lambda r: (str(r["backend"]), int(r["n"])))

        solver.close()
        del solver, near_op, pos_t, orient_t, vis, f3, apply_fn
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nwrote {len(rows)} rows -> {FAR_CSV}")

    print("\n N            far ms   M upd/s     rel_far    rel_asym   vs known")
    for r in rows:
        known = KNOWN_REL_ASYM.get(r["n"])
        delta = f"{r['rel_asym']/known:6.3f}x" if known else "     --"
        print(f" {r['n']:>9,}  {r['far_ms']:>9.3f}  {r['far_updates_per_sec']/1e6:>7.2f}  "
              f"{r['rel_far']:>10.3e}  {r['rel_asym']:>10.3e}  {delta}")

    for r in rows:
        ref_far = KNOWN_REL_FAR.get(r["n"])
        if ref_far and abs(r["rel_far"] / ref_far - 1.0) > 0.05:
            print(f"  !! rel_far at N={r['n']:,} is {r['rel_far']:.4e}, expected "
                  f"~{ref_far:.4e} -- check the reference wiring")

    vals = [r["rel_far"] for r in rows]
    asym = [r["rel_asym"] for r in rows]
    if len(vals) > 1:
        print(f"\n  flatness: rel_far spans {max(vals)/min(vals):.2f}x, "
              f"rel_asym spans {max(asym)/min(asym):.2f}x across "
              f"N={min(sizes):,}..{max(sizes):,}")


# ----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", choices=("left", "right", "both"), default="both")
    p.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",") if x],
                   default=None, help="panel (b) particle counts")
    p.add_argument("--max-batch", type=int, default=2**22,
                   help="panel (a) largest batch size")
    p.add_argument("--warmup", type=int, default=10, help="panel (a) warmup iters")
    p.add_argument("--iters", type=int, default=50, help="panel (a) timed iters")
    p.add_argument("--mac", type=float, default=0.8)
    p.add_argument("--pdeg", type=int, default=7)
    p.add_argument("--max-leaf", type=int, default=1024)
    p.add_argument("--loading", choices=("random", "gravity"), default="random",
                   help="random is the binding case: at mac 0.8 rel_far is "
                        "3.9e-4 under random but 8.7e-7 under coherent gravity")
    p.add_argument("--ref-samples", type=int, default=2048)
    p.add_argument("--far-repeats", type=int, default=7)
    p.add_argument("--K", type=int, default=24, help="Hutchinson probes")
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()

    if args.panel == "both":
        # Re-exec rather than loop: the two panels need opposite values of
        # TORCH_COMPILE_DISABLE and it is latched at torch._dynamo import.
        for panel, extra in (("right", {"TORCH_COMPILE_DISABLE": "1"}),
                             ("left", {})):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE")}
            env.update(extra)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--panel", panel]
            cmd += [a for a in sys.argv[1:]
                    if a not in ("--panel", "both")]
            print(f"\n=== {' '.join(cmd)} ===", flush=True)
            rc = subprocess.call(cmd, env=env, cwd=str(ROOT))
            if rc != 0:
                sys.exit(rc)
        return

    (run_left if args.panel == "left" else run_right)(args)


if __name__ == "__main__":
    main()
