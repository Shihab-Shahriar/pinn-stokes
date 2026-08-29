"""Pick the far-field operating point (mac, PDEG, maxLeaf) for NeMO.

The widebvh treecode was tuned for 1e-7..1e-8 relative error. NeMO does not need
that: its learned near field carries ~7.5% PRMSE at 10% volume fraction, so any
far-field error well under ~1e-3 of the total velocity is invisible end to end.
This script measures where that line actually falls, so the accuracy we do not
need can be traded for speed.

Reference is a dense float64 RPY translation-translation sum over r >= cutoff --
the same algebra as `exact_far_tt` in benchmarks/symmetry_treecode.py, but
evaluated only at sampled target rows so it scales to 1M particles (the dense
(3N,3N) form is 679 MB at N=3071 and impossible above that).

Two error measures are reported, and they answer different questions:

  rel_far   = ||u_tree - u_ref|| / ||u_ref||
      the treecode's own truncation error. Comparable to widebvh's published
      relL2err and to the theta sweep in artifacts/treecode_symmetry_report.md.

  rel_total = ||u_tree - u_ref|| / ||u_self + u_near + u_ref||
      the same error as a fraction of the velocity NeMO actually reports. This
      is the acceptance metric: it is what propagates into PRMSE.

`--policy` picks the multipole expansion. They calibrate on different axes --
bary on the compile-time Chebyshev degree (`--pdegs`), Cartesian on the runtime
Taylor order (`--orders`) -- and a `mac` from one is meaningless for the other,
so their results live in separate CSVs.

Usage:
    source ~/warp_env.sh
    export TORCH_COMPILE_DISABLE=1          # accuracy run
    python benchmarks/mac_calibration.py --case uniform100k
    python benchmarks/mac_calibration.py --case all --csv data/widebvh_mac_calibration.csv
    python benchmarks/mac_calibration.py --policy cart --orders 2,3,4 \
        --macs 0.2,0.25,0.3,0.4,0.5 --thetas "" \
        --csv data/cartesian_mac_calibration.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CUTOFF = 6.0
RADIUS = 1.0
PREF_TT = 1.0 / (8.0 * math.pi)

WIDEBVH_DISTROS = Path("/mnt/ffs24/home/khanmd/programs/widebvh/distros")


# ----------------------------------------------------------------------------
# reference
# ----------------------------------------------------------------------------
def far_ref_tt(pos: torch.Tensor, force: torch.Tensor, sample: torch.Tensor,
               cutoff: float = CUTOFF, budget_bytes: float = 1.5e9
               ) -> torch.Tensor:
    """Exact fp64 RPY-TT far field (r >= cutoff) at the sampled target rows.

    pos/force are (N,3); `sample` indexes the targets to evaluate. Chunked over
    targets so the (block,N,3) intermediates stay inside `budget_bytes`, which
    is what makes N = 1e6 feasible at all: a fixed block of 512 would want 12 GB
    for rvec alone. Cost is O(S*N), not O(N^2).
    """
    p = pos.double()
    f = force.double()
    n = p.shape[0]
    a2 = RADIUS * RADIUS
    # rvec dominates: (block, N, 3) fp64 = block*N*24 B, and a few same-shaped
    # temporaries live alongside it.
    block = max(1, min(512, int(budget_bytes / (n * 24.0 * 4))))
    out = torch.empty(sample.numel(), 3, dtype=torch.float64, device=p.device)

    for lo in range(0, sample.numel(), block):
        idx = sample[lo:lo + block]
        rvec = p[idx][:, None, :] - p[None, :, :]           # target - source
        r2 = (rvec * rvec).sum(-1)
        r = r2.sqrt()
        far = r >= cutoff                                    # also kills r == 0

        inv_r = torch.where(far, 1.0 / r, torch.zeros_like(r))
        inv_r3 = inv_r ** 3
        inv_r5 = inv_r ** 5

        # u = pref * [ (1/r + 2a^2/3r^3) f + (1/r^3 - 2a^2/r^5) r (r.f) ]
        rdf = (rvec * f[None, :, :]).sum(-1)
        coefI = inv_r + (2.0 * a2 / 3.0) * inv_r3
        coefR = (inv_r3 - 2.0 * a2 * inv_r5) * rdf
        out[lo:lo + idx.numel()] = PREF_TT * (
            (coefI[:, :, None] * f[None, :, :]).sum(1)
            + (coefR[:, :, None] * rvec).sum(1)
        )
    return out


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def selftest_reference(n: int = 400, seed: int = 7) -> None:
    """Check far_ref_tt against the independently-written dense reference.

    benchmarks/symmetry_treecode.py:exact_far_tt assembles the full (3N,3N) RPY-TT
    far-field matrix and was itself validated against benchmarks/bench_rpy.py.
    Agreeing with it to fp64 round-off means the sampled/chunked form here has
    the same signs, prefactor and cutoff convention.
    """
    from benchmarks.symmetry_treecode import exact_far_tt

    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 40.0, (n, 3))
    force = rng.standard_normal((n, 3))

    dense = exact_far_tt(pos) @ force.reshape(-1)
    dense = torch.as_tensor(dense.reshape(n, 3), dtype=torch.float64)

    dev = torch.device("cuda")
    got = far_ref_tt(torch.as_tensor(pos, device=dev),
                     torch.as_tensor(force, device=dev),
                     torch.arange(n, device=dev)).cpu()

    err = rel_l2(got, dense)
    print(f"[selftest] far_ref_tt vs exact_far_tt (N={n}): rel err {err:.3e}")
    assert err < 1e-12, f"reference mismatch: {err:.3e}"
    print("[selftest] OK")


# ----------------------------------------------------------------------------
# particle configurations
# ----------------------------------------------------------------------------
def load_soa_bin(path: Path) -> np.ndarray:
    """widebvh cloud format: headerless fp64 SoA blocks x[0:N], y[0:N], z[0:N]."""
    a = np.fromfile(path, dtype=np.float64)
    n = a.size // 3
    return np.stack([a[:n], a[n:2 * n], a[2 * n:]], axis=1)


def single_drop(radius: float = 40.0, phi: float = 0.048,
                seed: int = 0) -> np.ndarray:
    """3k-particle sedimenting drop, as in experiments/single_drop_sedimentation.

    Cubic lattice sized to hit the target volume fraction, clipped to a ball and
    jittered -- the same construction two_suspensions_1M.py uses.
    """
    a_lat = (4.0 * math.pi / 3.0 / phi) ** (1.0 / 3.0)
    n = int(math.ceil(2 * radius / a_lat)) + 2
    g = (np.arange(n) - (n - 1) / 2.0) * a_lat
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    pts = pts[np.linalg.norm(pts, axis=1) <= radius]
    rng = np.random.default_rng(seed)
    return pts + rng.uniform(-0.05 * a_lat, 0.05 * a_lat, pts.shape)


def load_csv_positions(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return np.ascontiguousarray(arr[:, :3], dtype=np.float64)


def two_drop_initial(seed: int = 0) -> np.ndarray:
    """The Figure-13 two-drop cloud at t=0, exactly as two_suspensions_1M.py
    builds it (same generator, same seed => bit-identical positions), so this
    case needs no dumped distribution file. N = 1,047,968."""
    from benchmarks.two_suspensions_1M import generate_suspension_drop
    np.random.seed(seed)
    r_drop, gap = 175.0, 100.0
    d1 = generate_suspension_drop((0, 0, 0.0), r_drop)
    d2 = generate_suspension_drop((0, 0, 2 * r_drop + gap), r_drop)
    return np.ascontiguousarray(np.vstack([d1, d2]), dtype=np.float64)


def get_case(name: str) -> tuple[np.ndarray, str]:
    if name == "drop3k":
        return single_drop(), "single drop R=40 phi=0.048"
    if name == "twodrop0":
        return two_drop_initial(), "two-drop sedimentation t=0 (generated, seed 0)"
    if name.startswith("npy:"):
        # Any (N,3) float array, e.g. a positions_<t>.npy snapshot written by
        # two_suspensions_1M.py --snapshots -- the deformed cloud later in the run.
        path = Path(name[4:])
        arr = np.load(path)
        assert arr.ndim == 2 and arr.shape[1] >= 3, f"{path}: expected (N,3)"
        return (np.ascontiguousarray(arr[:, :3], dtype=np.float64),
                f"positions from {path.name}")
    if name == "uniform100k":
        return (load_csv_positions(ROOT / "tmp" / "uniform_large_0.1_100000.csv"),
                "uniform phi=0.1")
    if name == "uniform750k":
        return (load_csv_positions(ROOT / "tmp" / "uniform_large_0.1_750000.csv"),
                "uniform phi=0.1")
    if name.startswith("twoball"):
        t = name.replace("twoball", "") or "0"
        return (load_soa_bin(WIDEBVH_DISTROS / f"two_ball_t{t}.bin"),
                f"two-drop sedimentation t={t}")
    raise ValueError(f"unknown case {name!r}")


CASES = ("drop3k", "uniform100k", "twoball0", "twoball100")
# Not in `all`: "twodrop0" (the Figure-13 t=0 cloud, generated in-process) and
# "npy:<path>" (any dumped snapshot) -- the cases used for the consumer-GPU
# sweep, where the widebvh distros directory is not available.


# ----------------------------------------------------------------------------
# loadings
# ----------------------------------------------------------------------------
def make_force(n: int, kind: str, seed: int = 2024) -> np.ndarray:
    if kind == "gravity":
        f = np.zeros((n, 3), dtype=np.float64)
        f[:, 2] = -9.81
        return f
    if kind == "random":
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def quiet():
    """Silence the per-call [MobFMM] chatter during a sweep."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def time_far(solver, pos_t, f3, repeats: int = 3) -> float:
    with quiet():
        solver.get_far_field_vel(pos_t, f3)          # warm up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            solver.get_far_field_vel(pos_t, f3)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / repeats


def far_asym_probe(solver, pos_t, K: int, seed: int = 11) -> dict:
    """||M_far - M_far^T||_F / ||M_far||_F of the far-field TT operator alone,
    by Hutchinson probing (the estimator of symmetry_treecode.hutchinson,
    applied to get_far_field_vel with (N,3) Rademacher probes).

    Far field only, not the grand operator: the near field is identical across
    every configuration swept here, so this isolates what the treecode settings
    (mac, PDEG, leaf, fp32 level) do to symmetry. The far-field operator's
    asymmetry is MAC-driven (widebvh_far_field_report.md: asym/trunc = sqrt 2).
    """
    n = pos_t.shape[0]
    g = torch.Generator(device=pos_t.device).manual_seed(seed)
    Z, MZ = [], []
    with quiet():
        for _ in range(K):
            z = (torch.randint(0, 2, (n, 3), device=pos_t.device, generator=g,
                               dtype=torch.float32) * 2 - 1)
            Z.append(z.double())
            MZ.append(solver.get_far_field_vel(pos_t, z).clone().double())
    diffs, cross = [], []
    for a in range(K):
        for b in range(a + 1, K):
            s_ab = float((Z[a] * MZ[b]).sum())
            s_ba = float((Z[b] * MZ[a]).sum())
            diffs.append((s_ab - s_ba) ** 2)
            cross.append(s_ab ** 2)
            cross.append(s_ba ** 2)
    fro_asym = math.sqrt(float(np.mean(diffs)))
    fro_M = math.sqrt(float(np.mean(cross)))
    spread = (float(np.std(diffs) / math.sqrt(K)) / (2 * fro_asym)
              if fro_asym > 0 else 0.0)
    return {"rel_asym": fro_asym / fro_M, "rel_asym_stderr": spread / fro_M}


def run_case(case: str, macs, pdegs, leaves, loadings, samples: int,
             rows: list, policy: str = "bary", orders=(None,),
             fp32_levels=(0,), pair_budget_gb=None, asym_probes: int = 0):
    pos, descr = get_case(case)
    n = pos.shape[0]
    print(f"\n=== {case}: N={n:,}  ({descr}) ===", flush=True)

    dev = torch.device("cuda")
    pos_t32 = torch.as_tensor(pos, dtype=torch.float32, device=dev).contiguous()
    pos_t64 = torch.as_tensor(pos, dtype=torch.float64, device=dev).contiguous()

    # Evenly spaced sample targets -- the protocol widebvh's own validation uses.
    s = min(samples, n)
    sample = torch.linspace(0, n - 1, s, device=dev).long()

    from src.treecode_widebvh import WidebvhFMM

    for loading in loadings:
        f_np = make_force(n, loading)
        f3 = torch.as_tensor(f_np, dtype=torch.float32, device=dev).contiguous()
        f64 = torch.as_tensor(f_np, dtype=torch.float64, device=dev)

        t0 = time.perf_counter()
        ref = far_ref_tt(pos_t64, f64, sample)
        ref_s = time.perf_counter() - t0
        ref_norm = float(torch.linalg.norm(ref))

        # Denominator for rel_total: the full reported velocity. The near field
        # is expensive to build here, so use the analytic self term plus the
        # exact far field -- self dominates it, and the near-field correction is
        # a few percent on top, so this is the right order and is loading-aware.
        self_u = f64[sample] / (6.0 * math.pi * RADIUS)
        total_norm = float(torch.linalg.norm(self_u + ref))
        print(f"  [{loading}] reference: {ref_s:.1f}s  "
              f"||u_far||={ref_norm:.4e}  ||u_self+u_far||={total_norm:.4e}",
              flush=True)

        # The Cartesian policy has no compile-time degree; its ladder is the
        # runtime `order`, so the two policies sweep different axes and the
        # inactive one collapses to a single pass.
        for pdeg in (pdegs if policy == "bary" else (0,)):
          for order in (orders if policy == "cart" else (None,)):
            for lvl in (fp32_levels if policy == "bary" else (0,)):
              for leaf in leaves:
                for mac in macs:
                    solver = WidebvhFMM(
                        near_field_operator=None, mac=mac, pdeg=pdeg or 7,
                        policy=policy, order=order, fp32_level=lvl,
                        max_leaf=leaf, near_field_cutoff=CUTOFF,
                        pair_budget_gb=pair_budget_gb,
                    )
                    try:
                        with quiet():
                            u = solver.get_far_field_vel(pos_t32, f3)
                        err = float(torch.linalg.norm(u[sample] - ref))
                        ms = time_far(solver, pos_t32, f3)
                        st = solver.last_stats
                        asym = (far_asym_probe(solver, pos_t32, asym_probes)
                                if asym_probes > 0 else {})
                    finally:
                        solver.close()

                    row = dict(
                        case=case, n=n, loading=loading,
                        backend="widebvh" if policy == "bary" else "widebvh-cart",
                        policy=policy, order=order or "",
                        pdeg=pdeg if policy == "bary" else "",
                        max_leaf=leaf, mac=mac, theta="",
                        rel_far=err / ref_norm, rel_total=err / total_norm,
                        far_ms=ms, near_pairs=st.get("near_pairs", 0),
                        traverse_ms=st.get("traverse_ms", 0),
                        p2p_ms=st.get("p2p_ms", 0),
                        build_ms=st.get("build_bvh_ms", 0) + st.get("bucket_ms", 0),
                        upward_ms=st.get("upward_ms", 0),
                        fp32_level=lvl,
                        rel_asym=asym.get("rel_asym", ""),
                        rel_asym_stderr=asym.get("rel_asym_stderr", ""),
                        asym_probes=asym_probes if asym else "",
                        pair_budget_gb=solver.env["TC_PAIR_BUDGET_GB"],
                        gpu=torch.cuda.get_device_name(0),
                    )
                    rows.append(row)
                    tag = (f"pdeg={pdeg} l={lvl}" if policy == "bary"
                           else f"order={order}")
                    asym_txt = (f"  asym={asym['rel_asym']:.3e}" if asym else "")
                    print(f"    {tag} leaf={leaf:<5} mac={mac:<5} "
                          f"rel_far={row['rel_far']:.3e}  "
                          f"rel_total={row['rel_total']:.3e}  "
                          f"far={ms:8.2f} ms (trav {row['traverse_ms']:.1f} "
                          f"p2p {row['p2p_ms']:.1f} up {row['upward_ms']:.1f} "
                          f"build {row['build_ms']:.1f})  "
                          f"pairs={int(row['near_pairs']):,}{asym_txt}",
                          flush=True)


def run_baseline(case: str, thetas, loadings, samples: int, rows: list):
    """Same measurement for the incumbent WarpFMM, as the head-to-head row."""
    pos, _ = get_case(case)
    n = pos.shape[0]
    dev = torch.device("cuda")
    pos_t32 = torch.as_tensor(pos, dtype=torch.float32, device=dev).contiguous()
    pos_t64 = torch.as_tensor(pos, dtype=torch.float64, device=dev).contiguous()
    s = min(samples, n)
    sample = torch.linspace(0, n - 1, s, device=dev).long()

    from src.treecode import WarpFMM

    print(f"\n=== {case}: WarpFMM baseline ===", flush=True)
    for loading in loadings:
        f_np = make_force(n, loading)
        f3 = torch.as_tensor(f_np, dtype=torch.float32, device=dev).contiguous()
        f64 = torch.as_tensor(f_np, dtype=torch.float64, device=dev)
        ref = far_ref_tt(pos_t64, f64, sample)
        ref_norm = float(torch.linalg.norm(ref))
        self_u = f64[sample] / (6.0 * math.pi * RADIUS)
        total_norm = float(torch.linalg.norm(self_u + ref))

        for theta in thetas:
            solver = WarpFMM(near_field_operator=None, theta=theta,
                             leaf_size=16, near_field_cutoff=CUTOFF)
            with quiet():
                u = solver.get_far_field_vel(pos_t32, f3)
            err = float(torch.linalg.norm(u[sample] - ref))
            ms = time_far(solver, pos_t32, f3)
            rows.append(dict(
                case=case, n=n, loading=loading, backend="warpfmm",
                policy="", order="", pdeg="", max_leaf=16, mac="", theta=theta,
                rel_far=err / ref_norm, rel_total=err / total_norm,
                far_ms=ms, near_pairs=0, traverse_ms=0, p2p_ms=0,
                build_ms=0, upward_ms=0,
            ))
            print(f"    theta={theta:<5} rel_far={err / ref_norm:.3e}  "
                  f"rel_total={err / total_norm:.3e}  far={ms:8.2f} ms",
                  flush=True)


FIELDS = ("case", "n", "loading", "backend", "policy", "order", "pdeg",
          "max_leaf", "mac", "theta", "rel_far", "rel_total", "far_ms",
          "near_pairs", "traverse_ms", "p2p_ms", "build_ms", "upward_ms",
          # added with the fp32 ladder / consumer-GPU sweep; blank in the
          # H200-era rows of data/widebvh_mac_calibration.csv
          "fp32_level", "rel_asym", "rel_asym_stderr", "asym_probes",
          "pair_budget_gb", "gpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="uniform100k",
                    help=f"one of {CASES + ('all',)}")
    ap.add_argument("--macs", default="0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--policy", default="bary", choices=("bary", "cart"),
                    help="multipole expansion: barycentric Lagrange (degree "
                         "--pdegs) or Cartesian Taylor (runtime --orders)")
    ap.add_argument("--pdegs", default="7", help="bary only")
    ap.add_argument("--orders", default="4", help="cart only (1..4)")
    ap.add_argument("--leaves", default="1024")
    ap.add_argument("--loadings", default="gravity")
    ap.add_argument("--thetas", default="0.3,0.28,0.2",
                    help="WarpFMM baseline opening angles (empty to skip)")
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--csv", default="")
    ap.add_argument("--selftest", action="store_true",
                    help="validate the fp64 reference and exit")
    ap.add_argument("--fp32-levels", default="0",
                    help="bary only: widebvh fp32 fast-path levels to sweep "
                         "(0 fp64, 1 fp32 M2P, 2 + fp32 P2P, 3 + fp32 upward); "
                         "each needs its libwidebvh_nemo*_f32l<L>.so")
    ap.add_argument("--pair-budget-gb", type=float, default=None,
                    help="widebvh pair-list budget override (default: the "
                         "wrapper's ~6%% of VRAM, min 1 GB). Raise it when "
                         "sweeping small leaves at 1M so the pair list does "
                         "not spill into the engine's tiled fallback")
    ap.add_argument("--asym-probes", type=int, default=0,
                    help="K > 0 adds a far-field-only Hutchinson symmetry "
                         "probe (K extra far-field applies per config)")
    args = ap.parse_args()

    if args.selftest:
        selftest_reference()
        return

    macs = [float(x) for x in args.macs.split(",") if x]
    pdegs = [int(x) for x in args.pdegs.split(",") if x]
    fp32_levels = [int(x) for x in args.fp32_levels.split(",") if x]
    orders = [int(x) for x in args.orders.split(",") if x]
    leaves = [int(x) for x in args.leaves.split(",") if x]
    loadings = [x for x in args.loadings.split(",") if x]
    thetas = [float(x) for x in args.thetas.split(",") if x]
    cases = list(CASES) if args.case == "all" else [args.case]

    rows: list[dict] = []
    for case in cases:
        run_case(case, macs, pdegs, leaves, loadings, args.samples, rows,
                 policy=args.policy, orders=orders, fp32_levels=fp32_levels,
                 pair_budget_gb=args.pair_budget_gb,
                 asym_probes=args.asym_probes)
        if thetas:
            run_baseline(case, thetas, loadings, args.samples, rows)

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out.exists()
        with out.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
