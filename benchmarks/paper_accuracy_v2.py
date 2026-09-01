#!/usr/bin/env python3
"""NeMO paper accuracy protocols (Figures 3 and 4, clustered near-contact table) for the grand mobility
operators, including the n-body models retrained on dataset v2.  Warp-free: the CPU operators run anywhere
(a CPU worker pool), the MFS truths and the two GPU yardsticks (mfs_coarse, the paper's GPU n-body operator)
run where a GPU is available.

Protocols (benchmarks/accuracy_grand_M.py):
  fig3     N in {200, 300} x phi in {.025,...,.2} x 10 seeds (123 + run); Fig 3 = run_experiment_fixed_size_diff_operators
  fig4     N in {20,...,200} x 8 phi x 10 seeds (123 + v_idx*1000 + p_idx*100 + run); Fig 4 = run_exp_different_sizes
  cluster  tmp/reference_sphere_{delta}.csv (N = 10 grown clusters, surface gap delta)
Truth = benchmarks.cluster.generate_uniform_testcase (Xfine MFS, tol 1e-8), cached per (N, phi, seed) in
tmp/nbody_moments_truth/ (the same generator, seeds and files as benchmarks/compare_nbody_moments.py).

    python benchmarks/paper_accuracy_v2.py --exp fig3 fig4 --phis 0.1 --truth-only [--gpu-ops] [--part]   # GPU job
    python benchmarks/paper_accuracy_v2.py --exp fig3 fig4 --phis 0.1 --workers 32 --part                  # CPU job
    python benchmarks/paper_accuracy_v2.py --exp cluster
    python benchmarks/paper_accuracy_v2.py --merge --summary --figures

Run with TORCH_COMPILE_DISABLE=1 (accuracy work).  Rows: data/paper_accuracy_v2.csv (long format, key
(exp, N, phi, seed, op); later runs replace earlier rows); --part writes data/paper_accuracy_v2/parts/*.csv
instead (SLURM array tasks), --merge folds the parts into the main CSV.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(ROOT)]
sys.path.insert(0, str(ROOT))  # this repo's src/ must shadow any other `src` package on the path
sys.path.insert(1, str(ROOT / "src"))  # grpy_tensors (RPY) is imported bare by mob_op_2b_combined
os.chdir(ROOT)

from benchmarks.compare_nbody_moments import compute_error_stats  # noqa: E402  (warp-free metric clone)

SHAPE = "sphere"
SELF_PATH = "data/models/self_interaction_model.pt"
TWO_BODY_PATH = "data/models/two_body_combined_model.pt"
TWO_BODY_WT = "data/models/combined_2body.wt"
MODELS = {
    "3b": "data/models/3body_cross.pt",
    "b1": "data/models/nbody_pinn_b1.pt",                      # paper Fig 3 (CPU)
    "b1_v2": "data/models/nbody_pinn_b1_v2.pt",
    "mom_old": "data/models/nbody_moments.pt",                 # moments model trained on the old rows
    "mom_v2_k10_rc6": "data/models/nbody_moments_v2_k10_rc6.pt",
    "mom_v2_kinf_rc6": "data/models/nbody_moments_v2_kinf_rc6.pt",
    "mom_v2_kinf_rc8": "data/models/nbody_moments_v2_kinf_rc8.pt",
    "mom_v2_kinf_rc8_pc8": "data/models/nbody_moments_v2_kinf_rc8_pc8.pt",  # pair_cutoff 8 ablation
    "diag_v2_pc8": "data/models/nbody_diag_v2_pc8.pt",         # per-particle diagonal correction (pc8 labels)
    "gpu_wt": "data/models/nbody_cross_tmp.wt",                # paper Fig 4 (GPU operator)
}
PHIS = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
FIG3_N = [200, 300]
FIG4_N = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
DELTAS = ["0.1", "0.2", "0.5", "1.0", "2.0", "3.0"]
NUM_REPEATS = 10
BASE_SEED = 123
NEARFIELD_CUTOFF = 6.0

TRUTH_DIR = ROOT / "tmp" / "nbody_moments_truth"
MAIN_CSV = ROOT / "data" / "paper_accuracy_v2.csv"
PARTS_DIR = ROOT / "data" / "paper_accuracy_v2" / "parts"
TABLES_MD = ROOT / "artifacts" / "paper_accuracy_v2_tables.md"
HIST_FIG3 = ROOT / "data" / "grand_M_acc_uniform_fixed_N.csv"
HIST_FIG4 = ROOT / "data" / "M_accuracy_nbody_diff_sizes.csv"
KEY = ["exp", "N", "phi", "seed", "op"]
CONFIG_COLS = ["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]
FORCE_COLS = ["f_x", "f_y", "f_z", "t_x", "t_y", "t_z"]
VEL_COLS = ["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]

# (max_neighbors, neighbor_cutoff, pair_cutoff) each moments model must be run with (= its training selection).
# The operator gets switch_dist=max(6, pair_cutoff): the 2b NN (trained to d=8) is the base wherever pairs are corrected.
SELECTION = {"mom_old": (10, 6.0, 6.0), "mom_v2_k10_rc6": (10, 6.0, 6.0), "mom_v2_kinf_rc6": (None, 6.0, 6.0),
             "mom_v2_kinf_rc8": (None, 8.0, 6.0), "mom_v2_kinf_rc8_pc8": (None, 8.0, 8.0)}
# ops that stack a per-particle diagonal model on a pair moments model: op -> (pair model key, diag model key).
# The diag model's labels subtract K_s over d <= its sidecar pair_cutoff, so it may only run at that pair_cutoff.
DIAG_OPS = {"M_mom_v2_kinf_rc8_pc8_diag": ("mom_v2_kinf_rc8_pc8", "diag_v2_pc8")}
PAPER_LABELS = {"M_rpy": "RPY", "M_2b": "NeMO 2-body", "M_3b": "NeMO 3-body summations", "M_nbody_b1": "NeMO n-body (b1)",
                "M_nbody_gpu": "NeMO n-body (GPU, Fig 4)", "mfs_coarse": "MFS coarse",
                "M_nbody_b1_v2": "b1 retrained on v2", "M_mom_old": "moments (old data)",
                "M_mom_v2_k10_rc6": "moments v2 (K=10, r_c=6)", "M_mom_v2_kinf_rc6": "moments v2 (all, r_c=6)",
                "M_mom_v2_kinf_rc8": "moments v2 (all, r_c=8)",
                "M_mom_v2_kinf_rc8_pc8": "moments v2 (all, r_c=8, pairs<=8)",
                "M_mom_v2_kinf_rc8_pc8_diag": "moments v2 (pairs<=8) + learned diagonal"}
OP_ORDER = ["M_rpy", "M_2b", "M_3b", "M_nbody_b1", "M_nbody_gpu", "M_nbody_b1_v2", "M_mom_old",
            "M_mom_v2_k10_rc6", "M_mom_v2_kinf_rc6", "M_mom_v2_kinf_rc8", "M_mom_v2_kinf_rc8_pc8",
            "M_mom_v2_kinf_rc8_pc8_diag", "mfs_coarse"]


# ----------------------------------------------------------------------------- cases
def cases(exp: str, Ns=None, phis=None, seeds=None) -> list[dict]:
    """The paper's (N, phi, seed) grid for an experiment; filters keep the paper's seed formula intact."""
    out = []
    if exp == "fig3":
        for N in (Ns or FIG3_N):
            for phi in PHIS:
                for run in range(NUM_REPEATS):
                    out.append({"exp": exp, "N": int(N), "phi": float(phi), "seed": BASE_SEED + run})
    elif exp == "fig4":
        for v_idx, phi in enumerate(PHIS):
            for p_idx, N in enumerate(FIG4_N):
                for run in range(NUM_REPEATS):
                    out.append({"exp": exp, "N": int(N), "phi": float(phi),
                                "seed": BASE_SEED + v_idx * 1000 + p_idx * 100 + run})
        if Ns is not None:
            out = [c for c in out if c["N"] in set(int(n) for n in Ns)]
    elif exp == "cluster":
        out = [{"exp": exp, "N": 10, "phi": float(d), "seed": 0} for d in DELTAS]
    else:
        raise ValueError(exp)
    if phis is not None:
        keep = np.asarray(phis, dtype=float)
        out = [c for c in out if np.any(np.isclose(c["phi"], keep, atol=1e-9))]
    if seeds is not None:
        out = [c for c in out if c["seed"] in set(int(s) for s in seeds)]
    return out


# ----------------------------------------------------------------------------- truth
def truth_path(N: int, phi: float, seed: int) -> Path:
    return TRUTH_DIR / f"uniform_N{N}_phi{phi:g}_seed{seed}.npz"


def load_case(case: dict, generate: bool = False):
    """(config (N,7), forces (N,6), velocity (N,6)) for a case; uniform cases from the truth cache."""
    if case["exp"] == "cluster":
        delta = {float(d): d for d in DELTAS}[float(case["phi"])]
        df = pd.read_csv(f"tmp/reference_sphere_{delta}.csv", float_precision="high")
        return (df[CONFIG_COLS].values.astype(np.float64), df[FORCE_COLS].values.astype(np.float64),
                df[VEL_COLS].values.astype(np.float64))
    p = truth_path(case["N"], case["phi"], case["seed"])
    if p.exists():
        d = np.load(p)
        return d["config"], d["forces"], d["velocity"]
    if not generate:
        raise FileNotFoundError(f"truth missing: {p} (run with --truth-only on a GPU box first)")
    from benchmarks.cluster import generate_uniform_testcase

    t0 = time.time()
    df = generate_uniform_testcase(shape=SHAPE, volume_fraction=case["phi"], numParticles=case["N"],
                                   seed=case["seed"], save_to_file=False)
    wall = time.time() - t0
    config = df[CONFIG_COLS].values.astype(np.float64)
    forces = df[FORCE_COLS].values.astype(np.float64)
    velocity = df[VEL_COLS].values.astype(np.float64)
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5(open(ROOT / "benchmarks" / "cluster.py", "rb").read()).hexdigest()
    np.savez(p, config=config, forces=forces, velocity=velocity, phi=case["phi"], N=case["N"], seed=case["seed"],
             acc="Xfine", wall=wall, cluster_md5=md5)
    print(f"[truth] N={case['N']} phi={case['phi']:g} seed={case['seed']}: {wall:.1f} s -> {p}", flush=True)
    return config, forces, velocity


# ----------------------------------------------------------------------------- operators
def _selection_for(key: str):
    """Selection parameters of a moments model: the published sidecar wins, the registry must agree."""
    want = SELECTION[key]
    side = Path(MODELS[key]).with_suffix(".json")
    if side.exists():
        meta = json.load(open(side))
        got = (meta.get("max_neighbors"), float(meta.get("neighbor_cutoff")), float(meta.get("pair_cutoff", 6.0)))
        assert got == want, f"{MODELS[key]}: trained with {got}, registry says {want}"
    return want


def _diag_sidecar_for(key: str) -> float:
    """The diag model's sidecar is mandatory: it pins the label convention (the pair_cutoff of the K_s
    subtraction -- running it at any other switch_dist double-counts K_s) and the selection/band layout.
    Returns diag_cutoff."""
    side = Path(MODELS[key]).with_suffix(".json")
    assert side.exists(), f"{MODELS[key]}: missing sidecar (required for a diag model)"
    meta = json.load(open(side))
    assert float(meta["pair_cutoff"]) == 8.0, f"diag model labelled at pair_cutoff {meta['pair_cutoff']}, op runs at 8"
    assert float(meta.get("diag_cutoff", 8.0)) == 8.0 and int(meta.get("nb", 8)) == 8, meta
    assert float(meta.get("band_lo", 2.0)) == 2.0 and float(meta.get("band_hi", 8.0)) == 8.0, meta
    return float(meta.get("diag_cutoff", 8.0))


def build_op(name: str):
    if name == "M_rpy" or name == "M_2b":
        from src.mob_op_2b_combined import NNMob
        return NNMob(SHAPE, SELF_PATH, TWO_BODY_PATH, nn_only=False, rpy_only=(name == "M_rpy"))
    if name == "M_3b":
        from src.mob_op_3body import NNMob3B
        return NNMob3B(shape=SHAPE, self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH, three_nn_path=MODELS["3b"],
                       nn_only=False, rpy_only=False, switch_dist=6.0, triplet_cutoff=6.0)
    if name in ("M_nbody_b1", "M_nbody_b1_v2"):
        from src.mob_op_nbody import Mob_Op_Nbody
        return Mob_Op_Nbody(shape=SHAPE, self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH,
                            nbody_nn_path=MODELS["b1" if name == "M_nbody_b1" else "b1_v2"],
                            nn_only=False, rpy_only=False, switch_dist=6.0)
    if name in DIAG_OPS:
        from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
        pair_key, diag_key = DIAG_OPS[name]
        max_neighbors, cutoff, pc = _selection_for(pair_key)
        return Mob_Op_Nbody_Moments(shape=SHAPE, self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH,
                                    nbody_nn_path=MODELS[pair_key], nn_only=False, rpy_only=False,
                                    switch_dist=max(6.0, pc), pair_cutoff=pc, neighbor_cutoff=cutoff,
                                    max_neighbors=max_neighbors, diag_nn_path=MODELS[diag_key],
                                    diag_cutoff=_diag_sidecar_for(diag_key))
    if name.startswith("M_mom_"):
        from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
        key = name[len("M_"):]
        max_neighbors, cutoff, pc = _selection_for(key)
        return Mob_Op_Nbody_Moments(shape=SHAPE, self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH,
                                    nbody_nn_path=MODELS[key], nn_only=False, rpy_only=False,
                                    switch_dist=max(6.0, pc), pair_cutoff=pc, neighbor_cutoff=cutoff,
                                    max_neighbors=max_neighbors)
    if name == "mfs_coarse":
        from src.triton_mfs import MobMFSTriton
        return MobMFSTriton(shape=SHAPE, acc="coarse")
    if name == "M_nbody_gpu":
        from src.gpu_nbody_mob import Mob_Nbody_Torch
        return Mob_Nbody_Torch(shape=SHAPE, self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_WT,
                               nbody_nn_path=MODELS["gpu_wt"], near_field_2b="nn", far_field_2b="rpy",
                               near_far_switch=6.0)
    raise KeyError(name)


CPU_OPS = ["M_rpy", "M_2b", "M_3b", "M_nbody_b1", "M_nbody_b1_v2", "M_mom_old", "M_mom_v2_k10_rc6",
           "M_mom_v2_kinf_rc6", "M_mom_v2_kinf_rc8", "M_mom_v2_kinf_rc8_pc8", "M_mom_v2_kinf_rc8_pc8_diag"]
GPU_OPS = ["mfs_coarse", "M_nbody_gpu"]
OP_MODEL = {"M_3b": "3b", "M_nbody_b1": "b1", "M_nbody_b1_v2": "b1_v2", "M_mom_old": "mom_old",
            "M_mom_v2_k10_rc6": "mom_v2_k10_rc6", "M_mom_v2_kinf_rc6": "mom_v2_kinf_rc6",
            "M_mom_v2_kinf_rc8": "mom_v2_kinf_rc8", "M_mom_v2_kinf_rc8_pc8": "mom_v2_kinf_rc8_pc8",
            "M_mom_v2_kinf_rc8_pc8_diag": ("mom_v2_kinf_rc8_pc8", "diag_v2_pc8"),
            "M_nbody_gpu": "gpu_wt"}


def available(names: list[str]) -> list[str]:
    out = []
    for n in names:
        key = OP_MODEL.get(n)
        keys = key if isinstance(key, tuple) else ((key,) if key is not None else ())
        missing = [k for k in keys if not Path(MODELS[k]).exists()]
        if missing:
            print(f"[ops] skipping {n}: {MODELS[missing[0]]} not found", flush=True)
            continue
        out.append(n)
    return out


def apply_op(op, config, forces):
    import torch
    with torch.no_grad():
        if hasattr(op, "apply_cpu"):  # GPU n-body operator (positions, orientations, force)
            pred = op.apply_cpu(config[:, :3], config[:, 3:], forces, viscosity=1.0)
        else:
            pred = op.apply(config, forces, 1.0)
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    return np.asarray(pred, dtype=np.float64)


def nearfield_interactions(config: np.ndarray) -> float:
    pos = config[:, :3]
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    return float(np.sum(D < NEARFIELD_CUTOFF) / len(pos))


def evaluate(ops: dict, case: dict, tag: str = "") -> list[dict]:
    config, forces, velocity = load_case(case)
    rows = []
    nf_int = nearfield_interactions(config)
    for name, op in ops.items():
        t0 = time.time()
        pred = apply_op(op, config, forces)
        wall = time.time() - t0
        stats = compute_error_stats(pred, velocity)
        rows.append({**case, "op": name + tag, "wall_s": wall, "nearfield_interactions": nf_int, **stats})
        print(f"{case['exp']} N={case['N']} phi={case['phi']:g} seed={case['seed']} {name + tag:<20} "
              f"rel_rmse={stats['rel_rmse']:7.3f}%  lin={stats['prmse_lin']:7.3f}%  ang={stats['prmse_ang']:7.3f}%  "
              f"max={stats['max_rel_rmse']:7.2f}%  ({wall:.1f} s)", flush=True)
    return rows


# ----------------------------------------------------------------------------- worker pool
_OPS: dict = {}
_TAG = ""


def _init_worker(op_names: list[str], tag: str, threads: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU operators only: never open a CUDA context per worker
    import torch
    torch.set_num_threads(threads)
    global _OPS, _TAG
    _TAG = tag
    _OPS = {n: build_op(n) for n in op_names}


def _worker(case: dict) -> list[dict]:
    return evaluate(_OPS, case, _TAG)


# ----------------------------------------------------------------------------- csv
def merge_rows(csv_path: Path, rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        merged = pd.concat([pd.read_csv(csv_path), new], ignore_index=True).drop_duplicates(subset=KEY, keep="last")
    else:
        merged = new
    merged = merged.sort_values(["exp", "N", "phi", "op", "seed"]).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False, float_format="%.6g")
    return merged


def merge_parts() -> pd.DataFrame:
    frames = [pd.read_csv(MAIN_CSV)] if MAIN_CSV.exists() else []
    parts = sorted(glob.glob(str(PARTS_DIR / "*.csv")))
    frames += [pd.read_csv(p) for p in parts if os.path.getsize(p) > 0]
    assert frames, "nothing to merge"
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=KEY, keep="last")
    df = df.sort_values(["exp", "N", "phi", "op", "seed"]).reset_index(drop=True)
    df.to_csv(MAIN_CSV, index=False, float_format="%.6g")
    print(f"[merge] {len(parts)} part files -> {MAIN_CSV} ({len(df)} rows)")
    return df


def done_keys() -> set:
    frames = [pd.read_csv(MAIN_CSV)] if MAIN_CSV.exists() else []
    frames += [pd.read_csv(p) for p in glob.glob(str(PARTS_DIR / "*.csv")) if os.path.getsize(p) > 0]
    if not frames:
        return set()
    df = pd.concat(frames, ignore_index=True)
    return set(zip(df["exp"], df["N"].astype(int), df["phi"].round(6), df["seed"].astype(int), df["op"]))


# ----------------------------------------------------------------------------- summary / figures
def _agg(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(["exp", "N", "phi", "op"])[col]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0), "n": g.count()}).reset_index()


def _ops_in(df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(df["op"]))
    return [o for o in OP_ORDER if o in present] + [o for o in present if o not in OP_ORDER]


def summary(df: pd.DataFrame, write_md: bool = True) -> str:
    lines = []
    hist3 = pd.read_csv(HIST_FIG3) if HIST_FIG3.exists() else None
    hist4 = pd.read_csv(HIST_FIG4) if HIST_FIG4.exists() else None

    def table(sub: pd.DataFrame, col: str, title: str, cols_key: str, col_values, fmt="{:.2f}", with_std=True, hist=None, col_label=None, n_expected=NUM_REPEATS):
        ops = _ops_in(sub)
        a = _agg(sub, col)
        hdr = f"| operator | " + " | ".join(f"{col_label or cols_key}={v:g}" for v in col_values) + " |"
        lines.append(f"\n**{title}**\n")
        lines.append(hdr)
        lines.append("|" + "---|" * (len(col_values) + 1))
        for op in ops:
            cells = []
            for v in col_values:
                r = a[(a["op"] == op) & np.isclose(a[cols_key], v)]
                if r.empty:
                    cells.append("-")
                else:
                    m, s, n = float(r["mean"].iloc[0]), float(r["std"].iloc[0]), int(r["n"].iloc[0])
                    cells.append((fmt.format(m) + (f" ± {s:.2f}" if with_std else "") + (f" ({n})" if n != n_expected else "")))
            lines.append(f"| {op} | " + " | ".join(cells) + " |")
        if hist is not None:
            for op, hop in hist:
                lines.append(f"| historic {op} (paper CSV) | " + " | ".join(fmt.format(hop.get(round(v, 6), np.nan)) for v in col_values) + " |")

    for exp in [e for e in ["fig3", "fig4", "cluster"] if e in set(df["exp"])]:
        d = df[df["exp"] == exp]
        if exp == "fig3":
            for N in sorted(d["N"].unique()):
                sub = d[d["N"] == N]
                phis = sorted(sub["phi"].unique())
                hist = None
                if hist3 is not None and N in set(hist3["num_particles"]):
                    h = hist3[hist3["num_particles"] == N]
                    hist = [(op, dict(zip(h["volume_fraction"].round(6), h[op]))) for op in ["M_rpy", "M_2b", "M_3b", "M_nbody"] if op in h]
                lines.append(f"\n## Fig 3 protocol, N = {N}: PRMSE (%) = rel_rmse, mean ± std over seeds (n = {NUM_REPEATS} unless noted)")
                table(sub, "rel_rmse", "rel_rmse (%)", "phi", phis, hist=hist)
                table(sub, "prmse_lin", "translational rel-L2 (%)", "phi", phis, with_std=False)
                table(sub, "prmse_ang", "rotational rel-L2 (%)", "phi", phis, with_std=False)
                table(sub, "max_rel_rmse", "max per-particle rel err (%)", "phi", phis, with_std=False)
                table(sub, "wall_s", "wall s per apply", "phi", phis, fmt="{:.1f}", with_std=False)
        elif exp == "fig4":
            lines.append(f"\n## Fig 4 protocol: PRMSE (%) vs N (mean over {NUM_REPEATS} seeds)")
            for op in _ops_in(d):
                sub = d[d["op"] == op]
                a = _agg(sub, "rel_rmse")
                Ns = sorted(sub["N"].unique())
                phis = sorted(sub["phi"].unique())
                lines.append(f"\n**{op}** (rows phi, columns N)\n")
                lines.append("| phi | " + " | ".join(str(n) for n in Ns) + " |")
                lines.append("|" + "---|" * (len(Ns) + 1))
                for phi in phis:
                    cells = []
                    for N in Ns:
                        r = a[(a["N"] == N) & np.isclose(a["phi"], phi)]
                        cells.append("-" if r.empty else f"{float(r['mean'].iloc[0]):.2f}")
                    lines.append(f"| {phi:g} | " + " | ".join(cells) + " |")
                if hist4 is not None and op in ("M_nbody_gpu", "M_nbody_b1"):
                    for phi in phis:
                        h = hist4[np.isclose(hist4["volume_fraction"], phi)]
                        cells = []
                        for N in Ns:
                            r = h[h["num_particles"] == N]
                            cells.append("-" if r.empty else f"{float(r['avg_rel_rmse'].iloc[0]):.2f}")
                        lines.append(f"| historic {phi:g} (paper CSV) | " + " | ".join(cells) + " |")
        else:
            deltas = sorted(d["phi"].unique())
            lines.append("\n## Clustered near-contact clusters (tmp/reference_sphere_delta.csv, N = 10)")
            table(d, "rel_rmse", "rel_rmse (%)", "phi", deltas, with_std=False, col_label="delta", n_expected=1)
            table(d, "max_rel_rmse", "max per-particle rel err (%)", "phi", deltas, with_std=False, col_label="delta", n_expected=1)
    text = "\n".join(lines)
    print(text)
    if write_md:
        TABLES_MD.parent.mkdir(parents=True, exist_ok=True)
        TABLES_MD.write_text("# Paper accuracy protocols (benchmarks/paper_accuracy_v2.py)\n" + text + "\n")
        print(f"\n-> {TABLES_MD}")
    # wide CSVs in the paper's layout
    d3 = df[df["exp"] == "fig3"]
    for N in sorted(d3["N"].unique()):
        a = _agg(d3[d3["N"] == N], "rel_rmse").pivot(index="phi", columns="op", values="mean").reset_index()
        a.insert(1, "num_particles", int(N))
        a.rename(columns={"phi": "volume_fraction"}).to_csv(ROOT / "data" / f"paper_accuracy_v2_fig3_P{N}.csv",
                                                             index=False, float_format="%.5g")
    return text


def figures(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (ROOT / "figures").mkdir(exist_ok=True)
    d3 = df[df["exp"] == "fig3"]
    hist3 = pd.read_csv(HIST_FIG3) if HIST_FIG3.exists() else None
    for N in sorted(d3["N"].unique()):
        sub = d3[d3["N"] == N]
        a = _agg(sub, "rel_rmse")
        fig, ax = plt.subplots(figsize=(9, 5))
        for op in _ops_in(sub):
            r = a[a["op"] == op].sort_values("phi")
            ax.errorbar(r["phi"], r["mean"], yerr=r["std"], marker="o", capsize=2, label=PAPER_LABELS.get(op, op))
        if hist3 is not None and N in set(hist3["num_particles"]):
            h = hist3[hist3["num_particles"] == N].sort_values("volume_fraction")
            for op in ["M_rpy", "M_2b", "M_3b", "M_nbody"]:
                ax.plot(h["volume_fraction"], h[op], ls="--", lw=0.8, color="grey", alpha=0.7,
                        label="paper CSV (historic)" if op == "M_rpy" else None)
        ax.set_xlabel("Volume fraction"); ax.set_ylabel("PRMSE (%)"); ax.set_title(f"Fig 3 protocol, N = {N}")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(ROOT / "figures" / f"paper_v2_fig3_P{N}.pdf"); fig.savefig(ROOT / "figures" / f"paper_v2_fig3_P{N}.png", dpi=150)
        plt.close(fig)
    d4 = df[df["exp"] == "fig4"]
    if not d4.empty:
        hist4 = pd.read_csv(HIST_FIG4) if HIST_FIG4.exists() else None
        ops = [o for o in _ops_in(d4) if o not in ("M_rpy", "M_2b", "M_3b", "mfs_coarse")]
        for target in ["rel_rmse", "max_rel_rmse"]:
            a = _agg(d4, target)
            ncol = min(3, len(ops)); nrow = int(np.ceil(len(ops) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.8 * nrow), squeeze=False, sharey=True)
            cmap = plt.get_cmap("viridis")
            show_phis = [p for p in [0.05, 0.1, 0.15, 0.2] if np.any(np.isclose(d4["phi"].unique()[:, None], p))]
            for k, op in enumerate(ops):
                ax = axes[k // ncol][k % ncol]
                for j, phi in enumerate(show_phis):
                    r = a[(a["op"] == op) & np.isclose(a["phi"], phi)].sort_values("N")
                    ax.plot(r["N"], r["mean"], marker="o", ms=3, color=cmap(j / max(1, len(show_phis) - 1)), label=f"phi={phi:g}")
                    if hist4 is not None and op in ("M_nbody_gpu", "M_nbody_b1"):
                        h = hist4[np.isclose(hist4["volume_fraction"], phi)].sort_values("num_particles")
                        ax.plot(h["num_particles"], h["avg_rel_rmse" if target == "rel_rmse" else "max_rel_rmse"],
                                ls="--", lw=0.8, color=cmap(j / max(1, len(show_phis) - 1)), alpha=0.6)
                ax.set_title(PAPER_LABELS.get(op, op), fontsize=9); ax.set_xlabel("num particles")
                if k % ncol == 0:
                    ax.set_ylabel("PRMSE (%)" if target == "rel_rmse" else "max per-particle rel err (%)")
            axes[0][0].legend(fontsize=7)
            fig.suptitle("Fig 4 protocol (dashed: paper CSV, historic)", fontsize=10); fig.tight_layout()
            fig.savefig(ROOT / "figures" / f"paper_v2_fig4_{target}.pdf"); fig.savefig(ROOT / "figures" / f"paper_v2_fig4_{target}.png", dpi=150)
            plt.close(fig)
    print("[figures] figures/paper_v2_fig3_P*.pdf, figures/paper_v2_fig4_*.pdf")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", default=["fig3"], choices=["fig3", "fig4", "cluster"])
    ap.add_argument("--N", type=int, nargs="+", default=None, help="restrict N (fig3 default 200 300; fig4 the paper list)")
    ap.add_argument("--phis", type=float, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--ops", nargs="+", default=None, help="default: every CPU operator whose model exists")
    ap.add_argument("--gpu-ops", action="store_true", help="also run mfs_coarse and (if warp imports) the paper's GPU n-body op")
    ap.add_argument("--truth-only", action="store_true", help="generate/cache the MFS truths (GPU), then exit (after --gpu-ops rows)")
    ap.add_argument("--workers", type=int, default=0, help="CPU worker processes (0 = in-process)")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    ap.add_argument("--part", action="store_true", help="write rows to data/paper_accuracy_v2/parts/ instead of the main CSV")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--skip-done", action="store_true", help="skip (case, op) rows already in the CSV/parts")
    ap.add_argument("--tag", default="", help="suffix appended to op names (model variants)")
    ap.add_argument("--models", nargs="*", default=[], metavar="KEY=PATH", help="override model paths, e.g. mom_v2_k10_rc6=runs/x/model.pt")
    args = ap.parse_args()
    for kv in args.models:
        k, v = kv.split("=", 1)
        assert k in MODELS, k
        MODELS[k] = v

    if args.merge:
        merge_parts()
    if args.summary or args.figures:
        df = pd.read_csv(MAIN_CSV)
        if args.summary:
            summary(df)
        if args.figures:
            figures(df)
    if args.merge or args.summary or args.figures:
        return

    all_cases = [c for e in args.exp for c in cases(e, args.N, args.phis, args.seeds)]
    print(f"[cases] {len(all_cases)} configurations: " + ", ".join(f"{e}:{sum(c['exp'] == e for c in all_cases)}" for e in args.exp))
    out_csv = MAIN_CSV
    if args.part:
        PARTS_DIR.mkdir(parents=True, exist_ok=True)
        jid = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        tid = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        out_csv = PARTS_DIR / f"{'_'.join(args.exp)}_{jid}{('_' + tid) if tid else ''}.csv"

    if args.truth_only or args.gpu_ops:
        for c in all_cases:
            if c["exp"] != "cluster":
                load_case(c, generate=True)
        if args.gpu_ops:
            names = ["mfs_coarse"]
            try:
                import warp  # noqa: F401
                names.append("M_nbody_gpu")
            except ImportError:
                print("[ops] warp not importable: skipping M_nbody_gpu")
            ops = {n: build_op(n) for n in available(names)}
            done = done_keys() if args.skip_done else set()
            for c in all_cases:
                todo = {n: o for n, o in ops.items() if (c["exp"], c["N"], round(c["phi"], 6), c["seed"], n + args.tag) not in done}
                if todo:
                    merge_rows(out_csv, evaluate(todo, c, args.tag))
        if args.truth_only:
            return

    op_names = available(args.ops or CPU_OPS)
    assert op_names, "no operators to run"
    done = done_keys() if args.skip_done else set()
    if args.skip_done:
        all_cases = [c for c in all_cases if any((c["exp"], c["N"], round(c["phi"], 6), c["seed"], n + args.tag) not in done for n in op_names)]
        print(f"[cases] {len(all_cases)} configurations still to do")
    for c in all_cases:  # every truth must be cached before the CPU pool starts
        load_case(c)
    t_start = time.time()
    if args.workers <= 0:
        _init_worker(op_names, args.tag, max(1, args.threads))
        for i, c in enumerate(all_cases):
            merge_rows(out_csv, _worker(c))
            print(f"[progress] {i + 1}/{len(all_cases)} ({time.time() - t_start:.0f} s)", flush=True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker, initargs=(op_names, args.tag, args.threads)) as pool:
            for i, rows in enumerate(pool.imap_unordered(_worker, all_cases)):
                merge_rows(out_csv, rows)
                print(f"[progress] {i + 1}/{len(all_cases)} ({time.time() - t_start:.0f} s)", flush=True)
    print(f"[done] {len(all_cases)} configurations x {len(op_names)} operators in {time.time() - t_start:.0f} s -> {out_csv}")


if __name__ == "__main__":
    main()
