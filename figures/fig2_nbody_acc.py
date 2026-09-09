#!/usr/bin/env python3
"""Figure 2 (left panel) refresh: progressive accuracy of the learned mobility stack on the
dataset-v2 validation subset.

The four stages are the shipped operators, each evaluated end-to-end on sampled validation
configurations (truth = the configuration's full MFS grand mobility matrix from data/multibody_v2/):
  2b       2-body only                    NNMob (self + 2-body NN within 6, RPY beyond)
  b1       + n-body (l=1)                 Mob_Op_Nbody, nbody_pinn_b1_v2.pt (K=10, r_c=6)
  moments  + n-body moments (l=2)         Mob_Op_Nbody_Moments, nbody_moments_v2_kinf_rc8_pc8.pt
                                          (all neighbours, r_c=8, pair_cutoff=switch_dist=8)
  diag     + learned diagonal             ... + nbody_diag_v2_pc8.pt (locked to pair_cutoff=8)
PRMSE is pooled over all sampled configurations (L2 of the stacked errors / L2 of the stacked truth,
as in the original Figure 2), translational and rotational separately.

    TORCH_COMPILE_DISABLE=1 python figures/fig2_nbody_acc.py                  # eval + figure
    python figures/fig2_nbody_acc.py --plot-only                              # re-render from the CSV

Outputs: figures/fig2_nbody_acc.{pdf,png} and figures/fig2_nbody_acc.csv (per-config errors).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(ROOT)]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))  # grpy_tensors is imported bare by mob_op_2b_combined
os.chdir(ROOT)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # accuracy work

FAMILIES = ["uniform", "grown", "lattice", "chain"]
SELF_PATH = "data/models/self_interaction_model.pt"
TWO_BODY_PATH = "data/models/two_body_combined_model.pt"
STAGES = ["2b", "b1", "moments", "diag"]
STAGE_LABELS = {"2b": "2-body only", "b1": r"+ $n$-body ($\ell=1$)",
                "moments": r"+ $n$-body moments ($\ell=2$)", "diag": "+ learned diagonal"}
# Okabe-Ito, warm -> cool; the endpoints are Figure 2's original pair (#D55E00 / #0072B2).
STAGE_COLORS = {"2b": "#D55E00", "b1": "#E69F00", "moments": "#56B4E9", "diag": "#0072B2"}
C_CONN = "#BDBDBD"
INK = "#2F2F2F"


def build_ops():
    from src.mob_op_2b_combined import NNMob
    from src.mob_op_nbody import Mob_Op_Nbody
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments

    common = dict(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH)
    mom = dict(common, nbody_nn_path="data/models/nbody_moments_v2_kinf_rc8_pc8.pt",
               switch_dist=8.0, pair_cutoff=8.0, neighbor_cutoff=8.0, max_neighbors=None)
    return {"2b": NNMob(**common),
            "b1": Mob_Op_Nbody(**common, nbody_nn_path="data/models/nbody_pinn_b1_v2.pt", switch_dist=6.0),
            "moments": Mob_Op_Nbody_Moments(**mom),
            "diag": Mob_Op_Nbody_Moments(**mom, diag_nn_path="data/models/nbody_diag_v2_pc8.pt", diag_cutoff=8.0)}


def evaluate(args) -> pd.DataFrame:
    import torch

    cfg = np.load(args.cache / "configs.npz")
    shards = sorted(glob.glob("data/multibody_v2/*/*/shard_*.npz"))
    rng = np.random.default_rng(args.seed)
    groups: dict[tuple, list[int]] = {}
    for i in np.nonzero(cfg["is_val"])[0]:
        groups.setdefault((int(cfg["family"][i]), round(float(cfg["param"][i]), 4), int(cfg["P"][i])), []).append(int(i))
    picked = np.concatenate([rng.choice(g, size=min(args.per_item, len(g)), replace=False) for _, g in sorted(groups.items())])
    if args.max_configs:
        picked = rng.permutation(picked)[:args.max_configs]
    picked = picked[np.argsort(cfg["shard"][picked], kind="stable")]  # one shard in memory at a time
    ops = build_ops()
    rows, t0 = [], time.time()
    shard_id, shard_M = -1, None
    for k, c in enumerate(picked):
        sh = int(cfg["shard"][c])
        if sh != shard_id:
            shard_id, shard_M = sh, np.load(shards[sh])["M"]
        P = int(cfg["P"][c])
        pos = cfg["positions"][c, :P].astype(np.float64)
        M = shard_M[int(cfg["index"][c])].astype(np.float64)
        F = np.random.default_rng([args.seed, int(c)]).normal(size=(P, 6))
        v_true = (M @ F.reshape(-1)).reshape(P, 6)
        config = np.zeros((P, 7)); config[:, :3] = pos; config[:, 6] = 1.0
        for stage in STAGES:
            with torch.no_grad():
                v = ops[stage].apply(config, F, 1.0)
            v = np.asarray(v.cpu() if torch.is_tensor(v) else v, dtype=np.float64)
            err = v - v_true
            rows.append({"cfg": int(c), "family": FAMILIES[int(cfg["family"][c])], "param": float(cfg["param"][c]),
                         "P": P, "stage": stage,
                         "err2_lin": float((err[:, :3] ** 2).sum()), "true2_lin": float((v_true[:, :3] ** 2).sum()),
                         "err2_ang": float((err[:, 3:] ** 2).sum()), "true2_ang": float((v_true[:, 3:] ** 2).sum())})
        if (k + 1) % 25 == 0 or k + 1 == len(picked):
            print(f"  {k + 1}/{len(picked)} configs  [{time.time() - t0:.0f} s]", flush=True)
    return pd.DataFrame(rows)


def pooled_prmse(df: pd.DataFrame) -> pd.DataFrame:
    """PRMSE (%) per stage, pooled over every sampled configuration (norms accumulated, then divided)."""
    g = df.groupby("stage", sort=False)[["err2_lin", "true2_lin", "err2_ang", "true2_ang"]].sum()
    out = pd.DataFrame({"lin": np.sqrt(g["err2_lin"] / g["true2_lin"]) * 100,
                        "ang": np.sqrt(g["err2_ang"] / g["true2_ang"]) * 100})
    return out.loc[STAGES]


def plot(df: pd.DataFrame, out_stem: Path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    p = pooled_prmse(df)
    print("\npooled PRMSE (%) on the validation subset:")
    print(p.round(2).to_string())
    for fam in FAMILIES:
        sub = df[df["family"] == fam]
        if len(sub):
            pf = pooled_prmse(sub)
            print(f"  {fam:8s} lin " + " -> ".join(f"{pf['lin'][s]:.2f}" for s in STAGES)
                  + "   ang " + " -> ".join(f"{pf['ang'][s]:.2f}" for s in STAGES))

    groups = [("Translational", "lin"), ("Rotational", "ang")]
    y = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    for yi, (_, col) in zip(y, groups):
        vals = p[col].values
        for a, b in zip(vals[:-1], vals[1:]):  # stage i -> i+1
            ax.annotate("", xy=(b, yi), xytext=(a, yi),
                        arrowprops=dict(arrowstyle="-|>", color=C_CONN, lw=1.8, shrinkA=5, shrinkB=5))
        for j, stage in enumerate(STAGES):
            ax.scatter(vals[j], yi, s=80, color=STAGE_COLORS[stage], edgecolor="white",
                       linewidth=1.1, zorder=3, label=STAGE_LABELS[stage] if yi == 0 else None)
            va, dy = ("bottom", -0.17) if j % 2 == 0 else ("top", 0.17)
            ax.text(vals[j], yi + dy, f"{vals[j]:.1f}%", ha="center", va=va, fontsize=8,
                    color=STAGE_COLORS[stage], fontweight="bold")
        ax.text(np.sqrt(vals[0] * vals[-1]), yi - 0.34, f"{vals[0] / vals[-1]:.1f}× overall", va="bottom",
                ha="center", fontsize=8, color=INK,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}%"))
    ax.xaxis.set_minor_locator(NullLocator())
    lo, hi = p.values.min(), p.values.max()
    ax.set_xlim(lo * 0.6, hi * 1.7)
    ax.set_yticks(y)
    ax.set_yticklabels([g[0] for g in groups], fontsize=9, ha="right", va="center", rotation_mode="anchor")
    ax.set_ylim(-0.62, len(groups) - 0.38)
    ax.invert_yaxis()
    ax.set_xlabel("PRMSE (%, log scale)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", which="both", color="#EEEEEE", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.002), ncol=2, frameon=False,
              fontsize=7.5, handletextpad=0.3, columnspacing=0.9, borderaxespad=0.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    print(f"-> {out_stem}.pdf, {out_stem}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("data/multibody_v2_cache_pc8"),
                    help="only configs.npz (positions, shard indices, is_val) is read")
    ap.add_argument("--per-item", type=int, default=6, help="validation configurations per (family, param, P)")
    ap.add_argument("--max-configs", type=int, default=None, help="cap on sampled configs (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("figures/fig2_nbody_acc"))
    ap.add_argument("--plot-only", action="store_true", help="re-render the figure from the existing CSV")
    args = ap.parse_args()
    csv = args.out.with_suffix(".csv")
    if args.plot_only:
        df = pd.read_csv(csv)
    else:
        df = evaluate(args)
        df.to_csv(csv, index=False, float_format="%.8g")
        print(f"-> {csv} ({df['cfg'].nunique()} configs x {len(STAGES)} stages)")
    plot(df, args.out)


if __name__ == "__main__":
    main()
