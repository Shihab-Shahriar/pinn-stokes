#!/usr/bin/env python3
"""Error decomposition ("ceiling") of the pairwise n-body design on dataset-v2 configurations.

For sampled validation configurations, the velocity error of the 2-body operator (NNMob: analytic self +
two-body NN within 6 radii + RPY beyond) is reduced step by step with the *exact* MFS residuals:
  e_2b     2-body operator
  e_near   + the exact cross-block residual R_ts = Mts_sym - M2b_ts of every near pair (what a perfect n-body model gives)
  e_diag   + the diagonal residual M_tt - self - sum K_s (no operator models it)
  e_far    + the residual of the far pairs (d > pair_cutoff, RPY): reproduces M F up to the label symmetrisation
so e_near is the floor of any pairwise near-field correction and e_diag - e_far is what the far field costs.

    CUDA_VISIBLE_DEVICES= python experiments/nbody_v2_ceiling.py --per-item 6 --out artifacts/nbody_v2_ceiling.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(ROOT)]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "src"))
os.chdir(ROOT)

FAMILIES = ["uniform", "grown", "lattice"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("data/multibody_v2_cache"))
    ap.add_argument("--per-item", type=int, default=6, help="configurations per (family, param, P)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/nbody_v2_ceiling.csv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair-cutoff", type=float, default=6.0,
                    help="corrected-pair cutoff; must equal the cache's (near = cached R rows, far = d > this)")
    args = ap.parse_args()
    from src.mob_op_2b_combined import NNMob
    cfg = np.load(args.cache / "configs.npz")
    meta = json.load(open(args.cache / "meta.json"))
    assert float(meta.get("pair_cutoff", 6.0)) == args.pair_cutoff, "cache built with a different pair_cutoff"
    pair_cfg = np.load(args.cache / "pair_cfg.npy"); pair_t = np.load(args.cache / "pair_t.npy"); pair_s = np.load(args.cache / "pair_s.npy")
    Mts = np.load(args.cache / "Mts_sym.npy", mmap_mode="r"); M2b = np.load(args.cache / "M2b.npy", mmap_mode="r")
    Mtt_res = np.load(args.cache / "Mtt_res.npy", mmap_mode="r")
    order = np.argsort(pair_cfg, kind="stable"); starts = np.searchsorted(pair_cfg[order], np.arange(len(cfg["P"]) + 1))
    shards = sorted(glob.glob("data/multibody_v2/*/*/shard_*.npz"))
    op = NNMob("sphere", "data/models/self_interaction_model.pt", "data/models/two_body_combined_model.pt",
               switch_dist=max(6.0, args.pair_cutoff))
    rng = np.random.default_rng(args.seed)
    keys = list(zip(cfg["family"], cfg["param"].round(4), cfg["P"]))
    groups = {}
    for i, k in enumerate(keys):
        if cfg["is_val"][i]:
            groups.setdefault(k, []).append(i)
    rows = []
    t0 = time.time()
    shard_cache = {}
    for k in sorted(groups):
        pick = rng.choice(groups[k], size=min(args.per_item, len(groups[k])), replace=False)
        for c in pick:
            P = int(cfg["P"][c]); pos = cfg["positions"][c, :P]
            sh = int(cfg["shard"][c])
            if sh not in shard_cache:
                shard_cache = {sh: np.load(shards[sh])["M"]}
            M = shard_cache[sh][int(cfg["index"][c])].astype(np.float64); Mb = M.reshape(P, 6, P, 6)
            F = rng.normal(size=(P, 6)); v_true = (M @ F.reshape(-1)).reshape(P, 6)
            config = np.zeros((P, 7)); config[:, :3] = pos; config[:, 6] = 1.0
            v_2b = op.apply(config, F, 1.0)
            r = order[starts[c]:starts[c + 1]]
            R = (np.asarray(Mts[r]) - np.asarray(M2b[r])).astype(np.float64).reshape(-1, 6, 6)
            t = pair_t[r].astype(np.int64); s = pair_s[r].astype(np.int64)
            v_near = v_2b.copy()
            np.add.at(v_near, t, np.einsum("nij,nj->ni", R, F[s])); np.add.at(v_near, s, np.einsum("nji,nj->ni", R, F[t]))
            v_diag = v_near + np.einsum("nij,nj->ni", np.asarray(Mtt_res[c, :P]).astype(np.float64).reshape(P, 6, 6), F)
            D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
            far = np.argwhere(D > args.pair_cutoff)
            v_far = v_diag.copy()
            for a, b in far:
                v_far[a] += (Mb[a, :, b, :] - op.compute_rpy_mobility(pos[b] - pos[a])) @ F[b]
            e = lambda v: float(np.linalg.norm(v - v_true) / np.linalg.norm(v_true) * 100)
            el = lambda v: float(np.linalg.norm((v - v_true)[:, :3]) / np.linalg.norm(v_true[:, :3]) * 100)
            ea = lambda v: float(np.linalg.norm((v - v_true)[:, 3:]) / np.linalg.norm(v_true[:, 3:]) * 100)
            rows.append({"family": FAMILIES[int(cfg["family"][c])], "param": float(cfg["param"][c]), "P": P, "cfg": int(c),
                         "n_near": int(len(r)), "n_far": int(len(far)) // 2,
                         "e_2b": e(v_2b), "e_near": e(v_near), "e_diag": e(v_diag), "e_far": e(v_far),
                         "lin_2b": el(v_2b), "lin_near": el(v_near), "lin_diag": el(v_diag),
                         "ang_2b": ea(v_2b), "ang_near": ea(v_near), "ang_diag": ea(v_diag)})
        print(f"  {FAMILIES[int(k[0])]:8s} param {k[1]:<6g} P {k[2]:3d}: " + "  ".join(
            f"{x}={np.mean([rw[x] for rw in rows[-len(pick):]]):6.2f}" for x in ["e_2b", "e_near", "e_diag", "e_far"]) + f"  [{time.time() - t0:.0f} s]", flush=True)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.5g")
    g = df.groupby(["family", "param"])[["e_2b", "e_near", "e_diag", "e_far", "lin_2b", "lin_near", "ang_2b", "ang_near", "n_near", "n_far"]].mean()
    print("\nmean over P and configs (rel L2 % of the velocity):")
    print(g.round(2).to_string())
    md = ["| family | param | e_2b | e_near (floor of any pairwise correction) | e_diag | e_far | lin 2b -> near | ang 2b -> near |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (fam, par), rr in g.iterrows():
        md.append(f"| {fam} | {par:g} | {rr['e_2b']:.2f} | {rr['e_near']:.2f} | {rr['e_diag']:.2f} | {rr['e_far']:.3f} | {rr['lin_2b']:.2f} -> {rr['lin_near']:.2f} | {rr['ang_2b']:.2f} -> {rr['ang_near']:.2f} |")
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"-> {args.out}, {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
