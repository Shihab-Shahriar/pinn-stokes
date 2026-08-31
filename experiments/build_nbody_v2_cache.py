#!/usr/bin/env python3
"""Geometry + label cache for training n-body corrections on dataset v2 (data/multibody_v2/).

Per configuration: padded positions and metadata.  Per unordered near pair (t < s, |x_s - x_t| <= pair_cutoff):
the symmetrised MFS cross block Mts_sym = 0.5 (M_ts + M_st^T), the operators' two-body cross block M2b
(TwoBodyCombined, median 5.01, +s_vec convention), and the neighbour lists of every selection variant
(k10_rc6 / kinf_rc6 / kinf_rc8 = (max_neighbors, cutoff about the pair midpoint); nbody_features.select_pair_neighbours,
the operators' own selection).  Features are computed on the fly by the trainer, so one cache serves every
variant and both architectures (~2.7 GB for the full set).

    python experiments/build_nbody_v2_cache.py --workers 8            # data/multibody_v2 -> data/multibody_v2_cache
    python experiments/build_nbody_v2_cache.py --max-configs 40 --out /tmp/x   # smoke
    python experiments/build_nbody_v2_cache.py --stats                 # summary of an existing cache

Also stored: the diagonal-block residual Mtt_res = M_tt - M_self(analytic) - sum_{s near t} K_s(t,s) (the two-body
self correction the operators apply), i.e. the part of the operator error no pairwise M_ts correction can fix.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(ROOT)]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src import nbody_features as nf  # noqa: E402

TWO_BODY_PATH = "data/models/two_body_combined_model.pt"
FAMILY_ID = {"uniform": 0, "grown": 1, "lattice": 2}
PMAX = 64
VARIANTS = nf.SELECTION_VARIANTS
MEAN_DIST_S = nf.MEAN_DIST_S
SELF_TT = 1.0 / (6.0 * np.pi)
SELF_RR = 1.0 / (8.0 * np.pi)


def self_block():
    return np.diag([SELF_TT] * 3 + [SELF_RR] * 3)


def list_shards(roots, families=None):
    shards = []
    for root in roots:
        shards += sorted(glob.glob(f"{root}/**/shard_*.npz", recursive=True))
    assert shards, f"no shards under {roots}"
    if families:
        keep = []
        for f in shards:
            meta = json.loads(str(np.load(f)["meta"]))
            if meta["family"] in families:
                keep.append(f)
        shards = keep
    return shards


# ----------------------------------------------------------------------------- per shard
def process_shard(job):
    """One shard -> dict of arrays (configs + pairs + variant neighbour lists)."""
    shard_path, shard_idx, cfg_offset, max_configs, pair_cutoff, variants, two_body_path, out_dir = job
    import torch
    torch.set_num_threads(1)
    two_nn = torch.jit.load(two_body_path, map_location="cpu").eval()
    d = np.load(shard_path)
    meta = json.loads(str(d["meta"]))
    n_cfg = d["positions"].shape[0] if max_configs is None else min(max_configs, d["positions"].shape[0])
    P = int(meta["P"])
    fam = FAMILY_ID[meta["family"]]
    params = meta["params"]
    param = float(params.get("phi", params.get("delta", 0.0)))
    jitter = float(params.get("jitter", 0.0))

    pos_all = np.zeros((n_cfg, PMAX, 3), dtype=np.float64)
    Mtt_res = np.zeros((n_cfg, PMAX, 36), dtype=np.float32)
    pair_cfg, pair_t, pair_s, pair_dist, Mts_sym, M2b, asym = [], [], [], [], [], [], []
    nbr_k10 = []
    csr = {v: ([], []) for v in variants if VARIANTS[v][0] is None}
    S = self_block()
    for i in range(n_cfg):
        pos = d["positions"][i].astype(np.float64)
        M = d["M"][i].astype(np.float64)
        Mb = M.reshape(P, 6, P, 6)
        pos_all[i, :P] = pos
        t_idx, s_idx, _, _ = nf.select_pair_neighbours(pos, pair_cutoff, 0.0, 0)  # pairs only
        n = len(t_idx)
        if n:
            s_vec = pos[s_idx] - pos[t_idx]
            dist = np.linalg.norm(s_vec, axis=1)
            Mts = Mb[t_idx, :, s_idx, :]
            Mst = Mb[s_idx, :, t_idx, :]
            sym = 0.5 * (Mts + np.transpose(Mst, (0, 2, 1)))
            a = np.linalg.norm(Mts - np.transpose(Mst, (0, 2, 1)), axis=(1, 2)) / np.maximum(np.linalg.norm(Mts, axis=(1, 2)), 1e-300)
            # two-body blocks in both orientations (cross block for the label, self correction for Mtt_res)
            with torch.no_grad():
                sv2 = np.concatenate([s_vec, -s_vec], 0)
                d2 = np.concatenate([dist, dist], 0)
                X2b = torch.as_tensor(np.concatenate([sv2, d2[:, None], ((d2 - nf.MEDIAN_2B_OPERATOR) ** 2)[:, None],
                                                      ((d2 - nf.MEDIAN_2B_OPERATOR) ** 4)[:, None], (d2 - 2.0)[:, None]], 1),
                                      dtype=torch.float32)
                K_s, K_t = two_nn.predict_mobility(X2b)
                K_s = K_s.numpy().astype(np.float64); K_t = K_t.numpy().astype(np.float64)
            M2b_ts = K_t[:n]                               # (t,s): source's force -> target's velocity
            # diagonal residual: M_tt - S - sum over near partners of the self correction K_s
            diag = Mb[np.arange(P), :, np.arange(P), :] - S[None]
            np.add.at(diag, t_idx, -K_s[:n])               # (t,s) row: self correction of t due to s
            np.add.at(diag, s_idx, -K_s[n:])               # (s,t) row: self correction of s due to t
            Mtt_res[i, :P] = diag.reshape(P, 36).astype(np.float32)
            pair_cfg.append(np.full(n, cfg_offset + i, dtype=np.int32))
            pair_t.append(t_idx.astype(np.int16)); pair_s.append(s_idx.astype(np.int16))
            pair_dist.append(dist.astype(np.float32))
            Mts_sym.append(sym.reshape(n, 36).astype(np.float32)); M2b.append(M2b_ts.reshape(n, 36).astype(np.float32))
            asym.append(a.astype(np.float32))
            for v in variants:
                K, rc = VARIANTS[v]
                tt, ss, indptr, indices = nf.select_pair_neighbours(pos, pair_cutoff, rc, K)
                assert np.array_equal(tt, t_idx) and np.array_equal(ss, s_idx)
                if K is None:
                    csr[v][0].append(np.diff(indptr).astype(np.int16)); csr[v][1].append(indices)
                else:
                    blk = np.full((n, K), -1, dtype=np.int16)
                    cnt = np.diff(indptr)
                    row = np.repeat(np.arange(n), cnt); col = np.arange(len(indices)) - np.repeat(indptr[:-1], cnt)
                    blk[row, col] = indices
                    nbr_k10.append(blk)
        else:
            diag = Mb[np.arange(P), :, np.arange(P), :] - S[None]
            Mtt_res[i, :P] = diag.reshape(P, 36).astype(np.float32)

    def cat(lst, dtype, shape_tail=()):
        return np.concatenate(lst, 0) if lst else np.zeros((0,) + shape_tail, dtype=dtype)

    out = {
        "cfg_positions": pos_all, "cfg_P": np.full(n_cfg, P, np.int16), "cfg_family": np.full(n_cfg, fam, np.int8),
        "cfg_param": np.full(n_cfg, param, np.float32), "cfg_jitter": np.full(n_cfg, jitter, np.float32),
        "cfg_seed": d["seed"][:n_cfg].astype(np.int64), "cfg_shard": np.full(n_cfg, shard_idx, np.int32),
        "cfg_index": np.arange(n_cfg, dtype=np.int32), "cfg_symm_err": d["symm_err"][:n_cfg].astype(np.float32),
        "cfg_Mtt_res": Mtt_res,
        "pair_cfg": cat(pair_cfg, np.int32), "pair_t": cat(pair_t, np.int16), "pair_s": cat(pair_s, np.int16),
        "pair_dist": cat(pair_dist, np.float32), "Mts_sym": cat(Mts_sym, np.float32, (36,)), "M2b": cat(M2b, np.float32, (36,)),
        "asym": cat(asym, np.float32),
    }
    if "k10_rc6" in variants:
        out["nbr_k10_rc6"] = cat(nbr_k10, np.int16, (10,))
    for v, (cnts, inds) in csr.items():
        out[f"nbr_{v}_counts"] = cat(cnts, np.int16)
        out[f"nbr_{v}_indices"] = cat(inds, np.int16)
    part = Path(out_dir) / "parts" / f"shard_{shard_idx:04d}.npz"
    part.parent.mkdir(parents=True, exist_ok=True)
    with open(str(part) + ".tmp", "wb") as fh:
        np.savez(fh, **out)
    os.replace(str(part) + ".tmp", part)
    return shard_idx, n_cfg, int(out["pair_cfg"].shape[0])


# ----------------------------------------------------------------------------- concatenate
def concatenate(out_dir: Path, n_shards: int, variants):
    parts = [np.load(out_dir / "parts" / f"shard_{k:04d}.npz") for k in range(n_shards)]
    C = sum(int(p["cfg_P"].shape[0]) for p in parts)
    n = sum(int(p["pair_cfg"].shape[0]) for p in parts)
    print(f"[concat] {C} configurations, {n} unordered near pairs", flush=True)
    cfg_keys = ["cfg_positions", "cfg_P", "cfg_family", "cfg_param", "cfg_jitter", "cfg_seed", "cfg_shard", "cfg_index", "cfg_symm_err"]
    cfg = {k: np.concatenate([p[k] for p in parts], 0) for k in cfg_keys}
    cfg["cfg_is_val"] = nf.v2_is_val(cfg["cfg_seed"])
    np.savez(out_dir / "configs.npz", **{k[4:]: v for k, v in cfg.items()})

    def mm(name, dtype, shape):
        return np.lib.format.open_memmap(out_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)

    streams = {"pair_cfg": (np.int32, ()), "pair_t": (np.int16, ()), "pair_s": (np.int16, ()), "pair_dist": (np.float32, ()),
               "Mts_sym": (np.float32, (36,)), "M2b": (np.float32, (36,)), "asym": (np.float32, ())}
    if "k10_rc6" in variants:
        streams["nbr_k10_rc6"] = (np.int16, (10,))
    for v in variants:
        if VARIANTS[v][0] is None:
            streams[f"nbr_{v}_counts"] = (np.int16, ())
    arrays = {k: mm(k, dt, (n,) + sh) for k, (dt, sh) in streams.items()}
    Mtt = mm("Mtt_res", np.float32, (C, PMAX, 36))
    csr_idx = {v: mm(f"nbr_{v}_indices", np.int16, (sum(int(p[f"nbr_{v}_indices"].shape[0]) for p in parts),))
               for v in variants if VARIANTS[v][0] is None}
    o = 0; oc = 0; oi = {v: 0 for v in csr_idx}
    for p in parts:
        m = int(p["pair_cfg"].shape[0]); c = int(p["cfg_P"].shape[0])
        for k in streams:
            arrays[k][o:o + m] = p[k]
        Mtt[oc:oc + c] = p["cfg_Mtt_res"]
        for v in csr_idx:
            q = int(p[f"nbr_{v}_indices"].shape[0])
            csr_idx[v][oi[v]:oi[v] + q] = p[f"nbr_{v}_indices"]; oi[v] += q
        o += m; oc += c
    for v in csr_idx:  # indptr from counts
        cnt = np.asarray(arrays[f"nbr_{v}_counts"], dtype=np.int64)
        indptr = np.zeros(n + 1, dtype=np.int64); indptr[1:] = np.cumsum(cnt)
        np.save(out_dir / f"nbr_{v}_indptr.npy", indptr)
        assert indptr[-1] == csr_idx[v].shape[0]
    for a in list(arrays.values()) + [Mtt] + list(csr_idx.values()):
        a.flush()
    return C, n


# ----------------------------------------------------------------------------- stats
def load_cache(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    cfg = dict(np.load(out_dir / "configs.npz"))
    d = {"cfg": cfg, "meta": json.load(open(out_dir / "meta.json"))}
    for f in sorted(out_dir.glob("*.npy")):
        d[f.stem] = np.load(f, mmap_mode="r")
    return d


def stats(out_dir: Path, truth_glob: str = "tmp/nbody_moments_truth/uniform_N200_*.npz") -> dict:
    import torch
    from src import nbody_moments as nbm
    d = load_cache(out_dir)
    cfg = d["cfg"]; meta = d["meta"]
    n = d["pair_cfg"].shape[0]
    fam_names = {v: k for k, v in FAMILY_ID.items()}
    pair_fam = cfg["family"][d["pair_cfg"]]
    pair_P = cfg["P"][d["pair_cfg"]]
    R = np.asarray(d["Mts_sym"]) - np.asarray(d["M2b"])
    st = {"n_configs": int(cfg["P"].shape[0]), "n_pairs": int(n), "val_frac_configs": float(cfg["is_val"].mean()),
          "val_frac_pairs": float(cfg["is_val"][d["pair_cfg"]].mean()), "by_family": {}, "variants": {}}
    print(f"configs {st['n_configs']}  unordered near pairs {n}  val configs {st['val_frac_configs']:.3f}  val pairs {st['val_frac_pairs']:.3f}")
    blocks = {"TT": (slice(0, 3), slice(0, 3)), "TR": (slice(0, 3), slice(3, 6)), "RT": (slice(3, 6), slice(0, 3)), "RR": (slice(3, 6), slice(3, 6))}

    def rms_blocks(A):  # A (m,36)
        A = A.reshape(-1, 6, 6)
        return {k: float(np.sqrt(np.mean(A[:, i, j] ** 2))) for k, (i, j) in blocks.items()}

    for f_id, name in fam_names.items():
        sel = pair_fam == f_id
        if not sel.any():
            continue
        Ps = sorted(np.unique(pair_P[sel]).tolist())
        row = {"pairs": int(sel.sum()), "configs": int((cfg["family"] == f_id).sum()), "P": Ps,
               "asym_median": float(np.median(d["asym"][sel])), "asym_p99": float(np.percentile(d["asym"][sel], 99)),
               "rms_R": rms_blocks(R[sel]), "rms_Mts": rms_blocks(np.asarray(d["Mts_sym"])[sel])}
        st["by_family"][name] = row
        print(f"  {name:8s} configs {row['configs']:6d} pairs {row['pairs']:8d} P {Ps}  asym median {row['asym_median']:.1e} p99 {row['asym_p99']:.1e}")
        print(f"           RMS |R| TT {row['rms_R']['TT']:.2e} TR {row['rms_R']['TR']:.2e} RR {row['rms_R']['RR']:.2e} | "
              f"|Mts| TT {row['rms_Mts']['TT']:.2e} TR {row['rms_Mts']['TR']:.2e} RR {row['rms_Mts']['RR']:.2e}")
    # residual by distance bin
    dist = np.asarray(d["pair_dist"])
    bins = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    if float(meta.get("pair_cutoff", 6.0)) > 6.0:
        bins += [7.0, float(meta["pair_cutoff"])]
    st["by_dist"] = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (dist >= lo) & (dist < hi)
        rr = rms_blocks(R[sel]); mm_ = rms_blocks(np.asarray(d["Mts_sym"])[sel])
        st["by_dist"][f"{lo}-{hi}"] = {"pairs": int(sel.sum()), "rms_R": rr, "rms_Mts": mm_}
        print(f"  d in [{lo},{hi}): pairs {sel.sum():8d}  |R|/|Mts| TT {rr['TT'] / mm_['TT']:.3f} TR {rr['TR'] / mm_['TR']:.3f} RR {rr['RR'] / mm_['RR']:.3f}")
    # diagonal residual
    Mtt = np.asarray(d["Mtt_res"])
    Pm = cfg["P"]
    valid = np.arange(PMAX)[None, :] < Pm[:, None]
    dd = Mtt[valid].reshape(-1, 6, 6)
    st["diag_res_rms"] = {k: float(np.sqrt(np.mean(dd[:, i, j] ** 2))) for k, (i, j) in blocks.items()}
    st["diag_res_frobenius_rel_to_self"] = float(np.sqrt(np.mean(np.sum(dd ** 2, (1, 2)))) / np.linalg.norm(self_block()))
    print(f"  diagonal residual (M_tt - self - sum K_s): RMS TT {st['diag_res_rms']['TT']:.2e} RR {st['diag_res_rms']['RR']:.2e} "
          f"(Frobenius {st['diag_res_frobenius_rel_to_self']:.3e} of the self block); pairwise |R| TT {rms_blocks(R)['TT']:.2e}")
    # neighbour counts per variant + band coverage
    truths = sorted(glob.glob(truth_glob))
    for v in meta["variants"]:
        K, rc = VARIANTS[v]
        cnt = np.asarray(d[f"nbr_{v}_counts"]) if K is None else (np.asarray(d["nbr_k10_rc6"]) >= 0).sum(1)
        q = np.percentile(cnt, [0, 5, 50, 95, 100])
        st["variants"][v] = {"nnbr_mean": float(cnt.mean()), "nnbr_pct_0_5_50_95_100": q.tolist(), "zero_frac": float((cnt == 0).mean())}
        print(f"  {v}: neighbours mean {cnt.mean():.1f} pct[0,5,50,95,100] {q.tolist()} zero {100 * (cnt == 0).mean():.3f}%")
        # band coverage: mean band counts s_a over a subsample of pairs (v2, by family) vs the N=200 truth configs
        rng = np.random.default_rng(0)
        cov = {}
        for f_id, name in fam_names.items():
            idx = np.nonzero(pair_fam == f_id)[0]
            if idx.size == 0:
                continue
            idx = np.sort(rng.choice(idx, size=min(20000, idx.size), replace=False))
            cov[name] = band_counts_v2(d, idx, v).tolist()
        for tp in truths[:4]:
            t = np.load(tp)
            pos = t["config"][:, :3]
            tt, ss, indptr, indices = nf.select_pair_neighbours(pos, float(meta.get("pair_cutoff", 6.0)), rc, K)
            nbr, mask = nf.pad_neighbours(pos, tt, indptr, indices)
            s_vec = pos[ss] - pos[tt]
            s, _, _ = nbm.band_moments(*[torch.as_tensor(a, dtype=torch.float64) for a in (s_vec, nbr, mask)])
            cov[f"N200_phi{float(t['phi']):g}_seed{int(t['seed'])}"] = s.mean(0).tolist()
        st["variants"][v]["band_counts"] = cov
        for k, val in cov.items():
            print(f"     mean band counts {k:22s} " + " ".join(f"{x:5.2f}" for x in val))
    json.dump(st, open(out_dir / "stats.json", "w"), indent=1)
    return st


def band_counts_v2(d, idx, variant):
    import torch
    from src import nbody_moments as nbm
    cfg = d["cfg"]
    K, rc = VARIANTS[variant]
    pc = np.asarray(d["pair_cfg"])[idx]; pt = np.asarray(d["pair_t"])[idx].astype(np.int64); ps = np.asarray(d["pair_s"])[idx].astype(np.int64)
    pos = cfg["positions"]
    s_vec = pos[pc, ps] - pos[pc, pt]
    if K is None:
        indptr = np.asarray(d[f"nbr_{variant}_indptr"]); ind = d[f"nbr_{variant}_indices"]
        cnt = (indptr[idx + 1] - indptr[idx]); Km = int(cnt.max())
        nb = np.full((len(idx), Km), -1, dtype=np.int64)
        for j, i in enumerate(idx):
            nb[j, :cnt[j]] = ind[indptr[i]:indptr[i + 1]]
    else:
        nb = np.asarray(d["nbr_k10_rc6"])[idx].astype(np.int64)
    mask = (nb >= 0).astype(np.float64)
    nbr = pos[pc[:, None], np.maximum(nb, 0)] - pos[pc, pt][:, None]
    nbr[mask == 0] = 0.0
    s, _, _ = nbm.band_moments(*[torch.as_tensor(a, dtype=torch.float64) for a in (s_vec, nbr, mask)])
    return s.mean(0).numpy()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=["data/multibody_v2"])
    ap.add_argument("--out", type=Path, default=Path("data/multibody_v2_cache"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pair-cutoff", type=float, default=6.0)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--max-configs", type=int, default=None, help="per shard (smoke tests)")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--two-body", default=TWO_BODY_PATH)
    ap.add_argument("--stats", action="store_true", help="only print statistics of an existing cache")
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args()
    if args.stats:
        stats(args.out)
        return
    shards = list_shards(args.roots, args.families)
    if args.max_shards:
        shards = shards[:args.max_shards]
    args.out.mkdir(parents=True, exist_ok=True)
    # global configuration ids follow the sorted shard order
    offsets = []; o = 0
    for f in shards:
        c = int(np.load(f)["positions"].shape[0])
        if args.max_configs is not None:
            c = min(c, args.max_configs)
        offsets.append(o); o += c
    jobs = [(f, k, offsets[k], args.max_configs, args.pair_cutoff, args.variants, args.two_body, str(args.out))
            for k, f in enumerate(shards) if not (args.out / "parts" / f"shard_{k:04d}.npz").exists()]
    print(f"[build] {len(shards)} shards ({len(jobs)} to do), {o} configurations, variants {args.variants}", flush=True)
    t0 = time.time()
    if jobs:
        if args.workers > 1:
            with get_context("spawn").Pool(args.workers) as pool:
                for i, (k, c, n) in enumerate(pool.imap_unordered(process_shard, jobs)):
                    print(f"  shard {k:4d}: {c:5d} configs {n:7d} pairs  [{i + 1}/{len(jobs)}, {time.time() - t0:.0f} s]", flush=True)
        else:
            for i, job in enumerate(jobs):
                k, c, n = process_shard(job)
                print(f"  shard {k:4d}: {c:5d} configs {n:7d} pairs  [{i + 1}/{len(jobs)}, {time.time() - t0:.0f} s]", flush=True)
    C, n = concatenate(args.out, len(shards), args.variants)
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git = "?"
    meta = {"roots": args.roots, "shards": shards, "n_configs": C, "n_pairs": n, "pair_cutoff": args.pair_cutoff,
            "variants": args.variants, "selection": {v: VARIANTS[v] for v in args.variants}, "median_2b": nf.MEDIAN_2B_OPERATOR,
            "two_body": args.two_body, "mean_dist_s": MEAN_DIST_S, "label": "Mts_sym = 0.5 (M_ts + M_st^T); R = Mts_sym - M2b",
            "git": git, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "build_s": time.time() - t0}
    json.dump(meta, open(args.out / "meta.json", "w"), indent=1)
    if not args.keep_parts:
        for f in (args.out / "parts").glob("shard_*.npz"):
            f.unlink()
        (args.out / "parts").rmdir()
    print(f"[done] {C} configurations, {n} pairs in {time.time() - t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
