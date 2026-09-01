"""Shared data loading / feature construction for the n-body correction models (warp-free).

Used by ``experiments/train_nbody_moments.py``, the CPU operators and the tests, so that the
training features and the inference features come from the same code.

Dataset (``src/create_dataset_multibody_cross.py``): each ragged row is
``[x_s, y_s, z_s, dist_s, dist_s - 2, F(3), T(3), then nk neighbour positions]`` with the
target at the origin, the force/torque applied to the source only, and ``Y`` the target's
6-velocity from an MFS solve.  ``neighbors_*.npy`` holds nk (1..10).
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from scipy.spatial.distance import pdist

try:
    from src import nbody_moments as nbm
except ImportError:  # imported with src/ on sys.path
    import nbody_moments as nbm

DEFAULT_ROOTS: Sequence[str] = ("data/multibody", "data/multibody_new")
MAX_NEIGHBORS = 10
MEDIAN_2B = 5.008307682776568      # 2-body training notebook; used for the residual labels
MEAN_DIST_S = 4.690027344329476    # Mob_Op_Nbody.DEFAULT_MEAN_DIST_S (= mean dist_s of the dataset)
FEATURE_NAMES = ['r_sum', 'r_diff', 'r_prod', 'u_over_ell_abs', 'u_over_ell_sq', 'rho_over_ell',
                 'inv_sum', 'inv_prod', 'inv_absdiff', 'cos_k']


def load_multibody_dataset(roots: Sequence[str] = DEFAULT_ROOTS, max_neighbors: int = MAX_NEIGHBORS,
                           check: bool = True) -> dict:
    """Load and pad the ragged multibody dataset in a deterministic (sorted) file order."""
    files = []
    for root in roots:
        for f in sorted(os.listdir(root)):
            if f.startswith("X_sphere_") and f.endswith(".npy"):
                files.append((root, f[len("X_sphere_"):-4]))
    assert files, f"no X_sphere_*.npy under {roots}"
    xs, ys, cs = [], [], []
    for root, tag in files:
        x = np.load(f"{root}/X_sphere_{tag}.npy", allow_pickle=True)
        y = np.load(f"{root}/Y_sphere_{tag}.npy")
        c = np.load(f"{root}/neighbors_{tag}.npy")
        assert len(x) == len(y) == len(c), (root, tag)
        xs.append(x); ys.append(y); cs.append(c)
    X_ragged = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0).astype(np.float64)
    counts = np.concatenate(cs, axis=0).astype(np.int64)
    N = len(Y)
    K = max_neighbors

    s_vec = np.zeros((N, 3)); dist = np.zeros(N); min_dist = np.zeros(N)
    force = np.zeros((N, 6)); nbr = np.zeros((N, K, 3)); mask = np.zeros((N, K))
    for i in range(N):
        row = np.asarray(X_ragged[i], dtype=np.float64)
        nk = int(counts[i])
        assert row.shape == (11 + 3 * nk,), (i, row.shape, nk)
        s_vec[i] = row[:3]; dist[i] = row[3]; min_dist[i] = row[4]; force[i] = row[5:11]
        nbr[i, :nk] = row[11:].reshape(nk, 3); mask[i, :nk] = 1.0

    if check:
        assert counts.min() >= 1 and counts.max() <= K, (counts.min(), counts.max())
        assert np.allclose(dist, np.linalg.norm(s_vec, axis=1))
        assert np.allclose(min_dist, dist - 2.0)
        for i in range(N):  # no overlaps anywhere (notebook sanity check)
            nk = int(counts[i])
            pos = np.concatenate([np.zeros((1, 3)), s_vec[i][None], nbr[i, :nk]], 0)
            assert pdist(pos).min() > 2.02, i
    return {"s_vec": s_vec, "dist": dist, "force": force, "nbr": nbr, "mask": mask, "Y": Y,
            "counts": counts, "mean_dist_s": float(dist.mean()), "files": files}


def pair_scalars(dist: np.ndarray, mean_dist_s: float) -> np.ndarray:
    """[d - mean, d - 2, (d - mean)^2, (d - mean)^4]  (baseline cols 33..36; nbody_moments.pair_scalars)."""
    dc = dist - mean_dist_s
    return np.stack([dc, dist - 2.0, dc ** 2, dc ** 4], -1)


def moment_features(s_vec: np.ndarray, nbr: np.ndarray, mask: np.ndarray, mean_dist_s: float,
                    chunk: int = 65536) -> np.ndarray:
    """Model input rows X[N, 111] (float32) from target-frame geometry, moments computed in float64."""
    out = []
    for i in range(0, len(s_vec), chunk):
        S = torch.as_tensor(s_vec[i:i + chunk], dtype=torch.float64)
        Nb = torch.as_tensor(nbr[i:i + chunk], dtype=torch.float64)
        Mk = torch.as_tensor(mask[i:i + chunk], dtype=torch.float64)
        out.append(nbm.moment_features(S, Nb, Mk, float(mean_dist_s)).to(torch.float32).numpy())
    return np.concatenate(out, 0)


def baseline_features(s_vec: np.ndarray, nbr: np.ndarray, mask: np.ndarray, mean_dist_s: float,
                      eps: float = 1e-9) -> np.ndarray:
    """The 147-column baseline row (branch1_multibody_pinn.ipynb == Mob_Op_Nbody._build_pair_feature_vector):
    [s_vec(3) | 10 neighbour vectors (30) | 4 pair scalars | 10 x 10 swap-even feats | 10 mask]."""
    s = np.asarray(s_vec, dtype=np.float64)
    ks = np.asarray(nbr, dtype=np.float64)
    k_mask = np.asarray(mask).astype(bool)
    N, K, _ = ks.shape
    t = np.zeros_like(s)
    ell = np.clip(np.linalg.norm(s - t, axis=1, keepdims=True), eps, None)
    zhat = (s - t) / ell
    m = 0.5 * (s + t)

    r_sk = np.linalg.norm(ks - s[:, None, :], axis=2)
    r_kt = np.linalg.norm(ks - t[:, None, :], axis=2)
    r_sk_c = np.clip(r_sk, eps, None)
    r_kt_c = np.clip(r_kt, eps, None)
    r_sum = r_sk + r_kt
    r_diff = np.abs(r_sk - r_kt)
    r_prod = r_sk * r_kt

    v = ks - m[:, None, :]
    u = np.sum(v * zhat[:, None, :], axis=2)
    v_perp = v - u[..., None] * zhat[:, None, :]
    rho = np.linalg.norm(v_perp, axis=2)
    ell_b = np.broadcast_to(ell, r_sk.shape)
    u_over_ell_abs = np.abs(u) / np.clip(ell_b, eps, None)
    u_over_ell_sq = (u / np.clip(ell_b, eps, None)) ** 2
    rho_over_ell = rho / np.clip(ell_b, eps, None)

    inv_sum = (1.0 / r_sk_c) + (1.0 / r_kt_c)
    inv_prod = 1.0 / (r_sk_c * r_kt_c)
    inv_absdiff = np.abs((1.0 / r_sk_c) - (1.0 / r_kt_c))

    a = s[:, None, :] - ks
    b = t[:, None, :] - ks
    cos_k = np.sum(a * b, axis=2) / np.clip(r_sk_c * r_kt_c, eps, None)
    cos_k = np.clip(cos_k, -1.0, 1.0)

    feats = np.stack([r_sum, r_diff, r_prod, u_over_ell_abs, u_over_ell_sq, rho_over_ell,
                      inv_sum, inv_prod, inv_absdiff, cos_k], axis=2)
    feats[~k_mask] = 0.0
    ks_masked = np.where(k_mask[..., None], ks, 0.0)

    dist = np.linalg.norm(s, axis=1)
    X = np.concatenate([s, ks_masked.reshape(N, 3 * K), pair_scalars(dist, mean_dist_s),
                        feats.reshape(N, 10 * K), k_mask.astype(np.float64)], axis=1)
    return X.astype(np.float32)


def two_body_velocity(two_nn, s_vec: np.ndarray, dist: np.ndarray, force_s: np.ndarray,
                      median_2b: float = MEDIAN_2B, mu: float = 1.0, device="cpu",
                      chunk: int = 65536) -> np.ndarray:
    """Two-body NN prediction of the target's velocity due to the source's force/torque, in the
    convention of the mobility operators: X2b = [+s_vec, d, (d-median)^2, (..)^4, d-2] with
    s_vec = source - target (``NNMob.get_two_vel``: ``center2 = pos[s] - pos[t]``), and
    ``two_nn.predict_velocity(X2b, F_target = 0, F_source, mu)``.

    The saved notebook (branch1_multibody_pinn.ipynb, ``predict_two_body_from_triplet``) feeds
    ``-s_vec`` instead.  L1/L2 are even in the axis but L3 is odd, so that flips the sign of the
    RT/TR blocks of the two-body prediction and its residual labels absorb a spurious -2*RT term
    (its "2-body only" errors of 38% / 199% are that artefact; with the operator convention they
    are 6.9% / 16.5% and the shipped nbody_pinn_b1.pt reproduces the notebook's 3.8% / 10.9%).
    A model trained on the notebook's labels is inconsistent with the operators, which compute the
    two-body term with +s_vec (tests/test_nbody_moments.py pins this against ``NNMob.apply``)."""
    out = []
    mu_t = torch.tensor(float(mu), dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(0, len(s_vec), chunk):
            sv = torch.as_tensor(s_vec[i:i + chunk], dtype=torch.float32, device=device)
            d = torch.as_tensor(dist[i:i + chunk], dtype=torch.float32, device=device)
            r2 = (d - median_2b) ** 2
            X2b = torch.cat([sv, d[:, None], r2[:, None], (r2 * r2)[:, None], (d - 2.0)[:, None]], 1)
            F = torch.as_tensor(force_s[i:i + chunk], dtype=torch.float32, device=device)
            out.append(two_nn.predict_velocity(X2b, torch.zeros_like(F), F, mu_t).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


def make_split(N: int, frac: float = 0.8, seed: int = 41):
    """Deterministic CPU train/val permutation split."""
    perm = np.random.default_rng(seed).permutation(N)
    n_train = int(frac * N)
    return perm[:n_train], perm[n_train:]


# ----------------------------------------------------------------------------- dataset v2 (grand mobility matrices)
def load_multibody_v2(roots: Sequence[str] = ("data/multibody_v2",), r_max: float = 8.0, families=None,
                      symmetrize: bool = False, check: bool = True, max_configs: int | None = None) -> dict:
    """Ordered-pair view of the v2 dataset (src/create_dataset_multibody_v2.py).

    Every shard holds configurations (positions (n,P,3)) and their full grand mobility matrices M (n,6P,6P),
    [U;Omega]_t = sum_s M_ts [F;T]_s.  Returns, for every ordered pair (t, s) with |x_s - x_t| <= r_max:
      s_vec (n,3) = x_s - x_t (operator convention), dist (n), M_ts (n,6,6), M_st (n,6,6), M_tt (n,6,6),
      nbr (n,K,3) = the other P-2 particles relative to the target (padded), mask (n,K), K = max P - 2,
      cfg (n) configuration index, t_idx, s_idx, plus per-configuration lists positions / M / family / params.
    """
    import glob
    import json
    shards = []
    for root in roots:
        shards += sorted(glob.glob(f"{root}/**/shard_*.npz", recursive=True))
    assert shards, f"no shards under {roots}"
    cfg_pos, cfg_M, cfg_meta = [], [], []
    for f in shards:
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        if families is not None and meta["family"] not in families:
            continue
        for i in range(d["positions"].shape[0]):
            cfg_pos.append(d["positions"][i].astype(np.float64))
            Mi = d["M"][i].astype(np.float64)
            if symmetrize:
                Mi = 0.5 * (Mi + Mi.T)
            cfg_M.append(Mi)
            cfg_meta.append({"family": meta["family"], "params": meta["params"], "P": int(meta["P"]),
                             "acc": meta["acc"], "shard": f, "index": i, "seed": int(d["seed"][i]),
                             "symm_err": float(d["symm_err"][i]), "residual": float(d["residual"][i])})
            if max_configs is not None and len(cfg_pos) >= max_configs:
                break
        if max_configs is not None and len(cfg_pos) >= max_configs:
            break
    K = max(p.shape[0] for p in cfg_pos) - 2
    s_vec, dist, M_ts, M_st, M_tt, nbr, mask, cfg_id, t_idx, s_idx = [], [], [], [], [], [], [], [], [], []
    for c, (pos, Mm) in enumerate(zip(cfg_pos, cfg_M)):
        P = pos.shape[0]
        if check:
            assert pdist(pos).min() >= 2.0 + MIN_GAP_V2 - 1e-9, (cfg_meta[c]["shard"], cfg_meta[c]["index"])
        D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        for t in range(P):
            for s in range(P):
                if s == t or D[t, s] > r_max:
                    continue
                others = [k for k in range(P) if k != t and k != s]
                nb = np.zeros((K, 3)); mk = np.zeros(K)
                nb[:len(others)] = pos[others] - pos[t]; mk[:len(others)] = 1.0
                s_vec.append(pos[s] - pos[t]); dist.append(D[t, s])
                M_ts.append(Mm[6 * t:6 * t + 6, 6 * s:6 * s + 6]); M_st.append(Mm[6 * s:6 * s + 6, 6 * t:6 * t + 6])
                M_tt.append(Mm[6 * t:6 * t + 6, 6 * t:6 * t + 6])
                nbr.append(nb); mask.append(mk); cfg_id.append(c); t_idx.append(t); s_idx.append(s)
    out = {"s_vec": np.array(s_vec), "dist": np.array(dist), "M_ts": np.array(M_ts), "M_st": np.array(M_st),
           "M_tt": np.array(M_tt), "nbr": np.array(nbr), "mask": np.array(mask), "cfg": np.array(cfg_id),
           "t_idx": np.array(t_idx), "s_idx": np.array(s_idx), "positions": cfg_pos, "M": cfg_M, "meta": cfg_meta,
           "K": K}
    return out


MIN_GAP_V2 = 0.05


def pair_rows_from_M(data: dict, rng: np.random.Generator, force_scale: float = 6 * np.pi) -> dict:
    """Old-convention training rows from the pair blocks: a random unit force and torque (scaled by
    force_scale) on the source, all other particles force-free, Y = M_ts [F;T]_s = velocity of the target.
    Resample with a fresh rng every epoch for augmentation.  Returns s_vec, dist, force (n,6), nbr, mask, Y."""
    n = data["s_vec"].shape[0]
    f = rng.normal(size=(n, 3)); f /= np.linalg.norm(f, axis=1, keepdims=True)
    t = rng.normal(size=(n, 3)); t /= np.linalg.norm(t, axis=1, keepdims=True)
    force = np.concatenate([f, t], 1) * force_scale
    Y = np.einsum("nij,nj->ni", data["M_ts"], force)
    return {"s_vec": data["s_vec"], "dist": data["dist"], "force": force, "nbr": data["nbr"], "mask": data["mask"],
            "Y": Y, "M_ts": data["M_ts"], "M_tt": data["M_tt"], "cfg": data["cfg"]}


def iter_multibody_v2_configs(roots: Sequence[str] = ("data/multibody_v2",), families=None, shuffle_seed=None):
    """Lazy iterator over configurations: yields (positions (P,3) f64, M (6P,6P) f64, meta dict) one at a time,
    for training-time pair extraction without materialising the whole ordered-pair table
    (`load_multibody_v2` builds ~2.4 KB per ordered pair; the full v2 set has ~5e7 pairs, so use it only with
    `max_configs`/`families`, and this iterator for full passes)."""
    import glob
    import json
    shards = []
    for root in roots:
        shards += sorted(glob.glob(f"{root}/**/shard_*.npz", recursive=True))
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(shards)
    for f in shards:
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        if families is not None and meta["family"] not in families:
            continue
        for i in range(d["positions"].shape[0]):
            yield d["positions"][i].astype(np.float64), d["M"][i].astype(np.float64), {
                "family": meta["family"], "params": meta["params"], "P": int(meta["P"]), "acc": meta["acc"],
                "shard": f, "index": i, "seed": int(d["seed"][i]), "symm_err": float(d["symm_err"][i])}


def pairs_from_config(pos: np.ndarray, M: np.ndarray, r_max: float = 8.0):
    """All ordered pairs (t, s) with |x_s - x_t| <= r_max of one configuration: s_vec (n,3), dist (n),
    M_ts (n,6,6), M_tt (n,6,6), nbr (n,P-2,3) relative to the target, t_idx, s_idx."""
    P = pos.shape[0]
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    t_idx, s_idx = np.nonzero((D <= r_max) & ~np.eye(P, dtype=bool))
    s_vec = pos[s_idx] - pos[t_idx]
    Mb = M.reshape(P, 6, P, 6)
    M_ts = Mb[t_idx, :, s_idx, :]
    M_tt = Mb[t_idx, :, t_idx, :]
    idx = np.arange(P)
    nbr = np.empty((len(t_idx), P - 2, 3))
    for k, (t, s) in enumerate(zip(t_idx, s_idx)):
        others = idx[(idx != t) & (idx != s)]
        nbr[k] = pos[others] - pos[t]
    return {"s_vec": s_vec, "dist": D[t_idx, s_idx], "M_ts": M_ts, "M_tt": M_tt, "nbr": nbr,
            "t_idx": t_idx, "s_idx": s_idx}


# ----------------------------------------------------------------------------- v2 training helpers (operator-consistent)
MEDIAN_2B_OPERATOR = 5.01   # NNMob.get_two_vel's constant; v2 labels use it (the old labels used MEDIAN_2B = 5.0083)
SELECTION_VARIANTS = {"k10_rc6": (10, 6.0), "kinf_rc6": (None, 6.0), "kinf_rc8": (None, 8.0)}  # (max_neighbors, cutoff)


def select_pair_neighbours(pos: np.ndarray, pair_cutoff: float = 6.0, neighbor_cutoff: float = 6.0,
                           max_neighbors: int | None = MAX_NEIGHBORS, chunk: int = 4096):
    """Neighbourhoods of all unordered near pairs of one configuration, with the operators' semantics.

    Pairs: t < s with |x_s - x_t| <= pair_cutoff.  Neighbours of a pair: every k not in {t, s} with
    |x_k - midpoint| <= neighbor_cutoff, the ``max_neighbors`` smallest d_kt * d_ks (stable sort, i.e. ties
    broken by index; ``None`` = all of them), returned in ascending index order — exactly
    ``Mob_Op_Nbody._select_neighbor_indices`` / ``Mob_Op_Nbody_Moments._select_neighbor_indices``.
    Returns t_idx (n,), s_idx (n,) int64 and the CSR neighbour lists indptr (n+1,) int64, indices (nnz,) int16.
    Pairs with zero neighbours are kept (empty list); the operators skip them, callers filter with
    ``np.diff(indptr) > 0``."""
    pos = np.asarray(pos, dtype=np.float64)
    P = pos.shape[0]
    assert P < 32768, "int16 neighbour indices"
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    iu = np.triu_indices(P, 1)
    near = D[iu] <= pair_cutoff
    t_idx = iu[0][near].astype(np.int64)
    s_idx = iu[1][near].astype(np.int64)
    n = len(t_idx)
    counts = np.zeros(n, dtype=np.int64)
    parts = []
    for i0 in range(0, n, chunk):
        t = t_idx[i0:i0 + chunk]; s = s_idx[i0:i0 + chunk]
        m = len(t)
        mid = 0.5 * (pos[t] + pos[s])
        dmid = np.linalg.norm(pos[None, :, :] - mid[:, None, :], axis=-1)      # (m, P)
        cand = dmid <= neighbor_cutoff
        ar = np.arange(m)
        cand[ar, t] = False
        cand[ar, s] = False
        if max_neighbors is None:
            counts[i0:i0 + m] = cand.sum(1)
            parts.append(np.nonzero(cand)[1])            # row-major -> ascending index within each pair
        else:
            K = int(max_neighbors)
            score = np.where(cand, D[t] * D[s], np.inf)
            order = np.argsort(score, axis=1, kind="stable")[:, :K]              # (m, K)
            chosen = np.take_along_axis(cand, order, axis=1)
            sel = np.where(chosen, order, P)                                     # P = sentinel, sorts last
            sel.sort(axis=1)
            counts[i0:i0 + m] = chosen.sum(1)
            parts.append(sel[sel < P])
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    indices = (np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)).astype(np.int16)
    return t_idx, s_idx, indptr, indices


def pad_neighbours(pos: np.ndarray, t_idx: np.ndarray, indptr: np.ndarray, indices: np.ndarray, K: int | None = None):
    """Target-relative padded neighbour tensor (n, K, 3) and mask (n, K) from CSR neighbour lists."""
    counts = np.diff(indptr)
    n = len(t_idx)
    K = int(counts.max()) if (K is None and n) else (K or 1)
    assert counts.max(initial=0) <= K
    nbr = np.zeros((n, K, 3), dtype=np.float64)
    mask = np.zeros((n, K), dtype=np.float64)
    if n == 0:
        return nbr, mask
    row = np.repeat(np.arange(n), counts)
    col = np.arange(len(indices)) - np.repeat(indptr[:-1], counts)
    nbr[row, col] = pos[indices.astype(np.int64)] - pos[t_idx[row]]
    mask[row, col] = 1.0
    return nbr, mask


def two_body_blocks(two_nn, s_vec: np.ndarray, dist: np.ndarray, median_2b: float = MEDIAN_2B_OPERATOR,
                    mu: float = 1.0, device="cpu", chunk: int = 65536) -> np.ndarray:
    """The two-body cross block M2b_ts (n, 6, 6) float64 that the operators apply to the source's force:
    ``predict_mobility(X2b)[1]`` (NNMob.get_two_vel stores it as M[t, s]), X2b = [+s_vec, d, (d-med)^2,
    (d-med)^4, d-2].  ``two_body_velocity`` == ``einsum('nij,nj->ni', two_body_blocks, F)``."""
    out = []
    with torch.no_grad():
        for i in range(0, len(s_vec), chunk):
            sv = torch.as_tensor(s_vec[i:i + chunk], dtype=torch.float32, device=device)
            d = torch.as_tensor(dist[i:i + chunk], dtype=torch.float32, device=device)
            r2 = (d - median_2b) ** 2
            X2b = torch.cat([sv, d[:, None], r2[:, None], (r2 * r2)[:, None], (d - 2.0)[:, None]], 1)
            _, M_t = two_nn.predict_mobility(X2b)
            out.append((M_t / mu).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


def baseline_features_torch(s_vec: torch.Tensor, nbr: torch.Tensor, mask: torch.Tensor, mean_dist_s: float,
                            eps: float = 1e-9) -> torch.Tensor:
    """Torch port of ``baseline_features`` (147 columns at K = 10), any device / float dtype; returns float32."""
    s = s_vec
    ks = nbr
    k_mask = mask > 0.5
    N, K, _ = ks.shape
    ell = torch.linalg.norm(s, dim=1, keepdim=True).clamp_min(eps)
    zhat = s / ell
    m = 0.5 * s
    r_sk = torch.linalg.norm(ks - s[:, None, :], dim=2)
    r_kt = torch.linalg.norm(ks, dim=2)
    r_sk_c = r_sk.clamp_min(eps)
    r_kt_c = r_kt.clamp_min(eps)
    v = ks - m[:, None, :]
    u = (v * zhat[:, None, :]).sum(2)
    rho = torch.linalg.norm(v - u[..., None] * zhat[:, None, :], dim=2)
    a = s[:, None, :] - ks
    b = -ks
    cos_k = ((a * b).sum(2) / (r_sk_c * r_kt_c).clamp_min(eps)).clamp(-1.0, 1.0)
    feats = torch.stack([r_sk + r_kt, (r_sk - r_kt).abs(), r_sk * r_kt, u.abs() / ell, (u / ell) ** 2, rho / ell,
                         1.0 / r_sk_c + 1.0 / r_kt_c, 1.0 / (r_sk_c * r_kt_c), (1.0 / r_sk_c - 1.0 / r_kt_c).abs(),
                         cos_k], dim=2)
    feats = torch.where(k_mask[..., None], feats, torch.zeros_like(feats))
    ks_masked = torch.where(k_mask[..., None], ks, torch.zeros_like(ks))
    dist = torch.linalg.norm(s, dim=1)
    dc = dist - mean_dist_s
    pair = torch.stack([dc, dist - 2.0, dc ** 2, dc ** 4], 1)
    X = torch.cat([s, ks_masked.reshape(N, 3 * K), pair, feats.reshape(N, 10 * K), k_mask.to(s.dtype)], 1)
    return X.to(torch.float32)


def predict_blocks(model, X: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """(n, 6, 6) correction blocks: ``predict_mobility`` when the model has it, else six unit-force
    ``predict_velocity`` calls (the old notebook-class ``nbody_pinn_b1.pt``)."""
    out = []
    has_pm = hasattr(model, "predict_mobility")
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            Xc = X[i:i + chunk]
            if has_pm:
                out.append(model.predict_mobility(Xc))
            else:
                cols = []
                for j in range(6):
                    F = torch.zeros((Xc.shape[0], 6), dtype=Xc.dtype, device=Xc.device)
                    F[:, j] = 1.0
                    cols.append(model.predict_velocity(Xc, F))
                out.append(torch.stack(cols, 2))
    return torch.cat(out, 0)


def v2_is_val(seed, val_every: int = 10):
    """Configuration-level validation flag for dataset v2: seeds are unique per configuration, so
    ``seed % val_every == 0`` is an order-independent, family-stratified 1/val_every split."""
    return (np.asarray(seed, dtype=np.int64) % val_every) == 0


# ----------------------------------------------------------------------------- per-particle diagonal (self-block) helpers
def select_particle_neighbours(pos: np.ndarray, cutoff: float = 8.0):
    """Neighbourhoods of every particle of one configuration: all k != t with |x_k - x_t| <= cutoff,
    in ascending index order.  THE single code path shared by the diagonal trainer
    (experiments/train_diag_v2.py), the operator (``Mob_Op_Nbody_Moments._diag_rows``) and the
    ceiling script.  Returns CSR indptr (N+1,) int64, indices (nnz,) int16.  Zero-neighbour
    particles keep an empty list (their moments are zero and the model output reduces to its
    constant coefficients); ``pad_neighbours(pos, np.arange(N), indptr, indices)`` builds the
    padded particle-relative tensor."""
    pos = np.asarray(pos, dtype=np.float64)
    N = pos.shape[0]
    assert N < 32768, "int16 neighbour indices"
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    cand = D <= cutoff
    np.fill_diagonal(cand, False)
    indptr = np.zeros(N + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(cand.sum(1))
    indices = np.nonzero(cand)[1].astype(np.int16)   # row-major -> ascending index per particle
    return indptr, indices


def self_moment_features(nbr: np.ndarray, mask: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """Self-model input rows X[N, 104] (float32) from particle-relative geometry, moments in float64."""
    out = []
    for i in range(0, len(nbr), chunk):
        Nb = torch.as_tensor(nbr[i:i + chunk], dtype=torch.float64)
        Mk = torch.as_tensor(mask[i:i + chunk], dtype=torch.float64)
        out.append(nbm.self_moment_features(Nb, Mk).to(torch.float32).numpy())
    return np.concatenate(out, 0)
