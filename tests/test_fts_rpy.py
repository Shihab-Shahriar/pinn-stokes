"""Tests for src/fts_rpy.py: the FTS (stresslet-level) RPY blocks and the stresslet reflection.

The block-level checks compare against Stokesian Dynamics' own generate_Minfinity (SD_ROOT, see
src/sd_ops.py) and are skipped without that checkout; the structural checks (symmetry, equivariance,
chunking, viscosity scaling) run everywhere.  Run from the repo root:
    TORCH_COMPILE_DISABLE=1 python -m pytest tests/test_fts_rpy.py -q
"""
import numpy as np
import pytest
import torch

from src import fts_rpy as fr
from src import sd_ops as S

HAS_SD = (S.sd_package_dir() / "settings.py").exists()
needs_sd = pytest.mark.skipif(not HAS_SD, reason=f"Stokesian Dynamics checkout not found at {S.SD_ROOT}")


def rsa(rng, N, side, min_dist=2.1):
    pts = []
    while len(pts) < N:
        p = rng.uniform(-side / 2, side / 2, 3)
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    return np.array(pts)


def cfg(pos):
    return np.hstack([pos, np.tile([0.0, 0.0, 0.0, 1.0], (len(pos), 1))])


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(7)
    pos = rsa(rng, 14, 11.0)
    F = rng.normal(size=(len(pos), 6))
    return pos, F


@needs_sd
@pytest.mark.parametrize("mu", [1.0, 2.5])
def test_blocks_match_stokesian_dynamics(case, mu):
    pos, _ = case
    S._import_sd()
    from functions.generate_Minfinity import generate_Minfinity
    Msd, _ = generate_Minfinity(S.SDMob().posdata(pos), mu=mu)
    Mme = fr.assemble_minfinity(pos, mu)
    N = len(pos)
    n3, n6, n11 = 3 * N, 6 * N, 11 * N
    scale = np.abs(Msd).max()
    for name, (r0, r1, c0, c1) in {"A": (0, n6, 0, n6), "G_U": (0, n3, n6, n11), "G_Omega": (n3, n6, n6, n11),
                                   "Mm": (n6, n11, n6, n11), "EF": (n6, n11, 0, n6)}.items():
        err = np.abs(Msd[r0:r1, c0:c1] - Mme[r0:r1, c0:c1]).max() / scale
        assert err < 1e-12, (name, err)


@needs_sd
def test_full_reflection_reproduces_sd_far_field(case):
    """RPY + the converged stresslet reflection is SD's far-field mobility (Minfinity, E = 0) exactly."""
    pos, F = case
    import benchmarks.paper_accuracy_v2 as pav
    v_rpy = pav.build_op("M_rpy").apply(cfg(pos), F, 1.0)
    v_sd = S.SDMob(minfinity_only=True).apply(cfg(pos), F, 1.0)
    B = fr.reflection_blocks(pos, 1.0, "full")
    dv = torch.einsum("tsab,sb->ta", B, torch.as_tensor(F)).numpy()
    assert np.linalg.norm(v_rpy + dv - v_sd) / np.linalg.norm(v_sd) < 1e-10
    # the truncated reflections converge towards it
    e = []
    for order in ("1", "2"):
        Bo = fr.reflection_blocks(pos, 1.0, order)
        dvo = torch.einsum("tsab,sb->ta", Bo, torch.as_tensor(F)).numpy()
        e.append(np.linalg.norm(v_rpy + dvo - v_sd) / np.linalg.norm(v_sd))
    assert e[1] < e[0] < 0.05, e


@pytest.mark.parametrize("order", ["1", "2", "full"])
def test_reflection_blocks_symmetric_and_batched(case, order):
    pos, _ = case
    B = fr.reflection_blocks(pos, 1.0, order)
    M = fr._flat(B)
    assert torch.linalg.norm(M - M.T) / torch.linalg.norm(M) < 1e-12
    # batched configurations give the same blocks
    Bb = fr.reflection_blocks(np.stack([pos, pos[::-1]]), 1.0, order)
    assert torch.allclose(Bb[0], B, atol=1e-14, rtol=1e-12)
    assert torch.allclose(Bb[1], fr.reflection_blocks(pos[::-1].copy(), 1.0, order), atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("order", ["1", "2"])
@pytest.mark.parametrize("excl", [None, 6.0])
def test_reflection_velocity_matches_blocks_and_chunking(case, order, excl):
    pos, F = case
    B = fr.reflection_blocks(pos, 1.0, order, diag_exclude_within=excl)
    dv = torch.einsum("tsab,sb->ta", B, torch.as_tensor(F))
    for rows in (None, 1, 5):
        V = fr.reflection_velocity(pos, F, 1.0, order, row_chunk=rows, diag_exclude_within=excl)
        assert torch.allclose(V, dv, atol=1e-14, rtol=1e-11)


def test_diag_exclusion_is_the_two_body_path(case):
    """Excluding within R removes exactly -G_tk G_tk^T / D of the near pairs from the diagonal, nothing else."""
    pos, _ = case
    B0 = fr.reflection_blocks(pos, 1.0, "1")
    B1 = fr.reflection_blocks(pos, 1.0, "1", diag_exclude_within=5.0)
    N = len(pos)
    off = ~torch.eye(N, dtype=torch.bool)
    assert torch.equal(B0[off], B1[off])
    D = fr.d_self(1.0)
    for t in range(N):
        expect = torch.zeros(6, 6, dtype=torch.float64)
        for k in range(N):
            if k != t and np.linalg.norm(pos[t] - pos[k]) <= 5.0:
                Bk = fr.reflection_blocks(pos[[t, k]], 1.0, "1")     # two-sphere: diagonal = -G G^T / D
                expect += Bk[0, 0]
        assert torch.allclose(B1[t, t] - B0[t, t], -expect, atol=1e-16, rtol=1e-12)


def test_equivariance_and_viscosity(case):
    pos, F = case
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.linalg.det(Q)
    V1 = fr.reflection_velocity(pos, F, 1.0, "1").numpy()
    Fr = np.concatenate([F[:, :3] @ Q.T, F[:, 3:] @ Q.T], 1)
    V2 = fr.reflection_velocity(pos @ Q.T + 5.0, Fr, 1.0, "1").numpy()
    V1r = np.concatenate([V1[:, :3] @ Q.T, V1[:, 3:] @ Q.T], 1)
    assert np.abs(V1r - V2).max() < 1e-12 * np.abs(V1).max()
    V3 = fr.reflection_velocity(pos, F, 3.0, "1").numpy()
    assert np.allclose(3.0 * V3, V1, atol=1e-15, rtol=1e-12)
    # G vanishes at r = 0 (no NaN) and the reflection is O(1/r^4) between two spheres
    B4 = fr.reflection_blocks(np.array([[0, 0, 0], [4.0, 0, 0]]), 1.0, "1")
    B8 = fr.reflection_blocks(np.array([[0, 0, 0], [8.0, 0, 0]]), 1.0, "1")
    assert torch.isfinite(B4).all() and torch.isfinite(B8).all()
    ratio = B4[0, 0, 0, 0] / B8[0, 0, 0, 0]
    assert B4[0, 0, 0, 0] < 0 and 12 < ratio < 20, float(ratio)   # 2^4 = 16 up to the Faxen terms
