"""Validation of the batched multi-RHS MFS solver (src/mfs_batched.py) against the fp64 references.

Fast tests run in ~1-2 min on the laptop GPU; `-m slow` adds the larger configurations.  All GPU tests are
skipped without CUDA.  Run from the repo root: python -m pytest tests/test_mfs_batched.py -q
"""
import glob
import math
import os
import sys

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from src.mfs_batched import BatchedMFS, MFSConvergenceError, load_template  # noqa: E402
from src.mfs import imp_mfs_mobility_vec  # noqa: E402


# ----------------------------------------------------------------------------- helpers
def uniform_cfg(phi, P, seed, gap=0.1):
    rng = np.random.default_rng(seed)
    L = (P * 4 / 3 * np.pi / phi) ** (1 / 3)
    half = L / 2
    pos = np.zeros((P, 3))
    for i in range(1, P):
        for _ in range(100000):
            x = rng.uniform(-half + 1, half - 1, 3)
            if np.all(np.linalg.norm(pos[:i] - x, axis=1) >= 2 + gap):
                pos[i] = x
                break
        else:
            raise RuntimeError("placement failed")
    return pos


def grown_cfg(P, delta, seed):
    rng = np.random.default_rng(seed)
    centers = [np.zeros(3)]
    d = 2.0 + delta
    for _ in range(1, P):
        for _ in range(10000):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            c = centers[rng.integers(len(centers))] + d * u
            if all(np.linalg.norm(c - x) >= d - 1e-12 for x in centers):
                centers.append(c)
                break
    return np.array(centers)


def cpu_reference(acc, pos, F, tol=1e-9, max_iter=3000):
    b, s, B_inv = load_template(acc)
    P = pos.shape[0]
    x = imp_mfs_mobility_vec([b + pos[p] for p in range(P)], [s + pos[p] for p in range(P)],
                             [F[p, :3] for p in range(P)], [F[p, 3:] for p in range(P)], [B_inv] * P,
                             max_iter=max_iter, tol=tol)
    M3 = 3 * s.shape[0]
    return np.stack([xp[M3:M3 + 6] for xp in x])


def rel(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.fixture(scope="module")
def m64():
    return BatchedMFS(acc="fine", backend="torch64")


@pytest.fixture(scope="module")
def m32():
    return BatchedMFS(acc="fine", backend="triton32")


# ----------------------------------------------------------------------------- (h) isolated sphere
@pytest.mark.parametrize("acc", ["fine", "Xfine"])
def test_h_isolated_sphere_self_mobility(acc):
    m = BatchedMFS(acc=acc, backend="torch64")
    Ms = m.self_mobility().cpu().numpy()
    ref = np.diag([1 / (6 * math.pi)] * 3 + [1 / (8 * math.pi)] * 3)
    assert np.abs(Ms - ref).max() / (1 / (6 * math.pi)) < 1e-6
    # a single isolated particle through the full solve
    vel, info = m.solve(np.zeros((1, 3)), np.eye(6)[None], tol_v=1e-10)
    assert np.allclose(vel[0].cpu().numpy(), Ms, atol=1e-12)


# ----------------------------------------------------------------------------- (a) fp64 vs CPU reference
@pytest.mark.parametrize("P,kind", [(2, "u"), (3, "u"), (4, "g"), (6, "u")])
def test_a_torch64_matches_cpu_reference(m64, P, kind):
    pos = uniform_cfg(0.05 if P <= 4 else 0.15, P, P) if kind == "u" else grown_cfg(P, 0.1, P)
    F = np.random.default_rng(P).normal(size=(P, 6))
    # the Gauss-Seidel reference converges in strength space and cannot reach 1e-9 for near-contact clusters
    ref_tol, gate = (1e-8, 1e-7) if kind == "g" else (1e-9, 1e-8)
    v_ref = cpu_reference("fine", pos, F, tol=ref_tol)
    for method in ("gmres", "jacobi"):
        vel, info = m64.solve(pos, F[:, :, None], method=method, tol=1e-10, tol_v=1e-10, max_iter=400)
        assert info.converged
        assert rel(vel[:, :, 0].cpu().numpy(), v_ref) < gate, method
    # grand mobility matrix columns are the same solve
    M, info = m64.solve_mobility_matrix(pos, tol=1e-10, tol_v=1e-10)
    vM = (M.cpu().numpy() @ F.reshape(-1)).reshape(P, 6)
    assert rel(vM, v_ref) < gate


@pytest.mark.slow
def test_a_torch64_matches_cpu_reference_xfine():
    m = BatchedMFS(acc="Xfine", backend="torch64")
    pos = uniform_cfg(0.15, 4, 11)
    F = np.random.default_rng(11).normal(size=(4, 6))
    v_ref = cpu_reference("Xfine", pos, F)
    vel, info = m.solve(pos, F[:, :, None], tol=1e-10, tol_v=1e-10)
    assert rel(vel[:, :, 0].cpu().numpy(), v_ref) < 1e-9


# ----------------------------------------------------------------------------- (g) same fixed point
def test_g_jacobi_gmres_same_fixed_point(m64):
    pos = grown_cfg(8, 0.1, 5)
    Mg, ig = m64.solve_mobility_matrix(pos, method="gmres", tol=1e-10, tol_v=1e-10)
    Mj, ij = m64.solve_mobility_matrix(pos, method="jacobi", tol=1e-10, tol_v=1e-10, max_iter=600)
    assert ig.converged and ij.converged
    assert rel(Mj.cpu().numpy(), Mg.cpu().numpy()) < 1e-9
    assert ig.n_matvec < ij.n_matvec


# ----------------------------------------------------------------------------- (b) fp32 kernel vs fp64
@pytest.mark.parametrize("P,phi", [(8, 0.15), (16, 0.2)])
def test_b_triton32_vs_torch64(m64, m32, P, phi):
    pos = uniform_cfg(phi, P, 3)
    M64, _ = m64.solve_mobility_matrix(pos, tol=1e-10, tol_v=1e-8)
    M32, i32 = m32.solve_mobility_matrix(pos)                       # default tol_v = 1e-5
    assert i32.converged and i32.converged_by == "velocity"
    assert rel(M32.cpu().numpy(), M64.cpu().numpy()) < 3e-5
    # single-level accumulation is not measurably worse here (error is the fp32 rounding of the strengths)
    m32b = BatchedMFS(acc="fine", backend="triton32", two_level=False)
    M32b, _ = m32b.solve_mobility_matrix(pos)
    assert rel(M32b.cpu().numpy(), M64.cpu().numpy()) < 3e-5


def test_b_fp32_gemm_is_unusable(m64):
    """Applying the ill-conditioned pseudo-inverse in fp32 breaks the labels (regression documentation)."""
    pos = uniform_cfg(0.15, 8, 3)
    M64, _ = m64.solve_mobility_matrix(pos, tol=1e-10, tol_v=1e-8)
    bad = BatchedMFS(acc="fine", backend="torch64", gemm_dtype=torch.float32)
    Mb, info = bad.solve_mobility_matrix(pos, tol=1e-10, tol_v=1e-8, raise_on_fail=False)
    assert rel(Mb.cpu().numpy(), M64.cpu().numpy()) > 1e-5


# ----------------------------------------------------------------------------- (c) old dataset rows
DATA_PRESENT = len(glob.glob("data/multibody/X_sphere_*.npy")) > 0


@pytest.mark.skipif(not DATA_PRESENT, reason="data/multibody missing")
def test_c_reproduce_old_dataset_rows(m64, m32):
    files = sorted(glob.glob("data/multibody/X_sphere_*.npy"))[:2]
    n_checked = 0
    for f in files:
        X = np.load(f, allow_pickle=True)
        Y = np.load(f.replace("X_sphere_", "Y_sphere_"))
        C = np.load(f.replace("X_sphere_", "neighbors_"))
        for i in range(0, len(Y), max(1, len(Y) // 6)):
            row = np.asarray(X[i], dtype=np.float64); nk = int(C[i])
            pos = np.concatenate([np.zeros((1, 3)), row[:3][None], row[11:].reshape(nk, 3)], 0)
            F = np.zeros((nk + 2, 6)); F[1] = row[5:11]
            # old rows: Gauss-Seidel tol 1e-7 absolute in strength space -> their own accuracy is ~1e-6;
            # the fp32 backend's floor on these tiny systems is ~4e-5
            for m, gate in ((m64, 3e-6), (m32, 1e-4)):
                vel, info = m.solve(pos, F[:, :, None])
                assert rel(vel[0, :, 0].cpu().numpy(), Y[i]) < gate, (f, i, m.backend)
            n_checked += 1
    assert n_checked >= 10


# ----------------------------------------------------------------------------- (d) cached truths
CSV = sorted(glob.glob("tmp/testcase_uniform_*_20.csv"))


@pytest.mark.skipif(not CSV, reason="no cached truth CSVs")
def test_d_vs_cached_truth_csv():
    import pandas as pd
    m = BatchedMFS(acc="Xfine", backend="torch64")
    df = pd.read_csv(CSV[0])
    pos = df[["x", "y", "z"]].values
    F = df[["f_x", "f_y", "f_z", "t_x", "t_y", "t_z"]].values
    V = df[["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]].values
    vel, info = m.solve(pos, F[:, :, None], tol_v=1e-8)
    assert rel(vel[:, :, 0].cpu().numpy(), V) < 5e-6      # truths: GS tol 1e-8 in strength space (~1e-6 velocity)


# ----------------------------------------------------------------------------- (e) symmetry, (f) batching
def test_e_symmetry_reported(m64):
    M, info = m64.solve_mobility_matrix(uniform_cfg(0.1, 12, 7))
    assert info.symm_err is not None and info.symm_err < 1e-3


def test_f_batch_invariance(m32):
    pos = uniform_cfg(0.15, 8, 9)
    M_all, _ = m32.solve_mobility_matrix(pos)
    # (i) columns one at a time == all at once (per-column arithmetic is identical up to reduction order)
    F = BatchedMFS.unit_forces(8)
    vel_single, _ = m32.solve(pos, F[:, :, 7:8])
    assert rel(vel_single[:, :, 0].cpu().numpy(), M_all.cpu().numpy()[:, 7].reshape(8, 6)) < 1e-6
    # (ii) four identical systems batched == one
    Ms, infos = m32.solve_mobility_matrix_batch([pos, pos, pos, pos])
    for Mi in Ms:
        assert torch.equal(Mi, M_all) or rel(Mi.cpu().numpy(), M_all.cpu().numpy()) < 1e-6
    # (iii) column chunking
    M_chunk, info = m32.solve_mobility_matrix(pos, cols_chunk=16)
    assert info.chunks == 3
    assert rel(M_chunk.cpu().numpy(), M_all.cpu().numpy()) < 1e-6


def test_convergence_error_raised():
    m = BatchedMFS(acc="fine", backend="triton32", max_restarts=1, m=2)
    with pytest.raises(MFSConvergenceError):
        m.solve_mobility_matrix(uniform_cfg(0.2, 8, 1), tol=1e-12, tol_v=1e-12)


# ----------------------------------------------------------------------------- generator + loader
def test_generator_and_loader(tmp_path):
    import subprocess
    out = tmp_path / "v2"
    cmd = [sys.executable, "src/create_dataset_multibody_v2.py", "--family", "grown", "--delta", "0.2", "--P", "8",
           "--n-configs", "3", "--backend", "triton32", "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    shards = list(out.glob("**/shard_*.npz"))
    assert len(shards) == 1 and (out / "manifest_worker0.csv").exists()
    # resume skips the existing shard
    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "0 shards" in r.stdout
    from src.nbody_features import load_multibody_v2, pair_rows_from_M
    d = load_multibody_v2(roots=(str(out),), check=True)
    assert len(d["positions"]) == 3 and d["M_ts"].shape[1:] == (6, 6)
    rows = pair_rows_from_M(d, np.random.default_rng(0))
    # convention: Y = velocity of the target due to the source's force, others force-free
    m = BatchedMFS(acc="fine", backend="torch64")
    n = 0
    c, t, s_ = d["cfg"][n], d["t_idx"][n], d["s_idx"][n]
    F = np.zeros((8, 6)); F[s_] = rows["force"][n]
    vel, _ = m.solve(d["positions"][c], F[:, :, None])
    assert rel(rows["Y"][n], vel[t, :, 0].cpu().numpy()) < 1e-4
