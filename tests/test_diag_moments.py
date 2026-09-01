"""Structural tests for the self-block (per-particle diagonal) correction.

CPU only, seconds.  Run from the repo root: python -m pytest tests -q
"""
import numpy as np
import pytest
import torch

from src import nbody_features as nf
from src import nbody_moments as nbm
from tests import diag_moments_ref as ref

torch.set_default_dtype(torch.float32)
DT = torch.float64


# ----------------------------------------------------------------------------- helpers
def random_env(rng, K=9, r_min=2.05, r_max=8.0):
    """K neighbours of a particle at the origin, at hard-sphere-feasible radii."""
    dirs = rng.normal(size=(K, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r = rng.uniform(r_min, r_max, size=K)
    return dirs * r[:, None]


def to_t(*arrays):
    return [torch.as_tensor(np.asarray(a), dtype=DT) for a in arrays]


def random_rotation(rng, det=1.0):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) * det < 0:
        Q[:, 0] *= -1
    return Q


def big_D(R):
    D = np.zeros((6, 6))
    D[:3, :3] = R
    D[3:, 3:] = np.linalg.det(R) * R
    return D


def random_model(seed=0):
    from src.model_archs import SelfBlockMoments
    torch.manual_seed(seed)
    m = SelfBlockMoments(zero_init_head=False).to(DT)
    with torch.no_grad():  # make the coefficients O(1) and non-trivial
        for p in m.parameters():
            p.mul_(3.0)
    return m.eval()


# ----------------------------------------------------------------------------- 1. bands
def test_self_partition_of_unity():
    r = torch.linspace(0.0, 12.0, 2001, dtype=DT)
    w = nbm.self_band_weights(r)
    assert w.shape == (2001, 8)
    assert torch.allclose(w.sum(-1), torch.ones_like(r), atol=1e-12)
    assert torch.all(w >= 0)
    # saturation below the first centre (2.375) and above the last (7.625); no zeroing beyond 8
    assert torch.all(w[r <= 2.375, 0] == 1.0)
    assert torch.all(w[r >= 7.625, 7] == 1.0)
    # continuity in r (grid step 0.006 x slope 4/3)
    assert torch.abs(w[1:] - w[:-1]).max() < 0.01


def test_self_band_counts_and_Q():
    rng = np.random.default_rng(1)
    P, K = 5, 9
    nbr = np.stack([random_env(rng, K) for _ in range(P)])
    mask = (rng.uniform(size=(P, K)) < 0.6).astype(np.float64)
    s, v, Q = nbm.self_band_moments(*to_t(nbr, mask))
    assert torch.allclose(s.sum(-1), torch.as_tensor(mask.sum(-1), dtype=DT), atol=1e-12)
    # Q symmetric traceless
    assert torch.allclose(Q, Q.transpose(-1, -2), atol=1e-12)
    assert torch.allclose(Q.diagonal(dim1=-2, dim2=-1).sum(-1), torch.zeros(P, 8, dtype=DT), atol=1e-12)


# ----------------------------------------------------------------------------- 2. numpy oracle
def test_matches_numpy_reference_self():
    rng = np.random.default_rng(2)
    for _ in range(5):
        nbr = random_env(rng, K=8)
        s_r, v_r, Q_r = ref.self_moments(nbr)
        inv_r = ref.self_invariants(s_r, v_r, Q_r)
        tt_r, tr_r = ref.self_bases(v_r, Q_r)
        c = rng.normal(size=90)
        M_r = ref.self_block(c, v_r, Q_r)

        N, Mk = to_t(nbr[None], np.ones((1, len(nbr))))
        s, v, Q = nbm.self_band_moments(N, Mk)
        assert np.allclose(s[0].numpy(), s_r, atol=1e-12)
        assert np.allclose(v[0].numpy(), v_r, atol=1e-12)
        assert np.allclose(Q[0].numpy(), Q_r, atol=1e-12)
        inv = nbm.self_invariants(s, v, Q)
        assert inv.shape == (1, 48)
        assert np.allclose(inv[0].numpy(), inv_r, atol=1e-10)
        tt, tr = nbm.self_bases(v, Q)
        assert tt.shape == (1, 33, 3, 3) and tr.shape == (1, 24, 3, 3)
        assert np.allclose(tt[0].numpy(), tt_r, atol=1e-12)
        assert np.allclose(tr[0].numpy(), tr_r, atol=1e-12)
        M = nbm.self_assemble_block(torch.as_tensor(c[None], dtype=DT), tt, tr)
        assert np.allclose(M[0].numpy(), M_r, atol=1e-12)


def test_self_pack_unpack_roundtrip():
    rng = np.random.default_rng(3)
    nbr = random_env(rng, K=6)
    N, Mk = to_t(nbr[None], np.ones((1, 6)))
    X = nbm.self_moment_features(N, Mk)
    assert X.shape == (1, nbm.X_DIM_SELF)
    s, v, Q = nbm.self_unpack_features(X)
    s0, v0, Q0 = nbm.self_band_moments(N, Mk)
    assert torch.allclose(s, s0) and torch.allclose(v, v0) and torch.allclose(Q, Q0)
    assert torch.allclose(nbm.self_pack_features(s, v, Q), X)
    # the numpy bridge is the float32 view of the same rows
    Xf = nf.self_moment_features(nbr[None], np.ones((1, 6)))
    assert Xf.shape == (1, nbm.X_DIM_SELF) and Xf.dtype == np.float32
    assert np.allclose(Xf[0], X[0].numpy(), rtol=1e-6, atol=1e-6)


# ----------------------------------------------------------------------------- 3. symmetry (the structural guarantee)
def test_block_symmetric():
    model = random_model(0)
    rng = np.random.default_rng(7)
    for _ in range(5):
        nbr = random_env(rng, K=7)
        X = nbm.self_moment_features(*to_t(nbr[None], np.ones((1, 7))))
        with torch.no_grad():
            K = model.predict_mobility(X)[0]
        assert torch.allclose(K, K.T, atol=1e-12)
        # non-trivial: TR coupling present and TT not isotropic -> the test has teeth
        assert K[:3, 3:].abs().max() > 1e-8 * K.abs().max()
        TT = K[:3, :3]
        assert (TT - TT.trace() / 3 * torch.eye(3, dtype=DT)).abs().max() > 1e-8 * TT.abs().max()


# ----------------------------------------------------------------------------- 4. O(3) equivariance
@pytest.mark.parametrize("kind", ["rotation", "reflection", "improper"])
def test_o3_equivariance(kind):
    model = random_model(1)
    rng = np.random.default_rng(8)
    if kind == "rotation":
        R = random_rotation(rng, +1.0)
    elif kind == "reflection":
        R = np.diag([1.0, 1.0, -1.0])
    else:
        R = random_rotation(rng, +1.0) @ np.diag([1.0, -1.0, 1.0])
    D = torch.as_tensor(big_D(R), dtype=DT)
    for _ in range(3):
        nbr = random_env(rng, K=6)
        X = nbm.self_moment_features(*to_t(nbr[None], np.ones((1, 6))))
        XR = nbm.self_moment_features(*to_t((nbr @ R.T)[None], np.ones((1, 6))))
        with torch.no_grad():
            K = model.predict_mobility(X)[0]
            KR = model.predict_mobility(XR)[0]
            inv = nbm.self_invariants(*nbm.self_unpack_features(X))
            invR = nbm.self_invariants(*nbm.self_unpack_features(XR))
        assert torch.allclose(inv, invR, atol=1e-9, rtol=1e-9)
        assert torch.allclose(KR, D @ K @ D.T, atol=1e-9, rtol=1e-8)
        F = torch.as_tensor(rng.normal(size=(1, 6)), dtype=DT)
        with torch.no_grad():
            v = model.predict_velocity(X, F)[0]
            vR = model.predict_velocity(XR, (D @ F[0])[None])[0]
        assert torch.allclose(vR, D @ v, atol=1e-9, rtol=1e-8)


# ----------------------------------------------------------------------------- 5. permutation / padding
def test_neighbour_permutation_and_padding_invariance():
    rng = np.random.default_rng(6)
    nbr = random_env(rng, K=5)
    nbr_pad = np.zeros((9, 3))
    nbr_pad[:5] = nbr
    mask = np.zeros(9)
    mask[:5] = 1
    X0 = nbm.self_moment_features(*to_t(nbr_pad[None], mask[None]))
    perm = rng.permutation(5)
    nbr2 = np.zeros((9, 3))
    nbr2[4:] = nbr[perm]                                   # shuffled and moved to the back
    mask2 = np.zeros(9)
    mask2[4:] = 1
    nbr2[:4] = rng.normal(size=(4, 3))                     # garbage in masked slots must not matter
    X1 = nbm.self_moment_features(*to_t(nbr2[None], mask2[None]))
    assert torch.allclose(X0, X1, atol=1e-12)


# ----------------------------------------------------------------------------- 6. isolated particle
def test_zero_moments_isolated_particle():
    model = random_model(2)
    X = nbm.self_moment_features(torch.zeros((1, 1, 3), dtype=DT), torch.zeros((1, 1), dtype=DT))
    assert torch.all(X == 0)
    with torch.no_grad():
        c = model.coefficients(X)[0]
        K = model.predict_mobility(X)[0]
    I = torch.eye(3, dtype=DT)
    assert torch.allclose(K[:3, :3], c[0] * I, atol=1e-12)
    assert torch.allclose(K[3:, 3:], c[33] * I, atol=1e-12)
    assert torch.all(K[:3, 3:] == 0) and torch.all(K[3:, :3] == 0)  # no isotropic rank-2 pseudotensor
    # a zero-init head is an exact no-op, neighbours or not
    from src.model_archs import SelfBlockMoments
    zero = SelfBlockMoments(zero_init_head=True).to(DT).eval()
    rng = np.random.default_rng(9)
    nbr = random_env(rng, K=7)
    Xn = nbm.self_moment_features(*to_t(nbr[None], np.ones((1, 7))))
    with torch.no_grad():
        assert torch.all(zero.predict_mobility(Xn) == 0)


# ----------------------------------------------------------------------------- 7. TorchScript
def test_torchscript_roundtrip(tmp_path):
    from src.model_archs import SelfBlockMoments
    torch.manual_seed(3)
    model = SelfBlockMoments(zero_init_head=False).eval()
    rng = np.random.default_rng(10)
    nbr = np.stack([random_env(rng, K=5) for _ in range(4)])
    X = torch.as_tensor(nf.self_moment_features(nbr, np.ones((4, 5))))
    with torch.no_grad():
        model.fit_normalisation(X)
    F = torch.as_tensor(rng.normal(size=(4, 6)), dtype=torch.float32)
    scripted = torch.jit.script(model)
    p = tmp_path / "m.pt"
    scripted.save(str(p))
    loaded = torch.jit.load(str(p)).eval()
    with torch.no_grad():
        v0 = model.predict_velocity(X, F)
        v1 = scripted.predict_velocity(X, F)
        v2 = loaded.predict_velocity(X, F)
        K0 = model.predict_mobility(X)
        K2 = loaded.predict_mobility(X)
    assert v0.abs().max() > 0
    assert torch.allclose(v0, v1, atol=1e-6) and torch.allclose(v0, v2, atol=1e-6)
    assert torch.allclose(K0, K2, atol=1e-6)
    assert loaded.inv_std.min() > 0 and loaded.basis_scale.min() > 0


# ----------------------------------------------------------------------------- 8. selection
def _grown(P, delta, seed):  # as in tests/test_nbody_v2.py
    rng = np.random.default_rng(seed)
    centers = [np.zeros(3)]
    while len(centers) < P:
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        c = centers[rng.integers(len(centers))] + d * (2.0 + delta)
        if all(np.linalg.norm(c - x) >= 2.0 + delta for x in centers):
            centers.append(c)
    return np.array(centers)


def _lattice(P, phi, jitter, seed):
    rng = np.random.default_rng(seed)
    a = ((4 * np.pi / 3) / phi) ** (1 / 3)
    g = np.arange(-4, 5)
    pts = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T * a
    pts = pts[np.argsort(np.linalg.norm(pts, axis=1), kind="stable")[:P]]
    return pts + rng.uniform(-jitter * a, jitter * a, size=pts.shape)


def test_select_particle_neighbours_matches_bruteforce():
    from benchmarks.cluster import uniform_sphere_cluster
    pos_u, _ = uniform_sphere_cluster(0.15, 50, seed=5)
    cfgs = {"uniform": pos_u, "grown": _grown(24, 0.2, 3), "lattice": _lattice(32, 0.1, 0.1, 7)}
    n_checked = 0
    for cutoff in [6.0, 8.0]:
        for name, pos in cfgs.items():
            indptr, indices = nf.select_particle_neighbours(pos, cutoff)
            assert indptr.shape == (len(pos) + 1,) and indptr[0] == 0 and indptr[-1] == len(indices)
            for t in range(len(pos)):
                want = [k for k in range(len(pos)) if k != t and np.linalg.norm(pos[k] - pos[t]) <= cutoff]
                got = indices[indptr[t]:indptr[t + 1]].tolist()
                assert got == want, (name, cutoff, t)
                n_checked += 1
    assert n_checked > 200
    # pad_neighbours consumes the per-particle CSR unchanged
    pos = cfgs["uniform"]
    indptr, indices = nf.select_particle_neighbours(pos, 8.0)
    nbr, mask = nf.pad_neighbours(pos, np.arange(len(pos)), indptr, indices)
    counts = np.diff(indptr)
    assert nbr.shape[0] == len(pos) and np.array_equal(mask.sum(1).astype(np.int64), counts)
    t = int(np.argmax(counts))
    ks = indices[indptr[t]:indptr[t + 1]].astype(np.int64)
    assert np.allclose(nbr[t, :counts[t]], pos[ks] - pos[t])


# ----------------------------------------------------------------------------- 9. operator / cache / trainer
import glob
import json
import os
import subprocess
import sys

SELF_PATH = "data/models/self_interaction_model.pt"
TWO_BODY_PATH = "data/models/two_body_combined_model.pt"
MODELS_PRESENT = all(os.path.exists(p) for p in (SELF_PATH, TWO_BODY_PATH))
V2_SHARDS = sorted(glob.glob("data/multibody_v2/*/*/shard_0000.npz"))
needs_models = pytest.mark.skipif(not MODELS_PRESENT, reason="model files missing")
needs_v2 = pytest.mark.skipif(not V2_SHARDS, reason="dataset v2 missing")


def _random_pair_wt(tmp_path, seed=0, scale=1.0):
    from src.model_archs import MultiBodyMoments
    torch.manual_seed(seed)
    m = MultiBodyMoments(4.69, zero_init_head=False)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(scale)
    wt = tmp_path / "pair.wt"
    torch.save(m.state_dict(), wt)
    return wt


def _diag_wt(tmp_path, zero_init, seed=0, scale=1.0):
    from src.model_archs import SelfBlockMoments
    torch.manual_seed(seed)
    d = SelfBlockMoments(zero_init_head=zero_init)
    with torch.no_grad():
        for p in d.parameters():
            p.mul_(scale)
    dwt = tmp_path / "diag.wt"
    torch.save(d.state_dict(), dwt)
    return dwt


PC8 = dict(switch_dist=8.0, pair_cutoff=8.0, neighbor_cutoff=8.0, max_neighbors=None)


@needs_models
def test_operator_diag_zero_init_is_noop(tmp_path):
    from benchmarks.cluster import uniform_sphere_cluster
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    common = dict(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH,
                  nbody_nn_path=str(_random_pair_wt(tmp_path)), **PC8)
    pos, _ = uniform_sphere_cluster(0.2, 20, seed=1)
    config = np.zeros((len(pos), 7))
    config[:, :3] = pos
    config[:, 6] = 1.0
    F = np.random.default_rng(2).normal(size=(len(pos), 6))
    v0 = Mob_Op_Nbody_Moments(**common).apply(config, F, 1.0)
    dwt = _diag_wt(tmp_path, zero_init=True)
    v1 = Mob_Op_Nbody_Moments(diag_nn_path=str(dwt), diag_cutoff=8.0, **common).apply(config, F, 1.0)
    assert np.array_equal(v0, v1)  # zero head -> K_diag = 0 exactly, bit-for-bit no-op


@needs_models
def test_operator_diag_guard(tmp_path):
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    common = dict(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH,
                  nbody_nn_path=str(_random_pair_wt(tmp_path)))
    dwt = _diag_wt(tmp_path, zero_init=True)
    with pytest.raises(AssertionError):  # K_s range mismatch: labels subtract d<=8, operator adds d<=switch
        Mob_Op_Nbody_Moments(switch_dist=8.0, pair_cutoff=6.0, neighbor_cutoff=8.0, max_neighbors=None,
                             diag_nn_path=str(dwt), **common)


@needs_models
def test_operator_grand_M_symmetric_with_diag(tmp_path):
    from benchmarks.cluster import uniform_sphere_cluster
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    op = Mob_Op_Nbody_Moments(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH,
                              nbody_nn_path=str(_random_pair_wt(tmp_path, seed=5, scale=3.0)),
                              diag_nn_path=str(_diag_wt(tmp_path, zero_init=False, seed=6, scale=3.0)), **PC8)
    pos, _ = uniform_sphere_cluster(0.2, 12, seed=0)
    n = len(pos)
    config = np.zeros((n, 7))
    config[:, :3] = pos
    config[:, 6] = 1.0
    M = np.zeros((6 * n, 6 * n))
    for j in range(6 * n):
        F = np.zeros((n, 6))
        F[j // 6, j % 6] = 1.0
        M[:, j] = op.apply(config, F, 1.0).reshape(-1)
    assert np.linalg.norm(M) > 0
    assert np.linalg.norm(M - M.T) / np.linalg.norm(M) < 1e-5     # float32 models
    Kd = op.diag_blocks(pos)
    assert np.abs(Kd).max() > 0
    assert np.allclose(Kd, np.transpose(Kd, (0, 2, 1)), atol=1e-6)  # per-particle blocks symmetric


def _build_smoke_cache(out, max_shards=2, max_configs=6, pair_cutoff=8.0):
    cmd = [sys.executable, "experiments/build_nbody_v2_cache.py", "--out", str(out), "--max-shards", str(max_shards),
           "--max-configs", str(max_configs), "--workers", "1", "--pair-cutoff", str(pair_cutoff)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


@needs_models
@needs_v2
def test_diag_cache_features_match_operator_rows(tmp_path):
    import experiments.train_diag_v2 as td
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    out = _build_smoke_cache(tmp_path / "cache")
    cache = td.DiagCache(out, "cpu", "cpu")
    cfg = np.load(out / "configs.npz")
    Mtt = np.load(out / "Mtt_res.npy")
    op = Mob_Op_Nbody_Moments(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH,
                              nbody_nn_path=str(_random_pair_wt(tmp_path)),
                              diag_nn_path=str(_diag_wt(tmp_path, zero_init=False)), **PC8)
    row_cfg = cache.row_cfg.numpy()
    row_p = cache.row_p.numpy()
    assert cache.n == int(cfg["P"].sum())
    for c in [0, int(cfg["P"].shape[0]) - 1]:
        P = int(cfg["P"][c])
        pos = cfg["positions"][c, :P]
        rows = np.nonzero(row_cfg == c)[0]
        assert np.array_equal(row_p[rows], np.arange(P))
        X = cache.features(rows).numpy()
        X_op = op._diag_rows(pos)                     # the operator's own rows: same selection, same encoder
        assert np.allclose(X, X_op, rtol=1e-5, atol=1e-6)
        A = Mtt[c, :P].reshape(P, 6, 6).astype(np.float64)
        Rlab = cache.R[rows].numpy().reshape(P, 6, 6)
        assert np.allclose(Rlab, 0.5 * (A + np.transpose(A, (0, 2, 1))), rtol=1e-5, atol=1e-8)


@needs_models
@needs_v2
def test_trainer_smoke_diag(tmp_path):
    from benchmarks.cluster import uniform_sphere_cluster
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    out = _build_smoke_cache(tmp_path / "cache", max_shards=3, max_configs=12)
    run = tmp_path / "run"
    cmd = [sys.executable, "experiments/train_diag_v2.py", "--cache", str(out), "--epochs", "2", "--max-steps", "6",
           "--batch", "64", "--fit-rows", "500", "--eval-rows", "300", "--eval-every", "1",
           "--device", "cpu", "--data-on", "cpu", "--out", str(run), "--publish", "--publish-name", "test_diag"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (run / "model.pt").exists() and (run / "metrics.json").exists()
    m = json.load(open(run / "metrics.json"))
    assert "by_family" in m and m["capture"] > 0 and m["pair_cutoff"] == 8.0
    side = json.load(open("data/models/test_diag.json"))
    assert side["pair_cutoff"] == 8.0 and side["diag_cutoff"] == 8.0
    assert side["nb"] == 8 and side["band_lo"] == 2.0 and side["band_hi"] == 8.0
    for p in ["data/models/test_diag.pt", "data/models/test_diag.json", "experiments/test_diag.wt"]:
        os.remove(p)
    # the published TorchScript model runs inside the operator
    op = Mob_Op_Nbody_Moments(shape="sphere", self_nn_path=SELF_PATH, two_nn_path=TWO_BODY_PATH,
                              nbody_nn_path=str(_random_pair_wt(tmp_path)),
                              diag_nn_path=str(run / "model.pt"), **PC8)
    pos, _ = uniform_sphere_cluster(0.15, 15, seed=2)
    config = np.zeros((15, 7))
    config[:, :3] = pos
    config[:, 6] = 1.0
    v = op.apply(config, np.ones((15, 6)), 1.0)
    assert np.all(np.isfinite(v)) and v.shape == (15, 6)


# ----------------------------------------------------------------------------- 10. harness
def test_harness_diag_sidecar(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pav2_diag", "benchmarks/paper_accuracy_v2.py")
    pav2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pav2)
    assert isinstance(pav2.OP_MODEL["M_mom_v2_kinf_rc8_pc8_diag"], tuple)
    mdl = tmp_path / "d.pt"
    mdl.touch()
    pav2.MODELS["diag_v2_pc8"] = str(mdl)
    with pytest.raises(AssertionError):
        pav2._diag_sidecar_for("diag_v2_pc8")     # sidecar is mandatory for a diag model
    (tmp_path / "d.json").write_text(json.dumps(
        {"pair_cutoff": 8.0, "diag_cutoff": 8.0, "nb": 8, "band_lo": 2.0, "band_hi": 8.0}))
    assert pav2._diag_sidecar_for("diag_v2_pc8") == 8.0
    (tmp_path / "d.json").write_text(json.dumps({"pair_cutoff": 6.0, "diag_cutoff": 8.0}))
    with pytest.raises(AssertionError):
        pav2._diag_sidecar_for("diag_v2_pc8")     # a pc6-labelled diag model must not run at pc8
    # available() requires BOTH model files of a multi-model op
    if os.path.exists(pav2.MODELS["mom_v2_kinf_rc8_pc8"]):
        assert pav2.available(["M_mom_v2_kinf_rc8_pc8_diag"]) == ["M_mom_v2_kinf_rc8_pc8_diag"]
    pav2.MODELS["diag_v2_pc8"] = str(tmp_path / "missing.pt")
    assert pav2.available(["M_mom_v2_kinf_rc8_pc8_diag"]) == []
    assert pav2.available(["M_2b"]) == ["M_2b"]   # ops without model files still pass
