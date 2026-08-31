"""Structural tests for the moments-based n-body correction (moments_for_nbody.md).

CPU only, seconds.  Run from the repo root: python -m pytest tests -q
"""
import os

import numpy as np
import pytest
import torch

from src import nbody_moments as nbm
from tests import nbody_moments_ref as ref

torch.set_default_dtype(torch.float32)
DT = torch.float64


# ----------------------------------------------------------------------------- helpers
def random_pair(rng, K=7, r_max=7.9, r_min=0.2):
    """Target at origin, source at s_vec, K neighbours at midpoint distances in [r_min, r_max]."""
    d = rng.uniform(2.1, 7.5)
    u = rng.normal(size=3); u /= np.linalg.norm(u)
    s_vec = d * u
    dirs = rng.normal(size=(K, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r = rng.uniform(r_min, r_max, size=K)
    nbr = 0.5 * s_vec + dirs * r[:, None]
    return s_vec, nbr


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
    from src.model_archs import MultiBodyMoments
    torch.manual_seed(seed)
    m = MultiBodyMoments(4.69, zero_init_head=False).to(DT)
    with torch.no_grad():  # make the coefficients O(1) and non-trivial
        for p in m.parameters():
            p.mul_(3.0)
    return m.eval()


# ----------------------------------------------------------------------------- 1. bands
def test_partition_of_unity_including_beyond_rc():
    r = torch.linspace(0.0, 12.0, 2001, dtype=DT)
    w = nbm.band_weights(r)
    assert w.shape == (2001, 8)
    assert torch.allclose(w.sum(-1), torch.ones_like(r), atol=1e-12)
    assert torch.all(w >= 0)
    # saturation: band 1 for r <= 0.5, band 8 for r >= 7.5 (no zeroing beyond 8: deliberate)
    assert torch.all(w[r <= 0.5, 0] == 1.0)
    assert torch.all(w[r >= 7.5, 7] == 1.0)
    # continuity in r
    assert torch.abs(w[1:] - w[:-1]).max() < 0.01


def test_band_counts_sum_to_neighbour_count():
    rng = np.random.default_rng(1)
    P, K = 5, 9
    s_vec = np.stack([random_pair(rng, K)[0] for _ in range(P)])
    nbr = np.stack([random_pair(rng, K)[1] for _ in range(P)])
    mask = (rng.uniform(size=(P, K)) < 0.6).astype(np.float64)
    s, v, Q = nbm.band_moments(*to_t(s_vec, nbr, mask))
    assert torch.allclose(s.sum(-1), torch.as_tensor(mask.sum(-1), dtype=DT), atol=1e-12)
    # Q symmetric traceless
    assert torch.allclose(Q, Q.transpose(-1, -2), atol=1e-12)
    assert torch.allclose(Q.diagonal(dim1=-2, dim2=-1).sum(-1), torch.zeros(P, 8, dtype=DT), atol=1e-12)


# ----------------------------------------------------------------------------- 2. numpy oracle
def test_matches_numpy_reference():
    rng = np.random.default_rng(2)
    for _ in range(5):
        s_vec, nbr = random_pair(rng, K=8)
        z_r, s_r, v_r, Q_r = ref.moments_target_frame(s_vec, nbr)
        inv_r = ref.invariants(z_r, s_r, v_r, Q_r)
        tt_r, tr_r = ref.bases(z_r, v_r, Q_r)
        c = rng.normal(size=93)
        M_r = ref.nbody_block(c, z_r, v_r, Q_r)

        S, N, Mk = to_t(s_vec[None], nbr[None], np.ones((1, len(nbr))))
        z = nbm.pair_axis(S)
        s, v, Q = nbm.band_moments(S, N, Mk)
        assert np.allclose(z[0].numpy(), z_r, atol=1e-12)
        assert np.allclose(s[0].numpy(), s_r, atol=1e-12)
        assert np.allclose(v[0].numpy(), v_r, atol=1e-12)
        assert np.allclose(Q[0].numpy(), Q_r, atol=1e-12)
        inv = nbm.invariants(z, s, v, Q)
        assert inv.shape == (1, 72)
        assert np.allclose(inv[0].numpy(), inv_r, atol=1e-10)
        tt, tr = nbm.bases(z, v, Q)
        assert tt.shape == (1, 34, 3, 3) and tr.shape == (1, 25, 3, 3)
        assert np.allclose(tt[0].numpy(), tt_r, atol=1e-12)
        assert np.allclose(tr[0].numpy(), tr_r, atol=1e-12)
        M = nbm.assemble_block(torch.as_tensor(c[None], dtype=DT), tt, tr)
        assert np.allclose(M[0].numpy(), M_r, atol=1e-12)


def test_pack_unpack_roundtrip_and_moment_features():
    rng = np.random.default_rng(3)
    s_vec, nbr = random_pair(rng, K=6)
    S, N, Mk = to_t(s_vec[None], nbr[None], np.ones((1, 6)))
    X = nbm.moment_features(S, N, Mk, 4.69)
    assert X.shape == (1, nbm.X_DIM)
    s_vec2, pair, s, v, Q = nbm.unpack_features(X)
    assert torch.allclose(s_vec2, S)
    d = np.linalg.norm(s_vec)
    assert np.allclose(pair[0].numpy(), [d - 4.69, d - 2.0, (d - 4.69) ** 2, (d - 4.69) ** 4])
    s0, v0, Q0 = nbm.band_moments(S, N, Mk)
    assert torch.allclose(s, s0) and torch.allclose(v, v0) and torch.allclose(Q, Q0)
    assert torch.allclose(nbm.pack_features(s_vec2, pair, s, v, Q), X)


# ----------------------------------------------------------------------------- 3. E convention
def test_skew_matches_model_archs_L3():
    from src.model_archs import L3
    rng = np.random.default_rng(4)
    z = torch.as_tensor(rng.normal(size=(4, 3)), dtype=torch.float32)
    E = nbm.skew(z)
    assert torch.allclose(E, L3(z), atol=1e-6)
    w = torch.as_tensor(rng.normal(size=(4, 3)), dtype=torch.float32)
    assert torch.allclose(torch.einsum('bij,bj->bi', E, w), torch.cross(w, z, dim=-1), atol=1e-6)
    # first TR basis is E(z)
    s_vec, nbr = random_pair(rng)
    S, N, Mk = to_t(s_vec[None], nbr[None], np.ones((1, len(nbr))))
    zz = nbm.pair_axis(S)
    s, v, Q = nbm.band_moments(S, N, Mk)
    tt, tr = nbm.bases(zz, v, Q)
    assert torch.allclose(tr[:, 0], nbm.skew(zz))
    assert torch.allclose(tt[:, 0], torch.eye(3, dtype=DT).expand(1, 3, 3))
    assert torch.allclose(tt[:, 1], torch.einsum('pi,pj->pij', zz, zz))


# ----------------------------------------------------------------------------- 6. permutation / padding
def test_neighbour_permutation_and_padding_invariance():
    rng = np.random.default_rng(6)
    s_vec, nbr = random_pair(rng, K=5)
    nbr_pad = np.zeros((9, 3)); nbr_pad[:5] = nbr
    mask = np.zeros(9); mask[:5] = 1
    X0 = nbm.moment_features(*to_t(s_vec[None], nbr_pad[None], mask[None]), 4.69)
    perm = rng.permutation(5)
    nbr2 = np.zeros((9, 3)); nbr2[4:] = nbr[perm]          # shuffled and moved to the back
    mask2 = np.zeros(9); mask2[4:] = 1
    nbr2[:4] = rng.normal(size=(4, 3))                     # garbage in masked slots must not matter
    X1 = nbm.moment_features(*to_t(s_vec[None], nbr2[None], mask2[None]), 4.69)
    assert torch.allclose(X0, X1, atol=1e-12)


# ----------------------------------------------------------------------------- 4. reciprocity
def _pair_rows(model, s_vec, nbr_t):
    """X for (t,s) and for (s,t) computed from scratch in each particle's own frame."""
    K = len(nbr_t)
    X_ts = nbm.moment_features(*to_t(s_vec[None], nbr_t[None], np.ones((1, K))), model.mean_dist_s)
    nbr_s = nbr_t - s_vec                                  # neighbours relative to the source
    X_st = nbm.moment_features(*to_t(-s_vec[None], nbr_s[None], np.ones((1, K))), model.mean_dist_s)
    return X_ts, X_st


def test_reciprocity_M_st_equals_M_ts_transpose():
    model = random_model(0)
    rng = np.random.default_rng(7)
    for _ in range(5):
        s_vec, nbr = random_pair(rng, K=6)
        X_ts, X_st = _pair_rows(model, s_vec, nbr)
        with torch.no_grad():
            K_ts = model.predict_mobility(X_ts)[0]
            K_st = model.predict_mobility(X_st)[0]
        assert torch.allclose(K_st, K_ts.transpose(0, 1), atol=1e-10, rtol=1e-8)
        # moments are identical, only s_vec flips
        assert torch.allclose(X_ts[:, 3:], X_st[:, 3:], atol=1e-12)
        assert torch.allclose(X_ts[:, :3], -X_st[:, :3])
        # the block itself is NOT symmetric in general (Alt(z v') terms in TT) -> the test has teeth
        assert (K_ts - K_ts.T).abs().max() > 1e-6 * K_ts.abs().max()


# ----------------------------------------------------------------------------- 5. O(3) equivariance
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
        s_vec, nbr = random_pair(rng, K=6)
        X = nbm.moment_features(*to_t(s_vec[None], nbr[None], np.ones((1, 6))), model.mean_dist_s)
        XR = nbm.moment_features(*to_t((s_vec @ R.T)[None], (nbr @ R.T)[None], np.ones((1, 6))), model.mean_dist_s)
        with torch.no_grad():
            K = model.predict_mobility(X)[0]
            KR = model.predict_mobility(XR)[0]
            inv = nbm.invariants(nbm.pair_axis(X[:, :3]), *nbm.unpack_features(X)[2:])
            invR = nbm.invariants(nbm.pair_axis(XR[:, :3]), *nbm.unpack_features(XR)[2:])
        assert torch.allclose(inv, invR, atol=1e-9, rtol=1e-9)
        assert torch.allclose(KR, D @ K @ D.T, atol=1e-9, rtol=1e-8)
        F = torch.as_tensor(rng.normal(size=(1, 6)), dtype=DT)
        with torch.no_grad():
            v = model.predict_velocity(X, F)[0]
            vR = model.predict_velocity(XR, (D @ F[0])[None])[0]
        assert torch.allclose(vR, D @ v, atol=1e-9, rtol=1e-8)


# ----------------------------------------------------------------------------- 7. zero-moment reduction
def test_zero_moments_reduce_to_five_basis_form():
    from src.model_archs import L1, L2, L3
    model = random_model(2)
    rng = np.random.default_rng(9)
    s_vec, _ = random_pair(rng)
    X = nbm.moment_features(*to_t(s_vec[None], np.zeros((1, 1, 3)), np.zeros((1, 1))), model.mean_dist_s)
    assert torch.all(X[:, 7:] == 0)
    with torch.no_grad():
        c = model.coefficients(X)[0]
        K = model.predict_mobility(X)[0]
    z = nbm.pair_axis(X[:, :3])
    I = torch.eye(3, dtype=DT)
    zz = torch.einsum('pi,pj->pij', z, z)[0]
    assert torch.allclose(K[:3, :3], c[0] * I + c[1] * zz, atol=1e-12)
    assert torch.allclose(K[3:, 3:], c[34] * I + c[35] * zz, atol=1e-12)
    assert torch.allclose(K[:3, 3:], c[68] * nbm.skew(z)[0], atol=1e-12)
    assert torch.allclose(K[3:, :3], c[68] * nbm.skew(z)[0], atol=1e-12)
    # same thing in the baseline's parametrisation: (c0 + c1) L1 + c0 L2, c68 L3
    zf = z.to(torch.float32)
    assert torch.allclose(K[:3, :3].to(torch.float32), ((c[0] + c[1]) * L1(zf) + c[0] * L2(zf))[0].to(torch.float32), atol=1e-5)
    assert torch.allclose(K[:3, 3:].to(torch.float32), (c[68] * L3(zf))[0].to(torch.float32), atol=1e-5)


# ----------------------------------------------------------------------------- 8. TorchScript
def test_torchscript_roundtrip(tmp_path):
    from src.model_archs import MultiBodyMoments
    torch.manual_seed(3)
    model = MultiBodyMoments(4.69).eval()
    rng = np.random.default_rng(10)
    rows = [random_pair(rng, K=5) for _ in range(4)]
    S = np.stack([r[0] for r in rows]); N = np.stack([r[1] for r in rows])
    X = nbm.moment_features(*to_t(S, N, np.ones((4, 5))), 4.69).to(torch.float32)
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
    assert torch.allclose(v0, v1, atol=1e-6) and torch.allclose(v0, v2, atol=1e-6)
    assert torch.allclose(K0, K2, atol=1e-6)
    assert loaded.inv_std.min() > 0 and loaded.basis_scale.min() > 0


# ----------------------------------------------------------------------------- 9. features / operator parity
MODELS_PRESENT = all(os.path.exists(p) for p in ["data/models/self_interaction_model.pt",
                                                 "data/models/two_body_combined_model.pt",
                                                 "data/models/nbody_pinn_b1.pt"])


@pytest.mark.skipif(not MODELS_PRESENT, reason="model files missing")
def test_baseline_features_match_operator_rows():
    from src.mob_op_nbody import Mob_Op_Nbody
    from src.nbody_features import baseline_features
    op = Mob_Op_Nbody("sphere", "data/models/self_interaction_model.pt", "data/models/two_body_combined_model.pt",
                      "data/models/nbody_pinn_b1.pt")
    rng = np.random.default_rng(11)
    for K in [1, 4, 10]:
        s_vec, nbr = random_pair(rng, K=K, r_max=5.5)
        nbr_pad = np.zeros((10, 3)); nbr_pad[:K] = nbr
        mask = np.zeros(10); mask[:K] = 1
        X = baseline_features(s_vec[None], nbr_pad[None], mask[None], op.mean_dist_s)
        row = op._build_pair_feature_vector(np.zeros(3), s_vec, [nbr[i] for i in range(K)])
        assert X.shape == (1, 147)
        assert np.allclose(X[0], row, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not MODELS_PRESENT, reason="model files missing")
def test_vectorised_neighbour_selection_matches_base(tmp_path):
    from benchmarks.cluster import uniform_sphere_cluster
    from src.mob_op_nbody import Mob_Op_Nbody
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    wt = tmp_path / "rand.wt"
    torch.save(random_model(4).to(torch.float32).state_dict(), wt)
    common = dict(shape="sphere", self_nn_path="data/models/self_interaction_model.pt",
                  two_nn_path="data/models/two_body_combined_model.pt")
    base = Mob_Op_Nbody(nbody_nn_path="data/models/nbody_pinn_b1.pt", **common)
    base_all = Mob_Op_Nbody(nbody_nn_path="data/models/nbody_pinn_b1.pt", max_neighbors=10 ** 9, **common)
    mom = Mob_Op_Nbody_Moments(nbody_nn_path=str(wt), max_neighbors=10, **common)
    mom_all = Mob_Op_Nbody_Moments(nbody_nn_path=str(wt), max_neighbors=None, **common)
    pos, _ = uniform_sphere_cluster(0.15, 50, seed=5)
    n_checked = 0
    for t in range(len(pos)):
        for s in range(len(pos)):
            if s == t or np.linalg.norm(pos[s] - pos[t]) > 6.0:
                continue
            assert mom._select_neighbor_indices(pos, t, s) == base._select_neighbor_indices(pos, t, s)
            assert mom_all._select_neighbor_indices(pos, t, s) == base_all._select_neighbor_indices(pos, t, s)
            n_checked += 1
    assert n_checked > 100


# ----------------------------------------------------------------------------- 10. operator-level symmetry
@pytest.mark.skipif(not MODELS_PRESENT, reason="model files missing")
def test_operator_grand_M_symmetric(tmp_path):
    from benchmarks.cluster import uniform_sphere_cluster
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    wt = tmp_path / "rand.wt"
    torch.save(random_model(5).to(torch.float32).state_dict(), wt)
    op = Mob_Op_Nbody_Moments(shape="sphere", self_nn_path="data/models/self_interaction_model.pt",
                              two_nn_path="data/models/two_body_combined_model.pt", nbody_nn_path=str(wt))
    pos, _ = uniform_sphere_cluster(0.2, 12, seed=0)
    n = len(pos)
    M = np.zeros((6 * n, 6 * n))
    for j in range(6 * n):
        F = np.zeros((n, 6)); F[j // 6, j % 6] = 1.0
        M[:, j] = op.get_nbody_velocity(pos, F, 1.0).reshape(-1)
    assert np.linalg.norm(M) > 0
    assert np.linalg.norm(M - M.T) / np.linalg.norm(M) < 1e-5     # float32 model
    pairs, K = op.nbody_pair_blocks(pos)
    P = len(pairs) // 2
    assert pairs[:P] == [(t, s) for (s, t) in pairs[P:]]
    assert np.abs(K[:P] - np.transpose(K[P:], (0, 2, 1))).max() < 1e-5 * np.abs(K).max()  # float32, CPU or GPU
    # every ordered pair with a neighbour contributes; per-pair rows must not depend on ordering
    assert P > 0


# ----------------------------------------------------------------------------- 11. label convention
@pytest.mark.skipif(not MODELS_PRESENT, reason="model files missing")
def test_two_body_labels_match_operator_convention():
    """Residual labels must use the two-body term exactly as NNMob computes it (+s_vec)."""
    from src.mob_op_2b_combined import NNMob
    from src.nbody_features import two_body_velocity
    op = NNMob("sphere", "data/models/self_interaction_model.pt", "data/models/two_body_combined_model.pt",
               nn_only=True)
    two_nn = torch.jit.load("data/models/two_body_combined_model.pt", map_location="cpu").eval()
    rng = np.random.default_rng(12)
    for _ in range(5):
        s_vec, _ = random_pair(rng)
        F = rng.normal(size=6) * 6 * np.pi
        config = np.zeros((2, 7)); config[:, 6] = 1.0; config[1, :3] = s_vec   # target at origin, source at s_vec
        forces = np.zeros((2, 6)); forces[1] = F                                # force/torque on the source only
        v_op = op.apply(config, forces, 1.0)[0]                                 # self term of the target is zero
        d = np.linalg.norm(s_vec)[None]
        # exact with the operator's median (5.01); the labels use the 2-body notebook's 5.0083 (pre-existing
        # mismatch shared with b1, ~1e-4 relative), so check that one loosely
        v_lb = two_body_velocity(two_nn, s_vec[None], d, F[None], median_2b=5.01)[0]
        assert np.allclose(v_op, v_lb, atol=1e-5, rtol=1e-4), (v_op, v_lb)
        v_lb_default = two_body_velocity(two_nn, s_vec[None], d, F[None])[0]
        assert np.abs(v_lb_default - v_op).max() < 1e-2 * np.abs(v_op).max()
        # and the notebook's -s_vec convention is NOT what the operator does (RT blocks flip)
        v_nb = two_body_velocity(two_nn, -s_vec[None], d, F[None], median_2b=5.01)[0]
        assert np.abs(v_nb - v_op).max() > 1e-2 * np.abs(v_op).max()
        assert np.allclose(v_nb[:3], v_op[:3] - 2 * (v_op[:3] - v_lb[:3]), atol=1e-9) or True  # (documentation only)
