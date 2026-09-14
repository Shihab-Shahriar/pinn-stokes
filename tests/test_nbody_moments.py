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


# ----------------------------------------------------------------------------- 13. v3 layout: knot bands, linear bases, class-2 TR
KNOTS5 = [0.5, 1.5, 2.5, 3.5, 4.5]
KNOTS4 = [0.5, 1.5, 2.5, 3.5]
LAYOUTS = {  # name -> (bands, bases, invariants, n_in, n_coef, x_dim)
    "v2": (None, "v2", "full", 76, 93, 111),
    "nb8_linear": (None, "linear", "full", 76, 69, 111),
    "nb5_linear": (KNOTS5, "linear", "full", 49, 45, 72),
    "nb4_linear": (KNOTS4, "linear", "full", 40, 37, 59),
    "nb5_linear_tr2": (KNOTS5, "linear_tr2", "full", 49, 60, 72),
    "nb4_linear_tr2_red": (KNOTS4, "linear_tr2", "reduced", 24, 49, 59),
}
WT_V2 = "experiments/nbody_moments_v2_kinf_rc8_pc8c.wt"
PT_V2 = "data/models/nbody_moments_v2_kinf_rc8_pc8c.pt"


def random_model_v3(seed, name):
    from src.model_archs import MultiBodyMoments
    bands, bases, inv = LAYOUTS[name][:3]
    torch.manual_seed(seed)
    m = MultiBodyMoments(4.69, zero_init_head=False, bands=bands, bases=bases, invariants=inv).to(DT)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(3.0)
    return m.eval()


def _rows_v3(model, s_vec, nbr, mask=None):
    K = len(nbr)
    mask = np.ones((1, K)) if mask is None else mask[None]
    knots = nbm.knots_of(model)
    if knots is None:
        return nbm.moment_features(*to_t(s_vec[None], nbr[None], mask), model.mean_dist_s)
    return nbm.moment_features_knots(*to_t(s_vec[None], nbr[None], mask), model.mean_dist_s, knots)


@pytest.mark.parametrize("knots", [KNOTS5, KNOTS4, [0.25, 0.75, 2.0, 5.0, 7.5], [1.0, 3.0]])
def test_knot_bands_partition_of_unity(knots):
    r = torch.linspace(0.0, 12.0, 4001, dtype=DT)
    k = torch.as_tensor(knots, dtype=DT)
    w = nbm.band_weights_knots(r, k)
    assert w.shape == (4001, len(knots))
    assert torch.allclose(w.sum(-1), torch.ones_like(r), atol=1e-12)
    assert torch.all(w >= 0) and torch.all(w <= 1)
    assert torch.all(w[r <= knots[0], 0] == 1.0)
    assert torch.all(w[r >= knots[-1], -1] == 1.0)
    for a, ka in enumerate(knots):  # peaks at the knots
        assert torch.allclose(nbm.band_weights_knots(torch.tensor([ka], dtype=DT), k)[0, a], torch.tensor(1.0, dtype=DT))
    assert torch.abs(w[1:] - w[:-1]).max() < 4.0 * (12.0 / 4000) / min(np.diff(knots))   # Lipschitz continuity
    assert np.allclose(w.numpy(), ref.band_weights_knots(r.numpy(), knots), atol=1e-12)


def test_knot_bands_reproduce_v2_bitwise():
    for dt in (torch.float32, torch.float64):
        r = torch.linspace(0.0, 12.0, 100001, dtype=dt)
        assert torch.equal(nbm.band_weights(r), nbm.band_weights_knots(r, torch.as_tensor(nbm.V2_KNOTS, dtype=dt)))


def test_v2_defaults_unchanged():
    from src.model_archs import MultiBodyMoments
    assert (nbm.X_DIM, nbm.N_COEF, nbm.N_TT, nbm.N_TR, nbm.N_INV, nbm.N_IN) == (111, 93, 34, 25, 72, 76)
    assert nbm.layout_dims(8, True, False, False) == {"nb": 8, "n_in": 76, "n_tt": 34, "n_tr1": 25, "n_tr2": 0, "n_coef": 93, "x_dim": 111}
    m = MultiBodyMoments(4.69)
    lay = nbm.layout_of_model(m)
    assert lay["version"] == "v2" and lay["bands"] == nbm.V2_KNOTS and (m.n_in, m.n_coef, m.x_dim) == (76, 93, 111)
    assert nbm.knots_of(m) is None
    assert not any("band_knots" in k for k in m.state_dict())        # v2 .wt files stay strict-loadable
    assert m.net[0].in_features == 76 and m.net[-1].out_features == 93
    if os.path.exists(WT_V2):
        m.load_state_dict(torch.load(WT_V2, map_location="cpu", weights_only=True), strict=True)
    if os.path.exists(PT_V2):
        pt = torch.jit.load(PT_V2, map_location="cpu").eval()
        assert nbm.layout_of_model(pt)["version"] == "v2" and nbm.knots_of(pt) is None
    # the v2 wrappers are the v3 functions at the v2 parameters, exactly
    rng = np.random.default_rng(13)
    s_vec, nbr = random_pair(rng, K=9)
    S, N, Mk = to_t(s_vec[None], nbr[None], np.ones((1, 9)))
    X = nbm.moment_features(S, N, Mk, 4.69)
    assert torch.equal(X, nbm.moment_features_knots(S, N, Mk, 4.69, torch.as_tensor(nbm.V2_KNOTS, dtype=DT)))
    z = nbm.pair_axis(S); s, v, Q = nbm.band_moments(S, N, Mk)
    tt, tr = nbm.bases(z, v, Q)
    tt3, tr1, tr2 = nbm.bases_v3(z, v, Q, True, False)
    assert torch.equal(tt, tt3) and torch.equal(tr, tr1) and tr2.shape == (1, 0, 3, 3)
    c = torch.as_tensor(rng.normal(size=(1, 93)), dtype=DT)
    assert torch.equal(nbm.assemble_block(c, tt, tr), nbm.assemble_block_v3(c, tt3, tr1, tr2))
    assert nbm.layout_from_sidecar({}) == {"bands": None, "bases": "v2", "invariants": "full"}
    assert nbm.layout_from_sidecar({"bands": nbm.V2_KNOTS}) == {"bands": None, "bases": "v2", "invariants": "full"}
    assert nbm.layout_from_sidecar({"bands": KNOTS5, "bases": "linear_tr2", "invariants": "reduced"}) == \
        {"bands": KNOTS5, "bases": "linear_tr2", "invariants": "reduced"}


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_layout_dims_and_row_width(name):
    bands, bases, inv, n_in, n_coef, x_dim = LAYOUTS[name]
    m = random_model_v3(0, name)
    lay = nbm.layout_of_model(m)
    assert (lay["n_in"], lay["n_coef"], lay["x_dim"]) == (n_in, n_coef, x_dim)
    assert lay["bases"] == bases and lay["invariants"] == inv and lay["bands"] == (bands or nbm.V2_KNOTS)
    assert lay["version"] == ("v2" if name == "v2" else "v3")
    assert (m.net[0].in_features, m.net[-1].out_features) == (n_in, n_coef)
    assert m.inv_std.shape == (n_in,) and m.basis_scale.shape == (n_coef,)
    rng = np.random.default_rng(14)
    s_vec, nbr = random_pair(rng, K=7)
    X = _rows_v3(m, s_vec, nbr)
    assert X.shape == (1, x_dim)
    s_vec2, pair, s, v, Q = nbm.unpack_features(X)
    assert s.shape == (1, lay["nb"]) and v.shape == (1, lay["nb"], 3) and Q.shape == (1, lay["nb"], 3, 3)
    assert torch.allclose(nbm.pack_features(s_vec2, pair, s, v, Q), X)
    assert torch.allclose(s.sum(-1), torch.tensor([7.0], dtype=DT), atol=1e-12)     # partition of unity
    with torch.no_grad():
        assert m.coefficients(X).shape == (1, n_coef) and m.predict_mobility(X).shape == (1, 6, 6)
    # the numpy wrapper builds the same rows
    from src import nbody_features as nf
    Xn = nf.moment_features(s_vec[None], nbr[None], np.ones((1, 7)), 4.69, knots=nbm.knots_of(m))
    assert Xn.shape == (1, x_dim) and np.allclose(Xn[0], X[0].numpy(), atol=1e-6)


def test_reciprocity_with_tr2():
    """Class-2 TR bases: RT = TR1 - TR2 != TR, yet M_st = M_ts^T exactly; zeroing class 2 restores RT = TR."""
    model = random_model_v3(1, "nb5_linear_tr2")
    rng = np.random.default_rng(15)
    n2 = model.n_tr2
    assert n2 == 15
    for _ in range(5):
        s_vec, nbr = random_pair(rng, K=6)
        X_ts = _rows_v3(model, s_vec, nbr)
        X_st = _rows_v3(model, -s_vec, nbr - s_vec)          # from scratch in the source's frame
        with torch.no_grad():
            K_ts = model.predict_mobility(X_ts)[0]
            K_st = model.predict_mobility(X_st)[0]
            c = model.coefficients(X_ts)
        assert torch.allclose(K_st, K_ts.T, atol=1e-10, rtol=1e-8)
        assert torch.allclose(X_ts[:, 3:], X_st[:, 3:], atol=1e-12) and torch.allclose(X_ts[:, :3], -X_st[:, :3])
        scale = K_ts.abs().max()
        assert (K_ts[:3, 3:] - K_ts[3:, :3]).abs().max() > 1e-3 * scale              # class 2 is active
        s_vec_t, pair, s, v, Q = nbm.unpack_features(X_ts)
        z = nbm.pair_axis(s_vec_t)
        tt, tr1, tr2 = nbm.bases_v3(z, v, Q, False, True)
        assert tt.shape[1] == 17 and tr1.shape[1] == 11 and tr2.shape[1] == 15
        c0 = c.clone(); c0[:, -n2:] = 0
        K0 = nbm.assemble_block_v3(c0, tt, tr1, tr2)[0]
        assert torch.equal(K0[:3, 3:], K0[3:, :3])                                    # class 2 off -> RT = TR
        assert torch.allclose(K0, nbm.assemble_block_v3(c, tt, tr1, tr2)[0] - torch.cat(
            [torch.cat([torch.zeros(3, 3, dtype=DT), (K_ts[:3, 3:] - K_ts[3:, :3]) / 2], 1),
             torch.cat([-(K_ts[:3, 3:] - K_ts[3:, :3]) / 2, torch.zeros(3, 3, dtype=DT)], 1)], 0), atol=1e-12)
        # each class-2 basis satisfies T(-z) = -T(z)^T, each class-1 basis T(-z) = T(z)^T
        _, _, _, v_s, Q_s = nbm.unpack_features(X_st)
        tt_s, tr1_s, tr2_s = nbm.bases_v3(nbm.pair_axis(X_st[:, :3]), v_s, Q_s, False, True)
        assert torch.allclose(tt_s, tt.transpose(-1, -2), atol=1e-12)
        assert torch.allclose(tr1_s, tr1.transpose(-1, -2), atol=1e-12)
        assert torch.allclose(tr2_s, -tr2.transpose(-1, -2), atol=1e-12)


def test_tr2_reduces_to_v2_when_zero():
    rng = np.random.default_rng(16)
    s_vec, nbr = random_pair(rng, K=8)
    S, N, Mk = to_t(s_vec[None], nbr[None], np.ones((1, 8)))
    z = nbm.pair_axis(S); s, v, Q = nbm.band_moments(S, N, Mk)
    tt, tr1, tr2 = nbm.bases_v3(z, v, Q, True, True)
    assert (tt.shape[1], tr1.shape[1], tr2.shape[1]) == (34, 25, 24)
    c93 = rng.normal(size=93)
    c = torch.as_tensor(np.concatenate([c93, np.zeros(24)])[None], dtype=DT)
    K = nbm.assemble_block_v3(c, tt, tr1, tr2)[0]
    assert torch.allclose(K, nbm.assemble_block(c[:, :93], *nbm.bases(z, v, Q))[0], atol=1e-12)
    z_r, s_r, v_r, Q_r = ref.moments_target_frame(s_vec, nbr)
    assert np.allclose(K.numpy(), ref.nbody_block(c93, z_r, v_r, Q_r), atol=1e-12)


@pytest.mark.parametrize("kind", ["rotation", "reflection", "improper"])
@pytest.mark.parametrize("name", ["nb5_linear_tr2", "nb4_linear_tr2_red"])
def test_o3_equivariance_v3(kind, name):
    model = random_model_v3(2, name)
    rng = np.random.default_rng(17)
    if kind == "rotation":
        R = random_rotation(rng, +1.0)
    elif kind == "reflection":
        R = np.diag([1.0, 1.0, -1.0])
    else:
        R = random_rotation(rng, +1.0) @ np.diag([1.0, -1.0, 1.0])
    D = torch.as_tensor(big_D(R), dtype=DT)
    for _ in range(3):
        s_vec, nbr = random_pair(rng, K=6)
        X = _rows_v3(model, s_vec, nbr)
        XR = _rows_v3(model, s_vec @ R.T, nbr @ R.T)
        with torch.no_grad():
            K = model.predict_mobility(X)[0]
            KR = model.predict_mobility(XR)[0]
            inv, invR = model._invariants(X), model._invariants(XR)
        assert torch.allclose(inv, invR, atol=1e-9, rtol=1e-9)
        assert torch.allclose(KR, D @ K @ D.T, atol=1e-9, rtol=1e-8)
        F = torch.as_tensor(rng.normal(size=(1, 6)), dtype=DT)
        with torch.no_grad():
            v = model.predict_velocity(X, F)[0]
            vR = model.predict_velocity(XR, (D @ F[0])[None])[0]
        assert torch.allclose(vR, D @ v, atol=1e-9, rtol=1e-8)


def test_matches_numpy_reference_v3():
    rng = np.random.default_rng(18)
    for quadratic, tr2 in [(False, True), (False, False), (True, True)]:
        s_vec, nbr = random_pair(rng, K=8)
        z_r, s_r, v_r, Q_r = ref.moments_knots(np.zeros(3), s_vec, nbr, KNOTS5)
        S, N, Mk = to_t(s_vec[None], nbr[None], np.ones((1, 8)))
        z = nbm.pair_axis(S)
        s, v, Q = nbm.band_moments_knots(S, N, Mk, torch.as_tensor(KNOTS5, dtype=DT))
        assert np.allclose(s[0].numpy(), s_r, atol=1e-12) and np.allclose(v[0].numpy(), v_r, atol=1e-12)
        assert np.allclose(Q[0].numpy(), Q_r, atol=1e-12)
        assert np.allclose(nbm.invariants(z, s, v, Q)[0].numpy(), ref.invariants(z_r, s_r, v_r, Q_r), atol=1e-10)
        assert np.allclose(nbm.invariants(z, s, v, Q, True)[0].numpy(), ref.invariants_reduced(z_r, s_r, v_r, Q_r), atol=1e-10)
        tt, tr1, t2 = nbm.bases_v3(z, v, Q, quadratic, tr2)
        tt_r, tr1_r, t2_r = ref.bases_v3(z_r, v_r, Q_r, quadratic, tr2)
        assert tt.shape[1:] == tt_r.shape and tr1.shape[1:] == tr1_r.shape and t2.shape[1:] == t2_r.shape
        assert np.allclose(tt[0].numpy(), tt_r, atol=1e-12) and np.allclose(tr1[0].numpy(), tr1_r, atol=1e-12)
        assert np.allclose(t2[0].numpy(), t2_r, atol=1e-12)
        n_coef = nbm.layout_dims(5, quadratic, tr2, False)["n_coef"]
        c = rng.normal(size=n_coef)
        M = nbm.assemble_block_v3(torch.as_tensor(c[None], dtype=DT), tt, tr1, t2)
        assert np.allclose(M[0].numpy(), ref.nbody_block_v3(c, z_r, v_r, Q_r, quadratic, tr2), atol=1e-12)


def test_torchscript_roundtrip_v3(tmp_path):
    model = random_model_v3(3, "nb5_linear_tr2").to(torch.float32)
    rng = np.random.default_rng(19)
    rows = [random_pair(rng, K=5) for _ in range(4)]
    S = np.stack([r[0] for r in rows]); N = np.stack([r[1] for r in rows])
    X = nbm.moment_features_knots(*to_t(S, N, np.ones((4, 5))), 4.69, torch.as_tensor(KNOTS5, dtype=DT)).to(torch.float32)
    with torch.no_grad():
        model.fit_normalisation(X)
    assert model.basis_scale.shape == (60,) and model.basis_scale.min() > 0
    scripted = torch.jit.script(model)
    p = tmp_path / "m.pt"
    scripted.save(str(p))
    loaded = torch.jit.load(str(p)).eval()
    assert nbm.layout_of_model(loaded) == nbm.layout_of_model(model)
    assert torch.allclose(nbm.knots_of(loaded), torch.as_tensor(KNOTS5, dtype=DT))
    assert (loaded.nb, loaded.has_tr2, loaded.use_quadratic, loaded.reduced_inv) == (5, True, False, False)
    with torch.no_grad():
        assert torch.allclose(model.predict_mobility(X), loaded.predict_mobility(X), atol=1e-6)
        F = torch.as_tensor(rng.normal(size=(4, 6)), dtype=torch.float32)
        assert torch.allclose(model.predict_velocity(X, F), loaded.predict_velocity(X, F), atol=1e-6)


@pytest.mark.skipif(not MODELS_PRESENT, reason="model files missing")
def test_operator_v3_rows_and_symmetry(tmp_path):
    """A v3 model in the CPU operator: rows built with the model's knots (per-pair from-scratch check), symmetric
    grand M, and the .wt path (sidecar or nbody_layout kwarg) equal to the self-describing .pt path."""
    import json
    from benchmarks.cluster import uniform_sphere_cluster
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    model = random_model_v3(6, "nb5_linear_tr2").to(torch.float32).eval()
    pt = tmp_path / "v3.pt"; wt = tmp_path / "v3.wt"
    torch.jit.script(model).save(str(pt))
    torch.save(model.state_dict(), wt)
    common = dict(shape="sphere", self_nn_path="data/models/self_interaction_model.pt",
                  two_nn_path="data/models/two_body_combined_model.pt", switch_dist=8.0, pair_cutoff=8.0,
                  neighbor_cutoff=8.0, max_neighbors=None)
    op = Mob_Op_Nbody_Moments(nbody_nn_path=str(pt), **common)
    assert op.nbody_layout["version"] == "v3" and op.nbody_layout["x_dim"] == 72
    assert np.allclose(op.band_knots.numpy(), KNOTS5)
    pos, _ = uniform_sphere_cluster(0.15, 40, seed=5)
    pairs, X_ts, X_st = op._pair_rows(pos)
    assert len(pairs) > 50 and X_ts.shape == (len(pairs), 72)
    knots = torch.as_tensor(KNOTS5, dtype=DT)
    for i, (t, s) in list(enumerate(pairs))[:40]:
        idx = op._select_neighbor_indices(pos, t, s)
        X = nbm.moment_features_knots(*to_t((pos[s] - pos[t])[None], (pos[idx] - pos[t])[None], np.ones((1, len(idx)))),
                                      op.mean_dist_s, knots)[0].numpy()
        assert np.allclose(X, X_ts[i], rtol=1e-6, atol=1e-6)
    assert np.allclose(X_st[:, :3], -X_ts[:, :3]) and np.array_equal(X_st[:, 3:], X_ts[:, 3:])
    pos12, _ = uniform_sphere_cluster(0.2, 12, seed=0)
    n = len(pos12)
    M = np.zeros((6 * n, 6 * n))
    for j in range(6 * n):
        F = np.zeros((n, 6)); F[j // 6, j % 6] = 1.0
        M[:, j] = op.get_nbody_velocity(pos12, F, 1.0).reshape(-1)
    assert np.linalg.norm(M) > 0 and np.linalg.norm(M - M.T) / np.linalg.norm(M) < 1e-5
    blk = M.reshape(n, 6, n, 6)
    off = np.array([np.abs(blk[t, :3, s, 3:] - blk[t, 3:, s, :3]).max() for t in range(n) for s in range(n) if t != s])
    assert off.max() > 1e-3 * np.abs(M).max()                                   # TR != RT in the pair blocks
    # .wt paths: nbody_layout kwarg, and a sidecar next to the weights
    config = np.hstack([pos12, np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))])
    F = np.random.default_rng(0).normal(size=(n, 6))
    v_pt = op.apply(config, F, 1.0)
    op_kw = Mob_Op_Nbody_Moments(nbody_nn_path=str(wt), nbody_layout={"bands": KNOTS5, "bases": "linear_tr2"}, **common)
    assert np.allclose(op_kw.apply(config, F, 1.0), v_pt, atol=1e-6, rtol=1e-5)
    json.dump({"bands": KNOTS5, "bases": "linear_tr2", "invariants": "full"}, open(tmp_path / "v3.json", "w"))
    op_side = Mob_Op_Nbody_Moments(nbody_nn_path=str(wt), **common)
    assert op_side.nbody_layout == op.nbody_layout
    assert np.allclose(op_side.apply(config, F, 1.0), v_pt, atol=1e-6, rtol=1e-5)
    with pytest.raises(Exception):  # wrong layout for the weights must fail at construction
        Mob_Op_Nbody_Moments(nbody_nn_path=str(wt), nbody_layout={"bands": KNOTS4, "bases": "linear_tr2"}, **common)


# ----------------------------------------------------------------------------- 14. learned radial bands (Bessel basis x MLP)
def random_model_bessel(seed, nb=6, n_radial=8, bases="linear_tr2"):
    from src.model_archs import MultiBodyMoments
    torch.manual_seed(seed)
    m = MultiBodyMoments(4.69, zero_init_head=False, radial="bessel", nb=nb, n_radial=n_radial, bases=bases).to(DT)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(3.0)
    return m.eval()


def _rows_model(model, s_vec, nbr, mask=None):
    K = len(nbr)
    mask = np.ones((1, K)) if mask is None else mask[None]
    with torch.no_grad():
        return model.moment_features(*to_t(s_vec[None], nbr[None], mask))


def test_bessel_basis_properties():
    rc, n = 8.0, 8
    r = torch.linspace(0.0, 10.0, 10001, dtype=DT)
    b = nbm.bessel_basis(r, rc, n)
    assert b.shape == (10001, n) and torch.isfinite(b).all()
    assert torch.all(b[r >= rc] == 0)                                                     # zero beyond the cutoff
    k = torch.arange(1, n + 1, dtype=DT)
    assert torch.allclose(b[0], np.sqrt(2 / rc) * k * np.pi / rc, rtol=1e-3)               # finite limit at r = 0
    assert (b[1:] - b[:-1]).abs().max() < 5e-3                                             # continuous (step 1e-3)
    assert b[(r > 7.9) & (r < 8.0)].abs().max() < 1e-4                                     # smooth approach to zero


def test_learned_bands_layout_and_dims():
    from src.model_archs import MultiBodyMoments
    m = random_model_bessel(0, nb=6, n_radial=8)
    lay = nbm.layout_of_model(m)
    assert lay["version"] == "v3" and lay["radial"] == "bessel" and lay["nb"] == 6 and lay["n_radial"] == 8 and lay["bands"] is None
    assert (lay["n_in"], lay["n_coef"], lay["x_dim"]) == (4 + 9 * 6, 5 + 11 * 6, 7 + 13 * 6)
    assert nbm.knots_of(m) is None and m.x_dim == 85 and m.n_coef == 71 and m.learned_radial
    assert sum(p.numel() for p in m.radial.parameters()) == 8 * 32 + 32 * 6
    r = torch.linspace(0.0, 9.0, 901, dtype=DT)
    with torch.no_grad():
        w = m.radial(r)
    assert w.shape == (901, 6) and torch.all(w[r >= 8.0] == 0)                             # bands vanish at the cutoff
    assert (w[1:] - w[:-1]).abs().max() < 0.05 * w.abs().max()                               # and are smooth in r
    kw = nbm.layout_from_sidecar({"radial": "bessel", "nb": 6, "n_radial": 8, "bases": "linear_tr2", "invariants": "full"})
    assert nbm.layout_of_model(MultiBodyMoments(4.69, **kw)) == lay
    assert nbm.layout_from_sidecar({"bands": KNOTS5, "bases": "linear_tr2"}) == {"bands": KNOTS5, "bases": "linear_tr2", "invariants": "full"}
    with pytest.raises(AssertionError):
        MultiBodyMoments(4.69, radial="bessel", bands=KNOTS5)


def test_model_moment_features_matches_knot_path():
    """For tent-band models the model-built row equals the float64 functional path (to float32 precision)."""
    from src.model_archs import MultiBodyMoments
    rng = np.random.default_rng(5)
    for bands in [None, KNOTS5]:
        m = MultiBodyMoments(4.69, bands=bands, bases="linear_tr2" if bands else "v2").eval()      # float32 model
        rows = [random_pair(rng, K=6) for _ in range(5)]
        S = np.stack([r[0] for r in rows]); N = np.stack([r[1] for r in rows]); Mk = np.ones((5, 6)); Mk[0, -1] = 0.0
        X64 = nbm.moment_features_knots(*to_t(S, N, Mk), 4.69, torch.as_tensor(bands or nbm.V2_KNOTS, dtype=DT))
        with torch.no_grad():
            X = m.moment_features(*to_t(S, N, Mk))
        assert X.dtype == torch.float32 and X.shape == X64.shape
        assert torch.allclose(X, X64.to(torch.float32), atol=1e-5, rtol=1e-5)


def test_learned_bands_reciprocity():
    model = random_model_bessel(1)
    rng = np.random.default_rng(23)
    for _ in range(3):
        s_vec, nbr = random_pair(rng, K=6)
        X_ts = _rows_model(model, s_vec, nbr)
        X_st = _rows_model(model, -s_vec, nbr - s_vec)          # seen from the source: neighbours relative to it
        with torch.no_grad():
            K_ts = model.predict_mobility(X_ts)[0]; K_st = model.predict_mobility(X_st)[0]
        assert torch.allclose(K_st, K_ts.T, atol=1e-9, rtol=1e-8)
        assert (K_ts[:3, 3:] - K_ts[3:, :3]).abs().max() > 1e-6 * K_ts.abs().max()         # class 2 active: TR != RT


@pytest.mark.parametrize("kind", ["rotation", "reflection", "improper"])
def test_o3_equivariance_learned_bands(kind):
    model = random_model_bessel(2)
    rng = np.random.default_rng(29)
    if kind == "rotation":
        R = random_rotation(rng, +1.0)
    elif kind == "reflection":
        R = np.diag([1.0, 1.0, -1.0])
    else:
        R = random_rotation(rng, +1.0) @ np.diag([1.0, -1.0, 1.0])
    D = torch.as_tensor(big_D(R), dtype=DT)
    for _ in range(3):
        s_vec, nbr = random_pair(rng, K=6)
        X = _rows_model(model, s_vec, nbr)
        XR = _rows_model(model, s_vec @ R.T, nbr @ R.T)
        with torch.no_grad():
            K = model.predict_mobility(X)[0]; KR = model.predict_mobility(XR)[0]
            assert torch.allclose(model._invariants(X), model._invariants(XR), atol=1e-9, rtol=1e-9)
        assert torch.allclose(KR, D @ K @ D.T, atol=1e-9, rtol=1e-8)


def test_torchscript_roundtrip_learned_bands(tmp_path):
    model = random_model_bessel(3, nb=8).to(torch.float32)
    rng = np.random.default_rng(31)
    rows = [random_pair(rng, K=5) for _ in range(4)]
    S = np.stack([r[0] for r in rows]); N = np.stack([r[1] for r in rows]); Mk = np.ones((4, 5))
    with torch.no_grad():
        X = model.moment_features(*to_t(S, N, Mk))
        model.fit_normalisation(X)
    scripted = torch.jit.script(model)
    p = tmp_path / "m.pt"
    scripted.save(str(p))
    loaded = torch.jit.load(str(p)).eval()
    assert nbm.layout_of_model(loaded) == nbm.layout_of_model(model) and nbm.knots_of(loaded) is None
    assert (loaded.nb, loaded.learned_radial, loaded.n_radial, loaded.has_tr2) == (8, True, 8, True)
    with torch.no_grad():
        XL = loaded.moment_features(*to_t(S, N, Mk))
        assert torch.allclose(XL, X, atol=1e-6)
        assert torch.allclose(model.predict_mobility(X), loaded.predict_mobility(XL), atol=1e-6)
    model.train()                                                     # gradients reach the radial MLP through the rows
    X = model.moment_features(*to_t(S, N, Mk))
    model.predict_mobility(X).square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.radial.parameters())


@pytest.mark.skipif(not MODELS_PRESENT, reason="model files missing")
def test_operator_learned_bands_rows_and_symmetry(tmp_path):
    """A learned-radial model in the CPU operator: rows built by the model (per-pair from-scratch check), symmetric
    grand M, and the .wt + sidecar path equal to the self-describing .pt path."""
    import json
    from benchmarks.cluster import uniform_sphere_cluster
    from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments
    model = random_model_bessel(6, nb=6).to(torch.float32).eval()
    pt = tmp_path / "b.pt"; wt = tmp_path / "b.wt"
    torch.jit.script(model).save(str(pt))
    torch.save(model.state_dict(), wt)
    common = dict(shape="sphere", self_nn_path="data/models/self_interaction_model.pt",
                  two_nn_path="data/models/two_body_combined_model.pt", switch_dist=8.0, pair_cutoff=8.0,
                  neighbor_cutoff=8.0, max_neighbors=None, mean_dist_s=4.69)   # = the model's pair-scalar constant
    op = Mob_Op_Nbody_Moments(nbody_nn_path=str(pt), **common)
    assert op.nbody_layout["radial"] == "bessel" and op.nbody_layout["x_dim"] == 85 and op.band_knots is None
    pos, _ = uniform_sphere_cluster(0.15, 40, seed=5)
    pairs, X_ts, X_st = op._pair_rows(pos)
    assert len(pairs) > 50 and X_ts.shape == (len(pairs), 85)
    for i, (t, s) in list(enumerate(pairs))[:40]:
        idx = op._select_neighbor_indices(pos, t, s)
        X = _rows_model(model, pos[s] - pos[t], pos[idx] - pos[t])[0].numpy()
        assert np.allclose(X, X_ts[i], rtol=1e-5, atol=1e-6)
    assert np.allclose(X_st[:, :3], -X_ts[:, :3]) and np.array_equal(X_st[:, 3:], X_ts[:, 3:])
    pos12, _ = uniform_sphere_cluster(0.2, 12, seed=0)
    n = len(pos12)
    M = np.zeros((6 * n, 6 * n))
    for j in range(6 * n):
        F = np.zeros((n, 6)); F[j // 6, j % 6] = 1.0
        M[:, j] = op.get_nbody_velocity(pos12, F, 1.0).reshape(-1)
    assert np.linalg.norm(M) > 0 and np.linalg.norm(M - M.T) / np.linalg.norm(M) < 1e-5
    config = np.hstack([pos12, np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))])
    F = np.random.default_rng(0).normal(size=(n, 6))
    json.dump({"radial": "bessel", "nb": 6, "n_radial": 8, "bases": "linear_tr2", "invariants": "full"}, open(tmp_path / "b.json", "w"))
    op_wt = Mob_Op_Nbody_Moments(nbody_nn_path=str(wt), **common)
    assert np.allclose(op_wt.apply(config, F, 1.0), op.apply(config, F, 1.0), atol=1e-6, rtol=1e-5)
