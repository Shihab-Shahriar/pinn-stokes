"""Structural tests for the HIGNN baseline adapter (src/hignn_ops.py).

CPU only, seconds. Skipped entirely when the hignn checkout (HIGNN_ROOT) is not present.
Run from the repo root: python -m pytest tests/test_hignn_ops.py -q
"""
import math

import numpy as np
import pytest
import torch

from src import hignn_ops as H

pytestmark = pytest.mark.skipif(not H.weight_path("2b").exists(), reason=f"hignn checkout not found at {H.HIGNN_ROOT}")
SIX_PI = 6.0 * math.pi


@pytest.fixture(scope="module")
def op2():
    return H.HignnMob(variant="2b", device="cpu")


@pytest.fixture(scope="module")
def opf():
    return H.HignnMob(variant="full", device="cpu", pair_chunk=997, triple_chunk=131)  # tiny chunks: exercise the chunking


def rsa(rng, N, side, min_dist=2.1):
    """Random non-overlapping unit spheres in a cube of the given side (centre distance >= min_dist)."""
    pts = []
    while len(pts) < N:
        p = rng.uniform(-side / 2, side / 2, 3)
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    return np.array(pts)


# ----------------------------------------------------------------------------- kernels / units
def test_isolated_sphere(op2, opf):
    for op in (op2, opf):
        v = op.apply_cpu(np.zeros((1, 3)), np.array([[0, 0, 0, 1.0]]), np.array([[1.0, 0, 0, 0, 0, 0]]))
        assert v.shape == (1, 6) and v.dtype == np.float64
        assert np.allclose(v[0, :3], [1 / SIX_PI, 0, 0], atol=1e-9) and np.all(v[0, 3:] == 0)
        v = op.apply_cpu(np.zeros((1, 3)), np.array([[0, 0, 0, 1.0]]), np.array([[0, 0, -9.81, 0, 0, 0]]), viscosity=2.0)
        assert np.isclose(v[0, 2], -9.81 / SIX_PI / 2.0)


def test_two_sphere_block(op2):
    """Their kernel's own values at rel = (0, 0, 3) (RPY would give zz 0.46296, xx 0.26852)."""
    M = op2.pair_mobility(torch.tensor([[0.0, 0.0, 3.0]]))[0].numpy()
    assert abs(M[2, 2] - 0.47403) < 2e-5 and abs(M[0, 0] - 0.26801) < 2e-5
    # two spheres, unit force on sphere 0 only: u0 = F/6pi (self), u1 = M(x0 - x1) F / 6pi
    X = np.array([[0, 0, 0], [0, 0, 3.0]])
    F = np.array([[0, 0, -1.0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    v = op2.apply_cpu(X, None, F)
    assert np.allclose(v[0, :3], [0, 0, -1 / SIX_PI], atol=1e-9)
    Mrel = op2.pair_mobility(torch.tensor([[0.0, 0.0, -3.0]]))[0].numpy()  # x_source - x_target = x0 - x1
    assert np.allclose(v[1, :3], Mrel @ np.array([0, 0, -1.0]) / SIX_PI, atol=1e-7)


def test_block_symmetry_parity_rotation(op2):
    rng = np.random.default_rng(0)
    d = rng.normal(size=(64, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    x = torch.tensor(d * rng.uniform(2.1, 12.0, size=(64, 1)), dtype=torch.float32)
    M, Mm = op2.pair_mobility(x), op2.pair_mobility(-x)
    assert (M - M.transpose(1, 2)).abs().max() < 1e-6                        # learned blocks are symmetric
    assert (M - Mm).abs().max() < 5e-3                                        # ... but only approximately parity-even (3.2e-3 seen)
    # rotation equivariance is learned, not imposed: loose bound only
    from scipy.spatial.transform import Rotation
    R = torch.tensor(Rotation.random(64, random_state=1).as_matrix(), dtype=torch.float32)
    lhs = R @ M @ R.transpose(1, 2)
    rhs = op2.pair_mobility((R @ x.unsqueeze(2)).squeeze(2))
    assert ((lhs - rhs).norm(dim=(1, 2)) / rhs.norm(dim=(1, 2))).max() < 0.05


# ----------------------------------------------------------------------------- edge enumeration vs brute force
def brute_edges(X, eps3):
    N = len(X)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    nb = [[j for j in range(N) if j != i and D[i, j] < eps3] for i in range(N)]
    pairs = {(a, b) for a in range(N) for b in nb[a]}
    chains = {(s, m, t) for s in range(N) for m in nb[s] for t in nb[m] if t != s}   # NeighborLists::BuildThreeBodyInfo
    return nb, pairs, chains


def test_neighbour_pairs_and_chains(opf):
    rng = np.random.default_rng(3)
    X = rsa(rng, 40, 9.0)
    Xt = torch.tensor(X, dtype=torch.float32)
    a, b = opf.neighbour_pairs(Xt)
    nb, pairs, chains = brute_edges(X, opf.eps3)
    assert set(zip(a.tolist(), b.tolist())) == pairs and len(a) == len(pairs)
    assert torch.all(a[1:] >= a[:-1])                                          # sorted by a (the CSR assumption)
    # run the chunked three-body assembly with an identity-like probe: count triples per target
    F = torch.ones((40, 3))
    opf.three_body_velocity(Xt, F, a, b)
    assert opf.last["triples"] == len(chains)
    assert len(chains) > 0 and len(pairs) > 0


# ----------------------------------------------------------------------------- reference forward (HIGNN_mdoel.forward) parity
def reference_velocity(op, X, F, eps3, full):
    """Transcription of HIGNN_mdoel.forward with brute-force NeighborLists edges; HIGNN units."""
    N = len(X)
    Xt = torch.tensor(X, dtype=torch.float32); Ft = torch.tensor(F, dtype=torch.float32)
    v = Ft.double().clone()                                                   # one-body term (edge_attr_one)
    e2 = [(j, i) for i in range(N) for j in range(N) if j != i]               # 2-body edge (j, i), attr F_j
    j, i = torch.tensor([e[0] for e in e2]), torch.tensor([e[1] for e in e2])
    v.index_add_(0, i, torch.bmm(op.pair_mobility(Xt[j] - Xt[i]), Ft[j].unsqueeze(2)).squeeze(2).double())
    if full:
        nb, pairs, chains = brute_edges(X, eps3)
        if chains:
            s, m, t = (torch.tensor(c) for c in zip(*sorted(chains)))         # 3-body edge (j, k, i) = (s, m, t)
            x_in = torch.cat([Xt[m] - Xt[s], Xt[t] - Xt[m]], dim=1)
            v.index_add_(0, t, torch.bmm(op.triple_mobility(x_in), Ft[s].unsqueeze(2)).squeeze(2).double())
        n, t = (torch.tensor(c) for c in zip(*sorted(pairs)))                # self edge (j, i) = (n, t), attr F_t
        v.index_add_(0, t, torch.bmm(op.self_mobility(Xt[n] - Xt[t]), Ft[t].unsqueeze(2)).squeeze(2).double())
    return v.numpy()


@pytest.mark.parametrize("variant", ["2b", "full"])
def test_reference_forward_parity(op2, opf, variant):
    rng = np.random.default_rng(5)
    X = rsa(rng, 50, 11.0)                                                    # ~phi 0.16 in the cube
    F = rng.normal(size=(50, 3))
    op = op2 if variant == "2b" else opf
    ref = reference_velocity(op, X, F, op.eps3, full=(variant == "full"))
    got = op.apply_cpu(X, None, np.concatenate([F, np.zeros((50, 3))], axis=1))[:, :3] * SIX_PI
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-5


def test_full_differs_from_2b(op2, opf):
    rng = np.random.default_rng(7)
    X = rsa(rng, 30, 9.0)
    F = np.tile([0, 0, -1.0, 0, 0, 0], (30, 1))
    v2, vf = op2.apply_cpu(X, None, F), opf.apply_cpu(X, None, F)
    assert np.linalg.norm(vf - v2) / np.linalg.norm(v2) > 1e-4               # the corrections are not silently empty


# ----------------------------------------------------------------------------- gravity truth spot check
SPOT = {(200, 0.025, 1423): (1.35, 1.13), (200, 0.1, 4423): (4.29, 3.82)}   # prmse_lin % (2b, full), this session


@pytest.mark.parametrize("cell", sorted(SPOT))
def test_gravity_spot(op2, opf, cell):
    p = H.truth_file(*cell)
    if not p.exists():
        pytest.skip(f"truth cache missing: {p}")
    d = np.load(p)
    assert np.abs(d["forces"][:, 3:]).max() == 0                              # torque-free protocol
    e2 = H.prmse_lin(op2.apply(d["config"], d["forces"]), d["velocity"])
    ef = H.prmse_lin(opf.apply(d["config"], d["forces"]), d["velocity"])
    assert abs(e2 - SPOT[cell][0]) < 0.02 and abs(ef - SPOT[cell][1]) < 0.02, (e2, ef)
