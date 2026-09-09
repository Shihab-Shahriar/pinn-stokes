"""Tests for the Stokesian Dynamics harness operator (src/sd_ops.py).

CPU only; the first test pays the numba JIT of the SD kernels (~20 s), the rest are seconds. Skipped entirely
when the SD checkout (SD_ROOT) is not present. Run from the repo root: python -m pytest tests/test_sd_ops.py -q
"""
import math

import numpy as np
import pytest

from src import sd_ops as S

pytestmark = pytest.mark.skipif(not (S.sd_package_dir() / "settings.py").exists(),
                                reason=f"Stokesian Dynamics checkout not found at {S.SD_ROOT}")
SIX_PI, EIGHT_PI = 6.0 * math.pi, 8.0 * math.pi


def cfg(positions):
    positions = np.asarray(positions, dtype=np.float64)
    return np.hstack([positions, np.tile([0.0, 0.0, 0.0, 1.0], (len(positions), 1))])


def rsa(rng, N, side, min_dist=2.1):
    pts = []
    while len(pts) < N:
        p = rng.uniform(-side / 2, side / 2, 3)
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    return np.array(pts)


@pytest.fixture(scope="module")
def op():
    return S.SDMob()


@pytest.fixture(scope="module")
def op_far():
    return S.SDMob(minfinity_only=True)


def test_isolated_sphere(op, op_far):
    F = np.array([[0.3, -1.2, 2.0, 0.7, 0.1, -0.4]])
    for o in (op, op_far):
        v = o.apply(cfg([[1.0, 2.0, 3.0]]), F, 1.0)
        assert np.allclose(v[0, :3], F[0, :3] / SIX_PI, atol=1e-12)
        assert np.allclose(v[0, 3:], F[0, 3:] / EIGHT_PI, atol=1e-12)
    v2 = op.apply(cfg([[0.0, 0.0, 0.0]]), F, 2.0)  # viscosity scaling
    assert np.allclose(v2, op.apply(cfg([[0.0, 0.0, 0.0]]), F, 1.0) / 2.0, atol=1e-12)


def test_matches_native_sd_two_sphere_solve(op):
    """The wrapper equals the package's own FTE path (tests/test_all.py) on its two-sphere test setups."""
    sd = S._import_sd()
    import settings  # noqa: F401  (already imported argv-safe by _import_sd)
    from functions.timestepping import generate_output_FTSUOE
    from setups.tests.positions import pos_setup_tests
    from settings import cutoff_factor, timestep

    # setup -2: two unit spheres at s' = 3.9 (inside the lubrication table, outside the far-field-only zone);
    # inputs -1 (F on sphere 0), -5 (T on sphere 0), -12 (opposite forces)
    posdata, _ = pos_setup_tests(-2)
    positions = np.array(posdata[1], dtype=np.float64)
    wrenches = {-1: [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                -5: [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
                -12: [[1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]]}
    for input_number, F in wrenches.items():
        out = generate_output_FTSUOE(posdata, 0, timestep, input_number, [], True, "fte", cutoff_factor, 0,
                                     False, False, False, [], [])
        Ua, Oa = np.asarray(out[5]), np.asarray(out[6])
        v = op.apply(cfg(positions), np.array(F, dtype=np.float64), 1.0)
        assert np.allclose(v[:, :3], Ua, rtol=0, atol=1e-12), input_number
        assert np.allclose(v[:, 3:], Oa, rtol=0, atol=1e-12), input_number
    assert sd.package_dir == S.sd_package_dir()


def test_lubrication_only_inside_cutoff(op, op_far):
    """R2Bexact changes the answer inside cutoff_factor*(a1+a2) = 4 and is absent beyond it."""
    F = np.array([[1.0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0]])
    near = op.apply(cfg([[0, 0, 0], [2.1, 0, 0]]), F), op_far.apply(cfg([[0, 0, 0], [2.1, 0, 0]]), F)
    far = op.apply(cfg([[0, 0, 0], [4.4, 0, 0]]), F), op_far.apply(cfg([[0, 0, 0], [4.4, 0, 0]]), F)
    assert abs(near[0][0, 0] - near[1][0, 0]) / abs(near[1][0, 0]) > 0.05   # squeezing pair: lubrication matters
    assert np.allclose(far[0], far[1], atol=1e-13)


def test_reciprocity_and_invariance(op):
    rng = np.random.default_rng(0)
    pos = rsa(rng, 8, 9.0)
    Fa, Fb = rng.normal(size=(8, 6)), rng.normal(size=(8, 6))
    va, vb = op.apply(cfg(pos), Fa), op.apply(cfg(pos), Fb)
    assert np.isclose(np.sum(va * Fb), np.sum(vb * Fa), rtol=1e-9)  # M symmetric (Lorentz reciprocity)
    shifted = op.apply(cfg(pos + np.array([5.0, -3.0, 2.0])), Fa)
    assert np.allclose(shifted, va, atol=1e-10)
    assert op.last_info["N"] == 8 and op.last_info["wall_s"] > 0


def test_many_body_matches_pairwise_limit(op):
    """A distant third sphere barely changes a close pair's velocities (sanity of assembly / indexing)."""
    F = np.array([[1.0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0]])
    pair = op.apply(cfg([[0, 0, 0], [2.2, 0, 0]]), F[:2])
    triple = op.apply(cfg([[0, 0, 0], [2.2, 0, 0], [0, 60.0, 0]]), F)
    assert np.allclose(triple[:2], pair, rtol=1e-3, atol=1e-6)
