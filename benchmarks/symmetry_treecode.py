"""Symmetry / SPD diagnostics for the NeMO grand mobility with the treecode far field.

Question this answers: the near field is (believed) symmetric, but is the grand mobility
still symmetric once `WarpFMM` (src/treecode.py) handles the far field for large systems?

The treecode is a target-centric Barnes-Hut cluster->point scheme: each target opens the tree
with its own acceptance test (warp/native/bvh.h:test_acceptance_criterion) and sources are
lumped into a monopole + first-order dipole Taylor expansion about the *source* centroid
(treecode.py:rpy_far_velocity_multipole). There is no target-side local expansion / M2L, so
M_eff[i,j] != M_eff[j,i]^T in general. This script measures how large that is.

Subcommands
-----------
near    Test 1: near-field-only operator (far_field_2b=None, no treecode). Checks the premise.
theta   Test 2: far-field block in isolation, swept over the acceptance criterion. The
        tightest value in the sweep is the causal control -- no node passes, traversal
        degenerates to all-direct pairs, and the far field must come out symmetric to
        float precision.
grand   Test 3: full 6N x 6N grand mobility through apply(), at the production setting.
probe   Test 4: Hutchinson probes -- relative asymmetry at 10k / 100k / 1M without assembly.
ttonly  Test 5: cost of the TT-only far field (torques dropped, RT/TR/RR zero beyond r=6).
all     Everything, and dump results to artifacts/treecode_symmetry_results.json.

Backend
-------
NEMO_FAR_FIELD selects the far field: `widebvh` (production, BaryStokes, tuned by `mac`,
default 0.8 via NEMO_MAC), `widebvh-cart` (the same engine's Cartesian Taylor expansion,
also tuned by `mac` but at its own calibrated value and NEMO_CART_ORDER), or `warp` (the
previous WarpFMM, tuned by opening angle `theta`). The criteria are measured against
different node radii, and the two widebvh policies truncate different series, so no value
transfers between them: every sweep and default here is expressed in whichever knob the
selected backend uses -- see KNOB / CONTROL / PRODUCTION below.

Note on reading old results: the numbers in artifacts/treecode_symmetry_report.md were
taken with the Warp far field AND before the n-body chunking bug was fixed
(src/gpu_nbody_mob.py:_per_particle_topk), which inflated every measurement above
~400k particles. artifacts/widebvh_far_field_report.md section 5 carries the current table.

Usage
-----
    source ~/warp_env.sh
    export TORCH_COMPILE_DISABLE=1          # accuracy run, per CLAUDE.md
    python benchmarks/symmetry_treecode.py theta
    NEMO_FAR_FIELD=warp python benchmarks/symmetry_treecode.py theta
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# insert(0): see the note in two_suspensions_1M.py -- PYTHONPATH carries an
# older pinn-stokes checkout that would otherwise shadow this repo's `src`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gpu_nbody_mob import Mob_Nbody_Torch
from src.treecode import WarpFMM
from src.spdness_of_nn_mobility import test_M
from benchmarks.bench_rpy import two_body_rpy_batch

DEVICE = "cuda"
CUTOFF = 6.0  # near/far switch; also hardcoded as 36.0 in warp/native/bvh.h:640
RADIUS = 1.0
PREF_TT = 1.0 / (8.0 * math.pi)

SELF_NN = "data/models/self_interaction_model.pt"
TWO_NN = "data/models/combined_2body.wt"
NBODY_NN = "data/models/nbody_cross_tmp.wt"

DEFAULT_CFG = "tmp/uniform_sphere_0.1_800.csv"


# --------------------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------------------

@contextlib.contextmanager
def quiet():
    """WarpFMM.apply / get_far_field_vel print timing on every call; a 6N-column
    assembly would emit ~15k lines. Suppress without touching src/."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def load_config(path: str, n: int | None = None):
    """Return (positions (N,3) f32, orientations (N,4) f32).

    Subsampling keeps the `n` particles closest to the centroid so the local number
    density -- and therefore the near/far pair balance -- is preserved. Taking a random
    subset would thin the suspension and change what we are measuring.
    """
    df = pd.read_csv(path, float_precision="high")
    pos = df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)

    if {"q_x", "q_y", "q_z", "q_w"}.issubset(df.columns):
        orient = df[["q_x", "q_y", "q_z", "q_w"]].to_numpy(dtype=np.float32, copy=True)
    else:  # tmp/uniform_large_*.csv carry positions only; spheres -> identity quaternion
        orient = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (pos.shape[0], 1))

    if n is not None and n < pos.shape[0]:
        d = np.linalg.norm(pos - pos.mean(axis=0), axis=1)
        keep = np.argsort(d)[:n]
        pos, orient = pos[keep], orient[keep]

    return np.ascontiguousarray(pos), np.ascontiguousarray(orient)


def to_gpu(pos, orient):
    return (
        torch.as_tensor(pos, device=DEVICE),
        torch.as_tensor(orient, device=DEVICE),
        torch.ones(pos.shape[0], device=DEVICE, dtype=torch.float32),
    )


def make_nbody(far_field):
    """The production near-field operator (benchmarks/two_suspensions_1M.py:178-202).

    far_field=None  -> near field only; pairs beyond `switch_dist` contribute nothing,
                       which is exactly the operator WarpFMM wraps.
    far_field='rpy' -> dense O(N^2) analytic far field, the no-treecode baseline.
    """
    return Mob_Nbody_Torch(
        shape="sphere",
        self_nn_path=SELF_NN,
        two_nn_path=TWO_NN,
        nbody_nn_path=NBODY_NN,
        near_field_2b="nn",
        far_field_2b=far_field,
        near_far_switch=CUTOFF,
    )


# Which far field the diagnostics measure. "widebvh" is production; "warp"
# reproduces the numbers in artifacts/treecode_symmetry_report.md, which were all
# taken against WarpFMM.
FAR_FIELD_BACKEND = os.environ.get("NEMO_FAR_FIELD", "widebvh")
MAC = float(os.environ.get("NEMO_MAC", "0.8"))
CART_ORDER = int(os.environ.get("NEMO_CART_ORDER", "0")) or None

# The two backends have different acceptance criteria -- Warp's opening angle `theta`
# and widebvh's `mac` -- and their values do not transfer. The sweeps below therefore
# ask the backend for its own knob rather than hardcoding theta, which they used to do:
# make_solver silently ignored its theta argument under widebvh, so `theta` and
# `thetascale` produced a column of identical rows and the theta=0 control (the
# "no node is acceptable" clean-partition check) never actually ran.
WARP_SWEEP = (0.0, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)
WIDEBVH_SWEEP = (0.02, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
# The Cartesian expansion needs a tighter mac for the same accuracy, so its sweep
# is shifted down; the top of the bary range carries no information there.
WIDEBVH_CART_SWEEP = (0.02, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8)

# The value at which no node passes the criterion, so traversal degenerates to all
# direct pairs and the far field must come out symmetric to float precision.
CONTROL = {"warp": 0.0, "widebvh": 0.02, "widebvh-cart": 0.02}[FAR_FIELD_BACKEND]
KNOB = {"warp": "theta", "widebvh": "mac", "widebvh-cart": "mac"}[FAR_FIELD_BACKEND]

# The production setting for whichever backend is selected. Tests that want "the
# operator as shipped" must take this rather than a hardcoded theta -- 0.3 is a
# reasonable Warp opening angle but a needlessly tight (and much slower) widebvh mac.
def _production():
    if FAR_FIELD_BACKEND == "warp":
        return 0.3
    if FAR_FIELD_BACKEND == "widebvh-cart" and "NEMO_MAC" not in os.environ:
        from src.treecode_widebvh import DEFAULT_CART_MAC
        return DEFAULT_CART_MAC
    return MAC


PRODUCTION = _production()


def default_sweep():
    return {"warp": WARP_SWEEP, "widebvh": WIDEBVH_SWEEP,
            "widebvh-cart": WIDEBVH_CART_SWEEP}[FAR_FIELD_BACKEND]


def make_solver(near_op, knob):
    """Build the configured far field. `knob` is theta for warp, mac for widebvh."""
    if FAR_FIELD_BACKEND.startswith("widebvh"):
        from src.treecode_widebvh import WidebvhFMM, DEFAULT_CART_ORDER
        cart = FAR_FIELD_BACKEND == "widebvh-cart"
        return WidebvhFMM(
            near_field_operator=near_op,
            mac=knob,
            policy="cart" if cart else "bary",
            order=(CART_ORDER or DEFAULT_CART_ORDER) if cart else None,
            near_field_cutoff=CUTOFF,
            device=DEVICE,
        )
    return WarpFMM(
        near_field_operator=near_op,
        theta=knob,
        leaf_size=16,
        near_field_cutoff=CUTOFF,
        device=DEVICE,
        block_dim=256,
    )


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

def sym_stats(M: np.ndarray) -> dict:
    """Absolute and relative measures of how far M is from symmetric."""
    M = np.asarray(M, dtype=np.float64)
    A = M - M.T
    fro_M = np.linalg.norm(M)
    return {
        "dim": int(M.shape[0]),
        "fro_M": float(fro_M),
        "fro_asym": float(np.linalg.norm(A)),
        "rel_asym": float(np.linalg.norm(A) / fro_M),
        "max_asym": float(np.abs(A).max()),
        "max_abs_M": float(np.abs(M).max()),
    }


def rel_err(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


def print_stats(label, s):
    print(f"  {label:<34s} rel_asym = {s['rel_asym']:.3e}   "
          f"max|M-M^T| = {s['max_asym']:.3e}   ||M||_F = {s['fro_M']:.4e}")


# --------------------------------------------------------------------------------------
# dense assembly
# --------------------------------------------------------------------------------------

def assemble_far(solver, pos_t) -> np.ndarray:
    """(3N,3N) far-field block: column (j,b) is a unit force e_b on particle j.

    Calls get_far_field_vel directly, so the near-field NN never runs and the treecode
    is measured on its own.
    """
    N = pos_t.shape[0]
    M = np.zeros((3 * N, 3 * N), dtype=np.float64)
    f = torch.zeros(N, 3, device=DEVICE, dtype=torch.float32)

    with quiet():
        for col in range(3 * N):
            f.zero_()
            f[col // 3, col % 3] = 1.0
            v = solver.get_far_field_vel(pos_t, f)
            M[:, col] = v.reshape(-1).double().cpu().numpy()
    return M


def assemble_grand(apply_fn, N) -> np.ndarray:
    """(6N,6N) grand mobility: column (j,b) is a unit wrench e_b on particle j.

    Same convention as compute_M in src/spdness_of_nn_mobility.py, so test_M applies.
    """
    M = np.zeros((6 * N, 6 * N), dtype=np.float64)
    f = torch.zeros(N, 6, device=DEVICE, dtype=torch.float32)

    with quiet():
        for col in range(6 * N):
            f.zero_()
            f[col // 6, col % 6] = 1.0
            v = apply_fn(f)
            M[:, col] = v.reshape(-1).double().cpu().numpy()
    return M


def exact_far_tt(pos: np.ndarray, cutoff: float = CUTOFF) -> np.ndarray:
    """(3N,3N) dense RPY translation-translation block over pairs with r >= cutoff.

    Same algebra as rpy_far_velocity_pair3x3 (treecode.py:103) but in float64 and with
    no tree, so it is symmetric by construction and serves as the truncation reference.
    """
    N = pos.shape[0]
    p = pos.astype(np.float64)
    rvec = p[:, None, :] - p[None, :, :]              # (N,N,3), target - source
    r = np.linalg.norm(rvec, axis=-1)
    np.fill_diagonal(r, np.inf)
    far = r >= cutoff

    inv_r = np.where(far, 1.0 / r, 0.0)
    inv_r3 = inv_r ** 3
    inv_r5 = inv_r ** 5
    a2 = RADIUS * RADIUS

    eye = np.eye(3)
    rrT = rvec[:, :, :, None] * rvec[:, :, None, :]
    blocks = PREF_TT * (
        (inv_r + (2.0 * a2 / 3.0) * inv_r3)[:, :, None, None] * eye
        + (inv_r3 - 2.0 * a2 * inv_r5)[:, :, None, None] * rrT
    )
    return blocks.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)


# --------------------------------------------------------------------------------------
# dense full-6x6 far field (for Test 5)
# --------------------------------------------------------------------------------------

def dense_far_apply(pos_t, force_t, cutoff: float = CUTOFF, block: int = 256):
    """Full 6x6 analytic RPY far field (r >= cutoff), chunked over targets.

    Returns (U, Omega), each (N,3). Signs follow benchmarks/bench_rpy.py: the coupling
    block is -scale * [r_hat]_x, i.e. u = scale * (T x r_hat) -- verified against
    two_body_rpy_batch in check_dense_far_matches_bench_rpy().
    """
    N = pos_t.shape[0]
    p = pos_t.double()
    F, T = force_t[:, :3].double(), force_t[:, 3:].double()

    U = torch.zeros(N, 3, device=DEVICE, dtype=torch.float64)
    W = torch.zeros(N, 3, device=DEVICE, dtype=torch.float64)
    a2 = RADIUS * RADIUS

    for s in range(0, N, block):
        e = min(s + block, N)
        rvec = p[s:e, None, :] - p[None, :, :]         # (B,N,3), target - source
        r = rvec.norm(dim=-1)
        mask = (r >= cutoff)
        r_safe = torch.where(mask, r, torch.ones_like(r))
        rhat = rvec / r_safe[..., None]

        inv_r = torch.where(mask, 1.0 / r_safe, torch.zeros_like(r))
        inv_r2 = inv_r ** 2
        inv_r3 = inv_r ** 3

        tt_id = PREF_TT * inv_r * (1.0 + 2.0 * a2 * inv_r2 / 3.0)
        tt_rh = PREF_TT * inv_r * (1.0 - 2.0 * a2 * inv_r2)
        rr_id = -(1.0 / (16.0 * math.pi)) * inv_r3
        rr_rh = (3.0 / (16.0 * math.pi)) * inv_r3
        rt_sc = PREF_TT * inv_r2

        # translation from force
        U[s:e] += (tt_id[..., None] * F[None]).sum(1)
        U[s:e] += (tt_rh[..., None] * rhat * (rhat * F[None]).sum(-1, keepdim=True)).sum(1)
        # translation from torque:  u = scale * (T x r_hat)
        U[s:e] += (rt_sc[..., None] * torch.cross(T[None].expand_as(rhat), rhat, dim=-1)).sum(1)

        # rotation from force: TR block equals the RT block (see module docstring of bench_rpy)
        W[s:e] += (rt_sc[..., None] * torch.cross(F[None].expand_as(rhat), rhat, dim=-1)).sum(1)
        # rotation from torque
        W[s:e] += (rr_id[..., None] * T[None]).sum(1)
        W[s:e] += (rr_rh[..., None] * rhat * (rhat * T[None]).sum(-1, keepdim=True)).sum(1)

    return U, W


def check_dense_far_matches_bench_rpy():
    """Validate dense_far_apply's block structure and signs against the production
    RPY path (benchmarks/bench_rpy.py:two_body_rpy_batch)."""
    torch.manual_seed(0)
    # two particles well outside the cutoff
    pos = torch.tensor([[0.0, 0.0, 0.0], [7.3, -2.1, 4.4]], device=DEVICE)
    f = torch.randn(2, 6, device=DEVICE)

    U, W = dense_far_apply(pos, f)
    mine = torch.cat([U, W], dim=1)                                  # (2,6) per-particle

    rel = (pos[0] - pos[1]).reshape(1, 3)
    radii = torch.ones(1, 2, device=DEVICE)
    K = two_body_rpy_batch(rel, radii)[0].double()                   # (12,12), [T0 T1 | R0 R1]

    # bench_rpy layout is [all translations; all rotations]; reorder our per-particle wrench
    fv = torch.cat([f[0, :3], f[1, :3], f[0, 3:], f[1, 3:]]).double()
    ref = K @ fv
    ref = torch.stack([
        torch.cat([ref[0:3], ref[6:9]]),
        torch.cat([ref[3:6], ref[9:12]]),
    ])

    err = (mine - ref).norm() / ref.norm()
    assert err < 1e-5, f"dense_far_apply disagrees with two_body_rpy_batch: rel err {err:.2e}"
    return float(err)


# --------------------------------------------------------------------------------------
# Test 1 -- near field only
# --------------------------------------------------------------------------------------

def test_near(cfg=DEFAULT_CFG, n=150):
    print(f"\n=== Test 1: near-field-only operator (no treecode)  [{cfg}, N={n}] ===")
    pos, orient = load_config(cfg, n)
    pos_t, orient_t, _ = to_gpu(pos, orient)

    op = make_nbody(far_field=None)   # pairs beyond 6 contribute nothing
    M = assemble_grand(lambda f: op.apply(pos_t, orient_t, f, 1.0), n)

    s = sym_stats(M)
    print_stats("near field (self + 2b + nbody)", s)
    eig = test_M(M, sym_tol=1e-5, eig_tol=1e-4)
    s["min_eig"] = float(eig.min())
    s["n_neg_eig"] = int((eig < -1e-4).sum())
    return s


# --------------------------------------------------------------------------------------
# Test 2 -- far field vs opening angle
# --------------------------------------------------------------------------------------

def test_theta(cfg=DEFAULT_CFG, n=800, thetas=None):
    thetas = default_sweep() if thetas is None else thetas
    print(f"\n=== Test 2: far-field block in isolation vs {KNOB}  "
          f"[{cfg}, N={n}, {FAR_FIELD_BACKEND}] ===")
    pos, orient = load_config(cfg, n)
    pos_t, _, _ = to_gpu(pos, orient)

    print("  building dense float64 reference (all pairs r >= 6) ...")
    M_ref = exact_far_tt(pos)
    ref_stats = sym_stats(M_ref)
    print_stats("dense reference (must be exact 0)", ref_stats)
    assert ref_stats["rel_asym"] < 1e-14, "reference is not symmetric -- bug in exact_far_tt"

    near_op = make_nbody(far_field=None)
    rows = []
    for th in thetas:
        solver = make_solver(near_op, th)
        t0 = time.perf_counter()
        M = assemble_far(solver, pos_t)
        dt = time.perf_counter() - t0

        s = sym_stats(M)
        s[KNOB] = th
        s["theta"] = th          # kept so old result JSONs stay comparable
        s["trunc_err"] = rel_err(M, M_ref)
        s["assemble_s"] = dt
        # If the treecode error E = M - M_exact were symmetric, asymmetry would vanish;
        # if E were entrywise uncorrelated with its transpose, the ratio would be sqrt(2).
        # Where this ratio lands says how much of the error is "extra" asymmetry.
        s["asym_over_trunc"] = (s["rel_asym"] / s["trunc_err"]) if s["trunc_err"] > 1e-12 else 0.0
        rows.append(s)
        print(f"  {KNOB}={th:<5} rel_asym = {s['rel_asym']:.3e}   "
              f"trunc_err = {s['trunc_err']:.3e}   ratio = {s['asym_over_trunc']:.3f}   "
              f"max|M-M^T| = {s['max_asym']:.3e}   ({dt:.0f}s)")

    zero = next(r for r in rows if r[KNOB] == CONTROL)
    print(f"\n  [control] {KNOB}={CONTROL} asymmetry     : {zero['rel_asym']:.3e}  "
          f"(expect ~1e-7, float32 pair math)")
    print(f"  [control] {KNOB}={CONTROL} vs dense ref  : {zero['trunc_err']:.3e}  "
          f"(expect ~1e-7 -> near/far split is the clean r>=6 partition)")
    return {"reference": ref_stats, "sweep": rows}


# --------------------------------------------------------------------------------------
# Test 3 -- full grand mobility
# --------------------------------------------------------------------------------------

def test_grand(cfg=DEFAULT_CFG, n=150, theta=None):
    theta = PRODUCTION if theta is None else theta
    print(f"\n=== Test 3: full grand M through {FAR_FIELD_BACKEND} [{cfg}, N={n}, {KNOB}={theta}] ===")
    pos, orient = load_config(cfg, n)
    pos_t, orient_t, vis = to_gpu(pos, orient)

    near_op = make_nbody(far_field=None)
    solver = make_solver(near_op, theta)
    M_fmm = assemble_grand(lambda f: solver.apply(pos_t, orient_t, f, vis).clone(), n)
    s_fmm = sym_stats(M_fmm)
    print_stats("treecode far field", s_fmm)
    eig = test_M(M_fmm, sym_tol=1e-5, eig_tol=1e-4)
    s_fmm["min_eig"] = float(eig.min())
    s_fmm["n_neg_eig"] = int((eig < -1e-4).sum())

    print("\n  baseline: same operator, dense analytic RPY far field, no treecode")
    base_op = make_nbody(far_field="rpy")
    M_base = assemble_grand(lambda f: base_op.apply(pos_t, orient_t, f, 1.0), n)
    s_base = sym_stats(M_base)
    print_stats("dense RPY far field", s_base)
    eig_b = test_M(M_base, sym_tol=1e-5, eig_tol=1e-4)
    s_base["min_eig"] = float(eig_b.min())
    s_base["n_neg_eig"] = int((eig_b < -1e-4).sum())

    return {"fmm": s_fmm, "dense_rpy_baseline": s_base,
            "fmm_vs_baseline_rel": rel_err(M_fmm, M_base)}


# --------------------------------------------------------------------------------------
# Test 4 -- Hutchinson probes at scale
# --------------------------------------------------------------------------------------

def hutchinson(apply_fn, N, K=24, seed=0):
    """Estimate ||M - M^T||_F / ||M||_F from K matrix-vector products, no assembly.

    For iid Rademacher u, v:  E[(u^T M v)^2] = ||M||_F^2  and
    E[(u^T M v - v^T M u)^2] = ||M - M^T||_F^2.  With K probes we form all K(K-1)/2
    ordered pairs; the pair estimates share probes, so the reported spread uses K
    (not the pair count) as the effective sample size.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    Z, MZ = [], []
    with quiet():
        for k in range(K):
            z = (torch.randint(0, 2, (N, 6), device=DEVICE, generator=g,
                               dtype=torch.float32) * 2 - 1)
            Z.append(z)
            MZ.append(apply_fn(z).clone().double())

    Zd = [z.double() for z in Z]
    diffs, cross = [], []
    for a in range(K):
        for b in range(a + 1, K):
            s_ab = float((Zd[a] * MZ[b]).sum())     # z_a^T M z_b
            s_ba = float((Zd[b] * MZ[a]).sum())     # z_b^T M z_a
            diffs.append((s_ab - s_ba) ** 2)
            cross.append(s_ab ** 2)
            cross.append(s_ba ** 2)

    fro_asym = math.sqrt(float(np.mean(diffs)))
    fro_M = math.sqrt(float(np.mean(cross)))
    # per-probe jackknife-style spread, conservative
    spread = float(np.std(diffs) / math.sqrt(K)) / (2 * fro_asym) if fro_asym > 0 else 0.0
    return {"K": K, "fro_asym": fro_asym, "fro_M": fro_M,
            "rel_asym": fro_asym / fro_M, "rel_asym_stderr": spread / fro_M}


def test_probe(theta=None, K=24):
    theta = PRODUCTION if theta is None else theta
    print(f"\n=== Test 4: Hutchinson probes at scale  [{KNOB}={theta}, K={K}] ===")
    results = []

    # --- validation: must reproduce the dense answer at N=150 ---
    print("  validating estimator against dense assembly (N=150) ...")
    pos, orient = load_config(DEFAULT_CFG, 150)
    pos_t, orient_t, vis = to_gpu(pos, orient)
    near_op = make_nbody(far_field=None)
    solver = make_solver(near_op, theta)
    fn = lambda f: solver.apply(pos_t, orient_t, f, vis).clone()

    dense = sym_stats(assemble_grand(fn, 150))
    est = hutchinson(fn, 150, K=K, seed=1)
    ratio = est["rel_asym"] / dense["rel_asym"] if dense["rel_asym"] > 0 else float("nan")
    print(f"    dense rel_asym = {dense['rel_asym']:.4e}")
    print(f"    probe rel_asym = {est['rel_asym']:.4e} +/- {est['rel_asym_stderr']:.1e}"
          f"   (ratio {ratio:.3f})")
    validation = {"N": 150, "dense": dense, "probe": est, "ratio": ratio}

    # --- production scales ---
    for path, N in [("tmp/uniform_large_0.1_10000.csv", 10_000),
                    ("tmp/uniform_large_0.1_100000.csv", 100_000),
                    ("tmp/uniform_large_0.1_1000000.csv", 1_000_000)]:
        if not os.path.exists(path):
            print(f"    (skip {path} -- not found)")
            continue
        pos, orient = load_config(path)
        pos_t, orient_t, vis = to_gpu(pos, orient)
        op = make_nbody(far_field=None)
        sol = make_solver(op, theta)
        f_ap = lambda f: sol.apply(pos_t, orient_t, f, vis).clone()

        t0 = time.perf_counter()
        e = hutchinson(f_ap, pos.shape[0], K=K, seed=2)
        e["N"] = int(pos.shape[0])
        e["wall_s"] = time.perf_counter() - t0
        results.append(e)
        print(f"    N={e['N']:>9,}  rel_asym = {e['rel_asym']:.4e} "
              f"+/- {e['rel_asym_stderr']:.1e}   ({e['wall_s']:.0f}s)")

        del sol, op, pos_t, orient_t, vis
        torch.cuda.empty_cache()

    return {"validation": validation, "scales": results}


def test_thetascale(path="tmp/uniform_large_0.1_100000.csv", thetas=None, K=16):
    """Prices the accuracy lever at production scale: asymmetry AND far-field wall time
    as a function of the acceptance criterion (theta for warp, mac for widebvh)."""
    # Each backend is priced over its own useful range. The Cartesian policy
    # reaches the production error at mac 0.33, so bary's 0.6-1.0 band is pure
    # noise there -- it was the last place a sweep still ignored the backend.
    if thetas is None:
        thetas = {"warp": (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
                  "widebvh": (0.6, 0.7, 0.8, 0.9, 1.0),
                  "widebvh-cart": (0.25, 0.3, 0.33, 0.4, 0.5)}[FAR_FIELD_BACKEND]
    print(f"\n=== Test 6: asymmetry vs cost at scale  "
          f"[{path}, K={K}, {FAR_FIELD_BACKEND}] ===")
    pos, orient = load_config(path)
    pos_t, orient_t, vis = to_gpu(pos, orient)
    N = pos.shape[0]
    op = make_nbody(far_field=None)

    f3 = torch.randn(N, 3, device=DEVICE)
    rows = []
    for th in thetas:
        sol = make_solver(op, th)
        e = hutchinson(lambda f: sol.apply(pos_t, orient_t, f, vis).clone(), N, K=K, seed=3)

        f6 = torch.randn(N, 6, device=DEVICE)
        with quiet():                       # time far field alone and full step, after warmup
            sol.get_far_field_vel(pos_t, f3)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                sol.get_far_field_vel(pos_t, f3)
            torch.cuda.synchronize()
            e["far_ms"] = (time.perf_counter() - t0) * 1000.0 / 5

            sol.apply(pos_t, orient_t, f6, vis)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                sol.apply(pos_t, orient_t, f6, vis)
            torch.cuda.synchronize()
            e["total_ms"] = (time.perf_counter() - t0) * 1000.0 / 3
        e[KNOB] = th
        e["theta"] = th          # kept so old result JSONs stay comparable
        e["N"] = N
        rows.append(e)
        print(f"  {KNOB}={th:<5} rel_asym = {e['rel_asym']:.3e} +/- {e['rel_asym_stderr']:.1e}"
              f"   far-field = {e['far_ms']:7.2f} ms   full step = {e['total_ms']:7.2f} ms")
        del sol
        torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------------------
# Test 5 -- cost of the TT-only far field
# --------------------------------------------------------------------------------------

def far_block_norms(cfg=DEFAULT_CFG, n=150):
    """Loading-independent view of the TT-only truncation: assemble the dense full-6x6
    far-field operator and report how much of its Frobenius norm lives outside TT.

    This is what explains the treecode-vs-dense-RPY gap in Test 3 -- that gap is the
    dropped blocks, not the treecode's tree approximation.
    """
    pos, orient = load_config(cfg, n)
    pos_t, _, _ = to_gpu(pos, orient)
    N = pos.shape[0]

    M = np.zeros((6 * N, 6 * N))
    f = torch.zeros(N, 6, device=DEVICE)
    for col in range(6 * N):
        f.zero_()
        f[col // 6, col % 6] = 1.0
        U, W = dense_far_apply(pos_t, f)
        M[:, col] = torch.cat([U, W], dim=1).reshape(-1).cpu().numpy()

    idx = np.arange(6 * N)
    t = idx[(idx % 6) < 3]
    r = idx[(idx % 6) >= 3]
    nrm = lambda a: float(np.linalg.norm(a))
    tot = nrm(M)
    out = {
        "N": N,
        "fro_total": tot,
        "fro_TT": nrm(M[np.ix_(t, t)]),
        "fro_RT": nrm(M[np.ix_(t, r)]),   # translation from torque
        "fro_TR": nrm(M[np.ix_(r, t)]),   # rotation from force
        "fro_RR": nrm(M[np.ix_(r, r)]),
        "sym_check": sym_stats(M)["rel_asym"],
    }
    out["frac_dropped"] = float(
        math.sqrt(out["fro_RT"] ** 2 + out["fro_TR"] ** 2 + out["fro_RR"] ** 2) / tot)
    return out


def test_ttonly(sizes=((DEFAULT_CFG, 800), ("tmp/uniform_large_0.1_10000.csv", 10_000))):
    print("\n=== Test 5: cost of the TT-only far field (torques dropped, RT/TR/RR = 0) ===")
    err = check_dense_far_matches_bench_rpy()
    print(f"  dense_far_apply validated against two_body_rpy_batch: rel err {err:.2e}")

    b = far_block_norms()
    print(f"\n  Far-field operator block norms (dense full 6x6 RPY, N={b['N']}):")
    print(f"    ||TT|| = {b['fro_TT']:.4e}   (kept by the treecode)")
    print(f"    ||RT|| = {b['fro_RT']:.4e}   ||TR|| = {b['fro_TR']:.4e}   "
          f"||RR|| = {b['fro_RR']:.4e}   (all set to zero)")
    print(f"    dropped fraction of ||M_far||_F = {b['frac_dropped']:.3%}")
    print(f"    (dense far-field operator is symmetric: rel_asym = {b['sym_check']:.2e})\n")

    torch.manual_seed(0)
    rows = []
    for cfg, n in sizes:
        if not os.path.exists(cfg):
            print(f"  (skip {cfg} -- not found)")
            continue
        pos, orient = load_config(cfg, n)
        pos_t, _, _ = to_gpu(pos, orient)
        N = pos.shape[0]

        # representative loading: uniform sedimentation force + random torques
        f = torch.zeros(N, 6, device=DEVICE)
        f[:, 2] = -1.0
        f[:, 3:] = torch.randn(N, 3, device=DEVICE)

        U_full, W_full = dense_far_apply(pos_t, f)
        # what the treecode keeps: TT block only, torques ignored
        f_tt = f.clone()
        f_tt[:, 3:] = 0.0
        U_tt, _ = dense_far_apply(pos_t, f_tt)

        e_u = float((U_tt - U_full).norm() / U_full.norm())

        row = {"config": cfg, "N": N,
               "rel_err_U": e_u,
               "rel_err_Omega": 1.0,     # far-field Omega is entirely dropped
               "far_Omega_norm": float(W_full.norm()),
               "far_U_norm": float(U_full.norm())}

        # split Omega into its two far-field sources so we can say what is lost
        f_only_F = f.clone(); f_only_F[:, 3:] = 0.0
        _, W_from_F = dense_far_apply(pos_t, f_only_F)
        f_only_T = f.clone(); f_only_T[:, :3] = 0.0
        U_from_T, W_from_T = dense_far_apply(pos_t, f_only_T)
        row["omega_from_force_norm"] = float(W_from_F.norm())
        row["omega_from_torque_norm"] = float(W_from_T.norm())
        row["u_from_torque_norm"] = float(U_from_T.norm())
        row["u_from_force_norm"] = float((U_full - U_from_T).norm())

        # Absolute norms mean nothing on their own -- compare against what the near field
        # actually delivers for the same loading, since that is all the operator keeps.
        op = make_nbody(far_field=None)
        with quiet():
            v_near = op.apply(pos_t, torch.zeros(N, 4, device=DEVICE), f, 1.0).double()
        u_near_n = float(v_near[:, :3].norm())
        w_near_n = float(v_near[:, 3:].norm())
        row["u_near_norm"] = u_near_n
        row["omega_near_norm"] = w_near_n
        row["omega_far_over_near"] = float(W_full.norm()) / w_near_n
        row["u_far_over_near"] = float(U_full.norm()) / u_near_n
        del op
        torch.cuda.empty_cache()

        rows.append(row)
        print(f"  {cfg}  N={N}")
        print(f"    U:     far-field error from dropping torque coupling = {e_u:.3e}")
        print(f"    Omega: far-field contribution is dropped entirely (100%)")
        print(f"      ||Omega_far|| / ||Omega_near|| = {row['omega_far_over_near']:.3e}"
              f"   <-- systematic error in angular velocity")
        print(f"      ||U_far||     / ||U_near||     = {row['u_far_over_near']:.3e}"
              f"   (retained)")
        print(f"      breakdown: ||Omega_far|| from forces = {row['omega_from_force_norm']:.4e}, "
              f"from torques = {row['omega_from_torque_norm']:.4e}")
    return {"blocks": b, "loadings": rows}


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["near", "theta", "grand", "probe", "ttonly",
                                     "thetascale", "all"])
    ap.add_argument("--n", type=int, default=150,
                    help="particles for the 6N-column dense tests (near, grand)")
    ap.add_argument("--n-far", type=int, default=800, dest="n_far",
                    help="particles for the 3N-column far-field-only theta sweep")
    ap.add_argument("--theta", type=float, default=None,
                    help="acceptance criterion; theta for warp, mac for widebvh. "
                         "Default is that backend's production value.")
    ap.add_argument("--K", type=int, default=24, help="Hutchinson probes")
    args = ap.parse_args()

    if os.environ.get("TORCH_COMPILE_DISABLE") != "1":
        print("WARNING: TORCH_COMPILE_DISABLE is not set to 1; accuracy runs should set it.")

    out = {}
    if args.mode in ("near", "all"):
        out["near"] = test_near(n=args.n)
    if args.mode in ("theta", "all"):
        out["theta"] = test_theta(n=args.n_far)
    if args.mode in ("grand", "all"):
        out["grand"] = test_grand(n=args.n, theta=args.theta)
    if args.mode in ("probe", "all"):
        out["probe"] = test_probe(theta=args.theta, K=args.K)
    if args.mode in ("ttonly", "all"):
        out["ttonly"] = test_ttonly()
    if args.mode in ("thetascale", "all"):
        out["thetascale"] = test_thetascale(K=max(8, args.K // 2))

    os.makedirs("artifacts", exist_ok=True)
    suffix = "" if args.mode == "all" else f"_{args.mode}"
    dst = f"artifacts/treecode_symmetry_results{suffix}.json"
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
