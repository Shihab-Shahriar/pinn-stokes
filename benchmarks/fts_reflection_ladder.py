"""FTS (force-torque-stresslet) reflection ladder: how much of NeMO's gravity-protocol error is the
uncorrected many-body far field, and how much of it a single stresslet reflection recovers.

Step 0 of the "close the gap to SD's far field" plan.  For each fig4g configuration at N = 200 (the
gravity comparison of artifacts/sd_comparison_report.md) with a FINE BatchedMFS grand mobility M
(cached in tmp/fts_ladder/M_cache/, solved by tmp/fts_ladder/solve_M.py) and truth v = M F for both
forcings (gravity F = (0,0,-9.81), T = 0; random unit wrench), the cumulative ladder at
switch_dist = pair_cutoff = 8:

  rpy / 2b / near_nn / stack_nn        RPY; self + 2b NN (RPY beyond 8); + moments pair model; + learned diagonal
  near_ex / diag_ex / far_ex           exact pair residual (d <= 8) / + exact diagonal / + exact far (sanity ~0)
                                       -> diag_ex is the far-field floor of ANY pairwise near correction
  rpy+ref1 / rpy+ref2 / rpy+full       RPY + the stresslet reflection at 1 / 2 / infinite reflections
                                       (rpy+full == SD_Minf, asserted against src/sd_ops.py)
  stack+ref1 / stack+ref2 / stack+full the shipped stack + the same, unmasked (double-counts the near
                                       triplets the learned models already saw: an upper bound)
  stack+ref1_uncov                     the shipped stack + ref1 restricted to triplets (t, k, s) the learned
                                       models do NOT cover (the no-retrain estimate)
  pnear+ref1_far / pnear+full_far      exact near + diag residuals + the reflection on far pair blocks (d > 8)
                                       only: the exact residuals already contain every k, so this is the floor
                                       after retraining the models on the reflected base (pair labels minus the
                                       in-box reflection, diagonal labels minus its far-k part, operator adds
                                       the global term back)

The FTS blocks come from Stokesian Dynamics' generate_Minfinity (the RPY-with-Faxen FTS grand mobility,
layout [U/F | Omega/T | E/S] with 5-vector stresslets), so nothing about the kernels is re-derived here.

    TORCH_COMPILE_DISABLE=1 python benchmarks/fts_reflection_ladder.py --phis 0.1 0.2 --n-seeds 3
    python benchmarks/fts_reflection_ladder.py --summary        # tables from the CSV -> artifacts/fts_reflection_ladder.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(ROOT)]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))
os.chdir(ROOT)

import benchmarks.paper_accuracy_v2 as pav  # noqa: E402
from benchmarks.compare_nbody_moments import compute_error_stats  # noqa: E402

MC = Path("tmp/fts_ladder/M_cache")
OUT_CSV = Path("data/fts_reflection_ladder.csv")
OUT_MD = Path("artifacts/fts_reflection_ladder.md")
D_SELF = 3.0 / (20.0 * np.pi)      # E<-S self block of a unit sphere at mu = 1: 1 / ((20/3) pi mu a^3)
PC, RC, DIAG_CUTOFF = 8.0, 8.0, 8.0  # the pc8c stack's pair_cutoff / neighbor_cutoff / diag_cutoff
METRICS = ["prmse_lin", "prmse_fluct", "err_mean_pct", "prmse_ang", "rel_rmse", "max_rel_lin"]
RUNGS = ["rpy", "2b", "near_nn", "stack_nn", "near_ex", "diag_ex", "far_ex",
         "rpy+ref1", "rpy+ref2", "rpy+full", "sd_minf",
         "stack+ref1", "stack+ref2", "stack+full", "stack+ref1_uncov",
         "pnear+ref1_far", "pnear+full_far"]

_solver = None


# ----------------------------------------------------------------------------- truth M
def full_M(pos, N, phi, seed, acc, backend, tol_v):
    global _solver
    p = MC / f"N{N}_phi{phi:g}_seed{seed}_{acc}.npz"
    if p.exists():
        return np.load(p)["M"]
    if _solver is None:
        from src.mfs_batched import BatchedMFS
        _solver = BatchedMFS(acc=acc, backend=backend, verbose=True, tol_v=tol_v, max_restarts=1)
    print(f"[M] solving N={N} phi={phi:g} seed={seed} acc={acc} ...", flush=True)
    t0 = time.time()
    M, info = _solver.solve_mobility_matrix(np.ascontiguousarray(pos), raise_on_fail=False)
    M = M.cpu().numpy()
    MC.mkdir(parents=True, exist_ok=True)
    np.savez(p, M=M, wall=time.time() - t0, acc=acc, backend=backend, tol_v=tol_v, symm_err=float(info.symm_err))
    return M


# ----------------------------------------------------------------------------- FTS blocks from SD
def to_blocks(M6, N):
    """SD-layout (6N, 6N) [U(3N) | Omega(3N)] x [F(3N) | T(3N)] -> (N, N, 6, 6) per-particle blocks."""
    return M6.reshape(2, N, 3, 2, N, 3).transpose(1, 4, 0, 2, 3, 5).reshape(N, N, 6, 6)


def fts_parts(pos):
    """A (N,N,6,6), G (6N,5N), Gb (N,N,6,5), Mm (5N,5N), Mm_off (5N,5N) of SD's FTS grand mobility at mu = 1."""
    from src.sd_ops import SDMob, _import_sd
    _import_sd()
    from functions.generate_Minfinity import generate_Minfinity
    N = len(pos)
    Minf, _ = generate_Minfinity(SDMob().posdata(pos), mu=1.0)
    assert Minf.shape == (11 * N, 11 * N)
    A = to_blocks(Minf[:6 * N, :6 * N], N)
    G = Minf[:6 * N, 6 * N:]
    Gb = G.reshape(2, N, 3, N, 5).transpose(1, 3, 0, 2, 4).reshape(N, N, 6, 5)
    Mm = Minf[6 * N:, 6 * N:]
    Mmb = Mm.reshape(N, 5, N, 5).transpose(0, 2, 1, 3)
    idx = np.arange(N)
    assert np.allclose(Mmb[idx, idx], D_SELF * np.eye(5), atol=1e-12), "E-S self block is not (3/20pi) I"
    assert np.allclose(np.linalg.norm(Gb[idx, idx], axis=(1, 2)), 0.0), "G self block must vanish"
    Mm_off = Mm - D_SELF * np.eye(5 * N)
    return A, G, Gb, Mm, Mm_off


def reflection_matrices(G, Mm, Mm_off, N):
    """(N,N,6,6) blocks of the stresslet reflection at 1, 2 and infinitely many reflections."""
    ref1 = -(G @ G.T) / D_SELF
    ref2 = ref1 + (G @ Mm_off @ G.T) / D_SELF ** 2
    full = -G @ np.linalg.solve(Mm, G.T)
    return to_blocks(ref1, N), to_blocks(ref2, N), to_blocks(full, N)


def uncovered_mask(pos):
    """u[t, s, k] = True where the triplet (t, k, s) is NOT covered by the learned corrections:
    near pairs (d_ts <= PC): k outside RC of the pair midpoint (the moments model's neighbourhood,
    nbody_features.select_pair_neighbours with max_neighbors=None); the diagonal (t == s): k beyond DIAG_CUTOFF
    (select_particle_neighbours); far pairs: every k."""
    N = len(pos)
    D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    eye = np.eye(N, dtype=bool)
    near = (D <= PC) & ~eye
    mid = 0.5 * (pos[:, None, :] + pos[None, :, :])
    dmid = np.linalg.norm(pos[None, None, :, :] - mid[:, :, None, :], axis=-1)   # (t, s, k)
    covered = near[:, :, None] & (dmid <= RC)
    idx = np.arange(N)
    covered[idx, idx, :] = (D <= DIAG_CUTOFF) & ~eye
    return ~covered, D


def masked_ref1(Gb, u):
    """-(1/D) sum_k u[t,s,k] G_tk G_sk^T, per target row (memory ~N^2 * 36 doubles per row)."""
    N = Gb.shape[0]
    out = np.empty((N, N, 6, 6))
    for t in range(N):
        out[t] = -np.einsum("sk,kab,skcb->sac", u[t].astype(np.float64), Gb[t], Gb) / D_SELF
    return out


def apply_blocks(B, F, mask=None):
    if mask is None:
        return np.einsum("tsab,sb->ta", B, F)
    return np.einsum("ts,tsab,sb->ta", mask.astype(np.float64), B, F)


# ----------------------------------------------------------------------------- exact residual rungs (phase_c)
def exact_residual_terms(pos, Mb, F, v_2b, two_nn, S):
    """(v_near_ex, v_diag_ex, v_far_ex): the phase_c ladder at pair_cutoff 8 with the pc8 label convention."""
    import torch
    from src import nbody_features as nf
    P = len(pos)
    t_idx, s_idx, _, _ = nf.select_pair_neighbours(pos, PC, 0.0, 0)
    n = len(t_idx)
    s_vec = pos[s_idx] - pos[t_idx]
    dist = np.linalg.norm(s_vec, axis=1)
    Mts = Mb[t_idx, :, s_idx, :]
    Mst = Mb[s_idx, :, t_idx, :]
    sym = 0.5 * (Mts + np.transpose(Mst, (0, 2, 1)))
    with torch.no_grad():
        sv2 = np.concatenate([s_vec, -s_vec], 0)
        d2 = np.concatenate([dist, dist], 0)
        X2b = torch.as_tensor(np.concatenate([sv2, d2[:, None], ((d2 - nf.MEDIAN_2B_OPERATOR) ** 2)[:, None],
                                              ((d2 - nf.MEDIAN_2B_OPERATOR) ** 4)[:, None], (d2 - 2.0)[:, None]], 1),
                              dtype=torch.float32)
        K_s, K_t = two_nn.predict_mobility(X2b)
        K_s = K_s.numpy().astype(np.float64)
        K_t = K_t.numpy().astype(np.float64)
    R = sym - K_t[:n]
    v_near_ex = v_2b.copy()
    np.add.at(v_near_ex, t_idx, np.einsum("nij,nj->ni", R, F[s_idx]))
    np.add.at(v_near_ex, s_idx, np.einsum("nji,nj->ni", R, F[t_idx]))
    diag = Mb[np.arange(P), :, np.arange(P), :] - S[None]
    np.add.at(diag, t_idx, -K_s[:n])
    np.add.at(diag, s_idx, -K_s[n:])
    v_diag_ex = v_near_ex + np.einsum("nij,nj->ni", diag, F)
    from grpy_tensors import mu as grpy_mu
    Mr = grpy_mu(pos, np.ones(P), blockmatrix=True)
    Krpy = np.empty((P, P, 6, 6))
    Krpy[:, :, :3, :3] = Mr[0, 0]; Krpy[:, :, :3, 3:] = Mr[0, 1]
    Krpy[:, :, 3:, :3] = Mr[1, 0]; Krpy[:, :, 3:, 3:] = Mr[1, 1]
    D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    Rfar = Mb.transpose(0, 2, 1, 3) - Krpy
    v_far_ex = v_diag_ex + apply_blocks(Rfar, F, D > PC)
    return v_near_ex, v_diag_ex, v_far_ex, n


# ----------------------------------------------------------------------------- main
def run(args):
    import torch
    from src.mob_op_2b_combined import NNMob
    from src.sd_ops import SDMob
    from experiments.build_nbody_v2_cache import self_block

    op_rpy = pav.build_op("M_rpy")
    op_2b = NNMob(pav.SHAPE, pav.SELF_PATH, pav.TWO_BODY_PATH, nn_only=False, rpy_only=False, switch_dist=PC)
    op_pair = pav.build_op("M_mom_v2_kinf_rc8_pc8c")
    op_stack = pav.build_op("M_mom_v2_kinf_rc8_pc8c_diag")
    op_sd = SDMob(minfinity_only=True)
    two_nn = torch.jit.load(pav.TWO_BODY_PATH, map_location="cpu").eval()
    S = self_block()

    rows = []
    for phi in args.phis:
        for case in pav.cases("fig4g", Ns=[args.N], phis=[phi])[:args.n_seeds]:
            N, seed = case["N"], case["seed"]
            config, F_grav, v_grav_file = pav.load_case(case)
            _, F_rand, v_rand_file = pav.load_case(dict(case, exp="fig4"))
            pos = config[:, :3]
            M = full_M(pos, N, phi, seed, args.acc, args.backend, args.tol_v)
            Mb = M.reshape(N, 6, N, 6)

            t0 = time.time()
            A, G, Gb, Mm, Mm_off = fts_parts(pos)
            ref1, ref2, full = reflection_matrices(G, Mm, Mm_off, N)
            u, D = uncovered_mask(pos)
            ref1_unc = masked_ref1(Gb, u)
            far = D > PC
            print(f"[fts] N={N} phi={phi:g} seed={seed}: blocks + masks {time.time() - t0:.1f} s, "
                  f"near pairs {int(((D <= PC).sum() - N) // 2)}, uncovered triplets {u.sum() / u.size:.3f}", flush=True)

            for forcing, F, v_file in (("gravity", F_grav, v_grav_file), ("random", F_rand, v_rand_file)):
                if forcing not in args.forcing:
                    continue
                v_true = (M @ F.reshape(-1)).reshape(N, 6)
                truth_gap = float(np.linalg.norm(v_true - v_file) / np.linalg.norm(v_file))
                v = {}
                v["rpy"] = pav.apply_op(op_rpy, config, F)
                v_A = apply_blocks(A, F)
                layout_err = float(np.abs(v_A - v["rpy"]).max() / np.abs(v["rpy"]).max())
                assert layout_err < 1e-8, f"SD A-block vs M_rpy layout mismatch: {layout_err:.2e}"
                v["2b"] = pav.apply_op(op_2b, config, F)
                v["near_nn"] = pav.apply_op(op_pair, config, F)
                v["stack_nn"] = pav.apply_op(op_stack, config, F)
                v["near_ex"], v["diag_ex"], v["far_ex"], n_near = exact_residual_terms(pos, Mb, F, v["2b"], two_nn, S)
                for tag, B in (("ref1", ref1), ("ref2", ref2), ("full", full)):
                    dv = apply_blocks(B, F)
                    v[f"rpy+{tag}"] = v["rpy"] + dv
                    v[f"stack+{tag}"] = v["stack_nn"] + dv
                v["sd_minf"] = pav.apply_op(op_sd, config, F)
                sd_err = float(np.linalg.norm(v["sd_minf"] - v["rpy+full"]) / np.linalg.norm(v["sd_minf"]))
                assert sd_err < 1e-8, f"rpy+full != SD_Minf: {sd_err:.2e}"
                v["stack+ref1_uncov"] = v["stack_nn"] + apply_blocks(ref1_unc, F)
                v["pnear+ref1_far"] = v["diag_ex"] + apply_blocks(ref1, F, far)
                v["pnear+full_far"] = v["diag_ex"] + apply_blocks(full, F, far)

                line = []
                for rung in RUNGS:
                    st = compute_error_stats(v[rung], v_true)
                    row = dict(N=N, phi=phi, seed=seed, forcing=forcing, rung=rung, truth_gap=truth_gap,
                               n_near=n_near, sd_check=sd_err, acc=args.acc)
                    row.update({k: float(st[k]) for k in METRICS})
                    rows.append(row)
                    line.append(f"{rung}={st['prmse_lin']:.2f}/{st['prmse_fluct']:.1f}")
                print(f"  {forcing:7s} seed={seed} gap={truth_gap:.1e} sd={sd_err:.1e} | lin/fluct: " + " ".join(line),
                      flush=True)
    df = pd.DataFrame(rows)
    if OUT_CSV.exists():
        df = pd.concat([pd.read_csv(OUT_CSV), df], ignore_index=True).drop_duplicates(
            subset=["N", "phi", "seed", "forcing", "rung"], keep="last")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df


def summary(df):
    lines = ["# FTS reflection ladder (N = 200, fig4g configurations, fine BatchedMFS grand mobility)", "",
             "Mean over seeds of each metric (%), per forcing and volume fraction. Rungs are cumulative; see the",
             "docstring of `benchmarks/fts_reflection_ladder.py` for their definitions.", ""]
    g = df.groupby(["forcing", "phi", "rung"])
    means = g[METRICS].mean()
    n_seeds = g.size().groupby(level=[0, 1]).max()
    for metric in ("prmse_lin", "prmse_fluct", "err_mean_pct", "prmse_ang"):
        lines += [f"## {metric}", ""]
        for forcing in sorted(df.forcing.unique()):
            phis = sorted(df[df.forcing == forcing].phi.unique())
            lines.append(f"**{forcing}** | " + " | ".join(f"φ={p:g} (n={int(n_seeds[(forcing, p)])})" for p in phis))
            lines.append("|---|" + "---|" * len(phis))
            for rung in RUNGS:
                vals = []
                for p in phis:
                    try:
                        vals.append(f"{means.loc[(forcing, p, rung), metric]:.2f}")
                    except KeyError:
                        vals.append("—")
                lines.append(f"| {rung} | " + " | ".join(vals) + " |")
            lines.append("")
    gap = df.groupby(["forcing", "phi"])["truth_gap"].agg(["mean", "max"])
    lines += ["## truth gap ‖MF − v_file‖/‖v_file‖ (fine M vs the cached truth files)", "",
              gap.to_string(float_format=lambda x: f"{x:.2e}"), ""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phis", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--forcing", nargs="+", default=["gravity", "random"])
    ap.add_argument("--acc", default="fine")
    ap.add_argument("--backend", default="triton32")
    ap.add_argument("--tol-v", type=float, default=1e-5)
    ap.add_argument("--summary", action="store_true", help="only re-render the tables from the CSV")
    args = ap.parse_args()
    if args.summary:
        summary(pd.read_csv(OUT_CSV))
        return
    df = run(args)
    summary(df)


if __name__ == "__main__":
    main()
