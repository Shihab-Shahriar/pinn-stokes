"""Parity check: GPU moments operator (src/gpu_nbody_moments.py) vs the CPU
reference (src/mob_op_nbody_moments.py) with the published pc8 models.

Runs inside the docker image (needs warp + CUDA):

    bash docker/run_local.sh python benchmarks/compare_gpu_moments.py            # compiled
    TORCH_COMPILE_DISABLE=1 bash docker/run_local.sh python benchmarks/compare_gpu_moments.py

Checks, per random RSA configuration:
  0. the .wt weights the GPU loads == the published TorchScript .pt (pair + diag);
  1. pair moments correction alone (CPU get_nbody_velocity vs GPU get_moments_velocity);
  2. diagonal correction alone;
  3. full apply (CPU: NN near + RPY far, dense; GPU: near_field="nn", far_field="rpy").
Expected disagreement is fp32-vs-fp64 feature/accumulation noise, ~1e-5 relative."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# grpy_tensors (RPY) is imported bare by src/mob_op_2b_combined.py
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.gpu_nbody_moments import Mob_Nbody_Moments_Torch
from src.mob_op_nbody_moments import Mob_Op_Nbody_Moments

SELF_PT = "data/models/self_interaction_model.pt"
TWO_PT = "data/models/two_body_combined_model.pt"
TWO_WT = "data/models/combined_2body.wt"
PAIR_PT = "data/models/nbody_moments_v2_kinf_rc8_pc8.pt"
PAIR_WT = "experiments/nbody_moments_v2_kinf_rc8_pc8.wt"
DIAG_PT = "data/models/nbody_diag_v2_pc8.pt"
DIAG_WT = "experiments/nbody_diag_v2_pc8.wt"


def rsa_box(n: int, phi: float, seed: int) -> np.ndarray:
    """Random sequential adsorption of unit spheres, gap >= 0.1 (the v2 uniform family)."""
    rng = np.random.default_rng(seed)
    L = (n * (4.0 / 3.0) * np.pi / phi) ** (1.0 / 3.0)
    pts = np.empty((0, 3))
    tries = 0
    while len(pts) < n:
        cand = rng.uniform(1.0, L - 1.0, size=3)
        if len(pts) == 0 or np.min(np.linalg.norm(pts - cand, axis=1)) >= 2.1:
            pts = np.vstack([pts, cand])
        tries += 1
        assert tries < 500_000, "RSA failed; lower phi"
    return pts


def rel(a, b, axis_slice=slice(None)):
    d = a[:, axis_slice] - b[:, axis_slice]
    return np.linalg.norm(d) / max(np.linalg.norm(b[:, axis_slice]), 1e-300)


def report(tag, gpu, cpu):
    print(f"  {tag:28s} rel_total {rel(gpu, cpu):.3e}  "
          f"rel_lin {rel(gpu, cpu, slice(0, 3)):.3e}  "
          f"rel_ang {rel(gpu, cpu, slice(3, 6)):.3e}  "
          f"max_abs {np.abs(gpu - cpu).max():.3e}")
    return rel(gpu, cpu)


def check_wt_vs_pt(mob_gpu):
    """The .wt-loaded modules must reproduce the published TorchScript models."""
    from src import nbody_moments as nbm
    dev = mob_gpu.device
    rng = np.random.default_rng(0)
    # realistic pair rows via the feature builder
    P, K = 256, 12
    s_vec = rng.normal(size=(P, 3)); s_vec *= (rng.uniform(2.2, 7.8, P) / np.linalg.norm(s_vec, axis=1))[:, None]
    nbr = rng.normal(size=(P, K, 3)) * 3.0 + 0.5 * s_vec[:, None, :]
    mask = (rng.uniform(size=(P, K)) < 0.7).astype(np.float64)
    X = nbm.moment_features(torch.as_tensor(s_vec), torch.as_tensor(nbr),
                            torch.as_tensor(mask), mob_gpu.mean_dist_s).float()
    pt = torch.jit.load(PAIR_PT, map_location="cpu").eval()
    with torch.no_grad():
        K_pt = pt.predict_mobility(X)
        K_wt = mob_gpu.moments_nn.predict_mobility(X.to(dev)).cpu()
    err = (K_wt - K_pt).abs().max().item()
    print(f"  pair .wt vs published .pt   max_abs {err:.3e}")
    assert err < 1e-4, "pair .wt does not match the published .pt"

    Xs = nbm.self_moment_features(torch.as_tensor(nbr + 2.5), torch.as_tensor(mask)).float()
    pt_d = torch.jit.load(DIAG_PT, map_location="cpu").eval()
    with torch.no_grad():
        Kd_pt = pt_d.predict_mobility(Xs)
        Kd_wt = mob_gpu.diag_nn.predict_mobility(Xs.to(dev)).cpu()
    err = (Kd_wt - Kd_pt).abs().max().item()
    print(f"  diag .wt vs published .pt   max_abs {err:.3e}")
    assert err < 1e-4, "diag .wt does not match the published .pt"


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")

    mob_gpu = Mob_Nbody_Moments_Torch(
        shape="sphere", self_nn_path=SELF_PT, two_nn_path=TWO_WT,
        moments_nn_path=PAIR_WT, diag_nn_path=DIAG_WT,
        near_field_2b="nn", far_field_2b="rpy", switch_dist=8.0,
        moments_backend="torch")
    # Second instance on the fused warp accumulation path; shares nothing with
    # the first, so both backends are pinned against the CPU reference.
    mob_gpu_warp = Mob_Nbody_Moments_Torch(
        shape="sphere", self_nn_path=SELF_PT, two_nn_path=TWO_WT,
        moments_nn_path=PAIR_WT, diag_nn_path=DIAG_WT,
        near_field_2b="nn", far_field_2b="rpy", switch_dist=8.0,
        moments_backend="warp", moments_mlp_fp16=False)
    # Production configuration: fp16 MLP. Checked against the CPU reference with
    # its own budget (half-precision coefficients, ~1e-3 of the correction).
    mob_gpu_fp16 = Mob_Nbody_Moments_Torch(
        shape="sphere", self_nn_path=SELF_PT, two_nn_path=TWO_WT,
        moments_nn_path=PAIR_WT, diag_nn_path=DIAG_WT,
        near_field_2b="nn", far_field_2b="rpy", switch_dist=8.0,
        moments_backend="warp", moments_mlp_fp16=True)

    mob_cpu = Mob_Op_Nbody_Moments(
        shape="sphere", self_nn_path=SELF_PT, two_nn_path=TWO_PT,
        nbody_nn_path=PAIR_PT, nn_only=False, rpy_only=False,
        switch_dist=8.0, pair_cutoff=8.0, neighbor_cutoff=8.0,
        max_neighbors=None, diag_nn_path=DIAG_PT, diag_cutoff=8.0)

    check_wt_vs_pt(mob_gpu)

    worst = 0.0
    for n, phi, seed, visc in ((300, 0.10, 1, 1.0), (300, 0.20, 2, 1.0), (200, 0.15, 3, 1.5)):
        pos = rsa_box(n, phi, seed)
        rng = np.random.default_rng(100 + seed)
        force = rng.standard_normal((n, 6))
        # NNMob.apply reads quaternions scalar-LAST (x, y, z, w); identity = (0,0,0,1)
        config = np.concatenate([pos, np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))], axis=1)

        pos_t = torch.as_tensor(pos, dtype=torch.float32, device=dev).contiguous()
        force_t = torch.as_tensor(force, dtype=torch.float32, device=dev).contiguous()
        orient_t = torch.zeros((n, 4), dtype=torch.float32, device=dev)
        orient_t[:, 3] = 1.0
        t_idx, s_idx = mob_gpu.get_neighbor_pairs(pos_t)

        print(f"\n=== N={n} phi={phi} mu={visc} (near pairs: {t_idx.numel()}) ===")

        v_pair_gpu = mob_gpu.get_moments_velocity(pos_t, force_t, t_idx, s_idx).cpu().numpy()
        v_pair_cpu = mob_cpu.get_nbody_velocity(pos, force, visc)
        worst = max(worst, report("pair moments (torch)", v_pair_gpu, v_pair_cpu))

        v_pair_warp = mob_gpu_warp.get_moments_velocity(pos_t, force_t, t_idx, s_idx).cpu().numpy()
        worst = max(worst, report("pair moments (warp fused)", v_pair_warp, v_pair_cpu))

        v_pair_h = mob_gpu_fp16.get_moments_velocity(pos_t, force_t, t_idx, s_idx).cpu().numpy()
        fp16_rel = report("pair moments (fp16 MLP)", v_pair_h, v_pair_cpu)
        assert fp16_rel < 1e-2, "fp16 MLP drifted: %.3e" % fp16_rel

        v_diag_gpu = mob_gpu.get_diag_velocity(pos_t, force_t, t_idx, s_idx, visc).cpu().numpy()
        v_diag_cpu = mob_cpu.get_diag_velocity(pos, force, visc)
        worst = max(worst, report("diag correction", v_diag_gpu, v_diag_cpu))

        # Full apply is compared at mu = 1 only: the CPU reference never scales its
        # RPY far field by viscosity (grpy_tensors.mu has no viscosity argument and
        # NNMob.get_two_vel applies the blocks raw), while the GPU path correctly
        # multiplies by 1/mu -- a latent CPU quirk, invisible at the mu = 1 every
        # benchmark/dataset uses. Component checks above still run at mu != 1.
        if visc == 1.0:
            v_gpu = mob_gpu.apply(pos_t, orient_t, force_t, visc, t_idx=t_idx, s_idx=s_idx).cpu().numpy()
            v_cpu = mob_cpu.apply(config, force, visc)
            worst = max(worst, report("full apply (near+far)", v_gpu, v_cpu))

    print(f"\nworst relative disagreement: {worst:.3e}")
    assert worst < 5e-4, "GPU port disagrees with the CPU reference"
    print("PARITY OK")


if __name__ == "__main__":
    main()
