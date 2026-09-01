"""Stage-level timing of the moments n-body path on the 1M two-drop cloud.

Splits the pair-moments stage into (grid build | accumulate | finish) and the
diag stage into (scatter | finish), each timed with CUDA syncs, so optimization
effort lands where the time actually is.

    bash docker/run_local.sh python benchmarks/profile_moments_stage.py [n_sub]

n_sub (optional) subsamples the cloud for quicker iterations; default full 1M.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.two_suspensions_1M import generate_suspension_drop
from src.gpu_nbody_moments import Mob_Nbody_Moments_Torch


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, out


def main():
    n_sub = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    np.random.seed(0)
    drop1 = generate_suspension_drop((0, 0, 0.0), 175.0)
    drop2 = generate_suspension_drop((0, 0, 450.0), 175.0)
    cloud = np.vstack([drop1, drop2])
    if n_sub:
        cloud = cloud[np.random.choice(len(cloud), n_sub, replace=False)]
    N = len(cloud)
    print(f"N = {N}")

    dev = torch.device("cuda")
    pos = torch.as_tensor(cloud, dtype=torch.float32, device=dev).contiguous()
    force = torch.zeros((N, 6), dtype=torch.float32, device=dev)
    force[:, 2] = -9.81

    mob = Mob_Nbody_Moments_Torch(
        shape="sphere",
        self_nn_path="data/models/self_interaction_model.pt",
        two_nn_path="data/models/combined_2body.wt",
        moments_nn_path="experiments/nbody_moments_v2_kinf_rc8_pc8.wt",
        diag_nn_path="experiments/nbody_diag_v2_pc8.wt",
        near_field_2b="nn", far_field_2b=None, switch_dist=8.0)

    ms, (t_idx, s_idx) = sync_ms(lambda: mob.get_neighbor_pairs(pos))
    print(f"near pair search: {ms:8.1f} ms  ({t_idx.numel()} ordered pairs)")

    # Warmup (compiles all kernels)
    for _ in range(2):
        mob.get_moments_velocity(pos, force, t_idx, s_idx)
        mob.get_diag_velocity(pos, force, t_idx, s_idx, 1.0)
    torch.cuda.synchronize()

    # ---- pair stage, split ------------------------------------------------
    keep = t_idx < s_idx
    t_u, s_u = t_idx[keep].contiguous(), s_idx[keep].contiguous()
    P = int(t_u.shape[0])
    ms_build, _ = sync_ms(lambda: mob._mid_search.build(pos, mob.neighbor_cutoff))

    acc_ms = fin_ms = inv_ms = mlp_ms = 0.0
    for start in range(0, P, mob.moments_pair_chunk):
        end = min(start + mob.moments_pair_chunk, P)
        t_c, s_c = t_u[start:end], s_u[start:end]
        mid = 0.5 * (pos[t_c] + pos[s_c])
        Pc = end - start
        buf = mob._get_buf("_pair_buf", Pc * 8, zero=False).view(Pc, 80)
        ms, _ = sync_ms(lambda: mob._mid_search.accumulate(
            pos, mid, t_c, s_c, mob.neighbor_cutoff, buf))
        acc_ms += ms
        from src.gpu_nbody_moments import (
            pair_invariants_kernel, pair_assemble_apply_kernel)
        buf80 = buf.view(Pc, 80)
        X76 = mob._get_flat("X76", Pc, 76)
        ms, _ = sync_ms(lambda: mob._mid_search.launch_pair_finish(
            pos, t_c, s_c, buf80, pair_invariants_kernel,
            [float(mob.mean_dist_s)], [X76]))
        inv_ms += ms
        torch._dynamo.mark_dynamic(X76, 0)
        ms, c = sync_ms(lambda: mob._coeff_k(X76).contiguous())
        mlp_ms += ms
        v_t = mob._get_flat("v_t", Pc, 6)
        v_s = mob._get_flat("v_s", Pc, 6)
        ms, _ = sync_ms(lambda: mob._mid_search.launch_pair_finish(
            pos, t_c, s_c, buf80, pair_assemble_apply_kernel,
            [c, force.contiguous()], [v_t, v_s]))
        fin_ms += ms
    print(f"pair stage ({P} unordered pairs):")
    print(f"  grid build      {ms_build:8.1f} ms")
    print(f"  warp accumulate {acc_ms:8.1f} ms")
    print(f"  warp invariants {inv_ms:8.1f} ms")
    print(f"  torch MLP       {mlp_ms:8.1f} ms")
    print(f"  warp assemble   {fin_ms:8.1f} ms")

    ms, _ = sync_ms(lambda: mob.get_moments_velocity(pos, force, t_idx, s_idx))
    print(f"  end-to-end      {ms:8.1f} ms")

    # ---- diag stage, split ------------------------------------------------
    buf = mob._get_buf("_diag_buf", N * 8)
    E = int(t_idx.shape[0])
    sc_ms = 0.0
    for e0 in range(0, E, mob.edge_chunk):
        e1 = min(e0 + mob.edge_chunk, E)
        t_c, s_c = t_idx[e0:e1], s_idx[e0:e1]
        torch._dynamo.mark_dynamic(t_c, 0)
        torch._dynamo.mark_dynamic(s_c, 0)
        ms, _ = sync_ms(lambda: mob._diag_scatter_k(pos, t_c, s_c, buf))
        sc_ms += ms
    inv_mu = torch.tensor(1.0, device=dev)
    fin2_ms = 0.0
    buf3 = buf.view(N, 8, 10)
    for p0 in range(0, N, mob.diag_row_chunk):
        p1 = min(p0 + mob.diag_row_chunk, N)
        chunk, f_chunk = buf3[p0:p1], force[p0:p1]
        torch._dynamo.mark_dynamic(chunk, 0)
        torch._dynamo.mark_dynamic(f_chunk, 0)
        ms, _ = sync_ms(lambda: mob._diag_finish_k(chunk, f_chunk, inv_mu))
        fin2_ms += ms
    print("diag stage:")
    print(f"  edge scatter    {sc_ms:8.1f} ms")
    print(f"  torch finish    {fin2_ms:8.1f} ms")
    ms, _ = sync_ms(lambda: mob.get_diag_velocity(pos, force, t_idx, s_idx, 1.0))
    print(f"  end-to-end      {ms:8.1f} ms")


if __name__ == "__main__":
    main()
