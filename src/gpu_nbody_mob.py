from __future__ import annotations

import os
import sys
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.profiler as profiler
from torch_geometric.nn import radius_graph

torch.set_float32_matmul_precision('high')

# Ensure relative imports work if run as a script
sys.path.append(os.path.dirname(__file__))

from src.gpu_mob_2b import NNMobTorch, TensorLike, check_against_ref, check_against_ref_gpu
from src.model_archs import MultiBodyCorrection

class Mob_Nbody_Torch(NNMobTorch):
    """NNMob augmented with an n-body correction network, running on GPU."""

    DEFAULT_MEAN_DIST_S: float = 4.690027344329476
    DEFAULT_MAX_NEIGHBORS: int = 10 # K, consider at most 10 neighbors for nbody effect
    DEFAULT_NEIGHBOR_CUTOFF: float = 6.0
    MAX_PAIR_NEIGHBORS: int = 32 # max particle pairs <6.0 cutoff
    _EPS: float = 1e-9

    def __init__(
        self,
        shape: str,
        self_nn_path: str,
        two_nn_path: str,
        nbody_nn_path: str,
        nn_only: bool = False,
        rpy_only: bool = False,
        switch_dist: float = 6.0,
        neighbor_cutoff: float = DEFAULT_NEIGHBOR_CUTOFF,
        max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
        mean_dist_s: float = DEFAULT_MEAN_DIST_S,
    ) -> None:
        super().__init__(
            shape=shape,
            self_nn_path=self_nn_path,
            two_nn_path=two_nn_path,
            nn_only=nn_only,
            rpy_only=rpy_only,
            switch_dist=switch_dist,
        )
        assert shape == "sphere", "Only sphere shape currently supported for n-body operator"

        median_2b = 5.008307682776568 #copied from 2b training notebook
        # state_dict = torch.load("experiments/nbody_cross_tmp.wt", weights_only=True)
        # two_nn_keys = [k for k in state_dict.keys() if k.startswith("two_nn.")]
        # for k in two_nn_keys:
        #     del state_dict[k]
        # model = MultiBodyCorrection(114,median_2b, mean_dist_s, 33).to(self.device)
        # model.load_state_dict(state_dict)
        # self.nbody_nn = model.eval()
        self.nbody_nn = torch.jit.load(nbody_nn_path, map_location=self.device).eval()

        self.max_neighbors = int(max_neighbors)
        self.neighbor_cutoff = float(neighbor_cutoff)
        self.mean_dist_s = float(mean_dist_s)

        # Build and (optionally) compile a wrapper module so we can JIT/Inductor
        # optimize the n-body correction path and reuse it inside apply().
        class _NBodyKernelModule(torch.nn.Module):
            def __init__(self, parent: 'Mob_Nbody_Torch'):
                super().__init__()
                self.parent = parent

            def forward(self, pos: torch.Tensor,
                        force: torch.Tensor, t_idx: torch.Tensor, 
                        s_idx: torch.Tensor, viscosity: TensorLike) -> torch.Tensor:
                # Delegate to the Python implementation; torch.compile may insert
                # graph breaks around calls to TorchScript model, but still speeds up
                # surrounding tensor ops.
                return self.parent.get_nbody_velocity(pos, force, t_idx, s_idx, viscosity, print_dim=False)

        self._nbody_kernel = _NBodyKernelModule(self).to(self.device)
        self._nbody_kernel_compiled = torch.compile(self._nbody_kernel, mode="max-autotune", backend="inductor")

    def get_close_pairs(
        self,
        pos: torch.Tensor,   # shape (N, 3
    ):
        # Build neighbor graph within cutoff using spatial indexing
        edge_index = radius_graph(
            pos,                        # (N, 3)
            r=self.neighbor_cutoff, # cutoff radius
            loop=False,                 # no self-edges
            max_num_neighbors=self.MAX_PAIR_NEIGHBORS,       # adjust as needed
        )

        # edge_index: (2, num_pairs); directed edges i -> j
        t_idx, s_idx = edge_index[0], edge_index[1]  # both (num_pairs,)
        return t_idx, s_idx


    def get_k_neighbors(self, pos: torch.Tensor, 
                        pos_t: torch.Tensor, t_idx: torch.Tensor,
                        pos_s: torch.Tensor, s_idx: torch.Tensor):
        """For each pair of particles within cutoff, get top-K neighbors for n-body effect, along with masks."""
        N = pos.shape[0]
        NP = pos_t.shape[0]

        # 2. Neighbor candidate distances relative to pair midpoint (NP, N)
        midpoints = 0.5 * (pos_t + pos_s)  # (NP,3)
        dist_to_midpoint = torch.linalg.norm(pos.unsqueeze(0) - midpoints.unsqueeze(1), dim=2)  # (NP,N)

        # Exclude self indices t and s from candidates
        k_indices = torch.arange(N, device=self.device).unsqueeze(0)  # (1,N)
        neighbor_exclude_mask = (k_indices == t_idx.unsqueeze(1)) | (k_indices == s_idx.unsqueeze(1))  # (NP,N) bool
        dist_to_midpoint = torch.where(neighbor_exclude_mask, torch.inf, dist_to_midpoint)  # (NP,N)

        # 3. Define neighbor score = d_kt * d_ks (NP, N), mask with cutoff and invalid pairs
        d_kt = torch.linalg.norm(pos.unsqueeze(0) - pos[t_idx].unsqueeze(1), dim=2)  # (NP,N)
        d_ks = torch.linalg.norm(pos.unsqueeze(0) - pos[s_idx].unsqueeze(1), dim=2)  # (NP,N)
        score = d_kt * d_ks  # (NP,N)
        score_mask = (dist_to_midpoint > self.neighbor_cutoff) | neighbor_exclude_mask
        score = torch.where(score_mask, torch.inf, score)  # (NP,N)

        # 4. Select static top-K neighbors (K << N; no special-casing)
        K = self.max_neighbors
        M = score.shape[1]
        _, top_k_indices = torch.topk(score, K, dim=1, largest=False)  # indices:(NP,K)
        gathered_scores = score.gather(1, top_k_indices)  # (NP,K)

        # neighbor_mask identifies finite (valid) neighbors among top-K
        neighbor_mask = torch.isfinite(gathered_scores)  # (NP,K) bool

        # Sort neighbors so valids come first; use a sentinel for invalid entries for sorting only
        sentinel = torch.full_like(top_k_indices, fill_value=M)  # (NP,K)
        sort_keys = torch.where(neighbor_mask, top_k_indices, sentinel)  # (NP,K)
        sort_perm = torch.argsort(sort_keys, dim=1)  # (NP,K)
        top_k_indices = torch.gather(top_k_indices, 1, sort_perm)  # (NP,K)
        gathered_scores = torch.gather(gathered_scores, 1, sort_perm)  # (NP,K)
        neighbor_mask = torch.take_along_dim(neighbor_mask, sort_perm, dim=1)  # (NP,K) bool
        return top_k_indices, neighbor_mask


    def get_nbody_velocity(
        self,
        pos: torch.Tensor,     # shape (N, 3)
        force: torch.Tensor,   # shape (N, 6)
        t_idx: torch.Tensor,   # shape (num_pairs,)
        s_idx: torch.Tensor,   # shape (num_pairs,)
        viscosity: float,      # scalar
        print_dim: bool = False,
    ) -> torch.Tensor:
        """Compute the learned n-body correction for each particle using PyTorch."""
        # Viscous scaling is absorbed by the learned model.
        _ = viscosity

        N = pos.shape[0]
        K = self.max_neighbors

        dtype = pos.dtype

        # Gather positions for each pair
        pos_t = pos[t_idx]                  # (num_pairs, 3)
        pos_s = pos[s_idx]                  # (num_pairs, 3)
        s_vec = pos_s - pos_t               # (num_pairs, 3)

        top_k_indices, neighbor_mask = self.get_k_neighbors(
            pos, pos_t, t_idx, pos_s, s_idx
        )

        # 5. Build feature vector with fixed shapes
        P = top_k_indices.shape[0]
        neighbor_vectors = pos[top_k_indices] - pos[t_idx].unsqueeze(1)  # (NP,K,3)

        if print_dim:
            print("neighbor_vectors shape:", neighbor_vectors.shape)

        # Zero out invalid neighbors and invalid pairs
        neighbor_vectors = torch.where(
            neighbor_mask.unsqueeze(-1), neighbor_vectors, torch.zeros_like(neighbor_vectors)
        )  # (NP,K,3)

        # Positional features (NP, 3 + 3K)
        idx_pos_feat_end = 3 + K * 3
        pos_feats_padded = torch.zeros(P, idx_pos_feat_end, device=self.device, dtype=dtype)  # (NP,3+3K)
        pos_feats_padded[:, :3] = s_vec  # (NP,3)

        flat_neighbors = neighbor_vectors.reshape(P, K * 3)  # (NP,3K)
        pos_feats_padded[:, 3:3 + K * 3] = flat_neighbors  # (NP,3K)

        if print_dim:
            print("pos_feats_padded shape:", pos_feats_padded.shape)

        # Pair distance scalars (NP, 4)
        dist_raw = torch.linalg.norm(s_vec, dim=1)  # (NP,)
        dist_centered = dist_raw - self.mean_dist_s  # (NP,)
        dist_sq = dist_centered * dist_centered  # (NP,)
        dist_sqsq = dist_sq * dist_sq  # (NP,)
        dist_feats = torch.stack([dist_centered, dist_raw - 2.0, dist_sq, dist_sqsq], dim=1)  # (NP,4)
        if print_dim:
            print("dist_feats shape:", dist_feats.shape)

        # Symmetric neighbor features
        ell = dist_raw.unsqueeze(1).clamp_min(self._EPS)  # (NP,1)
        zhat = s_vec / ell  # (NP,3)
        midpoint = 0.5 * s_vec  # (NP,3)

        r_sk_vec = neighbor_vectors - s_vec.unsqueeze(1)  # (NP,K,3)
        r_sk = torch.linalg.norm(r_sk_vec, dim=2)  # (NP,K)
        r_kt = torch.linalg.norm(neighbor_vectors, dim=2)  # (NP,K)
        r_sk_c = r_sk.clamp_min(self._EPS); r_kt_c = r_kt.clamp_min(self._EPS)  # (NP,K) each

        v = neighbor_vectors - midpoint.unsqueeze(1)  # (NP,K,3)
        u = torch.einsum('bki,bi->bk', v, zhat)  # (NP,K)
        rho = torch.linalg.norm(v - u.unsqueeze(-1) * zhat.unsqueeze(1), dim=2)  # (NP,K)

        a = s_vec.unsqueeze(1) - neighbor_vectors  # (NP,K,3)
        b = -neighbor_vectors  # (NP,K,3)
        num = torch.einsum('bki,bki->bk', a, b)  # (NP,K)
        den = r_sk_c * r_kt_c  # (NP,K)
        cos_k = (num / den.clamp_min(self._EPS)).clamp(-1.0, 1.0)  # (NP,K)

        neighbor_mask_f = neighbor_mask.to(dtype) 
        sym_feats_stacked = torch.stack([
            r_sk + r_kt,
            torch.abs(r_sk - r_kt),
            r_sk * r_kt,
            torch.abs(u) / ell,
            (u / ell) ** 2,
            rho / ell,
            (1.0 / r_sk_c) + (1.0 / r_kt_c),
            1.0 / (r_sk_c * r_kt_c),
            torch.abs((1.0 / r_sk_c) - (1.0 / r_kt_c)),
            cos_k,
        ], dim=2) * neighbor_mask_f.unsqueeze(-1)  # (NP,K,10)
        sym_feats = sym_feats_stacked.reshape(P, -1)  # (NP,10K)
        if print_dim:
            print("sym_feats shape:", sym_feats.shape)

        # Assemble final feature tensor X with fixed width
        X = torch.cat([pos_feats_padded, dist_feats, sym_feats, neighbor_mask_f], dim=1)  # (NP,7+14K)
        if print_dim:
            print("Final feature tensor X shape:", X.shape)

        # 6. Predict velocities for all pairs, zeroing invalid pairs
        Fs = force[s_idx]  # (NP,6)
        with torch.no_grad():
            pred = self.nbody_nn.predict_velocity(X, Fs)  # (NP,6)

            if print_dim:
                print("Predicted velocities shape:", pred.shape)

        # 7. Sum velocities for each target particle (fixed size N x 6)
        velocities = torch.zeros(N, 6, device=self.device, dtype=dtype)  # (N,6)
        velocities.index_add_(0, t_idx, pred)  # (N,6)

        return velocities  # (N,6)

    def apply(self, config: torch.Tensor, 
           force: torch.Tensor, viscosity: TensorLike) -> torch.Tensor:
        """Override base apply with additional n-body correction."""
        N = config.shape[0]
        assert config.shape == (N, 7)

        v_base = super().apply(config, force, viscosity)

        pos = config[:, :3]
        t_idx, s_idx = self.get_close_pairs(pos)

        print("nbody: t_idx shape:", t_idx.shape, "s_idx shape:", s_idx.shape)

        if t_idx.numel() == 0:
            print("No valid pairs within neighbor cutoff; returning base velocities.")
            print("Press any key to continue...")
            input()
            return v_base

        # Compute and add n-body correction
        with torch.no_grad():
            # Use compiled kernel if available
            torch.cuda.synchronize()
            start = time.perf_counter()
            v_nbody = self._nbody_kernel(pos, force, t_idx, s_idx, viscosity)
            torch.cuda.synchronize()
            end = time.perf_counter()
            print(f"Nbody kernel execution time: {(end - start)*1000:.6f} ms")

            v_total = v_base + v_nbody
            return v_total

import numpy as np
from src.mob_op_nbody import Mob_Op_Nbody


def accuracy_test():
    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/two_body_combined_model.pt"

    mob_cpu = Mob_Op_Nbody(
        shape=shape,
        self_nn_path=self_path,
        two_nn_path=two_body,
        nbody_nn_path="data/models/nbody_pinn_b1.pt",
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
    )

    mob_gpu = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_path,
        two_nn_path=two_body,
        nbody_nn_path="data/models/nbody_pinn_b1.pt",
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
    )

    for d in ["0.1", "0.2", "0.5", "1.0", "2.0", "3.0"]:
        print(f"\n=== Separation {d} ===")

        ref_path = f"tmp/reference_sphere_{d}.csv"

        print("\n=== N-body CPU accuracy ===")
        check_against_ref(mob_cpu, ref_path)
        print("-----")

        print("\n=== N-body GPU accuracy recovered ===")
        check_against_ref_gpu(mob_gpu, ref_path)
        print("-----")



def perftest():
    path = "tmp/uniform_sphere_0.1_800.csv"
    df = pd.read_csv(path, float_precision="high")
    expected_cols = ["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]
    config = df[expected_cols].to_numpy(dtype=np.float32, copy=True)
    config = np.ascontiguousarray(config)
    force = np.random.RandomState(2024).randn(config.shape[0], 6).astype(np.float32)

    config = torch.as_tensor(config, dtype=torch.float32, device="cuda")
    force = torch.as_tensor(force, dtype=torch.float32, device="cuda")

    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/two_body_combined_model.pt"
    mob_gpu = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_path,
        two_nn_path=two_body,
        nbody_nn_path="data/models/nbody_pinn_b1.pt",
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
    )

    # warm-up
    dev = torch.device("cuda")
    for i in range(3):
        v = mob_gpu.apply(config, force, viscosity=1.0)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    v = mob_gpu.apply(config, force, viscosity=1.0)
    end.record()
    torch.cuda.synchronize()
    print(f"GPU Time: {start.elapsed_time(end)} ms")


def profile_get_nbody_velocity(
    path: str = "tmp/uniform_sphere_0.1_400.csv",
    trace_dir: str = "tmp/profiler/nbody_kernel",
    wait_steps: int = 2,
    warmup_steps: int = 2,
    active_steps: int = 6,
) -> None:
    """Capture per-op timings for get_nbody_velocity using torch.profiler."""
    os.makedirs(trace_dir, exist_ok=True)

    df = pd.read_csv(path, float_precision="high")
    expected_cols = ["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]
    config = df[expected_cols].to_numpy(dtype=np.float32, copy=True)
    config = np.ascontiguousarray(config)
    force = np.random.RandomState(2024).randn(config.shape[0], 6).astype(np.float32)

    config = torch.as_tensor(config, dtype=torch.float32, device="cuda")
    force = torch.as_tensor(force, dtype=torch.float32, device="cuda")
    pos = config[:, :3]

    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/two_body_combined_model.pt"
    mob_gpu = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_path,
        two_nn_path=two_body,
        nbody_nn_path="data/models/nbody_pinn_b1.pt",
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
    )

    # Warm-up outside profiler to stabilize kernels
    for _ in range(3):
        mob_gpu.get_nbody_velocity(pos, force, viscosity=1.0)
    torch.cuda.synchronize()

    activities = [profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(profiler.ProfilerActivity.CUDA)

    schedule = profiler.schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=1)
    total_steps = wait_steps + warmup_steps + active_steps

    trace_handler = profiler.tensorboard_trace_handler(trace_dir, use_gzip=True)
    with profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=trace_handler,
    ) as prof:
        for _ in range(total_steps):
            mob_gpu.get_nbody_velocity(pos, force, viscosity=1.0)
            torch.cuda.synchronize()
            prof.step()

    sort_key = "cuda_time_total" 
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by=sort_key,
            row_limit=60,
        )
    )


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        profile_get_nbody_velocity()
    else:
        #perftest()
        accuracy_test()