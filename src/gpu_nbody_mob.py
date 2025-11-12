from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch

# Ensure relative imports work if run as a script
sys.path.append(os.path.dirname(__file__))

from src.gpu_mob_2b import NNMobTorch, TensorLike, check_against_ref, check_against_ref_gpu


class Mob_Nbody_Torch(NNMobTorch):
    """NNMob augmented with an n-body correction network, running on GPU."""

    DEFAULT_MEAN_DIST_S: float = 4.668671912939677
    DEFAULT_MAX_NEIGHBORS: int = 10
    DEFAULT_NEIGHBOR_CUTOFF: float = 6.0
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

        self.nbody_nn = torch.jit.load(nbody_nn_path, map_location=self.device).eval()
        self.max_neighbors = int(max_neighbors)
        self.neighbor_cutoff = float(neighbor_cutoff)
        self.mean_dist_s = float(mean_dist_s)

    def get_nbody_velocity(
        self,
        pos: torch.Tensor,
        force: torch.Tensor,
        viscosity: float,
        print_dim: bool = False,
    ) -> torch.Tensor:
        """Compute the learned n-body correction for each particle using PyTorch."""
        _ = viscosity  # viscous scaling is absorbed by the learned model
        N = pos.shape[0]
        dtype = pos.dtype
        
        # 1. Create all pairs (t, s)
        indices = torch.arange(N, device=self.device)
        t_idx, s_idx = torch.cartesian_prod(indices, indices).unbind(1)
        pair_mask = t_idx != s_idx
        t_idx, s_idx = t_idx[pair_mask], s_idx[pair_mask]
        if print_dim:
            print("initial idx shape:", t_idx.shape, s_idx.shape)
        
        pos_t, pos_s = pos[t_idx], pos[s_idx]
        s_vec = pos_s - pos_t
        dist_ts = torch.linalg.norm(s_vec, dim=1)
        if print_dim:
            print("pair distances shape: ", dist_ts.shape)

        # 2. Filter pairs by distance
        dist_mask = dist_ts < self.neighbor_cutoff
        if not dist_mask.any():
            return torch.zeros(N, 6, device=self.device, dtype=dtype)

        t_idx, s_idx = t_idx[dist_mask], s_idx[dist_mask]
        s_vec = s_vec[dist_mask]
        if print_dim:
            print("after distance filter idx shape:", t_idx.shape, s_idx.shape)
        
        # 3. Find neighbors for each valid pair
        midpoints = 0.5 * (pos[t_idx] + pos[s_idx])
        dist_to_midpoint = torch.linalg.norm(pos.unsqueeze(0) - midpoints.unsqueeze(1), dim=2)
        
        k_indices = torch.arange(N, device=self.device).unsqueeze(0)
        neighbor_exclude_mask = (k_indices == t_idx.unsqueeze(1)) | (k_indices == s_idx.unsqueeze(1))
        # Replace in-place mask write with torch.where for better kernel fusion
        dist_to_midpoint = torch.where(neighbor_exclude_mask, torch.inf, dist_to_midpoint)
        if print_dim:
            print("k_indices shape:", k_indices.shape)
            print("neighbor_exclude_mask shape:", neighbor_exclude_mask.shape)

        # Score is d_kt * d_ks
        d_kt = torch.linalg.norm(pos.unsqueeze(0) - pos[t_idx].unsqueeze(1), dim=2)
        d_ks = torch.linalg.norm(pos.unsqueeze(0) - pos[s_idx].unsqueeze(1), dim=2)
        score = d_kt * d_ks
        # Combine masks and apply with torch.where (avoid multiple in-place writes)
        score_mask = neighbor_exclude_mask | (dist_to_midpoint > self.neighbor_cutoff)
        score = torch.where(score_mask, torch.inf, score)

        if print_dim:
            print("score shape:", score.shape)

        # 4. Select top-K neighbors
        _, top_k_indices = torch.topk(score, self.max_neighbors, dim=1, largest=False)
        
        valid_pair_mask = torch.isfinite(score.min(dim=1).values)
        if not valid_pair_mask.any():
            return torch.zeros(N, 6, device=self.device, dtype=dtype)

        t_idx = t_idx[valid_pair_mask]
        s_idx = s_idx[valid_pair_mask]
        s_vec = s_vec[valid_pair_mask]
        top_k_indices = top_k_indices[valid_pair_mask]
        if print_dim:
            print("After select top_k_indices:", t_idx.shape, s_idx.shape)

        # 5. Build feature vector
        num_valid_pairs = top_k_indices.shape[0]
        score_valid = score[valid_pair_mask]
        gathered_scores = score_valid.gather(1, top_k_indices)
        neighbor_mask = torch.isfinite(gathered_scores)

        sentinel = torch.full_like(top_k_indices, fill_value=pos.shape[0])
        sort_keys = torch.where(neighbor_mask, top_k_indices, sentinel)
        sort_perm = torch.argsort(sort_keys, dim=1)

        top_k_indices = torch.gather(top_k_indices, 1, sort_perm)
        gathered_scores = torch.gather(gathered_scores, 1, sort_perm)
        neighbor_mask = torch.take_along_dim(neighbor_mask, sort_perm, dim=1)

        neighbor_vectors = pos[top_k_indices] - pos[t_idx].unsqueeze(1)

        if print_dim:
            print("neighbor_vectors shape:", neighbor_vectors.shape)

        # Use where instead of in-place masked zeroing for potential fusion
        neighbor_vectors = torch.where(
            neighbor_mask.unsqueeze(-1), neighbor_vectors, torch.zeros_like(neighbor_vectors)
        )

        # Positional features
        # The feature vector format needs to match the training script:
        # 1. s_vec (3)
        # 2. K neighbor vectors (max_neighbors * 3)
        idx_pos_feat_end = 3 + self.max_neighbors * 3
        
        pos_feats_padded = torch.zeros(num_valid_pairs, idx_pos_feat_end, device=self.device, dtype=dtype)
        pos_feats_padded[:, :3] = s_vec
        flat_neighbors = neighbor_vectors.reshape(num_valid_pairs, self.max_neighbors * 3)
        pos_feats_padded[:, 3:3 + self.max_neighbors * 3] = flat_neighbors

        if print_dim:
            print("pos_feats_padded shape:", pos_feats_padded.shape)

        # Pair distance scalars
        dist_raw = torch.linalg.norm(s_vec, dim=1)
        dist_centered = dist_raw - self.mean_dist_s
        dist_sq = dist_centered * dist_centered
        dist_sqsq = dist_sq * dist_sq
        dist_feats = torch.stack([dist_centered, dist_raw - 2.0, dist_sq, dist_sqsq], dim=1)
        if print_dim:
            print("dist_feats shape:", dist_feats.shape)

        # Symmetric neighbor features
        ell = dist_raw.unsqueeze(1).clamp_min(self._EPS)
        zhat = s_vec / ell
        midpoint = 0.5 * s_vec
        
        r_sk_vec = neighbor_vectors - s_vec.unsqueeze(1)
        r_sk = torch.linalg.norm(r_sk_vec, dim=2)
        r_kt = torch.linalg.norm(neighbor_vectors, dim=2)
        r_sk_c = r_sk.clamp_min(self._EPS); r_kt_c = r_kt.clamp_min(self._EPS)

        v = neighbor_vectors - midpoint.unsqueeze(1)
        u = torch.einsum('bki,bi->bk', v, zhat)
        rho = torch.linalg.norm(v - u.unsqueeze(-1) * zhat.unsqueeze(1), dim=2)
        
        a = s_vec.unsqueeze(1) - neighbor_vectors
        b = -neighbor_vectors
        num = torch.einsum('bki,bki->bk', a, b)
        den = r_sk_c * r_kt_c
        cos_k = (num / den.clamp_min(self._EPS)).clamp(-1.0, 1.0)

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
        ], dim=2) * neighbor_mask_f.unsqueeze(-1)
        sym_feats = sym_feats_stacked.reshape(num_valid_pairs, -1)
        if print_dim:
            print("sym_feats shape:", sym_feats.shape)

        # Assemble final feature tensor X
        X = torch.cat([pos_feats_padded, dist_feats, sym_feats, neighbor_mask_f], dim=1)
        if print_dim:
            print("Final feature tensor X shape:", X.shape)

        # 6. Predict velocities
        Fs = force[s_idx]
        with torch.no_grad():
            pred = self.nbody_nn.predict_velocity(X, Fs)

            if print_dim:
                print("Predicted velocities shape:", pred.shape)

        # 7. Sum velocities for each target particle
        velocities = torch.zeros(N, 6, device=self.device, dtype=dtype)
        velocities.index_add_(0, t_idx, pred)
        
        return velocities

    def apply(self, config: torch.Tensor, 
		   force: torch.Tensor, viscosity: TensorLike) -> torch.Tensor:
        """Override base apply with additional n-body correction."""
        N = config.shape[0]
        assert config.shape == (N, 7)

        v_base = super().apply(config, force, viscosity)

        # Compute and add n-body correction
        pos = config[:, :3]
        with torch.no_grad():
            v_nbody = self.get_nbody_velocity(pos, force, viscosity)
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

    ref_path = f"tmp/reference_sphere_0.1.csv"
    check_against_ref(mob_cpu, ref_path)
    check_against_ref_gpu(mob_gpu, ref_path)


def perftest():
    path = "tmp/uniform_sphere_0.1_1600.csv"
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


if __name__ == "__main__":
    accuracy_test()
    