from __future__ import annotations

import os
import sys
import time
from typing import Tuple

from benchmarks.cluster import uniform_sphere_cluster
import numpy as np
import pandas as pd
from src.triton_mfs import MobMFSTriton
import torch
import torch.profiler as profiler
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

torch.set_float32_matmul_precision('high')

"""
Lots of optimizations still possible, like making sure we use
fixed sized buffers for max neighbors, avoiding cuda graph breaks.

For now, profiling shows out of 40ms, 32ms is in two-body model.
"""

# Ensure relative imports work if run as a script
sys.path.append(os.path.dirname(__file__))

from src.gpu_mob_2b import (
    NNMobTorch, TensorLike, DEFAULT_TWO_BODY_CHUNK,
    check_against_ref, check_against_ref_gpu)
from src.model_archs import MultiBodyCorrection
from benchmarks.cluster import uniform_sphere_cluster

# Pairs per n-body NN chunk; see the sweep in Mob_Nbody_Torch.__init__.
DEFAULT_PAIR_CHUNK = 2_000_000


class Mob_Nbody_Torch(NNMobTorch):
    """NNMob augmented with an n-body correction network, running on GPU."""

    DEFAULT_MEAN_DIST_S: float = 4.690027344329476
    DEFAULT_MAX_K_NEIGHBORS: int = 10  # K, consider at most 10 neighbors for nbody effect
    DEFAULT_NEIGHBOR_CUTOFF: float = 6.0 # Line between near and far field
    MAX_PARTICLE_NEIGHBORS: int = 128  # max neighbors cached per particle
    _EPS: float = 1e-9

    def __init__(
        self,
        shape: str,
        self_nn_path: str,
        two_nn_path: str,
        nbody_nn_path: str,
        near_field_2b: str,
        far_field_2b: str,
        near_far_switch: float = DEFAULT_NEIGHBOR_CUTOFF,
        neighbor_cutoff: float = DEFAULT_NEIGHBOR_CUTOFF,
        max_k_neighbors: int = DEFAULT_MAX_K_NEIGHBORS,
        mean_dist_s: float = DEFAULT_MEAN_DIST_S,
        # 8M was chosen when the unchunked two-body path set a ~10.3 GiB floor, so
        # any value below that was free and none of it showed up in peak VRAM. With
        # that path chunked the n-body chunk becomes the binding term. Swept at
        # N=1.05M / 21.2M pairs with two_body_chunk_size fixed at 4M:
        #
        #   nb chunk | n-body ms | peak alloc
        #   8M       |  154.26   | 8.37 GiB
        #   4M       |  154.88   | 4.46 GiB
        #   2M       |  156.00   | 2.51 GiB   <- 5.86 GiB saved for 1.7 ms
        #   1M       |  158.24   | 2.40 GiB   <- 0.11 GiB more for another 2.2 ms
        #
        # 2M is where the trade stops paying: below it the curve flattens on memory
        # and keeps rising on time.
        pair_chunk_size: int = DEFAULT_PAIR_CHUNK,
        two_body_chunk_size: int = DEFAULT_TWO_BODY_CHUNK,
    ) -> None:
        super().__init__(
            shape=shape,
            self_nn_path=self_nn_path,
            two_nn_path=two_nn_path,
            near_field=near_field_2b,
            far_field=far_field_2b,
            switch_dist=near_far_switch,
            two_body_chunk_size=two_body_chunk_size,
        )
        assert shape == "sphere", "Only sphere shape currently supported for n-body operator"

        median_2b = 5.008307682776568 #copied from 2b training notebook
        assert nbody_nn_path.endswith(".wt"), "Expected .wt weights for nbody_nn_path; TorchScript .pt is not supported here"
        state_dict = torch.load(nbody_nn_path, weights_only=True)
        two_nn_keys = [k for k in state_dict.keys() if k.startswith("two_nn.")]
        for k in two_nn_keys:
            del state_dict[k]
        model = MultiBodyCorrection(114,median_2b, mean_dist_s, 33).to(self.device)
        model.load_state_dict(state_dict)
        self.nbody_nn = model.eval()
        #self.nbody_nn = torch.jit.load(nbody_nn_path, map_location=self.device).eval()

        self.max_k_neighbors = int(max_k_neighbors)
        self.neighbor_cutoff = float(neighbor_cutoff)
        self.mean_dist_s = float(mean_dist_s)
        self.pair_chunk_size = int(pair_chunk_size)

        # Build and (optionally) compile a wrapper module so we can JIT/Inductor
        # optimize the n-body correction path and reuse it inside apply().
        class _NBodyKernelModule(torch.nn.Module):
            def __init__(self, parent: 'Mob_Nbody_Torch'):
                super().__init__()
                self.parent = parent

            def forward(self, pos: torch.Tensor,
                    force: torch.Tensor, t_idx_chunk: torch.Tensor,
                    s_idx_chunk: torch.Tensor, topk_indices: torch.Tensor,
                    topk_mask: torch.Tensor) -> torch.Tensor:
                # Delegate to the Python implementation; torch.compile may insert
                # graph breaks around calls to TorchScript model, but still speeds up
                # surrounding tensor ops.
                #
                # The compiled unit is ONE chunk, not the whole chunk loop. The loop
                # bound is the pair count, which drifts every step of a dynamics run;
                # inside a compiled region that forces either a guard on its exact value
                # (recompile per step -> config.recompile_limit -> silent eager fallback,
                # ~3.5x slower) or, once the dimension is marked dynamic, a `range` over
                # a symbolic int, which dynamo cannot represent at all. Keeping the loop
                # in Python leaves one compiled graph serving every chunk size.
                return self.parent._nbody_chunk_velocity(
                    pos, force, t_idx_chunk, s_idx_chunk, topk_indices, topk_mask)
                #viscosity, print_dim=False)

        self._nbody_kernel = _NBodyKernelModule(self).to(self.device)
        self._nbody_kernel_compiled = torch.compile(
            self._nbody_kernel, mode="max-autotune", 
            backend="inductor", fullgraph=False, dynamic=True)

    @torch.no_grad()
    def _per_particle_topk(
        self,
        pos: torch.Tensor,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Each particle's K nearest neighbors, as (N,K) indices and a validity mask.

        Must be computed over the *complete* edge list, never per pair-chunk. The table
        is indexed by both endpoints of every pair in get_k_per_pair, but a chunk holds
        only a contiguous range of `t_idx` (fill_edge_indexes_kernel emits edges grouped
        by target), so a per-chunk table leaves every source outside that range unwritten.
        Those pairs then silently fall back to the target's neighbors alone, which is a
        directed truncation: (t,s) keeps t's neighbors while (s,t) keeps s's, and the
        assembled mobility stops being symmetric. See get_nbody_velocity.
        """
        device = pos.device
        dtype = pos.dtype
        index_dtype = t_idx.dtype
        num_particles = pos.shape[0]
        K = self.max_k_neighbors
        max_neighbors = self.MAX_PARTICLE_NEIGHBORS

        edge_dist = torch.linalg.norm(pos[s_idx] - pos[t_idx], dim=1)

        # One spare column absorbs the overflow from any particle with more than
        # max_neighbors neighbors; it is sliced off before the top-K. Without it those
        # writes land out of bounds. The original guard was dropped for being a
        # CPU-GPU sync -- this one costs neither a sync nor a compaction.
        neighbors = torch.full((num_particles, max_neighbors + 1), -1,
                               dtype=index_dtype, device=device)
        neighbor_dists = torch.full((num_particles, max_neighbors + 1), torch.inf,
                                    dtype=dtype, device=device)

        # t_idx, s_idx already carry both directions, so no concatenation is needed.
        # Edges arrive sorted and grouped by target, which is what makes the running
        # offset below a valid within-particle slot index.
        counts = torch.bincount(t_idx, minlength=num_particles)
        prefix = torch.cumsum(counts, dim=0) - counts
        local_pos = torch.arange(t_idx.shape[0], device=device) - prefix[t_idx]
        local_pos = torch.where(local_pos < max_neighbors, local_pos,
                                torch.full_like(local_pos, max_neighbors))

        neighbors[t_idx, local_pos] = s_idx
        neighbor_dists[t_idx, local_pos] = edge_dist

        topk_dists, topk_pos = torch.topk(neighbor_dists[:, :max_neighbors], K,
                                          dim=1, largest=False, sorted=True)
        topk_indices = neighbors[:, :max_neighbors].gather(1, topk_pos)
        topk_mask = torch.isfinite(topk_dists) & (topk_indices >= 0)
        return topk_indices, topk_mask

    @torch.no_grad()
    def get_k_per_pair(
        self,
        pos: torch.Tensor,
        pos_t: torch.Tensor,
        t_idx: torch.Tensor,
        pos_s: torch.Tensor,
        s_idx: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-K neighbors for each pair, drawn from the union of both endpoints'.

        `topk_indices` / `topk_mask` come from _per_particle_topk and cover all N
        particles, not just this chunk's.
        """
        device = pos.device
        index_dtype = t_idx.dtype
        K = self.max_k_neighbors

        if t_idx.numel() == 0:
            empty_idx = torch.zeros(t_idx.shape[0], K, dtype=index_dtype, device=device)
            empty_mask = torch.zeros(t_idx.shape[0], K, dtype=torch.bool, device=device)
            return empty_idx, empty_mask

        # Identify duplicates: mask out candidates in s that appear in t
        c_t = topk_indices[t_idx]
        c_s = topk_indices[s_idx]
        # (P, K, 1) == (P, 1, K) -> (P, K, K)
        matches = (c_t.unsqueeze(2) == c_s.unsqueeze(1))
        is_duplicate = matches.any(dim=1) # (P, K)

        candidates = torch.cat([c_t, c_s], dim=1)
        candidate_mask = torch.cat([topk_mask[t_idx], topk_mask[s_idx]], dim=1)

        # Apply duplicate mask to the second half
        duplicate_mask = torch.cat([torch.zeros_like(is_duplicate, dtype=torch.bool), is_duplicate], dim=1)
        candidate_mask = candidate_mask & ~duplicate_mask

        if candidates.numel() == 0:
            empty_idx = torch.zeros(t_idx.shape[0], K, dtype=index_dtype, device=device)
            empty_mask = torch.zeros(t_idx.shape[0], K, dtype=torch.bool, device=device)
            return empty_idx, empty_mask

        # Exclude t_idx and s_idx themselves from candidates (they are not valid neighbors for the pair)
        t_idx_exp = t_idx.unsqueeze(1)  # (num_pairs, 1)
        s_idx_exp = s_idx.unsqueeze(1)  # (num_pairs, 1)
        exclude_mask = (candidates == t_idx_exp) | (candidates == s_idx_exp)
        candidate_mask = candidate_mask & ~exclude_mask

        candidate_indices = torch.where(candidate_mask, candidates, torch.zeros_like(candidates))
        candidate_pos = pos[candidate_indices]

        pos_t_exp = pos_t.unsqueeze(1)
        pos_s_exp = pos_s.unsqueeze(1)

        d_kt = torch.linalg.norm(candidate_pos - pos_t_exp, dim=2)
        d_ks = torch.linalg.norm(candidate_pos - pos_s_exp, dim=2)
        score = d_kt * d_ks
        score = torch.where(candidate_mask, score, torch.full_like(score, torch.inf))

        topk_scores, topk_pos_pair = torch.topk(score, K, dim=1, largest=False, sorted=True)
        final_indices = candidates.gather(1, topk_pos_pair)
        final_mask = candidate_mask.gather(1, topk_pos_pair) & torch.isfinite(topk_scores)
        final_indices = torch.where(final_mask, final_indices, torch.zeros_like(final_indices))

        return final_indices, final_mask


    @torch.no_grad()
    def _nbody_chunk_velocity(
        self,
        pos: torch.Tensor,
        force: torch.Tensor,
        t_idx_chunk: torch.Tensor,
        s_idx_chunk: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_mask: torch.Tensor,
    ) -> torch.Tensor:
        """The n-body correction contributed by one chunk of pairs, as (P,6).

        Returns the per-pair prediction rather than scattering it, so the scatter (and
        with it the chunk loop) can stay outside the compiled region.
        """
        K = self.max_k_neighbors
        dtype = pos.dtype

        pos_t = pos[t_idx_chunk]
        pos_s = pos[s_idx_chunk]
        s_vec = pos_s - pos_t

        with torch.profiler.record_function("get_k_per_pair"):
            top_k_indices, neighbor_mask = self.get_k_per_pair(
                pos, pos_t, t_idx_chunk, pos_s, s_idx_chunk,
                topk_indices, topk_mask
            )

        P = top_k_indices.shape[0]
        neighbor_vectors = pos[top_k_indices] - pos_t.unsqueeze(1)  # (P,K,3)

        neighbor_vectors = torch.where(
            neighbor_mask.unsqueeze(-1), neighbor_vectors, torch.zeros_like(neighbor_vectors)
        )

        # Pair distance scalars
        dist_raw = torch.linalg.norm(s_vec, dim=1)
        dist_centered = dist_raw - self.mean_dist_s
        dist_sq = dist_centered * dist_centered
        dist_sqsq = dist_sq * dist_sq
        dist_feats = torch.stack([dist_centered, dist_raw - 2.0, dist_sq, dist_sqsq], dim=1)

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
        sym_feats = sym_feats_stacked.reshape(P, -1)

        X = torch.cat([s_vec, dist_feats, sym_feats, neighbor_mask_f], dim=1)

        Fs = force[s_idx_chunk]
        with torch.profiler.record_function("nbody_neural_net"):
            pred = self.nbody_nn.predict_velocity(X, Fs)

        return pred

    @torch.no_grad()
    def get_nbody_velocity(
        self,
        pos: torch.Tensor,     # shape (N, 3)
        force: torch.Tensor,   # shape (N, 6)
        t_idx: torch.Tensor,   # shape (num_pairs,)
        s_idx: torch.Tensor,   # shape (num_pairs,)
        topk_indices: torch.Tensor | None = None,
        topk_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the learned n-body correction for each particle using PyTorch.

        Processes pairs in chunks of self.pair_chunk_size to bound peak VRAM. Only the
        per-pair work is chunked; the per-particle neighbor table covers all N particles,
        because chunking it changes the answer (see _per_particle_topk). Chunking is
        unaffected: that table is sized by particle count, not chunk length, so it was
        never what chunking bounded.

        `apply` builds the table itself and passes it in, which keeps it out of the
        compiled region -- see _NBodyKernelModule.forward. Standalone callers can omit
        it and get it built here.
        """
        N = pos.shape[0]
        dtype = pos.dtype
        num_pairs = int(t_idx.shape[0])

        velocities = torch.zeros(N, 6, device=self.device, dtype=dtype)

        if topk_indices is None or topk_mask is None:
            topk_indices, topk_mask = self._per_particle_topk(pos, t_idx, s_idx)

        for start in range(0, num_pairs, self.pair_chunk_size):
            end = min(start + self.pair_chunk_size, num_pairs)
            t_c, s_c = t_idx[start:end], s_idx[start:end]
            # The last chunk is a different size from the full ones, and in a dynamics
            # run its size changes every step. Declaring it dynamic keeps that to one
            # compiled graph instead of one per size.
            torch._dynamo.mark_dynamic(t_c, 0)
            torch._dynamo.mark_dynamic(s_c, 0)
            pred = self._nbody_kernel_compiled(
                pos, force, t_c, s_c, topk_indices, topk_mask)
            velocities.index_add_(0, t_c, pred)

        return velocities


    @torch.no_grad()
    def apply(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        force: torch.Tensor,
        viscosity: TensorLike,
        t_idx: torch.Tensor | None = None,
        s_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Override base apply with additional n-body correction."""
        _ = orientations  # ignored for spheres
        N = positions.shape[0]
        assert positions.shape == (N, 3)

        # assert all inputs are on the correct cuda
        assert positions.is_cuda, "positions tensor must be on CUDA device"
        assert force.is_cuda, "force tensor must be on CUDA device"
        assert t_idx is None or t_idx.is_cuda
        assert s_idx is None or s_idx.is_cuda

        pos = positions.contiguous()

        # if t_idx not provided, compute neighbor pairs
        if t_idx is None or s_idx is None:
            print("Computing neighbor pairs for n-body correction...")
            t_idx, s_idx = self.get_neighbor_pairs(pos)

        torch.cuda.synchronize()
        base_start = time.perf_counter()
        v_base = super().apply(
            pos, orientations, force, viscosity,
            t_idx=t_idx, s_idx=s_idx
        )
        torch.cuda.synchronize()
        base_end = time.perf_counter()
        print(f"[Mob_Nbody] Base velocity compute time: {(base_end - base_start) * 1000:.3f} ms")


        torch.cuda.synchronize()

        if t_idx.numel() == 0:
            return v_base

        rest_start = time.perf_counter()

        # Compute and add n-body correction
        torch.cuda.synchronize()
        start = time.perf_counter()
        topk_indices, topk_mask = self._per_particle_topk(pos, t_idx, s_idx)
        v_nbody = self.get_nbody_velocity(
            pos, force, t_idx, s_idx, topk_indices, topk_mask)
        torch.cuda.synchronize()
        end = time.perf_counter()
        #print(f"Nbody kernel execution time: {(end - start)*1000:.6f} ms")

        v_total = v_base + v_nbody
        torch.cuda.synchronize()
        rest_end = time.perf_counter()
        print(f"[Mob_Nbody] Post-base path time: {(rest_end - rest_start) * 1000:.3f} ms")
        return v_total

    def apply_cpu(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        force: np.ndarray,
        viscosity: TensorLike,
    ) -> np.ndarray:
        """NumPy convenience wrapper that reuses the GPU-backed apply()."""

        positions_t = torch.as_tensor(
            np.ascontiguousarray(positions, dtype=np.float32), device=self.device
        )
        orientations_t = torch.as_tensor(
            np.ascontiguousarray(orientations, dtype=np.float32), device=self.device
        )
        force_t = torch.as_tensor(
            np.ascontiguousarray(force, dtype=np.float32), device=self.device
        )
        assert positions_t.is_cuda, f"positions tensor must be on CUDA device (got {positions_t.device})"
        velocities = self.apply(positions_t, orientations_t, force_t, viscosity)
        return velocities.detach().cpu().numpy()

import numpy as np
from src.mob_op_nbody import Mob_Op_Nbody


def accuracy_test():
    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body_wt = "data/models/combined_2body.wt"
    nbody_wt = "data/models/nbody_cross_tmp.wt"
    two_body_script = "data/models/two_body_combined_model.pt"
    nbody_script = "data/models/nbody_pinn_b1.pt"

    mob_cpu = Mob_Op_Nbody(
        shape=shape,
        self_nn_path=self_path,
        two_nn_path=two_body_script,
        nbody_nn_path=nbody_script,
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
    )

    mob_gpu = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_path,
        two_nn_path=two_body_wt,
        nbody_nn_path=nbody_wt,
        near_field_2b="nn",
        far_field_2b="rpy",
        near_far_switch=6.0,
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
    positions = df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)
    orientations = df[["q_x", "q_y", "q_z", "q_w"]].to_numpy(dtype=np.float32, copy=True)
    positions = np.ascontiguousarray(positions)
    orientations = np.ascontiguousarray(orientations)
    force = np.random.RandomState(2024).randn(positions.shape[0], 6).astype(np.float32)

    positions = torch.as_tensor(positions, dtype=torch.float32, device="cuda")
    orientations = torch.as_tensor(orientations, dtype=torch.float32, device="cuda")
    force = torch.as_tensor(force, dtype=torch.float32, device="cuda")

    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/combined_2body.wt"
    mob_gpu = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_path,
        two_nn_path=two_body,
        nbody_nn_path="data/models/nbody_cross_tmp.wt",
        near_field_2b="nn",
        far_field_2b="rpy",
        near_far_switch=6.0,
    )

    # warm-up
    dev = torch.device("cuda")
    for i in range(3):
        v = mob_gpu.apply(positions, orientations, force, viscosity=1.0)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    v = mob_gpu.apply(positions, orientations, force, viscosity=1.0)
    end.record()
    torch.cuda.synchronize()
    print(f"GPU Time: {start.elapsed_time(end)} ms")


def profile_get_nbody_velocity(
    path: str = "tmp/uniform_sphere_0.1_1600.csv",
    trace_dir: str = "tmp/profiler/nbody_kernel",
    wait_steps: int = 2,
    warmup_steps: int = 2,
    active_steps: int = 6,
) -> None:
    """Capture per-op timings for get_nbody_velocity using torch.profiler."""
    os.makedirs(trace_dir, exist_ok=True)

    df = pd.read_csv(path, float_precision="high")
    positions = df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)
    orientations = df[["q_x", "q_y", "q_z", "q_w"]].to_numpy(dtype=np.float32, copy=True)
    positions = np.ascontiguousarray(positions)
    orientations = np.ascontiguousarray(orientations)
    force = np.random.RandomState(2024).randn(positions.shape[0], 6).astype(np.float32)

    positions = torch.as_tensor(positions, dtype=torch.float32, device="cuda")
    orientations = torch.as_tensor(orientations, dtype=torch.float32, device="cuda")
    force = torch.as_tensor(force, dtype=torch.float32, device="cuda")
    pos = positions

    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/combined_2body.wt"
    mob_gpu = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_path,
        two_nn_path=two_body,
        nbody_nn_path="data/models/nbody_cross_tmp.wt",
        near_field_2b="nn",
        far_field_2b="rpy",
        near_far_switch=6.0,
    )

    t_idx, s_idx = mob_gpu.get_neighbor_pairs(pos)

    # Warm-up outside profiler to stabilize kernels
    for _ in range(3):
        mob_gpu.get_nbody_velocity(pos, force, t_idx, s_idx)
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
            mob_gpu.get_nbody_velocity(pos, force, t_idx, s_idx)
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



    
