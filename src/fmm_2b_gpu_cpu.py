import math
import os
import time
from typing import Sequence, Union
import numpy as np
import pandas as pd
import torch
import torch.profiler as profiler
from torch_geometric.nn import radius_graph

np.set_printoptions(suppress=True, formatter={"float_kind": lambda x: f"{x:.12f}"})

from src.gpu_mob_2b import NNMobTorch
from src.gpu_nbody_mob import Mob_Nbody_Torch
import fmm3dpy as fmm

# torch disable all gradient computations globally
torch.set_grad_enabled(False)

class MobFMM():
    def __init__(self, shape, near_field_operator, eps, near_field_cutoff: float = 16.0):
        assert shape=="sphere", "FMM only implemented for spheres currently."
        self.near_field_operator = near_field_operator
        self.eps = eps
        self.near_field_cutoff = float(near_field_cutoff)
        assert self.near_field_cutoff > 0.0, "near-field cutoff must be positive"

    @staticmethod
    def _ensure_numpy(array, dtype):
        """Convert torch tensors or array-likes to contiguous numpy arrays."""
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        return np.ascontiguousarray(np.asarray(array), dtype=dtype)

    @staticmethod
    def _viscosity_array(viscosity, n, dtype):
        """Normalize viscosity input to an array of length n."""
        if isinstance(viscosity, torch.Tensor):
            viscosity = viscosity.detach().cpu().numpy()
        vis_arr = np.asarray(viscosity, dtype=dtype)
        if vis_arr.ndim == 0:
            vis_arr = np.full(n, float(vis_arr), dtype=dtype)
        elif vis_arr.shape[0] != n:
            raise ValueError("Viscosity must be scalar or length N")
        else:
            vis_arr = np.ascontiguousarray(vis_arr, dtype=dtype)
        if np.any(vis_arr <= 0):
            raise ValueError("Viscosity must be positive")
        return vis_arr

    @staticmethod
    def _index_array(indices):
        """Convert index-like inputs to contiguous int64 numpy arrays."""
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        return np.ascontiguousarray(np.asarray(indices), dtype=np.int64).ravel()

    def get_far_field_vel(self, pos, forces, viscosity):
        pos_np = pos #self._ensure_numpy(pos, dtype=np.float64)
        force_np = forces #self._ensure_numpy(forces, dtype=np.float64)
        n_src = pos_np.shape[0]
        assert force_np.shape[1] == 6, "Forces array must have shape (N, 6), althoguh torque will be ignored here"

        sources = pos_np.T.copy(order="F")
        stoklet = force_np[:, :3].T.copy(order="F")


        fmm_out = fmm.stfmm3d(
            eps=self.eps,
            sources=sources,
            stoklet=stoklet,
            ifppreg=1,
        )

        pot = np.asarray(fmm_out.pot, dtype=np.float64)
        if pot.ndim == 3 and pot.shape[0] == 1:
            pot = pot[0]
        vel = pot.T
        #vis_arr = self._viscosity_array(viscosity, n_src, dtype=np.float64)
        vel = vel #/ vis_arr[:, None]
        zeros_rot = np.zeros((n_src, 3), dtype=vel.dtype)
        return np.concatenate([vel, zeros_rot], axis=1)
    
    def remove_stokeslets_for_near_pairs(self, vel, pos, forces, viscosity, target_idx, source_idx):
        if target_idx is None or source_idx is None:
            raise ValueError("Target/source indices are required to remove near-field duplicates")

        tidx = target_idx.reshape(-1).long()
        sidx = source_idx.reshape(-1).long()
        #print("No of 2b interactions passed to near field:", tidx.numel())
        assert tidx.numel() != 0, "No interactions available for near-field removal"

        rel = pos[tidx] - pos[sidx]
        dist = rel.norm(dim=1)
        valid = dist > 1e-9
        assert bool(valid.any()), "No valid RPY interactions with distance > 1e-9"

        tidx = tidx[valid]
        sidx = sidx[valid]
        rel = rel[valid]
        dist = dist[valid]

        r_hat = rel / dist.unsqueeze(1)
        kernels = torch.einsum("bi,bj->bij", r_hat, r_hat)
        kernels = kernels + torch.eye(3, device=vel.device, dtype=vel.dtype).unsqueeze(0)

        vis_sel = viscosity[tidx]
        coeff = 1.0 / (8.0 * math.pi * vis_sel * dist)
        kernels = kernels * coeff.view(-1, 1, 1)
        forces_sel = forces[sidx, :3].to(dtype=vel.dtype)
        contrib = torch.einsum("bij,bj->bi", kernels, forces_sel)

        delta = torch.zeros((vel.shape[0], 3), device=vel.device, dtype=vel.dtype)
        delta.index_add_(0, tidx, contrib)
        vel_adj = vel.clone()
        vel_adj[:, :3] -= delta
        return vel_adj

    def apply(self, config, forces, viscosity=1.0):
        """Apply combined near-field and FMM far-field mobility operator.

        Parameters
        ----------
        config : numpy.ndarray
            Particle configuration array shaped (N, 7). Must be C-contiguous.
        forces : numpy.ndarray
            Force/torque array shaped (N, 6). Must be C-contiguous.
        viscosity : float or array-like
            Dynamic viscosity per particle or scalar.
        """

        assert isinstance(config, np.ndarray), "config must be a numpy array"
        assert isinstance(forces, np.ndarray), "forces must be a numpy array"
        assert config.dtype == np.float32, "config array must be of dtype float32"
        assert forces.dtype == np.float32, "forces array must be of dtype float32"

        assert isinstance(config, np.ndarray) and config.ndim == 2 and config.shape[1] == 7, (
            "config must be an (N, 7) numpy array"
        )
        assert isinstance(forces, np.ndarray) and forces.ndim == 2 and forces.shape[1] == 6, (
            "forces must be an (N, 6) numpy array"
        )
        assert config.flags["C_CONTIGUOUS"], "config array must be C-contiguous"
        assert forces.flags["C_CONTIGUOUS"], "forces array must be C-contiguous"

        n_particles = config.shape[0]
        assert n_particles > 1, "Need at least two particles for MobFMM"

        vis_arr = self._viscosity_array(viscosity, n_particles, dtype=np.float32)
        positions_np = config[:, :3]
        
        fmm_start = time.perf_counter()
        v_far_np = self.get_far_field_vel(positions_np, forces, vis_arr)
        fmm_elapsed = time.perf_counter() - fmm_start
        #print(f"[MobFMM] FMM solve: {fmm_elapsed*1e3:.3f} ms")
        

        start_gpu_overall = time.perf_counter()
        gpu_device = torch.device("cuda")

        config_t = torch.as_tensor(config, dtype=torch.float32, device=gpu_device)
        forces_t = torch.as_tensor(forces, dtype=torch.float32, device=gpu_device)
        positions_t = config_t[:, :3]

        cutoff = self.near_field_cutoff
        max_neighbors = max(1, n_particles - 1)

        edge_index = radius_graph(
            positions_t,
            r=cutoff,
            loop=False,
            max_num_neighbors=max_neighbors,
        )
        t_idx = edge_index[0]
        s_idx = edge_index[1]

        # edge_index_cpu = edge_index.detach().cpu()
        # self.near_field_operator.near_pair_edge_index = edge_index_cpu

        torch.cuda.synchronize()
        nf_start = time.perf_counter()
        assert config_t.is_cuda, "config tensor not on GPU"
        v_near = self.near_field_operator.apply(
            config_t,
            forces_t,
            viscosity,
            t_idx=t_idx,
            s_idx=s_idx,
        )
        torch.cuda.synchronize()
        nf_elapsed = time.perf_counter() - nf_start
        #print(f"[MobFMM] near-field operator apply: {nf_elapsed*1e3:.3f} ms")
        
        assert v_near.is_cuda, "v_near tensor not on GPU"

        torch.cuda.synchronize()
        remove_start = time.perf_counter()
        vis_tensor = torch.from_numpy(vis_arr).to(device=gpu_device, dtype=torch.float32)
        v_far = torch.from_numpy(v_far_np).to(device=gpu_device, dtype=torch.float32)
        v_far = self.remove_stokeslets_for_near_pairs(
            v_far,
            positions_t,
            forces_t,
            vis_tensor,
            t_idx,
            s_idx,
        )
        torch.cuda.synchronize()
        remove_elapsed = time.perf_counter() - remove_start
        #print(f"[MobFMM] near-field Stokeslet removal: {remove_elapsed*1e3:.3f} ms")

        overall_elapsed = time.perf_counter() - start_gpu_overall
        #print(f"[MobFMM] overall GPU time: {overall_elapsed*1e3:.3f} ms")
        result = v_near + v_far
        return result.detach().cpu().numpy()


def accuracy_test(
    eps: float = 1e-10,
) -> list:
    """Compare direct-sum RPY velocities against MobFMM results.

    Returns a list of dicts with absolute and relative L2 disagreements
    for each available reference configuration.
    """
    
    reference_template: str = "tmp/uniform_sphere_0.1_{sep}.csv"
    #reference_separations = ("0.1", "0.2", "0.5", "1.0", "2.0", "3.0")
    reference_separations = [800, 1600]

    viscosity = 1.0

    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/two_body_combined_model.pt"

    # mob_fmm = NNMobTorch(
    #     shape=shape,
    #     self_nn_path=self_path,
    #     two_nn_path=two_body,
    #     nn_only=False,
    #     rpy_only=False,
    #     switch_dist=6.0,
    # )

    # baseline_mob = NNMobTorch(
    #     shape=shape,
    #     self_nn_path=self_path,
    #     two_nn_path=two_body,
    #     nn_only=False,
    #     rpy_only=False,
    #     switch_dist=6.0,
    # )

    mob_fmm = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_path,
        two_nn_path=two_body,
        nbody_nn_path="data/models/nbody_pinn_b1.pt",
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
    )

    baseline_mob = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_path,
        two_nn_path=two_body,
        nbody_nn_path="data/models/nbody_pinn_b1.pt",
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
    )

    fmm_solver = MobFMM(shape=shape, near_field_operator=mob_fmm, eps=eps)

    config_cols = ["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]

    results = []
    for sep in reference_separations:
        #print(f"Testing separation {sep}...\n")
        ref_path = reference_template.format(sep=sep)
        if not os.path.exists(ref_path):
            #print(f"Skipping {ref_path}: file not found.")
            continue

        df = pd.read_csv(ref_path, float_precision="high")
        config = df[config_cols].to_numpy(dtype=np.float32, copy=True)

        # Random unit-force vectors per particle to stress-test direction-dependent behavior
        force = np.random.randn(config.shape[0], 6).astype(np.float32)
        norms = np.linalg.norm(force, axis=1, keepdims=True)
        norms = norms.astype(np.float32, copy=False)
        norms[norms == 0.0] = 1.0
        force = force / norms

        # turn off torque
        force[:, 3:] = 0.0
        

        config = np.ascontiguousarray(config)
        force = np.ascontiguousarray(force)

        baseline_vel = baseline_mob.apply_cpu(config, force, viscosity)
        if isinstance(baseline_vel, torch.Tensor):
            baseline_vel = baseline_vel.detach().cpu().numpy()
        baseline_vel = np.asarray(baseline_vel, dtype=np.float64)

        fmm_vel = fmm_solver.apply(config, force, viscosity)
        if isinstance(fmm_vel, torch.Tensor):
            fmm_vel = fmm_vel.detach().cpu().numpy()
        fmm_vel = np.asarray(fmm_vel, dtype=np.float64)



        edge_index_attr = getattr(mob_fmm, "near_pair_edge_index", None)
        if edge_index_attr is None:
            neighbors_per_particle = np.zeros(config.shape[0], dtype=int)
            total_neighbors = 0
        else:
            if isinstance(edge_index_attr, torch.Tensor):
                edge_np = edge_index_attr.detach().cpu().numpy()
            else:
                edge_np = np.asarray(edge_index_attr, dtype=np.int64)
            if edge_np.size == 0:
                neighbors_per_particle = np.zeros(config.shape[0], dtype=int)
                total_neighbors = 0
            else:
                targets = np.asarray(edge_np[0], dtype=np.int64).ravel()
                neighbors_per_particle = np.bincount(targets, minlength=config.shape[0])
                total_neighbors = int(targets.size)
        # #print(
        #     "RPY-limit neighbors: "
        #     f"total directed pairs {total_neighbors}, per-particle counts {neighbors_per_particle.tolist()}"
        # )

        # Report particle with largest disagreement between FMM and baseline
        particle_diffs = np.linalg.norm(fmm_vel - baseline_vel, axis=1)
        worst_idx = int(np.argmax(particle_diffs))
        print(
            f"Max disagreement particle {worst_idx}:\n"
            f"  baseline vel {baseline_vel[worst_idx]}\n"
            f"  fmm       vel {fmm_vel[worst_idx]}\n"
            f"neighbors: {neighbors_per_particle[worst_idx]}"
        )


        baseline_delta = fmm_vel - baseline_vel
        baseline_abs_l2 = float(np.linalg.norm(baseline_delta))
        baseline_ref_norm = float(np.linalg.norm(baseline_vel))
        baseline_rel_l2 = baseline_abs_l2 / max(baseline_ref_norm, np.finfo(np.float64).eps)

        l2_per_particle = np.linalg.norm(baseline_delta) / len(baseline_delta)

        result = {
            "separation": sep,
            "ref_path": ref_path,
            "baseline_abs_l2": baseline_abs_l2,
            "baseline_rel_l2": baseline_rel_l2,
        }
        results.append(result)
        # #print()
        print(
            f"l2_per_particle = {l2_per_particle:.6f}\n"
            f"baseline L2 disagreement = {baseline_abs_l2:.6f} ",
            f"avg neighbors {total_neighbors/len(config):.2f} ", 
            f"(relative {baseline_rel_l2:.6f})\n\n"
        )


    if not results:
        #print("No reference files were found; accuracy test skipped.")
        return []
    
    return results


def perf_compare(
    eps: float = 1e-9,
    warmup_runs: int = 3,
    timed_runs: int = 5,
    seed: int = 1234,
) -> list:
    """Benchmark MobFMM against the baseline GPU NNMobTorch operator.

    Returns a list of timing summaries per reference configuration.
    """

    ref_path: str = "tmp/uniform_sphere_0.1_1600.csv"
    rng = np.random.default_rng(seed)

    shape = "sphere"
    viscosity = 1.0

    self_model = "data/models/self_interaction_model.pt"
    two_body_model = "data/models/two_body_combined_model.pt"
    nbody_path = "data/models/nbody_pinn_b1.pt"

    mob_fmm = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_model,
        two_nn_path=two_body_model,
        nbody_nn_path= nbody_path,
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
    )

    baseline_mob = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_model,
        two_nn_path=two_body_model,
        nbody_nn_path= nbody_path,
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
    )

    fmm_solver = MobFMM(shape=shape, near_field_operator=mob_fmm, eps=eps)

    config_cols = ["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]

    def _time_callable(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return time.perf_counter() - start
    
    df = pd.read_csv(ref_path, float_precision="high")
    config = df[config_cols].to_numpy(dtype=np.float32, copy=True)
    n_particles = config.shape[0]
    assert n_particles > 1, "Need at least two particles for benchmarking"

    force = rng.standard_normal(size=(n_particles, 6), dtype=np.float32)
    norms = np.linalg.norm(force, axis=1, keepdims=True)
    norms = norms.astype(np.float32, copy=False)
    norms[norms == 0.0] = 1.0
    force = force / norms
    force[:, 3:] = 0.0 #no torque for any fmm tests

    config = np.ascontiguousarray(config)
    force = np.ascontiguousarray(force)

    baseline_fn = lambda: baseline_mob.apply_cpu(config, force, viscosity)
    fmm_fn = lambda: fmm_solver.apply(config, force, viscosity)

    baseline_times = []
    fmm_times = []

    for _ in range(warmup_runs):
        baseline_fn()
    torch.cuda.synchronize()
    for _ in range(timed_runs):
        baseline_times.append(_time_callable(baseline_fn))
    torch.cuda.synchronize()


    for _ in range(warmup_runs):
        fmm_fn()
    torch.cuda.synchronize()
    for _ in range(timed_runs):
        fmm_times.append(_time_callable(fmm_fn))
    torch.cuda.synchronize()

    baseline_times = np.asarray(baseline_times)
    fmm_times = np.asarray(fmm_times)

    baseline_mean = float(baseline_times.mean())
    baseline_std = float(baseline_times.std(ddof=1)) if timed_runs > 1 else 0.0
    fmm_mean = float(fmm_times.mean())
    fmm_std = float(fmm_times.std(ddof=1)) if timed_runs > 1 else 0.0

    speedup = baseline_mean / fmm_mean if fmm_mean > 0 else float("inf")
    per_particle_us = fmm_mean * 1e6 / n_particles

    print(
        f"Separation: baseline {baseline_mean*1e3:.3f}±{baseline_std*1e3:.3f} ms, "
        f"MobFMM {fmm_mean*1e3:.3f}±{fmm_std*1e3:.3f} ms, "
        f"speedup {speedup:.2f}x, per-particle {per_particle_us:.2f} us"
    )


def perf_test(
    eps,
    warmup_runs: int = 3,
    timed_runs: int = 5,
    seed: int = 1234,
) -> dict:
    """Benchmark MobFMM alone and report timing statistics."""

    ref_path: str = "tmp/uniform_large_0.1_5000.csv"
    rng = np.random.default_rng(seed)

    shape = "sphere"
    viscosity = 1.0

    self_model = "data/models/self_interaction_model.pt"
    two_body_model = "data/models/two_body_combined_model.pt"
    nbody_path = "data/models/nbody_pinn_b1.pt"

    mob_fmm = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_model,
        two_nn_path=two_body_model,
        nbody_nn_path=nbody_path,
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
    )
    fmm_solver = MobFMM(shape=shape, near_field_operator=mob_fmm, eps=eps)

    config_cols = ["x", "y", "z"]

    def _time_callable(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return time.perf_counter() - start

    df = pd.read_csv(ref_path, float_precision="high")
    positions = df[config_cols].to_numpy(dtype=np.float32, copy=True)
    n_particles = positions.shape[0]
    orient = np.zeros((n_particles, 4), dtype=np.float32)
    orient[:, 3] = 1.0  # identity quaternion per sphere
    config = np.concatenate((positions, orient), axis=1)
    assert n_particles > 1, "Need at least two particles for benchmarking"

    force = rng.standard_normal(size=(n_particles, 6), dtype=np.float32)
    norms = np.linalg.norm(force, axis=1, keepdims=True)
    norms = norms.astype(np.float32, copy=False)
    norms[norms == 0.0] = 1.0
    force = force / norms
    force[:, 3:] = 0.0

    config = np.ascontiguousarray(config)
    force = np.ascontiguousarray(force)

    fmm_fn = lambda: fmm_solver.apply(config, force, viscosity)

    for _ in range(warmup_runs):
        fmm_fn()
    torch.cuda.synchronize()

    fmm_times = []
    for _ in range(timed_runs):
        fmm_times.append(_time_callable(fmm_fn))
    torch.cuda.synchronize()

    fmm_times = np.asarray(fmm_times)
    fmm_mean = float(fmm_times.mean())
    fmm_std = float(fmm_times.std(ddof=1)) if timed_runs > 1 else 0.0
    per_particle_us = fmm_mean * 1e6 / n_particles

    print(
        f"MobFMM: {fmm_mean*1e3:.3f}±{fmm_std*1e3:.3f} ms "
        f"({per_particle_us:.2f} us per particle)"
    )

    return {
        "ref_path": ref_path,
        "n_particles": n_particles,
        "mean_seconds": fmm_mean,
        "std_seconds": fmm_std,
        "per_particle_us": per_particle_us,
    }



if __name__ == "__main__":
    import sys
    if sys.argv[-1] == "acc":
        accuracy_test(eps=1e-9)

    elif sys.argv[-1] == "perf":
        perf_test(eps=1e-6)

    elif sys.argv[-1] == "compare":
        perf_compare(eps=1e-6)