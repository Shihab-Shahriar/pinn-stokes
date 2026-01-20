import time
from typing import Optional, Tuple, Union

import os
import numpy as np
import pandas as pd
import torch
import torch.profiler as profiler

import warp as wp
wp.init()

torch.set_grad_enabled(False)

import torch._inductor.config as config
config.freezing = True
np.set_printoptions(suppress=True, formatter={"float_kind": lambda x: f"{x:.12f}"})


from src.gpu_nbody_mob import Mob_Nbody_Torch
from src.hashgrid_neighbors import HashGridNeighborSearch

# TODO: Different streams, but syncing doesn't
# seem to impact timing. SO left as is for now.
warp_stream = wp.get_stream("cuda:0")
torch_from_warp = wp.stream_to_torch(warp_stream)

print("torch current cudaStream_t :", torch.cuda.current_stream().cuda_stream)
print("warp current cudaStream_t  :", torch_from_warp.cuda_stream)


def _get_wp_stream(device=None):
    torch_stream = torch.cuda.current_stream(device=device)
    return wp.stream_from_torch(torch_stream)


@wp.func
def rpy_far_velocity_multipole(
    rvec: wp.vec3,
    monopole: wp.vec3,
    dipole: wp.mat33,
    a: float,
    mu: float,
) -> wp.vec3:
    r2 = wp.dot(rvec, rvec)
    if r2 <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)

    r = wp.sqrt(r2)
    if r < 2.0 * a:
        return wp.vec3(0.0, 0.0, 0.0)

    inv_r = 1.0 / r
    inv_r3 = inv_r / r2
    inv_r5 = inv_r3 / r2
    inv_r7 = inv_r5 / r2

    a2 = a * a
    pref = 1.0 / (8.0 * wp.pi * mu)

    I = wp.mat33(
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    rrT = wp.outer(rvec, rvec)

    M = (
        I * inv_r
        + rrT * inv_r3
        + (2.0 * a2 / 3.0) * I * inv_r3
        - 2.0 * a2 * rrT * inv_r5
    ) * pref

    u = wp.mul(M, monopole)

    for k in range(3):
        rk = rvec[k]
        ek = wp.vec3(0.0, 0.0, 0.0)
        ek[k] = 1.0

        E = wp.outer(ek, rvec) + wp.outer(rvec, ek)

        d_termA = (-rk) * I * inv_r3
        d_termB = E * inv_r3 - (3.0 * rk) * rrT * inv_r5
        d_termC = (-2.0 * a2 * rk) * I * inv_r5
        d_termD = -2.0 * a2 * (E * inv_r5 - (5.0 * rk) * rrT * inv_r7)

        dM = (d_termA + d_termB + d_termC + d_termD) * pref
        p_k = dipole[k]
        u -= wp.mul(dM, p_k)

    return u


@wp.func
def rpy_far_velocity_pair3x3(rvec: wp.vec3, force: wp.vec3, a: float, mu: float) -> wp.vec3:
    r2 = wp.dot(rvec, rvec)
    if r2 <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)

    r = wp.sqrt(r2)
    if r < 2.0 * a:
        return wp.vec3(0.0, 0.0, 0.0)

    inv_r = 1.0 / r
    inv_r3 = inv_r / r2
    inv_r5 = inv_r3 / r2

    a2 = a * a
    pref = 1.0 / (8.0 * wp.pi * mu)

    I = wp.mat33(
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    rrT = wp.outer(rvec, rvec)

    M = (
        I * inv_r
        + rrT * inv_r3
        + (2.0 * a2 / 3.0) * I * inv_r3
        - 2.0 * a2 * rrT * inv_r5
    ) * pref

    return wp.mul(M, force)


@wp.kernel(enable_backward=False)
def compute_long_range_velocity(
    bvh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    item_forces: wp.array(dtype=wp.vec3),
    centroids: wp.array(dtype=wp.vec3),
    monopoles: wp.array(dtype=wp.vec3),
    dipoles: wp.array(dtype=wp.mat33),
    velocities: wp.array(dtype=wp.vec3d),
    theta: float,
    near_cutoff2: float,
):
    thread_id = wp.tid()
    idx = wp.bvh_primitive_id(bvh_id, thread_id) # morton code order

    pos = points[idx]
    query = wp.bvh_mp_query(bvh_id, pos, theta)

    cur_neighbor = wp.int32(-1)
    is_tree_node = wp.bool(False)

    u = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

    while wp.multipole_query_next(query, cur_neighbor, is_tree_node):
        if is_tree_node:
            center = centroids[cur_neighbor]
            monopole = monopoles[cur_neighbor]
            dipole = dipoles[cur_neighbor]
            rvec = pos - center

            vel = rpy_far_velocity_multipole(rvec, monopole, dipole, 1.0, 1.0)
            u += wp.vec3d(wp.float64(vel[0]), wp.float64(vel[1]), wp.float64(vel[2]))
        else:
            src_pos = points[cur_neighbor]
            force = item_forces[cur_neighbor]
            rvec = pos - src_pos
            if wp.dot(rvec, rvec) < near_cutoff2: # We handle near field in a different kernel
                continue
            else:
                vel = rpy_far_velocity_pair3x3(rvec, force, 1.0, 1.0)
                u += wp.vec3d(wp.float64(vel[0]), wp.float64(vel[1]), wp.float64(vel[2]))

    velocities[idx] = u



@wp.kernel(enable_backward=False)
def compute_aabb(
    points: wp.array(dtype=wp.vec3),
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
):
    idx = wp.tid()
    p = points[idx]
    lowers[idx] = p
    uppers[idx] = p



class WarpFMM:
    """
    Treecode-like FMM using a modified BVH from Warp.
    """

    def __init__(
        self,
        near_field_operator,
        theta,                  # controls far-field accuracy
        leaf_size: int = 16,    # max particles per leaf node
        near_field_cutoff: float = 6.0,
        device: str = "cuda",
        block_dim: int = 256,
    ) -> None:
        self.near_field_operator = near_field_operator
        self.leaf_size = leaf_size
        self.near_field_cutoff = near_field_cutoff
        self.theta = theta
        self.device = device
        self.block_dim = block_dim

        # Hashgrid neighbor search
        wp_stream = _get_wp_stream(device=torch.device(device))
        with wp.ScopedStream(wp_stream):
            self._wp_device = wp.get_device(device)
            max_pairs = 1_000_000 * 50  # preallocate for 50 million pairs
            self._neighbor_search = HashGridNeighborSearch(
                device=device,
                max_pairs=max_pairs,
            )

        self.fmm_buffer_allocated_ = False



    def get_edge_indexes(self, positions_t, radius):
        return self._neighbor_search.get_edge_indexes(positions_t, radius, verbose=True)


    def _gpu_near_field_pass(self, positions_t, orientations_t, forces, vis_arr, viscosity, device_index):
        """Run GPU near-field path in a worker thread for CPU/FMM overlap."""
        torch.cuda.synchronize()
        nf_start_evt = torch.cuda.Event(enable_timing=True)
        nf_end_evt = torch.cuda.Event(enable_timing=True)
        nf_start_evt.record()

        torch.cuda.set_device(device_index)
        gpu_device = torch.device(f"cuda:{device_index}")
        torch.cuda.reset_peak_memory_stats(gpu_device)

        forces_t = torch.as_tensor(forces, dtype=torch.float32, device=gpu_device)
        forces_t = forces_t.contiguous()

        cutoff = self.near_field_cutoff
        max_neighbors = int((cutoff ** 3) /2)  # Max 50% volume fraction

        rgraph_start_evt = torch.cuda.Event(enable_timing=True)
        rgraph_end_evt = torch.cuda.Event(enable_timing=True)
        nf_operator_evt_start = torch.cuda.Event(enable_timing=True)
        nf_operator_evt_end = torch.cuda.Event(enable_timing=True)


        rgraph_start_evt.record()
        wp_stream = _get_wp_stream(device=positions_t.device)
        with wp.ScopedStream(wp_stream):
            with wp.ScopedTimer("Warp::HashGrid", synchronize=True):
                t_idx, s_idx = self.get_edge_indexes(positions_t, cutoff)

        #edge_index = sort_edge_index(edge_index)
        rgraph_end_evt.record()

        # t_idx = edge_index[0]
        # s_idx = edge_index[1]

        # Assert targets are nondecreasing (grouped/sorted by target):
        assert torch.all(t_idx[1:] >= t_idx[:-1]), f"radius_graph edge_index[0] (targets) is not sorted: {t_idx[:60]}"

        N = positions_t.size(0)
        #assert is_undirected(edge_index, num_nodes=N), "Radius graph contains directed edges"

        # edge_index_cpu = edge_index.detach().cpu().numpy()
        # self.near_field_operator.near_pair_edge_index = edge_index_cpu

        assert positions_t.is_cuda, "positions tensor not on GPU"
        nf_operator_evt_start.record()

        v_near = self.near_field_operator.apply(
            positions_t,
            orientations_t,
            forces_t,
            viscosity,
            t_idx=t_idx,
            s_idx=s_idx,
        )
        nf_operator_evt_end.record()

        assert v_near.is_cuda, "v_near tensor not on GPU"

        nf_end_evt.record()

        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated(gpu_device) / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved(gpu_device) / (1024**2)
        print(f"[MobFMM] peak GPU memory: allocated {peak_alloc:.2f} MB, reserved {peak_reserved:.2f} MB")

        rgraph_elapsed = rgraph_start_evt.elapsed_time(rgraph_end_evt)
        print(f"[MobFMM] near-field construction: {rgraph_elapsed:.3f} ms")

        nf_operator_elapsed = nf_operator_evt_start.elapsed_time(nf_operator_evt_end)
        print(f"[MobFMM] near-field operator time: {nf_operator_elapsed:.3f} ms")

        nf_elapsed = nf_start_evt.elapsed_time(nf_end_evt)
        print(f"[MobFMM] overall Nearfield time: {nf_elapsed:,.3f} ms")

        particle_updates_per_sec = positions_t.shape[0] / (nf_elapsed * 1e-3)
        print(f"[MobFMM] near-field particles updates per sec: {particle_updates_per_sec:,.2f}")


        return v_near

    # FIXME: This is fluctuating a lot; need better timing strategy
    def get_far_field_vel(
        self,
        positions: torch.Tensor,
        forces: torch.Tensor,
    ) -> torch.Tensor:
        """Retuns 3D far-field velocities (linear only) using treecode-like FMM."""
        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        positions = positions.contiguous()
        forces = forces.contiguous()

        N = positions.shape[0]
        num_nodes = 2 * N - 1  # max possible nodes in binary tree

        torch_stream = torch.cuda.current_stream(device=positions.device)
        wp_stream = wp.stream_from_torch(torch_stream)
        with wp.ScopedStream(wp_stream):
            start_evt.record(stream=torch_stream)
            # This buffer caching didn't help at all. Warp must have 
            # pretty good internal memory allocator.
            if not self.fmm_buffer_allocated_ or self.fmm_lowers_.shape[0] != N: # Reallocate if N changes
                self.fmm_buffer_allocated_ = True
                self.fmm_lowers_ = wp.zeros(N, dtype=wp.vec3, device=self.device)
                self.fmm_uppers_ = wp.zeros(N, dtype=wp.vec3, device=self.device)
                self.fmm_subtree_sizes_ = wp.zeros(num_nodes, dtype=wp.int32, device=self.device)
                self.fmm_centroids_ = wp.zeros(num_nodes, dtype=wp.vec3, device=self.device)
                self.fmm_monopoles_ = wp.zeros(num_nodes, dtype=wp.vec3, device=self.device)
                self.fmm_dipoles_ = wp.zeros(num_nodes, dtype=wp.mat33, device=self.device)
                self.fmm_long_range_vel_ = wp.zeros(N, dtype=wp.vec3d, device=self.device)


            points = wp.from_torch(positions, dtype=wp.vec3)
            item_forces = wp.from_torch(forces, dtype=wp.vec3, )


            wp.launch(
                compute_aabb,
                dim=N,
                inputs=[points, self.fmm_lowers_, self.fmm_uppers_],
                device=self._wp_device,
                stream=wp_stream,
            )

            # Attempt to avoid build from scratch using rebuild or refit did not help with perf
            bvh = wp.Bvh(self.fmm_lowers_, self.fmm_uppers_, "lbvh", leaf_size=self.leaf_size)

            bvh.update_multipoles(item_forces, self.fmm_subtree_sizes_, 
                                  self.fmm_centroids_, self.fmm_monopoles_, self.fmm_dipoles_)

            near_cutoff2 = float(self.near_field_cutoff * self.near_field_cutoff)
            wp.launch(
                compute_long_range_velocity,
                dim=N,
                inputs=[
                    bvh.id,
                    points,
                    item_forces,
                    self.fmm_centroids_,
                    self.fmm_monopoles_,
                    self.fmm_dipoles_,
                    self.fmm_long_range_vel_,
                    self.theta,
                    near_cutoff2,
                ],
                outputs=[],
                device=self._wp_device,
                block_dim=self.block_dim,
                stream=wp_stream,
            )
            linear_long_range = wp.to_torch(self.fmm_long_range_vel_)
            wp.synchronize()  # ensure all warp work on this stream is complete before timing ends
            end_evt.record(stream=torch_stream)

        end_evt.synchronize()
        elapsed = start_evt.elapsed_time(end_evt)
        print(f"[MobFMM] far-field FMM GPU time: {elapsed:.3f} ms")
        assert linear_long_range.shape == (N, 3)
        return linear_long_range
    


    def apply(self, positions, orientations, forces, vis_arr):
        """Apply the FMM mobility to get velocities."""
        _ = orientations  # ignored for spheres

        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

        positions = positions.contiguous()

        force3d = forces[:, :3]
        force3d = force3d.contiguous()
        far_vel = self.get_far_field_vel(positions, force3d)

        near_vel = self._gpu_near_field_pass(
            positions,
            orientations,
            forces,
            vis_arr,
            viscosity=1.0,
            device_index=0,
        )

        total_vel = near_vel
        total_vel[:, :3] += far_vel

        end_evt.record()
        torch.cuda.synchronize()
        elapsed = start_evt.elapsed_time(end_evt)
        print(f"[MobFMM] total GPU time: {elapsed:.3f} ms")

        particle_updates_per_sec = positions.shape[0] / (elapsed * 1e-3)
        print(f"[MobFMM] total particles updates per sec: {particle_updates_per_sec:,.2f}\n\n")

        return total_vel


    def apply_cpu(self, positions, orientations, forces, viscosity):
        """Apply the FMM mobility on CPU (for testing)."""
        assert isinstance(positions, np.ndarray)
        assert isinstance(orientations, np.ndarray)
        assert isinstance(forces, np.ndarray)
        
        vis_arr = np.full((positions.shape[0],), viscosity, dtype=np.float32)

        assert positions.shape[0] == forces.shape[0] == vis_arr.shape[0]

        # Convert to torch tensors on the device
        device = torch.device(self.device)
        positions_t = torch.from_numpy(positions).to(device, dtype=torch.float32)
        orientations_t = torch.from_numpy(orientations).to(device, dtype=torch.float32)
        forces_t = torch.from_numpy(forces).to(device, dtype=torch.float32)
        vis_arr_t = torch.from_numpy(vis_arr).to(device, dtype=torch.float32)

        # Apply FMM
        res_t = self.apply(positions_t, orientations_t, forces_t, vis_arr_t)

        # Convert back to CPU
        return res_t.detach().cpu().numpy()
    




def accuracy_test(
    theta,
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


    mob_fmm = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_path,
        two_nn_path=two_body,
        nbody_nn_path="data/models/nbody_pinn_b1.pt",
        near_field_2b="nn",
        far_field_2b=None,
        near_far_switch=6.0,
    )

    warp_solver = WarpFMM(
        near_field_operator=mob_fmm,
        theta=theta,
        leaf_size=16,
        near_field_cutoff=6.0,
        device="cuda",
        block_dim=256,
    )

    baseline_mob = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_path,
        two_nn_path=two_body,
        nbody_nn_path="data/models/nbody_pinn_b1.pt",
        near_field_2b="nn",
        far_field_2b="rpy",
        near_far_switch=6.0,
    )



    config_cols = ["x", "y", "z"]
    orient_cols = ["q_x", "q_y", "q_z", "q_w"]

    results = []
    for sep in reference_separations:
        #print(f"Testing separation {sep}...\n")
        ref_path = reference_template.format(sep=sep)


        df = pd.read_csv(ref_path, float_precision="high")
        positions = df[config_cols].to_numpy(dtype=np.float32, copy=True)
        orientations = df[orient_cols].to_numpy(dtype=np.float32, copy=True)

        force = np.random.randn(positions.shape[0], 6).astype(np.float32)
        norms = np.linalg.norm(force, axis=1, keepdims=True)
        norms = norms.astype(np.float32, copy=False)
        norms[norms == 0.0] = 1.0
        force = force / norms

        # turn off torque
        force[:, 3:] = 0.0
        

        positions = np.ascontiguousarray(positions)
        orientations = np.ascontiguousarray(orientations)
        force = np.ascontiguousarray(force)

        fmm_vel = warp_solver.apply_cpu(positions, orientations, force, viscosity)
        fmm_vel = np.asarray(fmm_vel, dtype=np.float64)

        baseline_vel = baseline_mob.apply_cpu(positions, orientations, force, viscosity)
        baseline_vel = np.asarray(baseline_vel, dtype=np.float64)

        edge_index_attr = getattr(mob_fmm, "near_pair_edge_index", None)
        if edge_index_attr is None:
            neighbors_per_particle = np.zeros(positions.shape[0], dtype=int) - 1.0
            total_neighbors = -1
        else:
            assert not isinstance(edge_index_attr, torch.Tensor), "near_pair_edge_index must be on CPU"
            targets = np.asarray(edge_index_attr[0], dtype=np.int64).ravel()
            neighbors_per_particle = np.bincount(targets, minlength=positions.shape[0])
            total_neighbors = int(targets.size)


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

        rel_linear_err = np.linalg.norm(baseline_delta[:, :3]) / max(
            np.linalg.norm(baseline_vel[:, :3]), np.finfo(np.float64).eps
        )
        rel_angular_err = np.linalg.norm(baseline_delta[:, 3:]) / max(
            np.linalg.norm(baseline_vel[:, 3:]), np.finfo(np.float64).eps
        )

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
            f"avg neighbors {total_neighbors/len(positions):.2f} ", 
            f"(relative {baseline_rel_l2:.6f})\n",
            f"rel_linear_err = {rel_linear_err:.6f}, rel_angular_err = {rel_angular_err:.6f}\n"
        )


    if not results:
        #print("No reference files were found; accuracy test skipped.")
        return []
    
    return results



def perf_test(
    theta: float,
    warmup_runs: int = 3,
    timed_runs: int = 5,
    seed: int = 1234,
    profile_run: bool = False,
) -> dict:
    """Benchmark MobFMM alone and report timing statistics."""

    ref_path: str = "tmp/uniform_large_0.1_1000000.csv"
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
        near_field_2b="nn",
        far_field_2b=None,
        near_far_switch=6.0,
    )

    warp_solver = WarpFMM(
        near_field_operator=mob_fmm,
        theta=theta,
        leaf_size=16,
        near_field_cutoff=6.0,
        device="cuda",
        block_dim=256,
    )

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
    orientations = np.zeros((n_particles, 4), dtype=np.float32)
    orientations[:, 3] = 1.0  # identity quaternion per sphere
    assert n_particles > 1, "Need at least two particles for benchmarking"

    force = rng.standard_normal(size=(n_particles, 6), dtype=np.float32)
    norms = np.linalg.norm(force, axis=1, keepdims=True)
    norms = norms.astype(np.float32, copy=False)
    norms[norms == 0.0] = 1.0
    force = force / norms
    force[:, 3:] = 0.0

    positions = np.ascontiguousarray(positions)
    orientations = np.ascontiguousarray(orientations)
    force = np.ascontiguousarray(force)

    fmm_fn = lambda: warp_solver.apply_cpu(positions, orientations, force, viscosity)

    for _ in range(warmup_runs):
        fmm_fn()
    torch.cuda.synchronize()

    if profile_run:
        print("Profiling multiple runs...")
        trace_dir = "./profiler/fmm_stkfmm"
        os.makedirs(trace_dir, exist_ok=True)

        # Profile 3 active runs to get averages
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            schedule=profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
            #on_trace_ready=profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for _ in range(1 + 2 + 3):
                fmm_fn()
                torch.cuda.synchronize()
                prof.step()
        
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print("\n--- Specific Tag Stats ---")
        key_averages = prof.key_averages()
        for evt in key_averages:
            if evt.key == "nbody_nn" or evt.key == "get_k_per_pair":
                print(f"Tag: {evt.key}")
                print(f"Total CPU Time: {evt.cpu_time_total_str}") # Human readable (e.g., 5.4ms)
                print(f"Self CPU Time:  {evt.self_cpu_time_total_str}")
                print(f"Number of calls: {evt.count}")

    else:
        print("Timing multiple runs...")
        fmm_times = []
        for _ in range(timed_runs):
            fmm_times.append(_time_callable(fmm_fn))
            print()
        torch.cuda.synchronize()

        fmm_times = np.asarray(fmm_times)
        fmm_mean = float(fmm_times.mean())
        fmm_std = float(fmm_times.std(ddof=1)) if timed_runs > 1 else 0.0
        per_particle_us = fmm_mean * 1e6 / n_particles
        particle_updates_per_sec = n_particles / fmm_mean

        print(
            f"MobFMM: {fmm_mean*1e3:.3f}±{fmm_std*1e3:.3f} ms "
            f"({per_particle_us:.2f} us per particle)",
            f"{particle_updates_per_sec:.2f} particles/sec"
        )




if __name__ == "__main__":
    import sys
    if sys.argv[-1] == "acc":
        accuracy_test(theta=0.25)

    elif sys.argv[-1] == "perf":
        perf_test(theta=0.25, profile_run=False)
