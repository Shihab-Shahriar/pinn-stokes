"""GPU-accelerated mobility operator for sphere suspensions.

This module implements a pared-down variant of the CPU-oriented
`NNMob` class from `mob_op_2b_combined.py`.  Leveraging sphere-only
assumptions lets us replace quaternion-based rotations with closed-form
expressions and the analytic self-mobility of a unit-radius sphere.
The focus is squarely on fast batched velocity evaluation on the GPU,
so diagnostics (SPD checks, matrix assembly, etc.) are omitted.
"""

from __future__ import annotations

import math
import os
import time
from typing import Union
import numpy as np
import pandas as pd
import torch
import torch.profiler as profiler
from benchmarks.bench_rpy import _two_body_mu_batch, two_body_rpy_batch
from src.model_archs import TwoBodyCombined



TensorLike = Union[torch.Tensor, float]


def _ensure_device(device: Union[str, torch.device, None]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _as_float_tensor(data, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(data, dtype=torch.float32, device=device)


class PairVelKernel(torch.nn.Module):
    """Fused pairwise NN contribution with feature construction.

    This wraps the previous _two_body_velocity body so it can be compiled
    as a single graph and called from the main apply() path.
    """

    def __init__(self, model: torch.nn.Module, median: torch.Tensor, contact_distance: torch.Tensor):
        super().__init__()
        self.model = model
        # register buffers so device/dtype moves are handled automatically
        self.register_buffer("median", median.detach().clone())
        self.register_buffer("contact_distance", contact_distance.detach().clone())

    def forward(
        self,
        rel_vecs: torch.Tensor,
        target_indices: torch.Tensor,
        source_indices: torch.Tensor,
        force: torch.Tensor,
        viscosity: TensorLike,
    ) -> torch.Tensor:
        if target_indices.numel() == 0:
            return torch.zeros_like(force)

        rel_selected = rel_vecs[target_indices, source_indices]
        dist = torch.linalg.norm(rel_selected, dim=-1, keepdim=True).clamp_min(1e-8)

        r_shift = dist - self.median
        r_sq = r_shift * r_shift
        r_quad = r_sq * r_sq
        min_dist = dist - self.contact_distance
        features = torch.cat([rel_selected, dist, r_sq, r_quad, min_dist], dim=-1)

        force_target = force[target_indices]
        force_source = force[source_indices]

        mu_tensor = torch.as_tensor(
            viscosity, dtype=force.dtype, device=force.device
        )

        with torch.no_grad():
            vel = self.model.predict_velocity(features, force_target, force_source, mu_tensor)

        contribution = torch.zeros_like(force)
        contribution.index_add_(0, target_indices, vel)
        return contribution


class NNMobTorch:
    """GPU-ready mobility operator using TorchScript surrogates."""

    def __init__(
        self,
        shape: str,
        self_nn_path: str,
        two_nn_path: str,
        nn_only: bool = False,
        rpy_only: bool = False,
        switch_dist: float = 6.0,
        rpy_tile_size: int = 512,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        if nn_only and rpy_only:
            raise ValueError("`nn_only` and `rpy_only` are mutually exclusive")

        if shape != "sphere":
            raise ValueError("gpu_mob_2b.NNMob currently supports spheres only")

        self.shape = shape
        self.nn_only = nn_only
        self.rpy_only = rpy_only
        self.switch_dist = switch_dist
        self.rpy_tile_size = int(rpy_tile_size)
        self.device = _ensure_device(device)

        self.contact_distance = torch.tensor(2.0, dtype=torch.float32, device=self.device)
        self.median = torch.tensor(5.01, dtype=torch.float32, device=self.device)
        
        state_dict = torch.load("experiments/combined_2body.wt", weights_only=True)
        model = TwoBodyCombined(input_dim=4)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(self.device)
        # Keep base model compiled for its internal ops
        try:
            model = torch.compile(model, mode="max-autotune", backend="inductor")
        except Exception:
            pass
        
        self.two_nn = model
        # Build a fused, compiled kernel for pairwise NN velocity contribution
        self._pair_kernel = PairVelKernel(self.two_nn, self.median, self.contact_distance).to(self.device)
        self._pair_kernel_compiled = torch.compile(
            self._pair_kernel, mode="max-autotune", backend="inductor"
        )
        self.self_nn_path = self_nn_path  # kept for API compatibility; not used.

        # Optionally compile the RPY velocity computation. We wrap the existing
        # method logic in a nn.Module so torch.compile can capture the graph.
        class _RPYKernelModule(torch.nn.Module):
            def __init__(self, parent: 'NNMobTorch'):
                super().__init__()
                self.parent = parent

            def forward(self, rel_vecs: torch.Tensor, src_wrench: torch.Tensor, viscosity: TensorLike) -> torch.Tensor:
                return self.parent._rpy_velocity(rel_vecs, src_wrench, viscosity)

        self._rpy_kernel = _RPYKernelModule(self).to(self.device)
        self._rpy_velocity_compiled = torch.compile(self._rpy_kernel, mode="max-autotune", backend="inductor")


        ## Resure nearest neighbor indices and positions in nbody. TODO: Preallocate buffers
        self.t_idx_ = torch.empty(0, dtype=torch.long, device=self.device)
        self.s_idx_ = torch.empty(0, dtype=torch.long, device=self.device)
        # self.pos_t_ = torch.empty(0, dtype=torch.float32, device=self.device)
        # self.pos_s_ = torch.empty(0, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Self interaction (analytic for unit-radius spheres)
    # ------------------------------------------------------------------
    def _self_velocity(self, force: torch.Tensor, viscosity: TensorLike) -> torch.Tensor:
        torch.cuda.synchronize()
        start = time.perf_counter()
        mu_tensor = torch.as_tensor(viscosity, dtype=torch.float32, device=self.device)
        if mu_tensor.ndim == 0 or mu_tensor.numel() == 1:
            mu_tensor = mu_tensor.repeat(force.shape[0])
        elif mu_tensor.shape[0] != force.shape[0]:
            raise ValueError("Viscosity tensor must be scalar or batch-aligned")

        mu_tensor = mu_tensor.to(force.dtype)

        inv_6pi_mu = (1.0 / (6.0 * math.pi)) / mu_tensor
        inv_8pi_mu = (1.0 / (8.0 * math.pi)) / mu_tensor

        u = inv_6pi_mu.unsqueeze(-1) * force[:, :3]
        omega = inv_8pi_mu.unsqueeze(-1) * force[:, 3:]
        res = torch.cat([u, omega], dim=-1)
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"Self velocity computation time: {(end - start)*1000:.6f} ms")
        return res


    def _rpy_velocity(
        self,
        rel_vecs: torch.Tensor,
        Ft: torch.Tensor,
        viscosity: TensorLike,
    ) -> torch.Tensor:
        """Far-field Rotne-Prager-Yamakawa mobility via validated reference."""
        inv_mu = torch.as_tensor(1 / viscosity, dtype=torch.float32, device=rel_vecs.device)


        dtype = rel_vecs.dtype
        device = rel_vecs.device
        assert Ft.device == rel_vecs.device
        assert Ft.dtype == rel_vecs.dtype
        
        batch_size = rel_vecs.shape[0]
        radii = torch.ones((batch_size, 2), dtype=dtype, device=device)

        mobility_batch = two_body_rpy_batch(rel_vecs, radii)
                
        # origin = torch.zeros((batch_size, 1, 3), dtype=dtype, device=device)
        # rel_vecs_reshaped = rel_vecs.unsqueeze(1)
        # centres = torch.cat((origin, rel_vecs_reshaped), dim=1)
        # mobility_batch = _two_body_mu_batch(centres, radii)


        target_idx = 0
        source_idx = 1
        offset_rot = 3 * 2  # 3 * n

        trans_rows = slice(3 * target_idx, 3 * (target_idx + 1))
        rot_rows = slice(offset_rot + 3 * target_idx, offset_rot + 3 * (target_idx + 1))
        force_cols = slice(3 * source_idx, 3 * (source_idx + 1))
        torque_cols = slice(offset_rot + 3 * source_idx, offset_rot + 3 * (source_idx + 1))

        tt = mobility_batch[:, trans_rows, force_cols]
        tr = mobility_batch[:, trans_rows, torque_cols]
        rt = mobility_batch[:, rot_rows, force_cols]
        rr = mobility_batch[:, rot_rows, torque_cols]

        force = Ft[:, :3].unsqueeze(-1)
        torque = Ft[:, 3:].unsqueeze(-1)

        trans_vel = torch.bmm(tt, force).squeeze(-1) + torch.bmm(tr, torque).squeeze(-1)
        rot_vel = torch.bmm(rt, force).squeeze(-1) + torch.bmm(rr, torque).squeeze(-1)

        result = torch.cat((trans_vel, rot_vel), dim=-1)

        return result * inv_mu

    def _batch_rpy(
        self,
        positions: torch.Tensor,
        force: torch.Tensor,
        viscosity: TensorLike,
        pair_mask: torch.Tensor | None = None,
        min_distance: float | None = None,
        tile_size: int | None = None,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Evaluate far-field RPY interactions in cache-friendly tiles.

        Parameters
        ----------
        positions:
            Tensor of shape ``(N, 3)`` containing particle coordinates.
        force:
            Tensor of shape ``(N, 6)`` with force/torque for each particle.
        viscosity:
            Fluid viscosity passed to ``_rpy_velocity``.
        pair_mask:
            Optional boolean mask of shape ``(N, N)`` specifying which ordered
            pairs should be evaluated. When omitted, ``min_distance`` must be
            provided and the mask is generated on-the-fly by thresholding the
            distances within each tile.
        min_distance:
            Minimum separation for including a pair when ``pair_mask`` is
            ``None``.
        tile_size:
            Optional override for the square tile edge length. Defaults to
            ``self.rpy_tile_size``.
        out:
            Optional tensor of shape ``(N, 6)`` to accumulate the results into.

        Returns
        -------
        (torch.Tensor, int)
            The tensor receiving the accumulated velocities and the total
            number of evaluated pair interactions.
        """

        if pair_mask is None and min_distance is None:
            raise ValueError("pair_mask or min_distance must be provided")

        tile = int(tile_size or self.rpy_tile_size)
        tile = max(tile, 1)

        positions = positions.contiguous()
        force = force.contiguous()
        N = positions.shape[0]
        device = positions.device
        idx = torch.arange(N, device=device)

        if out is None:
            out = torch.zeros_like(force)

        pair_counter = 0

        for tgt_start in range(0, N, tile):
            tgt_end = min(tgt_start + tile, N)
            pos_t = positions[tgt_start:tgt_end]
            tgt_idx = idx[tgt_start:tgt_end]

            for src_start in range(0, N, tile):
                src_end = min(src_start + tile, N)
                pos_s = positions[src_start:src_end]
                src_idx = idx[src_start:src_end]

                rel = pos_t[:, None, :] - pos_s[None, :, :]

                if pair_mask is not None:
                    valid = pair_mask[tgt_start:tgt_end, src_start:src_end]
                else:
                    dist = torch.linalg.norm(rel, dim=-1)
                    threshold = 0.0 if min_distance is None else float(min_distance)
                    valid = dist > threshold
                    if tgt_start == src_start:
                        diag = torch.arange(pos_t.shape[0], device=device)
                        valid[diag, diag] = False

                if not torch.any(valid):
                    continue

                tgt_local, src_local = torch.nonzero(valid, as_tuple=True)
                if tgt_local.numel() == 0:
                    continue

                rel_pairs = rel[tgt_local, src_local]
                src_wrench = force[src_idx[src_local]]
                tgt_global = tgt_idx[tgt_local]

                vel = self._rpy_velocity_compiled(rel_pairs, src_wrench, viscosity)
                out.index_add_(0, tgt_global, vel)
                pair_counter += tgt_local.numel()

        return out, pair_counter

    # ------------------------------------------------------------------
    # Two-body NN contribution
    # ------------------------------------------------------------------
    def _two_body_velocity(
        self,
        rel_vecs: torch.Tensor,
        target_indices: torch.Tensor,
        source_indices: torch.Tensor,
        force: torch.Tensor,
        viscosity: TensorLike,
    ) -> torch.Tensor:
        if target_indices.numel() == 0:
            return torch.zeros_like(force)

        rel_selected = rel_vecs[target_indices, source_indices]
        dist = torch.linalg.norm(rel_selected, dim=-1, keepdim=True).clamp_min(1e-8)

        r_shift = dist - self.median
        r_sq = r_shift * r_shift
        r_quad = r_sq * r_sq
        min_dist = dist - self.contact_distance
        features = torch.cat([rel_selected, dist, r_sq, r_quad, min_dist], dim=-1)

        force_target = force[target_indices]
        force_source = force[source_indices]
        mu_tensor = torch.as_tensor(viscosity, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            vel = self.two_nn.predict_velocity(features, force_target, force_source, mu_tensor)

        contribution = torch.zeros_like(force)
        contribution.index_add_(0, target_indices, vel)
        return contribution

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(self, config: torch.Tensor, 
           force: torch.Tensor, viscosity: TensorLike) -> torch.Tensor:
        """Return particle velocities for the supplied configuration.

        Parameters
        ----------
        config : array-like, shape (N, 7)
            Particle positions and quaternions (x, y, z, qx, qy, qz, qw).
        force : array-like, shape (N, 6)
            Force and torque vectors per particle.
        viscosity : float or tensor
            Dynamic viscosity of the fluid.
        """

        with torch.no_grad():
            config_t = _as_float_tensor(config, self.device)
            force_t = _as_float_tensor(force, self.device)

            positions = config_t[:, :3]

            v_self = self._self_velocity(force_t, viscosity)

            N = positions.shape[0]
            if N == 1:
                return v_self.detach().cpu().numpy()

            rel = positions.unsqueeze(0) - positions.unsqueeze(1)
            dist = torch.linalg.norm(rel, dim=-1)
            mask_offdiag = ~torch.eye(N, dtype=torch.bool, device=self.device)

            rpy_mask = torch.zeros_like(mask_offdiag)
            if self.shape == "sphere":
                if self.rpy_only:
                    rpy_mask = mask_offdiag
                elif not self.nn_only:
                    rpy_mask = mask_offdiag & (dist > self.switch_dist)
            else:
                if self.rpy_only:
                    raise ValueError("RPY fallback currently supports spheres only")

            nn_mask = mask_offdiag & ~rpy_mask

            velocities = torch.zeros_like(force_t)

            if rpy_mask.any():
                print("no of RPY interactions:", rpy_mask.sum().item())
                torch.cuda.synchronize()
                start = time.perf_counter()
                _, _ = self._batch_rpy(
                    positions,
                    force_t,
                    viscosity,
                    pair_mask=rpy_mask,
                    tile_size=self.rpy_tile_size,
                    out=velocities,
                )
                torch.cuda.synchronize()
                end = time.perf_counter()
                print(f"RPY kernel execution time: {(end - start)*1000:.6f} ms")

            if nn_mask.any():
                print("no of NN interactions:", nn_mask.sum().item())
                torch.cuda.synchronize()
                start = time.perf_counter()
                tgt_idx, src_idx = torch.nonzero(nn_mask, as_tuple=True)
                self.t_idx_ = tgt_idx
                self.s_idx_ = src_idx
                velocities += self._pair_kernel_compiled(rel, tgt_idx, src_idx, force_t, viscosity)
                torch.cuda.synchronize()
                end = time.perf_counter()
                print(f"NN kernel execution time: {(end - start)*1000:.6f} ms")

            result = v_self + velocities
            return result


from src.mob_op_2b_combined import NNMob
from src.mob_op_2b_combined import check_against_ref 

def check_against_ref_gpu(mob, path, print_stuff=False):
    # Load the reference data
    viscosity = 1.0
    dev = torch.device("cuda")
    df = pd.read_csv(path, float_precision="high",
                        header=0, index_col=False)

    numParticles = df.shape[0]    
    config = df[["x","y","z","q_x","q_y","q_z","q_w"]].values
    forces = df[["f_x","f_y","f_z","t_x","t_y","t_z"]].values
    velocity = df[["v_x","v_y","v_z","w_x","w_y","w_z"]].values

    config_t = _as_float_tensor(config, dev)
    force_t = _as_float_tensor(forces, dev)
    v = mob.apply(config_t, force_t, viscosity).cpu().numpy()
    
    np.set_printoptions(precision=5, suppress=True)

    lin_avg_rmse = 0
    ang_avg_rmse = 0
    for i in range(numParticles):
        lin_rmse = np.sqrt(np.mean((v[i, :3] - velocity[i, :3])**2))
        ang_rmse = np.sqrt(np.mean((v[i, 3:] - velocity[i, 3:])**2))
        if print_stuff:
            print(i)
            print(f"linear: {v[i, :3]} {velocity[i, :3]} {lin_rmse:.4f}")
            print(f"angular: {v[i, 3:]} {velocity[i, 3:]} {ang_rmse:.4f}")

        lin_avg_rmse += lin_rmse
        ang_avg_rmse += ang_rmse

    lin_avg_rmse /= numParticles
    ang_avg_rmse /= numParticles
    print(f"Avg linear RMSE: {lin_avg_rmse:.4f}")
    print(f"Avg angular RMSE: {ang_avg_rmse:.4f}")

    err_2b = np.linalg.norm(velocity - v, axis=1).mean()
    print(f"Avg 2-norm error: {err_2b:.4f}")
    return config

def accuracy_test():
    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body = "experiments/combined_2body.wt"

    just_rpy = False

    mob = NNMobTorch(shape, self_path, two_body,
                     nn_only=False, rpy_only=just_rpy)

    mob_cpu = NNMob(shape, self_path, "data/models/two_body_combined_model.pt",
                    nn_only=False, rpy_only=just_rpy)

    # max dist between spheres in the following is 12.78, something 
    # the model not trained to handle on nn_only mode
    path = "tmp/reference_sphere_0.2.csv"

    print("GPU result:")
    config = check_against_ref_gpu(mob, path)
    print()
    print("CPU result:")
    check_against_ref(mob_cpu, path)


def profile_apply(
    path: str = "tmp/uniform_sphere_0.1_1600.csv",
    trace_dir: str = "tmp/profiler/2b_apply",
    wait_steps: int = 2,
    warmup_steps: int = 3,
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

    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/two_body_combined_model.pt"
    mob_gpu = NNMobTorch(shape, self_path, two_body,
                         nn_only=False, rpy_only=False)

    # Warm-up outside profiler to stabilize kernels
    for _ in range(3):
        _ = mob_gpu.apply(config, force, viscosity=1.0)
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
            _ = mob_gpu.apply(config, force, viscosity=1.0)
            torch.cuda.synchronize()
            prof.step()

    sort_key = "cuda_time_total" 
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by=sort_key,
            row_limit=60,
        )
    )


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
    two_body = "experiments/combined_2body.wt"

    just_rpy = False

    mob = NNMobTorch(shape, self_path, two_body,
                     nn_only=False, rpy_only=just_rpy, switch_dist=6.0)

    dev = torch.device("cuda")
    for i in range(4):
        v = mob.apply(config, force, viscosity=1.0)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    v = mob.apply(config, force, viscosity=1.0)
    end.record()
    torch.cuda.synchronize()
    print(f"GPU Time: {start.elapsed_time(end)} ms")



if __name__ == "__main__":
    #accuracy_test()
    #perftest()
    profile_apply()