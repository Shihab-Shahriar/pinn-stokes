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
from typing import Union

import torch

from benchmarks.bench_rpy import _two_body_mu_batch, mu as bench_mu


TensorLike = Union[torch.Tensor, float]


def _ensure_device(device: Union[str, torch.device, None]) -> torch.device:
	if device is None:
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(device)


def _as_float_tensor(data, device: torch.device) -> torch.Tensor:
	return torch.as_tensor(data, dtype=torch.float32, device=device)


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
		self.device = _ensure_device(device)

		self.contact_distance = torch.tensor(2.0, dtype=torch.float32, device=self.device)
		self.median = torch.tensor(5.01, dtype=torch.float32, device=self.device)

		self.two_nn = torch.jit.load(two_nn_path, map_location=self.device).eval()
		self.self_nn_path = self_nn_path  # kept for API compatibility; not used.

	# ------------------------------------------------------------------
	# Self interaction (analytic for unit-radius spheres)
	# ------------------------------------------------------------------
	def _self_velocity(self, force: torch.Tensor, viscosity: TensorLike) -> torch.Tensor:
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
		return torch.cat([u, omega], dim=-1)


	def _rpy_velocity(
		self,
		rel_vecs: torch.Tensor,
		src_wrench: torch.Tensor,
		viscosity: TensorLike,
	) -> torch.Tensor:
		"""Far-field Rotne-Prager-Yamakawa mobility via validated reference."""

		if rel_vecs.numel() == 0:
			return torch.zeros((0, 6), dtype=rel_vecs.dtype, device=rel_vecs.device)

		mu_tensor = torch.as_tensor(viscosity, dtype=torch.float32, device=rel_vecs.device)
		if mu_tensor.numel() == 0:
			raise ValueError("viscosity must be a scalar or non-empty tensor")
		mu_scalar = mu_tensor.reshape(-1)[0].to(rel_vecs.dtype)
		if not torch.isfinite(mu_scalar) or mu_scalar <= 0:
			raise ValueError("viscosity must be a positive finite value")

		dtype = rel_vecs.dtype
		device = rel_vecs.device
		src_wrench = src_wrench.to(dtype=dtype, device=device)
		
		batch_size = rel_vecs.shape[0]
		origin = torch.zeros((batch_size, 1, 3), dtype=dtype, device=device)
		rel_vecs_reshaped = rel_vecs.unsqueeze(1)
		centres = torch.cat((origin, rel_vecs_reshaped), dim=1)
		radii = torch.ones((batch_size, 2), dtype=dtype, device=device)

		mobility_batch = _two_body_mu_batch(centres, radii)

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

		block = torch.cat((torch.cat((tt, tr), dim=2), torch.cat((rt, rr), dim=2)), dim=1)
		
		result = torch.bmm(block, src_wrench.unsqueeze(-1)).squeeze(-1)

		return result / mu_scalar

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
	def apply(self, config, force, viscosity: TensorLike) -> torch.Tensor:
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

			if config_t.ndim != 2 or config_t.shape[1] != 7:
				raise ValueError("config must have shape (N, 7)")
			if force_t.shape != (config_t.shape[0], 6):
				raise ValueError("force must have shape (N, 6)")

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
				tgt_idx, src_idx = torch.nonzero(rpy_mask, as_tuple=True)
				rel_rpy = rel[tgt_idx, src_idx]
				src_wrench = force_t[src_idx]
				vel_rpy = self._rpy_velocity(rel_rpy, src_wrench, viscosity)
				velocities.index_add_(0, tgt_idx, vel_rpy)

			if nn_mask.any():
				tgt_idx, src_idx = torch.nonzero(nn_mask, as_tuple=True)
				velocities += self._two_body_velocity(rel, tgt_idx, src_idx, force_t, viscosity)

			result = v_self + velocities
			return result.detach().cpu().numpy()


from src.mob_op_2b_combined import check_against_ref, NNMob

if __name__ == "__main__":
	shape = "sphere"
	self_path = "data/models/self_interaction_model.pt"
	two_body = "data/models/two_body_combined_model.pt"

	just_rpy = False

	mob = NNMobTorch(shape, self_path, two_body,
					 nn_only=False, rpy_only=just_rpy)

	mob_cpu = NNMob(shape, self_path, two_body,
					nn_only=False, rpy_only=just_rpy)

	# max dist between spheres in the following is 12.78, something 
	# the model not trained to handle on nn_only mode
	path = "/home/shihab/repo/tmp/reference_sphere_1.0.csv"

	config = check_against_ref(mob, path)
	check_against_ref(mob_cpu, path)