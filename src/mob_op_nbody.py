"""Mobility operator with learned multi-body correction for spheres."""

from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

import numpy as np
import torch


EPS = 1e-12

def normalize(v: torch.Tensor) -> torch.Tensor:
    nrm = torch.sqrt(torch.sum(v * v, dim=-1, keepdim=True))
    nrm = torch.clamp(nrm, min=EPS)
    return v / nrm

def outer(u):      return torch.einsum('...i,...j->...ij', u, u)
def sym_outer(u,v):return 0.5*(torch.einsum('...i,...j->...ij', u, v) +
                               torch.einsum('...i,...j->...ij', v, u))

def cross_mat(u):
    ux, uy, uz = u.unbind(dim=-1)
    O = torch.zeros(u.shape[:-1]+(3,3), device=u.device, dtype=u.dtype)
    O[...,0,1], O[...,0,2] = -uz,  uy
    O[...,1,0], O[...,1,2] =  uz, -ux
    O[...,2,0], O[...,2,1] = -uy,  ux
    return O

def fallback_axes(e_vec: torch.Tensor) -> torch.Tensor:
    ex = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(e_vec)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand_as(e_vec)

    cand = torch.where((torch.abs((e_vec * ex).sum(dim=-1, keepdim=True)) < 0.9), ex, ey)
    g_vec = torch.cross(e_vec, cand, dim=-1)
    return normalize(g_vec)

def create_basis(pos_flat: torch.Tensor, mask: torch.Tensor):
    """Build a pair-symmetric orthonormal frame (e, ĝ) from target-centred inputs."""
    B = pos_flat.size(0)
    assert pos_flat.size(1) == 33
    assert mask.size(0) == B and mask.size(1) == 10

    device, dtype = pos_flat.device, pos_flat.dtype
    eps = EPS

    pos = pos_flat.reshape(B, 11, 3)
    x_s = pos[:, 0, :]
    x_k = pos[:, 1:, :]
    m_k = mask.to(dtype)

    e = normalize(-x_s)

    midpoint = 0.5 * x_s
    rel_mid = x_k - midpoint.unsqueeze(1)
    rel_t = x_k
    rel_s = x_k - x_s.unsqueeze(1)

    d_t = rel_t.norm(dim=-1).clamp_min(eps)
    d_s = rel_s.norm(dim=-1).clamp_min(eps)
    w = m_k * (1.0 / (d_t * d_s))

    weight_sum = w.sum(dim=1, keepdim=True).clamp_min(eps)

    proj = torch.eye(3, device=device, dtype=dtype).expand(B, 3, 3) - e.unsqueeze(-1) * e.unsqueeze(-2)
    rel_perp = torch.einsum("bij,bkj->bki", proj, rel_mid)

    cov = torch.einsum("bk,bki,bkj->bij", w, rel_perp, rel_perp) / weight_sum.unsqueeze(-1)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    cov_plane = torch.einsum("bij,bjk->bik", proj, torch.einsum("bij,bjk->bik", cov, proj))

    evals, evecs = torch.linalg.eigh(cov_plane)
    ghat = evecs[..., -1]
    ghat = normalize(ghat)

    g_mean = torch.einsum("bk,bki->bi", w, rel_perp) / weight_sum
    sign = torch.sign((ghat * g_mean).sum(dim=-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    ghat = ghat * sign

    ghat_norm = ghat.norm(dim=-1, keepdim=True)
    use_fallback = (weight_sum.squeeze(-1) <= 5 * eps) | (ghat_norm.squeeze(-1) < 1e-4)
    if use_fallback.any():
        ghat_fb = fallback_axes(e[use_fallback])
        ghat = ghat.clone()
        ghat[use_fallback] = ghat_fb

    return e, ghat


# Ensure relative imports work if run as a script
sys.path.append(os.path.dirname(__file__))

from mob_op_2b_combined import NNMob as TwoBodyNNMob


class Mob_Op_Nbody(TwoBodyNNMob):
	"""NNMob augmented with an n-body correction network.

	This operator augments the two-body neural mobility with a single neural
	network that, for each ordered target/source pair (t, s), consumes the
	positions of up to ``max_neighbors`` other particles around the pair and
	predicts a 6-vector velocity correction on ``t`` due to the force/torque on
	``s``.
	"""

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
		assert shape == "sphere", "Only sphere shape currently supported for n-body operator"
		super().__init__(
			shape=shape,
			self_nn_path=self_nn_path,
			two_nn_path=two_nn_path,
			nn_only=nn_only,
			rpy_only=rpy_only,
			switch_dist=switch_dist,
		)

		self.nbody_nn = torch.jit.load(nbody_nn_path, map_location=self.device).eval()
		self.max_neighbors = int(max_neighbors)
		self.neighbor_cutoff = float(neighbor_cutoff)
		self.mean_dist_s = float(mean_dist_s)

	# ------------------------------------------------------------------
	# Feature construction helpers
	# ------------------------------------------------------------------
	def _select_neighbor_indices(
		self,
		pos: np.ndarray,
		idx_target: int,
		idx_source: int,
	) -> List[int]:
		"""Return up to ``max_neighbors`` indices ordered by d_kt * d_ks."""

		center_t = pos[idx_target]
		center_s = pos[idx_source]
		midpoint = 0.5 * (center_t + center_s)

		selected: List[Tuple[float, int]] = []
		for k in range(pos.shape[0]):
			if k == idx_target or k == idx_source:
				continue

			# dist from midpoint, not t or s, to ensure symmetry
			if np.linalg.norm(pos[k] - midpoint) > self.neighbor_cutoff:
				continue

			d_kt_vec = pos[k] - center_t
			d_kt = float(np.linalg.norm(d_kt_vec))

			d_ks = float(np.linalg.norm(pos[k] - center_s))
			score = d_kt * d_ks
			selected.append((score, k))

		selected.sort(key=lambda x: x[0])
		neighbors = [idx for (_, idx) in selected[: self.max_neighbors]]
		return sorted(neighbors) # return sorted indexes for symmetry

	def _compute_neighbor_features(
		self,
		s_vec: np.ndarray,
		neighbor_vectors: Sequence[np.ndarray],
	) -> Tuple[np.ndarray, np.ndarray]:
		"""Compute symmetric geometric features for neighbor particles.

		Parameters
		----------
		s_vec : np.ndarray
			Vector from target ``t`` to source ``s`` (shape (3,)).
		neighbor_vectors : sequence of np.ndarray
			Each entry is the vector from target ``t`` to neighbor ``k``.

		Returns
		-------
		tuple[np.ndarray, np.ndarray]
			Flattened feature array of shape (max_neighbors * 10,) and a mask of
			shape (max_neighbors,) indicating which rows are valid (1.0) vs padding (0.0).
		"""
		K = self.max_neighbors
		feats = np.zeros((K, 10), dtype=np.float64)
		mask = np.zeros(K, dtype=np.float64)

		# Pair geometry
		ell = float(np.linalg.norm(s_vec))
		ell_c = max(ell, self._EPS)
		zhat = s_vec / ell_c
		midpoint = 0.5 * s_vec

		for idx, k_vec in enumerate(neighbor_vectors):
			if idx >= K:
				break
			mask[idx] = 1.0

			# Distances
			r_sk_vec = k_vec - s_vec          # (k - s)
			r_sk = float(np.linalg.norm(r_sk_vec))
			r_kt = float(np.linalg.norm(k_vec))  # (k - t) with t at origin
			r_sk_c = max(r_sk, self._EPS)
			r_kt_c = max(r_kt, self._EPS)

			# Symmetric distance combos
			r_sum  = r_sk + r_kt
			r_diff = abs(r_sk - r_kt)
			r_prod = r_sk * r_kt

			# Axial/radial in pair frame (even-only features)
			v = k_vec - midpoint
			u = float(np.dot(v, zhat))                  # signed axial, but we take even forms
			v_perp = v - u * zhat
			rho = float(np.linalg.norm(v_perp))

			u_over_ell_abs = abs(u) / ell_c
			u_over_ell_sq  = (u / ell_c) ** 2
			rho_over_ell   = rho / ell_c

			# Symmetric "inverse distance" features
			inv_sum     = (1.0 / r_sk_c) + (1.0 / r_kt_c)
			inv_prod    = 1.0 / (r_sk_c * r_kt_c)
			inv_absdiff = abs((1.0 / r_sk_c) - (1.0 / r_kt_c))

			# Angle at k (symmetric): cos_k = ((s-k)·(t-k)) / (r_sk r_kt)
			a = s_vec - k_vec            # (s - k)
			b = -k_vec                   # (t - k), since t = 0
			num = float(np.dot(a, b))
			den = r_sk_c * r_kt_c
			cos_k = num / max(den, self._EPS)
			cos_k = float(np.clip(cos_k, -1.0, 1.0))

			feats[idx] = [
				r_sum,
				r_diff,
				r_prod,
				u_over_ell_abs,
				u_over_ell_sq,
				rho_over_ell,
				inv_sum,
				inv_prod,
				inv_absdiff,
				cos_k,
			]

		return feats.reshape(-1), mask

	def _build_pair_feature_vector(
		self,
		pos_t: np.ndarray,
		pos_s: np.ndarray,
		neighbor_vectors: Sequence[np.ndarray],
	) -> np.ndarray:
		"""Return feature vector matching the multibody training pipeline (symmetric neighbor features)."""

		s_vec = pos_s - pos_t
		dist_raw = float(np.linalg.norm(s_vec))
		dist_centered = dist_raw - self.mean_dist_s
		dist_sq = dist_centered * dist_centered
		dist_sqsq = dist_sq * dist_sq
		K = len(neighbor_vectors)
		assert K <= self.max_neighbors

		neighbor_feats, neighbor_mask = self._compute_neighbor_features(s_vec, neighbor_vectors)

		idx_pos_feat_end = 3 + self.max_neighbors * 3
		# 4 pair scalars + (10 features per neighbor) + neighbor mask
		n_feat = idx_pos_feat_end + 4 + 10 * self.max_neighbors + self.max_neighbors

		feature = np.zeros(n_feat, dtype=np.float32)

		# Pair positions chunk: s_vec first, then K neighbor vectors (pad to max_neighbors)
		feature[0:3] = s_vec.astype(np.float32)
		feature[3:3 + K * 3] = np.array(neighbor_vectors, dtype=np.float32).reshape(-1)
		feature[3 + K * 3 : idx_pos_feat_end] = 0.0  # padding

		# Pair distance scalars
		feature[idx_pos_feat_end + 0] = dist_centered
		feature[idx_pos_feat_end + 1] = dist_raw - 2.0
		feature[idx_pos_feat_end + 2] = dist_sq
		feature[idx_pos_feat_end + 3] = dist_sqsq

		# Symmetric neighbor features + mask
		start = idx_pos_feat_end + 4
		end = start + 10 * self.max_neighbors
		feature[start:end] = neighbor_feats.astype(np.float32)
		feature[end:] = neighbor_mask.astype(np.float32)

		return feature

	# ------------------------------------------------------------------
	# Velocity computation
	# ------------------------------------------------------------------
	def get_nbody_velocity(
		self,
		pos: np.ndarray,
		force: np.ndarray,
		viscosity: float,
	) -> np.ndarray:
		"""Compute the learned n-body correction for each particle."""

		assert pos.ndim == 2 and pos.shape[1] == 3
		assert force.ndim == 2 and force.shape[1] == 6

		_ = viscosity  # viscous scaling is absorbed by the learned model

		N = pos.shape[0]
		velocities = np.zeros((N, 6), dtype=np.float64)

		#force[:, 3:] = 0.0  # only use forces for n-body correction

		pair_features: List[np.ndarray] = []
		pair_forces: List[np.ndarray] = []
		target_indices: List[int] = []

		idx_cachee = {}
		feat_cache = {}

		for t in range(N):
			pos_t = pos[t]

			for s in range(N):
				if s == t:
					continue

				pos_s = pos[s]
				dist_ts = float(np.linalg.norm(pos_s - pos_t))
				if dist_ts > self.neighbor_cutoff:
					continue

				neighbor_idx = self._select_neighbor_indices(pos, t, s)

				if not neighbor_idx:
					continue

				if (s, t) in idx_cachee:
					assert idx_cachee[(s, t)] == neighbor_idx
				else:
					idx_cachee[(t, s)] = neighbor_idx
				neighbor_vectors = [pos[k] - pos_t for k in neighbor_idx]

				features = self._build_pair_feature_vector(pos_t, pos_s, neighbor_vectors)
				pair_features.append(features)
				pair_forces.append(force[s].astype(np.float32))
				target_indices.append((t,s))

				if (s, t) in feat_cache:
					assert np.allclose(feat_cache[(s, t)][33:], features[33:])
					feat_cache[(t, s)] = features.copy()
				else:
					feat_cache[(t, s)] = features.copy()


		if not pair_features:
			return velocities

		X = torch.tensor(np.stack(pair_features), dtype=torch.float32, device=self.device)
		Fs = torch.tensor(np.stack(pair_forces), dtype=torch.float32, device=self.device)

		# e, ghat = create_basis(X[:,:33], torch.tensor(X[:,-self.max_neighbors:], dtype=torch.float32, device=self.device))
		# for idx, (t, s) in enumerate(target_indices):
		# 	for idx2, (t2, s2) in enumerate(target_indices):
		# 		if t == s2 and s == t2:
		# 			# check e, ghat for symmetry
		# 			assert torch.allclose(e[idx], -e[idx2]), f"e not antisymmetric for {(t,s)} vs {(t2,s2)}"
		# 			#assert torch.allclose(ghat[idx], ghat[idx2]), f"ghat not symmetric for {(t,s)} vs {(t2,s2)}"

		with torch.no_grad():
			pred = self.nbody_nn.predict_velocity(X, Fs)

		pred_np = pred.cpu().numpy().astype(np.float64)

		for idx, (t, s) in enumerate(target_indices):
			velocities[t] += pred_np[idx]

		return velocities

	def apply(
		self,
		config: np.ndarray,
		force: np.ndarray,
		viscosity: float,
	) -> np.ndarray:
		"""Override base apply with additional n-body correction."""
		N = config.shape[0]
		assert config.shape == (N, 7)
		v_base = super().apply(config, force, viscosity, spd_diagnostics=False)

		pos = config[:, :3]
		v_nbody = self.get_nbody_velocity(pos, force, viscosity)
		#v_nbody[:, 3:] = 0.0  # angular prediction bad for nbody

		return v_base + v_nbody


if __name__ == "__main__":
	try:
		from mob_op_nn import check_against_ref, helens_3body_sphere
	except Exception:
		sys.path.append(os.path.dirname(__file__))
		from mob_op_nn import check_against_ref

	from mob_op_3body import NNMob3B

	shape = "sphere"
	self_path = "data/models/self_interaction_model.pt"
	two_body = "data/models/two_body_combined_model.pt"


	mob_2b = TwoBodyNNMob(shape, self_path, two_body, 
				nn_only=False, rpy_only=False)

	mob_nbody = Mob_Op_Nbody(
		shape=shape,
		self_nn_path=self_path,
		two_nn_path=two_body,
		nbody_nn_path="data/models/nbody_pinn.pt",
		nn_only=False,
		rpy_only=False,
		switch_dist=6.0,
	)

	mob_nbody_recovered = Mob_Op_Nbody(
		shape=shape,
		self_nn_path=self_path,
		two_nn_path=two_body,
		nbody_nn_path="data/models/nbody_pinn_recovered.pt",
		nn_only=False,
		rpy_only=False,
		switch_dist=6.0,
	)

	mob_nbody_b1 = Mob_Op_Nbody(
		shape=shape,
		self_nn_path=self_path,
		two_nn_path=two_body,
		nbody_nn_path="data/models/nbody_pinn_b1.pt",
		nn_only=False,
		rpy_only=False,
		switch_dist=6.0,
	)

	mob_3body = NNMob3B(
		shape=shape,
		self_nn_path=self_path,
		two_nn_path=two_body,
		three_nn_path="data/models/3body_cross.pt",
		nn_only=False,
		rpy_only=False,
		switch_dist=6.0,
		triplet_cutoff=6.0,
	)

	# check_against_ref(mob_nbody_recovered, f"tmp/reference_sphere_0.5.csv")
	# print("-----")



	for d in ["0.1", "0.2", "0.5", "1.0", "2.0", "3.0"]:
		print(f"\n=== Separation {d} ===")

		ref_path = f"tmp/reference_sphere_{d}.csv"

		# print("\n=== N-body operator accuracy ===")
		# check_against_ref(mob_nbody, ref_path)
		# print("-----")

		print("\n=== N-body operator accuracy recovered ===")
		check_against_ref(mob_nbody_recovered, ref_path)
		print("-----")

		# print("\n=== 2-body operator accuracy ===")
		# check_against_ref(mob_2b, ref_path)
		# print("-----")

		print("\n=== N-body operator accuracy b1 ===")
		check_against_ref(mob_nbody_b1, ref_path)
		print("-----")

		# print("\n=== 3-body operator accuracy ===")
		# check_against_ref(mob_3body, ref_path)
		# print("-----")


	# for S in [.1, 0.5, 1.0, 2.0, 4.0]:
	# 	print(f"Separation {S}:")
	# 	helens_3body_sphere(mob_nbody, S, shape=shape)
	# 	print("-----")