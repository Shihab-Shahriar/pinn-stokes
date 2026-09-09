"""CPU grand-mobility operator with the moments-based n-body correction (moments_for_nbody.md).

``Mob_Op_Nbody_Moments`` keeps ``Mob_Op_Nbody``'s pair gate and neighbour *selection* (all
particles within ``neighbor_cutoff`` of the pair midpoint, the ``max_neighbors`` smallest
d_kt * d_ks, or all of them with ``max_neighbors=None``) and replaces the per-neighbour feature
rows with the band moments (s_a, v_a, Q_a) consumed by ``MultiBodyMoments``.  The moments of an
unordered pair are computed once (they are identical for (t, s) and (s, t); only s_vec flips),
so the assembled n-body term satisfies M_ji = M_ij^T exactly.

Inference-side neighbourhood defaults match the baseline (radius 6.0 about the midpoint,
K = 10) because the existing training rows carry at most 10 neighbours; ``max_neighbors=None``
and ``neighbor_cutoff=8.0`` are the (out-of-distribution) variants of the design doc.
``pair_cutoff`` gates which pairs receive a correction and must not exceed ``switch_dist`` so
RPY pairs are never corrected.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))

from src.mob_op_nbody import Mob_Op_Nbody
from src.model_archs import MultiBodyMoments, SelfBlockMoments
from src import nbody_features as nf


class Mob_Op_Nbody_Moments(Mob_Op_Nbody):
	DEFAULT_PAIR_CUTOFF: float = 6.0
	PREDICT_CHUNK: int = 100_000  # ordered-pair rows per model call
	PAIR_ROW_CHUNK: int = 25_000  # unordered pairs per feature-build chunk in get_nbody_velocity:
	# the padded-neighbour + band-moment intermediates are O(pairs x K_max) and reached ~5 GB at
	# N=3000, phi=0.15 unchunked (OOM on a 15 GB box); chunking caps them at a few hundred MB.

	def __init__(
		self,
		shape: str,
		self_nn_path: str,
		two_nn_path: str,
		nbody_nn_path: str,
		nn_only: bool = False,
		rpy_only: bool = False,
		switch_dist: float = 6.0,
		pair_cutoff: float = DEFAULT_PAIR_CUTOFF,
		neighbor_cutoff: float = Mob_Op_Nbody.DEFAULT_NEIGHBOR_CUTOFF,
		max_neighbors: Optional[int] = Mob_Op_Nbody.DEFAULT_MAX_NEIGHBORS,
		mean_dist_s: float = Mob_Op_Nbody.DEFAULT_MEAN_DIST_S,
		diag_nn_path: Optional[str] = None,
		diag_cutoff: float = 8.0,
	) -> None:
		super().__init__(
			shape=shape,
			self_nn_path=self_nn_path,
			two_nn_path=two_nn_path,
			nbody_nn_path=nbody_nn_path,
			nn_only=nn_only,
			rpy_only=rpy_only,
			switch_dist=switch_dist,
			neighbor_cutoff=neighbor_cutoff,
			max_neighbors=10 if max_neighbors is None else max_neighbors,
			mean_dist_s=mean_dist_s,
		)
		self.max_neighbors = None if max_neighbors is None else int(max_neighbors)
		self.pair_cutoff = float(pair_cutoff)
		assert self.pair_cutoff <= float(switch_dist), "n-body correction must not be applied to RPY pairs"
		self.diag_cutoff = float(diag_cutoff)
		self.diag_nn = self._load_diag_model(diag_nn_path) if diag_nn_path else None
		if self.diag_nn is not None:
			# The diagonal labels subtract the two-body self correction K_s of every pair with
			# d <= the training cache's pair_cutoff, while the operator adds K_s for every pair
			# with d <= switch_dist.  Both ranges must equal the corrected-pair range, or K_s is
			# double-counted / missed in the shell between them (the published pc8 sidecar pins 8).
			assert self.pair_cutoff == float(switch_dist), "diag correction needs pair_cutoff == switch_dist"

	def _load_nbody_model(self, model_path: str, mean_dist_s: float):
		if model_path.endswith(".pt"):
			model = torch.jit.load(model_path, map_location=self.device).eval()
		elif model_path.endswith(".wt"):
			model = MultiBodyMoments(float(mean_dist_s)).to(self.device)
			model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
			model = model.eval()
		else:
			raise ValueError(f"Unsupported n-body model format for '{model_path}'. Expected '.pt' or '.wt'.")
		assert hasattr(model, "predict_mobility"), "not a MultiBodyMoments model"
		assert float(model.inv_std.min()) > 0 and float(model.basis_scale.min()) > 0
		return model

	def _load_diag_model(self, model_path: str):
		if model_path.endswith(".pt"):
			model = torch.jit.load(model_path, map_location=self.device).eval()
		elif model_path.endswith(".wt"):
			model = SelfBlockMoments().to(self.device)
			model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
			model = model.eval()
		else:
			raise ValueError(f"Unsupported diag model format for '{model_path}'. Expected '.pt' or '.wt'.")
		assert hasattr(model, "predict_mobility"), "not a SelfBlockMoments model"
		assert float(model.inv_std.min()) > 0 and float(model.basis_scale.min()) > 0
		return model

	# ------------------------------------------------------------------
	# Neighbour selection (same semantics as Mob_Op_Nbody, vectorised)
	# ------------------------------------------------------------------
	def _select_neighbor_indices(self, pos: np.ndarray, idx_target: int, idx_source: int) -> List[int]:
		center_t = pos[idx_target]
		center_s = pos[idx_source]
		midpoint = 0.5 * (center_t + center_s)
		d_mid = np.linalg.norm(pos - midpoint, axis=1)
		cand = d_mid <= self.neighbor_cutoff
		cand[idx_target] = False
		cand[idx_source] = False
		idx = np.nonzero(cand)[0]
		if idx.size == 0:
			return []
		score = np.linalg.norm(pos[idx] - center_t, axis=1) * np.linalg.norm(pos[idx] - center_s, axis=1)
		order = np.argsort(score, kind="stable")
		if self.max_neighbors is not None:
			order = order[: self.max_neighbors]
		return sorted(idx[order].tolist())

	# ------------------------------------------------------------------
	# Pair rows
	# ------------------------------------------------------------------
	def _selected_pairs(self, pos: np.ndarray):
		"""Unordered near pairs (t < s) with >= 1 neighbour, CSR neighbour lists (zero-neighbour pairs dropped).

		Selection is ``nf.select_pair_neighbours`` (the one code path shared with the v2 training cache), so
		training rows and inference rows are built identically."""
		t_idx, s_idx, indptr, indices = nf.select_pair_neighbours(
			pos, self.pair_cutoff, self.neighbor_cutoff, self.max_neighbors)
		keep = np.diff(indptr) > 0
		if not keep.any():
			return t_idx[:0], s_idx[:0], indptr[:1], indices[:0]
		if not keep.all():  # drop zero-neighbour pairs (no correction, as in Mob_Op_Nbody)
			counts = np.diff(indptr)[keep]
			indices = np.concatenate([indices[indptr[i]:indptr[i + 1]] for i in np.nonzero(keep)[0]])
			indptr = np.concatenate([[0], np.cumsum(counts)])
			t_idx, s_idx = t_idx[keep], s_idx[keep]
		return t_idx, s_idx, indptr, indices

	def _build_rows(self, pos: np.ndarray, t_idx, s_idx, indptr, indices) -> Tuple[np.ndarray, np.ndarray]:
		"""Model rows (X_ts, X_st) for the given selected pairs + CSR neighbour lists."""
		nbr, mask = nf.pad_neighbours(pos, t_idx, indptr, indices)
		svecs = pos[s_idx] - pos[t_idx]
		X_ts = nf.moment_features(svecs, nbr, mask, self.mean_dist_s)
		X_st = X_ts.copy()
		X_st[:, :3] *= -1.0  # moments are midpoint-based and identical; only the pair axis flips
		return X_ts, X_st

	def _pair_rows(self, pos: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
		"""Unordered near pairs and their model rows for (t, s) and (s, t) -- unchunked (diagnostics path)."""
		t_idx, s_idx, indptr, indices = self._selected_pairs(pos)
		if len(t_idx) == 0:
			empty = np.zeros((0, 111), dtype=np.float32)
			return [], empty, empty
		X_ts, X_st = self._build_rows(pos, t_idx, s_idx, indptr, indices)
		pairs = [(int(t), int(s)) for t, s in zip(t_idx, s_idx)]
		return pairs, X_ts, X_st

	def _predict(self, X: np.ndarray, F: np.ndarray, blocks: bool = False) -> np.ndarray:
		out = []
		with torch.no_grad():
			for i in range(0, X.shape[0], self.PREDICT_CHUNK):
				Xc = torch.as_tensor(X[i:i + self.PREDICT_CHUNK], dtype=torch.float32, device=self.device)
				if blocks:
					out.append(self.nbody_nn.predict_mobility(Xc).cpu().numpy().astype(np.float64))
				else:
					Fc = torch.as_tensor(F[i:i + self.PREDICT_CHUNK], dtype=torch.float32, device=self.device)
					out.append(self.nbody_nn.predict_velocity(Xc, Fc).cpu().numpy().astype(np.float64))
		return np.concatenate(out, 0)

	# ------------------------------------------------------------------
	# Velocity computation
	# ------------------------------------------------------------------
	def get_nbody_velocity(self, pos: np.ndarray, force: np.ndarray, viscosity: float) -> np.ndarray:
		assert pos.ndim == 2 and pos.shape[1] == 3
		assert force.ndim == 2 and force.shape[1] == 6
		_ = viscosity  # absorbed by the learned model (as in Mob_Op_Nbody)
		N = pos.shape[0]
		velocities = np.zeros((N, 6), dtype=np.float64)
		t_idx, s_idx, indptr, indices = self._selected_pairs(pos)
		P = len(t_idx)
		for i0 in range(0, P, self.PAIR_ROW_CHUNK):  # bounded feature-build memory; rows are independent
			i1 = min(i0 + self.PAIR_ROW_CHUNK, P)
			X_ts, X_st = self._build_rows(pos, t_idx[i0:i1], s_idx[i0:i1],
			                              indptr[i0:i1 + 1] - indptr[i0], indices[indptr[i0]:indptr[i1]])
			X = np.concatenate([X_ts, X_st], 0)
			F = np.concatenate([force[s_idx[i0:i1]], force[t_idx[i0:i1]]], 0)  # (t,s): force on s moves t; (s,t): vice versa
			pred = self._predict(X, F)
			n = i1 - i0
			np.add.at(velocities, t_idx[i0:i1], pred[:n])
			np.add.at(velocities, s_idx[i0:i1], pred[n:])
		return velocities

	def nbody_pair_blocks(self, pos: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
		"""Ordered pairs [(t,s)...] + [(s,t)...] and their 6x6 correction blocks (for reciprocity checks)."""
		pairs, X_ts, X_st = self._pair_rows(pos)
		if not pairs:
			return [], np.zeros((0, 6, 6))
		ordered = pairs + [(s, t) for (t, s) in pairs]
		K = self._predict(np.concatenate([X_ts, X_st], 0), None, blocks=True)
		return ordered, K

	# ------------------------------------------------------------------
	# Per-particle diagonal (self-block) correction
	# ------------------------------------------------------------------
	def _diag_rows(self, pos: np.ndarray) -> np.ndarray:
		"""Self-model rows (N, 104) via the shared selection path (training == inference):
		all k != t within ``diag_cutoff`` of particle t (nf.select_particle_neighbours)."""
		indptr, indices = nf.select_particle_neighbours(pos, self.diag_cutoff)
		nbr, mask = nf.pad_neighbours(pos, np.arange(len(pos)), indptr, indices)
		return nf.self_moment_features(nbr, mask)

	def diag_blocks(self, pos: np.ndarray) -> np.ndarray:
		"""(N, 6, 6) learned symmetric diagonal corrections (viscosity 1), for tests/diagnostics."""
		assert self.diag_nn is not None
		X = self._diag_rows(pos)
		out = []
		with torch.no_grad():
			for i in range(0, X.shape[0], self.PREDICT_CHUNK):
				Xc = torch.as_tensor(X[i:i + self.PREDICT_CHUNK], dtype=torch.float32, device=self.device)
				out.append(self.diag_nn.predict_mobility(Xc).cpu().numpy().astype(np.float64))
		return np.concatenate(out, 0)

	def get_diag_velocity(self, pos: np.ndarray, force: np.ndarray, viscosity: float) -> np.ndarray:
		"""v_t += K_diag(t) F_t / mu -- the labels are at viscosity 1 and mobility scales as 1/mu."""
		assert self.diag_nn is not None
		X = self._diag_rows(pos)
		out = []
		with torch.no_grad():
			for i in range(0, X.shape[0], self.PREDICT_CHUNK):
				Xc = torch.as_tensor(X[i:i + self.PREDICT_CHUNK], dtype=torch.float32, device=self.device)
				Fc = torch.as_tensor(force[i:i + self.PREDICT_CHUNK], dtype=torch.float32, device=self.device)
				out.append(self.diag_nn.predict_velocity(Xc, Fc).cpu().numpy().astype(np.float64))
		return np.concatenate(out, 0) / float(viscosity)

	def apply(self, config: np.ndarray, force: np.ndarray, viscosity: float) -> np.ndarray:
		"""Base (self + 2b/RPY) + pair moments correction + optional learned diagonal correction."""
		v = super().apply(config, force, viscosity)
		if self.diag_nn is not None:
			v = v + self.get_diag_velocity(config[:, :3], force, viscosity)
		return v


if __name__ == "__main__":
	from src.mob_op_2b_combined import check_against_ref

	shape = "sphere"
	self_path = "data/models/self_interaction_model.pt"
	two_body = "data/models/two_body_combined_model.pt"
	mob = Mob_Op_Nbody_Moments(shape=shape, self_nn_path=self_path, two_nn_path=two_body,
		nbody_nn_path="data/models/nbody_moments.pt", nn_only=False, rpy_only=False, switch_dist=6.0)
	for d in ["0.1", "0.2", "0.5", "1.0", "2.0", "3.0"]:
		print(f"\n=== Separation {d} ===")
		check_against_ref(mob, f"tmp/reference_sphere_{d}.csv")
