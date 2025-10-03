"""
Three-body (triplet) cross mobility operator built on top of the 2-body combined NNMob.

For each target particle t, we consider ordered neighbour pairs (s, k) such that
both s and k are within a cutoff distance (default 6.0) from t. For each triplet
(t, s, k), a neural network predicts the velocity on t due to the force/torque on s
in the presence of k. These triplet contributions are added to the self- and two-body
contributions from the base operator.
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Ensure relative imports work if run as a script
sys.path.append(os.path.dirname(__file__))

from mob_op_2b_combined import NNMob as TwoBodyNNMob


class NNMob3B(TwoBodyNNMob):
    """NNMob with an additional additive three-body cross term."""

    def __init__(
        self,
        shape: str,
        self_nn_path: str,
        two_nn_path: str,
        three_nn_path: str,
        nn_only: bool = False,
        rpy_only: bool = False,
        switch_dist: float = 6.0,
        triplet_cutoff: float = 6.0,
    ) -> None:
        super().__init__(
            shape=shape,
            self_nn_path=self_nn_path,
            two_nn_path=two_nn_path,
            nn_only=nn_only,
            rpy_only=rpy_only,
            switch_dist=switch_dist,
        )

        self.three_nn = torch.jit.load(three_nn_path, map_location=self.device).eval()
        self.triplet_cutoff = float(triplet_cutoff)

        # Copied from notebook
        self.mean_dist_s = 4.724224262198524
        self.mean_dist_k = 4.701194833185605
        self.mean_dist_sk = 6.676046130020521


    def _build_triplet_features(
        self,
        center_t: np.ndarray,
        center_s: np.ndarray,
        center_k: np.ndarray,
    ) -> np.ndarray:
        """
        ['x_s', 'y_s', 'z_s', 'x_k', 'y_k', 'z_k', 
        'dist_s', 'dist_s_sq', 'dist_s_sqsq', 'min_dist_s', 
        'dist_k', 'dist_k_sq', 'dist_k_sqsq', 'min_dist_k', 
        'dist_sk', 'dist_sk_sq', 'dist_sk_sqsq'
        """
        dvec_s = center_t - center_s
        dvec_k = center_t - center_k

        dist_s = float(np.linalg.norm(dvec_s)) - self.mean_dist_s
        dist_k = float(np.linalg.norm(dvec_k)) - self.mean_dist_k
        dist_sk = float(np.linalg.norm(center_s - center_k)) - self.mean_dist_sk
        min_dist_s = float(np.linalg.norm(dvec_s)) - 2.0
        min_dist_k = float(np.linalg.norm(dvec_k)) - 2.0

        dist_s_sq = dist_s * dist_s
        dist_s_sqsq = dist_s_sq * dist_s_sq
        dist_k_sq = dist_k * dist_k
        dist_k_sqsq = dist_k_sq * dist_k_sq
        dist_sk_sq = dist_sk * dist_sk
        dist_sk_sqsq = dist_sk_sq * dist_sk_sq

        feat_s = [
            dvec_s[0], dvec_s[1], dvec_s[2],
            dvec_k[0], dvec_k[1], dvec_k[2],
            dist_s, dist_s_sq, dist_s_sqsq, min_dist_s,
            dist_k, dist_k_sq, dist_k_sqsq, min_dist_k,
            dist_sk, dist_sk_sq, dist_sk_sqsq
        ]
        feat = np.array(feat_s, dtype=np.float64)
        assert feat.shape == (17,)
        return feat

    def get_3b_vel(
        self,
        pos: np.ndarray,
        orientations,  # unused (sphere only)
        force: np.ndarray,
        viscosity: float,
    ) -> np.ndarray:
        """Compute additive 3-body cross velocity for each particle in lab frame.

        For every target t, and each ordered neighbour pair (s, k) within cutoff
        from t, predict the velocity on t due to force/torque on s in the presence
        of k, and sum over all such ordered pairs.
        """
        assert pos.ndim == 2 and pos.shape[1] == 3
        assert force.ndim == 2 and force.shape[1] == 6

        N = pos.shape[0]
        velocities = np.zeros((N, 6), dtype=np.float64)

        cutoff = self.triplet_cutoff

        for t in range(N):
            dvec_t = pos - pos[t]  # (N,3) vectors from t to i in lab frame
            dists_t = np.linalg.norm(dvec_t, axis=1)

            neighbours = [i for i in range(N) if i != t and dists_t[i] <= cutoff]
            if len(neighbours) < 2:
                continue

            del dvec_t, dists_t

            center_t = pos[t]

            triplet_features: List[np.ndarray] = []
            Fs_list: List[np.ndarray] = []

            for s in neighbours:
                center_s = pos[s]
                F_s = force[s]

                for k in neighbours:
                    if k == s:
                        continue
                    center_k = pos[k]

                    feat_sk = self._build_triplet_features(center_t, center_s, center_k)
                    triplet_features.append(feat_sk)
                    Fs_list.append(F_s)

            if not triplet_features:
                continue

            X = torch.tensor(np.asarray(triplet_features), dtype=torch.float32, device=self.device)
            Fs = torch.tensor(np.asarray(Fs_list), dtype=torch.float32, device=self.device)
            mu = torch.tensor(float(viscosity), dtype=torch.float32, device=self.device)

            with torch.no_grad():
                pred_v = self.three_nn.predict_velocity(X, Fs).cpu().numpy()

            velocities[t] = pred_v.sum(axis=0)
            velocities[t, 3:] = 0.0 #angular prediction bad for 3b_cross

            print("3b terms for ", t, ":", len(triplet_features), len(neighbours))

        return velocities


    def apply(
        self,
        config: np.ndarray,
        force: np.ndarray,
        viscosity: float,
    ) -> np.ndarray:
        """Override: self + 2-body + 3-body cross contributions."""
        N = config.shape[0]
        assert config.shape == (N, 7)

        pos = config[:, :3]
        orientations = Rotation.from_quat(config[:, 3:], scalar_first=False)

        v_self = self.get_self_vel_analytical(orientations, force, viscosity)
        v_two = self.get_two_vel(pos, orientations, force, viscosity)
        v_three = self.get_3b_vel(pos, None, force, viscosity)
        return v_self + v_two + v_three


if __name__ == "__main__":
    # Small harness to run check_against_ref on reference_sphere.csv and report accuracy
    try:
        from mob_op_nn import check_against_ref
    except Exception:
        # If import path issues, add src to sys.path explicitly
        sys.path.append(os.path.dirname(__file__))
        from mob_op_nn import check_against_ref

    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/two_body_combined_model.pt"
    three_body = "data/models/3body_cross.pt"

    mob = NNMob3B(
        shape=shape,
        self_nn_path=self_path,
        two_nn_path=two_body,
        three_nn_path=three_body,
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
        triplet_cutoff=6.0,
    )

    ref_path = "data/reference_sphere.csv"
    check_against_ref(mob, ref_path)

