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


def outer(u):                        
    return torch.einsum('bi,bj->bij', u, u)

def sym(u, v):                        
    return 0.5*(torch.einsum('bi,bj->bij', u, v) +
                torch.einsum('bi,bj->bij', v, u))

def cross_mat(u):                     # [u]×
    levi = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],
                         [[0,0,-1],[0,0,0],[1,0,0]],
                         [[0,1,0],[-1,0,0],[0,0,0]]],
                        dtype=u.dtype, device=u.device)
    return torch.einsum('ijk,bk->bij', levi, u)

def triplet_basis(r12, r13):          # r12 = x2–x1 , r13 = x3–x1
    e = r12 / r12.norm(p=2, dim=1, keepdim=True)
    f = r13 / r13.norm(p=2, dim=1, keepdim=True)
    I  = torch.eye(3, device=e.device).expand(len(e),3,3)
    B  = {                          # six independent pieces
        'I' : I,        'ee': outer(e),
        'ff': outer(f), 'ef': sym(e,f),
        'Xe': cross_mat(e), 'Xf': cross_mat(f)
    }
    return B

def triplet_block_corrected(r12, r13, coeff):
    B = triplet_basis(r12, r13)

    TT = (coeff[:,0,None,None]*B['I']  + coeff[:,1,None,None]*B['ee']
        + coeff[:,2,None,None]*B['ff'] + coeff[:,3,None,None]*B['ef'])
    RT = (coeff[:,4,None,None]*B['Xe'] + coeff[:,5,None,None]*B['Xf'])
    RR = (coeff[:,6,None,None]*B['I']  + coeff[:,7,None,None]*B['ee']
        + coeff[:,8,None,None]*B['ff'] + coeff[:,9,None,None]*B['ef'])

    K   = torch.zeros(len(r12),6,6, device=r12.device)
    K[:,:3,:3] = TT           # TT block
    K[:,3:,:3] = RT           # RT block
    K[:,:3,3:] = 0.0          # TR = 0 (Ignore spurious torque coupling)
    K[:,3:,3:] = RR           # RR block
    return K                 # shape [batch,6,6]

def velocity_triplet_corrected(r12, r13, coeff, force_s):
    K   = triplet_block_corrected(r12, r13, coeff)      # [B,6,6]
    v  = torch.einsum('bij,bj->bi', K, force_s)   # [B,6]
    return v


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
        assert shape=="sphere", "Only sphere shape currently supported for 3B"
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
                # pred_v = self.three_nn.predict_velocity(X, Fs).cpu().numpy()
                
                # Use corrected velocity calculation
                coeffs = self.three_nn(X[:, 6:])
                r12 = X[:, 0:3]
                r13 = X[:, 3:6]
                pred_v = velocity_triplet_corrected(r12, r13, coeffs, Fs).cpu().numpy()

            velocities[t] = pred_v.sum(axis=0)
            velocities[t, 3:] = 0.0 #angular prediction bad for 3b_cross

            #print("3b terms for ", t, ":", len(triplet_features), len(neighbours))

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

        # Initialize M matrix as in the parent class, required for get_two_vel
        self.M = np.zeros((6*N, 6*N), dtype=np.float64)

        pos = config[:, :3]
        # orientations = [Rotation.identity() for _ in range(N)]  # unused for spheres
        orientations = Rotation.from_quat(config[:, 3:], scalar_first=False)

        # v_s_and_2b = super().apply(config, force, viscosity)
        
        # 1. Self velocity
        v_self = self.get_self_vel_analytical(orientations, force, viscosity)

        # 2. Two-body velocity
        v_two = self.get_two_vel(pos, orientations, force, viscosity)

        # 3. Three-body velocity
        v_three = self.get_3b_vel(pos, None, force, viscosity)

        #print(v_s_and_2b.shape, v_three.shape)
        return v_self + v_two + v_three


if __name__ == "__main__":
    # Small harness to run check_against_ref on reference_sphere.csv and report accuracy
    try:
        from mob_op_nn import check_against_ref, helens_3body_sphere
    except Exception:
        # If import path issues, add src to sys.path explicitly
        sys.path.append(os.path.dirname(__file__))
        from mob_op_nn import check_against_ref, helens_3body_sphere

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
    #check_against_ref(mob, ref_path)

    for S in [.1, 0.5, 1.0, 2.0, 4.0]:
        print(f"Separation {S}:")
        helens_3body_sphere(mob, S, shape=shape)
        print("-----")

