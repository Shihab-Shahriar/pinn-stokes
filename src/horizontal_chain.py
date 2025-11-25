"""
Fig. 1 of Durlofsky et al. (1987) (non-periodic)

This test case looks at horizontal chains of 5, 9 and 15 spheres 
sedimenting vertically. The instantaneous drag coefficient, 
λ=F/(6πμaU), is measured for each sphere in the chain, 
in each case. Here we set up the chain of length 15. 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from mfs_utils import build_B
from mfs import imp_mfs_mobility_vec
from src.mob_op_3body import NNMob3B
from src.mob_op_nbody import Mob_Op_Nbody


def get_mfs_drags(sphere_positions, F_ext_list, T_ext_list):
    num_spheres = sphere_positions.shape[0]
    acc = "fine"
    shape = "sphere"
    # Load target (particle #1) geometry from files.
    boundary1 = np.loadtxt(f'/home/shihab/src/mfs/points/b_{shape}_{acc}.txt', dtype=np.float64)
    source1 = np.loadtxt(f'/home/shihab/src/mfs/points/s_{shape}_{acc}.txt', dtype=np.float64)

    B_orig = build_B(boundary1, source1, np.zeros(3))
    B1_inv = np.linalg.pinv(B_orig)

    B_inv_list = [B1_inv.copy() for _ in range(num_spheres)]
    b_list = []
    s_list = []

    for i in range(num_spheres):
        b_list.append(boundary1 + sphere_positions[i].reshape(1,3))
        s_list.append(source1 + sphere_positions[i].reshape(1,3))

    V_tilde_list = imp_mfs_mobility_vec(b_list, s_list, F_ext_list, T_ext_list, B_inv_list,
                                        max_iter=300, tol=1e-7)

    velocities_mfs = np.zeros((num_spheres, 6), dtype=np.float64)
    M1 = s_list[0].shape[0]
    for i in range(num_spheres):
        velocities_mfs[i] = V_tilde_list[i][3*M1 : 3*M1 + 6]

    drag_coeffs = np.zeros(num_spheres, dtype=np.float64)
    mu = 1.0  # also in townsend
    for i in range(num_spheres):
        #λ=F/(6πμaU),
        drag_coeffs[i] = -1 / (6 * np.pi * sphere_sizes[i] * velocities_mfs[i][2])

    print("Drag coefficients for each sphere in the horizontal chain of length 15:")
    for i in range(num_spheres):
        print(f"Sphere {i+1}: λ = {drag_coeffs[i]:.6f}")

    return drag_coeffs

def get_mob_op_drags(pos, F, T, is_3b):
    N = pos.shape[0]
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/two_body_combined_model.pt"
    three_body = "data/models/3body_cross.pt"

    mob = Mob_Op_Nbody(
        shape="sphere",
        self_nn_path=self_path,
        two_nn_path=two_body,
        nbody_nn_path="data/models/nbody_pinn_b1.pt",
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
    )

    config = np.ones((N, 7))
    orientations = [Rotation.identity() for _ in range(N)]
    orientations = [rot.as_quat(scalar_first=False) for rot in orientations]
    
    config[:, :3] = pos
    config[:, 3:] = orientations

    
    forces = np.hstack((F, T))
    assert forces.shape == (N, 6)
    v = mob.apply(config, forces, viscosity=1.0)

    drag_coeffs = np.zeros(N, dtype=np.float64)
    mu = 1.0  # also in townsend
    for i in range(N):
        #λ=F/(6πμaU),
        drag_coeffs[i] = -1 / (6 * np.pi * sphere_sizes[i] * v[i][2])

    print("Drag coefficients for each sphere in the horizontal chain of length 15 (MOB-OP):")
    for i in range(N):
        print(f"Sphere {i+1}: λ = {drag_coeffs[i]:.6f}")
    return drag_coeffs


num_spheres = 15
sphere_sizes = np.array([1 for _ in range(num_spheres)])
sphere_positions = np.array([[4*i, 0, 0] for i in range(num_spheres)])

F_ext_list = np.array([[0, 0, -1] for _ in range(num_spheres)])
T_ext_list = np.array([[0, 0, 0] for _ in range(num_spheres)])

#mfs_coeffs = get_mfs_drags(sphere_positions, F_ext_list, T_ext_list)
mfs_coeffs = [0.61787316, 0.55555363, 0.53188422, 0.51888724, 0.51095464,
              0.50609488, 0.50343569, 0.502587,   0.50343569, 0.50609488,
              0.51095464, 0.51888724, 0.53188422, 0.55555363, 0.61787315]

mob_coeffs = get_mob_op_drags(sphere_positions, F_ext_list, T_ext_list,
                              is_3b=False)

print("MFS coeffs:", mfs_coeffs)

N = (num_spheres+1)//2

townsend_data = [0.61764888, 0.55527634, 0.53156096, 0.51856039,
                0.51063075, 0.50577411,0.50311695, 0.50226895,
                0.50311695, 0.50577411, 0.51063075, 0.51856039,
                0.53156096, 0.55527634, 0.61764888]
durlofsky_data = [0.5018, 0.5029, 0.5054, 0.5102,
                  0.5183, 0.5321, 0.5559, 0.6170]  # Extracted from the paper

legends = []
plt.figure(figsize=(14, 8), dpi=150)
#plt.plot(range(N),townsend_data[N-1:]); legends.append('townsend_data')
plt.plot(range(N),durlofsky_data,'x'); legends.append('Durlofsky et al.')
#plt.plot(range(N),mfs_coeffs[N-1:]); legends.append("MFS")
plt.plot(range(N),mob_coeffs[N-1:]); legends.append("M_nbody")
plt.tight_layout()
plt.legend(legends)
plt.xlabel('Sphere number')
plt.ylabel('λ', rotation=0, fontsize=16)
plt.savefig("horizontal_chain_drag.pdf", dpi=600, format="pdf")
plt.show()
