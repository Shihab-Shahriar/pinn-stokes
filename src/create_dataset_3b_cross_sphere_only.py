"""
Generate a three-body interactions dataset for spheroids.

Target (particle #1) is fixed at the origin 

for each particle k: it should consist of d_kt, d_ks, d_kt*d_ks, Perpendicular distance to the t-s axis, cos(∠stk) and cos(∠tsk)

"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from mfs import imp_mfs_mobility_vec
from mfs_utils import build_B, get_QM_QN, createNewEllipsoid, min_distance_ellipsoids
#from ..utils import save_multiple_ellipsoids_legacy_vtk


def random_unit_vector():
    """Return a random unit vector in R^3."""
    vec = np.random.randn(3)
    return vec / np.linalg.norm(vec)


def random_orientation_spheroid():
    """
    Return a random rotation (scipy Rotation) that sends the z-axis to a random direction.
    Suitable for prolate/oblate spheroids.
    """
    z_axis = np.array([0, 0, 1], dtype=float)
    v = random_unit_vector()
    dot_zv = np.dot(z_axis, v)
    if np.allclose(v, z_axis):
        return R.from_rotvec([0, 0, 0])
    elif np.allclose(v, -z_axis):
        return R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
    else:
        axis = np.cross(z_axis, v)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot_zv)
        return R.from_rotvec(axis * angle)





def generate_dataset_spheroid_3body(N_data, dist_min, dist_max, no_overlap_thresh=0.02):
    """
    Generate N_data samples of three-body interactions.
    
    Particle #1 (target) is fixed at the origin.
    Particles #2 and #3 (sources) are randomly placed (with biased distance)
    """
    acc = "fine"
    # Load target (particle #1) geometry from files.
    boundary1 = np.loadtxt(f'/home/shihab/src/mfs/points/b_sphere_{acc}.txt', dtype=np.float64)  # boundary nodes
    source1 = np.loadtxt(f'/home/shihab/src/mfs/points/s_sphere_{acc}.txt', dtype=np.float64)
    print(f"Target: {boundary1.shape[0]} boundary nodes, {source1.shape[0]} source points")
    
    B_orig = build_B(boundary1, source1, np.zeros(3))
    t0 = time.time()
    B1_inv = np.linalg.pinv(B_orig)
    print(f"B-matrix built in {time.time()-t0:.2f} seconds.")
    
    X_features = []
    y_targets  = []
    n_accepted = 0
    rejected = 0
    index = 0

    def biased_distance():
        L = dist_max - dist_min
        u = np.random.uniform(0, 1)
        # Inverse CDF for a linearly decaying PDF.
        x = 2 - np.sqrt(4 - 3 * u)
        return dist_min + x * L

    while n_accepted < N_data:
        # For each source particle, generate a biased distance and random direction.
        dist_s = biased_distance()
        center_s = dist_s * random_unit_vector()

        boundary_s = boundary1 + center_s.reshape(1,3)
        source_s = source1 + center_s.reshape(1,3)

        dist_k = biased_distance()
        center_k = dist_k * random_unit_vector()
        boundary_k = boundary1 + center_k.reshape(1,3)
        source_k = source1 + center_k.reshape(1,3)

        index += 1


        # Compute minimal distances (target is at origin, identity orientation).
        min_dist_2 = np.sqrt(center_s[0]**2 + center_s[1]**2 + center_s[2]**2) - 2.0
        min_dist_3 = np.sqrt(center_k[0]**2 + center_k[1]**2 + center_k[2]**2) - 2.0
        min_dist_23 = np.sqrt((center_k[0] - center_s[0])**2 + (center_k[1] - center_s[1])**2 + (center_k[2] - center_s[2])**2) - 2.0

        if (min_dist_2 <= no_overlap_thresh or
            min_dist_3 <= no_overlap_thresh or
            min_dist_23 <= no_overlap_thresh):
            print(f"Rejected sample {index}: min_dists=({min_dist_2:.3f}, {min_dist_3:.3f}, {min_dist_23:.3f})")
            rejected += 1
            continue

        print(f"Iter {index}: d2={dist_s:.3f}, d3={dist_k:.3f}, "
              f"min_dists=({min_dist_2:.3f}, {min_dist_3:.3f}, {min_dist_23:.3f})")
        
        # Define force on each source.
        scalar = 6 * np.pi * -1.0 * 1.0
        force_on_s = random_unit_vector()* scalar
        torque_on_s = random_unit_vector() * scalar * 1.0

        b_list = [boundary1, boundary_s, boundary_k]
        s_list = [source1, source_s, source_k]
        F_ext_list = [np.zeros(3), force_on_s, np.zeros(3)]
        T_ext_list = [np.zeros(3), torque_on_s, np.zeros(3)]
        B_inv_list = [B1_inv, B1_inv, B1_inv]


        try:
            V_tilde_list = imp_mfs_mobility_vec(b_list, s_list, 
                                                F_ext_list, T_ext_list, B_inv_list,
                                                max_iter=200, tol=1e-7)
            M1 = source1.shape[0]
            solution_1 = V_tilde_list[0]
            velocity_1 = solution_1[3*M1 : 3*M1 + 3]
            omega_1    = solution_1[3*M1 + 3 : 3*M1 + 6]
            velocity_6d = np.concatenate([velocity_1, omega_1]) 

        except Exception as e:
            print(f"Error in sample {index}: {e}")
            continue

        # 16D
        feat_s = np.concatenate([
            center_s,
            [np.linalg.norm(center_s), min_dist_2],
            force_on_s, 
            torque_on_s
        ])

        feat_k = np.concatenate([
            center_k,
            [np.linalg.norm(center_k), min_dist_3]
        ])

        feature_i = np.concatenate([feat_s, feat_k])

        X_features.append(feature_i)
        y_targets.append(velocity_6d)  # 6D target

        n_accepted += 1
        if n_accepted % 5 == 0:
            print(f"{n_accepted} samples generated...")

    X_features = np.array(X_features)
    y_targets  = np.array(y_targets)
    print("Rejected samples %: ", rejected/N_data)
    return X_features, y_targets


def main():
    N_samples = 4000


    # Set the (biased) distance range between the target and source particles.
    dist_min = 2.04
    dist_max = 8.0

    # Generate the dataset.
    X, Y = generate_dataset_spheroid_3body(N_samples, dist_min, dist_max)

    print("Dataset shapes:")
    print(" Features:", X.shape)  # (N_samples, 24)
    print(" Targets:", Y.shape)   # (N_samples, 6)

    # Save the dataset to disk.
    import random
    tmp = random.randint(0, 1000)
    current_time = time.strftime("%H_%M", time.localtime())
    np.save(f"data/3b_cross/X_sphere_{current_time}_{tmp}.npy", X)
    np.save(f"data/3b_cross/Y_sphere_{current_time}_{tmp}.npy",  Y)
    print(f"Saved {N_samples} samples.")


if __name__ == "__main__":
    main()
