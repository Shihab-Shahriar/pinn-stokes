"""
Generate a three-body interactions dataset for spheroids.

Target (particle #1) is fixed at the origin (with the reference orientation),
and two other particles are randomly placed around it. Random forces and torques
are applied to these two particles.

The feature vector is:
  - For each source particle (2 and 3), we record:
      * Center coordinates (3)
      * The translation distance (norm of center) (1)
      * The minimal distance (min_dist) between the source and the target (1)
      * The orientation quaternion (4) [x,y,z,w]
      * The force vector (3)
      * The torque vector (3)
    Total per source: 3+1+1+4+3+3 = 15. For two sources, features are 30D.
      
The target is the 6D velocity (translation and rotation) of particle #1,
computed by solving the multi-particle IMP-MFS mobility problem.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from mfs import imp_mfs_mobility_vec
from mfs_utils import (build_B, get_QM_QN, createNewEllipsoid, 
                       random_unit_vector)
#from ..utils import save_multiple_ellipsoids_legacy_vtk


def solve_three_interaction_spheroid(
    B1_inv, b_list, s_list, F_ext_list, T_ext_list
):
    """
    High-level routine to solve the three-body mobility problem.
    
    Particle #1 (target) is at the origin.
    Particles #2 and #3 (sources) have external forces and torques.
    
    Returns the 6D velocity (translation and rotation) of particle #1.
    """

    # For spheres, orientation is irrelevant, so B_inv is the same for all.
    B_inv_list = [B1_inv, B1_inv, B1_inv]

    V_tilde_list = imp_mfs_mobility_vec(b_list, s_list, F_ext_list, T_ext_list, B_inv_list,
                                        max_iter=200, tol=1e-7)
    M1 = s_list[0].shape[0]
    solution_1 = V_tilde_list[0]
    velocity_1 = solution_1[3*M1 : 3*M1 + 3]
    omega_1    = solution_1[3*M1 + 3 : 3*M1 + 6]
    return np.concatenate([velocity_1, omega_1])  # shape (6,)


def generate_dataset_spheroid_3body(shape, N_data, a, b, c,
                                    dist_min, dist_max, no_overlap_thresh=0.02):
    
    """
    Generate a dataset of three-body interactions for spheroids.

    Force on particle #t (target) and #s(source), k is 3rd body whose force don't matter.
    """
    acc = "fine"
    # Load target (particle #1) geometry from files.
    boundary1 = np.loadtxt(f'/home/shihab/src/mfs/points/b_{shape}_{acc}.txt', dtype=np.float64)
    source1 = np.loadtxt(f'/home/shihab/src/mfs/points/s_{shape}_{acc}.txt', dtype=np.float64)
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
        index += 1
        # For each source particle, generate a biased distance and random direction.
        dist_s = biased_distance()
        center_s = dist_s * random_unit_vector()
        orientation_s = R.identity() # Identity rotation for sphere
        boundary_s, source_s = createNewEllipsoid(center_s, orientation_s, source1, boundary1)

        dist_k = biased_distance()

        center_k = dist_k * random_unit_vector()

        if n_accepted%10==2: # same plane for every 10 samples
            dim = n_accepted % 3
            center_s[dim] = 0
            center_k[dim] = 0
        if n_accepted%10==1: # equal distance for every 10 samples
            center_k = dist_k * random_unit_vector()


        orientation_k = R.identity() # Identity rotation for sphere
        boundary_k, source_k = createNewEllipsoid(center_k, orientation_k, source1, boundary1)

        # save_multiple_ellipsoids_legacy_vtk(
        #     f"out/3body_{index}.vtk", 
        #     [boundary1, boundary2, boundary3], 
        #     [source1, source2, source3]
        # )

        # Compute minimal distances for spheres (radius = a)
        min_dist_2 = np.linalg.norm(center_s) - 2*a
        min_dist_3 = np.linalg.norm(center_k) - 2*a
        min_dist_23 = np.linalg.norm(center_k - center_s) - 2*a

        if (min_dist_2 <= no_overlap_thresh or
            min_dist_3 <= no_overlap_thresh or
            min_dist_23 <= no_overlap_thresh):
            print(f"Rejected sample {index}: min_dists=({min_dist_2:.3f}, {min_dist_3:.3f}, {min_dist_23:.3f})")
            rejected += 1
            continue

        print(f"Iter {index}: d2={dist_s:.3f}, d3={dist_k:.3f}, "
              f"min_dists=({min_dist_2:.3f}, {min_dist_3:.3f}, {min_dist_23:.3f})")
        
        # Define force and torque on each source.
        scalar = 6 * np.pi * 1.0 * min(a, b, c)
        # force_on_t = random_unit_vector() * scalar
        # torque_on_t = random_unit_vector() * scalar * 1.0
        force_on_t = np.zeros(3)
        torque_on_t = np.zeros(3)

        force_on_s = random_unit_vector() * scalar
        torque_on_s = random_unit_vector() * scalar * 1.0

        if n_accepted % 10 == 3: # every 10 samples, make forces equal
            force_on_s = force_on_t.copy()
            torque_on_s = torque_on_t.copy()

        force_on_k = np.zeros(3)  # No force on particle #k
        torque_on_k = np.zeros(3) # No torque on particle #k

        force_list = [force_on_t, force_on_s, force_on_k]
        torque_list = [torque_on_t, torque_on_s, torque_on_k]

        b_list = [boundary1, boundary_s, boundary_k]
        s_list = [source1, source_s, source_k]

        try:
            velocity_6d = solve_three_interaction_spheroid(
                B1_inv, b_list, s_list,
                force_list, torque_list
            )
        except Exception as e:
            print(f"Error in sample {index}: {e}")
            # Optionally, save error case for visualization:
            # save_multiple_ellipsoids_legacy_vtk(
            #     f"out/ERROR_ellipsoids{index}.vtk", 
            #     [boundary1, boundary2, boundary3], 
            #     [source1, source2, source3]
            # )
            continue

        # 6D+11D+11D = 28D feature vector
        feat_t = np.concatenate([
            # [0, 0, 0],  # Target at origin
            # [0, 0],     # Target distance and min_dist
            force_on_t,
            torque_on_t
        ])
        feat_s = np.concatenate([
            center_s,
            [np.linalg.norm(center_s), min_dist_2],
            force_on_s, 
            torque_on_s
        ])
        feat_k = np.concatenate([
            center_k,
            [np.linalg.norm(center_k), min_dist_3],
            force_on_k,
            torque_on_k
        ])

        feature_i = np.concatenate([feat_t, feat_s, feat_k])
        assert feature_i.shape == (28,)

        X_features.append(feature_i)
        y_targets.append(velocity_6d)  # 6D target

        n_accepted += 1
        if n_accepted % 5 == 0:
            print(f"{n_accepted} samples generated...")

    X_features = np.array(X_features)
    y_targets  = np.array(y_targets)
    print("Rejected samples %: ", rejected/N_data if N_data > 0 else 0)
    return X_features, y_targets


def main():
    N_samples = 4000
    shape = "sphere"
    a, b, c = 1.0, 1.0, 1.0

    # Set the (biased) distance range between the target and source particles.
    dist_min = 2.04
    dist_max = 8.0

    print("Generating dataset for", shape)
    print("Axes lengths:", a, b, c)
    print("Number of samples:", N_samples)
    print("Distance range:", dist_min, dist_max)

    # Generate the dataset.
    X, Y = generate_dataset_spheroid_3body(shape, N_samples, a, b, c, 
                                           dist_min, dist_max)
    print("Dataset shapes:")
    print(" Features:", X.shape)
    print(" Targets:", Y.shape)

    # Save the dataset to disk.
    import random
    tmp = random.randint(0, 1000)
    current_time = time.strftime("%H_%M", time.localtime())
    np.save(f"data/X_3body_sphere_{current_time}_{tmp}.npy", X)
    np.save(f"data/Y_3body_sphere_{current_time}_{tmp}.npy",  Y)
    print(f"Saved {N_samples} samples.")


if __name__ == "__main__":
    main()
