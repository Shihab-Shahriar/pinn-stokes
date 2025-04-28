"""
Generate a three-body interactions dataset for spheroids.

Target (particle #1) is fixed at the origin (with the reference orientation),
second particle placed on x-axis at a random distance. Third one is randomly 
placed, with randomly directed unit force applied.

The feature vector is 24D:
  - For each source, we record:
      * Center coordinates (3)
      * The translation distance (norm of center) (1)
      * The minimal distance (min_dist) between the source and the target (1)
      * The orientation quaternion (4) [x,y,z,w]
      * The force vector (3)
    Total per source: 3+1+1+4+3 = 12. For two sources, features are 24D.
      
The target particles' 6D velocity (translation and rotation) is computed by solving
the multi-particle IMP-MFS mobility problem.
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


def solve_three_interaction_spheroid(
    boundary1, source1, B1_inv, B_orig,
    boundary2, source2, orientation2, force2, torque_on_2,
    boundary3, source3, orientation3, force3
):
    """
    High-level routine to solve the three-body mobility problem.
    
    Particle #1 (target) is at the origin.
    Particles #2 and #3 (sources) have external forces (no torque applied).
    
    Returns the 6D velocity (translation and rotation) of particle #1.
    """
    F_ext_list = [np.zeros(3), force2, force3]
    T_ext_list = [np.zeros(3), torque_on_2, np.zeros(3)]  # no torque on any particle

    # Compute transformed B-inverses for the sources.
    QM2, QN2 = get_QM_QN(orientation2, boundary2.shape[0], source2.shape[0])
    B2_inv = QM2 @ B1_inv @ QN2.T
    QM3, QN3 = get_QM_QN(orientation3, boundary3.shape[0], source3.shape[0])
    B3_inv = QM3 @ B1_inv @ QN3.T

    B_inv_list = [B1_inv, B2_inv, B3_inv]
    b_list = [boundary1, boundary2, boundary3]
    s_list = [source1, source2, source3]

    V_tilde_list = imp_mfs_mobility_vec(b_list, s_list, F_ext_list, T_ext_list, B_inv_list,
                                        max_iter=200, tol=1e-7)
    M1 = source1.shape[0]
    solution_1 = V_tilde_list[0]
    velocity_1 = solution_1[3*M1 : 3*M1 + 3]
    omega_1    = solution_1[3*M1 + 3 : 3*M1 + 6]
    return np.concatenate([velocity_1, omega_1])  # shape (6,)


def generate_dataset_spheroid_3body(shape, N_data, a, b, c,
                                    dist_min, dist_max, no_overlap_thresh=0.02):
    """
    Generate N_data samples of three-body interactions.
    
    Particle #1 (target) is fixed at the origin.
    Particles #2 and #3 (sources) are randomly placed (with biased distance)
    and rotated. Each sample is accepted only if the minimal distance
    between any pair (target-source2, target-source3, source2-source3)
    is greater than no_overlap_thresh.
    
    For each source, features are (center (3), distance (1), min_dist (1),
    orientation quaternion (4), force (3)) -> 12D per source.
    The total feature vector is 24D.
    
    The target output is the 6D (velocity, angular velocity) of the target.
    """
    acc = "fine"
    # Load target (particle #1) geometry from files.
    boundary1 = np.loadtxt(f'/home/shihab/src/mfs/points/b_{shape}_{acc}.txt', dtype=np.float64)  # boundary nodes
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
        # For each source particle, generate a biased distance and random direction.
        dist2 = biased_distance()
        center2 = dist2 * random_unit_vector()
        if center2[0] < 0:
            continue  # skip negative x-coordinates
        orientation2 = random_orientation_spheroid()
        boundary2, source2 = createNewEllipsoid(center2, orientation2, source1, boundary1)

        dist3 = biased_distance()
        #center3 = dist3 * random_unit_vector()
        center3 = np.array([dist3, 0, 0], dtype=float)
        orientation3 = random_orientation_spheroid()
        boundary3, source3 = createNewEllipsoid(center3, orientation3, source1, boundary1)

        index += 1
        # save_multiple_ellipsoids_legacy_vtk(
        #     f"out/3body_{index}.vtk", 
        #     [boundary1, boundary2, boundary3], 
        #     [source1, source2, source3]
        # )

        # Compute minimal distances (target is at origin, identity orientation).
        min_dist_2 = min_distance_ellipsoids(a, b, c, center2, orientation2, n_starts=10)
        min_dist_3 = min_distance_ellipsoids(a, b, c, center3, orientation3, n_starts=10)
        # For source2 and source3, compute relative configuration.
        rel_center = center3 - center2
        rel_orientation = orientation2.inv() * orientation3
        min_dist_23 = min_distance_ellipsoids(a, b, c, rel_center, rel_orientation, n_starts=10)

        if (min_dist_2 <= no_overlap_thresh or
            min_dist_3 <= no_overlap_thresh or
            min_dist_23 <= no_overlap_thresh):
            print(f"Rejected sample {index}: min_dists=({min_dist_2:.3f}, {min_dist_3:.3f}, {min_dist_23:.3f})")
            rejected += 1
            continue

        print(f"Iter {index}: d2={dist2:.3f}, d3={dist3:.3f}, "
              f"min_dists=({min_dist_2:.3f}, {min_dist_3:.3f}, {min_dist_23:.3f})")
        
        # Define force on each source.
        scalar = 6 * np.pi * -1.0 * min(a, b, c)
        force_on_2 = random_unit_vector()* scalar
        torque_on_2 = random_unit_vector() * scalar *.4

        #force_on_3 = random_unit_vector() * scalar
        force_on_3 = np.zeros(3)  # no force on source3

        try:
            velocity_6d = solve_three_interaction_spheroid(
                boundary1, source1, B1_inv, B_orig,
                boundary2, source2, orientation2, force_on_2, torque_on_2,
                boundary3, source3, orientation3, force_on_3
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

        # 16D
        feat_source2 = np.concatenate([
            center2,
            [np.linalg.norm(center2), min_dist_2],
            [np.linalg.norm(center2 - center3)],
            orientation2.as_quat(scalar_first=False),
            force_on_2, torque_on_2
        ])
        # 12D
        feat_source3 = np.concatenate([
            center3,
            [np.linalg.norm(center3), min_dist_3],
            orientation3.as_quat(scalar_first=False),
            force_on_3
        ])

        feature_i = np.concatenate([feat_source2, feat_source3])

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
    N_samples = 8000
    shape = "prolateSpheroid"  
    axes_length = {
        "prolateSpheroid": (1.0, 1.0, 3.0),
        "oblateSpheroid":  (2.0, 2.0, 1.0),
        "sphere":          (1.0, 1.0, 1.0),
    }
    a, b, c = axes_length[shape]

    # Set the (biased) distance range between the target and source particles.
    dist_min = 2.04
    dist_max = 8.0

    # Generate the dataset.
    X, Y = generate_dataset_spheroid_3body(shape, N_samples, a, b, c, 
                                           dist_min, dist_max)
    print("Dataset shapes:")
    print(" Features:", X.shape)  # (N_samples, 24)
    print(" Targets:", Y.shape)   # (N_samples, 6)

    # Save the dataset to disk.
    current_time = time.strftime("%H_%M", time.localtime())
    np.save(f"data/X_3body_torque_tri_s3onXaxis_posx{current_time}.npy", X)
    np.save(f"data/Y_3body_torque_tri_s3onXaxis_posx{current_time}.npy",  Y)
    print(f"Saved {N_samples} samples.")


if __name__ == "__main__":
    main()
