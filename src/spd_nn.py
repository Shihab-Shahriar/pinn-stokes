import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.cluster import grow_cluster
from mob_op_nn import NNMob
from src.mfs_utils import min_distance_two_ellipsoids

def build_and_analyze_mobility_matrix(mob, config):
    """
    Builds the mobility matrix M and performs analysis.
    """
    P = config.shape[0]
    N = P * 6
    M = np.zeros((N, N))
    for i in range(N):
        print(f"Computing column {i}/{N}")
        delta = np.zeros(N)
        delta[i] = 1.0
        delta = delta.reshape(P, 6)
        v = mob.apply(config, delta, viscosity=1.0)
        M[:, i] = v.flatten()

    eigens = np.linalg.eigvals(M)
    print("Eigenvalues of M:", eigens)
    print("Is M SPD?", np.all(eigens > -1e-4))
    print("Min eigenvalue:", eigens.min())
    print("Shape:", M.shape)
    print("Is M symmetric?", np.allclose(M, M.T, atol=1e-4))
    return M, eigens

def get_sphere_mob_op(nn_only=False, rpy_only=False):
    """
    Initializes and returns the NNMob operator for spheres.
    """
    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/two_body_sphere_model.pt"
    two_body_F1 = "data/models/two_body_sphere_model_F1.pt"
    return NNMob(shape, self_path, two_body, two_body_F1,
                 nn_only=nn_only, rpy_only=rpy_only)

def create_identity_orientations(P):
    """
    Creates an array of identity orientations for P particles.
    """
    r = Rotation.identity().as_quat(scalar_first=False)
    return np.tile(r, (P, 1))

def sphere_test():
    R = 1.0
    S = 0.48
    P = 20
    print(f"--- Running sphere_test: {P=}, {S=} ---")

    mob = get_sphere_mob_op()
    centers = grow_cluster(P, R, S)
    orientations = create_identity_orientations(P)
    config = np.concatenate((centers, orientations), axis=1)

    return build_and_analyze_mobility_matrix(mob, config)

def random_spheres_test():
    """
    Test mobility with randomly placed non-overlapping spheres.
    """
    np.random.seed(42)
    P = 20
    R = 1.0
    sphere_volume = P * (4/3) * np.pi * R**3
    vol_frac = 0.25
    total_volume = sphere_volume / vol_frac
    box_size = np.power(total_volume, 1/3)
    print(f"Box size: {box_size} with {P} spheres of radius {R}")

    mob = get_sphere_mob_op()

    centers = np.zeros((P, 3))
    for i in range(P):
        overlap = True
        attempts = 0
        while overlap and attempts < 1000:
            pos = np.random.uniform(0, box_size, 3)
            overlap = False
            for j in range(i):
                if np.linalg.norm(pos - centers[j]) < 2 * R:
                    overlap = True
                    break
            attempts += 1
        if attempts == 1000:
            print("Warning: Could not place all spheres without overlap.")
            break
        centers[i] = pos
    print("Sphere centers generated")

    orientations = create_identity_orientations(P)
    config = np.concatenate((centers, orientations), axis=1)

    return build_and_analyze_mobility_matrix(mob, config)

def prolate_test():
    print("--- Running prolate_test ---")
    shape = "prolateSpheroid"
    a, b, c = 1.0, 1.0, 3.0
    delta = 1.0

    mob = NNMob(shape,
                "data/models/self_interaction_model.pt",
                "data/models/two_body_prolate_model.pt")

    df = pd.read_csv("data/reference_prolate.csv", float_precision="high")
    config = df[["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].values
    num_particles = config.shape[0]

    centers = config[:, :3]
    orients = [Rotation.from_quat(config[i, 3:], scalar_first=False) for i in range(num_particles)]
    for i in range(num_particles):
        my_min = np.inf
        for j in range(i):
            dd = min_distance_two_ellipsoids(a, b, c, centers[i], orients[i],
                                             a, b, c, centers[j], orients[j])
            assert dd >= delta - 1e-4, f"Separation violation: {dd} < {delta}"
            my_min = min(my_min, dd)
        print(f"Min distance for ellipsoid {i}: {my_min}")
    print("Cluster integrity verified.")

    return build_and_analyze_mobility_matrix(mob, config)

def bryce_test():
    print("--- Running bryce_test ---")
    mob = get_sphere_mob_op()

    with open("src/pos 1.csv", "r") as f:
        data = list(map(float, f.read().split(",")))

    P = len(data) // 3
    centers = np.array(data).reshape(P, 3)

    # Original file had a bug here, hardcoding P=90.
    # This version uses the actual number of particles from the file.
    # If the original behavior is desired, uncomment the following lines:
    # P = 90
    # centers = centers[:P, :]

    print(f"Loaded {P} particle positions.")

    orientations = create_identity_orientations(P)
    config = np.concatenate((centers, orientations), axis=1)

    return build_and_analyze_mobility_matrix(mob, config)

if __name__ == "__main__":
    # The original file called random_spheres_test()
    M, eigens = random_spheres_test()
