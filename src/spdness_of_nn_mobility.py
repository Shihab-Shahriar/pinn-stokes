import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.cluster import grow_cluster
#from mob_op_nn import NNMob
from mob_op_combined import NNMob
from src.mfs_utils import min_distance_two_ellipsoids

"""
Sphere: tested for P=20 at S=.1 (min eigen=0.00194) 

With fs term, P=20 ceases to be SPD at S=2.45 or below.
"""

def sphere():
    R = 1.0  
    S = .4   # separation between sphere centers equals R
    P = 20
    print(f"sphere: {P=}, {S=}")

    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/two_body_combined_model.pt"
    #two_body_F1 = "data/models/two_body_sphere_model_F1.pt"
    mob = NNMob(shape, self_path, two_body, #two_body_F1,
                nn_only=False, rpy_only=False)
    
    centers = grow_cluster(P, R, S)
    r = Rotation.identity().as_quat(scalar_first=False)
    orientations = np.tile(r, (P, 1))

    config = np.concatenate((centers, orientations), axis=1)
    assert config.shape == (P, 7)


    N = P*6
    M = np.zeros((N, N))
    for i in range(N):
        print("column ", i)
        delta = np.zeros(N)
        delta[i] = 1.0
        delta = delta.reshape(P, 6)
        v = mob.apply(config, delta, viscosity=1.0)
        M[:, i] = v.flatten()

    eigens = np.linalg.eigvals(M)
    print("Eigenvalues of M:", eigens)
    print("Is M SPD?", np.all(eigens > -1e-4))
    print("min eigen: ", eigens.min())

    # Symmetry check. tol can't be too low cause NN
    print("Is M symmetric?", np.allclose(M, M.T, atol=1e-4))
    return M, sorted(eigens)


def random_spheres_test():
    """
    Test mobility with randomly placed non-overlapping spheres at 10% volume fraction.
    """
    np.random.seed(42)  # For reproducibility
    # Setup parameters
    N = 20  # Number of spheres
    R = 1.0  # Radius of each sphere
    

    sphere_volume = N * (4/3) * np.pi * R**3
    vol_fac = .25
    total_volume = sphere_volume / vol_fac  
    box_size = np.power(total_volume, 1/3)
    
    print(f"Box size: {box_size} with {N} spheres of radius {R}")
    
    # Initialize NN mobility model
    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/two_body_sphere_model.pt"
    #two_body_F1 = "data/models/two_body_sphere_model_F1.pt"
    mob = NNMob(shape, self_path, two_body, #two_body_F1, 
                nn_only=False, rpy_only=False)
    
    # Generate random positions without overlap
    centers = np.zeros((N, 3))
    for i in range(N):
        overlap = True
        attempts = 0
        while overlap and attempts < 1000:
            # Random position within box
            pos = np.random.uniform(0, box_size, 3)
            overlap = False
            
            # Check overlap with existing spheres
            for j in range(i):
                dist = np.linalg.norm(pos - centers[j])
                if dist < 2*R:  # Overlapping if distance < 2*radius
                    overlap = True
                    break
            attempts += 1
        
        if attempts == 1000:
            print("Warning: Could not place all spheres without overlap")
            break
            
        centers[i] = pos

    print("Sphere centers generated")
    
    # Create orientations (identity rotation for all spheres)
    r = Rotation.identity().as_quat(scalar_first=False)
    orientations = np.tile(r, (N, 1))
    
    # Combine positions and orientations
    config = np.concatenate((centers, orientations), axis=1)
    assert config.shape == (N, 7)
    
    # Build mobility matrix
    dim = N * 6
    M = np.zeros((dim, dim))
    for i in range(dim):
        print(f"Computing column {i}/{dim}")
        delta = np.zeros(dim)
        delta[i] = 1.0
        delta = delta.reshape(N, 6)
        v = mob.apply(config, delta, viscosity=1.0)
        M[:, i] = v.flatten()
    
    # Check properties
    eigens = np.linalg.eigvals(M)
    print("Eigenvalues of M:", eigens)
    print("Is M SPD?", np.all(eigens > -1e-4))
    print("Min eigenvalue:", eigens.min())
    print("Shape:", M.shape)
    print("Is M symmetric?", np.allclose(M, M.T, atol=1e-4))
    
    return M, eigens


def prolate():
    shape = "prolateSpheroid"
    a,b,c = 1.0, 1.0, 3.0
    delta = 1.0

    mob = NNMob(shape,
                "/home/shihab/repo/data/models/self_interaction_model.pt", 
                "/home/shihab/repo/data/models/two_body_prolate_model.pt")
    
    df = pd.read_csv("/home/shihab/repo/data/reference_prolate.csv", float_precision="high",
                        header=0, index_col=False)

    # df = pd.read_csv("/home/shihab/repo/data/n100.csv", float_precision="high",
    #                     header=0, index_col=False)
    numParticles = df.shape[0]    

    config = df[["x","y","z","q_x","q_y","q_z","q_w"]].values

    # Check that the minimum separation is maintained
    centers = config[:, :3]
    orients = [Rotation.from_quat(config[i, 3:], scalar_first=False) for i in range(numParticles)]
    for i in range(numParticles):
        my_min = np.inf
        for j in range(i):
            dd = min_distance_two_ellipsoids(a,b,c,centers[i],orients[i],
                                        a,b,c,centers[j],orients[j])

            assert dd >= delta-1e-4, f"Separation violation: {dd} < {delta}"

            my_min = min(my_min, dd)
        print(f"Min distance for ellipsoid {i}: {my_min}")
    print("cluster created")
    del centers, orients

    P = numParticles
    N = P*6
    M = np.zeros((N, N))
    for i in range(N):
        print("column ", i)
        delta = np.zeros(N)
        delta[i] = 1.0
        delta = delta.reshape(P, 6)
        v = mob.apply(config, delta, viscosity=1.0)
        M[:, i] = v.flatten()

    eigens = np.linalg.eigvals(M)
    print("Eigenvalues of M:", eigens)
    print("Is M SPD?", np.all(eigens > -1e-4))
    print("min eigen: ", eigens.min())

    # Symmetry check. tol can't be too low cause NN
    print("Is M symmetric?", np.allclose(M, M.T, atol=1e-4))
    return M


def bryce():
    R = 1.0  
    shape = "sphere"
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/two_body_sphere_model.pt"
    two_body_F1 = "data/models/two_body_sphere_model_F1.pt"
    mob = NNMob(shape, self_path, two_body, two_body_F1,
                nn_only=False, rpy_only=False)

    
    with open("/home/shihab/repo/src/pos 1.csv", "r") as f:
        data = f.read()
        data = list(map(float, data.split(",")))

        assert len(data) %3 == 0
        P = len(data) // 3
        centers = np.zeros((P, 3))
        for i in range(P):
            centers[i, 0] = data[i*3]
            centers[i, 1] = data[i*3+1]
            centers[i, 2] = data[i*3+2]

    P = 90
    centers = centers[:P, :]
    
    print(centers[0])
    r = Rotation.identity().as_quat(scalar_first=False)
    orientations = np.tile(r, (P, 1))

    config = np.concatenate((centers, orientations), axis=1)
    assert config.shape == (P, 7)



    N = P*6
    M = np.zeros((N, N))
    for i in range(N):
        print("column ", i)
        delta = np.zeros(N)
        delta[i] = 1.0
        delta = delta.reshape(P, 6)
        v = mob.apply(config, delta, viscosity=1.0)
        M[:, i] = v.flatten()

    eigens = np.linalg.eigvals(M)
    print("Eigenvalues of M:", eigens)
    print("Is M SPD?", np.all(eigens > -1e-4))
    print("min eigen: ", eigens.min())
    print("Shape:", M.shape)

    # Symmetry check. tol can't be too low cause NN
    print("Is M symmetric?", np.allclose(M, M.T, atol=1e-4))
    return M, eigens


if __name__ == "__main__":
    M, eigens = sphere()

    for ii in range(len(eigens)):
        block = M[ii*6:(ii+1)*6, ii*6:(ii+1)*6]
        if not np.allclose(block, block.T, atol=1e-4):
            print(f"Block {ii} is not symmetric")
            break 

    
