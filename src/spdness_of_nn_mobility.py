import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import sys

from benchmarks.cluster import grow_cluster
from mob_op_nn import NNMob
from src.mfs_utils import min_distance_two_ellipsoids

"""
Sphere: tested for P=20 at S=.1 (min eigen=0.00194) 

"""

def sphere():
    R = 1.0  
    S = .1   # separation between sphere centers equals R
    P = 20
    print(P, S)
    mob = NNMob(
                "/home/shihab/repo/experiments/self_interaction.wt", 
                "/home/shihab/repo/experiments/sphere_2body.wt")
    
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
        v = mob.apply(config, delta)
        M[:, i] = v.flatten()

    eigens = np.linalg.eigvals(M)
    print("Eigenvalues of M:", eigens)
    print("Is M SPD?", np.all(eigens > -1e-4))
    print("min eigen: ", eigens.min())

    # Symmetry check. tol can't be too low cause NN
    print("Is M symmetric?", np.allclose(M, M.T, atol=1e-4))


def prolate():
    shape = "prolateSpheroid"
    a,b,c = 1.0, 1.0, 3.0
    delta = 1.0

    mob = NNMob(shape,
                "/home/shihab/repo/experiments/self_interaction.wt", 
                "/home/shihab/repo/experiments/prolate_2body.wt")
    
    df = pd.read_csv("/home/shihab/repo/data/reference_ellipsoid.csv", float_precision="high",
                        header=0, index_col=False)
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
        v = mob.apply(config, delta)
        M[:, i] = v.flatten()

    eigens = np.linalg.eigvals(M)
    print("Eigenvalues of M:", eigens)
    print("Is M SPD?", np.all(eigens > -1e-4))
    print("min eigen: ", eigens.min())

    # Symmetry check. tol can't be too low cause NN
    print("Is M symmetric?", np.allclose(M, M.T, atol=1e-4))
    return M





if __name__ == "__main__":
    M = prolate()