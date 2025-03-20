import time
import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from mfs_utils import (build_B, createNewEllipsoid, random_unit_vector,
                       random_orientation_spheroid, get_QM_QN)


# NOTE THE SEEDS. Set since we don't expect to run same script multiple times here.
np.random.seed(42)
random.seed(43)


def generate_dataset_spheroid(shape, N_samples, a, b, c):
    boundary = np.loadtxt(f'/home/shihab/src/mfs/points/b_{shape}_{acc}.txt', dtype=np.float64)  # boundary nodes
    source = np.loadtxt(f'/home/shihab/src/mfs/points/s_{shape}_{acc}.txt', dtype=np.float64)  # source points
    N = boundary.shape[0]
    M = source.shape[0]
    print(f"{N=} boundary nodes, {M=} source points")

    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # center of mass
    scalar = 6*np.pi*1.0*min(a, b, c)
    B = build_B(boundary, source, center)
    B_inv = np.linalg.pinv(B)

    X_features = np.zeros((N_samples, 15))
    y_targets = np.zeros((N_samples, 6))

    for i in range(N_samples):
        if i % 200 == 0:
            print(f"{i=}")

        # Orientation is probably not necessary, since we have
        # random directional force. 
        orientation = random_orientation_spheroid()
        if shape == "prolateSpheroid" or shape == "oblateSpheroid":
            bnd, src = createNewEllipsoid(center, orientation, source, boundary)
            QM, QN = get_QM_QN(orientation, bnd.shape[0], src.shape[0])
            Bpp = QM @ B_inv @ QN.T
        else:
            bnd, src = boundary, source
            Bpp = B_inv

        F_ext = random_unit_vector() * scalar
        T_ext = random_unit_vector() * scalar * 0.4

        # F_ext = np.array([0.0, 0.0, -scalar], dtype=np.float64)
        # T_ext = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        F_tilde = np.concatenate([np.zeros(3*N), F_ext, T_ext])
        solution = Bpp @ F_tilde

        assert solution.shape == (3*M+6,)  
    
        V = solution[3*M:3*M+3]
        omega = solution[3*M+3:]

        # feature vector should be particle shape (a, b, c), orientation
        # force and torque (position ignored)
        orientation = orientation.as_matrix()[:, :2].flatten() # first 2 cols

        abc = np.array([a, b, c])
        feature = np.concatenate((abc, orientation, F_ext, T_ext))
        assert feature.shape == (15,)

        # target vector should be velocity and angular velocity
        target = np.concatenate([V, omega])

        X_features[i] = feature
        y_targets[i] = target

    return X_features, y_targets


if __name__ == "__main__":
    shape = "sphere"
    acc = "fine"
    axes_length = {
        "prolateSpheroid": (1.0, 1.0, 3.0),
        "oblateSpheroid":  (2.0, 2.0, 1.0),
        "sphere":          (1.0, 1.0, 1.0),
    }
    a, b, c = axes_length[shape]
    N_samples = 10_000
    X,y = generate_dataset_spheroid(shape, N_samples, a, b, c)

    t = time.localtime()
    current_time = time.strftime("%H:%M", t)+"_"+str(random.randint(0, 100))
    np.save(f"data/X_self_{shape}_{current_time}.npy", X)
    np.save(f"data/Y_self_{shape}_{current_time}.npy", y)
    print(f"Saved {N_samples} samples.")