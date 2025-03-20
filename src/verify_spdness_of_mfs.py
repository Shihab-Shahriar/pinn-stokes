"""
For two-body interactions, verify that the mobility matrix is SPD.

try this for all shapes: sphere, prolate, oblate, at different distances

observations:
- Even for sphere, at Xfine, the two-body mobility matrix is not SPD
- Closer distance leads to smaller condition number
- Farther distance leads to smaller magintude of negative eigenvalue (as expected)



At the end of this file, some important comments. DO NOT DELETE.
"""

import time
import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from mfs_utils import build_B
from create_dataset_2body import imp_mfs_mobility_vec

# --------------------------------------------------------------------


def get_6x6_mobility():
    shape = "sphere"
    axes_length = {
        "prolateSpheroid": (1.0, 1.0, 3.0),
        "oblateSpheroid":  (2.0, 2.0, 1.0),
        "sphere":          (1.0, 1.0, 1.0),
    }
    a, b, c = axes_length[shape]
    acc = "Xfine"
    boundary1 = np.loadtxt(f'points/b_{shape}_{acc}.txt', dtype=np.float64)  # boundary nodes
    source1 = np.loadtxt(f'points/s_{shape}_{acc}.txt', dtype=np.float64)  # source points
    N, M = boundary1.shape[0], source1.shape[0]
    
    print(f"N={N} boundary nodes, M={M} source points")
    print(f"{acc=}")

    B_orig = build_B(boundary1, source1, np.zeros(3))
    start_time = time.time()
    B1_inv = np.linalg.pinv(B_orig)
    print(f"Built B-matrix in {time.time() - start_time:.2f} seconds.")
    
    dist = -2.5  # NOTE the negative sign
    center2 = np.array([dist, 0.0, 0.0]) 
    
    boundary2 = boundary1.copy() + center2
    source2 = source1.copy() + center2

    B2_inv = B1_inv.copy()
    B_inv_list = [B1_inv, B2_inv]
    print(B_orig.shape, B1_inv.shape, B2_inv.shape)
    
    b_list = [boundary1, boundary2]
    s_list = [source1, source2]

    F_ext_list = [np.zeros(3), np.array([1, 0, 1])]
    T_ext_list = [np.zeros(3), np.zeros(3)]

    V_tilde_list = imp_mfs_mobility_vec(
        b_list, s_list, F_ext_list, T_ext_list, B_inv_list, 
        max_iter=200, tol=1e-9, print_interval=50
    )

    print(V_tilde_list[0][3*M : 3*M + 6])

    K = np.zeros((6, 6))
    for i in range(6):
        F_ext_list = [np.zeros(3), np.zeros(3)]
        T_ext_list = [np.zeros(3), np.zeros(3)]

        if i < 3:
            F_ext_list[1][i] = 1.0
        else:
            T_ext_list[1][i-3] = 1.0

        print(f"Force on sphere #2: {F_ext_list[1]}, Torque on sphere #2: {T_ext_list[1]}")
        
        V_tilde_list = imp_mfs_mobility_vec(
            b_list, s_list, F_ext_list, T_ext_list, B_inv_list, 
            max_iter=200, tol=1e-9, print_interval=50
        )

        vel6d = V_tilde_list[0][3*M : 3*M + 6]
        K[:, i] = vel6d

        total_force = np.concatenate([F_ext_list[1], T_ext_list[1]])
        print("dot:", np.dot(vel6d, total_force))

    print("Symmetric?", np.allclose(K, K.T, atol=1e-6))
    eigens = np.linalg.eigvalsh(K)
    print("Positive?", np.all(eigens >= -1e-6))
    # get condition number
    cond = np.linalg.cond(K)
    print("Condition number:", cond)

    np.set_printoptions(suppress=True)
    print(K)


    # print in scientific notation
    np.set_printoptions(formatter={'float_kind':'{:e}'.format})
    print("Eigenvalues:", eigens)

    return K


"""
The code below is what I used to verify MFS has exact same output as Helen's
threesphere code she shared with me (not townsend's one). 

In helen's code, force direction isn't defined for usual cartesian system,
rather in coordinate system relative to the particles. So a force 

Ff(i,1) = +1.0

means force applied on sphere i along direction "1" (line connecting the 
centers). Importantly, +1.0 means the force is _towards_ sphere j along this
line. This caused me quite some headache since I was defning force  
by x,y,z axis. At one instance, it was giving me a opposite sign *only* for 
omega, rest of the values and magnitudes remained exactly the same.

In code below, fix was simply placing sphere2 on negative x axis, so a positive
force along x-dir meant sphere2 was being pushed _towards_ sphere1.

Goal of the experiment was to validate my MFS code against Helen's. Now that the
velocities are proven to be exactly similar, next- we will extract mobility
matrices for both and compare them. 

Update 2 days later:
Velocities didn't prove to be exactly similar. When torque was applied, some glaring
issue was found, the direction along with torque direction for me was wrong.

One day later (Mar 07):
Even after solving my code, there was still sign difference between my code and helen
on get_a_single_vel(). Today I realized I was comparing rows, and I should have
compared against columns.

The issue is fixed now I believe. Both matrices completely - oh wait. Shouldn't
Helen's matrix that I have now be transposed?
"""

def get_a_single_vel():
    shape = "sphere"
    axes_length = {
        "prolateSpheroid": (1.0, 1.0, 3.0),
        "oblateSpheroid":  (2.0, 2.0, 1.0),
        "sphere":          (1.0, 1.0, 1.0),
    }
    a, b, c = axes_length[shape]
    acc = "Xfine"
    boundary1 = np.loadtxt(f'points/b_{shape}_{acc}.txt', dtype=np.float64)  # boundary nodes
    source1 = np.loadtxt(f'points/s_{shape}_{acc}.txt', dtype=np.float64)  # source points
    N, M = boundary1.shape[0], source1.shape[0]
    
    print(f"N={N} boundary nodes, M={M} source points")
    print(f"{acc=}")

    B_orig = build_B(boundary1, source1, np.zeros(3))
    start_time = time.time()
    B1_inv = np.linalg.pinv(B_orig)
    #print(f"Built B-matrix in {time.time() - start_time:.2f} seconds.")
    
    dist = -2.5  # NOTE the negative sign
    dir_vec = random_unit_vector()
    center2 = np.array([dist, 0.0, 0.0]) #dist * dir_vec
    print(f"Center2: {center2}")
    
    orientation2 = random_orientation_spheroid()
    boundary2, source2 = createNewEllipsoid(center2, orientation2, source1, boundary1)
    min_dist = min_distance_ellipsoids(a, b, c, center2, orientation2, n_starts=10)

    if min_dist <= 0.02: # TODO: think about this threshold
        print(f"Config too close: {min_dist}")
         

    QM, QN = get_QM_QN(orientation2, boundary2.shape[0], source2.shape[0])
    B2_inv = QM @ B1_inv @ QN.T
    B_inv_list = [B1_inv, B2_inv]
    #print(B_orig.shape, B1_inv.shape, B2_inv.shape)
    
    b_list = [boundary1, boundary2]
    s_list = [source1, source2]

    F_ext_list = [np.zeros(3), np.array([0, 0, 1])]
    T_ext_list = [np.zeros(3), np.array([0, 0, 0])]

    print("Force on sphere #2: ", F_ext_list[1])
    print("Torque on sphere #2: ", T_ext_list[1])

    V_tilde_list = imp_mfs_mobility_vec(
        b_list, s_list, F_ext_list, T_ext_list, B_inv_list, 
        max_iter=200, tol=1e-9, print_interval=50
    )

    print(V_tilde_list[0][3*M : 3*M + 6])
    
    v = map(str, list(V_tilde_list[0][3*M : 3*M + 6]))
    print(','.join(v))
    return V_tilde_list[0][3*M : 3*M + 6]




# output matrices

# mfs = np.array([
#     [2.98515007e-02,  6.39322853e-09, -9.29828996e-09, -5.76745198e-10,7.99338467e-09, 8.50689222e-09],
#     [-5.7194294181901924e-11,0.0176388143189311,-3.843513755431891e-10,-4.846312021665206e-10,6.892756418731025e-11,0.006383956934534444],
#     [8.974985727716112e-10,8.081135947722974e-10,0.017638814216487386,-8.831193787996227e-10,-0.006383955764887445,1.1162755131068704e-09],
#     [2.3930043121818294e-10,-3.020198290949478e-11,5.783949899500836e-11,0.0025549409921350203,1.400858185148738e-10,-1.7095559865599802e-10],
# ])

helenM = np.array([
    [2.9851e-02, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.7638814e-02, 0.0, 0.0, 0.0, 6.3839565573539534e-03],
    [0.0, 0.0, 1.76388144e-02, 0.0, -6.3839565573539534e-03, 0.0],
    [0.0, 0.0, 0.0, 2.55494099e-03, 0.0, 0.0],
    [0.0, 0.0, 6.3839565573539534e-03, 0.0, -1.1076507077248238e-03, 0.0],
    [0.0, -6.3839564835124893e-03, 0.0, 0.0, 0.0, -1.107650707e-03],
])

helenM = helenM.T # transpose since vel are columns, not rows



print("Symmetric?", np.allclose(helenM, helenM.T, atol=1e-6))
eigens = np.linalg.eigvalsh(helenM)
print("Positive?", np.all(eigens >= -1e-6))

np.set_printoptions(precision=5, suppress=True)
print(helenM)
print("---------------------------------------")
K = get_6x6_mobility()


print("All close: ", np.allclose(helenM, K, rtol=1e-4))
print("All close with helen.T: ", np.allclose(helenM.T, K, rtol=1e-4))
