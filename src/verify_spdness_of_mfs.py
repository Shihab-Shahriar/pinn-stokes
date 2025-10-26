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
    root = "/home/shihab/src/mfs/"

    boundary1 = np.loadtxt(f'{root}points/b_{shape}_{acc}.txt', dtype=np.float64)  # boundary nodes
    source1 = np.loadtxt(f'{root}points/s_{shape}_{acc}.txt', dtype=np.float64)  # source points
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
        max_iter=200, tol=1e-8, 
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
            max_iter=200, tol=1e-8
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


def get_18x18_mobility_equilateral():
    """
    Build a 3-sphere configuration (radius=1.0) at the vertices of an
    equilateral triangle in the xy-plane with center-to-center distance 2.5
    (surface separation = 0.5 since radius=1) and compute the full 18x18
    mobility matrix mapping [F1,T1,F2,T2,F3,T3] -> [U1,Ω1,U2,Ω2,U3,Ω3].

    Returns
    -------
    K : (18, 18) np.ndarray
        The assembled mobility matrix. Also prints symmetry and SPD checks.
    """
    # Geometry/shape setup (reuse the same discretization for all three spheres)
    shape = "sphere"
    acc = "fine"
    root = "/home/shihab/src/mfs/"

    # Load boundary and source nodes for a single unit sphere (radius=1)
    boundary = np.loadtxt(f"{root}points/b_{shape}_{acc}.txt", dtype=np.float64)
    source = np.loadtxt(f"{root}points/s_{shape}_{acc}.txt", dtype=np.float64)
    N, M = boundary.shape[0], source.shape[0]
    print(f"[3-body] N={N} boundary nodes, M={M} source points, acc={acc}")

    # Precompute (pseudo-)inverse of the single-body B-matrix once; all 3 spheres
    # are identical and unrotated, so the inverse is reused.
    B = build_B(boundary, source, np.zeros(3))
    B_inv = np.linalg.pinv(B)

    # Place sphere centers at an equilateral triangle of side L in the xy-plane.
    L = 2.1  # center-to-center spacing
    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([L, 0.0, 0.0])
    c3 = np.array([0.5 * L, 0.5 * np.sqrt(3.0) * L, 0.0])
    centers = [c1, c2, c3]

    # Shift the template nodes to each center; reuse B_inv for all since same shape.
    b_list = [boundary + c for c in centers]
    s_list = [source + c for c in centers]
    B_inv_list = [B_inv, B_inv, B_inv]

    # Assemble the 18x18 mobility by applying unit loads for each DOF and
    # concatenating the resulting rigid-body velocities from all three spheres.
    K = np.zeros((18, 18), dtype=np.float64)
    zeros3 = np.zeros(3)

    for j in range(18):
        # Determine which body and which component this column corresponds to
        body = j // 6              # 0, 1, or 2
        local = j % 6              # 0..2 => Fx,Fy,Fz; 3..5 => Tx,Ty,Tz

        # Initialize external forces/torques
        F_ext_list = [zeros3.copy(), zeros3.copy(), zeros3.copy()]
        T_ext_list = [zeros3.copy(), zeros3.copy(), zeros3.copy()]

        if local < 3:
            F_ext_list[body][local] = 1.0
        else:
            T_ext_list[body][local - 3] = 1.0

        # Solve for velocities given this unit load
        V_tilde_list = imp_mfs_mobility_vec(
            b_list, s_list, F_ext_list, T_ext_list, B_inv_list,
            max_iter=200, tol=1e-8
        )

        # Extract rigid-body 6-velocities [U(3), Omega(3)] for each body and
        # stack them as a single 18-vector in the order (1,2,3)
        col = np.zeros(18, dtype=np.float64)
        for k in range(3):
            vel6 = V_tilde_list[k][3 * M : 3 * M + 6]
            col[6 * k : 6 * (k + 1)] = vel6

        K[:, j] = col

    # Report symmetry and SPD (up to small numerical tolerance)
    is_sym = np.allclose(K, K.T, atol=1e-5)
    eigs = np.linalg.eigvalsh(K)
    is_spd = np.all(eigs >= -1e-5)
    cond = np.linalg.cond(K)

    print(f"[3-body] Symmetric? {is_sym}")
    print(f"[3-body] Positive (SPD up to tol)? {is_spd}")
    print(f"[3-body] Condition number: {cond:.3e}")

    return K, eigs


def get_6N6N_mobility_from_csv(csv_path: str = "tmp/reference_sphere_2.0.csv"):
    """
    Read a sphere configuration (radius=1.0) from CSV and compute the full
    6N×6N mobility matrix using MFS. Also perform a quick validation by
    applying the forces/torques from the CSV and comparing the predicted
    velocities to the CSV velocities.

    CSV is expected to contain columns:
      - positions: x, y, z
      - quaternions (unused for spheres): q_x, q_y, q_z, q_w
      - external loads: f_x, f_y, f_z, t_x, t_y, t_z
      - velocities: v_x, v_y, v_z, w_x, w_y, w_z

    Returns
    -------
    K : (6N, 6N) np.ndarray
        The assembled mobility matrix.
    eigs : (6N,) np.ndarray
        Eigenvalues of K (for SPD check).
    """
    import pandas as pd

    # Load particle configuration and reference data
    df = pd.read_csv(csv_path)
    pos = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    forces = df[["f_x", "f_y", "f_z", "t_x", "t_y", "t_z"]].to_numpy(dtype=np.float64)
    vel_ref = df[["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]].to_numpy(dtype=np.float64)

    Np = pos.shape[0]
    assert forces.shape == (Np, 6)
    assert vel_ref.shape == (Np, 6)
    print(f"[csv] Loaded {Np} spheres from {csv_path}")

    # Geometry/shape setup (same discretization for all spheres)
    shape = "sphere"
    acc = "fine"
    root = "/home/shihab/src/mfs/"

    boundary = np.loadtxt(f"{root}points/b_{shape}_{acc}.txt", dtype=np.float64)
    source = np.loadtxt(f"{root}points/s_{shape}_{acc}.txt", dtype=np.float64)
    N, M = boundary.shape[0], source.shape[0]
    print(f"[csv] N={N} boundary nodes, M={M} source points, acc={acc}")

    # Single-body B inverse, reused for all identical spheres
    B = build_B(boundary, source, np.zeros(3))
    B_inv = np.linalg.pinv(B)

    # Translate nodes to particle centers
    b_list = [boundary + c for c in pos]
    s_list = [source + c for c in pos]
    B_inv_list = [B_inv for _ in range(Np)]

    # Assemble 6N×6N mobility matrix by applying unit loads
    K = np.zeros((6 * Np, 6 * Np), dtype=np.float64)
    zeros3 = np.zeros(3)

    for j in range(6 * Np):
        print("Col:", j)
        body = j // 6
        local = j % 6

        F_ext_list = [zeros3.copy() for _ in range(Np)]
        T_ext_list = [zeros3.copy() for _ in range(Np)]

        if local < 3:
            F_ext_list[body][local] = 1.0
        else:
            T_ext_list[body][local - 3] = 1.0

        V_tilde_list = imp_mfs_mobility_vec(
            b_list, s_list, F_ext_list, T_ext_list, B_inv_list,
            max_iter=200, tol=1e-8,
        )

        col = np.zeros(6 * Np, dtype=np.float64)
        for k in range(Np):
            vel6 = V_tilde_list[k][3 * M : 3 * M + 6]
            col[6 * k : 6 * (k + 1)] = vel6

        K[:, j] = col

    # Symmetry/SPD diagnostics
    is_sym = np.allclose(K, K.T, atol=1e-5)
    eigs = np.linalg.eigvals(K)
    is_spd = np.all(eigs >= -1e-5)
    cond = np.linalg.cond(K)
    print(f"[csv] Symmetric? {is_sym}")
    print(f"[csv] Positive (SPD up to tol)? {is_spd}")
    print(f"[csv] Condition number: {cond:.3e}")

    # Quick validation: use CSV forces/torques and compare velocities
    F_ext_list = [forces[i, :3].copy() for i in range(Np)]
    T_ext_list = [forces[i, 3:].copy() for i in range(Np)]

    V_tilde_list = imp_mfs_mobility_vec(
        b_list, s_list, F_ext_list, T_ext_list, B_inv_list,
        max_iter=200, tol=1e-8,
    )

    vel_pred = np.zeros_like(vel_ref)
    for i in range(Np):
        vel_pred[i] = V_tilde_list[i][3 * M : 3 * M + 6]

    # Error metrics
    abs_err = np.abs(vel_pred - vel_ref)
    max_abs_per_comp = abs_err.max(axis=0)
    rmse_per_comp = np.sqrt(np.mean((vel_pred - vel_ref) ** 2, axis=0))
    max_abs = abs_err.max()
    rmse = np.sqrt(np.mean((vel_pred - vel_ref) ** 2))

    np.set_printoptions(precision=5, suppress=True)
    print("[csv] Max abs error per component:", max_abs_per_comp)
    print("[csv] RMSE per component:", rmse_per_comp)
    print(f"[csv] Overall max abs error: {max_abs:.3e}")
    print(f"[csv] Overall RMSE: {rmse:.3e}")

    return K, eigs

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

def helen():
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

if __name__ == '__main__':
    #K = get_6x6_mobility()
    #K, eigs = get_18x18_mobility_equilateral()
    K, eigs = get_6N6N_mobility_from_csv("tmp/reference_sphere_0.1.csv")
    np.set_printoptions(precision=5, suppress=True)
    print(eigs)