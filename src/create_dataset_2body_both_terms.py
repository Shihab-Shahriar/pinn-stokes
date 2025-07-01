"""
Create a set of two-body interactions dataset
for a pair of prolate spheroids.

Using twobody.py as starting point.

We will generalize to other shapes later.

Viscosity is assumed to be 1.0 throughout.
"""

import time
import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from mfs_utils import (build_B, createNewEllipsoid, get_QM_QN, 
                      min_distance_ellipsoids, random_unit_vector,
                      random_orientation_spheroid)
from create_dataset_self import get_self_vel

# np.random.seed(42)
# random.seed(43)
# --------------------------------------------------------------------

def G_vec(r):
    """
    Vectorized Oseen tensor for Stokes flow in 3D.
    r is shape (N, M, 3) representing displacements from M source points
    to N field (boundary) points.
    
    Returns G_out of shape (N, M, 3, 3).
    If r[n,m] is near zero, we set G_out[n,m,:,:] to zero.
    """
    # r_norm: shape (N, M)
    r_norm = np.linalg.norm(r, axis=-1)
    # Mask for near-zero distances
    mask = (r_norm < 1e-10)

    # Prepare output
    G_out = np.zeros((r.shape[0], r.shape[1], 3, 3), dtype=float)

    factor = 1.0/(8.0*np.pi)

    # We'll build (I / r_norm + r⊗r / r_norm^3) in a vectorized manner
    # r has shape (N, M, 3). Let's expand so r_r has shape (N, M, 3, 3).
    r_expanded = r[..., None]  # shape (N, M, 3, 1)
    r_r = r_expanded * np.swapaxes(r_expanded, -2, -1)  # (N, M, 3, 3)

    # np.eye(3) broadcast
    I = np.eye(3).reshape(1, 1, 3, 3)
    # shape (N, M, 1, 1)
    r_norm_4d = r_norm.reshape(r.shape[0], r.shape[1], 1, 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        # G_out = factor * [ I / r_norm + (r_r) / r_norm^3 ]
        G_out = factor * (I / r_norm_4d + r_r / (r_norm_4d**3))

    # Where r_norm is too small, set G_out to 0
    G_out[mask] = 0.0

    return G_out

def imp_mfs_mobility_vec(
    b_list,      # list of (N,3) boundary nodes per particle
    s_list,      # list of (M,3) source points per particle
    F_ext_list,  # list of external forces (3,) per particle
    T_ext_list,  # list of external torques (3,) per particle
    B_inv_list,  # list of B_inv ( (3N+6) x (3N+6) ) per particle
    max_iter=1000,
    tol=1e-7
):
    """
    Multi-particle IMP-MFS mobility problem solver including force and torque.
    This version is vectorized over boundary and source points (the big NxM loops).
    We still keep loops over particles.
    """
    P = len(b_list)
    N = b_list[0].shape[0]  
    M = s_list[0].shape[0]

    # x[p] has shape (3M + 6): [f_1, f_2, ..., f_M, V, Omega]
    # f_j is in R^3, total 3M
    # V in R^3
    # Omega in R^3

    # Each particle's system has size 3N+6 in the equations:
    #  - 3N boundary conditions
    #  - 3 force-balance conditions
    #  - 3 torque-balance conditions

    # We'll store the solution for each particle in x[p]
    x = [np.zeros(3*M + 6, dtype=np.float64) for _ in range(P)]

    # Construct F_tilde for each particle: a (3N+6, ) vector
    # that has [0,...0, F_ext, T_ext]
    F_tilde_list = []
    for p in range(P):
        F_tilde_p = np.zeros(3*N + 6, dtype=np.float64)
        F_tilde_p[3*N:3*N+3]   = F_ext_list[p]  # external force
        F_tilde_p[3*N+3:3*N+6] = T_ext_list[p]  # external torque
        F_tilde_list.append(F_tilde_p)

    for iteration in range(max_iter):
        old_solutions = [sol.copy() for sol in x]

        for p in range(P):
            # We'll compute w = "induced velocity" + zeros for the last 6 components
            # shape (3N+6,) but only the top 3N gets contributions
            w = np.zeros(3*N + 6, dtype=np.float64)
            bp = b_list[p]  # boundary nodes for particle p, shape (N, 3)

            # Summation of velocity from all other particles q
            # at these boundary points
            for q in range(P):
                if q == p:
                    continue
                x_q = x[q]
                # f_q: shape (M, 3)
                f_q = x_q[:3*M].reshape(M, 3)
                sq = s_list[q]  # shape (M,3)

                # Vector of displacements: shape (N, M, 3)
                # r[n, m, :] = bp[n] - sq[m]
                r = bp[:, None, :] - sq[None, :, :]

                # Gmatrix: shape (N, M, 3, 3)
                Gmatrix = G_vec(r)

                # velocity at each boundary node n => shape (N,3)
                # via sum_{m} Gmatrix[n,m,:,:] @ f_q[m,:].
                v = np.einsum('nmij,mj->ni', Gmatrix, f_q)

                # Flatten and add to w[0:3N]
                w[:3*N] += v.reshape(3*N)

            # Now solve for x[p] using the pre-inverted matrix B_inv_list[p]
            # The right-hand side is F_tilde[p] - w
            rhs = F_tilde_list[p] - w
            x[p] = B_inv_list[p] @ rhs

        # Check convergence
        max_diff = 0.0
        for p in range(P):
            diff = np.linalg.norm(x[p] - old_solutions[p])
            if diff > max_diff:
                max_diff = diff

        if max_diff < tol:
            print(f"Converged after {iteration+1} iterations with max diff = {max_diff:e}")
            break
        else:
            print(f"Iteration {iteration+1}: max diff = {max_diff:e}")
    else:
        print(f"Warning: Did not converge within {max_iter} iterations. Final max diff = {max_diff:e}")
        raise RuntimeError("Solver did not converge")

    return x

# --------------------------------------------------------------------

def createNewEllipsoid(center, rot, source_ref, boundary_ref):
    """
    Create a new ellipsoid by rotating and shifting 'boundary_ref'/'source_ref'.
    
    Parameters
    ----------
    center : (3,) array-like
        The desired center for ellipsoid #2.
    rot : scipy.spatial.transform.Rotation
        Rotation object defining the orientation for ellipsoid #2.
    source_ref : (M,3) ndarray
        Reference (base) source points for ellipsoid #2.
    boundary_ref : (N,3) ndarray
        Reference (base) boundary points for ellipsoid #2.

    Returns
    -------
    boundary2 : (N,3) ndarray
    source2   : (M,3) ndarray
    R2        : scipy.spatial.transform.Rotation
        The final orientation used for ellipsoid #2.
    """
    center = np.array(center, dtype=float)
    
    # Rotate each point
    boundary2 = rot.apply(boundary_ref) + center
    source2   = rot.apply(source_ref)   + center
    
    # Return the rotation object itself as R2
    return boundary2, source2



def solve_pair_interaction_spheroid(
    boundary1, source1, B1_inv, 
    boundary2, source2, orientation2, 
    force1, torque1, force2, torque2
):
    """
    High-level routine that:
    1) Builds or loads the relevant B-matrices for each spheroid
    2) Calls the mobility solver
    3) Returns the 6D velocity (V, Omega) of the *first* spheroid
    """
    F_ext_list = [force1, force2]
    T_ext_list = [torque1, torque2]

    QM, QN = get_QM_QN(orientation2, boundary2.shape[0], source2.shape[0])
    B2_inv = QM @ B1_inv @ QN.T
    B_inv_list = [B1_inv, B2_inv]
    print(B1_inv.shape, B2_inv.shape)
    
    b_list = [boundary1, boundary2]
    s_list = [source1, source2]
    
    try:
        V_tilde_list = imp_mfs_mobility_vec(
            b_list, s_list, F_ext_list, T_ext_list, B_inv_list, 
            max_iter=200, tol=1e-6  # No need for high precision
        )
    except Exception as e:
        raise e
    

    M1 = source1.shape[0]
    solution_1 = V_tilde_list[0]
    # last 6 are translational + rotational velocity
    velocity_1 = solution_1[3*M1 : 3*M1 + 3]
    omega_1    = solution_1[3*M1 + 3 : 3*M1 + 6]
    return np.concatenate([velocity_1, omega_1])  # shape (6,)


def generate_dataset_spheroid(shape, N_data, a, b, c,
                             dist_min, dist_max):
    """
    Generates N_data samples. Each sample:
      Features: (relative_position, orientation_axis, force_dir)
      Target: 6D velocity of spheroid #1
    Prolate spheroids: #1 at origin, #2 at random orientation & position 
      but no overlap, with a random unit force on #2.

      force_on_two: If True, force is applied on spheroid #2, else on #1.
    """
    acc = "fine"
    boundary1 = np.loadtxt(f'/home/shihab/src/mfs/points/b_{shape}_{acc}.txt', dtype=np.float64)  # boundary nodes
    source1 = np.loadtxt(f'/home/shihab/src/mfs/points/s_{shape}_{acc}.txt', dtype=np.float64)  # source points
    print(f"N={boundary1.shape[0]} boundary nodes, M={source1.shape[0]} source points")

    B_orig = build_B(boundary1, source1, np.zeros(3))
    start_time = time.time()
    B1_inv = np.linalg.pinv(B_orig)
    print(f"Built B-matrix in {time.time() - start_time:.2f} seconds.")
    
    X_features = []
    y_targets  = []
    
    n_accepted = 0
    rejected = 0
    index = 0
    while n_accepted < N_data:
        # introducde bias for smaller distance
        def biased_distance():
            L = dist_max - dist_min
            u = np.random.uniform(0, 1)
            # Inverse CDF for linearly decaying PDF (first bin has 2× the density of the last bin)
            x = 2 - np.sqrt(4 - 3 * u)
            dist = dist_min + x * L
            return dist
        dist = biased_distance()
        #dist = np.random.uniform(dist_min, dist_max)
        
        dir_vec = random_unit_vector()
        center2 = dist * dir_vec
        
        orientation2 = random_orientation_spheroid()
        boundary2, source2 = createNewEllipsoid(center2, orientation2, source1, boundary1)

        index += 1

        scalar = 6*np.pi*1.0*min(a, b, c)

        force_on_1 = random_unit_vector() * scalar
        force_on_2 = random_unit_vector() * scalar

        # TODO: Add random torque. All LLMs agree torque dir should be random
        torque_scalar = scalar * 1.0 
        torque_on_1 = random_unit_vector() * torque_scalar
        torque_on_2 = random_unit_vector() * torque_scalar


        min_dist = min_distance_ellipsoids(a, b, c, center2, orientation2, n_starts=10)
        if min_dist <= 0.05: # TODO: think about this threshold
            print(f"Rejected sample {index}: {min_dist}")
            rejected += 1
            continue
        
        print(f"Iter {index}: Min dist: {min_dist}, center2: {center2}")

        try:
            velocity_6d = solve_pair_interaction_spheroid(
                boundary1, source1, B1_inv, 
                boundary2, source2, orientation2, 
                force_on_1, torque_on_1, force_on_2, torque_on_2
            )
            self_vel = get_self_vel(shape, boundary1.shape[0], source1.shape[0],
                force_on_1, torque_on_1, B1_inv, Rotation.identity(3) 
            )
            self_vel = np.concatenate([self_vel[0], self_vel[1]])  # shape (6,)
            velocity_6d = velocity_6d - self_vel

        except Exception as e:
            print(f"Error in sample {index}: {e}")
            print(f"{center2}, {orientation2.as_matrix()}, {force_on_2}")
            # save_multiple_ellipsoids_legacy_vtk(
            #     f"out/ERROR_ellipsoids{index}.vtk", 
            #     [boundary1, boundary2], 
            #     [source1, source2]
            # )
            continue
        
        # Features: 21 Dimension
        feature_i = np.concatenate([
            center2, 
            [dist,min_dist],
            orientation2.as_quat(scalar_first=False),  # x,y,z,w format
            force_on_1,
            torque_on_1,
            force_on_2,
            torque_on_2      
        ])  
        assert feature_i.shape == (21,) 
        
        X_features.append(feature_i)
        y_targets.append(velocity_6d)  # shape (6,)
        
        n_accepted += 1
        if n_accepted % 200 == 0:
            print(f"{n_accepted} samples generated...")


    
    # Convert to arrays
    X_features = np.array(X_features)
    y_targets  = np.array(y_targets)
    

    print("Rejected samples %", rejected/N_data)
    return X_features, y_targets


def main():
    N_samples = 8000
    shape = "sphere"
    axes_length = {
        "prolateSpheroid": (1.0, 1.0, 3.0),
        "oblateSpheroid":  (2.0, 2.0, 1.0),
        "sphere":          (1.0, 1.0, 1.0),
    }
    a, b, c = axes_length[shape]
    
    # translation distance between ellipsoids
    dist_min = 2.02
    dist_max = 8.0 

    print("Generating data for both 2bdy terms at once...")
    print("Generating dataset for", shape)
    print("Axes lengths:", a, b, c)
    print("Number of samples:", N_samples)
    print("Distance range:", dist_min, dist_max)
    
    # Generate dataset
    X, Y = generate_dataset_spheroid(shape, N_samples, a, b, c, dist_min, dist_max)
    print("Dataset shapes:")
    print(" Features:", X.shape)
    print(" Targets:", Y.shape)
    
    # Save to disk
    import random 
    tmp = random.randint(0, 1000)

    t = time.localtime()
    current_time = time.strftime("%H:%M", t)
    np.save(f"data/X_{shape}_{current_time}_both_{tmp}.npy", X)
    np.save(f"data/Y_{shape}_{current_time}_both_{tmp}.npy",  Y)
    print(f"Saved {N_samples} samples.")

if __name__ == "__main__":
    main()
