"""
Dec 19.
Random cluster of unit spheres. "Grown" from origin,
ensuring new sphere has exactly min_sep distance from
any of the existing spheres.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from src.mfs_utils import (build_B, createNewEllipsoid, get_QM_QN,
                            min_distance_two_ellipsoids)
from src.mfs import G_vec, imp_mfs_mobility_vec
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

import time

def imp_mfs_multiparticle(
    b_list,      # list of (N,3) boundary nodes per particle
    s_list,      # list of (M,3) source points per particle
    center_list, # list of (3,) center of each particle
    F_ext_list,  # list of external forces (3,) per particle
    T_ext_list,  # list of external torques (3,) per particle
    B_inv_list,  # list of B_inv ( (3N+6) x (3N+6) ) per particle
    L_cut = 25.0,
    max_iter=1000,
    tol=1e-7,
    print_interval=None
):
    """
    Copied from create_dataset.py to adapt to point force approximation.
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

    stokeslet_sum = np.zeros((P, 3), dtype=np.float64)


    # Construct F_tilde for each particle: a (3N+6, ) vector
    # that has [0,...0, F_ext, T_ext]
    F_tilde_list = []
    for p in range(P):
        F_tilde_p = np.zeros(3*N + 6, dtype=np.float64)
        F_tilde_p[3*N:3*N+3]   = F_ext_list[p]  # external force
        F_tilde_p[3*N+3:3*N+6] = T_ext_list[p]  # external torque
        F_tilde_list.append(F_tilde_p)

    for iteration in range(max_iter):
        if print_interval is not None and iteration % print_interval == 0:
            print(f"Iteration {iteration} out of {max_iter}")
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
                #x_q = old_solutions[q]
                # f_q: shape (M, 3)
                f_q = x_q[:3*M].reshape(M, 3)
                # if iteration==1:
                #     f_q = f_q[:2]  

                dist_pq = np.linalg.norm(center_list[p] - center_list[q])

                if dist_pq < L_cut:
                    sq = s_list[q]  # shape (M,3)
                    # if iteration==1:
                    #     sq = sq[:2]  

                    # Vector of displacements: shape (N, M, 3)
                    # r[n, m, :] = bp[n] - sq[m]
                    r = bp[:, None, :] - sq[None, :, :]
                    # if iteration==1:
                    #     r = r[:, :2, :]

                    # Gmatrix: shape (N, M, 3, 3)
                    Gmatrix = G_vec(r)

                    # velocity at each boundary node n => shape (N,3)
                    # via sum_{m} Gmatrix[n,m,:,:] @ f_q[m,:].
                    v = np.einsum('nmij,mj->ni', Gmatrix, f_q)

                else: # point force approximation
                    f_net = stokeslet_sum[q]  # shape (3,)
                    rC = bp - center_list[q]     # shape (N,3)
                    rr = np.linalg.norm(rC, axis=1)  # shape (N,)
                    inv_r  = 1.0 / rr            # shape (N,)
                    inv_r3 = inv_r**3            # shape (N,)
                    dotfr  = np.sum(rC * f_net, axis=1)  # shape (N,) dot product for each boundary node
                    v = inv_r[:, None] * f_net \
                        + (dotfr * inv_r3)[:, None] * rC   # shape (N,3)
                    v =  v / (8.0 *np.pi)

                # Flatten and add to w[0:3N]
                w[:3*N] += v.reshape(3*N)

                # if p==0:
                #     np.set_printoptions(suppress=True, precision=8)
                #     print(f"p0, v0 after {q}: ", w[3:6])
                #     print(f"posx_accum for {q}: {sq[:, 0].sum():.9f}")
                #     print(f"fx_accum for {q}: {f_q[:, 0].sum():.9f}")
                #     if iteration==1:
                #         assert len(sq) == len(f_q) ==2

            # Now solve for x[p] using the pre-inverted matrix B_inv_list[p]
            # The right-hand side is F_tilde[p] - w
            rhs = F_tilde_list[p] - w
            x[p] = B_inv_list[p] @ rhs
            # print("sum of rhs[p]:", p, rhs.sum(), F_tilde_list[p].sum(), w.sum())
            # print("v0:" , p, w[:3])
            
            F_p = x[p][:3*M]
            stokeslet_sum[p] = np.sum(F_p.reshape(M, 3), axis=0)



        # print("sum of old solutions:", iteration, np.array(old_solutions).sum())
        # print("sum of new solutions:", iteration, np.array(x).sum())

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
            print(f"Iteration {iteration}: max diff = {max_diff:e}")
    else:
        print(f"Warning: Did not converge within {max_iter} iterations. Final max diff = {max_diff:e}")
        raise RuntimeError("Solver did not converge")

    return x


def random_unit_vector():
    # Sample a random direction uniformly on the unit sphere
    # Using spherical coordinates: 
    # phi in [0,2pi), cos(theta) in [-1,1]
    phi = 2*np.pi*np.random.rand()
    u = 2*np.random.rand()-1  # u = cos(theta)
    theta = np.arccos(u)
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return np.array([x,y,z], dtype=float)


def grow_cluster(P=10, R=1.0, delta=0.1):
    """
    Grow a cluster of P spheres of radius R, with minimum separation delta.
    The first sphere is at the origin. Each subsequent sphere is placed along a random direction
    such that it just touches the cluster (min distance 2R+delta to at least one sphere).
    """
    centers = [np.zeros(3)]  # first sphere at origin
    min_dist = 2*R+delta
    for i in range(1,P):
        # Try random directions until we find a valid placement
        for attempt in range(10000): # safeguard loop
            d = random_unit_vector()
            c = centers[np.random.randint(0,i)]  # pick a random existing center
 
            new_center = c + d * min_dist

            # check if new_center is at least min_dist from all existing centers
            if all(np.linalg.norm(new_center - c) >= min_dist for c in centers):
                centers.append(new_center)
                break
    return np.array(centers)




def create_another_ellipsoid(a, b, c, center1, orientation1, d, n_random_starts=5):
    """
    Randomly choose an orientation for the second ellipsoid (same semi-axes a,b,c),
    then numerically solve for center2 so that the global minimal distance to the first
    ellipsoid is as close as possible to d.

    Returns (center2, orientation2).
    """
    # Convert orientation1 to a 3x3 rotation matrix
    R1 = orientation1.as_matrix()

    # Step 1: Random orientation for second ellipsoid
    rot2 = Rotation.random()
    R2 = rot2.as_matrix()
    orientation2 = rot2

    # Step 2: We'll define an objective that tries to match min_distance_two_ellipsoids(...) = d
    def distance_gap(center2):
        dist = min_distance_two_ellipsoids(a, b, c, center1, R1,
                                           a, b, c, center2, R2)
        return (dist - d)**2  # We want this gap to be zero => dist == d

    best_center2 = None
    best_obj_val = np.inf

    # Try multiple random starts for center2
    for _ in range(n_random_starts):
        # We'll pick an initial guess "somewhere" near center1, e.g. random direction * some random distance
        rand_dir = np.random.normal(size=3)
        rand_dir /= np.linalg.norm(rand_dir)
        # Start about (a + b + c + d) away, or something big enough
        center2_init = center1 + (a + b + c + d) * rand_dir

        # Solve an unconstrained optimization in 3D for center2
        res = minimize(
            distance_gap,
            center2_init,
            method='L-BFGS-B',    # 3D unconstrained
            options={'maxiter': 500, 'ftol': 1e-10}
        )

        if res.success and res.fun < best_obj_val:
            best_obj_val = res.fun
            best_center2 = res.x

    # If we never found a solution or best_obj_val is large, the solver might not have converged well.
    # But we'll return the best we found.
    if best_center2 is None:
        # fallback: just do naive "line-based" placement
        # see note: won't guarantee the global min distance = d
        # but won't fail
        rand_dir = np.random.normal(size=3)
        rand_dir /= np.linalg.norm(rand_dir)
        best_center2 = center1 + (2*a + d) * rand_dir

    return best_center2, orientation2


def grow_ellipsoid_cluster(P, a, b, c, delta, max_attempts=10000):
    """
    Grow a cluster of P axis-aligned ellipsoids, each with half-axes (a, b, c).
    We maintain at least 'delta' surface distance between any pair of ellipsoids.
    
    Parameters
    ----------
    P : int
        Number of ellipsoids to place.
    a, b, c : floats
        Half-axes of all ellipsoids (assuming axis-aligned).
    delta : float
        Minimum separation between surfaces of any two ellipsoids.
    max_attempts : int
        Safeguard limit for attempts at placing each new ellipsoid.
    
    Returns
    -------
    centers : np.ndarray, shape (P, 3)
        The 3D coordinates of the centers of the placed ellipsoids.
    """
    centers = [np.zeros(3)]  # First ellipsoid at the origin
    orientation = [Rotation.identity()]  # First ellipsoid is axis-aligned
    
    for i in range(1, P):
        # Attempt to place the i-th ellipsoid
        placed = False
        for aatmpt in range(max_attempts):
            # Pick a random existing ellipsoid to grow from
            if aatmpt % 10 == 0:
                print(f"Attempt {aatmpt} for ellipsoid {i+1}")

            ref_idx = np.random.randint(0, i)
            ref_center = centers[ref_idx]
            ref_orient = orientation[ref_idx]
            
            new_center, new_orient = create_another_ellipsoid(
                a, b, c, ref_center, ref_orient, delta
            )
            
            # -------------------------------------------------------
            # 2) Check if this new ellipsoid is >= delta away from
            #    all other ellipsoids in the cluster.
            # -------------------------------------------------------
            for j in range(i):
                surface_dist = min_distance_two_ellipsoids(
                    a, b, c, centers[j], orientation[j],
                    a, b, c, new_center, new_orient, n_starts=6
                )
                if surface_dist< delta-1e-4:
                    print(surface_dist, delta)
                    break
            else:
                # If we didn't break out of the loop, the new ellipsoid is placed
                centers.append(new_center)
                orientation.append(new_orient)
                placed = True
                print(f"Placed ellipsoid {i+1} after {aatmpt+1} attempts.")
                break
        
        if not placed:
            # If we exhaust max_attempts, we either return what we have
            # or raise an exception. Here we'll just return partial cluster.
            print(f"Warning: Could only place {len(centers)} ellipsoids out of {P}.")
            break
    
    return np.array(centers), orientation


def write_vtk(centers, filename="cluster.vtk"):
    """
    Write the cluster configuration as points to a VTK legacy file.
    This will allow visualization in ParaView or VisIt.
    """
    P = len(centers)
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("Cluster of spheres\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {P} float\n")
        for c in centers:
            f.write(f"{c[0]} {c[1]} {c[2]}\n")
        # No cells, just points
        f.write(f"VERTICES {P} {2*P}\n")
        for i in range(P):
            f.write(f"1 {i}\n")
        # Optionally, add a scalar field
        f.write("POINT_DATA {0}\n".format(P))
        f.write("SCALARS sphere_id int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(P):
            f.write(f"{i}\n")



def reference_data_generation(shape, delta, numParticles, tol=1e-7):
    """
    Was used to both generate reference dataset, and compare cuda impl
    accuracy against this.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    TOLERANCE = tol
    acc = "Xfine"
    axes_length = {
        "prolateSpheroid": (1.0, 1.0, 3.0),
        "oblateSpheroid":  (2.0, 2.0, 1.0),
        "sphere":          (1.0, 1.0, 1.0),
    }
    a, b, c = axes_length[shape]

    print(f"{TOLERANCE=}")
    print(f"{acc=}")
    print(f"{a=}, {b=}, {c=}")
    print(f"{numParticles=}")
    print(f"{delta=}")

    start = time.time()
    if shape != "sphere":
        centers, orients = grow_ellipsoid_cluster(numParticles,a,b,c,delta)
    else:
        centers = grow_cluster(numParticles, a, delta)
        orients = [Rotation.identity() for _ in range(numParticles)]

    
    end = time.time()
    print(f"Elapsed time: {end-start:.3f} seconds")

    # Check that the minimum separation is maintained
    for i in range(numParticles):
        my_min = np.inf
        for j in range(i):
            dd = min_distance_two_ellipsoids(a,b,c,centers[i],orients[i],
                                        a,b,c,centers[j],orients[j])

            assert dd >= delta-1e-4, f"Separation violation: {dd} < {delta}"

            my_min = min(my_min, dd)
        print(f"Min distance for ellipsoid {i}: {my_min}")

    print("cluster created")

    # write_vtk(centers, "cluster.vtk")
    # print("Wrote cluster.vtk with sphere centers.")

    root = "/home/shihab/src/mfs/"
    b_single = np.loadtxt(f'{root}points/b_{shape}_{acc}.txt', dtype=np.float64)  # boundary nodes
    s_single = np.loadtxt(f'{root}points/s_{shape}_{acc}.txt', dtype=np.float64)
    
    N = b_single.shape[0]  # number of boundary nodes
    M = s_single.shape[0]  # number of source points

    
    b_list = [b_single]
    s_list = [s_single]
    for i in range(1, numParticles):
        boundary_i, source_i = createNewEllipsoid(centers[i], orients[i],
                                                s_single, b_single)
        b_list.append(boundary_i)
        s_list.append(source_i)

    # print xyz coordinates of first pos of p1
    print("b_list[i][0]:", b_list[1][0], b_list[0].shape)


    F = 1.0 * 6* np.pi  # choose some magnitude
    F_ext_list = [
        np.array([0.0, 0.0, F], dtype=np.float64) for _ in centers
    ]
    T_ext_list = [
        np.array([0.0, 0.0, 0.0], dtype=np.float64) for _ in centers
    ]

    B = build_B(b_single, s_single, np.zeros(3))
    Bpp_inv = np.linalg.pinv(B)
    Bpp_inv_list = [Bpp_inv.copy()]
    for i in range(1, numParticles):
        QM2, QN2 = get_QM_QN(orients[i], b_single.shape[0], s_single.shape[0])
        B2_inv = QM2 @ Bpp_inv @ QN2.T
        Bpp_inv_list.append(B2_inv)
        #print("QM QN sum:", i, QM2.sum(), QN2.sum())
        assert QM2.shape == (3*M+6, 3*M+6)
        assert QN2.shape == (3*N+6, 3*N+6)

    #print("Sum of template binv:", Bpp_inv.sum())
    #print(Bpp_inv.shape)

    #print("Sum of b_list:", np.array(b_list).sum())
    #print("Sum of s_list:", np.array(s_list).sum())
    ft = np.concatenate((F_ext_list, T_ext_list), axis=1)
    print("Sum of F_ext_list:", ft.sum())

    b_inv = np.array(Bpp_inv_list)


    # Run the IMP-MFS mobility solver
    start_time = time.time()
    V_tilde_list = imp_mfs_multiparticle(
        b_list, s_list, centers, F_ext_list, T_ext_list, Bpp_inv_list,
        L_cut=2500.0, max_iter=1000, tol=TOLERANCE, print_interval=50
    )
    end_time = time.time()
    print(f"Elapsed time: {end_time-start_time:.3f} seconds")


    columns = ["x", "y", "z",  
               "q_x", "q_y", "q_z", "q_w", 
               "f_x", "f_y", "f_z",
               "t_x", "t_y", "t_z",
               "v_x", "v_y", "v_z",
               "w_x", "w_y", "w_z"]
    df = pd.DataFrame(columns=columns)

    M = s_list[i].shape[0]
    for i in range(numParticles):
        solution_i = V_tilde_list[i]
        velocity_i = solution_i[3*M : 3*M + 3]
        omega_i    = solution_i[3*M + 3 : 3*M + 6]
        df.loc[i] = np.concatenate([centers[i], orients[i].as_quat(scalar_first=False),
                                    F_ext_list[i], T_ext_list[i], velocity_i, omega_i])
        # print(f"p {i}")
        # print("Lin velocity:", velocity_i)
        # print("Ang velocity:", omega_i)

    #save to csv
    # df.to_csv(f"tmp/reference_{shape}_{delta}.csv", index=False, header=True, float_format="%.16g")

    # # test saving didn't lose precision
    # df2 = pd.read_csv("reference_ellipsoid.csv", float_precision="high",
    #                   header=0, index_col=False)
    # assert np.allclose(df.values, df2.values)

    return df


def uniform_sphere_cluster(volume_fraction, numParticles, radius=1.0, max_attempts=10000):
    """Generate a random non-overlapping cluster of spheres."""
    # Calculate bounding box size from desired volume fraction
    sphere_volume = (4 / 3) * np.pi * radius ** 3
    total_sphere_volume = numParticles * sphere_volume
    bbox_volume = total_sphere_volume / volume_fraction
    bbox_side = bbox_volume ** (1 / 3)

    MIN_SEPARATION = .05

    positions = np.zeros((numParticles, 3), dtype=np.float64)
    for i in range(numParticles):
        attempts = 0
        while attempts < max_attempts:
            pos = np.random.uniform(radius, bbox_side - radius, 3)
            if i == 0:
                positions[i] = pos
                break

            # Check against all existing positions to ensure no overlap
            dd = cdist(positions[:i], pos[None, :])
            if np.all(dd >= 2 * radius + MIN_SEPARATION):
                positions[i] = pos
                break
            attempts += 1

        if attempts == max_attempts:
            print(f"Warning: Could not place sphere {i+1} without overlap")

    centers = np.array(positions)
    orients = [Rotation.identity() for _ in range(numParticles)]

    # Verify minimum separation using exact ellipsoid distance
    for i in range(numParticles):
        my_min = np.inf
        for j in range(i):
            dd = np.linalg.norm(centers[i] - centers[j]) - 2*radius
            assert dd >= MIN_SEPARATION, f"Separation violation: {dd} < {MIN_SEPARATION}, between {i} and {j}"
            my_min = min(my_min, dd)
        print(f"Min distance for ellipsoid {i}: {my_min}")

    print("cluster created")
    return centers, orients


def uniform_data_generation(shape, volume_fraction, numParticles, TOLERANCE=1e-8):
    """
    Generate uniform spheres with radius 1 and specified volume fraction.
    
    Args:
        shape: Shape identifier for output filename
        volume_fraction: Fraction of bounding box volume occupied by spheres
        numParticles: Number of spheres to generate
    """
    assert shape=="sphere", "Currently only 'sphere' shape is supported"
    acc = "Xfine"
    axes_length = {
        "prolateSpheroid": (1.0, 1.0, 3.0),
        "oblateSpheroid":  (2.0, 2.0, 1.0),
        "sphere":          (1.0, 1.0, 1.0),
    }
    a, b, c = axes_length[shape]

    centers, orients = uniform_sphere_cluster(volume_fraction, numParticles)
    print("centers")
    print(centers)

    # write_vtk(centers, "cluster.vtk")
    # print("Wrote cluster.vtk with sphere centers.")

    root = "/home/shihab/src/mfs/"
    b_single = np.loadtxt(f'{root}points/b_{shape}_{acc}.txt', dtype=np.float64)  # boundary nodes
    s_single = np.loadtxt(f'{root}points/s_{shape}_{acc}.txt', dtype=np.float64)
    
    N = b_single.shape[0]  # number of boundary nodes
    M = s_single.shape[0]  # number of source points

    
    b_list = []
    s_list = []
    for i in range(numParticles):
        # boundary_i, source_i = createNewEllipsoid(centers[i], orients[i],
        #                                         s_single, b_single)
        boundary_i = b_single + centers[i][None, :]
        source_i   = s_single + centers[i][None, :]

        b_list.append(boundary_i)
        s_list.append(source_i)

    # print xyz coordinates of first pos of p1
    print("b_list[i][0]:", b_list[1][0], b_list[0].shape)


    F = 1.0 * 6* np.pi  # choose some magnitude
    # F_ext_list = [
    #     np.array([0.0, 0.0, F], dtype=np.float64) for _ in centers
    # ]
    F_ext_list = [
        np.random.uniform(-1, 1, 3).astype(np.float64) for _ in centers
    ]
    F_ext_list = [F / np.linalg.norm(f) * f for f in F_ext_list]  # normalize to magnitude F
    T_ext_list = [
        np.array([0.0, 0.0, 0.0], dtype=np.float64) for _ in centers
    ]

    B = build_B(b_single, s_single, np.zeros(3))
    Bpp_inv = np.linalg.pinv(B)
    Bpp_inv_list = [Bpp_inv.copy()]
    for i in range(1, numParticles):
        if shape!="sphere":
            QM2, QN2 = get_QM_QN(orients[i], b_single.shape[0], s_single.shape[0])
            B2_inv = QM2 @ Bpp_inv @ QN2.T
            Bpp_inv_list.append(B2_inv)
            #print("QM QN sum:", i, QM2.sum(), QN2.sum())
            assert QM2.shape == (3*M+6, 3*M+6)
            assert QN2.shape == (3*N+6, 3*N+6)
        else:
            Bpp_inv_list.append(Bpp_inv.copy())

    print("Sum of template binv:", Bpp_inv.sum())
    print(Bpp_inv.shape)

    print("Sum of b_list:", np.array(b_list).sum())
    print("Sum of s_list:", np.array(s_list).sum())
    ft = np.concatenate((F_ext_list, T_ext_list), axis=1)
    print("Sum of F_ext_list:", ft.sum())

    b_inv = np.array(Bpp_inv_list)
    print("Sum of Bpp_inv_list:", b_inv.sum())
    print("Sum of Bpp_inv_list[0]:", b_inv[0].sum())
    print("Sum of Bpp_inv_list[1]:", b_inv[1].sum())
    print("Sum of Bpp_inv_list[2]:", b_inv[2].sum())

    print("B_inv total terms:", np.prod(b_inv.shape))

    # Run the IMP-MFS mobility solver
    start_time = time.time()
    V_tilde_list = imp_mfs_multiparticle(
        b_list, s_list, centers, F_ext_list, T_ext_list, Bpp_inv_list,
        L_cut=2500.0, max_iter=1000, tol=TOLERANCE, print_interval=50
    )
    end_time = time.time()
    print(f"Elapsed time: {end_time-start_time:.3f} seconds")


    columns = ["x", "y", "z",  
               "q_x", "q_y", "q_z", "q_w", 
               "f_x", "f_y", "f_z",
               "t_x", "t_y", "t_z",
               "v_x", "v_y", "v_z",
               "w_x", "w_y", "w_z"]
    df = pd.DataFrame(columns=columns)

    M = s_list[i].shape[0]
    for i in range(numParticles):
        solution_i = V_tilde_list[i]
        velocity_i = solution_i[3*M : 3*M + 3]
        omega_i    = solution_i[3*M + 3 : 3*M + 6]
        df.loc[i] = np.concatenate([centers[i], orients[i].as_quat(scalar_first=False),
                                    F_ext_list[i], T_ext_list[i], velocity_i, omega_i])
        print(f"p {i}")
        print("Lin velocity:", velocity_i)
        print("Ang velocity:", omega_i)

    #save to csv
    df.to_csv(f"tmp/uniform_{shape}_{volume_fraction}.csv", index=False, header=True, float_format="%.16g")
    return df


def create_and_save_ellipsoid_cluster(numParticles):
    import random
    random.seed(42)
    np.random.seed(42)

    delta=1.0
    TOLERANCE = 1e-7
    acc = "fine"
    shape = "prolateSpheroid"
    a,b,c = 1.0, 1.0, 3.0

    print(f"{TOLERANCE=}")
    print(f"{acc=}")
    print(f"{a=}, {b=}, {c=}")
    print(f"{numParticles=}")
    print(f"{delta=}")

    start = time.time()
    centers, orients = grow_ellipsoid_cluster(numParticles,a,b,c,delta)
    end = time.time()
    print(f"Elapsed time: {end-start:.3f} seconds")

    df = pd.DataFrame(columns=["x", "y", "z",  
                            "q_x", "q_y", "q_z", "q_w"])
    for i in range(numParticles):
        df.loc[i] = np.concatenate([centers[i], orients[i].as_quat(scalar_first=False)])
        print(f"p {i}")
        print("Lin velocity:", centers[i])
        print("Ang velocity:", orients[i].as_quat(scalar_first=False))

    #save to csv
    df.to_csv("n100.csv", index=False, header=True, float_format="%.16g")


if __name__ == '__main__':
    shape = "sphere"
    delta = 0.8
    numParticles = 800
    # df = reference_data_generation(shape, delta=delta, 
    #                                numParticles=numParticles, tol=1e-8)
    # df.to_csv(f"tmp/reference_{shape}_{delta}.csv", index=False, header=True, float_format="%.16g")
    
    
    #uniform_data_generation("sphere", volume_fraction=0.05, numParticles=20)

    centers, orients = uniform_sphere_cluster(0.1, numParticles)
    quats = np.array([r.as_quat(scalar_first=False) for r in orients])
    config = np.concatenate([centers, quats], axis=1)

    columns=["x", "y", "z",  "q_x", "q_y", "q_z", "q_w"]

    df = pd.DataFrame(config, columns=columns)

    df.to_csv(f"tmp/uniform_sphere_0.1_{numParticles}.csv", index=False, header=True, float_format="%.16g")


