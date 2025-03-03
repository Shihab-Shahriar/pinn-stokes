"""
Dec 19.
Random cluster of unit spheres. "Grown" from origin,
ensuring new sphere has exactly min_sep distance from
any of the existing spheres.
"""

import numpy as np
from scipy.spatial.distance import cdist
from simulation_singleP import build_B
from create_dataset import G_vec

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
                # f_q: shape (M, 3)
                f_q = x_q[:3*M].reshape(M, 3)

                dist_pq = np.linalg.norm(center_list[p] - center_list[q])

                if dist_pq < L_cut:
                    sq = s_list[q]  # shape (M,3)

                    # Vector of displacements: shape (N, M, 3)
                    # r[n, m, :] = bp[n] - sq[m]
                    r = bp[:, None, :] - sq[None, :, :]

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

            # Now solve for x[p] using the pre-inverted matrix B_inv_list[p]
            # The right-hand side is F_tilde[p] - w
            rhs = F_tilde_list[p] - w
            x[p] = B_inv_list[p] @ rhs
            
            F_p = x[p][:3*M]
            stokeslet_sum[p] = np.sum(F_p.reshape(M, 3), axis=0)

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

if __name__=="__main__":
    numParticles=500
    R=1.0
    delta=.8
    TOLERANCE = 1e-7
    acc = "fine"
    shape = "sphere"

    print(f"{TOLERANCE=}")
    print(f"{acc=}")
    print(f"Cluster of {numParticles} spheres, radius {R}, min separation {delta}")

    centers = grow_cluster(numParticles,R,delta)
    non_diag = ~np.eye(numParticles, dtype=bool)
    dd = cdist(centers, centers)[non_diag].reshape(numParticles,numParticles-1)
    dd = dd.min(axis=1)
    assert np.allclose(dd, 2*R+delta)

    print("cluster created")

    write_vtk(centers, "cluster.vtk")
    print("Wrote cluster.vtk with sphere centers.")

    b_single = np.loadtxt(f'points/b_{shape}_{acc}.txt', dtype=np.float64)  # boundary nodes
    s_single = np.loadtxt(f'points/s_{shape}_{acc}.txt', dtype=np.float64)
    
    N = b_single.shape[0]  # number of boundary nodes
    M = s_single.shape[0]  # number of source points

    b_list = [
        b_single + c for c in centers
    ]
    s_list = [
        s_single + c for c in centers
    ]
    F = 1.0 * 6* np.pi  # choose some magnitude
    F_ext_list = [
        np.array([0.0, 0.0, F], dtype=np.float64) for _ in centers
    ]
    T_ext_list = [
        np.array([0.0, 0.0, 0.0], dtype=np.float64) for _ in centers
    ]

    B = build_B(b_single, s_single, np.zeros(3))
    Bpp_inv = np.linalg.pinv(B)

    Bpp_inv_list = [Bpp_inv.copy() for _ in range(numParticles)]

    # Run the IMP-MFS mobility solver
    start_time = time.time()
    V_tilde_list = imp_mfs_multiparticle(
        b_list, s_list, centers, F_ext_list, T_ext_list, Bpp_inv_list,
        L_cut=25.0, max_iter=1000, tol=TOLERANCE, print_interval=50
    )

    end_time = time.time()
    print(f"Elapsed time: {end_time-start_time:.3f} seconds")



