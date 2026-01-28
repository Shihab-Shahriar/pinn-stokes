import time
import random
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.mfs_utils import build_B



INV_8PI = 1.0 / (8.0 * math.pi)
EPS_R2 = 1.0e-20

def _compute_centers(b_list, center_list):
    if center_list is None:
        centers = np.stack([bp.mean(axis=0) for bp in b_list], axis=0)
    else:
        centers = np.asarray(center_list, dtype=np.float64)
        if centers.shape != (len(b_list), 3):
            raise ValueError("center_list must have shape (P, 3)")
    return centers

def G_vec(r):
    """
    Vectorized Oseen tensor for Stokes flow in 3D.
    r is shape (N, M, 3) representing displacements from M source points
    to N field (boundary) points.

    Returns G_out of shape (N, M, 3, 3).
    For near-zero r, G_out is set to zero.
    """
    r_norm = np.linalg.norm(r, axis=-1)
    mask = (r_norm < 1e-10)
    G_out = np.zeros((r.shape[0], r.shape[1], 3, 3), dtype=float)
    factor = INV_8PI
    r_expanded = r[..., None]  # shape (N, M, 3, 1)
    r_r = r_expanded * np.swapaxes(r_expanded, -2, -1)  # (N, M, 3, 3)
    I = np.eye(3).reshape(1, 1, 3, 3)
    r_norm_4d = r_norm.reshape(r.shape[0], r.shape[1], 1, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        G_out = factor * (I / r_norm_4d + r_r / (r_norm_4d**3))
    G_out[mask] = 0.0
    return G_out

def imp_mfs_mobility_vec(
    b_list,      # list of (N,3) boundary nodes per particle
    s_list,      # list of (M,3) source points per particle
    F_ext_list,  # list of external forces (3,) per particle
    T_ext_list,  # list of external torques (3,) per particle
    B_inv_list,  # list of pre-inverted B-matrices ((3N+6) x (3N+6)) per particle
    max_iter=1000,
    tol=1e-7,
    print_steps=False,
    center_list=None,
    L_cut=None
):
    """
    Multi-particle IMP-MFS mobility solver including force and torque.
    If L_cut is provided, particle pairs beyond the cutoff use a point-force
    approximation based on the net stokeslet.
    """
    P = len(b_list)
    N = b_list[0].shape[0]
    M = s_list[0].shape[0]
    use_cutoff = L_cut is not None
    if use_cutoff:
        centers = _compute_centers(b_list, center_list)
        stokeslet_sum = np.zeros((P, 3), dtype=np.float64)

    # Initialize solution vector for each particle.
    x = [np.zeros(3*M + 6, dtype=np.float64) for _ in range(P)]

    # Build the F_tilde vector for each particle.
    F_tilde_list = []
    for p in range(P):
        F_tilde_p = np.zeros(3*N + 6, dtype=np.float64)
        F_tilde_p[3*N:3*N+3]   = F_ext_list[p]  # external force
        F_tilde_p[3*N+3:3*N+6] = T_ext_list[p]  # external torque
        F_tilde_list.append(F_tilde_p)

    if print_steps:
        print(
            f"imp_mfs_mobility_vec: start P={P}, N={N}, M={M}, use_cutoff={use_cutoff}",
            flush=True,
        )

    for iteration in range(max_iter):
        if print_steps:
            print(f"Iteration {iteration+1}: start", flush=True)
        old_solutions = [sol.copy() for sol in x]

        for p in range(P):
            if print_steps:
                print(f"Iteration {iteration+1}: particle {p+1}/{P} start", flush=True)
            w = np.zeros(3*N + 6, dtype=np.float64)
            bp = b_list[p]
            # Sum induced velocity from all other particles.
            for q in range(P):
                if q == p:
                    continue
                x_q = x[q]
                f_q = x_q[:3*M].reshape(M, 3)
                if use_cutoff:
                    dist_pq = np.linalg.norm(centers[p] - centers[q])
                    if dist_pq < L_cut:
                        sq = s_list[q]
                        r = bp[:, None, :] - sq[None, :, :]
                        Gmatrix = G_vec(r)
                        v = np.einsum('nmij,mj->ni', Gmatrix, f_q)
                    else:
                        f_net = stokeslet_sum[q]
                        rC = bp - centers[q]
                        rr = np.linalg.norm(rC, axis=1)
                        mask = rr < 1e-10
                        inv_r = np.zeros_like(rr)
                        inv_r[~mask] = 1.0 / rr[~mask]
                        inv_r3 = inv_r**3
                        dotfr = np.sum(rC * f_net, axis=1)
                        v = inv_r[:, None] * f_net + (dotfr * inv_r3)[:, None] * rC
                        v = v * INV_8PI
                        v[mask] = 0.0
                else:
                    sq = s_list[q]
                    r = bp[:, None, :] - sq[None, :, :]
                    Gmatrix = G_vec(r)
                    v = np.einsum('nmij,mj->ni', Gmatrix, f_q)
                w[:3*N] += v.reshape(3*N)

            rhs = F_tilde_list[p] - w
            x[p] = B_inv_list[p] @ rhs
            if use_cutoff:
                F_p = x[p][:3*M].reshape(M, 3)
                stokeslet_sum[p] = np.sum(F_p, axis=0)
            if print_steps:
                print(f"Iteration {iteration+1}: particle {p+1}/{P} done", flush=True)

        max_diff = max(np.linalg.norm(x[p]-old_solutions[p]) for p in range(P))
        if max_diff < tol:
            #print(f"Converged after {iteration+1} iterations (max diff = {max_diff:e})")
            break
        else:
            if print_steps:
                print(f"Iteration {iteration+1}: max diff = {max_diff:e}", flush=True)
    else:
        raise RuntimeError("Solver did not converge")
    return x

class MobOpMFS:
    def __init__(self, shape, acc):
        assert shape=="sphere", "Only sphere shape is implemented in this example."

        self.boundary = np.loadtxt(f'data/points/b_{shape}_{acc}.txt', dtype=np.float64)
        self.source = np.loadtxt(f'data/points/s_{shape}_{acc}.txt', dtype=np.float64)
        print(f"Loaded geometry: {self.boundary.shape[0]} boundary nodes, {self.source.shape[0]} source points")

        self.B_orig = build_B(self.boundary, self.source, np.zeros(3))
        self.B_inv = np.linalg.pinv(self.B_orig)


    def apply(self, config, forces, viscosity=1.0):
        """
        Sphere-only
        """
        N_particles = config.shape[0]
        b_list = []
        s_list = []
        F_ext_list = []
        T_ext_list = []
        for i in range(N_particles):
            center = config[i, :3]
            b_i = self.boundary + center[None, :]
            s_i = self.source + center[None, :]
            b_list.append(b_i)
            s_list.append(s_i)
            F_ext_list.append(forces[i, :3])
            T_ext_list.append(forces[i, 3:6])

        B_inv_list = [self.B_inv for _ in range(N_particles)]

        V_tilde_list = imp_mfs_mobility_vec(
            b_list, s_list, F_ext_list, T_ext_list, B_inv_list,
            max_iter=1000, tol=1e-7, print_steps=True
        )

        velocities = np.zeros((N_particles, 6), dtype=np.float64)
        M1 = s_list[0].shape[0]
        for i in range(N_particles):
            solution_i = V_tilde_list[i]
            velocities[i, :3] = solution_i[3*M1 : 3*M1 + 3]
            velocities[i, 3:6] = solution_i[3*M1 + 3 : 3*M1 + 6]
        return velocities
    
 
if __name__ == "__main__":
    from src.mob_op_2b_combined import check_against_ref 
    from src.triton_mfs import MobMFSTriton

    shape = "sphere"
    acc = "Xfine"
    #mob_mfs = MobOpMFS(shape, acc)
    mob_mfs = MobMFSTriton(shape, acc)


    ref_path = "tmp/testcase_uniform_0.1_40.csv"
    
    import pandas as pd
    df = pd.read_csv(ref_path, float_precision="high",
                    header=0, index_col=False)
    check_against_ref(mob_mfs, ref_path, print_stuff=True)
