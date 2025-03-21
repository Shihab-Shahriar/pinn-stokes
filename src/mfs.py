import time
import random
import numpy as np
from scipy.spatial.transform import Rotation as R

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
    factor = 1.0/(8.0*np.pi)
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
    tol=1e-7
):
    """
    Multi-particle IMP-MFS mobility solver including force and torque.
    """
    P = len(b_list)
    N = b_list[0].shape[0]
    M = s_list[0].shape[0]

    # Initialize solution vector for each particle.
    x = [np.zeros(3*M + 6, dtype=np.float64) for _ in range(P)]

    # Build the F_tilde vector for each particle.
    F_tilde_list = []
    for p in range(P):
        F_tilde_p = np.zeros(3*N + 6, dtype=np.float64)
        F_tilde_p[3*N:3*N+3]   = F_ext_list[p]  # external force
        F_tilde_p[3*N+3:3*N+6] = T_ext_list[p]  # external torque
        F_tilde_list.append(F_tilde_p)

    for iteration in range(max_iter):
        old_solutions = [sol.copy() for sol in x]

        for p in range(P):
            w = np.zeros(3*N + 6, dtype=np.float64)
            bp = b_list[p]
            # Sum induced velocity from all other particles.
            for q in range(P):
                if q == p:
                    continue
                x_q = x[q]
                f_q = x_q[:3*M].reshape(M, 3)
                sq = s_list[q]
                r = bp[:, None, :] - sq[None, :, :]
                Gmatrix = G_vec(r)
                v = np.einsum('nmij,mj->ni', Gmatrix, f_q) # boundary velocity on p due to particle q
                w[:3*N] += v.reshape(3*N)

            rhs = F_tilde_list[p] - w
            x[p] = B_inv_list[p] @ rhs

        max_diff = max(np.linalg.norm(x[p]-old_solutions[p]) for p in range(P))
        if max_diff < tol:
            print(f"Converged after {iteration+1} iterations (max diff = {max_diff:e})")
            break
        else:
            print(f"Iteration {iteration+1}: max diff = {max_diff:e}")
    else:
        raise RuntimeError("Solver did not converge")
    return x

