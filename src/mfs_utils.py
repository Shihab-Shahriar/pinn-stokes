import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

def S(w):
    """Skew-symmetric matrix for cross products: S(w)*v = w x v."""
    return np.array([
        [0,     -w[2],  w[1]],
        [w[2],   0,    -w[0]],
        [-w[1],  w[0],  0   ]
    ], dtype=float)

def G(r):
    """Oseen tensor for Stokes flow in 3D."""
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-15:
        return np.zeros((3,3))
    factor = 1.0/(8.0*np.pi)
    return factor * (np.eye(3)/r_norm + np.outer(r, r)/(r_norm**3))



def build_B(b, s, x0):
    """
    create matrix for mobility problem 
    """
    N = b.shape[0]
    M = s.shape[0]

    # A: 3N x 3M
    A_blocks = []
    for i in range(N):
        row_blocks = []
        for j in range(M):
            r_diff = b[i, :] - s[j, :]
            row_blocks.append(G(r_diff))
        A_blocks.append(np.hstack(row_blocks))
    A = np.vstack(A_blocks)

    # -V_block: N copies of I3
    V_block = np.tile(np.eye(3), (N, 1))  # (3N, 3)

    # -Omega_block: S(b_i - x0) for each i
    Omega_blocks = [S(b[i, :] - x0) for i in range(N)]
    Omega_block = np.vstack(Omega_blocks)  # (3N, 3)

    # Force eq: sum_j f_j = -F_ext => horizontally M copies of I3
    f_force_block = np.hstack([np.eye(3) for _ in range(M)])  # (3, 3M)

    # Torque eq: sum_j ( (s_j - x0) x f_j ) = -T_ext
    f_torque_blocks = [S(s[j, :] - x0) for j in range(M)]
    f_torque_block = np.hstack(f_torque_blocks)  # (3, 3M)

    B_top = np.hstack([A, -V_block, -Omega_block])
    B_mid_force = np.hstack([f_force_block, np.zeros((3,3)), np.zeros((3,3))])
    B_mid_torque = np.hstack([f_torque_block, np.zeros((3,3)), np.zeros((3,3))])
    B = np.vstack([B_top, B_mid_force, B_mid_torque])

    return B


def build_Q_transform(R, n):
    dim = 3 * (n + 2)  # total dimension
    Q = np.zeros((dim, dim), dtype=float)
    
    for i in range(n + 2):
        r0 = 3*i
        r1 = r0 + 3
        Q[r0:r1, r0:r1] = R
    return Q


def get_QM_QN(R, N, M):
    """
    Build the two large transformation matrices, QM and QN, so that:
        Binv1 = QM @ Binv1 @ QN.T
    
    Parameters
    ----------
    R : (3,3) ndarray or scipy.spatial.transform.Rotation
        Rotation matrix to apply.
    N : int
        Number of boundary points (so row dimension = 3N + 6).
    M : int
        Number of source points (so column dimension = 3M + 6).
    
    Returns
    -------
    QM : (3N+6, 3N+6) ndarray
        Block-diagonal rotation transform for the 'row side'.
    QN : (3M+6, 3M+6) ndarray
        Block-diagonal rotation transform for the 'column side'.
    """
    # if scipy rotation, convert to 3x3 matrix
    if hasattr(R, 'as_matrix'):
        R = R.as_matrix()
    assert R.shape == (3, 3)

    QM = build_Q_transform(R, M)
    QN = build_Q_transform(R, N)
    
    return QM, QN

def random_unit_vector():
    """Return a random unit vector in R^3, uniformly distributed on the sphere.
    
    TODO: Does this have any bias? 
    """
    vec = np.random.randn(3)
    return vec / np.linalg.norm(vec)



def random_orientation_spheroid():
    """
    Returns a Rotation (from scipy) that sends the z-axis to a random direction.
    Works for *only* prolate and oblate spheroids.
    """
    z_axis = np.array([0, 0, 1], dtype=float)
    v = random_unit_vector()  # random direction
    
    # If v is almost exactly z or -z, handle degenerate cases
    dot_zv = np.dot(z_axis, v)
    
    if np.allclose(v, z_axis):
        # no rotation needed
        return R.from_rotvec([0, 0, 0])
    elif np.allclose(v, -z_axis):
        # 180-degree rotation about x-axis (or y-axis, etc.)
        return R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
    else:
        # axis = z x v, angle = arccos(z·v)
        axis = np.cross(z_axis, v)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot_zv)
        return R.from_rotvec(axis * angle)


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


def min_distance_ellipsoids(a, b, c, center2, R2, n_starts=5):
    """
    Returns the minimum distance between two identical ellipsoids.
    """
    # Convert rotation input to matrix format
    if isinstance(R2, Rotation):
        R2_matrix = R2.as_matrix()
    else:
        R2_matrix = np.array(R2)
    
    # Objective function to minimize
    def objective(vars):
        x, y, z, u, v, w = vars
        point1 = np.array([x, y, z])
        point2 = R2_matrix @ np.array([u, v, w]) + center2
        return np.linalg.norm(point1 - point2)
    
    # Constraints for ellipsoid surfaces
    def con1(vars):
        x, y, z, u, v, w = vars
        return (x**2)/(a**2) + (y**2)/(b**2) + (z**2)/(c**2) - 1
    
    def con2(vars):
        x, y, z, u, v, w = vars
        return (u**2)/(a**2) + (v**2)/(b**2) + (w**2)/(c**2) - 1
    
    # Jacobians for constraints
    def con1_jac(vars):
        x, y, z, u, v, w = vars
        return [2*x/a**2, 2*y/b**2, 2*z/c**2, 0, 0, 0]
    
    def con2_jac(vars):
        x, y, z, u, v, w = vars
        return [0, 0, 0, 2*u/a**2, 2*v/b**2, 2*w/c**2]
    
    constraints = [
        {'type': 'eq', 'fun': con1, 'jac': con1_jac},
        {'type': 'eq', 'fun': con2, 'jac': con2_jac}
    ]
    
    # Track best solution
    min_dist = np.inf
    
    # Generate multiple initial guesses
    for _ in range(n_starts):
        # Random points on ellipsoids
        rand_dir1 = np.random.normal(size=3)
        rand_dir1 /= np.linalg.norm(rand_dir1)
        X0 = np.array([a*rand_dir1[0], b*rand_dir1[1], c*rand_dir1[2]])
        
        rand_dir2 = np.random.normal(size=3)
        rand_dir2 /= np.linalg.norm(rand_dir2)
        Y0 = np.array([a*rand_dir2[0], b*rand_dir2[1], c*rand_dir2[2]])
        
        initial_guess = np.concatenate([X0, Y0])
        
        # Optimize from this starting point
        result = minimize(objective, initial_guess, method='SLSQP',
                          constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-8})
        
        if result.success and result.fun < min_dist:
            min_dist = result.fun
            
    return min_dist

