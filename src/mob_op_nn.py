"""
Grand Mobility operator (M) using self and two body neural networks.


+ Only for sphere for now
+ O(N^2) complexity. 
+ Uses M_ij=RPY when distance(M_ij) > 6 for two body.
"""

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation

from analysis_utils import quaternion_to_6d_batch
from model_archs import SelfInteraction, TwoBodySphere, TwoBodyProlate
from mfs_utils import min_distance_two_ellipsoids

import sys 
sys.path.append("/home/shihab/repo/utils")
from viz3d import save_multiple_ellipsoids_legacy_vtk 

# NOTE orientation[i] is always the rotation needed to convert z-axis to the 
# major axis of the particle. That is,
#   - from body->lab: orientation[i]
#   - from lab->body: orientation[i].inv()

# chatgpt
def compute_stokeslet_mobility(r_vec, viscosity):
    """
    Far-field Stokeslet (Oseen) tensor for a point-force approximation.
    Returns a 6×6 block with only the 3×3 force→velocity sub-block filled.
    """
    r = np.linalg.norm(r_vec)
    if r == 0:
        raise ValueError("Distance between particles must be non-zero.")
    r_hat = r_vec / r
    I3 = np.eye(3)
    M_tt = (I3 + np.outer(r_hat, r_hat)) / (8 * np.pi * viscosity * r)

    K = np.zeros((6, 6))
    K[:3, :3] = M_tt                # force → translational velocity
    # all other couplings are O(r⁻³) or higher → neglected here
    return K

def compute_rot_batch(rot_partial_flat):
    # from 6d to Rotation object
    N = rot_partial_flat.shape[0]
    R_partial = rot_partial_flat.reshape(N, 3, 2)
    R3 = np.cross(R_partial[:, :, 0], R_partial[:, :, 1])
    R_full = np.concatenate((R_partial, R3[:, :, np.newaxis]), axis=2)  # shape (N,3,3)
    #rotvecs = Rotation.from_matrix(R_full).as_rotvec()  # shape (N,3)
    return Rotation.from_matrix(R_full)

def rotate_forces_torques(forces_lab, torques_lab, rot, lab_to_body=True):
    """
    Rotate forces & torques.
    
    forces_lab, torques_lab, rotvec all have shape (N,3).
    """
    # Create a batch of Rotation objects from each rotvec
    #rot = Rotation.from_rotvec(rotvec)   # shape: (N,) - a stack of N rotations
    
    # 'inverse=True' rotates vectors *into* the body frame
    forces_body = rot.apply(forces_lab, inverse=lab_to_body)
    torques_body = rot.apply(torques_lab, inverse=lab_to_body)

    # rotate back for testing
    forces_back = rot.inv().apply(forces_body, inverse=not lab_to_body)
    torques_back = rot.inv().apply(torques_body, inverse=not lab_to_body)
    assert np.allclose(forces_back, forces_lab)
    assert np.allclose(torques_back, torques_lab)
    
    return forces_body, torques_body


class NNMob:
    """
    Forces and torque has to be converted to particle's body (i.e. local) frame
    for self interaction. For two body interaction, we need to rotate the forces
    """
    def __init__(self, shape, self_nn_path, two_nn_path, two_nn_path_F1,
                 nn_only: bool = False,          
                 rpy_only: bool = False,         
                 switch_dist: float = 6.0):      # distance threshold
        assert not (nn_only and rpy_only), "`nn_only` and `rpy_only` are mutually exclusive."

        self.shape = shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        axes = {
            'sphere': [1.0, 1.0, 1.0],
            'prolateSpheroid': [1.0, 1.0, 3.0],
            'oblateSpheroid': [2.0, 2.0, 1.0],
        }
        self.abc = np.array(axes[shape], dtype=np.float64) 

        self.self_nn= torch.jit.load(self_nn_path, map_location=self.device).eval()
        self.two_nn = torch.jit.load(two_nn_path, map_location=self.device).eval()
        self.two_nn_F1 = torch.jit.load(two_nn_path_F1, map_location=self.device).eval()
        
        self.nn_only   = nn_only
        self.rpy_only  = rpy_only
        self.switch_dist = switch_dist  # distance at which we fall back to RPY

        # print("NNMob shape: ", shape)
        # print("NNMob axes: ", axes)

    def get_self_vel_analytical(self, orientations, force, mu):
        """
        force:  (B,6) = [Fx,Fy,Fz, Tx,Ty,Tz]
        radius: (B,)   sphere radius per sample
        returns (B,6) = [Ux,Uy,Uz, Ox,Oy,Oz]
        """
        F = force[:, :3]
        T = force[:, 3:]
        inv_6pi_mu_a  = (1.0 / (6.0 * torch.pi * mu))
        inv_8pi_mu_a3 = (1.0 / (8.0 * torch.pi * mu))
        U = inv_6pi_mu_a  * F
        Omega = inv_8pi_mu_a3 * T
        return np.concatenate([U, Omega], axis=1)


    def get_self_vel(self, orientations, force, viscosity):
        N = len(orientations)
        assert force.shape == (len(orientations), 6)
        
        # source of a previous bug
        orient6d = quaternion_to_6d_batch(orientations)
        
        abc = np.repeat(self.abc[None, :], N, axis=0)
        input_self = np.concatenate((abc, orient6d), axis=1) # shape (N, 9)

        # Rotate forces and torques from lab to body frame
        f, t = rotate_forces_torques(force[:, :3], force[:, 3:], 
                                                        orientations, lab_to_body=True)
        force_body = np.concatenate((f, t), axis=1) # shape (N, 6)

        # Convert to tensors
        X_self = torch.tensor(input_self, dtype=torch.float32, device=self.device)
        force_tensor = torch.tensor(force_body, dtype=torch.float32, device=self.device)

        print("Sums of tensors going to NN:", X_self.sum(), force_tensor.sum())

        viscosity_tensor = torch.tensor(viscosity, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v_self = self.self_nn.predict_velocity(X_self, force_tensor, viscosity_tensor).cpu().numpy()
        
        # Rotate the velocities back to the lab frame
        v_self[:, :3], v_self[:, 3:] = rotate_forces_torques(
                                        v_self[:, :3], v_self[:, 3:], 
                                        orientations, lab_to_body=False) #body to lab

        return v_self
    
    
    def compute_rpy_mobility(self, c2):
        import sys
        sys.path.insert(0, '/home/shihab/hignn')
        from grpy_tensors import mu 
    
        c = np.array([[0.0, 0.0, 0.0], list(c2)])
        radii = np.array([1.0, 1.0])  # Example radii of the particles
        
        M = mu(c, radii,blockmatrix=True)
        res = M[:,:,0,1,:,:] #M_ji, not M_ij. (RT,TR components vary)
        K = np.zeros((6,6))
        K[:3,:3] = res[0,0]
        K[:3,3:] = res[0,1]
        K[3:,:3] = res[1,0]
        K[3:,3:] = res[1,1]

        return K

    def get_two_vel_rpy(self, pos, orientations, force):
        """
        Compute pairwise RPY velocity from all other particles (ignoring self-interaction).

        Parameters
        ----------
        pos : (N, 3) array
            Positions of N particles.
        orientations : (N, 3) array (or (N, 6) if including rotation)
            Unused here, but provided for future extension.
        force : (N, 6) array
            Force (F_x, F_y, F_z, torque_x, torque_y, torque_z) on each particle.

        Returns
        -------
        velocities : (N, 6) array
            The resulting velocity (v_x, v_y, v_z, ω_x, ω_y, ω_z) for each particle,
            summing contributions from all other particles only.
        """
        N = len(pos)
        velocities = np.zeros((N, 6), dtype=float)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue  # skip self-interaction

                c2 = pos[j] - pos[i]

                # Compute the 2-body mobility block: a 6x6 array
                # describing how j's force contributes to i's velocity
                K_ij = self.compute_rpy_mobility(c2)

                # Accumulate contribution from particle j's force
                velocities[i] += K_ij @ force[j]

        return velocities
                    

    def get_two_vel(self, pos, orientations, force, viscosity):
        """
        Assumptions: Particles don't already overlap.

        Features for NN are:
        ['center_x', 'center_y', 'center_z', 'dist', 'min_dist', 'quat_6d_1', 
        'quat_6d_2', 'quat_6d_3', 'quat_6d_4', 'quat_6d_5', 'quat_6d_6']
        """

        N = len(orientations)
        #print("for N=", N)
        assert pos.shape == (len(orientations), 3)
        assert force.shape == (len(orientations), 6)

        velocities = np.zeros((N, 6))

        rpy_freq = []
        rpy_dists_all = []

        for t in range(N):
            features = []
            forces_t = []
            forces_s = []
            min_dist_all = np.inf
            
            rpy_count = 0
            rpy_dists = np.inf
            for s in range(N):
                if s == t:
                    continue

                # NOTE src-target here, we're just translating so p_i is at the origin
                # predict_velocity() reverses the direction 
                center2 = pos[s] - pos[t] 
                dist = np.linalg.norm(center2)

                # if dist> self.switch_dist:
                #     continue

                #print(dist)
                
                use_rpy       = (
                    self.shape == 'sphere'
                    and (self.rpy_only or (not self.nn_only and dist > self.switch_dist))
                )
                use_stokeslet = (
                    self.shape != 'sphere'
                    and (not self.nn_only)            # honour nn_only
                    and dist > 10.0                   # hard-coded cut-off
                )

                # RPY path ------------------------------------------------------
                if use_rpy:          
                    assert self.shape == 'sphere'
                    K_ij = self.compute_rpy_mobility(center2)
                    velocities[t] += K_ij @ force[s]
                    rpy_count += 1
                    rpy_dists = min(rpy_dists, dist)
                    continue  # next neighbour

                if use_stokeslet:
                    assert self.shape != 'sphere'
                    K_ij = compute_stokeslet_mobility(center2, viscosity)
                    velocities[t] += K_ij @ force[s]
                    continue  # next neighbour

                if self.shape=='sphere':
                    min_dist = dist - 2.0
                else: 
                    min_dist = min_distance_two_ellipsoids(
                        self.abc[0], self.abc[1], self.abc[2], pos[t], orientations[t],
                        self.abc[0], self.abc[1], self.abc[2], pos[s], orientations[s],
                        n_starts=20) 
                
                #assert np.allclose(min_dist, dist-2.0, 1e-3), f"{min_dist=} {dist=}" # for spheres of radius 1.0
                min_dist_all = min(min_dist_all, min_dist)


                # TODO: I am not very confident about these rotations. Even if there 
                # is a bug, we might not catch it with sphere.

                # rotation needed to align p_t to z-axis
                R_sys = orientations[t].inv()
                
                # Apply this rotation to force2, torque2
                F1_new, T1_new = R_sys.apply(force[t, :3]), R_sys.apply(force[t, 3:])
                F2_new, T2_new = R_sys.apply(force[s, :3]), R_sys.apply(force[s, 3:])
                center2 = R_sys.apply(center2)

                # But not to R_j. In training, we defined R_j as the rotation needed
                # to align p1's major axis to p2's major axis. 
                R_rel = orientations[s] * orientations[t].inv()

                # TODO Update: I think center2 needs to be rotated as well

                rot6d = quaternion_to_6d_batch([R_rel])[0]

                if self.shape=='sphere':
                    median = 5.01 # Constant based on dataset
                    r2 = (dist - median)**2
                    r4 = r2 * r2
                    inp_feat = np.concatenate((center2, [dist, r2, r4, min_dist]))
                    assert inp_feat.shape == (7,)
                else:
                    median = 7.034 # Constant based on dataset
                    r2 = (dist - median)**2
                    r4 = r2 * r2
                    inp_feat = np.concatenate((center2, [dist, min_dist], rot6d, [r2, r4]))
                    assert inp_feat.shape == (13,)

                force_torque_s = np.concatenate((F1_new, T1_new))
                force_torque_t = np.concatenate((F2_new, T2_new))

                features.append(inp_feat)
                forces_s.append(force_torque_s)
                forces_t.append(force_torque_t)

                assert force_torque_s.shape == (6,) == force_torque_t.shape

                # if t==1 and s==2:
                #     print("features", inp_feat)
                #     print("forces_s", force_torque_s)
                #     print("forces_t", force_torque_t)
                #     print("min_dist", min_dist)
                #     print("dist", dist)
                #     print("min_dist_all", min_dist_all)

            
            rpy_freq.append(rpy_count)
            rpy_dists_all.append(rpy_dists)
            if features:  # could be empty for N=1
                assert self.rpy_only==False, "If rpy_only, we should not have reached here."
                X   = torch.tensor(np.array(features), dtype=torch.float32, device=self.device)
                Fs  = torch.tensor(np.array(forces_s), dtype=torch.float32, device=self.device)
                Ft  = torch.tensor(np.array(forces_t), dtype=torch.float32, device=self.device)
                mu  = torch.tensor(viscosity, dtype=torch.float32, device=self.device)
                
                # if t==0:
                #     print("X:", X[1])  
                #     print("Ft:", Ft[1])
                #     print("mu:", mu)

                with torch.no_grad():
                    v_t = self.two_nn.predict_velocity(X, Ft, mu).cpu().numpy()
                    v_s = self.two_nn_F1.predict_velocity(X, Fs, mu).cpu().numpy()
                    v_two = v_t + v_s

                    assert v_two.shape == (len(features), 6)

                # rotate back to lab frame & accumulate
                v_two = v_two.astype(np.float64)
                # if t==0:
                #     print("v_two before rot:", v_two[1])

                v_two[:, :3] = orientations[t].apply(v_two[:, :3])
                v_two[:, 3:] = orientations[t].apply(v_two[:, 3:])
                # if t==0:
                #     print("v_two after rot:", v_two[1])
                velocities[t] += v_two.sum(axis=0)
        
        print("RPY count per particle:", rpy_freq)
        # print("RPY distances per particle:", rpy_dists_all)
        return velocities

        
    def apply(self, config, force, viscosity):
        """
        For a given particle configuration, apply forces on the particles
        and return the velocity.

        Parameters
        ----------
        config : (N, 7) ndarray
            Positions and orientations (quaternion, scalar-last) of the particles.
        force : (N, 6) ndarray
            Force and torque vector to apply.
        
        Returns
        -------
        velocity : (N, 6) ndarray
            Translational and angular velocity of the rigid particle.
        """

        # Extract the positions and orientations
        N = config.shape[0]
        assert config.shape == (N, 7)

        orientations = Rotation.from_quat(config[:, 3:], scalar_first=False) # (x, y, z, w) 
        pos = config[:, :3]

        #v_self = self.get_self_vel(orientations, force, viscosity)
        v_self = self.get_self_vel_analytical(orientations, force, viscosity)
        
        v_two = self.get_two_vel(pos, orientations, force, viscosity)

        return v_self + v_two
        #return v_two

# --------------------------------------------------------
# BASIC TESTING CODE BELOW

def check_against_ref(mob, path, print_stuff=False):
    # Load the reference data
    viscosity = 1.0
    df = pd.read_csv(path, float_precision="high",
                        header=0, index_col=False)
    #print(df.head())

    numParticles = df.shape[0]    
    config = df[["x","y","z","q_x","q_y","q_z","q_w"]].values
    forces = df[["f_x","f_y","f_z","t_x","t_y","t_z"]].values
    velocity = df[["v_x","v_y","v_z","w_x","w_y","w_z"]].values
    
    v = mob.apply(config, forces, viscosity)
    
    np.set_printoptions(precision=5, suppress=True)

    lin_avg_rmse = 0
    ang_avg_rmse = 0
    for i in range(numParticles):
        lin_rmse = np.sqrt(np.mean((v[i, :3] - velocity[i, :3])**2))
        ang_rmse = np.sqrt(np.mean((v[i, 3:] - velocity[i, 3:])**2))
        
        if print_stuff:
            print(i)
            print(f"linear: {v[i, :3]} {velocity[i, :3]} {lin_rmse:.4f}")
            print(f"angular: {v[i, 3:]} {velocity[i, 3:]} {ang_rmse:.4f}")

        lin_avg_rmse += lin_rmse
        ang_avg_rmse += ang_rmse

    lin_avg_rmse /= numParticles
    ang_avg_rmse /= numParticles
    print(f"Avg linear RMSE: {lin_avg_rmse:.9f}")
    print(f"Avg angular RMSE: {ang_avg_rmse:.9f}")

    err_2b = np.linalg.norm(velocity - v, axis=1).mean()
    print(f"Avg 2-norm error: {err_2b:.9f}")
    return config

def check_against_ref_old(mob, path, print_stuff=False):
    # Load the reference data
    viscosity = 1.0
    df = pd.read_csv(path, float_precision="high",
                        header=0, index_col=False)
    #print(df.head())

    # Create 3D scatter plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract position data
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    
    # Create scatter plot
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.6)
    
    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Particle Positions 3D Scatter Plot')
    
    # Add colorbar
    plt.colorbar(scatter)
    
    # Save as PNG
    plt.savefig('particle_positions_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    #print("3D scatter plot saved as 'particle_positions_3d.png'")


    numParticles = df.shape[0]    
    config = df[["x","y","z","q_x","q_y","q_z","q_w"]].values
    forces = df[["f_x","f_y","f_z","t_x","t_y","t_z"]].values
    velocity = df[["v_x","v_y","v_z","w_x","w_y","w_z"]].values


    
    v = mob.apply(config, forces, viscosity)
    
    np.set_printoptions(precision=5, suppress=True)

    lin_avg_rmse = 0
    ang_avg_rmse = 0
    lin_rmse_list = []
    ang_rmse_list = []
    for i in range(numParticles):
        #lin_rmse = np.sqrt(np.mean((v[i, :3] - velocity[i, :3])**2))
        lin_rmse = max(np.abs(v[i, :3] - velocity[i, :3]))
        ang_rmse = max(np.abs(v[i, 3:] - velocity[i, 3:]))
        lin_rmse_list.append(round(lin_rmse, 4))
        ang_rmse_list.append(round(ang_rmse, 4))
        if print_stuff:
            #print(i)
            # print(f"linear: {v[i, :3]} {velocity[i, :3]} {lin_rmse:.4f}")
            # print(f"angular: {v[i, 3:]} {velocity[i, 3:]} {ang_rmse:.4f}")
            pass
        

        lin_avg_rmse += lin_rmse
        ang_avg_rmse += ang_rmse
    
    print(lin_rmse_list)
    print(ang_rmse_list)

    lin_avg_rmse /= numParticles
    ang_avg_rmse /= numParticles
    print(f"Avg linear RMSE: {lin_avg_rmse:.4f}")
    print(f"Avg angular RMSE: {ang_avg_rmse:.4f}")

    # component-wise error
    print("Component-wise RMS error:")
    print(np.sqrt(np.mean((v - velocity)**2, axis=0)))


    return config

def helens_3body_sphere(mob, S, shape="sphere", ):
    R = 1.0  

    D = 2*R + S

    # Centers. Notice the plane they are on and force direction
    centers = np.array([
        [0.0,    0.0,     0.0],
        [D,      0.0,     0.0],
        [D/2.0,  0.0, (np.sqrt(3)/2.0)*D]
    ], dtype=np.float64)

    F = 1.0 * 6* np.pi  # choose some magnitude
    print(F)
    F_ext_list = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], 
        [0.0, 0.0, -F]
    ], dtype=np.float64)

    # No external torque
    T_ext_list = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float64)

    r = Rotation.identity().as_quat(scalar_first=False)
    orientations = np.array([r, r, r], dtype=np.float64)
    C = np.concatenate((centers, orientations), axis=1)
    forces = np.concatenate((F_ext_list, T_ext_list), axis=1)

    v = mob.apply(C, forces, 1.0)
    print("NNOP:")
    print("U1:", v[2, 2])
    print("U2:", v[0, 2])
    print("U3:", v[0, 0])
    print("O:", v[0, 4])

    # MFS
    acc = "fine"
    # Load target (particle #1) geometry from files.
    boundary1 = np.loadtxt(f'/home/shihab/src/mfs/points/b_{shape}_{acc}.txt', dtype=np.float64)
    source1 = np.loadtxt(f'/home/shihab/src/mfs/points/s_{shape}_{acc}.txt', dtype=np.float64)
    
    from mfs_utils import build_B
    from mfs import imp_mfs_mobility_vec

    B_orig = build_B(boundary1, source1, np.zeros(3))
    B1_inv = np.linalg.pinv(B_orig)

    B_inv_list = [B1_inv, B1_inv, B1_inv]
    boundary2 = boundary1 + centers[1].reshape(1,3)
    boundary3 = boundary1 + centers[2].reshape(1,3)
    # same source points, just translated
    source2 = source1 + centers[1].reshape(1,3)
    source3 = source1 + centers[2].reshape(1,3)

    b_list = [boundary1, boundary2, boundary3]
    s_list = [source1, source2, source3]
    V_tilde_list = imp_mfs_mobility_vec(b_list, s_list, F_ext_list, T_ext_list, B_inv_list,
                                        max_iter=300, tol=1e-7)
    

    M1 = s_list[0].shape[0]
    solution_1 = V_tilde_list[0]
    vel_1 = solution_1[3*M1 : 3*M1 + 6]

    solution_3 = V_tilde_list[2]
    vel_3 = solution_3[3*M1 : 3*M1 + 6]

    np.set_printoptions(precision=5, suppress=True)
    print("MFS:",)
    print("U1:", vel_3[2])
    print("U2:", vel_1[2])
    print("U3:", vel_1[0])
    print("O:", vel_1[4])

    print("L2 norm diff:", np.linalg.norm(v[0] - vel_1))

    # print()
    # print(v)





def mainnnn():
    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/two_body_sphere_model.pt"
    two_body_F1 = "data/models/two_body_sphere_model_F1.pt"
    mob = NNMob("sphere", self_path, two_body, two_body_F1,
                nn_only=False, rpy_only=False, switch_dist=6.0)

    #path = "/home/shihab/repo/tmp/reference_sphere_4.0.csv"
    #path = "/home/shihab/repo/tmp/uniform_sphere_0.05.csv"
    path = "/home/shihab/repo/data/reference_sphere.csv"
    config = check_against_ref(mob, path)

    print("Just RPY...\n")
    mob = NNMob("sphere", self_path, two_body, two_body_F1,
            nn_only=False, rpy_only=True)
    config = check_against_ref(mob, path)





if __name__ == "__main__":
    mainnnn()