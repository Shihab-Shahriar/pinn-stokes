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

from analysis_utils import shuffle_and_split, convert_to_tensors
from model_archs import SelfInteraction, TwoBodyInteraction
from mfs_utils import min_distance_ellipsoids

import sys 
sys.path.append("/home/shihab/repo/utils")
from viz3d import save_multiple_ellipsoids_legacy_vtk 

# NOTE orientation[i] is always the rotation needed to convert z-axis to the 
# major axis of the particle. That is,
#   - from body->lab: orientation[i]
#   - from lab->body: orientation[i].inv()

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
    Rotate forces & torques from lab frame to the body frame (which we'll
    assume is aligned with the spheroid's major axis = z-axis in the body frame).
    
    forces_lab, torques_lab, rotvec all have shape (N,3).
    """
    # Create a batch of Rotation objects from each rotvec
    #rot = Rotation.from_rotvec(rotvec)   # shape: (N,) - a stack of N rotations
    
    # 'inverse=True' rotates vectors *into* the body frame
    forces_body = rot.apply(forces_lab, inverse=lab_to_body)
    torques_body = rot.apply(torques_lab, inverse=lab_to_body)
    
    return forces_body, torques_body


class NNMob:
    """
    Forces and torque has to be converted to particle's body (i.e. local) frame
    for self interaction. For two body interaction, we need to rotate the forces
    """
    def __init__(self, self_nn_path, two_nn_path):
        self.self_nn = SelfInteraction(9)
        self.two_nn = TwoBodyInteraction(5)
        self.abc = np.array([1.0, 1.0, 1.0]) # Hardcoded for spheres

        self.self_nn.load_state_dict(torch.load(self_nn_path,weights_only=True))
        self.two_nn.load_state_dict(torch.load(two_nn_path,weights_only=True))
        self.self_nn.eval()
        self.two_nn.eval()

    def get_self_vel(self, pos, orientations, force):
        N = len(orientations)
        assert force.shape == (len(orientations), 6)
        
        orient6d = orientations.as_matrix()[:, :, :2].reshape(N, 6)
        
        abc = np.repeat(self.abc[None, :], N, axis=0)
        input_self = np.concatenate((abc, orient6d), axis=1) # shape (N, 9)

        # Rotate forces and torques from lab to body frame
        f, t = rotate_forces_torques(force[:, :3], force[:, 3:], 
                                                        orientations, lab_to_body=True)
        force_body = np.concatenate((f, t), axis=1) # shape (N, 6)

        # Convert to tensors
        X_self = torch.tensor(input_self, dtype=torch.float32)
        force_tensor = torch.tensor(force_body, dtype=torch.float32)

        with torch.no_grad():
            v_self = self.self_nn.predict_velocity(X_self, force_tensor).numpy()
        
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
                    



    def get_two_vel(self, pos, orientations, force):
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

        for t in range(N):
            features = []
            forces = []
            for s in range(N):
                if s == t:
                    continue

                # NOTE src-target here, we're just translating so p_i is at the origin
                # predict_velocity() reverses the direction 
                center2 = pos[s] - pos[t] 
                dist = np.linalg.norm(center2)
                min_dist = min_distance_ellipsoids(1.0, 1.0, 1.0, center2, orientations[j], n_starts=10)
                assert np.allclose(min_dist, dist-2.0, 1e-3), f"{min_dist=} {dist=}" # for spheres of radius 1.0

                # TODO: I am not very confident about these rotations. Even if there 
                # is a bug, we might not catch it with sphere.

                # rotation needed to align p_i to z-axis
                R_sys = orientations[t].inv()
                
                # Apply this rotation to force2, torque2
                F2_new, T2_new = R_sys.apply(force[s, :3]), R_sys.apply(force[s, 3:])

                # But not to R_j. In training, we defined R_j as the rotation needed
                # to align p1's major axis to p2's major axis. 
                R_rel = orientations[s] * orientations[t].inv()

                # TODO Update: I think center2 needs to be rotated as well
                center2 = R_sys.apply(center2)

                rot6d = R_rel.as_matrix()[:, :2].flatten()
                inp_feat = np.concatenate((center2, [dist, min_dist]))
                force_torque = np.concatenate((F2_new, T2_new))

                features.append(inp_feat)
                forces.append(force_torque)

                assert inp_feat.shape == (5,)
                assert force_torque.shape == (6,)

            features = torch.tensor(np.array(features), dtype=torch.float32)
            forces = torch.tensor(np.array(forces), dtype=torch.float32)
            #print(features.shape, forces.shape)

            with torch.no_grad():
                v_two = self.two_nn.predict_velocity(features, forces).numpy()
                assert v_two.shape == (N-1, 6)
            
            # project the velocities back to the lab frame
            # TODO: what if i rotate after summing up the velocities? - yes
            v_two = v_two.astype(np.float64)
            transv, rotv = v_two[:, :3], v_two[:, 3:]
            transv, rotv = orientations[t].apply(transv), orientations[t].apply(rotv)
            v_two = np.concatenate((transv, rotv), axis=1)

            velocities[t] = v_two.sum(axis=0)
            assert velocities[t].shape == (6,)
        
        return velocities

        
    def apply(self, config, force, rpy=False):
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

        v_self = self.get_self_vel(pos, orientations, force)
        
        print("Two body rpy?", rpy)
        if rpy:
            v_two = self.get_two_vel_rpy(pos, orientations, force)
        else:
            v_two = self.get_two_vel(pos, orientations, force)

        return v_self + v_two




if __name__ == "__main__":
    R = 1.0  
    S = 1.0   # separation between sphere centers equals R
    
    mob = NNMob("/home/shihab/repo/experiments/self_interaction.wt", 
                "/home/shihab/repo/experiments/sphere_2body.wt")
    
    D = 2*R + S

    # Centers. Notice the plane they are on and force direction
    # x0_1 = np.array([0.0,    0.0,     0.0], dtype=np.float64)
    # x0_2 = np.array([D,      0.0,     0.0], dtype=np.float64)
    # x0_3 = np.array([D/2.0,  0.0, (np.sqrt(3)/2.0)*D], dtype=np.float64)
    centers = np.array([
        [0.0,    0.0,     0.0],
        [D,      0.0,     0.0],
        [D/2.0,  0.0, (np.sqrt(3)/2.0)*D]
    ], dtype=np.float64)

    F = 1.0 * 6* np.pi  # choose some magnitude
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

    v = mob.apply(C, forces, True)

    np.set_printoptions(precision=5, suppress=True)
    print()
    print(v)

    # SPD check
    # M = np.zeros((18, 18))
    # for i in range(18):
    #     delta = np.zeros(18)
    #     delta[i] = 1.0
    #     delta = delta.reshape(3, 6)
    #     v = mob.apply(C, delta)
    #     M[:, i] = v.flatten()

    # eigens = np.linalg.eigvals(M)
    # print("Eigenvalues of M:", eigens)
    # print("Is M SPD?", np.all(eigens > 1e-4))

    # # Symmetry check. tol can't be too low cause NN
    # print("Is M symmetric?", np.allclose(M, M.T, atol=1e-4))


# if __name__ == "__main__":
#     mob = NNMob("/home/shihab/repo/experiments/self_interaction.wt", 
#                 "/home/shihab/repo/experiments/sphere_2body.wt")


#     xs,ys = [],[]

#     dataRoot = "/home/shihab/repo/data/"
#     xs.append(np.load(dataRoot+"X_self_oblateSpheroid_23:07_4.npy"))
#     ys.append(np.load(dataRoot+"Y_self_oblateSpheroid_23:07_4.npy"))

#     xs.append(np.load(dataRoot+"X_self_prolateSpheroid_23:09_4.npy"))
#     ys.append(np.load(dataRoot+"Y_self_prolateSpheroid_23:09_4.npy"))


#     xs.append(np.load(dataRoot+"X_self_sphere_22:42_4.npy"))
#     ys.append(np.load(dataRoot+"Y_self_sphere_22:42_4.npy"))

#     X = np.concatenate(xs, axis=0)
#     Y = np.concatenate(ys, axis=0)
#     print(X.shape, Y.shape) 

#     v = mob.apply(X[:, :9],X[:, 9:])


#     def mean_abs_err(val_output, val_velocity_tensor, npp=False):
#         # 6D vector: median % error for each vel component
#         valid_mask = np.abs(val_velocity_tensor) > 1e-6
        
#         filtered_y_tensor = np.where(valid_mask, val_velocity_tensor, np.nan)
#         relative_error = np.abs((val_output - filtered_y_tensor) / filtered_y_tensor)
        
#         a = np.nanmean(relative_error, axis=0)
#         return a*100

#     print(mean_abs_err(v, Y))

