"""
Grand Mobility operator (M) using self and two body neural networks.


+ Only for sphere for now
+ O(N^2) complexity. 
+ Uses M_ij=RPY when distance(M_ij) > 6 for two body.
"""

from experiments.all_models import CombinedModel
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation


import sys 
sys.path.append("/home/shihab/repo/utils")

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



class NNMob:
    """
    Forces and torque has to be converted to particle's body (i.e. local) frame
    for self interaction. For two body interaction, we need to rotate the forces
    """
    def __init__(self, nn_path,
                 nn_only: bool = False,          
                 rpy_only: bool = False,         
                 switch_dist: float = 6.0):      # distance threshold
        assert not (nn_only and rpy_only), "`nn_only` and `rpy_only` are mutually exclusive."

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.nn= CombinedModel(4, 4, 9).to(self.device)
        self.nn.load_state_dict(torch.load(nn_path, map_location=self.device))


        self.nn_only   = nn_only
        self.rpy_only  = rpy_only
        self.switch_dist = switch_dist  # distance at which we fall back to RPY


    def get_self_vel(self, force, viscosity):
        """
        Compute the isolated (single–sphere) self mobility contribution.

        For a rigid sphere of radius R in Stokes flow:
            V = F / (6 π μ R)
            Ω = T / (8 π μ R^3)

        Here the training/data assumes unit radius R = 1.0, but we keep R as
        a variable for clarity / future generalization.
        """
        assert force.ndim == 2 and force.shape[1] == 6, "force must be (N,6)"
        R = 1.0  # TODO: make configurable if non‑unit spheres are introduced

        F = force[:, :3]
        T = force[:, 3:]

        mob_T = 1.0 / (6.0 * np.pi * viscosity * R)
        mob_R = 1.0 / (8.0 * np.pi * viscosity * R**3)

        V = mob_T * F
        W = mob_R * T

        return np.concatenate([V, W], axis=1)

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

    def get_two_vel_rpy(self, pos, force):
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
                    

    def get_two_vel(self, pos, force, viscosity):
        """
        Assumptions: Particles don't already overlap.

        Features for NN are:
        ['center_x', 'center_y', 'center_z', 'dist', 'min_dist', 'quat_6d_1', 
        'quat_6d_2', 'quat_6d_3', 'quat_6d_4', 'quat_6d_5', 'quat_6d_6']
        """

        N = len(force)
        #print("for N=", N)
        assert pos.shape == (N, 3)
        assert force.shape == (N, 6)

        velocities = np.zeros((N, 6))

        mean_dist_s = 4.668671912939677

        rpy_dists_all = []

        for t in range(N):
            v_2b_self = np.zeros(6)
            v_2b_cross = np.zeros(6)
            features = []
            d_vecs   = []
            force_s  = []
            force_t =  []

            rpy_count = 0
            for s in range(N):
                if s == t:
                    continue

                # relative position
                dvec = pos[t] - pos[s]
                dist = np.linalg.norm(dvec)

                if self.rpy_only or (self.nn_only==False and dist > self.switch_dist):
                    # use RPY
                    rpy_dists_all.append(dist)
                    K_rpy = self.compute_rpy_mobility(-dvec) #src-target
                    v_2b_cross += K_rpy @ force[s]
                    rpy_count  += 1
                else:
                    # use NN
                    d_vec_s = dvec / dist  # unit vector from target to source

                    # ['dist_s', 'dist_s_sq', 'dist_s_sqsq', 'min_dist_s']
                    dist_s = dist - mean_dist_s
                    dist_s_sq = dist_s * dist_s
                    x_2b_s = [dist_s, dist_s_sq, dist_s_sq*dist_s_sq, dist - 2.0]

                    features.append(x_2b_s)
                    d_vecs.append(d_vec_s)
                    force_s.append(force[s])
                    force_t.append(force[t])


            if len(features) > 0:
                features = np.array(features, dtype=np.float32)
                d_vecs   = np.array(d_vecs, dtype=np.float32)
                force_s  = np.array(force_s, dtype=np.float32)
                force_t  = np.array(force_t, dtype=np.float32)

                # use nn
                with torch.no_grad():
                    X_2b_s = torch.tensor(features, device=self.device)
                    d_vec_s = torch.tensor(d_vecs, device=self.device)
                    force_s = torch.tensor(force_s, device=self.device)
                    force_t = torch.tensor(force_t, device=self.device)

                    K_2b_self_s = self.nn.two_body_self.predict_mobility(X_2b_s, d_vec_s)
                    K_2b_cross_s = self.nn.two_body_cross.predict_mobility(X_2b_s, d_vec_s)
                    v_2b_self_s = torch.bmm(K_2b_self_s, force_t.unsqueeze(-1)).squeeze(-1)
                    v_2b_cross_s = torch.bmm(K_2b_cross_s, force_s.unsqueeze(-1)).squeeze(-1)
                    v_2b_cross += v_2b_cross_s.sum(dim=0).cpu().numpy()
                    v_2b_self  += v_2b_self_s.sum(dim=0).cpu().numpy()

            velocities[t] = v_2b_self + v_2b_cross
            rpy_dists_all.append(rpy_count)

        print("RPY COUNT:", rpy_dists_all)
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

        pos = config[:, :3]

        v_self = self.get_self_vel(force, viscosity)
        v_two = self.get_two_vel(pos, force, viscosity)

        return v_self + v_two

# --------------------------------------------------------
# BASIC TESTING CODE BELOW

def check_against_ref(mob, path, print_stuff=False):
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
    
    print("3D scatter plot saved as 'particle_positions_3d.png'")


    numParticles = df.shape[0]    
    config = df[["x","y","z","q_x","q_y","q_z","q_w"]].values
    forces = df[["f_x","f_y","f_z","t_x","t_y","t_z"]].values
    velocity = df[["v_x","v_y","v_z","w_x","w_y","w_z"]].values
    
    v = mob.apply(config, forces, viscosity)
    
    np.set_printoptions(precision=5, suppress=True)

    lin_avg_rmse = 0
    ang_avg_rmse = 0
    for i in range(numParticles):
        #lin_rmse = np.sqrt(np.mean((v[i, :3] - velocity[i, :3])**2))
        lin_rmse = max(np.abs(v[i, :3] - velocity[i, :3]))
        ang_rmse = max(np.abs(v[i, 3:] - velocity[i, 3:]))
        
        if print_stuff:
            print(i)
            print(f"linear: {v[i, :3]} {velocity[i, :3]} {lin_rmse:.4f}")
            print(f"angular: {v[i, 3:]} {velocity[i, 3:]} {ang_rmse:.4f}")

        lin_avg_rmse += lin_rmse
        ang_avg_rmse += ang_rmse

    lin_avg_rmse /= numParticles
    ang_avg_rmse /= numParticles
    print(f"Avg linear RMSE: {lin_avg_rmse:.4f}")
    print(f"Avg angular RMSE: {ang_avg_rmse:.4f}")

    # component-wise error
    print("Component-wise RMS error:")
    print(np.sqrt(np.mean((v - velocity)**2, axis=0)))


    return config

def helens_3body_sphere():
    R = 1.0  
    S = 1.0   # separation between sphere centers equals R

    just_rpy = False
    print("Just RPY? -- ", just_rpy)

    self_path = "data/models/self_interaction_model.pt"
    two_body = "data/models/two_body_sphere_model.pt"
    two_body_F1 = "data/models/two_body_sphere_model_F1.pt"
    mob = NNMob(shape, self_path, two_body, two_body_F1,
                nn_only=False, rpy_only=just_rpy, switch_dist=6.0)
    

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

    np.set_printoptions(precision=5, suppress=True)
    print()
    print(v)




def main():
    nn_path = "/home/shihab/repo/experiments/all_models_sphere.wt"
    mob = NNMob(nn_path, nn_only=False, rpy_only=False, switch_dist=6.0)

    # max dist between spheres in the following is 12.78, something 
    # the model not trained to handle on nn_only mode
    path = "/home/shihab/repo/data/reference_sphere.csv"
    config = check_against_ref(mob, path, print_stuff=True)

    print("Just RPY...\n")
    mob = NNMob(nn_path, nn_only=False, rpy_only=True)
    config = check_against_ref(mob, path, print_stuff=True)


if __name__ == "__main__":
    main()
    #helens_3body_sphere()
