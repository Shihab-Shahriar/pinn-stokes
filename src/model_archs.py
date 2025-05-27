import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

viscosity = 1.0 #hardcoded

class SelfInteraction(nn.Module):
    """
    Neural network that predicts the 4 key self-mobility scalars for an
    axisymmetric particle in Stokes flow:
       mu_T_parallel, mu_T_perp, mu_R_parallel, mu_R_perp
    """
    def __init__(self, input_dim):
        super(SelfInteraction, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            #nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)

    @torch.jit.export
    def predict_velocity(self, x, force, viscosity):
        """
        force:  tensor of shape (batch_size, 6) -> [Fx, Fy, Fz, Tx, Ty, Tz]
        
        Steps:
        1) Forward pass to get the 4 mobility scalars.
        2) Construct the diagonal 3x3 blocks M^{TT}, M^{RR} because
           the particle's major axis is z-axis.
        3) Multiply by force => velocity [Ux, Uy, Uz, Ox, Oy, Oz].
        """
        # 1) Get the mobility scalars
        mus = self.forward(x)  
        mu_T_para = mus[:, 0]  # (batch_size,)
        mu_T_perp = mus[:, 1]
        mu_R_para = mus[:, 2]
        mu_R_perp = mus[:, 3]

        F_3    = force[:, 0:3]  # (batch_size, 3)
        tau_3  = force[:, 3:6]  # (batch_size, 3)

        # 2) For each sample, build M^{TT} and M^{RR} as diag:
        #    M^{TT} = diag(mu_T_perp, mu_T_perp, mu_T_para)
        #    M^{RR} = diag(mu_R_perp, mu_R_perp, mu_R_para)
        # We'll do it in a batched way:
        
        # expand dims for broadcasting
        mu_T_para_ = mu_T_para.view(-1, 1)  # (batch_size,1)
        mu_T_perp_ = mu_T_perp.view(-1, 1)
        mu_R_para_ = mu_R_para.view(-1, 1)
        mu_R_perp_ = mu_R_perp.view(-1, 1)

        # velocity in the z-axis aligned frame:
        #   Ux = mu_T_perp * Fx
        #   Uy = mu_T_perp * Fy
        #   Uz = mu_T_para * Fz
        Ux = mu_T_perp_ * F_3[:,0:1]
        Uy = mu_T_perp_ * F_3[:,1:2]
        Uz = mu_T_para_ * F_3[:,2:3]

        # angular velocity:
        #   Ox = mu_R_perp * Tx
        #   Oy = mu_R_perp * Ty
        #   Oz = mu_R_para * Tz
        Ox = mu_R_perp_ * tau_3[:,0:1]
        Oy = mu_R_perp_ * tau_3[:,1:2]
        Oz = mu_R_para_ * tau_3[:,2:3]

        U = torch.cat([Ux, Uy, Uz], dim=1)       # (batch_size, 3)
        Omega = torch.cat([Ox, Oy, Oz], dim=1)   # (batch_size, 3)

        # Combine
        velocity = torch.cat([U, Omega], dim=1)  # (batch_size, 6)
        return velocity
    
@torch.jit.script
def L1(d):
    """ Computes the outer product of each 3D vector in the batch with itself. """
    # d: [batch_size, 3]
    return torch.einsum('bi,bj->bij', d, d)  # [batch_size, 3, 3]

@torch.jit.script
def L2(d):
    """ Returns the matrix (I - outer(d, d)) for each vector in the batch. """
    # Identity tensor expanded to batch size
    batch_size = d.shape[0]
    I = torch.eye(3, device=d.device).unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, 3, 3]
    ddT = torch.einsum('bi,bj->bij', d, d)  # [batch_size, 3, 3]
    return I - ddT


@torch.jit.script
def L3(d):
    """ Computes the cross product matrix for each 3D vector in the batch. """
    # Using einsum for batched matrix-vector multiplication:
    levi_civita_data = torch.zeros(3, 3, 3, device=d.device)
    levi_civita_data[0, 1, 2] = 1
    levi_civita_data[1, 2, 0] = 1
    levi_civita_data[2, 0, 1] = 1
    levi_civita_data[0, 2, 1] = -1
    levi_civita_data[2, 1, 0] = -1
    levi_civita_data[1, 0, 2] = -1
    return torch.einsum('ijk,bk->bij', levi_civita_data, d)  # [batch_size, 3, 3]


class TwoBodySphere(nn.Module):
    def __init__(self, input_dim):
        super(TwoBodySphere, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
            #nn.Tanh()
        )
        
    def forward(self, r):
        x = self.layers(r)
        return x

    @torch.jit.export
    def predict_mobility(self, X):
        d_vec, r = X[:,:3], X[:,3]
        sc = self.forward(X[:, 3:])

        d_vec = -d_vec/ r.unsqueeze(-1) # negative,cz dvec=target-src
        TT = sc[:, 0].unsqueeze(1).unsqueeze(2) * L1(d_vec) + \
                sc[:, 1].unsqueeze(1).unsqueeze(2) * L2(d_vec) # TODO: d_vec or r?
        RT = sc[:, 2].unsqueeze(1).unsqueeze(2) * L3(d_vec)
        RR = sc[:, 3].unsqueeze(1).unsqueeze(2) * L1(d_vec) + \
                sc[:, 4].unsqueeze(1).unsqueeze(2) * L2(d_vec)
    
        K = torch.zeros((len(X), 6, 6), dtype=torch.float32, device=X.device)

        # After experiments, the kernel is NOT symmetric. 
        # Top-right and bottem left should NOT be transpose of each other
        K[:, :3, :3] = TT  # Top-left block
        K[:, 3:, :3] = RT  # Bottom-left block
        K[:, :3, 3:] = RT  # Top-right block (transpose of B)
        K[:, 3:, 3:] = RR  # Bottom-right block

        for i in range(len(X)):
            if not torch.linalg.eigvals(K[i]).real.min() > -1e-4:
                print(i, X[i])
                assert False, "Mobility Kernel is not SPD"
        return K
    
    @torch.jit.export
    def predict_velocity(self, X, force, viscosity):
        M = self.predict_mobility(X)/viscosity
        # print("M", M.shape)
        # print("force", force.unsqueeze(-1).shape)
        velocity = torch.bmm(M, force.unsqueeze(-1)).squeeze(-1)
        return velocity
    

class TwoBodyProlate(TwoBodySphere):
    def __init__(self, input_dim):
        super(TwoBodyProlate, self).__init__(input_dim)
        H1, H2 = 64, 128
        self.layers = nn.Sequential(
            nn.Linear(input_dim, H1),
            nn.ReLU(),
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.Linear(H2, H1),
            nn.ReLU(),
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.Linear(H2, H1),
            nn.Tanh(),
            nn.Linear(H1, 5),
            #nn.Tanh()
        )


if __name__ == "__main__":
    do_models = {
        "self_interaction": False,
        "two_body_sphere": False,
        "two_body_prolate": True
    }
    r = "/home/shihab/repo/experiments/"

    if do_models["self_interaction"]:
        self_model = SelfInteraction(9).to(device)

        self_model.load_state_dict(torch.load(r+"self_interaction.wt", weights_only=True))
        self_model.eval()

        # TorchScript-compile (script) the entire model
        scripted_self_model = torch.jit.script(self_model)
        scripted_self_model.save("data/models/self_interaction_model.pt")
        print("Saved TorchScript model to self_interaction_model.pt")

    if do_models["two_body_prolate"]:
        model = TwoBodyProlate(10).to(device)
        model.load_state_dict(torch.load(r+"prolate_2body.wt", weights_only=True))
        model.eval()

        # TorchScript-compile (script) the entire model
        scripted_model = torch.jit.script(model)
        scripted_model.save("data/models/two_body_prolate_model.pt")
        print("Saved TorchScript model to two_body_sphere_model.pt")

    if do_models["two_body_sphere"]:
        # Sphere
        model = TwoBodySphere(4).to(device)
        model.load_state_dict(torch.load(r+"sphere_2body.wt", weights_only=True))
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.save("data/models/two_body_sphere_model.pt")
        print("Saved TorchScript model to two_body_sphere_model.pt")