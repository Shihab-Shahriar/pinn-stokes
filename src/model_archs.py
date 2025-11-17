import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime


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
        assert mus.min() >= -1e-8, "Mobility scalars must be non-negative"

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

        # for i in range(len(X)):
        #     if not torch.linalg.eigvals(K[i]).real.min() > -1e-4:
        #         print(i, X[i])
        #         assert False, "Mobility Kernel is not SPD"
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


class TwoBodyCombined(nn.Module):
    def __init__(self, input_dim):
        super(TwoBodyCombined, self).__init__()
        self.FtModel = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
            #nn.Tanh()
        )
        self.FsModel = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
            #nn.Tanh()
        )
        
        
    @torch.jit.export                    
    def make_mobility(
        self,
        sc: torch.Tensor,
        d_vec: torch.Tensor,
        is_Ms: bool         
    ) -> torch.Tensor:
        TT = sc[:, 0].unsqueeze(1).unsqueeze(2) * L1(d_vec) + \
                sc[:, 1].unsqueeze(1).unsqueeze(2) * L2(d_vec) # TODO: d_vec or r?
        RT = sc[:, 2].unsqueeze(1).unsqueeze(2) * L3(d_vec)
        RR = sc[:, 3].unsqueeze(1).unsqueeze(2) * L1(d_vec) + \
                sc[:, 4].unsqueeze(1).unsqueeze(2) * L2(d_vec)
    
        K = torch.zeros((len(sc), 6, 6), dtype=torch.float32, device=sc.device)

        # After experiments, the kernel is NOT symmetric. 
        # Top-right and bottem left should NOT be transpose of each other
        K[:, :3, :3] = TT  # Top-left block
        K[:, 3:, :3] = RT  # Bottom-left block
        K[:, :3, 3:] = RT.transpose(1, 2) if is_Ms else RT
        K[:, 3:, 3:] = RR  # Bottom-right block
        return K

    @torch.jit.export
    def predict_mobility(self, X):
        d_vec, r = X[:,:3], X[:,3]
        sc_t = self.FtModel(X[:, 3:]) #exclude d_vec
        sc_s = self.FsModel(X[:, 3:]) #exclude d_vec
        
        d_vec = -d_vec/ r.unsqueeze(-1) # negative,cz dvec=target-src

        K_t = self.make_mobility(sc_t, d_vec, False)
        K_s = self.make_mobility(sc_s, d_vec, True)
        return K_s, K_t
        
    @torch.jit.export
    def predict_velocity(self, X, force_s, force_t, viscosity):
        M_s, M_t = self.predict_mobility(X)
        M_s, M_t = M_s/viscosity, M_t/viscosity
        
        v_s = torch.bmm(M_s, force_s.unsqueeze(-1)).squeeze(-1)
        v_t = torch.bmm(M_t, force_t.unsqueeze(-1)).squeeze(-1)
        v =  v_t + v_s
        return v


class SelfAndTwoBody(nn.Module):
    def __init__(self, input_dim):
        super(SelfAndTwoBody, self).__init__()
        self.self_nn = nn.Sequential(
            nn.Linear(3, 64), # (a,b,c)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            #nn.Tanh()
        )
        self.FtModel = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
            #nn.Tanh()
        )
        self.FsModel = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
            #nn.Tanh()
        )

    @torch.jit.export
    def self_mobility(self, x):
        mus = self.self_nn.forward(x)  
        # if not self.training:
        #     assert mus.min()>=0, "negative coeff for self interaction {mus.min()}"
            
        diag_vals = torch.stack([
            mus[:,1],  # mu_T_perp → Ux
            mus[:,1],  # mu_T_perp → Uy
            mus[:,0],  # mu_T_para → Uz
            mus[:,3],  # mu_R_perp → Ox
            mus[:,3],  # mu_R_perp → Oy
            mus[:,2],  # mu_R_para → Oz
        ], dim=1)  # shape (N,6)
    
        # produces (N,6,6) with diag_vals on the diagonal
        return torch.diag_embed(diag_vals)
        
    @torch.jit.export
    def make_mobility(self, sc, d_vec, Ms=False):
        TT = sc[:, 0].unsqueeze(1).unsqueeze(2) * L1(d_vec) + \
                sc[:, 1].unsqueeze(1).unsqueeze(2) * L2(d_vec) # TODO: d_vec or r?
        RT = sc[:, 2].unsqueeze(1).unsqueeze(2) * L3(d_vec)
        RR = sc[:, 3].unsqueeze(1).unsqueeze(2) * L1(d_vec) + \
                sc[:, 4].unsqueeze(1).unsqueeze(2) * L2(d_vec)
    
        K = torch.zeros((len(sc), 6, 6), dtype=torch.float32, device=X.device)

        # After experiments, the kernel is NOT symmetric for M_t. 
        # Top-right and bottem left should NOT be transpose of each other
        K[:, :3, :3] = TT  # Top-left block
        K[:, 3:, :3] = RT  # Bottom-left block
        K[:, :3, 3:] = RT.transpose(1, 2) if Ms else RT
        K[:, 3:, 3:] = RR  # Bottom-right block

        if Ms:
            assert torch.allclose(K, K.transpose(1, 2), atol=1e-6)
        return K

    @torch.jit.export
    def predict_mobility(self, X):
        d_vec, r = X[:,:3], X[:,3]
        sc_t = self.FtModel(X[:, 3:]) #exclude d_vec
        sc_s = self.FsModel(X[:, 3:]) #exclude d_vec
        
        d_vec = -d_vec/ r.unsqueeze(-1) # negative,cz dvec=target-src

        K_t = self.make_mobility(sc_t, d_vec)
        K_s = self.make_mobility(sc_s, d_vec, Ms=True)
        return K_s, K_t
        
    @torch.jit.export
    def predict_velocity(self, X, force_s, force_t, return_M=False):
        M_s, M_t = self.predict_mobility(X)
        M_s, M_t = M_s/viscosity, M_t/viscosity
        
        v_s = torch.bmm(M_s, force_s.unsqueeze(-1)).squeeze(-1)
        v_t = torch.bmm(M_t, force_t.unsqueeze(-1)).squeeze(-1)

        self_feat = torch.ones((len(X), 3), device=v_s.device) # Just for sphere 
        M_self = self.self_mobility(self_feat)
        
        
        N = M_self.shape[0]
        M12 = torch.zeros((N, 12, 12),
                          device=M_self.device,
                          dtype=M_self.dtype)
        # diagonal blocks
        M12[:, :6, :6]   = M_self + M_s
        M12[:, 6:, 6:]   = M_self + M_s
        # off-diagonals
        M12[:, :6, 6:]   = M_t
        M12[:, 6:, :6]   = M_t.transpose(1,2)
    
        # 3) stack the 12D force vector [force_s; force_t]
        F12 = torch.cat([force_s, force_t], dim=1).unsqueeze(-1)  # (N,12,1)
    
        # 4) compute velocities: V12 = M12 @ F12 -> (N,12,1) -> squeeze to (N,12)
        V12 = torch.bmm(M12, F12).squeeze(-1)
    
        # split back into two 6D velocities
        v1 = V12[:, :6]
        v2 = V12[:, 6:]
    
        if return_M:
            return (torch.cat([v1, v2], dim=1), M12)
        else:
            return torch.cat([v1, v2], dim=1)


class MultiBodyCorrection(nn.Module):
    def __init__(self, input_dim, median_2b, mean_dist_s, dist_s_feat_loc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 5)
        )
        #self.two_nn = torch.jit.load(two_nn_path, map_location=device).eval()
        self.median_2b = median_2b
        self.mean_dist_s = mean_dist_s
        self.dist_s_feat_loc = dist_s_feat_loc

    def forward(self, X):
        return self.net(X)   

    def predict_velocity(self, X, force_s): 
        coeff = self.net(X[:,33:]) # 3:33 cols not being used
        d_vec = X[:,:3]
        dist_s_centered = X[:, self.dist_s_feat_loc]                  # col 'dist_s'
        r = dist_s_centered + self.mean_dist_s
        
        d_vec = -d_vec/ r.unsqueeze(-1) # negative,cz dvec=target-src
        
        par = L1(d_vec)                                   # [B,3,3]
        perp = L2(d_vec)
        angular = L3(d_vec)

        B0 = coeff[:,0][:,None,None]; B1 = coeff[:,1][:,None,None]
        C0 = coeff[:,2][:,None,None]; 
        D0 = coeff[:,3][:,None,None]; D1 = coeff[:,4][:,None,None]

        # if not self.training:
        #     print(torch.abs(B0.mean()),torch.abs(B1.mean()), torch.abs(C0.mean()), torch.abs(D0.mean()), torch.abs(D1.mean()))

        TT = B0*par + B1*perp 
        RT = C0*angular 
        RR = D0*par + D1*perp 

        K   = torch.zeros(X.shape[0],6,6, device=X.device, dtype=X.dtype)
        K[:,:3,:3] = TT
        K[:,3:,:3] = RT
        K[:,:3,3:] = RT
        K[:,3:,3:] = RR

        v = torch.einsum('bij,bj->bi', K, force_s)      # [B,6]
        return v


if __name__ == "__main__":
    do_models = {
        "self_interaction": False,
        "two_body_sphere": False,
        "two_body_prolate": False,
        "two_body_sphere_F1": False,
        "two_body_combined": True,
        "two_body_combined_all": False
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
        filename = r+"prolate_2body.wt"

        # when last modified r+"prolate_2body.wt"?
        ts = os.path.getmtime(filename)
        # Convert to human-readable datetime
        last_modified = datetime.datetime.fromtimestamp(ts)

        print("Last modified:", last_modified)

        model.load_state_dict(torch.load(filename, weights_only=True))
        model.eval()

        # TorchScript-compile (script) the entire model
        scripted_model = torch.jit.script(model)
        scripted_model.save("data/models/two_body_prolate_model.pt")
        print("Saved TorchScript model to two_body_sphere_model.pt")

    if do_models["two_body_sphere"]:
        # Sphere
        model = TwoBodySphere(4).to(device)

        filename = r+"sphere_2body.wt"

        # when last modified r+"prolate_2body.wt"?
        ts = os.path.getmtime(filename)
        # Convert to human-readable datetime
        last_modified = datetime.datetime.fromtimestamp(ts)

        print("Last modified:", last_modified)
        model.load_state_dict(torch.load(filename, weights_only=True))
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.save("data/models/two_body_sphere_model.pt")
        print("Saved TorchScript model to two_body_sphere_model.pt")

    if do_models["two_body_sphere_F1"]:
        model = TwoBodySphere(4).to(device)
        model.load_state_dict(torch.load(r+"sphere_2body_F1.wt", weights_only=True))
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.save("data/models/two_body_sphere_model_F1.pt")
        print("Saved TorchScript model to two_body_sphere_model_F1.pt")

    if do_models["two_body_combined"]:
        model = TwoBodyCombined(4).to(device)
        model.load_state_dict(torch.load(r+"combined_2body.wt", weights_only=True))
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.save("data/models/two_body_combined_model.pt")
        print("Saved TorchScript model to two_body_combined_model.pt")

    if do_models["two_body_combined_all"]:
        model = TwoBodyCombined(4).to(device)
        state = torch.load(r+"combined_all.wt", weights_only=True)
        for k in ["self_nn.0.weight", "self_nn.0.bias", "self_nn.2.weight", "self_nn.2.bias", "self_nn.4.weight", "self_nn.4.bias"]:
            del state[k]
        model.load_state_dict(state)
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.save("data/models/two_body_combined_all.pt")
        print("Saved TorchScript model to two_body_combined_all.pt")