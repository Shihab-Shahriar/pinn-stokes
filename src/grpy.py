import torch
import math

@torch.compile
def mu(centers, radii, blockmatrix=False):
    """
    Vectorized, pytorch implementation of rpy tensor, to be run on GPU. Does not add 
    laplace correction. Outputs full 6x6 matrix for each pair of particles.

    Keeping in line with our two-body neural net, this rpy only handles the
    case when particles don't overlap: `dist_ij > a_i + a_j`. Also no 
    self-interaction.

    This code is an adaptation of:
    https://github.com/RadostW/PyGRPY/blob/main/pygrpy/grpy_tensors.py

    Parameters
    ----------
    centers : (N,3) torch.Tensor
        Positions of the N beads (float).
    radii   : (N,)  torch.Tensor
        Radii of the N beads (float).
    blockmatrix : bool
        If True, returns shape (2,2,N,N,3,3).
        Else, returns shape (6N, 6N).

    Returns
    -------
    If blockmatrix=True:
      out => (2,2,N,N,3,3)  (the block form)
    else:
      out => (6N, 6N)       (flattened)
    """
    device = centers.device
    dtype  = centers.dtype
    n = centers.shape[0]

    # --------------------------------------------------------------------
    # 1) Compute displacements / distances
    # --------------------------------------------------------------------
    disp = centers.unsqueeze(1) - centers.unsqueeze(0)  # shape (n,n,3)
    dist = disp.norm(dim=-1)                            # shape (n,n)

    # --------------------------------------------------------------------
    # 2) Build rHat[i,j] = disp[i,j]/|disp[i,j]| for i != j, else 0
    # --------------------------------------------------------------------
    rHatMatrix = torch.zeros_like(disp)  # (n,n,3)
    eye_mask   = torch.eye(n, dtype=torch.bool, device=device)
    offdiag    = ~eye_mask

    rHatMatrix[offdiag] = disp[offdiag] / dist[offdiag].unsqueeze(-1)

    # --------------------------------------------------------------------
    # 3) Expand radii to shape (n,n)
    #    We assume always dist_ij > a_i + a_j. So we skip checks & errors.
    #    We'll just apply the far-field formula to all i != j.
    # --------------------------------------------------------------------
    a_i = radii.view(n,1).expand(n,n)  # (n,n)
    a_j = radii.view(1,n).expand(n,n)  # (n,n)

    # --------------------------------------------------------------------
    # 4) For i != j, compute the far separation RPY scalars
    #    For i == j, set everything to zero (no self terms)
    # --------------------------------------------------------------------
    # shape (n,n)
    dist_mask = offdiag  # only fill these for i != j

    # Distances for i != j
    dist_off = dist[dist_mask]
    a_i_off  = a_i[dist_mask]
    a_j_off  = a_j[dist_mask]

    # TT identity, TT rhat
    TTidentityScale = torch.zeros((n,n), dtype=dtype, device=device)
    TTrHatScale     = torch.zeros((n,n), dtype=dtype, device=device)
    TTidentityScale[dist_mask] = (1.0/(8.0*math.pi*dist_off)) * \
        (1.0 + (a_i_off**2 + a_j_off**2)/(3.0*dist_off**2))
    TTrHatScale[dist_mask] = (1.0/(8.0*math.pi*dist_off)) * \
        (1.0 - (a_i_off**2 + a_j_off**2)/(dist_off**2))

    # RR identity, RR rhat
    RRidentityScale = torch.zeros_like(TTidentityScale)
    RRrHatScale     = torch.zeros_like(TTidentityScale)
    RRidentityScale[dist_mask] = -1.0/(16.0*math.pi*(dist_off**3))
    RRrHatScale[dist_mask]     =  3.0/(16.0*math.pi*(dist_off**3))

    # RT scale
    RTScale = torch.zeros_like(TTidentityScale)
    RTScale[dist_mask] = 1.0/(8.0*math.pi*(dist_off**2))

    # --------------------------------------------------------------------
    # 5) Build (n,n,3,3) blocks:
    #    muTT[i,j] = TTid[i,j]*I + TTrr[i,j]* (rHat outer rHat)
    #    muRR[i,j] = RRid[i,j]*I + RRrr[i,j]* (rHat outer rHat)
    #    muRT[i,j] = RTScale[i,j]* cross(rHat)
    #
    # i == j => 0 block
    # --------------------------------------------------------------------
    I_3 = torch.eye(3, dtype=dtype, device=device)
    r_outer = rHatMatrix.unsqueeze(-1)*rHatMatrix.unsqueeze(-2)  # (n,n,3,3)

    muTT = (TTidentityScale.unsqueeze(-1).unsqueeze(-1)*I_3
            + TTrHatScale.unsqueeze(-1).unsqueeze(-1)*r_outer)
    muRR = (RRidentityScale.unsqueeze(-1).unsqueeze(-1)*I_3
            + RRrHatScale.unsqueeze(-1).unsqueeze(-1)*r_outer)

    muRT = RTScale.unsqueeze(-1).unsqueeze(-1)*_build_cross(rHatMatrix)

    # --------------------------------------------------------------------
    # 6) Return either block form or flattened (6N,6N)
    # --------------------------------------------------------------------
    if blockmatrix:
        top = torch.stack([muTT, muRT], dim=0)         # shape (2,n,n,3,3)
        bot = torch.stack([_transTranspose(muRT), muRR], dim=0)
        return torch.stack([top, bot], dim=0)          # (2,2,n,n,3,3)
    else:
        # Assemble the full matrix
        # Optimization suggestion from Deepseek R1

        # Reshape blocks into 3n x3n matrices
        muTT_flat = muTT.permute(0, 2, 1, 3).reshape(3*n, 3*n)
        muRT_flat = muRT.permute(0, 2, 1, 3).reshape(3*n, 3*n)
        muRR_flat = muRR.permute(0, 2, 1, 3).reshape(3*n, 3*n)
        tr_block = -muRT_flat.T

        # full matrix
        M_full = torch.zeros((6*n, 6*n), dtype=dtype, device=device)
        M_full[:3*n, :3*n] = muTT_flat
        M_full[:3*n, 3*n:] = muRT_flat
        M_full[3*n:, :3*n] = tr_block
        M_full[3*n:, 3*n:] = muRR_flat

        return M_full


def _build_cross(rVec):
    """
    rVec: (n,n,3)
    Returns cross-product matrix for each [i,j],
    shape => (n,n,3,3), representing cross(rVec[i,j], .).
    """
    x = rVec[...,0]
    y = rVec[...,1]
    z = rVec[...,2]

    row0 = torch.stack([torch.zeros_like(x), -z,               y], dim=-1)
    row1 = torch.stack([              z, torch.zeros_like(x), -x], dim=-1)
    row2 = torch.stack([             -y,               x, torch.zeros_like(x)], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def _transTranspose(muRT):
    """Return - (muRT^T) in the last two dims."""
    return -muRT.transpose(-1, -2)



