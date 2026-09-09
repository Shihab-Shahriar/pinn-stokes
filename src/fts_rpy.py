"""FTS (force-torque-stresslet) far-field blocks of the RPY-with-Faxen grand mobility, and the
stresslet single-reflection velocity built from them.

Physics.  Beyond the force/torque (FT) level every sphere in a straining flow carries an induced
stresslet S; the FTS grand mobility of unit spheres reads

    [U; Omega] = A [F; T] + G S,        E = G^T [F; T] + Mm S,

with A the RPY tensor (``grpy_tensors.mu``), G the (6 x 5) velocity<-stresslet blocks, Mm the
(5 x 5) strain<-stresslet blocks and the self term Mm_kk = D I, D = 3 / (20 pi mu a^3).  Rigid
spheres have E = 0, so S = -Mm^{-1} G^T [F; T] and the many-body "screening" correction to the FT
mobility is -G Mm^{-1} G^T.  Its first term, the single reflection

    M_ref1 = -(1/D) sum_k G_tk G_sk^T          (all k; G_kk = 0),

is what this module evaluates as a global O(N^2) velocity term (``reflection_velocity``) and as
dense per-pair blocks for label building (``reflection_blocks``).  It is symmetric by construction
and is exactly the leading term of what Stokesian Dynamics' far field (``SD_Minf`` in
``src/sd_ops.py``) inverts.

Diagonal two-body paths.  For the pair block (t, s) every path t -> k -> s is a genuine three-body
term (G_tt = 0), but the diagonal path t -> k -> t is a TWO-body effect that the two-body model's
self correction K_s(t, k) already contains exactly for d_tk <= pair_cutoff.  ``diag_exclude_within``
therefore removes -G_tk G_tk^T / D from the diagonal for those near pairs (the operator and the
label builder both pass the pair cutoff); beyond it the base is RPY and the term is genuinely missing.

Conventions.  Everything is transcribed from Townsend's Stokesian Dynamics
(``stokesian_dynamics/functions/generate_Minfinity.py``, thesis table 2.1 / section A.2.3):
``r = pos[velocity particle] - pos[stresslet particle]``; stresslets and strain rates live in the
orthonormal condensed 5-vector basis ``cond_E`` applied to the (xx, xy, yy, xz, yz) components; unit
radii; ``c = 1 / (8 pi mu)``.  ``assemble_minfinity`` rebuilds SD's 11N x 11N matrix from these
blocks and is checked against ``generate_Minfinity`` in ``tests/test_fts_rpy.py``.

Layouts.  Positions (N, 3); wrenches (N, 6) = [F, T]; velocities (N, 6) = [U, Omega]; a 6 x 6 block
``B[t, s]`` maps the wrench of s to the velocity of t.  torch, float64, any device.
"""
from __future__ import annotations

import math

import numpy as np
import torch

_S2 = math.sqrt(2.0)
_HP = 0.5 * (math.sqrt(3.0) + 1.0)
_HM = 0.5 * (math.sqrt(3.0) - 1.0)
COND_IDX = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2))
COND_E = np.zeros((5, 5))
COND_E[0, 0] = _HP; COND_E[0, 2] = _HM; COND_E[1, 1] = _S2
COND_E[2, 0] = _HM; COND_E[2, 2] = _HP; COND_E[3, 3] = _S2; COND_E[4, 4] = _S2
# contraction matrix E_ajk: symmetric traceless 3x3 -> condensed 5-vector (shared.py::contraction)
CONTRACTION = np.zeros((5, 3, 3))
CONTRACTION[0, 0, 0] = _HP; CONTRACTION[0, 1, 1] = _HM
CONTRACTION[1, 0, 1] = _S2
CONTRACTION[2, 0, 0] = _HM; CONTRACTION[2, 1, 1] = _HP
CONTRACTION[3, 0, 2] = _S2
CONTRACTION[4, 1, 2] = _S2
DEFAULT_PAIR_CHUNK = 100_000


def d_self(mu: float = 1.0) -> float:
    """E <- S self block of a unit sphere: 1 / ((20/3) pi mu)."""
    return 3.0 / (20.0 * math.pi * mu)


def _consts(device, dtype):
    I = torch.eye(3, device=device, dtype=dtype)
    eps = torch.zeros(3, 3, 3, device=device, dtype=dtype)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
    con = torch.as_tensor(CONTRACTION, device=device, dtype=dtype)
    condE = torch.as_tensor(COND_E, device=device, dtype=dtype)
    return I, eps, con, condE


def fts_blocks(r: torch.Tensor, mu: float = 1.0, want_es: bool = False):
    """G (n, 6, 5) = [U; Omega] of the first particle per condensed stresslet of the second, for
    r = pos[first] - pos[second] (n, 3); zero for |r| = 0.  With ``want_es`` also Mm (n, 5, 5), the
    condensed strain<-stresslet block (off-diagonal only; the self block is ``d_self`` I)."""
    r = r.to(torch.float64)
    dev, dt = r.device, r.dtype
    I, eps, con, condE = _consts(dev, dt)
    c = 1.0 / (8.0 * math.pi * mu)
    s = torch.linalg.norm(r, dim=1)
    zero = s < 1e-10
    s = torch.where(zero, torch.ones_like(s), s)
    s3, s5, s7 = s ** 3, s ** 5, s ** 7
    v = lambda x: x.view(-1, 1, 1, 1)
    # D_l J_ij  -> [n, l, i, j]
    DJ = ((-torch.einsum("ij,nl->nlij", I, r) + torch.einsum("il,nj->nlij", I, r) + torch.einsum("jl,ni->nlij", I, r))
          / v(s3) - 3.0 * torch.einsum("ni,nj,nl->nlij", r, r, r) / v(s5))
    # K_ijk = 1/2 (D_k J_ij + D_j J_ik) -> [n, i, j, k]
    K = 0.5 * (DJ.permute(0, 2, 3, 1) + DJ.permute(0, 2, 1, 3))
    # D_k Lap J_ij (fully symmetric) = Lap K_ijk -> [n, i, j, k]
    LapK = (-6.0 / v(s5)) * (torch.einsum("jk,ni->nijk", I, r) + torch.einsum("ij,nk->nijk", I, r)
                             + torch.einsum("ik,nj->nijk", I, r)) + 30.0 * torch.einsum("ni,nj,nk->nijk", r, r, r) / v(s7)
    g = -c * (K + (1.0 / 6.0 + 1.0 / 10.0) * LapK)                                   # M13, g tilde
    # D_m D_l J_ij -> [n, m, l, i, j]
    v4 = lambda x: x.view(-1, 1, 1, 1, 1)
    T1 = (-torch.einsum("ij,lm->mlij", I, I) + torch.einsum("il,jm->mlij", I, I) + torch.einsum("jl,im->mlij", I, I))
    rr = torch.einsum("ni,nj->nij", r, r)
    T2 = (-torch.einsum("ij,nlm->nmlij", I, rr) + torch.einsum("il,njm->nmlij", I, rr)
          + torch.einsum("jl,nim->nmlij", I, rr) + torch.einsum("im,njl->nmlij", I, rr)
          + torch.einsum("jm,nil->nmlij", I, rr) + torch.einsum("lm,nij->nmlij", I, rr))
    T3 = torch.einsum("ni,nj,nl,nm->nmlij", r, r, r, r)
    DDJ = T1[None] / v4(s3) - 3.0 * T2 / v4(s5) + 15.0 * T3 / v4(s7)
    # D_l K_ijk = 1/2 (D_l D_k J_ij + D_l D_j J_ik) -> [n, l, i, j, k]
    DK = 0.5 * (DDJ.permute(0, 1, 3, 4, 2) + DDJ.permute(0, 1, 3, 2, 4))
    h = -0.5 * c * torch.einsum("ilm,nlmjk->nijk", eps, DK)                         # M23, h tilde
    gc = torch.einsum("ajk,nijk->nia", con, g)                                        # con_M13_row
    hc = torch.einsum("ajk,nijk->nia", con, h)                                        # con_M23_row
    G = torch.cat([gc, hc], dim=1)
    G = torch.where(zero.view(-1, 1, 1), torch.zeros_like(G), G)
    if not want_es:
        return G
    s9 = s ** 9
    # D_l Lap K_ijk (fully symmetric) -> [n, l, i, j, k]
    T4 = (torch.einsum("ij,kl->lijk", I, I) + torch.einsum("ik,jl->lijk", I, I) + torch.einsum("jk,il->lijk", I, I))
    T5 = (torch.einsum("ij,nkl->nlijk", I, rr) + torch.einsum("ik,njl->nlijk", I, rr) + torch.einsum("jk,nil->nlijk", I, rr)
          + torch.einsum("il,njk->nlijk", I, rr) + torch.einsum("jl,nik->nlijk", I, rr) + torch.einsum("kl,nij->nlijk", I, rr))
    T6 = torch.einsum("ni,nj,nk,nl->nlijk", r, r, r, r)
    DLapK = (-6.0 / v4(s5)) * T4[None] - 210.0 * T6 / v4(s9) + 30.0 * T5 / v4(s7)
    # m_ijkl = -c/2 [(D_j K_ikl + D_i K_jkl) + (2/10)(D_j Lap K_ikl + D_i Lap K_jkl)] -> [n, i, j, k, l]
    m = -0.5 * c * ((DK.permute(0, 2, 1, 3, 4) + DK) + 0.2 * (DLapK.permute(0, 2, 1, 3, 4) + DLapK))
    ci = torch.tensor([p[0] for p in COND_IDX], device=dev)
    cj = torch.tensor([p[1] for p in COND_IDX], device=dev)
    M33 = m[:, ci[:, None], cj[:, None], ci[None, :], cj[None, :]]                     # (n, 5, 5)
    Mm = condE @ M33 @ condE
    Mm = torch.where(zero.view(-1, 1, 1), torch.zeros_like(Mm), Mm)
    return G, Mm


def _pair_blocks(pos: torch.Tensor, mu: float, want_es: bool, chunk: int):
    """Dense (..., N, N, 6, 5) G blocks (and (..., N, N, 5, 5) Mm) for r = pos[t] - pos[s], pos (..., N, 3);
    small N only (the cache builder batches configurations of equal size along the leading axis)."""
    lead, N = pos.shape[:-2], pos.shape[-2]
    r = (pos[..., :, None, :] - pos[..., None, :, :]).reshape(-1, 3)
    Gs, Ms = [], []
    for i in range(0, r.shape[0], chunk):
        out = fts_blocks(r[i:i + chunk], mu, want_es)
        if want_es:
            Gs.append(out[0]); Ms.append(out[1])
        else:
            Gs.append(out)
    G = torch.cat(Gs, 0).view(*lead, N, N, 6, 5)
    if not want_es:
        return G
    return G, torch.cat(Ms, 0).view(*lead, N, N, 5, 5)


def _flat(B: torch.Tensor) -> torch.Tensor:
    """(..., N, N, a, b) blocks -> (..., N a, N b) matrix, particle-major."""
    *lead, N, _, a, b = B.shape
    return B.transpose(-3, -2).reshape(*lead, N * a, N * b)


def _blocks(M: torch.Tensor, N: int, a: int, b: int) -> torch.Tensor:
    lead = M.shape[:-2]
    return M.view(*lead, N, a, N, b).transpose(-3, -2)


def reflection_blocks(pos, mu: float = 1.0, order="1", chunk: int = DEFAULT_PAIR_CHUNK,
                      diag_exclude_within: float | None = None) -> torch.Tensor:
    """(P, P, 6, 6) blocks of the stresslet reflection correction, diagonal included (pos (P, 3), or
    (B, P, 3) for a batch of equal-size configurations -> (B, P, P, 6, 6)):
    order 1: -G D^{-1} G^T; order 2: + G D^{-1} Mm_off D^{-1} G^T; ``"full"``: -G Mm^{-1} G^T.
    ``diag_exclude_within``: drop the two-body diagonal path -G_tk G_tk^T / D for |x_k - x_t| <= it.
    Dense (P <= a few hundred): the cache builder's and the ladder's path."""
    pos = torch.as_tensor(pos, dtype=torch.float64)
    P = pos.shape[-2]
    order = str(order)
    D = d_self(mu)
    if order == "1":
        Gb = _pair_blocks(pos, mu, False, chunk)
        G = _flat(Gb)
        M = -(G @ G.transpose(-1, -2)) / D
    else:
        Gb, Mmb = _pair_blocks(pos, mu, True, chunk)
        G, Mm_off = _flat(Gb), _flat(Mmb)
        GT = G.transpose(-1, -2)
        if order == "2":
            M = -(G @ GT) / D + (G @ Mm_off @ GT) / D ** 2
        elif order == "full":
            Mm = Mm_off + D * torch.eye(5 * P, dtype=torch.float64, device=pos.device)
            M = -G @ torch.linalg.solve(Mm, GT)
        else:
            raise ValueError(order)
    B = _blocks(M, P, 6, 6).clone()
    if diag_exclude_within is not None:
        dist = torch.linalg.norm(pos[..., :, None, :] - pos[..., None, :, :], dim=-1)
        near = ((dist <= diag_exclude_within) & (dist > 0)).to(torch.float64)
        corr = torch.einsum("...tkab,...tk,...tkcb->...tac", Gb, near, Gb) / D      # = -sum_near (-G G^T / D)
        idx = torch.arange(P, device=pos.device)
        B[..., idx, idx, :, :] += corr
    return B


def reflection_velocity(pos, force, mu: float = 1.0, order="1", row_chunk: int | None = None,
                        pair_chunk: int = DEFAULT_PAIR_CHUNK, diag_exclude_within: float | None = None) -> torch.Tensor:
    """(N, 6) velocity of the stresslet reflection for wrenches ``force`` (N, 6): matrix-free O(N^2),
    target rows chunked so no (N, N) block tensor is ever materialised.
    order 1: S = -E/D with E_k = sum_s G_sk^T F_s;  order 2: one Jacobi sweep, S = -(E + Mm_off S1)/D.
    ``diag_exclude_within``: remove the two-body diagonal path -G_tk G_tk^T F_t / D for |x_k - x_t| <= it."""
    pos = torch.as_tensor(pos, dtype=torch.float64)
    F = torch.as_tensor(force, dtype=torch.float64, device=pos.device)
    N = pos.shape[0]
    assert F.shape == (N, 6), F.shape
    order = str(order)
    assert order in ("1", "2"), order
    D = d_self(mu)
    rows = row_chunk or max(1, min(N, pair_chunk // N))

    def strain(FT):  # E_k = sum_s G(pos[s] - pos[k])^T FT_s   (the E<-F block is the transpose of the U<-S block)
        E = torch.empty(N, 5, dtype=torch.float64, device=pos.device)
        for k0 in range(0, N, rows):
            k1 = min(N, k0 + rows)
            r = (pos[None, :, :] - pos[k0:k1, None, :]).reshape(-1, 3)     # r_sk = pos[s] - pos[k]
            G = fts_blocks(r, mu).view(k1 - k0, N, 6, 5)
            E[k0:k1] = torch.einsum("ksab,sa->kb", G, FT)
        return E

    S = -strain(F) / D
    if order == "2":
        E2 = torch.empty(N, 5, dtype=torch.float64, device=pos.device)
        for k0 in range(0, N, rows):
            k1 = min(N, k0 + rows)
            r = (pos[k0:k1, None, :] - pos[None, :, :]).reshape(-1, 3)     # r_kl = pos[k] - pos[l]
            _, Mm = fts_blocks(r, mu, want_es=True)
            E2[k0:k1] = torch.einsum("klab,lb->ka", Mm.view(k1 - k0, N, 5, 5), S)
        S = -(strain(F) + E2) / D
    V = torch.empty(N, 6, dtype=torch.float64, device=pos.device)
    for t0 in range(0, N, rows):
        t1 = min(N, t0 + rows)
        r = pos[t0:t1, None, :] - pos[None, :, :]                           # r_tk = pos[t] - pos[k]
        G = fts_blocks(r.reshape(-1, 3), mu).view(t1 - t0, N, 6, 5)
        V[t0:t1] = torch.einsum("tkab,kb->ta", G, S)
        if diag_exclude_within is not None:
            dist = torch.linalg.norm(r, dim=-1)
            near = ((dist <= diag_exclude_within) & (dist > 0)).to(torch.float64)
            V[t0:t1] += torch.einsum("tkab,tk,tkcb,tc->ta", G, near, G, F[t0:t1]) / D
    return V


def assemble_minfinity(pos, mu: float = 1.0) -> np.ndarray:
    """SD's 11N x 11N FTS grand mobility [U/F | Omega/T | E/S] from ``grpy_tensors.mu`` (A) and these
    blocks (test helper: compared with ``generate_Minfinity`` in tests/test_fts_rpy.py)."""
    from grpy_tensors import mu as grpy_mu
    pos = np.asarray(pos, dtype=np.float64)
    N = pos.shape[0]
    Mr = grpy_mu(pos, np.ones(N), blockmatrix=True) / mu     # (2, 2, N, N, 3, 3) at mu = 1
    A = np.zeros((6 * N, 6 * N))
    for bi in range(2):
        for bj in range(2):
            A[3 * N * bi:3 * N * (bi + 1), 3 * N * bj:3 * N * (bj + 1)] = Mr[bi, bj].transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    Gb, Mmb = _pair_blocks(torch.as_tensor(pos), mu, True, DEFAULT_PAIR_CHUNK)
    Gb, Mmb = Gb.numpy(), Mmb.numpy()
    G = np.zeros((6 * N, 5 * N))
    G[:3 * N] = Gb[:, :, :3, :].transpose(0, 2, 1, 3).reshape(3 * N, 5 * N)
    G[3 * N:] = Gb[:, :, 3:, :].transpose(0, 2, 1, 3).reshape(3 * N, 5 * N)
    Mm = Mmb.transpose(0, 2, 1, 3).reshape(5 * N, 5 * N) + d_self(mu) * np.eye(5 * N)
    M = np.zeros((11 * N, 11 * N))
    M[:6 * N, :6 * N] = A
    M[:6 * N, 6 * N:] = G
    M[6 * N:, :6 * N] = G.T
    M[6 * N:, 6 * N:] = Mm
    return M
