"""Band-moment neighbourhood encoding for the n-body correction.

Implements moments_for_nbody.md sections 3-5 in batched, TorchScript-scriptable
torch: radial band weights, the (s_a, v_a, Q_a) moments of a pair's
neighbourhood, the 72 pair-symmetric rotational invariants that feed the MLP,
and the 34 + 25 tensor bases the predicted coefficients multiply onto.

Conventions (shared by training and inference; see plan / report):
  * ``s_vec = x_source - x_target`` (the data/operator convention); the pair
    axis is ``z = -s_vec / |s_vec|`` = the doc's ``\\hat z`` = the baseline's
    ``d_vec`` after negation.
  * Neighbour vectors ``nbr`` are relative to the target, so the pair midpoint
    is ``s_vec / 2`` and ``r_k = nbr_k - s_vec / 2``.
  * ``skew(u)_ab = eps_abc u_c`` = ``model_archs.L3``; ``skew(u) @ w = w x u``.
  * Band weights are a tent partition of unity on [0, inf): band 1 saturates
    for r <= 0.5, band 8 for r >= 7.5.  Deliberate deviation from the doc:
    NO zeroing beyond r_c = 8.  The neighbourhood cutoff is enforced by
    neighbour *selection* (operator side), not by the weight function, because
    the existing training rows contain neighbours farther than 8 from the
    midpoint that do influence the label.
  * Padded neighbour slots must be masked: a zero neighbour coincides with the
    target, which is |s_vec|/2 from the midpoint and would otherwise be counted.

Model input row (``X_DIM`` = 111 columns):
  [0:3]   s_vec
  [3:7]   pair scalars: d - mean, d - 2, (d - mean)^2, (d - mean)^4
  [7:15]  s_a         (8)
  [15:39] v_a         (8 x 3, row-major)
  [39:111] Q_a        (8 x 3 x 3, row-major, full symmetric traceless matrix)
"""

from typing import List, Tuple

import torch
from torch import Tensor

NB: int = 8                       # radial bands, centres 0.5 .. 7.5
EPS: float = 1e-6
N_PAIR: int = 4                   # pair scalars
N_INV: int = 9 * NB               # 72 neighbourhood invariants
N_IN: int = N_PAIR + N_INV        # 76 MLP inputs
N_TT: int = 2 + 4 * NB            # 34 TT / RR bases
N_TR: int = 1 + 3 * NB            # 25 TR = RT bases
N_COEF: int = 2 * N_TT + N_TR     # 93 coefficients
OFF_PAIR: int = 3
OFF_S: int = OFF_PAIR + N_PAIR    # 7
OFF_V: int = OFF_S + NB           # 15
OFF_Q: int = OFF_V + 3 * NB       # 39
X_DIM: int = OFF_Q + 9 * NB       # 111


# TorchScript cannot read closed-over module globals, so scripted code reads the
# constants through these trivial functions.
def _nb() -> int:
    return 8


def _eps() -> float:
    return 1e-6


def _n_tt() -> int:
    return 34


def skew(u: Tensor) -> Tensor:
    """E(u)_ab = eps_abc u_c for u[..., 3] -> [..., 3, 3] (E(u) @ w = w x u)."""
    ux, uy, uz = u.unbind(-1)
    zero = torch.zeros_like(ux)
    r0 = torch.stack([zero, uz, -uy], -1)
    r1 = torch.stack([-uz, zero, ux], -1)
    r2 = torch.stack([uy, -ux, zero], -1)
    return torch.stack([r0, r1, r2], -2)


def sym(X: Tensor) -> Tensor:
    return X + X.transpose(-1, -2)


def alt(X: Tensor) -> Tensor:
    return X - X.transpose(-1, -2)


def pair_axis(s_vec: Tensor) -> Tensor:
    """z = -s_vec / |s_vec|  (target - source, unit)."""
    n = torch.sqrt((s_vec * s_vec).sum(-1, keepdim=True)).clamp_min(_eps())
    return -s_vec / n


def pair_scalars(dist: Tensor, mean_dist_s: float) -> Tensor:
    """[d - mean, d - 2, (d - mean)^2, (d - mean)^4] -> [..., 4] (baseline cols 33..36)."""
    dc = dist - mean_dist_s
    dc2 = dc * dc
    return torch.stack([dc, dist - 2.0, dc2, dc2 * dc2], -1)


def band_weights(r: Tensor) -> Tensor:
    """Tent partition of unity over NB unit-width bands; r[...] -> [..., NB]."""
    cols: List[Tensor] = []
    for a in range(_nb()):
        c = float(a) + 0.5
        w = torch.clamp(1.0 - torch.abs(r - c), min=0.0)
        if a == 0:
            w = torch.where(r <= c, torch.ones_like(w), w)
        if a == _nb() - 1:
            w = torch.where(r >= c, torch.ones_like(w), w)
        cols.append(w)
    return torch.stack(cols, -1)


def band_moments(s_vec: Tensor, nbr: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Band moments of a pair's neighbourhood.

    s_vec [P, 3]  source - target
    nbr   [P, K, 3]  neighbour positions relative to the target (padded)
    mask  [P, K]  1.0 for real neighbours, 0.0 for padding
    returns s [P, NB], v [P, NB, 3], Q [P, NB, 3, 3] (symmetric, traceless)
    """
    r = nbr - 0.5 * s_vec.unsqueeze(1)                       # [P,K,3] from midpoint
    rn = torch.sqrt((r * r).sum(-1))                          # [P,K]
    rh = r / rn.clamp_min(_eps()).unsqueeze(-1)                  # [P,K,3]
    W = band_weights(rn) * mask.unsqueeze(-1)                 # [P,K,NB]
    s = W.sum(1)                                              # [P,NB]
    v = torch.einsum('pka,pki->pai', [W, rh])                 # [P,NB,3]
    Q = torch.einsum('pka,pki,pkj->paij', [W, rh, rh])        # [P,NB,3,3]
    I3 = torch.eye(3, device=nbr.device, dtype=nbr.dtype)
    Q = Q - (s / 3.0).unsqueeze(-1).unsqueeze(-1) * I3
    return s, v, Q


def invariants(z: Tensor, s: Tensor, v: Tensor, Q: Tensor) -> Tensor:
    """72 rotation-invariant, swap-even scalars, doc section 4 order, block-major
    (index = block * NB + band): s | |v|^2 | (z.v)^2 | z'Qz | |Qz|^2 | trQ^2 | trQ^3 | v'Qv | (z.v)(z'Qv)."""
    P = z.shape[0]
    z1 = z.unsqueeze(1)                                       # [P,1,3]
    zv = (v * z1).sum(-1)                                     # [P,NB]
    Qz = torch.einsum('paij,pj->pai', [Q, z])                 # [P,NB,3]
    zQz = (Qz * z1).sum(-1)
    Qz2 = (Qz * Qz).sum(-1)
    trQ2 = (Q * Q).sum(-1).sum(-1)                            # Q symmetric
    trQ3 = torch.einsum('paij,pajk,paki->pa', [Q, Q, Q])
    Qv = torch.einsum('paij,paj->pai', [Q, v])
    vQv = (v * Qv).sum(-1)
    zQv = (Qv * z1).sum(-1)
    inv = torch.stack([s, (v * v).sum(-1), zv * zv, zQz, Qz2, trQ2, trQ3, vQv, zv * zQv], 1)
    return inv.reshape(P, 9 * _nb())


def bases(z: Tensor, v: Tensor, Q: Tensor) -> Tuple[Tensor, Tensor]:
    """Tensor bases, doc section 5.2, band-major.

    tt [P, 34, 3, 3]: I, zz', then per band {Q_a, S(zz'Q_a), v_a v_a', Alt(z v_a')}
    tr [P, 25, 3, 3]: E(z), then per band {E(Q_a z), S(z (z x v_a)'), (z.v_a) E(v_a)}
    """
    P = z.shape[0]
    I3 = torch.eye(3, device=z.device, dtype=z.dtype).unsqueeze(0).expand(P, 3, 3)
    zz = torch.einsum('pi,pj->pij', [z, z])
    z1 = z.unsqueeze(1)
    zv = (v * z1).sum(-1)                                     # [P,NB]
    zzQ = torch.einsum('pij,pajk->paik', [zz, Q])
    vv = torch.einsum('pai,paj->paij', [v, v])
    zvT = torch.einsum('pi,paj->paij', [z, v])
    Qz = torch.einsum('paij,pj->pai', [Q, z])
    zxv = torch.cross(z1.expand_as(v), v, dim=-1)
    z_zxv = torch.einsum('pi,paj->paij', [z, zxv])
    per_tt = torch.stack([Q, sym(zzQ), vv, alt(zvT)], 2)     # [P,NB,4,3,3]
    tt = torch.cat([I3.unsqueeze(1), zz.unsqueeze(1), per_tt.reshape(P, 4 * _nb(), 3, 3)], 1)
    per_tr = torch.stack([skew(Qz), sym(z_zxv), zv.unsqueeze(-1).unsqueeze(-1) * skew(v)], 2)
    tr = torch.cat([skew(z).unsqueeze(1), per_tr.reshape(P, 3 * _nb(), 3, 3)], 1)
    return tt, tr


def assemble_block(c: Tensor, tt: Tensor, tr: Tensor) -> Tensor:
    """c [P, 93] -> 6x6 block: TT = c[:34].tt, RR = c[34:68].tt, TR = RT = c[68:].tr."""
    ntt = _n_tt()
    TT = torch.einsum('pb,pbij->pij', [c[:, :ntt], tt])
    RR = torch.einsum('pb,pbij->pij', [c[:, ntt:2 * ntt], tt])
    TR = torch.einsum('pb,pbij->pij', [c[:, 2 * ntt:], tr])
    top = torch.cat([TT, TR], 2)
    bot = torch.cat([TR, RR], 2)
    return torch.cat([top, bot], 1)


def pack_features(s_vec: Tensor, pair: Tensor, s: Tensor, v: Tensor, Q: Tensor) -> Tensor:
    P = s_vec.shape[0]
    nb = _nb()
    return torch.cat([s_vec, pair, s, v.reshape(P, 3 * nb), Q.reshape(P, 9 * nb)], 1)


def unpack_features(X: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    P = X.shape[0]
    nb = _nb()
    off_s = 3 + 4
    off_v = off_s + nb
    off_q = off_v + 3 * nb
    s_vec = X[:, 0:3]
    pair = X[:, 3:off_s]
    s = X[:, off_s:off_v]
    v = X[:, off_v:off_q].reshape(P, nb, 3)
    Q = X[:, off_q:off_q + 9 * nb].reshape(P, nb, 3, 3)
    return s_vec, pair, s, v, Q


def moment_features(s_vec: Tensor, nbr: Tensor, mask: Tensor, mean_dist_s: float) -> Tensor:
    """Full model input row [P, X_DIM] from raw geometry (any float dtype)."""
    dist = torch.sqrt((s_vec * s_vec).sum(-1))
    pair = pair_scalars(dist, mean_dist_s)
    s, v, Q = band_moments(s_vec, nbr, mask)
    return pack_features(s_vec, pair, s, v, Q)
