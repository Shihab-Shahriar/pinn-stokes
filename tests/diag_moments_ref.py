"""Numpy reference implementation of the self-block (per-particle diagonal) encoder --
an independent twin of ``src/nbody_moments.py``'s ``self_*`` functions.  Bands are the
remapped tents on [LO, HI] = [2, 8] (width 0.75, centres 2.375 .. 7.625) with saturating
end bands and NO zeroing beyond HI (the cutoff is enforced by neighbour selection).
Used only by the tests."""
import numpy as np
from numpy import einsum, eye

I3, NB, LO, HI = eye(3), 8, 2.0, 8.0
H = (HI - LO) / NB                     # 0.75


def E(u):                              # E(u)_{ab} = eps_{abc} u_c
    ux, uy, uz = u
    return np.array([[0, uz, -uy], [-uz, 0, ux], [uy, -ux, 0]])


def S(X):
    return X + X.T


def self_band_weights(r):              # (K,) -> (K,NB), tent partition of unity on [0, inf)
    centres = LO + (np.arange(NB) + 0.5) * H               # 2.375 ... 7.625
    w = np.clip(1.0 - np.abs(r[:, None] - centres[None, :]) / H, 0.0, None)
    w[r <= centres[0], 0] = 1.0
    w[r >= centres[-1], -1] = 1.0
    return w


def self_moments(xk):                  # xk: (K,3) neighbours relative to the particle
    rn = np.linalg.norm(xk, axis=1)
    rh = xk / rn[:, None]
    W = self_band_weights(rn)                                   # (K,NB)
    s = W.sum(0)                                                # (NB,)      l=0
    v = einsum('ka,ki->ai', W, rh)                              # (NB,3)     l=1
    Q = einsum('ka,ki,kj->aij', W, rh, rh) - s[:, None, None] * I3[None] / 3   # (NB,3,3) l=2
    return s, v, Q


def self_invariants(s, v, Q):          # 6 per band, all true rotation scalars (reflection-invariant)
    Qv = einsum('aij,aj->ai', Q, v)
    return np.concatenate([
        s,
        (v * v).sum(1),
        einsum('aij,aji->a', Q, Q),
        einsum('aij,ajk,aki->a', Q, Q, Q),
        (v * Qv).sum(1),
        (Qv * Qv).sum(1),
    ])                                                          # (48,)


def self_bases(v, Q):                  # TT/RR bases (33,3,3) and TR bases (24,3,3)
    tt = [I3]
    tr = []
    for a in range(NB):
        vv = np.outer(v[a], v[a])
        tt += [Q[a], vv, Q[a] @ Q[a], S(Q[a] @ vv)]
        Ev = E(v[a])
        tr += [Ev, E(Q[a] @ v[a]), Q[a] @ Ev - Ev @ Q[a]]       # commutator: symmetric pseudotensor
    return np.array(tt), np.array(tr)


def self_block(c, v, Q):               # c: (90,) coefficients -> symmetric 6x6
    tt, tr = self_bases(v, Q)
    TT = einsum('b,bij->ij', c[:33], tt)
    RR = einsum('b,bij->ij', c[33:66], tt)
    TR = einsum('b,bij->ij', c[66:90], tr)
    M = np.zeros((6, 6))
    M[:3, :3] = TT
    M[:3, 3:] = TR
    M[3:, :3] = TR.T
    M[3:, 3:] = RR
    return M
