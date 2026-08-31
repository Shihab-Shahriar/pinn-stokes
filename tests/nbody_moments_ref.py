"""Numpy reference implementation from moments_for_nbody.md section 7 (verbatim, doc conventions),
plus a target-frame wrapper matching the repo's data layout.  Used only by the tests."""
import numpy as np
from numpy import einsum, eye

I3, NB, RC = eye(3), 8, 8.0


def E(u):                         # E(u)_{ab} = eps_{abc} u_c
    ux, uy, uz = u
    return np.array([[0, uz, -uy], [-uz, 0, ux], [uy, -ux, 0]])


def S(X):
    return X + X.T


def Alt(X):
    return X - X.T


def band_weights(r):              # (K,) -> (K,NB), tent partition of unity on [0, RC]
    centres = np.arange(1, NB + 1) - 0.5                    # 0.5 ... 7.5
    w = np.clip(1.0 - np.abs(r[:, None] - centres[None, :]), 0.0, None)
    w[r <= centres[0], 0] = 1.0
    w[r >= centres[-1], -1] = 1.0
    w[r > RC, :] = 0.0
    return w


def moments(xi, xj, xk):          # xk: (K,3) neighbours within RC of the midpoint
    m = 0.5 * (xi + xj)
    rij = xi - xj; ell = np.linalg.norm(rij); z = rij / ell
    rk = xk - m; rn = np.linalg.norm(rk, axis=1); rh = rk / rn[:, None]
    W = band_weights(rn)                                        # (K,NB)
    s = W.sum(0)                                                # (NB,)      l=0
    v = einsum('ka,ki->ai', W, rh)                              # (NB,3)     l=1
    Q = einsum('ka,ki,kj->aij', W, rh, rh) - s[:, None, None] * I3[None] / 3   # (NB,3,3) l=2
    return z, s, v, Q


def invariants(z, s, v, Q):       # 9 per band, all rotation-invariant and swap-even
    zv  = v @ z
    Qz  = einsum('aij,j->ai', Q, z)
    return np.concatenate([
        s,
        (v * v).sum(1),
        zv ** 2,
        einsum('i,aij,j->a', z, Q, z),
        (Qz * Qz).sum(1),
        einsum('aij,aji->a', Q, Q),
        einsum('aij,ajk,aki->a', Q, Q, Q),
        einsum('ai,aij,aj->a', v, Q, v),
        zv * einsum('i,aij,aj->a', z, Q, v),
    ])                                                          # (72,)


def bases(z, v, Q):               # TT/RR bases (34,3,3) and TR/RT bases (25,3,3)
    zz = np.outer(z, z); zv = v @ z
    tt = [I3, zz]
    tr = [E(z)]
    for a in range(NB):
        tt += [Q[a], S(zz @ Q[a]), np.outer(v[a], v[a]), Alt(np.outer(z, v[a]))]
        tr += [E(Q[a] @ z), S(np.outer(z, np.cross(z, v[a]))), zv[a] * E(v[a])]
    return np.array(tt), np.array(tr)


def nbody_block(c, z, v, Q):      # c: (93,) coefficients
    tt, tr = bases(z, v, Q)
    TT = einsum('b,bij->ij', c[:34],   tt)
    RR = einsum('b,bij->ij', c[34:68], tt)
    TR = einsum('b,bij->ij', c[68:93], tr)
    M = np.zeros((6, 6)); M[:3, :3] = TT; M[:3, 3:] = TR; M[3:, :3] = TR; M[3:, 3:] = RR
    return M


def moments_target_frame(s_vec, nbr):
    """Repo layout: target at the origin, source at s_vec, neighbours nbr (K,3) relative to the target."""
    return moments(np.zeros(3), np.asarray(s_vec, dtype=np.float64), np.asarray(nbr, dtype=np.float64))
