"""Validate the RPY term added to CartesianStokes::m2p, in isolation.

Mirrors the CUDA contraction line for line (node-scaled moments, the 1/r
derivative recurrence evaluated at R/scale, the (-1)^n/(a!b!c!) slot
coefficients) and compares one node's M2P against the exact RPY sum over that
node's sources.

The point of the exercise: two_ball only supports coherent gravity loading,
where the RPY correction is 48x weaker than under random loading (1.4e-4 vs
6.8e-3 at phi=0.1), so it cannot tell a correct RPY term from a missing one.
This can: with the term ON the error must fall with expansion order; with it
OFF it must floor at the size of the correction itself.
"""
import itertools
import math

import numpy as np

RPY_A = 1.0
RPY_C = 2.0 * RPY_A * RPY_A / 3.0


def laplace_sym(R, deriv_order):
    """Derivatives of 1/|R| up to `deriv_order`, keyed by multi-index (a,b,c).

    Same recurrence as CartesianStokes::laplaceSym.
    """
    Rx, Ry, Rz = R
    r2 = float(R @ R)
    ir = 1.0 / math.sqrt(r2)
    ir3 = ir / r2
    T = {(0, 0, 0): ir, (1, 0, 0): -Rx * ir3,
         (0, 1, 0): -Ry * ir3, (0, 0, 1): -Rz * ir3}
    Rc = (Rx, Ry, Rz)
    for n in range(2, deriv_order + 1):
        for a in range(n, -1, -1):
            for b in range(n - a, -1, -1):
                c = n - a - b
                alpha = (a, b, c)
                t = 0.0
                for d in range(3):
                    ad = alpha[d]
                    if ad:
                        prev = list(alpha)
                        prev[d] -= 1
                        t += (2 * n - 1) * ad * Rc[d] * T[tuple(prev)]
                    if ad > 1:
                        prev = list(alpha)
                        prev[d] -= 2
                        t += (n - 1) * ad * (ad - 1) * T[tuple(prev)]
                T[alpha] = t * (-1.0 / (r2 * n))
    return T


def m2p(order, center, scale, moments, target, rpy_on):
    """One node's far-field contribution. Mirrors CartesianStokes::m2p."""
    invS = 1.0 / scale
    rpy_c_s = RPY_C * invS * invS
    R = (target - center) * invS
    L = laplace_sym(R, order + (2 if rpy_on else 1))

    uf = np.zeros(3)
    for n in range(order + 1):
        for a in range(n, -1, -1):
            for b in range(n - a, -1, -1):
                c = n - a - b
                alpha = (a, b, c)
                coeff = ((-1.0) ** n) / (
                    math.factorial(a) * math.factorial(b) * math.factorial(c))
                Mc = moments[alpha]
                Talpha = L[alpha]
                for i in range(3):
                    beta = list(alpha)
                    beta[i] += 1
                    Ti = L[tuple(beta)]
                    acc = 0.0
                    for j in range(3):
                        g = (Talpha if i == j else 0.0) - R[j] * Ti
                        if alpha[j]:
                            d = list(alpha)
                            d[j] -= 1
                            d[i] += 1
                            g -= alpha[j] * L[tuple(d)]
                        if rpy_on:
                            d = list(beta)
                            d[j] += 1
                            g -= rpy_c_s * L[tuple(d)]
                        acc += g * Mc[j]
                    uf[i] += coeff * acc
    return uf * invS


def build_moments(order, center, scale, pos, force):
    """P2M: M[alpha] = sum_a f_a * ((y_a - c)/scale)^alpha."""
    a_scaled = (pos - center) / scale
    out = {}
    for n in range(order + 1):
        for a in range(n, -1, -1):
            for b in range(n - a, -1, -1):
                c = n - a - b
                w = (a_scaled[:, 0] ** a) * (a_scaled[:, 1] ** b) \
                    * (a_scaled[:, 2] ** c)
                out[(a, b, c)] = w @ force
    return out


def exact(kind, pos, force, target):
    """Dense sum over the node's sources: Oseen, or RPY with a = 1."""
    R = target[None, :] - pos
    r2 = (R * R).sum(1)
    r = np.sqrt(r2)
    ir, ir3, ir5 = 1.0 / r, 1.0 / r ** 3, 1.0 / r ** 5
    rdf = (R * force).sum(1)
    if kind == "stokeslet":
        cI, cR = ir, ir3 * rdf
    else:
        cI = ir + RPY_C * ir3
        cR = (ir3 - 2.0 * RPY_A ** 2 * ir5) * rdf
    return (cI[:, None] * force).sum(0) + (cR[:, None] * R).sum(0)


def main():
    rng = np.random.default_rng(11)
    n_src = 64
    center = np.zeros(3)
    half_diag = 1.0
    scale = half_diag                       # NodeM2P::scale is the half-diagonal
    pos = rng.uniform(-1.0, 1.0, (n_src, 3))
    pos /= np.maximum(1.0, np.linalg.norm(pos, axis=1, keepdims=True))
    force = rng.standard_normal((n_src, 3))

    # mac = half_diag / distance, so distance = half_diag / mac.
    for mac in (0.5, 0.3):
        target = np.array([1.0, 0.6, -0.3])
        target *= (half_diag / mac) / np.linalg.norm(target)
        ref_rpy = exact("rpy", pos, force, target)
        ref_stk = exact("stokeslet", pos, force, target)
        gap = np.linalg.norm(ref_rpy - ref_stk) / np.linalg.norm(ref_rpy)
        print(f"\nmac={mac}  |target|={np.linalg.norm(target):.3f}  "
              f"RPY-vs-Stokeslet gap at this node = {gap:.3e}")
        print(f"  {'order':>5} {'RPY term ON':>14} {'RPY term OFF':>14}")
        for order in range(1, 5):
            mom = build_moments(order, center, scale, pos, force)
            on = m2p(order, center, scale, mom, target, rpy_on=True)
            off = m2p(order, center, scale, mom, target, rpy_on=False)
            e_on = np.linalg.norm(on - ref_rpy) / np.linalg.norm(ref_rpy)
            e_off = np.linalg.norm(off - ref_rpy) / np.linalg.norm(ref_rpy)
            print(f"  {order:>5} {e_on:>14.3e} {e_off:>14.3e}")
            # The RPY-off branch must still be an exact Oseen expansion.
            e_stk = np.linalg.norm(off - ref_stk) / np.linalg.norm(ref_stk)
            assert e_stk < e_off * 1.01 or e_stk < 1e-10, (order, e_stk, e_off)
        print(f"  (RPY OFF converges to the Stokeslet answer instead: "
              f"order 4 err vs Stokeslet = "
              f"{np.linalg.norm(m2p(4, center, scale, build_moments(4, center, scale, pos, force), target, False) - ref_stk) / np.linalg.norm(ref_stk):.3e})")


if __name__ == "__main__":
    main()
