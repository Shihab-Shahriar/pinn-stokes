"""Band-moment neighbourhood encoding for the n-body correction.

Implements moments_for_nbody.md sections 3-5 in batched, TorchScript-scriptable
torch: radial band weights, the (s_a, v_a, Q_a) moments of a pair's
neighbourhood, the pair-symmetric rotational invariants that feed the MLP, and
the tensor bases the predicted coefficients multiply onto.

Two layouts share this code:

* **v2** (the published ``nbody_moments_v2_*`` models): 8 unit-width tent bands
  with centres 0.5 .. 7.5, 9 invariants per band (72), TT/RR bases
  {I, zz'} + per band {Q_a, S(zz'Q_a), v_a v_a', Alt(z v_a')} (34), TR bases
  {E(z)} + per band {E(Q_a z), S(z (z x v_a)'), (z.v_a) E(v_a)} (25) written
  identically into both off-diagonal corners (RT = TR): 93 coefficients.
  The v2-named functions below (``band_weights``, ``band_moments``, ``bases``,
  ``assemble_block``, ``moment_features``) are that layout, unchanged.

* **v3** (configurable): the same 8 bands and rows, optional quadratic bases
  and an optional second class of TR bases (``bases_v3``,
  ``assemble_block_v3``).  Class-1 TR bases satisfy T(-z) = T(z)' and are
  written as TR = RT = T; class-2 bases satisfy T(-z) = -T(z)' ((z-even,
  antisymmetric) or (z-odd, symmetric), one epsilon each) and are written as
  TR = +T, RT = -T.  Both keep M_ji = M_ij' with swap-even coefficients; the
  pair block's TR and RT are no longer tied, which the residual labels need
  (||TR - RT|| / ||TR|| = 0.8 on dataset v2).  Class 2 per band:
  {E(v_a), (z.v_a) E(z), [E(z), Q_a]}; all v3 bases are linear in the moments.

Conventions (shared by training and inference; see plan / report):
  * ``s_vec = x_source - x_target`` (the data/operator convention); the pair
    axis is ``z = -s_vec / |s_vec|`` = the doc's ``\\hat z`` = the baseline's
    ``d_vec`` after negation.
  * Neighbour vectors ``nbr`` are relative to the target, so the pair midpoint
    is ``s_vec / 2`` and ``r_k = nbr_k - s_vec / 2``.
  * ``skew(u)_ab = eps_abc u_c`` = ``model_archs.L3``; ``skew(u) @ w = w x u``.
  * Band weights are a tent partition of unity on [0, inf): the first band
    saturates below its knot, the last band above its knot.  Deliberate
    deviation from the doc: NO zeroing beyond r_c = 8.  The neighbourhood
    cutoff is enforced by neighbour *selection* (operator side), not by the
    weight function, because the training rows contain neighbours farther
    than 8 from the midpoint that do influence the label.
  * Padded neighbour slots must be masked: a zero neighbour coincides with the
    target, which is |s_vec|/2 from the midpoint and would otherwise be counted.

Model input row (``X_DIM`` = 7 + 13 * NB = 111 columns):
  [0:3]          s_vec
  [3:7]          pair scalars: d - mean, d - 2, (d - mean)^2, (d - mean)^4
  [7:7+NB]       s_a
  [.. +3NB]      v_a   (NB x 3, row-major)
  [.. +9NB]      Q_a   (NB x 3 x 3, row-major, full symmetric traceless matrix)

Coefficient layout: ``[TT (n_tt) | RR (n_tt) | TR class 1 (n_tr1) | TR class 2 (n_tr2)]``,
``n_tt = 2 + (3 + q) NB``, ``n_tr1 = 1 + (2 + q) NB``, ``n_tr2 = 3 t NB`` with q = quadratic
bases on, t = class 2 on; ``N_COEF = 5 + (8 + 3q + 3t) NB`` (93 for v2 and the adopted v3).
"""

from typing import Dict, List, Tuple

import torch
from torch import Tensor

NB: int = 8                       # v2 radial bands, centres 0.5 .. 7.5
EPS: float = 1e-6
N_PAIR: int = 4                   # pair scalars
N_INV: int = 9 * NB               # 72 neighbourhood invariants (v2)
N_IN: int = N_PAIR + N_INV        # 76 MLP inputs (v2)
N_TT: int = 2 + 4 * NB            # 34 TT / RR bases (v2)
N_TR: int = 1 + 3 * NB            # 25 TR = RT bases (v2)
N_COEF: int = 2 * N_TT + N_TR     # 93 coefficients (v2)
OFF_PAIR: int = 3
OFF_S: int = OFF_PAIR + N_PAIR    # 7
OFF_V: int = OFF_S + NB           # 15
OFF_Q: int = OFF_V + 3 * NB       # 39
X_DIM: int = OFF_Q + 9 * NB       # 111
N_INV_PER_BAND: int = 9           # full invariant set
N_INV_PER_BAND_REDUCED: int = 5   # {s, |v|^2, (z.v)^2, z'Qz, trQ^2}
N_TR2_PER_BAND: int = 3           # class-2 TR bases per band

# Named layouts (plain Python; the trainer / operator / sidecars use these names).
BASES: Dict[str, Tuple[bool, bool]] = {      # name -> (use_quadratic, has_tr2)
    "v2": (True, False),                    # the published layout
    "linear": (False, False),               # drop v_a v_a' (TT, RR) and (z.v_a) E(v_a) (TR)
    "linear_tr2": (False, True),            # linear + class-2 TR bases (RT = -TR part)
    "v2_tr2": (True, True),                 # v2 + class-2 (ablation)
}
INVARIANTS: Dict[str, bool] = {"full": False, "reduced": True}   # name -> reduced flag


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


# ----------------------------------------------------------------------------- bands
def band_weights(r: Tensor) -> Tensor:
    """v2: tent partition of unity over 8 unit-width bands; r[...] -> [..., 8]."""
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
    rh = r / rn.clamp_min(_eps()).unsqueeze(-1)               # [P,K,3]
    W = band_weights(rn) * mask.unsqueeze(-1)                 # [P,K,NB]
    s = W.sum(1)                                              # [P,NB]
    v = torch.einsum('pka,pki->pai', [W, rh])                 # [P,NB,3]
    Q = torch.einsum('pka,pki,pkj->paij', [W, rh, rh])        # [P,NB,3,3]
    I3 = torch.eye(3, device=rh.device, dtype=rh.dtype)
    Q = Q - (s / 3.0).unsqueeze(-1).unsqueeze(-1) * I3
    return s, v, Q


# ----------------------------------------------------------------------------- invariants
def invariants(z: Tensor, s: Tensor, v: Tensor, Q: Tensor, reduced: bool = False) -> Tensor:
    """Rotation-invariant, swap-even scalars, block-major (index = block * NB + band), NB = s.shape[1].

    full (9 per band, doc section 4 order): s | |v|^2 | (z.v)^2 | z'Qz | |Qz|^2 | trQ^2 | trQ^3 | v'Qv | (z.v)(z'Qv)
    reduced (5 per band):                    s | |v|^2 | (z.v)^2 | z'Qz | trQ^2
    """
    P = z.shape[0]
    nb = s.shape[1]
    z1 = z.unsqueeze(1)                                       # [P,1,3]
    zv = (v * z1).sum(-1)                                     # [P,NB]
    Qz = torch.einsum('paij,pj->pai', [Q, z])                 # [P,NB,3]
    zQz = (Qz * z1).sum(-1)
    trQ2 = (Q * Q).sum(-1).sum(-1)                            # Q symmetric
    if reduced:
        inv = torch.stack([s, (v * v).sum(-1), zv * zv, zQz, trQ2], 1)
        return inv.reshape(P, 5 * nb)
    Qz2 = (Qz * Qz).sum(-1)
    trQ3 = torch.einsum('paij,pajk,paki->pa', [Q, Q, Q])
    Qv = torch.einsum('paij,paj->pai', [Q, v])
    vQv = (v * Qv).sum(-1)
    zQv = (Qv * z1).sum(-1)
    inv = torch.stack([s, (v * v).sum(-1), zv * zv, zQz, Qz2, trQ2, trQ3, vQv, zv * zQv], 1)
    return inv.reshape(P, 9 * nb)


# ----------------------------------------------------------------------------- bases
def bases_v3(z: Tensor, v: Tensor, Q: Tensor, use_quadratic: bool, has_tr2: bool) -> Tuple[Tensor, Tensor, Tensor]:
    """Tensor bases, band-major, NB = v.shape[1].

    tt  [P, 2 + (3+q) NB, 3, 3]: I, zz', per band {Q_a, S(zz'Q_a), [v_a v_a'], Alt(z v_a')}      (T(-z) = T')
    tr1 [P, 1 + (2+q) NB, 3, 3]: E(z), per band {E(Q_a z), S(z (z x v_a)'), [(z.v_a) E(v_a)]}    (T(-z) = T', RT = +TR)
    tr2 [P, 3 NB or 0, 3, 3]:    per band {E(v_a), (z.v_a) E(z), E(z) Q_a - Q_a E(z)}             (T(-z) = -T', RT = -TR)
    The bracketed (quadratic) entries are present iff ``use_quadratic``; with (True, False) this is the v2 layout.
    """
    P = z.shape[0]
    nb = v.shape[1]
    I3 = torch.eye(3, device=z.device, dtype=z.dtype).unsqueeze(0).expand(P, 3, 3)
    zz = torch.einsum('pi,pj->pij', [z, z])
    z1 = z.unsqueeze(1)
    zv = (v * z1).sum(-1)                                     # [P,NB]
    zzQ = torch.einsum('pij,pajk->paik', [zz, Q])
    zvT = torch.einsum('pi,paj->paij', [z, v])
    Qz = torch.einsum('paij,pj->pai', [Q, z])
    zxv = torch.cross(z1.expand_as(v), v, dim=-1)
    z_zxv = torch.einsum('pi,paj->paij', [z, zxv])
    Ez = skew(z)                                              # [P,3,3]
    per_tt: List[Tensor] = [Q, sym(zzQ)]
    if use_quadratic:
        per_tt.append(torch.einsum('pai,paj->paij', [v, v]))
    per_tt.append(alt(zvT))
    tt = torch.cat([I3.unsqueeze(1), zz.unsqueeze(1), torch.stack(per_tt, 2).reshape(P, len(per_tt) * nb, 3, 3)], 1)
    per_tr: List[Tensor] = [skew(Qz), sym(z_zxv)]
    if use_quadratic:
        per_tr.append(zv.unsqueeze(-1).unsqueeze(-1) * skew(v))
    tr1 = torch.cat([Ez.unsqueeze(1), torch.stack(per_tr, 2).reshape(P, len(per_tr) * nb, 3, 3)], 1)
    if has_tr2:
        Ez1 = Ez.unsqueeze(1)                                 # [P,1,3,3]
        comm = torch.matmul(Ez1, Q) - torch.matmul(Q, Ez1)    # [E(z), Q_a]: z-odd, symmetric, pseudo
        per_tr2: List[Tensor] = [skew(v), zv.unsqueeze(-1).unsqueeze(-1) * Ez1, comm]
        tr2 = torch.stack(per_tr2, 2).reshape(P, 3 * nb, 3, 3)
    else:
        tr2 = tt[:, :0]
    return tt, tr1, tr2


def bases(z: Tensor, v: Tensor, Q: Tensor) -> Tuple[Tensor, Tensor]:
    """v2 bases: tt [P, 34, 3, 3], tr [P, 25, 3, 3] (``bases_v3`` with quadratic on, class 2 off)."""
    tt, tr1, _ = bases_v3(z, v, Q, True, False)
    return tt, tr1


def assemble_block_v3(c: Tensor, tt: Tensor, tr1: Tensor, tr2: Tensor) -> Tensor:
    """c [P, 2 n_tt + n_tr1 + n_tr2] -> 6x6 block: TT = c[:n].tt, RR = c[n:2n].tt,
    TR = T1 + T2, RT = T1 - T2 with T1 = c[2n:2n+n1].tr1, T2 = c[2n+n1:].tr2 (empty tr2 -> RT = TR)."""
    n = tt.shape[1]
    n1 = tr1.shape[1]
    n2 = tr2.shape[1]
    assert c.shape[1] == 2 * n + n1 + n2, "coefficient count does not match the basis layout"
    TT = torch.einsum('pb,pbij->pij', [c[:, :n], tt])
    RR = torch.einsum('pb,pbij->pij', [c[:, n:2 * n], tt])
    T1 = torch.einsum('pb,pbij->pij', [c[:, 2 * n:2 * n + n1], tr1])
    if n2 > 0:
        T2 = torch.einsum('pb,pbij->pij', [c[:, 2 * n + n1:], tr2])
        TR = T1 + T2
        RT = T1 - T2
    else:
        TR = T1
        RT = T1
    top = torch.cat([TT, TR], 2)
    bot = torch.cat([RT, RR], 2)
    return torch.cat([top, bot], 1)


def assemble_block(c: Tensor, tt: Tensor, tr: Tensor) -> Tensor:
    """v2: c [P, 93] -> 6x6 block, TT = c[:34].tt, RR = c[34:68].tt, TR = RT = c[68:].tr."""
    return assemble_block_v3(c, tt, tr, tr[:, :0])


# ----------------------------------------------------------------------------- rows
def pack_features(s_vec: Tensor, pair: Tensor, s: Tensor, v: Tensor, Q: Tensor) -> Tensor:
    P = s_vec.shape[0]
    nb = s.shape[1]
    return torch.cat([s_vec, pair, s, v.reshape(P, 3 * nb), Q.reshape(P, 9 * nb)], 1)


def unpack_features(X: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    P = X.shape[0]
    nb = (X.shape[1] - 7) // 13
    assert 7 + 13 * nb == X.shape[1], "row width must be 7 + 13 * NB"
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
    """Model input row [P, 111] from raw geometry (any float dtype); the same for every layout."""
    dist = torch.sqrt((s_vec * s_vec).sum(-1))
    pair = pair_scalars(dist, mean_dist_s)
    s, v, Q = band_moments(s_vec, nbr, mask)
    return pack_features(s_vec, pair, s, v, Q)


# ----------------------------------------------------------------------------- layout bookkeeping (plain Python)
def layout_dims(use_quadratic: bool, has_tr2: bool, reduced_inv: bool) -> Dict[str, int]:
    """Widths of a layout: MLP inputs, basis counts, coefficient count and row width."""
    nb = NB
    q = 1 if use_quadratic else 0
    n_tt = 2 + (3 + q) * nb
    n_tr1 = 1 + (2 + q) * nb
    n_tr2 = N_TR2_PER_BAND * nb if has_tr2 else 0
    n_in = N_PAIR + (N_INV_PER_BAND_REDUCED if reduced_inv else N_INV_PER_BAND) * nb
    return {"nb": nb, "n_in": n_in, "n_tt": n_tt, "n_tr1": n_tr1, "n_tr2": n_tr2,
            "n_coef": 2 * n_tt + n_tr1 + n_tr2, "x_dim": X_DIM}


def bases_name(use_quadratic: bool, has_tr2: bool) -> str:
    for name, flags in BASES.items():
        if flags == (bool(use_quadratic), bool(has_tr2)):
            return name
    raise ValueError((use_quadratic, has_tr2))


def layout_of_model(model) -> dict:
    """Layout of a ``MultiBodyMoments`` (eager or TorchScript-loaded).  Models without a ``use_quadratic``
    attribute (the published v2 ``.pt`` files) are the v2 layout.  Models exported while the band knots were
    configurable carry them in ``band_knots``; only the standard bands are supported."""
    if not hasattr(model, "use_quadratic"):
        use_quadratic, has_tr2, reduced_inv = True, False, False
    else:
        use_quadratic = bool(model.use_quadratic)
        has_tr2 = bool(model.has_tr2)
        reduced_inv = bool(model.reduced_inv)
    knots = getattr(model, "band_knots", None)
    assert knots is None or knots.detach().cpu().tolist() == [a + 0.5 for a in range(NB)], \
        f"non-standard band knots are no longer supported: {knots}"
    assert not bool(getattr(model, "learned_radial", False)), "learned radial bands are no longer supported"
    is_v2 = use_quadratic and not has_tr2 and not reduced_inv
    return {"version": "v2" if is_v2 else "v3", "bases": bases_name(use_quadratic, has_tr2),
            "invariants": "reduced" if reduced_inv else "full", **layout_dims(use_quadratic, has_tr2, reduced_inv)}


def layout_from_sidecar(meta: dict) -> dict:
    """``MultiBodyMoments`` layout kwargs (bases / invariants) from a model sidecar; v2 when absent."""
    bands = meta.get("bands")
    assert bands is None or list(bands) == [a + 0.5 for a in range(NB)], f"non-standard bands: {bands}"
    assert (meta.get("radial") or "knots") == "knots", "learned radial bands are no longer supported"
    return {"bases": meta.get("bases", "v2"), "invariants": meta.get("invariants", "full")}

# ---------------------------------------------------------------------------
# Self-block (per-particle diagonal) correction: moments about the particle.
#
# A particle's neighbourhood (all k != t within a cutoff of x_t; selection in
# nbody_features.select_particle_neighbours) is encoded by band moments about
# the particle itself (r_k = x_k - x_t) -- there is no pair axis.  The
# NB_SELF tent bands are remapped to [BAND_LO, BAND_HI] = [2, 8] (width 0.75):
# hard spheres put every neighbour at r >= 2, so the pair layout's unit bands
# 1-2 would be structurally empty.  Bands 0 / NB-1 saturate below/above their
# centres, so the partition of unity holds on [0, inf) (selection enforces
# r <= BAND_HI; the tails are safety margins only).
#
# Invariants (6 per band, 48): s_a, |v_a|^2, tr Q_a^2, tr Q_a^3, v_a'Q_a v_a,
# |Q_a v_a|^2 -- all true rotation scalars (no pseudoscalar), so the MLP input
# is invariant under the full O(3), reflections included.
# Bases: TT/RR (symmetric true tensors) {I} + per band {Q_a, v_a v_a', Q_a^2,
# S(Q_a v_a v_a')} = 33 each; TR (pseudotensors, one eps each) per band
# {E(v_a), E(Q_a v_a), [Q_a, E(v_a)]} = 24 with NO constant term (there is no
# isotropic rank-2 pseudotensor), so an isolated particle reduces to
# TT = c0 I, RR = c33 I, TR = 0.  Assembly ties RT = TR^T: the block is
# *symmetric by construction* (unlike assemble_block's RT = TR, which encodes
# pair reciprocity) and the grand mobility stays symmetric.
#
# Self model input row (X_DIM_SELF = 104 columns):
#   [0:8]    s_a  (8)
#   [8:32]   v_a  (8 x 3, row-major)
#   [32:104] Q_a  (8 x 3 x 3, row-major, full symmetric traceless matrix)
# ---------------------------------------------------------------------------

NB_SELF: int = 8                          # radial bands, centres 2.375 .. 7.625
BAND_LO: float = 2.0
BAND_HI: float = 8.0
N_INV_SELF: int = 6 * NB_SELF             # 48 invariants
N_TT_SELF: int = 1 + 4 * NB_SELF          # 33 TT / RR bases
N_TR_SELF: int = 3 * NB_SELF              # 24 TR bases (RT = TR^T)
N_COEF_SELF: int = 2 * N_TT_SELF + N_TR_SELF   # 90 coefficients
X_DIM_SELF: int = 13 * NB_SELF            # 104


def _band_lo() -> float:
    return 2.0


def _band_hi() -> float:
    return 8.0


def _n_tt_self() -> int:
    return 33


def self_band_weights(r: Tensor) -> Tensor:
    """Tent partition of unity over NB_SELF width-h bands on [BAND_LO, BAND_HI]; r[...] -> [..., NB]."""
    h = (_band_hi() - _band_lo()) / float(_nb())
    cols: List[Tensor] = []
    for a in range(_nb()):
        c = _band_lo() + (float(a) + 0.5) * h
        w = torch.clamp(1.0 - torch.abs(r - c) / h, min=0.0)
        if a == 0:
            w = torch.where(r <= c, torch.ones_like(w), w)
        if a == _nb() - 1:
            w = torch.where(r >= c, torch.ones_like(w), w)
        cols.append(w)
    return torch.stack(cols, -1)


def self_band_moments(nbr: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Band moments of a particle's neighbourhood.

    nbr   [P, K, 3]  neighbour positions relative to the particle (padded)
    mask  [P, K]  1.0 for real neighbours, 0.0 for padding
    returns s [P, NB], v [P, NB, 3], Q [P, NB, 3, 3] (symmetric, traceless)
    """
    rn = torch.sqrt((nbr * nbr).sum(-1))                      # [P,K]
    rh = nbr / rn.clamp_min(_eps()).unsqueeze(-1)             # [P,K,3]
    W = self_band_weights(rn) * mask.unsqueeze(-1)            # [P,K,NB]
    s = W.sum(1)                                              # [P,NB]
    v = torch.einsum('pka,pki->pai', [W, rh])                 # [P,NB,3]
    Q = torch.einsum('pka,pki,pkj->paij', [W, rh, rh])        # [P,NB,3,3]
    I3 = torch.eye(3, device=nbr.device, dtype=nbr.dtype)
    Q = Q - (s / 3.0).unsqueeze(-1).unsqueeze(-1) * I3
    return s, v, Q


def self_invariants(s: Tensor, v: Tensor, Q: Tensor) -> Tensor:
    """48 rotation invariants (no pseudoscalars), block-major (index = block * NB + band):
    s | |v|^2 | trQ^2 | trQ^3 | v'Qv | |Qv|^2."""
    P = s.shape[0]
    trQ2 = (Q * Q).sum(-1).sum(-1)                            # Q symmetric
    trQ3 = torch.einsum('paij,pajk,paki->pa', [Q, Q, Q])
    Qv = torch.einsum('paij,paj->pai', [Q, v])
    vQv = (v * Qv).sum(-1)
    Qv2 = (Qv * Qv).sum(-1)
    inv = torch.stack([s, (v * v).sum(-1), trQ2, trQ3, vQv, Qv2], 1)
    return inv.reshape(P, 6 * _nb())


def self_bases(v: Tensor, Q: Tensor) -> Tuple[Tensor, Tensor]:
    """Tensor bases for the self block, band-major.

    tt [P, 33, 3, 3]: I, then per band {Q_a, v_a v_a', Q_a^2, S(Q_a v_a v_a')} -- all symmetric true tensors
    tr [P, 24, 3, 3]: per band {E(v_a), E(Q_a v_a), [Q_a, E(v_a)] = S(Q_a E(v_a))} -- pseudotensors, no constant
    """
    P = v.shape[0]
    I3 = torch.eye(3, device=v.device, dtype=v.dtype).unsqueeze(0).expand(P, 3, 3)
    vv = torch.einsum('pai,paj->paij', [v, v])
    QQ = torch.einsum('paij,pajk->paik', [Q, Q])
    Qvv = torch.einsum('paij,pajk->paik', [Q, vv])
    Ev = skew(v)                                              # [P,NB,3,3]
    Qvec = torch.einsum('paij,paj->pai', [Q, v])
    EQv = skew(Qvec)
    QEv = torch.einsum('paij,pajk->paik', [Q, Ev])
    per_tt = torch.stack([Q, vv, QQ, sym(Qvv)], 2)            # [P,NB,4,3,3]
    tt = torch.cat([I3.unsqueeze(1), per_tt.reshape(P, 4 * _nb(), 3, 3)], 1)
    per_tr = torch.stack([Ev, EQv, sym(QEv)], 2)              # sym(QE) = QE - EQ (E antisym, Q sym)
    tr = per_tr.reshape(P, 3 * _nb(), 3, 3)
    return tt, tr


def self_assemble_block(c: Tensor, tt: Tensor, tr: Tensor) -> Tensor:
    """c [P, 90] -> symmetric 6x6 block: TT = c[:33].tt, RR = c[33:66].tt, TR = c[66:].tr, RT = TR^T."""
    ntt = _n_tt_self()
    TT = torch.einsum('pb,pbij->pij', [c[:, :ntt], tt])
    RR = torch.einsum('pb,pbij->pij', [c[:, ntt:2 * ntt], tt])
    TR = torch.einsum('pb,pbij->pij', [c[:, 2 * ntt:], tr])
    top = torch.cat([TT, TR], 2)
    bot = torch.cat([TR.transpose(-1, -2), RR], 2)
    return torch.cat([top, bot], 1)


def self_pack_features(s: Tensor, v: Tensor, Q: Tensor) -> Tensor:
    P = s.shape[0]
    nb = _nb()
    return torch.cat([s, v.reshape(P, 3 * nb), Q.reshape(P, 9 * nb)], 1)


def self_unpack_features(X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    P = X.shape[0]
    nb = _nb()
    off_v = nb
    off_q = off_v + 3 * nb
    s = X[:, 0:off_v]
    v = X[:, off_v:off_q].reshape(P, nb, 3)
    Q = X[:, off_q:off_q + 9 * nb].reshape(P, nb, 3, 3)
    return s, v, Q


def self_moment_features(nbr: Tensor, mask: Tensor) -> Tensor:
    """Self-model input rows X[P, 104] from particle-relative geometry."""
    s, v, Q = self_band_moments(nbr, mask)
    return self_pack_features(s, v, Q)
