"""GPU moments-based n-body mobility operator: pair correction + learned diagonal.

Port of the CPU ``Mob_Op_Nbody_Moments`` (src/mob_op_nbody_moments.py) to the GPU
stack, for the published pc8 configuration (``nbody_moments_v2_kinf_rc8_pc8`` +
``nbody_diag_v2_pc8``): switch_dist = pair_cutoff = neighbor_cutoff = diag_cutoff = 8,
max_neighbors = None (every particle within 8 of the pair midpoint contributes).

Design notes (what makes this fast, and where it deliberately differs from the CPU
operator in *mechanics* while matching it in *math*):

* **One model call per unordered pair.** The CPU operator evaluates the network for
  (t,s) and (s,t) separately. The moments of an unordered pair are identical for both
  directions (only the pair axis flips) and the invariants are swap-even, so the
  coefficients agree exactly and the assembled blocks satisfy K_st = K_ts^T *by
  construction* (pinned by tests/test_nbody_moments.py). We therefore evaluate K_ts
  once and apply v_t += K_ts F_s, v_s += K_ts^T F_t.

* **No per-pair padded neighbour tensors.** With max_neighbors = None a padded
  (P, K, 3) tensor is unbounded; instead the band moments (s_a, v_a, Q_a) are *sums*
  over neighbours, accumulated by index_add over a (pair -> neighbour) COO edge list.
  The tent partition of unity touches at most 2 adjacent bands per neighbour, so each
  edge contributes 2 x 10 values (w, w*rh, w*rh rh^T upper triangle).

* **Midpoint neighbour search in Warp.** Neighbours of a pair live within
  ``neighbor_cutoff`` of the pair *midpoint*, which is NOT a subset of either
  endpoint's neighbour list (a particle 8 from the midpoint can be ~12 from both
  endpoints), so the near-field pair list cannot be reused. A hash grid over particle
  positions (cell = cutoff/2) is queried at midpoints; count + exclusive-scan + fill
  emits the edge list per pair chunk.

* **The diagonal reuses the near-field edge list.** With diag_cutoff == switch_dist,
  the particle neighbourhood (all k != t within 8) is exactly the ordered near-pair
  list, so the self band moments come from one scatter pass over edges that already
  exist.

* **Chunked, compiled, dynamic.** Same invariants as the other GPU paths (see
  CLAUDE.md): the chunk loops live in Python outside torch.compile; each compiled
  unit takes chunk-sized tensors with the leading dim marked dynamic. Pair chunks
  bound the edge list; edge sub-chunks bound the scatter temporaries.

Semantics matched to the CPU operator:
  * pairs with zero neighbours receive no pair correction (the CPU drops them);
  * zero-neighbour particles DO receive the diagonal correction (constant term kept,
    as in training);
  * the pair correction is not divided by viscosity (absorbed by the model, labels at
    mu = 1), the diagonal correction IS divided by viscosity;
  * features are bit-compatible with ``nbody_features.moment_features`` /
    ``self_moment_features`` up to fp32-vs-fp64 accumulation order.
"""
from __future__ import annotations

import copy
import os
import time
from typing import Optional, Tuple

import torch
import warp as wp

from src.gpu_mob_2b import NNMobTorch, TensorLike, DEFAULT_TWO_BODY_CHUNK
from src.model_archs import MultiBodyMoments, SelfBlockMoments

wp.init()

DEFAULT_MEAN_DIST_S = 4.690027344329476
# Unordered pairs per moments chunk. Sized so the per-chunk edge list (~50-110
# neighbours/pair at the 1M benchmark's densities) stays a few hundred MB.
DEFAULT_MOMENTS_PAIR_CHUNK = 524_288
# (pair -> neighbour) edges per scatter sub-chunk; bounds the (E, 10) value
# temporaries at ~320 MB each.
DEFAULT_EDGE_CHUNK = 8_000_000
# Particles per diagonal model chunk; bounds the (P, 57, 3, 3) basis temporaries.
DEFAULT_DIAG_ROW_CHUNK = 262_144


# ---------------------------------------------------------------------------
# Warp kernels: neighbours of pair midpoints via hash grid
# ---------------------------------------------------------------------------
@wp.kernel
def count_midpoint_neighbors_kernel(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    mids: wp.array(dtype=wp.vec3),
    t_idx: wp.array(dtype=wp.int32),
    s_idx: wp.array(dtype=wp.int32),
    radius: float,
    radius_sq: float,
    counts: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    m = mids[tid]
    t = t_idx[tid]
    s = s_idx[tid]
    c = int(0)
    neighbors = wp.hash_grid_query(grid, m, radius)
    for index in neighbors:
        if index != t and index != s:
            d = wp.length_sq(positions[index] - m)
            # <= to match nbody_features.select_pair_neighbours (the training path)
            if d <= radius_sq:
                c += 1
    counts[tid] = c


@wp.kernel
def fill_midpoint_edges_kernel(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    mids: wp.array(dtype=wp.vec3),
    t_idx: wp.array(dtype=wp.int32),
    s_idx: wp.array(dtype=wp.int32),
    idx_start: wp.array(dtype=wp.int32),
    radius: float,
    radius_sq: float,
    edge_pair: wp.array(dtype=wp.int32),
    edge_nbr: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    m = mids[tid]
    t = t_idx[tid]
    s = s_idx[tid]
    k = idx_start[tid]
    neighbors = wp.hash_grid_query(grid, m, radius)
    for index in neighbors:
        if index != t and index != s:
            d = wp.length_sq(positions[index] - m)
            if d <= radius_sq:
                edge_pair[k] = tid
                edge_nbr[k] = index
                k += 1


# Fused single-pass variant: accumulate the 8x10 band-moment table of one pair
# directly during the hash-grid traversal (per-thread local accumulator, written
# once). Removes the second traversal, the edge list and the index_add atomics of
# the two-pass + torch-scatter path; on a 1M/phi=0.1 step that path moved ~100 GB
# through global atomics alone.
# Per-pair band-moment table type: 8 bands x [w, w*rh (3), w*rh rh^T upper (6)].
# Dynamically indexed per-thread local array (Warp places it in local memory).
# Measured against a register-resident variant with a 7-way band branch: the
# branch divergence made it ~5x SLOWER (7.8 s vs 1.7 s at 27M pairs on the 4060);
# the L1-backed local array wins because only ~80 B of it is touched per
# neighbour and there is no warp serialization.
mat810 = wp.types.matrix(shape=(8, 10), dtype=wp.float32)


@wp.kernel
def accumulate_pair_moments_kernel(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    mids: wp.array(dtype=wp.vec3),
    t_idx: wp.array(dtype=wp.int32),
    s_idx: wp.array(dtype=wp.int32),
    radius: float,
    radius_sq: float,
    out: wp.array2d(dtype=wp.float32),   # (P, 80), overwritten
):
    tid = wp.tid()
    m = mids[tid]
    t = t_idx[tid]
    s = s_idx[tid]
    acc = mat810(0.0)
    neighbors = wp.hash_grid_query(grid, m, radius)
    for index in neighbors:
        if index != t and index != s:
            d = positions[index] - m
            r2 = wp.length_sq(d)
            if r2 <= radius_sq:
                rn = wp.sqrt(r2)
                rh = d / wp.max(rn, 1.0e-6)
                tt = wp.clamp(rn - 0.5, 0.0, 7.0)
                a0 = wp.min(int(wp.floor(tt)), 6)
                f = tt - float(a0)
                x = rh[0]
                y = rh[1]
                z = rh[2]
                w0 = 1.0 - f
                acc[a0, 0] += w0
                acc[a0, 1] += w0 * x
                acc[a0, 2] += w0 * y
                acc[a0, 3] += w0 * z
                acc[a0, 4] += w0 * x * x
                acc[a0, 5] += w0 * x * y
                acc[a0, 6] += w0 * x * z
                acc[a0, 7] += w0 * y * y
                acc[a0, 8] += w0 * y * z
                acc[a0, 9] += w0 * z * z
                a1 = a0 + 1
                acc[a1, 0] += f
                acc[a1, 1] += f * x
                acc[a1, 2] += f * y
                acc[a1, 3] += f * z
                acc[a1, 4] += f * x * x
                acc[a1, 5] += f * x * y
                acc[a1, 6] += f * x * z
                acc[a1, 7] += f * y * y
                acc[a1, 8] += f * y * z
                acc[a1, 9] += f * z * z
    for a in range(8):
        for j in range(10):
            out[tid, a * 10 + j] = acc[a, j]


@wp.func
def _band_Q(mom: wp.array2d(dtype=wp.float32), tid: int, a: int) -> wp.mat33:
    """Traceless symmetric Q_a from the band-moment row (trace of the raw
    second moment is s_a, exactly as in nbody_moments.band_moments)."""
    t3 = mom[tid, a * 10 + 0] / 3.0
    xx = mom[tid, a * 10 + 4] - t3
    xy = mom[tid, a * 10 + 5]
    xz = mom[tid, a * 10 + 6]
    yy = mom[tid, a * 10 + 7] - t3
    yz = mom[tid, a * 10 + 8]
    zz = mom[tid, a * 10 + 9] - t3
    return wp.mat33(xx, xy, xz, xy, yy, yz, xz, yz, zz)


@wp.func
def _band_v(mom: wp.array2d(dtype=wp.float32), tid: int, a: int) -> wp.vec3:
    return wp.vec3(mom[tid, a * 10 + 1], mom[tid, a * 10 + 2], mom[tid, a * 10 + 3])


@wp.func
def _skew(u: wp.vec3) -> wp.mat33:
    """E(u)_ab = eps_abc u_c (nbody_moments.skew; E(u) @ w = w x u)."""
    return wp.mat33(0.0, u[2], -u[1],
                    -u[2], 0.0, u[0],
                    u[1], -u[0], 0.0)


@wp.kernel
def pair_invariants_kernel(
    positions: wp.array(dtype=wp.vec3),
    t_idx: wp.array(dtype=wp.int32),
    s_idx: wp.array(dtype=wp.int32),
    mom: wp.array2d(dtype=wp.float32),    # (P, 80) band moments
    mean_dist_s: float,
    out: wp.array2d(dtype=wp.float32),    # (P, 76) raw model inputs
):
    """The 4 pair scalars + 72 rotation invariants of nbody_moments.invariants,
    block-major (col = 4 + block * 8 + band), computed per pair in registers --
    replaces ~100 GB of (P, 8, 3, 3) torch intermediates at 27M pairs."""
    tid = wp.tid()
    s_vec = positions[s_idx[tid]] - positions[t_idx[tid]]
    dist = wp.length(s_vec)
    z = -s_vec / wp.max(dist, 1.0e-6)
    dc = dist - mean_dist_s
    out[tid, 0] = dc
    out[tid, 1] = dist - 2.0
    out[tid, 2] = dc * dc
    out[tid, 3] = dc * dc * dc * dc
    for a in range(8):
        sa = mom[tid, a * 10 + 0]
        v = _band_v(mom, tid, a)
        Q = _band_Q(mom, tid, a)
        zv = wp.dot(v, z)
        Qz = Q * z
        Qv = Q * v
        QQ = Q * Q
        out[tid, 4 + 0 * 8 + a] = sa
        out[tid, 4 + 1 * 8 + a] = wp.dot(v, v)
        out[tid, 4 + 2 * 8 + a] = zv * zv
        out[tid, 4 + 3 * 8 + a] = wp.dot(Qz, z)
        out[tid, 4 + 4 * 8 + a] = wp.dot(Qz, Qz)
        out[tid, 4 + 5 * 8 + a] = wp.ddot(Q, Q)
        out[tid, 4 + 6 * 8 + a] = wp.trace(QQ * Q)
        out[tid, 4 + 7 * 8 + a] = wp.dot(v, Qv)
        out[tid, 4 + 8 * 8 + a] = zv * wp.dot(Qv, z)


@wp.kernel
def pair_assemble_apply_kernel(
    positions: wp.array(dtype=wp.vec3),
    t_idx: wp.array(dtype=wp.int32),
    s_idx: wp.array(dtype=wp.int32),
    mom: wp.array2d(dtype=wp.float32),    # (P, 80) band moments
    coef: wp.array2d(dtype=wp.float32),   # (P, 93) scaled coefficients
    force: wp.array2d(dtype=wp.float32),  # (N, 6)
    v_t: wp.array2d(dtype=wp.float32),    # (P, 6) out: K_ts F_s
    v_s: wp.array2d(dtype=wp.float32),    # (P, 6) out: K_ts^T F_t
):
    """nbody_moments.bases + assemble_block + both force products, in registers.
    Basis order must match bases(): tt = [I, zz'] + per band {Q, sym(zz'Q),
    vv', alt(zv')} (band-major, 4 per band); tr = [E(z)] + per band {E(Qz),
    sym(z (zxv)'), (z.v) E(v)}. TT = c[0:34].tt, RR = c[34:68].tt,
    TR = RT = c[68:93].tr; K = [[TT, TR], [TR, RR]]."""
    tid = wp.tid()
    t = t_idx[tid]
    s = s_idx[tid]
    s_vec = positions[s] - positions[t]
    z = -s_vec / wp.max(wp.length(s_vec), 1.0e-6)

    total = float(0.0)
    for a in range(8):
        total += mom[tid, a * 10 + 0]
    if total <= 0.0:    # zero-neighbour pair: no correction (CPU drops them)
        for j in range(6):
            v_t[tid, j] = 0.0
            v_s[tid, j] = 0.0
        return

    I3 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    zz = wp.outer(z, z)
    TT = coef[tid, 0] * I3 + coef[tid, 1] * zz
    RR = coef[tid, 34] * I3 + coef[tid, 35] * zz
    Ez = _skew(z)
    TR = coef[tid, 68] * Ez
    for a in range(8):
        v = _band_v(mom, tid, a)
        Q = _band_Q(mom, tid, a)
        zzQ = zz * Q
        s_zzQ = zzQ + wp.transpose(zzQ)
        vv = wp.outer(v, v)
        zvT = wp.outer(z, v)
        a_zvT = zvT - wp.transpose(zvT)
        b = 2 + 4 * a
        TT += coef[tid, b] * Q + coef[tid, b + 1] * s_zzQ \
            + coef[tid, b + 2] * vv + coef[tid, b + 3] * a_zvT
        TT_off = 34
        RR += coef[tid, TT_off + b] * Q + coef[tid, TT_off + b + 1] * s_zzQ \
            + coef[tid, TT_off + b + 2] * vv + coef[tid, TT_off + b + 3] * a_zvT
        Qz = Q * z
        zxv = wp.cross(z, v)
        z_zxv = wp.outer(z, zxv)
        s_z_zxv = z_zxv + wp.transpose(z_zxv)
        zv = wp.dot(z, v)
        c = 68 + 1 + 3 * a
        TR += coef[tid, c] * _skew(Qz) + coef[tid, c + 1] * s_z_zxv \
            + coef[tid, c + 2] * (zv * _skew(v))

    Fs_f = wp.vec3(force[s, 0], force[s, 1], force[s, 2])
    Fs_t = wp.vec3(force[s, 3], force[s, 4], force[s, 5])
    Ft_f = wp.vec3(force[t, 0], force[t, 1], force[t, 2])
    Ft_t = wp.vec3(force[t, 3], force[t, 4], force[t, 5])

    ut = TT * Fs_f + TR * Fs_t          # top row of K = [TT, TR]
    ot = TR * Fs_f + RR * Fs_t          # bottom row  = [TR, RR]
    TTt = wp.transpose(TT)
    TRt = wp.transpose(TR)
    RRt = wp.transpose(RR)
    us = TTt * Ft_f + TRt * Ft_t        # K^T rows
    os = TRt * Ft_f + RRt * Ft_t
    for j in range(3):
        v_t[tid, j] = ut[j]
        v_t[tid, 3 + j] = ot[j]
        v_s[tid, j] = us[j]
        v_s[tid, 3 + j] = os[j]


class MidpointNeighborSearch:
    """Hash-grid search for particles within ``radius`` of pair midpoints.

    The grid is built once per configuration over particle positions with cell size
    ``radius / 2`` (a query of radius r over cell h scans ((2r+h)/h)^3 cells; h = r/2
    scans ~40% less volume than h = r) and then queried per pair chunk. Edge buffers
    are grow-only to avoid per-chunk allocation."""

    def __init__(self, device: torch.device, grid_dim: int = 160) -> None:
        self.device = torch.device(device)
        self.grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device=str(self.device))
        self._edge_pair: Optional[torch.Tensor] = None
        self._edge_nbr: Optional[torch.Tensor] = None
        self._capacity = 0

    def _ensure_capacity(self, total: int) -> None:
        if total <= self._capacity:
            return
        new_size = max(total, 2 * self._capacity)
        self._edge_pair = torch.empty(new_size, dtype=torch.int32, device=self.device)
        self._edge_nbr = torch.empty(new_size, dtype=torch.int32, device=self.device)
        self._capacity = new_size

    def build(self, positions: torch.Tensor, radius: float,
              cell_scale: float = 0.5) -> None:
        """cell_scale sets the grid cell edge as a fraction of the query radius:
        a query of radius r over cells h scans a ((2r+h)/h)^3-cell box, so smaller
        cells trade wasted candidate volume against per-cell probe overhead."""
        assert positions.is_cuda and positions.dtype == torch.float32
        stream = wp.stream_from_torch(torch.cuda.current_stream(positions.device))
        with wp.ScopedStream(stream):
            p = wp.from_torch(positions.contiguous(), dtype=wp.vec3)
            self.grid.build(points=p, radius=float(cell_scale) * float(radius))

    def query(
        self,
        positions: torch.Tensor,
        mids: torch.Tensor,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
        radius: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Edges (pair_local int32, neighbour int32) for one chunk of pairs."""
        P = mids.shape[0]
        radius = float(radius)
        counts = torch.empty(P, dtype=torch.int32, device=self.device)
        stream = wp.stream_from_torch(torch.cuda.current_stream(positions.device))
        with wp.ScopedStream(stream):
            p = wp.from_torch(positions.contiguous(), dtype=wp.vec3)
            m = wp.from_torch(mids.contiguous(), dtype=wp.vec3)
            t_w = wp.from_torch(t_idx.contiguous(), dtype=wp.int32)
            s_w = wp.from_torch(s_idx.contiguous(), dtype=wp.int32)
            c_w = wp.from_torch(counts, dtype=wp.int32)
            wp.launch(
                count_midpoint_neighbors_kernel, dim=P,
                inputs=(self.grid.id, p, m, t_w, s_w, radius, radius * radius, c_w),
                stream=stream)
            cum = torch.cumsum(counts, dim=0, dtype=torch.int32)
            total = int(cum[-1].item())          # syncs the stream
            if total == 0:
                empty = torch.empty(0, dtype=torch.int32, device=self.device)
                return empty, empty
            self._ensure_capacity(total)
            starts = cum - counts                # exclusive prefix
            st_w = wp.from_torch(starts.contiguous(), dtype=wp.int32)
            ep_w = wp.from_torch(self._edge_pair, dtype=wp.int32)
            en_w = wp.from_torch(self._edge_nbr, dtype=wp.int32)
            wp.launch(
                fill_midpoint_edges_kernel, dim=P,
                inputs=(self.grid.id, p, m, t_w, s_w, st_w, radius, radius * radius,
                        ep_w, en_w),
                stream=stream)
        return self._edge_pair[:total], self._edge_nbr[:total]

    def accumulate(
        self,
        positions: torch.Tensor,
        mids: torch.Tensor,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
        radius: float,
        out: torch.Tensor,
    ) -> None:
        """Fused path: fill out[P, 80] with each pair's 8x10 band-moment table."""
        P = mids.shape[0]
        radius = float(radius)
        assert out.shape == (P, 80) and out.dtype == torch.float32
        stream = wp.stream_from_torch(torch.cuda.current_stream(positions.device))
        with wp.ScopedStream(stream):
            p = wp.from_torch(positions.contiguous(), dtype=wp.vec3)
            m = wp.from_torch(mids.contiguous(), dtype=wp.vec3)
            t_w = wp.from_torch(t_idx.contiguous(), dtype=wp.int32)
            s_w = wp.from_torch(s_idx.contiguous(), dtype=wp.int32)
            o_w = wp.from_torch(out)
            wp.launch(
                accumulate_pair_moments_kernel, dim=P,
                inputs=(self.grid.id, p, m, t_w, s_w, radius, radius * radius, o_w),
                stream=stream)

    @staticmethod
    def launch_pair_finish(
        positions: torch.Tensor,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
        mom: torch.Tensor,
        kernel,
        extra,
        outs,
    ) -> None:
        """Launch one of the per-pair finish kernels (invariants / assemble)."""
        P = t_idx.shape[0]
        stream = wp.stream_from_torch(torch.cuda.current_stream(positions.device))
        with wp.ScopedStream(stream):
            args = [wp.from_torch(positions.contiguous(), dtype=wp.vec3),
                    wp.from_torch(t_idx.contiguous(), dtype=wp.int32),
                    wp.from_torch(s_idx.contiguous(), dtype=wp.int32),
                    wp.from_torch(mom)]
            args += [a if isinstance(a, float) else wp.from_torch(a) for a in extra]
            args += [wp.from_torch(o) for o in outs]
            wp.launch(kernel, dim=P, inputs=args, stream=stream)


# ---------------------------------------------------------------------------
# Compiled torch kernels
# ---------------------------------------------------------------------------
def _edge_base_values(rel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-edge (r_norm, [1, rh(3), rh rh^T upper(6)]) shared by both band paths."""
    rn = torch.linalg.norm(rel, dim=1)
    rh = rel / rn.clamp_min(1e-6).unsqueeze(1)   # EPS matches nbody_moments._eps
    x, y, z = rh.unbind(1)
    base = torch.stack(
        [torch.ones_like(rn), x, y, z, x * x, x * y, x * z, y * y, y * z, z * z], 1)
    return rn, base


def _tent_bands(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Band index a0 (long, <= 6) and fraction f for tent coordinate t in [0, 7]:
    band a0 gets weight 1-f, band a0+1 gets f. Reproduces nbody_moments.band_weights
    / self_band_weights including the saturating end bands."""
    a0 = t.floor().clamp(max=6.0)
    f = t - a0
    return a0.to(torch.int64), f


def _moments_from_buf(buf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(P, 8, 10) accumulator -> s_a (P,8), v_a (P,8,3), Q_a (P,8,9) traceless."""
    s_a = buf[:, :, 0]
    v_a = buf[:, :, 1:4]
    xx, xy, xz = buf[:, :, 4], buf[:, :, 5], buf[:, :, 6]
    yy, yz, zz = buf[:, :, 7], buf[:, :, 8], buf[:, :, 9]
    tr3 = s_a / 3.0     # trace of sum(w rh rh^T) = sum w = s_a
    Q = torch.stack([xx - tr3, xy, xz, xy, yy - tr3, yz, xz, yz, zz - tr3], 2)
    return s_a, v_a, Q


class _PairScatterKernel(torch.nn.Module):
    """Accumulate one edge sub-chunk of pair band moments into ``buf`` (P*8, 10)."""

    def forward(self, pos, t_c, s_c, edge_pair, edge_nbr, buf):
        mid = 0.5 * (pos[t_c] + pos[s_c])                 # (P, 3)
        ep = edge_pair.to(torch.int64)
        rel = pos[edge_nbr] - mid[ep]
        rn, base = _edge_base_values(rel)
        # Pair bands: unit width, centres 0.5..7.5, ends saturating.
        a0, f = _tent_bands((rn - 0.5).clamp(0.0, 7.0))
        idx0 = ep * 8 + a0
        buf.index_add_(0, idx0, base * (1.0 - f).unsqueeze(1))
        buf.index_add_(0, idx0 + 1, base * f.unsqueeze(1))
        return buf


class _PairFinishKernel(torch.nn.Module):
    """Moments buffer -> model rows -> 6x6 blocks -> per-pair velocity contributions."""

    def __init__(self, model: torch.nn.Module, mean_dist_s: float) -> None:
        super().__init__()
        self.model = model
        self.mean_dist_s = float(mean_dist_s)

    def forward(self, pos, force, t_c, s_c, buf):
        P = t_c.shape[0]
        s_vec = pos[s_c] - pos[t_c]
        s_a, v_a, Q = _moments_from_buf(buf.view(P, 8, 10))
        dist = torch.linalg.norm(s_vec, dim=1)
        dc = dist - self.mean_dist_s
        dc2 = dc * dc
        pair = torch.stack([dc, dist - 2.0, dc2, dc2 * dc2], 1)
        X = torch.cat([s_vec, pair, s_a, v_a.reshape(P, 24), Q.reshape(P, 72)], 1)
        K = self.model.predict_mobility(X)                # (P, 6, 6), = K_ts
        # Zero-neighbour pairs receive no correction (CPU operator drops them).
        K = K * (s_a.sum(1) > 0).to(K.dtype).view(P, 1, 1)
        v_t = torch.einsum('pij,pj->pi', K, force[s_c])   # K_ts F_s
        v_s = torch.einsum('pij,pi->pj', K, force[t_c])   # K_st F_t = K_ts^T F_t
        return v_t, v_s


class _CoeffKernel(torch.nn.Module):
    """Standardisation + MLP + per-basis rescaling (MultiBodyMoments.coefficients
    minus the invariant construction, which the warp kernel already did)."""

    def __init__(self, model: torch.nn.Module, fp16: bool = False) -> None:
        super().__init__()
        self.model = model
        # fp16 GEMMs (tensor cores): the MLP input is standardised O(1) and the
        # activations are tanh-bounded, so half evaluation perturbs the predicted
        # coefficients by ~1e-3 relative -- two orders below the model's own
        # residual. Standardisation and the basis_scale division stay fp32.
        self.fp16 = bool(fp16)
        if self.fp16:
            self.net_h = copy.deepcopy(model.net).half()

    def forward(self, X76: torch.Tensor) -> torch.Tensor:
        m = self.model
        x = (X76 - m.inv_mean) / m.inv_std
        if self.fp16:
            return self.net_h(x.half()).float() / m.basis_scale
        return m.net(x) / m.basis_scale


class _DiagScatterKernel(torch.nn.Module):
    """Accumulate one near-edge sub-chunk of particle self band moments into buf."""

    def forward(self, pos, t_c, s_c, buf):
        rel = pos[s_c] - pos[t_c]
        rn, base = _edge_base_values(rel)
        # Self bands: width 0.75 on [2, 8], centres 2.375..7.625, ends saturating.
        a0, f = _tent_bands(((rn - 2.0) / 0.75 - 0.5).clamp(0.0, 7.0))
        idx0 = t_c.to(torch.int64) * 8 + a0
        buf.index_add_(0, idx0, base * (1.0 - f).unsqueeze(1))
        buf.index_add_(0, idx0 + 1, base * f.unsqueeze(1))
        return buf


class _DiagFinishKernel(torch.nn.Module):
    """Self-moment rows -> symmetric 6x6 diagonal correction -> velocity / mu."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, buf_chunk, force_chunk, inv_mu):
        P = buf_chunk.shape[0]
        s_a, v_a, Q = _moments_from_buf(buf_chunk)
        X = torch.cat([s_a, v_a.reshape(P, 24), Q.reshape(P, 72)], 1)
        K = self.model.predict_mobility(X)
        return torch.einsum('pij,pj->pi', K, force_chunk) * inv_mu


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------
class Mob_Nbody_Moments_Torch(NNMobTorch):
    """GPU grand-mobility operator: self + two-body NN + moments pair correction
    (+ optional learned per-particle diagonal), all over the switch-distance
    neighbour list. The published pc8 models require switch_dist = 8."""

    def __init__(
        self,
        shape: str,
        self_nn_path: str,
        two_nn_path: str,
        moments_nn_path: str,
        near_field_2b: str = "nn",
        far_field_2b: Optional[str] = None,
        switch_dist: float = 8.0,
        neighbor_cutoff: float = 8.0,
        mean_dist_s: float = DEFAULT_MEAN_DIST_S,
        diag_nn_path: Optional[str] = None,
        two_body_chunk_size: int = DEFAULT_TWO_BODY_CHUNK,
        moments_pair_chunk: int = DEFAULT_MOMENTS_PAIR_CHUNK,
        edge_chunk: int = DEFAULT_EDGE_CHUNK,
        diag_row_chunk: int = DEFAULT_DIAG_ROW_CHUNK,
        compile_kernels: bool = True,
        moments_backend: str = "warp",
        moments_mlp_fp16: bool = True,
        mid_cell_scale: float = 0.5,
        fts_reflection: Optional[str] = None,
        fts_pair_chunk: int = 2_000_000,
    ) -> None:
        super().__init__(
            shape=shape,
            self_nn_path=self_nn_path,
            two_nn_path=two_nn_path,
            near_field=near_field_2b,
            far_field=far_field_2b,
            switch_dist=switch_dist,
            two_body_chunk_size=two_body_chunk_size,
        )
        assert shape == "sphere", "moments n-body operator is sphere-only"
        # The pair gate equals the near list (pair_cutoff == switch_dist): every
        # near pair is corrected, no RPY pair ever is. The diagonal labels subtract
        # K_s over d <= pair_cutoff while the operator adds K_s over d <= switch;
        # they must coincide or K_s is double-counted (see the CPU operator).
        self.neighbor_cutoff = float(neighbor_cutoff)
        self.mean_dist_s = float(mean_dist_s)
        self.moments_pair_chunk = int(moments_pair_chunk)
        self.edge_chunk = int(edge_chunk)
        self.diag_row_chunk = int(diag_row_chunk)
        # "warp": fused single-pass moment accumulation (production);
        # "torch": two-pass edge list + index_add scatter (reference/fallback).
        assert moments_backend in ("warp", "torch"), moments_backend
        self.moments_backend = moments_backend
        self.moments_mlp_fp16 = bool(moments_mlp_fp16)
        self.mid_cell_scale = float(
            os.environ.get("NEMO_MID_CELL_SCALE", mid_cell_scale))

        self.moments_nn = self._load_model(
            moments_nn_path, lambda: MultiBodyMoments(self.mean_dist_s))
        self.diag_nn = (
            self._load_model(diag_nn_path, SelfBlockMoments) if diag_nn_path else None)
        # Stresslet single reflection (src/fts_rpy.py) as a global analytic term, for models whose
        # labels subtract it (sidecar fts_base = "refl1"); the diagonal two-body path within the
        # switch distance is excluded because the 2b model's K_s already holds it.  Dense O(N^2)
        # torch passes in fp64 (chunked by target rows): fine for the harness sweeps (N <= 1e4),
        # the treecode version for production sizes is a separate step.
        self.fts_order = {None: 0, "none": 0, "refl1": 1, "refl2": 2}[fts_reflection]
        self.fts_diag_exclude = float(switch_dist)
        self.fts_pair_chunk = int(fts_pair_chunk)

        self._mid_search = MidpointNeighborSearch(self.device)
        self._pair_buf: Optional[torch.Tensor] = None
        self._diag_buf: Optional[torch.Tensor] = None
        self._flat_bufs: dict = {}   # name -> grow-only (rows, cols) fp32 buffers

        def _maybe_compile(mod: torch.nn.Module) -> torch.nn.Module:
            mod = mod.to(self.device)
            if not compile_kernels:
                return mod
            return torch.compile(mod, mode="max-autotune", backend="inductor",
                                 fullgraph=False, dynamic=True)

        self._pair_scatter_k = _maybe_compile(_PairScatterKernel())
        self._pair_finish_k = _maybe_compile(
            _PairFinishKernel(self.moments_nn, self.mean_dist_s))
        if self.moments_backend == "warp":
            # The warp invariants kernel implements the plain (extensive)
            # invariants; inv_norm models would need it taught the s_a division.
            assert not getattr(self.moments_nn, "inv_norm", False), \
                "warp backend does not implement inv_norm invariants"
            self._coeff_k = _maybe_compile(
                _CoeffKernel(self.moments_nn, fp16=self.moments_mlp_fp16))
        if self.diag_nn is not None:
            self._diag_scatter_k = _maybe_compile(_DiagScatterKernel())
            self._diag_finish_k = _maybe_compile(_DiagFinishKernel(self.diag_nn))

    # ------------------------------------------------------------------
    def _load_model(self, path: str, factory):
        """Load .wt weights into a fresh module (compilable end-to-end; preferred)
        or fall back to TorchScript .pt (runs with graph breaks)."""
        if path.endswith(".wt"):
            model = factory()
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        elif path.endswith(".pt"):
            model = torch.jit.load(path, map_location=self.device)
        else:
            raise ValueError(f"expected .wt or .pt model, got {path!r}")
        model = model.eval().to(self.device)
        assert hasattr(model, "predict_mobility")
        assert float(model.inv_std.min()) > 0 and float(model.basis_scale.min()) > 0
        return model

    def _get_flat(self, name: str, rows: int, cols: int) -> torch.Tensor:
        buf = self._flat_bufs.get(name)
        if buf is None or buf.shape[0] < rows or buf.shape[1] != cols:
            buf = torch.empty(rows, cols, dtype=torch.float32, device=self.device)
            self._flat_bufs[name] = buf
        return buf[:rows]

    def _get_buf(self, attr: str, rows: int, zero: bool = True) -> torch.Tensor:
        buf = getattr(self, attr)
        if buf is None or buf.shape[0] < rows:
            buf = torch.empty(rows, 10, dtype=torch.float32, device=self.device)
            setattr(self, attr, buf)
        out = buf[:rows]
        if zero:
            out.zero_()
        return out

    # ------------------------------------------------------------------
    # Pair moments correction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_moments_velocity(
        self,
        pos: torch.Tensor,
        force: torch.Tensor,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Learned pair n-body correction summed per particle, (N, 6)."""
        v = torch.zeros_like(force)
        keep = t_idx < s_idx                     # unordered pairs, once each
        t_u = t_idx[keep].contiguous()
        s_u = s_idx[keep].contiguous()
        P = int(t_u.shape[0])
        if P == 0:
            return v
        self._mid_search.build(pos, self.neighbor_cutoff, self.mid_cell_scale)
        n_edges = 0
        for start in range(0, P, self.moments_pair_chunk):
            end = min(start + self.moments_pair_chunk, P)
            t_c, s_c = t_u[start:end], s_u[start:end]
            mid = 0.5 * (pos[t_c] + pos[s_c])
            Pc = end - start
            if self.moments_backend == "warp":
                # Accumulate -> invariants -> MLP -> assemble+apply: only the
                # MLP runs in torch; the tensor algebra around it lives in the
                # two warp kernels (registers, one read/write per pair).
                mom = self._get_buf("_pair_buf", Pc * 8, zero=False).view(Pc, 80)
                self._mid_search.accumulate(
                    pos, mid, t_c, s_c, self.neighbor_cutoff, mom)
                X76 = self._get_flat("X76", Pc, 76)
                self._mid_search.launch_pair_finish(
                    pos, t_c, s_c, mom, pair_invariants_kernel,
                    [float(self.mean_dist_s)], [X76])
                torch._dynamo.mark_dynamic(X76, 0)
                c = self._coeff_k(X76).contiguous()
                v_t = self._get_flat("v_t", Pc, 6)
                v_s = self._get_flat("v_s", Pc, 6)
                self._mid_search.launch_pair_finish(
                    pos, t_c, s_c, mom, pair_assemble_apply_kernel,
                    [c, force.contiguous()], [v_t, v_s])
            else:
                edge_pair, edge_nbr = self._mid_search.query(
                    pos, mid, t_c, s_c, self.neighbor_cutoff)
                E = int(edge_pair.shape[0])
                n_edges += E
                buf = self._get_buf("_pair_buf", Pc * 8)
                for e0 in range(0, E, self.edge_chunk):
                    e1 = min(e0 + self.edge_chunk, E)
                    ep, en = edge_pair[e0:e1], edge_nbr[e0:e1]
                    torch._dynamo.mark_dynamic(ep, 0)
                    torch._dynamo.mark_dynamic(en, 0)
                    torch._dynamo.mark_dynamic(buf, 0)
                    self._pair_scatter_k(pos, t_c, s_c, ep, en, buf)
                torch._dynamo.mark_dynamic(t_c, 0)
                torch._dynamo.mark_dynamic(s_c, 0)
                torch._dynamo.mark_dynamic(buf, 0)
                v_t, v_s = self._pair_finish_k(pos, force, t_c, s_c, buf)
            v.index_add_(0, t_c, v_t)
            v.index_add_(0, s_c, v_s)
        self.last_moment_edges = n_edges
        self.last_moment_pairs = P
        return v

    # ------------------------------------------------------------------
    # Learned per-particle diagonal correction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_diag_velocity(
        self,
        pos: torch.Tensor,
        force: torch.Tensor,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
        viscosity: TensorLike,
    ) -> torch.Tensor:
        """v_t = K_diag(t) F_t / mu from the ordered near edge list (== all k != t
        within switch_dist, which the pc8 diagonal requires to equal diag_cutoff)."""
        assert self.diag_nn is not None
        N = pos.shape[0]
        mu = torch.as_tensor(viscosity, dtype=torch.float32, device=self.device)
        if mu.ndim >= 1 and mu.numel() > 1:
            mu = mu[0]
        inv_mu = 1.0 / mu
        buf = self._get_buf("_diag_buf", N * 8)
        E = int(t_idx.shape[0])
        for e0 in range(0, E, self.edge_chunk):
            e1 = min(e0 + self.edge_chunk, E)
            t_c, s_c = t_idx[e0:e1], s_idx[e0:e1]
            torch._dynamo.mark_dynamic(t_c, 0)
            torch._dynamo.mark_dynamic(s_c, 0)
            self._diag_scatter_k(pos, t_c, s_c, buf)
        v = torch.empty_like(force)
        buf3 = buf.view(N, 8, 10)
        for p0 in range(0, N, self.diag_row_chunk):
            p1 = min(p0 + self.diag_row_chunk, N)
            chunk = buf3[p0:p1]
            f_chunk = force[p0:p1]
            torch._dynamo.mark_dynamic(chunk, 0)
            torch._dynamo.mark_dynamic(f_chunk, 0)
            v[p0:p1] = self._diag_finish_k(chunk, f_chunk, inv_mu)
        return v

    # ------------------------------------------------------------------
    @torch.no_grad()
    def apply(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        force: torch.Tensor,
        viscosity: TensorLike,
        t_idx: torch.Tensor | None = None,
        s_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = orientations  # spheres
        assert positions.is_cuda and force.is_cuda
        pos = positions.contiguous()
        if t_idx is None or s_idx is None:
            t_idx, s_idx = self.get_neighbor_pairs(pos)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        v = super().apply(pos, orientations, force, viscosity,
                          t_idx=t_idx, s_idx=s_idx)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"[Mob_Nbody] Base velocity compute time: {(t1 - t0) * 1000:.3f} ms")

        if t_idx.numel() == 0:
            return v

        force_t = torch.as_tensor(force, dtype=torch.float32, device=self.device)
        t2 = time.perf_counter()
        v = v + self.get_moments_velocity(pos, force_t, t_idx, s_idx)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        edges = (f", {self.last_moment_edges} edges"
                 if self.moments_backend == "torch" else "")
        print(f"[MobMoments] pair moments time: {(t3 - t2) * 1000:.3f} ms "
              f"({self.last_moment_pairs} pairs{edges})")

        if self.diag_nn is not None:
            v = v + self.get_diag_velocity(pos, force_t, t_idx, s_idx, viscosity)
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            print(f"[MobMoments] diag correction time: {(t4 - t3) * 1000:.3f} ms")

        if self.fts_order:
            v = v + self.get_fts_velocity(pos, force, viscosity)
            torch.cuda.synchronize()
            t45 = time.perf_counter()
            print(f"[MobMoments] FTS reflection time: {(t45 - t3) * 1000:.3f} ms")

        t5 = time.perf_counter()
        print(f"[Mob_Nbody] Post-base path time: {(t5 - t2) * 1000:.3f} ms")
        return v

    @torch.no_grad()
    def get_fts_velocity(self, pos: torch.Tensor, force: torch.Tensor, viscosity: TensorLike) -> torch.Tensor:
        """fts_rpy.reflection_velocity on the device (fp64), returned in the velocity dtype."""
        from src import fts_rpy
        mu = float(viscosity.item() if torch.is_tensor(viscosity) else viscosity)
        v = fts_rpy.reflection_velocity(pos.to(torch.float64), force.to(torch.float64), mu=mu,
                                        order=str(self.fts_order), pair_chunk=self.fts_pair_chunk,
                                        diag_exclude_within=self.fts_diag_exclude)
        return v.to(torch.float32)
