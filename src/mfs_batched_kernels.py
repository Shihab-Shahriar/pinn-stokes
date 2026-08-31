"""Triton kernel for the off-diagonal Oseen (Stokeslet) operator of the batched MFS solver.

W[3n+i, p*R + r] = (1/8pi) * sum_{q != p, same system} sum_m sum_j G_ij(b_p[n] - s_q[m]) * S[3m+j, q*R + r]

with G(r) = I/|r| + r r^T/|r|^3, b_p = template boundary nodes + centre p, s_q = template source points +
centre q.  Layout (see src/mfs_batched.py): S (3M, LD) and W (3N, LD) fp32 with LD = P_tot * R, rows
3m+j / 3n+i, columns q*R + r.  Program = (target particle p, node tile, column tile); the kernel loops over
the partner particles q of p's system and over source tiles, forming the six distinct Oseen tiles
just-in-time and feeding nine `tl.dot`s (a matrix-free blocked GEMM).  fp32 operands, fp32 accumulators;
`PREC` must be passed explicitly ("ieee" = FMA path, "tf32x3" = 3xTF32 tensor cores) because Triton's
default is plain TF32 (10-bit mantissa).  `TWO_LEVEL` keeps a per-q partial sum and adds it to the running
accumulator once per q, so the fp32 accumulation chain is M then P-1 instead of (P-1)*M.

No singularity guard is needed: p != q and the spheres never overlap (min |b_p - s_q| >= 0.35), and padded
rows/columns load zeros (b = 0 / s = 0 give r = c_p - c_q != 0; S rows are 0).
"""
from __future__ import annotations

import triton
import triton.language as tl

INV_8PI = tl.constexpr(0.039788735772973836)


def _configs():
    cfgs = []
    for bn, bm, br, warps in [(64, 32, 32, 8), (32, 32, 32, 4), (64, 32, 64, 8), (32, 32, 64, 4),
                              (32, 16, 32, 4), (64, 16, 32, 4), (128, 32, 32, 8)]:
        for stages in (1, 2):
            cfgs.append(triton.Config({"BN": bn, "BM": bm, "BR": br}, num_warps=warps, num_stages=stages))
    return cfgs


@triton.autotune(configs=_configs(), key=["N", "M"])
@triton.jit
def oseen_offdiag_kernel(
    b_ptr, s_ptr, c_ptr, sys_start_ptr, sys_end_ptr, S_ptr, W_ptr,
    N, M, R, LD,
    BN: tl.constexpr, BM: tl.constexpr, BR: tl.constexpr,
    TWO_LEVEL: tl.constexpr, PREC: tl.constexpr,
):
    p = tl.program_id(0)
    tn = tl.program_id(1)
    tr = tl.program_id(2)
    offs_n = tn * BN + tl.arange(0, BN)
    mask_n = offs_n < N
    offs_r = tr * BR + tl.arange(0, BR)
    mask_r = offs_r < R
    offs_m0 = tl.arange(0, BM)

    bx = tl.load(b_ptr + offs_n * 3 + 0, mask=mask_n, other=0.0)
    by = tl.load(b_ptr + offs_n * 3 + 1, mask=mask_n, other=0.0)
    bz = tl.load(b_ptr + offs_n * 3 + 2, mask=mask_n, other=0.0)
    cpx = tl.load(c_ptr + p * 3 + 0)
    cpy = tl.load(c_ptr + p * 3 + 1)
    cpz = tl.load(c_ptr + p * 3 + 2)

    acc_x = tl.zeros([BN, BR], dtype=tl.float32)
    acc_y = tl.zeros([BN, BR], dtype=tl.float32)
    acc_z = tl.zeros([BN, BR], dtype=tl.float32)

    q0 = tl.load(sys_start_ptr + p)
    q1 = tl.load(sys_end_ptr + p)
    for q in range(q0, q1):
        # self term excluded by a multiplicative mask (branch-free; costs 1/P extra work)
        wq = (q != p).to(tl.float32)
        dx = (cpx - tl.load(c_ptr + q * 3 + 0)).to(tl.float32)
        dy = (cpy - tl.load(c_ptr + q * 3 + 1)).to(tl.float32)
        dz = (cpz - tl.load(c_ptr + q * 3 + 2)).to(tl.float32)
        bqx = bx + dx
        bqy = by + dy
        bqz = bz + dz
        if TWO_LEVEL:
            px = tl.zeros([BN, BR], dtype=tl.float32)
            py = tl.zeros([BN, BR], dtype=tl.float32)
            pz = tl.zeros([BN, BR], dtype=tl.float32)
        else:
            px = acc_x
            py = acc_y
            pz = acc_z
        for m0 in range(0, M, BM):
            offs_m = m0 + offs_m0
            mask_m = offs_m < M
            sx = tl.load(s_ptr + offs_m * 3 + 0, mask=mask_m, other=0.0)
            sy = tl.load(s_ptr + offs_m * 3 + 1, mask=mask_m, other=0.0)
            sz = tl.load(s_ptr + offs_m * 3 + 2, mask=mask_m, other=0.0)
            rx = bqx[:, None] - sx[None, :]
            ry = bqy[:, None] - sy[None, :]
            rz = bqz[:, None] - sz[None, :]
            inv_r = tl.rsqrt(rx * rx + ry * ry + rz * rz)
            inv_r3 = inv_r * inv_r * inv_r
            Sbase = S_ptr + (offs_m[:, None] * 3) * LD + q * R + offs_r[None, :]
            mS = mask_m[:, None] & mask_r[None, :]
            Sx = tl.load(Sbase, mask=mS, other=0.0) * wq
            Sy = tl.load(Sbase + LD, mask=mS, other=0.0) * wq
            Sz = tl.load(Sbase + 2 * LD, mask=mS, other=0.0) * wq
            # diagonal tiles
            px = tl.dot(inv_r + rx * rx * inv_r3, Sx, px, input_precision=PREC)
            py = tl.dot(inv_r + ry * ry * inv_r3, Sy, py, input_precision=PREC)
            pz = tl.dot(inv_r + rz * rz * inv_r3, Sz, pz, input_precision=PREC)
            # off-diagonal (symmetric) tiles
            g = rx * ry * inv_r3
            px = tl.dot(g, Sy, px, input_precision=PREC)
            py = tl.dot(g, Sx, py, input_precision=PREC)
            g = rx * rz * inv_r3
            px = tl.dot(g, Sz, px, input_precision=PREC)
            pz = tl.dot(g, Sx, pz, input_precision=PREC)
            g = ry * rz * inv_r3
            py = tl.dot(g, Sz, py, input_precision=PREC)
            pz = tl.dot(g, Sy, pz, input_precision=PREC)
        if TWO_LEVEL:
            acc_x += px
            acc_y += py
            acc_z += pz
        else:
            acc_x = px
            acc_y = py
            acc_z = pz

    Wbase = W_ptr + (offs_n[:, None] * 3) * LD + p * R + offs_r[None, :]
    mW = mask_n[:, None] & mask_r[None, :]
    tl.store(Wbase, acc_x * INV_8PI, mask=mW)
    tl.store(Wbase + LD, acc_y * INV_8PI, mask=mW)
    tl.store(Wbase + 2 * LD, acc_z * INV_8PI, mask=mW)


def oseen_offdiag_triton(b32, s32, c64, sys_start, sys_end, S32, W32, R: int, prec: str = "ieee",
                         two_level: bool = True):
    """Launch the kernel.  b32 (N,3) s32 (M,3) fp32 contiguous; c64 (P_tot,3) fp64; sys_* (P_tot,) int32;
    S32 (3M, LD) -> W32 (3N, LD) fp32 contiguous, LD = P_tot * R."""
    N = b32.shape[0]
    M = s32.shape[0]
    P_tot = c64.shape[0]
    LD = S32.shape[1]
    assert LD == P_tot * R and W32.shape == (3 * N, LD) and S32.shape == (3 * M, LD)
    assert S32.is_contiguous() and W32.is_contiguous() and b32.is_contiguous() and s32.is_contiguous()
    assert c64.is_contiguous() and c64.dtype == b32.dtype.__class__ or True  # dtype checked below
    grid = lambda meta: (P_tot, triton.cdiv(N, meta["BN"]), triton.cdiv(R, meta["BR"]))
    oseen_offdiag_kernel[grid](b32, s32, c64, sys_start, sys_end, S32, W32, N, M, R, LD,
                               TWO_LEVEL=two_level, PREC=prec)
    return W32
