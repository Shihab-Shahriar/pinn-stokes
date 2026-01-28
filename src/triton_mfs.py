import math

import numpy as np
import torch
import triton
import triton.language as tl

from src.mfs_utils import build_B

_TRITON_AVAILABLE = True
INV_8PI = 1.0 / (8.0 * math.pi)


@triton.jit
def _oseen_sum_qblock_kernel(
    bp_ptr,
    sq_ptr,
    f_ptr,
    centers_ptr,
    stokeslet_ptr,
    near_ptr,
    out_ptr,
    N,
    M,
    P,
    p_idx,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    bp_base = bp_ptr + p_idx * (N * 3)
    bp_ptrs = bp_base + offs_n * 3
    bp_x = tl.load(bp_ptrs + 0, mask=mask_n, other=0.0)
    bp_y = tl.load(bp_ptrs + 1, mask=mask_n, other=0.0)
    bp_z = tl.load(bp_ptrs + 2, mask=mask_n, other=0.0)

    v_x = tl.zeros([BLOCK_N], dtype=tl.float64)
    v_y = tl.zeros([BLOCK_N], dtype=tl.float64)
    v_z = tl.zeros([BLOCK_N], dtype=tl.float64)
    eps = 1.0e-20

    q_base = pid_q * BLOCK_Q
    for q_off in tl.static_range(0, BLOCK_Q):
        q = q_base + q_off
        q_valid = q < P
        not_self = q != p_idx
        active = q_valid & not_self
        active_f = active.to(tl.float64)

        near = tl.load(near_ptr + p_idx * P + q, mask=q_valid, other=0).to(tl.float64)
        near_f = near * active_f
        far_f = active_f - near_f

        center_base = centers_ptr + q * 3
        center_q_x = tl.load(center_base + 0, mask=q_valid, other=0.0)
        center_q_y = tl.load(center_base + 1, mask=q_valid, other=0.0)
        center_q_z = tl.load(center_base + 2, mask=q_valid, other=0.0)

        stokes_base = stokeslet_ptr + q * 3
        fnet_x = tl.load(stokes_base + 0, mask=q_valid, other=0.0)
        fnet_y = tl.load(stokes_base + 1, mask=q_valid, other=0.0)
        fnet_z = tl.load(stokes_base + 2, mask=q_valid, other=0.0)

        r_pf_x = bp_x - center_q_x
        r_pf_y = bp_y - center_q_y
        r_pf_z = bp_z - center_q_z
        r2_pf = r_pf_x * r_pf_x + r_pf_y * r_pf_y + r_pf_z * r_pf_z
        valid_pf = mask_n & (r2_pf > eps)
        inv_r_pf = tl.where(valid_pf, 1.0 / tl.sqrt(r2_pf), 0.0)
        inv_r3_pf = inv_r_pf * inv_r_pf * inv_r_pf
        dot_pf = r_pf_x * fnet_x + r_pf_y * fnet_y + r_pf_z * fnet_z
        v_x += (fnet_x * inv_r_pf + r_pf_x * dot_pf * inv_r3_pf) * far_f
        v_y += (fnet_y * inv_r_pf + r_pf_y * dot_pf * inv_r3_pf) * far_f
        v_z += (fnet_z * inv_r_pf + r_pf_z * dot_pf * inv_r3_pf) * far_f

        f_stride0 = 3 * M + 6
        for m in range(0, M, BLOCK_M):
            offs_m = m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < M

            sq_base = sq_ptr + q * (M * 3) + offs_m * 3
            sq_x = tl.load(sq_base + 0, mask=mask_m & q_valid, other=0.0)
            sq_y = tl.load(sq_base + 1, mask=mask_m & q_valid, other=0.0)
            sq_z = tl.load(sq_base + 2, mask=mask_m & q_valid, other=0.0)

            f_base = f_ptr + q * f_stride0 + offs_m * 3
            f_x = tl.load(f_base + 0, mask=mask_m & q_valid, other=0.0) * near_f
            f_y = tl.load(f_base + 1, mask=mask_m & q_valid, other=0.0) * near_f
            f_z = tl.load(f_base + 2, mask=mask_m & q_valid, other=0.0) * near_f

            r_nm_x = bp_x[:, None] - sq_x[None, :]
            r_nm_y = bp_y[:, None] - sq_y[None, :]
            r_nm_z = bp_z[:, None] - sq_z[None, :]
            r2 = r_nm_x * r_nm_x + r_nm_y * r_nm_y + r_nm_z * r_nm_z

            mask_nm = mask_n[:, None] & mask_m[None, :]
            valid_mask = mask_nm & (r2 > eps)
            r2_safe = tl.where(valid_mask, r2, 1.0)
            inv_r = 1.0 / tl.sqrt(r2_safe)
            inv_r = tl.where(valid_mask, inv_r, 0.0)
            inv_r3 = inv_r * inv_r * inv_r

            dot = r_nm_x * f_x[None, :] + r_nm_y * f_y[None, :] + r_nm_z * f_z[None, :]
            term_x = f_x[None, :] * inv_r + r_nm_x * dot * inv_r3
            term_y = f_y[None, :] * inv_r + r_nm_y * dot * inv_r3
            term_z = f_z[None, :] * inv_r + r_nm_z * dot * inv_r3

            term_x = tl.where(valid_mask, term_x, 0.0)
            term_y = tl.where(valid_mask, term_y, 0.0)
            term_z = tl.where(valid_mask, term_z, 0.0)

            v_x += tl.sum(term_x, axis=1)
            v_y += tl.sum(term_y, axis=1)
            v_z += tl.sum(term_z, axis=1)

    factor = 0.039788735772973836
    v_x = v_x * factor
    v_y = v_y * factor
    v_z = v_z * factor

    out_base = out_ptr + pid_q * (N * 3)
    out_ptrs = out_base + offs_n * 3
    tl.store(out_ptrs + 0, v_x, mask=mask_n)
    tl.store(out_ptrs + 1, v_y, mask=mask_n)
    tl.store(out_ptrs + 2, v_z, mask=mask_n)

@triton.jit
def _oseen_sum_qblock_kernel_sphere(
    b_single_ptr,
    s_single_ptr,
    f_ptr,
    centers_ptr,
    stokeslet_ptr,
    near_ptr,
    out_ptr,
    N,
    M,
    P,
    p_idx,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    b_base = b_single_ptr + offs_n * 3
    b_x = tl.load(b_base + 0, mask=mask_n, other=0.0)
    b_y = tl.load(b_base + 1, mask=mask_n, other=0.0)
    b_z = tl.load(b_base + 2, mask=mask_n, other=0.0)

    center_p_base = centers_ptr + p_idx * 3
    center_p_x = tl.load(center_p_base + 0)
    center_p_y = tl.load(center_p_base + 1)
    center_p_z = tl.load(center_p_base + 2)

    bp_x = b_x + center_p_x
    bp_y = b_y + center_p_y
    bp_z = b_z + center_p_z

    v_x = tl.zeros([BLOCK_N], dtype=tl.float64)
    v_y = tl.zeros([BLOCK_N], dtype=tl.float64)
    v_z = tl.zeros([BLOCK_N], dtype=tl.float64)
    eps = 1.0e-20

    q_base = pid_q * BLOCK_Q
    for q_off in tl.static_range(0, BLOCK_Q):
        q = q_base + q_off
        q_valid = q < P
        not_self = q != p_idx
        active = q_valid & not_self
        active_f = active.to(tl.float64)

        near = tl.load(near_ptr + p_idx * P + q, mask=q_valid, other=0).to(tl.float64)
        near_f = near * active_f
        far_f = active_f - near_f

        center_q_base = centers_ptr + q * 3
        center_q_x = tl.load(center_q_base + 0, mask=q_valid, other=0.0)
        center_q_y = tl.load(center_q_base + 1, mask=q_valid, other=0.0)
        center_q_z = tl.load(center_q_base + 2, mask=q_valid, other=0.0)

        stokes_base = stokeslet_ptr + q * 3
        fnet_x = tl.load(stokes_base + 0, mask=q_valid, other=0.0)
        fnet_y = tl.load(stokes_base + 1, mask=q_valid, other=0.0)
        fnet_z = tl.load(stokes_base + 2, mask=q_valid, other=0.0)

        r_pf_x = bp_x - center_q_x
        r_pf_y = bp_y - center_q_y
        r_pf_z = bp_z - center_q_z
        r2_pf = r_pf_x * r_pf_x + r_pf_y * r_pf_y + r_pf_z * r_pf_z
        valid_pf = mask_n & (r2_pf > eps)
        inv_r_pf = tl.where(valid_pf, 1.0 / tl.sqrt(r2_pf), 0.0)
        inv_r3_pf = inv_r_pf * inv_r_pf * inv_r_pf
        dot_pf = r_pf_x * fnet_x + r_pf_y * fnet_y + r_pf_z * fnet_z
        v_x += (fnet_x * inv_r_pf + r_pf_x * dot_pf * inv_r3_pf) * far_f
        v_y += (fnet_y * inv_r_pf + r_pf_y * dot_pf * inv_r3_pf) * far_f
        v_z += (fnet_z * inv_r_pf + r_pf_z * dot_pf * inv_r3_pf) * far_f

        f_stride0 = 3 * M + 6
        for m in range(0, M, BLOCK_M):
            offs_m = m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < M

            s_base = s_single_ptr + offs_m * 3
            s_x = tl.load(s_base + 0, mask=mask_m & q_valid, other=0.0) + center_q_x
            s_y = tl.load(s_base + 1, mask=mask_m & q_valid, other=0.0) + center_q_y
            s_z = tl.load(s_base + 2, mask=mask_m & q_valid, other=0.0) + center_q_z

            f_base = f_ptr + q * f_stride0 + offs_m * 3
            f_x = tl.load(f_base + 0, mask=mask_m & q_valid, other=0.0) * near_f
            f_y = tl.load(f_base + 1, mask=mask_m & q_valid, other=0.0) * near_f
            f_z = tl.load(f_base + 2, mask=mask_m & q_valid, other=0.0) * near_f

            r_nm_x = bp_x[:, None] - s_x[None, :]
            r_nm_y = bp_y[:, None] - s_y[None, :]
            r_nm_z = bp_z[:, None] - s_z[None, :]
            r2 = r_nm_x * r_nm_x + r_nm_y * r_nm_y + r_nm_z * r_nm_z

            mask_nm = mask_n[:, None] & mask_m[None, :]
            valid_mask = mask_nm & (r2 > eps)
            r2_safe = tl.where(valid_mask, r2, 1.0)
            inv_r = 1.0 / tl.sqrt(r2_safe)
            inv_r = tl.where(valid_mask, inv_r, 0.0)
            inv_r3 = inv_r * inv_r * inv_r

            dot = r_nm_x * f_x[None, :] + r_nm_y * f_y[None, :] + r_nm_z * f_z[None, :]
            term_x = f_x[None, :] * inv_r + r_nm_x * dot * inv_r3
            term_y = f_y[None, :] * inv_r + r_nm_y * dot * inv_r3
            term_z = f_z[None, :] * inv_r + r_nm_z * dot * inv_r3

            term_x = tl.where(valid_mask, term_x, 0.0)
            term_y = tl.where(valid_mask, term_y, 0.0)
            term_z = tl.where(valid_mask, term_z, 0.0)

            v_x += tl.sum(term_x, axis=1)
            v_y += tl.sum(term_y, axis=1)
            v_z += tl.sum(term_z, axis=1)

    factor = 0.039788735772973836
    v_x = v_x * factor
    v_y = v_y * factor
    v_z = v_z * factor

    out_base = out_ptr + pid_q * (N * 3)
    out_ptrs = out_base + offs_n * 3
    tl.store(out_ptrs + 0, v_x, mask=mask_n)
    tl.store(out_ptrs + 1, v_y, mask=mask_n)
    tl.store(out_ptrs + 2, v_z, mask=mask_n)

def _point_force_velocity_torch(bp, center_q, f_net):
    rC = bp - center_q
    rr = torch.linalg.norm(rC, dim=1)
    mask = rr < 1e-10
    inv_r = torch.where(mask, torch.zeros_like(rr), 1.0 / rr)
    inv_r3 = inv_r * inv_r * inv_r
    dotfr = torch.sum(rC * f_net, dim=1)
    v = inv_r[:, None] * f_net + (dotfr * inv_r3)[:, None] * rC
    v = v * INV_8PI
    v = torch.where(mask[:, None], torch.zeros_like(v), v)
    return v





def imp_mfs_mobility_vec_triton(
    b_list,
    s_list,
    F_ext_list,
    T_ext_list,
    B_inv_list,
    max_iter=1000,
    tol=1e-7,
    print_steps=False,
    center_list=None,
    L_cut=25.0,
    device=None,
    block_n=64,
    block_m=64,
    block_q=4,
    num_warps=4,
):
    """
    Triton-accelerated IMP-MFS mobility solver.
    Uses a cutoff (default 25.0) with point-force approximation for far pairs.
    Returns a list of torch tensors on the requested device.
    """
    device = torch.device("cuda" if device is None else device)
    dtype = torch.float64
    P = len(b_list)
    N = b_list[0].shape[0]
    M = s_list[0].shape[0]
    use_cutoff = L_cut is not None

    b_t = torch.stack(
        [torch.as_tensor(b, device=device, dtype=dtype) for b in b_list],
        dim=0,
    ).contiguous()
    s_t = torch.stack(
        [torch.as_tensor(s, device=device, dtype=dtype) for s in s_list],
        dim=0,
    ).contiguous()
    B_inv_t = torch.stack(
        [torch.as_tensor(B_inv, device=device, dtype=dtype) for B_inv in B_inv_list],
        dim=0,
    ).contiguous()
    F_ext_t = torch.stack(
        [torch.as_tensor(F_ext, device=device, dtype=dtype) for F_ext in F_ext_list],
        dim=0,
    ).contiguous()
    T_ext_t = torch.stack(
        [torch.as_tensor(T_ext, device=device, dtype=dtype) for T_ext in T_ext_list],
        dim=0,
    ).contiguous()

    if center_list is None:
        centers_t = b_t.mean(dim=1)
    else:
        centers_t = torch.as_tensor(center_list, device=device, dtype=dtype)
        if centers_t.shape != (P, 3):
            raise ValueError("center_list must have shape (P, 3)")

    if use_cutoff:
        delta = centers_t[:, None, :] - centers_t[None, :, :]
        dist2 = (delta * delta).sum(dim=-1)
        near_mask = dist2 < (float(L_cut) * float(L_cut))
    else:
        near_mask = torch.ones((P, P), device=device, dtype=torch.bool)
    eye = torch.eye(P, device=device, dtype=torch.bool)
    near_mask = (near_mask & ~eye).to(torch.int8).contiguous()

    x = torch.zeros((P, 3 * M + 6), device=device, dtype=dtype)
    x_prev = torch.empty_like(x)
    stokeslet_sum = torch.zeros((P, 3), device=device, dtype=dtype)

    F_tilde = torch.zeros((P, 3 * N + 6), device=device, dtype=dtype)
    F_tilde[:, 3 * N:3 * N + 3] = F_ext_t
    F_tilde[:, 3 * N + 3:3 * N + 6] = T_ext_t

    v_buf = torch.zeros((N, 3), device=device, dtype=dtype)
    q_blocks = triton.cdiv(P, block_q)
    v_partial = torch.empty((q_blocks, N, 3), device=device, dtype=dtype)
    w_buf = torch.zeros(3 * N + 6, device=device, dtype=dtype)
    rhs_buf = torch.empty(3 * N + 6, device=device, dtype=dtype)

    grid = (triton.cdiv(N, block_n), q_blocks)

    if print_steps:
        print(
            f"imp_mfs_mobility_vec_triton: start P={P}, N={N}, M={M}, use_cutoff={use_cutoff}",
            flush=True,
        )

    for iteration in range(max_iter):
        if print_steps:
            print(f"Iteration {iteration+1}: start", flush=True)
        x_prev.copy_(x)
        f_t = x[:, :3 * M].reshape(P, M, 3)

        for p in range(P):
            _oseen_sum_qblock_kernel[grid](
                b_t,
                s_t,
                f_t,
                centers_t,
                stokeslet_sum,
                near_mask,
                v_partial,
                N,
                M,
                P,
                p,
                BLOCK_N=block_n,
                BLOCK_M=block_m,
                BLOCK_Q=block_q,
                num_warps=num_warps,
            )
            torch.sum(v_partial, dim=0, out=v_buf)

            w_buf.zero_()
            w_buf[:3 * N].copy_(v_buf.reshape(3 * N))
            rhs_buf.copy_(F_tilde[p])
            rhs_buf.sub_(w_buf)
            torch.mv(B_inv_t[p], rhs_buf, out=x[p])

            if use_cutoff:
                torch.sum(x[p, :3 * M].reshape(M, 3), dim=0, out=stokeslet_sum[p])


        max_diff = torch.linalg.norm(x - x_prev, dim=1).max().item()
        if max_diff < tol:
            print(f"Converged after {iteration+1} iterations (max diff = {max_diff:e})")
            break
        if print_steps:
            print(f"Iteration {iteration+1}: max diff = {max_diff:e}", flush=True)
    else:
        raise RuntimeError("Solver did not converge")

    return [x[p] for p in range(P)]

def imp_mfs_mobility_sphere_triton(
    b_single,
    s_single,
    centers,
    F_ext_list,
    T_ext_list,
    B_inv,
    max_iter=1000,
    tol=1e-7,
    print_steps=False,
    L_cut=25.0,
    device=None,
    block_n=64,
    block_m=64,
    block_q=4,
    num_warps=4,
):
    """
    Sphere-optimized Triton IMP-MFS solver using shared templates and B_inv.
    """
    device = torch.device("cuda" if device is None else device)
    dtype = torch.float64

    if torch.is_tensor(b_single):
        b_single_t = b_single.to(device=device, dtype=dtype) if (b_single.device != device or b_single.dtype != dtype) else b_single
    else:
        b_single_t = torch.as_tensor(b_single, device=device, dtype=dtype)
    b_single_t = b_single_t.contiguous()

    if torch.is_tensor(s_single):
        s_single_t = s_single.to(device=device, dtype=dtype) if (s_single.device != device or s_single.dtype != dtype) else s_single
    else:
        s_single_t = torch.as_tensor(s_single, device=device, dtype=dtype)
    s_single_t = s_single_t.contiguous()

    if torch.is_tensor(centers):
        centers_t = centers.to(device=device, dtype=dtype) if (centers.device != device or centers.dtype != dtype) else centers
    else:
        centers_t = torch.as_tensor(centers, device=device, dtype=dtype)
    centers_t = centers_t.contiguous()

    if torch.is_tensor(B_inv):
        B_inv_t = B_inv.to(device=device, dtype=dtype) if (B_inv.device != device or B_inv.dtype != dtype) else B_inv
    else:
        B_inv_t = torch.as_tensor(B_inv, device=device, dtype=dtype)
    B_inv_t = B_inv_t.contiguous()

    if torch.is_tensor(F_ext_list):
        F_ext_t = F_ext_list.to(device=device, dtype=dtype)
    else:
        F_ext_t = torch.as_tensor(np.asarray(F_ext_list, dtype=np.float64), device=device, dtype=dtype)
    if torch.is_tensor(T_ext_list):
        T_ext_t = T_ext_list.to(device=device, dtype=dtype)
    else:
        T_ext_t = torch.as_tensor(np.asarray(T_ext_list, dtype=np.float64), device=device, dtype=dtype)
    F_ext_t = F_ext_t.contiguous()
    T_ext_t = T_ext_t.contiguous()

    if centers_t.ndim != 2 or centers_t.shape[1] != 3:
        raise ValueError("centers must have shape (P, 3)")
    if b_single_t.ndim != 2 or b_single_t.shape[1] != 3:
        raise ValueError("b_single must have shape (N, 3)")
    if s_single_t.ndim != 2 or s_single_t.shape[1] != 3:
        raise ValueError("s_single must have shape (M, 3)")

    P = centers_t.shape[0]
    N = b_single_t.shape[0]
    M = s_single_t.shape[0]

    if F_ext_t.shape != (P, 3):
        raise ValueError("F_ext_list must have shape (P, 3)")
    if T_ext_t.shape != (P, 3):
        raise ValueError("T_ext_list must have shape (P, 3)")
    if B_inv_t.shape != (3 * M + 6, 3 * N + 6):
        raise ValueError("B_inv must have shape (3M+6, 3N+6)")

    use_cutoff = L_cut is not None
    if use_cutoff:
        delta = centers_t[:, None, :] - centers_t[None, :, :]
        dist2 = (delta * delta).sum(dim=-1)
        near_mask = dist2 < (float(L_cut) * float(L_cut))
    else:
        near_mask = torch.ones((P, P), device=device, dtype=torch.bool)
    eye = torch.eye(P, device=device, dtype=torch.bool)
    near_mask = (near_mask & ~eye).to(torch.int8).contiguous()

    x = torch.zeros((P, 3 * M + 6), device=device, dtype=dtype)
    x_prev = torch.empty_like(x)
    stokeslet_sum = torch.zeros((P, 3), device=device, dtype=dtype)

    F_tilde = torch.zeros((P, 3 * N + 6), device=device, dtype=dtype)
    F_tilde[:, 3 * N:3 * N + 3] = F_ext_t
    F_tilde[:, 3 * N + 3:3 * N + 6] = T_ext_t

    v_buf = torch.zeros((N, 3), device=device, dtype=dtype)
    q_blocks = triton.cdiv(P, block_q)
    v_partial = torch.empty((q_blocks, N, 3), device=device, dtype=dtype)
    w_buf = torch.zeros(3 * N + 6, device=device, dtype=dtype)
    rhs_buf = torch.empty(3 * N + 6, device=device, dtype=dtype)

    grid = (triton.cdiv(N, block_n), q_blocks)

    if print_steps:
        print(
            f"imp_mfs_mobility_sphere_triton: start P={P}, N={N}, M={M}, use_cutoff={use_cutoff}",
            flush=True,
        )

    for iteration in range(max_iter):
        if print_steps:
            print(f"Iteration {iteration+1}: start", flush=True)
        x_prev.copy_(x)
        f_t = x[:, :3 * M].reshape(P, M, 3)

        for p in range(P):
            _oseen_sum_qblock_kernel_sphere[grid](
                b_single_t,
                s_single_t,
                f_t,
                centers_t,
                stokeslet_sum,
                near_mask,
                v_partial,
                N,
                M,
                P,
                p,
                BLOCK_N=block_n,
                BLOCK_M=block_m,
                BLOCK_Q=block_q,
                num_warps=num_warps,
            )
            torch.sum(v_partial, dim=0, out=v_buf)

            w_buf.zero_()
            w_buf[:3 * N].copy_(v_buf.reshape(3 * N))
            rhs_buf.copy_(F_tilde[p])
            rhs_buf.sub_(w_buf)
            torch.mv(B_inv_t, rhs_buf, out=x[p])

            if use_cutoff:
                torch.sum(x[p, :3 * M].reshape(M, 3), dim=0, out=stokeslet_sum[p])

        max_diff = torch.linalg.norm(x - x_prev, dim=1).max().item()
        if max_diff < tol:
            print(f"Converged after {iteration+1} iterations (max diff = {max_diff:e})")
            break
        if print_steps:
            print(f"Iteration {iteration+1}: max diff = {max_diff:e}", flush=True)
    else:
        raise RuntimeError("Solver did not converge")

    return [x[p] for p in range(P)]



class MobMFSTriton:
    def __init__(self, shape, acc):
        assert shape=="sphere", "Only sphere shape is implemented in this example."

        self.boundary = np.loadtxt(f'data/points/b_{shape}_{acc}.txt', dtype=np.float64)
        self.source = np.loadtxt(f'data/points/s_{shape}_{acc}.txt', dtype=np.float64)
        print(f"Loaded geometry: {self.boundary.shape[0]} boundary nodes, {self.source.shape[0]} source points")

        self.B_orig = build_B(self.boundary, self.source, np.zeros(3))
        self.B_inv = np.linalg.pinv(self.B_orig)

        self.L_cut = 25.0
        self.block_n = 64
        self.block_m = 64
        self.block_q = 4
        self.num_warps = 4
        self.device = None
        self._cached_device = None
        self._b_t = None
        self._s_t = None
        self._B_inv_t = None

    def _get_device_tensors(self, device):
        device = torch.device("cuda" if device is None else device)
        if self._cached_device != device:
            dtype = torch.float64
            self._b_t = torch.as_tensor(self.boundary, device=device, dtype=dtype).contiguous()
            self._s_t = torch.as_tensor(self.source, device=device, dtype=dtype).contiguous()
            self._B_inv_t = torch.as_tensor(self.B_inv, device=device, dtype=dtype).contiguous()
            self._cached_device = device
        return self._b_t, self._s_t, self._B_inv_t

    def apply(self, config, forces, viscosity=1.0):
        """
        Sphere-only (CPU inputs, GPU compute).
        """
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        N_particles = config.shape[0]
        centers = config[:, :3]
        F_ext_list = forces[:, :3]
        T_ext_list = forces[:, 3:6]
        b_t, s_t, B_inv_t = self._get_device_tensors(self.device)

        V_tilde_list = imp_mfs_mobility_sphere_triton(
            b_t,
            s_t,
            centers,
            F_ext_list,
            T_ext_list,
            B_inv_t,
            max_iter=1000,
            tol=1e-7,
            print_steps=False,
            L_cut=self.L_cut,
            device=b_t.device,
            block_n=self.block_n,
            block_m=self.block_m,
            block_q=self.block_q,
            num_warps=self.num_warps,
        )

        M1 = self.source.shape[0]
        sol = torch.stack(V_tilde_list, dim=0)
        velocities = sol[:, 3 * M1:3 * M1 + 6].detach().cpu().numpy()

        end.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end)
        print(f"Mobility computation time: {elapsed_time_ms:.2f} ms")
        return velocities


if __name__ == "__main__":
    from src.mob_op_2b_combined import check_against_ref 
    from pathlib import Path

    shape = "sphere"
    acc = "Xfine"
    mob_mfs = MobMFSTriton(shape, acc)

    tmp_dir = Path("tmp")
    ref_paths = sorted(tmp_dir.glob("testcase_*"))
    assert ref_paths, f"No testcase_* files found in {tmp_dir.resolve()}"

    # for ref_path in ref_paths:
    #     print(f"\n=== Running check_against_ref on {ref_path} ===", flush=True)
    #     check_against_ref(mob_mfs, str(ref_path), print_stuff=True)

    ref_path = "tmp/testcase_uniform_0.15_40.csv"
    check_against_ref(mob_mfs, str(ref_path), print_stuff=False)