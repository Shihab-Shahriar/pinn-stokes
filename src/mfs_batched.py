"""Batched multi-right-hand-side MFS mobility solver for rigid spheres (Triton + torch).

Per particle p the MFS unknowns are x_p = [f_p (3M source strengths); V_p; Omega_p] with
x_p = B_inv [-W_p; F_p; T_p], where B (3N+6)x(3M+6) is the single-sphere MFS matrix built once at the
origin (src/mfs_utils.build_B), B_inv its pseudo-inverse (shared by all spheres) and
W_p = sum_{q != p} A_pq f_q the velocity induced at p's N boundary nodes by the other spheres' sources
(exact Oseen sum, mu = 1).  The reference solvers (src/mfs.py, src/triton_mfs.py) iterate this
Gauss-Seidel style, one right-hand side at a time, in fp64.

Here the coupling is solved for R right-hand sides of n_sys independent configurations at once, in the
space of boundary velocities W (the strengths are ill-determined MFS internals; W is the physical,
well-conditioned quantity).  Splitting B_inv = [K | C] into the columns acting on the boundary rows (K)
and on the force/torque rows (C):

    f = X0_f - K_f W,   [V; Omega] = X0_v - K_v W,   X0 = C [F; T]  (isolated-sphere solution)
    T W = b,   T = I + Wop K_f,   b = Wop X0_f,     Wop = off-diagonal Oseen operator (Triton kernel)

`method="jacobi"` is plain Richardson on T (the reference fixed point, used for validation);
`method="gmres"` is batched restarted GMRES (production).  `backend="torch64"` evaluates Wop exactly in
fp64 with torch (reference / truth mode); `backend="triton32"` uses the fp32 Triton kernel.  K_f is
always applied by an fp64 cuBLAS GEMM: the pseudo-inverse is ill-conditioned (cond ~ 1e6) and an fp32
product carries a net-force error that shows up in the labels (see the plan / tests).

Layouts: S (3M, LD) and W (3N, LD) with LD = P_tot * R, rows 3m+j / 3n+i, columns q*R + r; systems are
concatenated along the particle axis (all with the same P_c) and coupled only inside [sys_start, sys_end).
Convention of the result: [U; Omega]_t = sum_s M_ts [F; T]_s, mu = 1, sphere radius 1.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    from src.mfs_utils import build_B
except ImportError:  # imported with src/ on sys.path
    from mfs_utils import build_B

INV_8PI = 1.0 / (8.0 * math.pi)
POINTS_DIR = "data/points"
SOLVER_VERSION = "mfs_batched-1.0"
_TEMPLATE_CACHE = {}


def load_template(acc: str, points_dir: str = POINTS_DIR):
    """Boundary nodes (N,3) and source points (M,3) of the unit sphere, and B_inv (3M+6, 3N+6), fp64."""
    key = (acc, points_dir)
    if key not in _TEMPLATE_CACHE:
        b = np.loadtxt(f"{points_dir}/b_sphere_{acc}.txt", dtype=np.float64)
        s = np.loadtxt(f"{points_dir}/s_sphere_{acc}.txt", dtype=np.float64)
        B = build_B(b, s, np.zeros(3))
        B_inv = np.linalg.pinv(B)
        _TEMPLATE_CACHE[key] = (b, s, B_inv)
    return _TEMPLATE_CACHE[key]


@dataclass
class SolveInfo:
    method: str
    backend: str
    prec: str
    tol: float
    P_c: int
    n_sys: int
    R: int
    iters: int = 0
    restarts: int = 0
    n_matvec: int = 0
    max_rel_residual: float = float("nan")
    rel_residual_per_col: Optional[np.ndarray] = None   # (n_sys, R)
    tol_v: float = float("nan")
    max_rel_dv: float = float("nan")                    # last relative change of the velocities per column (max)
    rel_dv_per_col: Optional[np.ndarray] = None
    converged: bool = False
    converged_mask: Optional[np.ndarray] = None         # (n_sys, R) bool
    converged_by: str = ""
    wall_s: float = 0.0
    symm_err: Optional[float] = None
    chunks: int = 1
    notes: List[str] = field(default_factory=list)


class MFSConvergenceError(RuntimeError):
    def __init__(self, msg: str, info: SolveInfo):
        super().__init__(msg)
        self.info = info


class _Ctx:
    """Concatenated systems: centres, system ranges, column layout."""

    def __init__(self, c64: torch.Tensor, P_c: int, n_sys: int, R: int):
        self.c64 = c64
        self.P_c, self.n_sys, self.R = P_c, n_sys, R
        self.P_tot = P_c * n_sys
        self.LD = self.P_tot * R
        dev = c64.device
        p = torch.arange(self.P_tot, device=dev, dtype=torch.int32)
        self.sys_start = ((p // P_c) * P_c).to(torch.int32).contiguous()
        self.sys_end = (self.sys_start + P_c).contiguous()


class BatchedMFS:
    def __init__(self, acc: str = "fine", device="cuda", backend: str = "triton32",
                 gemm_dtype=torch.float64, prec: str = "ieee", two_level: bool = True,
                 tol: Optional[float] = None, tol_v: Optional[float] = None, max_iter: int = 60, m: int = 20,
                 max_restarts: int = 3,
                 mem_budget_gb: float = 4.0, points_dir: str = POINTS_DIR, verbose: bool = False):
        assert backend in ("triton32", "torch64"), backend
        assert prec in ("ieee",), f"{prec}: only the FMA path is supported (tf32x3/tf32 crash the Triton 3.2 compiler on register operands)"
        self.acc, self.backend, self.prec, self.two_level = acc, backend, prec, two_level
        self.gemm_dtype = gemm_dtype
        # Convergence is judged on the velocities (tol_v, per column, every Krylov step).  The W-space
        # residual tolerance `tol` is only a safety stop: for fp64 it is set below what tol_v needs so that
        # the velocity criterion governs; for the fp32 kernel the W residual stalls at ~1e-4 (MFS strength
        # cancellation ~1e3 x fp32 eps) and only the velocity criterion can be met (floor ~1e-6..1e-5).
        if tol_v is None:
            tol_v = 1e-8 if backend == "torch64" else 1e-5
        if tol is None:
            tol = 1e-10 if backend == "torch64" else 1e-6
        self.tol, self.tol_v = float(tol), float(tol_v)
        self.max_iter, self.m, self.max_restarts = int(max_iter), int(m), int(max_restarts)
        self.mem_budget = float(mem_budget_gb) * (1 << 30)
        self.verbose = verbose
        self.device = torch.device(device)
        self.wdtype = torch.float64 if backend == "torch64" else torch.float32

        b, s, B_inv = load_template(acc, points_dir)
        self.N, self.M = b.shape[0], s.shape[0]
        N3, M3 = 3 * self.N, 3 * self.M
        dev = self.device
        self.b64 = torch.as_tensor(b, dtype=torch.float64, device=dev).contiguous()
        self.s64 = torch.as_tensor(s, dtype=torch.float64, device=dev).contiguous()
        self.b32 = self.b64.to(torch.float32).contiguous()
        self.s32 = self.s64.to(torch.float32).contiguous()
        Bt = torch.as_tensor(B_inv, dtype=torch.float64, device=dev)
        self.K_f = Bt[:M3, :N3].contiguous()          # (3M, 3N)
        self.K_v = Bt[M3:, :N3].contiguous()          # (6, 3N)
        self.C = Bt[:, N3:].contiguous()              # (3M+6, 6)
        self.M_self = self.C[M3:].clone()             # (6, 6)
        self.K_f_gemm = self.K_f.to(gemm_dtype).contiguous()
        self._I3_64 = torch.eye(3, dtype=torch.float64, device=dev)

    # ------------------------------------------------------------------ operator
    def self_mobility(self) -> torch.Tensor:
        return self.M_self.clone()

    def oseen_offdiag(self, S: torch.Tensor, ctx: _Ctx) -> torch.Tensor:
        """W = Wop S: S (3M, LD) -> W (3N, LD), both in the working dtype."""
        if self.backend == "torch64":
            return self._oseen_torch64(S, ctx)
        from src.mfs_batched_kernels import oseen_offdiag_triton
        S32 = S.to(torch.float32).contiguous()
        W32 = torch.empty((3 * self.N, ctx.LD), dtype=torch.float32, device=self.device)
        oseen_offdiag_triton(self.b32, self.s32, ctx.c64, ctx.sys_start, ctx.sys_end, S32, W32, ctx.R,
                             prec=self.prec, two_level=self.two_level)
        return W32

    def _oseen_torch64(self, S: torch.Tensor, ctx: _Ctx) -> torch.Tensor:
        """Exact fp64 operator: for every target p, one dgemm over all partners q of its system,
        W_p (3N x R) = G_p (3N x 3M*Q) @ S_{q's} (3M*Q x R), with G built elementwise in fp64."""
        S = S.to(torch.float64)
        N, M, R = self.N, self.M, ctx.R
        P_c, n_sys = ctx.P_c, ctx.n_sys
        W = torch.empty((3 * N, ctx.LD), dtype=torch.float64, device=self.device)
        I3 = self._I3_64
        for c in range(n_sys):
            base = c * P_c
            cs = ctx.c64[base:base + P_c]                                          # (P_c, 3)
            Ssys = S[:, base * R:(base + P_c) * R].reshape(3 * M, P_c, R)           # rows 3m+j
            Ssys = Ssys.permute(1, 0, 2).reshape(P_c * 3 * M, R)                    # (P_c*3M, R): (q, 3m+j)
            for pl in range(P_c):
                p = base + pl
                bp = self.b64 + ctx.c64[p]                                          # (N, 3)
                sq = self.s64[None, :, :] + cs[:, None, :]                          # (P_c, M, 3)
                r = bp[None, :, None, :] - sq[:, None, :, :]                        # (Q, N, M, 3)
                inv = 1.0 / torch.linalg.norm(r, dim=-1)                            # (Q, N, M)
                inv[pl] = 0.0                                                       # exclude self
                inv3 = inv * inv * inv
                G = inv[..., None, None] * I3 + (r[..., :, None] * r[..., None, :]) * inv3[..., None, None]
                # (Q, N, M, 3, 3) -> (N, 3, Q, M, 3) -> (3N, Q*3M)
                Gm = G.permute(1, 3, 0, 2, 4).reshape(3 * N, P_c * 3 * M)
                W[:, p * R:(p + 1) * R] = (Gm @ Ssys) * INV_8PI
        return W

    def apply_Kf(self, W: torch.Tensor) -> torch.Tensor:
        """K_f W in the GEMM dtype, returned in the working dtype (3M, LD)."""
        KW = self.K_f_gemm @ W.to(self.gemm_dtype)
        return KW.to(self.wdtype)

    def matvec(self, W: torch.Tensor, ctx: _Ctx) -> torch.Tensor:
        """T W = W + Wop (K_f W)."""
        return W + self.oseen_offdiag(self.apply_Kf(W), ctx)

    # ------------------------------------------------------------------ column reductions
    def _col_view(self, V: torch.Tensor, ctx: _Ctx) -> torch.Tensor:
        return V.view(3 * self.N, ctx.n_sys, ctx.P_c, ctx.R)

    def _dot_cols(self, U: torch.Tensor, V: torch.Tensor, ctx: _Ctx) -> torch.Tensor:
        prod = self._col_view(U, ctx) * self._col_view(V, ctx)
        return prod.sum(dim=(0, 2), dtype=torch.float64)                  # (n_sys, R)

    def _norm_cols(self, V: torch.Tensor, ctx: _Ctx) -> torch.Tensor:
        return torch.sqrt(self._dot_cols(V, V, ctx))

    def _bcast(self, a: torch.Tensor, ctx: _Ctx) -> torch.Tensor:
        """(n_sys, R) -> broadcastable against the (3N, n_sys, P_c, R) view."""
        return a.to(self.wdtype)[None, :, None, :]

    def _axpy_cols(self, V: torch.Tensor, a: torch.Tensor, X: torch.Tensor, ctx: _Ctx) -> None:
        """V += a[col] * X  (in place)."""
        self._col_view(V, ctx).add_(self._bcast(a, ctx) * self._col_view(X, ctx))

    # ------------------------------------------------------------------ solvers
    def _vel_view(self, V6: torch.Tensor, ctx: _Ctx) -> torch.Tensor:
        return V6.view(6, ctx.n_sys, ctx.P_c, ctx.R)

    def _vel_rel_change(self, Vn: torch.Tensor, Vp: torch.Tensor, ctx: _Ctx) -> torch.Tensor:
        """Per-column relative change of the 6*P_c velocity vector, (n_sys, R) fp64."""
        d = self._vel_view(Vn - Vp, ctx).pow(2).sum(dim=(0, 2)).sqrt()
        n = self._vel_view(Vn, ctx).pow(2).sum(dim=(0, 2)).sqrt()
        return d / torch.where(n > 0, n, torch.ones_like(n))

    def _velocities(self, X0_v: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return X0_v - self.K_v @ W.to(torch.float64)

    def _jacobi(self, b: torch.Tensor, ctx: _Ctx, X0_v: torch.Tensor, tol: float, tol_v: float,
                max_iter: int, info: SolveInfo):
        W = b.clone()
        Vp = self._velocities(X0_v, W)
        dv_prev = torch.full((ctx.n_sys, ctx.R), float("inf"), dtype=torch.float64, device=self.device)
        dv = dv_prev
        for it in range(max_iter):
            Wn = b - self.oseen_offdiag(self.apply_Kf(W), ctx)
            info.n_matvec += 1
            W = Wn
            Vn = self._velocities(X0_v, W)
            dv_prev, dv = dv, self._vel_rel_change(Vn, Vp, ctx)
            Vp = Vn
            info.iters = it + 1
            if bool((torch.maximum(dv, dv_prev) <= tol_v).all()):
                break
        r = b - self.matvec(W, ctx)
        info.n_matvec += 1
        beta0 = self._norm_cols(b, ctx)
        rel = self._norm_cols(r, ctx) / torch.where(beta0 > 0, beta0, torch.ones_like(beta0))
        self._finish(info, rel, torch.maximum(dv, dv_prev), tol, tol_v)
        return W

    def _gmres(self, b: torch.Tensor, ctx: _Ctx, X0_v: torch.Tensor, tol: float, tol_v: float, m: int,
               max_restarts: int, info: SolveInfo):
        n_sys, R = ctx.n_sys, ctx.R
        dev = self.device
        beta0 = self._norm_cols(b, ctx)                                   # (n_sys, R)
        beta0_safe = torch.where(beta0 > 0, beta0, torch.ones_like(beta0))
        W = torch.zeros_like(b)
        r = b.clone()
        rel = beta0 / beta0_safe
        Vbase = self._velocities(X0_v, W)                                 # velocities at the restart base
        Vp = Vbase
        dv = torch.full((n_sys, R), float("inf"), dtype=torch.float64, device=dev)
        dv_prev = dv
        done = False
        for cycle in range(max_restarts):
            beta = self._norm_cols(r, ctx)
            beta_safe = torch.where(beta > 0, beta, torch.ones_like(beta))
            V = [r / self._bcast(beta_safe, ctx).expand_as(self._col_view(r, ctx)).reshape_as(r)]
            KV = [self.K_v @ V[0].to(torch.float64)]                       # (6, LD) per basis vector
            H = torch.zeros((m + 1, m, n_sys, R), dtype=torch.float64, device=dev)
            cs = torch.zeros((m, n_sys, R), dtype=torch.float64, device=dev)
            sn = torch.zeros((m, n_sys, R), dtype=torch.float64, device=dev)
            g = torch.zeros((m + 1, n_sys, R), dtype=torch.float64, device=dev)
            g[0] = beta
            k_done = 0
            y = None
            for k in range(m):
                w = self.matvec(V[k], ctx)
                info.n_matvec += 1
                for _pass in range(2):                                    # MGS + one re-orthogonalisation
                    for i in range(k + 1):
                        h = self._dot_cols(V[i], w, ctx)
                        self._axpy_cols(w, -h, V[i], ctx)
                        H[i, k] += h
                hk1 = self._norm_cols(w, ctx)
                H[k + 1, k] = hk1
                hk1_safe = torch.where(hk1 > 0, hk1, torch.ones_like(hk1))
                V.append(w / self._bcast(hk1_safe, ctx).expand_as(self._col_view(w, ctx)).reshape_as(w))
                KV.append(self.K_v @ V[-1].to(torch.float64))
                for i in range(k):
                    t1 = cs[i] * H[i, k] + sn[i] * H[i + 1, k]
                    t2 = -sn[i] * H[i, k] + cs[i] * H[i + 1, k]
                    H[i, k], H[i + 1, k] = t1, t2
                denom = torch.sqrt(H[k, k] ** 2 + H[k + 1, k] ** 2)
                ok = denom > 0
                denom_safe = torch.where(ok, denom, torch.ones_like(denom))
                cs[k] = torch.where(ok, H[k, k] / denom_safe, torch.ones_like(denom))
                sn[k] = torch.where(ok, H[k + 1, k] / denom_safe, torch.zeros_like(denom))
                H[k, k] = denom
                H[k + 1, k] = 0.0
                g[k + 1] = -sn[k] * g[k]
                g[k] = cs[k] * g[k]
                k_done = k + 1
                info.iters += 1
                # current iterate's velocities from the small triangular solve (no full update needed)
                y = self._back_substitute(H, g, k_done)
                Vn = Vbase.clone()
                for i in range(k_done):
                    self._vel_view(Vn, ctx).sub_(y[i][None, :, None, :] * self._vel_view(KV[i], ctx))
                dv_prev, dv = dv, self._vel_rel_change(Vn, Vp, ctx)
                Vp = Vn
                res_est = g[k + 1].abs() / beta0_safe
                if bool((res_est <= tol).all()) or bool((torch.maximum(dv, dv_prev) <= tol_v).all()):
                    break
            for i in range(k_done):
                self._axpy_cols(W, y[i], V[i], ctx)
            del V, KV
            r = b - self.matvec(W, ctx)
            info.n_matvec += 1
            rel = self._norm_cols(r, ctx) / beta0_safe
            Vbase = self._velocities(X0_v, W)
            info.restarts = cycle + 1
            if bool((rel <= tol).all()) or bool((torch.maximum(dv, dv_prev) <= tol_v).all()):
                break
        self._finish(info, rel, torch.maximum(dv, dv_prev), tol, tol_v)
        return W

    @staticmethod
    def _back_substitute(H: torch.Tensor, g: torch.Tensor, k: int) -> torch.Tensor:
        """Solve the k x k upper-triangular systems H[:k,:k] y = g[:k] for every (system, column)."""
        y = torch.zeros((k, H.shape[2], H.shape[3]), dtype=H.dtype, device=H.device)
        for i in range(k - 1, -1, -1):
            acc = g[i].clone()
            for j in range(i + 1, k):
                acc -= H[i, j] * y[j]
            hii = H[i, i]
            ok = hii != 0
            y[i] = torch.where(ok, acc / torch.where(ok, hii, torch.ones_like(hii)), torch.zeros_like(acc))
        return y

    def _finish(self, info: SolveInfo, rel: torch.Tensor, dv: torch.Tensor, tol: float, tol_v: float) -> None:
        rel_np = rel.detach().cpu().numpy()
        dv_np = dv.detach().cpu().numpy()
        info.rel_residual_per_col = rel_np
        info.rel_dv_per_col = dv_np
        info.max_rel_residual = float(rel_np.max()) if rel_np.size else 0.0
        info.max_rel_dv = float(dv_np.max()) if dv_np.size else 0.0
        info.tol_v = tol_v
        by_res = rel_np <= tol
        by_v = dv_np <= tol_v
        info.converged_mask = by_res | by_v
        info.converged = bool(info.converged_mask.all())
        info.converged_by = "residual" if by_res.all() else ("velocity" if info.converged else "none")

    # ------------------------------------------------------------------ high level
    def _to_positions_list(self, positions) -> List[np.ndarray]:
        if isinstance(positions, (list, tuple)):
            lst = [np.asarray(p, dtype=np.float64) for p in positions]
        else:
            lst = [np.asarray(positions, dtype=np.float64)]
        P_c = lst[0].shape[0]
        for p in lst:
            assert p.shape == (P_c, 3), "all systems in a batch must have the same particle count (P, 3)"
        return lst

    def _make_ctx(self, positions_list: List[np.ndarray], R: int) -> _Ctx:
        c64 = torch.as_tensor(np.concatenate(positions_list, 0), dtype=torch.float64, device=self.device).contiguous()
        return _Ctx(c64, positions_list[0].shape[0], len(positions_list), R)

    def columns_per_budget(self, P_c: int, m: Optional[int] = None) -> int:
        """Max number of (system, column) pairs that fit the memory budget."""
        m = self.m if m is None else m
        wb = 8 if self.wdtype == torch.float64 else 4
        per_col = ((m + 6) * 3 * self.N * wb + (3 * self.M * 8 * 2 + 3 * self.N * 8 + 3 * self.M * wb)) * P_c
        return max(1, int(self.mem_budget // per_col))

    def _solve_ctx(self, ctx: _Ctx, X0: torch.Tensor, method: str, tol: float, tol_v: float, max_iter: int,
                   m: int, max_restarts: int, info: SolveInfo) -> torch.Tensor:
        M3 = 3 * self.M
        X0_f = X0[:M3].to(self.wdtype).contiguous()
        X0_v = X0[M3:].contiguous()
        b = self.oseen_offdiag(X0_f, ctx)
        info.n_matvec += 1
        if method == "jacobi":
            W = self._jacobi(b, ctx, X0_v, tol, tol_v, max_iter, info)
        elif method == "gmres":
            W = self._gmres(b, ctx, X0_v, tol, tol_v, m, max_restarts, info)
        else:
            raise ValueError(method)
        return self._velocities(X0_v, W)                                   # (6, LD)

    def solve(self, positions, forces, *, tol: Optional[float] = None, tol_v: Optional[float] = None,
              method: str = "gmres", max_iter: Optional[int] = None, m: Optional[int] = None,
              max_restarts: Optional[int] = None, raise_on_fail: bool = True, cols_chunk: Optional[int] = None):
        """Velocities (P_tot, 6, R) fp64 (a list per system if `positions` is a list) and SolveInfo.

        positions: (P,3) or list of (P_c,3) with equal P_c; forces: (P,6,R) or list of (P_c,6,R): column r of
        system c is one right-hand side (force/torque on every particle of that system)."""
        t0 = time.time()
        tol = self.tol if tol is None else float(tol)
        tol_v = self.tol_v if tol_v is None else float(tol_v)
        max_iter = self.max_iter if max_iter is None else int(max_iter)
        m = self.m if m is None else int(m)
        max_restarts = self.max_restarts if max_restarts is None else int(max_restarts)
        pos_list = self._to_positions_list(positions)
        as_list = isinstance(positions, (list, tuple))
        f_list = list(forces) if isinstance(forces, (list, tuple)) else [forces]
        F = torch.as_tensor(np.concatenate([np.asarray(f, dtype=np.float64) for f in f_list], 0),
                            dtype=torch.float64, device=self.device)                  # (P_tot, 6, R)
        P_c, n_sys = pos_list[0].shape[0], len(pos_list)
        assert F.shape[0] == P_c * n_sys and F.shape[1] == 6, F.shape
        R = F.shape[2]
        # column chunking: every system keeps all its particles; columns are independent
        max_cols = self.columns_per_budget(P_c, m)
        chunk = R if cols_chunk is None else int(cols_chunk)
        if n_sys * chunk > max_cols:
            chunk = max(1, max_cols // n_sys)
            chunk = min(chunk, R)
        info = SolveInfo(method=method, backend=self.backend, prec=self.prec, tol=tol, P_c=P_c, n_sys=n_sys, R=R)
        vel = torch.empty((P_c * n_sys, 6, R), dtype=torch.float64, device=self.device)
        rel_cols, dv_cols, mask_cols, bys = [], [], [], []
        n_chunks = 0
        for c0 in range(0, R, chunk):
            c1 = min(c0 + chunk, R)
            Rc = c1 - c0
            ctx = self._make_ctx(pos_list, Rc)
            FT = F[:, :, c0:c1].permute(1, 0, 2).reshape(6, ctx.LD)                    # column p*Rc + r
            X0 = self.C @ FT                                                           # (3M+6, LD) fp64
            sub = SolveInfo(method=method, backend=self.backend, prec=self.prec, tol=tol, P_c=P_c, n_sys=n_sys, R=Rc)
            Vc = self._solve_ctx(ctx, X0, method, tol, tol_v, max_iter, m, max_restarts, sub)
            vel[:, :, c0:c1] = Vc.reshape(6, ctx.P_tot, Rc).permute(1, 0, 2)
            info.iters = max(info.iters, sub.iters)
            info.restarts = max(info.restarts, sub.restarts)
            info.n_matvec += sub.n_matvec
            rel_cols.append(sub.rel_residual_per_col)
            dv_cols.append(sub.rel_dv_per_col)
            mask_cols.append(sub.converged_mask)
            bys.append(sub.converged_by)
            n_chunks += 1
        info.chunks = n_chunks
        info.tol_v = tol_v
        info.rel_residual_per_col = np.concatenate(rel_cols, axis=1)
        info.rel_dv_per_col = np.concatenate(dv_cols, axis=1)
        info.converged_mask = np.concatenate(mask_cols, axis=1)
        info.max_rel_residual = float(info.rel_residual_per_col.max())
        info.max_rel_dv = float(info.rel_dv_per_col.max())
        info.converged = bool(info.converged_mask.all())
        info.converged_by = bys[0] if len(set(bys)) == 1 else "mixed"
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        info.wall_s = time.time() - t0
        if self.verbose:
            print(f"[BatchedMFS] {method} {self.backend} P={P_c}x{n_sys} R={R} iters={info.iters} "
                  f"matvecs={info.n_matvec} maxres={info.max_rel_residual:.2e} maxdv={info.max_rel_dv:.2e} "
                  f"by={info.converged_by} chunks={n_chunks} {info.wall_s:.2f}s")
        if raise_on_fail and not info.converged:
            raise MFSConvergenceError(f"MFS did not converge: max rel residual {info.max_rel_residual:.3e} > tol "
                                      f"and max rel velocity change {info.max_rel_dv:.3e} > tol_v", info)
        if as_list:
            return [vel[i * P_c:(i + 1) * P_c] for i in range(n_sys)], info
        return vel, info

    @staticmethod
    def unit_forces(P: int) -> np.ndarray:
        """(P, 6, 6P): column k = 6 s + i is a unit force/torque component i on particle s."""
        F = np.zeros((P, 6, 6 * P))
        for s in range(P):
            for i in range(6):
                F[s, i, 6 * s + i] = 1.0
        return F

    def solve_mobility_matrix(self, positions, **kw):
        """Grand mobility matrix M (6P, 6P) fp64 (torch, on device) with [U;Omega]_t = sum_s M_ts [F;T]_s."""
        pos = np.asarray(positions, dtype=np.float64)
        P = pos.shape[0]
        vel, info = self.solve(pos, self.unit_forces(P), **kw)
        Mmat = vel.reshape(6 * P, 6 * P)                                 # rows 6t+i', columns 6s+i
        info.symm_err = float(torch.linalg.norm(Mmat - Mmat.T) / torch.linalg.norm(Mmat))
        return Mmat, info

    def solve_mobility_matrix_batch(self, positions_list: Sequence, **kw):
        """Same for a list of equal-P configurations, batched on the GPU (chunked by the memory budget)."""
        pos_list = self._to_positions_list(list(positions_list))
        P = pos_list[0].shape[0]
        R = 6 * P
        max_cols = self.columns_per_budget(P, kw.get("m", None))
        n_per_batch = max(1, max_cols // R)
        Ms, infos = [], []
        F1 = self.unit_forces(P)
        for i0 in range(0, len(pos_list), n_per_batch):
            batch = pos_list[i0:i0 + n_per_batch]
            vels, info = self.solve(batch, [F1] * len(batch), **kw)
            for j, v in enumerate(vels):
                Mmat = v.reshape(6 * P, 6 * P)
                sub = SolveInfo(**{k: getattr(info, k) for k in ("method", "backend", "prec", "tol", "P_c", "R")},
                                n_sys=1, iters=info.iters, restarts=info.restarts, n_matvec=info.n_matvec,
                                wall_s=info.wall_s / len(batch), chunks=info.chunks, tol_v=info.tol_v,
                                converged_by=info.converged_by)
                sub.rel_residual_per_col = info.rel_residual_per_col[j:j + 1]
                sub.rel_dv_per_col = info.rel_dv_per_col[j:j + 1]
                sub.converged_mask = info.converged_mask[j:j + 1]
                sub.max_rel_residual = float(sub.rel_residual_per_col.max())
                sub.max_rel_dv = float(sub.rel_dv_per_col.max())
                sub.converged = bool(sub.converged_mask.all())
                sub.symm_err = float(torch.linalg.norm(Mmat - Mmat.T) / torch.linalg.norm(Mmat))
                Ms.append(Mmat)
                infos.append(sub)
        return Ms, infos
