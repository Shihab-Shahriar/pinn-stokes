#!/usr/bin/env python3
"""HIGNN (Ma, Ye & Pan, CMAME 400 (2022) 115496; H-HIGNN 2025) as a NeMO harness mobility operator.

Pure-torch re-evaluation of the shipped HIGNN weights (github.com/Pan-Group-UW-Madison/hignn), so the
baseline can be scored on this repo's truth cache without their C++/Kokkos/MPI build. Two variants:

  "2b"    what their H-matrix engine (`HignnModel.dot`) computes:
              u_i = F_i + sum_{j != i} M2(x_j - x_i) F_j   over ALL pairs (no cutoff, no RPY),
              M2(x) = (I + net2(x)) / |x|                    (python/convert.py::Net, nn/two_body_unbounded.pkl)
          evaluated densely here; their engine adds ACA compression on top, so dense is their best case.
  "full"  + the two corrections that exist only in their Python path (python/HIGNN/model_structure.py::
          HIGNN_mdoel.forward, driven by python/gravity_field.py):
              3-body:  u_t += sum_{(s,m,t): s~m, m~t, t != s} net3([x_m - x_s | x_t - x_m]) / (|x_m - x_s| |x_t - x_m|) F_s
              self:    u_t += sum_{n ~ t} netself(x_n - x_t) / |x_n - x_t|^2 F_t
          with "~" = centre distance < eps3 (5.0 in their scripts; an inference-time choice, not stored
          with the weights). Weights python/Saved_Model/Unbounded_try1/HIGNN_nn_{2body,3body,self}.pkl
          (plain nn.Sequential pickles; the 2-body one is bit-identical to nn/two_body_unbounded.pkl, so
          this is the consistent model their gravity_field.py runs).

HIGNN is translational-only and torque-free: radius a = 1, mu = 1, self mobility = I, i.e. velocities in
units of F/(6 pi mu a) (divided out here). Torques in the input are ignored and the returned angular
velocities are 0, so score it with the translational metrics only (prmse_lin, prmse_fluct, err_mean_pct,
max_rel_lin). Nothing here imports the hignn package, torch_scatter or torch_geometric.

  HIGNN_ROOT=/path/to/hignn python src/hignn_ops.py --case 200 0.1 4423            # both variants, one truth
  python src/hignn_ops.py --case 200 0.1 4423 --variant full --eps3-sweep 3 4 5 6 8   # cutoff sensitivity
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SIX_PI = 6.0 * math.pi
HIGNN_ROOT = Path(os.environ.get("HIGNN_ROOT", "/home/shihab/throwaway/hignn"))
WEIGHTS = {"2b": "python/Saved_Model/Unbounded_try1/HIGNN_nn_2body.pkl",
           "3b": "python/Saved_Model/Unbounded_try1/HIGNN_nn_3body.pkl",
           "self": "python/Saved_Model/Unbounded_try1/HIGNN_nn_self.pkl"}
ENGINE_2B = "nn/two_body_unbounded.pkl"   # what their C++ engine converts to TorchScript (same weights as WEIGHTS["2b"])
DEFAULT_EPS3 = 5.0                        # python/gravity_field.py:36, python/neighbor_lists.py:49
ARCH = {"2b": [3, 128, 512, 9], "3b": [6, 256, 1024, 512, 9], "self": [3, 256, 1024, 512, 9]}


def weight_path(key: str, root=None) -> Path:
    return Path(root or HIGNN_ROOT) / WEIGHTS[key]


def load_sequential(path) -> torch.nn.Sequential:
    """One of their pickled nn.Sequential MLPs (torch.nn globals only): float32, eval, frozen."""
    path = Path(path)
    assert path.exists(), f"HIGNN weights not found: {path} (set HIGNN_ROOT to the hignn checkout)"
    net = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(net, torch.nn.Sequential), (path, type(net))
    return net.float().eval().requires_grad_(False)


def md5(path) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def _widths(net: torch.nn.Sequential) -> list[int]:
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    return [m.in_features for m in lin] + [lin[-1].out_features]


class HignnMob:
    """HIGNN mobility (translational), behind the harness `apply_cpu` / `apply` contracts (module docstring)."""

    def __init__(self, two_body_path=None, three_body_path=None, self_path=None, *, variant: str = "full",
                 eps3: float = DEFAULT_EPS3, device=None, pair_chunk: int = 262_144, triple_chunk: int = 131_072):
        assert variant in ("2b", "full"), variant
        self.variant, self.eps3 = variant, float(eps3)
        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pair_chunk, self.triple_chunk = int(pair_chunk), int(triple_chunk)
        self.paths = {"2b": Path(two_body_path or weight_path("2b"))}
        self.net2 = load_sequential(self.paths["2b"]).to(self.device)
        assert _widths(self.net2) == ARCH["2b"], _widths(self.net2)
        self.net3 = self.netself = None
        if variant == "full":
            self.paths["3b"] = Path(three_body_path or weight_path("3b"))
            self.paths["self"] = Path(self_path or weight_path("self"))
            self.net3 = load_sequential(self.paths["3b"]).to(self.device)
            self.netself = load_sequential(self.paths["self"]).to(self.device)
            assert _widths(self.net3) == ARCH["3b"], _widths(self.net3)
            assert _widths(self.netself) == ARCH["self"], _widths(self.netself)
        self.last: dict = {}      # edge counts / wall of the last apply (report material)
        self._warned_torque = False

    # ------------------------------------------------------------ kernels (HIGNN units; rel = x_source - x_target)
    @torch.no_grad()
    def pair_mobility(self, rel: torch.Tensor) -> torch.Tensor:
        """(P,3) -> (P,3,3): (I + net2(rel)) / |rel|  == Two_body_net_Hmatrix == convert.py::Net for r > 1e-5."""
        r = rel.norm(dim=1, keepdim=True)
        y = self.net2(rel).clone()
        y[:, 0] += 1.0
        y[:, 4] += 1.0
        y[:, 8] += 1.0
        return (y / r).reshape(-1, 3, 3)

    @torch.no_grad()
    def triple_mobility(self, x6: torch.Tensor) -> torch.Tensor:
        """(T,6) = [x_m - x_s | x_t - x_m] -> (T,3,3): net3(x6) / (|x_m - x_s| |x_t - x_m|)  == Three_body_net."""
        r1 = x6[:, :3].norm(dim=1, keepdim=True)
        r2 = x6[:, 3:].norm(dim=1, keepdim=True)
        return (self.net3(x6) / r1 / r2).reshape(-1, 3, 3)

    @torch.no_grad()
    def self_mobility(self, rel: torch.Tensor) -> torch.Tensor:
        """(S,3) = x_n - x_t -> (S,3,3): netself(rel) / |rel|^2  == Two_body_self_net."""
        r = rel.norm(dim=1, keepdim=True)
        return (self.netself(rel) / r / r).reshape(-1, 3, 3)

    # ------------------------------------------------------------ terms
    @torch.no_grad()
    def two_body_velocity(self, X: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """F_i + sum_{j != i} M2(x_j - x_i) F_j, dense over all pairs in target-row blocks; float64 accumulation
        (their engine accumulates the float32 kernel into a double `u` as well)."""
        N = X.shape[0]
        u = F.double().clone()
        rows = max(1, self.pair_chunk // max(N, 1))
        ar = torch.arange(N, device=X.device)
        for r0 in range(0, N, rows):
            r1 = min(N, r0 + rows)
            i = torch.arange(r0, r1, device=X.device).repeat_interleave(N)
            j = ar.repeat(r1 - r0)
            keep = i != j
            i, j = i[keep], j[keep]
            M = self.pair_mobility(X[j] - X[i])
            u.index_add_(0, i, torch.bmm(M, F[j].unsqueeze(2)).squeeze(2).double())
        self.last["pairs"] = N * (N - 1)
        return u

    @torch.no_grad()
    def neighbour_pairs(self, X: torch.Tensor):
        """Ordered pairs (a, b), a != b, |x_a - x_b| < eps3, sorted by a (their NeighborLists radius search)."""
        N = X.shape[0]
        rows = max(1, self.pair_chunk // max(N, 1))
        A, B = [], []
        for r0 in range(0, N, rows):
            r1 = min(N, r0 + rows)
            m = torch.cdist(X[r0:r1], X) < self.eps3
            m[torch.arange(r1 - r0, device=X.device), torch.arange(r0, r1, device=X.device)] = False
            a, b = m.nonzero(as_tuple=True)
            A.append(a + r0)
            B.append(b)
        return torch.cat(A), torch.cat(B)

    @torch.no_grad()
    def self_velocity(self, X, F, a, b) -> torch.Tensor:
        """sum_{n ~ t} Mself(x_n - x_t) F_t: edge (n = a, t = b), attr F_t, scattered to t
        (HIGNN_mdoel.forward: x_in = x[edge[0]] - x[edge[1]], attr = F[edge[1]], scatter -> edge[1])."""
        u = torch.zeros(F.shape, dtype=torch.float64, device=F.device)
        for p0 in range(0, a.numel(), self.triple_chunk):
            n, t = a[p0:p0 + self.triple_chunk], b[p0:p0 + self.triple_chunk]
            M = self.self_mobility(X[n] - X[t])
            u.index_add_(0, t, torch.bmm(M, F[t].unsqueeze(2)).squeeze(2).double())
        self.last["self_edges"] = int(a.numel())
        return u

    @torch.no_grad()
    def three_body_velocity(self, X, F, a, b) -> torch.Tensor:
        """Chains (s, m, t) with s ~ m, m ~ t, t != s (NeighborLists::BuildThreeBodyInfo):
        u_t += M3([x_m - x_s | x_t - x_m]) F_s  (HIGNN_mdoel.forward: x_in = cat(x[e1] - x[e0], x[e2] - x[e1]),
        attr = F[e0], scatter -> e2). For each ordered pair (s = a, m = b) the targets are the neighbours of m
        except s; the neighbour list of m is the contiguous block of `b` where `a == m` (pairs sorted by a)."""
        N = X.shape[0]
        u = torch.zeros(F.shape, dtype=torch.float64, device=F.device)
        self.last["triples"] = 0
        if a.numel() == 0:
            return u
        counts = torch.bincount(a, minlength=N)
        offsets = torch.cumsum(counts, 0) - counts
        per_pair = counts[b]                                  # targets generated by each pair (before t != s)
        cum = torch.cumsum(per_pair, 0)
        total = int(cum[-1])
        if total == 0:
            return u
        marks = torch.arange(0, total, self.triple_chunk, device=X.device)
        bounds = sorted(set(torch.searchsorted(cum, marks).tolist()) | {0, int(per_pair.numel())})
        for p0, p1 in zip(bounds[:-1], bounds[1:]):
            cnt = per_pair[p0:p1]
            n_out = int(cnt.sum())
            if n_out == 0:
                continue
            seg = torch.cumsum(cnt, 0) - cnt
            local = torch.arange(n_out, device=X.device) - seg.repeat_interleave(cnt)
            m = b[p0:p1].repeat_interleave(cnt)
            s = a[p0:p1].repeat_interleave(cnt)
            t = b[offsets[m] + local]
            keep = t != s
            s, m, t = s[keep], m[keep], t[keep]
            x6 = torch.cat([X[m] - X[s], X[t] - X[m]], dim=1)
            u.index_add_(0, t, torch.bmm(self.triple_mobility(x6), F[s].unsqueeze(2)).squeeze(2).double())
            self.last["triples"] += int(t.numel())
        return u

    @torch.no_grad()
    def velocity_hignn_units(self, X: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """(N,3) float64 velocities in HIGNN units (an isolated sphere moves at F)."""
        t0 = time.time()
        u = self.two_body_velocity(X, F)
        if self.variant == "full":
            a, b = self.neighbour_pairs(X)
            u += self.self_velocity(X, F, a, b)
            u += self.three_body_velocity(X, F, a, b)
        if X.is_cuda:
            torch.cuda.synchronize()
        self.last["wall_s"] = time.time() - t0
        return u

    # ------------------------------------------------------------ harness contracts
    def apply_cpu(self, positions, orientations, forces, viscosity: float = 1.0) -> np.ndarray:
        """(N,3) positions, (N,4) orientations (unused: spheres), (N,6) wrench -> (N,6) velocity, angular = 0."""
        forces = np.asarray(forces, dtype=np.float64)
        if forces.shape[1] > 3 and np.abs(forces[:, 3:]).max() > 0 and not self._warned_torque:
            print("[hignn] torques ignored: HIGNN is translational-only (torque-free)", flush=True)
            self._warned_torque = True
        X = torch.as_tensor(np.ascontiguousarray(positions, dtype=np.float32), device=self.device)
        F = torch.as_tensor(np.ascontiguousarray(forces[:, :3], dtype=np.float32), device=self.device)
        u = self.velocity_hignn_units(X, F).cpu().numpy() / (SIX_PI * float(viscosity))
        out = np.zeros((X.shape[0], 6), dtype=np.float64)
        out[:, :3] = u
        return out

    def apply(self, config, forces, viscosity: float = 1.0) -> np.ndarray:
        config = np.asarray(config)
        return self.apply_cpu(config[:, :3], config[:, 3:], forces, viscosity)


# ------------------------------------------------------------------ CLI
def truth_file(N: int, phi: float, seed: int, forcing: str = "gravity") -> Path:
    suffix = {"random": "", "gravity": "_grav"}[forcing]
    return ROOT / "tmp" / "nbody_moments_truth" / f"uniform_N{N}_phi{phi:g}_seed{seed}{suffix}.npz"


def prmse_lin(pred: np.ndarray, vel: np.ndarray) -> float:
    return 100.0 * float(np.linalg.norm(vel[:, :3] - pred[:, :3]) / np.linalg.norm(vel[:, :3]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs=3, metavar=("N", "PHI", "SEED"), default=["200", "0.1", "4423"],
                    help="gravity truth cell (tmp/nbody_moments_truth/uniform_N{N}_phi{phi}_seed{seed}_grav.npz)")
    ap.add_argument("--variant", choices=["2b", "full", "both"], default="both")
    ap.add_argument("--eps3", type=float, default=DEFAULT_EPS3)
    ap.add_argument("--eps3-sweep", type=float, nargs="*", default=None, help="full variant at each cutoff")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dump", type=Path, default=None, help="save truth + per-particle predictions (npz)")
    args = ap.parse_args()
    N, phi, seed = int(args.case[0]), float(args.case[1]), int(args.case[2])
    p = truth_file(N, phi, seed)
    assert p.exists(), p
    d = np.load(p)
    config, forces, vel = d["config"], d["forces"], d["velocity"]
    print("weights: " + ", ".join(f"{k}={weight_path(k).relative_to(HIGNN_ROOT)} md5={md5(weight_path(k))[:8]}"
                                  for k in WEIGHTS))
    variants = ["2b", "full"] if args.variant == "both" else [args.variant]
    dump = {}
    for v in variants:
        eps_list = args.eps3_sweep if (args.eps3_sweep and v == "full") else [args.eps3]
        for eps in eps_list:
            op = HignnMob(variant=v, eps3=eps, device=args.device)
            pred = op.apply(config, forces)
            print(f"N={N} phi={phi:g} seed={seed} HIGNN_{v:<4} eps3={eps:<4g} prmse_lin={prmse_lin(pred, vel):7.3f}%  "
                  f"{op.last} device={op.device}", flush=True)
            dump[f"{v}_eps{eps:g}"] = pred
    if args.dump:
        np.savez(args.dump, config=config, forces=forces, velocity=vel, **dump)
        print(f"-> {args.dump}")


if __name__ == "__main__":
    main()
