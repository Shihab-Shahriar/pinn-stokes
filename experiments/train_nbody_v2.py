#!/usr/bin/env python3
"""Train / evaluate n-body correction models on the dataset-v2 cache (experiments/build_nbody_v2_cache.py).

    python experiments/train_nbody_v2.py --model moments  --variant k10_rc6  --out experiments/runs_v2/mom_k10_rc6
    python experiments/train_nbody_v2.py --model moments  --variant kinf_rc8 --out experiments/runs_v2/mom_kinf_rc8
    python experiments/train_nbody_v2.py --model baseline --variant k10_rc6  --out experiments/runs_v2/b1_k10_rc6
    python experiments/train_nbody_v2.py --eval-only data/models/nbody_moments.pt --features moments --variant k10_rc6 \\
                                         --out experiments/runs_v2/eval_mom_old

Rows = unordered near pairs (t < s, d <= 6) with >= 1 neighbour under the selection variant; features are built on
the fly from the cached geometry with the operators' own code paths (nbody_moments.moment_features /
nbody_features.baseline_features_torch); the label is the residual block R = Mts_sym - M2b (operator conventions:
+s_vec, median 5.01), and both architectures are reciprocal by construction so unordered rows suffice.
Split = configuration level (configs.is_val, seed % 10 == 0).  Loss (default): L1 over the 36 entries of
6*pi*(predict_mobility(X) - R); Adam + cosine over all steps; the moments model fits its normalisation buffers on a
train subsample.  Metrics (validation configs, plus 2b-only): block rel-Frobenius error (all / TT / TR / RR),
residual capture, and the old-style PRMSE lin / ang under fixed random unit forces x 6*pi, by family and by
neighbour-count bin.  Outputs: metrics.json, log.csv, config.json, model.wt, model.pt (TorchScript);
--publish copies model.pt to data/models/<name>.pt with a <name>.json sidecar (selection parameters).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(ROOT)]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src import nbody_features as nf  # noqa: E402
from src import nbody_moments as nbm  # noqa: E402
from src.model_archs import MultiBodyCorrectionB1, MultiBodyMoments  # noqa: E402

MEAN_DIST_S = nf.MEAN_DIST_S
FAMILIES = ["uniform", "grown", "lattice"]
NNBR_BINS = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 10 ** 6)]
DIST_BINS = [(2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.01), (6.01, 8.01)]
PUBLISH = {("moments", "k10_rc6"): "nbody_moments_v2_k10_rc6", ("moments", "kinf_rc6"): "nbody_moments_v2_kinf_rc6",
           ("moments", "kinf_rc8"): "nbody_moments_v2_kinf_rc8", ("baseline", "k10_rc6"): "nbody_pinn_b1_v2"}
BLOCKS = {"TT": (slice(0, 3), slice(0, 3)), "TR": (slice(0, 3), slice(3, 6)), "RT": (slice(3, 6), slice(0, 3)), "RR": (slice(3, 6), slice(3, 6))}


# ----------------------------------------------------------------------------- cache on device
class V2Cache:
    """The cache as device tensors + on-the-fly feature construction for one selection variant."""

    def __init__(self, root: Path, variant: str, device, data_on: str = "gpu", families=None):
        self.root = Path(root); self.variant = variant; self.device = torch.device(device)
        self.K, self.cutoff = nf.SELECTION_VARIANTS[variant]
        cfg = np.load(self.root / "configs.npz")
        self.meta = json.load(open(self.root / "meta.json"))
        assert variant in self.meta["variants"], (variant, self.meta["variants"])
        ld = lambda name: np.load(self.root / f"{name}.npy", mmap_mode="r")
        pair_cfg = np.asarray(ld("pair_cfg")).astype(np.int64)
        n_all = pair_cfg.shape[0]
        if self.K is None:
            counts = np.asarray(ld(f"nbr_{variant}_counts")).astype(np.int64)
        else:
            nbr_blk = np.asarray(ld("nbr_k10_rc6"))
            counts = (nbr_blk >= 0).sum(1).astype(np.int64)
        keep = counts > 0                                   # the operators skip zero-neighbour pairs
        fam_all = cfg["family"][pair_cfg]
        if families:
            keep &= np.isin(fam_all, [FAMILIES.index(f) for f in families])
        self.rows = np.nonzero(keep)[0]                     # cache row of every training row
        n = len(self.rows)
        store = self.device if data_on == "gpu" else torch.device("cpu")
        self.store = store
        T = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=store)
        self.positions = T(cfg["positions"], torch.float64)                   # (C, 64, 3)
        self.cfg_P = cfg["P"]; self.cfg_family = cfg["family"]; self.cfg_is_val = cfg["is_val"]; self.cfg_seed = cfg["seed"]
        self.pair_cfg = T(pair_cfg[self.rows], torch.int64)
        self.pair_t = T(np.asarray(ld("pair_t"))[self.rows], torch.int64)
        self.pair_s = T(np.asarray(ld("pair_s"))[self.rows], torch.int64)
        self.dist = np.asarray(ld("pair_dist"))[self.rows].astype(np.float32)
        Mts = np.asarray(ld("Mts_sym"))[self.rows]; M2b = np.asarray(ld("M2b"))[self.rows]
        self.R = T(Mts - M2b, torch.float32)                                   # (n, 36) residual labels
        self.Mts_np = Mts; self.M2b_np = M2b                                    # numpy (eval only)
        self.nnbr = counts[self.rows]
        if self.K is None:
            indptr = np.asarray(ld(f"nbr_{variant}_indptr")).astype(np.int64)
            self.indptr = T(indptr[self.rows], torch.int64)                     # start of each kept row's list
            self.indices = T(np.asarray(ld(f"nbr_{variant}_indices")), torch.int16)
            self.counts = T(counts[self.rows], torch.int64)
        else:
            self.nbr_blk = T(nbr_blk[self.rows], torch.int16)                   # (n, 10), -1 padded
        self.family = fam_all[self.rows]
        self.is_val = cfg["is_val"][pair_cfg[self.rows]]
        self.train_idx = np.nonzero(~self.is_val)[0]
        self.val_idx = np.nonzero(self.is_val)[0]
        self.n = n
        print(f"[cache] {variant}: {n} rows ({n_all - n} dropped) train {len(self.train_idx)} val {len(self.val_idx)} "
              f"configs {len(cfg['P'])} on {store}", flush=True)

    def _idx(self, idx):
        return torch.as_tensor(idx, dtype=torch.int64, device=self.store)

    def gather(self, idx):
        """s_vec (B,3), nbr (B,K,3), mask (B,K) float64 on the compute device, K = max count in the batch."""
        i = self._idx(idx)
        cfg = self.pair_cfg[i]; t = self.pair_t[i]; s = self.pair_s[i]
        pos_t = self.positions[cfg, t]; pos_s = self.positions[cfg, s]
        if self.K is None:
            cnt = self.counts[i]; Km = int(cnt.max().item()) if len(i) else 1
            ar = torch.arange(Km, device=self.store)
            valid = ar[None, :] < cnt[:, None]
            flat = (self.indptr[i][:, None] + ar[None, :]).clamp_(max=self.indices.shape[0] - 1)
            nb = self.indices[flat].long()
        else:
            nb = self.nbr_blk[i].long(); valid = nb >= 0
        nb = torch.where(valid, nb, torch.zeros_like(nb))
        nbr = self.positions[cfg[:, None], nb] - pos_t[:, None, :]
        nbr = torch.where(valid[..., None], nbr, torch.zeros_like(nbr))
        s_vec = pos_s - pos_t
        dev = self.device
        return s_vec.to(dev), nbr.to(dev), valid.to(torch.float64).to(dev)

    def features(self, idx, kind: str) -> torch.Tensor:
        s_vec, nbr, mask = self.gather(idx)
        if kind == "moments":
            return nbm.moment_features(s_vec, nbr, mask, MEAN_DIST_S).to(torch.float32)
        assert self.K == 10, "the baseline layout needs the k10 selection"
        return nf.baseline_features_torch(s_vec, nbr, mask, MEAN_DIST_S)

    def labels(self, idx) -> torch.Tensor:
        return self.R[self._idx(idx)].to(self.device).view(-1, 6, 6)


# ----------------------------------------------------------------------------- metrics
def rel(a, b):
    return float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-300) * 100.0)


def block_metrics(pred, Mts, M2b):
    """pred/Mts/M2b (m,6,6) numpy -> relative Frobenius errors (%) of the total and of the 2b-only operator."""
    tot = M2b + pred
    R = Mts - M2b
    out = {"n": int(len(pred)), "rel_total": rel(tot - Mts, Mts), "rel_2b": rel(M2b - Mts, Mts),
           "capture": rel(pred - R, R), "blocks": {}}
    for k, (i, j) in BLOCKS.items():
        out["blocks"][k] = {"rel_total": rel((tot - Mts)[:, i, j], Mts[:, i, j]), "rel_2b": rel((M2b - Mts)[:, i, j], Mts[:, i, j])}
    return out


def velocity_metrics(pred, Mts, M2b, F):
    """Old-style PRMSE: total velocity (M2b + pred) F vs Mts F, translational / rotational."""
    v = np.einsum("nij,nj->ni", Mts, F); vt = np.einsum("nij,nj->ni", M2b + pred, F); v2 = np.einsum("nij,nj->ni", M2b, F)
    return {"prmse_lin": rel((vt - v)[:, :3], v[:, :3]), "prmse_ang": rel((vt - v)[:, 3:], v[:, 3:]),
            "prmse_all": rel(vt - v, v), "rmse": float(np.sqrt(np.mean((vt - v) ** 2))),
            "twobody_only": {"prmse_lin": rel((v2 - v)[:, :3], v[:, :3]), "prmse_ang": rel((v2 - v)[:, 3:], v[:, 3:]),
                             "prmse_all": rel(v2 - v, v), "rmse": float(np.sqrt(np.mean((v2 - v) ** 2)))}}


def fixed_forces(n, seed=12345, scale=6 * np.pi):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, 3)); f /= np.linalg.norm(f, axis=1, keepdims=True)
    t = rng.normal(size=(n, 3)); t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.concatenate([f, t], 1) * scale


def predict(model, cache: V2Cache, idx, kind: str, chunk: int = 16384) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), chunk):
            X = cache.features(idx[i:i + chunk], kind)
            out.append(nf.predict_blocks(model, X).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


def evaluate(model, cache: V2Cache, idx, kind: str, full: bool = True) -> dict:
    pred = predict(model, cache, idx, kind)
    Mts = cache.Mts_np[idx].reshape(-1, 6, 6).astype(np.float64); M2b = cache.M2b_np[idx].reshape(-1, 6, 6).astype(np.float64)
    F = fixed_forces(len(idx))
    m = {**block_metrics(pred, Mts, M2b), **velocity_metrics(pred, Mts, M2b, F)}
    if not full:
        return m
    fam = cache.family[idx]; nn_ = cache.nnbr[idx]; dist = cache.dist[idx]
    m["by_family"] = {}
    for k, name in enumerate(FAMILIES):
        sel = fam == k
        if sel.any():
            m["by_family"][name] = {**block_metrics(pred[sel], Mts[sel], M2b[sel]), **velocity_metrics(pred[sel], Mts[sel], M2b[sel], F[sel])}
    m["by_nnbr"] = {}
    for lo, hi in NNBR_BINS:
        sel = (nn_ >= lo) & (nn_ <= hi)
        if sel.any():
            m["by_nnbr"][f"{lo}-{hi if hi < 10 ** 6 else 'inf'}"] = {"n": int(sel.sum()), **velocity_metrics(pred[sel], Mts[sel], M2b[sel], F[sel]),
                                                                    "rel_total": rel(M2b[sel] + pred[sel] - Mts[sel], Mts[sel])}
    m["by_dist"] = {}
    for lo, hi in DIST_BINS:
        sel = (dist >= lo) & (dist < hi)
        if sel.any():
            m["by_dist"][f"{lo}-{hi}"] = {"n": int(sel.sum()), **velocity_metrics(pred[sel], Mts[sel], M2b[sel], F[sel])}
    return m


def fmt(m: dict) -> str:
    return (f"PRMSE lin {m['prmse_lin']:.3f}% ang {m['prmse_ang']:.3f}% | block rel {m['rel_total']:.3f}% "
            f"(TT {m['blocks']['TT']['rel_total']:.2f} TR {m['blocks']['TR']['rel_total']:.2f} RR {m['blocks']['RR']['rel_total']:.2f}) "
            f"capture {m['capture']:.1f}% | 2b-only lin {m['twobody_only']['prmse_lin']:.2f}% ang {m['twobody_only']['prmse_ang']:.2f}% "
            f"block {m['rel_2b']:.2f}%")


def load_model(path: str, features: str, device, inv_norm=False):
    if path.endswith(".pt"):
        return torch.jit.load(path, map_location=device).eval()
    model = (MultiBodyMoments(MEAN_DIST_S, inv_norm=inv_norm) if features == "moments" else MultiBodyCorrectionB1(MEAN_DIST_S)).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("data/multibody_v2_cache"))
    ap.add_argument("--variant", choices=list(nf.SELECTION_VARIANTS), default="k10_rc6")
    ap.add_argument("--model", choices=["moments", "baseline"], default="moments")
    ap.add_argument("--features", choices=["moments", "baseline"], default=None, help="(eval-only) feature layout")
    ap.add_argument("--eval-only", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["l1", "l2"], default="l1")
    ap.add_argument("--block-weights", choices=["none", "rms"], default="none")
    ap.add_argument("--loss-form", choices=["block", "velocity"], default="block")
    ap.add_argument("--scale", type=float, default=6 * math.pi)
    ap.add_argument("--seed", type=int, default=411)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-on", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--fit-rows", type=int, default=262144)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-rows", type=int, default=500000)
    ap.add_argument("--families", nargs="+", default=None, choices=FAMILIES)
    ap.add_argument("--family-weights", type=float, nargs=3, default=None, metavar=("W_UNIFORM", "W_GROWN", "W_LATTICE"),
                    help="relative sampling weight per row of each family (default: uniform over rows)")
    ap.add_argument("--inv-norm", action="store_true")
    ap.add_argument("--zero-init-head", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--publish-name", default=None)
    args = ap.parse_args()
    features = args.features or args.model
    if args.eval_only and args.features is None:
        ap.error("--eval-only requires --features")
    if features == "baseline":
        assert args.variant == "k10_rc6", "the baseline (147-column) layout is only defined for the k10_rc6 selection"
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, open(args.out / "config.json", "w"), indent=1)

    t0 = time.time()
    cache = V2Cache(args.cache, args.variant, device, args.data_on, args.families)
    rng = np.random.default_rng(args.seed)
    val_all = cache.val_idx
    val_sub = np.sort(rng.choice(val_all, size=min(args.eval_rows, len(val_all)), replace=False))
    print(f"[data] loaded in {time.time() - t0:.0f} s; periodic eval on {len(val_sub)} val rows, final on {len(val_all)}", flush=True)

    # ---------------------------------------------------------------- eval-only
    if args.eval_only:
        model = load_model(args.eval_only, features, device, args.inv_norm)
        t0 = time.time()
        m = evaluate(model, cache, val_all, features)
        m.update({"model_path": args.eval_only, "features": features, "variant": args.variant, "eval_s": time.time() - t0})
        print(f"[eval-only] {args.eval_only} ({args.variant}): {fmt(m)}")
        for name, mm in m["by_family"].items():
            print(f"   {name:8s} {fmt(mm)}")
        json.dump(m, open(args.out / "metrics.json", "w"), indent=1)
        return

    # ---------------------------------------------------------------- model + recipe
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    zero_init = (args.model == "moments") if args.zero_init_head == "auto" else (args.zero_init_head == "on")
    model = (MultiBodyMoments(MEAN_DIST_S, zero_init_head=zero_init, inv_norm=args.inv_norm) if args.model == "moments"
             else MultiBodyCorrectionB1(MEAN_DIST_S, zero_init_head=zero_init)).to(device)
    train_idx = cache.train_idx
    fit_idx = np.sort(rng.choice(train_idx, size=min(args.fit_rows, len(train_idx)), replace=False))
    if args.model == "moments":
        Xfit = torch.cat([cache.features(fit_idx[i:i + 16384], features) for i in range(0, len(fit_idx), 16384)], 0)
        model.fit_normalisation(Xfit)
        print(f"[norm] fitted on {len(fit_idx)} rows: inv_std range [{model.inv_std.min():.3g}, {model.inv_std.max():.3g}]  "
              f"basis_scale range [{model.basis_scale.min():.3g}, {model.basis_scale.max():.3g}]", flush=True)
        del Xfit
    bw = torch.ones((6, 6), device=device)
    if args.block_weights == "rms":
        Rf = cache.R[torch.as_tensor(fit_idx, device=cache.store)].to(device).view(-1, 6, 6)
        for k, (i, j) in BLOCKS.items():
            bw[i, j] = 1.0 / Rf[:, i, j].pow(2).mean().sqrt().clamp_min(1e-9)
        bw = bw / bw[:3, :3].mean()
        print(f"[loss] block weights TT {bw[0, 0]:.2f} TR {bw[0, 3]:.2f} RR {bw[3, 3]:.2f}")
    weights = None
    if args.family_weights is not None:
        w = np.asarray(args.family_weights, dtype=np.float64)[cache.family[train_idx]]
        weights = torch.as_tensor(w / w.sum(), dtype=torch.float64, device=device)
    n_train = len(train_idx)
    steps_per_epoch = max(1, n_train // args.batch)
    total_steps = args.epochs * steps_per_epoch if args.max_steps is None else min(args.max_steps, args.epochs * steps_per_epoch)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    n_params = sum(p.numel() for p in model.parameters())
    gen = torch.Generator(device=device).manual_seed(args.seed)
    train_idx_t = torch.as_tensor(train_idx, device=device)
    print(f"[train] model={args.model} variant={args.variant} params={n_params} rows={n_train} batch={args.batch} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} loss={args.loss}/{args.loss_form} scale={args.scale:.3f}", flush=True)

    def eval_now(idx, full):
        model.eval(); m = evaluate(model, cache, idx, features, full); model.train(); return m

    log = []; step = 0; t_start = time.time(); epoch = 0; avg = 0.0; n_avg = 0
    while step < total_steps:
        if weights is None:
            perm = train_idx_t[torch.randperm(n_train, generator=gen, device=device)]
        else:
            perm = train_idx_t[torch.multinomial(weights, n_train, replacement=True, generator=gen)]
        model.train()
        for it in range(steps_per_epoch):
            if step >= total_steps:
                break
            idx = perm[it * args.batch:(it + 1) * args.batch]
            X = cache.features(idx, features)
            R = cache.labels(idx)
            optimizer.zero_grad(set_to_none=True)
            if args.loss_form == "block":
                diff = (model.predict_mobility(X) - R) * bw
            else:
                F = torch.randn((len(idx), 6), device=device, generator=gen)
                F = F / F.norm(dim=1, keepdim=True).clamp_min(1e-12)
                diff = model.predict_velocity(X, F) - torch.einsum("bij,bj->bi", R, F)
            diff = diff * args.scale
            loss = diff.abs().mean() if args.loss == "l1" else diff.pow(2).mean()
            loss.backward()
            optimizer.step(); scheduler.step()
            avg += loss.item(); n_avg += 1; step += 1
        row = {"epoch": epoch, "step": step, "train_loss": avg / max(n_avg, 1), "lr": scheduler.get_last_lr()[0], "time_s": time.time() - t_start}
        avg = 0.0; n_avg = 0
        if epoch % args.eval_every == 0 or step >= total_steps:
            torch.save(model.state_dict(), args.out / "model.ckpt.wt")  # rolling checkpoint: a killed run resumes via --eval-only
            m = eval_now(val_sub, False)
            row.update({"val_prmse_lin": m["prmse_lin"], "val_prmse_ang": m["prmse_ang"], "val_rel_total": m["rel_total"], "val_capture": m["capture"]})
            print(f"epoch {epoch:4d} step {step:7d}  train {row['train_loss']:.5f}  val {fmt(m)}  [{row['time_s']:.0f} s]", flush=True)
        log.append(row); epoch += 1

    # ---------------------------------------------------------------- final eval + save
    torch.save(model.state_dict(), args.out / "model.wt")  # checkpoint before the (long) final eval: a kill there must not lose the run
    model.eval()
    m = evaluate(model, cache, val_all, features)
    m.update({"model": args.model, "features": features, "variant": args.variant, "max_neighbors": cache.K, "neighbor_cutoff": cache.cutoff,
              "epochs": epoch, "steps": step, "batch": args.batch, "lr": args.lr, "loss": args.loss, "loss_form": args.loss_form,
              "block_weights": args.block_weights, "scale": args.scale, "seed": args.seed, "n_params": n_params, "n_train": n_train,
              "n_val": int(len(val_all)), "zero_init_head": zero_init, "inv_norm": args.inv_norm, "families": args.families,
              "family_weights": args.family_weights, "train_time_s": time.time() - t_start, "device": str(device), "cache": str(args.cache)})
    print(f"[final] {fmt(m)}")
    for name, mm in m["by_family"].items():
        print(f"   {name:8s} {fmt(mm)}")
    print("   by nnbr: " + "  ".join(f"{k}:{v['prmse_lin']:.2f}/{v['prmse_ang']:.2f}%" for k, v in m["by_nnbr"].items()))
    json.dump(m, open(args.out / "metrics.json", "w"), indent=1)
    with open(args.out / "log.csv", "w", newline="") as f:
        keys = ["epoch", "step", "train_loss", "lr", "val_prmse_lin", "val_prmse_ang", "val_rel_total", "val_capture", "time_s"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in log:
            w.writerow({k: r.get(k, "") for k in keys})
    torch.save(model.state_dict(), args.out / "model.wt")
    scripted = torch.jit.script(model.cpu().eval())
    scripted.save(str(args.out / "model.pt"))
    chk = torch.jit.load(str(args.out / "model.pt")).eval()
    Xc = cache.features(val_all[:64], features).cpu()
    with torch.no_grad():
        a = model.predict_mobility(Xc); b = chk.predict_mobility(Xc)
    assert torch.allclose(a, b, atol=1e-6), "TorchScript export mismatch"
    print(f"[saved] {args.out}/model.wt, model.pt, metrics.json, log.csv, config.json")
    if args.publish:
        name = args.publish_name or PUBLISH[(args.model, args.variant)]
        shutil.copy(args.out / "model.pt", Path("data/models") / f"{name}.pt")
        shutil.copy(args.out / "model.wt", Path("experiments") / f"{name}.wt")
        side = {"name": name, "model": args.model, "features": features, "variant": args.variant, "max_neighbors": cache.K,
                "neighbor_cutoff": cache.cutoff, "pair_cutoff": cache.meta["pair_cutoff"], "mean_dist_s": MEAN_DIST_S,
                "median_2b": cache.meta["median_2b"], "run": str(args.out), "prmse_lin": m["prmse_lin"], "prmse_ang": m["prmse_ang"],
                "rel_total": m["rel_total"], "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(side, open(Path("data/models") / f"{name}.json", "w"), indent=1)
        print(f"[published] data/models/{name}.pt (+ .json sidecar), experiments/{name}.wt")


if __name__ == "__main__":
    main()
