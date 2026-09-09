#!/usr/bin/env python3
"""Train / evaluate the per-particle diagonal (self-block) correction on the dataset-v2 pc8 cache.

    python experiments/train_diag_v2.py --out experiments/runs_v2/diag_pc8 --publish
    python experiments/train_diag_v2.py --inv-norm --out experiments/runs_v2/diag_pc8_invnorm
    python experiments/train_diag_v2.py --eval-only data/models/nbody_diag_v2_pc8.pt --out experiments/runs_v2/eval_diag

Rows = every particle of every configuration (zero-neighbour particles are kept: the operator evaluates all N
particles, so the constant coefficients must be anchored at zero moments).  Features are built on the fly with
the operators' own code paths (nbody_features.select_particle_neighbours -> nbody_moments.self_moment_features:
all k != t within --cutoff of the particle, band moments on [2, 8]).  The label is the symmetrised diagonal
residual R = 0.5 (Mtt_res + Mtt_res^T) with Mtt_res = M_tt - diag(1/6pi, 1/8pi) - sum_{d <= pair_cutoff} K_s
(the cache's convention, experiments/build_nbody_v2_cache.py), so a model trained on the pc8 cache must run
with pair_cutoff = switch_dist = 8 -- the published sidecar records this and the harness asserts it.
Split = configuration level (configs.is_val, seed % 10 == 0).  Loss (default): L1 over the 36 entries of
6*pi*(predict_mobility(X) - R); Adam + cosine over all steps; normalisation buffers fitted on a train
subsample.  Outputs: metrics.json, log.csv, config.json, model.wt, model.pt (TorchScript); --publish copies
model.pt to data/models/<name>.pt with a <name>.json sidecar (pair_cutoff, diag_cutoff, band layout).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if p != str(ROOT)]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src import nbody_features as nf  # noqa: E402
from src import nbody_moments as nbm  # noqa: E402
from src.model_archs import SelfBlockMoments  # noqa: E402
from experiments.train_nbody_v2 import BLOCKS, FAMILIES, fixed_forces, rel  # noqa: E402

DEFAULT_CACHE = Path("data/multibody_v2_cache_pc8")
PUBLISH_NAME = "nbody_diag_v2_pc8"
NNBR_BINS = [(0, 0), (1, 5), (6, 15), (16, 30), (31, 10 ** 6)]


# ----------------------------------------------------------------------------- cache on device
class DiagCache:
    """Per-particle rows of the v2 cache as device tensors + on-the-fly feature construction."""

    def __init__(self, root: Path, device, data_on: str = "gpu", families=None, cutoff: float = 8.0,
                 label_base: str = "none"):
        self.root = Path(root)
        self.device = torch.device(device)
        self.cutoff = float(cutoff)
        self.label_base = str(label_base)
        cfg = np.load(self.root / "configs.npz")
        self.meta = json.load(open(self.root / "meta.json"))
        positions_np = cfg["positions"]                          # materialise once: npz re-extracts per access
        Mtt = np.load(self.root / "Mtt_res.npy", mmap_mode="r")
        C, PMAX = Mtt.shape[0], Mtt.shape[1]
        P = cfg["P"].astype(np.int64)
        valid = np.arange(PMAX)[None, :] < P[:, None]
        row_cfg, row_p = np.nonzero(valid)                       # row-major: by config, then particle
        keep = np.ones(len(row_cfg), dtype=bool)
        if families:
            keep &= np.isin(cfg["family"][row_cfg], [FAMILIES.index(f) for f in families])
        row_cfg, row_p = row_cfg[keep], row_p[keep]
        n = len(row_cfg)

        # labels: symmetrised diagonal residual (the model output is symmetric by construction)
        A = np.asarray(Mtt[valid])[keep].reshape(n, 6, 6).astype(np.float32)
        if self.label_base != "none":
            # base = analytic self + K_s over d <= pair_cutoff + the in-configuration stresslet reflection
            assert self.label_base in self.meta.get("fts", {}), f"{self.root}: no {self.label_base} reflection arrays"
            Mref = np.load(self.root / f"Mref_{self.label_base}_tt.npy", mmap_mode="r")
            A = A - np.asarray(Mref[valid])[keep].reshape(n, 6, 6).astype(np.float32)
        R = 0.5 * (A + np.transpose(A, (0, 2, 1)))

        # per-particle neighbour CSR through the shared selection code path, config by config
        t0 = time.time()
        cfgs_used = np.unique(row_cfg)
        counts_all = np.zeros(len(row_cfg), dtype=np.int64)
        ind_parts = []
        base = np.searchsorted(row_cfg, cfgs_used)               # first row of each used config
        for k, c in enumerate(cfgs_used):
            pc = int(P[c])
            indptr_c, indices_c = nf.select_particle_neighbours(positions_np[c, :pc], self.cutoff)
            counts_all[base[k]:base[k] + pc] = np.diff(indptr_c)
            ind_parts.append(indices_c)
        indices = np.concatenate(ind_parts) if ind_parts else np.zeros(0, dtype=np.int16)
        indptr = np.zeros(n + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(counts_all)
        print(f"[csr] {n} particle rows, {len(indices)} neighbour entries "
              f"(mean {len(indices) / max(n, 1):.1f}/particle) in {time.time() - t0:.1f} s", flush=True)

        store = self.device if data_on == "gpu" else torch.device("cpu")
        self.store = store
        T = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=store)
        self.positions = T(positions_np, torch.float64)          # (C, 64, 3)
        self.row_cfg = T(row_cfg, torch.int64)
        self.row_p = T(row_p, torch.int64)
        self.R = T(R.reshape(n, 36), torch.float32)
        self.counts = T(counts_all, torch.int64)
        self.indptr = T(indptr[:-1], torch.int64)                # start of each row's list
        self.indices = T(indices, torch.int16)
        self.nnbr = counts_all
        self.family = cfg["family"][row_cfg]
        self.cfg_P = P[row_cfg]
        self.is_val = cfg["is_val"][row_cfg]
        self.train_idx = np.nonzero(~self.is_val)[0]
        self.val_idx = np.nonzero(self.is_val)[0]
        self.n = n
        print(f"[cache] diag cutoff {self.cutoff}: {n} rows train {len(self.train_idx)} val {len(self.val_idx)} "
              f"configs {C} pair_cutoff {self.meta['pair_cutoff']} on {store}", flush=True)

    def _idx(self, idx):
        return torch.as_tensor(idx, dtype=torch.int64, device=self.store)

    def gather(self, idx):
        """nbr (B,K,3), mask (B,K) float64 on the compute device, K = max count in the batch."""
        i = self._idx(idx)
        cfg = self.row_cfg[i]
        pos_t = self.positions[cfg, self.row_p[i]]
        cnt = self.counts[i]
        Km = max(int(cnt.max().item()) if len(i) else 1, 1)
        ar = torch.arange(Km, device=self.store)
        valid = ar[None, :] < cnt[:, None]
        if self.indices.shape[0]:
            flat = (self.indptr[i][:, None] + ar[None, :]).clamp_(max=self.indices.shape[0] - 1)
            nb = self.indices[flat].long()
        else:
            nb = torch.zeros((len(i), Km), dtype=torch.int64, device=self.store)
        nb = torch.where(valid, nb, torch.zeros_like(nb))
        nbr = self.positions[cfg[:, None], nb] - pos_t[:, None, :]
        nbr = torch.where(valid[..., None], nbr, torch.zeros_like(nbr))
        dev = self.device
        return nbr.to(dev), valid.to(torch.float64).to(dev)

    def features(self, idx) -> torch.Tensor:
        nbr, mask = self.gather(idx)
        return nbm.self_moment_features(nbr, mask).to(torch.float32)

    def labels(self, idx) -> torch.Tensor:
        return self.R[self._idx(idx)].to(self.device).view(-1, 6, 6)


# ----------------------------------------------------------------------------- metrics
def diag_metrics(pred, R):
    """pred/R (m,6,6) numpy float64 -> residual capture (all / per block) and velocity-space capture."""
    d = pred - R
    m = {"n": int(len(pred)), "capture": rel(d, R),
         "rms_label": {k: float(np.sqrt(np.mean(R[:, i, j] ** 2))) for k, (i, j) in BLOCKS.items()},
         "rms_resid": {k: float(np.sqrt(np.mean(d[:, i, j] ** 2))) for k, (i, j) in BLOCKS.items()},
         "capture_blocks": {k: rel(d[:, i, j], R[:, i, j]) for k, (i, j) in BLOCKS.items()}}
    F = fixed_forces(len(pred))
    v = np.einsum("nij,nj->ni", R, F)
    vp = np.einsum("nij,nj->ni", pred, F)
    m["vel_capture"] = rel(vp - v, v)
    m["vel_capture_lin"] = rel((vp - v)[:, :3], v[:, :3])
    m["vel_capture_ang"] = rel((vp - v)[:, 3:], v[:, 3:])
    return m


def predict(model, cache: DiagCache, idx, chunk: int = 16384) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), chunk):
            X = cache.features(idx[i:i + chunk])
            out.append(model.predict_mobility(X).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


def evaluate(model, cache: DiagCache, idx, full: bool = True) -> dict:
    pred = predict(model, cache, idx)
    R = cache.R[cache._idx(idx)].cpu().numpy().reshape(-1, 6, 6).astype(np.float64)
    m = diag_metrics(pred, R)
    if not full:
        return m
    fam = cache.family[idx]
    nn_ = cache.nnbr[idx]
    m["by_family"] = {}
    for k, name in enumerate(FAMILIES):
        sel = fam == k
        if sel.any():
            m["by_family"][name] = diag_metrics(pred[sel], R[sel])
    m["by_nnbr"] = {}
    for lo, hi in NNBR_BINS:
        sel = (nn_ >= lo) & (nn_ <= hi)
        if sel.any():
            m["by_nnbr"][f"{lo}-{hi if hi < 10 ** 6 else 'inf'}"] = {"n": int(sel.sum()),
                                                                     "capture": rel(pred[sel] - R[sel], R[sel])}
    return m


def fmt(m: dict) -> str:
    cb = m["capture_blocks"]
    return (f"capture {m['capture']:.1f}% (TT {cb['TT']:.1f} TR {cb['TR']:.1f} RR {cb['RR']:.1f}) | "
            f"vel capture {m['vel_capture']:.1f}% lin {m['vel_capture_lin']:.1f}% ang {m['vel_capture_ang']:.1f}%")


def load_model(path: str, device, inv_norm=False):
    if path.endswith(".pt"):
        return torch.jit.load(path, map_location=device).eval()
    model = SelfBlockMoments(inv_norm=inv_norm).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--cutoff", type=float, default=8.0, help="particle neighbour cutoff (selection radius)")
    ap.add_argument("--label-base", choices=["none", "refl1", "refl2"], default="none",
                    help="subtract the cache's stresslet-reflection diagonal blocks from the labels (needs --add-fts)")
    ap.add_argument("--eval-only", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["l1", "l2"], default="l1")
    ap.add_argument("--scale", type=float, default=6 * math.pi)
    ap.add_argument("--seed", type=int, default=411)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-on", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--fit-rows", type=int, default=262144)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-rows", type=int, default=500000)
    ap.add_argument("--families", nargs="+", default=None, choices=FAMILIES)
    ap.add_argument("--inv-norm", action="store_true")
    ap.add_argument("--zero-init-head", choices=["on", "off"], default="on")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--publish-name", default=None)
    args = ap.parse_args()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              open(args.out / "config.json", "w"), indent=1)

    t0 = time.time()
    cache = DiagCache(args.cache, device, args.data_on, args.families, args.cutoff, args.label_base)
    rng = np.random.default_rng(args.seed)
    val_all = cache.val_idx
    val_sub = np.sort(rng.choice(val_all, size=min(args.eval_rows, len(val_all)), replace=False))
    print(f"[data] loaded in {time.time() - t0:.0f} s; periodic eval on {len(val_sub)} val rows, "
          f"final on {len(val_all)}", flush=True)

    # ---------------------------------------------------------------- eval-only
    if args.eval_only:
        model = load_model(args.eval_only, device, args.inv_norm)
        t0 = time.time()
        m = evaluate(model, cache, val_all)
        m.update({"model_path": args.eval_only, "cutoff": args.cutoff, "eval_s": time.time() - t0})
        print(f"[eval-only] {args.eval_only}: {fmt(m)}")
        for name, mm in m["by_family"].items():
            print(f"   {name:8s} {fmt(mm)}")
        json.dump(m, open(args.out / "metrics.json", "w"), indent=1)
        return

    # ---------------------------------------------------------------- model + recipe
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = SelfBlockMoments(zero_init_head=args.zero_init_head == "on", inv_norm=args.inv_norm).to(device)
    train_idx = cache.train_idx
    fit_idx = np.sort(rng.choice(train_idx, size=min(args.fit_rows, len(train_idx)), replace=False))
    Xfit = torch.cat([cache.features(fit_idx[i:i + 16384]) for i in range(0, len(fit_idx), 16384)], 0)
    model.fit_normalisation(Xfit)
    print(f"[norm] fitted on {len(fit_idx)} rows: inv_std range [{model.inv_std.min():.3g}, {model.inv_std.max():.3g}]  "
          f"basis_scale range [{model.basis_scale.min():.3g}, {model.basis_scale.max():.3g}]", flush=True)
    del Xfit
    n_train = len(train_idx)
    steps_per_epoch = max(1, n_train // args.batch)
    total_steps = args.epochs * steps_per_epoch if args.max_steps is None else min(args.max_steps, args.epochs * steps_per_epoch)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    n_params = sum(p.numel() for p in model.parameters())
    gen = torch.Generator(device=device).manual_seed(args.seed)
    train_idx_t = torch.as_tensor(train_idx, device=device)
    print(f"[train] diag cutoff={args.cutoff} params={n_params} rows={n_train} batch={args.batch} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} loss={args.loss} scale={args.scale:.3f}", flush=True)

    def eval_now(idx, full):
        model.eval()
        m = evaluate(model, cache, idx, full)
        model.train()
        return m

    log = []
    step = 0
    t_start = time.time()
    epoch = 0
    avg = 0.0
    n_avg = 0
    while step < total_steps:
        perm = train_idx_t[torch.randperm(n_train, generator=gen, device=device)]
        model.train()
        for it in range(steps_per_epoch):
            if step >= total_steps:
                break
            idx = perm[it * args.batch:(it + 1) * args.batch]
            X = cache.features(idx)
            R = cache.labels(idx)
            optimizer.zero_grad(set_to_none=True)
            diff = (model.predict_mobility(X) - R) * args.scale
            loss = diff.abs().mean() if args.loss == "l1" else diff.pow(2).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg += loss.item()
            n_avg += 1
            step += 1
        row = {"epoch": epoch, "step": step, "train_loss": avg / max(n_avg, 1), "lr": scheduler.get_last_lr()[0],
               "time_s": time.time() - t_start}
        avg = 0.0
        n_avg = 0
        if epoch % args.eval_every == 0 or step >= total_steps:
            torch.save(model.state_dict(), args.out / "model.ckpt.wt")  # rolling checkpoint
            m = eval_now(val_sub, False)
            row.update({"val_capture": m["capture"], "val_vel_capture": m["vel_capture"],
                        "val_vel_lin": m["vel_capture_lin"], "val_vel_ang": m["vel_capture_ang"]})
            print(f"epoch {epoch:4d} step {step:7d}  train {row['train_loss']:.5f}  val {fmt(m)}  "
                  f"[{row['time_s']:.0f} s]", flush=True)
        log.append(row)
        epoch += 1

    # ---------------------------------------------------------------- final eval + save
    torch.save(model.state_dict(), args.out / "model.wt")
    model.eval()
    m = evaluate(model, cache, val_all)
    m.update({"model": "diag_moments", "cutoff": args.cutoff, "epochs": epoch, "steps": step, "batch": args.batch,
              "lr": args.lr, "loss": args.loss, "scale": args.scale, "seed": args.seed, "n_params": n_params,
              "n_train": n_train, "n_val": int(len(val_all)), "zero_init_head": args.zero_init_head == "on",
              "inv_norm": args.inv_norm, "families": args.families, "train_time_s": time.time() - t_start,
              "device": str(device), "cache": str(args.cache), "pair_cutoff": cache.meta["pair_cutoff"]})
    print(f"[final] {fmt(m)}")
    for name, mm in m["by_family"].items():
        print(f"   {name:8s} {fmt(mm)}")
    print("   by nnbr: " + "  ".join(f"{k}:{v['capture']:.1f}%" for k, v in m["by_nnbr"].items()))
    json.dump(m, open(args.out / "metrics.json", "w"), indent=1)
    with open(args.out / "log.csv", "w", newline="") as f:
        keys = ["epoch", "step", "train_loss", "lr", "val_capture", "val_vel_capture", "val_vel_lin", "val_vel_ang", "time_s"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in log:
            w.writerow({k: r.get(k, "") for k in keys})
    torch.save(model.state_dict(), args.out / "model.wt")
    scripted = torch.jit.script(model.cpu().eval())
    scripted.save(str(args.out / "model.pt"))
    chk = torch.jit.load(str(args.out / "model.pt")).eval()
    Xc = cache.features(val_all[:64]).cpu()
    with torch.no_grad():
        a = model.predict_mobility(Xc)
        b = chk.predict_mobility(Xc)
    assert torch.allclose(a, b, atol=1e-6), "TorchScript export mismatch"
    print(f"[saved] {args.out}/model.wt, model.pt, metrics.json, log.csv, config.json")
    if args.publish:
        name = args.publish_name or PUBLISH_NAME
        shutil.copy(args.out / "model.pt", Path("data/models") / f"{name}.pt")
        shutil.copy(args.out / "model.wt", Path("experiments") / f"{name}.wt")
        side = {"name": name, "model": "diag_moments", "features": "diag_moments",
                "pair_cutoff": cache.meta["pair_cutoff"], "diag_cutoff": args.cutoff, "fts_base": args.label_base,
                "nb": nbm.NB_SELF, "band_lo": nbm.BAND_LO, "band_hi": nbm.BAND_HI,
                "cache": str(args.cache), "run": str(args.out), "capture": m["capture"],
                "vel_capture_lin": m["vel_capture_lin"], "vel_capture_ang": m["vel_capture_ang"],
                "inv_norm": args.inv_norm, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(side, open(Path("data/models") / f"{name}.json", "w"), indent=1)
        print(f"[published] data/models/{name}.pt (+ .json sidecar), experiments/{name}.wt")


if __name__ == "__main__":
    main()
