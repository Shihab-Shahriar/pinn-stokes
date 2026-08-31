#!/usr/bin/env python3
"""Train / evaluate n-body correction models on the existing multibody datasets.

    python experiments/train_nbody_moments.py --model moments  --out experiments/runs/moments_s411
    python experiments/train_nbody_moments.py --model baseline --out experiments/runs/baseline_s411
    python experiments/train_nbody_moments.py --eval-only data/models/nbody_pinn_b1.pt --features baseline \\
                                              --out experiments/runs/b1_shipped

Recipe = experiments/branch1_multibody_pinn.ipynb: residual labels y - v2b (two-body TorchScript
model, evaluated in the *operator's* +s_vec convention -- see nbody_features.two_body_velocity; the
saved notebook flips the RT sign), L1 loss, Adam 1e-3, CosineAnnealingLR(T_max=epochs), 500 epochs,
batch 256, 80/20 split.
The split is deterministic (sorted file order + numpy Generator(split_seed)) and saved to
<out>/split.npz so every model is compared on identical validation rows.  Outputs: metrics.json,
log.csv, split.npz, model.wt (state_dict incl. buffers), model.pt (TorchScript).  --publish copies
model.wt to experiments/<name>.wt and model.pt to data/models/<name>.pt.
"""
from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT))  # this repo's src/ must shadow any other `src` package on the path
os.chdir(ROOT)

from src import nbody_features as nf  # noqa: E402
from src.model_archs import MultiBodyCorrectionB1, MultiBodyMoments  # noqa: E402

TWO_BODY_PATH = "data/models/two_body_combined_model.pt"
PUBLISH = {"moments": ("nbody_moments.wt", "nbody_moments.pt"),
           "baseline": ("nbody_b1_retrained.wt", "nbody_pinn_b1_retrained.pt")}
COMPS = ["Ux", "Uy", "Uz", "Ox", "Oy", "Oz"]


def build_features(kind: str, data: dict, mean_dist_s: float) -> np.ndarray:
    if kind == "moments":
        return nf.moment_features(data["s_vec"], data["nbr"], data["mask"], mean_dist_s)
    if kind == "baseline":
        return nf.baseline_features(data["s_vec"], data["nbr"], data["mask"], mean_dist_s)
    raise ValueError(kind)


def rel_l2(err: np.ndarray, ref: np.ndarray, axis=None) -> np.ndarray:
    return np.linalg.norm(err, axis=axis) / np.maximum(np.linalg.norm(ref, axis=axis), 1e-300) * 100.0


def metrics(total: np.ndarray, Y: np.ndarray, v2b: np.ndarray, counts: np.ndarray) -> dict:
    """The notebook's validation metrics (errors of 2b-only and 2b + nbody against the MFS targets)."""
    err = total - Y
    err2b = v2b - Y
    res = Y - v2b
    m = {
        "n": int(len(Y)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "rmse_comp": np.sqrt(np.mean(err ** 2, axis=0)).tolist(),
        "mae_comp": np.mean(np.abs(err), axis=0).tolist(),
        "prmse_comp": rel_l2(err, Y, axis=0).tolist(),
        "prmse_lin": float(rel_l2(err[:, :3], Y[:, :3])),
        "prmse_ang": float(rel_l2(err[:, 3:], Y[:, 3:])),
        "prmse_all": float(rel_l2(err, Y)),
        "twobody_only": {
            "rmse": float(np.sqrt(np.mean(err2b ** 2))),
            "prmse_comp": rel_l2(err2b, Y, axis=0).tolist(),
            "prmse_lin": float(rel_l2(err2b[:, :3], Y[:, :3])),
            "prmse_ang": float(rel_l2(err2b[:, 3:], Y[:, 3:])),
        },
        "residual": {  # how well the correction itself is predicted
            "rmse_comp": np.sqrt(np.mean(err ** 2, axis=0)).tolist(),
            "magnitude_comp": np.mean(np.abs(res), axis=0).tolist(),
            "prmse_lin": float(rel_l2(err[:, :3], res[:, :3])),
            "prmse_ang": float(rel_l2(err[:, 3:], res[:, 3:])),
        },
        "target_magnitude_comp": np.mean(np.abs(Y), axis=0).tolist(),
        "prmse_by_K": {},
    }
    for k in np.unique(counts).tolist():
        sel = counts == k
        m["prmse_by_K"][str(int(k))] = {"n": int(sel.sum()), "prmse": float(rel_l2(err[sel], Y[sel])),
                                        "prmse_2b": float(rel_l2(err2b[sel], Y[sel]))}
    return m


def predict_residual(model, X: torch.Tensor, F: torch.Tensor, chunk: int = 8192) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            out.append(model.predict_velocity(X[i:i + chunk], F[i:i + chunk]).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


def fmt(m: dict) -> str:
    return (f"RMSE {m['rmse']:.5f} | PRMSE lin {m['prmse_lin']:.3f}% ang {m['prmse_ang']:.3f}% "
            f"(2b-only lin {m['twobody_only']['prmse_lin']:.2f}% ang {m['twobody_only']['prmse_ang']:.2f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["moments", "baseline"], default="moments")
    ap.add_argument("--features", choices=["moments", "baseline"], default=None,
                    help="feature layout (default: implied by --model; required with --eval-only)")
    ap.add_argument("--eval-only", type=str, default=None, help="evaluate this .pt/.wt on the val split")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=411)
    ap.add_argument("--split-seed", type=int, default=41)
    ap.add_argument("--split-frac", type=float, default=0.8)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--no-check", action="store_true", help="skip the dataset overlap sanity check")
    ap.add_argument("--inv-norm", action="store_true",
                    help="moments model: MLP sees invariants of count-normalised moments (extrapolation in K)")
    ap.add_argument("--zero-init-head", choices=["auto", "on", "off"], default="auto",
                    help="zero-initialise the output layer (auto: on for moments, off for baseline = notebook recipe)")
    args = ap.parse_args()
    features = args.features or args.model
    if args.eval_only and args.features is None:
        ap.error("--eval-only requires --features")
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    t0 = time.time()
    data = nf.load_multibody_dataset(check=not args.no_check)
    N = len(data["Y"])
    mean_dist_s = nf.MEAN_DIST_S
    assert abs(data["mean_dist_s"] - mean_dist_s) < 1e-6, data["mean_dist_s"]
    X = build_features(features, data, mean_dist_s)
    two_nn = torch.jit.load(TWO_BODY_PATH, map_location=device).eval()
    v2b = nf.two_body_velocity(two_nn, data["s_vec"], data["dist"], data["force"], device=device)
    Y = data["Y"]
    y_res = Y - v2b
    train_idx, val_idx = nf.make_split(N, args.split_frac, args.split_seed)
    np.savez(args.out / "split.npz", train_idx=train_idx, val_idx=val_idx, split_seed=args.split_seed,
             files=np.array([f"{r}/{t}" for r, t in data["files"]]))
    print(f"[data] N={N} features={features} X.shape={X.shape} train={len(train_idx)} val={len(val_idx)} "
          f"|res| lin {np.abs(y_res[:, :3]).mean():.4f} ang {np.abs(y_res[:, 3:]).mean():.4f} "
          f"({time.time() - t0:.1f} s)", flush=True)

    tt = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    Xtr, Ftr, Rtr = tt(X[train_idx]), tt(data["force"][train_idx]), tt(y_res[train_idx])
    Xva, Fva = tt(X[val_idx]), tt(data["force"][val_idx])
    Yva, v2b_va, cva = Y[val_idx], v2b[val_idx], data["counts"][val_idx]

    def evaluate(model):
        pred = predict_residual(model, Xva, Fva)
        return metrics(v2b_va + pred, Yva, v2b_va, cva)

    # ---------------------------------------------------------------- eval-only
    if args.eval_only:
        path = args.eval_only
        if path.endswith(".pt"):
            model = torch.jit.load(path, map_location=device).eval()
        else:
            model = (MultiBodyMoments(mean_dist_s, inv_norm=args.inv_norm) if features == "moments"
                     else MultiBodyCorrectionB1(mean_dist_s)).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
        m = evaluate(model)
        m["model_path"] = path
        m["features"] = features
        m["note"] = ("shipped model: trained on an unknown 80% split of the same rows -> validation rows "
                     "are partly in its training set (contaminated)" if "nbody_pinn_b1.pt" in path else "")
        print(f"[eval-only] {path}: {fmt(m)}")
        json.dump(m, open(args.out / "metrics.json", "w"), indent=2)
        return

    # ---------------------------------------------------------------- model + recipe
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    zero_init = (args.model == "moments") if args.zero_init_head == "auto" else (args.zero_init_head == "on")
    model = (MultiBodyMoments(mean_dist_s, zero_init_head=zero_init, inv_norm=args.inv_norm) if args.model == "moments"
             else MultiBodyCorrectionB1(mean_dist_s, zero_init_head=zero_init)).to(device)
    if args.model == "moments":
        model.fit_normalisation(Xtr)
        print(f"[norm] inv_std range [{model.inv_std.min():.3g}, {model.inv_std.max():.3g}]  "
              f"basis_scale range [{model.basis_scale.min():.3g}, {model.basis_scale.max():.3g}]")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.L1Loss()
    n_train = Xtr.shape[0]
    n_batches = n_train // args.batch
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model={args.model} params={n_params} epochs={args.epochs} batches/epoch={n_batches} device={device}",
          flush=True)

    log = []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, generator=gen).to(device)
        avg = 0.0
        for it in range(n_batches):
            idx = perm[it * args.batch:(it + 1) * args.batch]
            optimizer.zero_grad()
            pred = model.predict_velocity(Xtr[idx], Ftr[idx])
            loss = criterion(pred, Rtr[idx])
            loss.backward()
            optimizer.step()
            avg += loss.item()
        scheduler.step()
        avg /= n_batches
        row = {"epoch": epoch, "train_l1": avg, "time_s": time.time() - t_start}
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            m = evaluate(model)
            row.update({"val_rmse": m["rmse"], "val_prmse_lin": m["prmse_lin"], "val_prmse_ang": m["prmse_ang"]})
            print(f"epoch {epoch:4d}  train L1 {avg:.6f}  val {fmt(m)}  [{row['time_s']:.0f} s]", flush=True)
        log.append(row)

    # ---------------------------------------------------------------- final eval + save
    model.eval()
    m = evaluate(model)
    m.update({"model": args.model, "features": features, "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
              "seed": args.seed, "split_seed": args.split_seed, "n_params": n_params, "zero_init_head": zero_init, "inv_norm": args.inv_norm,
              "train_time_s": time.time() - t_start, "device": str(device)})
    print(f"[final] {fmt(m)}")
    print("  per-component RMSE :", " ".join(f"{c}={v:.5f}" for c, v in zip(COMPS, m["rmse_comp"])))
    print("  per-component PRMSE:", " ".join(f"{c}={v:.2f}%" for c, v in zip(COMPS, m["prmse_comp"])))
    print("  PRMSE by K         :", " ".join(f"{k}:{v['prmse']:.2f}%" for k, v in m["prmse_by_K"].items()))
    json.dump(m, open(args.out / "metrics.json", "w"), indent=2)
    import csv
    with open(args.out / "log.csv", "w", newline="") as f:
        keys = ["epoch", "train_l1", "val_rmse", "val_prmse_lin", "val_prmse_ang", "time_s"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in log:
            w.writerow({k: r.get(k, "") for k in keys})
    torch.save(model.state_dict(), args.out / "model.wt")
    scripted = torch.jit.script(model.cpu().eval())
    scripted.save(str(args.out / "model.pt"))
    # round-trip check of the exported model on a few rows
    chk = torch.jit.load(str(args.out / "model.pt")).eval()
    with torch.no_grad():
        a = model.predict_velocity(Xva[:64].cpu(), Fva[:64].cpu())
        b = chk.predict_velocity(Xva[:64].cpu(), Fva[:64].cpu())
    assert torch.allclose(a, b, atol=1e-6), "TorchScript export mismatch"
    print(f"[saved] {args.out}/model.wt, model.pt, metrics.json, log.csv, split.npz")
    if args.publish:
        wt_name, pt_name = PUBLISH[args.model]
        shutil.copy(args.out / "model.wt", Path("experiments") / wt_name)
        shutil.copy(args.out / "model.pt", Path("data/models") / pt_name)
        print(f"[published] experiments/{wt_name}, data/models/{pt_name}")


if __name__ == "__main__":
    main()
