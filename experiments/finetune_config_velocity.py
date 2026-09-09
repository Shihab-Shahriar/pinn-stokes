#!/usr/bin/env python3
"""Configuration-level velocity fine-tune of the pair moments + diagonal models (dataset v2 cache).

The published models are trained one block at a time (L1 over the 36 entries of each pair / diagonal
residual block).  Under aligned forcing (gravity) the per-pair errors of a configuration add coherently,
which that loss cannot see.  This script starts from the published weights and minimises, per batch of
whole configurations and per force pattern F,

    L = lambda_v * mean |6 pi (v_pred - v_tgt)|  +  lambda_b * (block L1 of the pair model + block L1 of the diag model)

where v_pred = sum over the cached near pairs of K_ts F_s (+ K_ts^T F_t) + sum over particles of K_tt F_t and
v_tgt is the same assembly with the exact residual blocks R_ts / R_tt of the cache -- i.e. the residual-space
velocity error of the two learned terms (the far-pair residual beyond the pair cutoff is not representable by
either model and is left out on purpose).  Force patterns: half random unit wrenches per particle, half uniform
gravity in a random direction per configuration (T = 0), all scaled by 6 pi as in train_nbody_v2.fixed_forces.

    TORCH_COMPILE_DISABLE=1 python experiments/finetune_config_velocity.py --cache data/multibody_v2_cache_pc8c \
        --label-base refl1 --pair-init experiments/nbody_moments_v2_kinf_rc8_pc8c_fts.wt \
        --diag-init experiments/nbody_diag_v2_pc8c_fts.wt --out experiments/runs_v2/ft_pc8c_fts --publish

Publishes <parent>_ft.{pt,json} for both models (sidecars copied from the parents + a "finetune" record), so the
harness runs them through --models overrides or a registered op.
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

from experiments import train_diag_v2 as td  # noqa: E402
from experiments import train_nbody_v2 as tr  # noqa: E402

PMAX = 64
SCALE = 6 * math.pi


# ----------------------------------------------------------------------------- data
class ConfigBatches:
    """Whole-configuration batches over the pair (V2Cache) and diagonal (DiagCache) row tables."""

    def __init__(self, pair: tr.V2Cache, diag: td.DiagCache):
        self.pair, self.diag = pair, diag
        C = len(pair.cfg_P)
        self.C = C
        self.P = np.asarray(pair.cfg_P, dtype=np.int64)
        pc = pair.pair_cfg.cpu().numpy()
        assert np.all(np.diff(pc) >= 0), "kept pair rows must stay configuration-sorted"
        self.pair_start = np.searchsorted(pc, np.arange(C + 1))
        rc = diag.row_cfg.cpu().numpy()
        assert np.all(np.diff(rc) >= 0)
        self.diag_start = np.searchsorted(rc, np.arange(C + 1))
        self.is_val = np.asarray(pair.cfg_is_val, dtype=bool)
        self.train_cfgs = np.nonzero(~self.is_val)[0]
        self.val_cfgs = np.nonzero(self.is_val)[0]

    def rows(self, cfgs):
        """(pair rows, pair local-config ids, diag rows, diag local-config ids) for a list of configurations."""
        pr, pl, dr, dl = [], [], [], []
        for j, c in enumerate(cfgs):
            a, b = self.pair_start[c], self.pair_start[c + 1]
            pr.append(np.arange(a, b)); pl.append(np.full(b - a, j))
            a, b = self.diag_start[c], self.diag_start[c + 1]
            dr.append(np.arange(a, b)); dl.append(np.full(b - a, j))
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, dtype=np.int64)
        return cat(pr), cat(pl), cat(dr), cat(dl)


def force_patterns(P, n_pat, gen, device):
    """(n_pat, B*PMAX, 6) wrenches: even patterns random unit force + torque per particle, odd patterns uniform
    gravity (random direction per configuration, zero torque); zero on padding slots."""
    B = len(P)
    F = torch.zeros(n_pat, B * PMAX, 6, dtype=torch.float32, device=device)
    valid = (torch.arange(PMAX, device=device)[None, :] < torch.as_tensor(P, device=device)[:, None]).reshape(-1)
    for p in range(n_pat):
        if p % 2 == 0:
            f = torch.randn(B * PMAX, 3, generator=gen, device=device)
            t = torch.randn(B * PMAX, 3, generator=gen, device=device)
            F[p, :, :3] = f / f.norm(dim=1, keepdim=True)
            F[p, :, 3:] = t / t.norm(dim=1, keepdim=True)
        else:
            g = torch.randn(B, 3, generator=gen, device=device)
            g = g / g.norm(dim=1, keepdim=True)
            F[p, :, :3] = g.repeat_interleave(PMAX, 0)
    return F * SCALE * valid[None, :, None].to(F.dtype), valid


def assemble(K, Kd, F, slot_t, slot_s, slot_d, n_slots):
    """(n_pat, n_slots, 6) velocities from pair blocks K (rows,6,6) (reciprocal: K_st = K_ts^T) and diagonal Kd."""
    n_pat = F.shape[0]
    v = torch.zeros(n_pat, n_slots, 6, dtype=K.dtype, device=K.device)
    if K.shape[0]:
        v.index_add_(1, slot_t, torch.einsum("nab,pnb->pna", K, F[:, slot_s]))
        v.index_add_(1, slot_s, torch.einsum("nba,pnb->pna", K, F[:, slot_t]))
    if Kd.shape[0]:
        v.index_add_(1, slot_d, torch.einsum("nab,pnb->pna", Kd, F[:, slot_d]))
    return v


# ----------------------------------------------------------------------------- one batch
def batch_terms(data: ConfigBatches, pair_model, diag_model, cfgs, n_pat, gen, device):
    pr, pl, dr, dl = data.rows(cfgs)
    P = data.P[cfgs]
    F, valid = force_patterns(P, n_pat, gen, device)
    n_slots = len(cfgs) * PMAX
    pl_t = torch.as_tensor(pl, device=device); dl_t = torch.as_tensor(dl, device=device)
    pi = data.pair._idx(pr)
    slot_t = pl_t * PMAX + data.pair.pair_t[pi].to(device)
    slot_s = pl_t * PMAX + data.pair.pair_s[pi].to(device)
    di = data.diag._idx(dr)
    slot_d = dl_t * PMAX + data.diag.row_p[di].to(device)
    if len(pr):
        X = data.pair.features(pr, "moments")
        K = pair_model.predict_mobility(X)
        R = data.pair.labels(pr)
    else:
        K = R = torch.zeros(0, 6, 6, device=device)
    Xd = data.diag.features(dr)
    Kd = diag_model.predict_mobility(Xd)
    Rd = data.diag.labels(dr)
    v_pred = assemble(K, Kd, F, slot_t, slot_s, slot_d, n_slots)
    v_tgt = assemble(R, Rd, F, slot_t, slot_s, slot_d, n_slots)
    return K, R, Kd, Rd, v_pred, v_tgt, valid


def vel_stats(v_pred, v_tgt, valid):
    """Residual-velocity capture (%): ||v_pred - v_tgt|| / ||v_tgt||, translational / angular, per pattern kind."""
    out = {}
    d = (v_pred - v_tgt)[:, valid]; t = v_tgt[:, valid]
    for name, sl in (("lin", slice(0, 3)), ("ang", slice(3, 6))):
        for kind, ps in (("random", slice(0, None, 2)), ("gravity", slice(1, None, 2))):
            num = d[ps][..., sl].norm(); den = t[ps][..., sl].norm().clamp_min(1e-30)
            out[f"{kind}_{name}"] = float(num / den * 100)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("data/multibody_v2_cache_pc8c"))
    ap.add_argument("--variant", default="kinf_rc8")
    ap.add_argument("--label-base", choices=["none", "refl1", "refl2"], default="none")
    ap.add_argument("--pair-init", required=True, help=".wt state dict of the pair moments model to start from")
    ap.add_argument("--diag-init", required=True, help=".wt state dict of the diagonal model to start from")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--configs-per-step", type=int, default=64)
    ap.add_argument("--patterns", type=int, default=4, help="force patterns per configuration per step (even: half random, half gravity)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-vel", type=float, default=1.0)
    ap.add_argument("--lambda-block", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--freeze", choices=["none", "diag", "pair"], default="none")
    ap.add_argument("--seed", type=int, default=411)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-on", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--eval-configs", type=int, default=800)
    ap.add_argument("--eval-rows", type=int, default=300000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--publish-suffix", default="_ft")
    args = ap.parse_args()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              open(args.out / "config.json", "w"), indent=1)

    t0 = time.time()
    pair = tr.V2Cache(args.cache, args.variant, device, args.data_on, None, args.label_base)
    diag = td.DiagCache(args.cache, device, args.data_on, None, 8.0, args.label_base)
    data = ConfigBatches(pair, diag)
    print(f"[data] {data.C} configurations ({len(data.train_cfgs)} train / {len(data.val_cfgs)} val), "
          f"{pair.n} pair rows, {diag.n} particle rows, loaded in {time.time() - t0:.0f} s", flush=True)

    pair_model = tr.load_model(args.pair_init, "moments", device)
    diag_model = td.load_model(args.diag_init, device)
    for m, name in ((pair_model, "pair"), (diag_model, "diag")):
        assert not isinstance(m, torch.jit.ScriptModule), f"{name}: start from a .wt state dict, not TorchScript"
    params = []
    if args.freeze != "pair":
        params += list(pair_model.parameters())
    if args.freeze != "diag":
        params += list(diag_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    eval_cfgs = np.sort(rng.choice(data.val_cfgs, size=min(args.eval_configs, len(data.val_cfgs)), replace=False))
    eval_gen_seed = args.seed + 999
    pair_val = np.sort(rng.choice(pair.val_idx, size=min(args.eval_rows, len(pair.val_idx)), replace=False))
    diag_val = np.sort(rng.choice(diag.val_idx, size=min(args.eval_rows, len(diag.val_idx)), replace=False))

    steps_per_epoch = max(1, len(data.train_cfgs) // args.configs_per_step)
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    def evaluate_now():
        pair_model.eval(); diag_model.eval()
        with torch.no_grad():
            g = torch.Generator(device=device).manual_seed(eval_gen_seed)
            acc = {}
            vp_all, vt_all, va_all = [], [], []
            for i in range(0, len(eval_cfgs), args.configs_per_step):
                cfgs = eval_cfgs[i:i + args.configs_per_step]
                _, _, _, _, vp, vt, valid = batch_terms(data, pair_model, diag_model, cfgs, args.patterns, g, device)
                vp_all.append(vp[:, valid]); vt_all.append(vt[:, valid])
            vp = torch.cat(vp_all, 1); vt = torch.cat(vt_all, 1)
            acc = vel_stats(vp, vt, torch.ones(vp.shape[1], dtype=torch.bool, device=device))
            mp = tr.evaluate(pair_model, pair, pair_val, "moments", full=False)
            md = td.evaluate(diag_model, diag, diag_val, full=False)
            acc.update({"pair_prmse_lin": mp["prmse_lin"], "pair_prmse_ang": mp["prmse_ang"], "pair_capture": mp["capture"],
                        "pair_rel_total": mp["rel_total"], "diag_capture": md["capture"]})
        pair_model.train(); diag_model.train()
        return acc

    def fmt(m):
        return (f"vel-capture random {m['random_lin']:.1f}/{m['random_ang']:.1f}%  gravity {m['gravity_lin']:.1f}/{m['gravity_ang']:.1f}% | "
                f"pair PRMSE {m['pair_prmse_lin']:.3f}/{m['pair_prmse_ang']:.3f}% capture {m['pair_capture']:.1f}% | diag capture {m['diag_capture']:.1f}%")

    m0 = evaluate_now()
    print(f"[init] {fmt(m0)}", flush=True)
    log = [{"epoch": -1, "step": 0, "train_loss": "", "loss_vel": "", "loss_block": "", "lr": args.lr, "time_s": 0.0, **m0}]
    print(f"[train] {len(data.train_cfgs)} train configs, {args.configs_per_step}/step, {steps_per_epoch} steps/epoch, "
          f"{total_steps} steps, {args.patterns} patterns, lambda_v {args.lambda_vel} lambda_b {args.lambda_block}, "
          f"lr {args.lr}, freeze {args.freeze}", flush=True)
    step = 0; t_start = time.time()
    for epoch in range(args.epochs):
        perm = rng.permutation(data.train_cfgs)
        pair_model.train(); diag_model.train()
        sums = np.zeros(3); n_avg = 0
        for it in range(steps_per_epoch):
            cfgs = perm[it * args.configs_per_step:(it + 1) * args.configs_per_step]
            K, R, Kd, Rd, vp, vt, valid = batch_terms(data, pair_model, diag_model, cfgs, args.patterns, gen, device)
            loss_vel = ((vp - vt)[:, valid] * SCALE).abs().mean()
            loss_block = ((K - R) * SCALE).abs().mean() + ((Kd - Rd) * SCALE).abs().mean()
            loss = args.lambda_vel * loss_vel + args.lambda_block * loss_block
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step(); scheduler.step(); step += 1
            sums += (loss.item(), loss_vel.item(), loss_block.item()); n_avg += 1
        m = evaluate_now()
        row = {"epoch": epoch, "step": step, "train_loss": sums[0] / n_avg, "loss_vel": sums[1] / n_avg,
               "loss_block": sums[2] / n_avg, "lr": scheduler.get_last_lr()[0], "time_s": time.time() - t_start, **m}
        log.append(row)
        torch.save(pair_model.state_dict(), args.out / "pair.ckpt.wt"); torch.save(diag_model.state_dict(), args.out / "diag.ckpt.wt")
        print(f"epoch {epoch:3d} step {step:6d} loss {row['train_loss']:.4f} (vel {row['loss_vel']:.4f} block {row['loss_block']:.4f}) "
              f"| {fmt(m)} [{row['time_s']:.0f} s]", flush=True)

    # ---------------------------------------------------------------- save + publish
    with open(args.out / "log.csv", "w", newline="") as f:
        keys = list(log[0].keys()); w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in log:
            w.writerow({k: r.get(k, "") for k in keys})
    final = log[-1]
    json.dump({"init": m0, "final": {k: final[k] for k in m0}, "steps": step, "epochs": args.epochs}, open(args.out / "metrics.json", "w"), indent=1)
    outs = {}
    for model, name, init, kind in ((pair_model, "pair", args.pair_init, "moments"), (diag_model, "diag", args.diag_init, "diag")):
        model = model.cpu().eval()
        torch.save(model.state_dict(), args.out / f"{name}.wt")
        scripted = torch.jit.script(model)
        scripted.save(str(args.out / f"{name}.pt"))
        chk = torch.jit.load(str(args.out / f"{name}.pt")).eval()
        X = (pair.features(pair_val[:64], "moments") if kind == "moments" else diag.features(diag_val[:64])).cpu()
        with torch.no_grad():
            assert torch.allclose(model.predict_mobility(X), chk.predict_mobility(X), atol=1e-6), "TorchScript export mismatch"
        outs[name] = init
    print(f"[saved] {args.out}/{{pair,diag}}.{{wt,pt}}, metrics.json, log.csv")
    if args.publish:
        for name, init in outs.items():
            parent = Path(init).stem
            side_p = Path("data/models") / f"{parent}.json"
            assert side_p.exists(), f"parent sidecar missing: {side_p}"
            side = json.load(open(side_p))
            new = parent + args.publish_suffix
            shutil.copy(args.out / f"{name}.pt", Path("data/models") / f"{new}.pt")
            shutil.copy(args.out / f"{name}.wt", Path("experiments") / f"{new}.wt")
            side.update({"name": new, "parent": parent, "fts_base": args.label_base, "run": str(args.out),
                         "finetune": {"loss": "config-velocity L1 + block L1", "lambda_vel": args.lambda_vel,
                                      "lambda_block": args.lambda_block, "epochs": args.epochs, "lr": args.lr,
                                      "patterns": args.patterns, "configs_per_step": args.configs_per_step,
                                      "init": m0, "final": {k: final[k] for k in m0}},
                         "created": time.strftime("%Y-%m-%d %H:%M:%S")})
            json.dump(side, open(Path("data/models") / f"{new}.json", "w"), indent=1)
            print(f"[published] data/models/{new}.pt (+ .json), experiments/{new}.wt")


if __name__ == "__main__":
    main()
