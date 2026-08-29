"""
Sanity check for the Figure 3 N=300 recompute.

Validates, on ONE config per volume fraction at seed=123, that:
  (a) benchmarks.accuracy_grand_M resolves to the pinn-stokes repo (no shadowing),
  (b) the Fig-3 CPU operators reproduce the task's expected seed-123 rel_rmse table,
  (c) how long an Xfine ground-truth MFS solve takes at N=300 (launch-decision timing),
  (d) whether the Fig-4 GPU n-body operator (Mob_Nbody_Torch) produces the SAME
      per-particle velocity as the Fig-3 CPU n-body operator (Mob_Op_Nbody) on the
      identical config -> i.e. are the two figures measuring the same real operator.

Run: TORCH_COMPILE_DISABLE=1 python sanity_n300.py
"""
import time
import numpy as np
import benchmarks.accuracy_grand_M as A

assert "pinn-stokes" in A.__file__, f"WRONG MODULE: {A.__file__}"
assert A.UNIFORM_CONFIG is True, A.UNIFORM_CONFIG
print("module:", A.__file__, flush=True)

from benchmarks.cluster import generate_uniform_testcase

SHAPE = A.SHAPE
N = 300
SEED = A.DEFAULT_SEED  # 123
VFS = [0.10, 0.20]

# Expected seed-123 single-config rel_rmse values from the task description.
EXPECTED = {
    0.10: dict(M_rpy=17.01, M_2b=13.72, M_3b=10.19, M_nbody=10.40, mfs_coarse=0.85),
    0.20: dict(M_rpy=35.41, M_2b=30.82, M_3b=20.28, M_nbody=24.01, mfs_coarse=1.74),
}

t0 = time.time()
mob_ops = A.build_mobility_ops(SHAPE)
print(f"[built CPU fig-3 operators in {time.time()-t0:.1f}s]", flush=True)

# --- Fig-4's real operator (GPU n-body), OPTIONAL. Needs .wt weights. ---
# 2-body .wt is the same model that converts to two_body_combined_model.pt.
# n-body .wt candidates (no conversion block links these to nbody_pinn_b1.pt):
mob_nbody_gpu = None
gpu_nbody_wt = None
try:
    from src.gpu_nbody_mob import Mob_Nbody_Torch
    TWO_WT = "data/models/combined_2body.wt"
    for cand in ["data/models/nbody_cross_tmp.wt", "experiments/nbody_cross_tmp.wt"]:
        try:
            mob_nbody_gpu = Mob_Nbody_Torch(
                shape=SHAPE, self_nn_path=A.SELF_PATH, two_nn_path=TWO_WT,
                nbody_nn_path=cand, near_field_2b="nn", far_field_2b="rpy",
                near_far_switch=6.0,
            )
            gpu_nbody_wt = cand
            print(f"[built GPU fig-4 n-body operator with nbody wt={cand}]", flush=True)
            break
        except Exception as e:
            print(f"[GPU nbody build FAILED for {cand}: {type(e).__name__}: {e}]", flush=True)
except Exception as e:
    print(f"[GPU operator import failed: {e}]", flush=True)

for vf in VFS:
    print(f"\n===== vf={vf}, N={N}, seed={SEED} =====", flush=True)
    t0 = time.time()
    df = generate_uniform_testcase(
        shape=SHAPE, volume_fraction=vf, numParticles=N,
        seed=SEED, save_to_file=False,
    )
    t_gt = time.time() - t0
    print(f"[ground-truth Xfine MFS solve: {t_gt:.1f}s]", flush=True)

    config = df[["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].values
    forces = df[["f_x", "f_y", "f_z", "t_x", "t_y", "t_z"]].values
    velocity = df[["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]].values

    t0 = time.time()
    errs = A.compute_errors(mob_ops, config, forces, velocity)
    t_ops = time.time() - t0

    exp = EXPECTED[vf]
    print(f"[operator evals: {t_ops:.1f}s]")
    print(f"{'op':<12s} {'rel_rmse':>10s} {'expected':>10s} {'d%':>8s}")
    for key in ["M_rpy", "M_2b", "M_3b", "M_nbody", "mfs_coarse"]:
        got = errs[key]["rel_rmse"]
        e = exp[key]
        dd = (got - e) / e * 100.0
        print(f"{key:<12s} {got:>10.4f} {e:>10.2f} {dd:>+7.1f}%")

    # --- Consistency: compare Fig-3 CPU vs Fig-4 GPU n-body directly ---
    if mob_nbody_gpu is not None:
        pos = config[:, :3]
        ori = config[:, 3:]
        import torch
        pos_t = torch.as_tensor(pos, device=mob_nbody_gpu.device, dtype=torch.float32)
        ori_t = torch.as_tensor(ori, device=mob_nbody_gpu.device, dtype=torch.float32)
        f_t = torch.as_tensor(forces, device=mob_nbody_gpu.device, dtype=torch.float32)
        gpu_pred = mob_nbody_gpu.apply(pos_t, ori_t, f_t, viscosity=1.0).detach().cpu().numpy()
        # CPU n-body prediction (re-run the operator to get raw velocities)
        cpu_op = mob_ops["M_nbody"]
        cpu_pred = cpu_op.apply_cpu(pos, ori, forces, viscosity=1.0) \
            if hasattr(cpu_op, "apply_cpu") else cpu_op.apply(config, forces, viscosity=1.0)
        cpu_pred = np.asarray(cpu_pred)
        max_abs = np.max(np.abs(gpu_pred - cpu_pred))
        rel = np.linalg.norm(gpu_pred - cpu_pred) / (np.linalg.norm(cpu_pred) + 1e-12) * 100
        gpu_rel_rmse = A._compute_error_stats(gpu_pred, velocity)["rel_rmse"]
        print(f"[NBODY CPU-vs-GPU] max_abs_diff={max_abs:.3e}  rel_diff={rel:.4f}%  "
              f"CPU rel_rmse={errs['M_nbody']['rel_rmse']:.4f}  GPU rel_rmse={gpu_rel_rmse:.4f}  "
              f"(gpu wt={gpu_nbody_wt})", flush=True)

print("\nDone.", flush=True)
