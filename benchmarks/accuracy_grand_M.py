from pathlib import Path
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.mfs_utils import build_B
from src.mob_op_2b_combined import NNMob as TwoBodyNNMob
from src.mob_op_3body import NNMob3B
from src.mob_op_nbody import Mob_Op_Nbody
from src.gpu_nbody_mob import Mob_Nbody_Torch
from src.triton_mfs import imp_mfs_mobility_sphere_triton, MobMFSTriton

from benchmarks.cluster import reference_data_generation, generate_uniform_testcase, uniform_sphere_cluster


SHAPE = "sphere"
SELF_PATH = "data/models/self_interaction_model.pt"
TWO_BODY_PATH = "data/models/two_body_combined_model.pt"
THREE_BODY_PATH = "data/models/3body_cross.pt"
NBODY_PATH = "data/models/nbody_pinn_b1.pt"

NUM_PARTICLES_LIST = [200]
VOLUME_FRACTIONS = [.025, 0.05, .075, 0.1, 0.125, 0.15, .175, 0.2]
DELTAS = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]

UNIFORM_CONFIG = True
GENERATE_DATA = True
DEFAULT_SEED = 123


def build_mobility_ops(shape):
    """
    This uses CPU implementations.
    """
    mob_2b = TwoBodyNNMob(shape, SELF_PATH, TWO_BODY_PATH,
                          nn_only=False, rpy_only=False)
    mob_rpy = TwoBodyNNMob(shape, SELF_PATH, TWO_BODY_PATH,
                           nn_only=False, rpy_only=True)
    mob_3b = NNMob3B(
        shape=shape,
        self_nn_path=SELF_PATH,
        two_nn_path=TWO_BODY_PATH,
        three_nn_path=THREE_BODY_PATH,
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
        triplet_cutoff=6.0,
    )
    mob_nbody = Mob_Op_Nbody(
        shape=shape,
        self_nn_path=SELF_PATH,
        two_nn_path=TWO_BODY_PATH,
        nbody_nn_path=NBODY_PATH,
        nn_only=False,
        rpy_only=False,
        switch_dist=6.0,
    )

    mfs_coarse = MobMFSTriton(shape=shape, acc="coarse")
    print("Initialized MFS coarse mobility operator with Triton backend.")

    return {
        "M_2b": mob_2b,
        "M_rpy": mob_rpy,
        "M_3b": mob_3b,
        "M_nbody": mob_nbody,
        "mfs_coarse": mfs_coarse,
    }


def load_dataset(shape, param_value, num_particles, uniform_config,
                 generate_data=False, seed=DEFAULT_SEED):
    if uniform_config:
        path = Path(f"tmp/uniform_{param_value}_{num_particles}.csv")
        if generate_data or not path.exists():
            df = generate_uniform_testcase(
                shape=shape,
                volume_fraction=param_value,
                numParticles=num_particles,
                seed=seed,
            )
        else:
            df = pd.read_csv(path, float_precision="high",
                             header=0, index_col=False)
    else:
        path = Path(f"tmp/reference_{param_value}_{num_particles}.csv")
        if generate_data or not path.exists():
            df = reference_data_generation(shape, param_value, num_particles, seed=seed)
            df.to_csv(path, index=False, header=True, float_format="%.16g")
        else:
            df = pd.read_csv(path, float_precision="high",
                             header=0, index_col=False)

    return df


def _mob_accepts_positions(mob):
    try:
        sig = inspect.signature(mob.apply)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    if not params:
        return False
    return params[0].name in ("positions", "pos")


def _to_numpy(array):
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return array


def _as_torch(array, device, dtype):
    if torch.is_tensor(array):
        if array.device != device or array.dtype != dtype:
            return array.to(device=device, dtype=dtype)
        return array
    return torch.as_tensor(array, device=device, dtype=dtype)


def _compute_error_stats(predicted, velocity):
    use_torch = torch.is_tensor(predicted) or torch.is_tensor(velocity)
    if use_torch:
        if torch.is_tensor(velocity):
            device = velocity.device
            dtype = velocity.dtype
        else:
            device = predicted.device
            dtype = predicted.dtype
        pred_t = _as_torch(predicted, device=device, dtype=dtype)
        vel_t = _as_torch(velocity, device=device, dtype=dtype)

        diff = vel_t - pred_t

        rmse = torch.sqrt(torch.mean(torch.mean(diff ** 2, dim=1)))
        ref_rms = torch.sqrt(torch.mean(torch.mean(vel_t ** 2, dim=1)))
        ref_rms_val = ref_rms.item()
        rmse_val = rmse.item()
        rel_rmse = 0.0 if ref_rms_val < 1e-12 else (rmse_val / ref_rms_val) * 100.0

        mae = torch.mean(torch.linalg.norm(diff, dim=1))
        mae_val = mae.item()

        v_norms = torch.linalg.norm(vel_t, dim=1)
        mean_v_norm = torch.mean(v_norms)
        mean_v_norm_val = mean_v_norm.item()
        rel_mae = 0.0 if mean_v_norm_val < 1e-12 else (mae_val / mean_v_norm_val) * 100.0

        element_rel = torch.nan_to_num(
            torch.linalg.norm(diff, dim=1) / v_norms,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        max_rel_rmse = element_rel.max().item() * 100.0

        return {
            "rmse": rmse_val,
            "rel_rmse": rel_rmse,
            "mae": mae_val,
            "rel_mae": rel_mae,
            "max_rel_rmse": max_rel_rmse,
        }

    diff = velocity - predicted

    rmse = np.sqrt(np.mean(np.mean(diff ** 2, axis=1)))
    ref_rms = np.sqrt(np.mean(np.mean(velocity ** 2, axis=1)))

    if ref_rms < 1e-12:
        rel_rmse = 0.0
    else:
        rel_rmse = (rmse / ref_rms) * 100

    mae = np.mean(np.linalg.norm(diff, axis=1))

    v_norms = np.linalg.norm(velocity, axis=1)
    mean_v_norm = np.mean(v_norms)
    if mean_v_norm < 1e-12:
        rel_mae = 0.0
    else:
        rel_mae = (mae / mean_v_norm) * 100

    with np.errstate(divide="ignore", invalid="ignore"):
        element_rel = np.linalg.norm(diff, axis=1) / v_norms
        element_rel = np.nan_to_num(element_rel)

    max_rel_rmse = np.max(element_rel) * 100

    return {
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "mae": mae,
        "rel_mae": rel_mae,
        "max_rel_rmse": max_rel_rmse,
    }


def compute_errors(mob_ops, config, forces, velocity):
    results = {}
    positions = config[:, :3]
    orientations = config[:, 3:]

    for key, mob in mob_ops.items():
        if _mob_accepts_positions(mob):
            if hasattr(mob, "device"):
                pos_t = _as_torch(positions, device=mob.device, dtype=torch.float32)
                ori_t = _as_torch(orientations, device=mob.device, dtype=torch.float32)
                force_t = _as_torch(forces, device=mob.device, dtype=torch.float32)
                predicted = mob.apply(pos_t, ori_t, force_t, viscosity=1.0)
            else:
                predicted = mob.apply(positions, orientations, forces, viscosity=1.0)
        elif hasattr(mob, "apply_cpu"):
            positions_np = _to_numpy(positions)
            orientations_np = _to_numpy(orientations)
            forces_np = _to_numpy(forces)
            predicted = mob.apply_cpu(positions_np, orientations_np, forces_np, viscosity=1.0)
        else:
            config_np = _to_numpy(config)
            forces_np = _to_numpy(forces)
            predicted = mob.apply(config_np, forces_np, viscosity=1.0)

        results[key] = _compute_error_stats(predicted, velocity)

    return results


def examine(num_particles, vol_frac, mob_op, shape=SHAPE, seed=DEFAULT_SEED):
    df = generate_uniform_testcase(
        shape=shape,
        volume_fraction=vol_frac,
        numParticles=num_particles,
        seed=seed,
        save_to_file=False,
    )

    config = df[["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].values
    forces = df[["f_x", "f_y", "f_z", "t_x", "t_y", "t_z"]].values
    velocity = df[["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]].values

    positions = config[:, :3]
    orientations = config[:, 3:]

    if hasattr(mob_op, "apply_cpu"):
        predicted = mob_op.apply_cpu(positions, orientations, forces, viscosity=1.0)
    else:
        predicted = mob_op.apply(config, forces, viscosity=1.0)

    pair_dist = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
        axis=-1,
    )
    np.fill_diagonal(pair_dist, np.inf)
    neighbor_counts = np.sum(pair_dist < 6.0, axis=1)

    for idx in range(num_particles):
        print(
            f"particle {idx:03d} | neighbors<6.0: {int(neighbor_counts[idx])} | "
            f"true v={velocity[idx]} | pred v={predicted[idx]}"
        )


def plot_operator_errors(res_df, param_col, num_particles, output_prefix,  x_label):
    plot_df = res_df.melt(
        id_vars=[param_col],
        value_vars=["M_rpy", "M_2b", "M_3b", "M_nbody"],
        var_name="operator",
        value_name="error",
    )
    name_map = {
        "M_2b": "NeMO 2-body",
        "M_rpy": "RPY",
        "M_3b": "NeMO 3-body summations",
        "M_nbody": "NeMO n-body",
    }
    plot_df["operator"] = plot_df["operator"].map(name_map)

    op_order = ["RPY", "NeMO 2-body", "NeMO 3-body summations", "NeMO n-body"]
    plot_df["operator"] = pd.Categorical(plot_df["operator"],
                                         categories=op_order,
                                         ordered=True)

    sns.set(style="white")
    plt.figure(figsize=(9, 5))
    plot_df_sorted = plot_df.sort_values(param_col)

    ax = sns.lineplot(
        data=plot_df_sorted,
        x=param_col,
        y="error",
        hue="operator",
        hue_order=op_order,
        marker="o",
    )
    sns.scatterplot(
        data=plot_df_sorted,
        x=param_col,
        y="error",
        hue="operator",
        hue_order=op_order,
        legend=False,
        s=60,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("PRMSE (%)")
    #ax.set_title(f"Grand mobility accuracy vs operator type (P={num_particles})")
    plt.legend(title="Mobility operator", fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_P{num_particles}.pdf", dpi=1000)
    plt.close()


def run_experiment_fixed_size_diff_operators(skip_if_csv_exists=True):
    """
    Compare the accuracy of different mobility operators:
    (RPY, 2-body NN, 3-body NN, n-body NN)

    We run for different number of particles, for both uniform and clustered configs.
    Save a separate csv/png for each particle count.

    We might choose just one particle count for final pair of figures.
    """
    param_values = VOLUME_FRACTIONS if UNIFORM_CONFIG else DELTAS
    param_col = "volume_fraction" if UNIFORM_CONFIG else "delta"
    output_prefix = "grand_M_acc_uniform_fixed_N" if UNIFORM_CONFIG else "grand_M_acc_cluster_fixed_N"
    output_path = Path(f"data/{output_prefix}.csv")

    if skip_if_csv_exists and output_path.exists():
        print(f"Skipping dataset generation; found existing results at {output_path}")
        res_df = pd.read_csv(output_path, float_precision="high",
                             header=0, index_col=False)
        for num_particles in NUM_PARTICLES_LIST:
            subset = res_df[res_df["num_particles"] == num_particles]
            plot_operator_errors(subset, param_col, num_particles, f"figures/{output_prefix}", "Volume fraction")
        return

    mob_ops = build_mobility_ops(SHAPE)

    num_repeats = 10
    base_seed = DEFAULT_SEED

    results = []
    print("Uniform config:", UNIFORM_CONFIG)
    for num_particles in NUM_PARTICLES_LIST:
        for param_value in param_values:
            print(f"{param_col}={param_value}, num_particles={num_particles}")
            
            accumulated_metrics = {}

            for run_idx in range(num_repeats):
                seed = base_seed + run_idx
                if UNIFORM_CONFIG:
                    df = generate_uniform_testcase(
                        shape=SHAPE,
                        volume_fraction=param_value,
                        numParticles=num_particles,
                        seed=seed,
                        save_to_file=False,
                    )
                else:
                    df = reference_data_generation(SHAPE, param_value, num_particles, seed=seed)

                config = df[["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].values
                forces = df[["f_x", "f_y", "f_z", "t_x", "t_y", "t_z"]].values
                velocity = df[["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]].values

                err_values = compute_errors(mob_ops, config, forces, velocity)

                if not accumulated_metrics:
                    for key, stats in err_values.items():
                        accumulated_metrics[key] = {m: 0.0 for m in stats}

                for key, stats in err_values.items():
                    for metric, val in stats.items():
                        accumulated_metrics[key][metric] += val
            
            # Average
            avg_metrics = {}
            for key, stats in accumulated_metrics.items():
                avg_metrics[key] = {m: val / num_repeats for m, val in stats.items()}

            # Helper to flatten for CSV
            result = {
                 param_col: float(param_value),
                 "num_particles": int(num_particles),
            }
            
            for key, stats in avg_metrics.items():
                 # For backward compatibility with the plotting code which expects "err_2b" etc.
                 # We'll use the "mae" or "rmse" as the main "error" if not specified, 
                 # but the previous code used: np.linalg.norm(..., axis=1).mean() which is MAE of L2 norms.
                 # So 'mae' in new compute_errors corresponds to old metric.
                 result[key] = stats['rel_rmse'] 
                 
                 # We can also add detailed cols like "err_2b_rel_rmse"
                 result[f"{key}_rel_rmse"] = stats['rel_rmse']
                 result[f"{key}_rel_mae"] = stats['rel_mae']

            results.append(result)

            print(
                f"errors (MAE) -> 2b: {result['M_2b']:.6e}, "
                f"RPY: {result['M_rpy']:.6e}, "
                f"3b: {result['M_3b']:.6e}, "
                f"nbody: {result['M_nbody']:.6e}",
                f"mfs_coarse: {result['mfs_coarse']:.6e}"
            )

    res_df = pd.DataFrame(
        results,
        columns=[param_col, "num_particles", "M_rpy", "M_2b", "M_3b", "M_nbody", "mfs_coarse"],
    )
    res_df = res_df.round(5)
    res_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

    for num_particles in NUM_PARTICLES_LIST:
        subset = res_df[res_df["num_particles"] == num_particles]
        plot_operator_errors(subset, param_col, num_particles, f"figures/{output_prefix}", "Volume fraction")


def plot_nbody_diff_sizes(res_df, output_prefix, volume_fractions=(0.05, 0.10, 0.15, 0.20)):
    plot_df = res_df.copy()
    if volume_fractions is not None:
        volume_fractions = np.asarray(volume_fractions, dtype=float)
        vf_values = plot_df["volume_fraction"].to_numpy(dtype=float)
        mask = np.isclose(
            vf_values[:, np.newaxis],
            volume_fractions[np.newaxis, :],
            rtol=0.0,
            atol=1e-12,
        ).any(axis=1)
        plot_df = plot_df.loc[mask].copy()

        for volume_fraction in volume_fractions:
            matching_rows = np.isclose(
                plot_df["volume_fraction"].to_numpy(dtype=float),
                volume_fraction,
                rtol=0.0,
                atol=1e-12,
            )
            plot_df.loc[matching_rows, "volume_fraction"] = volume_fraction

    assert not plot_df.empty, "No rows match the requested volume fractions"

    for target in ['avg_rel_rmse',  'max_rel_rmse', 'avg_rel_mae', 'avg_nearfield_interactions']:

        sns.set(style="white")
        plt.figure(figsize=(8, 5))
        ax = sns.lineplot(
            data=plot_df,
            x="num_particles",
            y=target,
            hue="volume_fraction",
            hue_order=volume_fractions,
            marker="o",
            palette="viridis",
        )
        ax.set_xlabel("num particles")
        if target == "avg_rel_rmse":
            ax.set_ylabel("Relative RMSE of velocity")
            #ax.set_title("n-body NN accuracy vs particle count")
        else:
            ax.set_ylabel(target)
            #ax.set_title(f"{target} vs particle count")
            
        if target != 'avg_nearfield_interactions':
            ax.set_ybound(lower=1e-6, upper=plot_df.max()[target] * 1.5)
            
        plt.legend(title="Volume fraction")
        plt.tight_layout()
        #plt.savefig(f"figures/{output_prefix}.png", dpi=600)
        plt.savefig(f"figures/{output_prefix}_{target}.pdf", dpi=1000)
        plt.close()


def run_exp_different_sizes(skip_if_csv_exists=True):
    """
    Show how n-body NN accuracy varies with number of particles
    in randomly distributed configurations.
    """

    num_particles_list = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]  
    volume_fractions = [.025, 0.05, .075, 0.1, 0.125, 0.15, .175, 0.2]

    num_repeats = 10
    base_seed = DEFAULT_SEED
    output_prefix = "M_accuracy_nbody_diff_sizes"
    output_path = Path(f"data/{output_prefix}.csv")

    if skip_if_csv_exists and output_path.exists():
        print(f"Skipping dataset generation; found existing results at {output_path}")
        res_df = pd.read_csv(output_path, float_precision="high",
                             header=0, index_col=False)
        plot_nbody_diff_sizes(res_df, output_prefix)
        return

    # mob_nbody = Mob_Op_Nbody(
    #     shape=SHAPE,
    #     self_nn_path=SELF_PATH,
    #     two_nn_path=TWO_BODY_PATH,
    #     nbody_nn_path=NBODY_PATH,
    #     nn_only=False,
    #     rpy_only=False,
    #     switch_dist=6.0,
    # )

    mob_nbody = Mob_Nbody_Torch(
        shape=SHAPE,
        self_nn_path=SELF_PATH,
        two_nn_path=TWO_BODY_PATH,
        nbody_nn_path=NBODY_PATH,
        near_field_2b="nn",
        far_field_2b='rpy',
        near_far_switch=6.0,
    )

    results = []

    print("Generating uniform testcases...")
    for v_idx, volume_fraction in enumerate(volume_fractions):
        for p_idx, num_particles in enumerate(num_particles_list):
            print(f"volume_fraction={volume_fraction}, num_particles={num_particles}")

            rmse_runs = []
            mae_runs = []
            rel_mae_runs = []
            rel_rmse_runs = []
            max_rel_rmses = []
            avg_nearfield_interactions = []
            for run_idx in range(num_repeats):
                seed = base_seed + v_idx * 1000 + p_idx * 100 + run_idx
                df = generate_uniform_testcase(
                    shape=SHAPE,
                    volume_fraction=volume_fraction,
                    numParticles=num_particles,
                    seed=seed,
                    save_to_file=False,
                )
                config = df[["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].values
                positions = df[["x", "y", "z"]].values
                # orientations = df[["q_x", "q_y", "q_z", "q_w"]].values # Not needed if using common compute_errors which extracts it from config if needed, or we pass components.
                
                # The compute_errors function expects 'config' (Nx7), 'forces' (Nx6), 'velocity' (Nx6)
                forces = df[["f_x", "f_y", "f_z", "t_x", "t_y", "t_z"]].values
                velocity = df[["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]].values
                
                # We need to wrap mob_nbody in a dict to use compute_errors
                mob_ops = {"nbody": mob_nbody}
                
                # Use the shared compute_errors function
                err_dict = compute_errors(mob_ops, config, forces, velocity)
                metrics = err_dict["nbody"]

                rmse = metrics["rmse"]
                rel_rmse = metrics["rel_rmse"]
                mae = metrics["mae"]
                rel_mae = metrics["rel_mae"]
                max_rel_rmse = metrics["max_rel_rmse"]
                
                # Compatibility with previous print
                err = mae 
                
                rel_rmse_runs.append(rel_rmse)
                max_rel_rmses.append(max_rel_rmse)
                rmse_runs.append(rmse)
                mae_runs.append(mae)
                rel_mae_runs.append(rel_mae)


                # find avg no of nearfield interactions
                pair_dist = np.linalg.norm(
                    positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
                    axis=-1,
                )
                np.fill_diagonal(pair_dist, np.inf)
                nearfield_cutoff = 6.0
                num_nearfield = np.sum(pair_dist < nearfield_cutoff) // 2
                avg_nearfield = num_nearfield * 2 / num_particles
                avg_nearfield_interactions.append(avg_nearfield)


                print(
                    f"  run {run_idx + 1}/{num_repeats} (seed={seed}): "
                    f"rel RMSE={rel_rmse:.6e}, max rel RMSE={max_rel_rmse:.6e}, "
                )
            avg_rel_rmse = float(np.mean(rel_rmse_runs))
            results.append({
                "volume_fraction": float(volume_fraction),
                "num_particles": int(num_particles),
                "avg_rel_rmse": avg_rel_rmse,
                "max_rel_rmse": float(np.mean(max_rel_rmses)),
                "avg_nearfield_interactions": float(np.mean(avg_nearfield_interactions)),
                "avg_rmse": float(np.mean(rmse_runs)),
                "avg_mae": float(np.mean(mae_runs)),
                "avg_rel_mae": float(np.mean(rel_mae_runs))
            })
            print(f"avg rel RMSE: {avg_rel_rmse:.6e}")
            print()

    res_df = pd.DataFrame(results, columns=[
        "volume_fraction", "num_particles", "avg_rel_rmse", "max_rel_rmse", 
        "avg_nearfield_interactions", "avg_rmse", "avg_mae", "avg_rel_mae"
    ])
    res_df = res_df.round(6)
    res_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_prefix}.csv")

    plot_nbody_diff_sizes(res_df, output_prefix)



def run_exp_nearfield_only(model_name):
    """
    Docstring for run_exp_nearfield_only
    """
    volume_fractions = [0.05, 0.075, 0.1, 0.125, 0.15, 0.20]
    num_repeats = 10
    base_seed = DEFAULT_SEED 
    device = torch.device("cuda")
    dtype = torch.float64


    root = "data"
    acc = "Xfine"
    b_single = np.loadtxt(f'{root}/points/b_{SHAPE}_{acc}.txt', dtype=np.float64)  # boundary nodes
    s_single = np.loadtxt(f'{root}/points/s_{SHAPE}_{acc}.txt', dtype=np.float64)
    B = build_B(b_single, s_single, np.zeros(3))
    B_t = torch.as_tensor(B, device=device, dtype=dtype)
    Bpp_inv = torch.linalg.pinv(B_t).contiguous()

    b_single_t = torch.as_tensor(b_single, device=device, dtype=dtype).contiguous()
    s_single_t = torch.as_tensor(s_single, device=device, dtype=dtype).contiguous()
    
    N = b_single.shape[0]  # number of boundary nodes
    M = s_single.shape[0]  # number of source points

    max_particles = 100
    # force_mag = 6.0 * np.pi * 1.0 * 1.0  # 6 pi eta a U with eta=1, a=1, U=1
    force_mag = 1.0
    F_ext_base = torch.rand((max_particles, 3), device=device, dtype=dtype) * 2.0 - 1.0
    T_ext_base = torch.rand((max_particles, 3), device=device, dtype=dtype) * 2.0 - 1.0
    F_ext_base = F_ext_base / torch.linalg.norm(F_ext_base, dim=1, keepdim=True) * force_mag
    T_ext_base = T_ext_base / torch.linalg.norm(T_ext_base, dim=1, keepdim=True) * force_mag

    if model_name == "mfs_coarse":
        model = MobMFSTriton(shape=SHAPE, acc="coarse")
    elif model_name == "nbody":
        model = Mob_Nbody_Torch(
            shape=SHAPE,
            self_nn_path=SELF_PATH,
            two_nn_path=TWO_BODY_PATH,
            nbody_nn_path=NBODY_PATH,
            near_field_2b="nn",
            far_field_2b='rpy',
            near_far_switch=6.0,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    results = []
    output_prefix = f"nearfield_only_accuracy_{model_name}"
    output_path = Path(f"data/{output_prefix}.csv")

    for volume_fraction in volume_fractions:
        print(f"volume_fraction={volume_fraction}")

        rel_rmse_runs = []
        max_rel_rmses = []
        rmse_runs = []
        mae_runs = []
        rel_mae_runs = []
        num_nearfield_runs = []

        for run_idx in range(num_repeats):
            seed = base_seed + run_idx
            centers, orients = uniform_sphere_cluster(
                volume_fraction,
                max_particles,
                seed=seed,
            )

            orients = np.array(
                [r.as_quat(scalar_first=False) for r in orients],
                dtype=np.float64,
            )
            centers_t = torch.as_tensor(centers, device=device, dtype=dtype)
            orients_t = torch.as_tensor(orients, device=device, dtype=dtype)

            # Dist from origin (on GPU)
            dist = torch.linalg.norm(centers_t, dim=1)
            mask = dist <= 6.0
            num_particles = int(mask.sum().item())
            print(
                f"  run {run_idx + 1}/{num_repeats} (seed={seed}): "
                f"num nearfield particles={num_particles}"
            )

            centers_nf = centers_t[mask]
            orients_nf = orients_t[mask]
            assert centers_nf.shape[0] > 0
            config_nf = torch.cat((centers_nf, orients_nf), dim=1)

            F_ext_nf = F_ext_base[:num_particles]
            T_ext_nf = T_ext_base[:num_particles]
            forces_nf = torch.cat((F_ext_nf, T_ext_nf), dim=1)

            # ground truth using MFS
            V_tilde_gpu = imp_mfs_mobility_sphere_triton(
                b_single_t,
                s_single_t,
                centers_nf,
                F_ext_nf,
                T_ext_nf,
                Bpp_inv,
                max_iter=1000,
                tol=1e-8,
                print_steps=False,
                L_cut=25.0,
                device=device,
            )
            sol = torch.stack(V_tilde_gpu, dim=0)
            predicted = sol[:, 3 * M:3 * M + 6]

            # Compute predictions for all particles to account for interactions
            if model_name == "mfs_coarse":
                config_np = config_nf.detach().cpu().numpy()
                forces_np = forces_nf.detach().cpu().numpy()
                pred_all = model.apply(config_np, forces_np, viscosity=1.0)
            elif model_name == "nbody":
                pred_all = model.apply(
                    centers_nf.float(),
                    orients_nf.float(),
                    forces_nf.float(),
                    viscosity=1.0
                )

            # Ground truth (from MFS calculation above)
            gt_all = predicted.detach().cpu().numpy()

            # Measure error only on the first particle (guaranteed to be at origin)
            metrics = _compute_error_stats(pred_all[:1], gt_all[:1])

            rel_rmse_runs.append(metrics["rel_rmse"])
            max_rel_rmses.append(metrics["max_rel_rmse"])
            rmse_runs.append(metrics["rmse"])
            mae_runs.append(metrics["mae"])
            rel_mae_runs.append(metrics["rel_mae"])
            num_nearfield_runs.append(num_particles)

        results.append({
            "volume_fraction": float(volume_fraction),
            "avg_num_nearfield_particles": float(np.mean(num_nearfield_runs)),
            "avg_rel_rmse": float(np.mean(rel_rmse_runs)),
            "avg_max_rel_rmse": float(np.mean(max_rel_rmses)),
            "avg_rmse": float(np.mean(rmse_runs)),
            "avg_mae": float(np.mean(mae_runs)),
            "avg_rel_mae": float(np.mean(rel_mae_runs)),
        })

        print("\n")

    res_df = pd.DataFrame(results, columns=[
        "volume_fraction",
        "num_particles",
        "avg_num_nearfield_particles",
        "avg_rel_rmse",
        "avg_max_rel_rmse",
        "avg_rmse",
        "avg_mae",
        "avg_rel_mae",
    ])
    res_df = res_df.round(6)
    res_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_prefix}.csv")

    for target in ["avg_rel_rmse", "avg_max_rel_rmse"]:
        sns.set(style="white")
        plt.figure(figsize=(8, 5))
        plot_df = res_df.sort_values("avg_num_nearfield_particles")
        ax = sns.lineplot(
            data=plot_df,
            x="avg_num_nearfield_particles",
            y=target,
            hue="volume_fraction",
            marker="o",
            palette="viridis",
        )
        ax.set_xlabel("avg number of particles in nearfield")
        ax.set_ylabel(target)
        ax.set_title(f"{target} ({model_name}) vs nearfield particle count")
        plt.legend(title="Volume fraction")
        plt.tight_layout()
        plt.savefig(f"figures/{output_prefix}_{target}.png", dpi=600)
        plt.close()


def run_exp_diff_force_fields():
    """
    How does grand mobility accuracy vary with different types of
    applied forces/torques (random, sedimentation, ABC flow etc)
    """
    VOL_FRAC = 0.15
    NUM_PARTICLES = [40, 80, 120, 160, 200, 240, 280]
    FORCE_FIELDS = ["random", "sedimentation", "periodic", "abc"]

    mob_op = build_mobility_ops(SHAPE)["M_nbody"]
    mfs_truth = MobMFSTriton(shape=SHAPE, acc="Xfine")

    num_repeats = 5
    base_seed = DEFAULT_SEED+1
    output_prefix = "grand_M_force_field_accuracy"

    results = []
    periodic_snapshot = None

    def abc_velocity(pos, A=1.0, B=1.0, C=1.0, L=2 * np.pi):
        pos = np.asarray(pos, dtype=float)
        x, y, z = (2 * np.pi / L) * pos.T
        u = np.empty_like(pos)
        u[:, 0] = A * np.sin(z) + C * np.cos(y)
        u[:, 1] = B * np.sin(x) + A * np.cos(z)
        u[:, 2] = C * np.sin(y) + B * np.cos(x)
        return u

    def abc_forces(pos, f0=1.0, mean_zero=True, **abc_kwargs):
        F = f0 * abc_velocity(pos, **abc_kwargs)
        if mean_zero:
            F = F - F.mean(axis=0, keepdims=True)
        return F

    for num_particles in NUM_PARTICLES:
        for run_idx in range(num_repeats):
            seed = base_seed + num_particles * 100 + run_idx
            centers, _ = uniform_sphere_cluster(
                volume_fraction=VOL_FRAC,
                numParticles=num_particles,
                seed=seed,
            )
            orientations = np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                (num_particles, 1),
            )
            config = np.concatenate([centers, orientations], axis=1)

            for force_field in FORCE_FIELDS:
                if force_field == "random":
                    rng = np.random.default_rng(seed + 17)
                    F_ext = rng.uniform(-1.0, 1.0, size=(num_particles, 3)).astype(np.float64)
                    T_ext = rng.uniform(-1.0, 1.0, size=(num_particles, 3)).astype(np.float64)
                    F_ext = F_ext / (np.linalg.norm(F_ext, axis=1, keepdims=True) + 1e-12)
                    T_ext = T_ext / (np.linalg.norm(T_ext, axis=1, keepdims=True) + 1e-12)
                elif force_field == "sedimentation":
                    F_ext = np.tile(
                        np.array([0.0, 0.0, -9.81], dtype=np.float64),
                        (num_particles, 1),
                    )
                    T_ext = np.zeros((num_particles, 3), dtype=np.float64)
                elif force_field == "periodic":
                    x = centers[:, 0]
                    y = centers[:, 1]
                    fx = -np.sin(x) * np.cos(y)
                    fy = np.cos(x) * np.sin(y)
                    fz = np.zeros_like(fx)
                    F_ext = np.stack([fx, fy, fz], axis=1).astype(np.float64)
                    T_ext = np.zeros((num_particles, 3), dtype=np.float64)
                else:
                    F_ext = abc_forces(centers, f0=1.0, mean_zero=True).astype(np.float64)
                    T_ext = np.zeros((num_particles, 3), dtype=np.float64)

                forces = np.concatenate([F_ext, T_ext], axis=1)

                velocity = mfs_truth.apply(config, forces, viscosity=1.0)
                err_values = compute_errors({"nbody": mob_op}, config, forces, velocity)
                stats = err_values["nbody"]

                results.append({
                    "volume_fraction": float(VOL_FRAC),
                    "num_particles": int(num_particles),
                    "force_field": force_field,
                    "run": int(run_idx),
                    "rmse": stats["rmse"],
                    "rel_rmse": stats["rel_rmse"],
                    "mae": stats["mae"],
                    "rel_mae": stats["rel_mae"],
                    "max_rel_rmse": stats["max_rel_rmse"],
                })

                if force_field == "periodic" and periodic_snapshot is None:
                    periodic_snapshot = (centers.copy(), F_ext.copy())

            print(
                f"Completed run {run_idx + 1}/{num_repeats} for P={num_particles}"
            )

    raw_df = pd.DataFrame(results)
    summary_df = (
        raw_df.drop(columns=["run"])
        .groupby(["force_field", "num_particles"], as_index=False)
        .mean(numeric_only=True)
    )
    summary_df = summary_df.rename(columns={
        "rmse": "avg_rmse",
        "rel_rmse": "avg_rel_rmse",
        "mae": "avg_mae",
        "rel_mae": "avg_rel_mae",
        "max_rel_rmse": "avg_max_rel_rmse",
    })
    summary_df["num_runs"] = num_repeats
    summary_df = summary_df.sort_values(["force_field", "num_particles"])

    output_csv = Path(f"data/{output_prefix}.csv")
    summary_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

    plot_df = summary_df.copy()
    plot_df["force_field"] = pd.Categorical(
        plot_df["force_field"],
        categories=FORCE_FIELDS,
        ordered=True,
    )
    sns.set(style="whitegrid")
    plt.figure(figsize=(9, 5))
    ax = sns.lineplot(
        data=plot_df,
        x="num_particles",
        y="avg_rel_rmse",
        hue="force_field",
        marker="o",
    )
    ax.set_xlabel("num particles")
    ax.set_ylabel("n-body NN relative RMSE (%)")
    ax.set_title("Grand mobility accuracy vs number of particles")
    plt.legend(title="force field")
    plt.tight_layout()
    output_png = Path(f"figures/{output_prefix}.png")
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved plot to {output_png}")

    if periodic_snapshot is not None:
        centers, forces_xy = periodic_snapshot
        plt.figure(figsize=(6, 5))
        plt.quiver(
            centers[:, 0],
            centers[:, 1],
            forces_xy[:, 0],
            forces_xy[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
        )
        plt.scatter(centers[:, 0], centers[:, 1], s=10, c="k", alpha=0.35)
        plt.gca().set_aspect("equal", "box")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Periodic force field on particle positions")
        plt.tight_layout()
        periodic_png = Path("figures/periodic_force_field.png")
        plt.savefig(periodic_png, dpi=300)
        plt.close()
        print(f"Saved periodic force field plot to {periodic_png}")


def run_exp_force_field_variants():
    """
    Test how periodic/ABC force field modifications affect n-body NN accuracy.
    Variants: wavelength scaling, 3D extension, per-particle normalization, mean subtraction.
    Added: smooth random field with a controllable smoothing length (fraction of bbox).
    Ideas tested:
      1. Wavelength ≈ neighbor cutoff (L in {3,4,6,8}) to maximize within-shell diversity.
      2. ABC + random noise superposition (alpha blending) to break spatial correlation.
    """
    VOL_FRAC = 0.10
    NUM_PARTICLES = [50, 100, 150, 200]
    num_repeats = 3
    base_seed = DEFAULT_SEED + 7

    mob_op = build_mobility_ops(SHAPE)["M_nbody"]
    mfs_truth = MobMFSTriton(shape=SHAPE, acc="fine")

    def _bbox_side(num_particles, vol_frac, radius=1.0):
        sphere_vol = (4 / 3) * np.pi * radius ** 3
        return (num_particles * sphere_vol / vol_frac) ** (1 / 3)

    def make_periodic(centers, L, three_d=False, normalize=False, mean_sub=False):
        x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
        fx = -np.sin(x / L) * np.cos(y / L)
        fy = np.cos(x / L) * np.sin(y / L)
        if three_d:
            fz = -np.sin(z / L) * np.cos(x / L)
        else:
            fz = np.zeros_like(fx)
        F = np.stack([fx, fy, fz], axis=1).astype(np.float64)
        if mean_sub:
            F = F - F.mean(axis=0, keepdims=True)
        if normalize:
            norms = np.linalg.norm(F, axis=1, keepdims=True)
            F = np.where(norms > 1e-12, F / norms, F)
        return F

    def make_abc(centers, L, normalize=False, mean_sub=True,
                 A=1.0, B=1.0, C=1.0):
        pos = np.asarray(centers, dtype=float)
        x, y, z = (2 * np.pi / L) * pos.T
        F = np.empty_like(pos)
        F[:, 0] = A * np.sin(z) + C * np.cos(y)
        F[:, 1] = B * np.sin(x) + A * np.cos(z)
        F[:, 2] = C * np.sin(y) + B * np.cos(x)
        if mean_sub:
            F = F - F.mean(axis=0, keepdims=True)
        if normalize:
            norms = np.linalg.norm(F, axis=1, keepdims=True)
            F = np.where(norms > 1e-12, F / norms, F)
        return F.astype(np.float64)

    def make_abc_plus_noise(centers, L, alpha, seed, normalize=False, mean_sub=True):
        """Blend ABC flow with random unit forces: F = alpha*abc + (1-alpha)*random."""
        abc_F = make_abc(centers, L, normalize=False, mean_sub=False)
        rng = np.random.default_rng(seed + 9999)
        rand_F = rng.uniform(-1.0, 1.0, size=centers.shape).astype(np.float64)
        rand_norms = np.linalg.norm(rand_F, axis=1, keepdims=True)
        rand_F = rand_F / np.where(rand_norms > 1e-12, rand_norms, 1.0)
        # Match scales: normalize abc component to unit RMS before blending
        abc_rms = np.sqrt(np.mean(abc_F ** 2))
        abc_F_scaled = abc_F / abc_rms if abc_rms > 1e-12 else abc_F
        F = alpha * abc_F_scaled + (1.0 - alpha) * rand_F
        if mean_sub:
            F = F - F.mean(axis=0, keepdims=True)
        if normalize:
            norms = np.linalg.norm(F, axis=1, keepdims=True)
            F = np.where(norms > 1e-12, F / norms, F)
        return F.astype(np.float64)

    def make_smooth_random(centers, smooth_len, seed, normalize=True, mean_sub=True):
        """
        Smooth a random vector field with a Gaussian kernel.
        smooth_len controls the correlation length in the same units as centers.
        """
        centers = np.asarray(centers, dtype=np.float64)
        n = centers.shape[0]
        rng = np.random.default_rng(seed + 31337)
        base = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float64)

        if smooth_len <= 0:
            F = base
        else:
            diff = centers[:, None, :] - centers[None, :, :]
            dist2 = np.sum(diff ** 2, axis=2)
            sigma2 = float(smooth_len) ** 2
            weights = np.exp(-0.5 * dist2 / sigma2)
            wsum = weights.sum(axis=1, keepdims=True)
            weights = weights / np.where(wsum > 1e-12, wsum, 1.0)
            F = weights @ base

        if mean_sub:
            F = F - F.mean(axis=0, keepdims=True)
        if normalize:
            norms = np.linalg.norm(F, axis=1, keepdims=True)
            F = np.where(norms > 1e-12, F / norms, F)
        return F.astype(np.float64)

    def sedimentation_forces(centers):
        n = centers.shape[0]
        F = np.tile(
            np.array([0.0, 0.0, -9.81], dtype=np.float64),
            (n, 1),
        )
        return F

    def totally_random_forces(centers, seed):
        n = centers.shape[0]
        rng = np.random.default_rng(seed + 4242)
        F = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float64)
        norms = np.linalg.norm(F, axis=1, keepdims=True)
        F = F / np.where(norms > 1e-12, norms, 1.0)
        return F

    # builder_func(centers, bbox_side, seed) -> F_ext (N,3)
    # --- Original variants ---
    SMOOTH_RANDOM_SCALES = [.2, .5]  # fraction of bbox side length
    VARIANTS = {
        "periodic_baseline": lambda c, bs, s: make_periodic(c, L=2*np.pi),
        "periodic_baseline_16": lambda c, bs, s: make_periodic(c, L=16*np.pi),
        "periodic_baseline_64": lambda c, bs, s: make_periodic(c, L=64*np.pi),
        "periodic_3d_32": lambda c, bs, s: make_periodic(c, L=32*np.pi, three_d=True),
    }
    VARIANTS = {
        "sedimentation": lambda c, bs, s: sedimentation_forces(c),
        "totally_random": lambda c, bs, s: totally_random_forces(c, s),
    }
    for scale in SMOOTH_RANDOM_SCALES:
        VARIANTS[f"smooth_random_scale{scale:g}"] = (
            lambda c, bs, s, scale=scale: make_smooth_random(
                c, smooth_len=scale * bs, seed=s, normalize=True, mean_sub=True
            )
        )
    F = make_smooth_random(
        np.array([[0.0, 0.0, 0.0]]), smooth_len=0.2, seed=0
    )
    F[:, :2] = 0.0
    VARIANTS[f"smooth_random_scale0.2_1d"] = lambda c, bs, s: F

    results = []

    for num_particles in NUM_PARTICLES:
        bbox = _bbox_side(num_particles, VOL_FRAC)
        print(f"\n=== P={num_particles}, bbox_side={bbox:.2f} ===")

        for run_idx in range(num_repeats):
            seed = base_seed + num_particles * 100 + run_idx
            centers, _ = uniform_sphere_cluster(
                volume_fraction=VOL_FRAC,
                numParticles=num_particles,
                seed=seed,
            )
            orientations = np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                (num_particles, 1),
            )
            config = np.concatenate([centers, orientations], axis=1)

            for variant_name, builder in VARIANTS.items():
                F_ext = builder(centers, bbox, seed)
                T_ext = np.zeros((num_particles, 3), dtype=np.float64)
                forces = np.concatenate([F_ext, T_ext], axis=1)

                velocity = mfs_truth.apply(config, forces, viscosity=1.0)
                err_values = compute_errors({"nbody": mob_op}, config, forces, velocity)
                stats = err_values["nbody"]

                results.append({
                    "variant": variant_name,
                    "num_particles": int(num_particles),
                    "run": int(run_idx),
                    "rel_rmse": stats["rel_rmse"],
                    "max_rel_rmse": stats["max_rel_rmse"],
                    "rmse": stats["rmse"],
                    "mae": stats["mae"],
                    "rel_mae": stats["rel_mae"],
                })
                print(
                    f"  variant: {variant_name:<30s} "
                    f"rel RMSE: {stats['rel_rmse']:.6e}, "
                )

            print(f"  run {run_idx + 1}/{num_repeats} done")

    raw_df = pd.DataFrame(results)
    summary_df = (
        raw_df.drop(columns=["run"])
        .groupby(["variant", "num_particles"], as_index=False)
        .mean(numeric_only=True)
    )
    summary_df = summary_df.rename(columns={
        "rmse": "avg_rmse",
        "rel_rmse": "avg_rel_rmse",
        "mae": "avg_mae",
        "rel_mae": "avg_rel_mae",
        "max_rel_rmse": "avg_max_rel_rmse",
    })
    summary_df["num_runs"] = num_repeats
    summary_df = summary_df.sort_values(["variant", "num_particles"])

    output_csv = Path("data/force_field_variants_accuracy.csv")
    summary_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'variant':<32s} {'P':>5s} {'rel_rmse%':>10s} {'max_rel%':>10s} {'rel_mae%':>10s}")
    print("-" * 100)
    for _, row in summary_df.iterrows():
        print(
            f"{row['variant']:<32s} {int(row['num_particles']):>5d} "
            f"{row['avg_rel_rmse']:>10.4f} {row['avg_max_rel_rmse']:>10.4f} "
            f"{row['avg_rel_mae']:>10.4f}"
        )
    print("=" * 100)

    # Plot: rel_rmse vs num_particles, one line per variant
    sns.set(style="whitegrid")
    plt.figure(figsize=(13, 7))
    ax = sns.lineplot(
        data=summary_df,
        x="num_particles",
        y="avg_rel_rmse",
        hue="variant",
        marker="o",
    )
    ax.set_xlabel("num particles")
    ax.set_ylabel("n-body NN relative RMSE (%)")
    ax.set_title("Force field variant accuracy comparison")
    plt.legend(title="variant", fontsize=7, title_fontsize=8,
               bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    output_png = Path("figures/force_field_variants_accuracy.png")
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved plot to {output_png}")

    # Second plot: max_rel_rmse (the outlier metric most sensitive to correlation)
    plt.figure(figsize=(13, 7))
    ax2 = sns.lineplot(
        data=summary_df,
        x="num_particles",
        y="avg_max_rel_rmse",
        hue="variant",
        marker="s",
    )
    ax2.set_xlabel("num particles")
    ax2.set_ylabel("n-body NN max relative error (%)")
    ax2.set_title("Force field variant — worst-particle error comparison")
    plt.legend(title="variant", fontsize=7, title_fontsize=8,
               bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    output_png2 = Path("figures/force_field_variants_max_rel.png")
    plt.savefig(output_png2, dpi=300)
    plt.close()
    print(f"Saved plot to {output_png2}")


def run_exp_linear_gradient_fields():
    """
    Test linear gradient / shear force fields at different magnitudes.

    Force fields:
    - sedimentation: F = [0, 0, -mag], T = 0  (uniform baseline)
    - shear: F = [mag * z, 0, 0], T = 0  (simple shear, zero-mean linear)
    - extension: F = mag * [x, -y/2, -z/2], T = 0  (uniaxial extensional, div-free)
    - sed_plus_shear: F = [mag * z, 0, -9.81], T = 0  (sedimentation + shear perturbation)

    Magnitudes control the gradient strength (gamma/epsilon).
    For sed_plus_shear the uniform offset is fixed at g=9.81, so the ratio
    mag * (bbox/2) / 9.81 controls how non-uniform the field is.
    """
    VOL_FRAC = 0.15
    NUM_PARTICLES = [40, 120, 280]
    MAGNITUDES = [0.1, 1.0, 10.0]
    num_repeats = 3
    base_seed = DEFAULT_SEED + 42

    mob_op = build_mobility_ops(SHAPE)["M_nbody"]
    mfs_truth = MobMFSTriton(shape=SHAPE, acc="Xfine")

    def _bbox_side(num_particles, vol_frac, radius=1.0):
        sphere_vol = (4 / 3) * np.pi * radius ** 3
        return (num_particles * sphere_vol / vol_frac) ** (1 / 3)

    def make_sedimentation(centers, mag):
        N = centers.shape[0]
        F = np.zeros((N, 3), dtype=np.float64)
        F[:, 2] = -mag
        T = np.zeros((N, 3), dtype=np.float64)
        return F, T

    def make_shear(centers, mag):
        """Simple shear: F_x = mag * z_i"""
        N = centers.shape[0]
        F = np.zeros((N, 3), dtype=np.float64)
        F[:, 0] = mag * centers[:, 2]
        T = np.zeros((N, 3), dtype=np.float64)
        return F, T

    def make_extension(centers, mag):
        """Uniaxial extension: F = mag * [x, -y/2, -z/2] (divergence-free)"""
        N = centers.shape[0]
        F = np.zeros((N, 3), dtype=np.float64)
        F[:, 0] = mag * centers[:, 0]
        F[:, 1] = -0.5 * mag * centers[:, 1]
        F[:, 2] = -0.5 * mag * centers[:, 2]
        T = np.zeros((N, 3), dtype=np.float64)
        return F, T

    def make_sed_plus_shear(centers, mag):
        """Sedimentation + shear perturbation: F = [mag*z, 0, -9.81]"""
        N = centers.shape[0]
        F = np.zeros((N, 3), dtype=np.float64)
        F[:, 0] = mag * centers[:, 2]
        F[:, 2] = -9.81
        T = np.zeros((N, 3), dtype=np.float64)
        return F, T

    FORCE_FIELDS = {
        "sedimentation": make_sedimentation,
        "shear": make_shear,
        "extension": make_extension,
        "sed_plus_shear": make_sed_plus_shear,
    }

    results = []

    for num_particles in NUM_PARTICLES:
        bbox = _bbox_side(num_particles, VOL_FRAC)
        print(f"\n=== P={num_particles}, bbox_side={bbox:.2f} ===")

        # Pre-generate configs for this particle count (shared across force fields)
        configs_list = []
        centers_list = []
        for run_idx in range(num_repeats):
            seed = base_seed + num_particles * 100 + run_idx
            centers, _ = uniform_sphere_cluster(
                volume_fraction=VOL_FRAC,
                numParticles=num_particles,
                seed=seed,
            )
            orientations = np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                (num_particles, 1),
            )
            config = np.concatenate([centers, orientations], axis=1)
            configs_list.append(config)
            centers_list.append(centers)

        for ff_name, ff_builder in FORCE_FIELDS.items():
            for mag in MAGNITUDES:
                rel_rmse_runs = []
                max_rel_runs = []
                rmse_runs = []
                mae_runs = []
                rel_mae_runs = []
                force_std_runs = []

                for run_idx in range(num_repeats):
                    config = configs_list[run_idx]
                    centers = centers_list[run_idx]

                    F_ext, T_ext = ff_builder(centers, mag)
                    forces = np.concatenate([F_ext, T_ext], axis=1)

                    # Force variation metric: std of force magnitudes / mean
                    f_norms = np.linalg.norm(F_ext, axis=1)
                    f_mean = np.mean(f_norms)
                    f_std_ratio = np.std(f_norms) / f_mean if f_mean > 1e-12 else 0.0
                    force_std_runs.append(f_std_ratio)

                    velocity = mfs_truth.apply(config, forces, viscosity=1.0)
                    err_values = compute_errors({"nbody": mob_op}, config, forces, velocity)
                    stats = err_values["nbody"]

                    rel_rmse_runs.append(stats["rel_rmse"])
                    max_rel_runs.append(stats["max_rel_rmse"])
                    rmse_runs.append(stats["rmse"])
                    mae_runs.append(stats["mae"])
                    rel_mae_runs.append(stats["rel_mae"])

                # Compute gradient-to-mean ratio for sed_plus_shear
                grad_to_mean = mag * (bbox / 2) / 9.81 if ff_name == "sed_plus_shear" else np.nan

                results.append({
                    "force_field": ff_name,
                    "magnitude": mag,
                    "num_particles": int(num_particles),
                    "avg_rel_rmse": float(np.mean(rel_rmse_runs)),
                    "avg_max_rel_rmse": float(np.mean(max_rel_runs)),
                    "avg_rmse": float(np.mean(rmse_runs)),
                    "avg_mae": float(np.mean(mae_runs)),
                    "avg_rel_mae": float(np.mean(rel_mae_runs)),
                    "force_variation": float(np.mean(force_std_runs)),
                    "grad_to_mean_ratio": grad_to_mean,
                })

                print(
                    f"  P={num_particles:>4d} | {ff_name:<18s} | mag={mag:>6.1f} | "
                    f"rel_rmse={np.mean(rel_rmse_runs):>8.4f}% | "
                    f"max_rel={np.mean(max_rel_runs):>8.4f}%"
                )

    res_df = pd.DataFrame(results)
    output_csv = Path("data/linear_gradient_accuracy.csv")
    res_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    # Print summary tables — one per force field
    for ff_name in FORCE_FIELDS:
        ff_df = res_df[res_df["force_field"] == ff_name]
        print(f"\n{'=' * 72}")
        print(f" {ff_name.upper()}")
        print(f"{'=' * 72}")
        header = f"{'mag':>8s}"
        for p in NUM_PARTICLES:
            header += f" | P={p:>3d} rel%  max_rel%"
        print(header)
        print("-" * 72)
        for mag in MAGNITUDES:
            row_str = f"{mag:>8.1f}"
            for p in NUM_PARTICLES:
                match = ff_df[(ff_df["magnitude"] == mag) & (ff_df["num_particles"] == p)]
                if len(match) == 1:
                    r = match.iloc[0]
                    row_str += f" | {r['avg_rel_rmse']:>7.3f}  {r['avg_max_rel_rmse']:>8.3f}"
                else:
                    row_str += f" |     N/A       N/A"
            print(row_str)
        print(f"{'=' * 72}")

    # Comparison table: all force fields at mag=1.0
    print(f"\n{'=' * 72}")
    print(f" COMPARISON AT mag=1.0 (avg_rel_rmse %)")
    print(f"{'=' * 72}")
    header = f"{'force_field':<20s}"
    for p in NUM_PARTICLES:
        header += f" | P={p:>3d}"
    print(header)
    print("-" * 72)
    for ff_name in FORCE_FIELDS:
        row_str = f"{ff_name:<20s}"
        for p in NUM_PARTICLES:
            match = res_df[
                (res_df["force_field"] == ff_name)
                & (res_df["magnitude"] == 1.0)
                & (res_df["num_particles"] == p)
            ]
            if len(match) == 1:
                row_str += f" | {match.iloc[0]['avg_rel_rmse']:>6.3f}"
            else:
                row_str += f" |    N/A"
        print(row_str)
    print(f"{'=' * 72}")

    # Plot: rel_rmse vs magnitude for each force field, separate subplot per P
    fig, axes = plt.subplots(1, len(NUM_PARTICLES), figsize=(5 * len(NUM_PARTICLES), 5),
                             sharey=True)
    if len(NUM_PARTICLES) == 1:
        axes = [axes]

    for ax, num_p in zip(axes, NUM_PARTICLES):
        subset = res_df[res_df["num_particles"] == num_p]
        for ff_name in FORCE_FIELDS:
            ff_sub = subset[subset["force_field"] == ff_name].sort_values("magnitude")
            ax.plot(ff_sub["magnitude"], ff_sub["avg_rel_rmse"],
                    marker="o", label=ff_name)
        ax.set_xlabel("magnitude (gradient strength)")
        ax.set_ylabel("avg relative RMSE (%)")
        ax.set_title(f"P={num_p}")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Linear gradient force field accuracy vs magnitude", fontsize=13)
    plt.tight_layout()
    output_png = Path("figures/linear_gradient_accuracy.png")
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved plot to {output_png}")


if __name__ == "__main__":

    #run_exp_linear_gradient_fields()

    #run_exp_force_field_variants()

    run_exp_different_sizes()
    
    #run_experiment_fixed_size_diff_operators()

    # examine(
    #     num_particles=50,
    #     vol_frac=0.1,
    #     mob_op=build_mobility_ops(SHAPE)["M_nbody"],
    #     shape=SHAPE,
    #     seed=DEFAULT_SEED,
    # )


    #run_exp_nearfield_only("nbody")


    # mob = build_mobility_ops(SHAPE)["M_nbody"]
    # accum_stats = {"rel_rmse": 0.0, "max_rel_rmse": 0.0}
    # num_runs = 5
    # for i in range(num_runs):
    #     df = generate_uniform_testcase(
    #         shape=SHAPE,
    #         volume_fraction=.2,
    #         numParticles=50,
    #         tol=1e-7,
    #         seed=DEFAULT_SEED + i,
    #         save_to_file=False,
    #         random_force_directions=False
    #     )
    #     config = df[["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].values
    #     forces = df[["f_x", "f_y", "f_z", "t_x", "t_y", "t_z"]].values
    #     velocity = df[["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]].values
    #     print(config[2])

    #     err_values = compute_errors({"mob": mob}, config, forces, velocity)
    #     stats = err_values["mob"]
    #     accum_stats["rel_rmse"] += stats["rel_rmse"]
    #     accum_stats["max_rel_rmse"] += stats["max_rel_rmse"]

    # print(
    #     f"Avg rel RMSE={accum_stats['rel_rmse'] / num_runs:.6e}, "
    #     f"Avg max rel RMSE={accum_stats['max_rel_rmse'] / num_runs:.6e}, "
    # )
