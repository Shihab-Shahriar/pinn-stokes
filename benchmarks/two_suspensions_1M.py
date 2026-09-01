import sys
import os
import io
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import time
import torch

# For large simulations: disable CUDA graphs to avoid memory fragmentation
# while still keeping torch.compile for kernel fusion/optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch._inductor.config as config
config.triton.cudagraph_skip_dynamic_graphs = True
config.triton.cudagraphs = False  
config.freezing = True

# Add src to path
# insert(0), not append: PYTHONPATH carries an older pinn-stokes checkout that
# otherwise shadows this repo's `src` package when run as `python benchmarks/...`
# (sys.path[0] is then the script's own directory, not the cwd).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from src.fmm_stkfmm import MobStkFMM
from src.gpu_nbody_mob import Mob_Nbody_Torch
from src.gpu_nbody_moments import Mob_Nbody_Moments_Torch
from src.treecode import WarpFMM
from src.treecode_widebvh import (
    WidebvhFMM, DEFAULT_MAC, DEFAULT_MAX_LEAF, DEFAULT_CART_MAC,
    DEFAULT_CART_ORDER)
from src.gpu_mob_2b import DEFAULT_TWO_BODY_CHUNK
from src.gpu_nbody_mob import DEFAULT_PAIR_CHUNK
from benchmarks.figure11_breakdown import parse_components

# Far-field solver. "widebvh" is the calibrated default (see
# benchmarks/mac_calibration.py); "warp" reproduces the published baseline.
FAR_FIELD_BACKEND = os.environ.get("NEMO_FAR_FIELD", "widebvh")

def generate_suspension_drop(center, drop_radius):
    """
    Generates a spherical suspension drop based on the description:
    1. Primitive cubic lattice.
    2. Filter by spherical domain.
    3. Random perturbation.
    """
    # 1. Define Lattice Constant
    # a = 3.5 yields the target count for R=175 (R/a = 50 boundary)
    a = 3.5
    
    # 2. Generate Primitive Cubic Lattice
    # Determine grid range to cover the sphere diameter
    # We generate points centered at 0 relative to the drop
    n_pts = int(np.ceil(2 * drop_radius / a)) + 2
    grid_1d = np.arange(n_pts) * a
    grid_1d -= np.mean(grid_1d) # Center the grid
    
    x, y, z = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    coords = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    
    # 3. Discard particles outside the spherical domain
    dists = np.linalg.norm(coords, axis=1)
    mask = dists <= drop_radius
    print("to drop:", np.sum(~mask))
    particles = coords[mask]
    
    # 4. Perturb positions randomly ("slightly")
    # We use a small fraction of the lattice constant (e.g., 5%) to avoid significant overlap
    perturbation = np.random.uniform(-0.05 * a, 0.05 * a, size=particles.shape)
    particles += perturbation
    
    # Shift to the specified center coordinates
    particles += np.array(center)
    
    return particles

def save_vtk(particles, timestamp, output_dir="figures"):
    """Save particle positions in VTK format for ParaView visualization."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"particles_{timestamp:.1f}.vtk")
    
    n_particles = len(particles)
    
    with open(filepath, 'w') as f:
        # VTK header
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"Particle positions at t={timestamp:.2f}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        
        # Write points
        f.write(f"POINTS {n_particles} float\n")
        for p in particles:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        
        # Write vertices (each point is a vertex)
        f.write(f"VERTICES {n_particles} {2 * n_particles}\n")
        for i in range(n_particles):
            f.write(f"1 {i}\n")
        
        # Add point data (particle index as scalar for coloring)
        f.write(f"POINT_DATA {n_particles}\n")
        f.write("SCALARS particle_id int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(n_particles):
            f.write(f"{i}\n")
    
    print(f"Saved VTK file: {filepath}")


def save_plot(particles, timestamp, output_dir=r"figures/drop_1M/"):
    # Save VTK file for ParaView
    save_vtk(particles, timestamp, output_dir)

    # Binary dump alongside the VTK: same data, 12 MB vs 75 MB of ASCII, and it
    # is what an A/B comparison between two `mac` values reads back.
    np.save(os.path.join(output_dir, f"positions_{timestamp:.1f}.npy"), particles)
    
    # plot a random 1% subsample
    subsample_ratio = 0.01
    indices = np.random.choice(len(particles), int(len(particles) * subsample_ratio), replace=False)
    plot_particles = particles[indices]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(plot_particles[:, 0], plot_particles[:, 1], plot_particles[:, 2], 
               s=1, c='b', alpha=0.5, label='Suspension Particles')

    # Setup view
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Configuration at t={timestamp:.2f}\n(Subsampled {subsample_ratio*100}%)')

    # Ensure aspect ratio is roughly equal
    # Use fixed limits based on initial configuration to see movement
    # Initial range was approx -200 to 600 in Z, -200 to 200 in X, Y
    # Let's use a dynamic but consistent range or just auto
    # For comparison, fixed limits are better.
    
    # Based on R_drop=175, gap=100, z_center_2 = 2*175+100 = 450.
    # Drop 1 at 0. Drop 2 at 450.
    # Z range: -175 to 450+175 = 625.
    # X, Y range: -175 to 175.
    
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_zlim(-200, 700)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"config_{timestamp:.1f}.png"), dpi=300)
    plt.close(fig)

@torch.no_grad()
def main(theta, benchmark_mode=True, t_final=0.5, *, mac=None,
         max_leaf=DEFAULT_MAX_LEAF, hilbert_q=None, seed=0,
         out_dir="figures/drop_1M/", pdeg=7, fp32_level=0,
         pair_budget_gb=None, two_body_chunk=DEFAULT_TWO_BODY_CHUNK,
         pair_chunk=DEFAULT_PAIR_CHUNK, log_csv=None, near_op="baseline"):
    # --- Simulation Parameters ---
    R_drop = 175.0
    r_particle = 1.0
    phi = 0.10          # 10.0% volume fraction

    # --- Configuration ---
    gap = 100.0 
    z_center_1 = 0.0
    z_center_2 = 2 * R_drop + gap

    # --- Simulation Loop ---
    dt = 0.01
    viscosity = 1.0
    SAVE_STUFF = False if benchmark_mode else True
    

    # Seeded so that two runs at different `mac` see a bit-identical initial
    # cloud: generate_suspension_drop jitters the lattice with
    # np.random.uniform, and unseeded that made any A/B comparison of the
    # dynamics meaningless. Matches benchmarks/far_field_drift.py.
    np.random.seed(seed)

    print("Generating Drop 1...")
    drop1 = generate_suspension_drop((0, 0, z_center_1), R_drop )

    print("Generating Drop 2...")
    drop2 = generate_suspension_drop((0, 0, z_center_2), R_drop)

    # Combine all particles
    all_particles = np.vstack([drop1, drop2])
    n_particles = len(all_particles)

    print(f"Total particles: {n_particles}")

    # --- Setup Solver ---
    shape = "sphere"
    # Adjust paths to be relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    self_model = os.path.join(project_root, "data/models/self_interaction_model.pt")
    two_body_model = os.path.join(project_root, "data/models/combined_2body.wt")
    nbody_path = os.path.join(project_root, "data/models/nbody_cross_tmp.wt")

    # "baseline" is the published operator (old K=10 n-body, switch 6);
    # "moments" is the dataset-v2 stack (moments pair model kinf_rc8_pc8 + learned
    # diagonal, both locked to switch_dist = pair_cutoff = 8, so the near/far
    # switch -- and with it the treecode's nearCutoff -- moves to 8 as well).
    if near_op == "moments":
        print("Initializing Mob_Nbody_Moments_Torch (pc8 moments + diag)...")
        near_cutoff = 8.0
        mob_fmm = Mob_Nbody_Moments_Torch(
            shape=shape,
            self_nn_path=self_model,
            two_nn_path=two_body_model,
            moments_nn_path=os.path.join(
                project_root, "experiments/nbody_moments_v2_kinf_rc8_pc8.wt"),
            diag_nn_path=os.path.join(
                project_root, "experiments/nbody_diag_v2_pc8.wt"),
            near_field_2b="nn",
            far_field_2b=None,
            switch_dist=near_cutoff,
            two_body_chunk_size=two_body_chunk,
        )
    else:
        print("Initializing Mob_Nbody_Torch...")
        near_cutoff = 6.0
        mob_fmm = Mob_Nbody_Torch(
            shape=shape,
            self_nn_path=self_model,
            two_nn_path=two_body_model,
            nbody_nn_path=nbody_path,
            near_field_2b="nn",
            far_field_2b=None,
            near_far_switch=6.0,
            # Both pair paths are chunked (two-body 4M, n-body 2M by default), which
            # is what keeps this case inside a 20 GB card (4.1 GiB vs 15.4 GiB
            # process footprint at N=1M). Exposed as CLI knobs for smaller cards.
            two_body_chunk_size=two_body_chunk,
            pair_chunk_size=pair_chunk,
        )
    
    print("Initializing MobStkFMM...")
    # fmm_solver = MobStkFMM(
    #     shape=shape, 
    #     near_field_operator=mob_fmm,
    #     mult_order=8, 
    #     max_pts=256   
    # )

    if FAR_FIELD_BACKEND.startswith("widebvh"):
        cart = FAR_FIELD_BACKEND == "widebvh-cart"
        # `mac` is per-policy: the two expansions truncate different series and
        # no value transfers between them, so None means "this policy's
        # calibrated default" rather than a single shared number.
        if mac is None:
            mac = DEFAULT_CART_MAC if cart else DEFAULT_MAC
        # Two balls in a large bounding box occupy only ~35% of it, so the
        # bucket count for a given cells-per-axis is far below the uniform
        # case and cart_hilbert_q's fill=1 assumption does not hold. 26 is
        # what the sweep found here (6101 buckets, 172 particles each:
        # 345.8 ms of far field on auto, 135.2 at 26). bary is left on auto,
        # which is within 5% of its own optimum.
        if hilbert_q is None:
            hilbert_q = 26.0 if cart else None
        elif hilbert_q <= 0.0:
            hilbert_q = None          # explicit request for the engine's auto
        warp_solver = WidebvhFMM(
            near_field_operator=mob_fmm,
            near_field_cutoff=near_cutoff,
            device="cuda",
            policy="cart" if cart else "bary",
            mac=mac,
            max_leaf=max_leaf,
            order=DEFAULT_CART_ORDER if cart else None,
            hilbert_q=hilbert_q,
            pdeg=pdeg,
            fp32_level=fp32_level,
            pair_budget_gb=pair_budget_gb,
        )
        print(f"Far field: widebvh treecode, policy={warp_solver.policy}, "
              f"mac={warp_solver.mac}, PDEG={warp_solver.pdeg}, "
              f"order={warp_solver.order}, maxLeaf={warp_solver.max_leaf}, "
              f"fp32_level={warp_solver.fp32_level}, "
              f"pair_budget_gb={warp_solver.env['TC_PAIR_BUDGET_GB']}")
    else:
        warp_solver = WarpFMM(
            near_field_operator=mob_fmm,
            theta=theta,
            leaf_size=16,
            near_field_cutoff=near_cutoff,
            device="cuda",
            block_dim=256,
        )
        print(f"Far field: Warp treecode, theta={theta}")


    device = torch.device("cuda")

    # Initial state (GPU)
    positions = torch.from_numpy(all_particles.astype(np.float32)).to(device)
    initial_positions = positions.clone()
    # Orientation: (qx, qy, qz, qw) = (0, 0, 0, 1)
    orientations = torch.zeros((n_particles, 4), dtype=torch.float32, device=device)
    orientations[:, 3] = 1.0
    
    # Forces: Gravity in -z direction. F = (0, 0, -1)
    # Torques: 0
    forces = torch.zeros((n_particles, 6), dtype=torch.float32, device=device)
    forces[:, 2] = -9.81 # Fz = -9.81

    vis_arr = torch.full((n_particles,), viscosity, dtype=torch.float32, device=device)

    current_time = 0.0
    save_interval = 0.1
    save_timestamps = np.arange(0.0, t_final + save_interval / 2, save_interval)
    save_tol = 1e-5

    print("Starting simulation...")

    # Warmup: run a few steps to trigger compilation, then reset state
    warmup_steps = 5
    with torch.no_grad():
        for _ in range(warmup_steps):
            vel = warp_solver.apply(positions, orientations, forces, vis_arr)
            positions += vel[:, :3] * dt

    positions = initial_positions.clone()
    current_time = 0.0

    start_time = time.perf_counter()
    step_rows = []

    # Perf note: tried sorting data every 10 timesteps. didn't help.
    with torch.no_grad():
        while current_time <= t_final + save_tol:
            # Check if we need to save
            if SAVE_STUFF:
                for save_t in save_timestamps:
                    if abs(current_time - save_t) < save_tol:
                        print(f"Saving configuration at t={current_time:.2f}")
                        save_plot(positions.detach().cpu().numpy(), current_time,
                                  output_dir=out_dir)
                        #np.save(os.path.join("figures", f"config_{current_time:.1f}.npy"), positions.detach().cpu().numpy())
            
            if current_time >= t_final - save_tol:
                break

            # Compute velocity on GPU
            print(f"Step t={current_time:.2f}")
            start = time.perf_counter()
            if log_csv is None:
                vel = warp_solver.apply(positions, orientations, forces, vis_arr)
            else:
                # Capture the operator's own [MobFMM]/[Mob_Nbody] stdout keys
                # for this step so the per-step breakdown lands in a CSV as
                # well as in the log. Same parser as figure11_breakdown.py.
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    vel = warp_solver.apply(positions, orientations, forces, vis_arr)
                sys.stdout.write(buf.getvalue())
                raw = parse_components(buf.getvalue())
                step_rows.append({k: (v[-1] if v else "") for k, v in raw.items()})
                step_rows[-1]["t"] = round(current_time, 4)
            end = time.perf_counter()
            print(f"Velocity computed in {end - start:.3f} seconds.")
            if log_csv is not None:
                step_rows[-1]["wall_s"] = round(end - start, 4)
            
            # Extract linear velocity (first 3 components)
            v_linear = vel[:, :3]
            
            # Explicit Euler update
            positions += v_linear * dt
            
            current_time += dt

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print("Simulation complete.")
    print(f"Total simulation time: {total_time:.2f} seconds.")

    if log_csv is not None and step_rows:
        _write_step_csv(log_csv, step_rows, dict(
            n=n_particles, near_op=near_op, near_cutoff=near_cutoff,
            mac=mac, max_leaf=max_leaf, pdeg=pdeg,
            fp32_level=fp32_level, two_body_chunk=two_body_chunk,
            pair_chunk=pair_chunk, backend=FAR_FIELD_BACKEND,
            gpu=torch.cuda.get_device_name(0)))
        _print_summary(step_rows, total_time)


def _write_step_csv(path, rows, config):
    import csv
    keys = ["t", "wall_s", "far_ms", "nsearch_ms", "self2b_ms", "nbody_ms",
            "overall_near_ms", "total_gpu_ms", "near_pairs", "peak_alloc_mb",
            "peak_total_mb"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(config) + keys)
        w.writeheader()
        for r in rows:
            w.writerow({**config, **{k: r.get(k, "") for k in keys}})
    print(f"Wrote {len(rows)} per-step rows -> {path}")


def _print_summary(rows, total_time):
    """Mean/median of the timed steps, in the same '@@' style as
    far_field_drift.py so logs can be grepped uniformly."""
    def col(k):
        return [float(r[k]) for r in rows if r.get(k) not in ("", None)]
    def stats(xs):
        if not xs:
            return "n/a"
        xs = sorted(xs)
        med = xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs)//2 - 1] + xs[len(xs)//2])
        return f"mean {sum(xs)/len(xs):9.2f}  median {med:9.2f}  min {xs[0]:9.2f}  max {xs[-1]:9.2f}"
    print(f"@@ steps={len(rows)}  total_sim_time={total_time:.2f}s  "
          f"({1e3*total_time/len(rows):.1f} ms/step wall incl. Euler update)")
    for k, label in (("wall_s", "wall_s     "), ("far_ms", "far_ms     "),
                     ("overall_near_ms", "near_ms    "),
                     ("nsearch_ms", "nsearch_ms "), ("self2b_ms", "self2b_ms  "),
                     ("nbody_ms", "nbody_ms   "), ("total_gpu_ms", "total_gpu  ")):
        print(f"@@ {label} {stats(col(k))}")
    pk = col("peak_total_mb") or col("peak_alloc_mb")
    if pk:
        print(f"@@ peak_mem_mb  max {max(pk):.0f}  "
              f"({'process total' if col('peak_total_mb') else 'torch allocated'})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="1M two-drop sedimentation (Figure 13 / nemo.pdf sec 3.4). "
                    "Defaults reproduce the checked-in timing run exactly; "
                    "--snapshots --t-final 1.0 reproduces the figure.")
    ap.add_argument("--mac", type=float, default=None,
                    help="widebvh acceptance criterion; default is the "
                         "policy's calibrated value (bary %.2f, cart %.2f). "
                         "NOT WarpFMM's theta. Saturates at 1.0 -- above that "
                         "the tree has no further nodes to accept and both the "
                         "error and the cost stop moving."
                         % (DEFAULT_MAC, DEFAULT_CART_MAC))
    ap.add_argument("--max-leaf", type=int, default=DEFAULT_MAX_LEAF,
                    help="widebvh leaf size (default %(default)s). Trades "
                         "traversal against P2P at essentially constant "
                         "accuracy, so it is the free knob on a GPU whose "
                         "fp64 P2P is relatively more expensive than the H200's.")
    ap.add_argument("--hilbert-q", type=float, default=None,
                    help="source-bucket cells per axis; default is the "
                         "policy's value (26 for cart, engine auto for bary). "
                         "Pass 0 to force the engine's auto.")
    ap.add_argument("--theta", type=float, default=0.3,
                    help="WarpFMM opening angle, used only when "
                         "NEMO_FAR_FIELD=warp (default %(default)s)")
    ap.add_argument("--t-final", type=float, default=0.5,
                    help="end time; dt is 0.01, so 0.5 is 50 steps (default, "
                         "the timing run) and 1.0 is the 100 steps Figure 13 shows")
    ap.add_argument("--snapshots", action="store_true",
                    help="write VTK/npy/png every 0.1 (the figure run); off by "
                         "default, which is the timing run")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the lattice jitter (default %(default)s). "
                         "Two runs must share it to be comparable.")
    ap.add_argument("--out-dir", default="figures/drop_1M/",
                    help="snapshot output directory (default %(default)s)")
    ap.add_argument("--pdeg", type=int, default=7,
                    help="widebvh barycentric degree (compile-time; needs the "
                         "matching libwidebvh_nemo_p<N>.so). Default %(default)s")
    ap.add_argument("--fp32-level", type=int, default=0,
                    help="widebvh fp32 fast-path level: 0 = production fp64 "
                         "kernels, 1 = fp32 M2P, 2 = + fp32 P2P, 3 = + fp32 "
                         "upward pass (needs libwidebvh_nemo*_f32l<L>.so). "
                         "Default %(default)s")
    ap.add_argument("--pair-budget-gb", type=float, default=None,
                    help="widebvh P2P pair-list budget; default is ~6%% of "
                         "VRAM (min 1 GB), which caps at 26.8M pairs on an "
                         "8 GB card -- raise it (2-3) for leaf sizes below "
                         "1024, whose pair counts otherwise spill into the "
                         "engine's tiled fallback path")
    ap.add_argument("--two-body-chunk", type=int, default=DEFAULT_TWO_BODY_CHUNK,
                    help="pairs per two-body NN chunk (default %(default)s)")
    ap.add_argument("--pair-chunk", type=int, default=DEFAULT_PAIR_CHUNK,
                    help="pairs per n-body NN chunk (default %(default)s)")
    ap.add_argument("--near-op", choices=("baseline", "moments"),
                    default="baseline",
                    help="near-field operator: 'baseline' = published stack "
                         "(old n-body, switch 6), 'moments' = pc8 moments pair "
                         "model + learned diagonal (switch 8, far field cutoff "
                         "8 to match)")
    ap.add_argument("--log-csv", default=None,
                    help="write a per-step breakdown CSV (far/near/nsearch/"
                         "self2b/nbody/total ms, wall s, peak memory) parsed "
                         "from the operator's stdout, plus an @@ summary")
    a = ap.parse_args()

    main(theta=a.theta, benchmark_mode=not a.snapshots, t_final=a.t_final,
         mac=a.mac, max_leaf=a.max_leaf, hilbert_q=a.hilbert_q, seed=a.seed,
         out_dir=a.out_dir, pdeg=a.pdeg, fp32_level=a.fp32_level,
         pair_budget_gb=a.pair_budget_gb, two_body_chunk=a.two_body_chunk,
         pair_chunk=a.pair_chunk, log_csv=a.log_csv, near_op=a.near_op)
