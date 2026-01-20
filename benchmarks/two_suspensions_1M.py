import sys
import os
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from src.fmm_stkfmm import MobStkFMM
from src.gpu_nbody_mob import Mob_Nbody_Torch
from src.treecode import WarpFMM

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
def main(theta, benchmark_mode=True, t_final=0.5):
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
    two_body_model = os.path.join(project_root, "data/models/two_body_combined_model.pt")
    nbody_path = os.path.join(project_root, "data/models/nbody_pinn_b1.pt")

    print("Initializing Mob_Nbody_Torch...")
    mob_fmm = Mob_Nbody_Torch(
        shape=shape,
        self_nn_path=self_model,
        two_nn_path=two_body_model,
        nbody_nn_path=nbody_path,
        near_field_2b="nn",
        far_field_2b=None,
        near_far_switch=6.0,
    )
    
    print("Initializing MobStkFMM...")
    # fmm_solver = MobStkFMM(
    #     shape=shape, 
    #     near_field_operator=mob_fmm,
    #     mult_order=8, 
    #     max_pts=256   
    # )

    warp_solver = WarpFMM(
        near_field_operator=mob_fmm,
        theta=0.3,
        leaf_size=16,
        near_field_cutoff=6.0,
        device="cuda",
        block_dim=256,
    )


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
    
    # Perf note: tried sorting data every 10 timesteps. didn't help.
    with torch.no_grad():
        while current_time <= t_final + save_tol:
            # Check if we need to save
            if SAVE_STUFF:
                for save_t in save_timestamps:
                    if abs(current_time - save_t) < save_tol:
                        print(f"Saving configuration at t={current_time:.2f}")
                        save_plot(positions.detach().cpu().numpy(), current_time)
                        #np.save(os.path.join("figures", f"config_{current_time:.1f}.npy"), positions.detach().cpu().numpy())
            
            if current_time >= t_final - save_tol:
                break

            # Compute velocity on GPU
            print(f"Step t={current_time:.2f}")
            start = time.perf_counter()
            vel = warp_solver.apply(positions, orientations, forces, vis_arr)
            end = time.perf_counter()
            print(f"Velocity computed in {end - start:.3f} seconds.")
            
            # Extract linear velocity (first 3 components)
            v_linear = vel[:, :3]
            
            # Explicit Euler update
            positions += v_linear * dt
            
            current_time += dt

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print("Simulation complete.")
    print(f"Total simulation time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main(theta=.3, benchmark_mode=False, t_final=1.0)
