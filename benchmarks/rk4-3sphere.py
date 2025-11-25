import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import seaborn as sns

from src.mob_op_2b_combined import NNMob as TwoBodyNNMob
from src.mob_op_3body import NNMob3B
from src.mob_op_nbody import Mob_Op_Nbody
from src.mfs import MobOpMFS


shape = "sphere"
self_path = "data/models/self_interaction_model.pt"
two_body = "data/models/two_body_combined_model.pt"
mob_2b = TwoBodyNNMob(shape, self_path, two_body, 
            nn_only=False, rpy_only=False)

mob_rpy = TwoBodyNNMob(shape, self_path, two_body, 
            nn_only=False, rpy_only=True)

mob_3b = NNMob3B(
    shape=shape,
    self_nn_path=self_path,
    two_nn_path=two_body,
    three_nn_path="data/models/3body_cross.pt",
    nn_only=False,
    rpy_only=False,
    switch_dist=6.0,
    triplet_cutoff=6.0,
)

mob_nbody = Mob_Op_Nbody(
    shape=shape,
    self_nn_path=self_path,
    two_nn_path=two_body,
    nbody_nn_path="data/models/nbody_pinn_b1.pt",
    nn_only=False,
    rpy_only=False,
    switch_dist=6.0,
)

mob_mfs = MobOpMFS(shape, acc="coarse")


pos = np.array([
    [-5.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [7.0, 0.0, 0.0],
], dtype=np.float64)


forces = np.array([
    [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
], dtype=np.float64)

rots = [Rotation.identity().as_quat(scalar_first=False) for _ in range(pos.shape[0])]
config = np.concatenate((pos, rots), axis=1)  # shape (N, 7)
print(config.shape)



# RK4 time integration
HOW_MANY = 90
dt = 1.0 # no obvious difference between 0.02 vs 1.0 at t=128
T = 0
target_time = 128*HOW_MANY
steps = int(target_time / dt)
print_interval = int(128 / dt)
print(f"Total steps: {steps}, print every {print_interval} steps")


ops = {
    "nbody": mob_nbody,
    "2b": mob_2b,
    "3b": mob_3b,
    "rpy": mob_rpy,
    "mfs": mob_mfs,
}
METHOD = "2b"

poo = 0
def get_velocity(config, forces, radius):
    mob_op = ops[METHOD]
    return mob_op.apply(config, forces, radius)


def plot_method_curves(traj_xyz, *, label, color, linestyle, **plot_kwargs):
    """Plot each sphere's x-z trajectory using shared styles for one method."""
    for idx in range(traj_xyz.shape[1]):
        plt.plot(
            traj_xyz[:, idx, 0],
            traj_xyz[:, idx, 2],
            color=color,
            linestyle=linestyle,
            label=label if idx == 0 else None,
            **plot_kwargs,
        )


output_filename = "/home/shihab/programs/stokesian-dynamics/stokesian_dynamics/output/2510242122-s2-i1-100fr-t128p0-M1-gravity.npz"
data1 = np.load(output_filename)
sd_positions = data1['centres']

positions_over_time = []
for it in range(steps):
    T  = it*dt
    if  it % print_interval == 0:
        print(f"T= {T}, iter={it}")
        print(config[:, :3])
        print("---------\n")
        positions_over_time.append(config[:, :3].copy())
        print("error with sd")
        for i in range(3):
            err = (config[i, :3] - sd_positions[it//print_interval, i])
            #config[i, 0] = sd_positions[it//print_interval, i, 0] # Fix X axis
            print(f"  Sphere {i}: error = {err}")
        print("---------\n")


    # k1
    vel1 = get_velocity(config, forces, 1.0).astype(np.float64)
    #vel1[:, 1] = 0.0  # Constrain y-velocity to zero
    
    # k2
    config2 = config.copy()
    config2[:, :3] += 0.5 * dt * vel1[:, :3]
    vel2 = get_velocity(config2, forces, 1.0).astype(np.float64)
    #vel2[:, 1] = 0.0  # Constrain y-velocity to zero

    # k3
    config3 = config.copy()
    config3[:, :3] += 0.5 * dt * vel2[:, :3]
    vel3 = get_velocity(config3, forces, 1.0).astype(np.float64)
    #vel3[:, 1] = 0.0  # Constrain y-velocity to zero

    # k4
    config4 = config.copy()
    config4[:, :3] += dt * vel3[:, :3]
    vel4 = get_velocity(config4, forces, 1.0).astype(np.float64)
    #vel4[:, 1] = 0.0  # Constrain y-velocity to zero

    if it == 0:
        print("Velocities at first step (it=0):")
        print("v1:", vel1[:, :3])
        print("v2:", vel2[:, :3])
        print("v3:", vel3[:, :3])
        print("v4:", vel4[:, :3])

    # Update position
    config[:, :3] += (dt / 6.0) * (vel1[:, :3] + 2*vel2[:, :3] + 2*vel3[:, :3] + vel4[:, :3])


positions_over_time.append(config[:, :3].copy())
print(len(positions_over_time))
print("Final positions:")
print(config[:, :3])

positions_over_time = np.array(positions_over_time)  # shape (num_steps, N, 3)
np.save(f"tmp/rk4_3sphere_{METHOD}", positions_over_time)

plot_method_curves(
    sd_positions[:HOW_MANY],
    label="SD (Townsend et al.)",
    color="tab:blue",
    linestyle="--",
)

print("sd_positions final:")
print(sd_positions[-1])

durlofsky_fig5_flat = np.genfromtxt(
    "/home/shihab/programs/stokesian-dynamics/examples/data/durlofsky_fig5_data.txt",
    delimiter=",",
)
if durlofsky_fig5_flat.shape[0] % pos.shape[0] != 0:
    raise ValueError("Durlofsky data length is not divisible by number of spheres")
n_timesteps = durlofsky_fig5_flat.shape[0] // pos.shape[0]
# File stores (x, z) coordinates for each sphere sequentially per time step; rebuild xyz layout
durlofsky_fig5_flat = durlofsky_fig5_flat.reshape(n_timesteps, pos.shape[0], -1)
if durlofsky_fig5_flat.shape[2] != 2:
    raise ValueError("Durlofsky data is expected to have exactly two spatial columns (x, z)")
durlofsky_fig5_data = np.zeros((n_timesteps, pos.shape[0], 3), dtype=durlofsky_fig5_flat.dtype)
durlofsky_fig5_data[:, :, 0] = durlofsky_fig5_flat[:, :, 0]
durlofsky_fig5_data[:, :, 2] = durlofsky_fig5_flat[:, :, 1]
print(durlofsky_fig5_data.shape)
print(durlofsky_fig5_data[0])
plot_method_curves(
    durlofsky_fig5_data,
    label="SD (Durlofsky et al.)",
    color="gray",
    linestyle="None",
    marker=".",
    ms=2,
    zorder=0,
)
plot_method_curves(
    positions_over_time,
    label=f"M_{METHOD}",
    color="tab:orange",
    linestyle="-",
)
plt.legend()
plt.xlabel('x')
plt.ylabel('y',rotation=0)
plt.xlim([-6,13])
plt.ylim([-850,10])

plt.savefig(f"figs/rk4-{METHOD}.pdf", dpi=600, bbox_inches='tight')
plt.show()