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


mob_nbody = Mob_Op_Nbody(
    shape=shape,
    self_nn_path=self_path,
    two_nn_path=two_body,
    nbody_nn_path="data/models/nbody_pinn_b1.pt",
    nn_only=False,
    rpy_only=False,
    switch_dist=6.0,
)

mob_mfs = MobOpMFS(shape, acc="fine")


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
HOW_MANY = 100
dt = 1.0 # no obvious difference between 0.02 vs 1.0 at t=128
T = 0
target_time = 128*HOW_MANY
steps = int(target_time / dt)
print_interval = int(128 / dt)
print(f"Total steps: {steps}, print every {print_interval} steps")


ops = {
    "nbody": mob_nbody,
    "2b": mob_2b,
    "mfs": mob_mfs,
}
METHOD = "2b"

def get_velocity(config, forces, radius):
    mob_op = ops[METHOD]
    return mob_op.apply(config, forces, radius)


positions_over_time = []
for it in range(steps):
    T  = it*dt
    if  it % print_interval == 0:
        print(f"Step {T}")
        print(config[:, :3])
        print("---------\n")
        positions_over_time.append(config[:, :3].copy())

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


output_filename = "/home/shihab/programs/stokesian-dynamics/stokesian_dynamics/output/2510242122-s2-i1-100fr-t128p0-M1-gravity.npz"

data1 = np.load(output_filename)

particle_positions = data1['centres']

print("particle_positions[1]:")
print(particle_positions[1])

plt.plot(particle_positions[:HOW_MANY,:,0],particle_positions[:HOW_MANY,:,2])
plt.plot(positions_over_time[:,:,0],positions_over_time[:,:,2], linestyle='--')
plt.legend(['SD','ME'])
plt.xlabel('x')
plt.ylabel('y',rotation=0)
plt.xlim([-6,13])
plt.ylim([-810,10])

name = METHOD
plt.savefig(f"tmp/rk4-{name}.png")
plt.show()