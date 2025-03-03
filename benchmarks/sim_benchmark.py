"""
Unlike benchmark.py, this file is used to do an entire mobility matrix
application. We are starting with a O(N^2) algorithm, and will move to
a tree-based one later. Sphere only.

Next, 
We will apply two-body RPY or NN only for distance upto R (=16.0) for 
now. Particles beyond R will have simple stokes point approximation.
"""


import torch
from benchmark import ScNetwork
from grpy_tensors import mu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

def init_positions(radius, L, n_spheres):
    min_coord = radius        # So the sphere stays fully inside the domain
    max_coord = L - radius

    centers = torch.empty((n_spheres, 3))
    count = 0  # Number of accepted spheres
    attempts = 0

    while count < n_spheres:
        # Generate one candidate center

        candidate = (max_coord - min_coord) * torch.rand(3) + min_coord
        if count == 0:
            centers[count] = candidate
            count += 1
        else:
            dists = torch.norm(centers[:count] - candidate, dim=1)
            if (dists >= 2*radius+.05).all():
                centers[count] = candidate
                count += 1
        attempts += 1

    sphere_volume = 4/3 * torch.pi * radius**3
    domain_volume = L**3
    volume_fraction = n_spheres * sphere_volume / domain_volume

    print(f"Volume fraction of the particles: {volume_fraction:.4f}")

    return centers, attempts

# copied from 
# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

@torch.compile
def compute_rpy_velocities(centers, R, forces, viscosity):
    """
    centers   : (N,3) torch.Tensor of bead positions
    radii     : scalar, radius.
    forces    : (N,6) torch.Tensor of [Fx,Fy,Fz,Tx,Ty,Tz] for each bead
    viscosity : float or torch scalar

    Returns
    -------
    velocities : (N,6) torch.Tensor of velocities & angular velocities
                 [vx, vy, vz, wx, wy, wz] for each bead.
    """
    # 1) Build the (6N x 6N) RPY mobility matrix
    radii = torch.ones(N, device=device) * R

    M = mu(centers, radii, blockmatrix=False)  # shape (6N, 6N)

    # 2) Divide by viscosity (typical in low-Reynolds fluids)
    M = M / viscosity  # => shape still (6N, 6N)

    # 3) Flatten the (N,6) forces into (6N,)
    F_flat = forces.reshape(-1)

    # 4) Multiply
    V_flat = M @ F_flat  # => (6N,)

    # 5) Reshape => (N,6)
    return V_flat.view(-1, 6)

@torch.compile
def compute_nn_velocities(base_model, centers, force_torque):
    # create a (N,N,3) tensor of distances
    N = centers.shape[0]
    eye_mask   = torch.eye(N, dtype=torch.bool, device=device)

    d_vec = centers.unsqueeze(0) - centers.unsqueeze(1)
    d_vec[eye_mask] = 1.0 # dummy value to avoid divide by zero
    d_vec = d_vec.view(-1, 3).to(device) # (N^2, 3) tensor

    # compute l2 norm and l2 norm squared of d_vec
    d_vec_norm = torch.norm(d_vec, dim=1).unsqueeze(1)
    d_vec_norm_sq = d_vec_norm ** 2

    # concatenate d_vec with d_vec_norm and d_vec_norm_sq
    d_vec = torch.cat((d_vec, d_vec_norm, d_vec_norm_sq), dim=1)

    ft = force_torque.unsqueeze(0).repeat(N, 1, 1) # (N, N, 6) tensor
    ft = ft.view(-1, 6) # (N^2, 6) tensor

    velocities = base_model.forward(d_vec, ft) # (N^2, 6) tensor

    velocities = velocities.view(N, N, 6) # (N, N, 6) tensor
    # Zero out diagonal entries to exclude self-interactions
    diag_indices = torch.arange(N, device=device)
    velocities[diag_indices, diag_indices, :] = 0.0

    # Sum over the second dimension
    velocities = velocities.sum(dim=1)  # shape (N, 6)
    return velocities


# create a simulation domain of 40x40x40, and uniformly distribute 1000 particles
N = 2000
L = 50.0
R = 1.0
dt = 0.01
viscosity = 1.0

# generate random positions 
centers, attempts = init_positions(R, L, N)
print(attempts)

input_dim = 5  # matching the usage in forward pass
base_model = ScNetwork(input_dim).to(device)
base_model.eval()

# create (N,3) force and torque tensors, then concat to N,6 tensor
force = torch.randn(N, 3, device=device)
torque = torch.randn(N, 3, device=device)
force_torque = torch.cat((force, torque), dim=1).to(device)

centers = centers.to(device)

# first warmup run
velocities = compute_rpy_velocities(centers, R, force_torque, viscosity)

# with torch.no_grad():
#     velocities = compute_nn_velocities(base_model, centers, force_torque)
# assert velocities.requires_grad == False
del velocities

torch.cuda.synchronize(device)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()



velocities = compute_rpy_velocities(centers, R, force_torque, viscosity)
# with torch.no_grad():
#     velocities = compute_nn_velocities(base_model, centers, force_torque)



end.record()
torch.cuda.synchronize(device)
elapsed_sec = start.elapsed_time(end) / 1000.0

assert velocities.requires_grad == False
assert velocities.shape == (N, 6)

velocities = velocities.cpu().detach().numpy()

print(f"Elapsed time: {elapsed_sec:.4f} sec")
print(velocities[0][0])