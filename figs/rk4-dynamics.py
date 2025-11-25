import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import seaborn as sns


sd_out = "/home/shihab/programs/stokesian-dynamics/stokesian_dynamics/output/2510242122-s2-i1-100fr-t128p0-M1-gravity.npz"
data1 = np.load(sd_out)
sd_positions = data1['centres'][:91]


ours = np.load("tmp/rk4_3sphere_nbody.npz.npy")
mfs =  np.load("tmp/rk4_3sphere_mfs.npz.npy")
print(ours.shape)
print(mfs.shape)
print(sd_positions.shape)


durlofsky_fig5_flat = np.genfromtxt(
	"/home/shihab/programs/stokesian-dynamics/examples/data/durlofsky_fig5_data.txt",
	delimiter=",",
)
n_spheres = 3
n_timesteps = durlofsky_fig5_flat.shape[0] // n_spheres
# File stores (x, z) coordinates for sphere 0..2 sequentially per time step; rebuild xyz layout
durlofsky_fig5_flat = durlofsky_fig5_flat.reshape(n_timesteps, n_spheres, -1)
durlofsky_fig5_data = np.zeros((n_timesteps, n_spheres, 3), dtype=durlofsky_fig5_flat.dtype)
durlofsky_fig5_data[:, :, 0] = durlofsky_fig5_flat[:, :, 0]
durlofsky_fig5_data[:, :, 2] = durlofsky_fig5_flat[:, :, 1]
#durlofsky_fig5_data= durlofsky_fig5_data[:91]
print(durlofsky_fig5_data.shape)
print(durlofsky_fig5_data[0])

# Assign one color per method so all spheres within a dataset share the same hue
method_palette = sns.color_palette("colorblind", n_colors=4)
method_colors = {
	"M_mfs": method_palette[0],
	"SD (Durlofsky et al.)": method_palette[1],
	"SD (Townsend et al.)": method_palette[2],
	"M_nbody": method_palette[3],
}


def plot_method(points, label, **plot_kwargs):
	"""Plot all spheres for a dataset using one consistent color."""
	color = method_colors[label]
	for i in range(n_spheres):
		kwargs = plot_kwargs.copy()
		kwargs.setdefault("color", color)
		kwargs["label"] = label if i == 0 else None
		plt.plot(points[:, i, 0], points[:, i, 2], **kwargs)


plot_method(mfs, label="M_mfs", linestyle='--')
plot_method(durlofsky_fig5_data, label="SD (Durlofsky et al.)",
	marker='.',
	linestyle='None',
	ms=4,
	zorder=0,
)
plot_method(sd_positions[:90], label="SD (Townsend et al.)", linestyle='--')
plot_method(ours, label="M_nbody", linestyle='-')


plt.legend()
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.xlim([-6, 13])
plt.ylim([-850, 10])

plt.savefig("figs/rk4-dynamics.pdf", dpi=600, format='pdf', bbox_inches='tight')
plt.show()