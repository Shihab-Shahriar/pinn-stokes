"""Visualize and simulate a large-scale arrangement of spheres.

Features
--------
* Places 1,600 sphere particles in a 10 x 10 x 16 lattice with 3.0 unit
	spacing, centered at the origin.
* Colors the initial configuration by a velocity proxy (distance-based, mapped
	to the [0, 100] range) and renders true unit-radius spheres with PyVista plus
	a scalar bar.
* Imports :class:`src.gpu_nbody_mob.Mob_Nbody_Torch` to evaluate the learned
	mobility operator and advances the system via explicit Euler integration with
	a 1e-3 s timestep.
* Applies a constant gravity-like body force of [0, 0, -1] to every particle for
	all timesteps to maximize reuse of the force tensor.
* Records the simulation at 50–100 FPS (user configurable), saves the frames to
	a compressed ``.npz`` file, can optionally render an MP4 preview video, and can
	replay any previously saved NPZ bundle without re-running the dynamics.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.append(str(REPO_ROOT))

from src.gpu_nbody_mob import Mob_Nbody_Torch


GRID_SHAPE = (10, 10, 16)  # (nx, ny, nz)
SPACING = 3.0  # center-to-center distance between adjacent spheres
VELOCITY_CLIM = (0.0, 100.0)
DEFAULT_DT = 1e-3
DEFAULT_TOTAL_SECONDS = 2.0
DEFAULT_OUTPUT_FPS = 60.0
SIM_OUTPUT_PATH = REPO_ROOT / "data" / "simulations" / "large_scale_dynamics.npz"


def generate_grid_points(shape: tuple[int, int, int], spacing: float) -> np.ndarray:
	"""Return an (N, 3) array of xyz positions for an evenly spaced 3D grid."""

	nx, ny, nz = shape
	offsets = [((n - 1) / 2.0) for n in shape]

	xs = (np.arange(nx) - offsets[0]) * spacing
	ys = (np.arange(ny) - offsets[1]) * spacing
	zs = (np.arange(nz) - offsets[2]) * spacing

	grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
	return grid.reshape(-1, 3)


def compute_velocity_magnitudes(points: np.ndarray) -> np.ndarray:
	"""Compute velocities scaled to [0, 100] proportional to distance from origin."""

	distances = np.linalg.norm(points, axis=1)
	min_d, max_d = distances.min(), distances.max()

	if np.isclose(max_d, min_d):
		return np.zeros_like(distances)

	scaled = (distances - min_d) / (max_d - min_d)
	return scaled * (VELOCITY_CLIM[1] - VELOCITY_CLIM[0]) + VELOCITY_CLIM[0] #Dont clip velocities


def plot_particles(
	points: np.ndarray,
	velocities: np.ndarray,
	*,
	show: bool = True,
	sphere_radius: float = 1.0,
	off_screen: bool = False,
) -> None:
	"""Render the particle arrangement with PyVista, honoring unit sphere radius."""

	if not show:
		return

	try:
		pv = importlib.import_module("pyvista")
	except ImportError as exc:
		raise RuntimeError(
			"PyVista is required for plotting. Install it via `pip install pyvista`."
		) from exc

	if off_screen:
		pv.OFF_SCREEN = True

	cloud = pv.PolyData(points)
	cloud["velocity"] = velocities
	glyphs = cloud.glyph(scale=False, geom=pv.Sphere(radius=sphere_radius))

	plotter = pv.Plotter()
	plotter.add_mesh(
		glyphs,
		scalars="velocity",
		cmap="viridis",
		clim=VELOCITY_CLIM,
		smooth_shading=True,
	)
	plotter.add_axes(line_width=2, labels_off=True)
	plotter.add_scalar_bar(title="Velocity", fmt="{:.0f}")
	plotter.show(title="10x10x16 Sphere Arrangement", auto_close=True)


def build_initial_config(points: np.ndarray) -> np.ndarray:
	"""Pack xyz + identity quaternion for each sphere into an (N, 7) array."""

	num = points.shape[0]
	config = np.zeros((num, 7), dtype=np.float32)
	config[:, :3] = points.astype(np.float32)
	config[:, 6] = 1.0  # Identity quaternion (0, 0, 0, 1)
	return config


def build_constant_force(
	num_particles: int,
	force_vector: tuple[float, float, float],
	*,
	force_scale: float,
	device: torch.device,
	dtype: torch.dtype,
) -> torch.Tensor:
	"""Precompute a constant (N, 6) force tensor reused for every timestep."""

	force = torch.zeros(num_particles, 6, device=device, dtype=dtype)
	base = torch.tensor(force_vector, device=device, dtype=dtype)
	force[:, :3] = force_scale * base
	return force


def build_mobility_operator() -> Mob_Nbody_Torch:
	"""Instantiate the GPU mobility operator with canonical checkpoints."""

	models_dir = REPO_ROOT / "data" / "models"
	return Mob_Nbody_Torch(
		shape="sphere",
		self_nn_path=str(models_dir / "self_interaction_model.pt"),
		two_nn_path=str(models_dir / "two_body_combined_model.pt"),
		nbody_nn_path=str(models_dir / "nbody_pinn_b1.pt"),
		nn_only=False,
		rpy_only=False,
		switch_dist=6.0,
	)


def run_simulation(
	mobility: Mob_Nbody_Torch,
	config0: np.ndarray,
	*,
	dt: float = DEFAULT_DT,
	total_seconds: float = DEFAULT_TOTAL_SECONDS,
	output_fps: float = DEFAULT_OUTPUT_FPS,
	viscosity: float = 1.0,
	force_scale: float = 1.0,
	force_vector: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Advance the system with explicit Euler and sample frames at the requested FPS."""

	if output_fps <= 0:
		raise ValueError("output_fps must be positive")

	device = mobility.device
	config = torch.as_tensor(config0, device=device, dtype=torch.float32).clone()
	total_steps = int(np.ceil(total_seconds / dt))
	frame_dt = 1.0 / output_fps
	next_frame_time = frame_dt
	constant_force = build_constant_force(
		config.shape[0],
		force_vector,
		force_scale=force_scale,
		device=device,
		dtype=config.dtype,
	)

	frames = [config[:, :3].detach().cpu().numpy()]
	vel_hist = [np.zeros_like(frames[0])]
	times = [0.0]

	for step in range(1, total_steps + 1):
		velocities = mobility.apply(config, constant_force, viscosity)
		config[:, :3] = config[:, :3] + dt * velocities[:, :3]

		current_time = step * dt
		if current_time + 1e-12 >= next_frame_time:
			frames.append(config[:, :3].detach().cpu().numpy())
			vel_hist.append(velocities[:, :3].detach().cpu().numpy())
			times.append(current_time)
			next_frame_time += frame_dt
			print(f"Recorded frame at t = {current_time:.4f} s")

	return np.stack(frames), np.stack(vel_hist), np.asarray(times)


def save_simulation(
	output_path: Path,
	positions: np.ndarray,
	velocities: np.ndarray,
	times: np.ndarray,
	*,
	dt: float,
	fps: float,
	spacing: float,
) -> None:
	"""Persist simulation arrays as a compressed NPZ bundle."""

	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		output_path,
		positions=positions.astype(np.float32),
		velocities=velocities.astype(np.float32),
		times=times.astype(np.float32),
		metadata=dict(dt=dt, output_fps=fps, spacing=spacing),
	)
	print(f"Saved {positions.shape[0]} frames to {output_path}")


def load_simulation(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
	"""Load a previously saved simulation NPZ bundle."""

	if not npz_path.exists():
		raise FileNotFoundError(f"Saved simulation not found at {npz_path}")

	with np.load(npz_path, allow_pickle=True) as data:
		positions = data["positions"]
		velocities = data["velocities"]
		times = data["times"]
		metadata = data["metadata"].item() if "metadata" in data.files else {}

	return positions, velocities, times, metadata


def _draw_bounding_box(
	ax,
	lims: np.ndarray,
	*,
	color: str = "black",
	linewidth: float = 1.0,
) -> None:
	"""Render a rectangular prism defined by lims = [[xmin, ymin, zmin], [xmax, ymax, zmax]]."""

	x0, y0, z0 = lims[0]
	x1, y1, z1 = lims[1]
	corners = np.array(
		[
			[x0, y0, z0],
			[x1, y0, z0],
			[x0, y1, z0],
			[x1, y1, z0],
			[x0, y0, z1],
			[x1, y0, z1],
			[x0, y1, z1],
			[x1, y1, z1],
		]
	)
	edges = [
		(0, 1),
		(0, 2),
		(1, 3),
		(2, 3),
		(4, 5),
		(4, 6),
		(5, 7),
		(6, 7),
		(0, 4),
		(1, 5),
		(2, 6),
		(3, 7),
	]
	for start, end in edges:
		sx, sy, sz = corners[start]
		ex, ey, ez = corners[end]
		ax.plot([sx, ex], [sy, ey], [sz, ez], color=color, linewidth=linewidth, alpha=0.9)


def save_simulation_video(
	video_path: Path,
	positions: np.ndarray,
	velocities: np.ndarray,
	*,
	fps: float,
	times: np.ndarray | None = None,
) -> None:
	"""Render a distant-perspective scatter video that highlights the full trajectory."""

	if positions.ndim != 3:
		raise ValueError("positions must be an array of shape (frames, particles, 3)")

	try:
		matplotlib = importlib.import_module("matplotlib")
		matplotlib.use("Agg")
		plt = importlib.import_module("matplotlib.pyplot")
		cm = importlib.import_module("matplotlib.cm")
		colors_mod = importlib.import_module("matplotlib.colors")
	except ImportError as exc:
		raise RuntimeError(
			"Video export requires matplotlib. Install it via `pip install matplotlib` and retry."
		) from exc

	imageio_spec = importlib.util.find_spec("imageio")
	if imageio_spec is None:
		raise RuntimeError(
			"Video export requires imageio. Install it via `pip install imageio imageio-ffmpeg`."
		)
	imageio = importlib.import_module("imageio")

	video_path.parent.mkdir(parents=True, exist_ok=True)
	flat_positions = positions.reshape(-1, 3)
	mins = flat_positions.min(axis=0)
	maxs = flat_positions.max(axis=0)
	centers = 0.5 * (mins + maxs)
	span = np.max(maxs - mins)
	span = max(span, 1e-6)
	half = span / 2.0
	lims = np.stack([centers - half, centers + half], axis=0)
	span_vec = lims[1] - lims[0]
	padding = span_vec * np.array([0.15, 0.15, 0.6])
	lims_vis = np.stack([lims[0] - padding, lims[1] + padding], axis=0)
	frame_speeds = np.linalg.norm(velocities, axis=2)
	norm = colors_mod.Normalize(*VELOCITY_CLIM)
	cmap = plt.colormaps["viridis"]

	fig = plt.figure(figsize=(10, 8), dpi=200)
	ax = fig.add_subplot(111, projection="3d")
	fig.patch.set_facecolor("white")
	if hasattr(ax, "set_proj_type"):
		ax.set_proj_type("persp")
	if hasattr(ax, "dist"):
		ax.dist = 11

	with imageio.get_writer(video_path, fps=fps) as writer:
		for idx, pts in enumerate(positions):
			speeds = np.clip(frame_speeds[idx], *VELOCITY_CLIM)
			colors = cmap(norm(speeds))
			ax.clear()
			ax.set_facecolor("white")
			ax.set_box_aspect((lims_vis[1] - lims_vis[0]).tolist())
			if hasattr(ax, "set_proj_type"):
				ax.set_proj_type("persp")
			if hasattr(ax, "dist"):
				ax.dist = 11
			ax.view_init(elev=0, azim=-30)
			ax.grid(False)
			ax.scatter(
				pts[:, 0],
				pts[:, 1],
				pts[:, 2],
				c=colors,
				s=8,
				edgecolors="none",
				alpha=0.9,
			)
			_draw_bounding_box(ax, lims_vis, color="#111111", linewidth=1.2)
			ax.set_xlim(lims_vis[0, 0], lims_vis[1, 0])
			ax.set_ylim(lims_vis[0, 1], lims_vis[1, 1])
			ax.set_zlim(lims_vis[0, 2], lims_vis[1, 2])
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_zticks([])
			ax.set_title(
				f"Dense settling | frame {idx + 1}/{positions.shape[0]}",
				color="#222222",
			)
			t_current = None if times is None else times[min(idx, len(times) - 1)]
			t_label = f"t = {t_current:.2f} s" if t_current is not None else f"frame {idx + 1}"
			ax.text2D(0.02, 0.92, t_label, transform=ax.transAxes, color="#222222")
			fig.canvas.draw()
			buffer = np.asarray(fig.canvas.buffer_rgba())
			image = buffer[:, :, :3].copy()
			writer.append_data(image)

	plt.close(fig)
	print(f"Saved video preview to {video_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Explicit Euler step size (s)")
	parser.add_argument(
		"--seconds",
		type=float,
		default=DEFAULT_TOTAL_SECONDS,
		help="Total simulated duration in seconds",
	)
	parser.add_argument(
		"--fps",
		type=float,
		default=DEFAULT_OUTPUT_FPS,
		help="Sampling rate for saved frames",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=SIM_OUTPUT_PATH,
		help="Destination NPZ file for recorded frames",
	)
	parser.add_argument(
		"--video-path",
		type=Path,
		default=SIM_OUTPUT_PATH.with_suffix(".mp4"),
		help="Destination video file (MP4/GIF) for a quick preview",
	)
	parser.add_argument(
		"--video-fps",
		type=float,
		default=30.0,
		help="Playback FPS for the exported video",
	)
	parser.add_argument(
		"--no-video",
		action="store_true",
		help="Skip rendering a video preview",
	)
	parser.add_argument(
		"--replay-only",
		action="store_true",
		help="Skip running the simulator and reuse a saved NPZ bundle instead",
	)
	parser.add_argument(
		"--replay-from",
		type=Path,
		default=SIM_OUTPUT_PATH,
		help="Path to the saved NPZ bundle used when --replay-only is supplied",
	)
	parser.add_argument("--no-plot", action="store_true", help="Skip the PyVista preview window")
	parser.add_argument(
		"--off-screen",
		action="store_true",
		help="Render PyVista off-screen (requires xvfb on headless machines)",
	)
	parser.add_argument(
		"--skip-sim",
		action="store_true",
		help="Only build the initial arrangement without running dynamics",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.replay_only:
		positions, velocities, times, metadata = load_simulation(args.replay_from)
		frame_count, particle_count = positions.shape[0], positions.shape[1]
		print(
			f"Loaded {frame_count} frames for {particle_count} particles from {args.replay_from}"
		)
		if metadata:
			print(f"Metadata: {metadata}")

		if not args.no_video:
			try:
				save_simulation_video(
					args.video_path,
					positions,
					velocities,
					fps=args.video_fps,
					times=times,
				)
			except RuntimeError as err:
				print(f"Video export skipped: {err}")
		else:
			print("Replay requested without video export; nothing further to do.")

		return

	points = generate_grid_points(GRID_SHAPE, SPACING)
	color_vel = compute_velocity_magnitudes(points)
	print(f"Generated {len(points)} particles with spacing {SPACING}")

	if not args.no_plot:
		plot_particles(points, color_vel, show=True, sphere_radius=1.0, off_screen=args.off_screen)

	if args.skip_sim:
		return

	config0 = build_initial_config(points)
	mobility = build_mobility_operator()
	print("Running explicit Euler simulation via Mob_Nbody_Torch ...")

	positions, velocities, times = run_simulation(
		mobility,
		config0,
		dt=args.dt,
		total_seconds=args.seconds,
		output_fps=args.fps,
		force_scale=1.0
	)

	save_simulation(
		args.output,
		positions,
		velocities,
		times,
		dt=args.dt,
		fps=args.fps,
		spacing=SPACING,
	)

	if not args.no_video:
		try:
			save_simulation_video(
				args.video_path,
				positions,
				velocities,
				fps=args.video_fps,
				times=times,
			)
		except RuntimeError as err:
			print(f"Video export skipped: {err}")


if __name__ == "__main__":
	main()
