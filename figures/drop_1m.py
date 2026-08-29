
import os
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 7,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

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

def main(data_dir=None, output_path=None):
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / "figures" / "drop_1M"
    # Select 5 files: 0.1 to 0.5
    #files = [data_dir / f"particles_0.{i}.vtk" for i in range(1, 6)]

    tss = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Chosen time steps
    files = [data_dir / f"particles_{i}.vtk" for i in tss]  # 0
    
    print(f"Found {len(files)} files to plot.")

    meshes = []
    all_points = []

    for f in files:
        print(f"Reading {f}...")
        mesh = pv.read(f)
        meshes.append(mesh)
        all_points.append(mesh.points)

    # Compute global bounds
    all_points_concat = np.concatenate(all_points, axis=0)
    mins = all_points_concat.min(axis=0)
    maxs = all_points_concat.max(axis=0)
    
    # Add padding
    padding_xy = 35.0
    padding_z_top = 250.0
    padding_z_bottom = 100.0
    
    lims_vis = np.array([
        [mins[0] - padding_xy, mins[1] - padding_xy, mins[2] - padding_z_bottom],
        [maxs[0] + padding_xy, maxs[1] + padding_xy, maxs[2] + padding_z_top]
    ])
    
    print(f"Bounds: {lims_vis}")

    # Setup plot. The scene is very tall and narrow, so use narrow manual axes
    # instead of regular subplots; this removes the large empty gaps between panels.
    num_plots = len(files)
    fig = plt.figure(figsize=(4.4, 4.35), dpi=600)
    fig.patch.set_facecolor("white")

    left_margin = 0.015
    right_margin = 0.015
    bottom = 0.035
    panel_height = 0.88
    panel_width = 0.24
    base_step = (1.0 - left_margin - right_margin - panel_width) / (num_plots - 1)
    panel_step = 0.6 * base_step
    axes = [
        fig.add_axes(
            [left_margin + idx * panel_step, bottom, panel_width, panel_height],
            projection="3d",
        )
        for idx in range(num_plots)
    ]

    # Let's check the number of points in the first mesh.
    n_points = meshes[0].n_points
    print(f"Number of points per frame: {n_points}")

    downsample_factor = 1
    if n_points > 50000:
        downsample_factor = n_points // 35000
        print(f"Downsampling by factor of {downsample_factor} for visualization.")

    for i, ax in enumerate(axes):
        mesh = meshes[i]
        pts = mesh.points

        if downsample_factor > 1:
            pts = pts[::downsample_factor]

        colors = pts[:, 2]

        ax.set_facecolor("none")
        ax.patch.set_alpha(0)
        ax.set_box_aspect((lims_vis[1] - lims_vis[0]).tolist(), zoom=1.08)
        ax.set_anchor("N")

        if hasattr(ax, "set_proj_type"):
            ax.set_proj_type("ortho")

        ax.view_init(elev=-2, azim=-10)
        ax.grid(False)

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=colors,
            cmap="viridis",
            s=0.07,
            edgecolors="none",
            linewidths=0,
            alpha=0.72,
            rasterized=True,
        )

        _draw_bounding_box(ax, lims_vis, color="#222222", linewidth=0.32)

        ax.set_xlim(lims_vis[0, 0], lims_vis[1, 0])
        ax.set_ylim(lims_vis[0, 1], lims_vis[1, 1])
        ax.set_zlim(lims_vis[0, 2], lims_vis[1, 2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()

    output_path = (Path(output_path) if output_path
                   else REPO_ROOT / "figures" / "img_drop_1m.png")
    fig.savefig(output_path, format="png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Render Figure 13: six sedimentation snapshots from the "
                    "VTKs written by benchmarks/two_suspensions_1M.py.")
    ap.add_argument("--data-dir", default=None,
                    help="directory holding particles_<t>.vtk "
                         "(default figures/drop_1M)")
    ap.add_argument("--out", default=None,
                    help="output .png path; the .pdf is written alongside "
                         "(default figures/img_drop_1m.png)")
    a = ap.parse_args()
    main(data_dir=a.data_dir, output_path=a.out)
