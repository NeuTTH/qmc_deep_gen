"""
inference_latents.py — Run posterior inference on mouse USV spectrograms and
produce a label dictionary:
    {usv_unique_label: {"watershed_label": int, "latent_coord": [x, y]}}

Also saves watershed boundary and per-cluster spiral-sampled spectrogram plots.

Usage:
    python inference_latents.py \
        --model_path="path/to/checkpoint.tar" \
        --dataloc="path/to/mouse/data" \
        --output_dir="path/to/output/dir" \
        --lattice_m=20 --bandwidth=0.6 \
        --ws_sigma=3.0 --ws_compactness=0.0
"""

import datetime
import json
import os
import pickle

import fire
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter, minimum_filter
from skimage.segmentation import watershed
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from analysis.model_helpers import get_posterior_summaries, torus_forward, torus_reverse
from data.mouse_data import load_full_mouse_data, mouse_data
from models.qmc_base import QMCLVM, TorusBasis
from models.sampling import gen_fib_basis
from train.losses import binary_lp
from train.model_saving_loading import load


# ---------------------------------------------------------------------------
# Toroidal watershed
# ---------------------------------------------------------------------------

def toroidal_watershed(
    elevation: np.ndarray,
    markers: np.ndarray,
    compactness: float = 0.0,
) -> np.ndarray:
    """Watershed on T^2 via 3x3 periodic tiling.

    Standard watershed treats image borders as hard boundaries, which is
    incorrect for the QLVM's toroidal latent space where z=0 ≡ z=1.
    Tiling the elevation and marker maps 3x3 gives the central tile correct
    periodic neighbors on all sides.

    Args:
        elevation: (H, W) scalar field. Watershed floods from minima, so
                   low values mark cluster interiors and high values mark
                   boundaries (e.g. Jacobian norm of the decoder).
        markers: (H, W) integer seed image (0 = unlabeled). Each unique
                 nonzero label is replicated identically across all 9 tiles
                 so that basins wrapping around the torus get a single label.
        compactness: skimage watershed compactness parameter.

    Returns:
        labels: (H, W) integer cluster assignments from the central tile.
    """
    H, W = elevation.shape

    tiled_elevation = np.tile(elevation, (3, 3))
    tiled_markers = np.tile(markers, (3, 3))

    tiled_labels = watershed(tiled_elevation, markers=tiled_markers,
                             compactness=compactness)

    return tiled_labels[H:2*H, W:2*W]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(latent_dim: int, c_dim: int, device: torch.device):
    """Construct QMCLVM with the fixed decoder architecture used during training."""
    decoder = nn.Sequential(
        nn.Linear(2 * latent_dim + c_dim, 2048),
        nn.Linear(2048, 64 * 8 * 8),
        nn.Unflatten(1, (64, 8, 8)),
        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        nn.Sigmoid(),
    )
    model = QMCLVM(latent_dim=latent_dim, device=device, decoder=decoder, basis=TorusBasis())
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, optimizer


# ---------------------------------------------------------------------------
# Jacobian norm elevation map
# ---------------------------------------------------------------------------

def compute_jacobian_norm_fd(model, c_dim, res=200, eps=1e-3, device=None):
    """Finite-difference approximation of ||∇_z f_θ(z)||²_F on a [0,1]² grid.

    Uses central differences along each raw coordinate dimension with periodic
    wrapping. Cost: 4 forward passes over the full grid (2 per dimension).

    Low values = decoder is locally smooth (cluster interior).
    High values = decoder changes rapidly (cluster boundary).

    Returns:
        jac_norm: (res, res) float32 array, rows=y, cols=x.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    grid_x = np.linspace(0, 1, res)
    grid_y = np.linspace(0, 1, res)
    gx, gy = np.meshgrid(grid_x, grid_y)
    z_raw = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (res², 2)

    @torch.no_grad()
    def decode(z_np):
        z_torus = torus_forward(z_np)
        z_t = torch.tensor(z_torus, dtype=torch.float32, device=device)
        if c_dim > 0:
            z_t = torch.cat([z_t, torch.zeros(len(z_t), c_dim, device=device)], dim=1)
        outs = []
        for i in range(0, len(z_t), 512):
            chunk = z_t[i:i+512]
            outs.append(model.decoder(chunk).cpu().numpy().reshape(len(chunk), -1))
        return np.concatenate(outs, axis=0)  # (N, D)

    jac_sq_sum = np.zeros(len(z_raw), dtype=np.float64)
    for dim in range(2):
        z_plus = z_raw.copy()
        z_minus = z_raw.copy()
        z_plus[:, dim] = (z_plus[:, dim] + eps) % 1.0
        z_minus[:, dim] = (z_minus[:, dim] - eps) % 1.0
        df = (decode(z_plus) - decode(z_minus)) / (2 * eps)  # (N, D)
        jac_sq_sum += (df ** 2).sum(axis=1)

    return jac_sq_sum.reshape(res, res).astype(np.float32)


def _select_prominent_minima(jac_smooth, n_clusters, res):
    """Select n_clusters seed points from Jacobian norm local minima.

    Finds all local minima of jac_smooth, ranks by depth (lowest value = most
    interior to a decoder-smooth basin), then greedily picks with a minimum
    toroidal separation so no two seeds land in the same basin.

    Returns:
        marker_img: (res, res) int array, seeds labeled 1..K
        centers: (K, 2) array of (x, y) coords in [0, 1]
    """
    fp = max(3, int(res * 0.04))
    local_min_mask = jac_smooth == minimum_filter(jac_smooth, size=fp, mode='wrap')
    ys, xs = np.where(local_min_mask)
    order = np.argsort(jac_smooth[ys, xs])   # deepest first
    ys, xs = ys[order], xs[order]

    min_sep = max(5, int(res * 0.08))
    sel_ys, sel_xs = [], []
    for y, x in zip(ys, xs):
        if len(sel_ys) >= n_clusters:
            break
        if sel_ys:
            dy = np.minimum(np.abs(np.array(sel_ys) - y), res - np.abs(np.array(sel_ys) - y))
            dx = np.minimum(np.abs(np.array(sel_xs) - x), res - np.abs(np.array(sel_xs) - x))
            if np.sqrt(dx**2 + dy**2).min() < min_sep:
                continue
        sel_ys.append(y)
        sel_xs.append(x)

    if len(sel_ys) < n_clusters:
        print(f'  Warning: only {len(sel_ys)} minima found with min_sep={min_sep}px '
              f'(requested {n_clusters}); try smaller ws_sigma or n_clusters')

    marker_img = np.zeros((res, res), dtype=int)
    centers = []
    for i, (y, x) in enumerate(zip(sel_ys, sel_xs)):
        marker_img[y, x] = i + 1
        centers.append([x / res, y / res])
    centers = np.array(centers) if centers else np.empty((0, 2), dtype=np.float32)
    return marker_img, centers


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _scatter_by_labels(ax, latent_coords, ws_grid, res, centers, colors,
                       boundary_levels, xx, yy, title):
    """Shared scatter-coloring logic for watershed panels."""
    n_clusters = len(centers)
    px = np.clip((latent_coords[:, 0] * res).astype(int), 0, res - 1)
    py = np.clip((latent_coords[:, 1] * res).astype(int), 0, res - 1)
    sample_labels = ws_grid[py, px]
    for ci in range(n_clusters):
        mask = sample_labels == ci + 1
        ax.scatter(latent_coords[mask, 0], latent_coords[mask, 1],
                   c=[colors[ci]], s=3, alpha=0.3, linewidths=0,
                   rasterized=True, label=str(ci + 1))
    ax.contour(xx, yy, ws_grid, levels=boundary_levels,
               colors='black', linewidths=1.0, alpha=0.9)
    for i, (cx, cy) in enumerate(centers):
        ax.text(cx, cy, str(i + 1), color='black', fontsize=11,
                fontweight='bold', ha='center', va='center', zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xlabel('Latent dim 1'); ax.set_ylabel('Latent dim 2')
    ax.set_title(title)
    return sample_labels


def _heatmap_with_boundaries(ax, heatmap, ws_grid, centers, boundary_levels,
                              xx, yy, vmax, title):
    """Shared heatmap + contour logic for watershed panels."""
    ax.imshow(heatmap, origin='lower', extent=(0, 1, 0, 1),
              cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
    ax.contour(xx, yy, ws_grid, levels=boundary_levels,
               colors='white', linewidths=1.2)
    for i, (cx, cy) in enumerate(centers):
        ax.text(cx, cy, str(i + 1), color='white', fontsize=11,
                fontweight='bold', ha='center', va='center', zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xlabel('Latent dim 1'); ax.set_ylabel('Latent dim 2')
    ax.set_title(title)


def plot_watershed(latent_coords, heatmap, ws_labels, centers, save_path,
                   ws_labels_periodic=None):
    """2x2 comparison: top row = standard watershed, bottom row = periodic.
    Left column = heatmap + boundaries, right column = scatter colored by label.
    Uses identical colormap so visual comparison is consistent."""
    res = heatmap.shape[0]
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    n_clusters = len(centers)
    boundary_levels = np.arange(0.5, n_clusters + 1.5)

    cmap = cm.get_cmap('tab20', n_clusters)
    colors = [cmap(i) for i in range(n_clusters)]

    nz = heatmap[heatmap > 0]
    vmax = float(np.percentile(nz, 95)) if nz.size else None

    has_periodic = ws_labels_periodic is not None
    n_rows = 2 if has_periodic else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing

    # Row 0: standard watershed
    _heatmap_with_boundaries(axes[0, 0], heatmap, ws_labels, centers,
                              boundary_levels, xx, yy, vmax,
                              'Standard watershed — density + boundaries')
    _scatter_by_labels(axes[0, 1], latent_coords, ws_labels, res, centers,
                       colors, boundary_levels, xx, yy,
                       'Standard watershed — cluster scatter')

    # Row 1: periodic watershed
    if has_periodic:
        _heatmap_with_boundaries(axes[1, 0], heatmap, ws_labels_periodic, centers,
                                  boundary_levels, xx, yy, vmax,
                                  'Periodic watershed — density + boundaries')
        _scatter_by_labels(axes[1, 1], latent_coords, ws_labels_periodic, res,
                           centers, colors, boundary_levels, xx, yy,
                           'Periodic watershed — cluster scatter')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {os.path.basename(save_path)}')
    plt.close()


def plot_toroidal_comparison(latent_coords, heatmap, ws_labels, ws_labels_periodic,
                             centers, save_path):
    """3-panel figure for toroidal vs standard watershed comparison.

    Panel 1: 3x3 tiled scatter with periodic watershed coloring (shows wrapping).
    Panel 2: diff map highlighting samples that changed label (red).
    Panel 3: per-cluster size comparison (bar chart).
    """
    res = heatmap.shape[0]
    n_clusters = len(centers)
    boundary_levels = np.arange(0.5, n_clusters + 1.5)
    cmap = cm.get_cmap('tab20', n_clusters)
    colors = [cmap(i) for i in range(n_clusters)]

    px = np.clip((latent_coords[:, 0] * res).astype(int), 0, res - 1)
    py = np.clip((latent_coords[:, 1] * res).astype(int), 0, res - 1)
    ws_std = ws_labels[py, px]
    ws_per = ws_labels_periodic[py, px]
    changed = ws_std != ws_per

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ── Panel 1: 3x3 tiled scatter with periodic labels ─────────────────
    ax = axes[0]
    for di in range(3):
        for dj in range(3):
            offset_x = latent_coords[:, 0] + dj
            offset_y = latent_coords[:, 1] + di
            for ci in range(n_clusters):
                mask = ws_per == ci + 1
                ax.scatter(offset_x[mask], offset_y[mask],
                           c=[colors[ci]], s=0.5, alpha=0.15,
                           linewidths=0, rasterized=True)
    # Highlight central tile border
    ax.axhline(1, color='white', lw=1.5, ls='--', alpha=0.7)
    ax.axhline(2, color='white', lw=1.5, ls='--', alpha=0.7)
    ax.axvline(1, color='white', lw=1.5, ls='--', alpha=0.7)
    ax.axvline(2, color='white', lw=1.5, ls='--', alpha=0.7)
    # Mark central tile
    from matplotlib.patches import Rectangle
    rect = Rectangle((1, 1), 1, 1, linewidth=2.5, edgecolor='white',
                      facecolor='none', zorder=10)
    ax.add_patch(rect)
    ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.set_aspect('equal')
    ax.set_xlabel('Latent dim 1 (tiled)'); ax.set_ylabel('Latent dim 2 (tiled)')
    ax.set_title('3x3 tiled view — periodic watershed labels')

    # ── Panel 2: diff map ────────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(latent_coords[~changed, 0], latent_coords[~changed, 1],
               c='lightgray', s=2, alpha=0.15, linewidths=0, rasterized=True)
    ax.scatter(latent_coords[changed, 0], latent_coords[changed, 1],
               c='red', s=6, alpha=0.7, linewidths=0, rasterized=True,
               label=f'changed: {changed.sum()} ({100*changed.mean():.1f}%)')
    # Overlay both boundary sets
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    ax.contour(xx, yy, ws_labels, levels=boundary_levels,
               colors='black', linewidths=0.8, alpha=0.4, linestyles='dashed')
    ax.contour(xx, yy, ws_labels_periodic, levels=boundary_levels,
               colors='blue', linewidths=1.2, alpha=0.9)
    for i, (cx, cy) in enumerate(centers):
        ax.text(cx, cy, str(i + 1), color='black', fontsize=11,
                fontweight='bold', ha='center', va='center', zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xlabel('Latent dim 1'); ax.set_ylabel('Latent dim 2')
    ax.set_title('Label differences (black dashed=std, blue=periodic)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # ── Panel 3: per-cluster size comparison ─────────────────────────────
    ax = axes[2]
    cluster_ids = np.arange(1, n_clusters + 1)
    sizes_std = np.array([(ws_std == c).sum() for c in cluster_ids])
    sizes_per = np.array([(ws_per == c).sum() for c in cluster_ids])
    x_pos = np.arange(n_clusters)
    w = 0.35
    ax.bar(x_pos - w/2, sizes_std, w, label='Standard', color='gray', alpha=0.7)
    ax.bar(x_pos + w/2, sizes_per, w, label='Periodic', color='steelblue', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(c) for c in cluster_ids])
    ax.set_xlabel('Cluster ID'); ax.set_ylabel('Count')
    ax.set_title('Cluster sizes: standard vs periodic')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {os.path.basename(save_path)}')
    plt.close()


# ---------------------------------------------------------------------------
# Cluster sample-picking strategies
# ---------------------------------------------------------------------------

def _pick_cluster_samples(pts, mask, centroid, cluster_label, ws_labels_grid,
                           n_per, method, res=200):
    """Pick up to n_per dataset indices from one cluster.

    pts:           (M, 2) latent coords of in-cluster samples
    mask:          (M,) global dataset indices
    centroid:      (cx, cy) cluster centroid
    cluster_label: integer label in ws_labels_grid for this cluster
    method: 'spiral' | 'random' | 'nearest' | 'farthest_point' | 'grid'
    """
    if len(mask) == 0:
        return np.array([], dtype=int)

    n_take = min(n_per, len(mask))
    cx0, cy0 = centroid

    if method == 'random':
        idxs = np.random.choice(len(mask), n_take, replace=False)
        return mask[np.sort(idxs)]

    elif method == 'nearest':
        dists = np.sqrt((pts[:, 0] - cx0) ** 2 + (pts[:, 1] - cy0) ** 2)
        return mask[np.argsort(dists)[:n_take]]

    elif method == 'farthest_point':
        # Iterative farthest-point: maximally spread coverage of the cluster.
        # Start from centroid so the first pick is the point farthest from center.
        selected_local = []
        sel_xy = np.array([[cx0, cy0]])
        remaining = np.arange(len(mask))
        for _ in range(n_take):
            cands = pts[remaining]                             # (R, 2)
            diffs = cands[:, None, :] - sel_xy[None, :, :]   # (R, S, 2)
            d2_min = (diffs ** 2).sum(axis=-1).min(axis=-1)   # (R,)
            best_r = int(np.argmax(d2_min))
            selected_local.append(remaining[best_r])
            sel_xy = np.vstack([sel_xy, pts[remaining[best_r]]])
            remaining = np.delete(remaining, best_r)
        return mask[np.array(selected_local, dtype=int)]

    elif method == 'grid':
        # Dense grid over [0,1]^2; only use cells whose center falls inside
        # this cluster's watershed region, then evenly subsample to n_per.
        n_side = max(int(np.ceil(np.sqrt(n_per) * 3)), 30)
        gx_vals = np.linspace(0, 1, n_side + 2)[1:-1]
        gy_vals = np.linspace(0, 1, n_side + 2)[1:-1]
        cluster_cells = [
            (gx, gy)
            for gx in gx_vals
            for gy in gy_vals
            if ws_labels_grid[
                int(np.clip(gy * res, 0, res - 1)),
                int(np.clip(gx * res, 0, res - 1)),
            ] == cluster_label
        ]
        if len(cluster_cells) > n_per:
            sel_idx = np.linspace(0, len(cluster_cells) - 1, n_per).astype(int)
            cluster_cells = [cluster_cells[i] for i in sel_idx]
        used = set()
        picks = []
        for gx, gy in cluster_cells:
            d2 = (pts[:, 0] - gx) ** 2 + (pts[:, 1] - gy) ** 2
            for kk in np.argsort(d2):
                idx = int(mask[kk])
                if idx not in used:
                    used.add(idx)
                    picks.append(idx)
                    break
        return np.array(picks, dtype=int)

    elif method == 'spiral':
        dists_c = np.sqrt((pts[:, 0] - cx0) ** 2 + (pts[:, 1] - cy0) ** 2)
        r_max = float(dists_c.max()) if len(dists_c) > 0 else 0.1
        n_turns, n_dense = 4, 4000
        t_dense = np.linspace(0.0, 1.0, n_dense)
        theta_d = 2 * np.pi * n_turns * t_dense
        r_d = r_max * t_dense
        xs_d = cx0 + r_d * np.cos(theta_d)
        ys_d = cy0 + r_d * np.sin(theta_d)
        px_d = np.clip((xs_d * res).astype(int), 0, res - 1)
        py_d = np.clip((ys_d * res).astype(int), 0, res - 1)
        inside = (
            (xs_d >= 0) & (xs_d <= 1) & (ys_d >= 0) & (ys_d <= 1)
            & (ws_labels_grid[py_d, px_d] == cluster_label)
        )
        xs_in, ys_in = xs_d[inside], ys_d[inside]
        if len(xs_in) >= n_per:
            sel = np.linspace(0, len(xs_in) - 1, n_per).astype(int)
            xs_in, ys_in = xs_in[sel], ys_in[sel]
        used = set()
        picks = []
        for tx_, ty_ in zip(xs_in, ys_in):
            d2 = (pts[:, 0] - tx_) ** 2 + (pts[:, 1] - ty_) ** 2
            for kk in np.argsort(d2):
                idx = int(mask[kk])
                if idx not in used:
                    used.add(idx)
                    picks.append(idx)
                    break
        return np.array(picks, dtype=int)

    else:
        raise ValueError(
            f"Unknown sampling_method {method!r}. "
            "Choose from 'spiral', 'random', 'nearest', 'farthest_point', 'grid'."
        )


def plot_clusters_sampling(latent_coords, sample_ws, centers, ws_labels_grid,
                           full_ds, heatmap, save_dir, res=200,
                           sampling_method='spiral', n_per=30, max_plot_num=40):
    """One figure per cluster: left = spatial map with numbered picks;
    right = spectrogram grid (rows/cols determined by actual picks, capped at max_plot_num).

    sampling_method: 'spiral' | 'random' | 'nearest' | 'farthest_point' | 'grid'
    """
    n_clusters = len(centers)

    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    boundary_levels = np.arange(0.5, n_clusters + 1.5)
    nz = heatmap[heatmap > 0]
    vmax = float(np.percentile(nz, 95)) if nz.size else None

    import matplotlib.gridspec as gridspec

    for ci in range(n_clusters):
        mask = np.where(sample_ws == ci + 1)[0]
        pts = latent_coords[mask]  # (M, 2)

        picks = _pick_cluster_samples(
            pts, mask, tuple(centers[ci]), ci + 1,
            ws_labels_grid, n_per, sampling_method, res=res,
        )
        picks = picks[:max_plot_num]

        n_actual_cols = min(6, len(picks)) if len(picks) > 0 else 1
        n_actual_rows = int(np.ceil(len(picks) / n_actual_cols)) if n_actual_cols > 0 else 1

        fig = plt.figure(figsize=(6 + n_actual_cols * 1.5, max(6, n_actual_rows * 1.5 + 1)))
        outer = gridspec.GridSpec(1, 2, figure=fig,
                                  width_ratios=[1, 1.05], wspace=0.15)

        # Left: posterior heatmap + contours + numbered picks
        ax_l = fig.add_subplot(outer[0, 0])
        ax_l.imshow(heatmap, origin='lower', extent=(0, 1, 0, 1),
                    cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
        ax_l.contour(xx, yy, ws_labels_grid, levels=boundary_levels,
                     colors='black', linewidths=1.2)
        ax_l.contour(xx, yy, (ws_labels_grid == ci + 1).astype(int),
                     levels=[0.5], colors='red', linewidths=2.5)
        ax_l.scatter(centers[:, 0], centers[:, 1],
                     c='black', s=50, marker='x', linewidths=1.5, zorder=9)
        ax_l.scatter([centers[ci, 0]], [centers[ci, 1]],
                     c='red', s=100, marker='x', linewidths=2.5, zorder=10)
        for k, pidx in enumerate(picks):
            ax_l.text(latent_coords[pidx, 0], latent_coords[pidx, 1],
                      str(k + 1), color='black', fontsize=6,
                      fontweight='bold', ha='center', va='center', zorder=12)
        ax_l.set_xlim(0, 1); ax_l.set_ylim(0, 1); ax_l.set_aspect('equal')
        ax_l.set_xlabel('Latent dimension 1', fontsize=10)
        ax_l.set_ylabel('Latent dimension 2', fontsize=10)
        ax_l.set_title(f'Cluster {ci + 1} — {sampling_method} picks',
                       fontsize=11, fontweight='bold', color='red')

        # Right: spectrogram grid
        right_gs = gridspec.GridSpecFromSubplotSpec(
            n_actual_rows, n_actual_cols, subplot_spec=outer[0, 1],
            hspace=0.25, wspace=0.08)
        for j in range(n_actual_rows * n_actual_cols):
            rr, cc = divmod(j, n_actual_cols)
            ax_s = fig.add_subplot(right_gs[rr, cc])
            if j < len(picks):
                spec = full_ds[picks[j]][0].numpy().squeeze()
                ax_s.imshow(spec, cmap='viridis', origin='lower', aspect='auto')
                ax_s.set_title(str(j + 1), fontsize=7, color='black', pad=1)
                for spine in ax_s.spines.values():
                    spine.set_visible(False)
            else:
                ax_s.axis('off')
            ax_s.set_xticks([]); ax_s.set_yticks([])

        plt.suptitle(
            f'Cluster {ci + 1} — {sampling_method}-sampled spectrograms ({len(picks)} shown)',
            fontsize=11, fontweight='bold',
        )
        out_path = os.path.join(save_dir, f'cluster_{ci + 1:02d}_{sampling_method}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
    print(f'Saved: cluster_XX_{sampling_method}.png ({n_clusters} figures)')


# ---------------------------------------------------------------------------
# Toroidal boundary diagnostic
# ---------------------------------------------------------------------------

def plot_torus_boundary_diagnostic(
    latent_coords, sample_ws, sample_ws_periodic, full_ds,
    heatmap, ws_labels, ws_labels_periodic, centers, save_path,
    edge_threshold=0.05, n_show=24, res=200,
):
    """Two-row diagnostic for toroidal boundary effects.

    Row 0 — Edge samples: within edge_threshold of the x=0/1 or y=0/1 boundary.
             Spectrograms are sorted most-extreme first.
    Row 1 — Switched samples: standard and periodic watershed disagree.
             Spectrograms are grouped by (std_label → per_label) transition type.

    Each row: left = scatter/heatmap map (dashed white = standard boundary,
    solid cyan = periodic boundary); right = n_show spectrogram grid.
    Spectrogram titles show 'std→per' in red when labels differ, plain label otherwise.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle

    n_clusters = len(centers)
    boundary_levels = np.arange(0.5, n_clusters + 1.5)
    cmap = cm.get_cmap('tab20', n_clusters)
    colors = [cmap(i) for i in range(n_clusters)]
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    nz = heatmap[heatmap > 0]
    vmax = float(np.percentile(nz, 95)) if nz.size else None

    # Distance to the nearest edge on the torus (min over all four sides)
    edge_dist = np.minimum(
        np.minimum(latent_coords[:, 0], 1.0 - latent_coords[:, 0]),
        np.minimum(latent_coords[:, 1], 1.0 - latent_coords[:, 1]),
    )
    edge_mask = edge_dist < edge_threshold
    switched_mask = sample_ws != sample_ws_periodic

    n_cols_spec = 6
    n_rows_spec = max(1, n_show // n_cols_spec)
    row_h = max(5.0, n_rows_spec * 1.5 + 1.2)

    row_defs = [
        (edge_mask,
         f'Edge samples  |coord| < {edge_threshold} from boundary  (N={edge_mask.sum()})'),
        (switched_mask,
         f'Label-switched  std ≠ periodic  (N={switched_mask.sum()})'),
    ]

    fig = plt.figure(figsize=(4 + n_cols_spec * 1.6, row_h * 2 + 0.6))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45)

    for row_idx, (row_mask, row_title) in enumerate(row_defs):
        row_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[row_idx],
            width_ratios=[1, 1.8], wspace=0.12,
        )

        # ── Left: scatter / heatmap ───────────────────────────────────────
        ax_map = fig.add_subplot(row_gs[0, 0])
        ax_map.imshow(heatmap, origin='lower', extent=(0, 1, 0, 1),
                      cmap='viridis', vmin=0, vmax=vmax, aspect='equal', alpha=0.55)

        # Gray background
        ax_map.scatter(
            latent_coords[~row_mask, 0], latent_coords[~row_mask, 1],
            c='lightgray', s=1, alpha=0.15, linewidths=0, rasterized=True,
        )
        # Highlighted samples colored by periodic label
        idxs = np.where(row_mask)[0]
        for ci in range(n_clusters):
            sub = idxs[sample_ws_periodic[idxs] == ci + 1]
            if len(sub):
                ax_map.scatter(latent_coords[sub, 0], latent_coords[sub, 1],
                               c=[colors[ci]], s=8, alpha=0.85, linewidths=0,
                               rasterized=True)

        # Standard boundary (dashed white) and periodic boundary (solid cyan)
        ax_map.contour(xx, yy, ws_labels, levels=boundary_levels,
                       colors='white', linewidths=0.8, linestyles='dashed', alpha=0.6)
        ax_map.contour(xx, yy, ws_labels_periodic, levels=boundary_levels,
                       colors='cyan', linewidths=1.3, alpha=0.9)

        if row_idx == 0:
            # Shade the four edge bands
            for xy, w, h in [
                ((0, 0), edge_threshold, 1),
                ((1 - edge_threshold, 0), edge_threshold, 1),
                ((0, 0), 1, edge_threshold),
                ((0, 1 - edge_threshold), 1, edge_threshold),
            ]:
                ax_map.add_patch(Rectangle(
                    xy, w, h, facecolor='orange', alpha=0.18, linewidth=0,
                ))

        for i, (cx, cy) in enumerate(centers):
            ax_map.text(cx, cy, str(i + 1), color='white', fontsize=9,
                        fontweight='bold', ha='center', va='center', zorder=5)
        ax_map.set_xlim(0, 1); ax_map.set_ylim(0, 1); ax_map.set_aspect('equal')
        ax_map.set_xlabel('Latent dim 1', fontsize=9)
        ax_map.set_ylabel('Latent dim 2', fontsize=9)
        ax_map.set_title(row_title, fontsize=9)

        # ── Right: spectrogram grid ───────────────────────────────────────
        if len(idxs) == 0:
            # Nothing to show; leave grid blank
            picks = np.array([], dtype=int)
        elif row_idx == 0:
            order = np.argsort(edge_dist[idxs])    # most extreme (closest to edge) first
            picks = idxs[order[:n_show]]
        else:
            # Group by transition type (std_label * 1000 + per_label)
            pair_key = sample_ws[idxs].astype(np.int64) * 1000 + sample_ws_periodic[idxs]
            order = np.argsort(pair_key)
            picks = idxs[order[:n_show]]

        spec_gs = gridspec.GridSpecFromSubplotSpec(
            n_rows_spec, n_cols_spec, subplot_spec=row_gs[0, 1],
            hspace=0.38, wspace=0.08,
        )
        for j in range(n_rows_spec * n_cols_spec):
            rr, cc = divmod(j, n_cols_spec)
            ax_s = fig.add_subplot(spec_gs[rr, cc])
            if j < len(picks):
                pidx = picks[j]
                spec = full_ds[pidx][0].numpy().squeeze()
                ax_s.imshow(spec, cmap='viridis', origin='lower', aspect='auto')
                lbl_s = sample_ws[pidx]
                lbl_p = sample_ws_periodic[pidx]
                title_str = f'{lbl_s}→{lbl_p}' if lbl_s != lbl_p else str(lbl_s)
                title_col = 'red' if lbl_s != lbl_p else 'black'
                ax_s.set_title(title_str, fontsize=6, color=title_col, pad=1)
                for spine in ax_s.spines.values():
                    spine.set_visible(False)
            else:
                ax_s.axis('off')
            ax_s.set_xticks([]); ax_s.set_yticks([])

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {os.path.basename(save_path)}')
    plt.close()


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_inference(
    model_path: str,
    dataloc: str,
    output_dir: str,
    # Lattice
    lattice_m: int = 20,
    # Clustering
    n_clusters: int = 10,
    res: int = 200,
    # Watershed segmentation
    heatmap_sigma: float = 0.5,   # first smooth on sample-space histogram (notebook: 0.5)
    ws_sigma: float = 3.0,        # second smooth before watershed (notebook grid: swept 1–20)
    ws_compactness: float = 0.0,
    # Data loading
    batch_size: int = 256,
    total_samples: int = None,
    filter_mask: bool = True,
    lo: int = 0,
    hi: int = 15,  # Assuming the full dataset
    # Model
    c_dim: int = 0,
    # Caching
    cache_posteriors: bool = False,
    # Cluster spectrogram sampling
    sampling_method: str = 'spiral',
    n_per: int = 30,
    max_plot_num: int = 40,
    # Boundary diagnostic
    edge_threshold: float = 0.05,
    n_boundary_show: int = 24,
):
    """
    Run inference on mouse USV spectrograms and write a label dictionary.

    All outputs are saved inside output_dir:
        latents_full_YYYYMMDD_HHMMSS.pkl      — label dictionary + provenance
        latents_full_YYYYMMDD_HHMMSS_provenance.json
        watershed_check.png
        toroidal_vs_standard.png
        torus_boundary_diagnostic.png
        cluster_XX_<sampling_method>.png
    """
    t_start = datetime.datetime.now()
    date_tag = t_start.strftime('%Y%m%d')
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'latents_full_{date_tag}.pkl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Data ────────────────────────────────────────────────────────────────
    print('Loading mouse data...')
    full_dict = load_full_mouse_data(dataloc)
    full_ds = mouse_data(full_dict, filter_mask=filter_mask, lo=lo, hi=hi,
                         sampling_strategy='subsample', total_samples=total_samples)
    n_workers = min(24, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4)
    loader = DataLoader(full_ds, num_workers=n_workers, shuffle=False, batch_size=batch_size)
    print(f'  {len(full_ds)} samples')

    # ── Model ────────────────────────────────────────────────────────────────
    print('Loading model...')
    model, optimizer = build_model(latent_dim=2, c_dim=c_dim, device=device)
    model, optimizer, run_info = load(model, optimizer, model_path)
    model.to(device)
    model.eval()

    # ── Lattice ──────────────────────────────────────────────────────────────
    print(f'Generating Fibonacci lattice (m={lattice_m})...')
    lattice = gen_fib_basis(m=lattice_m)
    # gen_fib_basis returns the lattice UNWRAPPED; wrap it before anything treats a
    # coordinate as a position on the torus. torus_forward below is periodic either way.
    lattice_np = lattice.numpy() % 1.0
    lattice_embedded = torus_forward(lattice_np)   # (N_lattice, 4)
    print(f'  Lattice size: {len(lattice_np)}')

    # ── Posteriors ───────────────────────────────────────────────────────────
    cache_path = os.path.join(save_dir, 'posterior_cache.npz')
    if cache_posteriors and os.path.exists(cache_path):
        print(f'Loading cached posteriors from {cache_path}...')
        cache = np.load(cache_path)
        torus_weighted = cache['torus_weighted']
        aggregated = cache['aggregated']
        weights = cache['weights']
    else:
        print('Computing posteriors...')
        torus_weighted, aggregated, weights = get_posterior_summaries(
            model, lattice, loader, binary_lp, c_fn=None,
        )
        if cache_posteriors:
            np.savez(cache_path,
                     torus_weighted=torus_weighted,
                     aggregated=aggregated,
                     weights=weights)
            print(f'  Cached to {cache_path}')

    latent_coords = torus_reverse(torus_weighted, dim=2) % 1.0   # (N, 2)

    # ── Watershed segmentation ───────────────────────────────────────────────
    edges = np.linspace(0, 1, res + 1)

    # Sample histogram — retained for visualization overlays only
    heatmap_samples, _, _ = np.histogram2d(
        latent_coords[:, 0], latent_coords[:, 1],
        bins=[edges, edges],
    )
    heatmap_samples = heatmap_samples.T   # rows=y, cols=x
    heatmap = gaussian_filter(heatmap_samples, sigma=heatmap_sigma, mode='wrap')

    # ── Jacobian norm elevation map ──────────────────────────────────────────
    print('Computing decoder Jacobian norm (finite-difference)...')
    jac_norm = compute_jacobian_norm_fd(model, c_dim=c_dim, res=res, device=device)
    jac_smooth = (
        gaussian_filter(jac_norm, sigma=ws_sigma, mode='wrap')
        if ws_sigma is not None and ws_sigma > 0
        else jac_norm
    )

    # ── Marker seeding: K most prominent Jacobian minima ─────────────────────
    print(f'Selecting {n_clusters} prominent Jacobian minima...')
    marker_img, centers = _select_prominent_minima(jac_smooth, n_clusters, res)
    print(f'  Placed {len(centers)} markers')

    # ── Standard watershed on Jacobian elevation (basins = low Jacobian) ─────
    ws_labels = watershed(jac_smooth, markers=marker_img, compactness=ws_compactness)

    # Assign each sample to a watershed region (1-indexed)
    sample_px = np.clip((latent_coords[:, 0] * res).astype(int), 0, res - 1)
    sample_py = np.clip((latent_coords[:, 1] * res).astype(int), 0, res - 1)
    sample_ws = ws_labels[sample_py, sample_px]

    # ── Periodic-padded watershed (toroidal-aware) ───────────────────────────
    print('Running periodic-padded watershed...')
    ws_labels_periodic = toroidal_watershed(
        jac_smooth, marker_img, compactness=ws_compactness,
    )
    sample_ws_periodic = ws_labels_periodic[sample_py, sample_px]

    # Diagnostic: how many samples change label
    n_changed = int((sample_ws != sample_ws_periodic).sum())
    pct_changed = 100.0 * n_changed / len(sample_ws)
    print(f'  Periodic vs standard: {n_changed}/{len(sample_ws)} '
          f'samples differ ({pct_changed:.2f}%)')

    # ── Compute mean frequencies for spiral plots ────────────────────────────
    print('Computing mean frequencies...')
    mean_freqs_list = []
    H = None
    for batch in tqdm(loader, desc='mean freq'):
        spec = batch[0].squeeze(1).float()   # (B, H, W)
        if H is None:
            H = spec.shape[1]
            freq_bins = torch.arange(H, dtype=torch.float32)
        freq_profile = spec.sum(dim=2)
        total = freq_profile.sum(dim=1).clamp(min=1e-10)
        mean_freqs_list.append(((freq_profile * freq_bins).sum(dim=1) / total).numpy())
    mean_freqs = np.concatenate(mean_freqs_list)

    # ── Build output dictionary ──────────────────────────────────────────────
    # Group by session_id (YYYYMMDD_HHMMSS); each value is a DataFrame
    from collections import defaultdict
    session_rows = defaultdict(list)
    for i, spec_id in enumerate(full_ds.spec_ids):
        if spec_id is not None:
            parts = spec_id.split('_')
            # spec_id format: {session_key}_{orig_hdf5_index}
            # session_key may itself contain underscores (e.g. Female_Female conditions)
            session_id = '_'.join(parts[:-1])
            original_index = int(parts[-1])
        else:
            session_id = '__unknown__'
            original_index = i
        session_rows[session_id].append({
            'original_index': original_index,
            'ws_label': int(sample_ws[i]),
            'ws_label_periodic': int(sample_ws_periodic[i]),
            'x': float(latent_coords[i, 0]),
            'y': float(latent_coords[i, 1]),
        })

    result = {
        sid: pd.DataFrame(rows).sort_values('original_index').reset_index(drop=True)
        for sid, rows in session_rows.items()
    }

    # ── Provenance ───────────────────────────────────────────────────────────
    t_end = datetime.datetime.now()
    cluster_sizes = {i + 1: int((sample_ws == i + 1).sum()) for i in range(len(centers))}
    cluster_sizes_periodic = {i + 1: int((sample_ws_periodic == i + 1).sum())
                              for i in range(len(centers))}
    provenance = {
        'generated_at': t_end.isoformat(),
        'elapsed_seconds': (t_end - t_start).total_seconds(),
        'model_path': os.path.abspath(model_path),
        'dataloc': os.path.abspath(dataloc),
        'n_samples': len(full_ds),
        'n_clusters': len(centers),
        'cluster_centers': centers.tolist(),
        'parameters': {
            'lattice_m': lattice_m,
            'lattice_size': int(len(lattice_np)),
            'n_clusters': n_clusters,
            'res': res,
            'heatmap_sigma': heatmap_sigma,
            'ws_sigma': ws_sigma,
            'ws_compactness': ws_compactness,
            'filter_mask': filter_mask,
            'lo': lo,
            'hi': hi,
            'total_samples': total_samples,
            'c_dim': c_dim,
            'sampling_method': sampling_method,
            'n_per': n_per,
            'max_plot_num': max_plot_num,
        },
        'dataset_stats': {
            'n_filtered': len(full_ds),
            'cluster_sizes_standard': cluster_sizes,
            'cluster_sizes_periodic': cluster_sizes_periodic,
            'periodic_vs_standard_changed': n_changed,
            'periodic_vs_standard_pct': round(pct_changed, 4),
        },
    }

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    print(f'Saved: {output_path}')

    provenance_path = os.path.join(save_dir, f'latents_full_provenance.json')
    with open(provenance_path, 'w') as f:
        json.dump(provenance, f, indent=2)
    print(f'Saved: {os.path.basename(provenance_path)}')

    # ── Numpy arrays for viewer export ───────────────────────────────────────
    arrays_path = os.path.join(save_dir, 'arrays.npz')
    np.savez(
        arrays_path,
        latent_coords=latent_coords.astype(np.float32),        # (N, 2)
        sample_ws=sample_ws.astype(np.int16),                  # (N,)
        sample_ws_periodic=sample_ws_periodic.astype(np.int16),# (N,)
        ws_labels=ws_labels.astype(np.int16),                  # (res, res)
        ws_labels_periodic=ws_labels_periodic.astype(np.int16),# (res, res)
        heatmap=heatmap.astype(np.float32),                    # (res, res)
        jac_norm=jac_norm.astype(np.float32),                  # (res, res)
        jac_smooth=jac_smooth.astype(np.float32),              # (res, res)
        centers=centers.astype(np.float32),                    # (K, 2)
    )
    print(f'Saved: arrays.npz')

    spec_ids_path = os.path.join(save_dir, 'spec_ids.json')
    with open(spec_ids_path, 'w') as f:
        json.dump(list(full_ds.spec_ids), f)
    print(f'Saved: spec_ids.json')

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_watershed(
        latent_coords, jac_smooth, ws_labels, centers,
        save_path=os.path.join(save_dir, 'watershed_check.png'),
        ws_labels_periodic=ws_labels_periodic,
    )

    # Toroidal vs standard comparison (3-panel: tiled view, diff map, bar chart)
    plot_toroidal_comparison(
        latent_coords, heatmap, ws_labels, ws_labels_periodic, centers,
        save_path=os.path.join(save_dir, 'toroidal_vs_standard.png'),
    )

    # Boundary diagnostic: edge samples + label-switched samples
    plot_torus_boundary_diagnostic(
        latent_coords=latent_coords,
        sample_ws=sample_ws,
        sample_ws_periodic=sample_ws_periodic,
        full_ds=full_ds,
        heatmap=heatmap,
        ws_labels=ws_labels,
        ws_labels_periodic=ws_labels_periodic,
        centers=centers,
        save_path=os.path.join(save_dir, 'torus_boundary_diagnostic.png'),
        edge_threshold=edge_threshold,
        n_show=n_boundary_show,
        res=res,
    )

    # Per-cluster spectrogram grids (periodic labels by default)
    plot_clusters_sampling(
        latent_coords=latent_coords,
        sample_ws=sample_ws_periodic,
        centers=centers,
        ws_labels_grid=ws_labels_periodic,
        full_ds=full_ds,
        heatmap=heatmap,
        save_dir=save_dir,
        res=res,
        sampling_method=sampling_method,
        n_per=n_per,
        max_plot_num=max_plot_num,
    )

    print(f'\nDone. {len(result)} labels written, {len(centers)} clusters.')
    print(f'Elapsed: {provenance["elapsed_seconds"]:.1f} s')
    return


if __name__ == '__main__':
    fire.Fire(run_inference)
