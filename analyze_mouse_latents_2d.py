"""
Script to generate latent space visualizations for trained mouse vocalization model.
Reproduces Figure E/F style plots: embedded latents + aggregated posterior with centroids.

Usage:
    python analyze_mouse_latents.py \
        --model_path="path/to/checkpoint.tar" \
        --dataloc="path/to/mouse/data" \
        --save_dir="path/to/save/figures" \
        --lattice_m=20 \
        --bandwidth=0.1
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader
import os
import fire
from tqdm import tqdm

from models.qmc_base import QMCLVM, TorusBasis
from models.sampling import gen_fib_basis
from train.model_saving_loading import load
from train.losses import binary_lp
from torch.optim import Adam
from data.mouse_data import load_mouse_data, mouse_data
from analysis.model_helpers import get_posterior_summaries, torus_forward, torus_reverse
from analysis.clustering import run_mean_shift, run_mean_shift_fast


# ---------------------------------------------------------------------------
# Conditional variable registry  (mirrors bartul_mouse_cond.py)
#
# mouse_data.__getitem__ returns:
#   0: spec            (1 x H x W)
#   1: masks_len       scalar float
#   2: raw duration    scalar float
#   3: norm_duration   scalar float  [0, 1]
#   4: mean_freq       scalar float  [0, 1]
#   5: mask_count_onehot  (8,) float
#   6: mask            (H x W)
#   7: spec_id         str
# ---------------------------------------------------------------------------
CONDITIONAL_REGISTRY = {
    "mask_count": {"field_idx": 5, "c_dim": 8,  "label": "masks_len (one-hot)"},
    "duration":   {"field_idx": 3, "c_dim": 1,  "label": "normalized duration"},
    "mean_freq":  {"field_idx": 4, "c_dim": 1,  "label": "mean frequency"},
}


def _make_c_fn(cond_name, device):
    """Return a callable ``c_fn(batch) -> Tensor (1, c_dim)`` for use with
    ``get_posterior_summaries``.  ``batch`` comes from the default DataLoader
    collate so each element is already a batched tensor."""
    fi = CONDITIONAL_REGISTRY[cond_name]["field_idx"]

    def c_fn(batch):
        vals = batch[fi].float()          # (B,) or (B, c_dim)
        if vals.dim() == 1:
            vals = vals.unsqueeze(1)      # (B, 1)
        return vals.mean(dim=0, keepdim=True).to(device)  # (1, c_dim)

    return c_fn


def _get_sample_c(dataset, index, cond_name, device):
    """Extract the conditioning tensor for a single dataset sample → (1, c_dim)."""
    fi = CONDITIONAL_REGISTRY[cond_name]["field_idx"]
    val = dataset[index][fi].float()
    if val.dim() == 0:
        val = val.unsqueeze(0)    # scalar → (1,)
    return val.unsqueeze(0).to(device)   # (1, c_dim)


def plot_latent_scatter(
    latent_coords,
    color_values,
    save_path,
    label,
    cmap='viridis',
    vmin=None,
    vmax=None,
    color_map=None,
    category_order=None,
    scatter_size=7,
    scatter_alpha=0.4,
):
    """
    Plot 2-D latent scatter with continuous colormap or discrete category colors.

    Continuous mode (color_map is None):
        color_values: numeric array (N,), NaN entries are drawn gray behind.
        Adds a colorbar labeled `label`.

    Discrete mode (color_map is a {category: hex} dict):
        color_values: object array (N,), None/NaN entries drawn gray behind.
        category_order: draw order (default: sorted keys of color_map).
        Adds a legend.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    if color_map is None:
        # --- continuous ---
        vals = np.asarray(color_values, dtype=float)
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            ax.scatter(
                latent_coords[nan_mask, 0], latent_coords[nan_mask, 1],
                c='lightgray', s=scatter_size, alpha=scatter_alpha,
                rasterized=True, linewidth=0, marker='.', edgecolors='none',
            )
        valid = ~nan_mask
        sort_idx = np.argsort(vals[valid])
        sc = ax.scatter(
            latent_coords[valid][sort_idx, 0],
            latent_coords[valid][sort_idx, 1],
            c=vals[valid][sort_idx],
            cmap=cmap, vmin=vmin, vmax=vmax,
            s=scatter_size, alpha=scatter_alpha,
            rasterized=True, linewidth=0, marker='.', edgecolors='none',
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
        cbar.solids.set_alpha(1)
        cbar.set_label(label, fontsize=10)
    else:
        # --- discrete ---
        cats = np.asarray(color_values, dtype=object)
        nan_mask = np.array([c is None or (isinstance(c, float) and np.isnan(c)) for c in cats])
        if nan_mask.any():
            ax.scatter(
                latent_coords[nan_mask, 0], latent_coords[nan_mask, 1],
                c='lightgray', s=scatter_size, alpha=scatter_alpha,
                rasterized=True, linewidth=0, marker='.', edgecolors='none',
            )
        order = category_order if category_order is not None else sorted(color_map)
        for cat in order:
            mask = cats == cat
            if not mask.any():
                continue
            ax.scatter(
                latent_coords[mask, 0], latent_coords[mask, 1],
                c=color_map[cat], label=cat,
                s=scatter_size, alpha=scatter_alpha,
                rasterized=True, linewidth=0, marker='.', edgecolors='none',
            )
        ax.legend(markerscale=4, fontsize=9, loc='best')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Latent dimension 1', fontsize=12)
    ax.set_ylabel('Latent dimension 2', fontsize=12)
    ax.set_title('Embedded latents', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {os.path.basename(save_path)}')
    plt.close()


def compute_mean_frequency(spectrogram, freq_bins=None, freq_range_khz=None):
    """
    Compute mean frequency of a spectrogram.

    Args:
        spectrogram: (H, W) array, where H is frequency bins, W is time
        freq_bins: frequency values for each bin in kHz (optional)
        freq_range_khz: (min_freq, max_freq) in kHz to create linearly spaced bins

    Returns:
        mean_freq: weighted mean frequency in kHz (or bin index if no freq info provided)
    """
    if freq_bins is None:
        if freq_range_khz is not None:
            # Create linearly spaced frequency bins
            freq_bins = np.linspace(freq_range_khz[0], freq_range_khz[1], spectrogram.shape[0])
        else:
            # Use bin indices if actual frequencies not provided
            freq_bins = np.arange(spectrogram.shape[0])

    # Sum over time axis to get frequency profile
    freq_profile = spectrogram.sum(axis=1)

    # Compute weighted mean (center of mass in frequency)
    if freq_profile.sum() > 0:
        mean_freq = (freq_profile * freq_bins).sum() / freq_profile.sum()
    else:
        mean_freq = 0

    return mean_freq


def create_aggregated_posterior_heatmap(lattice, posteriors, resolution=200):
    """
    Create a 2D histogram/heatmap of aggregated posterior probability.

    Args:
        lattice: (n_lattice, 2) lattice points in [0,1]^2
        posteriors: (n_samples, n_lattice) posterior probabilities
        resolution: grid resolution for heatmap

    Returns:
        heatmap: (resolution, resolution) aggregated posterior density
        extent: (x_min, x_max, y_min, y_max) for imshow
    """
    # Sum posteriors across all samples
    aggregated = posteriors.sum(axis=0)  # (n_lattice,)

    # Create 2D histogram weighted by posterior
    x_edges = np.linspace(0, 1, resolution + 1)
    y_edges = np.linspace(0, 1, resolution + 1)

    heatmap, _, _ = np.histogram2d(
        lattice[:, 0],
        lattice[:, 1],
        bins=[x_edges, y_edges],
        weights=aggregated
    )

    return heatmap.T, (0, 1, 0, 1)  # Transpose for correct orientation


def _torus_segments(p1, p2):
    """
    Split geodesic p1->p2 on the [0,1]^2 torus into drawable pieces.
    Returns a list of (a, b) pairs, each in [0,1]^2.
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    delta = p2 - p1
    delta -= np.round(delta)  # shortest-path direction on torus

    crossings = set()
    for dim in range(2):
        d = delta[dim]
        if d > 1e-9:  # moving toward upper boundary (1)
            t = (1.0 - p1[dim]) / d
            if 1e-9 < t < 1 - 1e-9:
                crossings.add(t)
        elif d < -1e-9:  # moving toward lower boundary (0)
            t = (0.0 - p1[dim]) / d
            if 1e-9 < t < 1 - 1e-9:
                crossings.add(t)

    ts = [0.0] + sorted(crossings) + [1.0]
    result = []
    for i in range(len(ts) - 1):
        q0 = p1 + ts[i] * delta
        q1 = p1 + ts[i + 1] * delta
        offset = np.floor((q0 + q1) / 2)  # which torus cell the midpoint falls in
        result.append((q0 - offset, q1 - offset))
    return result


def plot_continuous_segments_overlay(
    latent_coords,
    base_color_values,
    segments_coords,
    segments_dist,
    save_path,
    title,
    direction='decreasing',
    cmap_name='RdYlGn',
    max_lines=10,
    scatter_size=5,
    scatter_alpha=0.2,
    vmin=0,
    vmax=85,
):
    """
    Plot duration-colored scatter with continuous segment trajectories overlaid.

    Args:
        latent_coords: (N, 2) array of all latent positions
        base_color_values: (N,) durations for background scatter coloring
        segments_coords: list of (N_i, 2) arrays, one per segment
        segments_dist: list of (N_i,) social distance arrays, one per segment
        save_path: output file path
        title: plot title
        direction: 'decreasing' or 'increasing' — controls annotation text
        cmap_name: colormap for segment lines (RdYlGn: green=close, red=far)
        max_lines: cap on number of segments drawn
        vmin/vmax: social distance colormap range
    """
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(6, 6))

    # Background scatter colored by duration
    vals = np.asarray(base_color_values, dtype=float)
    ax.scatter(
        latent_coords[:, 0], latent_coords[:, 1],
        c=vals, cmap='magma',
        s=scatter_size, alpha=scatter_alpha,
        rasterized=True, linewidth=0, marker='.', edgecolors='none',
    )

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for coords, dists in zip(segments_coords[:max_lines], segments_dist[:max_lines]):
        all_segs, all_colors = [], []
        for k in range(len(coords) - 1):
            avg_color = (dists[k] + dists[k + 1]) / 2
            for a, b in _torus_segments(coords[k], coords[k + 1]):
                all_segs.append([a, b])
                all_colors.append(avg_color)
        if not all_segs:
            continue
        lc = LineCollection(
            all_segs, cmap=cmap_name, norm=norm,
            linewidth=1, alpha=0.6, zorder=5,
        )
        lc.set_array(np.array(all_colors))
        ax.add_collection(lc)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Latent dimension 1', fontsize=12)
    ax.set_ylabel('Latent dimension 2', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_aspect('equal')

    sm = mpl.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    # cbar.solids.set_alpha(1)
    # cbar.set_label('Social distance (cm)', fontsize=10)

    if direction == 'decreasing':
        color_note = 'red = far apart  \u2192  green = close'
    else:
        color_note = 'green = close  \u2192  red = far apart'
    ax.text(
        0.5, -0.10, color_note,
        transform=ax.transAxes, ha='center', va='top',
        fontsize=9, color='dimgray',
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {os.path.basename(save_path)}')
    plt.close()


def make_segment_video(
    coords,
    dists,
    dataset_indices,
    times,
    session_id,
    full_ds,
    latent_coords_bg,
    durations_bg,
    save_path,
    vmin=0,
    vmax=85,
    scatter_size=5,
    scatter_alpha=0.2,
    fps=5,
    tail_coords=None,
    tail_dists=None,
    tail_dataset_indices=None,
    tail_times=None,
    frame_rate=150,
):
    """
    Animate a single continuous segment on the torus latent space alongside spectrograms.

    Left panel : background scatter + torus-aware trail growing frame by frame.
    Right panel: grid of all N spectrograms in the segment (+ optional tail);
                 current frame is highlighted with a distance-colored border.
    Labels     : session datetime, current USV start/end time (converted to seconds).
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation
    from datetime import datetime

    TRAIL_CMAP = 'winter'

    dt = datetime.strptime(session_id, '%Y%m%d_%H%M%S')
    dt_label = dt.strftime('%Y-%m-%d %H:%M:%S')
    seg_start = times[0][0] / frame_rate
    seg_end = times[-1][1] / frame_rate
    n_seg = len(coords)

    # ── Concatenate tail if provided ────────────────────────────────────────
    if tail_coords is not None and len(tail_coords) > 0:
        all_coords = np.concatenate([coords, tail_coords], axis=0)
        all_dists = np.concatenate([dists, tail_dists])
        all_didx = list(dataset_indices) + list(tail_dataset_indices)
        all_times = list(times) + list(tail_times)
    else:
        all_coords = coords
        all_dists = dists
        all_didx = list(dataset_indices)
        all_times = list(times)
    N = len(all_coords)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(TRAIL_CMAP)

    # ── Grid layout for spectrograms ────────────────────────────────────────
    n_cols = min(8, int(np.ceil(np.sqrt(N))))
    n_rows = int(np.ceil(N / n_cols))

    fig_w = 6 + 1.6 * n_cols
    fig_h = max(5, 1.5 * n_rows + 0.6)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f'{dt_label}  |  segment: {seg_start:.2f} s \u2013 {seg_end:.2f} s',
        fontsize=11,
    )

    outer_gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1],
                                  left=0.05, right=0.97, top=0.91, bottom=0.07,
                                  wspace=0.12)
    ax_lat = fig.add_subplot(outer_gs[0, 0])

    right_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=outer_gs[0, 1],
        hspace=0.08, wspace=0.06,
    )
    ax_specs = []
    for r in range(n_rows):
        for c in range(n_cols):
            k = r * n_cols + c
            ax = fig.add_subplot(right_gs[r, c])
            if k < N:
                ax_specs.append(ax)
            else:
                ax.set_visible(False)

    # ── Left panel: static background ──────────────────────────────────────
    bg_vals = np.asarray(durations_bg, dtype=float)
    ax_lat.scatter(
        latent_coords_bg[:, 0], latent_coords_bg[:, 1],
        c=bg_vals, cmap='magma',
        s=scatter_size, alpha=scatter_alpha,
        rasterized=True, linewidth=0, marker='.', edgecolors='none',
    )
    ax_lat.set_xlim(0, 1)
    ax_lat.set_ylim(0, 1)
    ax_lat.set_aspect('equal')
    ax_lat.set_xlabel('Latent dim 1', fontsize=10)
    ax_lat.set_ylabel('Latent dim 2', fontsize=10)

    trail_artists = []
    dot, = ax_lat.plot([], [], 'wo', markersize=6, markeredgecolor='k',
                       markeredgewidth=0.8, zorder=10)
    time_text = ax_lat.text(
        0.02, 0.97, '', transform=ax_lat.transAxes,
        va='top', ha='left', fontsize=8, color='white',
        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.4),
    )

    # ── Right panel: pre-load all spectrograms into the grid ───────────────
    spec_ims = []
    spec_overlays = []
    for k, ax_s in enumerate(ax_specs):
        spec = full_ds[all_didx[k]][0].squeeze().numpy()
        im = ax_s.imshow(
            spec, aspect='auto', origin='lower',
            cmap='magma', interpolation='nearest',
            vmin=0, vmax=1,
        )
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        for spine in ax_s.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor('#555555')
        spec_ims.append(im)

        overlay = ax_s.add_patch(
            mpl.patches.Rectangle(
                (0, 0), 1, 1,
                transform=ax_s.transAxes,
                color='black', alpha=0.55,
                zorder=5,
            )
        )
        spec_overlays.append(overlay)

    # ── Title artists for each grid cell (pre-created, updated each frame) ─
    spec_titles = []
    for ax_s in ax_specs:
        t = ax_s.set_title('', fontsize=9, pad=1)
        spec_titles.append(t)

    def _highlight(k):
        border_rgba = cmap_obj(norm(all_dists[k]))
        border_color = border_rgba[:3]
        for j, (ax_s, ov, tt) in enumerate(zip(ax_specs, spec_overlays, spec_titles)):
            if j == k:
                for spine in ax_s.spines.values():
                    spine.set_linewidth(3.0)
                    spine.set_edgecolor(border_color)
                tt.set_text(f'{all_dists[k]:.1f} cm')
                tt.set_color(border_color)
                ov.set_visible(False)
            else:
                for spine in ax_s.spines.values():
                    spine.set_linewidth(0.8)
                    spine.set_edgecolor('#444444')
                tt.set_text('')
                ov.set_visible(True)

    def init():
        dot.set_data([], [])
        time_text.set_text('')
        _highlight(0)
        return [dot, time_text] + spec_ims + spec_overlays + spec_titles

    def update(k):
        for art in trail_artists:
            art.remove()
        trail_artists.clear()

        if k > 0:
            sc = ax_lat.scatter(
                all_coords[:k, 0], all_coords[:k, 1],
                c=all_dists[:k], cmap=TRAIL_CMAP, norm=norm,
                s=50, alpha=0.9, linewidths=0, zorder=5,
            )
            trail_artists.append(sc)

        dot.set_data([all_coords[k, 0]], [all_coords[k, 1]])
        st, et = all_times[k]
        time_text.set_text(f'{st / frame_rate:.2f} s \u2013 {et / frame_rate:.2f} s')

        _highlight(k)

        return [dot, time_text] + spec_ims + spec_overlays + spec_titles + trail_artists

    ani = FuncAnimation(
        fig, update, frames=N,
        init_func=init, blit=False, interval=int(1000 / fps),
    )
    ani.save(save_path, writer='ffmpeg', fps=fps, dpi=150,
             extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    print(f'Saved: {os.path.basename(save_path)}')
    plt.close()


def grid_examples(model, grid_size, save_path, device, c=None):
    """Decode a grid_size x grid_size grid of latent points in [0,1]^2 and plot reconstructions.

    Args:
        c: optional conditioning tensor of shape (1, c_dim).  When provided it is
           tiled to match the grid batch and concatenated to the basis output before
           the decoder — matching the conditional model's forward pass.
    """
    xs = np.linspace(0, 1, grid_size, endpoint=False) + 0.5 / grid_size
    gx, gy = np.meshgrid(xs, xs)
    z = np.stack([gx.ravel(), gy.ravel()], axis=1)
    z_t = torch.tensor(z, dtype=torch.float32, device=device)
    with torch.no_grad():
        basis_out = model.basis(z_t)                     # (grid^2, 2*latent_dim)
        if c is not None:
            c_exp = c.to(device).expand(len(z_t), -1)   # (grid^2, c_dim)
            decoder_input = torch.cat([basis_out, c_exp], dim=-1)
        else:
            decoder_input = basis_out
        recon = model.decoder(decoder_input).cpu().numpy().squeeze(1)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            ax = axes[grid_size - 1 - i, j]  # flip y so row 0 is bottom
            ax.imshow(recon[i * grid_size + j], cmap='viridis', origin='lower', aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f'Grid reconstructions ({grid_size}x{grid_size})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {os.path.basename(save_path)}')
    plt.close()


def figure_H_watershed_variants(heatmap, centers, latent_coords, mean_freqs,
                                 scatter_size, scatter_alpha, save_path):
    """5x5 grid of watershed variants over (sigma, compactness).

    Background: the smoothed heatmap used by watershed in each cell, so
    boundaries can be judged against the density the algorithm actually sees.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.segmentation import watershed

    sigmas   = [1, 3, 6, 12, 20]
    compacts = [0, 0.05, 0.2, 0.5, 2.0]

    res = heatmap.shape[0]
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)

    marker_img_base = np.zeros((res, res), dtype=int)
    for i, (cx, cy) in enumerate(centers):
        px = int(np.clip(cx * res, 0, res - 1))
        py = int(np.clip(cy * res, 0, res - 1))
        marker_img_base[py, px] = i + 1

    fig, axes = plt.subplots(len(sigmas), len(compacts),
                             figsize=(len(compacts) * 4, len(sigmas) * 4))
    for row, sigma in enumerate(sigmas):
        smoothed = gaussian_filter(heatmap, sigma=sigma)
        nz = smoothed[smoothed > 0]
        vmax = float(np.percentile(nz, 95)) if nz.size else None
        for col, compact in enumerate(compacts):
            ax = axes[row, col]
            labels_ws = watershed(-smoothed, markers=marker_img_base.copy(), compactness=compact)
            # Show the smoothed heatmap watershed actually operates on — lets
            # you judge whether boundaries fall at density valleys.
            ax.imshow(smoothed, origin='lower', extent=(0, 1, 0, 1),
                      cmap='viridis', vmin=0, vmax=vmax, aspect='equal',
                      interpolation='nearest')
            boundary_levels = np.arange(0.5, len(centers) + 1.5)
            ax.contour(xx, yy, labels_ws, levels=boundary_levels,
                       colors='white', linewidths=1.5)
            ax.scatter(centers[:, 0], centers[:, 1],
                       c='white', s=80, marker='x', linewidths=2.0, zorder=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f'compact={compact}', fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'σ={sigma}', fontsize=10, fontweight='bold')
    plt.suptitle('Watershed segmentation grid  (rows=σ, cols=compactness)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def figure_watershed_overlay(
    latent_coords, ws_labels, centers,
    heatmap_right,
    durations,
    scatter_size, scatter_alpha,
    save_path,
):
    """
    Two-panel figure with watershed boundaries overlaid on:
      left  — scatter colored by duration
      right — aggregated posterior heatmap (slightly less smoothed than FG left)
    """
    res = ws_labels.shape[0]
    n_clusters = len(centers)
    boundary_levels = np.arange(0.5, n_clusters + 1.5)
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    extent = (0, 1, 0, 1)

    def _add_watershed(ax):
        ax.contour(xx, yy, ws_labels, levels=boundary_levels,
                   colors='white', linewidths=1.5, zorder=5)
        for i, (cx, cy) in enumerate(centers):
            ax.text(cx, cy, str(i + 1), color='white', fontsize=9,
                    fontweight='bold', ha='center', va='center', zorder=6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('Latent dim 1', fontsize=11)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- Left: scatter by duration ---
    ax = axes[0]
    dur = np.asarray(durations, dtype=float)
    nan_mask = np.isnan(dur)
    if nan_mask.any():
        ax.scatter(latent_coords[nan_mask, 0], latent_coords[nan_mask, 1],
                   c='lightgray', s=scatter_size, alpha=scatter_alpha,
                   rasterized=True, linewidth=0, marker='.', edgecolors='none')
    valid = ~nan_mask
    sort_idx = np.argsort(dur[valid])
    sc = ax.scatter(
        latent_coords[valid][sort_idx, 0], latent_coords[valid][sort_idx, 1],
        c=dur[valid][sort_idx], cmap='magma',
        s=scatter_size, alpha=scatter_alpha,
        rasterized=True, linewidth=0, marker='.', edgecolors='none',
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.solids.set_alpha(1)
    cbar.set_label('Duration (samples)', fontsize=10)
    ax.set_ylabel('Latent dim 2', fontsize=11)
    ax.set_title('Duration', fontsize=13, fontweight='bold')
    _add_watershed(ax)

    # --- Right: aggregated posterior heatmap (less smoothed, same style as FG left) ---
    ax = axes[1]
    nz = heatmap_right[heatmap_right > 0]
    agg_vmax = float(np.percentile(nz, 95)) if nz.size else None
    im2 = ax.imshow(
        heatmap_right, origin='lower', extent=extent, cmap='viridis',
        interpolation=None, vmin=0, vmax=agg_vmax, aspect='equal',
    )
    cbar = plt.colorbar(im2, ax=ax, shrink=0.6)
    cbar.solids.set_alpha(1)
    cbar.set_label('Aggregated posterior', fontsize=10)
    ax.set_title('Aggregated posterior', fontsize=13, fontweight='bold')
    _add_watershed(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {os.path.basename(save_path)}')
    plt.close()


def plot_recon_bars(mse_arr, mask_counts, durations, save_dir, n_dur_bins=5):
    """Bar chart of mean reconstruction MSE grouped by mask count and duration bin."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- by mask count ---
    ax = axes[0]
    unique_mls = np.sort(np.unique(mask_counts))
    means = [mse_arr[mask_counts == ml].mean() for ml in unique_mls]
    sems  = [mse_arr[mask_counts == ml].std() / max(1, np.sqrt((mask_counts == ml).sum()))
             for ml in unique_mls]
    ax.bar(np.arange(len(unique_mls)), means, yerr=sems, capsize=3,
           color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xticks(np.arange(len(unique_mls)))
    ax.set_xticklabels([str(int(m)) for m in unique_mls])
    ax.set_xlabel('masks_len (syllable length bins)', fontsize=11)
    ax.set_ylabel('Mean reconstruction MSE', fontsize=11)
    ax.set_title('Reconstruction MSE by mask count', fontsize=12, fontweight='bold')

    # --- by duration bin ---
    ax = axes[1]
    bin_edges = np.percentile(durations, np.linspace(0, 100, n_dur_bins + 1))
    bin_edges[0]  -= 1
    bin_edges[-1] += 1
    bin_labels, bin_means, bin_sems = [], [], []
    for b in range(n_dur_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (durations >= lo) & (durations < hi)
        if mask.sum() == 0:
            continue
        bin_labels.append(f'[{lo:.0f},{hi:.0f})')
        bin_means.append(mse_arr[mask].mean())
        bin_sems.append(mse_arr[mask].std() / max(1, np.sqrt(mask.sum())))
    ax.bar(np.arange(len(bin_means)), bin_means, yerr=bin_sems, capsize=3,
           color='darkorange', alpha=0.8, edgecolor='white')
    ax.set_xticks(np.arange(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=30, ha='right', fontsize=9)
    ax.set_xlabel('Duration (samples)', fontsize=11)
    ax.set_ylabel('Mean reconstruction MSE', fontsize=11)
    ax.set_title('Reconstruction MSE by duration', fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'figure_recon_mse_bars.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: figure_recon_mse_bars.png')
    plt.close()


def analyze_mouse_latents(
    model_path,
    dataloc,
    save_dir,
    lattice_m=20,
    bandwidth=0.1,
    batch_size=1,
    freq_range_khz=(20, 120),
    scatter_size=5,
    scatter_alpha=0.2,
    total_samples=None,
    beh_features_path="/jukebox/falkner/Dexter/vocal_beh/data/full_dataset/utils/usv_beh_features.pkl",
    consec_threshold=150,
    min_seg_len=20,
    grid_size=15,
    n_per_cluster=16,
    sample_from_centroid=False,
    cache_posteriors=False,
    # reconstruction diagnostics
    compute_recon=True,
    recon_batch_size=64,
    n_dur_bins=5,
    # mean-shift algorithm selection
    use_fast_mean_shift=False,
    ms_k_neighbors=8,
    ms_seed_percentile=50,
    ms_max_iter=300,
    ms_tol=1e-5,
    # conditional model support
    conditional=None,   # one of: "mask_count", "duration", "mean_freq", or None
    # data filtering
    filter_mask=True,
    lo=1,
    hi=8,
):
    """
    Generate latent space analysis figures for mouse vocalization model.

    Args:
        model_path: Path to trained model checkpoint (.tar file)
        dataloc: Path to mouse data directory
        save_dir: Directory to save output figures
        lattice_m: Fibonacci lattice parameter (m=20 gives ~10K points)
        bandwidth: Bandwidth for mean-shift clustering
        batch_size: Batch size for data loading
        freq_range_khz: (min, max) frequency range in kHz, e.g., (20, 120) for mouse USVs
        scatter_size: Point size for scatter plot (default 3)
        scatter_alpha: Transparency for scatter plot (default 0.3)
        beh_features_path: Path to pkl with {session_id -> DataFrame(avg_social_distance, emitter_sex)}
        cache_posteriors: If True, save/load posterior summaries to/from save_dir/posterior_cache.npz
        compute_recon: If True, compute round-trip MSE and save bar charts
        recon_batch_size: Batch size for reconstruction MSE computation
        n_dur_bins: Number of duration quantile bins for the bar chart
        use_fast_mean_shift: If True, use run_mean_shift_fast (lattice-based) instead of run_mean_shift
        ms_k_neighbors: k-neighbor size for local-max seed detection (fast mean-shift only)
        ms_seed_percentile: Weight percentile threshold for seed selection; 0=all local maxima,
            50=above-median local maxima. Higher = fewer seeds, faster, may miss shallow modes.
        ms_max_iter: Max iterations for fast mean-shift
        ms_tol: Convergence threshold relative to bandwidth (fast mean-shift only)
        conditional: name of conditioning variable from CONDITIONAL_REGISTRY, or None for
            unconditional models. When set the decoder's first linear layer is widened
            by c_dim and conditioning tensors are extracted per-batch from the dataset.
            Choices: "mask_count" (8-D one-hot), "duration" (scalar), "mean_freq" (scalar).
        filter_mask: If True, filter dataset to syllables whose masks_len is in [lo, hi].
        lo: Lower bound (inclusive) on masks_len when filter_mask=True.
        hi: Upper bound (inclusive) on masks_len when filter_mask=True.
    """

    # Setup
    os.makedirs(save_dir, exist_ok=True)

    if conditional is not None and conditional not in CONDITIONAL_REGISTRY:
        raise ValueError(
            f"Unknown conditional: {conditional!r}. "
            f"Choose from: {list(CONDITIONAL_REGISTRY)} or None."
        )
    c_dim = CONDITIONAL_REGISTRY[conditional]["c_dim"] if conditional is not None else 0
    if conditional is not None:
        print(f"Conditional mode: {conditional!r}  ({CONDITIONAL_REGISTRY[conditional]['label']},  c_dim={c_dim})")

    # Load behavioral features (session_id -> DataFrame)
    import pickle
    beh_features = {}
    if beh_features_path and os.path.exists(beh_features_path):
        with open(beh_features_path, "rb") as f:
            beh_features = pickle.load(f)
        print(f"Loaded behavioral features for {len(beh_features)} sessions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading mouse data...")
    # train_dict, val_dict = load_mouse_data(dataloc)
    # ## Using training data for better coverage
    # test_ds = mouse_data(train_dict, masks_len_range=(1, 8), equal_sampling=True, max_samples=max_samples)
    
    full_dict = torch.load(os.path.join(dataloc, 'full_data.pt'), mmap=True)
    full_ds = mouse_data(full_dict, filter_mask=filter_mask, lo=lo, hi=hi,
                         sampling_strategy='subsample', total_samples=total_samples)
    
    n_workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4
    test_loader = DataLoader(full_ds, num_workers=n_workers, shuffle=False, batch_size=batch_size)

    # Reconstruct model architecture (must match training)
    print("Loading model...")
    latent_dim = 2

    # Architecture must match training (bartul_mouse.py or bartul_mouse_cond.py).
    # When a conditional is used, the first linear layer is widened by c_dim.
    import torch.nn as nn
    decoder = nn.Sequential(
        nn.Linear(2*latent_dim + c_dim, 2048),
        nn.Linear(2048, 64*8*8),
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

    model, optimizer, run_info = load(model, optimizer, model_path)
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")

    # Generate lattice
    print(f"Generating Fibonacci lattice with m={lattice_m}...")
    lattice = gen_fib_basis(m=lattice_m)
    print(f"Lattice size: {len(lattice)} points")

    # Compute posteriors for all test samples (streaming — never materializes full matrix)
    lattice_np = lattice.numpy()
    cache_path = os.path.join(save_dir, 'posterior_cache.npz')
    if cache_posteriors and os.path.exists(cache_path):
        print(f"Loading cached posteriors from {cache_path}...")
        cache = np.load(cache_path)
        torus_weighted, aggregated, weights = cache['torus_weighted'], cache['aggregated'], cache['weights']
    else:
        print("Computing posteriors...")
        c_fn = _make_c_fn(conditional, device) if conditional is not None else None
        if conditional is not None and batch_size > 1:
            print(f"  Warning: batch_size={batch_size} with conditional model — posterior uses "
                  f"batch-mean c, not per-sample c. Set batch_size=1 for exact per-sample conditioning.")
        torus_weighted, aggregated, weights = get_posterior_summaries(
            model, lattice, test_loader, binary_lp, c_fn=c_fn
        )
        if cache_posteriors:
            np.savez(cache_path, torus_weighted=torus_weighted, aggregated=aggregated, weights=weights)
            print(f"Cached posteriors to {cache_path}")

    # Posterior mean embedding (torus-aware, continuous)
    latent_coords = torus_reverse(torus_weighted, dim=2)      # (n_samples, 2)
    
    # DEBUG: Check lattice and coordinates
    print(f"\n=== DEBUGGING LATENT COORDINATES ===")
    print(f"Lattice shape: {lattice.shape}")
    print(f"Lattice range: X=[{lattice[:, 0].min():.3f}, {lattice[:, 0].max():.3f}], Y=[{lattice[:, 1].min():.3f}, {lattice[:, 1].max():.3f}]")
    print(f"Number of samples: {len(latent_coords)}")
    print(f"Latent coords range BEFORE mod: X=[{latent_coords[:, 0].min():.3f}, {latent_coords[:, 0].max():.3f}], Y=[{latent_coords[:, 1].min():.3f}, {latent_coords[:, 1].max():.3f}]")
    print(f"Unique points: {len(np.unique(latent_coords, axis=0))}")

    # Force coordinates into [0, 1] range (they should already be there, but just in case)
    latent_coords = latent_coords % 1.0

    print(f"Latent coords range AFTER mod: X=[{latent_coords[:, 0].min():.3f}, {latent_coords[:, 0].max():.3f}], Y=[{latent_coords[:, 1].min():.3f}, {latent_coords[:, 1].max():.3f}]")
    print(f"===================================\n")

    # Compute mean frequency, mask counts, and durations in a batched pass
    print("Computing mean frequencies and extracting metadata...")
    mean_freqs_list = []
    mask_counts_list = []
    durations_list = []

    H = None  # infer freq bin count from first batch
    for batch in tqdm(test_loader, desc="metadata"):
        # batch: (spec, ml, duration, mask, spec_id)
        spec_batch = batch[0]                          # (B, 1, H, W)
        spec = spec_batch.squeeze(1).float()           # (B, H, W)
        if H is None:
            H = spec.shape[1]
            freq_bins = torch.arange(H, dtype=torch.float32)

        freq_profile = spec.sum(dim=2)                 # (B, H) — sum over time
        total = freq_profile.sum(dim=1).clamp(min=1e-10)
        mean_freqs_list.append(
            ((freq_profile * freq_bins).sum(dim=1) / total).numpy()
        )

        mask_counts_list.append(batch[1].numpy())  # masks_len = SAM mask count (time bins)

        durations_list.append(batch[2].numpy())

    mean_freqs = np.concatenate(mean_freqs_list)
    mask_counts = np.concatenate(mask_counts_list)
    durations   = np.concatenate(durations_list)

    # ============= Reconstruction MSE bar charts =============
    if compute_recon:
        print("\nComputing reconstruction MSE...")
        from train.losses import binary_lp as _lp_fnc
        all_mse = []
        model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(full_ds), recon_batch_size), desc="recon MSE"):
                end = min(start + recon_batch_size, len(full_ds))
                items = [full_ds[i] for i in range(start, end)]
                specs = torch.stack([it[0] for it in items]).to(torch.float32).to(device)
                if conditional is not None:
                    c_vals = torch.stack([
                        _get_sample_c(full_ds, idx, conditional, device).squeeze(0)
                        for idx in range(start, end)
                    ])                                          # (B, c_dim)
                    c_batch = c_vals.mean(dim=0, keepdim=True) # (1, c_dim)
                    recon = model.round_trip(lattice.to(device), specs, _lp_fnc, c=c_batch)
                else:
                    recon = model.round_trip(lattice.to(device), specs, _lp_fnc)
                mse = ((recon.cpu() - specs.cpu()) ** 2).mean(dim=(1, 2, 3)).numpy()
                all_mse.extend(mse.tolist())
        model.eval()  # keep in eval for subsequent figure generation
        mse_arr = np.array(all_mse, dtype=np.float32)
        print(f"  Mean MSE: {mse_arr.mean():.4f}  (std {mse_arr.std():.4f})")
        plot_recon_bars(mse_arr, mask_counts, durations, save_dir, n_dur_bins=n_dur_bins)

    print(f"Number of samples: {len(latent_coords)}")
    print(f"Latent coords range: X=[{latent_coords[:, 0].min():.3f}, {latent_coords[:, 0].max():.3f}], Y=[{latent_coords[:,1].min():.3f}, {latent_coords[:, 1].max():.3f}]")
    print(f"Unique points: {len(np.unique(latent_coords, axis=0))}")

    # ============= FIGURE E1: Embedded latents colored by mean frequency =============
    print("\nGenerating Figure E1: Embedded latents (colored by mean frequency)...")
    freq_unit = 'kHz' if freq_range_khz is not None else 'bins'
    plot_latent_scatter(
        latent_coords, mean_freqs,
        save_path=os.path.join(save_dir, 'figure_E_embedded_latents_by_freq.png'),
        label=f'Mean frequency ({freq_unit})',
        cmap='viridis',
        scatter_size=scatter_size, scatter_alpha=scatter_alpha,
    )

    # ============= FIGURE E2: Embedded latents colored by mask count =============
    print("\nGenerating Figure E2: Embedded latents (colored by mask count)...")
    plot_latent_scatter(
        latent_coords, mask_counts,
        save_path=os.path.join(save_dir, 'figure_E_embedded_latents_by_mask_count.png'),
        label='SAM mask count (time bins)',
        cmap='plasma',
        scatter_size=scatter_size, scatter_alpha=scatter_alpha,
    )

    # ============= FIGURE E3: Embedded latents colored by condition =============
    print("\nGenerating Figure E3: Embedded latents (colored by condition)...")

    lone_male_ids = {
        "20250912_155546",
        "20250912_170514",
        "20250919_145712",
        "20250921_155753",
        "20250927_135343",
    }

    # Parse spec_ids from dataset (already subsampled/indexed identically to latent_coords)
    raw_spec_ids = full_ds.spec_ids  # list of str, same length as latent_coords

    cond_labels = []
    for sid in raw_spec_ids:
        if sid is None:
            cond_labels.append("Unknown")
            continue
        parts = sid.split("_")
        # Format: YYYYMMDD_HHMMSS_cond_avg_idx
        datetime_str = f"{parts[0]}_{parts[1]}"
        cond = parts[2] if len(parts) > 2 else "Unknown"
        if cond == "ephys":
            cond = "Lone-Male" if datetime_str in lone_male_ids else "Male-Female"
        cond_labels.append(cond)

    cond_labels = np.array(cond_labels)

    cond_order = ["Female-Female", "Male-Female", "Lone-Male"]
    cond_colors = {"Female-Female": "#e377c2", "Male-Female": "#1f77b4", "Lone-Male": "#ff7f0e"}

    plot_latent_scatter(
        latent_coords, cond_labels,
        save_path=os.path.join(save_dir, "figure_E_embedded_by_cond.jpg"),
        label=None,
        color_map=cond_colors, category_order=cond_order,
        scatter_size=scatter_size, scatter_alpha=scatter_alpha,
    )

    # Print condition counts
    for cond in cond_order:
        print(f"  {cond}: {(cond_labels == cond).sum()} samples")

    # ============= FIGURE E4 & E5: Behavioral features from pkl =============
    # Parse session_id and row_idx from each spec_id: YYYYMMDD_HHMMSS_cond_avg_idx
    social_distances = []
    emitter_sexes = []
    for sid in raw_spec_ids:
        if sid is None:
            social_distances.append(np.nan)
            emitter_sexes.append(None)
            continue
        parts = sid.split("_")
        session_id = f"{parts[0]}_{parts[1]}"
        row_idx = int(parts[-1])
        if session_id not in beh_features:
            social_distances.append(np.nan)
            emitter_sexes.append(None)
        else:
            df = beh_features[session_id]
            if row_idx not in df.index:
                social_distances.append(np.nan)
                emitter_sexes.append(None)
            else:
                row = df.loc[row_idx]
                social_distances.append(row["avg_social_distance"])
                emitter_sexes.append(row["emitter_sex"])

    social_distances = np.array(social_distances, dtype=float)
    emitter_sexes = np.array(emitter_sexes, dtype=object)

    n_with_dist = np.sum(~np.isnan(social_distances))
    n_with_sex = np.sum(emitter_sexes != None)
    print(f"\nBehavioral features: {n_with_dist} samples with social distance, {n_with_sex} with emitter sex")

    # ---- Figure E4: Social distance (continuous) ----
    print("\nGenerating Figure E4: Embedded latents (colored by social distance)...")
    plot_latent_scatter(
        latent_coords, social_distances,
        save_path=os.path.join(save_dir, "figure_E_embedded_by_social_dist.jpg"),
        label="Social distance (cm)",
        cmap="YlOrRd_r", vmin=0, vmax=85,
        scatter_size=scatter_size, scatter_alpha=scatter_alpha,
    )

    # ---- Figure E5: Emitter sex (discrete) ----
    print("\nGenerating Figure E5: Embedded latents (colored by emitter sex)...")
    sex_colors = {"female": "#d44fa8", "male": "#5baed4"}
    sex_order = ["female", "male"]

    plot_latent_scatter(
        latent_coords, emitter_sexes,
        save_path=os.path.join(save_dir, "figure_E_embedded_by_emitter_sex.jpg"),
        label=None,
        color_map=sex_colors, category_order=sex_order,
        scatter_size=scatter_size, scatter_alpha=scatter_alpha,
    )
    nan_mask = np.array([s is None or (isinstance(s, float) and np.isnan(s)) for s in emitter_sexes])
    for sex in sex_order:
        print(f"  {sex}: {(emitter_sexes == sex).sum()} samples")
    print(f"  NaN/missing: {nan_mask.sum()} samples")

    # ============= FIGURE E6: Embedded latents colored by duration =============
    print("\nGenerating Figure E6: Embedded latents (colored by duration)...")
    plot_latent_scatter(
        latent_coords, durations,
        save_path=os.path.join(save_dir, "figure_E_embedded_by_duration.png"),
        label="Duration (samples)",
        cmap="magma",
        scatter_size=scatter_size, scatter_alpha=scatter_alpha,
    )

    # ============= FIGURES E7 & E8: Continuous segment overlays =============
    print("\nGenerating Figures E7 & E8: Continuous segment overlays...")
    import pandas as pd
    from collections import defaultdict

    # Group dataset items by session
    session_to_items = defaultdict(list)
    for i, sid in enumerate(raw_spec_ids):
        if sid is None:
            continue
        parts = sid.split("_")
        session_id = f"{parts[0]}_{parts[1]}"
        row_idx = int(parts[-1])
        session_to_items[session_id].append((i, row_idx))

    all_seg_coords, all_seg_dist, all_seg_trend = [], [], []
    all_seg_didx, all_seg_times, all_seg_session = [], [], []

    for session_id, items in session_to_items.items():
        if session_id not in beh_features:
            continue
        df = beh_features[session_id]
        timed = []
        for dataset_idx, row_idx in items:
            if row_idx not in df.index:
                continue
            row = df.loc[row_idx]
            st = row.get('start_times', np.nan)
            et = row.get('end_times', np.nan)
            sd = row.get('avg_social_distance', np.nan)
            if pd.isna(st) or pd.isna(sd):
                continue
            timed.append((float(st), float(et), float(sd), dataset_idx))
        if len(timed) < 2:
            continue
        timed.sort(key=lambda x: x[0])

        def _split_monotone_chunks(chunk):
            """Split a consecutive chunk at turning points into strictly monotone sub-chunks."""
            if len(chunk) < 2:
                return [chunk] if chunk else []
            d = [chunk[i + 1][2] - chunk[i][2] for i in range(len(chunk) - 1)]
            # Find sign (ignore zeros — treat as continuation)
            sign = [0] * len(d)
            for i, v in enumerate(d):
                if v > 0:
                    sign[i] = 1
                elif v < 0:
                    sign[i] = -1
            # Fill zeros with the previous non-zero sign
            cur_sign = 1
            for i in range(len(sign)):
                if sign[i] != 0:
                    cur_sign = sign[i]
                else:
                    sign[i] = cur_sign
            # Split at sign changes
            sub_chunks = []
            start = 0
            for i in range(1, len(sign)):
                if sign[i] != sign[i - 1]:
                    sub_chunks.append(chunk[start:i + 1])  # overlap at turning point
                    start = i
            sub_chunks.append(chunk[start:])
            return sub_chunks

        def _flush_seg(cur, _sid=session_id):
            for sub in _split_monotone_chunks(cur):
                if len(sub) > min_seg_len:
                    dists = np.array([t[2] for t in sub])
                    trend = 'decreasing' if dists[-1] < dists[0] else 'increasing'
                    all_seg_coords.append(np.array([latent_coords[t[3]] for t in sub]))
                    all_seg_dist.append(dists)
                    all_seg_trend.append(trend)
                    all_seg_didx.append([t[3] for t in sub])
                    all_seg_times.append([(t[0], t[1]) for t in sub])
                    all_seg_session.append(_sid)

        cur = [timed[0]]
        for j in range(1, len(timed)):
            gap = timed[j][0] - cur[-1][1]
            if gap < consec_threshold:
                cur.append(timed[j])
            else:
                _flush_seg(cur)
                cur = [timed[j]]
        _flush_seg(cur)

    print(f"  Found {len(all_seg_coords)} continuous segments (>{min_seg_len} USVs each)")

    def select_top_segments(direction, n=10):
        indices = [i for i, t in enumerate(all_seg_trend) if t == direction]
        # Score by rate of change (cm/s): favors large, time-compact monotone segments
        def _score(i):
            dists = all_seg_dist[i]
            times = all_seg_times[i]
            delta_dist = abs(dists[-1] - dists[0])
            duration_s = times[-1][1] - times[0][0]
            return delta_dist / duration_s if duration_s > 0 else 0.0
        scores = [_score(i) for i in indices]
        top = sorted(zip(scores, indices), reverse=True)[:n]
        sel = [idx for _, idx in top]
        return (
            [all_seg_coords[idx] for idx in sel],
            [all_seg_dist[idx] for idx in sel],
            [all_seg_didx[idx] for idx in sel],
            [all_seg_times[idx] for idx in sel],
            [all_seg_session[idx] for idx in sel],
        )

    dec_coords, dec_dist, dec_didx, dec_times, dec_sessions = select_top_segments('decreasing')
    inc_coords, inc_dist, inc_didx, inc_times, inc_sessions = select_top_segments('increasing')
    print(f"  Decreasing segments: {len(dec_coords)}, Increasing segments: {len(inc_coords)}")

    print("\nGenerating Figure E7: Continuous segments (decreasing social distance)...")
    plot_continuous_segments_overlay(
        latent_coords, durations, dec_coords, dec_dist,
        save_path=os.path.join(save_dir, 'figure_E7_segments_decreasing.jpg'),
        title='Continuous segments — decreasing social distance',
        direction='decreasing',
        cmap_name='RdYlGn', vmin=0, vmax=85,
        scatter_size=scatter_size, scatter_alpha=scatter_alpha,
    )

    print("\nGenerating Figure E7 videos: top-3 decreasing-distance segments...")
    for vi in range(min(2, len(dec_coords))):
        session_id = dec_sessions[vi]

        # ── Build tail: next 20 USVs after segment end (ignore gap constraint) ──
        tail_coords_vi = tail_dists_vi = tail_didx_vi = tail_times_vi = None
        if session_id in beh_features and session_id in session_to_items:
            import pandas as pd
            df = beh_features[session_id]
            seg_last_time = dec_times[vi][-1][1]
            seg_didx_set = set(dec_didx[vi])
            candidates = []
            for di, ri in session_to_items[session_id]:
                if ri not in df.index or di in seg_didx_set:
                    continue
                row = df.loc[ri]
                st = row.get('start_times', np.nan)
                sd = row.get('avg_social_distance', np.nan)
                if pd.isna(st) or pd.isna(sd):
                    continue
                if float(st) > seg_last_time:
                    candidates.append((float(st), float(row.get('end_times', st)), float(sd), di))
            candidates.sort(key=lambda x: x[0])
            tail_items = candidates[:20]
            if tail_items:
                tail_coords_vi = np.array([latent_coords[t[3]] for t in tail_items])
                tail_dists_vi = np.array([t[2] for t in tail_items])
                tail_didx_vi = [t[3] for t in tail_items]
                tail_times_vi = [(t[0], t[1]) for t in tail_items]

        make_segment_video(
            coords=dec_coords[vi],
            dists=dec_dist[vi],
            dataset_indices=dec_didx[vi],
            times=dec_times[vi],
            session_id=session_id,
            full_ds=full_ds,
            latent_coords_bg=latent_coords,
            durations_bg=durations,
            save_path=os.path.join(save_dir, f'figure_E7_video_{vi + 1}_{session_id}.mp4'),
            vmin=0, vmax=85,
            scatter_size=scatter_size, scatter_alpha=scatter_alpha,
            tail_coords=tail_coords_vi,
            tail_dists=tail_dists_vi,
            tail_dataset_indices=tail_didx_vi,
            tail_times=tail_times_vi,
        )

    print("\nGenerating Figure E8: Continuous segments (increasing social distance)...")
    plot_continuous_segments_overlay(
        latent_coords, durations, inc_coords, inc_dist,
        save_path=os.path.join(save_dir, 'figure_E8_segments_increasing.jpg'),
        title='Continuous segments — increasing social distance',
        direction='increasing',
        cmap_name='RdYlGn', vmin=0, vmax=85,
        scatter_size=scatter_size, scatter_alpha=scatter_alpha,
    )

    # ============= FIGURE F: Aggregated posterior with mean-shift centroids =============
    print("\nGenerating Figure F: Aggregated posterior...")

    # Build the aggregated posterior density as a 2D histogram.
    #
    # The description in the paper ("aggregated posterior density as a 2D histogram,
    # Fig. 5B") is the density on the latent torus obtained by *marginalizing* the
    # per-sample posteriors over the dataset. There are two equivalent ways to
    # obtain it, with different numerical trade-offs:
    #
    #   (A) 'samples'  — histogram of per-sample posterior-mean latent coords
    #                    (`latent_coords`, one point per test sample). Dense, robust,
    #                    and independent of lattice resolution. Loses within-sample
    #                    posterior spread (collapses each posterior to its mean).
    #   (B) 'lattice'  — histogram2d of the lattice points weighted by `aggregated`
    #                    = sum_s p(z_j | x_s). This is the exact marginal on the
    #                    lattice, but with a Fibonacci lattice of a few thousand
    #                    points into a 200x200 grid most bins are empty and a few
    #                    lattice-aligned bins dominate the colormap (the bug in the
    #                    previous version). Fixable with a torus-wrap Gaussian blur.
    #   (C) 'kde'      — same weights as (B), but splatted onto the grid via a
    #                    Gaussian kernel so the sparse lattice contribution is
    #                    smoothed out. Effectively (B) + heavy smoothing.
    #
    # We default to (A) because it reproduces the "2D histogram" wording literally
    # and gives a clean dense image; (B)+smoothing is computed alongside for the
    # watershed step below so clusters snap to the true posterior mass and not
    # just to where samples happened to land.
    from scipy.ndimage import gaussian_filter

    res = 200
    edges = np.linspace(0, 1, res + 1)

    # (A) Sample-space histogram — the image we actually plot.
    heatmap_samples, _, _ = np.histogram2d(
        latent_coords[:, 0], latent_coords[:, 1],
        bins=[edges, edges],
    )
    heatmap_samples = heatmap_samples.T  # rows=y, cols=x for imshow

    # (B) Lattice-weighted histogram, smoothed with a torus-wrap Gaussian so the
    # sparse Fibonacci lattice does not produce a near-empty grid. Used for the
    # watershed segmentation further down.
    heatmap_lattice, _, _ = np.histogram2d(
        lattice_np[:, 0], lattice_np[:, 1],
        bins=[edges, edges], weights=aggregated,
    )
    heatmap_lattice = gaussian_filter(heatmap_lattice.T, sigma=2.0, mode='wrap')

    # Light smoothing on the sample histogram too — purely cosmetic, keeps the
    # image readable without hiding real structure.
    heatmap = gaussian_filter(heatmap_samples, sigma=1.5, mode='wrap')
    extent = (0, 1, 0, 1)

    # Run mean-shift clustering
    print("Running mean-shift clustering...")
    embedded = torus_forward(latent_coords)

    if use_fast_mean_shift:
        print("  Using run_mean_shift_fast (lattice-based)...")
        lattice_embedded = torus_forward(lattice_np)   # (N_lattice, 4)
        centers, _lattice_labels, _seed_mask = run_mean_shift_fast(
            lattice_embedded,
            aggregated,
            bandwidth=bandwidth,
            k_neighbors=ms_k_neighbors,
            seed_percentile=ms_seed_percentile,
            max_iter=ms_max_iter,
            tol=ms_tol,
            embedded=True,
        )
        # Assign per-sample labels by nearest center in torus-embedded space
        from scipy.spatial import cKDTree as _cKDTree
        centers_embedded = torus_forward(centers)
        _, labels = _cKDTree(centers_embedded).query(embedded)
    else:
        centers, wms, labels = run_mean_shift(
            embedded,
            seeds=embedded,  # Use all points as seeds
            weights=weights,
            bandwidth=bandwidth,
            n_jobs=min(16, n_workers),
            p=2,
            embedded=True,
            normal=False
        )

    labels = np.atleast_1d(labels)

    # centers are in [0,1]^2 after torus_reverse inside run_mean_shift / run_mean_shift_fast
    print(f"Found {len(centers)} clusters")

    # ============= FIGURE FG: Aggregated posterior + cluster example spectrograms =============
    print("\nGenerating Figure FG: Aggregated posterior + cluster examples...")
    n_clusters_found = len(centers)
    n_cols_g = 5
    n_rows_g = int(np.ceil(n_clusters_found / n_cols_g))

    import matplotlib.gridspec as gridspec
    fig_height = max(5, n_rows_g * 2 + 1) if n_rows_g > 1 else 4
    fig = plt.figure(figsize=(6 + n_cols_g * 2, fig_height))
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.1], wspace=0.15)

    ax = fig.add_subplot(outer[0, 0])
    # Render the aggregated posterior as a proper 2D histogram image. `heatmap`
    # here is the (lightly-smoothed) histogram of per-sample posterior-mean
    # latent coordinates (option A above). Clip vmax to the 99th percentile so a
    # handful of dense bins do not flatten the colormap.
    nz = heatmap[heatmap > 0]
    agg_vmax = float(np.percentile(nz, 95)) if nz.size else None
    im = ax.imshow(
        heatmap, origin='lower', extent=extent, cmap='viridis',
        interpolation=None, vmin=0, vmax=agg_vmax, aspect='equal',
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # ax.scatter(centers[:, 0], centers[:, 1], c='white', s=100, marker='.',
    #            edgecolors='white', linewidths=0, zorder=10)
    for i, (x, y) in enumerate(centers):
        ax.text(x, y, str(i + 1), color='White', fontsize=20, fontweight='bold',
                ha='center', va='center', zorder=11)
    ax.set_xlabel('Latent dimension 1', fontsize=12)
    ax.set_ylabel('Latent dimension 2', fontsize=12)
    ax.set_title('Aggregated posterior', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.solids.set_alpha(1)
    cbar.set_label('Aggregated posterior', fontsize=10)

    right_gs = gridspec.GridSpecFromSubplotSpec(n_rows_g, n_cols_g, subplot_spec=outer[0, 1],
                                                 hspace=0.25, wspace=0.1)
    for cluster_id in range(n_rows_g * n_cols_g):
        r, c = divmod(cluster_id, n_cols_g)
        ax_s = fig.add_subplot(right_gs[r, c])
        if cluster_id < n_clusters_found:
            cluster_samples = np.where(labels == cluster_id)[0]
            if len(cluster_samples) > 0:
                example_idx = cluster_samples[np.random.choice(len(cluster_samples))]
                spec = full_ds[example_idx][0].numpy().squeeze()
                ax_s.imshow(spec, cmap='viridis', origin='lower', aspect='auto')
                ax_s.set_title(f'{cluster_id + 1}', color='red', fontsize=12, fontweight='bold')
            else:
                ax_s.text(0.5, 0.5, 'Empty', ha='center', va='center', transform=ax_s.transAxes)
        else:
            ax_s.axis('off')
        ax_s.set_xticks([]); ax_s.set_yticks([])

    plt.savefig(os.path.join(save_dir, 'figure_FG_posterior_and_examples.png'),
                dpi=300, bbox_inches='tight')
    print("Saved: figure_FG_posterior_and_examples.png")
    plt.close()

    # ============= Grid reconstructions =============
    print("\nGenerating grid reconstructions...")
    if conditional is not None:
        # For conditional models generate one grid per sweep value and also a
        # mean-c grid for a compact single overview.
        from bartul_mouse_cond import get_grid_sweep
        sweep = get_grid_sweep(conditional)
        for val_label, c_val in sweep:
            grid_examples(
                model, grid_size,
                save_path=os.path.join(save_dir, f'figure_grid_examples_{conditional}_{val_label}.png'),
                device=device, c=c_val.to(device),
            )
        # Also produce a mean-conditioning overview grid
        fi = CONDITIONAL_REGISTRY[conditional]["field_idx"]
        mean_c = torch.stack([
            full_ds[i][fi].float() if full_ds[i][fi].dim() > 0 else full_ds[i][fi].float().unsqueeze(0)
            for i in range(min(500, len(full_ds)))
        ]).mean(dim=0, keepdim=True).to(device)   # (1, c_dim)
        grid_examples(model, grid_size,
                      save_path=os.path.join(save_dir, 'figure_grid_examples.png'),
                      device=device, c=mean_c)
    else:
        grid_examples(model, grid_size,
                      save_path=os.path.join(save_dir, 'figure_grid_examples.png'),
                      device=device)

    # ============= FIGURE H: Random samples per watershed cluster (one fig per cluster) =============
    print("\nGenerating Figure H: Random samples per cluster (watershed σ=3, compact=0)...")
    from scipy.ndimage import gaussian_filter
    from skimage.segmentation import watershed

    res = heatmap.shape[0]
    # Watershed on the lattice-weighted posterior mass (already torus-smoothed
    # above); this is the true marginal over the torus rather than the empirical
    # sample histogram, so basins line up with the mean-shift centers.
    smoothed = gaussian_filter(heatmap_lattice, sigma=3, mode='wrap')
    marker_img = np.zeros((res, res), dtype=int)
    for i, (cx, cy) in enumerate(centers):
        px = int(np.clip(cx * res, 0, res - 1))
        py = int(np.clip(cy * res, 0, res - 1))
        marker_img[py, px] = i + 1
    ws_labels = watershed(-smoothed, markers=marker_img, compactness=0)

    sample_px = np.clip((latent_coords[:, 0] * res).astype(int), 0, res - 1)
    sample_py = np.clip((latent_coords[:, 1] * res).astype(int), 0, res - 1)
    sample_cluster = ws_labels[sample_py, sample_px]  # 1..n_clusters_found

    # ============= FIGURE FH: Duration scatter + aggregated posterior with watershed =============
    print("\nGenerating Figure FH: Watershed overlay...")
    # Slightly less smoothing than FG left (sigma=1.5) — for plotting only
    heatmap_fh_right = gaussian_filter(heatmap_samples, sigma=0.8, mode='wrap')
    figure_watershed_overlay(
        latent_coords=latent_coords,
        ws_labels=ws_labels,
        centers=centers,
        heatmap_right=heatmap_fh_right,
        durations=durations,
        scatter_size=scatter_size,
        scatter_alpha=scatter_alpha,
        save_path=os.path.join(save_dir, 'figure_FH_watershed_overlay.png'),
    )

    # ============= FIGURE H watershed grid search =============
    print("\nGenerating Figure H: Watershed grid search (σ × compactness)...")
    figure_H_watershed_variants(
        heatmap_lattice, centers, latent_coords, mean_freqs,
        scatter_size, scatter_alpha,
        save_path=os.path.join(save_dir, 'figure_H_watershed_grid.png'),
    )

    n_clusters_found = len(centers)
    n_cols_h = int(np.ceil(np.sqrt(n_per_cluster)))
    n_rows_h = int(np.ceil(n_per_cluster / n_cols_h))

    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    sort_idx_freq = np.argsort(mean_freqs)
    boundary_levels = np.arange(0.5, n_clusters_found + 1.5)

    import matplotlib.gridspec as gridspec

    for ci in range(n_clusters_found):
        mask = np.where(sample_cluster == ci + 1)[0]

        picks = []
        if len(mask) > 0:
            pts = latent_coords[mask]
            used = set()
            if sample_from_centroid:
                # Archimedean spiral outward from cluster centroid; skip points
                # falling outside this cluster's watershed region.
                cx0, cy0 = centers[ci]
                dists_c = np.sqrt((pts[:, 0] - cx0) ** 2 + (pts[:, 1] - cy0) ** 2)
                r_max = float(dists_c.max()) if len(dists_c) > 0 else 0.1
                n_turns = 4
                n_dense = 4000
                t_dense = np.linspace(0.0, 1.0, n_dense)
                theta_d = 2 * np.pi * n_turns * t_dense
                r_d = r_max * t_dense
                xs_d = cx0 + r_d * np.cos(theta_d)
                ys_d = cy0 + r_d * np.sin(theta_d)
                px_d = np.clip((xs_d * res).astype(int), 0, res - 1)
                py_d = np.clip((ys_d * res).astype(int), 0, res - 1)
                inside = (
                    (xs_d >= 0) & (xs_d <= 1) & (ys_d >= 0) & (ys_d <= 1)
                    & (ws_labels[py_d, px_d] == ci + 1)
                )
                xs_in = xs_d[inside]
                ys_in = ys_d[inside]
                if len(xs_in) >= n_per_cluster:
                    sel = np.linspace(0, len(xs_in) - 1, n_per_cluster).astype(int)
                    xs_in = xs_in[sel]
                    ys_in = ys_in[sel]
                targets = list(zip(xs_in, ys_in))
                for tx_, ty_ in targets:
                    d2 = (pts[:, 0] - tx_) ** 2 + (pts[:, 1] - ty_) ** 2
                    for kk in np.argsort(d2):
                        idx = int(mask[kk])
                        if idx not in used:
                            used.add(idx)
                            picks.append(idx)
                            break
            else:
                # Tile-based sampling: partition cluster bbox into tiles
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                x_max = max(x_max, x_min + 1e-6)
                y_max = max(y_max, y_min + 1e-6)
                tx = np.linspace(x_min, x_max, n_cols_h + 1)
                ty = np.linspace(y_min, y_max, n_rows_h + 1)
                # Iterate top -> bottom, left -> right
                for rr in range(n_rows_h - 1, -1, -1):
                    for cc in range(n_cols_h):
                        if len(picks) >= n_per_cluster:
                            break
                        cx_t = 0.5 * (tx[cc] + tx[cc + 1])
                        cy_t = 0.5 * (ty[rr] + ty[rr + 1])
                        d2 = (pts[:, 0] - cx_t) ** 2 + (pts[:, 1] - cy_t) ** 2
                        for k in np.argsort(d2):
                            idx = int(mask[k])
                            if idx not in used:
                                used.add(idx)
                                picks.append(idx)
                                break
        picks = np.array(picks, dtype=int)

        fig = plt.figure(figsize=(6 + n_cols_h * 1.6, max(6, n_rows_h * 1.6 + 1)))
        outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.05], wspace=0.15)

        # --- Left: watershed segmentation, current cluster highlighted ---
        ax_l = fig.add_subplot(outer[0, 0])
        ax_l.scatter(latent_coords[sort_idx_freq, 0], latent_coords[sort_idx_freq, 1],
                     c=mean_freqs[sort_idx_freq], cmap='magma',
                     s=scatter_size, alpha=scatter_alpha,
                     rasterized=True, linewidth=0, 
                     marker='.', edgecolors='none')
        ax_l.contour(xx, yy, ws_labels, levels=boundary_levels,
                     colors='black', linewidths=1.5)
        ax_l.contour(xx, yy, (ws_labels == ci + 1).astype(int), levels=[0.5],
                     colors='red', linewidths=3.0)
        ax_l.scatter(centers[:, 0], centers[:, 1],
                     c='black', s=60, marker='x', linewidths=1.5, zorder=9)
        ax_l.scatter([centers[ci, 0]], [centers[ci, 1]],
                     c='red', s=120, marker='x', linewidths=2.5, zorder=10)
        if len(picks) > 0:
            for k, pidx in enumerate(picks):
                ax_l.text(latent_coords[pidx, 0], latent_coords[pidx, 1], str(k + 1),
                          color='black', fontsize=7, fontweight='bold',
                          ha='center', va='center', zorder=12)
        ax_l.set_xlim(0, 1); ax_l.set_ylim(0, 1); ax_l.set_aspect('equal')
        ax_l.set_xlabel('Latent dimension 1', fontsize=11)
        ax_l.set_ylabel('Latent dimension 2', fontsize=11)
        ax_l.set_title(f'Cluster {ci + 1} (σ=3, compact=0)',
                       fontsize=12, fontweight='bold', color='red')

        # --- Right: tile-sampled spectrograms ---
        right_gs = gridspec.GridSpecFromSubplotSpec(
            n_rows_h, n_cols_h, subplot_spec=outer[0, 1], hspace=0.25, wspace=0.1)
        for j in range(n_rows_h * n_cols_h):
            rr, cc = divmod(j, n_cols_h)
            ax_s = fig.add_subplot(right_gs[rr, cc])
            if j < len(picks):
                spec = full_ds[picks[j]][0].numpy().squeeze()
                ax_s.imshow(spec, cmap='viridis', origin='lower', aspect='auto')
                ax_s.set_title(str(j + 1), fontsize=8, color='black', pad=1)
                for spine in ax_s.spines.values():
                    spine.set_visible(False)
            else:
                ax_s.axis('off')
            ax_s.set_xticks([]); ax_s.set_yticks([])

        plt.suptitle(f'Figure H — Cluster {ci + 1}: watershed region & tile-sampled examples',
                     fontsize=12, fontweight='bold')
        out_path = os.path.join(save_dir, f'figure_H_cluster_{ci + 1:02d}_samples.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    print(f"Saved: figure_H_cluster_XX_samples.png ({n_clusters_found} figures)")

    # Save cluster information
    cluster_info = {
        'n_clusters': n_clusters_found,
        'centroids': centers.tolist(),
        'cluster_sizes': [int(np.sum(labels == i)) for i in range(n_clusters_found)],
        'bandwidth': bandwidth
    }

    import json
    with open(os.path.join(save_dir, 'cluster_info.json'), 'w') as f:
        json.dump(cluster_info, f, indent=2)

    print(f"\n=== Analysis Complete ===")
    print(f"Figures saved to: {save_dir}")
    print(f"Number of clusters: {n_clusters_found}")
    print(f"Cluster sizes: {cluster_info['cluster_sizes']}")

    return {
        'latent_coords': latent_coords,
        'mean_freqs': mean_freqs,
        'mask_counts': mask_counts,
        'centers': centers,
        'labels': labels,
        'social_distances': social_distances,
        'emitter_sexes': emitter_sexes,
    }


if __name__ == '__main__':
    fire.Fire(analyze_mouse_latents)
