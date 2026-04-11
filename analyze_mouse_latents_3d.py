"""
3D latent space visualizations for trained mouse vocalization model.
Covers Figures E1–E6 (scatter colored by frequency, mask count, condition,
social distance, emitter sex, duration) plus decoder grid slices.

Usage:
    python analyze_mouse_latents_3D.py \
        --model_path="path/to/checkpoint.tar" \
        --dataloc="path/to/mouse/data" \
        --save_dir="path/to/save/figures" \
        --n_lattice_points=5000 \
        --n_slices=8
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from torch.utils.data import DataLoader
import os
import fire
from tqdm import tqdm
from torch.optim import Adam

from models.qmc_base import QMCLVM, TorusBasis
from models.sampling import roberts_sequence, gen_korobov_basis
from train.model_saving_loading import load
from train.losses import binary_lp
from data.mouse_data import load_mouse_data, mouse_data
from analysis.model_helpers import get_posterior_summaries, torus_forward, torus_reverse


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_latent_scatter_3d(
    latent_coords,
    color_values,
    save_path,
    label,
    cmap='viridis',
    vmin=None,
    vmax=None,
    color_map=None,
    category_order=None,
    scatter_size=5,
    scatter_alpha=0.3,
    elev=25,
    azim=45,
):
    """
    3-D latent scatter with continuous colormap or discrete category colors.

    Continuous mode (color_map is None):
        color_values: numeric array (N,), NaN entries drawn gray behind.
        Adds a colorbar labeled `label`.

    Discrete mode (color_map is {category: hex} dict):
        color_values: object array (N,), None/NaN entries drawn gray behind.
        Adds a legend.
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    if color_map is None:
        # --- continuous ---
        vals = np.asarray(color_values, dtype=float)
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            ax.scatter(
                latent_coords[nan_mask, 0],
                latent_coords[nan_mask, 1],
                latent_coords[nan_mask, 2],
                c='lightgray', s=scatter_size, alpha=scatter_alpha,
                rasterized=True, linewidth=0, marker='.', edgecolors='none',
            )
        valid = ~nan_mask
        sort_idx = np.argsort(vals[valid])
        sc = ax.scatter(
            latent_coords[valid][sort_idx, 0],
            latent_coords[valid][sort_idx, 1],
            latent_coords[valid][sort_idx, 2],
            c=vals[valid][sort_idx],
            cmap=cmap, vmin=vmin, vmax=vmax,
            s=scatter_size, alpha=scatter_alpha,
            rasterized=True, linewidth=0, marker='.', edgecolors='none',
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.4, pad=0.1)
        cbar.solids.set_alpha(1)
        if label:
            cbar.set_label(label, fontsize=10)
    else:
        # --- discrete ---
        cats = np.asarray(color_values, dtype=object)
        nan_mask = np.array([c is None or (isinstance(c, float) and np.isnan(c)) for c in cats])
        if nan_mask.any():
            ax.scatter(
                latent_coords[nan_mask, 0],
                latent_coords[nan_mask, 1],
                latent_coords[nan_mask, 2],
                c='lightgray', s=scatter_size, alpha=scatter_alpha,
                rasterized=True, linewidth=0, marker='.', edgecolors='none',
            )
        order = category_order if category_order is not None else sorted(color_map)
        for cat in order:
            mask = cats == cat
            if not mask.any():
                continue
            ax.scatter(
                latent_coords[mask, 0],
                latent_coords[mask, 1],
                latent_coords[mask, 2],
                c=color_map[cat], label=cat,
                s=scatter_size, alpha=scatter_alpha,
                rasterized=True, linewidth=0, marker='.', edgecolors='none',
            )
        ax.legend(markerscale=4, fontsize=9, loc='best')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('Latent dim 1', fontsize=10, labelpad=6)
    ax.set_ylabel('Latent dim 2', fontsize=10, labelpad=6)
    ax.set_zlabel('Latent dim 3', fontsize=10, labelpad=6)
    ax.set_title('Embedded latents', fontsize=13, fontweight='bold')
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass  # older matplotlib
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {os.path.basename(save_path)}')
    plt.close()


def save_decoder_grid_slices(model, device, save_dir, n_slices=8, n_samples_dim=15, cmap='viridis'):
    """
    For each of n_slices evenly spaced z-values, decode an (n_samples_dim x n_samples_dim)
    grid of (x, y) points and save one PNG per slice.

    Mirrors the approach in plotting/visualize_3d.py.
    """
    z_values = np.linspace(0, 1, n_slices)
    xx, yy = torch.meshgrid(
        [torch.linspace(0, 1, n_samples_dim)] * 2, indexing='ij'
    )

    with torch.no_grad():
        for si, z_val in enumerate(z_values):
            zz = torch.full((n_samples_dim * n_samples_dim, 1), float(z_val))
            grid = torch.cat([
                xx.flatten().unsqueeze(1),
                yy.flatten().unsqueeze(1),
                zz,
            ], dim=1).to(device)

            imgs = model(grid, mod=False, random=False)  # (N, 1, H, W)
            imgs = imgs.detach().cpu()

            mosaic = [
                [f"s{ii * n_samples_dim + jj}" for ii in range(n_samples_dim)]
                for jj in range(n_samples_dim)
            ]
            fig, axes = plt.subplot_mosaic(
                mosaic, figsize=(20, 20), sharex=True, sharey=True,
                gridspec_kw={'wspace': 0.01, 'hspace': 0.01},
            )
            for ii in range(n_samples_dim * n_samples_dim):
                ax = axes[f"s{ii}"]
                ax.imshow(imgs[ii, 0], cmap=cmap, origin='lower')
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(f'Decoder grid  —  z = {z_val:.3f}  (slice {si + 1}/{n_slices})', fontsize=14)
            fn = os.path.join(save_dir, f'decoder_grid_slice_{si + 1:02d}_z{z_val:.3f}.png')
            plt.savefig(fn, bbox_inches='tight')
            print(f'Saved: {os.path.basename(fn)}')
            plt.close()


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def analyze_mouse_latents_3d(
    model_path,
    dataloc,
    save_dir,
    n_lattice_points=5000,
    lattice_type='roberts',
    korobov_a=76,
    n_slices=8,
    n_samples_dim=15,
    batch_size=32,
    freq_range_khz=(20, 120),
    scatter_size=5,
    scatter_alpha=0.2,
    samples_per_mask=None,
    beh_features_path="/jukebox/falkner/Dexter/vocal_beh/data/full_dataset/utils/usv_beh_features.pkl",
    elev=25,
    azim=45,
):
    """
    Generate 3-D latent space analysis figures for a mouse vocalization model
    trained with latent_dim=3.

    Args:
        model_path:          Path to .tar checkpoint
        dataloc:             Directory with full_data.pt (or train/val split)
        save_dir:            Output directory
        n_lattice_points:    Number of lattice points for posterior computation
        lattice_type:        'roberts' or 'korobov'
        korobov_a:           Generator for Korobov lattice (only used if lattice_type='korobov')
        n_slices:            Number of z-slices for decoder grid visualisation
        n_samples_dim:       Grid side length per slice (n_samples_dim² images per slice)
        batch_size:          DataLoader batch size for posterior computation
        freq_range_khz:      (min, max) kHz range for mean-frequency colorbar
        scatter_size:        Point size in scatter plots
        scatter_alpha:       Transparency in scatter plots
        samples_per_mask:    Max samples per masks_len bin (quantile-evenly spaced by duration; None = all)
        beh_features_path:   Path to behavioral features pkl (optional)
        elev / azim:         3D view angles in degrees
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Behavioral features ────────────────────────────────────────────────
    import pickle
    beh_features = {}
    if beh_features_path and os.path.exists(beh_features_path):
        with open(beh_features_path, 'rb') as f:
            beh_features = pickle.load(f)
        print(f'Loaded behavioral features for {len(beh_features)} sessions')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Data ───────────────────────────────────────────────────────────────
    print('Loading mouse data...')
    full_dict = torch.load(os.path.join(dataloc, 'full_data.pt'), mmap=True)
    full_ds = mouse_data(full_dict, filter_mask=True, lo=1, hi=8,
                         sampling_strategy='mask_duration', samples_per_mask=samples_per_mask)

    n_workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4
    loader = DataLoader(full_ds, num_workers=n_workers, shuffle=False, batch_size=batch_size)

    # ── Model ──────────────────────────────────────────────────────────────
    print('Loading model...')
    latent_dim = 3
    decoder = nn.Sequential(
        nn.Linear(2 * latent_dim, 2048),
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
    model, optimizer, _ = load(model, optimizer, model_path)
    model.to(device).eval()
    print(f'Model loaded from {model_path}')

    # ── Lattice ────────────────────────────────────────────────────────────
    print(f'Generating 3-D {lattice_type} lattice ({n_lattice_points} points)...')
    if lattice_type == 'korobov':
        lattice = gen_korobov_basis(korobov_a, latent_dim, n_lattice_points)
    else:
        lattice = roberts_sequence(n_lattice_points, latent_dim)
    print(f'Lattice shape: {lattice.shape}')

    # ── Posteriors ─────────────────────────────────────────────────────────
    print('Computing posteriors...')
    torus_weighted, aggregated, weights = get_posterior_summaries(model, lattice, loader, binary_lp)
    latent_coords = torus_reverse(torus_weighted, dim=latent_dim) % 1.0  # (N, 3)
    print(f'Latent coords: {latent_coords.shape}, '
          f'ranges X=[{latent_coords[:,0].min():.2f},{latent_coords[:,0].max():.2f}] '
          f'Y=[{latent_coords[:,1].min():.2f},{latent_coords[:,1].max():.2f}] '
          f'Z=[{latent_coords[:,2].min():.2f},{latent_coords[:,2].max():.2f}]')

    # ── Metadata ───────────────────────────────────────────────────────────
    print('Extracting metadata...')
    mean_freqs_list, mask_counts_list, durations_list = [], [], []
    H = None
    for batch in tqdm(loader, desc='metadata'):
        spec = batch[0].squeeze(1).float()          # (B, H, W)
        if H is None:
            H = spec.shape[1]
            freq_bins = torch.arange(H, dtype=torch.float32)
            if freq_range_khz is not None:
                freq_bins = torch.linspace(freq_range_khz[0], freq_range_khz[1], H)
        freq_profile = spec.sum(dim=2)              # (B, H)
        total = freq_profile.sum(dim=1).clamp(min=1e-10)
        mean_freqs_list.append(((freq_profile * freq_bins).sum(dim=1) / total).numpy())
        mask_counts_list.append(batch[1].numpy())
        durations_list.append(batch[2].numpy())

    mean_freqs  = np.concatenate(mean_freqs_list)
    mask_counts = np.concatenate(mask_counts_list)
    durations   = np.concatenate(durations_list)

    raw_spec_ids = full_ds.spec_ids

    scatter_kw = dict(scatter_size=scatter_size, scatter_alpha=scatter_alpha, elev=elev, azim=azim)

    # ── Figure E1: mean frequency ──────────────────────────────────────────
    print('\nFigure E1: mean frequency...')
    freq_unit = 'kHz' if freq_range_khz is not None else 'bins'
    plot_latent_scatter_3d(
        latent_coords, mean_freqs,
        save_path=os.path.join(save_dir, 'figure_E1_latents_by_freq.png'),
        label=f'Mean frequency ({freq_unit})',
        cmap='viridis', **scatter_kw,
    )

    # ── Figure E2: mask count ──────────────────────────────────────────────
    print('\nFigure E2: mask count...')
    plot_latent_scatter_3d(
        latent_coords, mask_counts,
        save_path=os.path.join(save_dir, 'figure_E2_latents_by_mask_count.png'),
        label='SAM mask count (time bins)',
        cmap='plasma', **scatter_kw,
    )

    # ── Figure E3: condition ───────────────────────────────────────────────
    print('\nFigure E3: condition...')
    lone_male_ids = {
        "20250912_155546", "20250912_170514", "20250919_145712",
        "20250921_155753", "20250927_135343",
    }
    cond_labels = []
    for sid in raw_spec_ids:
        if sid is None:
            cond_labels.append('Unknown')
            continue
        parts = sid.split('_')
        dt_str = f'{parts[0]}_{parts[1]}'
        cond = parts[2] if len(parts) > 2 else 'Unknown'
        if cond == 'ephys':
            cond = 'Lone-Male' if dt_str in lone_male_ids else 'Male-Female'
        cond_labels.append(cond)
    cond_labels = np.array(cond_labels)

    cond_order  = ['Female-Female', 'Male-Female', 'Lone-Male']
    cond_colors = {'Female-Female': '#e377c2', 'Male-Female': '#1f77b4', 'Lone-Male': '#ff7f0e'}
    plot_latent_scatter_3d(
        latent_coords, cond_labels,
        save_path=os.path.join(save_dir, 'figure_E3_latents_by_condition.png'),
        label=None,
        color_map=cond_colors, category_order=cond_order, **scatter_kw,
    )
    for c in cond_order:
        print(f'  {c}: {(cond_labels == c).sum()}')

    # ── Figures E4 & E5: behavioral features ──────────────────────────────
    social_distances, emitter_sexes = [], []
    for sid in raw_spec_ids:
        if sid is None:
            social_distances.append(np.nan)
            emitter_sexes.append(None)
            continue
        parts = sid.split('_')
        session_id = f'{parts[0]}_{parts[1]}'
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
                social_distances.append(row['avg_social_distance'])
                emitter_sexes.append(row['emitter_sex'])
    social_distances = np.array(social_distances, dtype=float)
    emitter_sexes    = np.array(emitter_sexes, dtype=object)
    print(f'\nBehavioral features: {(~np.isnan(social_distances)).sum()} with distance, '
          f'{(emitter_sexes != None).sum()} with sex')

    print('\nFigure E4: social distance...')
    plot_latent_scatter_3d(
        latent_coords, social_distances,
        save_path=os.path.join(save_dir, 'figure_E4_latents_by_social_dist.png'),
        label='Social distance (cm)',
        cmap='YlOrRd_r', vmin=0, vmax=85, **scatter_kw,
    )

    print('\nFigure E5: emitter sex...')
    sex_colors = {'female': '#d44fa8', 'male': '#5baed4'}
    plot_latent_scatter_3d(
        latent_coords, emitter_sexes,
        save_path=os.path.join(save_dir, 'figure_E5_latents_by_emitter_sex.png'),
        label=None,
        color_map=sex_colors, category_order=['female', 'male'], **scatter_kw,
    )

    # ── Figure E6: duration ────────────────────────────────────────────────
    print('\nFigure E6: duration...')
    plot_latent_scatter_3d(
        latent_coords, durations,
        save_path=os.path.join(save_dir, 'figure_E6_latents_by_duration.png'),
        label='Duration (samples)',
        cmap='magma', **scatter_kw,
    )

    # ── Decoder grid slices ────────────────────────────────────────────────
    print(f'\nDecoder grid slices ({n_slices} slices, {n_samples_dim}x{n_samples_dim} each)...')
    save_decoder_grid_slices(
        model, device, save_dir,
        n_slices=n_slices,
        n_samples_dim=n_samples_dim,
    )

    print(f'\n=== Done. Figures saved to: {save_dir} ===')


if __name__ == '__main__':
    fire.Fire(analyze_mouse_latents_3d)
