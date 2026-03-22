"""
Create Figure E as a smoothed heatmap (like the paper) instead of scatter plot.
This aggregates samples into a 2D histogram weighted by mean frequency.

Usage:
    python analyze_mouse_latents_heatmap.py \
        --model_path="path/to/checkpoint.tar" \
        --dataloc="path/to/mouse/data" \
        --save_dir="path/to/save/figures" \
        --lattice_m=20 \
        --resolution=200 \
        --freq_range_khz="(20,120)"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader
import os
import fire
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from models.qmc_base import QMCLVM
from models.layers import TorusBasis
from models.sampling import gen_fib_basis
from train.model_saving_loading import load
from train.losses import binary_lp
from torch.optim import Adam
from data.mouse_data import load_mouse_data, mouse_data
from analysis.model_helpers import get_stacked_posterior


def compute_mean_frequency(spectrogram, freq_bins=None, freq_range_khz=None):
    """Compute mean frequency of a spectrogram."""
    if freq_bins is None:
        if freq_range_khz is not None:
            freq_bins = np.linspace(freq_range_khz[0], freq_range_khz[1], spectrogram.shape[0])
        else:
            freq_bins = np.arange(spectrogram.shape[0])

    freq_profile = spectrogram.sum(axis=1)

    if freq_profile.sum() > 0:
        mean_freq = (freq_profile * freq_bins).sum() / freq_profile.sum()
    else:
        mean_freq = 0

    return mean_freq


def create_frequency_heatmap(latent_coords, mean_freqs, resolution=200, sigma=2.0):
    """
    Create a smoothed 2D heatmap showing mean frequency across latent space.

    Args:
        latent_coords: (N, 2) array of latent coordinates in [0,1]^2
        mean_freqs: (N,) array of mean frequencies
        resolution: grid resolution for heatmap
        sigma: Gaussian smoothing kernel width

    Returns:
        heatmap: (resolution, resolution) array
        extent: (x_min, x_max, y_min, y_max)
    """
    # Create 2D histogram weighted by frequency
    x_edges = np.linspace(0, 1, resolution + 1)
    y_edges = np.linspace(0, 1, resolution + 1)

    # Accumulate frequency values in each bin
    freq_sum, _, _ = np.histogram2d(
        latent_coords[:, 0],
        latent_coords[:, 1],
        bins=[x_edges, y_edges],
        weights=mean_freqs
    )

    # Count samples in each bin
    count, _, _ = np.histogram2d(
        latent_coords[:, 0],
        latent_coords[:, 1],
        bins=[x_edges, y_edges]
    )

    # Average frequency per bin (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_freq_map = freq_sum / count
        mean_freq_map[~np.isfinite(mean_freq_map)] = np.nan

    # Apply Gaussian smoothing
    # Handle NaNs by interpolating from valid neighbors
    mask = ~np.isnan(mean_freq_map)
    if mask.sum() > 0:
        # Smooth only where we have data
        smoothed = gaussian_filter(np.nan_to_num(mean_freq_map), sigma=sigma)

        # Also smooth the mask to weight interpolation
        weight_map = gaussian_filter(mask.astype(float), sigma=sigma)

        # Combine: weighted average where we have nearby data
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = smoothed / (weight_map + 1e-10)
            # Set areas with no nearby data to NaN
            heatmap[weight_map < 0.1] = np.nan
    else:
        heatmap = mean_freq_map

    return heatmap.T, (0, 1, 0, 1)  # Transpose for correct orientation


def analyze_mouse_heatmap(
    model_path,
    dataloc,
    save_dir,
    lattice_m=20,
    resolution=200,
    sigma=3.0,
    batch_size=1,
    freq_range_khz=None,
    vmin=None,
    vmax=None,
    cmap='viridis'
):
    """
    Generate heatmap-style Figure E (like the paper).

    Args:
        model_path: Path to trained model checkpoint (.tar file)
        dataloc: Path to mouse data directory
        save_dir: Directory to save output figures
        lattice_m: Fibonacci lattice parameter (m=20 gives ~10K points)
        resolution: Grid resolution for heatmap (default 200)
        sigma: Gaussian smoothing width (default 3.0, increase for smoother)
        batch_size: Batch size for data loading
        freq_range_khz: (min, max) frequency range in kHz, e.g., (20, 120)
        vmin, vmax: Colorbar limits (None = auto)
        cmap: Colormap name (default 'viridis')
    """

    # Setup
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading mouse data...")
    train_dict, val_dict = load_mouse_data(dataloc)
    test_ds = mouse_data(val_dict, masks_len_range=(1, 8), equal_sampling=False)
    n_workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4
    test_loader = DataLoader(test_ds, num_workers=n_workers, shuffle=False, batch_size=batch_size)

    # Load model
    print("Loading model...")
    latent_dim = 2

    import torch.nn as nn
    decoder = nn.Sequential(
        nn.Linear(2*latent_dim, 2048),
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

    # Compute posteriors
    print("Computing posteriors...")
    posteriors = get_stacked_posterior(model, lattice, test_loader, binary_lp)
    print(f"Posteriors shape: {posteriors.shape}")

    # Get MAP estimates
    map_indices = np.argmax(posteriors, axis=1)
    latent_coords = lattice[map_indices].numpy() % 1.0

    # Compute mean frequencies
    print("Computing mean frequencies...")
    mean_freqs = []

    for i in tqdm(range(len(test_ds))):
        sample_data = test_ds[i]
        spec = sample_data[0].numpy().squeeze()
        mean_freq = compute_mean_frequency(spec, freq_range_khz=freq_range_khz)
        mean_freqs.append(mean_freq)

    mean_freqs = np.array(mean_freqs)

    print(f"\nData statistics:")
    print(f"  Number of samples: {len(latent_coords)}")
    print(f"  Frequency range: [{mean_freqs.min():.1f}, {mean_freqs.max():.1f}]")
    print(f"  Unique lattice points used: {len(np.unique(latent_coords, axis=0))}")

    # Create heatmap
    print(f"\nGenerating heatmap (resolution={resolution}, sigma={sigma})...")
    heatmap, extent = create_frequency_heatmap(
        latent_coords,
        mean_freqs,
        resolution=resolution,
        sigma=sigma
    )

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        heatmap,
        extent=extent,
        origin='lower',
        aspect='equal',
        cmap=cmap,
        interpolation='bilinear',
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Latent dimension 1', fontsize=12)
    ax.set_ylabel('Latent dimension 2', fontsize=12)
    ax.set_title('Embedded latents', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    cbar = plt.colorbar(im, ax=ax)
    freq_unit = 'kHz' if freq_range_khz is not None else 'bins'
    cbar.set_label(f'Mean frequency ({freq_unit})', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'figure_E_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'figure_E_heatmap.svg'), bbox_inches='tight')
    print(f"Saved: figure_E_heatmap.png/svg")
    plt.close()

    # Also save a version with scatter overlay
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        heatmap,
        extent=extent,
        origin='lower',
        aspect='equal',
        cmap=cmap,
        interpolation='bilinear',
        vmin=vmin,
        vmax=vmax,
        alpha=0.7
    )

    # Overlay scatter points
    scatter = ax.scatter(
        latent_coords[:, 0],
        latent_coords[:, 1],
        c=mean_freqs,
        cmap=cmap,
        s=1,
        alpha=0.2,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Latent dimension 1', fontsize=12)
    ax.set_ylabel('Latent dimension 2', fontsize=12)
    ax.set_title('Embedded latents (heatmap + scatter)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'Mean frequency ({freq_unit})', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'figure_E_heatmap_overlay.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: figure_E_heatmap_overlay.png")
    plt.close()

    print("\n=== Complete ===")
    print(f"Try adjusting:")
    print(f"  --sigma=5.0  (smoother, more like paper)")
    print(f"  --vmin=30 --vmax=90  (tighter frequency range)")
    print(f"  --resolution=300  (higher detail)")


if __name__ == '__main__':
    fire.Fire(analyze_mouse_heatmap)
