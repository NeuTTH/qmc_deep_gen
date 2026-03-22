"""
Create smooth aggregated posterior plots (Figure F) like the paper.
Uses Gaussian KDE or interpolation instead of raw histograms.

Usage:
    python fixed_aggregated_posterior.py \
        --model_path="path/to/checkpoint.tar" \
        --dataloc="path/to/mouse/data" \
        --save_dir="./output" \
        --lattice_m=20 \
        --method="gaussian_kde"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import fire
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

from models.qmc_base import QMCLVM
from models.layers import TorusBasis
from models.sampling import gen_fib_basis
from train.model_saving_loading import load
from train.losses import binary_lp
from torch.optim import Adam
from data.mouse_data import load_mouse_data, mouse_data
from analysis.model_helpers import get_stacked_posterior, torus_forward, torus_reverse
from analysis.clustering import run_mean_shift


def create_smooth_aggregated_posterior(lattice, posteriors, resolution=200, method='gaussian_kde', sigma=3.0):
    """
    Create smooth aggregated posterior heatmap using various methods.

    Args:
        lattice: (n_lattice, 2) lattice points in [0,1]^2
        posteriors: (n_samples, n_lattice) posterior probabilities
        resolution: grid resolution for output heatmap
        method: 'gaussian_kde', 'interpolation', or 'smoothed_histogram'
        sigma: smoothing parameter (for gaussian_kde and smoothed_histogram)

    Returns:
        heatmap: (resolution, resolution) array
        extent: (x_min, x_max, y_min, y_max)
    """
    # Sum posteriors across all samples to get aggregated weights
    aggregated = posteriors.sum(axis=0)  # (n_lattice,)

    # Normalize to get density
    aggregated = aggregated / aggregated.sum()

    # Create output grid
    x_grid = np.linspace(0, 1, resolution)
    y_grid = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)

    if method == 'gaussian_kde':
        # Use Gaussian kernel density estimation
        # For each grid point, sum Gaussian kernels centered at lattice points
        print(f"  Computing Gaussian KDE with sigma={sigma}...")

        heatmap = np.zeros((resolution, resolution))

        # Vectorized computation for efficiency
        lattice_np = lattice.numpy() if hasattr(lattice, 'numpy') else lattice

        for i in tqdm(range(len(lattice_np)), desc="KDE"):
            # Gaussian kernel centered at lattice point i
            dx = X - lattice_np[i, 0]
            dy = Y - lattice_np[i, 1]

            # Handle periodic boundary (torus topology)
            dx = np.minimum(np.abs(dx), 1 - np.abs(dx))
            dy = np.minimum(np.abs(dy), 1 - np.abs(dy))

            kernel = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            heatmap += aggregated[i] * kernel

        # Normalize
        heatmap = heatmap / heatmap.max()

    elif method == 'interpolation':
        # Use griddata interpolation (cubic)
        print("  Computing cubic interpolation...")

        lattice_np = lattice.numpy() if hasattr(lattice, 'numpy') else lattice

        # Interpolate from lattice points to regular grid
        points = lattice_np
        values = aggregated

        heatmap = griddata(
            points,
            values,
            (X, Y),
            method='cubic',
            fill_value=0
        )

        # Smooth slightly to remove artifacts
        heatmap = gaussian_filter(heatmap, sigma=1.0)
        heatmap = np.clip(heatmap, 0, None)  # Remove negative values from interpolation

    elif method == 'smoothed_histogram':
        # Original histogram method but with heavy smoothing
        print(f"  Computing smoothed histogram with sigma={sigma}...")

        x_edges = np.linspace(0, 1, resolution + 1)
        y_edges = np.linspace(0, 1, resolution + 1)

        lattice_np = lattice.numpy() if hasattr(lattice, 'numpy') else lattice

        heatmap, _, _ = np.histogram2d(
            lattice_np[:, 0],
            lattice_np[:, 1],
            bins=[x_edges, y_edges],
            weights=aggregated
        )

        # Apply strong Gaussian smoothing
        heatmap = gaussian_filter(heatmap.T, sigma=sigma)

    else:
        raise ValueError(f"Unknown method: {method}")

    return heatmap, (0, 1, 0, 1)


def analyze_posterior(
    model_path,
    dataloc,
    save_dir,
    lattice_m=20,
    resolution=200,
    method='gaussian_kde',
    sigma=3.0,
    bandwidth=0.1,
    batch_size=1
):
    """
    Generate smooth aggregated posterior figure (like paper's Figure F).

    Args:
        model_path: Path to trained model checkpoint
        dataloc: Path to mouse data directory
        save_dir: Directory to save output figures
        lattice_m: Fibonacci lattice parameter (m=20 gives ~10K points)
        resolution: Grid resolution for heatmap (default 200)
        method: 'gaussian_kde', 'interpolation', or 'smoothed_histogram'
        sigma: Smoothing parameter (default 3.0)
        bandwidth: Bandwidth for mean-shift clustering
        batch_size: Batch size for data loading
    """

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

    # Generate lattice
    print(f"Generating Fibonacci lattice with m={lattice_m}...")
    lattice = gen_fib_basis(m=lattice_m)
    print(f"Lattice size: {len(lattice)} points")

    # Compute posteriors
    print("Computing posteriors...")
    posteriors = get_stacked_posterior(model, lattice, test_loader, binary_lp)
    print(f"Posteriors shape: {posteriors.shape}")

    # Get MAP estimates for clustering
    map_indices = np.argmax(posteriors, axis=1)
    latent_coords = lattice[map_indices].numpy() % 1.0

    # Create smooth aggregated posterior
    print(f"\nCreating aggregated posterior using method='{method}'...")
    heatmap, extent = create_smooth_aggregated_posterior(
        lattice,
        posteriors,
        resolution=resolution,
        method=method,
        sigma=sigma
    )

    # Run mean-shift clustering
    print("\nRunning mean-shift clustering...")
    embedded = torus_forward(latent_coords)
    weights = posteriors.max(axis=1)

    centers, wms, labels = run_mean_shift(
        embedded,
        seeds=embedded,
        weights=weights,
        bandwidth=bandwidth,
        n_jobs=min(16, n_workers),
        p=2,
        embedded=True,
        normal=False
    )

    print(f"Found {len(centers)} clusters")

    # ========== PLOT ==========
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        heatmap,
        extent=extent,
        origin='lower',
        aspect='equal',
        cmap='viridis',
        interpolation='bilinear'
    )

    # Overlay centroids
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c='red',
        s=100,
        marker='o',
        edgecolors='white',
        linewidths=1.5,
        zorder=10
    )

    # Label centroids
    for i, (x, y) in enumerate(centers):
        ax.text(
            x, y, str(i+1),
            color='red',
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='center',
            zorder=11
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Latent dimension 1', fontsize=12)
    ax.set_ylabel('Latent dimension 2', fontsize=12)
    ax.set_title('Aggregated posterior', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Posterior density', fontsize=10)

    plt.tight_layout()

    # Save with method name
    save_name = f'figure_F_aggregated_posterior_{method}.png'
    plt.savefig(os.path.join(save_dir, save_name), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, save_name.replace('.png', '.svg')), bbox_inches='tight')
    print(f"\nSaved: {save_name}")
    plt.close()

    print("\n=== Complete ===")
    print(f"Cluster count: {len(centers)}")
    print(f"\nTry different methods:")
    print(f"  --method=gaussian_kde --sigma=5.0   (smooth, kernel-based)")
    print(f"  --method=interpolation              (cubic interpolation)")
    print(f"  --method=smoothed_histogram --sigma=10.0  (histogram + smoothing)")


if __name__ == '__main__':
    fire.Fire(analyze_posterior)
