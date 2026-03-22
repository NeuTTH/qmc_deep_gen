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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader
import os
import fire

from models.qmc_base import QMCLVM, TorusBasis
from models.sampling import gen_fib_basis
from train.model_saving_loading import load
from train.losses import binary_lp
from torch.optim import Adam
from data.mouse_data import load_mouse_data, mouse_data
from analysis.model_helpers import get_stacked_posterior, torus_forward, torus_reverse
from analysis.clustering import run_mean_shift


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


def analyze_mouse_latents(
    model_path,
    dataloc,
    save_dir,
    lattice_m=20,
    bandwidth=0.1,
    batch_size=1,
    freq_range_khz=(20, 120),
    scatter_size=.5,
    scatter_alpha=0.1
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
    """

    # Setup
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading mouse data...")
    train_dict, val_dict = load_mouse_data(dataloc)
    # test_ds = mouse_data(val_dict, masks_len_range=(1, 8), equal_sampling=True, max_samples=20000)
    ## Using training data for better coverage
    test_ds = mouse_data(train_dict, masks_len_range=(1, 8), equal_sampling=True, max_samples=100000)
    n_workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4
    test_loader = DataLoader(test_ds, num_workers=n_workers, shuffle=False, batch_size=batch_size)

    # Reconstruct model architecture (must match training)
    print("Loading model...")
    latent_dim = 2

    # This is the architecture from bartul_mouse.py - adjust if yours differs
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

    # Compute posteriors for all test samples
    print("Computing posteriors...")
    posteriors = get_stacked_posterior(model, lattice, test_loader, binary_lp)
    print(f"Posteriors shape: {posteriors.shape}")

    # Get MAP estimates for each sample
    map_indices = np.argmax(posteriors, axis=1)
    latent_coords = lattice[map_indices].numpy()
    
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

    # Compute mean frequency and extract mask counts for each sample
    print("Computing mean frequencies and extracting metadata...")
    mean_freqs = []
    mask_counts = []
    spectrograms = []

    for i in range(len(test_ds)):
        if i % 1000 == 0:
            print(f"Processing sample {i} of {len(test_ds)}")
        sample_data = test_ds[i]
        spec = sample_data[0].numpy().squeeze()  # (H, W)
        spectrograms.append(spec)

        # Compute mean frequency
        mean_freq = compute_mean_frequency(spec)
        mean_freqs.append(mean_freq)

        # Extract mask_count (syllable length) if available
        if len(sample_data) > 1:
            # Assuming sample_data[1] contains mask or mask_count
            mask = sample_data[1]
            if hasattr(mask, 'sum'):
                mask_count = mask.sum().item() if torch.is_tensor(mask) else mask.sum()
            elif hasattr(mask, '__len__'):
                mask_count = len(mask)
            else:
                mask_count = mask  # Assume it's already a count
        else:
            # Infer from spectrogram time length (number of non-zero time bins)
            mask_count = (spec.sum(axis=0) > 0).sum()

        mask_counts.append(mask_count)

    mean_freqs = np.array(mean_freqs)
    mask_counts = np.array(mask_counts)
    spectrograms = np.array(spectrograms)
    
    print(f"Number of samples: {len(latent_coords)}")
    print(f"Latent coords range: X=[{latent_coords[:, 0].min():.3f}, {latent_coords[:, 0].max():.3f}], Y=[{latent_coords[:,1].min():.3f}, {latent_coords[:, 1].max():.3f}]")
    print(f"Unique points: {len(np.unique(latent_coords, axis=0))}")

    # ============= FIGURE E1: Embedded latents colored by mean frequency =============
    print("\nGenerating Figure E1: Embedded latents (colored by mean frequency)...")
    fig, ax = plt.subplots(figsize=(6, 6))

    scatter = ax.scatter(
        latent_coords[:, 0],
        latent_coords[:, 1],
        c=mean_freqs,
        cmap='viridis',
        s=scatter_size,
        alpha=scatter_alpha,
        rasterized=True,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Latent dimension 1', fontsize=12)
    ax.set_ylabel('Latent dimension 2', fontsize=12)
    ax.set_title('Embedded latents', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    cbar = plt.colorbar(scatter, ax=ax)
    freq_unit = 'kHz' if freq_range_khz is not None else 'bins'
    cbar.set_label(f'Mean frequency ({freq_unit})', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'figure_E_embedded_latents_by_freq.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'figure_E_embedded_latents_by_freq.svg'), bbox_inches='tight')
    print(f"Saved: figure_E_embedded_latents_by_freq.png/svg")
    plt.close()

    # ============= FIGURE E2: Embedded latents colored by mask count =============
    print("\nGenerating Figure E2: Embedded latents (colored by mask count)...")
    fig, ax = plt.subplots(figsize=(6, 6))

    scatter = ax.scatter(
        latent_coords[:, 0],
        latent_coords[:, 1],
        c=mask_counts,
        cmap='plasma',
        s=scatter_size,
        alpha=scatter_alpha,
        rasterized=True,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Latent dimension 1', fontsize=12)
    ax.set_ylabel('Latent dimension 2', fontsize=12)
    ax.set_title('Embedded latents', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Syllable length (time bins)', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'figure_E_embedded_latents_by_length.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'figure_E_embedded_latents_by_length.svg'), bbox_inches='tight')
    print(f"Saved: figure_E_embedded_latents_by_length.png/svg")
    plt.close()

    # ============= FIGURE F: Aggregated posterior with mean-shift centroids =============
    print("\nGenerating Figure F: Aggregated posterior...")

    # Create aggregated posterior heatmap
    heatmap, extent = create_aggregated_posterior_heatmap(
        lattice.numpy(),
        posteriors,
        resolution=200
    )

    # Run mean-shift clustering
    print("Running mean-shift clustering...")
    embedded = torus_forward(latent_coords)
    weights = posteriors.max(axis=1)  # Use posterior confidence as weights

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

    # centers are in [0,1]^2 after torus_reverse inside run_mean_shift
    print(f"Found {len(centers)} clusters")

    # Plot
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

    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    ax.set_xlabel('Latent dimension 1', fontsize=12)
    ax.set_ylabel('Latent dimension 2', fontsize=12)
    ax.set_title('Aggregated posterior', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean frequency (MHz)', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'figure_F_aggregated_posterior.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'figure_F_aggregated_posterior.svg'), bbox_inches='tight')
    print(f"Saved: figure_F_aggregated_posterior.png/svg")
    plt.close()

    # ============= FIGURE G: Example spectrograms from each cluster =============
    print("\nGenerating Figure G: Example spectrograms per cluster...")

    # For each cluster, find samples and show examples
    n_clusters_found = len(centers)
    n_cols = 5
    n_rows = int(np.ceil(n_clusters_found / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axes = axes.flatten() if n_clusters_found > 1 else [axes]

    for cluster_id in range(n_clusters_found):
        ax = axes[cluster_id]

        # Find samples in this cluster
        cluster_samples = np.where(labels == cluster_id)[0]

        if len(cluster_samples) > 0:
            # Show a random example (or the closest to centroid)
            example_idx = cluster_samples[np.random.choice(len(cluster_samples))]
            spec = spectrograms[example_idx]

            ax.imshow(spec, cmap='viridis', origin='lower', aspect='auto')
            ax.set_title(f'{cluster_id+1}', color='red', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Empty', ha='center', va='center', transform=ax.transAxes)

        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i in range(n_clusters_found, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'figure_G_cluster_examples.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: figure_G_cluster_examples.png")
    plt.close()

    # Save cluster information
    cluster_info = {
        'n_clusters': n_clusters_found,
        'centroids': centers.tolist(),
        'cluster_sizes': [np.sum(labels == i) for i in range(n_clusters_found)],
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
        'posteriors': posteriors,
        'latent_coords': latent_coords,
        'mean_freqs': mean_freqs,
        'mask_counts': mask_counts,
        'centers': centers,
        'labels': labels,
        'spectrograms': spectrograms
    }


if __name__ == '__main__':
    fire.Fire(analyze_mouse_latents)
