"""
Diagnostic script to check if your model has learned good latent structure.
Helps identify whether the issue is visualization or model training.

Usage:
    python diagnose_model_structure.py \
        --model_path="path/to/checkpoint.tar" \
        --dataloc="path/to/mouse/data" \
        --lattice_m=20
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import fire
from tqdm import tqdm

from models.qmc_base import QMCLVM
from models.layers import TorusBasis
from models.sampling import gen_fib_basis
from train.model_saving_loading import load
from train.losses import binary_lp
from torch.optim import Adam
from data.mouse_data import load_mouse_data, mouse_data
from analysis.model_helpers import get_stacked_posterior


def compute_mean_frequency(spectrogram, freq_range_khz=(20, 120)):
    """Compute mean frequency of a spectrogram."""
    freq_bins = np.linspace(freq_range_khz[0], freq_range_khz[1], spectrogram.shape[0])
    freq_profile = spectrogram.sum(axis=1)
    if freq_profile.sum() > 0:
        return (freq_profile * freq_bins).sum() / freq_profile.sum()
    return 0


def diagnose_model(model_path, dataloc, lattice_m=20, n_samples=5000):
    """
    Run diagnostic checks on the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load data
    print("Loading data...")
    train_dict, val_dict = load_mouse_data(dataloc)
    test_ds = mouse_data(val_dict, masks_len_range=(1, 8), equal_sampling=False)
    n_workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4

    # Limit to n_samples for speed
    n_samples = min(n_samples, len(test_ds))
    subset_indices = np.random.choice(len(test_ds), n_samples, replace=False)
    test_loader = DataLoader(
        torch.utils.data.Subset(test_ds, subset_indices),
        num_workers=n_workers,
        shuffle=False,
        batch_size=1
    )

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
    print(f"Generating lattice (m={lattice_m})...")
    lattice = gen_fib_basis(m=lattice_m)

    # Compute posteriors
    print(f"Computing posteriors for {n_samples} samples...")
    posteriors = get_stacked_posterior(model, lattice, test_loader, binary_lp)
    map_indices = np.argmax(posteriors, axis=1)
    latent_coords = lattice[map_indices].numpy() % 1.0

    # Compute mean frequencies
    print("Computing mean frequencies...")
    mean_freqs = []
    for idx in tqdm(subset_indices):
        spec = test_ds[idx][0].numpy().squeeze()
        mean_freqs.append(compute_mean_frequency(spec))
    mean_freqs = np.array(mean_freqs)

    # ========== DIAGNOSTICS ==========
    print("\n" + "="*70)
    print("DIAGNOSTIC RESULTS")
    print("="*70)

    # 1. Latent space coverage
    unique_points = len(np.unique(latent_coords, axis=0))
    coverage_pct = 100 * unique_points / len(lattice)
    print(f"\n1. LATENT SPACE COVERAGE:")
    print(f"   - Total lattice points: {len(lattice)}")
    print(f"   - Unique points used: {unique_points}")
    print(f"   - Coverage: {coverage_pct:.1f}%")

    if coverage_pct < 10:
        print("   ⚠️  WARNING: Very low coverage - model may have collapsed!")
    elif coverage_pct < 30:
        print("   ⚠️  Low coverage - model needs more training or larger lattice")
    else:
        print("   ✓ Good coverage")

    # 2. Spatial organization by frequency
    print(f"\n2. FREQUENCY ORGANIZATION:")
    print(f"   - Frequency range: [{mean_freqs.min():.1f}, {mean_freqs.max():.1f}] kHz")

    # Compute correlation between frequency and position
    corr_x = np.corrcoef(latent_coords[:, 0], mean_freqs)[0, 1]
    corr_y = np.corrcoef(latent_coords[:, 1], mean_freqs)[0, 1]
    print(f"   - Correlation with dim 1: {corr_x:.3f}")
    print(f"   - Correlation with dim 2: {corr_y:.3f}")

    max_corr = max(abs(corr_x), abs(corr_y))
    if max_corr < 0.2:
        print("   ⚠️  WARNING: Weak frequency organization (|corr| < 0.2)")
        print("       Model hasn't learned to organize by frequency well")
    elif max_corr < 0.5:
        print("   ~ Moderate frequency organization (|corr| = 0.2-0.5)")
    else:
        print("   ✓ Strong frequency organization (|corr| > 0.5)")

    # 3. Local consistency check
    print(f"\n3. LOCAL CONSISTENCY:")
    # For each point, check if nearby points have similar frequencies
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean').fit(latent_coords)
    distances, indices = nbrs.kneighbors(latent_coords)

    freq_std_local = []
    for i in range(len(latent_coords)):
        neighbor_freqs = mean_freqs[indices[i]]
        freq_std_local.append(neighbor_freqs.std())

    avg_local_std = np.mean(freq_std_local)
    global_std = mean_freqs.std()
    consistency_ratio = avg_local_std / global_std

    print(f"   - Global frequency std: {global_std:.1f} kHz")
    print(f"   - Avg local frequency std (k=10): {avg_local_std:.1f} kHz")
    print(f"   - Local/global ratio: {consistency_ratio:.3f}")

    if consistency_ratio > 0.8:
        print("   ⚠️  WARNING: Poor local consistency - nearby points have very different frequencies")
    elif consistency_ratio > 0.5:
        print("   ~ Moderate local consistency")
    else:
        print("   ✓ Good local consistency - nearby points have similar frequencies")

    # 4. Training curve check
    print(f"\n4. TRAINING INFO:")
    if hasattr(run_info, '__len__'):
        losses = np.array(run_info)
        print(f"   - Total epochs: {len(losses)}")
        print(f"   - Initial loss: {-losses[0]:.2f}")
        print(f"   - Final loss: {-losses[-1]:.2f}")
        print(f"   - Improvement: {losses[-1] - losses[0]:.2f}")

        # Check if still improving
        last_100 = losses[-100:] if len(losses) > 100 else losses
        trend = np.polyfit(np.arange(len(last_100)), last_100, 1)[0]
        if trend > 0.01:
            print(f"   ✓ Still improving (trend: +{trend:.3f}/epoch)")
            print("      Consider training longer!")
        elif trend > -0.001:
            print(f"   ~ Plateaued (trend: {trend:.3f}/epoch)")
        else:
            print(f"   ⚠️  Degrading? (trend: {trend:.3f}/epoch)")

    # ========== VISUALIZATION ==========
    print(f"\n5. CREATING DIAGNOSTIC PLOTS...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Top-left: Scatter plot colored by frequency
    ax = axes[0, 0]
    scatter = ax.scatter(latent_coords[:, 0], latent_coords[:, 1],
                        c=mean_freqs, cmap='viridis', s=10, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.set_title(f'Scatter (n={n_samples})')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Mean freq (kHz)')

    # Top-right: 2D histogram of sample density
    ax = axes[0, 1]
    h, xedges, yedges = np.histogram2d(latent_coords[:, 0], latent_coords[:, 1], bins=50)
    extent = [0, 1, 0, 1]
    im = ax.imshow(h.T, origin='lower', extent=extent, aspect='equal', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.set_title('Sample density')
    plt.colorbar(im, ax=ax, label='Count')

    # Bottom-left: Frequency vs latent dim 1
    ax = axes[1, 0]
    ax.scatter(latent_coords[:, 0], mean_freqs, s=1, alpha=0.3)
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Mean frequency (kHz)')
    ax.set_title(f'Freq vs Dim1 (corr={corr_x:.3f})')
    ax.set_xlim(0, 1)

    # Bottom-right: Frequency vs latent dim 2
    ax = axes[1, 1]
    ax.scatter(latent_coords[:, 1], mean_freqs, s=1, alpha=0.3)
    ax.set_xlabel('Latent dim 2')
    ax.set_ylabel('Mean frequency (kHz)')
    ax.set_title(f'Freq vs Dim2 (corr={corr_y:.3f})')
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig('diagnostics.png', dpi=150)
    print("   ✓ Saved diagnostics.png")

    # ========== RECOMMENDATIONS ==========
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)

    if coverage_pct < 20 or max_corr < 0.3 or consistency_ratio > 0.7:
        print("\n❌ MODEL NEEDS MORE TRAINING")
        print("   Your model hasn't learned good structure yet.")
        print("   Actions:")
        print("   1. Train for more epochs (try 2-3x current)")
        print("   2. Check training loss is decreasing")
        print("   3. Try different hyperparameters (learning rate, batch size)")
    else:
        print("\n✓ MODEL STRUCTURE LOOKS REASONABLE")
        print("   The issue is likely visualization method.")
        print("   Actions:")
        print("   1. Use the heatmap visualization:")
        print("      python analyze_mouse_latents_heatmap.py \\")
        print(f"          --model_path={model_path} \\")
        print(f"          --dataloc={dataloc} \\")
        print("          --save_dir=./heatmap_output \\")
        print("          --sigma=5.0 \\")
        print("          --freq_range_khz='(20,120)'")
        print("   2. Tighten frequency range if needed:")
        print("      --vmin=30 --vmax=90")

    print("="*70 + "\n")


if __name__ == '__main__':
    fire.Fire(diagnose_model)
