"""
Debug script to check why Figure E is blank.
Run this to diagnose the issue.
"""

import numpy as np
import matplotlib.pyplot as plt

# Quick diagnostic - modify these paths
model_path = "path/to/your/checkpoint.tar"  # UPDATE THIS
dataloc = "path/to/your/data"  # UPDATE THIS

import torch
from torch.utils.data import DataLoader
from data.mouse_data import load_mouse_data, mouse_data
from models.qmc_base import QMCLVM
from models.layers import TorusBasis
from models.sampling import gen_fib_basis
from train.model_saving_loading import load
from train.losses import binary_lp
from torch.optim import Adam
from analysis.model_helpers import get_stacked_posterior
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
print("Loading data...")
train_dict, val_dict = load_mouse_data(dataloc)
test_ds = mouse_data(val_dict, masks_len_range=(1, 8), equal_sampling=False)
print(f"Test dataset size: {len(test_ds)}")

# Check first sample
sample = test_ds[0]
print(f"Sample shape: {sample[0].shape}")
print(f"Sample min/max: {sample[0].min():.3f} / {sample[0].max():.3f}")

# Load model
print("\nLoading model...")
latent_dim = 2
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

# Generate lattice and compute posteriors for first 100 samples
print("\nComputing posteriors for first 100 samples...")
lattice = gen_fib_basis(m=15)  # Smaller for speed
n_workers = 4
test_loader_small = DataLoader(
    torch.utils.data.Subset(test_ds, range(min(100, len(test_ds)))),
    batch_size=1,
    shuffle=False,
    num_workers=n_workers
)

posteriors = get_stacked_posterior(model, lattice, test_loader_small, binary_lp)
print(f"Posteriors shape: {posteriors.shape}")

# Get MAP estimates
map_indices = np.argmax(posteriors, axis=1)
latent_coords = lattice[map_indices].numpy()

print(f"\nLatent coordinates statistics:")
print(f"  Shape: {latent_coords.shape}")
print(f"  Dim 1 - min: {latent_coords[:, 0].min():.3f}, max: {latent_coords[:, 0].max():.3f}, mean: {latent_coords[:, 0].mean():.3f}")
print(f"  Dim 2 - min: {latent_coords[:, 1].min():.3f}, max: {latent_coords[:, 1].max():.3f}, mean: {latent_coords[:, 1].mean():.3f}")
print(f"  Unique points: {len(np.unique(latent_coords, axis=0))}")

# Check if all points are the same
if len(np.unique(latent_coords, axis=0)) == 1:
    print("\n⚠️  WARNING: All samples map to the SAME lattice point!")
    print(f"   Point: {latent_coords[0]}")
elif len(np.unique(latent_coords, axis=0)) < 10:
    print(f"\n⚠️  WARNING: Only {len(np.unique(latent_coords, axis=0))} unique points!")

# Plot with different settings
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

settings = [
    ("tiny (size=1, alpha=0.3)", {'s': 1, 'alpha': 0.3}),
    ("small (size=5, alpha=0.5)", {'s': 5, 'alpha': 0.5}),
    ("medium (size=20, alpha=0.8)", {'s': 20, 'alpha': 0.8}),
    ("large (size=50, alpha=1.0)", {'s': 50, 'alpha': 1.0}),
    ("huge (size=100, alpha=1.0)", {'s': 100, 'alpha': 1.0}),
    ("markers (size=50, marker='x')", {'s': 50, 'alpha': 1.0, 'marker': 'x'}),
]

for ax, (label, kwargs) in zip(axes.flatten(), settings):
    ax.scatter(latent_coords[:, 0], latent_coords[:, 1], c='red', **kwargs)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.set_title(label)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_figure_e.png', dpi=150)
print(f"\n✅ Saved debug plot to: debug_figure_e.png")
print("   Check this to see which settings make points visible!")

plt.close()

# Print recommendations
print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
if len(np.unique(latent_coords, axis=0)) == 1:
    print("❌ Problem: All samples collapse to one point!")
    print("   This means your model didn't learn good representations.")
    print("   Check:")
    print("   - Did training converge?")
    print("   - Are your training losses decreasing?")
    print("   - Try training longer or with different hyperparameters")
elif latent_coords.min() < -0.1 or latent_coords.max() > 1.1:
    print("❌ Problem: Points are outside [0,1] range!")
    print(f"   Range: [{latent_coords.min():.3f}, {latent_coords.max():.3f}]")
    print("   This shouldn't happen with lattice coordinates.")
elif len(latent_coords) < 10:
    print("❌ Problem: Very few test samples!")
    print(f"   Only {len(latent_coords)} samples in test set.")
else:
    print("✅ Data looks OK - problem is likely visualization settings")
    print(f"   You have {len(latent_coords)} points spanning:")
    print(f"   Dim 1: [{latent_coords[:, 0].min():.3f}, {latent_coords[:, 0].max():.3f}]")
    print(f"   Dim 2: [{latent_coords[:, 1].min():.3f}, {latent_coords[:, 1].max():.3f}]")
    print("\n   Try these settings:")
    print("   --scatter_size=50 --scatter_alpha=0.8")
