"""
QMC vs VAE comparison for mouse vocalization data.

Trains a comparable VAE (if not already saved) and computes:
  1. Held-out log-evidence  (QMC IS estimate vs VAE ELBO)
  2. MAP reconstruction BCE
  3. Jacobian norm map (QMC decoder coverage)
  4. Posterior sharpness histogram
  5. Latent space scatter side-by-side

Usage:
    python compare_qmc_vae_mouse.py \
        --qmc_model_path="path/to/qmc_checkpoint.tar" \
        --dataloc="path/to/mouse/data" \
        --save_dir="path/to/output" \
        --lattice_m=20 \
        --vae_epochs=300
"""

import json
import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import fire

from models.qmc_decoder import build_for_checkpoint
from models.qmc_base import QMCLVM, TorusBasis
from models.vae_base import VAE, Encoder
from models.sampling import gen_fib_basis
from train.model_saving_loading import load, save
from train.losses import binary_lp, binary_evidence, binary_elbo
from train.train_vae import train_loop as train_vae_loop
from data.mouse_data import load_mouse_data, mouse_data
from analysis.model_helpers import get_stacked_posterior
from analysis.jacobians import get_norms_lattice


# ── Architecture helpers ─────────────────────────────────────────────────────

def build_qmc_model(latent_dim, device, checkpoint):
    # Architecture comes from the checkpoint, not from a literal copied out of the
    # driver: models/qmc_decoder.build_for_checkpoint reads which head the weights
    # were trained with. This file used to carry its own copy of the Sequential,
    # one of seven, and every one of them had to be edited in lockstep.
    decoder, _head = build_for_checkpoint(checkpoint, latent_dim)
    return QMCLVM(latent_dim=latent_dim, device=device, decoder=decoder, basis=TorusBasis())


def build_vae_model(latent_dim, device):
    """
    VAE with same decoder capacity as QMC.
    latent_dim=8 matches 2D QMC per the paper.
    Uses LowRankMultivariateNormal directly, matching mnist.py / zebra_finch.py pattern.
    """
    from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal

    shared_net = nn.Sequential(
        nn.Conv2d(1, 8, 3, stride=2, padding=1),   # -> 64x64
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, stride=2, padding=1),  # -> 32x32
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> 16x16
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), # -> 8x8
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 2048),
        nn.ReLU(),
    )
    encoder = Encoder(
        net=shared_net,
        mu_net=nn.Linear(2048, latent_dim),
        l_net=nn.Linear(2048, latent_dim),
        d_net=nn.Linear(2048, latent_dim),
        latent_dim=latent_dim,
    )
    decoder = nn.Sequential(
        nn.Linear(latent_dim, 2048),
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
    # Pass LowRankMultivariateNormal directly, as in mnist.py / zebra_finch.py
    return VAE(decoder=decoder, encoder=encoder, distribution=LowRankMultivariateNormal, device=device)


# ── Metrics ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def qmc_test_metrics(model, lattice, loader, device):
    """
    Returns per-sample:
      - log_evidence: IS estimate of log p(x) over the lattice
      - map_bce:      BCE of the MAP reconstruction
      - posterior_entropy: entropy of the posterior distribution over lattice points
    """
    log_evidences, map_bces, posterior_entropies = [], [], []
    lattice_t = lattice.to(device).float()

    for batch in tqdm(loader, desc="QMC metrics"):
        data = batch[0].to(device).float()  # B x 1 x H x W
        B = data.shape[0]

        # Decode all lattice points once (S x 1 x H x W)
        with torch.no_grad():
            recons = model(lattice_t, mod=False, random=False)  # S x 1 x H x W

        # log p(x|z) for every (sample, lattice point): B x S
        lp = binary_lp(recons, data)  # B x S

        # 1. Log-evidence: log(1/S * sum_s exp(lp))
        log_ev = torch.logsumexp(lp, dim=1) - np.log(lp.shape[1])  # B
        log_evidences.append(log_ev.cpu().numpy())

        # 2. MAP reconstruction BCE
        map_idx = lp.argmax(dim=1)  # B
        map_recon = recons[map_idx]  # B x 1 x H x W
        bce = nn.functional.binary_cross_entropy(map_recon, data, reduction='none')
        map_bce = bce.sum(dim=(1, 2, 3))  # B
        map_bces.append(map_bce.cpu().numpy())

        # 3. Posterior entropy (sharpness)
        log_posterior = lp - torch.logsumexp(lp, dim=1, keepdim=True)
        posterior = log_posterior.exp()
        entropy = -(posterior * log_posterior.clamp(min=-1e8)).sum(dim=1)  # B
        posterior_entropies.append(entropy.cpu().numpy())

    return (
        np.concatenate(log_evidences),
        np.concatenate(map_bces),
        np.concatenate(posterior_entropies),
    )


@torch.no_grad()
def vae_test_metrics(model, loader, device, n_quad=50):
    """
    Returns per-sample:
      - elbo:     reconstruction_ll - KL  (lower bound on log p(x))
      - map_bce:  BCE of encoder-mean reconstruction
    """
    elbos, map_bces = [], []

    for batch in tqdm(loader, desc="VAE metrics"):
        data = batch[0].to(device).float().clamp(0, 1)
        recon, params = model(data)

        # ELBO
        neg_lp, kl = binary_elbo(recon, params, data)
        elbo_batch = -(neg_lp + kl)  # scalar; re-compute per-sample below

        # Per-sample ELBO
        mu, L, D = params
        L = L.squeeze(-1)
        z_dim = mu.shape[1]
        t12 = -0.5 * torch.log(D).sum(-1) - 0.5 * torch.log(1 + torch.einsum('bd,bd->b', L / D, L))
        t22 = 0.5 * (D.sum(-1) + (L ** 2).sum(-1))
        t32 = -z_dim / 2
        t42 = 0.5 * (mu ** 2).sum(-1)
        kl_per = t12 + t22 + t32 + t42

        neg_lp_per = nn.functional.binary_cross_entropy(
            recon, data, reduction='none'
        ).sum(dim=(1, 2, 3))

        elbo_per = -(neg_lp_per + kl_per)
        elbos.append(elbo_per.cpu().numpy())

        # MAP reconstruction (encoder mean)
        map_recon = model.round_trip(data)
        bce = nn.functional.binary_cross_entropy(map_recon, data, reduction='none')
        map_bces.append(bce.sum(dim=(1, 2, 3)).cpu().numpy())

    return np.concatenate(elbos), np.concatenate(map_bces)


# ── Jacobian norm heatmap ─────────────────────────────────────────────────────

def plot_jacobian_heatmap(model, lattice, save_path):
    """
    Compute Frobenius norm of the decoder Jacobian over the given lattice points.
    Uniform norms across the torus = full decoder coverage (no dead zones).
    lattice: already-subsampled tensor of shape (N, 2)
    """
    norms, log_norms = get_norms_lattice(model, lattice)

    coords = lattice.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc0 = axes[0].scatter(coords[:, 0], coords[:, 1], c=log_norms, cmap='inferno', s=5)
    axes[0].set_title('log Jacobian Frobenius norm\n(QMC decoder coverage)', fontsize=12)
    axes[0].set_xlabel('Latent dim 1')
    axes[0].set_ylabel('Latent dim 2')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    plt.colorbar(sc0, ax=axes[0], label='log ||J||_F')

    # Coefficient of variation as a single coverage quality number
    cv = norms.std() / (norms.mean() + 1e-8)
    axes[1].hist(log_norms, bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
    axes[1].set_title(f'log Jacobian norm distribution\nCV={cv:.3f} (lower = more uniform)', fontsize=12)
    axes[1].set_xlabel('log ||J||_F')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Jacobian CV: {cv:.4f}  (lower = more uniform decoder coverage)")
    return cv


# ── Latent scatter ────────────────────────────────────────────────────────────

@torch.no_grad()
def get_vae_latents(model, loader, device):
    mus, mask_counts = [], []
    for batch in tqdm(loader, desc="VAE latents"):
        data = batch[0].to(device).float()
        mu, _, _ = model.encode(data)
        mus.append(mu.cpu().numpy())
        # batch[1] = ml (masks_len scalar per sample), batch[2] = pixel mask
        ml = batch[1]
        mask_counts.append(ml.numpy() if torch.is_tensor(ml) else np.array(ml))
    return np.concatenate(mus), np.concatenate(mask_counts)


# ── VAE training for mouse data ───────────────────────────────────────────────

def _train_vae_mouse(model, optimizer, loader, device, n_epochs):
    """Training loop that handles mouse_data's multi-element batches."""
    recon_losses, kl_losses = [], []
    model.train()
    for epoch in tqdm(range(n_epochs), desc="VAE training"):
        for batch in loader:
            data = batch[0].to(device).float().clamp(0, 1)
            optimizer.zero_grad()
            recon, params = model(data)
            neg_lp, kl = binary_elbo(recon, params, data)
            loss = neg_lp + kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 200)
            optimizer.step()
            recon_losses.append(neg_lp.item())
            kl_losses.append(kl.item())
    return model, optimizer, [recon_losses, kl_losses]


# ── Main ──────────────────────────────────────────────────────────────────────

def compare_qmc_vae_mouse(
    qmc_model_path,
    dataloc,
    save_dir,
    lattice_m=20,
    vae_epochs=300,
    qmc_latent_dim=2,
    vae_latent_dim=32,       # 8D VAE matches 2D QMC per the paper
    total_samples=100000,     # None = use all; set e.g. 10000 for proportional subsample
    train_batch_size=512,
    test_batch_size=1,
    jac_samples=2000,
    use_train_data=False,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    train_dict, val_dict = load_mouse_data(dataloc)
    data_dict = train_dict if use_train_data else val_dict
    data_name = "TRAIN" if use_train_data else "TEST"

    train_ds = mouse_data(train_dict, filter_mask=True, lo=1, hi=8,
                          sampling_strategy='subsample', total_samples=total_samples, seed=seed)
    test_ds  = mouse_data(data_dict,  filter_mask=True, lo=1, hi=8, seed=seed)
    json.dump(train_ds.sampling_config,
              open(os.path.join(save_dir, 'sampling_config.json'), 'w'), indent=2)
    print(f"Train: {len(train_ds)}  |  {data_name}: {len(test_ds)}")
    
    # Diagnose spectrogram range — binary loss requires data in [0, 1]
    specs = train_ds.spectrograms
    print(f"Spectrogram range: min={specs.min():.4f}  max={specs.max():.4f}")
    if specs.min() < 0 or specs.max() > 1:
        print("WARNING: spectrograms are outside [0,1] — binary loss (BCE) will fail for VAE.")
        print("         Consider normalizing in your data preprocessing, or switching to Gaussian loss.")

    n_workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4
    train_loader = DataLoader(train_ds, num_workers=n_workers, shuffle=True,  batch_size=train_batch_size)
    test_loader  = DataLoader(test_ds,  num_workers=n_workers, shuffle=False, batch_size=test_batch_size)

    print(f"QMC latent_dim={qmc_latent_dim}  |  VAE latent_dim={vae_latent_dim} (paper: 8D VAE ~ 2D QMC)")

    # ── QMC model ─────────────────────────────────────────────────────────────
    print("\nLoading QMC model...")
    qmc_model = build_qmc_model(qmc_latent_dim, device, checkpoint=qmc_model_path)
    qmc_opt = Adam(qmc_model.parameters(), lr=1e-3)
    qmc_model, qmc_opt, _ = load(qmc_model, qmc_opt, qmc_model_path)
    qmc_model.to(device).eval()

    lattice = gen_fib_basis(m=lattice_m)
    print(f"Lattice: {len(lattice)} points")

    # ── VAE model ─────────────────────────────────────────────────────────────
    vae_save = os.path.join(save_dir, 'vae_mouse_comparison.tar')
    vae_model = build_vae_model(vae_latent_dim, device)
    vae_opt = Adam(vae_model.parameters(), lr=1e-3)

    if not os.path.isfile(vae_save):
        print(f"\nTraining VAE for {vae_epochs} epochs...")
        vae_model, vae_opt, vae_losses = _train_vae_mouse(
            vae_model, vae_opt, train_loader, device, vae_epochs
        )
        save(vae_model.to('cpu'), vae_opt, vae_losses, fn=vae_save)
        vae_model.to(device)
        print(f"VAE saved to {vae_save}")

        all_losses = vae_losses[0]  # recon losses
        plt.figure()
        plt.plot(all_losses)
        plt.xlabel('Batch step')
        plt.ylabel('Reconstruction loss + KL')
        plt.title('VAE training loss')
        plt.savefig(os.path.join(save_dir, 'vae_train_loss.png'), dpi=150)
        plt.close()
    else:
        vae_model, vae_opt, _ = load(vae_model, vae_opt, vae_save)
        print(f"VAE loaded from {vae_save}")

    vae_model.to(device).eval()

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\n── Computing QMC metrics ──")
    qmc_log_ev, qmc_map_bce, qmc_entropy = qmc_test_metrics(qmc_model, lattice, test_loader, device)

    print("\n── Computing VAE metrics ──")
    vae_elbo, vae_map_bce = vae_test_metrics(vae_model, test_loader, device)

    # Summary statistics
    print("\n" + "=" * 60)
    print(f"{'Metric':<35} {'QMC':>10} {'VAE':>10}")
    print("-" * 60)
    print(f"{'Mean log-evidence / ELBO':<35} {qmc_log_ev.mean():>10.2f} {vae_elbo.mean():>10.2f}")
    print(f"{'Median log-evidence / ELBO':<35} {np.median(qmc_log_ev):>10.2f} {np.median(vae_elbo):>10.2f}")
    print(f"{'Mean MAP reconstruction BCE':<35} {qmc_map_bce.mean():>10.2f} {vae_map_bce.mean():>10.2f}")
    print(f"{'Mean posterior entropy (QMC)':<35} {qmc_entropy.mean():>10.3f} {'N/A':>10}")
    print("=" * 60)
    print(f"\nNOTE: QMC metric = IS estimate of log p(x)  (tighter with more lattice points)")
    print(f"      VAE metric = ELBO  (lower bound; typically < log p(x))")
    print(f"      Lower MAP BCE = better reconstruction quality")

    np.savez(
        os.path.join(save_dir, 'metrics.npz'),
        qmc_log_ev=qmc_log_ev, qmc_map_bce=qmc_map_bce, qmc_entropy=qmc_entropy,
        vae_elbo=vae_elbo, vae_map_bce=vae_map_bce,
    )

    # ── Figure 1: metric distributions ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1a. Log-evidence / ELBO
    ax = axes[0]
    common_kw = dict(bins=60, alpha=0.7, density=True, edgecolor='none')
    all_vals = np.concatenate([qmc_log_ev, vae_elbo])
    bins = np.linspace(np.percentile(all_vals, 2), np.percentile(all_vals, 98), 60)
    ax.hist(qmc_log_ev, bins=bins, color='steelblue', label=f'QMC IS  μ={qmc_log_ev.mean():.1f}', **{k: v for k, v in common_kw.items() if k != 'bins'})
    ax.hist(vae_elbo,   bins=bins, color='salmon',    label=f'VAE ELBO μ={vae_elbo.mean():.1f}',  **{k: v for k, v in common_kw.items() if k != 'bins'})
    ax.axvline(qmc_log_ev.mean(), color='steelblue', linestyle='--', linewidth=1.5)
    ax.axvline(vae_elbo.mean(),   color='salmon',    linestyle='--', linewidth=1.5)
    ax.set_xlabel('log p(x) estimate', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Held-out log-evidence\n(higher = better)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # 1b. MAP reconstruction BCE
    ax = axes[1]
    all_bce = np.concatenate([qmc_map_bce, vae_map_bce])
    bins_bce = np.linspace(np.percentile(all_bce, 1), np.percentile(all_bce, 99), 60)
    ax.hist(qmc_map_bce, bins=bins_bce, color='steelblue', label=f'QMC  μ={qmc_map_bce.mean():.0f}', **{k: v for k, v in common_kw.items() if k != 'bins'})
    ax.hist(vae_map_bce, bins=bins_bce, color='salmon',    label=f'VAE  μ={vae_map_bce.mean():.0f}',  **{k: v for k, v in common_kw.items() if k != 'bins'})
    ax.axvline(qmc_map_bce.mean(), color='steelblue', linestyle='--', linewidth=1.5)
    ax.axvline(vae_map_bce.mean(), color='salmon',    linestyle='--', linewidth=1.5)
    ax.set_xlabel('BCE (summed over pixels)', fontsize=11)
    ax.set_title('MAP reconstruction quality\n(lower = better)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # 1c. QMC posterior entropy (sharpness)
    ax = axes[2]
    ax.hist(qmc_entropy, bins=50, color='steelblue', alpha=0.8, edgecolor='none')
    ax.axvline(qmc_entropy.mean(), color='navy', linestyle='--', linewidth=1.5,
               label=f'Mean={qmc_entropy.mean():.2f}')
    max_entropy = np.log(len(lattice))
    ax.axvline(max_entropy, color='red', linestyle=':', linewidth=1.5,
               label=f'Max possible={max_entropy:.1f}')
    ax.set_xlabel('Posterior entropy H[p(z|x)]', fontsize=11)
    ax.set_title('QMC posterior sharpness\n(lower = more confident assignments)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    plt.suptitle(f'QMC vs VAE — Mouse vocalization ({data_name} set, N={len(test_ds)})',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'comparison_metrics.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'comparison_metrics.svg'), bbox_inches='tight')
    plt.close()
    print("\nSaved: comparison_metrics.png/svg")

    # ── Figure 2: latent spaces side by side ─────────────────────────────────
    print("\nComputing VAE latents for scatter plot...")
    vae_mus, mask_counts = get_vae_latents(vae_model, test_loader, device)
    # mask_counts = masks_len scalar per sample (batch[1] from mouse_data)

    # QMC latents (MAP)
    posteriors = get_stacked_posterior(qmc_model, lattice, test_loader, binary_lp)
    map_idx = np.argmax(posteriors, axis=1)
    qmc_coords = (lattice[map_idx].numpy()) % 1.0

    # For VAE: project 8D latent to 2D via PCA for visualization
    from sklearn.decomposition import PCA
    vae_pca = PCA(n_components=2)
    vae_2d = vae_pca.fit_transform(vae_mus)
    vae_var_explained = vae_pca.explained_variance_ratio_.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    scatter_kw = dict(s=2, alpha=0.25, rasterized=True, cmap='plasma')

    sc0 = axes[0].scatter(qmc_coords[:, 0], qmc_coords[:, 1], c=mask_counts, **scatter_kw)
    axes[0].set_title(f'QMC — MAP latents\n(2D torus [0,1]²)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Latent dim 1')
    axes[0].set_ylabel('Latent dim 2')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_aspect('equal')
    plt.colorbar(sc0, ax=axes[0], label='Syllable length (time bins)')

    sc1 = axes[1].scatter(vae_2d[:, 0], vae_2d[:, 1], c=mask_counts, **scatter_kw)
    axes[1].set_title(
        f'VAE — encoder mean μ (PCA to 2D)\n({vae_latent_dim}D Gaussian, {vae_var_explained:.0f}% var explained)',
        fontsize=12, fontweight='bold'
    )
    axes[1].set_xlabel('PC 1')
    axes[1].set_ylabel('PC 2')
    axes[1].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[1], label='Syllable length (time bins)')

    plt.suptitle(f'Latent representations — colored by syllable length\n({data_name} set)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'latent_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: latent_comparison.png")

    # ── Figure 3: Jacobian norm map (QMC only) ────────────────────────────────
    print(f"\nComputing Jacobian norms on {jac_samples} lattice points...")
    jac_idx = np.random.choice(len(lattice), min(jac_samples, len(lattice)), replace=False)
    jac_lattice = lattice[jac_idx]
    jac_cv = plot_jacobian_heatmap(
        qmc_model, jac_lattice,
        save_path=os.path.join(save_dir, 'qmc_jacobian_coverage.png'),
    )
    print("Saved: qmc_jacobian_coverage.png")

    # ── Figure 4: Example round-trips ─────────────────────────────────────────
    print("\nGenerating round-trip comparison panel...")
    n_examples = 6
    sample_indices = np.random.choice(len(test_ds), n_examples, replace=False)
    lattice_t = lattice.to(device).float()

    fig, axes = plt.subplots(n_examples, 3, figsize=(9, n_examples * 2))
    axes[0, 0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('QMC MAP recon', fontsize=11, fontweight='bold')
    axes[0, 2].set_title('VAE μ recon', fontsize=11, fontweight='bold')

    with torch.no_grad():
        all_recons_qmc = qmc_model(lattice_t, mod=False, random=False)  # S x 1 x H x W

    for row, idx in enumerate(sample_indices):
        data_item = test_ds[idx]
        spec = data_item[0].float().unsqueeze(0).to(device)  # 1 x 1 x H x W

        # QMC MAP
        lp = binary_lp(all_recons_qmc, spec)  # 1 x S
        best_idx = lp.argmax().item()
        qmc_recon = all_recons_qmc[best_idx].cpu().squeeze().numpy()

        # VAE MAP (encoder mean)
        with torch.no_grad():
            vae_recon = vae_model.round_trip(spec).cpu().squeeze().numpy()

        orig = spec.cpu().squeeze().numpy()
        for col, img in enumerate([orig, qmc_recon, vae_recon]):
            axes[row, col].imshow(img, cmap='viridis', origin='lower', aspect='auto')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'round_trips_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: round_trips_comparison.png")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"QMC ({qmc_latent_dim}D torus) mean log-evidence: {qmc_log_ev.mean():.2f}  (IS estimate, improves with larger lattice)")
    print(f"VAE ({vae_latent_dim}D Gaussian) mean ELBO:     {vae_elbo.mean():.2f}  (lower bound on log p(x))")
    delta = qmc_log_ev.mean() - vae_elbo.mean()
    if delta > 0:
        print(f"\nQMC is {delta:.2f} nats better in terms of log-likelihood accounting.")
    else:
        print(f"\nVAE ELBO is {-delta:.2f} nats higher — but note ELBO < log p(x), so gap may be reversed.")
    print(f"\nQMC MAP BCE: {qmc_map_bce.mean():.1f}  |  VAE MAP BCE: {vae_map_bce.mean():.1f}")
    print(f"Jacobian CV: {jac_cv:.4f}  (lower = more uniform decoder coverage)")
    print(f"\nAll outputs saved to: {save_dir}")


if __name__ == '__main__':
    fire.Fire(compare_qmc_vae_mouse)
