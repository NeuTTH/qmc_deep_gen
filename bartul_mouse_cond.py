import torch
import torch.nn.functional as F
from models.sampling import *
from models.qmc_base import *
from models.layers import *
from train.losses import binary_lp, binary_evidence
import train.train as train_qmc
from torch.utils.data import DataLoader
import os
from torch.optim import Adam
from train.model_saving_loading import *
from plotting.visualize import format_plot_axis, conditional_qmc_grid_plot
from data.mouse_data import load_mouse_data, mouse_data

import matplotlib.pyplot as plt
import json
import random
import numpy as np
import fire
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Conditional variable registry
#
# Maps CLI name → {field_idx into __getitem__ tuple, c_dim, label}
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
#
# To add a new conditional:
#   1. Add precomputed field to mouse_data.__init__ and update __getitem__
#   2. Add an entry here (field_idx, c_dim, label)
# ---------------------------------------------------------------------------
CONDITIONAL_REGISTRY = {
    "mask_count": {"field_idx": 5, "c_dim": 8, "label": "masks_len (one-hot)"},
    "duration":   {"field_idx": 3, "c_dim": 1, "label": "normalized duration"},
    "mean_freq":  {"field_idx": 4, "c_dim": 1, "label": "mean frequency"},
}


def make_collate_fn(cond_names):
    """Return a collate fn that produces (specs, c, masks, spec_ids).

    c is the batch-mean of selected conditional(s), shape (1, total_c_dim).
    cond_names: list of keys from CONDITIONAL_REGISTRY (currently only one).
    """
    cfg = [CONDITIONAL_REGISTRY[n] for n in cond_names]

    def collate(batch):
        specs = torch.stack([b[0] for b in batch])
        c_parts = []
        for entry in cfg:
            fi = entry["field_idx"]
            vals = torch.stack([
                b[fi].float() if b[fi].dim() > 0 else b[fi].float().unsqueeze(0)
                for b in batch
            ])                                            # (B, c_dim) or (B, 1)
            if vals.dim() == 1:
                vals = vals.unsqueeze(1)                  # ensure 2-D
            c_parts.append(vals.mean(dim=0, keepdim=True))  # (1, c_dim)
        c = torch.cat(c_parts, dim=-1)                   # (1, total_c_dim)
        masks    = torch.stack([b[6] for b in batch])
        spec_ids = [b[7] for b in batch]
        return (specs, c, masks, spec_ids)

    return collate


def get_grid_sweep(cond_name):
    """Return list of (label_str, c_tensor shape (1, c_dim)) for grid plots."""
    if cond_name == "mask_count":
        return [
            (f"ml{i + 1}", F.one_hot(torch.tensor([i]), num_classes=8).float())
            for i in range(8)
        ]
    elif cond_name == "duration":
        vals = [0.1, 0.3, 0.5, 0.7, 0.9]
        return [(f"dur{v:.1f}", torch.tensor([[v]])) for v in vals]
    elif cond_name == "mean_freq":
        vals = [0.1, 0.3, 0.5, 0.7, 0.9]
        return [(f"mf{v:.1f}", torch.tensor([[v]])) for v in vals]
    else:
        raise ValueError(f"Unknown conditional: {cond_name!r}")


def _get_sample_c(dataset, index, cond_name):
    """Extract the conditioning tensor for a single sample → (1, c_dim)."""
    fi = CONDITIONAL_REGISTRY[cond_name]["field_idx"]
    item = dataset[index]
    val = item[fi].float()
    if val.dim() == 0:
        val = val.unsqueeze(0)   # scalar → (1,)
    return val.unsqueeze(0)      # (1, c_dim)


def _save_round_trip_panel(dataset, model, base_sequence, lp_fnc, device,
                            cond_name, save_path, n_per_mask=10, seed=42):
    """Save a round-trip panel grouped by masks_len.

    Layout: each unique masks_len value occupies a pair of consecutive rows —
    row 2i = originals, row 2i+1 = reconstructions. Columns = samples.
    """
    rng = np.random.default_rng(seed)
    ml_all = dataset.masks_len.numpy()
    unique_mls = np.unique(ml_all)

    n_rows = 2 * len(unique_mls)
    n_cols = n_per_mask

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    if n_rows == 1:
        axs = axs[np.newaxis, :]
    if n_cols == 1:
        axs = axs[:, np.newaxis]

    model.eval()
    with torch.no_grad():
        for row_pair, ml in enumerate(unique_mls):
            bin_inds = np.where(ml_all == ml)[0]
            chosen = rng.choice(bin_inds, size=min(n_per_mask, len(bin_inds)), replace=False)

            orig_row  = 2 * row_pair
            recon_row = 2 * row_pair + 1

            for col, idx in enumerate(chosen):
                spec = dataset[idx][0].to(torch.float32).to(device).unsqueeze(0)
                c = _get_sample_c(dataset, idx, cond_name).to(device)
                recon = model.round_trip(base_sequence, spec, lp_fnc, c=c).detach().cpu().squeeze()
                spec_cpu = spec.detach().cpu().squeeze()

                axs[orig_row,  col].imshow(spec_cpu.numpy(),  cmap="viridis", origin="lower", aspect="auto")
                axs[recon_row, col].imshow(recon.numpy(),      cmap="viridis", origin="lower", aspect="auto")

            for col in range(len(chosen), n_cols):
                axs[orig_row,  col].set_visible(False)
                axs[recon_row, col].set_visible(False)

            axs[orig_row,  0].set_ylabel(f"ml={int(ml)}\norig",  fontsize=7)
            axs[recon_row, 0].set_ylabel(f"ml={int(ml)}\nrecon", fontsize=7)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    cond_label = CONDITIONAL_REGISTRY[cond_name]["label"]
    plt.suptitle(f"Round trips — conditional: {cond_label}", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    model.train()


def compute_val_diagnostics(model, val_dataset, base_sequence, lp_fnc, device,
                             indices, cond_name, diag_batch_size=32):
    """Compute per-sample reconstruction MSE over a fixed set of val indices.

    Uses batch-mean c within each diagnostic batch (consistent with training).
    Returns float32 numpy array of length len(indices).
    """
    fi = CONDITIONAL_REGISTRY[cond_name]["field_idx"]
    base_sequence = base_sequence.to(device)
    all_mse = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), diag_batch_size):
            batch_idx = indices[start : start + diag_batch_size]
            items = [val_dataset[i] for i in batch_idx]
            specs = torch.stack([it[0] for it in items]).to(torch.float32).to(device)
            c_vals = torch.stack([
                it[fi].float() if it[fi].dim() > 0 else it[fi].float().unsqueeze(0)
                for it in items
            ])
            if c_vals.dim() == 1:
                c_vals = c_vals.unsqueeze(1)
            c = c_vals.mean(dim=0, keepdim=True).to(device)   # (1, c_dim)
            recon = model.round_trip(base_sequence, specs, lp_fnc, c=c)
            mse = ((recon.cpu() - specs.cpu()) ** 2).mean(dim=(1, 2, 3)).numpy()
            all_mse.extend(mse.tolist())
    model.train()
    return np.array(all_mse, dtype=np.float32)


def _save_diagnostic_plots(
    save_location, qmc_losses, val_loss_epochs, val_losses,
    diag_epochs, diag_mse, val_diag_ml, val_diag_dur, dur_bin_edges,
):
    """Write all three diagnostic plots to save_location."""
    # --- loss plot ---
    fig, ax = plt.subplots()
    ax.plot(-np.array(qmc_losses), label="train", alpha=0.8, color="tab:blue")
    if val_losses:
        n_batches = len(qmc_losses) // val_loss_epochs[-1]
        val_x = np.array(val_loss_epochs) * n_batches
        ax.plot(val_x, val_losses, marker="o", markersize=4, label="val", color="tab:orange")
    ax.set_xlabel("update number")
    ax.set_ylabel("log evidence")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, "qmc_cond_train_stats.png"))
    plt.close()

    if not diag_epochs:
        return

    epochs_arr = np.array(diag_epochs)
    mse_mat    = np.array(diag_mse)   # (n_checkpoints, n_diag_samples)

    # --- MSE by masks_len ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for ml in sorted(np.unique(val_diag_ml)):
        mask = val_diag_ml == ml
        mean_mse = [mse_mat[t][mask].mean() if mask.any() else np.nan for t in range(len(epochs_arr))]
        ax.plot(epochs_arr, mean_mse, marker="o", markersize=3, label=f"masks_len={int(ml)}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean MSE")
    ax.set_title("Val MSE by masks_len across training")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, "qmc_cond_val_mse_by_masks_len.png"))
    plt.close()

    # --- MSE by duration bin ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in range(len(dur_bin_edges) - 1):
        lo, hi = dur_bin_edges[b], dur_bin_edges[b + 1]
        in_bin  = (val_diag_dur >= lo) & (val_diag_dur < hi)
        mean_mse = [mse_mat[t][in_bin].mean() if in_bin.any() else np.nan for t in range(len(epochs_arr))]
        ax.plot(epochs_arr, mean_mse, marker="o", markersize=3, label=f"dur [{lo:.0f}, {hi:.0f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean MSE")
    ax.set_title("Val MSE by duration bin across training")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, "qmc_cond_val_mse_by_duration.png"))
    plt.close()


def print_gpu_memory(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved  = torch.cuda.memory_reserved()  / 1024**2
        peak      = torch.cuda.max_memory_allocated() / 1024**2
        total     = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"[GPU {label}] allocated={allocated:.1f}MB  reserved={reserved:.1f}MB  peak={peak:.1f}MB  total={total:.1f}MB")
    else:
        print(f"[GPU {label}] no CUDA device available")


def run_mouse_cond_experiments(
    save_location,
    dataloc,
    conditional="mask_count",   # one of: "mask_count", "duration", "mean_freq"
    train_grid_m=15,
    test_grid_m=20,
    nEpochs=300,
    train_batch_size=64,
    test_batch_size=1,
    print_gpu_mem=False,
    seed=42,
    # validation parameters
    val_freq=10,
    test_samples_per_mask=50,
    n_dur_bins=5,
    # dataset parameters
    filter_mask=True,
    lo=1,
    hi=8,
    sampling_strategy="mask_duration",
    samples_per_mask=5000,
    total_samples=None,
    duration_aware=False,
):
    if conditional not in CONDITIONAL_REGISTRY:
        raise ValueError(
            f"Unknown conditional: {conditional!r}. "
            f"Choose from: {list(CONDITIONAL_REGISTRY)}"
        )
    cond_names = [conditional]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not os.path.exists(save_location):
        print(f"Creating save directory: {save_location}")
        os.makedirs(save_location)

    train_dict, val_dict = load_mouse_data(dataloc)
    train_ds = mouse_data(
        train_dict,
        filter_mask=filter_mask, lo=lo, hi=hi,
        sampling_strategy=sampling_strategy,
        samples_per_mask=samples_per_mask,
        total_samples=total_samples,
        duration_aware=duration_aware,
        seed=seed,
    )
    test_ds  = mouse_data(val_dict, filter_mask=filter_mask, lo=lo, hi=hi, seed=seed)
    json.dump(train_ds.sampling_config,
              open(os.path.join(save_location, 'sampling_config.json'), 'w'), indent=2)

    collate_fn = make_collate_fn(cond_names)
    n_workers = len(os.sched_getaffinity(0))
    cond_label = CONDITIONAL_REGISTRY[conditional]["label"]
    c_dim = sum(CONDITIONAL_REGISTRY[n]["c_dim"] for n in cond_names)
    print(f"Conditional: {conditional!r}  ({cond_label})")
    print(f"c_dim={c_dim}, decoder input dim={2 * 2 + c_dim}")  # latent_dim=2 → 2*2=4 after TorusBasis
    print(f"Using train_batch_size={train_batch_size}, test_batch_size={test_batch_size}")

    train_loader = DataLoader(train_ds, num_workers=n_workers, shuffle=True,
                              batch_size=train_batch_size, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print_gpu_memory("before model init")

    qmc_latent_dim = 2
    qmc_loss_function = lambda samples, data: binary_evidence(samples, data)
    lp_fnc = lambda x, y: binary_lp(x, y)

    decoder_qmc = nn.Sequential(
        nn.Linear(2 * qmc_latent_dim + c_dim, 2048),
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

    qmc_model = QMCLVM(latent_dim=qmc_latent_dim, device=device, decoder=decoder_qmc, basis=TorusBasis())
    print_gpu_memory("after model init")
    train_base_sequence = gen_fib_basis(m=train_grid_m)
    test_base_sequence  = gen_fib_basis(m=test_grid_m)

    # --- Pre-compute fixed val diagnostic indices, balanced across masks_len bins ---
    val_ml_all  = test_ds.masks_len.numpy()
    val_dur_all = test_ds.durations.numpy().astype(np.float32)
    val_diag_indices = []
    for ml in np.unique(val_ml_all):
        bin_inds = np.where(val_ml_all == ml)[0]
        chosen = bin_inds if len(bin_inds) <= test_samples_per_mask \
                 else np.random.choice(bin_inds, test_samples_per_mask, replace=False)
        val_diag_indices.extend(chosen.tolist())
    val_diag_indices = np.array(sorted(val_diag_indices))
    val_diag_ml  = val_ml_all[val_diag_indices].astype(np.float32)
    val_diag_dur = val_dur_all[val_diag_indices]
    print(f"Val diagnostic set: {len(val_diag_indices)} samples "
          f"({test_samples_per_mask} per masks_len bin)")

    val_diag_loader = DataLoader(
        torch.utils.data.Subset(test_ds, val_diag_indices),
        num_workers=n_workers, shuffle=False, batch_size=test_batch_size,
        collate_fn=collate_fn,
    )

    # Duration bin edges computed from full val set (consistent across epochs)
    dur_bin_edges = np.percentile(val_dur_all, np.linspace(0, 100, n_dur_bins + 1))
    dur_bin_edges[0]  -= 1
    dur_bin_edges[-1] += 1

    save_qmc  = os.path.join(save_location, 'qmc_train_mouse_cond_experiment.tar')
    save_diag = os.path.join(save_location, 'qmc_cond_val_diagnostics.npz')

    if not os.path.isfile(save_qmc):
        print("now training conditional qmc model")
        torch.cuda.reset_peak_memory_stats()

        qmc_opt    = Adam(qmc_model.parameters(), lr=1e-3)
        qmc_losses = []
        diag_epochs, diag_mse         = [], []
        val_loss_epochs, val_losses   = [], []

        for epoch in tqdm(range(nEpochs)):
            batch_loss, qmc_model, qmc_opt = train_qmc.train_epoch(
                qmc_model, qmc_opt, train_loader,
                train_base_sequence.to(device),
                qmc_loss_function, conditional=True,
            )
            qmc_losses += batch_loss

            if (epoch + 1) % val_freq == 0 or epoch == nEpochs - 1:
                val_batch_losses = train_qmc.test_epoch(
                    qmc_model, val_diag_loader,
                    test_base_sequence.to(device),
                    qmc_loss_function, conditional=True,
                )
                val_losses.append(float(np.mean(val_batch_losses)))
                val_loss_epochs.append(epoch + 1)

                mse_arr = compute_val_diagnostics(
                    qmc_model, test_ds,
                    test_base_sequence, lp_fnc, device,
                    val_diag_indices, conditional,
                )
                diag_epochs.append(epoch + 1)
                diag_mse.append(mse_arr)

                _save_diagnostic_plots(
                    save_location, qmc_losses, val_loss_epochs, val_losses,
                    diag_epochs, diag_mse, val_diag_ml, val_diag_dur, dur_bin_edges,
                )

            if print_gpu_mem and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved  = torch.cuda.memory_reserved()  / 1024**2
                peak      = torch.cuda.max_memory_allocated() / 1024**2
                print(f"  [GPU epoch {epoch+1}] allocated={allocated:.1f}MB  reserved={reserved:.1f}MB  peak={peak:.1f}MB")

        print_gpu_memory("after training")
        save(qmc_model.to('cpu'), qmc_opt, qmc_losses, fn=save_qmc)
        qmc_model.to(device)
        np.savez(
            save_diag,
            epochs=np.array(diag_epochs),
            mse=np.array(diag_mse),
            val_diag_ml=val_diag_ml,
            val_diag_dur=val_diag_dur,
            dur_bin_edges=dur_bin_edges,
            val_loss_epochs=np.array(val_loss_epochs),
            val_losses=np.array(val_losses),
        )
    else:
        qmc_opt = Adam(qmc_model.parameters(), lr=1e-3)
        qmc_model, qmc_opt, qmc_losses = load(qmc_model, qmc_opt, save_qmc)
        print_gpu_memory("after model load")

    # --- Training loss plot ---
    qmc_losses = np.array(qmc_losses)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(-qmc_losses, color='tab:blue', alpha=0.8)
    ax = format_plot_axis(ax, ylabel='log evidence', xlabel='update number',
                          xticks=ax.get_xticks(), yticks=ax.get_yticks())
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, 'qmc_cond_train_stats.svg'))
    plt.close()

    qmc_model = qmc_model.to(device)

    # --- Grid plots: sweep over representative conditioning values ---
    sweep = get_grid_sweep(conditional)
    for val_label, c_val in sweep:
        conditional_qmc_grid_plot(
            qmc_model, n_samples_dim=20, c=c_val.to(device), show=False,
            fn=os.path.join(save_location, f'qmc_cond_grid_{conditional}_{val_label}.png'),
            title=f"{cond_label} = {val_label}",
            origin='lower', cm='viridis',
        )

    # --- Round-trip panels grouped by masks_len ---
    _save_round_trip_panel(
        train_ds, qmc_model, test_base_sequence.to(device), lp_fnc, device,
        conditional,
        os.path.join(save_location, 'qmc_cond_round_trips_train.png'),
        n_per_mask=10, seed=seed,
    )
    _save_round_trip_panel(
        test_ds, qmc_model, test_base_sequence.to(device), lp_fnc, device,
        conditional,
        os.path.join(save_location, 'qmc_cond_round_trips_val.png'),
        n_per_mask=10, seed=seed,
    )


if __name__ == '__main__':
    fire.Fire(run_mouse_cond_experiments)
