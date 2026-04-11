import torch
from models.sampling import *
from models.qmc_base import *
from models.layers import *
from train.losses import binary_lp, binary_evidence
import train.train as train_qmc
from torch.utils.data import DataLoader
import os
from torch.optim import Adam
from train.model_saving_loading import *
from plotting.visualize import *
from plotting.visualize_3d import model_grid_plot as model_grid_plot_3d
from data.mouse_data import load_mouse_data, mouse_data

import json
import random
import numpy as np
import fire
from tqdm import tqdm


def print_gpu_memory(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(
            f"[GPU {label}] allocated={allocated:.1f}MB  reserved={reserved:.1f}MB  peak={peak:.1f}MB  total={total:.1f}MB"
        )
    else:
        print(f"[GPU {label}] no CUDA device available")


def compute_val_diagnostics(model, val_dataset, base_sequence, lp_fnc, device, indices, diag_batch_size=32):
    """Compute reconstruction MSE for a fixed set of val indices.

    indices: pre-computed array of dataset indices (balanced across masks_len bins).
    Returns mse_arr (float32 numpy array of length len(indices)).
    """
    base_sequence = base_sequence.to(device)
    all_mse = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), diag_batch_size):
            batch_idx = indices[start : start + diag_batch_size]
            items = [val_dataset[i] for i in batch_idx]
            specs = torch.stack([it[0] for it in items]).to(torch.float32).to(device)
            recon = model.round_trip(base_sequence, specs, lp_fnc)
            mse = ((recon.cpu() - specs.cpu()) ** 2).mean(dim=(1, 2, 3)).numpy()
            all_mse.extend(mse.tolist())
    model.train()
    return np.array(all_mse, dtype=np.float32)


def _save_diagnostic_plots(
    save_location, qmc_losses, val_loss_epochs, val_losses,
    diag_epochs, diag_mse, val_diag_ml, val_diag_dur, dur_bin_edges,
):
    """Write all three diagnostic plots to save_location, overwriting any existing files."""
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
    plt.savefig(os.path.join(save_location, "qmc_train_stats.png"))
    plt.close()

    if not diag_epochs:
        return

    epochs_arr = np.array(diag_epochs)
    mse_mat = np.array(diag_mse)  # (n_checkpoints, n_diag_samples)

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
    plt.savefig(os.path.join(save_location, "qmc_val_mse_by_masks_len.png"))
    plt.close()

    # --- MSE by duration bin ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in range(len(dur_bin_edges) - 1):
        lo, hi = dur_bin_edges[b], dur_bin_edges[b + 1]
        in_bin = (val_diag_dur >= lo) & (val_diag_dur < hi)
        mean_mse = [mse_mat[t][in_bin].mean() if in_bin.any() else np.nan for t in range(len(epochs_arr))]
        ax.plot(epochs_arr, mean_mse, marker="o", markersize=3, label=f"dur [{lo:.0f}, {hi:.0f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean MSE")
    ax.set_title("Val MSE by duration bin across training")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, "qmc_val_mse_by_duration.png"))
    plt.close()


def _save_round_trip_panel(dataset, model, base_sequence, lp_fnc, device, save_path, n_per_mask=10, seed=42):
    """Save a single round-trip figure grouped by masks_len.

    Layout: each unique masks_len value occupies a pair of consecutive rows —
    row 2i = originals, row 2i+1 = reconstructions.  Columns = samples (up to
    n_per_mask per masks_len group).
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
            n_avail = len(bin_inds)
            chosen = rng.choice(bin_inds, size=min(n_per_mask, n_avail), replace=False)

            orig_row  = 2 * row_pair
            recon_row = 2 * row_pair + 1

            for col, idx in enumerate(chosen):
                spec = dataset[idx][0].to(torch.float32).to(device).unsqueeze(0)
                recon = model.round_trip(base_sequence, spec, lp_fnc).detach().cpu().squeeze()
                spec_cpu = spec.detach().cpu().squeeze()

                axs[orig_row,  col].imshow(spec_cpu.numpy(),  cmap="viridis", origin="lower", aspect="auto")
                axs[recon_row, col].imshow(recon.numpy(),     cmap="viridis", origin="lower", aspect="auto")

            for col in range(len(chosen), n_cols):
                axs[orig_row,  col].set_visible(False)
                axs[recon_row, col].set_visible(False)

            axs[orig_row,  0].set_ylabel(f"ml={int(ml)}\norig",  fontsize=7)
            axs[recon_row, 0].set_ylabel(f"ml={int(ml)}\nrecon", fontsize=7)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    model.train()


def run_mouse_experiments(
    save_location,
    dataloc,
    train_grid_m=15,
    test_grid_m=20,
    nEpochs=300,
    samples_per_mask=5000,
    train_batch_size=64,
    test_batch_size=1,
    print_gpu_mem=False,
    latent_dim=2,
    lattice_type="korobov",
    korobov_a=76,
    train_n_points=1021,
    test_n_points=2039,
    val_freq=10,
    test_samples_per_mask=50,
    n_dur_bins=5,
    seed=42,
    # dataset parameters
    filter_mask=True,
    lo=1,
    hi=8,
    sampling_strategy="mask_duration",
    total_samples=None,
    duration_aware=False,
):
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
    test_ds = mouse_data(val_dict, filter_mask=filter_mask, lo=lo, hi=hi, seed=seed)
    json.dump(train_ds.sampling_config,
              open(os.path.join(save_location, 'sampling_config.json'), 'w'), indent=2)
    n_workers = len(os.sched_getaffinity(0))
    print(
        f"Using train_batch_size={train_batch_size}, test_batch_size={test_batch_size}"
    )
    train_loader = DataLoader(
        train_ds, num_workers=n_workers, shuffle=True, batch_size=train_batch_size
    )
    test_loader = DataLoader(
        test_ds, num_workers=n_workers, shuffle=False, batch_size=test_batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print_gpu_memory("before model init")

    qmc_latent_dim = latent_dim
    qmc_loss_function = lambda samples, data: binary_evidence(samples, data)
    lp_fnc = lambda x, y: binary_lp(x, y)

    decoder_qmc = nn.Sequential(
        nn.Linear(2 * qmc_latent_dim, 2048),
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

    qmc_model = QMCLVM(
        latent_dim=qmc_latent_dim,
        device=device,
        decoder=decoder_qmc,
        basis=TorusBasis(),
    )
    print_gpu_memory("after model init")
    if lattice_type == "korobov":
        train_base_sequence = gen_korobov_basis(
            korobov_a, qmc_latent_dim, train_n_points
        )
        test_base_sequence = gen_korobov_basis(korobov_a, qmc_latent_dim, test_n_points)
    elif lattice_type == "roberts":
        train_base_sequence = roberts_sequence(train_n_points, qmc_latent_dim)
        test_base_sequence = roberts_sequence(test_n_points, qmc_latent_dim)
    else:  # 'fib', 2D only
        train_base_sequence = gen_fib_basis(m=train_grid_m)
        test_base_sequence = gen_fib_basis(m=test_grid_m)

    save_qmc  = os.path.join(save_location, "qmc_train_mouse_experiment.tar")
    save_diag = os.path.join(save_location, "qmc_val_diagnostics.npz")

    # --- pre-calculate fixed val diagnostic indices, balanced across masks_len bins ---
    val_ml_all  = test_loader.dataset.masks_len.numpy()
    val_dur_all = test_loader.dataset.durations.numpy().astype(np.float32)
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

    # small loader over the fixed diagnostic subset — used for val loss and MSE
    val_diag_loader = DataLoader(
        torch.utils.data.Subset(test_loader.dataset, val_diag_indices),
        num_workers=n_workers, shuffle=False, batch_size=test_batch_size,
    )

    # precompute duration bin edges from full val set (consistent across epochs)
    dur_bin_edges = np.percentile(val_dur_all, np.linspace(0, 100, n_dur_bins + 1))
    dur_bin_edges[0]  -= 1
    dur_bin_edges[-1] += 1

    if not os.path.isfile(save_qmc):
        print("now training qmc model")
        torch.cuda.reset_peak_memory_stats()

        qmc_opt    = Adam(qmc_model.parameters(), lr=1e-3)
        qmc_losses = []
        diag_epochs, diag_mse = [], []
        val_loss_epochs, val_losses = [], []

        for epoch in tqdm(range(nEpochs)):
            batch_loss, qmc_model, qmc_opt = train_qmc.train_epoch(
                qmc_model, qmc_opt, train_loader,
                train_base_sequence.to(device),
                qmc_loss_function,
            )
            qmc_losses += batch_loss

            if (epoch + 1) % val_freq == 0 or epoch == nEpochs - 1:
                val_batch_losses = train_qmc.test_epoch(
                    qmc_model, val_diag_loader,
                    test_base_sequence.to(device),
                    qmc_loss_function,
                )
                val_losses.append(float(np.mean(val_batch_losses)))
                val_loss_epochs.append(epoch + 1)

                mse_arr = compute_val_diagnostics(
                    qmc_model, test_loader.dataset,
                    test_base_sequence, lp_fnc, device,
                    val_diag_indices,
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
        save(qmc_model.to("cpu"), qmc_opt, qmc_losses, fn=save_qmc)
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

    qmc_losses = np.array(qmc_losses)
    ax = plt.gca()
    ax.plot(-qmc_losses)
    ax = format_plot_axis(
        ax,
        ylabel="log evidence",
        xlabel="update number",
        xticks=ax.get_xticks(),
        yticks=ax.get_yticks(),
    )
    plt.savefig(os.path.join(save_location, "qmc_train_stats.svg"))
    plt.close()

    _grid_plot_fn = model_grid_plot_3d if qmc_latent_dim == 3 else model_grid_plot
    _grid_fn = os.path.join(save_location, "qmc_grid") if qmc_latent_dim == 3 else os.path.join(save_location, "qmc_grid.png")
    _grid_plot_fn(
        qmc_model.to(device),
        n_samples_dim=20,
        show=False,
        fn=_grid_fn,
        origin="lower",
        cm="viridis",
    )

    _save_round_trip_panel(
        train_ds, qmc_model, test_base_sequence.to(device), lp_fnc, device,
        os.path.join(save_location, "qmc_round_trips_train.png"),
        n_per_mask=10, seed=seed,
    )
    _save_round_trip_panel(
        test_loader.dataset, qmc_model, test_base_sequence.to(device), lp_fnc, device,
        os.path.join(save_location, "qmc_round_trips_val.png"),
        n_per_mask=10, seed=seed,
    )


if __name__ == "__main__":
    fire.Fire(run_mouse_experiments)
