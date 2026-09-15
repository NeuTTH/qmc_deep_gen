"""Train a QMCLVM on mouse vocalization spectrograms.

This module is a ``fire`` CLI driver. Its single entry point,
:func:`run_mouse_experiments`, loads mouse USV spectrograms, builds a
QMC latent-variable model (a fixed quasi-random lattice in latent space
plus a learned ConvTranspose decoder), trains the decoder, and renders a
suite of diagnostic plots / round-trip panels into ``save_location``.

The lattice is fixed (not learned): each forward pass adds a random
torus shift ``r ~ U[0,1]^d`` to the whole lattice. The 2D latent
coordinates are expanded to ``(cos, sin)`` pairs by ``TorusBasis`` before
the decoder, which is why the decoder's input width is ``2 * latent_dim``.

Behavior is gated on the checkpoint file
``<save_location>/qmc_train_mouse_experiment.tar``: if it is absent the
model is trained (and the checkpoint + ``.npz`` diagnostics are written);
if it is present the model is loaded and only the final plots are
regenerated.

Usage (fire CLI)::

    # train (or reload) and render plots
    python bartul_mouse.py <save_location> <dataloc> \
        --nEpochs=300 --latent_dim=2 --lattice_type=korobov --korobov_a=76

    # 3D latent space
    python bartul_mouse.py <save_location> <dataloc> --latent_dim=3

``save_location`` and ``dataloc`` are required positional arguments; every
other argument is an optional ``--flag=value`` (note the camelCase
``--nEpochs`` and the ``--korobov_a`` spellings). See
:func:`run_mouse_experiments` for the full parameter list.

Star-import provenance (names used in this module):
    * ``nn``, ``QMCLVM``, ``TorusBasis``            -> models.qmc_base
    * ``gen_korobov_basis``, ``roberts_sequence``,
      ``gen_fib_basis``                             -> models.sampling
    * ``plt``                                       -> plotting.visualize
The original wildcard imports have been made explicit below.
"""

# --- standard library ---
import copy
import json
import os
import random

# --- third-party ---
import fire
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- local: models / training ---
from models.qmc_base import QMCLVM, TorusBasis
from models.qmc_decoder import build_qmc_decoder, build_for_checkpoint
from models.sampling import gen_fib_basis, gen_korobov_basis, roberts_sequence
from train.losses import binary_lp, binary_evidence
from train.model_saving_loading import load, save
import train.train as train_qmc

# --- local: data / plotting ---
from data.mouse_data import load_mouse_data, mouse_data
from plotting.visualize import plt
from plotting.figstyle import TRAIN_COLOR, VAL_COLOR


# ---------------------------------------------------------------------------
# GPU memory reporting
# ---------------------------------------------------------------------------
def print_gpu_memory(label="", compact=False):
    """Print current CUDA memory usage (allocated / reserved / peak).

    Single source of truth for the two GPU-memory print formats used in
    this module:

    * ``compact=False`` (default) -- full milestone report including the
      device total, e.g. ``[GPU <label>] allocated=...MB ... total=...MB``.
      Used at named milestones (model init, after training, after load).
    * ``compact=True`` -- the indented per-epoch line (two leading spaces,
      no device total), e.g. ``  [GPU <label>] allocated=...MB ... peak=...MB``.

    Both branches are reproduced byte-for-byte from the original inline
    prints. When CUDA is unavailable, the full format prints a "no CUDA
    device available" message; the compact (per-epoch) caller is guarded by
    its own ``torch.cuda.is_available()`` check and is never reached without
    a device.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        if compact:
            print(f"  [GPU {label}] allocated={allocated:.1f}MB  reserved={reserved:.1f}MB  peak={peak:.1f}MB")
        else:
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print(
                f"[GPU {label}] allocated={allocated:.1f}MB  reserved={reserved:.1f}MB  peak={peak:.1f}MB  total={total:.1f}MB"
            )
    else:
        print(f"[GPU {label}] no CUDA device available")


# ---------------------------------------------------------------------------
# Validation diagnostics: reconstruction MSE over a fixed val subset
# ---------------------------------------------------------------------------
def compute_val_diagnostics(model, val_dataset, base_sequence, lp_fnc, device, indices, diag_batch_size=32):
    """Reconstruction error for a fixed set of val indices, under both readouts.

    Each spectrogram in ``indices`` is round-tripped through the model and
    compared to the original. The model is switched to ``eval()`` for the pass and
    back to ``train()`` afterwards so the caller can keep training.

    WHY TWO READOUTS. ``'posterior'`` decodes at the circular mean of the posterior
    over the lattice. When the posterior is unimodal that is the mode and the two
    readouts agree; when a sharper decoder makes it multimodal the mean falls
    between modes and the reconstruction is scored at a latent no mode occupies.
    Measured over two seeds, the penalty is 0.0001 for the shipped architecture,
    0.0006 with the ReLU head and 0.0035 with a K=4 harmonic head -- it appears in
    proportion to how much the decoder sharpens, which is exactly the axis a
    capacity change moves along. ``'argmax'`` decodes at the MAP lattice point and
    carries no such bias, so it is the one to rank architectures on.

    WHY IN-MASK. Overall MSE is 98.6% background on a masked set, where the target
    is exactly zero and every model scores about 0.0008. The call occupies ~1.4% of
    the pixels and is the whole quantity of interest. Bracket the in-mask number
    against two baselines computed on the same val frame rather than reading it
    raw: the global mean image, which knows no mask, and the true mask filled with
    the global in-mask mean, which is an oracle. On the phase 2 val set those are
    0.267 and 0.0494 against a trained model's 0.144.

    indices: pre-computed array of dataset indices (balanced across masks_len bins).

    Returns a dict of float32 arrays aligned with ``indices``:
        'mse'             overall MSE at the 'posterior' readout. UNCHANGED in
                          meaning, because every qmc_val_diagnostics.npz written
                          for phases 0 to 4 holds this and the model cards compare
                          across them.
        'mse_argmax'      overall MSE at the 'argmax' readout.
        'mse_in_mask'     in-mask MSE at 'argmax'.
        'mse_out_mask'    out-of-mask MSE at 'argmax'.
    """
    base_sequence = base_sequence.to(device)
    out = {k: [] for k in ('mse', 'mse_argmax', 'mse_in_mask', 'mse_out_mask')}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), diag_batch_size):
            batch_idx = indices[start : start + diag_batch_size]
            items = [val_dataset[i] for i in batch_idx]
            specs = torch.stack([it[0] for it in items]).to(torch.float32).to(device)
            # slot 6 is the raw mask; binarize it the same way __getitem__ does
            masks = (torch.stack([it[6] for it in items]) > 0.5).unsqueeze(1).to(device)

            recon_post = model.round_trip(base_sequence, specs, lp_fnc)
            recon_amax = model.round_trip(base_sequence, specs, lp_fnc, recon_type='argmax')

            out['mse'].extend(((recon_post - specs) ** 2).mean(dim=(1, 2, 3)).cpu().tolist())
            err = (recon_amax - specs) ** 2
            out['mse_argmax'].extend(err.mean(dim=(1, 2, 3)).cpu().tolist())
            n_in  = masks.sum(dim=(1, 2, 3)).clamp(min=1)
            n_out = (~masks).sum(dim=(1, 2, 3)).clamp(min=1)
            out['mse_in_mask'].extend(((err * masks).sum(dim=(1, 2, 3)) / n_in).cpu().tolist())
            out['mse_out_mask'].extend(((err * ~masks).sum(dim=(1, 2, 3)) / n_out).cpu().tolist())
    model.train()
    return {k: np.array(v, dtype=np.float32) for k, v in out.items()}


def val_frame_baselines(val_dataset, indices):
    """The two numbers an in-mask MSE has to be read against, on this val frame.

    Returns ``(global_mean_image, mask_oracle)``. The first fills every pixel with
    the dataset's mean image and knows nothing about the mask; the second fills the
    TRUE mask with the global in-mask mean and is therefore an oracle on the
    support. A model's in-mask MSE means nothing on its own: what it recovers is
    the fraction of the distance between them, and that fraction is the only
    quantity comparable across a change of frame such as a canonicalization.
    """
    specs = torch.stack([val_dataset[i][0] for i in indices]).to(torch.float32)
    masks = (torch.stack([val_dataset[i][6] for i in indices]) > 0.5).unsqueeze(1)
    mean_image = specs.mean(dim=0, keepdim=True)
    err = (specs - mean_image) ** 2
    global_mean = ((err * masks).sum() / masks.sum().clamp(min=1)).item()
    oracle = specs[masks].var().item()
    return global_mean, oracle


# ---------------------------------------------------------------------------
# Diagnostic plotting (loss curve + MSE breakdowns) and round-trip panels
# ---------------------------------------------------------------------------
def _save_diagnostic_plots(
    save_location, qmc_losses, val_loss_epochs, val_losses,
):
    """Write the single training-stats figure, overwriting any existing file.

    One figure, ``qmc_train_stats`` in both .png and .svg, carrying the train and
    the val log-evidence trace together. It used to be three files that had to be
    read against each other: a .png with train and val, a .svg written at the end
    of the run with train only, and two val-MSE-per-group plots whose lines were
    flat from the first checkpoint and never separated a run from any other. The
    per-group MSE numbers are still written to ``qmc_val_diagnostics.npz``, which
    is what the model cards and the cross-run comparisons read.

    The y-axis is clipped to the settled range: the first few updates sit thousands
    of nats below everything that follows, and left unclipped they flattened the
    whole curve against the top of the axes.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    train_ev = -np.asarray(qmc_losses, dtype=float)
    ax.plot(train_ev, label="train", alpha=0.8, color=TRAIN_COLOR, linewidth=0.8)
    if val_losses:
        n_batches = max(1, len(qmc_losses) // val_loss_epochs[-1])
        val_x = np.array(val_loss_epochs) * n_batches
        ax.plot(val_x, val_losses, marker="o", markersize=4, label="val",
                color=VAL_COLOR)

    # Ignore the first 1 % of updates when setting the range, then pad.
    settled = train_ev[max(1, len(train_ev) // 100):]
    if settled.size:
        lo = float(np.min(settled))
        hi = float(np.max(settled))
        if val_losses:
            lo = min(lo, float(np.min(val_losses)))
            hi = max(hi, float(np.max(val_losses)))
        pad = 0.08 * max(hi - lo, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("update number")
    ax.set_ylabel("log evidence")
    ax.set_title("Training and validation log evidence", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(save_location, f"qmc_train_stats.{ext}"))
    plt.close(fig)


def precompute_round_trip_indices(dataset, n_per_mask=10, seed=42):
    """Pre-compute fixed per-masks_len sample indices for round-trip panels.

    Returns a dict mapping each unique masks_len value to a numpy array of
    dataset indices (length <= n_per_mask).  Pass this to _save_round_trip_panel
    to keep the same samples across every epoch.
    """
    rng = np.random.default_rng(seed)
    ml_all = dataset.masks_len.numpy()
    indices = {}
    for ml in np.unique(ml_all):
        bin_inds = np.where(ml_all == ml)[0]
        chosen = rng.choice(bin_inds, size=min(n_per_mask, len(bin_inds)), replace=False)
        indices[int(ml)] = chosen
    return indices


def _save_round_trip_panel(dataset, model, base_sequence, lp_fnc, device, save_path, n_per_mask=10, seed=42, fixed_indices=None):
    """Save a single round-trip figure grouped by masks_len.

    Layout: each unique masks_len value occupies a pair of consecutive rows —
    row 2i = originals, row 2i+1 = reconstructions.  Columns = samples (up to
    n_per_mask per masks_len group).

    fixed_indices: optional dict {masks_len -> array of dataset indices} from
    precompute_round_trip_indices().  When provided, the same samples are used
    every call (useful for per-epoch panels).  When None, samples are drawn
    randomly using seed.
    """
    ml_all = dataset.masks_len.numpy()
    unique_mls = np.unique(ml_all)

    if fixed_indices is None:
        rng = np.random.default_rng(seed)
        fixed_indices = {}
        for ml in unique_mls:
            bin_inds = np.where(ml_all == ml)[0]
            fixed_indices[int(ml)] = rng.choice(bin_inds, size=min(n_per_mask, len(bin_inds)), replace=False)

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
            chosen = fixed_indices[int(ml)]

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


# ---------------------------------------------------------------------------
# Model + lattice construction
# ---------------------------------------------------------------------------
# The decoder itself now lives in models/qmc_decoder.py, imported above and
# re-exported here because callers do `from bartul_mouse import build_qmc_decoder`.
# It used to be a literal copy in seven files; that is what made adding the missing
# activation a seven-file change rather than a one-line one.


def build_lattice_pair(
    lattice_type, latent_dim,
    korobov_a, train_n_points, test_n_points,
    train_grid_m, test_grid_m,
):
    """Return ``(train_base_sequence, test_base_sequence)`` lattices.

    Selects the QMC generator by ``lattice_type``:

    * ``"korobov"`` -- Korobov lattice ``gen_korobov_basis(a, dim, n_points)``.
    * ``"roberts"`` -- Roberts low-discrepancy sequence ``roberts_sequence(n_points, dim)``.
    * anything else -- Fibonacci lattice ``gen_fib_basis(m=grid_m)`` (2D only).

    Note the differing per-generator argument order, reproduced exactly from
    the original inline construction.
    """
    if lattice_type == "korobov":
        train_base_sequence = gen_korobov_basis(
            korobov_a, latent_dim, train_n_points
        )
        test_base_sequence = gen_korobov_basis(korobov_a, latent_dim, test_n_points)
    elif lattice_type == "roberts":
        train_base_sequence = roberts_sequence(train_n_points, latent_dim)
        test_base_sequence = roberts_sequence(test_n_points, latent_dim)
    else:  # 'fib', 2D only
        train_base_sequence = gen_fib_basis(m=train_grid_m)
        test_base_sequence = gen_fib_basis(m=test_grid_m)
    return train_base_sequence, test_base_sequence


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
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
    model_seed=None,
    num_workers=None,
    # dataset parameters
    filter_mask=True,
    lo=1,
    hi=8,
    sampling_strategy="mask_duration",
    total_samples=None,
    duration_aware=False,
    apply_mask=None,
):
    """Train (or reload) a QMCLVM on mouse spectrograms and render diagnostics.

    Required positional args:
        save_location: output directory for the checkpoint, ``.npz``
            diagnostics, ``sampling_config.json`` and all plots (created if
            missing). Re-running with an existing
            ``qmc_train_mouse_experiment.tar`` here skips training and only
            regenerates the final plots.
        dataloc: path passed to ``load_mouse_data`` (train/val split source).

    Lattice / model args:
        train_grid_m, test_grid_m: Fibonacci lattice order (used only when
            ``lattice_type`` is neither korobov nor roberts; 2D only).
        latent_dim: latent dimensionality (decoder input is ``2*latent_dim``
            via TorusBasis).
        lattice_type: "korobov" | "roberts" | other (-> Fibonacci).
        korobov_a: Korobov generating integer (only used for korobov lattices).
        train_n_points, test_n_points: lattice sizes for korobov/roberts.
        nEpochs: number of training epochs.

    Training / data args:
        samples_per_mask, total_samples, duration_aware, sampling_strategy,
        filter_mask, lo, hi: forwarded to ``mouse_data`` for the train set.
        apply_mask: override whether spectrograms are multiplied by their mask.
            Leave None to let the dataset decide (sets built by
            build-qlvm-training-set declare it; older sets are masked).
            The effective value is recorded in ``sampling_config.json``.
        train_batch_size, test_batch_size, num_workers: DataLoader settings
            (``num_workers`` defaults to the CPU affinity count).
        print_gpu_mem: if True, print a compact per-epoch GPU-memory line.

    Diagnostics args:
        val_freq: run val loss + MSE diagnostics every ``val_freq`` epochs
            (and on the final epoch). Round-trip panels are written on a fixed
            cadence of every 40 epochs.
        test_samples_per_mask: cap on val samples drawn per ``masks_len`` bin
            for the diagnostic subset.
        n_dur_bins: number of duration percentile bins recorded in
            ``qmc_val_diagnostics.npz`` (``dur_bin_edges``).

    Seeding args:
        seed: dataset / round-trip RNG seed (``mouse_data`` uses ``seed``).
        model_seed: global RNG seed for model init; defaults to ``seed``.

    Side effects (artifacts written to ``save_location``):
        sampling_config.json, qmc_train_mouse_experiment.tar,
        qmc_val_diagnostics.npz, qmc_train_stats.png, qmc_train_stats.svg,
        qmc_round_trips_val_<epoch>.png, qmc_round_trips_train.png and
        qmc_round_trips_val.png. The latent-decode grid is written by
        ``analyze_mouse_latents_2d.py`` as ``figure_grid_examples.png``.
    """
    # --- seeding: model_seed governs model init RNG; mouse_data uses `seed` ---
    if model_seed is None:
        model_seed = seed
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)

    # --- data loading: build train/val mouse_data datasets and loaders ---
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
        apply_mask=apply_mask,
        seed=seed,
    )
    test_ds = mouse_data(val_dict, filter_mask=filter_mask, lo=lo, hi=hi,
                         apply_mask=apply_mask, seed=seed)
    json.dump(train_ds.sampling_config,
              open(os.path.join(save_location, 'sampling_config.json'), 'w'), indent=2)
    n_workers = num_workers if num_workers is not None else len(os.sched_getaffinity(0))
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

    # --- model + losses: binary evidence for training, binary lp for round-trips ---
    qmc_latent_dim = latent_dim
    qmc_loss_function = lambda samples, data: binary_evidence(samples, data)
    lp_fnc = lambda x, y: binary_lp(x, y)

    save_qmc  = os.path.join(save_location, "qmc_train_mouse_experiment.tar")
    save_diag = os.path.join(save_location, "qmc_val_diagnostics.npz")

    # Head selection, and why it is decided here rather than by a flag. A new run
    # gets the ReLU head. A run being RE-OPENED gets whatever head its own
    # checkpoint holds, read out of the state dict, so every phase 0 to 4
    # checkpoint still loads unchanged and the figures regenerate from it.
    if os.path.isfile(save_qmc):
        decoder_qmc, decoder_head = build_for_checkpoint(save_qmc, qmc_latent_dim)
        print(f"Reopening an existing checkpoint; its decoder head is {decoder_head!r}.")
    else:
        decoder_head = "relu"
        decoder_qmc = build_qmc_decoder(qmc_latent_dim, head=decoder_head)
        print(f"New run; decoder head is {decoder_head!r}.")

    qmc_model = QMCLVM(
        latent_dim=qmc_latent_dim,
        device=device,
        decoder=decoder_qmc,
        basis=TorusBasis(),
    )
    print_gpu_memory("after model init")
    train_base_sequence, test_base_sequence = build_lattice_pair(
        lattice_type, qmc_latent_dim,
        korobov_a, train_n_points, test_n_points,
        train_grid_m, test_grid_m,
    )

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

    # The two numbers the in-mask MSE has to be read against on THIS val frame.
    # Computed once and stamped into the diagnostics, because a canonicalized frame
    # has a different mask area and a different in-mask variance, and an in-mask
    # MSE compared across frames without them is meaningless.
    baseline_global_mean, baseline_mask_oracle = val_frame_baselines(
        test_loader.dataset, val_diag_indices
    )
    print(f"In-mask baselines on this val frame: global mean image "
          f"{baseline_global_mean:.5f}, mask oracle {baseline_mask_oracle:.5f}")

    # small loader over the fixed diagnostic subset — used for val loss and MSE
    val_diag_loader = DataLoader(
        torch.utils.data.Subset(test_loader.dataset, val_diag_indices),
        num_workers=n_workers, shuffle=False, batch_size=test_batch_size,
    )

    # precompute duration bin edges from full val set (consistent across epochs)
    dur_bin_edges = np.percentile(val_dur_all, np.linspace(0, 100, n_dur_bins + 1))
    dur_bin_edges[0]  -= 1
    dur_bin_edges[-1] += 1

    # pre-compute fixed indices for per-epoch round-trip panels
    round_trip_val_indices = precompute_round_trip_indices(test_loader.dataset, n_per_mask=10, seed=seed)

    # --- train-vs-load gate: presence of the .tar checkpoint decides ---
    if not os.path.isfile(save_qmc):
        print("now training qmc model")
        torch.cuda.reset_peak_memory_stats()

        qmc_opt    = Adam(qmc_model.parameters(), lr=1e-3)
        # Decay the LR when val evidence stops improving, and keep the weights from
        # the best validation checkpoint rather than whatever the last epoch left.
        # Across the 24 phase 0 to 4 runs, 12 finished ABOVE their own minimum, by a
        # median of 0.06% and a worst case of 1.09%; the median run also reached 95%
        # of its total improvement at 63% of the way through. Neither fact is worth
        # much on its own, but both are free.
        qmc_sched = ReduceLROnPlateau(qmc_opt, mode='min', factor=0.5, patience=3)
        qmc_losses = []
        diag_epochs, diag_mse = [], []
        val_loss_epochs, val_losses, lr_trace = [], [], []
        best_val, best_epoch, best_state = np.inf, None, None

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
                val_now = float(np.mean(val_batch_losses))
                val_losses.append(val_now)
                val_loss_epochs.append(epoch + 1)
                qmc_sched.step(val_now)
                lr_trace.append(qmc_opt.param_groups[0]['lr'])

                if val_now < best_val:
                    best_val, best_epoch = val_now, epoch + 1
                    best_state = copy.deepcopy(qmc_model.state_dict())

                diag = compute_val_diagnostics(
                    qmc_model, test_loader.dataset,
                    test_base_sequence, lp_fnc, device,
                    val_diag_indices,
                )
                diag_epochs.append(epoch + 1)
                diag_mse.append(diag)

                _save_diagnostic_plots(
                    save_location, qmc_losses, val_loss_epochs, val_losses,
                )

            # round-trip panels on a fixed cadence of every 40 epochs
            if (epoch + 1) % 40 == 0:
                _save_round_trip_panel(
                    test_loader.dataset, qmc_model,
                    test_base_sequence.to(device), lp_fnc, device,
                    os.path.join(save_location, f"qmc_round_trips_val_{epoch + 1}.png"),
                    n_per_mask=10, fixed_indices=round_trip_val_indices,
                )

            if print_gpu_mem and torch.cuda.is_available():
                print_gpu_memory(f"epoch {epoch+1}", compact=True)

        print_gpu_memory("after training")
        # The checkpoint is the best-validation one, not the last one. The loss
        # trace and the diagnostics still cover every epoch, and `best_epoch` says
        # which of them the saved weights correspond to.
        if best_state is not None and best_epoch != val_loss_epochs[-1]:
            print(f"Restoring the epoch-{best_epoch} weights "
                  f"(val {best_val:.2f}) over the epoch-{val_loss_epochs[-1]} ones "
                  f"(val {val_losses[-1]:.2f}).")
            qmc_model.load_state_dict(best_state)
        save(qmc_model.to("cpu"), qmc_opt, qmc_losses, fn=save_qmc)
        qmc_model.to(device)
        np.savez(
            save_diag,
            epochs=np.array(diag_epochs),
            # 'mse' keeps its phase 0 to 4 meaning: overall MSE at the 'posterior'
            # readout. The argmax and in-mask columns are additions beside it.
            mse=np.array([d['mse'] for d in diag_mse]),
            mse_argmax=np.array([d['mse_argmax'] for d in diag_mse]),
            mse_in_mask=np.array([d['mse_in_mask'] for d in diag_mse]),
            mse_out_mask=np.array([d['mse_out_mask'] for d in diag_mse]),
            val_diag_ml=val_diag_ml,
            val_diag_dur=val_diag_dur,
            dur_bin_edges=dur_bin_edges,
            val_loss_epochs=np.array(val_loss_epochs),
            val_losses=np.array(val_losses),
            lr_trace=np.array(lr_trace),
            best_epoch=np.array(best_epoch if best_epoch is not None else -1),
            best_val_loss=np.array(best_val),
            decoder_head=np.array(decoder_head),
            baseline_global_mean=np.array(baseline_global_mean),
            baseline_mask_oracle=np.array(baseline_mask_oracle),
        )
    else:
        qmc_opt = Adam(qmc_model.parameters(), lr=1e-3)
        qmc_model, qmc_opt, qmc_losses = load(qmc_model, qmc_opt, save_qmc)
        val_loss_epochs, val_losses = [], []
        print_gpu_memory("after model load")

    # --- final plots: the merged loss figure and the train/val round-trips ---
    #
    # No latent-decode grid here any more. It was written twice per run, once as
    # `qmc_grid.png` by plotting.visualize.model_grid_plot and once as
    # `figure_grid_examples.png` by analyze_mouse_latents_2d.grid_examples. The
    # inference-side one is kept: it places the samples on torus cell centres
    # (`linspace(0, 1, n, endpoint=False) + 0.5/n`) rather than sampling the wrapped
    # edge twice, orients row 0 at the bottom to match every other latent panel, and
    # takes a conditioning vector. The training-side one is gone.
    qmc_losses = np.array(qmc_losses)

    # On the reload path the val trace is not in scope; recover it from the run's own
    # diagnostics file so a regenerated figure is the same figure, not a train-only one.
    if not val_losses and os.path.isfile(save_diag):
        _diag = np.load(save_diag)
        val_loss_epochs = _diag["val_loss_epochs"].tolist()
        val_losses = _diag["val_losses"].tolist()
    _save_diagnostic_plots(save_location, qmc_losses, val_loss_epochs, val_losses)

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
