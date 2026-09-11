"""Train a CONDITIONAL QMCLVM on mouse vocalization spectrograms.

A ``fire`` CLI driver, the conditional sibling of ``bartul_mouse.py``. The model
is the same fixed QMC lattice plus learned ConvTranspose decoder; the difference
is that a conditioning vector ``c`` is concatenated onto every lattice point
before decoding, so the decoder learns one family of images per conditioning
value. ``data/conditionals.py`` is the single source of truth for which
conditionals exist, how wide each one is, and how rows are grouped by it.

ONE ``c`` PER BATCH -- the constraint everything here is shaped around.
``QMCLVM.forward`` decodes the whole lattice against ``c.repeat(n_lattice, 1)``
and ``binary_evidence`` marginalizes every row of the batch over that one stack
of images. ``train/train.py`` therefore reads a single ``c`` per batch
(``batch[1]...view(1, -1)``). A per-row ``c`` is not affordable: it would need
``B x n_lattice`` decoder outputs, ~20 GB per forward at B=512, n_lattice=610,
128x128.

That is exact only if every row in the batch shares the ``c`` it is scored
against. It did not: ``make_collate_fn`` used to average the per-row values, and
the mean of 512 shuffled one-hots is just the dataset marginal -- a near-constant
vector, the same at every step, telling the decoder nothing about the batch. The
April-2025 checkpoint trained that way moved its output by 0.0031 (pixel std)
across the eight mask-count classes versus 0.0191 across lattice position, i.e.
it had learned to ignore ``c``.

The fix is CONDITIONING-HOMOGENEOUS BATCHES, not per-row decoding: rows are
grouped by conditioning value (``conditionals.group_ids``) and
``ConditionGroupedBatchSampler`` draws every batch from a single group. Same cost
per step, and the batch mean becomes the exact value (discrete) or a point inside
one quantile bin (continuous). ``make_collate_fn`` now guards that homogeneity
rather than assuming it, because a silent regression to mixed batches reproduces
the original bug with no visible symptom.

Usage (fire CLI)::

    python bartul_mouse_cond.py <save_location> <dataloc> \
        --conditional=mask_count --nEpochs=300 --train_batch_size=512

``save_location`` and ``dataloc`` are required positional arguments; everything
else is an optional ``--flag=value`` (note the camelCase ``--nEpochs``).

A ``fire`` TRAP WORTH KNOWING. fire does not reject unknown flags up front: it
runs the function to completion, writes every artifact, and only then exits 2 on
the unconsumed flag. The launcher used to pass ``--latent_dim`` and
``--lattice_type`` to a version of ``run_mouse_cond_experiments`` that had
neither, so those flags did nothing at all while the job "failed" after having
succeeded. That is why this driver now mirrors ``bartul_mouse.py``'s full
parameter list rather than a subset of it.
"""

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
from plotting.visualize import format_plot_axis, conditional_qmc_grid_plot
from data.mouse_data import ConditionGroupedBatchSampler, load_mouse_data, mouse_data
from data import conditionals
from data.conditionals import DEFAULT_COND_N_BINS, get_grid_sweep, sample_c
from bartul_mouse import build_lattice_pair

import matplotlib.pyplot as plt
import json
import random
import numpy as np
import fire
from tqdm import tqdm


# The conditional registry used to be duplicated here, in
# analyze_mouse_latents_2d.py and in analyze_all_sessions.py, and the three
# copies drifted. It now lives only in data/conditionals.py. ``get_grid_sweep``
# and ``sample_c`` are imported above purely to be re-exported: inference code
# does ``from bartul_mouse_cond import get_grid_sweep``, and that import is kept
# working rather than chased across every caller.
__all__ = [
    "get_grid_sweep",
    "sample_c",
    "make_collate_fn",
    "compute_val_diagnostics",
    "run_mouse_cond_experiments",
]


# Within-batch spread of the conditioning value above which a batch is not
# homogeneous. Identical one-hots differ by exactly nothing, so for a discrete
# conditional anything above float noise means the grouping broke. A continuous
# conditional is grouped into quantile bins and so has a real non-zero spread --
# the bin width -- which the training driver measures and passes in; this default
# is the fallback for callers that do not, and is set to catch the regression
# (a shuffled batch spans nearly the whole [0, 1] range) rather than to police
# the bin width.
_DISCRETE_SPREAD_TOL = 1e-6
_CONTINUOUS_SPREAD_TOL = 0.5


def make_collate_fn(cond_names, spread_tol=None, strict=None):
    """Return a collate fn producing ``(specs, c, masks, spec_ids)``.

    ``c`` has shape ``(1, total_c_dim)`` -- one conditioning vector for the whole
    batch, which is what ``train/train.py`` reads (``batch[1]...view(1, -1)``) and
    all ``QMCLVM`` can use (see the module docstring).

    The batch mean is still how ``c`` is formed, but it now means something
    different: batches arrive from ``ConditionGroupedBatchSampler`` already
    homogeneous, so for a discrete conditional the mean of identical one-hots IS
    that one-hot, exactly, and for a continuous one it is a representative point
    inside a single quantile bin. Averaged over a SHUFFLED batch the same line
    produced the dataset marginal and the model learned nothing from ``c`` -- so
    the homogeneity is checked here, loudly, instead of assumed. A regression to
    mixed batches is invisible in the loss curve; it just quietly stops
    conditioning.

    Args:
        cond_names: conditional names from ``data.conditionals`` (one, in
            practice; several are concatenated along the ``c`` axis).
        spread_tol: largest within-batch spread (max minus min, worst component)
            treated as homogeneous. Defaults to ``_DISCRETE_SPREAD_TOL`` when every
            conditional is discrete, else ``_CONTINUOUS_SPREAD_TOL``; the training
            driver passes the widest quantile bin it actually built, which is the
            exact bound.
        strict: raise on a violation rather than warn. Defaults to True when every
            conditional is discrete -- there a violation can only be a bug -- and
            False otherwise, where a fat tail bin can legitimately be wide.

    Note the warn-once flag is per worker process: with ``num_workers > 0`` the
    collate runs in the workers, so a warning may appear once per worker. A raise
    propagates to the main process as normal.
    """
    cfg = [conditionals.resolve(n) for n in cond_names]
    all_discrete = all(entry["kind"] == "discrete" for entry in cfg)
    if strict is None:
        strict = all_discrete
    if spread_tol is None:
        spread_tol = _DISCRETE_SPREAD_TOL if all_discrete else _CONTINUOUS_SPREAD_TOL
    warned = {"already": False}

    def collate(batch):
        specs = torch.stack([b[0] for b in batch])
        c_parts = []
        spread = 0.0
        for entry in cfg:
            fi = entry["field_idx"]
            vals = torch.stack([
                b[fi].float() if b[fi].dim() > 0 else b[fi].float().unsqueeze(0)
                for b in batch
            ])                                            # (B, c_dim) or (B, 1)
            if vals.dim() == 1:
                vals = vals.unsqueeze(1)                  # ensure 2-D
            if vals.shape[0] > 1:
                per_component = vals.max(dim=0).values - vals.min(dim=0).values
                spread = max(spread, float(per_component.max()))
            c_parts.append(vals.mean(dim=0, keepdim=True))  # (1, c_dim)
        c = torch.cat(c_parts, dim=-1)                   # (1, total_c_dim)

        if spread > spread_tol:
            message = (
                f"conditioning is NOT homogeneous within this batch: spread="
                f"{spread:.4g} > tol={spread_tol:.4g} over {len(batch)} rows for "
                f"{cond_names}. The batch mean of a mixed batch is the dataset "
                f"marginal, which is the bug ConditionGroupedBatchSampler exists "
                f"to fix -- pass batch_sampler=ConditionGroupedBatchSampler(...) "
                f"to the DataLoader instead of shuffle=True."
            )
            if strict:
                raise ValueError(message)
            if not warned["already"]:
                print(f"WARNING: {message}")
                warned["already"] = True

        masks    = torch.stack([b[6] for b in batch])
        spec_ids = [b[7] for b in batch]
        return (specs, c, masks, spec_ids)

    return collate


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
                c = sample_c(dataset, idx, cond_name, device=device)
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

    cond_label = conditionals.label_of(cond_name)
    plt.suptitle(f"Round trips — conditional: {cond_label}", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    model.train()


def compute_val_diagnostics(model, val_dataset, base_sequence, lp_fnc, device,
                             indices, cond_name, diag_batch_size=32,
                             cond_n_bins=DEFAULT_COND_N_BINS, groups=None):
    """Per-sample reconstruction MSE over a fixed set of val indices.

    Diagnostic batches are grouped by conditioning value before decoding, the
    same way training batches are: ``round_trip`` conditions a whole batch on one
    ``c`` (``c.repeat(n_lattice, 1)`` inside ``QMCLVM.forward``), so a mixed batch
    would score every row against a conditioning value that is nobody's. Within a
    group the mean of the per-row values IS the row value for a discrete
    conditional and a point inside one quantile bin for a continuous one -- which
    is exactly the guarantee training now relies on, so the diagnostic measures
    the model that is being trained.

    (This function used to take the mean over an arbitrary slice of ``indices``,
    which was "consistent with training" only while training was itself wrong.)

    Args:
        indices: val-set row indices to score.
        cond_name: conditional name, resolved through ``data.conditionals``.
        diag_batch_size: rows decoded at once WITHIN a group; groups are never
            mixed, so a group smaller than this just yields a smaller batch.
        cond_n_bins: quantile bins for a continuous conditional.
        groups: precomputed ``conditionals.group_ids(val_dataset, ...)``. Passed
            in by the training loop so the quantiles are not recomputed at every
            validation checkpoint.

    Returns a float32 array aligned with ``indices``, NOT with the grouped decode
    order -- callers pair it with ``val_diag_ml`` / ``val_diag_dur``, which are in
    ``indices`` order.
    """
    entry = conditionals.resolve(cond_name)
    fi = entry["field_idx"]
    base_sequence = base_sequence.to(device)
    indices = np.asarray(indices)
    if groups is None:
        groups = conditionals.group_ids(val_dataset, cond_name, cond_n_bins)
    diag_groups = np.asarray(groups)[indices]

    all_mse = np.zeros(len(indices), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for gid in np.unique(diag_groups):
            # positions into `indices` (and therefore into the output array)
            in_group = np.where(diag_groups == gid)[0]
            for start in range(0, len(in_group), diag_batch_size):
                positions = in_group[start : start + diag_batch_size]
                items = [val_dataset[i] for i in indices[positions]]
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
                all_mse[positions] = mse.astype(np.float32)
    model.train()
    return all_mse


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


def _update_run_config(save_location, updates):
    """Merge ``updates`` into ``<save_location>/run_config.json``.

    The SLURM launcher writes this file first (phase, mask_tag, dataset,
    dataset_path, latent_dim, seed) and ``analyze_all_sessions.py`` reads it back
    to work out which arm a checkpoint belongs to. Read-modify-write rather than
    overwrite, so neither side has to know the other's keys and whichever runs
    second does not erase the first. A file that is not readable JSON is reported
    and replaced, rather than aborting a training run over a record file.
    """
    path = os.path.join(save_location, "run_config.json")
    record = {}
    if os.path.isfile(path):
        try:
            with open(path) as handle:
                record = json.load(handle)
        except (ValueError, OSError) as err:
            print(f"WARNING: {path} is unreadable ({err}); rewriting it from scratch.")
            record = {}
    record.update(updates)
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)
    return record


def run_mouse_cond_experiments(
    save_location,
    dataloc,
    conditional="mask_count",   # any name in data/conditionals.py's registry
    train_grid_m=15,
    test_grid_m=20,
    nEpochs=300,
    train_batch_size=64,
    test_batch_size=1,
    print_gpu_mem=False,
    latent_dim=2,
    lattice_type="fib",
    korobov_a=76,
    train_n_points=1021,
    test_n_points=2039,
    seed=42,
    model_seed=None,
    num_workers=None,
    # validation parameters
    val_freq=10,
    test_samples_per_mask=50,
    n_dur_bins=5,
    # conditioning
    cond_n_bins=DEFAULT_COND_N_BINS,
    # dataset parameters
    filter_mask=False,
    lo=1,
    hi=8,
    sampling_strategy=None,
    samples_per_mask=5000,
    total_samples=None,
    duration_aware=False,
    apply_mask=None,
):
    """Train (or reload) a conditional QMCLVM and render its diagnostics.

    Required positional args:
        save_location: output directory for the checkpoint, ``.npz`` diagnostics,
            ``sampling_config.json``, ``run_config.json`` and every plot (created
            if missing). Re-running with an existing
            ``qmc_train_mouse_cond_experiment.tar`` here skips training and only
            regenerates the final plots.
        dataloc: path passed to ``load_mouse_data`` (train/val split source).

    Conditioning args:
        conditional: name from ``data/conditionals.py`` -- ``mask_count`` (5-class
            one-hot), ``mask_count8`` (the legacy 8-class width), ``duration`` or
            ``mean_freq``. It fixes ``c_dim``, the width of the mask-count one-hot
            the datasets build, and how rows are grouped into batches.
        cond_n_bins: quantile bins a CONTINUOUS conditional is grouped into for
            batching (ignored by the discrete ones). Each bin should hold several
            batches; 32 bins over ~106k rows is ~3,300 rows, ~6 batches of 512.

    Lattice / model args:
        latent_dim: latent dimensionality. The decoder input is
            ``2 * latent_dim + c_dim``: ``TorusBasis`` expands each coordinate to a
            ``(cos, sin)`` pair, and ``c`` is concatenated onto that.
        lattice_type: "korobov" | "roberts" | anything else (-> Fibonacci).
            Defaults to "fib" here, which is what this driver has always used.
        korobov_a: Korobov generating integer (korobov lattices only).
        train_n_points, test_n_points: lattice sizes for korobov/roberts.
        train_grid_m, test_grid_m: Fibonacci lattice order (2D only; used only
            when ``lattice_type`` is neither korobov nor roberts).
        nEpochs: number of training epochs.

    Training / data args:
        train_batch_size: rows per batch. Every batch is drawn from ONE
            conditioning group, so this is a cap rather than an exact size --
            a group smaller than this yields one short (still homogeneous) batch.
        test_batch_size: val / diagnostic batch size. Left at 1 by default: a
            single-row batch is exactly conditioned by construction.
        num_workers: DataLoader workers; defaults to the CPU affinity count.
        print_gpu_mem: if True, print a compact per-epoch GPU-memory line.

    Dataset args -- NOTE THE DEFAULTS, which differ from ``bartul_mouse.py``:
        filter_mask defaults to False and sampling_strategy to None, because every
        current dataset (the bbvfree / playpen sets built by usv-playpen's
        ``build-qlvm-training-set``) is ALREADY filtered to its mask-count strata
        and ALREADY drawn per session to a stratification that is the experimental
        variable. Re-filtering or re-sampling here would silently destroy exactly
        the thing the run is comparing. ``lo``, ``hi``, ``samples_per_mask``,
        ``total_samples`` and ``duration_aware`` remain for the legacy path (a raw
        ``.pt`` set), where ``filter_mask=True`` / ``sampling_strategy=
        'mask_duration'`` still make sense.
        apply_mask: override whether spectrograms are multiplied by their mask.
            Leave None to let the dataset declare it (build-qlvm-training-set sets
            write the flag; older sets are masked). The resolved value is recorded
            in both ``sampling_config.json`` and ``run_config.json``.

    Diagnostics args:
        val_freq: run val loss + MSE diagnostics every ``val_freq`` epochs (and on
            the final epoch).
        test_samples_per_mask: cap on val rows drawn per ``masks_len`` bin for the
            fixed diagnostic subset.
        n_dur_bins: number of duration percentile bins recorded in the ``.npz``.

    Seeding args:
        seed: dataset / sampler / round-trip RNG seed.
        model_seed: global RNG seed for model init; defaults to ``seed``.

    Side effects (written to ``save_location``): sampling_config.json,
    run_config.json (merged, not overwritten), qmc_train_mouse_cond_experiment.tar,
    qmc_cond_val_diagnostics.npz, the three diagnostic plots, one decoder grid per
    conditioning value in the sweep, and two round-trip panels.
    """
    entry = conditionals.resolve(conditional)      # raises, listing the options
    cond_names = [conditional]
    cond_label = conditionals.label_of(conditional)
    c_dim = conditionals.c_dim_of(conditional)
    mask_count_classes = conditionals.mask_count_classes_for(conditional)

    # model_seed governs model-init RNG; the datasets and the sampler use `seed`.
    if model_seed is None:
        model_seed = seed
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)

    if not os.path.exists(save_location):
        print(f"Creating save directory: {save_location}")
        os.makedirs(save_location)

    # Lattices first: they are cheap and they are two of the numbers the run
    # record needs, so the record can be written before anything slow happens.
    train_base_sequence, test_base_sequence = build_lattice_pair(
        lattice_type, latent_dim,
        korobov_a, train_n_points, test_n_points,
        train_grid_m, test_grid_m,
    )

    train_dict, val_dict = load_mouse_data(dataloc)
    train_ds = mouse_data(
        train_dict,
        filter_mask=filter_mask, lo=lo, hi=hi,
        sampling_strategy=sampling_strategy,
        samples_per_mask=samples_per_mask,
        total_samples=total_samples,
        duration_aware=duration_aware,
        apply_mask=apply_mask,
        mask_count_classes=mask_count_classes,
        seed=seed,
    )
    test_ds  = mouse_data(val_dict, filter_mask=filter_mask, lo=lo, hi=hi,
                          apply_mask=apply_mask,
                          mask_count_classes=mask_count_classes, seed=seed)
    json.dump(train_ds.sampling_config,
              open(os.path.join(save_location, 'sampling_config.json'), 'w'), indent=2)
    _update_run_config(save_location, {
        "conditional": conditional,
        "c_dim": c_dim,
        "cond_n_bins": cond_n_bins,
        "lattice_type": lattice_type,
        "n_lattice_train": int(len(train_base_sequence)),
        "n_lattice_test": int(len(test_base_sequence)),
        "mask_count_classes": mask_count_classes,
        "apply_mask_effective": bool(train_ds.apply_mask),
        "driver": "bartul_mouse_cond.py",
    })

    n_workers = num_workers if num_workers is not None else len(os.sched_getaffinity(0))

    # --- conditioning-homogeneous batching (the point of this driver) ---
    train_groups = conditionals.group_ids(train_ds, conditional, cond_n_bins)
    train_sampler = ConditionGroupedBatchSampler(
        train_groups, batch_size=train_batch_size,
        shuffle=True, drop_last=False, seed=seed,
    )

    # The collate guard needs to know what spread is legitimate. Zero, for a
    # discrete conditional. For a continuous one it is the width of the widest bin
    # the grouping actually produced -- measured rather than guessed, so a fat tail
    # bin does not raise a false alarm while a genuinely mixed batch still does.
    spread_tol = None
    if entry["kind"] != "discrete":
        raw = conditionals._raw_values(train_ds, conditional).reshape(-1)
        widest = max(np.ptp(raw[train_groups == g]) for g in np.unique(train_groups))
        spread_tol = float(widest) * 1.05 + 1e-6
    collate_fn = make_collate_fn(cond_names, spread_tol=spread_tol)

    sizes = train_sampler.group_sizes()
    print(f"Conditional: {conditionals.describe(conditional, cond_n_bins)}")
    print(f"c_dim={c_dim}, latent_dim={latent_dim}, "
          f"decoder input dim={2 * latent_dim + c_dim}")
    print(f"Grouped batching: {len(sizes)} groups over {len(train_ds)} rows; "
          f"group size min={int(sizes.min())} median={int(np.median(sizes))} "
          f"max={int(sizes.max())}; {len(train_sampler)} batches/epoch")
    if spread_tol is not None:
        print(f"Widest quantile bin spans {spread_tol:.4g} -- the within-batch "
              f"conditioning spread the collate guard will tolerate")
    print(f"Using train_batch_size={train_batch_size}, test_batch_size={test_batch_size}")

    # batch_sampler owns batching entirely: DataLoader forbids passing
    # batch_size / shuffle / drop_last alongside it.
    train_loader = DataLoader(train_ds, num_workers=n_workers,
                              batch_sampler=train_sampler, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print_gpu_memory("before model init")

    qmc_loss_function = lambda samples, data: binary_evidence(samples, data)
    lp_fnc = lambda x, y: binary_lp(x, y)

    # TorusBasis expands each latent coordinate to a (cos, sin) pair, and `c` is
    # concatenated onto that -- hence 2 * latent_dim + c_dim inputs.
    decoder_qmc = nn.Sequential(
        nn.Linear(2 * latent_dim + c_dim, 2048),
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

    qmc_model = QMCLVM(latent_dim=latent_dim, device=device,
                       decoder=decoder_qmc, basis=TorusBasis())
    print_gpu_memory("after model init")

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

    # Group ids for the val set, computed once: compute_val_diagnostics decodes
    # group by group, and the quantile edges must not wobble between checkpoints.
    val_groups = conditionals.group_ids(test_ds, conditional, cond_n_bins)

    # test_batch_size=1 needs no grouping -- a one-row batch is exactly
    # conditioned on its own row -- but the size is a CLI knob, and any larger
    # value would re-introduce the batch-mean bug here (and trip the collate
    # guard). So group this loader too as soon as it batches more than one row.
    # shuffle=False: the diagnostic subset is meant to be a fixed, repeatable pass.
    val_subset = torch.utils.data.Subset(test_ds, val_diag_indices)
    if test_batch_size > 1:
        val_diag_loader = DataLoader(
            val_subset, num_workers=n_workers, collate_fn=collate_fn,
            batch_sampler=ConditionGroupedBatchSampler(
                val_groups[val_diag_indices], batch_size=test_batch_size,
                shuffle=False, drop_last=False, seed=seed,
            ),
        )
    else:
        val_diag_loader = DataLoader(
            val_subset, num_workers=n_workers, shuffle=False,
            batch_size=test_batch_size, collate_fn=collate_fn,
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
                    cond_n_bins=cond_n_bins, groups=val_groups,
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
    # conditional_qmc_grid_plot tiles a 2D meshgrid of latent coordinates, so it
    # is meaningless (and a shape error) for any other latent_dim. Higher
    # dimensional runs get their decoder grids from analyze_mouse_latents_3d.py.
    # Raising here would fail the job AFTER the checkpoint and every diagnostic
    # had been written, which is the most expensive place to fail.
    if latent_dim == 2:
        sweep = get_grid_sweep(conditional)
        for val_label, c_val in sweep:
            conditional_qmc_grid_plot(
                qmc_model, n_samples_dim=20, c=c_val.to(device), show=False,
                fn=os.path.join(save_location, f'qmc_cond_grid_{conditional}_{val_label}.png'),
                title=f"{cond_label} = {val_label}",
                origin='lower', cm='viridis',
            )
    else:
        print(f"latent_dim={latent_dim}: skipping the decoder grid sweep "
              f"(conditional_qmc_grid_plot is 2D-only); the latent grids for this "
              f"run come from analyze_mouse_latents_3d.py")

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
