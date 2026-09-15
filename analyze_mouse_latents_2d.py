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
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader
import os
import fire
from tqdm import tqdm

from models.qmc_decoder import build_for_checkpoint
from models.qmc_base import QMCLVM, TorusBasis
from models.sampling import gen_fib_basis
from train.model_saving_loading import load
from train.losses import binary_lp
from torch.optim import Adam
from data.mouse_data import (
    ConditionGroupedBatchSampler, load_full_mouse_data, load_mouse_data, mouse_data,
)
from data import conditionals
from analysis.model_helpers import get_posterior_summaries, torus_forward, torus_reverse
from analysis.clustering import run_mean_shift, run_mean_shift_fast
from plotting.figstyle import (
    SESSION_TYPE_COLORS, SESSION_TYPE_ORDER, COND_COLORS, COND_ORDER,
    EMITTER_SEX_COLORS, EMITTER_SEX_ORDER, MISSING_COLOR,
    CMAP_SPEC, CMAP_POSTERIOR, CMAP_FREQ, CMAP_BANDWIDTH, CMAP_MASK_COUNT,
    CMAP_DURATION, CMAP_SOCIAL_DIST, CMAP_SEGMENT, OVERLAY_COLOR, HIGHLIGHT_COLOR,
    SOCIAL_DIST_RANGE, MASK_COUNT_RANGE, HIGH_MASK_COUNTS, HIGH_MASK_COLORS,
    session_type_color, bar_shades, scatter_params_for,
)


# ---------------------------------------------------------------------------
# Conditional variable registry -- now owned by data/conditionals.py.
#
# This module used to carry its own literal copy of the registry, as did
# bartul_mouse_cond.py (training). They drifted, and the drift was silent in the
# worst possible place: the mask-count one-hot width. Training moved to 5 classes
# (1/2/3/4/5+) while this copy still said 8, and a width disagreement between the
# script that trained the decoder and the script that rebuilds it does not raise
# -- it builds nn.Linear(2*latent_dim + c_dim, ...) at the wrong input size and
# either fails to load the state dict or, worse, loads something that runs.
#
# The alias below keeps the module-level name importable for anything that still
# reads it; every lookup in this file now goes through the shared helpers
# (conditionals.resolve / c_dim_of / describe / get_grid_sweep / sample_c), so
# there is nothing left here to drift.
# ---------------------------------------------------------------------------
CONDITIONAL_REGISTRY = conditionals.CONDITIONAL_REGISTRY


CONDITION_SUFFIX = {
    "courtship_mute_female": "mute",
    "courtship_intact_partners": "intact",
}


def _refine_session_types(session_types, session_ids, csv_path):
    """Split a sex-derived session_type by experimental condition.

    ``session_types`` from build-qlvm-training-set encodes the SEX PAIRING only, so
    two conditions run on the same pairing are indistinguishable afterwards. This
    reads a ``session_id,condition`` CSV (written from the source session lists, so
    it is authoritative rather than inferred) and appends a condition suffix where
    one is defined -- ``MF`` -> ``MF_mute`` / ``MF_intact``.

    Sessions absent from the table, and conditions with no suffix defined, keep
    their original label: this only ever refines, never relabels across pairings.
    """
    import csv as _csv

    condition_of = {}
    with open(csv_path, newline='') as handle:
        for row in _csv.DictReader(handle):
            condition_of[row['session_id']] = row['condition']

    types = np.array([str(t) for t in session_types], dtype=object)
    ids = np.array([str(s) for s in session_ids], dtype=object)
    refined, n_changed, unseen = types.copy(), 0, set()
    for i, (label, sid) in enumerate(zip(types, ids)):
        condition = condition_of.get(sid)
        if condition is None:
            unseen.add(sid)
            continue
        suffix = CONDITION_SUFFIX.get(condition)
        if suffix is not None:
            refined[i] = f"{label}_{suffix}"
            n_changed += 1
    counts = {k: int(v) for k, v in zip(*np.unique(refined, return_counts=True))}
    print(f"Session conditions: refined {n_changed:,} of {len(types):,} labels "
          f"from {csv_path}")
    if unseen:
        print(f"  WARNING: {len(unseen)} session(s) not in the condition table, "
              f"label unchanged (e.g. {sorted(unseen)[:3]})")
    print(f"  types now: {counts}")
    return list(refined)


# ---------------------------------------------------------------------------
# Conditioning-homogeneous batching at INFERENCE.
#
# The QMCLVM decodes ONE lattice per batch against ONE ``c``:
# ``QMCLVM.posterior_probability`` builds ``basis = [lattice_basis,
# c.repeat(n_lattice, 1)]`` and scores EVERY row of the batch against that single
# stack of decoded images. That is exact only when every row in the batch shares
# the ``c`` it is scored against -- the same structural constraint training has,
# and the reason training draws its batches through ``ConditionGroupedBatchSampler``
# (see data/mouse_data.py and data/conditionals.py).
#
# Inference did not do this. It ran the ordinary dataset-order loader and averaged
# the batch's conditioning values, so at the launcher's batch_size=512 each
# spectrogram's posterior was computed against a decoder conditioned on roughly the
# DATASET MARGINAL: a row whose normalized duration is 0.9 was scored by a decoder
# told about 0.27. The posterior is then wrong and the latent coordinate lands
# wherever the mismatched decoder happens to explain that row best -- which is to
# say every conditional embedding was wrong, with no visible symptom.
#
# batch_size=1 would be exact and is not affordable: ``posterior_probability``
# decodes the WHOLE LATTICE once per batch, so one row per batch means one full
# lattice decode per spectrogram -- 365,150 decodes of 46,368 points for the full
# corpus. Grouping the batches makes the existing one-``c``-per-batch decode
# correct instead, at one decode per batch exactly as before.
#
# The cost of grouping is that the loader no longer walks the dataset in order, and
# ``get_posterior_summaries`` assembles its outputs with vstack/concatenate over
# batches -- i.e. in LOADER order. Every consumer downstream joins those arrays
# positionally against dataset-order metadata (spec_id, durations, mask counts,
# session types, the recon breakdown, and the external probe in
# scripts/dataset_construct/14_fig11_latent_probe.py, which indexes
# posterior_cache.npz straight against full_data.npz). So the permutation is
# captured here and inverted on the way out; a silent misalignment there would be
# worse than the bug being fixed.
# ---------------------------------------------------------------------------

# Largest within-batch spread of the conditioning value still treated as
# homogeneous. These mirror ``bartul_mouse_cond._DISCRETE_SPREAD_TOL`` /
# ``_CONTINUOUS_SPREAD_TOL``; they are restated rather than imported because
# importing the training driver into an analysis script would drag train/,
# plotting.visualize and bartul_mouse in for two floats. Identical one-hots differ
# by exactly nothing, so for a discrete conditional anything above float noise
# means the grouping broke. A continuous conditional legitimately spans its
# quantile bin, so callers measure the widest bin they actually built and pass it
# in; the 0.5 fallback is set to catch the regression (an ungrouped batch spans
# nearly the whole [0, 1] range), not to police the bin width.
_DISCRETE_SPREAD_TOL = 1e-6
_CONTINUOUS_SPREAD_TOL = 0.5


def _c_batch_mean(vals, cond_name, spread_tol=None, strict=None, warned=None):
    """Collapse per-row conditioning values to the single ``(1, c_dim)`` vector the
    decoder takes, having first checked the collapse means anything.

    The inference twin of the guard in ``bartul_mouse_cond.make_collate_fn``, and
    deliberately the same behaviour: a discrete conditional RAISES (identical
    one-hots differ by exactly nothing, so any spread can only be a bug), a
    continuous one WARNS once against the widest quantile bin the grouping actually
    produced (a fat tail bin is legitimately wide).

    The batch mean is still how ``c`` is formed -- it is exact on a homogeneous
    batch -- but on a MIXED batch the mean of 512 shuffled values is just the
    dataset marginal: the same near-constant vector for every batch, telling the
    decoder nothing about the rows it is scoring. That is the bug the grouping
    exists to remove, and it is invisible in the output, so it is checked rather
    than assumed.

    Args:
        vals: ``(B, c_dim)`` per-row conditioning values for one batch.
        cond_name: key into ``data.conditionals``; its ``kind`` picks the defaults.
        spread_tol: largest tolerated max-minus-min, worst component. Default
            ``_DISCRETE_SPREAD_TOL`` for a discrete conditional, else
            ``_CONTINUOUS_SPREAD_TOL``; pass the measured widest bin for the exact
            bound.
        strict: raise instead of warn. Defaults to True for a discrete conditional.
        warned: optional ``{"already": bool}`` used to warn once per pass rather
            than once per batch. Unlike the training collate this runs in the MAIN
            process (``get_posterior_summaries`` calls it on the collated batch), so
            warn-once really is once.
    """
    entry = conditionals.resolve(cond_name)
    if strict is None:
        strict = entry["kind"] == "discrete"
    if spread_tol is None:
        spread_tol = (_DISCRETE_SPREAD_TOL if entry["kind"] == "discrete"
                      else _CONTINUOUS_SPREAD_TOL)

    if vals.shape[0] > 1:
        spread = float((vals.max(dim=0).values - vals.min(dim=0).values).max())
        if spread > spread_tol:
            message = (
                f"conditioning is NOT homogeneous within this inference batch: "
                f"spread={spread:.4g} > tol={spread_tol:.4g} over {vals.shape[0]} "
                f"rows for {cond_name!r}. The batch mean of a mixed batch is the "
                f"dataset marginal, so every posterior in it is computed against a "
                f"decoder that was told nothing about these rows -- pass "
                f"batch_sampler=ConditionGroupedBatchSampler(...) to the DataLoader "
                f"instead of a plain dataset-order loader."
            )
            if strict:
                raise ValueError(message)
            if warned is None or not warned.get("already"):
                print(f"WARNING: {message}")
                if warned is not None:
                    warned["already"] = True

    return vals.mean(dim=0, keepdim=True)


def _make_c_fn(cond_name, device, spread_tol=None, strict=None):
    """Return a callable ``c_fn(batch) -> Tensor (1, c_dim)`` for use with
    ``get_posterior_summaries``.  ``batch`` comes from the default DataLoader
    collate so each element is already a batched tensor.

    The batch mean is kept -- it is the exact conditioning value on a
    conditioning-homogeneous batch, which is what
    :func:`_grouped_inference_batches` now guarantees the loader yields -- but it is
    guarded rather than trusted. See :func:`_c_batch_mean`.
    """
    fi = conditionals.resolve(cond_name)["field_idx"]
    warned = {"already": False}

    def c_fn(batch):
        vals = batch[fi].float()          # (B,) or (B, c_dim)
        if vals.dim() == 1:
            vals = vals.unsqueeze(1)      # (B, 1)
        c = _c_batch_mean(vals, cond_name, spread_tol=spread_tol,
                          strict=strict, warned=warned)
        return c.to(device)               # (1, c_dim)

    return c_fn


def _grouped_inference_batches(dataset, cond_name, batch_size,
                               n_bins=conditionals.DEFAULT_COND_N_BINS, seed=42):
    """Conditioning-homogeneous batches over ``dataset``, plus the permutation they
    imply and the tolerance its batches deserve.

    Reuses ``ConditionGroupedBatchSampler`` -- the very class training batches
    through -- rather than a second implementation of the same grouping, so
    inference and training cannot drift in what "homogeneous" means.

    ``shuffle=False``: inference is one deterministic pass, and there is nothing to
    gain from varying batch membership. ``drop_last=False`` is not optional -- a
    dropped row is a row with no posterior, and every array downstream is joined
    positionally against dataset-order metadata.

    The batch LIST is materialized once here and is meant to be handed to the
    DataLoader as its ``batch_sampler`` (which forbids ``batch_size`` / ``shuffle``
    / ``drop_last`` alongside it). Handing over the same list that produced ``perm``
    is what makes the permutation a fact rather than a re-derivation that could
    disagree with what the loader actually walked.

    Returns:
        batches: ``list[list[int]]`` -- dataset indices per batch, in loader order.
        perm: ``(N,)`` int array; ``perm[k]`` is the dataset index of the k-th row a
            loader walking ``batches`` yields.
        inverse: ``(N,)`` int array; the inverse permutation, so
            ``loader_order_array[inverse]`` is back in dataset order and
            ``perm[inverse] == arange(N)``.
        spread_tol: float or None -- the within-batch spread the homogeneity guard
            should tolerate. None for a discrete conditional (the guard's own
            zero-tolerance default is right); for a continuous one, the width of the
            widest bin the grouping actually produced, measured rather than guessed
            so a fat tail bin does not raise a false alarm while a genuinely mixed
            batch still does.
        summary: one-line description for the log.
    """
    entry = conditionals.resolve(cond_name)
    gids = conditionals.group_ids(dataset, cond_name, n_bins)
    sampler = ConditionGroupedBatchSampler(
        gids, batch_size=batch_size, shuffle=False, drop_last=False, seed=seed,
    )
    batches = [[int(i) for i in b] for b in sampler]

    n = len(dataset)
    perm = (np.concatenate([np.asarray(b, dtype=np.int64) for b in batches])
            if batches else np.zeros(0, dtype=np.int64))
    if len(perm) != n or not np.array_equal(np.sort(perm), np.arange(n)):
        raise RuntimeError(
            f"grouped inference batches do not cover the dataset exactly: "
            f"{len(perm)} indices for {n} rows. Every row must appear exactly once "
            f"or the posterior arrays cannot be put back into dataset order."
        )
    inverse = np.argsort(perm, kind="stable")
    # The round-trip assertion lives in the CODE, not only in the tests. A permuted
    # posterior joined against dataset-order metadata produces plausible figures of
    # the wrong thing, and this repo has been bitten by exactly this class of
    # positional mismatch before (see CLAUDE.md, "prior bugs came from tuple-index
    # shifts breaking consumers").
    if not np.array_equal(perm[inverse], np.arange(n)):
        raise RuntimeError("grouped-batch permutation does not round-trip; refusing "
                           "to embed with an order that cannot be undone.")

    spread_tol, widest = None, None
    if entry["kind"] != "discrete":
        raw = conditionals._raw_values(dataset, cond_name).reshape(-1)
        widest = float(max(np.ptp(raw[gids == g]) for g in np.unique(gids)))
        spread_tol = widest * 1.05 + 1e-6

    sizes = sampler.group_sizes()
    summary = (
        f"grouped batching for {cond_name!r}: {len(sizes)} groups over {n} rows "
        f"(group size min={int(sizes.min())} median={int(np.median(sizes))} "
        f"max={int(sizes.max())}), {len(batches)} batches of at most {batch_size}"
    )
    if widest is not None:
        summary += (f"; widest bin spans {widest:.4g} in the conditioning value, "
                    f"which is the batch-mean error this leaves behind")
    return batches, perm, inverse, spread_tol, summary


# Bumped whenever the MEANING of the cached arrays changes, not merely their
# contents. v2 is the first version to record anything at all: a v1 file carries no
# stamp, was written by the code that conditioned every posterior on the BATCH MEAN
# of a 512-row dataset-order batch, and for a conditional run its numbers are wrong
# rather than stale -- no re-ordering can repair them. See _read_posterior_cache.
_POSTERIOR_CACHE_VERSION = 2


def _read_posterior_cache(cache_path, conditional, n_samples):
    """Load ``posterior_cache.npz`` if it is valid for THIS run, else None.

    The cache is keyed by directory alone -- nothing in the path says which model,
    dataset or conditioning produced it -- so the little that CAN be checked is
    checked here, loudly, and a file that fails is recomputed rather than trusted.

    ROW ORDER IS DATASET ORDER, always, in both the conditional and unconditional
    paths. That is what the arrays mean everywhere else: every consumer indexes them
    against dataset-order metadata, including one outside this repo
    (scripts/dataset_construct/14_fig11_latent_probe.py reads torus_weighted
    straight against the corpus's full_data.npz). The conditional path computes the
    posteriors in grouped-batch order and un-permutes BEFORE caching, so a cache
    never holds loader order.

    A v1 (unstamped) file is REFUSED FOR A CONDITIONAL RUN and accepted for an
    unconditional one. v1 is everything written before the conditional posterior
    pass was grouped, when every row's posterior was computed against the batch
    mean of a 512-row dataset-order batch, i.e. against the dataset marginal: for a
    conditional run those numbers are wrong rather than stale and no re-ordering
    can repair them. An unconditional posterior never involved a conditioning value
    at all, so a v1 file written by one is exactly what this code would write
    today, and the existing full-corpus caches (phases 2-3, hours of GPU each) stay
    usable. The two cases are not confusable in practice: a save_dir lives inside
    its own run directory, and an unconditional analysis pointed at a conditional
    run's directory cannot even load the checkpoint -- c_dim widens the decoder's
    first layer.
    """
    if not os.path.exists(cache_path):
        return None

    cache = np.load(cache_path, allow_pickle=False)

    def reject(reason):
        print(f"  IGNORING the posterior cache at {cache_path}: {reason}. "
              f"Recomputing (and overwriting it) instead.")
        return None

    version = int(cache["format_version"]) if "format_version" in cache.files else 1
    if version != _POSTERIOR_CACHE_VERSION:
        if conditional is not None:
            return reject(
                f"format_version={version}, this build writes "
                f"v{_POSTERIOR_CACHE_VERSION}. A v{version} file was written before "
                f"conditional posteriors were grouped, so every row in it was "
                f"conditioned on the batch mean -- on the dataset marginal, not on "
                f"its own value"
            )
        stamp = f"format v{version} (pre-stamp, unconditional: nothing to invalidate)"
    else:
        want = "" if conditional is None else str(conditional)
        got = str(cache["conditional"])
        if got != want:
            return reject(f"written for conditional={got or None!r} but this run is "
                          f"conditional={conditional!r}")
        stamp = (f"format v{version}, conditional={got or None}, "
                 f"row order={str(cache['row_order'])}")

    torus_weighted = cache["torus_weighted"]
    if len(torus_weighted) != n_samples:
        return reject(f"holds {len(torus_weighted)} rows but this dataset has "
                      f"{n_samples}")

    print(f"Loading cached posteriors from {cache_path} ({stamp})")
    return torus_weighted, cache["aggregated"], cache["weights"]


def _write_posterior_cache(cache_path, conditional, torus_weighted, aggregated, weights):
    """Write the posterior summaries plus the stamp that says what they mean."""
    np.savez(
        cache_path,
        torus_weighted=torus_weighted, aggregated=aggregated, weights=weights,
        format_version=np.array(_POSTERIOR_CACHE_VERSION),
        conditional=np.array("" if conditional is None else str(conditional)),
        # Recorded rather than implied: the conditional path computes these in
        # grouped-batch order, and a reader has no other way to know they were put
        # back before being written.
        row_order=np.array("dataset"),
        n_samples=np.array(len(torus_weighted)),
    )
    print(f"Cached posteriors to {cache_path} "
          f"(format v{_POSTERIOR_CACHE_VERSION}, row order=dataset)")


def _get_sample_c(dataset, index, cond_name, device):
    """Extract the conditioning tensor for a single dataset sample → (1, c_dim).

    A thin alias for ``conditionals.sample_c``, so the conditional-to-tuple-slot
    mapping lives in exactly one place. Its one call site -- the reconstruction
    loop -- no longer uses it: that loop now reads the conditioning value off the
    dataset items it has already loaded, rather than re-indexing the dataset and
    decoding the whole spectrogram a second time to reach one scalar. Kept because
    it is the readable way to ask for one row's ``c`` and the grid-sweep figures
    below are its natural next caller.
    """
    return conditionals.sample_c(dataset, index, cond_name, device=device)


def _style_latent_axis(ax, title=None, xlabel='Latent dimension 1',
                       ylabel='Latent dimension 2'):
    """Common framing for every panel that lives on the [0,1]^2 latent torus."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')


def draw_latent_scatter(
    ax,
    latent_coords,
    color_values,
    label,
    cmap=CMAP_FREQ,
    vmin=None,
    vmax=None,
    color_map=None,
    category_order=None,
    scatter_size=7,
    scatter_alpha=0.4,
    title=None,
    colorbar=True,
    missing_label='missing',
):
    """Draw one 2-D latent scatter onto an existing axis.

    Continuous mode (``color_map`` is None): ``color_values`` is numeric, NaN
    entries are drawn in the shared missing-data grey behind the valid points, and
    a colorbar labelled ``label`` is attached when ``colorbar``.

    Discrete mode (``color_map`` is a ``{category: hex}`` dict): ``color_values``
    is an object array, missing entries likewise drawn behind, and a legend
    carrying per-category counts replaces the colorbar.

    Returns the scatter mappable in continuous mode, else None.
    """
    sc = None
    scatter_size, scatter_alpha = scatter_params_for(
        len(latent_coords), scatter_size, scatter_alpha)
    if color_map is None:
        vals = np.asarray(color_values, dtype=float)
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            # A panel where most samples lack the variable (social distance covers a
            # minority of sessions) is otherwise a field of grey with the real points
            # lost in it, so the missing layer is drawn fainter than the data.
            ax.scatter(
                latent_coords[nan_mask, 0], latent_coords[nan_mask, 1],
                c=MISSING_COLOR, s=scatter_size * 0.8, alpha=scatter_alpha * 0.45,
                rasterized=True, linewidth=0, marker='.', edgecolors='none',
            )
        valid = ~nan_mask
        sort_idx = np.argsort(vals[valid])
        sc = ax.scatter(
            latent_coords[valid][sort_idx, 0],
            latent_coords[valid][sort_idx, 1],
            c=vals[valid][sort_idx],
            cmap=cmap, vmin=vmin, vmax=vmax,
            s=scatter_size, alpha=scatter_alpha,
            rasterized=True, linewidth=0, marker='.', edgecolors='none',
        )
        if colorbar:
            cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.72, pad=0.02)
            cbar.solids.set_alpha(1)
            cbar.set_label(label, fontsize=9)
    else:
        cats = np.asarray(color_values, dtype=object)
        nan_mask = np.array([c is None or (isinstance(c, float) and np.isnan(c))
                             for c in cats])
        if nan_mask.any():
            # A panel where most samples lack the variable (social distance covers a
            # minority of sessions) is otherwise a field of grey with the real points
            # lost in it, so the missing layer is drawn fainter than the data.
            ax.scatter(
                latent_coords[nan_mask, 0], latent_coords[nan_mask, 1],
                c=MISSING_COLOR, s=scatter_size * 0.8, alpha=scatter_alpha * 0.45,
                rasterized=True, linewidth=0, marker='.', edgecolors='none',
            )
        order = category_order if category_order is not None else sorted(color_map)
        # Proxy handles rather than the scatters themselves: the points are drawn
        # small and nearly transparent, and a legend swatch inheriting that is
        # invisible however far markerscale is turned up.
        handles = []
        for cat in order:
            mask = cats == cat
            if not mask.any():
                continue
            ax.scatter(
                latent_coords[mask, 0], latent_coords[mask, 1],
                c=color_map[cat],
                s=scatter_size, alpha=scatter_alpha,
                rasterized=True, linewidth=0, marker='.', edgecolors='none',
            )
            handles.append(mpl.lines.Line2D(
                [], [], marker='o', linestyle='none', markersize=6,
                color=color_map[cat], label=f'{cat} (n={int(mask.sum()):,})'))
        if nan_mask.any():
            handles.append(mpl.lines.Line2D(
                [], [], marker='o', linestyle='none', markersize=6,
                color=MISSING_COLOR,
                label=f'{missing_label} (n={int(nan_mask.sum()):,})'))
        ax.legend(handles=handles, fontsize=8, loc='upper right', framealpha=0.85)

    _style_latent_axis(ax, title=title)
    return sc


def draw_posterior_density(ax, heatmap, title='Aggregated posterior',
                           label='Aggregated posterior', colorbar=True,
                           percentile=95):
    """Draw the aggregated-posterior image on an axis, clipped at a percentile.

    Every figure that shows the posterior goes through here, so the density is
    always the same colormap and the same vmax rule.
    """
    nz = heatmap[heatmap > 0]
    vmax = float(np.percentile(nz, percentile)) if nz.size else None
    im = ax.imshow(
        heatmap, origin='lower', extent=(0, 1, 0, 1), cmap=CMAP_POSTERIOR,
        interpolation='nearest', vmin=0, vmax=vmax, aspect='equal',
    )
    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
        cbar.solids.set_alpha(1)
        cbar.set_label(label, fontsize=9)
    _style_latent_axis(ax, title=title)
    return im


def _grid_shape(n_panels, candidates=(3, 4)):
    """Column count leaving the fewest empty slots, then the fewest rows."""
    best = min(
        candidates,
        key=lambda c: (int(np.ceil(n_panels / c)) * c - n_panels,
                       int(np.ceil(n_panels / c))),
    )
    return best, int(np.ceil(n_panels / best))


def figure_E_grid(panels, save_path, n_cols=None, panel_size=5.0, dpi=200,
                  suptitle='Embedded latents'):
    """Lay every Figure-E panel on one grid, drawn on the same latent map.

    ``panels`` is a list of ``(title, draw_fn)``; ``draw_fn(ax)`` draws one panel.
    Panels whose variable is absent from the dataset are dropped by the caller
    rather than blanked, so a run without behavioural features produces a smaller
    grid instead of a grid of empty boxes. Each panel is lettered so a caption can
    point at one without repeating its title.
    """
    if not panels:
        print('  no Figure-E panels available; skipping figure_E_embedded_grid.png')
        return

    if n_cols is None:
        n_cols, n_rows = _grid_shape(len(panels))
    else:
        n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * panel_size * 1.18, n_rows * panel_size),
    )
    axes = np.atleast_1d(axes).ravel()

    for k, (title, draw_fn) in enumerate(panels):
        ax = axes[k]
        draw_fn(ax)
        ax.text(
            -0.08, 1.04, chr(ord('A') + k), transform=ax.transAxes,
            fontsize=15, fontweight='bold', ha='right', va='bottom',
        )
    for ax in axes[len(panels):]:
        ax.axis('off')

    fig.suptitle(suptitle, fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f'Saved: {os.path.basename(save_path)}  ({len(panels)} panels, '
          f'{n_rows}x{n_cols})')
    plt.close(fig)


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


def _torus_segments(p1, p2):
    """
    Split geodesic p1->p2 on the [0,1]^2 torus into drawable pieces.
    Returns a list of (a, b) pairs, each in [0,1]^2.
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    delta = p2 - p1
    delta -= np.round(delta)  # shortest-path direction on torus

    crossings = set()
    for dim in range(2):
        d = delta[dim]
        if d > 1e-9:  # moving toward upper boundary (1)
            t = (1.0 - p1[dim]) / d
            if 1e-9 < t < 1 - 1e-9:
                crossings.add(t)
        elif d < -1e-9:  # moving toward lower boundary (0)
            t = (0.0 - p1[dim]) / d
            if 1e-9 < t < 1 - 1e-9:
                crossings.add(t)

    ts = [0.0] + sorted(crossings) + [1.0]
    result = []
    for i in range(len(ts) - 1):
        q0 = p1 + ts[i] * delta
        q1 = p1 + ts[i + 1] * delta
        offset = np.floor((q0 + q1) / 2)  # which torus cell the midpoint falls in
        result.append((q0 - offset, q1 - offset))
    return result


def draw_segments_overlay(
    ax,
    latent_coords,
    base_color_values,
    segments_coords,
    segments_dist,
    title,
    direction='decreasing',
    cmap_name=CMAP_SEGMENT,
    max_lines=10,
    scatter_size=5,
    scatter_alpha=0.2,
    vmin=SOCIAL_DIST_RANGE[0],
    vmax=SOCIAL_DIST_RANGE[1],
    colorbar=True,
):
    """Draw duration-coloured scatter with social-distance trajectories on an axis.

    Args:
        latent_coords: (N, 2) array of all latent positions
        base_color_values: (N,) durations for the background scatter
        segments_coords: list of (N_i, 2) arrays, one per segment
        segments_dist: list of (N_i,) social distance arrays, one per segment
        title: panel title
        direction: 'decreasing' or 'increasing' -- controls the annotation text
        cmap_name: colormap for the segment lines (green = close, red = far)
        max_lines: cap on the number of segments drawn
        vmin/vmax: social distance colormap range
    """
    from matplotlib.collections import LineCollection

    scatter_size, scatter_alpha = scatter_params_for(
        len(latent_coords), scatter_size, scatter_alpha)
    vals = np.asarray(base_color_values, dtype=float)
    ax.scatter(
        latent_coords[:, 0], latent_coords[:, 1],
        c=vals, cmap=CMAP_DURATION,
        s=scatter_size, alpha=scatter_alpha,
        rasterized=True, linewidth=0, marker='.', edgecolors='none',
    )

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for coords, dists in zip(segments_coords[:max_lines], segments_dist[:max_lines]):
        all_segs, all_colors = [], []
        for k in range(len(coords) - 1):
            avg_color = (dists[k] + dists[k + 1]) / 2
            for a, b in _torus_segments(coords[k], coords[k + 1]):
                all_segs.append([a, b])
                all_colors.append(avg_color)
        if not all_segs:
            continue
        lc = LineCollection(
            all_segs, cmap=cmap_name, norm=norm,
            linewidth=1, alpha=0.6, zorder=5,
        )
        lc.set_array(np.array(all_colors))
        ax.add_collection(lc)

    _style_latent_axis(ax, title=title)

    if colorbar:
        sm = mpl.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.72, pad=0.02)
        cbar.set_label('Social distance (cm)', fontsize=9)

    # RdYlGn maps the low end of the range to red, and the low end here is 0 cm, so
    # red is close and green is far. The note used to claim the opposite.
    arrow = ('green \u2192 red' if direction == 'decreasing'
             else 'red \u2192 green')
    ax.text(
        0.5, -0.13,
        f'red = close, green = far apart   (over time: {arrow})',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=9, color='dimgray',
    )


def make_segment_video(
    coords,
    dists,
    dataset_indices,
    times,
    session_id,
    full_ds,
    latent_coords_bg,
    durations_bg,
    save_path,
    vmin=0,
    vmax=85,
    scatter_size=5,
    scatter_alpha=0.2,
    fps=5,
    tail_coords=None,
    tail_dists=None,
    tail_dataset_indices=None,
    tail_times=None,
    frame_rate=150,
):
    """
    Animate a single continuous segment on the torus latent space alongside spectrograms.

    Left panel : background scatter + torus-aware trail growing frame by frame.
    Right panel: grid of all N spectrograms in the segment (+ optional tail);
                 current frame is highlighted with a distance-colored border.
    Labels     : session datetime, current USV start/end time (converted to seconds).
    """
    # Writing an .mp4 needs an ffmpeg writer, and samv2_env has none -- matplotlib
    # registers only ['pillow', 'html'] there. Animation.save(writer='ffmpeg') does
    # not report that: it silently falls back to PillowWriter, whose __init__ takes
    # no extra_args, so the save below raises
    #     TypeError: AbstractMovieWriter.__init__() got an unexpected keyword
    #                argument 'extra_args'
    # and kills the whole figure run at figure E7. This went unnoticed because the
    # per-draw datasets never produced a segment longer than min_seg_len, so this
    # function was never called; the full-corpus set is the first input dense enough
    # to reach it. Skip the video and keep every other figure. Install ffmpeg into
    # the environment to get the videos back.
    from matplotlib.animation import writers as _mpl_writers
    if not _mpl_writers.is_available('ffmpeg'):
        print(f"  no ffmpeg writer available (matplotlib has {_mpl_writers.list()}); "
              f"skipping video {os.path.basename(save_path)}")
        return
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation
    from datetime import datetime

    TRAIL_CMAP = 'winter'

    dt = datetime.strptime(session_id, '%Y%m%d_%H%M%S')
    dt_label = dt.strftime('%Y-%m-%d %H:%M:%S')
    seg_start = times[0][0] / frame_rate
    seg_end = times[-1][1] / frame_rate
    n_seg = len(coords)

    # ── Concatenate tail if provided ────────────────────────────────────────
    if tail_coords is not None and len(tail_coords) > 0:
        all_coords = np.concatenate([coords, tail_coords], axis=0)
        all_dists = np.concatenate([dists, tail_dists])
        all_didx = list(dataset_indices) + list(tail_dataset_indices)
        all_times = list(times) + list(tail_times)
    else:
        all_coords = coords
        all_dists = dists
        all_didx = list(dataset_indices)
        all_times = list(times)
    N = len(all_coords)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(TRAIL_CMAP)

    # ── Grid layout for spectrograms ────────────────────────────────────────
    n_cols = min(8, int(np.ceil(np.sqrt(N))))
    n_rows = int(np.ceil(N / n_cols))

    fig_w = 6 + 1.6 * n_cols
    fig_h = max(5, 1.5 * n_rows + 0.6)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f'{dt_label}  |  segment: {seg_start:.2f} s \u2013 {seg_end:.2f} s',
        fontsize=11,
    )

    outer_gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1],
                                  left=0.05, right=0.97, top=0.91, bottom=0.07,
                                  wspace=0.12)
    ax_lat = fig.add_subplot(outer_gs[0, 0])

    right_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=outer_gs[0, 1],
        hspace=0.08, wspace=0.06,
    )
    ax_specs = []
    for r in range(n_rows):
        for c in range(n_cols):
            k = r * n_cols + c
            ax = fig.add_subplot(right_gs[r, c])
            if k < N:
                ax_specs.append(ax)
            else:
                ax.set_visible(False)

    # ── Left panel: static background ──────────────────────────────────────
    bg_vals = np.asarray(durations_bg, dtype=float)
    ax_lat.scatter(
        latent_coords_bg[:, 0], latent_coords_bg[:, 1],
        c=bg_vals, cmap=CMAP_DURATION,
        s=scatter_size, alpha=scatter_alpha,
        rasterized=True, linewidth=0, marker='.', edgecolors='none',
    )
    ax_lat.set_xlim(0, 1)
    ax_lat.set_ylim(0, 1)
    ax_lat.set_aspect('equal')
    ax_lat.set_xlabel('Latent dim 1', fontsize=10)
    ax_lat.set_ylabel('Latent dim 2', fontsize=10)

    trail_artists = []
    dot, = ax_lat.plot([], [], 'wo', markersize=6, markeredgecolor='k',
                       markeredgewidth=0.8, zorder=10)
    time_text = ax_lat.text(
        0.02, 0.97, '', transform=ax_lat.transAxes,
        va='top', ha='left', fontsize=8, color='white',
        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.4),
    )

    # ── Right panel: pre-load all spectrograms into the grid ───────────────
    spec_ims = []
    spec_overlays = []
    for k, ax_s in enumerate(ax_specs):
        spec = full_ds[all_didx[k]][0].squeeze().numpy()
        im = ax_s.imshow(
            spec, aspect='auto', origin='lower',
            cmap=CMAP_SPEC, interpolation='nearest',
            vmin=0, vmax=1,
        )
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        for spine in ax_s.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor('#555555')
        spec_ims.append(im)

        overlay = ax_s.add_patch(
            mpl.patches.Rectangle(
                (0, 0), 1, 1,
                transform=ax_s.transAxes,
                color='black', alpha=0.55,
                zorder=5,
            )
        )
        spec_overlays.append(overlay)

    # ── Title artists for each grid cell (pre-created, updated each frame) ─
    spec_titles = []
    for ax_s in ax_specs:
        t = ax_s.set_title('', fontsize=9, pad=1)
        spec_titles.append(t)

    def _highlight(k):
        border_rgba = cmap_obj(norm(all_dists[k]))
        border_color = border_rgba[:3]
        for j, (ax_s, ov, tt) in enumerate(zip(ax_specs, spec_overlays, spec_titles)):
            if j == k:
                for spine in ax_s.spines.values():
                    spine.set_linewidth(3.0)
                    spine.set_edgecolor(border_color)
                tt.set_text(f'{all_dists[k]:.1f} cm')
                tt.set_color(border_color)
                ov.set_visible(False)
            else:
                for spine in ax_s.spines.values():
                    spine.set_linewidth(0.8)
                    spine.set_edgecolor('#444444')
                tt.set_text('')
                ov.set_visible(True)

    def init():
        dot.set_data([], [])
        time_text.set_text('')
        _highlight(0)
        return [dot, time_text] + spec_ims + spec_overlays + spec_titles

    def update(k):
        for art in trail_artists:
            art.remove()
        trail_artists.clear()

        if k > 0:
            sc = ax_lat.scatter(
                all_coords[:k, 0], all_coords[:k, 1],
                c=all_dists[:k], cmap=TRAIL_CMAP, norm=norm,
                s=50, alpha=0.9, linewidths=0, zorder=5,
            )
            trail_artists.append(sc)

        dot.set_data([all_coords[k, 0]], [all_coords[k, 1]])
        st, et = all_times[k]
        time_text.set_text(f'{st / frame_rate:.2f} s \u2013 {et / frame_rate:.2f} s')

        _highlight(k)

        return [dot, time_text] + spec_ims + spec_overlays + spec_titles + trail_artists

    ani = FuncAnimation(
        fig, update, frames=N,
        init_func=init, blit=False, interval=int(1000 / fps),
    )
    ani.save(save_path, writer='ffmpeg', fps=fps, dpi=150,
             extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    print(f'Saved: {os.path.basename(save_path)}')
    plt.close()


def grid_examples(model, grid_size, save_path, device, c=None):
    """Decode a grid_size x grid_size grid of latent points in [0,1]^2 and plot reconstructions.

    Args:
        c: optional conditioning tensor of shape (1, c_dim).  When provided it is
           tiled to match the grid batch and concatenated to the basis output before
           the decoder — matching the conditional model's forward pass.
    """
    xs = np.linspace(0, 1, grid_size, endpoint=False) + 0.5 / grid_size
    gx, gy = np.meshgrid(xs, xs)
    z = np.stack([gx.ravel(), gy.ravel()], axis=1)
    z_t = torch.tensor(z, dtype=torch.float32, device=device)
    with torch.no_grad():
        basis_out = model.basis(z_t)                     # (grid^2, 2*latent_dim)
        if c is not None:
            c_exp = c.to(device).expand(len(z_t), -1)   # (grid^2, c_dim)
            decoder_input = torch.cat([basis_out, c_exp], dim=-1)
        else:
            decoder_input = basis_out
        recon = model.decoder(decoder_input).cpu().numpy().squeeze(1)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            ax = axes[grid_size - 1 - i, j]  # flip y so row 0 is bottom
            ax.imshow(recon[i * grid_size + j], cmap=CMAP_SPEC, origin='lower', aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f'Grid reconstructions ({grid_size}x{grid_size})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {os.path.basename(save_path)}')
    plt.close()


def figure_H_watershed_variants(posterior, centers, save_path,
                                sigmas=(1, 3, 6, 12, 20),
                                compacts=(0, 0.05, 0.2, 0.5, 2.0)):
    """Watershed sweep over (sigma, compactness), drawn on the aggregated posterior.

    Every cell shows the aggregated posterior the algorithm is segmenting, smoothed
    by that row's sigma, so a boundary can be judged against the density it is
    supposed to follow. The sweep and the shipped segmentation therefore run on the
    same field.

    This used to render as a flat purple square with four corner blobs, because the
    lattice it histogrammed came straight from ``gen_fib_basis``, whose second
    column is unwrapped (it runs to tens of thousands, the model applies the ``% 1``
    itself). Only the handful of lattice points whose raw coordinate happened to
    land inside [0, 1] were binned, so the "posterior" was two points wide and
    watershed segmented a flat field into Voronoi cells. The caller now wraps the
    lattice before histogramming it.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.segmentation import watershed

    res = posterior.shape[0]
    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)

    marker_img_base = np.zeros((res, res), dtype=int)
    for i, (cx, cy) in enumerate(centers):
        px = int(np.clip(cx * res, 0, res - 1))
        py = int(np.clip(cy * res, 0, res - 1))
        marker_img_base[py, px] = i + 1

    fig, axes = plt.subplots(len(sigmas), len(compacts),
                             figsize=(len(compacts) * 4, len(sigmas) * 4))
    for row, sigma in enumerate(sigmas):
        smoothed = gaussian_filter(posterior, sigma=sigma, mode='wrap')
        nz = smoothed[smoothed > 0]
        vmax = float(np.percentile(nz, 95)) if nz.size else None
        for col, compact in enumerate(compacts):
            ax = axes[row, col]
            labels_ws = watershed(-smoothed, markers=marker_img_base.copy(),
                                  compactness=compact)
            ax.imshow(smoothed, origin='lower', extent=(0, 1, 0, 1),
                      cmap=CMAP_POSTERIOR, vmin=0, vmax=vmax, aspect='equal',
                      interpolation='nearest')
            boundary_levels = np.arange(0.5, len(centers) + 1.5)
            ax.contour(xx, yy, labels_ws, levels=boundary_levels,
                       colors=OVERLAY_COLOR, linewidths=1.2)
            ax.scatter(centers[:, 0], centers[:, 1],
                       c=OVERLAY_COLOR, s=60, marker='x', linewidths=1.6, zorder=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f'compact={compact}', fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'\u03c3={sigma}', fontsize=11, fontweight='bold')
    fig.suptitle('Watershed of the aggregated posterior  '
                 '(rows = smoothing \u03c3, cols = compactness)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {os.path.basename(save_path)}')
    plt.close(fig)


def _mask_bin_labels(mask_counts, top_bin=5):
    """Bin mask counts as 1, 2, ... top_bin-1, '<top_bin>+'. Returns (labels, edges)."""
    binned = np.clip(np.asarray(mask_counts, dtype=int), 1, top_bin)
    labels = [str(b) for b in range(1, top_bin)] + [f"{top_bin}+"]
    return binned, labels


def _panel_mse_by_mask_count(ax, mse_arr, mask_counts):
    """Mean reconstruction MSE per SAM mask count."""
    unique_mls = np.sort(np.unique(mask_counts))
    means = [mse_arr[mask_counts == ml].mean() for ml in unique_mls]
    sems = [mse_arr[mask_counts == ml].std() / max(1, np.sqrt((mask_counts == ml).sum()))
            for ml in unique_mls]
    ax.bar(np.arange(len(unique_mls)), means, yerr=sems, capsize=3,
           color=bar_shades(CMAP_MASK_COUNT, len(unique_mls)),
           edgecolor='white', linewidth=0.6)
    ax.set_xticks(np.arange(len(unique_mls)))
    ax.set_xticklabels([str(int(m)) for m in unique_mls])
    ax.set_xlabel('SAM mask count (syllable length bins)', fontsize=11)
    ax.set_ylabel('Mean reconstruction MSE', fontsize=11)
    ax.set_title('By mask count', fontsize=12, fontweight='bold')


def _panel_mse_by_duration(ax, mse_arr, durations, n_dur_bins=5):
    """Mean reconstruction MSE per duration quantile bin."""
    bin_edges = np.percentile(durations, np.linspace(0, 100, n_dur_bins + 1))
    bin_edges[0] -= 1
    bin_edges[-1] += 1
    bin_labels, bin_means, bin_sems = [], [], []
    for b in range(n_dur_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        sel = (durations >= lo) & (durations < hi)
        if sel.sum() == 0:
            continue
        bin_labels.append(f'[{lo:.0f},{hi:.0f})')
        bin_means.append(mse_arr[sel].mean())
        bin_sems.append(mse_arr[sel].std() / max(1, np.sqrt(sel.sum())))
    ax.bar(np.arange(len(bin_means)), bin_means, yerr=bin_sems, capsize=3,
           color=bar_shades(CMAP_DURATION, len(bin_means)),
           edgecolor='white', linewidth=0.6)
    ax.set_xticks(np.arange(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=30, ha='right', fontsize=9)
    ax.set_xlabel('Duration (samples)', fontsize=11)
    ax.set_ylabel('Mean reconstruction MSE', fontsize=11)
    ax.set_title('By duration', fontsize=12, fontweight='bold')


def _panel_mse_by_type(ax, mse_arr, types, present):
    """Mean reconstruction MSE per session type."""
    means = [mse_arr[types == t].mean() for t in present]
    sems = [mse_arr[types == t].std() / max(1, np.sqrt((types == t).sum()))
            for t in present]
    ax.bar(np.arange(len(present)), means, yerr=sems, capsize=3,
           color=[session_type_color(t) for t in present],
           edgecolor='white', linewidth=0.6)
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([f'{t}\nn={int((types == t).sum()):,}' for t in present],
                       fontsize=9)
    ax.set_ylabel('Mean reconstruction MSE', fontsize=11)
    ax.set_title('By session type', fontsize=12, fontweight='bold')


def _panel_mse_by_type_and_mask(ax, mse_arr, types, present, binned, bin_labels,
                                top_bin=5):
    """Mean reconstruction MSE per session type x mask-count bin.

    This is the panel that shows whether a draw rule bought anything in the bins it
    up-weighted, so it gets the widest slot in the grid.
    """
    width = 0.8 / len(present)
    for i, t in enumerate(present):
        cell_means, cell_sems = [], []
        for b in range(1, top_bin + 1):
            sel = (types == t) & (binned == b)
            cell_means.append(mse_arr[sel].mean() if sel.any() else np.nan)
            cell_sems.append(mse_arr[sel].std() / np.sqrt(sel.sum())
                             if sel.sum() > 1 else 0.0)
        ax.bar(np.arange(top_bin) + i * width - 0.4 + width / 2, cell_means,
               width=width, yerr=cell_sems, capsize=2, label=t,
               color=session_type_color(t), edgecolor='white', linewidth=0.6)
    ax.set_xticks(np.arange(top_bin))
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel('SAM mask count', fontsize=11)
    ax.set_ylabel('Mean reconstruction MSE', fontsize=11)
    ax.set_title('By session type x mask count', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)


def figure_recon_mse(mse_arr, mask_counts, durations, save_dir, session_types=None,
                     n_dur_bins=5, top_bin=5, extra_columns=None):
    """Every reconstruction-MSE breakdown on one grid.

    Panels: mask count, duration, session type, session type x mask-count bin. The
    two session-type panels appear only when the dataset carries a ``session_type``
    column -- the multi-condition sets from usv-playpen's build-qlvm-training-set
    do, the older monolithic ones do not -- so a set without it produces a 1x2 row
    rather than a grid with two empty boxes.

    The raw per-spectrogram numbers go to ``recon_mse_breakdown.npz`` alongside, so
    a cross-run comparison never has to be read back off the pixels.
    """
    types = None
    present = []
    if session_types is not None:
        types = np.asarray(session_types, dtype=object)
        present = [t for t in SESSION_TYPE_ORDER if (types == t).any()]
        present += sorted({str(t) for t in np.unique(types)}
                          - set(SESSION_TYPE_ORDER) - {"None"})

    binned, bin_labels = _mask_bin_labels(mask_counts, top_bin=top_bin)

    if present:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
        print("  no session types present; reconstruction grid is mask count + "
              "duration only")

    _panel_mse_by_mask_count(axes[0], mse_arr, mask_counts)
    _panel_mse_by_duration(axes[1], mse_arr, durations, n_dur_bins=n_dur_bins)
    if present:
        _panel_mse_by_type(axes[2], mse_arr, types, present)
        _panel_mse_by_type_and_mask(axes[3], mse_arr, types, present,
                                    binned, bin_labels, top_bin=top_bin)

    for k, ax in enumerate(axes):
        ax.text(-0.08, 1.05, chr(ord('A') + k), transform=ax.transAxes,
                fontsize=14, fontweight='bold', ha='right', va='bottom')

    fig.suptitle('Reconstruction MSE', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(save_dir, 'figure_recon_mse.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print('Saved: figure_recon_mse.png')
    plt.close(fig)

    extra_columns = extra_columns or {}
    # session_types is always written at full length, 'unknown' where the dataset
    # does not carry the column. 09_fig8_qlvm_loss_compare.py indexes `mse` with
    # `session_types == <type>`, so a zero-length column here would raise
    # IndexError rather than select nothing.
    np.savez(
        os.path.join(save_dir, "recon_mse_breakdown.npz"),
        mse=mse_arr, mask_counts=np.asarray(mask_counts),
        durations=np.asarray(durations), mask_bin=binned,
        session_types=(types.astype(str) if types is not None
                       else np.full(len(mse_arr), "unknown", dtype="<U7")),
        **extra_columns,
    )
    print("Saved: recon_mse_breakdown.npz "
          f"(columns: mse, mask_counts, durations, mask_bin, session_types"
          f"{''.join(', ' + k for k in extra_columns)})")


def analyze_mouse_latents(
    model_path,
    dataloc,
    save_dir,
    lattice_m=20,
    bandwidth=0.1,
    batch_size=1,
    freq_range_khz=(30, 120),   # usv-playpen generate_spectrograms defaults
    scatter_size=5,
    scatter_alpha=0.2,
    total_samples=None,
    beh_features_path="/jukebox/falkner/Dexter/vocal_beh/data/full_dataset/utils/usv_beh_features.pkl",
    consec_threshold=150,
    min_seg_len=20,
    grid_size=15,
    n_per_cluster=16,
    cluster_sample_figures=True,
    watershed_max_clusters=10,
    sample_from_centroid=False,
    cache_posteriors=False,
    # reconstruction diagnostics
    compute_recon=True,
    recon_batch_size=64,
    n_dur_bins=5,
    # mean-shift algorithm selection
    use_fast_mean_shift=False,
    ms_k_neighbors=8,
    ms_seed_percentile=50,
    ms_max_iter=300,
    ms_tol=1e-5,
    # conditional model support
    conditional=None,         # a data.conditionals key, or None for unconditional
    mask_count_classes=None,  # slot-5 one-hot width; None = derive / dataset default
    cond_n_bins=conditionals.DEFAULT_COND_N_BINS,  # quantile bins for grouped batching
    # data filtering
    filter_mask=True,
    lo=1,
    hi=8,
    apply_mask=None,
    session_conditions=None,
):
    """
    Generate latent space analysis figures for mouse vocalization model.

    Args:
        model_path: Path to trained model checkpoint (.tar file)
        dataloc: Path to mouse data directory
        save_dir: Directory to save output figures
        lattice_m: Fibonacci lattice parameter (m=20 gives ~10K points)
        bandwidth: Bandwidth for mean-shift clustering
        batch_size: Batch size for the posterior and metadata dataloaders.
            With a conditional model the posterior batches are drawn grouped by
            conditioning value, so a large batch is now CORRECT as well as fast:
            every row in a batch shares the ``c`` its posterior is computed
            against. It used to be neither -- the batch mean of a dataset-order
            batch is the dataset marginal -- which is why the default is still the
            safe-but-slow 1 rather than a value tuned for the old behaviour.
        freq_range_khz: (min, max) frequency range in kHz of the spectrogram's
            frequency axis, used to convert mean frequency and bandwidth out of bin
            units. Default (30, 120) matches usv-playpen's generate_spectrograms
            min_freq/max_freq. Pass None to leave both in bins (the colorbars then
            say 'bins' rather than 'kHz').
        scatter_size: Point size for scatter plot (default 3)
        scatter_alpha: Transparency for scatter plot (default 0.3)
        beh_features_path: Path to pkl with {session_id -> DataFrame(avg_social_distance, emitter_sex)}
        cache_posteriors: If True, save/load posterior summaries to/from
            save_dir/posterior_cache.npz. The file is keyed by directory alone, so
            it carries a format version and the name of the conditional it was
            written for, and a file that does not match this run is ignored and
            recomputed rather than trusted. A cache predating that stamp is refused
            for a CONDITIONAL run -- it was written when every conditional posterior
            was computed against the batch mean, so its contents are wrong rather
            than merely stale -- and accepted for an unconditional one, which never
            had a conditioning value to get wrong. Rows are always in DATASET order.
        compute_recon: If True, compute round-trip MSE and save bar charts
        recon_batch_size: Batch size for reconstruction MSE computation
        n_dur_bins: Number of duration quantile bins for the bar chart
        use_fast_mean_shift: If True, use run_mean_shift_fast (lattice-based) instead of run_mean_shift
        ms_k_neighbors: k-neighbor size for local-max seed detection (fast mean-shift only)
        ms_seed_percentile: Weight percentile threshold for seed selection; 0=all local maxima,
            50=above-median local maxima. Higher = fewer seeds, faster, may miss shallow modes.
        ms_max_iter: Max iterations for fast mean-shift
        ms_tol: Convergence threshold relative to bandwidth (fast mean-shift only)
        conditional: name of a conditioning variable from ``data.conditionals``
            (the registry the training driver reads), or None for unconditional
            models. When set the decoder's first linear layer is widened by c_dim
            and conditioning tensors are extracted per-batch from the dataset.
            Choices: "mask_count" (5-D one-hot, 1/2/3/4/5+), "mask_count8" (the
            legacy 8-D width the April-2025 checkpoints were trained with),
            "duration" (scalar), "mean_freq" (scalar). It MUST match what the
            checkpoint was trained with -- c_dim sets the decoder's input width,
            which is not something loading the state dict can silently fix.
        mask_count_classes: one-hot width ``mouse_data`` builds for slot 5 of its
            tuple. Leave None: for a mask-count conditional it is derived from
            ``conditional`` (5 for "mask_count", 8 for "mask_count8") because that
            width IS the decoder's c_dim, and otherwise the dataset keeps its own
            default. Set it only to reproduce a run whose dataset was built with a
            non-default width.
        cond_n_bins: Quantile bins a CONTINUOUS conditional ("duration",
            "mean_freq") is grouped into for the posterior and reconstruction
            passes. Ignored for a discrete conditional, which groups by class, and
            for an unconditional run. The default matches the training driver's, so
            inference groups the rows exactly as training did. Raising it tightens
            the only approximation left -- the batch mean within a bin -- at a cost
            of roughly one extra lattice decode per extra bin, which is nothing
            against one decode per batch; the log line reports the widest bin the
            grouping actually produced so the size of that approximation is visible.
        apply_mask: Override whether spectrograms are multiplied by their mask.
            Leave None to let the dataset decide -- an unmasked set built with
            --no-apply-mask must be embedded unmasked, or the reconstruction
            numbers describe a different input than the model was trained on.
        session_conditions: Optional path to a session_id,condition CSV. Refines
            session_type by experimental condition before any figure reads it --
            build-qlvm-training-set derives session_type from subject sex alone, so
            mute-female courtship sessions otherwise sit inside 'MF' next to the
            intact-partner ones with nothing able to separate them.
        filter_mask: If True, filter dataset to syllables whose masks_len is in [lo, hi].
        lo: Lower bound (inclusive) on masks_len when filter_mask=True.
        hi: Upper bound (inclusive) on masks_len when filter_mask=True.
    """

    # Setup
    os.makedirs(save_dir, exist_ok=True)

    # resolve() raises with the full list of valid names; c_dim_of() returns 0 for
    # None, which is exactly the unconditional decoder's extra input width.
    if conditional is not None:
        conditionals.resolve(conditional)
    c_dim = conditionals.c_dim_of(conditional)
    if conditional is not None:
        print(f"Conditional mode: {conditionals.describe(conditional)}")

    # A mask-count conditional feeds slot 5 of the dataset tuple straight into the
    # decoder, so the one-hot width mouse_data builds IS c_dim. Derive it from the
    # conditional rather than trusting the dataset default to agree: "mask_count"
    # needs 5 and the legacy "mask_count8" needs 8, and a disagreement hands the
    # decoder a vector of the wrong width instead of raising.
    if mask_count_classes is None and conditional is not None:
        if conditionals.resolve(conditional)["field_idx"] == 5:
            mask_count_classes = conditionals.mask_count_classes_for(conditional)

    # Load behavioral features (session_id -> DataFrame)
    import pickle
    beh_features = {}
    if beh_features_path and os.path.exists(beh_features_path):
        with open(beh_features_path, "rb") as f:
            beh_features = pickle.load(f)
        print(f"Loaded behavioral features for {len(beh_features)} sessions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading mouse data...")
    # train_dict, val_dict = load_mouse_data(dataloc)
    # ## Using training data for better coverage
    # test_ds = mouse_data(train_dict, masks_len_range=(1, 8), equal_sampling=True, max_samples=max_samples)
    
    full_dict = load_full_mouse_data(dataloc)
    # Pass mask_count_classes only when something actually asked for a width, so
    # the unconditional path stays exactly the call it has always been.
    ds_kwargs = ({} if mask_count_classes is None
                 else {"mask_count_classes": int(mask_count_classes)})
    full_ds = mouse_data(full_dict, filter_mask=filter_mask, lo=lo, hi=hi,
                         apply_mask=apply_mask,
                         sampling_strategy='subsample', total_samples=total_samples,
                         **ds_kwargs)

    # Refine session_type from an experimental-condition table, before any figure
    # reads it. build-qlvm-training-set derives session_type from subject SEX alone
    # (dataset_session_types.py), so conditions that share a pairing collapse: a
    # mute-female courtship session is male+female and lands in 'MF' next to the
    # intact-partner sessions, with nothing downstream able to tell them apart.
    # The CSV is authoritative (it is the source session lists), so it refines
    # rather than guesses -- 'MF' becomes 'MF_mute' / 'MF_intact'.
    if session_conditions:
        full_ds.session_types = _refine_session_types(
            full_ds.session_types, full_ds.session_ids, session_conditions)


    n_workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4
    test_loader = DataLoader(full_ds, num_workers=n_workers, shuffle=False, batch_size=batch_size)

    # Reconstruct model architecture (must match training)
    print("Loading model...")
    latent_dim = 2

    # Architecture must match training (bartul_mouse.py or bartul_mouse_cond.py).
    # When a conditional is used, the first linear layer is widened by c_dim.
    # Architecture comes from the checkpoint rather than a literal copied out of
    # the driver: build_for_checkpoint reads which head the weights were trained
    # with, so a phase 0 to 4 checkpoint and a ReLU-head one both load here.
    decoder, decoder_head = build_for_checkpoint(model_path, latent_dim, c_dim=c_dim)
    print(f"Decoder head from the checkpoint: {decoder_head!r}")

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

    # Compute posteriors for all test samples (streaming — never materializes full matrix)
    #
    # gen_fib_basis returns the lattice UNWRAPPED: the second column is
    # arange(n) * fib(m-1) / n, which for m=24 runs to ~28656, and the model applies
    # the `% 1` itself on every forward pass. Anything here that treats a lattice
    # coordinate as a position on the [0,1]^2 torus -- the aggregated-posterior
    # histogram below -- has to wrap it first, or it keeps only the two points whose
    # raw coordinate happens to land inside the unit square.
    lattice_np = lattice.numpy() % 1.0
    cache_path = os.path.join(save_dir, 'posterior_cache.npz')
    cached = (_read_posterior_cache(cache_path, conditional, len(full_ds))
              if cache_posteriors else None)
    if cached is not None:
        torus_weighted, aggregated, weights = cached
    else:
        print("Computing posteriors...")
        if conditional is None:
            # The unconditional path, unchanged: the dataset-order loader built
            # above, no grouping and no permutation. There is no ``c`` here for a
            # batch to be homogeneous about.
            posterior_loader, posterior_inverse, c_fn = test_loader, None, None
        else:
            # One lattice decode per batch, exactly as before -- but every row in
            # the batch now shares the ``c`` it is decoded against, so the batch
            # mean IS that row's value (discrete) or a point inside the one
            # quantile bin the rows came from (continuous).
            batches, _perm, posterior_inverse, spread_tol, summary = \
                _grouped_inference_batches(full_ds, conditional, batch_size,
                                           n_bins=cond_n_bins)
            print(f"  {summary}")
            c_fn = _make_c_fn(conditional, device, spread_tol=spread_tol)
            # batch_sampler owns batching entirely: DataLoader forbids passing
            # batch_size / shuffle / drop_last alongside it. The list handed over is
            # the same object `posterior_inverse` was derived from, so the order
            # walked and the order inverted cannot disagree.
            posterior_loader = DataLoader(full_ds, num_workers=n_workers,
                                          batch_sampler=batches)

        torus_weighted, aggregated, weights = get_posterior_summaries(
            model, lattice, posterior_loader, binary_lp, c_fn=c_fn
        )

        if posterior_inverse is not None:
            # get_posterior_summaries vstacks/concatenates its per-batch outputs, so
            # these two come back in LOADER order. Put them back into dataset order
            # here, before anything else touches them: latent_coords, every metadata
            # column, the recon breakdown, each figure and the external probe in
            # scripts/dataset_construct/ all join them positionally on dataset index.
            #
            # `aggregated` is deliberately NOT re-indexed, and this is not an
            # oversight to be "fixed" later: it is the sum over all samples of the
            # posterior at each LATTICE point -- shape (n_lattice,), not
            # (n_samples,) -- so it is order-independent by construction. Permuting
            # it would scramble the lattice, not unscramble the samples.
            assert len(torus_weighted) == len(full_ds) == len(posterior_inverse), (
                f"posterior returned {len(torus_weighted)} rows for "
                f"{len(full_ds)} dataset rows; refusing to un-permute a length "
                f"mismatch")
            torus_weighted = torus_weighted[posterior_inverse]
            weights = weights[posterior_inverse]

        if cache_posteriors:
            _write_posterior_cache(cache_path, conditional,
                                   torus_weighted, aggregated, weights)

    # Posterior mean embedding (torus-aware, continuous)
    latent_coords = torus_reverse(torus_weighted, dim=2)      # (n_samples, 2)
    
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

    # Compute mean frequency, bandwidth, mask counts, and durations in one pass
    print("Computing mean frequencies and extracting metadata...")
    mean_freqs_list = []
    bandwidths_list = []
    mask_counts_list = []
    durations_list = []

    H = None  # infer freq bin count from first batch
    for batch in tqdm(test_loader, desc="metadata"):
        # batch: (spec, ml, duration, mask, spec_id)
        spec_batch = batch[0]                          # (B, 1, H, W)
        spec = spec_batch.squeeze(1).float()           # (B, H, W)
        if H is None:
            H = spec.shape[1]
            freq_bins = torch.arange(H, dtype=torch.float32)

        freq_profile = spec.sum(dim=2)                 # (B, H) — sum over time
        total = freq_profile.sum(dim=1).clamp(min=1e-10)
        mean_freq = (freq_profile * freq_bins).sum(dim=1) / total   # (B,)
        mean_freqs_list.append(mean_freq.numpy())

        # Spectral bandwidth = energy-weighted standard deviation of the frequency
        # profile about that centroid: how spread out in frequency the syllable is,
        # in the SAME units as mean frequency. A pure whistle comes out near zero, a
        # harmonic stack or a broadband call comes out wide. Computed here rather
        # than from the SAM mask so it does not inherit the mask's segmentation.
        deviation = freq_bins.unsqueeze(0) - mean_freq.unsqueeze(1)   # (B, H)
        variance = (freq_profile * deviation ** 2).sum(dim=1) / total
        bandwidths_list.append(torch.sqrt(variance.clamp(min=0)).numpy())

        mask_counts_list.append(batch[1].numpy())  # masks_len = SAM mask count (time bins)

        durations_list.append(batch[2].numpy())

    mean_freqs = np.concatenate(mean_freqs_list)
    bandwidths  = np.concatenate(bandwidths_list)
    mask_counts = np.concatenate(mask_counts_list)
    durations   = np.concatenate(durations_list)

    # Both quantities come out of the loop in frequency-BIN units. Convert them to
    # kHz here, or the panels carry a kHz colorbar over bin indices -- which is what
    # they did until this was fixed, and it is invisible: bin 0..127 over a 30-120
    # kHz band lands in the same numeric range the label implies.
    #
    # The axis is usv-playpen's own: generate_spectrograms band-limits the STFT to
    # [min_freq, max_freq] and resamples onto num_freq_bins rows, then writes the
    # axis as linspace(min_freq, max_freq, num_freq_bins) -- see
    # usv_playpen/processing/generate_spectrograms.py. Defaults there are 30 kHz,
    # 120 kHz and 128 bins, so one row spans ~709 Hz. Mean frequency is a POSITION
    # on that axis and takes the offset; bandwidth is a WIDTH and takes only the
    # scale.
    if freq_range_khz is not None:
        f_lo, f_hi = float(freq_range_khz[0]), float(freq_range_khz[1])
        khz_per_bin = (f_hi - f_lo) / max(1, H - 1)
        mean_freqs = f_lo + mean_freqs * khz_per_bin
        bandwidths = bandwidths * khz_per_bin
        print(f"  frequency axis: {H} bins over {f_lo:g}-{f_hi:g} kHz "
              f"({khz_per_bin * 1000:.0f} Hz/bin)")
    print(f"  mean frequency: {mean_freqs.min():.1f}-{mean_freqs.max():.1f}, "
          f"bandwidth: {bandwidths.min():.1f}-{bandwidths.max():.1f}")

    # ============= Aggregated posterior density =============
    # Built once here, before any figure draws it, so the Figure-E grid, Figure FG
    # and the Figure-H watershed all show the same density.

    # Build the aggregated posterior density as a 2D histogram.
    #
    # The description in the paper ("aggregated posterior density as a 2D histogram,
    # Fig. 5B") is the density on the latent torus obtained by *marginalizing* the
    # per-sample posteriors over the dataset. There are two equivalent ways to
    # obtain it, with different numerical trade-offs:
    #
    #   (A) 'samples'  — histogram of per-sample posterior-mean latent coords
    #                    (`latent_coords`, one point per test sample). Dense, robust,
    #                    and independent of lattice resolution. Loses within-sample
    #                    posterior spread (collapses each posterior to its mean).
    #   (B) 'lattice'  — histogram2d of the lattice points weighted by `aggregated`
    #                    = sum_s p(z_j | x_s). This is the exact marginal on the
    #                    lattice. It needs `lattice_np` wrapped onto the torus (see
    #                    above) and a torus-wrap Gaussian blur; without the wrap it
    #                    collapsed to two points and every watershed run on it came
    #                    out as plain Voronoi cells around the markers.
    #   (C) 'kde'      — same weights as (B), but splatted onto the grid via a
    #                    Gaussian kernel so the sparse lattice contribution is
    #                    smoothed out. Effectively (B) + heavy smoothing.
    #
    # We default to (A) because it reproduces the "2D histogram" wording literally
    # and gives a clean dense image; (B)+smoothing is computed alongside for the
    # watershed step below so clusters snap to the true posterior mass and not
    # just to where samples happened to land.
    from scipy.ndimage import gaussian_filter

    res = 200
    edges = np.linspace(0, 1, res + 1)

    # (A) Sample-space histogram — the image we actually plot.
    heatmap_samples, _, _ = np.histogram2d(
        latent_coords[:, 0], latent_coords[:, 1],
        bins=[edges, edges],
    )
    heatmap_samples = heatmap_samples.T  # rows=y, cols=x for imshow

    # (B) Lattice-weighted histogram, smoothed with a torus-wrap Gaussian. Used for
    # the watershed segmentation further down and as the background of Figure H.
    heatmap_lattice, _, _ = np.histogram2d(
        lattice_np[:, 0], lattice_np[:, 1],
        bins=[edges, edges], weights=aggregated,
    )
    heatmap_lattice = gaussian_filter(heatmap_lattice.T, sigma=2.0, mode='wrap')

    # Light smoothing on the sample histogram too — purely cosmetic, keeps the
    # image readable without hiding real structure.
    heatmap = gaussian_filter(heatmap_samples, sigma=1.5, mode='wrap')


    # ============= Reconstruction MSE bar charts =============
    if compute_recon:
        print("\nComputing reconstruction MSE...")
        from train.losses import binary_lp as _lp_fnc
        # Whole-image MSE is NOT comparable between a masked and an unmasked run:
        # a masked target is ~97% exact zeros, which is a far easier image to fit.
        # The in-mask error is measured on the same pixels either way, so it is the
        # metric that survives the comparison. The out-of-mask error is kept too --
        # for a masked run it measures how well the model reproduces the zeroing,
        # and for an unmasked run how well it reproduces the background.
        n_rows = len(full_ds)
        # The SAME grouping as the posterior pass, for the same reason: round_trip
        # goes through posterior_probability, which decodes one lattice against one
        # ``c``, so a contiguous dataset-order chunk of 64 rows was reconstructed
        # from a decoder conditioned on the mean of 64 unrelated conditioning values
        # -- the same batch-mean bug, in the numbers that go into
        # recon_mse_breakdown.npz. Unconditionally there is nothing to group, so the
        # batches stay the contiguous chunks they have always been.
        if conditional is None:
            recon_batches = [list(range(s, min(s + recon_batch_size, n_rows)))
                             for s in range(0, n_rows, recon_batch_size)]
            recon_spread_tol = None
        else:
            recon_batches, _, _, recon_spread_tol, recon_summary = \
                _grouped_inference_batches(full_ds, conditional, recon_batch_size,
                                           n_bins=cond_n_bins)
            print(f"  recon {recon_summary}")
        # Filled BY DATASET INDEX rather than appended in loader order, so the
        # grouping never reaches the arrays: these are joined positionally against
        # mask_counts, durations, session_types and spec_ids below. NaN-initialized
        # so a row that never got written is caught rather than read as zero.
        mse_arr = np.full(n_rows, np.nan, dtype=np.float32)
        mse_in_arr = np.full(n_rows, np.nan, dtype=np.float32)
        mse_out_arr = np.full(n_rows, np.nan, dtype=np.float32)
        recon_warned = {"already": False}
        fi_recon = (conditionals.resolve(conditional)["field_idx"]
                    if conditional is not None else None)
        model.eval()
        with torch.no_grad():
            for rows in tqdm(recon_batches, desc="recon MSE"):
                items = [full_ds[i] for i in rows]
                specs = torch.stack([it[0] for it in items]).to(torch.float32).to(device)
                if conditional is not None:
                    # Read the conditioning value off the items already loaded
                    # instead of re-indexing the dataset, which would decode and
                    # renormalize the whole spectrogram a second time to read one
                    # scalar. _c_batch_mean is the same guarded collapse the
                    # posterior pass uses.
                    c_vals = torch.stack([
                        it[fi_recon].float() if it[fi_recon].dim() > 0
                        else it[fi_recon].float().unsqueeze(0)
                        for it in items
                    ]).to(device)                               # (B, c_dim)
                    c_batch = _c_batch_mean(c_vals, conditional,
                                            spread_tol=recon_spread_tol,
                                            warned=recon_warned)   # (1, c_dim)
                    recon = model.round_trip(lattice.to(device), specs, _lp_fnc, c=c_batch)
                else:
                    recon = model.round_trip(lattice.to(device), specs, _lp_fnc)
                squared = (recon.cpu() - specs.cpu()) ** 2          # (B, 1, H, W)
                idx = np.asarray(rows, dtype=np.int64)
                mse_arr[idx] = squared.mean(dim=(1, 2, 3)).numpy()

                inside = (torch.stack([it[6] for it in items]).unsqueeze(1) > 0.5)
                n_in = inside.sum(dim=(1, 2, 3)).clamp(min=1)
                n_out = (~inside).sum(dim=(1, 2, 3)).clamp(min=1)
                mse_in_arr[idx] = ((squared * inside).sum(dim=(1, 2, 3)) / n_in).numpy()
                mse_out_arr[idx] = ((squared * ~inside).sum(dim=(1, 2, 3)) / n_out).numpy()
        model.eval()  # keep in eval for subsequent figure generation
        # Every row must have been written exactly once. An unfilled slot means the
        # batching dropped a row, which nothing downstream would notice.
        assert not np.isnan(mse_arr).any(), (
            f"{int(np.isnan(mse_arr).sum())} of {n_rows} rows got no reconstruction")
        print(f"  Mean MSE: {mse_arr.mean():.4f}  (std {mse_arr.std():.4f})")
        print(f"  Mean MSE inside the SAM mask:  {mse_in_arr.mean():.4f}")
        print(f"  Mean MSE outside the SAM mask: {mse_out_arr.mean():.4f}")
        figure_recon_mse(
            mse_arr, mask_counts, durations, save_dir,
            session_types=getattr(full_ds, 'session_types', None),
            n_dur_bins=n_dur_bins,
            extra_columns={"mse_in_mask": mse_in_arr, "mse_out_mask": mse_out_arr,
                           "apply_mask": np.array(full_ds.apply_mask),
                           # spec_id makes a cross-run comparison verifiable rather
                           # than an assumption about row order.
                           "spec_id": np.asarray(full_ds.spec_ids, dtype=str)})

    print(f"Number of samples: {len(latent_coords)}")
    print(f"Latent coords range: X=[{latent_coords[:, 0].min():.3f}, {latent_coords[:, 0].max():.3f}], Y=[{latent_coords[:,1].min():.3f}, {latent_coords[:, 1].max():.3f}]")
    print(f"Unique points: {len(np.unique(latent_coords, axis=0))}")

    # ============= FIGURE E: one grid, every colouring of the same latent map =====
    # Each panel is the same embedding drawn against a different variable, so they
    # only mean anything side by side. They used to be seven separate files at
    # seven slightly different sizes, which is why they were never compared.
    print("\nPreparing Figure E panels...")
    freq_unit = 'kHz' if freq_range_khz is not None else 'bins'

    # ---- condition / session type -------------------------------------------- #
    lone_male_ids = {
        "20250912_155546",
        "20250912_170514",
        "20250919_145712",
        "20250921_155753",
        "20250927_135343",
    }

    # The multi-condition sets built by usv-playpen carry a real session_type column,
    # so use it; only fall back to parsing spec_id for the older monolithic sets
    # whose ids encode the condition.
    #
    # The legacy parse expects "YYYYMMDD_HHMMSS_cond_avg_idx" and reads parts[2] as
    # the condition. build-qlvm-training-set writes "{session_id}_{row_index}"
    # instead, so parts[2] is a row number and EVERY label came out "Unknown" -- an
    # empty figure with no error. Hence the explicit check below rather than a
    # silent write.
    raw_spec_ids = full_ds.spec_ids  # list of str, same length as latent_coords
    session_types = getattr(full_ds, "session_types", None)
    if session_types is not None:
        type_values = np.array([str(t) for t in session_types], dtype=object)
        cond_values, cond_colors, cond_order = None, None, None
        print("  condition read from the dataset's session_type column")
    else:
        type_values = None
        cond_labels = []
        for sid in raw_spec_ids:
            if sid is None:
                cond_labels.append("Unknown")
                continue
            parts = sid.split("_")
            # Format: YYYYMMDD_HHMMSS_cond_avg_idx
            datetime_str = f"{parts[0]}_{parts[1]}"
            cond = parts[2] if len(parts) > 2 else "Unknown"
            if cond == "ephys":
                cond = "Lone-Male" if datetime_str in lone_male_ids else "Male-Female"
            cond_labels.append(cond)
        cond_values = np.array(cond_labels, dtype=object)
        cond_colors, cond_order = COND_COLORS, COND_ORDER
        print("  condition parsed from spec_id (no session_type column present)")
        # Refuse to draw an empty panel. An unparseable id set used to produce a
        # blank figure and a silent "0 samples" for every condition.
        if int(np.isin(cond_values, COND_ORDER).sum()) == 0:
            print(f"  no sample matched {COND_ORDER}; labels seen: "
                  f"{sorted(set(cond_values))[:6]} -- dropping the condition panel")
            cond_values = None

    # ---- behavioural features from the pkl ----------------------------------- #
    # Parse session_id and row_idx from each spec_id: YYYYMMDD_HHMMSS_..._idx
    social_distances = []
    emitter_sexes = []
    for sid in raw_spec_ids:
        if sid is None:
            social_distances.append(np.nan)
            emitter_sexes.append(None)
            continue
        parts = sid.split("_")
        session_id = f"{parts[0]}_{parts[1]}"
        row_idx = int(parts[-1])
        if session_id not in beh_features:
            social_distances.append(np.nan)
            emitter_sexes.append(None)
        else:
            df = beh_features[session_id]
            if row_idx not in df.index:
                social_distances.append(np.nan)
                emitter_sexes.append(None)
            else:
                row = df.loc[row_idx]
                social_distances.append(row["avg_social_distance"])
                emitter_sexes.append(row["emitter_sex"])

    social_distances = np.array(social_distances, dtype=float)
    emitter_sexes = np.array(emitter_sexes, dtype=object)

    n_with_dist = int(np.sum(~np.isnan(social_distances)))
    n_with_sex = int(np.sum(emitter_sexes != None))
    print(f"  behavioural features: {n_with_dist:,} samples with social distance, "
          f"{n_with_sex:,} with emitter sex")

    # ---- continuous social-distance segments --------------------------------- #
    print("  finding continuous segments...")
    import pandas as pd
    from collections import defaultdict

    # Group dataset items by session
    session_to_items = defaultdict(list)
    for i, sid in enumerate(raw_spec_ids):
        if sid is None:
            continue
        parts = sid.split("_")
        session_id = f"{parts[0]}_{parts[1]}"
        row_idx = int(parts[-1])
        session_to_items[session_id].append((i, row_idx))

    all_seg_coords, all_seg_dist, all_seg_trend = [], [], []
    all_seg_didx, all_seg_times, all_seg_session = [], [], []

    for session_id, items in session_to_items.items():
        if session_id not in beh_features:
            continue
        df = beh_features[session_id]
        timed = []
        for dataset_idx, row_idx in items:
            if row_idx not in df.index:
                continue
            row = df.loc[row_idx]
            st = row.get('start_times', np.nan)
            et = row.get('end_times', np.nan)
            sd = row.get('avg_social_distance', np.nan)
            if pd.isna(st) or pd.isna(sd):
                continue
            timed.append((float(st), float(et), float(sd), dataset_idx))
        if len(timed) < 2:
            continue
        timed.sort(key=lambda x: x[0])

        def _split_monotone_chunks(chunk):
            """Split a consecutive chunk at turning points into strictly monotone sub-chunks."""
            if len(chunk) < 2:
                return [chunk] if chunk else []
            d = [chunk[i + 1][2] - chunk[i][2] for i in range(len(chunk) - 1)]
            # Find sign (ignore zeros — treat as continuation)
            sign = [0] * len(d)
            for i, v in enumerate(d):
                if v > 0:
                    sign[i] = 1
                elif v < 0:
                    sign[i] = -1
            # Fill zeros with the previous non-zero sign
            cur_sign = 1
            for i in range(len(sign)):
                if sign[i] != 0:
                    cur_sign = sign[i]
                else:
                    sign[i] = cur_sign
            # Split at sign changes
            sub_chunks = []
            start = 0
            for i in range(1, len(sign)):
                if sign[i] != sign[i - 1]:
                    sub_chunks.append(chunk[start:i + 1])  # overlap at turning point
                    start = i
            sub_chunks.append(chunk[start:])
            return sub_chunks

        def _flush_seg(cur, _sid=session_id):
            for sub in _split_monotone_chunks(cur):
                if len(sub) > min_seg_len:
                    dists = np.array([t[2] for t in sub])
                    trend = 'decreasing' if dists[-1] < dists[0] else 'increasing'
                    all_seg_coords.append(np.array([latent_coords[t[3]] for t in sub]))
                    all_seg_dist.append(dists)
                    all_seg_trend.append(trend)
                    all_seg_didx.append([t[3] for t in sub])
                    all_seg_times.append([(t[0], t[1]) for t in sub])
                    all_seg_session.append(_sid)

        cur = [timed[0]]
        for j in range(1, len(timed)):
            gap = timed[j][0] - cur[-1][1]
            if gap < consec_threshold:
                cur.append(timed[j])
            else:
                _flush_seg(cur)
                cur = [timed[j]]
        _flush_seg(cur)

    print(f"  Found {len(all_seg_coords)} continuous segments (>{min_seg_len} USVs each)")

    def select_top_segments(direction, n=10):
        indices = [i for i, t in enumerate(all_seg_trend) if t == direction]
        # Score by rate of change (cm/s): favors large, time-compact monotone segments
        def _score(i):
            dists = all_seg_dist[i]
            times = all_seg_times[i]
            delta_dist = abs(dists[-1] - dists[0])
            duration_s = times[-1][1] - times[0][0]
            return delta_dist / duration_s if duration_s > 0 else 0.0
        scores = [_score(i) for i in indices]
        top = sorted(zip(scores, indices), reverse=True)[:n]
        sel = [idx for _, idx in top]
        return (
            [all_seg_coords[idx] for idx in sel],
            [all_seg_dist[idx] for idx in sel],
            [all_seg_didx[idx] for idx in sel],
            [all_seg_times[idx] for idx in sel],
            [all_seg_session[idx] for idx in sel],
        )

    dec_coords, dec_dist, dec_didx, dec_times, dec_sessions = select_top_segments('decreasing')
    inc_coords, inc_dist, inc_didx, inc_times, inc_sessions = select_top_segments('increasing')
    print(f"  Decreasing segments: {len(dec_coords)}, Increasing segments: {len(inc_coords)}")

    dec_coords, dec_dist, dec_didx, dec_times, dec_sessions = select_top_segments('decreasing')
    inc_coords, inc_dist, inc_didx, inc_times, inc_sessions = select_top_segments('increasing')
    print(f"  decreasing segments: {len(dec_coords)}, increasing segments: {len(inc_coords)}")

    # ---- assemble the grid ---------------------------------------------------- #
    # A panel whose variable is absent from this dataset is dropped rather than
    # drawn empty, so the grid shrinks instead of filling with blank boxes.
    #
    # The high-mask-count panel: only counts in HIGH_MASK_COUNTS carry a colour,
    # every other syllable is drawn once in the shared missing grey behind them. The
    # continuous mask-count panel saturates at MASK_COUNT_RANGE[1] and cannot show
    # which of the high counts a point actually is; this one can, at the cost of
    # showing nothing else.
    high_mask_values = np.array(
        [str(int(m)) if int(m) in HIGH_MASK_COUNTS else None for m in mask_counts],
        dtype=object,
    )
    high_mask_order = [str(m) for m in HIGH_MASK_COUNTS
                       if (high_mask_values == str(m)).any()]
    n_high_mask = int(np.sum(high_mask_values != None))  # noqa: E711

    panels = [
        ('Mean frequency', lambda ax: draw_latent_scatter(
            ax, latent_coords, mean_freqs, f'Mean frequency ({freq_unit})',
            cmap=CMAP_FREQ, title='Mean frequency',
            scatter_size=scatter_size, scatter_alpha=scatter_alpha)),
        ('Bandwidth', lambda ax: draw_latent_scatter(
            ax, latent_coords, bandwidths, f'Spectral bandwidth ({freq_unit})',
            cmap=CMAP_BANDWIDTH, title='Bandwidth',
            scatter_size=scatter_size, scatter_alpha=scatter_alpha)),
        ('Duration', lambda ax: draw_latent_scatter(
            ax, latent_coords, durations, 'Duration (samples)',
            cmap=CMAP_DURATION, title='Duration',
            scatter_size=scatter_size, scatter_alpha=scatter_alpha)),
        # Clipped at MASK_COUNT_RANGE, not autoscaled: the long thin tail past 20
        # otherwise takes the whole colorbar and flattens the 1-3 bulk. Points above
        # the cap keep their place on the map, drawn at the top colour.
        ('SAM mask count', lambda ax: draw_latent_scatter(
            ax, latent_coords, mask_counts,
            f'SAM mask count (time bins, \u2265{MASK_COUNT_RANGE[1]} clipped)',
            cmap=CMAP_MASK_COUNT,
            vmin=MASK_COUNT_RANGE[0], vmax=MASK_COUNT_RANGE[1],
            title='SAM mask count',
            scatter_size=scatter_size, scatter_alpha=scatter_alpha)),
        # Drawn larger and more opaque than the other panels on purpose: counts 4-7
        # are a small minority of syllables, and at the marker weight that suits a
        # full-embedding scatter they vanish into the grey backdrop entirely.
        ('High mask count', lambda ax: draw_latent_scatter(
            ax, latent_coords, high_mask_values, None,
            color_map=HIGH_MASK_COLORS, category_order=high_mask_order,
            title='High mask count', missing_label='other',
            scatter_size=scatter_size * 2.5,
            scatter_alpha=min(0.85, scatter_alpha * 3.0))),
    ]
    print(f"    high mask count (in {HIGH_MASK_COUNTS}): {n_high_mask:,} samples")

    if type_values is not None:
        order = [t for t in SESSION_TYPE_ORDER if (type_values == t).any()]
        order += sorted({str(t) for t in np.unique(type_values)}
                        - set(SESSION_TYPE_ORDER) - {"None"})
        colors = {t: session_type_color(t) for t in order}
        panels.append(('Session type', lambda ax, o=order, c=colors: draw_latent_scatter(
            ax, latent_coords, type_values, None, color_map=c, category_order=o,
            title='Session type',
            scatter_size=scatter_size, scatter_alpha=scatter_alpha)))
        for session_type in order:
            print(f"    {session_type}: {(type_values == session_type).sum():,} samples")
    elif cond_values is not None:
        panels.append(('Condition', lambda ax: draw_latent_scatter(
            ax, latent_coords, cond_values, None,
            color_map=cond_colors, category_order=cond_order, title='Condition',
            scatter_size=scatter_size, scatter_alpha=scatter_alpha)))
        for cond in cond_order:
            print(f"    {cond}: {(cond_values == cond).sum():,} samples")

    if n_with_sex > 0:
        panels.append(('Emitter sex', lambda ax: draw_latent_scatter(
            ax, latent_coords, emitter_sexes, None,
            color_map=EMITTER_SEX_COLORS, category_order=EMITTER_SEX_ORDER,
            title='Emitter sex',
            scatter_size=scatter_size, scatter_alpha=scatter_alpha)))

    if n_with_dist > 0:
        panels.append(('Social distance', lambda ax: draw_latent_scatter(
            ax, latent_coords, social_distances, 'Social distance (cm)',
            cmap=CMAP_SOCIAL_DIST,
            vmin=SOCIAL_DIST_RANGE[0], vmax=SOCIAL_DIST_RANGE[1],
            title='Social distance',
            scatter_size=scatter_size, scatter_alpha=scatter_alpha)))

    # The decreasing/increasing social-distance segment overlays are deliberately
    # NOT panels here. They are trajectory plots over ten hand-picked segments, not
    # a colouring of the whole embedding like every other panel, so they never read
    # against their neighbours. The segments themselves are still computed above and
    # still drive the Figure E7 videos below.

    # Reference panel: the density every other panel's points were drawn from, and
    # the same image Figures FG and H segment.
    panels.append(('Aggregated posterior', lambda ax: draw_posterior_density(
        ax, heatmap, title='Aggregated posterior')))

    figure_E_grid(panels, os.path.join(save_dir, 'figure_E_embedded_grid.png'))

    print("\nGenerating Figure E7 videos: top-3 decreasing-distance segments...")
    for vi in range(min(2, len(dec_coords))):
        session_id = dec_sessions[vi]

        # ── Build tail: next 20 USVs after segment end (ignore gap constraint) ──
        tail_coords_vi = tail_dists_vi = tail_didx_vi = tail_times_vi = None
        if session_id in beh_features and session_id in session_to_items:
            import pandas as pd
            df = beh_features[session_id]
            seg_last_time = dec_times[vi][-1][1]
            seg_didx_set = set(dec_didx[vi])
            candidates = []
            for di, ri in session_to_items[session_id]:
                if ri not in df.index or di in seg_didx_set:
                    continue
                row = df.loc[ri]
                st = row.get('start_times', np.nan)
                sd = row.get('avg_social_distance', np.nan)
                if pd.isna(st) or pd.isna(sd):
                    continue
                if float(st) > seg_last_time:
                    candidates.append((float(st), float(row.get('end_times', st)), float(sd), di))
            candidates.sort(key=lambda x: x[0])
            tail_items = candidates[:20]
            if tail_items:
                tail_coords_vi = np.array([latent_coords[t[3]] for t in tail_items])
                tail_dists_vi = np.array([t[2] for t in tail_items])
                tail_didx_vi = [t[3] for t in tail_items]
                tail_times_vi = [(t[0], t[1]) for t in tail_items]

        make_segment_video(
            coords=dec_coords[vi],
            dists=dec_dist[vi],
            dataset_indices=dec_didx[vi],
            times=dec_times[vi],
            session_id=session_id,
            full_ds=full_ds,
            latent_coords_bg=latent_coords,
            durations_bg=durations,
            save_path=os.path.join(save_dir, f'figure_E7_video_{vi + 1}_{session_id}.mp4'),
            vmin=0, vmax=85,
            scatter_size=scatter_size, scatter_alpha=scatter_alpha,
            tail_coords=tail_coords_vi,
            tail_dists=tail_dists_vi,
            tail_dataset_indices=tail_didx_vi,
            tail_times=tail_times_vi,
        )

    # ============= FIGURE F: mean-shift centroids over the aggregated posterior ====
    print("\nGenerating Figure F: Aggregated posterior...")

    # Run mean-shift clustering
    print("Running mean-shift clustering...")
    embedded = torus_forward(latent_coords)

    if use_fast_mean_shift:
        print("  Using run_mean_shift_fast (lattice-based)...")
        lattice_embedded = torus_forward(lattice_np)   # (N_lattice, 4)
        centers, _lattice_labels, _seed_mask = run_mean_shift_fast(
            lattice_embedded,
            aggregated,
            bandwidth=bandwidth,
            k_neighbors=ms_k_neighbors,
            seed_percentile=ms_seed_percentile,
            max_iter=ms_max_iter,
            tol=ms_tol,
            embedded=True,
        )
        # Assign per-sample labels by nearest center in torus-embedded space
        from scipy.spatial import cKDTree as _cKDTree
        centers_embedded = torus_forward(centers)
        _, labels = _cKDTree(centers_embedded).query(embedded)
    else:
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

    labels = np.atleast_1d(labels)

    # centers are in [0,1]^2 after torus_reverse inside run_mean_shift / run_mean_shift_fast
    print(f"Found {len(centers)} clusters")

    # ============= FIGURE FG: Aggregated posterior + cluster example spectrograms =============
    print("\nGenerating Figure FG: Aggregated posterior + cluster examples...")
    n_clusters_found = len(centers)
    n_cols_g = 5
    n_rows_g = int(np.ceil(n_clusters_found / n_cols_g))

    import matplotlib.gridspec as gridspec
    fig_height = max(5, n_rows_g * 2 + 1) if n_rows_g > 1 else 4
    fig = plt.figure(figsize=(6 + n_cols_g * 2, fig_height))
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.1], wspace=0.15)

    ax = fig.add_subplot(outer[0, 0])
    # Same drawer, same colormap and same vmax rule as every other panel that shows
    # the posterior -- the Figure-E reference panel and the Figure-H background.
    draw_posterior_density(ax, heatmap)
    for i, (x, y) in enumerate(centers):
        ax.text(x, y, str(i + 1), color=OVERLAY_COLOR, fontsize=18,
                fontweight='bold', ha='center', va='center', zorder=11)

    right_gs = gridspec.GridSpecFromSubplotSpec(n_rows_g, n_cols_g, subplot_spec=outer[0, 1],
                                                 hspace=0.25, wspace=0.1)
    for cluster_id in range(n_rows_g * n_cols_g):
        r, c = divmod(cluster_id, n_cols_g)
        ax_s = fig.add_subplot(right_gs[r, c])
        if cluster_id < n_clusters_found:
            cluster_samples = np.where(labels == cluster_id)[0]
            if len(cluster_samples) > 0:
                example_idx = cluster_samples[np.random.choice(len(cluster_samples))]
                spec = full_ds[example_idx][0].numpy().squeeze()
                ax_s.imshow(spec, cmap=CMAP_SPEC, origin='lower', aspect='auto')
                ax_s.set_title(f'{cluster_id + 1}', color=HIGHLIGHT_COLOR,
                               fontsize=12, fontweight='bold')
            else:
                ax_s.text(0.5, 0.5, 'Empty', ha='center', va='center', transform=ax_s.transAxes)
        else:
            ax_s.axis('off')
        ax_s.set_xticks([]); ax_s.set_yticks([])

    plt.savefig(os.path.join(save_dir, 'figure_FG_posterior_and_examples.png'),
                dpi=300, bbox_inches='tight')
    print("Saved: figure_FG_posterior_and_examples.png")
    plt.close()

    # ============= Grid reconstructions =============
    print("\nGenerating grid reconstructions...")
    if conditional is not None:
        # For conditional models generate one grid per sweep value and also a
        # mean-c grid for a compact single overview.
        # From data.conditionals, NOT bartul_mouse_cond: the training driver's own
        # copy of the sweep was the 8-class one, so a 5-class mask_count model got
        # eight grids, three of them decoded from a c wider than the decoder's c_dim.
        sweep = conditionals.get_grid_sweep(conditional)
        for val_label, c_val in sweep:
            grid_examples(
                model, grid_size,
                save_path=os.path.join(save_dir, f'figure_grid_examples_{conditional}_{val_label}.png'),
                device=device, c=c_val.to(device),
            )
        # Also produce a mean-conditioning overview grid
        fi = conditionals.resolve(conditional)["field_idx"]
        mean_c = torch.stack([
            full_ds[i][fi].float() if full_ds[i][fi].dim() > 0 else full_ds[i][fi].float().unsqueeze(0)
            for i in range(min(500, len(full_ds)))
        ]).mean(dim=0, keepdim=True).to(device)   # (1, c_dim)
        grid_examples(model, grid_size,
                      save_path=os.path.join(save_dir, 'figure_grid_examples.png'),
                      device=device, c=mean_c)
    else:
        grid_examples(model, grid_size,
                      save_path=os.path.join(save_dir, 'figure_grid_examples.png'),
                      device=device)

    # ============= FIGURE H: Random samples per watershed cluster (one fig per cluster) =============
    print("\nGenerating Figure H: Random samples per cluster (watershed σ=3, compact=0)...")
    from scipy.ndimage import gaussian_filter
    from skimage.segmentation import watershed

    res = heatmap.shape[0]
    # Watershed on the lattice-weighted posterior mass (already torus-smoothed
    # above); this is the true marginal over the torus rather than the empirical
    # sample histogram, so basins line up with the mean-shift centers.
    smoothed = gaussian_filter(heatmap_lattice, sigma=3, mode='wrap')
    marker_img = np.zeros((res, res), dtype=int)
    for i, (cx, cy) in enumerate(centers):
        px = int(np.clip(cx * res, 0, res - 1))
        py = int(np.clip(cy * res, 0, res - 1))
        marker_img[py, px] = i + 1
    ws_labels = watershed(-smoothed, markers=marker_img, compactness=0)

    sample_px = np.clip((latent_coords[:, 0] * res).astype(int), 0, res - 1)
    sample_py = np.clip((latent_coords[:, 1] * res).astype(int), 0, res - 1)
    sample_cluster = ws_labels[sample_py, sample_px]  # 1..n_clusters_found

    # ============= FIGURE H watershed grid search =============
    # Figure FH used to sit here, pairing a duration scatter with the posterior under
    # the same boundaries. It is gone: its posterior-plus-watershed view is what the
    # sweep below and each per-cluster panel now draw, and its duration scatter is a
    # panel of the Figure-E grid.
    print("\nGenerating Figure H: Watershed grid search (σ × compactness)...")
    figure_H_watershed_variants(
        heatmap_lattice, centers,
        save_path=os.path.join(save_dir, 'figure_H_watershed_grid.png'),
    )

    # The sweep above is seeded from the mean-shift centroids, so every cell in it
    # returns exactly len(centers) basins -- it varies the BOUNDARIES and nothing
    # else, and you cannot pick a cluster count out of it. This second sweep seeds
    # from the local maxima of the smoothed posterior instead, which makes sigma
    # the knob that sets the count, and reports the setting with the most clusters
    # still under watershed_max_clusters. Both are kept: the first answers "where
    # do these centroids put the boundaries", the second answers "how many groups
    # does this density actually have at a given scale".
    from analysis.watershed_sweep import figure_watershed_sweep
    ws_choice, ws_n, ws_labels_sel = figure_watershed_sweep(
        heatmap_lattice,
        save_path=os.path.join(save_dir, 'figure_H_watershed_sweep.png'),
        max_clusters=watershed_max_clusters,
        cmap=CMAP_POSTERIOR, overlay_color=OVERLAY_COLOR,
    )
    if ws_choice is None:
        print(f"  NOTE: no setting in the sweep falls under "
              f"{watershed_max_clusters} clusters; nothing selected.")
    else:
        print(f"  Selected: sigma={ws_choice[0]}, compactness={ws_choice[1]} "
              f"-> {ws_n} clusters (most structure under {watershed_max_clusters})")
    print("Saved: figure_H_watershed_sweep.png")

    n_clusters_found = len(centers)
    n_cols_h = int(np.ceil(np.sqrt(n_per_cluster)))
    n_rows_h = int(np.ceil(n_per_cluster / n_cols_h))

    xx = np.linspace(0, 1, res)
    yy = np.linspace(0, 1, res)
    boundary_levels = np.arange(0.5, n_clusters_found + 1.5)

    import matplotlib.gridspec as gridspec

    # One figure per cluster, and there can be dozens (the phase-2 full-corpus
    # run wrote 30, ~18 MB). They are the slowest and bulkiest part of the
    # figure set and are off for the conditional runs, which want the Figure-E
    # grid and the watershed sweep only.
    if cluster_sample_figures:
        for ci in range(n_clusters_found):
            mask = np.where(sample_cluster == ci + 1)[0]

            picks = []
            if len(mask) > 0:
                pts = latent_coords[mask]
                used = set()
                if sample_from_centroid:
                    # Archimedean spiral outward from cluster centroid; skip points
                    # falling outside this cluster's watershed region.
                    cx0, cy0 = centers[ci]
                    dists_c = np.sqrt((pts[:, 0] - cx0) ** 2 + (pts[:, 1] - cy0) ** 2)
                    r_max = float(dists_c.max()) if len(dists_c) > 0 else 0.1
                    n_turns = 4
                    n_dense = 4000
                    t_dense = np.linspace(0.0, 1.0, n_dense)
                    theta_d = 2 * np.pi * n_turns * t_dense
                    r_d = r_max * t_dense
                    xs_d = cx0 + r_d * np.cos(theta_d)
                    ys_d = cy0 + r_d * np.sin(theta_d)
                    px_d = np.clip((xs_d * res).astype(int), 0, res - 1)
                    py_d = np.clip((ys_d * res).astype(int), 0, res - 1)
                    inside = (
                        (xs_d >= 0) & (xs_d <= 1) & (ys_d >= 0) & (ys_d <= 1)
                        & (ws_labels[py_d, px_d] == ci + 1)
                    )
                    xs_in = xs_d[inside]
                    ys_in = ys_d[inside]
                    if len(xs_in) >= n_per_cluster:
                        sel = np.linspace(0, len(xs_in) - 1, n_per_cluster).astype(int)
                        xs_in = xs_in[sel]
                        ys_in = ys_in[sel]
                    targets = list(zip(xs_in, ys_in))
                    for tx_, ty_ in targets:
                        d2 = (pts[:, 0] - tx_) ** 2 + (pts[:, 1] - ty_) ** 2
                        for kk in np.argsort(d2):
                            idx = int(mask[kk])
                            if idx not in used:
                                used.add(idx)
                                picks.append(idx)
                                break
                else:
                    # Tile-based sampling: partition cluster bbox into tiles
                    x_min, y_min = pts.min(axis=0)
                    x_max, y_max = pts.max(axis=0)
                    x_max = max(x_max, x_min + 1e-6)
                    y_max = max(y_max, y_min + 1e-6)
                    tx = np.linspace(x_min, x_max, n_cols_h + 1)
                    ty = np.linspace(y_min, y_max, n_rows_h + 1)
                    # Iterate top -> bottom, left -> right
                    for rr in range(n_rows_h - 1, -1, -1):
                        for cc in range(n_cols_h):
                            if len(picks) >= n_per_cluster:
                                break
                            cx_t = 0.5 * (tx[cc] + tx[cc + 1])
                            cy_t = 0.5 * (ty[rr] + ty[rr + 1])
                            d2 = (pts[:, 0] - cx_t) ** 2 + (pts[:, 1] - cy_t) ** 2
                            for k in np.argsort(d2):
                                idx = int(mask[k])
                                if idx not in used:
                                    used.add(idx)
                                    picks.append(idx)
                                    break
            picks = np.array(picks, dtype=int)

            fig = plt.figure(figsize=(6 + n_cols_h * 1.6, max(6, n_rows_h * 1.6 + 1)))
            outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.05], wspace=0.15)

            # --- Left: the posterior this cluster was carved out of, boundaries on top ---
            # This panel used to be a mean-frequency scatter, which showed where samples
            # landed but not the density the watershed actually cut. It now draws the
            # same aggregated posterior as Figures E and FG, so a basin can be checked
            # against the mass it claims.
            ax_l = fig.add_subplot(outer[0, 0])
            # heatmap_lattice, not heatmap: this panel exists to show the basin the
            # watershed cut, so it draws the field the watershed was run on. The two
            # estimators of the posterior agree closely; Figures E and FG use the
            # sample-space one because they are drawn against per-sample points.
            draw_posterior_density(ax_l, heatmap_lattice, title=None, colorbar=False)
            ax_l.contour(xx, yy, ws_labels, levels=boundary_levels,
                         colors=OVERLAY_COLOR, linewidths=1.0)
            ax_l.contour(xx, yy, (ws_labels == ci + 1).astype(int), levels=[0.5],
                         colors=HIGHLIGHT_COLOR, linewidths=2.5)
            ax_l.scatter(centers[:, 0], centers[:, 1],
                         c=OVERLAY_COLOR, s=50, marker='x', linewidths=1.2, zorder=9)
            ax_l.scatter([centers[ci, 0]], [centers[ci, 1]],
                         c=HIGHLIGHT_COLOR, s=120, marker='x', linewidths=2.5, zorder=10)
            if len(picks) > 0:
                for k, pidx in enumerate(picks):
                    ax_l.text(latent_coords[pidx, 0], latent_coords[pidx, 1], str(k + 1),
                              color=HIGHLIGHT_COLOR, fontsize=7, fontweight='bold',
                              ha='center', va='center', zorder=12)
            ax_l.set_title(f'Cluster {ci + 1} (σ=3, compact=0)',
                           fontsize=12, fontweight='bold', color=HIGHLIGHT_COLOR)

            # --- Right: tile-sampled spectrograms ---
            right_gs = gridspec.GridSpecFromSubplotSpec(
                n_rows_h, n_cols_h, subplot_spec=outer[0, 1], hspace=0.25, wspace=0.1)
            for j in range(n_rows_h * n_cols_h):
                rr, cc = divmod(j, n_cols_h)
                ax_s = fig.add_subplot(right_gs[rr, cc])
                if j < len(picks):
                    spec = full_ds[picks[j]][0].numpy().squeeze()
                    ax_s.imshow(spec, cmap=CMAP_SPEC, origin='lower', aspect='auto')
                    ax_s.set_title(str(j + 1), fontsize=8, color='black', pad=1)
                    for spine in ax_s.spines.values():
                        spine.set_visible(False)
                else:
                    ax_s.axis('off')
                ax_s.set_xticks([]); ax_s.set_yticks([])

            plt.suptitle(f'Figure H — Cluster {ci + 1}: watershed region & tile-sampled examples',
                         fontsize=12, fontweight='bold')
            out_path = os.path.join(save_dir, f'figure_H_cluster_{ci + 1:02d}_samples.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
        print(f"Saved: figure_H_cluster_XX_samples.png ({n_clusters_found} figures)")

    # Save cluster information
    cluster_info = {
        'n_clusters': n_clusters_found,
        'centroids': centers.tolist(),
        'cluster_sizes': [int(np.sum(labels == i)) for i in range(n_clusters_found)],
        'bandwidth': bandwidth,
        # The local-maxima sweep's pick, which is a different segmentation from the
        # mean-shift centroids above; recorded so a reader knows which figure the
        # cluster count in figure_H_watershed_sweep.png came from.
        'watershed_sweep': (
            None if ws_choice is None
            else {'sigma': ws_choice[0], 'compactness': ws_choice[1],
                  'n_clusters': int(ws_n), 'max_clusters': watershed_max_clusters}
        ),
        'cluster_sample_figures': bool(cluster_sample_figures),
    }

    import json
    with open(os.path.join(save_dir, 'cluster_info.json'), 'w') as f:
        json.dump(cluster_info, f, indent=2)

    print(f"\n=== Analysis Complete ===")
    print(f"Figures saved to: {save_dir}")
    print(f"Number of clusters: {n_clusters_found}")
    print(f"Cluster sizes: {cluster_info['cluster_sizes']}")

    return {
        'latent_coords': latent_coords,
        'mean_freqs': mean_freqs,
        'bandwidths': bandwidths,
        'mask_counts': mask_counts,
        'centers': centers,
        'labels': labels,
        'social_distances': social_distances,
        'emitter_sexes': emitter_sexes,
    }


if __name__ == '__main__':
    fire.Fire(analyze_mouse_latents)
