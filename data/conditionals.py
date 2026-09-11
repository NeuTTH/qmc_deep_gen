"""Conditioning variables for the conditional QLVM.

Single source of truth for the conditional registry. ``bartul_mouse_cond.py``
(training), ``analyze_mouse_latents_2d.py`` and ``analyze_all_sessions.py``
(inference) all import from here. Each of them used to carry its own literal
copy of the registry, and they drifted: a change to the mask-count width in one
silently disagreed with the others, and nothing checked.

``mouse_data.__getitem__`` returns, positionally::

    0: spec               (1 x H x W)
    1: masks_len          scalar
    2: raw duration       scalar
    3: norm_duration      scalar, [0, 1]
    4: mean_freq          scalar, [0, 1]
    5: mask_count_onehot  (K,)     K = mouse_data(mask_count_classes=K)
    6: mask               (H x W)
    7: spec_id            str

Adding a conditional: precompute the field in ``mouse_data.__init__``, expose it
in ``__getitem__`` WITHOUT reordering the existing slots (downstream consumers
index this tuple positionally), then add an entry here.

WHY ``kind`` MATTERS.
The QMCLVM decodes ONE lattice per batch conditioned on ONE ``c`` vector. That
is exact only when every member of the batch shares that ``c``. Training draws
batches through ``ConditionGroupedBatchSampler``, which groups rows by
conditioning value: exactly for a ``discrete`` conditional, and by quantile bin
for a ``continuous`` one. ``kind`` is what tells the sampler which to do.
"""

import numpy as np
import torch
import torch.nn.functional as F


# Mask-count one-hot width. 5 classes: 1, 2, 3, 4, and 5-or-more lumped into the
# top class. The 5-strata datasets are built on exactly these bins, and the tail
# beyond 5 is thin -- masks_len reaches 18 in the bbvfree sets but only 0.66% of
# training rows exceed 8. The legacy 8-class width is kept reachable as
# 'mask_count8' so the April 2025 checkpoints still load.
DEFAULT_MASK_COUNT_CLASSES = 5

# Number of quantile bins a continuous conditional is grouped into for batching.
# Each bin must be comfortably larger than one batch or the sampler emits mostly
# short batches: 32 bins over a 106k-row training set is ~3,300 rows per bin,
# about 6 full batches of 512.
DEFAULT_COND_N_BINS = 32


CONDITIONAL_REGISTRY = {
    "mask_count": {
        "field_idx": 5,
        "c_dim": DEFAULT_MASK_COUNT_CLASSES,
        "n_classes": DEFAULT_MASK_COUNT_CLASSES,
        "kind": "discrete",
        "label": "mask count (one-hot, 1/2/3/4/5+)",
    },
    "mask_count8": {
        "field_idx": 5,
        "c_dim": 8,
        "n_classes": 8,
        "kind": "discrete",
        "label": "mask count (one-hot, 1..8, legacy)",
    },
    "duration": {
        "field_idx": 3,
        "c_dim": 1,
        "n_classes": None,
        "kind": "continuous",
        "label": "normalized duration",
    },
    "mean_freq": {
        "field_idx": 4,
        "c_dim": 1,
        "n_classes": None,
        "kind": "continuous",
        "label": "mean frequency",
    },
}


def resolve(cond_name):
    """Return the registry entry for ``cond_name``, or raise with the options."""
    if cond_name not in CONDITIONAL_REGISTRY:
        raise ValueError(
            f"Unknown conditional: {cond_name!r}. "
            f"Choose from: {sorted(CONDITIONAL_REGISTRY)}."
        )
    return CONDITIONAL_REGISTRY[cond_name]


def c_dim_of(cond_name):
    """Conditioning width for ``cond_name``; 0 for an unconditional model."""
    if cond_name is None:
        return 0
    return resolve(cond_name)["c_dim"]


def mask_count_classes_for(cond_name):
    """One-hot width ``mouse_data`` should build for this conditional.

    Only the mask-count conditionals care. Everything else gets the default, so
    that slot 5 of the tuple has a stable width across runs.
    """
    if cond_name is None:
        return DEFAULT_MASK_COUNT_CLASSES
    entry = resolve(cond_name)
    if entry["kind"] == "discrete" and entry["n_classes"]:
        return int(entry["n_classes"])
    return DEFAULT_MASK_COUNT_CLASSES


def label_of(cond_name):
    """Human-readable name, for figure titles and logs."""
    return None if cond_name is None else resolve(cond_name)["label"]


def get_grid_sweep(cond_name):
    """Representative conditioning values for decoder-grid figures.

    Returns a list of ``(label_str, c_tensor)`` with ``c_tensor`` of shape
    ``(1, c_dim)``. Discrete conditionals sweep every class; continuous ones
    sweep five points across [0, 1].
    """
    entry = resolve(cond_name)
    if entry["kind"] == "discrete":
        k = int(entry["n_classes"])
        labels = [f"ml{i + 1}" for i in range(k)]
        labels[-1] = f"ml{k}plus"          # the top class is open-ended
        return [
            (labels[i], F.one_hot(torch.tensor([i]), num_classes=k).float())
            for i in range(k)
        ]
    prefix = {"duration": "dur", "mean_freq": "mf"}.get(cond_name, "c")
    return [(f"{prefix}{v:.1f}", torch.tensor([[v]])) for v in (0.1, 0.3, 0.5, 0.7, 0.9)]


def sample_c(dataset, index, cond_name, device=None):
    """Conditioning tensor for a single sample -> ``(1, c_dim)``."""
    entry = resolve(cond_name)
    val = dataset[index][entry["field_idx"]]
    val = val.float() if torch.is_tensor(val) else torch.tensor([float(val)])
    if val.dim() == 0:
        val = val.unsqueeze(0)
    out = val.unsqueeze(0)
    return out if device is None else out.to(device)


def group_ids(dataset, cond_name, n_bins=DEFAULT_COND_N_BINS):
    """Integer group id per row, for ``ConditionGroupedBatchSampler``.

    Rows sharing a group id share (discrete) or nearly share (continuous) a
    conditioning value, so a batch drawn from one group can be decoded with a
    single ``c`` without averaging away the signal.

    Discrete conditionals group by class. Continuous ones group by quantile bin,
    which keeps groups near-equal in size; equal-width bins would leave the tails
    almost empty and produce batches of one or two rows.
    """
    entry = resolve(cond_name)
    values = _raw_values(dataset, cond_name)

    if entry["kind"] == "discrete":
        # values here are the one-hot rows; the class is the argmax.
        return values.argmax(axis=1).astype(np.int64)

    values = values.reshape(-1)
    # Quantile edges. Duplicate edges (a spike in the distribution) collapse to
    # fewer than n_bins groups, which is correct rather than an error: it means
    # those rows genuinely share a value.
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        return np.zeros(len(values), dtype=np.int64)
    ids = np.digitize(values, edges[1:-1], right=False)
    return ids.astype(np.int64)


def _raw_values(dataset, cond_name):
    """The conditioning column for every row, as ``(N,)`` or ``(N, c_dim)``.

    Reads the precomputed attribute directly rather than iterating
    ``__getitem__``, which would decode every spectrogram just to read a scalar.
    """
    entry = resolve(cond_name)
    attr = {
        3: "norm_durations",
        4: "mean_freqs",
        5: "mask_count_onehot",
    }[entry["field_idx"]]
    column = getattr(dataset, attr)
    return column.detach().cpu().numpy() if torch.is_tensor(column) else np.asarray(column)


def describe(cond_name, n_bins=DEFAULT_COND_N_BINS):
    """One-line summary for logs and run records."""
    if cond_name is None:
        return "unconditional (c_dim=0)"
    entry = resolve(cond_name)
    how = (
        f"grouped by class ({entry['n_classes']} classes)"
        if entry["kind"] == "discrete"
        else f"grouped by {n_bins} quantile bins"
    )
    return f"{cond_name} -- {entry['label']}, c_dim={entry['c_dim']}, {how}"
