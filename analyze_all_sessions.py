"""Run the QLVM latent-space figure set over the FULL session corpus.

``analyze_mouse_latents_2d.py`` embeds whatever dataset it is pointed at. That is
the right contract for the per-cell inference jobs, which embed the same
stratified DRAW each model was trained on (151k / 81k specs). This module is the
other question: embed **every spectrogram in the session pool** -- the
``full_data.npz`` written by usv-playpen's ``build-qlvm-training-set
--full-dataset`` over all 325 session roots (365,150 specs after
``0 < duration < 128`` and ``require_mask``) -- through a model trained on one of
those draws.

It is a wrapper, not a fork: every figure comes from ``analyze_mouse_latents``
with the same settings the per-cell inference launcher uses, so an
``latents_all_sessions/`` figure is directly comparable to the ``latents_all/``
one beside it and the only thing that changed is which spectrograms went in.

WHAT THE WRAPPER ADDS
---------------------
1. **Preflight.** The mistakes this path invites are silent, not loud:

   * embedding a model on the wrong MASKING ARM. ``--masking-type sam
     --apply-mask`` zeroes the background on disk and that is irreversible, so
     masked and unmasked are separate datasets. ``mouse_data`` reads the scalar
     ``apply_mask`` each set carries and honours it, which is why inference must
     never pass ``apply_mask`` explicitly -- but nothing stops you handing it the
     other arm's directory. :func:`check_arm` compares the dataset's declared
     ``apply_mask`` against the run's own ``sampling_config.json`` and refuses on
     a mismatch.
   * pointing at a DRAW instead of the full corpus and calling the result
     "all sessions". :func:`check_full_corpus` requires a ``full_data`` file and,
     when the build's ``metadata.npz`` is present, that it was built with
     ``full_dataset=True`` and an all-take-all ``session_type_targets``.
   * rebuilding a CONDITIONAL decoder at the wrong input width. ``conditional``
     widens the decoder's first linear layer by ``c_dim``, so passing the wrong
     name -- or none at all, against a model that was trained with one -- builds a
     different architecture and then tries to load a checkpoint into it.
     :func:`check_conditional` compares the requested conditioning against the
     run's own ``run_config.json``. The trap it exists for is the mask-count
     width: that one-hot was 8 classes until the 5-class (1/2/3/4/5+) build
     landed, so an April-2025 checkpoint needs ``mask_count8`` and a recent one
     needs ``mask_count``, and the two differ by nothing a reader would notice.
   * a set whose ``session_type`` column is all ``'unknown'`` -- what the builder
     writes when ``session_type_targets`` is empty, which also silently disables
     ``require_mask``. That set drops the session-type panel out of
     ``figure_E_embedded_grid``, so it is rejected here rather than discovered in
     the figure.

2. **A manifest.** ``manifest.json`` next to the figures records the model, the
   dataset, N, the masking arm, the lattice, the analysis settings, wall time and
   the figures actually written, so a figure set can be traced back without
   re-deriving it from a launcher.

Usage (fire CLI)::

    python analyze_all_sessions.py \
        --model_path /scratch/.../qmc_train_mouse_experiment.tar \
        --dataloc  /mnt/cup/labs/.../qlvm_all_sessions_unmasked/all-sessions_n325_len128_seed42 \
        --save_dir /scratch/.../latents_all_sessions

Every keyword of ``analyze_mouse_latents`` is forwarded; the defaults here are
the ones ``scripts/inference_qlvm_all_sessions.sh`` uses. See
``docs/qlvm_playpen_runs/ALL_SESSIONS.md`` for the full option table.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time

import fire
import numpy as np

# Import by module so the wrapper can live beside the driver and still be run
# from anywhere (the driver itself does `from models... import` relative to the
# qmc_deep_gen root, so that root has to be on sys.path either way).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from analyze_mouse_latents_2d import analyze_mouse_latents  # noqa: E402
from data import conditionals  # noqa: E402


# The four labels the session-type panel of figure_E_embedded_grid draws. A
# full-corpus set built with an empty session_type_targets writes 'unknown' for
# every row instead, which drops the panel; see check_full_corpus.
EXPECTED_SESSION_TYPES = ("MF", "FF", "MM", "lone_male")


def _scalar(value):
    """Unwrap a 0-d numpy array / 1-element array to a plain Python value."""
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
    elif array.size == 1:
        item = array.reshape(-1)[0]
    else:
        return array.tolist()
    return item.decode() if isinstance(item, bytes) else item


def read_dataset_metadata(dataloc):
    """Return the build record ``metadata.npz`` holds, as plain Python.

    Returns ``{}`` when the directory has no ``metadata.npz`` -- older sets and
    hand-assembled directories have none, and the caller decides whether that is
    fatal.
    """
    path = os.path.join(dataloc, "metadata.npz")
    if not os.path.isfile(path):
        return {}
    out = {}
    with np.load(path, allow_pickle=False) as handle:
        for key in handle.files:
            value = handle[key]
            if value.dtype.kind in "US" and value.ndim <= 1 and value.size <= 4096:
                out[key] = [str(v) for v in value.ravel()] if value.ndim else str(value)
            elif value.ndim == 0 or value.size <= 32:
                out[key] = _scalar(value)
            else:
                out[key] = f"<{value.dtype} {value.shape}>"
    for key in ("session_type_targets", "type_report", "session_type_by_key", "split_sessions"):
        if isinstance(out.get(key), str):
            try:
                out[key] = json.loads(out[key])
            except (TypeError, ValueError):
                pass
    return out


def read_run_record(model_path):
    """Return ``{run_config, sampling_config, run_dir}`` for a checkpoint.

    Both JSONs are written next to the checkpoint by the training launcher;
    ``sampling_config.json`` is the authoritative record of the masking arm the
    model was TRAINED on (``mouse_data`` writes its resolved ``apply_mask``
    there), and ``run_config.json`` carries the dataset path and a ``mask_tag``.
    Either may be absent for a hand-run checkpoint.
    """
    run_dir = os.path.dirname(os.path.abspath(model_path))
    record = {"run_dir": run_dir, "run_config": {}, "sampling_config": {}}
    for name, key in (("run_config.json", "run_config"),
                      ("sampling_config.json", "sampling_config")):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            with open(path) as handle:
                record[key] = json.load(handle)
    return record


def model_apply_mask(run_record):
    """The masking arm the model was trained on, or None if unrecorded.

    ``sampling_config.json`` wins: it is what ``mouse_data`` actually resolved at
    training time. ``run_config.json``'s ``mask_tag`` is the fallback for runs
    predating that file.
    """
    sampling = run_record.get("sampling_config") or {}
    if "apply_mask" in sampling:
        return bool(sampling["apply_mask"]), "sampling_config.json"
    tag = (run_record.get("run_config") or {}).get("mask_tag")
    if tag in ("masked", "unmasked"):
        return tag == "masked", "run_config.json:mask_tag"
    return None, "unrecorded"


def check_full_corpus(dataloc, metadata, min_n=None, strict=True):
    """Refuse a dataset that is not the full corpus. Returns a report dict.

    Checks, in order of how quietly they would otherwise go wrong:

    * a ``full_data.{pt,npz}`` exists -- ``load_full_mouse_data`` would silently
      fall back to concatenating a train/val DRAW otherwise;
    * the build says ``full_dataset=True`` and every ``session_type_targets``
      entry is null (take-all). A budgeted draw with a big N is still a draw;
    * ``require_mask`` is on and ``masking_type == 'sam'`` -- unless the build is
      maskless (``masking_type == 'none'``), which the BBV corpora are, in which
      case the requirement is instead that it does NOT ask for masking;
    * the ``session_type_by_key`` map is populated with real labels. An empty
      ``session_type_targets`` takes the builder's untyped path, which writes
      ``session_type='unknown'`` for every row (empty ``_cond`` figure) AND
      ignores ``require_mask`` -- one flag, two silent failures.
    """
    problems, notes = [], []

    present = [stem for stem in ("full_data.pt", "full_data.npz")
               if os.path.isfile(os.path.join(dataloc, stem))]
    if not present:
        problems.append(
            f"no full_data.pt / full_data.npz in {dataloc}; load_full_mouse_data would "
            f"fall back to concatenating train_data+val_data, which is a DRAW, not the "
            f"full corpus. Build with --full-dataset."
        )
    else:
        notes.append(f"full-data file: {present[0]}")

    if not metadata:
        notes.append("no metadata.npz in the dataset directory; build flags unverified")
        if strict:
            problems.append(
                f"{dataloc} has no metadata.npz, so the build flags cannot be verified; "
                f"pass strict=False to embed anyway."
            )
        return {"problems": problems, "notes": notes}

    if not bool(metadata.get("full_dataset", False)):
        problems.append("metadata.npz says full_dataset=False: this directory holds a draw.")

    targets = metadata.get("session_type_targets")
    if isinstance(targets, dict) and targets:
        budgeted = {k: v for k, v in targets.items() if v is not None}
        if budgeted:
            problems.append(
                f"session_type_targets budgets {budgeted}; a budgeted draw is not the full "
                f"corpus. Every entry must be null (take-all)."
            )
        else:
            notes.append(f"session_type_targets take-all for {sorted(targets)}")
    elif isinstance(targets, dict):
        problems.append(
            "session_type_targets is empty, so the build took the untyped path: every row's "
            "session_type is 'unknown' (the session-type panel drops out of the "
            "Figure-E grid) and "
            "--require-mask was ignored. Rebuild with an all-null mapping."
        )

    # A MASKLESS corpus is a legitimate full corpus, not a misbuilt masked one. The
    # broadband-vocalization builds have no SAM segmentation at all: masking_type is
    # 'none', the stored mask plane is all zeros, require_mask is off because there was
    # never a detector to require, and the set declares apply_mask=False. Treated as a
    # masked build, these two gates reject every BBV corpus for the wrong reason.
    #
    # This is not a loosening of the masked case. A set that says masking_type='sam'
    # still has to have require_mask on, which is the failure these lines were written
    # to catch. And a maskless set that nonetheless asks for masking is caught here
    # instead, because multiplying by an all-zero plane would zero every spectrogram.
    if str(metadata.get("masking_type", "")) == "none":
        if bool(metadata.get("apply_mask", True)):
            problems.append("masking_type='none' but apply_mask is true: there are no masks "
                            "to apply, and masking would zero every spectrogram.")
        else:
            notes.append("maskless corpus (masking_type='none', apply_mask=False); the "
                         "in-mask / out-of-mask reconstruction split carries no meaning "
                         "here -- read mean_recon_mse, not mse_in_mask")
    else:
        if not bool(metadata.get("require_mask", False)):
            problems.append("require_mask=False: rows the detector found no mask for carry an "
                            "all-ones mask, which asserts the whole spectrogram is signal.")
        if str(metadata.get("masking_type", "")) != "sam":
            problems.append(f"masking_type={metadata.get('masking_type')!r}, expected 'sam'.")

    types_by_key = metadata.get("session_type_by_key")
    if isinstance(types_by_key, dict) and types_by_key:
        labels = sorted(set(types_by_key.values()))
        notes.append(f"{len(types_by_key)} sessions typed: {labels}")
        if labels == ["unknown"]:
            problems.append("every session typed 'unknown'; the session-type YAMLs were not read.")
    elif isinstance(types_by_key, dict):
        problems.append("session_type_by_key is empty; the build never typed its sessions.")

    n_full = metadata.get("n_full")
    if n_full is not None:
        notes.append(f"build wrote n_full={int(n_full):,}")
        if min_n is not None and int(n_full) < int(min_n):
            problems.append(f"n_full={int(n_full):,} < min_n={int(min_n):,}: not the full corpus.")
    return {"problems": problems, "notes": notes}


def check_arm(metadata, run_record):
    """Refuse to embed a model on the other masking arm. Returns a report dict."""
    problems, notes = [], []
    data_arm = metadata.get("apply_mask")
    model_arm, source = model_apply_mask(run_record)

    if data_arm is None:
        notes.append("dataset does not record apply_mask in metadata.npz; "
                     "mouse_data will still read the scalar inside the split")
    else:
        data_arm = bool(data_arm)
        notes.append(f"dataset apply_mask={data_arm}")

    if model_arm is None:
        notes.append("model does not record which arm it was trained on "
                     f"({source}); cannot cross-check")
    else:
        notes.append(f"model apply_mask={model_arm} (from {source})")

    if data_arm is not None and model_arm is not None and data_arm != model_arm:
        problems.append(
            f"MASKING ARM MISMATCH: the model was trained with apply_mask={model_arm} "
            f"({source}) but this dataset declares apply_mask={data_arm}. The background "
            f"zeroing is burned into the stored spectrograms, so these are different "
            f"inputs, not a flag -- point at the "
            f"{'masked' if model_arm else 'unmasked'} full-corpus set instead."
        )
    return {"problems": problems, "notes": notes, "data_arm": data_arm, "model_arm": model_arm}


def _recorded_c_dim(run_config):
    """The conditioning width the run record claims, or None if unknowable.

    ``c_dim`` is written explicitly by the conditional driver; for a record that
    predates that field it is re-derived from the recorded NAME. A name the
    registry no longer knows (renamed, or a typo in a hand-written record) gives
    None rather than raising -- the caller reports that as a mismatch it cannot
    quantify, which is more useful than a traceback out of the preflight.
    """
    if run_config.get("c_dim") is not None:
        return int(run_config["c_dim"])
    try:
        return conditionals.c_dim_of(run_config.get("conditional"))
    except ValueError:
        return None


def _recorded_mask_count_classes(run_record):
    """The slot-5 one-hot width the run built, and which file said so.

    ``sampling_config.json`` wins for the same reason it wins in
    :func:`model_apply_mask`: ``mouse_data`` writes its own RESOLVED
    ``mask_count_classes`` there, whereas ``run_config.json`` records what the
    driver asked for. They agree on anything the current driver wrote; on older
    or hand-run checkpoints only one of them exists, and on none of them before
    the 5-class build does either.
    """
    sampling = run_record.get("sampling_config") or {}
    if sampling.get("mask_count_classes") is not None:
        return int(sampling["mask_count_classes"]), "sampling_config.json"
    run_config = run_record.get("run_config") or {}
    if run_config.get("mask_count_classes") is not None:
        return int(run_config["mask_count_classes"]), "run_config.json"
    return None, None


def _same_field_key_at(cond_name, c_dim):
    """A registry name for the same field at width ``c_dim``, if one exists.

    Today this only ever finds the mask-count pair: ``mask_count`` (5) and
    ``mask_count8`` (8) both read slot 5 of the dataset tuple and differ only in
    the one-hot width. Finding it turns a "these widths disagree" message into a
    "pass ``mask_count8``" instruction, which is the actual fix.
    """
    try:
        field = conditionals.resolve(cond_name)["field_idx"]
    except ValueError:
        return None
    for name, entry in conditionals.CONDITIONAL_REGISTRY.items():
        if name != cond_name and entry["field_idx"] == field and int(entry["c_dim"]) == int(c_dim):
            return name
    return None


def check_conditional(run_record, conditional, strict=True):
    """Refuse to rebuild a conditional decoder at the wrong input width.

    ``analyze_mouse_latents`` reconstructs the architecture from scratch and then
    loads the checkpoint into it; the only thing that tells it how wide the
    decoder's first layer is, is the ``conditional`` argument. So this is the same
    class of silent mistake as :func:`check_arm`: nothing about the checkpoint
    file announces that it was trained with conditioning, and the failure modes
    are a confusing size error at load time or -- when the widths happen to be
    compatible -- a model that runs and produces figures of nothing.

    The four ways to get it wrong, all treated as problems:

    * the model was trained conditional and none was requested (the decoder comes
      out ``c_dim`` inputs too narrow);
    * the model was trained unconditional and a conditional was requested (too
      wide);
    * both conditional, but on DIFFERENT variables;
    * both conditional on the same variable, but at different widths. This is the
      mask-count trap: the one-hot was 8 classes through April 2025 and is 5
      (1/2/3/4/5+) now, so the same name means two different decoders depending on
      when the checkpoint was trained. The message names ``mask_count8`` when that
      is what the record implies.

    An absent record is NOT a problem by itself: hand-run checkpoints and every
    run predating the conditional driver have no ``conditional`` key, and every
    full-corpus run so far is unconditional, so refusing them would break the
    ordinary path. It only becomes a problem when a conditional was REQUESTED
    against an unverifiable record, which is when the width is a guess -- and
    then only under ``strict``, matching how :func:`check_full_corpus` downgrades
    a missing ``metadata.npz``.

    Returns the usual ``{"notes", "problems"}`` report plus the resolved
    ``conditional`` / ``c_dim`` for the manifest, and ``mask_count_classes``: the
    slot-5 one-hot width the run recorded, so the dataset built here can be built
    at the same width rather than at whatever ``mouse_data`` currently defaults to.
    """
    problems, notes = [], []
    run_config = run_record.get("run_config") or {}

    # Raises with the full list of valid names for an unknown one; returns 0 for
    # None, which is exactly the unconditional decoder's extra input width.
    requested_dim = conditionals.c_dim_of(conditional)
    notes.append(f"requested conditioning: {conditionals.describe(conditional)}")

    recorded_classes, classes_source = _recorded_mask_count_classes(run_record)
    report = {"problems": problems, "notes": notes,
              "conditional": conditional, "c_dim": requested_dim,
              "mask_count_classes": recorded_classes}

    if "conditional" not in run_config:
        notes.append(
            "run_config.json does not record 'conditional' (no record, or a run "
            "predating the conditional driver); the conditioning cannot be "
            "cross-checked here"
        )
        if conditional is not None and strict:
            problems.append(
                f"conditional={conditional!r} was requested but "
                f"{os.path.join(run_record.get('run_dir', '?'), 'run_config.json')} does "
                f"not record what the model was trained with, so the decoder width "
                f"(2*latent_dim + {requested_dim}) is a guess and a wrong guess is not "
                f"recoverable. Pass strict=False to embed anyway."
            )
        return report

    trained = run_config.get("conditional")
    trained_dim = _recorded_c_dim(run_config)

    if trained is None and conditional is None:
        notes.append("model trained unconditional (run_config.json: conditional=null)")
    elif trained is not None and conditional is None:
        problems.append(
            f"CONDITIONING MISMATCH: the model was trained with conditional={trained!r} "
            f"(c_dim={trained_dim}) but no conditional was passed. The decoder's first "
            f"layer would be rebuilt {trained_dim} inputs too narrow, so the checkpoint "
            f"either refuses to load or loads into the wrong architecture. Pass "
            f"conditional={trained!r}."
        )
    elif trained is None and conditional is not None:
        problems.append(
            f"CONDITIONING MISMATCH: the model was trained UNCONDITIONAL "
            f"(run_config.json: conditional=null) but conditional={conditional!r} was "
            f"requested, which widens the decoder's first layer by {requested_dim}. "
            f"Pass conditional=None."
        )
    elif trained != conditional:
        problems.append(
            f"CONDITIONING MISMATCH: the model was trained on {trained!r} "
            f"(c_dim={trained_dim}) but {conditional!r} (c_dim={requested_dim}) was "
            f"requested. Even at equal widths these are different variables and the "
            f"figures would be conditioned on the wrong one. Pass "
            f"conditional={trained!r}."
        )
    elif trained_dim is None:
        notes.append(f"conditioning name matches ({conditional!r}) but the run record "
                     f"carries no c_dim to check the width against")
    elif int(trained_dim) != int(requested_dim):
        alternative = _same_field_key_at(conditional, trained_dim)
        fix = (f"Pass conditional={alternative!r}: same variable, c_dim={trained_dim}."
               if alternative else
               f"The registry has no {conditional!r} variant at c_dim={trained_dim}.")
        problems.append(
            f"CONDITIONING WIDTH MISMATCH: the model was trained on {conditional!r} at "
            f"c_dim={trained_dim}, but {conditional!r} is c_dim={requested_dim} in the "
            f"current registry. The name is the same and the decoder is not. {fix}"
        )
    else:
        notes.append(f"conditioning matches the run record: {conditional!r} "
                     f"(c_dim={requested_dim})")

    # The slot-5 one-hot width is a property of the DATASET, not of the checkpoint,
    # so a mask-count model needs the inference set built the same way the training
    # set was or the decoder is fed a vector of the wrong length. Only the
    # mask-count conditionals read slot 5; for anything else the width is recorded
    # for provenance and forwarded unchanged.
    if recorded_classes is not None:
        notes.append(f"run record built slot 5 as a {int(recorded_classes)}-class "
                     f"one-hot (from {classes_source})")
        if conditional is not None:
            try:
                reads_slot5 = conditionals.resolve(conditional)["field_idx"] == 5
            except ValueError:                           # already reported above
                reads_slot5 = False
            wanted = conditionals.mask_count_classes_for(conditional)
            if reads_slot5 and int(recorded_classes) != int(wanted):
                alternative = _same_field_key_at(conditional, recorded_classes)
                fix = (f"Pass conditional={alternative!r}."
                       if alternative else
                       f"No registry entry builds a {int(recorded_classes)}-class one-hot.")
                problems.append(
                    f"MASK-COUNT WIDTH MISMATCH: the training set built slot 5 as a "
                    f"{int(recorded_classes)}-class one-hot, but conditional="
                    f"{conditional!r} asks mouse_data for {wanted} classes. {fix}"
                )
    return report


def _git_commit(path):
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(path)), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:                                    # noqa: BLE001 - provenance only
        return None


def _driver_fingerprint():
    """SHA-256 (first 12) of the analysis modules that actually produced the figures.

    ``git_commit`` alone is not enough provenance here: this tree carries the
    analysis code as uncommitted working-tree changes, so HEAD can sit still while
    the driver changes underneath two runs. That has already bitten -- two
    full-corpus runs recorded the SAME git_commit and wrote DIFFERENT figure sets
    (individual figure_E_embedded_by_*.jpg vs the consolidated
    figure_E_embedded_grid.png), with nothing in either manifest able to tell them
    apart. Hash the files instead, so the manifest identifies the code that ran.
    """
    import hashlib

    out = {}
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("analyze_all_sessions.py", "analyze_mouse_latents_2d.py"):
        path = os.path.join(here, name)
        try:
            with open(path, "rb") as handle:
                out[name] = hashlib.sha256(handle.read()).hexdigest()[:12]
        except OSError:                                  # noqa: PERF203 - provenance only
            out[name] = None
    return out


def analyze_all_sessions(
    model_path,
    dataloc,
    save_dir,
    # --- preflight ---
    min_n=300000,
    strict=True,
    dry_run=False,
    # --- analysis settings: the inference-launcher values, forwarded verbatim ---
    lattice_m=24,
    bandwidth=0.4,
    batch_size=512,
    grid_size=15,
    n_per_cluster=36,
    sample_from_centroid=True,
    cache_posteriors=True,
    use_fast_mean_shift=True,
    compute_recon=True,
    n_dur_bins=5,
    filter_mask=False,
    # --- conditioning: must match what the checkpoint was trained with ---
    conditional=None,
    **analyze_kwargs,
):
    """Embed every spectrogram in the full session corpus and write the figure set.

    Args:
        model_path: Trained QLVM checkpoint (.tar). Its directory must hold the
            run's ``run_config.json`` / ``sampling_config.json`` for the arm check.
        dataloc: Directory holding the FULL-CORPUS ``full_data.npz`` (or .pt) plus
            the build's ``metadata.npz`` -- what
            ``scripts/dataset_construct/build_qlvm_all_sessions.sh`` writes.
        save_dir: Output directory. Use ``latents_all_sessions/``, NOT
            ``latents_all/`` -- the latter holds the draw-scoped comparison set.
        min_n: Refuse a dataset whose build wrote fewer than this many samples.
            Default 300000: the 325-session pool yields 365,150 and the largest
            draw is 151,421, so this separates them with room to spare. Pass None
            to skip the size check (e.g. for a smoke subset of session roots).
        strict: Refuse a dataset directory with no ``metadata.npz`` (the build
            flags would be unverifiable). False downgrades that to a warning.
        dry_run: Run the preflight, print the report and the resolved settings,
            and stop without loading data or writing anything.
        lattice_m: Fibonacci lattice parameter. 24 -> 46,368 lattice points, the
            value the per-cell inference jobs use.
        bandwidth: Mean-shift bandwidth (0.4). With use_fast_mean_shift the shift
            runs over the LATTICE, not the samples, so its cost does not grow with
            the corpus; only the per-sample nearest-centre assignment does.
        batch_size: Posterior/metadata dataloader batch size.
        grid_size: Decoder grid figure resolution (grid_size x grid_size).
        n_per_cluster: Example spectrograms per watershed cluster figure.
        sample_from_centroid: Spiral outward from each cluster centroid when
            picking those examples (True) instead of tiling its bounding box.
        cache_posteriors: Write/reuse ``posterior_cache.npz`` in save_dir. Keep
            True -- the posterior pass is the expensive GPU stage, and a re-run
            for a figure tweak should not repeat it. DELETE the cache if the
            dataset changes; it is keyed by nothing but the directory.
        use_fast_mean_shift: Lattice-based mean shift (True). The O(N^2) sample
            based path is not viable at 365k.
        compute_recon: Round-trip MSE bars + ``recon_mse_breakdown.npz``.
        n_dur_bins: Duration quantile bins in the MSE bar chart.
        filter_mask: Leave False. The shipped [1,8] mask-count filter would drop
            the open-ended top stratum of the corpus.
        conditional: The conditioning variable the CHECKPOINT was trained with --
            a key of ``data.conditionals`` ("duration", "mean_freq", "mask_count",
            "mask_count8") -- or None for an unconditional model. Default None
            because every full-corpus run to date is unconditional, and because
            this is the one analysis argument that changes the ARCHITECTURE rather
            than the figures: ``c_dim`` widens the decoder's first linear layer, so
            a wrong value does not produce a worse embedding, it produces a decoder
            the checkpoint does not fit. :func:`check_conditional` cross-checks it
            against the run's own ``run_config.json`` before anything is loaded.
            Note that the mask-count one-hot changed width -- checkpoints trained
            before the 5-class (1/2/3/4/5+) build need "mask_count8", not
            "mask_count".
        **analyze_kwargs: Anything else ``analyze_mouse_latents`` takes
            (``scatter_size``, ``scatter_alpha``, ``beh_features_path``,
            ``recon_batch_size``, ``ms_*``, ...). ``apply_mask`` and
            ``total_samples`` are REFUSED: the dataset declares its own masking
            arm, and subsampling contradicts the point of this entry point.

    Returns:
        The manifest dict that was written to ``save_dir/manifest.json``.
    """
    for banned, why in (
        ("apply_mask", "the dataset declares its own masking arm and mouse_data honours it; "
                       "overriding it here is how a model gets evaluated on inputs it was "
                       "not trained on"),
        ("total_samples", "this entry point exists to embed EVERY spectrogram; use "
                          "analyze_mouse_latents directly for a subsample"),
    ):
        if banned in analyze_kwargs:
            raise ValueError(f"analyze_all_sessions refuses {banned!r}: {why}.")

    # Fail on an unknown conditioning name here rather than several minutes into a
    # corpus load. resolve() raises with the list of valid names.
    if conditional is not None:
        conditionals.resolve(conditional)

    model_path = os.path.abspath(model_path)
    dataloc = os.path.abspath(dataloc)
    save_dir = os.path.abspath(save_dir)

    if os.path.basename(save_dir) == "latents_all":
        raise ValueError(
            "save_dir ends in 'latents_all', which is the draw-scoped figure set the "
            "per-cell inference jobs publish. Write to 'latents_all_sessions' instead."
        )

    metadata = read_dataset_metadata(dataloc)
    run_record = read_run_record(model_path)
    corpus = check_full_corpus(dataloc, metadata, min_n=min_n, strict=strict)
    arm = check_arm(metadata, run_record)
    cond = check_conditional(run_record, conditional, strict=strict)

    print("=== analyze_all_sessions preflight ===")
    print(f"  model    : {model_path}")
    print(f"  dataloc  : {dataloc}")
    print(f"  save_dir : {save_dir}")
    for note in corpus["notes"] + arm["notes"] + cond["notes"]:
        print(f"  ok       : {note}")
    problems = corpus["problems"] + arm["problems"] + cond["problems"]
    for problem in problems:
        print(f"  PROBLEM  : {problem}")
    if problems:
        raise ValueError(
            f"{len(problems)} preflight problem(s); refusing to run. "
            f"See the PROBLEM lines above."
        )
    print("  preflight passed.\n")

    settings = dict(
        lattice_m=lattice_m, bandwidth=bandwidth, batch_size=batch_size,
        grid_size=grid_size, n_per_cluster=n_per_cluster,
        sample_from_centroid=sample_from_centroid, cache_posteriors=cache_posteriors,
        use_fast_mean_shift=use_fast_mean_shift, compute_recon=compute_recon,
        n_dur_bins=n_dur_bins, filter_mask=filter_mask, conditional=conditional,
        **analyze_kwargs,
    )
    # Reproduce the slot-5 one-hot width the training set was built with, so the
    # conditioning vector this run feeds the decoder is the length it was trained
    # on. Only forwarded when the run actually recorded a width and the caller did
    # not pin one: left out, analyze_mouse_latents derives it from `conditional`
    # and mouse_data keeps its own default, which is the pre-existing behaviour.
    if cond["mask_count_classes"] is not None and "mask_count_classes" not in analyze_kwargs:
        settings["mask_count_classes"] = int(cond["mask_count_classes"])
    if dry_run:
        print("dry_run=True; settings that WOULD be used:")
        print(json.dumps(settings, indent=2, default=str))
        return {"dry_run": True, "settings": settings,
                "dataset_metadata": metadata, "run_record": run_record}

    os.makedirs(save_dir, exist_ok=True)
    started = time.time()
    result = analyze_mouse_latents(
        model_path=model_path, dataloc=dataloc, save_dir=save_dir, **settings,
    )
    elapsed = time.time() - started

    figures = sorted(
        (name, os.path.getsize(os.path.join(save_dir, name)))
        for name in os.listdir(save_dir)
        if name.lower().endswith((".png", ".jpg", ".svg", ".pdf", ".mp4"))
    )
    type_report = metadata.get("type_report") if isinstance(metadata.get("type_report"), dict) else {}

    manifest = {
        "entry_point": "qmc_deep_gen/analyze_all_sessions.py:analyze_all_sessions",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_time_s": round(elapsed, 1),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "git_commit": _git_commit(__file__),
        # git_commit records HEAD, which does not move when the driver is an
        # uncommitted change; the hashes below identify the code that actually ran.
        "driver_sha256": _driver_fingerprint(),
        "slurm": {key: os.environ[key] for key in
                  ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                   "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES") if key in os.environ},
        "model": {
            "path": model_path,
            "run_dir": run_record["run_dir"],
            "run_config": run_record["run_config"],
            "sampling_config": run_record["sampling_config"],
            "trained_apply_mask": arm["model_arm"],
            # The conditioning the decoder was actually rebuilt with. Recorded
            # beside trained_apply_mask because it is the same kind of fact: not a
            # figure setting but a statement about which model this figure set is of.
            "conditional": cond["conditional"],
            "c_dim": cond["c_dim"],
        },
        "dataset": {
            "path": dataloc,
            "apply_mask": arm["data_arm"],
            "n_built": metadata.get("n_full"),
            "n_embedded": int(len(result["latent_coords"])),
            "length_threshold": metadata.get("length_threshold"),
            "target_shape": metadata.get("target_shape"),
            "masking_type": metadata.get("masking_type"),
            "require_mask": metadata.get("require_mask"),
            "time_stretch": metadata.get("time_stretch"),
            "random_state": metadata.get("random_state"),
            "session_type_targets": metadata.get("session_type_targets"),
            "n_sessions": len(metadata.get("session_type_by_key") or {}),
            "per_type": {t: r.get("drawn") for t, r in type_report.items()},
        },
        "analysis": dict(settings, n_lattice_points=None),
        "clusters": {
            "n_clusters": int(len(result["centers"])),
            "sizes": [int(np.sum(result["labels"] == i)) for i in range(len(result["centers"]))],
            "bandwidth": bandwidth,
        },
        "figures": [{"name": name, "bytes": size} for name, size in figures],
        "n_figures": len(figures),
    }
    # Lattice size is a property of lattice_m, but recording the realized count
    # saves a reader from having to know gen_fib_basis' convention.
    try:
        from models.sampling import gen_fib_basis
        manifest["analysis"]["n_lattice_points"] = int(len(gen_fib_basis(m=lattice_m)))
    except Exception:                                    # noqa: BLE001 - provenance only
        manifest["analysis"].pop("n_lattice_points", None)

    manifest_path = os.path.join(save_dir, "manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print(f"\n=== analyze_all_sessions complete ===")
    print(f"  embedded  : {manifest['dataset']['n_embedded']:,} spectrograms "
          f"from {manifest['dataset']['n_sessions']} sessions")
    print(f"  arm       : apply_mask={arm['data_arm']}")
    print(f"  clusters  : {manifest['clusters']['n_clusters']}")
    print(f"  figures   : {len(figures)}")
    print(f"  wall time : {elapsed / 60:.1f} min")
    print(f"  manifest  : {manifest_path}")
    return manifest


if __name__ == "__main__":
    fire.Fire(analyze_all_sessions)
