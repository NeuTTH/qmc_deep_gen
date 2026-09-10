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
    * ``require_mask`` is on and ``masking_type == 'sam'``;
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

    print("=== analyze_all_sessions preflight ===")
    print(f"  model    : {model_path}")
    print(f"  dataloc  : {dataloc}")
    print(f"  save_dir : {save_dir}")
    for note in corpus["notes"] + arm["notes"]:
        print(f"  ok       : {note}")
    problems = corpus["problems"] + arm["problems"]
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
        n_dur_bins=n_dur_bins, filter_mask=filter_mask, **analyze_kwargs,
    )
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
