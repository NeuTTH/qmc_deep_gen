"""Embed a QLVM in its full corpus, then cluster the latent torus: one command, two stages.

``analyze_mouse_latents_2d.py`` no longer clusters (``segment_clusters=False``): it computes
the posterior over the lattice, writes ``posterior_cache.npz`` and draws the figure set,
including the decoder grid ``figure_grid_examples.png``. The clustering is done by
``scripts/qlvm_latent_clustering`` in the MMMmB repo: mean shift, a stitched periodic
watershed, a valley/ridge merge of unsupported boundaries, bootstrap stability and external
validity, reported at the frozen parameters. This wrapper runs the two back to back.

STAGES
------
1. **Embed** -- :func:`analyze_all_sessions.analyze_all_sessions`, unchanged: preflight
   (masking arm, full corpus, conditioning), posterior pass, ``posterior_cache.npz``, the
   figure set (``figure_grid_examples.png``, ``figure_E_embedded_grid.png``,
   ``figure_recon_mse.png``) and ``manifest.json``. A valid cache in ``save_dir`` is read
   back instead of recomputed.
2. **Cluster** -- ``pipeline.run_one`` from the clustering package, in a fresh Python
   process (so the ~48 GB corpus the embedding held is gone), on ``save_dir``'s cache. It
   writes ``clusters/`` (final per-call labels and figures) and ``supplementary/`` (every
   step's figures and metrics), then records where in ``manifest.json``
   (``latent_clustering``) -- unless ``--out_dir`` / ``--out_root`` made it an experiment.

WHERE THE CLUSTERING IS WRITTEN (first that applies)
----------------------------------------------------
1. ``--out_dir``: exactly there.
2. ``--run_key stage/phase/cell``, or a results-tree cell whose ``latent_full_dset`` link
   resolves to ``save_dir`` (found automatically): ``<out_root>/<stage>/<phase>/<cell>/``,
   where ``out_root`` defaults to ``results/qlvm_playpen_runs/inference`` -- the same place
   ``cluster_latents.py run`` writes.
3. Otherwise ``<save_dir>/latent_clustering/`` (``<out_root>/adhoc/...`` if ``--out_root``
   is given).

Parameters come from ``<params_root>/frozen_params.json`` (default: the canonical
``inference/frozen_params.json``, i.e. bandwidth 0.5, σ 3, merged), then the override flags.
A run whose parameters differ from the frozen ones is refused when its destination is inside
the canonical ``inference/`` tree, so an experiment cannot overwrite the frozen results; give
it an ``--out_dir`` or ``--out_root``.

Usage (fire CLI)::

    # both stages
    python analyze_and_cluster_latents.py run \\
        --model_path /scratch/.../qmc_train_mouse_experiment.tar \\
        --dataloc /mnt/cup/labs/.../qlvm_all_sessions/all-plus-mute_n402_len128_seed42 \\
        --save_dir /scratch/.../latent_full_dset

    # stage 2 only, on an existing embedding (reads model/corpus from its manifest.json)
    python analyze_and_cluster_latents.py cluster --save_dir /scratch/.../latent_full_dset

    python analyze_and_cluster_latents.py run -- --help       # authoritative flag list

Environment: ``QLVM_CLUSTERING_DIR`` (default ``<MMMmB>/scripts/qlvm_latent_clustering``,
where ``<MMMmB>`` is this checkout's parent directory), ``MMMMB_ROOT`` (default that parent),
``QMC_DEEP_GEN_ROOT`` (default this file's directory).
"""

import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
QMC_ROOT = os.environ.get("QMC_DEEP_GEN_ROOT", _HERE)
MMMMB_ROOT = os.environ.get("MMMMB_ROOT", os.path.dirname(QMC_ROOT))
CLUSTERING_DIR = os.environ.get("QLVM_CLUSTERING_DIR",
                                os.path.join(MMMMB_ROOT, "scripts", "qlvm_latent_clustering"))
SESSION_CONDITIONS = os.path.join(MMMMB_ROOT, "results", "model_training_datasets",
                                  "session_conditions.csv")
if QMC_ROOT not in sys.path:
    sys.path.insert(0, QMC_ROOT)

# The clustering overrides both commands accept; None = the frozen value.
PARAM_FLAGS = ("bandwidth", "sigma", "compactness", "n_boot", "valley_thr", "ridge_thr",
               "min_basin_frac", "final_level", "seed", "rough_max_clusters")


def _clustering_modules():
    """Import the clustering package's ``runs`` and ``pipeline`` modules.

    Imported lazily, and only in the stage-2 process: the package runs its mean-shift
    ladders and bootstrap resamples in *spawned* workers, which re-import this file, so
    nothing heavy may be imported at module level.
    """
    if not os.path.isfile(os.path.join(CLUSTERING_DIR, "pipeline.py")):
        raise FileNotFoundError(
            f"no clustering package at {CLUSTERING_DIR}; set QLVM_CLUSTERING_DIR to the "
            f"MMMmB scripts/qlvm_latent_clustering directory")
    # The package reads these at import; point them at this checkout unless set.
    os.environ.setdefault("QMC_DEEP_GEN_ROOT", QMC_ROOT)
    os.environ.setdefault("MMMMB_ROOT", MMMMB_ROOT)
    if CLUSTERING_DIR not in sys.path:
        sys.path.insert(0, CLUSTERING_DIR)
    import pipeline
    import runs
    return runs, pipeline


def _read_manifest(save_dir):
    path = os.path.join(save_dir, "manifest.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def _find_tree_run(runs, cache):
    """The results-tree run whose posterior cache is this file, or None."""
    real = os.path.realpath(cache)
    for run in runs.discover():
        if os.path.realpath(run.cache) == real:
            return run
    return None


def _plan(save_dir, run_key=None, out_dir=None, out_root=None, params_root=None,
          checkpoint=None, corpus=None, conditional=None, family=None, overrides=None):
    """Resolve everything stage 2 needs without running it. Returns a dict."""
    import dataclasses

    runs, pipeline = _clustering_modules()
    save_dir = os.path.abspath(save_dir)
    cache = os.path.join(save_dir, "posterior_cache.npz")
    manifest = _read_manifest(save_dir)
    checkpoint = checkpoint or (manifest.get("model") or {}).get("path")
    corpus = corpus or (manifest.get("dataset") or {}).get("path")
    if conditional is None:
        conditional = (manifest.get("model") or {}).get("conditional")
    if conditional is None and os.path.isfile(cache):
        conditional = runs._conditional_of(cache, "")

    tree_run = None
    if run_key is None and os.path.isfile(cache):
        tree_run = _find_tree_run(runs, cache)
        run_key = tree_run.key if tree_run else None
    if run_key is not None:
        parts = run_key.strip("/").split("/")
        if len(parts) != 3:
            raise ValueError(f"run_key must be stage/phase/cell, got {run_key!r}")
        stage, phase, cell = parts
    else:
        model_dir = os.path.dirname(os.path.abspath(checkpoint)) if checkpoint else save_dir
        stage, phase, cell = "adhoc", os.path.basename(os.path.dirname(model_dir)), \
            os.path.basename(model_dir)
    family = family or (tree_run.family if tree_run else "stock")

    if out_dir:
        dest = os.path.abspath(out_dir)
    elif run_key is not None:
        dest = os.path.join(os.path.abspath(out_root or runs.INFERENCE_OUT), stage, phase, cell)
    elif out_root:
        dest = os.path.join(os.path.abspath(out_root), stage, phase, cell)
    else:
        dest = os.path.join(save_dir, "latent_clustering")

    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    params_root = os.path.abspath(params_root or runs.INFERENCE_OUT)
    params = pipeline.load_params(params_root, overrides)
    frozen = pipeline.load_params(runs.INFERENCE_OUT)
    departures = {k: params[k] for k in pipeline.DEFAULTS if params[k] != frozen[k]}
    canonical = os.path.realpath(runs.INFERENCE_OUT)
    if departures and (os.path.realpath(dest) + os.sep).startswith(canonical + os.sep):
        raise ValueError(
            f"parameters {departures} differ from {frozen['_source']}, and {dest} is inside the "
            f"canonical inference tree, which holds the frozen-parameter results. Pass --out_dir "
            f"or --out_root for this run.")

    @dataclasses.dataclass
    class _Run(runs.Run):
        """A ``runs.Run`` whose output directory is fixed rather than derived from a root."""
        dest: str = ""

        def out_dir(self, root=None):
            return self.dest

    run = _Run(stage=stage, phase=phase, cell=cell, cache=cache, checkpoint=checkpoint,
               family=family, corpus=corpus, conditional=conditional or None, dest=dest)
    return {"run": run, "params": params, "departures": departures, "manifest": manifest,
            "found_in_tree": tree_run is not None}


def _print_plan(plan):
    run, params = plan["run"], plan["params"]
    print("=== latent clustering plan ===")
    print(f"  run       : {run.key}" + ("  (found in the results tree)" if plan["found_in_tree"] else ""))
    print(f"  cache     : {run.cache}")
    print(f"  checkpoint: {run.checkpoint}  (family {run.family}, conditional {run.conditional})")
    print(f"  corpus    : {run.corpus}")
    print(f"  output    : {run.dest}")
    print(f"  params    : bandwidth {params['bandwidth']}, sigma {params['sigma']}, "
          f"compactness {params['compactness']}, final_level {params['final_level']}, "
          f"min_basin_frac {params['min_basin_frac']}, n_boot {params['n_boot']}, "
          f"rough_max_clusters {params.get('rough_max_clusters')}")
    print(f"  from      : {params['_source']}"
          + (f"; differs from frozen: {plan['departures']}" if plan["departures"] else ""))


def cluster(save_dir, run_key=None, out_dir=None, out_root=None, params_root=None,
            checkpoint=None, corpus=None, conditional=None, family=None, n_jobs=8,
            device="cuda", jacobian=True, bootstrap=True, external=True, behaviour=True,
            report=True, features=True, freq_range_khz=(30.0, 120.0), dry_run=False,
            bandwidth=None, sigma=None, compactness=None, n_boot=None, valley_thr=None,
            ridge_thr=None, min_basin_frac=None, final_level=None, seed=None,
            rough_max_clusters=None):
    """Stage 2 only: cluster the latent torus of an existing embedding.

    Args:
        save_dir: directory holding ``posterior_cache.npz`` (and normally the
            ``manifest.json`` stage 1 wrote). *(required)*
        run_key: ``stage/phase/cell`` to name and place the run in the results tree.
            Default: found from the results-tree link to ``save_dir``, else ``adhoc``.
        out_dir: write here exactly, overriding ``run_key`` / ``out_root`` placement.
        out_root: root for ``<stage>/<phase>/<cell>/``. *(config)* -- default
            ``results/qlvm_playpen_runs/inference`` for a tree run, else
            ``<save_dir>/latent_clustering`` is used instead of a root.
        params_root: directory whose ``frozen_params.json`` supplies the parameters.
            *(config)* -- default ``results/qlvm_playpen_runs/inference``.
        checkpoint: decoder checkpoint (Jacobian, decoded report panels). *(config)* --
            default ``manifest.json`` ``model.path``.
        corpus: corpus directory with ``full_data.npz`` (metadata, examples). *(config)* --
            default ``manifest.json`` ``dataset.path``.
        conditional: conditioning variable. *(config)* -- default ``manifest.json``
            ``model.conditional``, else the cache's own stamp.
        family: decoder family: stock | canonical | hybrid | plain. *(config)* -- default
            the tree run's family, else ``stock`` (the only one this driver embeds).
        n_jobs: processes for mean-shift ladders and bootstrap resamples.
        device: torch device for the Jacobian and the decoder report panels.
        jacobian: compute the decoder Jacobian (ridge tests). Needs a checkpoint and a GPU.
        bootstrap: step 6, session bootstrap stability and parameter sensitivity.
        external: step 7, η², enrichment and repertoire. Needs the corpus metadata.
        behaviour: join emitter sex / social distance from the behaviour pickle.
        report: write the headline ``clusters/`` report after the steps.
        features: compute the corpus' mean-frequency / bandwidth sidecar (needed for their
            η²) when it does not exist yet. About 2 min for the 445,725-call corpus.
        freq_range_khz: frequency axis of the spectrogram rows, for that sidecar.
        dry_run: print the resolved plan and stop.
        bandwidth, sigma, compactness, n_boot, valley_thr, ridge_thr, min_basin_frac,
            final_level, seed, rough_max_clusters: override the frozen parameters for this run
            only. *(config)* -- default from ``<params_root>/frozen_params.json``, else
            ``pipeline.DEFAULTS`` (``rough_max_clusters`` 9). See the package README,
            "Controlling the cluster count".
    """
    overrides = dict(bandwidth=bandwidth, sigma=sigma, compactness=compactness, n_boot=n_boot,
                     valley_thr=valley_thr, ridge_thr=ridge_thr, min_basin_frac=min_basin_frac,
                     final_level=final_level, seed=seed, rough_max_clusters=rough_max_clusters)
    plan = _plan(save_dir, run_key, out_dir, out_root, params_root, checkpoint, corpus,
                 conditional, family, overrides)
    _print_plan(plan)
    if dry_run:
        return None
    run, params = plan["run"], plan["params"]
    if not os.path.isfile(run.cache):
        raise FileNotFoundError(f"{run.cache} does not exist; run stage 1 (run) first")

    runs, pipeline = _clustering_modules()
    if features and run.corpus and not os.path.exists(runs.features_path(run.corpus)):
        print(f"computing the acoustic-feature sidecar for {run.corpus}", flush=True)
        runs.compute_acoustic_features(run.corpus, None, tuple(freq_range_khz))

    started = time.time()
    summary = pipeline.run_one(run, os.path.dirname(run.dest), params, n_jobs=n_jobs,
                               device=device, jacobian=jacobian, bootstrap=bootstrap,
                               external=external, behaviour=behaviour, report=report)
    final = summary.get("final") or {}
    record = {
        "entry_point": "qmc_deep_gen/analyze_and_cluster_latents.py:cluster",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_time_s": round(time.time() - started, 1),
        "clustering_package": CLUSTERING_DIR,
        "run": run.key, "output": run.dest,
        "clusters_dir": final.get("dir"),
        "params_source": params["_source"], "departures_from_frozen": plan["departures"],
        "bandwidth": params["bandwidth"], "sigma": params["sigma"],
        "final_level": final.get("level", params["final_level"]),
        "n_clusters": final.get("n_clusters"), "n_stable": final.get("n_stable"),
        "fine_k": summary.get("fine_k"), "merged_k": summary.get("merged_k"),
        "rough_k": summary.get("rough_k"), "rough_stable": summary.get("rough_stable_k"),
        "rough_dir": (summary.get("rough") or {}).get("dir"),
        "coarse_k": summary.get("coarse_k"),
    }
    manifest_path = os.path.join(os.path.dirname(run.cache), "manifest.json")
    # Only the default placement is the embedding's clustering; an --out_dir / --out_root run
    # is an experiment and must not overwrite that record.
    if os.path.isfile(manifest_path) and not (out_dir or out_root):
        manifest = _read_manifest(os.path.dirname(run.cache))
        manifest["latent_clustering"] = record
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2, default=str)

    print("\n=== latent clustering complete ===")
    print(f"  fine {record['fine_k']} -> merged {record['merged_k']} basins; rough {record['rough_k']} "
          f"({record['rough_stable']} stable); coarse {record['coarse_k']}")
    print(f"  final     : {record['n_clusters']} {record['final_level']} clusters, "
          f"{record['n_stable']} stable")
    print(f"  labels    : {os.path.join(record['clusters_dir'] or run.dest, 'cluster_labels.csv')}")
    print(f"  wall time : {record['wall_time_s'] / 60:.1f} min")
    return None


def run(model_path, dataloc, save_dir, embed=True, cluster_latents=True, min_n=300000,
        strict=True, conditional=None, session_conditions=SESSION_CONDITIONS, dry_run=False,
        run_key=None, out_dir=None, out_root=None, params_root=None, n_jobs=8, device="cuda",
        jacobian=True, bootstrap=True, external=True, behaviour=True, report=True,
        features=True, bandwidth=None, sigma=None, compactness=None, n_boot=None,
        valley_thr=None, ridge_thr=None, min_basin_frac=None, final_level=None, seed=None,
        rough_max_clusters=None, **embed_kwargs):
    """Stage 1 (posterior, decoder grid and figure set) then stage 2 (latent clustering).

    Args:
        model_path: trained QLVM checkpoint (.tar); its directory holds the run's
            ``run_config.json`` / ``sampling_config.json``. *(required)*
        dataloc: full-corpus directory (``full_data.npz`` + ``metadata.npz``). *(required)*
        save_dir: embedding output, conventionally ``<run_dir>/latent_full_dset``. *(required)*
        embed: run stage 1. False clusters an existing ``save_dir`` only.
        cluster_latents: run stage 2. False stops after the embedding.
        min_n: stage 1 preflight: refuse a corpus that built fewer rows (None skips).
        strict: stage 1 preflight: refuse a corpus with no ``metadata.npz``.
        conditional: conditioning variable the checkpoint was trained with, or None.
        session_conditions: ``session_id,condition`` CSV that splits ``MF`` into
            ``MF_intact`` / ``MF_mute``, as every phase launcher passes. *(config)* --
            default ``<MMMmB>/results/model_training_datasets/session_conditions.csv``;
            pass None to leave session types unrefined.
        dry_run: stage 1 preflight and the stage-2 plan only; nothing is loaded or written.
        run_key, out_dir, out_root, params_root, n_jobs, device, jacobian, bootstrap,
            external, behaviour, report, features: stage 2, as in :func:`cluster`.
        bandwidth, sigma, compactness, n_boot, valley_thr, ridge_thr, min_basin_frac,
            final_level, seed, rough_max_clusters: stage 2 parameter overrides, as in
            :func:`cluster`.
        **embed_kwargs: every other keyword goes to ``analyze_all_sessions`` and on to
            ``analyze_mouse_latents`` (``batch_size`` 512, ``grid_size`` 15,
            ``compute_recon`` True, ``lattice_m`` 24, ``freq_range_khz`` (30, 120), ...;
            see ``analyze_all_sessions.py -- --help``). ``segment_clusters`` is refused:
            the clustering is stage 2.
    """
    if embed_kwargs.get("segment_clusters"):
        raise ValueError("segment_clusters=True reruns the old in-driver clustering; the "
                         "clustering here is stage 2 (scripts/qlvm_latent_clustering)")
    if embed_kwargs.get("lattice_m", 24) != 24:
        raise ValueError("the clustering package reads a lattice_m=24 cache; keep lattice_m 24")
    model_path, dataloc, save_dir = (os.path.abspath(p) for p in (model_path, dataloc, save_dir))
    run_config_path = os.path.join(os.path.dirname(model_path), "run_config.json")
    run_config = {}
    if os.path.isfile(run_config_path):
        with open(run_config_path) as handle:
            run_config = json.load(handle)
    if (embed or dry_run) and (run_config.get("dataset_floor") is not None
                               or "floor" in str(run_config.get("phase", ""))):
        # The floor was baked into the training arrays; no corpus carries it, and the
        # masking-arm preflight passes on the unmasked corpus, so the stock driver would
        # embed unfloored inputs without complaint.
        raise ValueError(
            f"{run_config_path} describes a floor run ({run_config.get('phase')!r}); the stock driver "
            f"would embed it WITHOUT the floor. Embed it with its adapter (phase 6: "
            f"qlvm_phase6_floor/scripts/embed_phase6_all_sessions.py), then cluster with "
            f"`cluster --save_dir <its latent_full_dset>`.")

    if embed or dry_run:
        import analyze_all_sessions as aas
        settings = dict(embed_kwargs)
        if session_conditions is not None:
            if os.path.isfile(session_conditions):
                settings["session_conditions"] = session_conditions
            elif session_conditions == SESSION_CONDITIONS:
                print(f"  no {SESSION_CONDITIONS}; session types left unrefined", flush=True)
            else:
                raise FileNotFoundError(f"session_conditions: {session_conditions} does not exist")
        print("=== stage 1: posterior, decoder grid and figure set ===", flush=True)
        aas.analyze_all_sessions(model_path=model_path, dataloc=dataloc, save_dir=save_dir,
                                 min_n=min_n, strict=strict, dry_run=dry_run,
                                 conditional=conditional, **settings)
    if not cluster_latents:
        return None

    print("\n=== stage 2: latent clustering (scripts/qlvm_latent_clustering) ===", flush=True)
    flags = dict(save_dir=save_dir, run_key=run_key, out_dir=out_dir, out_root=out_root,
                 params_root=params_root, checkpoint=model_path, corpus=dataloc,
                 conditional=conditional, n_jobs=n_jobs, device=device, jacobian=jacobian,
                 bootstrap=bootstrap, external=external, behaviour=behaviour, report=report,
                 features=features, dry_run=dry_run, bandwidth=bandwidth, sigma=sigma,
                 compactness=compactness, n_boot=n_boot, valley_thr=valley_thr,
                 ridge_thr=ridge_thr, min_basin_frac=min_basin_frac, final_level=final_level,
                 seed=seed, rough_max_clusters=rough_max_clusters)
    if "freq_range_khz" in embed_kwargs and embed_kwargs["freq_range_khz"] is not None:
        flags["freq_range_khz"] = list(embed_kwargs["freq_range_khz"])
    # A fresh process: the embedding's corpus arrays and CUDA context do not follow it.
    argv = [sys.executable, os.path.abspath(__file__), "cluster"]
    argv += [f"--{k}={v}" for k, v in flags.items() if v is not None]
    subprocess.run(argv, check=True)
    return None


if __name__ == "__main__":
    import fire
    fire.Fire({"run": run, "cluster": cluster})
