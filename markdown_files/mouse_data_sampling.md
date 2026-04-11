# Mouse Data Sampling

## Overview

`data/mouse_data.py` — `mouse_data` dataset class.

Sampling is two-stage: **filtering** then **strategy**. Both are optional and controlled entirely by constructor arguments, which map 1-to-1 to `fire` CLI flags.

---

## Stage 1 — Mask Filtering

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filter_mask` | bool | `False` | Enable filtering by `masks_len` range |
| `lo` | int | `1` | Minimum `masks_len` (inclusive) |
| `hi` | int | `8` | Maximum `masks_len` (inclusive) |

When `filter_mask=False`, the full dataset is passed to Stage 2 unchanged.

---

## Stage 2 — Sampling Strategy

Controlled by `sampling_strategy`. Default (`None`) returns the full (possibly filtered) dataset.

### `None` — Full dataset
No sampling applied. Use when you want all data.

---

### `"mask_duration"` — Per-bin quota with duration coverage

Groups data by `masks_len`, then within each bin:
- If `bin_size <= samples_per_mask`: keep all
- Else: pick exactly `samples_per_mask` entries, **quantile-evenly spaced** across the duration distribution (`_quantile_sample`)

Duration-awareness is always on in this mode. No oversampling in any case.

| Parameter | Type | Description |
|---|---|---|
| `samples_per_mask` | int | Max samples per `masks_len` bin |

---

### `"subsample"` — Proportional subsample

Draws `total_samples` total while preserving the natural `masks_len` distribution.

For each bin:
```
n_bin = floor(total_samples * bin_size / total)
n_bin = min(n_bin, bin_size)   # no oversampling
```

If `total_samples >= len(dataset)`, the full dataset is returned unchanged.

Within each bin, selection is either random or duration-aware:

| `duration_aware` | Within-bin selection |
|---|---|
| `False` (default) | `rng.choice(bin_inds, n_bin, replace=False)` |
| `True` | Quantile-evenly spaced by duration (same as `mask_duration` mode) |

| Parameter | Type | Default | Description |
|---|---|---|---|
| `total_samples` | int | `None` | Target total number of samples |
| `duration_aware` | bool | `False` | Use duration-quantile selection within bins |

> Note: returned count may be slightly below `total_samples` due to `floor()` rounding across bins.

---

## Seeding & Traceability

### Consistent seeding

`seed` is a first-class CLI parameter in all training scripts (default `42`). It controls:

| Scope | Call |
|---|---|
| Python stdlib | `random.seed(seed)` |
| NumPy global | `np.random.seed(seed)` |
| PyTorch CPU | `torch.manual_seed(seed)` |
| PyTorch GPU | `torch.cuda.manual_seed_all(seed)` |
| Data sampling | `mouse_data(..., seed=seed)` → `np.random.default_rng(seed)` |

The seed block runs at the top of each training function, before any model init or data loading. `_quantile_sample` is fully deterministic and needs no seed. `dataset.seed` stores the value used.

### Config record

After construction, `dataset.sampling_config` holds the full record:

```python
{
    'filter_mask':       bool,
    'lo':                int | None,
    'hi':                int | None,
    'sampling_strategy': None | "mask_duration" | "subsample",
    'samples_per_mask':  int | None,
    'total_samples':     int | None,
    'duration_aware':    bool,
    'seed':              int,
    'n_after_filter':    int,   # count after Stage 1
    'n_final':           int,   # count after Stage 2
}
```

Training scripts (`bartul_mouse.py`, `bartul_mouse_cond.py`, `compare_qmc_vae_mouse.py`) automatically write `sampling_config.json` to the run's save directory after dataset creation.

---

## Quick Reference

| Use case | Parameters |
|---|---|
| All data, no filtering | (defaults) |
| Filter only | `filter_mask=True, lo=1, hi=8` |
| Per-bin duration-aware quota | `filter_mask=True, sampling_strategy='mask_duration', samples_per_mask=N` |
| Proportional subsample, random | `filter_mask=True, sampling_strategy='subsample', total_samples=N` |
| Proportional subsample, duration-aware | `filter_mask=True, sampling_strategy='subsample', total_samples=N, duration_aware=True` |

---

## Job Submission (Slurm)

Training uses `bartul_mouse.py` via `fire`. Submit with `sbatch`.

### Template script

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=fail
#SBATCH --mail-user=tt1131@princeton.edu
#SBATCH --output=/usr/people/tt1131/projects/MMMmB/qmc_deep_gen/scripts/out/slurm-%A_%a.out

module purge
module load anacondapy/2023.07
conda activate /usr/people/tt1131/.conda/envs/samv2_env

train_n_samples=30000
latent_dim=2

python /usr/people/tt1131/projects/MMMmB/qmc_deep_gen/bartul_mouse.py \
    --save_location /usr/people/tt1131/projects/MMMmB/qmc_deep_gen/results/mouse_${train_n_samples}_${latent_dim}D \
    --dataloc /jukebox/falkner/Dexter/vocal_beh/data/full_dataset/preprocessed_500k_110_centered_512/ \
    --nEpochs 300 \
    --samples_per_mask ${train_n_samples} \
    --train_batch_size 512 \
    --test_batch_size 1 \
    --print_gpu_mem False \
    --latent_dim ${latent_dim} \
    --lattice_type fib \
    --train_grid_m 15 \
    --test_grid_m 20 \
    --val_freq 10 \
    --test_samples_per_mask 100 \
    --n_dur_bins 8 \
    --seed 42
```

### Key parameters

| Flag | Maps to | Notes |
|---|---|---|
| `--samples_per_mask` | `mouse_data(sampling_strategy='mask_duration', samples_per_mask=N)` | Max samples per `masks_len` bin in training set |
| `--latent_dim` | Model latent dimensionality | `2` for 2D torus latent space |
| `--lattice_type` | `fib` / `korobov` / `roberts` | QMC lattice type |
| `--train_grid_m` | Lattice size for training | Approx Fibonacci number index |
| `--test_grid_m` | Lattice size for evaluation | Use larger than train |
| `--val_freq` | Validation every N epochs | |
| `--test_samples_per_mask` | Samples per bin in test set | |
| `--n_dur_bins` | Duration bins for analysis plots | |
| `--seed` | Global random seed | Controls model init, data sampling, shuffling |

### Submit

```bash
sbatch scripts/training.sh
```

Output logs go to `scripts/out/slurm-<jobid>.out`.

To adjust training size, change `train_n_samples` at the top of the script — the save directory name updates automatically.
