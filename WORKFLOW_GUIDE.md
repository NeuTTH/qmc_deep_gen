# QMC Deep Generative Models - Complete Workflow Guide

This guide shows the complete workflow from training to analysis, starting with MNIST and progressing to more complex datasets.

## Repository Structure

```
qmc_deep_gen/
├── mnist_example/          # Self-contained MNIST example (read this first!)
│   ├── run_mnist.py       # Train QLVM, VAE, IWAE and compare
│   ├── qlvm.py            # Simplified QMC model
│   ├── vae.py             # Simplified VAE model
│   └── train.py           # Training loops
│
├── models/                 # Production models
│   ├── qmc_base.py        # QMCLVM - main QMC model
│   ├── vae_base.py        # VAE baseline
│   ├── sampling.py        # QMC sequences (Fibonacci, Korobov, Roberts)
│   ├── layers.py          # Basis functions (TorusBasis, GaussianICDF, etc.)
│   └── utils.py           # Architecture builders
│
├── train/                  # Training infrastructure
│   ├── train.py           # QMC training loop
│   ├── train_vae.py       # VAE training loop
│   ├── losses.py          # Evidence, ELBO, IWAE losses
│   └── model_saving_loading.py
│
├── data/                   # Dataset loaders
│   ├── mouse_data.py      # Mouse vocalizations (HDF5/PT)
│   ├── bird_data.py       # Bird vocalizations (HDF5)
│   ├── dynamics_data.py   # Motion capture
│   └── utils.py           # Generic loader
│
├── analysis/               # Post-training analysis
│   ├── model_helpers.py   # Posterior computation, torus operations
│   ├── clustering.py      # Mean-shift, k-means on torus
│   ├── jacobians.py       # Frobenius norm analysis
│   ├── geodesics.py       # Geodesic distances
│   └── tda.py             # Topological data analysis
│
├── plotting/               # Visualization
│   ├── visualize.py       # 2D plots (grid plots, reconstructions)
│   ├── visualize_1d.py    # 1D latent visualizations
│   └── visualize_3d.py    # 3D latent visualizations
│
└── Experiment scripts (root):
    ├── mnist.py                      # MNIST: QMC vs VAE comparison
    ├── qmc_vae_comparison.py        # General: Train QMC + VAE across datasets
    ├── compare_embeddings.py        # Compare learned representations
    ├── bartul_mouse.py              # Mouse: Unconditional model
    ├── bartul_mouse_cond.py         # Mouse: Conditional on syllable length
    ├── analyze_mouse_latents.py     # Mouse: Figure E/F/G analysis
    └── diagnose_model_structure.py  # Diagnostic tool
```

---

## Workflow 1: MNIST (Start Here!)

### Step 1: Simple MNIST Comparison (mnist_example/)

**Purpose**: Understand the basic QMC-LVM concept with a clean, self-contained implementation.

```bash
cd mnist_example
python run_mnist.py \
    --dataloc="/path/to/data" \
    --save_plots=True \
    --batch_size=128
```

**What it does**:
- Trains QLVM (latent_dim=2, m=15)
- Trains VAE (latent_dim=2)
- Trains IWAE (latent_dim=2)
- Compares test losses
- **Output**: `mnist_losses.png`

**Key insight**: QLVM uses a fixed lattice (no encoder), VAE uses amortized inference.

---

### Step 2: Production MNIST Training (root)

**Purpose**: Full-featured training with detailed analysis and comparison across VAE dimensions.

```bash
python mnist.py \
    --save_location="./save/mnist_full" \
    --dataloc="/path/to/data" \
    --train_grid_m=15 \
    --test_grid_m=20 \
    --nEpochs=300
```

**What it does**:
- Trains QMC model (2D latent, Fibonacci lattice)
- Trains VAE model (32D latent)
- Creates grid plots showing decoder coverage
- Generates reconstruction comparisons
- **Outputs**:
  - `qmc_train_stats.svg` - Training curve
  - `vae_train_stats.svg` - VAE losses (recon + KL)
  - `qmc_vae_stats_comparison.svg` - Side-by-side
  - `qmc_grid.png` - QMC decoder samples across [0,1]²
  - `qmc_vae_round_trips_sample_*.png` - Reconstruction quality

---

### Step 3: Detailed QMC vs VAE Comparison

**Purpose**: Train QMC (2D) against multiple VAE dimensions (2D, 4D, ..., 128D).

```bash
python qmc_vae_comparison.py \
    --save_location="./save/comparison" \
    --dataloc="/path/to/data" \
    --dataset="mnist" \
    --nEpochs=300 \
    --train_lattice_m=15 \
    --test_lattice_m=18 \
    --make_comparison_plots=True
```

**What it does**:
- Trains QMC model (2D)
- Trains VAEs with dimensions [2, 4, 8, 16, 32, 64, 128]
- Compares evidence vs ELBO vs reconstruction likelihood
- **Outputs**:
  - `vae_qmc_evidence_elbo_comparison_by_dim_mnist.svg` - Main comparison plot
  - `qmc_mnist_grid_2d.png` - QMC decoder grid
  - `vae_mnist_2d_grid.png` - VAE decoder grid
  - `qmc_vae_recon_comparison_*d_set.png` - Reconstruction comparisons
  - `vae_posterior_comparison_2d_{sample_num}.png` - True vs encoder posterior

**Key insight**: Shows how QMC 2D compares to VAE across increasing dimensions.

---

### Step 4: Embedding Comparison (Latent Space Quality)

**Purpose**: Compare learned representations: QMC 2D vs VAE 2D vs VAE 128D (UMAP'd).

```bash
python compare_embeddings.py \
    --model_save_loc="./save/comparison/mnist" \
    --dataset="mnist" \
    --dataloc="/path/to/data" \
    --save_location="./save/embeddings" \
    --lattice_m=15
```

**What it does**:
- Loads trained models from Step 3
- Computes MAP latent embeddings for all test samples
- For VAE 128D: applies UMAP to reduce to 2D
- Creates side-by-side scatter plots colored by digit class
- **Output**: `latent_rep_comparison_mnist.png`

**Key insight**: Visualizes how well each model organizes the latent space by category.

---

## Workflow 2: Mouse Vocalizations (Your Case!)

### Step 1: Train Unconditional Model

```bash
python bartul_mouse.py \
    --save_location="./save/mouse" \
    --dataloc="/path/to/mouse/data" \
    --train_grid_m=15 \
    --test_grid_m=20 \
    --nEpochs=300 \
    --train_batch_size=64 \
    --test_batch_size=1
```

**What it does**:
- Trains QMC model on mouse spectrograms
- Uses TorusBasis (maps [0,1]² → ℝ⁴ via cos/sin)
- Architecture: Linear → ConvTranspose (128x128 spectrograms)
- **Outputs**:
  - `qmc_train_mouse_experiment.tar` - Checkpoint
  - `qmc_train_stats.svg` - Training curve
  - `qmc_grid.png` - Grid of generated spectrograms
  - `qmc_round_trips_sample_*.png` - Reconstruction quality

**Important**: Check training curve - ensure loss is still improving! If not converged, increase `nEpochs`.

---

### Step 2: Diagnose Model Quality (NEW!)

```bash
python diagnose_model_structure.py \
    --model_path="./save/mouse/qmc_train_mouse_experiment.tar" \
    --dataloc="/path/to/mouse/data" \
    --lattice_m=20 \
    --n_samples=5000
```

**What it does**:
- Computes frequency organization metrics:
  - Latent space coverage (% of lattice used)
  - Correlation between latent position and mean frequency
  - Local consistency (do nearby points have similar frequencies?)
- Checks training curve for convergence
- **Outputs**:
  - `diagnostics.png` - Scatter, density, freq vs dim1/dim2
  - Terminal output with actionable recommendations

**Interpretation**:
- Correlation > 0.5: Good frequency organization ✓
- Correlation < 0.3: Model needs more training ⚠️
- Coverage < 20%: Model has collapsed ❌

---

### Step 3: Generate Figure E/F/G (Paper-style Analysis)

#### Option A: Original Script (Scatter + Smoothed Posterior)

```bash
python analyze_mouse_latents.py \
    --model_path="./save/mouse/qmc_train_mouse_experiment.tar" \
    --dataloc="/path/to/mouse/data" \
    --save_dir="./analysis_output" \
    --lattice_m=20 \
    --bandwidth=0.1 \
    --freq_range_khz="(20,120)" \
    --scatter_size=1 \
    --scatter_alpha=0.15 \
    --posterior_sigma=8.0
```

**Outputs**:
- `figure_E_embedded_latents_by_freq.png` - Scatter plot colored by mean frequency
- `figure_E_embedded_latents_by_length.png` - Scatter plot colored by syllable length
- `figure_F_aggregated_posterior.png` - **Smoothed heatmap** + centroids (FIXED!)
- `figure_G_cluster_examples.png` - Example spectrograms per cluster
- `cluster_info.json` - Cluster metadata

**Parameters to tune**:
- `--scatter_alpha=0.1` - More transparent (for dense plots)
- `--scatter_size=1` - Smaller points (paper-like)
- `--posterior_sigma=10.0` - Smoother aggregated posterior
- `--bandwidth=0.15` - Affects number of clusters found

#### Option B: Heatmap-style Figure E (NEW!)

If you want Figure E as a smooth heatmap (not scatter):

```bash
python analyze_mouse_latents_heatmap.py \
    --model_path="./save/mouse/qmc_train_mouse_experiment.tar" \
    --dataloc="/path/to/mouse/data" \
    --save_dir="./heatmap_output" \
    --lattice_m=20 \
    --resolution=200 \
    --sigma=5.0 \
    --freq_range_khz="(20,120)" \
    --vmin=30 \
    --vmax=90
```

**Outputs**:
- `figure_E_heatmap.png` - Smoothed density heatmap
- `figure_E_heatmap_overlay.png` - Heatmap + scatter overlay

#### Option C: Advanced Aggregated Posterior (NEW!)

For publication-quality Figure F with multiple smoothing methods:

```bash
python fixed_aggregated_posterior.py \
    --model_path="./save/mouse/qmc_train_mouse_experiment.tar" \
    --dataloc="/path/to/mouse/data" \
    --save_dir="./posterior_output" \
    --lattice_m=20 \
    --method="gaussian_kde" \
    --sigma=5.0 \
    --bandwidth=0.1
```

**Methods**:
- `gaussian_kde`: Kernel density estimation (slowest, smoothest)
- `interpolation`: Cubic interpolation (fast, smooth)
- `smoothed_histogram`: Histogram + Gaussian blur (fastest)

**Outputs**:
- `figure_F_aggregated_posterior_gaussian_kde.png`

---

### Step 4: Conditional Model (Optional)

If syllable length affects generation:

```bash
python bartul_mouse_cond.py \
    --save_location="./save/mouse_cond" \
    --dataloc="/path/to/mouse/data" \
    --train_grid_m=15 \
    --test_grid_m=20 \
    --nEpochs=300
```

**What it does**:
- Trains QMC model with syllable length as conditional input
- Decoder takes `[cos(z1), sin(z1), cos(z2), sin(z2), mask_length]` (5D input)
- Creates grid plots for each syllable length (1-8 time bins)
- **Outputs**: `qmc_cond_grid_ml{1-8}.png` - Conditional generations

---

## Workflow 3: General Dataset Workflow

### Step 1: Train QMC vs Multiple VAEs

```bash
python qmc_vae_comparison.py \
    --save_location="./save" \
    --dataloc="/path/to/data" \
    --dataset="finch" \  # or "gerbil", "celeba", "mocap", etc.
    --nEpochs=300 \
    --batch_size=256 \
    --var=0.1  # For Gaussian likelihood datasets
```

### Step 2: Compare Embeddings

```bash
python compare_embeddings.py \
    --model_save_loc="./save/finch" \
    --dataset="finch" \
    --dataloc="/path/to/data" \
    --save_location="./save/embeddings"
```

---

## Key Hyperparameters

### Lattice Size (`m` parameter)
- **Training**: `m=15` → ~10K points (fast)
- **Testing**: `m=20` → ~20K points (better coverage)
- **Large datasets**: `m=25` → ~50K points (even finer)

Formula: `n_points ≈ (m choose 2) = m*(m-1)/2`

### Basis Functions
- **TorusBasis** (most common): [0,1]² → ℝ⁴ via `[cos(2πz₁), sin(2πz₁), cos(2πz₂), sin(2πz₂)]`
- **GaussianICDFBasis**: [0,1]² → ℝ² via inverse Gaussian CDF
- **IdentityBasis**: [0,1]² → ℝ² (no transformation)
- **FourierBasis**: Multi-frequency Fourier features

### Loss Functions
- **Binary data** (MNIST, mouse/bird spectrograms): `binary_evidence`, `binary_lp`
- **Continuous data** (motion capture): `gaussian_evidence`, `gaussian_lp` with `var=0.1`

---

## Troubleshooting

### Issue: Figure E shows weak frequency organization

**Diagnosis**:
```bash
python diagnose_model_structure.py \
    --model_path="your_checkpoint.tar" \
    --dataloc="your_data_path"
```

**If correlation < 0.3**:
1. Train longer (`--nEpochs=500`)
2. Check training curve is still improving
3. Verify data preprocessing matches paper
4. Try different architecture (check `models/utils.py`)

**If correlation > 0.5**:
- Just visualization issue
- Use `--scatter_alpha=0.1 --scatter_size=1`
- Or use heatmap version

### Issue: Figure F shows discrete dots

**Fix**: Apply smoothing!
```bash
python analyze_mouse_latents.py \
    --posterior_sigma=10.0  # Increase for smoother
```

Or use the advanced script:
```bash
python fixed_aggregated_posterior.py \
    --method="gaussian_kde" \
    --sigma=5.0
```

### Issue: Training is very slow

**Solutions**:
- Reduce lattice size: `--train_grid_m=12` (fewer points)
- Increase batch size: `--batch_size=256`
- Use smaller test lattice: `--test_grid_m=18`
- Enable CUDA if available

---

## Recommended Order for Your Mouse Data

1. ✅ **Train unconditional model** (`bartul_mouse.py`) - You likely did this
2. ✅ **Check training curve** (`quick_training_check.py`) - Did it converge?
3. 🔄 **Diagnose structure** (`diagnose_model_structure.py`) - Is frequency organized?
4. 🔄 **Generate figures** (`analyze_mouse_latents.py` with `--posterior_sigma=8.0`)
5. 📊 **Compare to VAE** (optional, `qmc_vae_comparison.py --dataset=mouse`)

---

## Next Steps for Publication-Quality Figures

### For Figure E (Embedded Latents):

**If your model has good structure (corr > 0.5)**:
```bash
python analyze_mouse_latents.py \
    --scatter_size=0.5 \
    --scatter_alpha=0.08 \
    --freq_range_khz="(20,120)"
```

**If you prefer heatmap style**:
```bash
python analyze_mouse_latents_heatmap.py \
    --sigma=6.0 \
    --vmin=25 \
    --vmax=95
```

### For Figure F (Aggregated Posterior):

```bash
python fixed_aggregated_posterior.py \
    --method="gaussian_kde" \
    --sigma=4.0 \
    --resolution=250
```

### For Figure G (Cluster Examples):
Already generated by `analyze_mouse_latents.py`!

---

## Summary: Start with MNIST, Then Apply to Mouse

```bash
# 1. Understand the concept (5 minutes)
cd mnist_example && python run_mnist.py --dataloc="./data"

# 2. Full MNIST workflow (30 minutes)
python mnist.py --save_location="./save/mnist" --dataloc="./data"

# 3. Check your mouse model (2 minutes)
python diagnose_model_structure.py \
    --model_path="your_mouse_checkpoint.tar" \
    --dataloc="your_mouse_data"

# 4. Generate mouse figures (10 minutes)
python analyze_mouse_latents.py \
    --model_path="your_mouse_checkpoint.tar" \
    --dataloc="your_mouse_data" \
    --save_dir="./mouse_figs" \
    --posterior_sigma=8.0 \
    --scatter_alpha=0.1 \
    --scatter_size=1
```

That's it! 🎉
