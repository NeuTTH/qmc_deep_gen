# Understanding Figure E: Embedded Latents

## What are the "dots" in Figure E?

**Each dot = one individual vocalization from your test dataset**

- If you have 10,000 mouse vocalizations in your test set → 10,000 dots
- The fine detail comes from the density of your dataset, not the lattice resolution
- More data = more dots = finer visual detail

## How are samples mapped to the latent space?

### Step-by-step process:

1. **Generate a lattice** (e.g., 10,000 points) covering [0,1]²
   - This is your "grid" for inference
   - Lattice points are fixed, quasi-random positions

2. **For each vocalization**, compute posterior probability at every lattice point:
   ```
   P(z_i | x) for i = 1, 2, ..., 10,000 lattice points
   ```

3. **Find the MAP (Maximum A Posteriori) estimate**:
   ```
   z_MAP = argmax P(z_i | x)
   ```
   - This gives you the single lattice point with highest posterior
   - This becomes the dot's (x, y) coordinate

4. **Compute a summary statistic** (mean frequency or mask count) from the spectrogram

5. **Plot**: (z_MAP[0], z_MAP[1]), colored by the summary statistic

## How is mean frequency computed?

Mean frequency is the **weighted average frequency** (center of mass):

```python
# Spectrogram shape: (frequency_bins, time_steps)
# Example: (128 freq bins, 100 time steps) for a syllable

freq_bins = np.linspace(20, 120, 128)  # 20-120 kHz (mouse USV range)

# Sum across time to get frequency energy profile
freq_profile = spectrogram.sum(axis=1)  # (128,)

# Weighted average
mean_freq = (freq_profile * freq_bins).sum() / freq_profile.sum()
```

### Example:
- Low-pitched vocalization centered at 30 kHz → **mean_freq = 30**
- High-pitched vocalization centered at 80 kHz → **mean_freq = 80**
- Broadband vocalization (20-100 kHz) → **mean_freq ≈ 60**

## How is mask count computed?

**Mask count = syllable duration** (length in time bins)

```python
# Method 1: If mask is provided in dataset
mask_count = mask.sum()  # number of active time bins

# Method 2: Infer from spectrogram
mask_count = (spectrogram.sum(axis=0) > 0).sum()  # non-zero time bins
```

- Short syllable (20ms) → small mask_count
- Long syllable (100ms) → large mask_count

## Why color by these variables?

### Mean frequency
- Shows if the model organizes vocalizations by **pitch**
- Low frequencies cluster together, high frequencies cluster together
- Reveals spectral organization in latent space

### Mask count (syllable length)
- Shows if the model organizes by **temporal structure**
- Short syllables vs. long syllables
- Reveals temporal organization in latent space

## Visual Interpretation

Looking at the paper's Figure E:
- **Purple regions** (low values) = low-frequency vocalizations
- **Green/cyan regions** (mid values) = mid-frequency vocalizations
- **Yellow regions** (high values) = high-frequency vocalizations

The smooth color gradients show that **nearby points in latent space have similar acoustic properties**, which is exactly what you want from a good generative model!

## Key Differences from Typical VAEs

In a VAE:
- You sample from a continuous Gaussian distribution
- Each sample can be anywhere in latent space

In QMC-LVM:
- Samples can only map to **discrete lattice points**
- But with 10,000+ lattice points, it appears continuous
- Multiple samples can map to the **same lattice point** (this is good - similar sounds cluster)

## Adjusting the visualization

### If dots are too small/invisible:
```bash
--scatter_size=5  # Increase point size
```

### If there's too much overlap:
```bash
--scatter_alpha=0.1  # Make more transparent (0.1-0.5)
```

### If you want actual frequency values (not bin indices):
```bash
--freq_range_khz="(20,120)"  # Mouse USV range in kHz
```

## Example usage with proper parameters

```bash
python analyze_mouse_latents.py \
    --model_path="./save/qmc_train_mouse_experiment.tar" \
    --dataloc="./data/mouse_vocalizations/" \
    --save_dir="./analysis_output" \
    --lattice_m=20 \
    --freq_range_khz="(20,120)" \
    --scatter_size=3 \
    --scatter_alpha=0.3 \
    --bandwidth=0.1
```

This will create:
- `figure_E_embedded_latents_by_freq.png` - colored by mean frequency (kHz)
- `figure_E_embedded_latents_by_length.png` - colored by syllable length
- `figure_F_aggregated_posterior.png` - heatmap with cluster centroids
- `figure_G_cluster_examples.png` - example spectrograms per cluster
