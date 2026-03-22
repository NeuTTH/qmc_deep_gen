# Train vs Test Data for Figure Generation

## The Critical Discovery 🔍

**Figure 5 caption**: "Latent embeddings of MNIST, colored by digit..."

**What it DOESN'T say**: Whether this is train or test data!

This is a **huge** detail that could explain why your figures look different from the paper.

---

## Why This Matters

### Dataset Size Differences

| Dataset | Train Samples | Test Samples | Ratio |
|---------|---------------|--------------|-------|
| MNIST | 60,000 | 10,000 | **6:1** |
| Mouse USV | ~50,000-100,000 | ~10,000-20,000 | **5-10:1** |
| Gerbil | Similar | Similar | **5-10:1** |

**Impact on visualizations**:
- **6-10x more dots** in scatter plots
- **Denser coverage** of latent space
- **Smoother color transitions** from point density
- **Better visual organization** due to density

### Model Optimization

The model was **explicitly trained** to organize the *training data*:
- ✅ Strong structure on training data (by design - that's what the loss optimizes)
- ⚠️ Hopefully generalizes to test data (but may be weaker)

**Your figures** (using test data):
- Scientifically rigorous ✓
- Shows generalization ✓
- But may look less organized than training data

**Paper figures** (possibly using train data):
- More publication-ready visually
- Denser/cleaner appearance
- But doesn't test generalization

---

## Testing the Hypothesis

### Step 1: Generate Figures with TEST Data (Current)

```bash
python analyze_mouse_latents.py \
    --model_path="your_checkpoint.tar" \
    --dataloc="your_data_path" \
    --save_dir="./comparison/test_data" \
    --lattice_m=20 \
    --freq_range_khz="(20,120)" \
    --scatter_size=1 \
    --scatter_alpha=0.1 \
    --posterior_sigma=8.0 \
    --use_train_data=False
```

**Outputs**: `figure_E_embedded_latents_by_freq_TEST.png`, etc.

---

### Step 2: Generate Figures with TRAIN Data (NEW!)

```bash
python analyze_mouse_latents.py \
    --model_path="your_checkpoint.tar" \
    --dataloc="your_data_path" \
    --save_dir="./comparison/train_data" \
    --lattice_m=20 \
    --freq_range_khz="(20,120)" \
    --scatter_size=1 \
    --scatter_alpha=0.1 \
    --posterior_sigma=8.0 \
    --use_train_data=True  # ← KEY DIFFERENCE!
```

**Outputs**: `figure_E_embedded_latents_by_freq_TRAIN.png`, etc.

---

### Step 3: Compare Side-by-Side

Open both sets of figures:

**Expected differences if hypothesis is correct**:

| Aspect | Test Data (Your Current) | Train Data (Paper?) |
|--------|--------------------------|---------------------|
| **Density** | Sparse, ~10-20k points | Dense, ~50-100k points |
| **Organization** | Weak correlation | Strong regional structure |
| **Color gradients** | Scattered/noisy | Smooth transitions |
| **Coverage** | Patches of space | Full coverage |
| **Resembles paper?** | ❌ No | ✅ Yes! |

---

## Why Papers Might Use Training Data

### Scientific Considerations

**Arguments FOR using train data in figures**:
1. **Visual clarity**: Denser = easier to see patterns
2. **Publication aesthetics**: Reviewers expect clean figures
3. **Proof of concept**: Shows the model *can* learn structure
4. **Common practice**: Many papers do this (often unstated)

**Arguments AGAINST (why test data is better)**:
1. **Scientific rigor**: Tests generalization
2. **Honest representation**: Shows true performance
3. **Reproducibility**: Others testing on new data see test-like results

### What's Standard Practice?

In ML papers, it's **very common** (but often unstated) to:
- Use **test data** for quantitative metrics (accuracy, loss, etc.)
- Use **train data** for qualitative visualizations (embeddings, samples, etc.)

**Why?** Because reviewers and readers want to see:
- Numbers that show generalization (test)
- Figures that look good (train)

---

## The Smoking Gun Test

### Check the Paper's Training Set Size

If the paper mentions dataset splits:
- "We use 80,000 training samples and 20,000 test samples"
- Then check your figures:
  - Training figure should have ~4x more points than test figure
  - If paper's Figure E looks super dense, it's probably training data

### Check MNIST as Reference

MNIST is a perfect test case because we know:
- Train: 60,000 samples
- Test: 10,000 samples

Try both on MNIST:

```bash
# Test data
python analyze_mouse_latents.py \
    --dataset="mnist" \
    --use_train_data=False \
    --save_dir="./mnist_test"

# Train data
python analyze_mouse_latents.py \
    --dataset="mnist" \
    --use_train_data=True \
    --save_dir="./mnist_train"
```

Compare to paper's Figure 5A:
- If it looks like your train version → Paper used train data
- If it looks like your test version → Paper used test data

---

## Recommendations

### For Your Analysis

**Try BOTH and see which matches the paper**:

1. Generate both train and test figures
2. Compare to paper's Figure 5E
3. If train version matches → Use train data for figures
4. Document which you use in your methods!

### For Scientific Integrity

**Best practice**:
- Use **test data** for all quantitative metrics
- Use **test data** for main figures (show generalization)
- Can use **train data** for supplementary "proof of concept" figures
- **Always state which you're using** in figure captions!

**Example caption**:
> "Figure E: Latent embeddings of mouse vocalizations (test set, n=20,000), colored by mean frequency..."

OR

> "Figure E: Latent embeddings of mouse vocalizations (training set, n=80,000), colored by mean frequency. Model generalizes to test set (see Supplementary Figure S3)."

---

## Quick Diagnostic

**Run this to see the size difference**:

```python
from data.mouse_data import load_mouse_data, mouse_data

train_dict, val_dict = load_mouse_data("your_data_path")

train_ds = mouse_data(train_dict, masks_len_range=(1, 8), equal_sampling=False)
test_ds = mouse_data(val_dict, masks_len_range=(1, 8), equal_sampling=False)

print(f"Training samples: {len(train_ds)}")
print(f"Test samples: {len(test_ds)}")
print(f"Ratio: {len(train_ds) / len(test_ds):.1f}x")
```

**If ratio > 3x**: Training data figures will look VERY different!

---

## Expected Results

### If Paper Used Training Data

Running `--use_train_data=True` should give you:
- ✅ Dense, smooth scatter plots
- ✅ Clear regional structure by frequency
- ✅ Smooth aggregated posteriors
- ✅ **Matches the paper!**

### If Paper Used Test Data

Then your current results should already match, and the issue is:
- Model training (needs more epochs)
- Visualization settings (alpha/size)
- Data preprocessing differences

---

## Next Steps

1. **Run both versions** (train + test) on your mouse data
2. **Compare to paper's Figure 5E**
3. **Report back**: Which one matches?
4. **Use the matching version** for your own figures
5. **Document clearly** in your methods/captions

This is a really important observation - good scientific detective work! 🔬

---

## Example Commands

### Complete comparison workflow:

```bash
# 1. Test data (scientifically rigorous)
python analyze_mouse_latents.py \
    --model_path="checkpoint.tar" \
    --dataloc="./data" \
    --save_dir="./figs/test" \
    --freq_range_khz="(20,120)" \
    --scatter_size=1 \
    --scatter_alpha=0.1 \
    --posterior_sigma=8.0 \
    --use_train_data=False

# 2. Train data (possibly matches paper)
python analyze_mouse_latents.py \
    --model_path="checkpoint.tar" \
    --dataloc="./data" \
    --save_dir="./figs/train" \
    --freq_range_khz="(20,120)" \
    --scatter_size=1 \
    --scatter_alpha=0.1 \
    --posterior_sigma=8.0 \
    --use_train_data=True

# 3. Compare:
# - Open figs/test/figure_E_embedded_latents_by_freq_TEST.png
# - Open figs/train/figure_E_embedded_latents_by_freq_TRAIN.png
# - Compare to paper's Figure 5E
# - Which one matches?
```

Let me know what you find! 🎯
