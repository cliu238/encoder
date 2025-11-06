# Further Improvements Guide

Three approaches have been implemented to address the overfitting problem and improve domain adaptation. This guide explains each approach, when to use it, and how to run experiments.

---

## Quick Decision Matrix

| Approach | Best For | Stability | Speed | Complexity |
|----------|----------|-----------|-------|------------|
| **Stronger Adversarial (DANN)** | When current DANN shows promise | ⚠️ Can be unstable | Fast | Low |
| **MMD Loss** | Small datasets, stability needed | ✅ Very stable | Fast | Medium |
| **Data Augmentation** | Any approach (complementary) | ✅ Stable | Same | Low |

**Recommendation**: Try in this order:
1. Start with **MMD** (most stable)
2. Add **Data Augmentation** to whichever works best
3. Try **Ultra-Strong Adversarial** if MMD plateaus

---

## Approach 1: Ultra-Strong Adversarial Training

### What It Does

Increases the adversarial loss weight from `lambda_adv=1.0` → `5.0` and uses constant alpha schedule (`const` instead of `exp`) to apply full adversarial pressure from epoch 1.

### Why It Might Help

The current model achieves 50% domain accuracy during training but 90% on test set. Stronger adversarial force may push the encoders harder to learn truly domain-invariant features.

### Configuration

**File**: `config_ultra_adversarial.json`

```json
{
  "lambda_adv": 5.0,           // 5x stronger than original
  "alpha_schedule": "const",   // Full weight from epoch 1
  "encoder_hidden": 128,       // Reduced capacity
  "embedding_dim": 32,
  "dropout": 0.5,
  "weight_decay": 0.001
}
```

### How to Run

```bash
python train.py --config config_ultra_adversarial.json
```

### How to Monitor

```bash
# Watch training in real-time
python monitor_training.py --watch results/ultra_adversarial_v1.json --interval 5
```

**What to look for:**
- ✅ Domain accuracy stays near 50% during training AND on test set
- ✅ Train-val gap <15%
- ⚠️ Watch for training instability (loss spikes)

### Pros & Cons

**Pros:**
- Simple - just changes hyperparameters
- Fast - no new code or architecture
- Can be effective if properly tuned

**Cons:**
- May be unstable (adversarial training can collapse)
- Might hurt classification accuracy
- Requires careful tuning of `lambda_adv` weight

### If This Works

- Domain accuracy should drop to ~50-60% on test set
- Try gradually increasing to `lambda_adv=7.0` or `10.0`

### If This Fails

- Training loss becomes unstable or NaN
- Classification accuracy drops below 30%
- → Switch to **Approach 2: MMD**

---

## Approach 2: MMD Loss (Recommended)

### What It Does

Replaces adversarial training (gradient reversal) with **Maximum Mean Discrepancy (MMD)** - a kernel-based method that directly minimizes the distance between domain distributions.

### Why It Might Help

MMD is:
- **More stable** than adversarial training (no minimax game)
- **Better for small datasets** (fewer parameters to tune)
- **Theoretically grounded** (minimizes actual distribution distance)

### Architecture

Same as DANN, but instead of:
```
L = L_cls + lambda_adv * L_adversarial
```

MMD uses:
```
L = L_cls + lambda_mmd * MMD(embedding_iv5, embedding_phmrc)
```

Where MMD is computed using multi-scale Gaussian kernels.

### How to Run

```bash
python train_mmd.py
```

Or with custom config:

```bash
# Create config file
cat > config_mmd.json <<EOF
{
  "lambda_mmd": 1.0,
  "kernel_sigmas": [0.01, 0.1, 1, 10, 100],
  "encoder_hidden": 128,
  "embedding_dim": 32,
  "dropout": 0.5,
  "weight_decay": 0.001,
  "num_epochs": 150
}
EOF

python train_mmd.py --config config_mmd.json
```

### How to Monitor

```bash
# Check training progress
python monitor_training.py results/mmd_model_v1.json
```

**What to look for:**
- ✅ Steadily decreasing MMD loss (indicates domains are aligning)
- ✅ Classification accuracy improving or stable
- ✅ No training instability

### Tuning MMD

If results are suboptimal, tune `lambda_mmd`:

```bash
# Try different weights
lambda_mmd: 0.5   # Less domain adaptation
lambda_mmd: 1.0   # Balanced (default)
lambda_mmd: 2.0   # Stronger alignment
lambda_mmd: 5.0   # Very strong alignment
```

### Pros & Cons

**Pros:**
- ✅ Much more stable than DANN
- ✅ No gradient reversal tricks
- ✅ Works well on small datasets
- ✅ Interpretable loss (actual distance between distributions)

**Cons:**
- ⚠️ Slightly slower (kernel computation)
- ⚠️ Kernel bandwidth selection matters
- ⚠️ May need tuning of `lambda_mmd`

### Expected Results

Based on literature and your dataset size:
- Domain classification accuracy: 60-70% (better than 90%, not as good as 50%)
- Classification accuracy: Similar or slightly better than DANN
- Training stability: Much better than DANN

---

## Approach 3: Data Augmentation

### What It Does

Applies three augmentation strategies during training:
1. **Feature dropout** (10%): Randomly set features to -1 (missing)
2. **Gaussian noise** (σ=0.1): Add noise to simulate measurement uncertainty
3. **Binary flip** (5%): Randomly flip binary values to simulate errors

### Why It Helps

With only 3,199 samples, augmentation effectively increases training diversity and forces the model to learn robust patterns.

### How to Use

#### Option A: With DANN

```python
from src.augmented_dataset import create_augmented_dataloaders

# In your training script, replace:
dataloaders = preprocessor.create_dataloaders(...)

# With:
dataloaders = create_augmented_dataloaders(
    splits,
    batch_size=32,
    use_balanced_sampling=True,
    augment_train=True,
    dropout_prob=0.1,
    noise_std=0.1,
    flip_prob=0.05
)
```

#### Option B: With MMD

Same as above - just use in `train_mmd.py` instead of `train.py`.

### Configuration

Create `config_with_augmentation.json`:

```json
{
  "use_augmentation": true,
  "augmentation": {
    "dropout_prob": 0.1,
    "noise_std": 0.1,
    "flip_prob": 0.05
  },
  "encoder_hidden": 128,
  "embedding_dim": 32,
  "dropout": 0.5,
  "weight_decay": 0.001
}
```

### Tuning Augmentation

**If model underfits** (low training accuracy):
- Reduce `dropout_prob`: 0.1 → 0.05
- Reduce `noise_std`: 0.1 → 0.05
- Reduce `flip_prob`: 0.05 → 0.02

**If model still overfits**:
- Increase `dropout_prob`: 0.1 → 0.15
- Increase `noise_std`: 0.1 → 0.15
- Increase `flip_prob`: 0.05 → 0.10

### Pros & Cons

**Pros:**
- ✅ Complementary to any approach (DANN or MMD)
- ✅ Simple to implement
- ✅ Proven effective for small datasets
- ✅ No hyperparameter sensitivity

**Cons:**
- ⚠️ Slight training slowdown (negligible)
- ⚠️ Need to tune augmentation strength

---

## Recommended Experimental Plan

### Phase 1: Test All Approaches

Run experiments in parallel to identify the best approach:

```bash
# Experiment 1: Ultra-strong adversarial
python train.py --config config_ultra_adversarial.json &

# Experiment 2: MMD (recommended)
python train_mmd.py &

# Experiment 3: High regularization (baseline - already done)
# (You already have results for this)

# Wait for all to finish
wait
```

### Phase 2: Analyze Results

```bash
# Compare all three
python compare_results.py results/high_regularization_v1.json results/ultra_adversarial_v1.json
python compare_results.py results/high_regularization_v1.json results/mmd_model_v1.json
python compare_results.py results/ultra_adversarial_v1.json results/mmd_model_v1.json

# Check overfitting
python monitor_training.py results/ultra_adversarial_v1.json
python monitor_training.py results/mmd_model_v1.json
```

**Key metrics to compare:**

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Test domain accuracy | ~50-60% | Domain confusion achieved |
| Train-val gap | <15% | Generalization good |
| Cross-dataset NN agreement | >40% | Better than 27% baseline |
| Classification accuracy | >45% | Improvement over 40% |

### Phase 3: Add Augmentation to Winner

Once you identify the best base approach (likely MMD), add augmentation:

```bash
# If MMD won, create config_mmd_augmented.json
# Then train with augmentation
python train_mmd.py --config config_mmd_augmented.json
```

### Phase 4: Fine-Tune Winner

Based on Phase 3 results, fine-tune:

**If MMD + augmentation:**
- Tune `lambda_mmd`: Try 0.5, 1.0, 2.0, 5.0
- Tune kernel bandwidths if needed

**If Ultra-adversarial + augmentation:**
- Tune `lambda_adv`: Try 5.0, 7.0, 10.0
- Try `alpha_schedule`: 'const' vs 'linear'

---

## Implementation Checklist

### Files Created

- ✅ `config_ultra_adversarial.json` - Ultra-strong DANN config
- ✅ `src/models/mmd_model.py` - MMD model implementation
- ✅ `train_mmd.py` - MMD training script
- ✅ `src/augmented_dataset.py` - Data augmentation
- ✅ `monitor_training.py` - Overfitting detection
- ✅ `compare_results.py` - Experiment comparison

### Ready to Run

**Experiment 1: Ultra-Strong Adversarial**
```bash
python train.py --config config_ultra_adversarial.json
```

**Experiment 2: MMD**
```bash
python train_mmd.py
```

**Experiment 3: Augmented DANN**
```python
# Modify train.py to use create_augmented_dataloaders
# Then run
python train.py
```

---

## Troubleshooting

### Problem: Training is Unstable (Loss → NaN)

**Solution:**
- Reduce `lambda_adv` or `lambda_mmd`
- Reduce learning rate: 0.001 → 0.0005
- Increase dropout: 0.5 → 0.6
- Switch from DANN to MMD

### Problem: Domain Accuracy Still >80% on Test

**Solution:**
- Increase `lambda_mmd` (if using MMD): 1.0 → 5.0
- Increase `lambda_adv` (if using DANN): 5.0 → 10.0
- Try MMD if using DANN

### Problem: Classification Accuracy <30%

**Solution:**
- Reduce domain adaptation weight
- Increase model capacity: `encoder_hidden` 128 → 192
- Reduce dropout: 0.5 → 0.4
- Check data preprocessing

### Problem: Train-Val Gap Still >20%

**Solution:**
- Add or increase augmentation
- Increase weight_decay: 0.001 → 0.005
- Increase dropout: 0.5 → 0.6
- Reduce model capacity further

---

## Expected Timeline

**Phase 1** (Testing all approaches): ~6-8 hours
- Each training run: ~2 hours
- Can run in parallel

**Phase 2** (Analysis): ~30 minutes
- Compare results
- Identify winner

**Phase 3** (Add augmentation): ~2-3 hours
- Implement augmentation for winner
- Train and evaluate

**Phase 4** (Fine-tuning): ~4-6 hours
- Hyperparameter search
- Final evaluation

**Total**: ~1-2 days of experimentation

---

## Success Criteria

Your experiments will be successful if you achieve:

1. ✅ **Test domain accuracy** < 70% (currently 90%)
2. ✅ **Train-val gap** < 15% (currently 21%)
3. ✅ **Cross-dataset NN agreement** > 40% (currently 27%)
4. ✅ **Classification accuracy** ≥ 40% (currently 40%)

**Stretch goals:**
- Test domain accuracy ~50-60%
- Cross-dataset NN agreement >50%
- Classification accuracy >45%

---

## Next Steps

1. **Run Phase 1** experiments (ultra-adversarial + MMD)
2. **Analyze** using comparison tools
3. **Add augmentation** to the winner
4. **Fine-tune** and achieve target metrics

Good luck! All the tools and code are ready to go. 🚀
