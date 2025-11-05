# Shared Embedding Space for Verbal Autopsy Cross-Dataset Classification

A PyTorch implementation of Domain-Adversarial Neural Networks (DANN) for learning unified representations across heterogeneous verbal autopsy datasets (IV5 and PHMRC).

## Project Overview

This project addresses the challenge of **cross-dataset classification** in verbal autopsy data, where different questionnaires (IV5 with 353 features, PHMRC with 109 features) have **zero overlapping features** but share the same cause-of-death labels.

### Key Features

- **Domain-Adversarial Learning**: Learns dataset-agnostic embeddings via gradient reversal
- **Heterogeneous Input Handling**: Dataset-specific encoders for different feature spaces
- **Comprehensive Validation**: Embedding quality metrics + cross-dataset transfer evaluation
- **Balanced Sampling**: Handles severe class imbalance (e.g., HIV: 123 samples vs Pneumonia: 662)
- **Missing Value Encoding**: 3-state encoding (Y/N/Missing) for PHMRC's 76.5% sparsity

### Architecture

```
IV5 (353 features)  ──→  IV5Encoder ──┐
                                       ├──→  SharedEmbedding  ──┬──→  Classifier (7 classes)
PHMRC (109 features) ──→ PHMRCEncoder ─┘                        │
                                                                 └──→  DomainDiscriminator
                                                                       (with gradient reversal)
```

## Setup Instructions

### 1. Create Conda Environment

```bash
conda create -n encoder python=3.10
conda activate encoder
```

### 2. Install Dependencies with `uv`

```bash
# Install uv if not already installed
pip install uv

# Install dependencies
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy pandas scikit-learn matplotlib seaborn
uv pip install umap-learn  # Optional, for UMAP visualization
```

**Note**: If you encounter `cudf.pandas` compatibility issues, ensure you're using CPU-only PyTorch or install CUDA-compatible versions explicitly.

### 3. Prepare Data

Place your datasets in the `data/` directory:
- `data/IV5_child_8categories.csv`
- `data/PHMRC_child_8categories.csv`

Both datasets should have a `broader_category` column with labels: `diarrhea`, `hiv`, `injury`, `malaria`, `other_ch`, `other_infections`, `pneumonia`.

## Usage Examples

### Training

Train a DANN model with default hyperparameters:

```bash
python train.py
```

Train with custom configuration:

```bash
python train.py --config config/experiment1.json
```

**Example config.json:**
```json
{
  "batch_size": 64,
  "num_epochs": 150,
  "learning_rate": 0.0005,
  "embedding_dim": 128,
  "alpha_schedule": "exp",
  "lambda_cls": 1.0,
  "lambda_adv": 0.5
}
```

**Training outputs:**
- Model checkpoints: `results/checkpoints/checkpoint_best.pt`
- Training logs: `results/<timestamp>.json`
- Visualizations: `results/embeddings_tsne.png`, `results/cm_*.png`

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py --checkpoint results/checkpoints/checkpoint_best.pt
```

**Evaluation generates:**
- Comprehensive text report: `results/evaluation/evaluation_report.txt`
- Embedding visualizations: t-SNE and UMAP plots
- Confusion matrices: IV5→PHMRC and PHMRC→IV5 transfers

### Testing Components

Test individual modules:

```bash
# Test data preprocessing
python src/data_preprocessing.py

# Test DANN architecture
python src/models/dann.py

# Test validation framework
python src/validation.py

# Test utilities
python src/utils.py
```

## File Structure

```
encoder/
├── data/                                 # Datasets
│   ├── IV5_child_8categories.csv        # InterVA-5 dataset (353 features)
│   └── PHMRC_child_8categories.csv      # PHMRC dataset (109 features)
│
├── src/                                  # Source code
│   ├── data_preprocessing.py            # Data loading, encoding, balanced sampling
│   ├── validation.py                    # Embedding quality & transfer metrics
│   ├── utils.py                         # Training utilities, logging, checkpointing
│   └── models/
│       └── dann.py                      # DANN architecture with gradient reversal
│
├── train.py                             # Main training script
├── evaluate.py                          # Standalone evaluation script
│
├── results/                             # Outputs (created during training)
│   ├── checkpoints/                     # Model checkpoints
│   ├── evaluation/                      # Evaluation results
│   ├── *.json                          # Experiment logs
│   └── *.png                           # Visualizations
│
├── CLAUDE.md                            # Development guidelines
└── README.md                            # This file
```

## Validation Metrics

### Embedding Quality Metrics

1. **Silhouette Score** (range: -1 to 1, higher = better)
   - Measures cluster separation for classes
   - Target: > 0.3 for good separation

2. **5-NN Purity** (range: 0 to 1, higher = better)
   - Fraction of nearest neighbors with same label
   - Target: > 0.7 for cohesive clusters

3. **Cross-Dataset 5-NN Agreement** (range: 0 to 1, higher = better)
   - Label agreement for cross-dataset neighbors
   - Target: > 0.6 for good alignment
   - **Critical metric for shared embedding validation**

4. **Domain Classification Accuracy** (range: 0 to 1, closer to 0.5 = better)
   - Ability to distinguish IV5 vs PHMRC from embeddings
   - Target: 0.45-0.55 (random guessing = domain invariance)

### Cross-Dataset Transfer Metrics

- **IV5 → PHMRC**: Train on IV5, test on PHMRC
- **PHMRC → IV5**: Train on PHMRC, test on IV5
- Metrics: Accuracy, Macro F1, Weighted F1, Per-class F1

## Class Distribution

| Category            | IV5   | PHMRC | Notes               |
|---------------------|-------|-------|---------------------|
| diarrhea            | 103   | 256   | PHMRC has 2.5x more |
| hiv                 | 103   | 20    | IV5 has 5.2x more   |
| injury              | 69    | 416   | PHMRC has 6.0x more |
| malaria             | 204   | 116   | IV5 has 1.8x more   |
| other_ch            | 397*  | 387   | *Includes severe_malnutrition merger |
| other_infections    | 129   | 337   | PHMRC has 2.6x more |
| pneumonia           | 130   | 532   | PHMRC has 4.1x more |
| **TOTAL**           | 1,135 | 2,064 |                     |

*Note: `severe_malnutrition` (291 samples) merged into `other_ch` for label harmonization.*

## Hyperparameter Guide

### Architecture Hyperparameters

- `encoder_hidden`: 256 (hidden layer size for encoders)
- `encoder_output`: 128 (encoder output dimension)
- `embedding_dim`: 64 (shared embedding dimension)
- `dropout`: 0.3 (dropout rate for regularization)

### Training Hyperparameters

- `batch_size`: 32 (larger = more stable gradients, smaller = better generalization)
- `learning_rate`: 0.001 (use 0.0005-0.002 range)
- `num_epochs`: 100 (increase to 150 if underfitting)
- `patience`: 15 (early stopping patience)

### Domain Adaptation Hyperparameters

- `lambda_cls`: 1.0 (classification loss weight)
- `lambda_adv`: 1.0 (adversarial/domain loss weight)
  - Increase if domain accuracy is high (>0.7)
  - Decrease if classification accuracy is poor
- `alpha_schedule`: 'exp' (domain adaptation schedule)
  - 'exp': Slow start, fast ramp-up (recommended)
  - 'linear': Linear increase from 0 to 1
  - 'const': Constant weight of 1.0

## Troubleshooting

### Low Cross-Dataset Transfer Performance

- **Increase `lambda_adv`**: Forces stronger domain invariance
- **Try 'exp' alpha schedule**: Gradual domain adaptation
- **Increase `embedding_dim`**: More capacity for shared representations
- **Check domain accuracy**: Should be near 0.5; if >0.7, increase adversarial weight

### Overfitting

- **Increase `dropout`**: Try 0.4-0.5
- **Reduce model capacity**: Smaller `encoder_hidden` or `embedding_dim`
- **Early stopping**: Reduce `patience` to 10
- **Regularization**: Increase `weight_decay` (default: 0.0001)

### Class Imbalance Issues

- **Enable balanced sampling**: `use_balanced_sampling=True` (default)
- **Adjust class weights**: Modify loss function weights in `train.py`
- **Oversample minority classes**: Increase sampling rate for HIV, injury, etc.

## Technical Details

### Why DANN for Verbal Autopsy?

Traditional transfer learning fails because IV5 and PHMRC have **zero overlapping features**. DANN solves this by:

1. **Dataset-Specific Encoders**: Handle different input dimensions (353 vs 109)
2. **Shared Semantic Space**: Map both datasets to common embedding space
3. **Gradient Reversal**: Forces encoders to hide dataset-specific patterns
4. **Adversarial Training**: Minimax game ensures domain-invariant features

### Gradient Reversal Mechanism

The gradient reversal layer creates an adversarial dynamic:
- **Forward pass**: Identity function (embeddings pass through unchanged)
- **Backward pass**: Gradients are reversed (negated)

This teaches the encoder to:
- ✓ Learn features that classify diseases accurately
- ✗ Avoid features that reveal which dataset a sample came from

### Missing Value Handling

PHMRC is 76.5% sparse. We use **3-state encoding**:
- `Y` → 1 (yes)
- `N` / `.` → 0 (no / don't know)
- `NaN` / empty → -1 (missing / not asked)

The network learns that -1 is informative (e.g., "question not applicable" may correlate with age or symptoms).

## Citation

If you use this code, please cite:

```bibtex
@software{va_shared_embedding_2025,
  title={Shared Embedding Space for Verbal Autopsy Cross-Dataset Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/encoder}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- DANN architecture based on Ganin et al. (2016) "Domain-Adversarial Training of Neural Networks"
- IV5 dataset: InterVA-5 verbal autopsy questionnaire
- PHMRC dataset: Population Health Metrics Research Consortium
