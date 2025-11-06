# Development Guidelines

## Code Style & Structure
- **Simplicity first**: Use the simplest possible code and structure
- **File management**: Do not create unnecessary files. When replacing a file, archive or delete the legacy version
- **No duplication**: Remove legacy/deprecated code immediately

## Environment
- **Virtual environment**: Use Conda virtual environment
- **Package management**: Use `uv` exclusively for Python packages and execution
- **Dependencies**: Keep requirements.txt minimal and up-to-date

## Documentation Workflow
When making changes, update `README.md` to reflect:
1. **Project overview** - What the project does and key achievements
2. **Setup instructions** - Conda + `uv` installation steps
3. **Usage examples** - Clear command-line examples for common tasks
4. **File structure** - Tree format showing project organization

## Project-Specific Notes (Updated Nov 2025)

### Known Issues & Solutions
1. **Overfitting problem**: Original DANN architecture overfits severely
   - Train domain accuracy: 51% (good) → Test: 90% (bad)
   - **Solution implemented**: High-regularization config (-75% parameters)
   - **Status**: Train-val gap reduced from 34% → 21%

2. **Domain adaptation challenge**: Target is 50% domain accuracy on test set
   - **Solutions available**: Ultra-adversarial training, MMD loss, data augmentation
   - **See**: `IMPROVEMENTS_GUIDE.md` for experimental protocol

### Current Best Practices
- **Training**: Use `config_high_regularization.json` or `train_mmd.py` (more stable)
- **Monitoring**: Always use `monitor_training.py` to detect overfitting
- **Comparison**: Use `compare_results.py` to evaluate experiments

### File Organization
```
encoder/
├── README.md                    # Main documentation (start here)
├── CLAUDE.md                    # This file - dev guidelines
├── IMPROVEMENTS_GUIDE.md        # Experimental protocols for further improvement
├── NEXT_STEPS_SUMMARY.md        # Quick reference guide
├── docs/
│   ├── ANALYSIS_SUMMARY.md      # Overfitting diagnosis & evidence
│   └── archive/                 # Legacy documentation
├── src/                         # Source code
│   ├── models/
│   │   ├── dann.py             # DANN with gradient reversal
│   │   └── mmd_model.py        # MMD alternative (more stable)
│   ├── data_preprocessing.py
│   ├── augmented_dataset.py    # Data augmentation
│   ├── validation.py
│   └── utils.py
├── config_high_regularization.json     # Recommended config
├── config_ultra_adversarial.json       # Experimental (lambda_adv=5.0)
├── train.py                     # DANN training
├── train_mmd.py                 # MMD training
├── evaluate.py                  # Model evaluation
├── monitor_training.py          # Real-time overfitting detection
└── compare_results.py           # Experiment comparison
```

### When to Archive Files
Move to `docs/archive/` when:
- Contains intermediate/test results no longer relevant
- Documentation superseded by newer versions
- Early development verification that's now obsolete

## Quick Reference Commands

```bash
# Training (recommended approaches)
python train.py --config config_high_regularization.json  # DANN with strong regularization
python train_mmd.py                                       # MMD (more stable)

# Monitor training in real-time
python monitor_training.py --watch results/experiment.json --interval 5

# Compare experiments
python compare_results.py results/old.json results/new.json

# Evaluate model
python evaluate.py --checkpoint results/checkpoints/checkpoint_best.pt
```

## Critical Reminders
- ⚠️ **Always monitor for overfitting** - Check train vs test domain accuracy gap
- ⚠️ **Target metric**: Test domain accuracy ~50%, not 90%
- ⚠️ **Don't collect more data** - Problem is overfitting, not data insufficiency (3,199 samples is sufficient)
- ✅ **Use diagnostic tools** - `monitor_training.py` detects overfitting automatically
