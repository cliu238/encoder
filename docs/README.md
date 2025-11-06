# Documentation Directory

## Active Documentation

### [`ANALYSIS_SUMMARY.md`](ANALYSIS_SUMMARY.md)
**Purpose**: Comprehensive diagnosis of the overfitting problem

**Contents**:
- Evidence that the issue is overfitting, not insufficient data
- Train vs test domain accuracy gap analysis (51% → 90%)
- High-regularization solution and results
- Tools created for diagnosis (monitoring, comparison)
- Recommendations for further improvement

**When to reference**:
- Understanding why the model overfits
- Explaining the diagnosis to others
- Justifying architectural decisions

---

## Archived Documentation

The `archive/` subdirectory contains legacy documentation from early development:

### [`archive/IMPLEMENTATION_SUMMARY.md`](archive/IMPLEMENTATION_SUMMARY.md)
- Early implementation completion report
- Initial test results (20 epochs)
- Superseded by current README.md

### [`archive/QUICKSTART.md`](archive/QUICKSTART.md)
- Initial quick start guide
- Now redundant with main README.md usage section

### [`archive/VERIFICATION_RESULTS.md`](archive/VERIFICATION_RESULTS.md)
- Component test results from initial development
- Unit test verification
- No longer needed as code is tested and stable

**Why archived**: These documents were useful during initial development but are now superseded by the main README.md and more comprehensive documentation.

---

## Documentation Hierarchy

```
Root Level (Essential, frequently referenced)
├── README.md                    # Start here - main documentation
├── CLAUDE.md                    # Development guidelines
├── IMPROVEMENTS_GUIDE.md        # Experimental protocols (MMD, augmentation, etc.)
└── NEXT_STEPS_SUMMARY.md        # Quick reference for next experiments

docs/ (Reference material)
├── ANALYSIS_SUMMARY.md          # Overfitting diagnosis (important reference)
└── README.md                    # This file

docs/archive/ (Historical)
├── IMPLEMENTATION_SUMMARY.md    # Early completion report
├── QUICKSTART.md                # Legacy quick start
└── VERIFICATION_RESULTS.md      # Initial verification tests
```

---

## When to Update

### Add to `docs/`
- Detailed analysis or research findings
- Technical deep-dives on specific topics
- Historical reference material that's still relevant

### Move to `docs/archive/`
- Documentation superseded by newer versions
- Intermediate development artifacts
- Test results from early development
- Anything no longer actively referenced

### Keep in root
- README.md (always main entry point)
- CLAUDE.md (development guidelines)
- Active guides (IMPROVEMENTS_GUIDE.md, NEXT_STEPS_SUMMARY.md)
