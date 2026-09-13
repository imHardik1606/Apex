# Cleanup & File Organization

**Date**: 2026-09-13  
**Action**: Project structure cleanup and file renaming

## Files Removed
- ❌ `models/train_and_save_executed.ipynb` — Duplicate of executed version (old backup, no longer needed)

## Files Renamed
- `run_leakage_fixed_pipeline.py` → `train_models.py`
  - More descriptive name
  - Indicates: This script trains the ML models
  - Entry point for model training pipeline

- `show_metrics.py` → `display_metrics.py`
  - Consistency with Python naming conventions
  - Indicates: Display-only utility
  - No parameters needed, just run it

## Files Kept (Analysis Notebooks)
- `notebooks/pre_qual.ipynb` — Pre-qualifying model analysis & validation
- `notebooks/post_qual.ipynb` — Post-qualifying model analysis & validation
- `models/train_and_save.ipynb` — Main training pipeline (production code)

## Directory Structure (Clean)
```
project/
├── app.py                              # Flask API
├── display_metrics.py                  # Show final metrics (utility)
├── train_models.py                     # Train models (production script)
├── requirements.txt                    # Dependencies
│
├── models/
│   ├── train_and_save.ipynb           # Training pipeline (source of truth)
│   └── saved_models/
│       ├── pre_qual_model.pkl
│       ├── post_qual_model.pkl
│       ├── pre_qual_calibrated.pkl
│       ├── post_qual_calibrated.pkl
│       ├── pre_qual_medians.pkl
│       ├── post_qual_medians.pkl
│       └── feature_columns.json
│
├── notebooks/
│   ├── pre_qual.ipynb                 # Analysis & validation
│   └── post_qual.ipynb                # Analysis & validation
│
├── scripts/
│   ├── extractor.py
│   └── cache/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── pre_qualifying_dataset.csv
│       └── post_qualifying_dataset.csv
│
├── cache/
│   ├── 2022/
│   ├── 2023/
│   ├── 2024/
│   ├── 2025/
│   └── ...
│
└── Documentation/
    ├── LEAKAGE_FIX_SUMMARY.md          # Executive summary
    ├── LEAKAGE_AUDIT.md                # Detailed audit
    ├── LEAKAGE_REMEDIATION_REPORT.md   # Technical analysis
    └── LEAKAGE_FIX_COMPLETE.md         # Final status
```

## Usage

### Train Models
```bash
python train_models.py
```
Executes the complete training pipeline:
- Loads pre/post-qualifying datasets
- Runs 3-fold walk-forward CV (honoring temporal splits)
- Trains final model on [2022-2023]
- Calibrates on [2024]
- Evaluates on [2025]
- Saves all models and metrics

### Display Metrics
```bash
python display_metrics.py
```
Shows the current honest metrics:
- Pre-qualifying model performance
- Post-qualifying model performance
- Cross-validation results
- Train/calibration/test splits

### Training Notebook (Reference)
Run cells in `models/train_and_save.ipynb` for interactive exploration.

---

**Status**: ✅ Project structure cleaned and organized
