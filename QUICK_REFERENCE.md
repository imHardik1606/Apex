# QUICK REFERENCE GUIDE

## File Organization Summary

### ✅ What Changed
| Old File | New File | Reason |
|----------|----------|--------|
| `run_leakage_fixed_pipeline.py` | `train_models.py` | Clearer purpose |
| `show_metrics.py` | `display_metrics.py` | Python convention |
| `models/train_and_save_executed.ipynb` | ❌ Deleted | Duplicate backup |
| ❌ None | `README.md` | Comprehensive docs |
| ❌ None | `CLEANUP_NOTES.md` | Cleanup summary |
| ❌ None | `PROJECT_COMPLETION.md` | Final status |

---

## How To Use Each File

### 🎯 START HERE
- **`README.md`** — Everything you need to know (10 min read)

### 🚀 Run Training
```bash
python train_models.py
```
Trains models with:
- Train [2022-2023]
- Calibrate [2024]
- Test [2025]

### 📊 Display Metrics
```bash
python display_metrics.py
```
Shows:
- Pre-qualifying: 29.17% top-1
- Post-qualifying: 41.67% top-1
- Cross-validation results
- Split confirmation

### 🌐 Start API
```bash
python app.py
```
Launches Flask server on `http://localhost:5000`

### 📓 Explore Interactively
```bash
jupyter notebook models/train_and_save.ipynb
```
Interactive training pipeline (same as `train_models.py`)

### 📚 Read Documentation
| Document | Read For |
|----------|----------|
| `README.md` | Full project guide |
| `LEAKAGE_FIX_SUMMARY.md` | Why metrics changed |
| `LEAKAGE_AUDIT.md` | What was wrong |
| `LEAKAGE_REMEDIATION_REPORT.md` | How we fixed it |
| `LEAKAGE_FIX_COMPLETE.md` | Final details |
| `PROJECT_COMPLETION.md` | Overall status |
| `CLEANUP_NOTES.md` | What was cleaned |

---

## Key Metrics (Honest & Validated)

### Pre-Qualifying
- **Top-1 Hit Rate**: 29.17%
- **Log Loss**: 0.1315
- **Random Baseline**: 4.5%
- **Better than Random**: 6.5x ✅

### Post-Qualifying
- **Top-1 Hit Rate**: 41.67%
- **Log Loss**: 0.1077
- **Random Baseline**: 4.5%
- **Better than Random**: 9x ✅

**Why Honest?**: Evaluated on held-out 2025 test data, never seen in training/calibration.

---

## 3 Critical Fixes

### 1. Evaluation on Calibration Set
```
BEFORE: Train [2022-2024] → Cal+Test [2025] → 100% accuracy ❌
AFTER:  Train [2022-2023] → Cal [2024] → Test [2025] → 29-42% ✅
```

### 2. Preprocessing Leakage
```
BEFORE: NaN stats computed on full dataset (includes future) ❌
AFTER:  NaN stats computed on training fold only ✅
```

### 3. Temporal CV Leakage
```
BEFORE: CV recency weighting gave 2x boost to test years ❌
AFTER:  CV uses uniform weights (no temporal info) ✅
```

---

## Project Structure (Visual)

```
📁 F1-Race-Winner-Prediction
├── 📄 README.md ..................... Comprehensive guide
├── 📄 train_models.py ............... Train ML models
├── 📄 display_metrics.py ............ Show metrics
├── 📄 app.py ........................ Flask API
│
├── 📁 models/
│   ├── train_and_save.ipynb ........ Main training code
│   └── 📁 saved_models/
│       └── *.pkl, feature_columns.json
│
├── 📁 notebooks/
│   ├── pre_qual.ipynb .............. Analysis
│   └── post_qual.ipynb ............. Analysis
│
├── 📁 data/
│   ├── raw/
│   └── processed/ .................. Datasets
│
├── 📁 Documentation/
│   ├── LEAKAGE_FIX_SUMMARY.md ....... Executive summary
│   ├── LEAKAGE_AUDIT.md ............ Audit findings
│   ├── LEAKAGE_REMEDIATION_REPORT.md Technical analysis
│   ├── LEAKAGE_FIX_COMPLETE.md ..... Final status
│   ├── PROJECT_COMPLETION.md ....... Overall summary
│   └── CLEANUP_NOTES.md ............ Cleanup details
```

---

## Common Tasks

### Task: Retrain Everything
```bash
python train_models.py
# Output: Updated models in models/saved_models/
# Time: ~5 minutes
```

### Task: Check Model Performance
```bash
python display_metrics.py
# Output: Metrics table + cross-validation results
# Time: <1 second
```

### Task: Make a Prediction
```bash
curl -X POST http://localhost:5000/post_qual_predict \
  -H "Content-Type: application/json" \
  -d '{"season": 2026, "round": 1, "circuit_id": "bahrain"}'
```

### Task: Update Documentation
1. Edit `README.md` (general)
2. Edit respective `LEAKAGE_*.md` (if about leakage)
3. Run `git add` and `git commit` with clear message

### Task: Add New Feature
1. Edit `train_models.py` to add feature to `pre_qual_features` or `post_qual_features`
2. Make sure feature is in dataset CSV
3. Run `python train_models.py`
4. Commit changes

---

## Important Files to Never Delete

⚠️ **CRITICAL**:
- `models/train_and_save.ipynb` — Main training code
- `models/saved_models/` — All saved models
- `data/processed/` — Input datasets
- `README.md` — Project documentation
- `LEAKAGE_*.md` — Audit trail (keep for reference)

---

## Git Commit History

```
6e6b8cc - Add: Project completion summary
818022c - Refactor: Organize project structure
fe79ebe - Add: Leakage fix completion summary
e3b98cf - Fix: Critical data leakage in training pipeline
```

All commits are self-documenting with detailed messages.

---

## Status Dashboard

```
✅ Leakage Fixed           ✅ Files Organized
✅ Models Retrained        ✅ Documentation Complete
✅ Metrics Honest          ✅ Ready for Deployment
✅ Cross-Validation OK     ✅ Git History Clean
✅ API Ready               ✅ Code Quality High
```

---

## Next Steps (When Ready)

1. **Code Review**: Have team review the 4 commits
2. **Stakeholder Briefing**: Share LEAKAGE_FIX_SUMMARY.md
3. **Deployment**: Deploy `app.py` to production
4. **Frontend Integration**: Connect Next.js frontend to API
5. **Monitoring**: Watch model performance on 2025 data
6. **Quarterly Retraining**: Add new season data, retrain

---

**Last Updated**: 2026-09-13  
**Status**: ✅ PRODUCTION READY
