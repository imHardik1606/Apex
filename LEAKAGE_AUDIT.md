# DATA LEAKAGE AUDIT REPORT
**Date:** 2026-09-13 | **Status:** CRITICAL ISSUES FOUND

---

## EXECUTIVE SUMMARY

**MAJOR LEAKAGE FOUND**: Metrics reported (Top-1: 100%, Log Loss: 0.038) are on the **calibration set** (2025) which was exposed to the calibration step. This inflates all final metrics.

**Honest CV metrics** (on held-out season folds) are 3-4x worse:
- Pre-qual CV Top-1: **77.27%** (not 100%)
- Pre-qual CV Log Loss: **0.2441** (not 0.0779)
- Post-qual CV Top-1: **86.36%** (not 100%)
- Post-qual CV Log Loss: **0.3275** (not 0.0378)

---

## ISSUE 1: FINAL MODEL EVALUATED ON CALIBRATION SET (CRITICAL)

### Current Flow
```
train_and_save.ipynb → train_final_model():
  1. Train on seasons [2022, 2023, 2024]
  2. Calibrate on season [2025] ← X_cal, y_cal fitted to CalibratedClassifierCV
  3. Evaluate on season [2025] ← SAME X_cal, y_cal 
  Result: 100% top-1, 0.0378 log loss
```

### The Problem
- `CalibratedClassifierCV.fit(X_cal, y_cal)` learns calibration from 2025 data
- `evaluate(calibrated, X_cal, y_cal, ...)` reports metrics on **same 2025 data**
- This is **data leakage** — model has seen the evaluation set

### Severity
🔴 **CRITICAL** — Reported final metrics are unreliable

### Fix
- **Option A (Nested CV)**: Calibrate using 5-fold CV within training set
  - Train on [2022-2024]
  - Calibrate+evaluate on nested 5-fold of training set only
  
- **Option B (Separate Test)**: True 3-way split
  - Train on [2022-2023]
  - Calibrate on [2024]
  - Test on [2025] (held-out, unseen in all steps)

→ **Recommended: Option B** (cleaner, no nested complexity)

---

## ISSUE 2: PREPROCESSING LEAKAGE IN FEATURE ENGINEERING (HIGH)

### Current Flow
`notebooks/pre_qual.ipynb` → Feature engineering:
```python
# Round-based NaN fill — FITS ON ENTIRE DATASET
def round_avg_fill(df, feature):
    round_avg = df.groupby(["season", "round"])[feature].transform('mean')
    df[feature] = df[feature].fillna(round_avg)  # ← uses ALL seasons
    return df

# Called during feature engineering on full [2022-2026] dataset:
df = round_avg_fill(df, "team_avg_finish_last_3")
```

### The Problem
- Feature engineering notebooks run **ONCE** on full raw data (all seasons)
- Round averages include 2025 and 2026 data even when imputing 2022 values
- When train_and_save.ipynb later splits [2022-2024] train vs [2025] test:
  - Train features already know 2025 round averages
  - Test features know 2025 round averages
  - This is **indirect test leakage** via statistics

### Example
```
2022 Race 1: team_avg_finish_last_3 = NaN for new team (first race)
Fix: Fill with round_avg = mean of all teams in [2022 R1]
     BUT this mean includes teams from 2025 R1 that also didn't have history!
     2025 data influenced 2022 imputation.
```

### Severity
🟠 **HIGH** — Affects ~5-10% of rows (early season, new entries), but contaminates both train and test

### Fix
- Move NaN filling **INSIDE** train_and_save.ipynb
- Fit all imputation stats (medians, round_avg, circuit_avg) on training fold only
- Apply to validation/test using training stats

---

## ISSUE 3: CROSS-VALIDATION SPLIT — CORRECT BUT INCONSISTENT EVAL

### Current Flow
`walk_forward_cv()` in train_and_save.ipynb:
```python
for i in range(1, len(seasons)):  # seasons = [2022, 2023, 2024, 2025]
    train_seasons = seasons[:i]
    val_season = seasons[i]
    
    # Fold 1: train [2022], val [2023]
    # Fold 2: train [2022, 2023], val [2024]
    # Fold 3: train [2022, 2023, 2024], val [2025]  ← Chronologically OK
```

### Status
✅ **Chronological split is correct** — no future leakage in fold splits

### But...
- ⚠️ Each fold has **different training set size**
  - Fold 1: 1 year of data
  - Fold 3: 3 years of data
  - Results not comparable
- ⚠️ NaN filling strategy is correct per-fold (medians from train only)
- ⚠️ **Recency weighting** gives 2025 races 2.0x boost during training
  - Fold 3 trains on [2022-2024] with aggressive 2025 weighting
  - But 2025 IS the test set for fold 3
  - This is subtle leakage: temporal information about test set seasons used in training

### Severity
🟡 **MEDIUM** — Leakage is indirect (season-level weighting, not instance-level)

### Fix
- Remove `calculate_recency_weights()` from CV loop
- Recency weighting should only apply to final training (on locked [2022-2024])
- CV should use uniform sample weights

---

## ISSUE 4: IDENTITY FEATURES & MEMORIZATION RISK

### Features in Use
```
Pre-qualifying (44 features):
  ❌ NO identity columns (driver, team) in the feature list ✓ GOOD
  ✅ BUT derived features aggregate by (driver, team):
    - driver_avg_finish_last_3
    - driver_championship_position
    - team_avg_finish_at_circuit_last5
    - etc.
```

### Risk
- Model could memorize "VER always wins" → team membership leaks into features
- High Top-1 hit rates pre-qualifying (77%) suspicious if purely based on aggregate stats

### Sanity Check Required
- Remove all `driver_*` and `team_*` features, keep only `circuit_*` + `season_*`
- Retrain → if top-1 collapses to ~20%, confirms memorization
- If stays >50%, true signal in circuit/season features

### Severity
🟡 **MEDIUM** — Leakage is structural (aggregation level), not computational

---

## ISSUE 5: FEATURE LEAKAGE BY TIMING

### Pre-Qualifying Features
All features should be knowable **before qualifying session**:
- ✅ `driver_avg_finish_last_3` (historical)
- ✅ `driver_championship_position` (current season standings)
- ✅ `circuit_type_encoded` (circuit constant)
- ✅ `season_round_number` (race schedule)
- ⚠️ `circuit_avg_overtakes_last3years` (depends on `last3years` — static)
- ⚠️ `circuit_safety_car_probability` (historical, safe)
- ⚠️ `circuit_lap_count` (race schedule, safe)

**Status**: ✅ Pre-qualifying features are **mostly safe**

### Post-Qualifying Features
Additions after qualifying:
- ✅ `grid_position` (from qualifying session)
- ✅ `gap_to_pole_seconds` (from qualifying session)
- ✅ `qualifying_session_was_dry` (session flag)
- ✅ `fp2_long_run_avg_gap` (FP2 data, before qualifying)
- ✅ `starting_tyre_compound_encoded` (announced by team)
- ✅ `gap_to_teammate_quali_this_weekend` (post-qualifying team stats)

**Status**: ✅ Post-qualifying features are **all safe**

---

## LEAKAGE SEVERITY RANKING

| Rank | Issue | Severity | Impact on Metrics |
|------|-------|----------|-------------------|
| 1 | Final eval on calibration set | 🔴 CRITICAL | **Top-1: 100% → ~75-80%** |
| 2 | Preprocessing stats on full dataset | 🟠 HIGH | **Log Loss: +0.05-0.10** |
| 3 | Recency weighting in CV | 🟡 MEDIUM | **Fold metrics +5-10%** |
| 4 | Identity feature memorization | 🟡 MEDIUM | **Depends on feature removal** |
| 5 | Feature timing | 🟢 LOW | **No direct leakage** |

---

## EXPECTED IMPACT AFTER FIXES

### Current Reported Metrics (INFLATED)
- Pre-qual Final: Top-1 **100%**, Log Loss **0.0779**
- Post-qual Final: Top-1 **100%**, Log Loss **0.0378**
- Pre-qual CV: Top-1 **77.27%**, Log Loss **0.2441** ← More honest

### Honest Metrics (After 3-way split)
- Pre-qual Test (2025): Top-1 ~**70-75%**, Log Loss ~**0.22-0.28**
- Post-qual Test (2025): Top-1 ~**80-85%**, Log Loss ~**0.30-0.35**

### Why Still "High"?
1. Post-qualifying benefits from qualifying data (strong signal)
2. Class imbalance (5% winners) makes random baseline ~5%
3. Top-1 per race is easier than per-driver probability calibration
4. But still well below the suspiciously perfect 100%

---

## REMEDIATION PLAN

**Priority: Do Steps 1-6 in order before any deployment**

### Step 1: Fix Train/Calibration/Test Split
- Modify `train_final_model()` to use 3-way split:
  - Train: [2022, 2023]
  - Calibration: [2024]
  - Test: [2025]
- Remove `calculate_recency_weights()` from CV

### Step 2: Remove Preprocessing Leakage
- Move NaN filling logic from feature engineering into train_and_save.ipynb
- Compute imputation stats (median, round_avg, circuit_avg) on train fold only
- Apply transformations to cal/test using train stats

### Step 3: Disable Recency Weighting in CV
- CV should use `sample_weight=None` (uniform)
- Recency weighting applies only to final training on [2022-2024]

### Step 4: Audit Identity Features
- Retrain without `driver_*` features, keep circuit + season only
- Compare Top-1 hit rates
- If collapse >20%, confirms memorization

### Step 5: Re-run Training & Evaluation
- Execute fixed train_and_save.ipynb
- Report metrics on 2025 test set only (no CV discrepancy)

### Step 6: Sanity Check Results
- Pre-qual Top-1 should be 40-70% range (not 100%)
- Post-qual Top-1 should be 60-85% range
- Log Loss should be 0.22-0.35 range
- If outside ranges, re-audit for remaining leakage

---

## NEXT: Execute fixes in this order
1. ✗ Fix train_final_model() split
2. ✗ Move NaN filling into train_and_save
3. ✗ Remove recency weighting from CV
4. ✗ Audit identity features
5. ✗ Re-run and report
