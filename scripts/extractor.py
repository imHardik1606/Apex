"""
extractor.py
============
Script 1 of 3 — Raw Data Extraction from FastF1

Extracts all raw fields needed to engineer features for both
pre-qualifying and post-qualifying F1 race winner prediction models.

Sessions extracted per race weekend:
  R   → Race results (finish position, grid, points, status)
  Q   → Qualifying (position, gap to pole, wet flag)
  FP2 → Practice 2 (long run pace only, wet flag)

Outputs:
  data/raw/raw_race_data.csv   → one row per driver per race
  data/raw/raw_quali_data.csv  → one row per driver per qualifying session
  data/raw/raw_fp2_data.csv    → one row per driver per FP2 session

Run this script first before either feature engineering script.
"""

import fastf1
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")


# ── Setup ──────────────────────────────────────────────────────────────────────

os.makedirs("cache", exist_ok=True)
os.makedirs("data", exist_ok=True)
fastf1.Cache.enable_cache("cache")

# Include 2026 to capture current season performance (Antonelli, etc)
# Models will weight 2026 data more heavily via recency weighting in train_and_save.ipynb
SEASONS = [2022, 2023, 2024, 2025, 2026]

# Statuses that count as a finish (not DNF)
FINISHED_STATUSES = {
    "Finished", "+1 Lap", "+2 Laps", "+3 Laps", "+4 Laps",
    "+5 Laps", "+6 Laps", "+7 Laps", "+8 Laps", "+9 Laps"
}

# Street circuits → circuit_type_encoded = 1
STREET_CIRCUITS = {
    "Monaco", "Baku", "Singapore", "Jeddah", "Melbourne",
    "Las Vegas", "Miami", "Montréal", "Montreal"
}

# Historical average overtakes per race (2019-2024 averages)
CIRCUIT_OVERTAKES = {
    "Bahrain":       35, "Saudi Arabia":  22, "Australia":    18,
    "Japan":         12, "China":         28, "Miami":        20,
    "Imola":         10, "Monaco":         3, "Canada":       30,
    "Spain":         15, "Austria":       25, "Britain":      22,
    "Hungary":        8, "Belgium":       18, "Netherlands":  10,
    "Italy":         35, "Azerbaijan":    28, "Singapore":     8,
    "United States": 20, "Mexico":        15, "Brazil":       30,
    "Las Vegas":     18, "Qatar":         12, "Abu Dhabi":    14,
}

# Historical safety car probability per circuit (2018-2024)
CIRCUIT_SC_PROB = {
    "Bahrain":       0.40, "Saudi Arabia":  0.55, "Australia":    0.50,
    "Japan":         0.35, "China":         0.45, "Miami":        0.60,
    "Imola":         0.45, "Monaco":        0.75, "Canada":       0.65,
    "Spain":         0.30, "Austria":       0.40, "Britain":      0.50,
    "Hungary":       0.35, "Belgium":       0.55, "Netherlands":  0.40,
    "Italy":         0.35, "Azerbaijan":    0.70, "Singapore":    0.80,
    "United States": 0.50, "Mexico":        0.40, "Brazil":       0.65,
    "Las Vegas":     0.55, "Qatar":         0.35, "Abu Dhabi":    0.30,
}

# Average pit stop time loss in seconds per circuit
PIT_STOP_DELTA = {
    "Monaco":        24.0, "Singapore":     23.5, "Baku":         22.0,
    "Melbourne":     22.5, "Jeddah":        22.0, "Las Vegas":    21.5,
    "Miami":         22.0, "Montréal":      23.0, "Montreal":     23.0,
    "Bahrain":       21.0, "Saudi Arabia":  22.0, "Australia":    22.5,
    "Japan":         22.0, "China":         21.5, "Imola":        22.0,
    "Spain":         21.0, "Austria":       20.5, "Britain":      21.5,
    "Hungary":       22.0, "Belgium":       21.0, "Netherlands":  21.5,
    "Italy":         20.0, "Azerbaijan":    22.0, "United States":21.5,
    "Mexico":        22.0, "Brazil":        21.5, "Qatar":        21.0,
    "Abu Dhabi":     21.0,
}


# ── Helper Functions ────────────────────────────────────────────────────────────

def get_dnf_flag(status):
    """
    Returns 1 if driver did not finish, 0 if they finished.
    Treats missing status as DNF to be conservative.
    """
    if pd.isna(status):
        return 1
    return 0 if str(status).strip() in FINISHED_STATUSES else 1


def normalise_circuit(location):
    """
    Strips and normalises circuit location string
    to match lookup dictionary keys.
    """
    if pd.isna(location):
        return "Unknown"
    return str(location).strip()


CIRCUIT_OVERRIDES = {
    # Relocated due to regional conflict; treat as novel because neither Sakhir nor historical Sepang applies.
    (2026, "Bahrain Grand Prix"): "Sepang_relocated_2026",
}


def extract_fp2_long_run(year, round_no):
    """
    Extracts FP2 long run average pace per driver.

    Long run = laps that are 3rd or later in their stint.
    This simulates race conditions and is the most honest
    practice signal available — teams cannot easily fake
    long run pace without hurting their own preparation.

    Only extracted when session was dry.
    Wet FP2 long run data is meaningless for race pace prediction.

    Returns:
        drivers_data  : dict keyed by driver abbreviation
        session_was_dry : bool
    """
    try:
        fp2 = fastf1.get_session(year, round_no, "FP2")
        fp2.load(laps=True, telemetry=False, weather=True, messages=False)

        laps = fp2.laps.pick_quicklaps()
        if laps is None or laps.empty:
            return {}, True

        # Determine if session was dry
        try:
            weather = fp2.weather_data
            was_wet = bool(weather["Rainfall"].any()) if not weather.empty else False
            was_dry = not was_wet
        except Exception:
            was_dry = True

        # Tag each lap with its position within its stint
        all_laps = laps.copy().sort_values(["Driver", "Stint", "LapNumber"])
        all_laps["stint_lap"] = (
            all_laps.groupby(["Driver", "Stint"]).cumcount() + 1
        )

        # Long run laps = 3rd lap or later in a stint
        long_run_all = all_laps[all_laps["stint_lap"] >= 3].dropna(subset=["LapTime"])

        if long_run_all.empty:
            return {}, was_dry

        # Session best long run reference time for gap calculation
        session_best_long_run = long_run_all["LapTime"].dt.total_seconds().min()

        drivers_data = {}
        for driver in all_laps["Driver"].unique():
            d_long = long_run_all[long_run_all["Driver"] == driver]

            # Require at least 3 long run laps for meaningful average
            if len(d_long) < 3:
                continue

            long_run_avg = d_long["LapTime"].dt.total_seconds().mean()
            long_run_gap = long_run_avg - session_best_long_run

            drivers_data[driver] = {
                "fp2_long_run_avg_gap": long_run_gap,
                "fp2_session_was_dry":  int(was_dry),
            }

        return drivers_data, was_dry

    except Exception:
        # FP2 frequently missing or corrupted — return empty silently
        # Feature engineering handles NaN with circuit median fallback
        return {}, True


# ── Main Extraction Loop ────────────────────────────────────────────────────────

race_rows  = []
quali_rows = []
fp2_rows   = []

for year in SEASONS:
    print(f"\n{'=' * 55}")
    print(f"  Processing season {year}")
    print(f"{'=' * 55}")

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"  ✗ Could not load schedule for {year}: {e}")
        continue

    for _, event in tqdm(schedule.iterrows(),
                         total=len(schedule),
                         desc=f"  {year} rounds"):

        round_no   = int(event["RoundNumber"])
        location   = event.get("Location", event.get("EventName", "Unknown"))
        circuit_id = CIRCUIT_OVERRIDES.get(
            (year, event.get("EventName")),
            normalise_circuit(location),
        )

        # ── Race Session ────────────────────────────────────────────────────

        try:
            race = fastf1.get_session(year, round_no, "R")
            race.load(laps=False, telemetry=False,
                      weather=False, messages=False)
            results = race.results

            if results is None or results.empty:
                continue

            # Total laps in race for circuit_lap_count feature
            try:
                total_laps = int(race.total_laps)
            except Exception:
                total_laps = np.nan

            for _, row in results.iterrows():
                driver = str(row.get("Abbreviation", "UNK"))
                team   = str(row.get("TeamName",     "Unknown"))
                status = str(row.get("Status",       ""))

                try:
                    position = int(float(row.get("Position",    np.nan)))
                except (ValueError, TypeError):
                    position = np.nan

                try:
                    grid = int(float(row.get("GridPosition", np.nan)))
                except (ValueError, TypeError):
                    grid = np.nan

                try:
                    points = float(row.get("Points", 0.0))
                except (ValueError, TypeError):
                    points = 0.0

                race_rows.append({
                    # ── Identity ───────────────────────────────────────────
                    "season":     year,
                    "round":      round_no,
                    "circuit_id": circuit_id,
                    "driver":     driver,
                    "team":       team,
                    # ── Raw race fields ────────────────────────────────────
                    "finish_position": position,
                    "grid_position":   grid,
                    "points":          points,
                    "status":          status,
                    "dnf_flag":        get_dnf_flag(status),
                    "winner":          1 if position == 1 else 0,
                    # ── Circuit context (same for all drivers this race) ───
                    "circuit_type_encoded":
                        1 if circuit_id in STREET_CIRCUITS else 0,
                    "circuit_avg_overtakes_last3years":
                        CIRCUIT_OVERTAKES.get(circuit_id, 18),
                    "circuit_safety_car_probability":
                        CIRCUIT_SC_PROB.get(circuit_id, 0.45),
                    "circuit_lap_count":
                        total_laps,
                    "pit_stop_avg_time_loss_at_circuit":
                        PIT_STOP_DELTA.get(circuit_id, 21.5),
                })

        except Exception as e:
            print(f"\n    ✗ Race failed  {year} R{round_no}: {e}")
            continue

        # ── Qualifying Session ──────────────────────────────────────────────

        try:
            quali = fastf1.get_session(year, round_no, "Q")
            quali.load(laps=True, telemetry=False,
                       weather=True, messages=False)

            q_results = quali.results

            # Wet flag
            try:
                w       = quali.weather_data
                was_wet = bool(w["Rainfall"].any()) if not w.empty else False
                was_dry = not was_wet
            except Exception:
                was_dry = True

            # Pole time for gap calculation
            try:
                q_laps    = quali.laps.pick_quicklaps()
                pole_time = (
                    q_laps["LapTime"].min().total_seconds()
                    if not q_laps.empty else np.nan
                )
            except Exception:
                pole_time = np.nan

            if q_results is not None and not q_results.empty:
                for _, qrow in q_results.iterrows():
                    driver = str(qrow.get("Abbreviation", "UNK"))
                    team   = str(qrow.get("TeamName",     "Unknown"))

                    try:
                        q_pos = int(float(qrow.get("Position", np.nan)))
                    except (ValueError, TypeError):
                        q_pos = np.nan

                    # Driver best qualifying lap
                    try:
                        d_laps = quali.laps.pick_driver(driver).pick_quicklaps()
                        d_best = (
                            d_laps["LapTime"].min().total_seconds()
                            if not d_laps.empty else np.nan
                        )
                        gap_to_pole = (
                            d_best - pole_time
                            if pd.notna(d_best) and pd.notna(pole_time)
                            else np.nan
                        )
                    except Exception:
                        d_best      = np.nan
                        gap_to_pole = np.nan

                    quali_rows.append({
                        # ── Identity ───────────────────────────────────────
                        "season":     year,
                        "round":      round_no,
                        "circuit_id": circuit_id,
                        "driver":     driver,
                        "team":       team,
                        # ── Qualifying fields ──────────────────────────────
                        "quali_position":            q_pos,
                        "quali_best_lap_seconds":    d_best,
                        "gap_to_pole_seconds":       gap_to_pole,
                        "qualifying_session_was_dry": int(was_dry),
                    })

        except Exception as e:
            print(f"\n    ✗ Quali failed {year} R{round_no}: {e}")

        # ── FP2 Session (Long Run Only) ─────────────────────────────────────

        try:
            fp2_data, _ = extract_fp2_long_run(year, round_no)

            for driver, feats in fp2_data.items():
                fp2_rows.append({
                    # ── Identity ───────────────────────────────────────────
                    "season":     year,
                    "round":      round_no,
                    "circuit_id": circuit_id,
                    "driver":     driver,
                    # ── FP2 long run features ──────────────────────────────
                    "fp2_long_run_avg_gap": feats["fp2_long_run_avg_gap"],
                    "fp2_session_was_dry":  feats["fp2_session_was_dry"],
                })

        except Exception as e:
            print(f"\n    ✗ FP2 failed   {year} R{round_no}: {e}")


# ── Save Raw CSV Files ──────────────────────────────────────────────────────────

race_df  = pd.DataFrame(race_rows)
quali_df = pd.DataFrame(quali_rows)
fp2_df   = pd.DataFrame(fp2_rows)

race_df.to_csv("data/raw/raw_race_data.csv",  index=False)
quali_df.to_csv("data/raw/raw_quali_data.csv", index=False)
fp2_df.to_csv("data/raw/raw_fp2_data.csv",    index=False)