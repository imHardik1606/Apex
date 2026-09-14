import json
import pickle
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask app

app = Flask(__name__)

fastf1.Cache.enable_cache('cache')

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models" / "saved_models"
DATA_DIR = BASE_DIR / "data" / "processed"

# Load Models

with open(MODELS_DIR / "pre_qual_calibrated.pkl", "rb") as f:
    pre_cal = pickle.load(f)

with open(MODELS_DIR / "post_qual_calibrated.pkl", "rb") as f:
    post_cal = pickle.load(f)

with open(MODELS_DIR / "pre_qual_medians.pkl", "rb") as f:
    pre_med = pickle.load(f)

with open(MODELS_DIR / "post_qual_medians.pkl", "rb") as f:
    post_med = pickle.load(f)

with open(MODELS_DIR / "feature_columns.json", "r") as f:
    meta = json.load(f)

PRE_QUAL_FEATURES = meta["pre_qual_features"]
POST_QUAL_FEATURES = meta["post_qual_features"]

# Load historical data
pre_qual_history = pd.read_csv(DATA_DIR / "pre_qualifying_dataset.csv")

TEAM_ALIASES = {
    "Red Bull": "Red Bull Racing",
    "Red Bull Racing": "Red Bull Racing",
    "Haas": "Haas F1 Team",
    "Haas F1 Team": "Haas F1 Team",
    "RB": "Racing Bulls",
    "Visa Cash App RB": "Racing Bulls",
    "Racing Bulls": "Racing Bulls",
}


def normalize_team_name(team: str) -> str:
    """Return one canonical display name for a team label."""
    team_name = str(team).strip()
    return TEAM_ALIASES.get(team_name, team_name)


pre_qual_history["team"] = pre_qual_history["team"].map(normalize_team_name)
print(f"Loaded {len(pre_qual_history)} historical rows")
print(f"Seasons available: {sorted(pre_qual_history['season'].unique())}")

# Static circuit data
PIT_STOP_DELTA = {
    "Monaco": 24.0, "Singapore": 23.5, "Baku": 22.0,
    "Melbourne": 22.5, "Jeddah": 22.0, "Las Vegas": 21.5,
    "Miami": 22.0, "Montréal": 23.0, "Bahrain": 21.0,
    "Saudi Arabia": 22.0, "Australia": 22.5, "Japan": 22.0,
    "China": 21.5, "Imola": 22.0, "Spain": 21.0,
    "Austria": 20.5, "Britain": 21.5, "Hungary": 22.0,
    "Belgium": 21.0, "Netherlands": 21.5, "Italy": 20.0,
    "Azerbaijan": 22.0, "United States": 21.5, "Mexico": 22.0,
    "Brazil": 21.5, "Qatar": 21.0, "Abu Dhabi": 21.0,
}

# Current configured 2026 lineup used before qualifying is available.
CURRENT_GRID_2026 = {
    "VER": "Red Bull Racing", "LAW": "Red Bull Racing",
    "LEC": "Ferrari", "HAM": "Ferrari",
    "RUS": "Mercedes", "ANT": "Mercedes",
    "NOR": "McLaren", "PIA": "McLaren",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "GAS": "Alpine", "COL": "Alpine",
    "TSU": "Racing Bulls", "LIN": "Racing Bulls",
    "OCO": "Haas F1 Team", "BEA": "Haas F1 Team",
    "ALB": "Williams", "SAI": "Williams",
    "HUL": "Audi", "BOR": "Audi",
    "PER": "Cadillac", "BOT": "Cadillac",
}

GRID_2026 = set(CURRENT_GRID_2026)

# FastF1 Fetching Functions

@lru_cache(maxsize=128)
def fetch_quali_results(season: int, round: int) -> pd.DataFrame:
    """Fetch qualifying results for a given season and round."""
    try:
        session = fastf1.get_session(season, round, 'Q')
        session.load(telemetry=False, weather=True, messages=False)
        results = session.results[['Abbreviation', 'TeamName', 'Position']].copy()
        results.columns = ['driver', 'team', 'quali_position']
        results['team'] = results['team'].map(normalize_team_name)
        results['quali_position'] = pd.to_numeric(results['quali_position'], errors='coerce').fillna(20)
        
        # Check if session was dry
        if hasattr(session, 'weather_data') and session.weather_data is not None:
            rain_pct = session.weather_data['Rainfall'].mean() if 'Rainfall' in session.weather_data.columns else 0
            results['qualifying_session_was_dry'] = 1 if rain_pct < 0.1 else 0
        else:
            results['qualifying_session_was_dry'] = 1
        
        return results
    except Exception as e:
        print(f"Error fetching qualifying results for {season} Round {round}: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=128)
def fetch_fp2_data(season: int, round: int) -> pd.DataFrame:
    """Fetch FP2 data for a given season and round."""
    COMPOUND_MAP = {'SOFT': 2, 'MEDIUM': 1, 'HARD': 0, 'INTERMEDIATE': 3, 'WET': 4}

    try:
        session = fastf1.get_session(season, round, 'FP2')
        session.load(telemetry=False, weather=True, messages=False)
        
        laps = session.laps.copy()
        laps = laps[laps['TrackStatus'] == '1']  # Green flag only
        laps = laps[laps['LapTime'].notna()]
        laps['lap_s'] = laps['LapTime'].dt.total_seconds()

        # Calculate best long run pace per driver
        driver_pace = {}
        for driver in laps['Driver'].unique():
            driver_laps = laps[laps['Driver'] == driver].sort_values('LapNumber')
            
            for stint in driver_laps['Stint'].unique():
                stint_laps = driver_laps[driver_laps['Stint'] == stint]
                if len(stint_laps) >= 5:  # Long run = 5+ laps
                    median_pace = stint_laps['lap_s'].median()
                    compound = stint_laps['Compound'].mode().iloc[0] if not stint_laps['Compound'].mode().empty else 'MEDIUM'
                    
                    if driver not in driver_pace or median_pace < driver_pace[driver]['pace']:
                        driver_pace[driver] = {'pace': median_pace, 'compound': compound}
        
        if not driver_pace:
            return pd.DataFrame()
        
        fp2_df = pd.DataFrame([
            {'driver': d, 'avg_pace': data['pace'], 'compound': data['compound']}
            for d, data in driver_pace.items()
        ])
        
        best_pace = fp2_df['avg_pace'].min()
        fp2_df['fp2_long_run_avg_gap'] = fp2_df['avg_pace'] - best_pace
        fp2_df['starting_tyre_compound_encoded'] = fp2_df['compound'].map(COMPOUND_MAP).fillna(1).astype(int)
        
        # Session conditions
        if hasattr(session, 'weather_data') and session.weather_data is not None:
            rain_pct = session.weather_data['Rainfall'].mean() if 'Rainfall' in session.weather_data.columns else 0
            fp2_df['fp2_session_was_dry'] = 1 if rain_pct < 0.1 else 0
        else:
            fp2_df['fp2_session_was_dry'] = 1
        
        return fp2_df[['driver', 'fp2_long_run_avg_gap', 'fp2_session_was_dry', 'starting_tyre_compound_encoded']]
    
    except Exception as e:
        print(f"Error fetching FP2 data for {season} Round {round}: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=128)
def fetch_race_results(season: int, round: int) -> pd.DataFrame:
    """Fetch actual race results for a given season and round."""
    try:
        session = fastf1.get_session(season, round, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        
        results = session.results.copy()
        
        # Extract relevant columns - keep as strings first for cleaning
        result_df = pd.DataFrame({
            'position': results['Position'].astype(str).str.replace('+', '').str.replace('R', '').fillna('DNF'),
            'driver': results['Abbreviation'],
            'team': results['TeamName'],
            'grid_position': results['GridPosition'].astype(str).str.replace('+', '').str.replace('R', '').fillna('20'),
            'points': results['Points'].fillna(0).astype(int),
            'status': results['Status'] if 'Status' in results.columns else 'Finished',
            'time': results['Time'].astype(str) if 'Time' in results.columns else '',
            'laps': results['Laps'].fillna(0).astype(int),
        })
        
        # Clean up position column - handle DNF and non-finishes
        result_df['position'] = pd.to_numeric(result_df['position'], errors='coerce').fillna(999).astype(int)
        
        # Clean up grid_position column - convert to numeric
        result_df['grid_position'] = pd.to_numeric(result_df['grid_position'], errors='coerce').fillna(999).astype(int)
        
        # Sort by finishing position
        result_df = result_df.sort_values('position').reset_index(drop=True)
        
        return result_df
    
    except Exception as e:
        print(f"Error fetching race results for {season} Round {round}: {e}")
        return pd.DataFrame()

# Feature Engineering Functions
def get_pre_qual_features(season: int, round: int, circuit_id: str) -> pd.DataFrame:
    """Get pre-qualifying features from history"""
    # Try exact match first
    exact = pre_qual_history[
        (pre_qual_history['season'] == season) &
        (pre_qual_history['round'] == round) &
        (pre_qual_history['circuit_id'] == circuit_id)
    ].copy()
    
    if not exact.empty:
        return exact
    
    # For current season, use most recent round as proxy
    if season == 2026:
        season_data = pre_qual_history[pre_qual_history['season'] == season].copy()
        if not season_data.empty:
            last_round = season_data['round'].max()
            proxy = season_data[season_data['round'] == last_round].copy()
            # Update round-specific features
            proxy['season_round_number'] = round
            proxy['races_remaining_this_season'] = max(0, 24 - round)
            proxy['is_season_opener'] = int(round == 1)
            return proxy
    
    # Fallback to last season
    last_season = pre_qual_history['season'].max()
    last_round = pre_qual_history[pre_qual_history['season'] == last_season]['round'].max()
    return pre_qual_history[
        (pre_qual_history['season'] == last_season) &
        (pre_qual_history['round'] == last_round)
    ].copy()

def build_post_qual_features(season: int, round: int, circuit_id: str) -> pd.DataFrame:
    """Build post-qualifying features using FastF1"""
    # Get base features
    feature_df = get_pre_qual_features(season, round, circuit_id)
    
    if feature_df.empty:
        return pd.DataFrame()
    
    # Fetch live data
    quali_df = fetch_quali_results(season, round)
    fp2_df = fetch_fp2_data(season, round)
    
    if quali_df.empty:
        print("No qualifying data available")
        return pd.DataFrame()

    live_drivers = set(quali_df['driver'])
    feature_df = feature_df[feature_df['driver'].isin(live_drivers)].copy()
    missing_drivers = live_drivers - set(feature_df['driver'])

    if missing_drivers:
        fallback_rows = []
        for driver in missing_drivers:
            new_row = feature_df.median(numeric_only=True)
            new_row['driver'] = driver
            new_row['team'] = quali_df.loc[
                quali_df['driver'] == driver, 'team'
            ].iloc[0]
            fallback_rows.append(new_row)

        feature_df = pd.concat(
            [feature_df, pd.DataFrame(fallback_rows)],
            ignore_index=True,
        )
    
    # Merge qualifying data
    feature_df = feature_df.merge(
        quali_df[
            ['driver', 'team', 'quali_position', 'qualifying_session_was_dry']
        ].rename(columns={'team': 'live_team'}),
        on='driver', how='inner'
    )
    feature_df['team'] = feature_df['live_team']
    feature_df.drop(columns=['live_team'], inplace=True)
    
    # Merge FP2 data
    if not fp2_df.empty:
        feature_df = feature_df.merge(
            fp2_df[['driver', 'fp2_long_run_avg_gap', 'fp2_session_was_dry', 'starting_tyre_compound_encoded']],
            on='driver', how='left'
        )
    else:
        # Defaults when FP2 unavailable
        feature_df['fp2_long_run_avg_gap'] = 0.5
        feature_df['fp2_session_was_dry'] = 1
        feature_df['starting_tyre_compound_encoded'] = 1
    
    # Fill missing values
    feature_df['quali_position'] = feature_df['quali_position'].fillna(15)
    feature_df['qualifying_session_was_dry'] = feature_df['qualifying_session_was_dry'].fillna(1)
    feature_df['gap_to_pole_seconds'] = 1.0  # Default gap
    
    # Grid position (assuming no penalties for live predictions)
    feature_df['grid_position'] = feature_df['quali_position']
    feature_df['grid_penalty_applied'] = 0
    
    # Pit stop loss at this circuit
    feature_df['pit_stop_avg_time_loss_at_circuit'] = PIT_STOP_DELTA.get(circuit_id, 21.5)
    
    # Cross-driver features
    team_avg_quali = feature_df.groupby('team')['grid_position'].transform('mean')
    feature_df['gap_to_teammate_quali_this_weekend'] = feature_df['grid_position'] - team_avg_quali
    
    feature_df['quali_position_vs_season_avg'] = (
        feature_df['grid_position'] - feature_df['driver_avg_quali_position_this_season']
    ).fillna(0)
    
    # Tyre compound vs majority
    if not feature_df['starting_tyre_compound_encoded'].empty:
        majority = feature_df['starting_tyre_compound_encoded'].mode().iloc[0]
    else:
        majority = 1
    feature_df['tyre_compound_vs_majority'] = (feature_df['starting_tyre_compound_encoded'] != majority).astype(int)
    
    # Count faster cars ahead
    def count_faster_ahead(df):
        results = []
        for _, row in df.iterrows():
            grid = row.get('grid_position', 20)
            pace = row.get('team_avg_finish_last_3', 10)
            ahead = df[df['grid_position'] < grid]
            faster = ahead[ahead['team_avg_finish_last_3'] < pace]
            results.append(len(faster))
        return results
    
    feature_df['drivers_ahead_with_higher_team_pace'] = count_faster_ahead(feature_df)
    
    return feature_df

# Helper Functions

@lru_cache(maxsize=32)
def get_current_season_drivers(season: int) -> set:
    """Get set of drivers currently participating in a given season"""
    if season == 2026:
        # Return actual 2026 grid
        return GRID_2026
    
    # For other seasons, use historical data
    season_data = pre_qual_history[pre_qual_history['season'] == season]
    return set(season_data['driver'].unique()) if not season_data.empty else set()

def filter_results_by_current_drivers(results: list, season: int, fallback_to_all: bool = False) -> list:
    """
    Filter prediction results to only include drivers in current season.
    For 2026+, this ensures only registered drivers are shown.
    """
    current_drivers = get_current_season_drivers(season)
    
    if not current_drivers and fallback_to_all:
        # Fallback: show all if season data incomplete
        return results
    
    if current_drivers:
        # Filter results to only include drivers in current season
        filtered = [r for r in results if r['driver'] in current_drivers]
        return filtered if filtered else results
    
    return results

def normalize_probs(probs: np.ndarray) -> np.ndarray:
    """Normalize probabilities to sum to 1"""
    total = probs.sum()
    return probs / total if total > 0 else probs


def fill_missing_2026_drivers(
    feature_df: pd.DataFrame,
    season: int,
    active_grid: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Ensure feature_df contains all configured drivers for 2026 season.
    For missing drivers, create rows using teammate data or median values.
    Removes drivers not in the active configured grid.
    """
    if season != 2026:
        return feature_df

    active_grid = active_grid or CURRENT_GRID_2026
    active_drivers = set(active_grid)
    
    feature_df = feature_df[feature_df['driver'].isin(active_drivers)].copy()
    
    current_drivers = set(feature_df['driver'].unique())
    missing_drivers = active_drivers - current_drivers
    
    if missing_drivers:
        print(f"Missing drivers in feature_df: {missing_drivers}")
        
        # Create rows for missing drivers
        new_rows = []
        for driver in missing_drivers:
            team = active_grid[driver]
            
            # Try to find teammate data
            teammate_rows = feature_df[feature_df['team'] == team]
            
            if not teammate_rows.empty:
                # Copy from teammate
                new_row = teammate_rows.iloc[0].copy()
                new_row['driver'] = driver
            else:
                # Create row from medians
                new_row = feature_df.median(numeric_only=True)
                new_row['driver'] = driver
                new_row['team'] = team
            
            new_rows.append(new_row)
        
        # Append new rows to feature_df
        new_df = pd.DataFrame(new_rows)
        feature_df = pd.concat([feature_df, new_df], ignore_index=True)
        print(f"Added {len(new_rows)} missing drivers. Total drivers now: {len(feature_df)}")
    
    # Validate
    final_drivers = set(feature_df['driver'].unique())
    missing = active_drivers - final_drivers
    extra = final_drivers - active_drivers
    
    if missing:
        print(f"WARNING: Still missing drivers after fill: {missing}")
    if extra:
        print(f"WARNING: Extra drivers found (should be removed): {extra}")
    
    return feature_df


# API Endpoints

@app.route('/predict/pre-qualifying', methods=['POST'])
def predict_pre_qual():
    """Pre-qualifying prediction - uses only historical data"""
    try:

        data = request.get_json()

        # Validate input
        required = ['season', 'round', 'circuit_id']
        if not all(k in data for k in required):
            return jsonify({'error': f'Missing fields. Required: {required}'}), 400

        season = int(data['season'])
        round_no = int(data['round'])
        circuit_id = data.get('circuit_id')
        
        # Get features
        feature_df = get_pre_qual_features(season, round_no, circuit_id)
        
        if feature_df.empty:
            return jsonify({'error': 'No historical data available'}), 404

        # Fill missing drivers for 2026 season
        feature_df = fill_missing_2026_drivers(
            feature_df,
            season,
            CURRENT_GRID_2026,
        )

        if season == 2026:
            feature_df['team'] = feature_df['driver'].map(
                CURRENT_GRID_2026
            ).fillna(feature_df['team'])
        
        # Validation for 2026
        if season == 2026:
            current_drivers = set(feature_df['driver'].unique())
            configured_drivers = set(CURRENT_GRID_2026)
            missing = configured_drivers - current_drivers
            extra = current_drivers - configured_drivers
            
            if missing or extra:
                error_msg = []
                if missing:
                    error_msg.append(f"Missing drivers: {missing}")
                if extra:
                    error_msg.append(f"Extra drivers: {extra}")
                print(f"ERROR: {', '.join(error_msg)}")
                return jsonify({'error': ', '.join(error_msg)}), 400
            
            assert len(current_drivers) == len(CURRENT_GRID_2026), (
                f"Expected {len(CURRENT_GRID_2026)} drivers, "
                f"got {len(current_drivers)}"
            )

        # Prepare features
        available = [c for c in PRE_QUAL_FEATURES if c in feature_df.columns]
        X = feature_df[available].fillna(pre_med)
        
        # Predict
        raw_probs = pre_cal.predict_proba(X)[:, 1]
        norm_probs = normalize_probs(raw_probs)

        # Format response
        results = []
        for i, (_, row) in enumerate(feature_df.iterrows()):
            results.append({
                'driver': str(row['driver']),
                'team': str(row['team']),
                'win_probability': round(float(norm_probs[i]), 4)
            })
        
        results.sort(key=lambda x: x['win_probability'], reverse=True)
        
        # Filter to only current season drivers
        if season != 2026:
            results = filter_results_by_current_drivers(results, season, fallback_to_all=True)   

        return jsonify({
            'model': 'pre-qualifying',
            'season': season,
            'round': round_no,
            'circuit': circuit_id,
            'predictions': results,
            'total_drivers': len(results),
            'note': f'Showing {len(results)} drivers registered for {season} season'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
@app.route('/predict/post-qualifying', methods=['POST'])
def predict_post_qual():
    """Post-qualifying prediction - includes live session data"""
    try:
        data = request.get_json()

        # Validate input
        required = ['season', 'round', 'circuit_id']
        if not all(k in data for k in required):
            return jsonify({'error': f'Missing fields. Required: {required}'}), 400
        
        season = int(data['season'])
        round_no = int(data['round'])
        circuit_id = str(data['circuit_id'])
        
        # Build features with live data
        feature_df = build_post_qual_features(season, round_no, circuit_id)

        if feature_df.empty:
            return jsonify({'error': 'Could not build features for this race'}), 404
        
        # Prepare features
        available = [c for c in POST_QUAL_FEATURES if c in feature_df.columns]
        X = feature_df[available].fillna(post_med)
        
        # Predict
        raw_probs = post_cal.predict_proba(X)[:, 1]
        norm_probs = normalize_probs(raw_probs)

         # Format response
        results = []
        for i, (_, row) in enumerate(feature_df.iterrows()):
            results.append({
                'driver': str(row['driver']),
                'team': str(row['team']),
                'grid_position': int(row.get('grid_position', 0)),
                'win_probability': round(float(norm_probs[i]), 4)
            })
        
        results.sort(key=lambda x: x['win_probability'], reverse=True)
        
        return jsonify({
            'model': 'post-qualifying',
            'season': season,
            'round': round_no,
            'circuit': circuit_id,
            'predictions': results,
            'total_drivers': len(results),
            'timestamp': datetime.utcnow().isoformat(),
            'note': f'Showing {len(results)} drivers registered for {season} season'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/race-results', methods=['POST'])
def get_race_results():
    """Fetch actual race results from FastF1"""
    try:
        data = request.get_json()

        # Validate input
        required = ['season', 'round']
        if not all(k in data for k in required):
            return jsonify({'error': f'Missing fields. Required: {required}'}), 400

        season = int(data['season'])
        round_no = int(data['round'])
        circuit_id = str(data.get('circuit_id', 'Unknown'))
        
        # Fetch race results
        results_df = fetch_race_results(season, round_no)
        
        if results_df.empty:
            return jsonify({'error': 'No race results available for this round'}), 404
        
        # Format response
        results = []
        for _, row in results_df.iterrows():
            # Ensure position and grid_position are integers for comparison
            pos = int(row['position']) if pd.notna(row['position']) else 999
            grid_pos = int(row['grid_position']) if pd.notna(row['grid_position']) else 999
            
            results.append({
                'position': pos if pos < 999 else None,
                'driver': str(row['driver']),
                'team': str(row['team']),
                'grid_position': grid_pos if grid_pos < 999 else None,
                'points': int(row['points']),
                'status': str(row['status']),
                'laps': int(row['laps']),
                'time': str(row['time']) if str(row['time']) != '' else 'DNF'
            })
        
        # Get winner (first finished driver with position)
        winner = next(
            (r for r in results if r['position'] is not None),
            results[0] if results else None
        )
        
        return jsonify({
            'season': season,
            'round': round_no,
            'circuit': circuit_id,
            'results': results,
            'total_finishers': len([r for r in results if r['position'] is not None]),
            'total_drivers': len(results),
            'winner': winner['driver'] if winner else None,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'pre_qual_features': len(PRE_QUAL_FEATURES),
        'post_qual_features': len(POST_QUAL_FEATURES),
        'historical_rows': len(pre_qual_history),
        'seasons': sorted(pre_qual_history['season'].unique().tolist())
    })


@app.route('/', methods=['GET'])
def home():
    """API information"""
    return jsonify({
        'name': 'F1 Race Winner Prediction API',
        'version': '1.0.0',
        'endpoints': {
            'POST /predict/pre-qualifying': 'Predict before qualifying (historical data only)',
            'POST /predict/post-qualifying': 'Predict after qualifying (includes live session data)',
            'POST /race-results': 'Get actual race results from FastF1',
            'GET /health': 'Health check',
            'GET /': 'This information'
        },
        'example': {
            'season': 2026,
            'round': 8,
            'circuit_id': 'Monaco'
        }
    })

if __name__ == '__main__':
    app.run(debug=True)