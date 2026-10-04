import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

TARGET = "winner"
REDUNDANT_FEATURE = "driver_avg_quali_position_last_3"
GRID_FEATURE = "grid_position"
DIAGNOSTIC_COLSAMPLE_BYTREE = 0.8
DIAGNOSTIC_COLSAMPLE_BYLEVEL = 1.0
DIAGNOSTIC_COLSAMPLE_BYNODE = 0.8

FEATURES = [
    "team_avg_finish_last_3", "team_avg_finish_last_5", "team_points_last_3",
    "team_points_this_season", "team_championship_position", "team_points_gap_to_leader",
    "team_dnf_rate_this_season", "team_finish_rate_this_season",
    "driver_avg_finish_last_3", "driver_avg_finish_last_5", "driver_points_this_season",
    "driver_championship_position", "driver_points_gap_to_leader",
    "driver_dnf_rate_this_season", "driver_finish_rate_this_season",
    "driver_top5_rate_this_season", "driver_avg_quali_position_last_3",
    "driver_avg_quali_position_last_5", "driver_avg_quali_position_this_season",
    "driver_quali_to_finish_delta_last_5", "avg_quali_gap_to_teammate_this_season",
    "avg_finish_gap_to_teammate_this_season", "teammate_head_to_head_ratio_this_season",
    "driver_avg_finish_at_circuit_last5", "driver_best_finish_at_circuit_last5",
    "driver_podiums_at_circuit_last5", "driver_avg_quali_position_at_circuit_last5",
    "driver_dnf_rate_at_circuit", "driver_visits_to_circuit", "driver_circuit_vs_season_delta",
    "team_avg_finish_at_circuit_last5", "team_wins_at_circuit_last5",
    "team_podium_rate_at_circuit", "team_avg_quali_position_at_circuit_last5",
    "team_dnf_rate_at_circuit", "team_circuit_vs_season_delta",
    "circuit_type_encoded", "circuit_avg_overtakes_last3years",
    "circuit_safety_car_probability", "circuit_lap_count", "season_round_number",
    "races_remaining_this_season", "is_season_opener", "seasons_since_regulation_change",
    "grid_position", "gap_to_pole_seconds", "gap_to_teammate_quali_this_weekend",
    "quali_position_vs_season_avg", "qualifying_session_was_dry", "grid_penalty_applied",
    "fp2_long_run_avg_gap", "fp2_session_was_dry", "starting_tyre_compound_encoded",
    "tyre_compound_vs_majority", "pit_stop_avg_time_loss_at_circuit",
    "drivers_ahead_with_higher_team_pace",
]


def make_model(scale_pos_weight):
    return XGBClassifier(
        objective="binary:logistic", scale_pos_weight=scale_pos_weight,
        max_depth=3, n_estimators=300, learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=DIAGNOSTIC_COLSAMPLE_BYTREE,
        colsample_bylevel=DIAGNOSTIC_COLSAMPLE_BYLEVEL,
        colsample_bynode=DIAGNOSTIC_COLSAMPLE_BYNODE,
        min_child_weight=5,
        gamma=1, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="aucpr", random_state=42, verbosity=0,
    )


def normalize_per_race(meta, probabilities):
    values = pd.Series(probabilities, index=meta.index, name="probability")
    return values.groupby([meta["season"], meta["round"]]).transform(
        lambda group: group / group.sum() if group.sum() > 0 else group
    ).to_numpy()


def calculate_metrics(model, X, y, meta):
    raw = model.predict_proba(X)[:, 1]
    normalized = normalize_per_race(meta, raw)
    ranking = meta[["season", "round", TARGET]].copy()
    ranking["probability"] = normalized
    top1 = ranking.groupby(["season", "round"], group_keys=False).apply(
        lambda group: group.loc[group["probability"].idxmax(), TARGET]
    ).mean()
    return {
        "top1_accuracy": float(top1),
        "log_loss": float(log_loss(y, raw)),
        "brier_score": float(brier_score_loss(y, raw)),
        "auc_pr": float(average_precision_score(y, raw)),
    }


def split_data(dataset, feature_columns):
    seasons = sorted(dataset["season"].unique())
    train_seasons = seasons[:-2]
    calibration_season = seasons[-2]
    test_season = seasons[-1]
    available = [column for column in feature_columns if column in dataset.columns]

    train = dataset[dataset["season"].isin(train_seasons)].copy()
    calibration = dataset[dataset["season"] == calibration_season].copy()
    test = dataset[dataset["season"] == test_season].copy()
    scale_pos_weight = round(
        (train[TARGET] == 0).sum() / max((train[TARGET] == 1).sum(), 1)
    )

    X_train = train[available].copy()
    X_calibration = calibration[available].copy()
    X_test = test[available].copy()
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_calibration = X_calibration.fillna(medians)
    X_test = X_test.fillna(medians)

    return {
        "available": available,
        "train": train,
        "calibration": calibration,
        "test": test,
        "X_train": X_train,
        "X_calibration": X_calibration,
        "X_test": X_test,
        "scale_pos_weight": scale_pos_weight,
        "split": (train_seasons, calibration_season, test_season),
    }


def train_variant(parts, label, feature_weights=None):
    model = make_model(parts["scale_pos_weight"])
    fit_kwargs = {"verbose": False}
    if feature_weights is not None:
        fit_kwargs["feature_weights"] = feature_weights
    model.fit(parts["X_train"], parts["train"][TARGET], **fit_kwargs)

    calibrated = CalibratedClassifierCV(estimator=model, method="isotonic", cv=5)
    calibrated.fit(parts["X_calibration"], parts["calibration"][TARGET])
    meta = parts["test"][["season", "round", TARGET]]
    result = calculate_metrics(calibrated, parts["X_test"], parts["test"][TARGET], meta)
    result["label"] = label
    result["model"] = model
    result["calibrated"] = calibrated
    result["features"] = len(parts["available"])
    result["available"] = parts["available"]
    return result


def print_metrics(results):
    print("\nGRID POSITION ABLATION RESULTS")
    print("metric                 baseline    grid_weight_2x    grid_weight_4x")
    for metric, label in [
        ("log_loss", "log_loss"),
        ("brier_score", "brier_score"),
        ("auc_pr", "auc_pr"),
        ("top1_accuracy", "top1_accuracy"),
    ]:
        print(
            f"{label:<22} {results[0][metric]:>8.4f}"
            f" {results[1][metric]:>16.4f}"
            f" {results[2][metric]:>16.4f}"
        )


def print_redundancy_metrics(baseline, removed):
    print("\nLAST_3 REDUNDANCY PERFORMANCE")
    print("metric                 baseline    last_3_removed")
    for metric in ["log_loss", "brier_score", "auc_pr", "top1_accuracy"]:
        print(
            f"{metric:<22} {baseline[metric]:>8.4f}"
            f" {removed[metric]:>16.4f}"
        )


def print_importance_comparison(baseline, removed, feature_columns):
    baseline_importance = dict(zip(feature_columns, baseline["model"].feature_importances_))
    removed_importance = dict(zip(removed["available"], removed["model"].feature_importances_))
    print("\nQUALIFYING TREND REDUNDANCY")
    print("feature                                      baseline    last_3_removed")
    for feature in ["driver_avg_quali_position_last_5", REDUNDANT_FEATURE, GRID_FEATURE]:
        print(
            f"{feature:<45} {baseline_importance.get(feature, 0):>8.4f}"
            f" {removed_importance.get(feature, 0):>16.4f}"
        )


def print_front_grid_analysis(baseline, parts):
    test = parts["test"].copy()
    probabilities = baseline["calibrated"].predict_proba(parts["X_test"])[:, 1]
    test["model_probability"] = probabilities
    rows = []
    race_rows = []

    for (season, round_number), race in test.groupby(["season", "round"]):
        front = race[race[GRID_FEATURE] <= 3].sort_values(GRID_FEATURE)
        if front.empty:
            continue
        model_pick = race.loc[race["model_probability"].idxmax()]
        winner_rows = race[race[TARGET] == 1]
        if winner_rows.empty:
            continue
        winner = winner_rows.iloc[0]
        winner_grid = int(winner[GRID_FEATURE])
        race_rows.append({
            "season": season,
            "round": round_number,
            "winner": winner["driver"],
            "winner_grid": winner_grid,
            "result_type": "chalk (pole/front-row winner)" if winner_grid <= 2 else "upset (winner outside top 3)" if winner_grid > 3 else "top-3 non-front-row",
        })
        if model_pick["driver"] in set(front["driver"]):
            continue
        rows.append({
            "season": season,
            "round": round_number,
            "front_grid_drivers": ", ".join(
                f"{row.driver} P{int(row[GRID_FEATURE])} ({row.model_probability:.4f})"
                for row in front.itertuples()
            ),
            "winner": winner["driver"],
            "winner_grid": int(winner[GRID_FEATURE]),
            "winner_probability": float(winner["model_probability"]),
            "model_pick": model_pick["driver"],
            "model_pick_grid": int(model_pick[GRID_FEATURE]),
            "model_pick_probability": float(model_pick["model_probability"]),
            "model_pick_correct": bool(model_pick[TARGET] == 1),
            "front_grid_winner": bool(winner["driver"] in set(front["driver"])),
        })

    print("\nTEST RACE GRID ANALYSIS")
    race_table = pd.DataFrame(race_rows).sort_values(["season", "round"])
    print(race_table.to_string(index=False))
    print(
        f"\nExact test races: {len(race_table)} | "
        f"chalk: {(race_table['winner_grid'] <= 2).sum()} | "
        f"winner started P3: {(race_table['winner_grid'] == 3).sum()} | "
        f"genuine upsets (outside top 3): {(race_table['winner_grid'] > 3).sum()}"
    )

    print("\nFRONT-GRID DISAGREEMENT ANALYSIS")
    if not rows:
        print("No test races found where the model pick was outside the actual P1-P3 group.")
        return
    analysis = pd.DataFrame(rows)
    print(analysis.to_string(index=False))
    print(
        f"\nCases: {len(analysis)} | Model picks correct: "
        f"{int(analysis['model_pick_correct'].sum())}/{len(analysis)}"
    )
    print(
        "Interpretation: a correct model pick means historical form beat the front-grid "
        "candidate in that race; an incorrect pick means grid position would have been safer."
    )


def main():
    dataset = pd.read_csv("data/processed/post_qualifying_dataset.csv")
    feature_columns = [column for column in FEATURES if column in dataset.columns]
    parts = split_data(dataset, feature_columns)

    print("DIAGNOSTIC XGBOOST PARAMETERS")
    print(f"colsample_bytree={DIAGNOSTIC_COLSAMPLE_BYTREE}")
    print(f"colsample_bylevel={DIAGNOSTIC_COLSAMPLE_BYLEVEL}")
    print(f"colsample_bynode={DIAGNOSTIC_COLSAMPLE_BYNODE}")
    print("feature_weights test uses node-level column subsampling at 0.8")

    baseline = train_variant(parts, "baseline")
    removed_columns = [column for column in feature_columns if column != REDUNDANT_FEATURE]
    removed_parts = split_data(dataset, removed_columns)
    removed = train_variant(removed_parts, "last_3_removed")

    weights_2x = np.ones(len(feature_columns))
    weights_2x[feature_columns.index(GRID_FEATURE)] = 2.0
    weights_4x = np.ones(len(feature_columns))
    weights_4x[feature_columns.index(GRID_FEATURE)] = 4.0
    weighted_2x = train_variant(parts, "grid_weight_2x", weights_2x)
    weighted_4x = train_variant(parts, "grid_weight_4x", weights_4x)

    print(
        f"\nSplit: train {parts['split'][0]}, calibration {parts['split'][1]}, "
        f"test {parts['split'][2]}"
    )
    print_importance_comparison(baseline, removed, feature_columns)
    print_redundancy_metrics(baseline, removed)
    print_metrics([baseline, weighted_2x, weighted_4x])
    print_front_grid_analysis(baseline, parts)


if __name__ == "__main__":
    main()
