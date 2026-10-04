import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

TARGET = "winner"
FP2_FEATURES = {"fp2_long_run_avg_gap", "fp2_session_was_dry"}
BASE_FEATURES = [
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
    "fp2_long_run_avg_gap", "fp2_session_was_dry",
    "starting_tyre_compound_encoded", "tyre_compound_vs_majority",
    "pit_stop_avg_time_loss_at_circuit", "drivers_ahead_with_higher_team_pace",
]


def make_model(scale_pos_weight):
    return XGBClassifier(
        objective="binary:logistic", scale_pos_weight=scale_pos_weight,
        max_depth=3, n_estimators=300, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=1, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="aucpr", random_state=42, verbosity=0,
    )


def normalize_per_race(meta, probabilities):
    values = pd.Series(probabilities, index=meta.index, name="probability")
    return values.groupby([meta["season"], meta["round"]]).transform(
        lambda group: group / group.sum() if group.sum() > 0 else group
    ).to_numpy()


def metrics(model, X, y, meta):
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


def run_experiment(dataset, feature_columns, label):
    seasons = sorted(dataset["season"].unique())
    train_seasons = seasons[:-2]
    calibration_season = seasons[-2]
    test_season = seasons[-1]
    available = [column for column in feature_columns if column in dataset.columns]

    train = dataset[dataset["season"].isin(train_seasons)].copy()
    calibration = dataset[dataset["season"] == calibration_season].copy()
    test = dataset[dataset["season"] == test_season].copy()
    scale_pos_weight = round((train[TARGET] == 0).sum() / max((train[TARGET] == 1).sum(), 1))

    X_train = train[available].copy()
    X_calibration = calibration[available].copy()
    X_test = test[available].copy()
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_calibration = X_calibration.fillna(medians)
    X_test = X_test.fillna(medians)

    model = make_model(scale_pos_weight)
    model.fit(X_train, train[TARGET], verbose=False)
    calibrated = CalibratedClassifierCV(estimator=model, method="isotonic", cv=5)
    calibrated.fit(X_calibration, calibration[TARGET])
    result = metrics(calibrated, X_test, test[TARGET], test[["season", "round", TARGET]])
    result["label"] = label
    result["features"] = len(available)
    return result


def main():
    dataset = pd.read_csv("data/processed/post_qualifying_dataset.csv")
    with_fp2 = [column for column in BASE_FEATURES if column in dataset.columns]
    without_fp2 = [column for column in with_fp2 if column not in FP2_FEATURES]
    results = [
        run_experiment(dataset, with_fp2, "With FP2"),
        run_experiment(dataset, without_fp2, "Without FP2"),
    ]
    print("\nFP2 ABLATION RESULTS")
    print(f"Split: train {sorted(dataset['season'].unique())[:-2]}, calibration {sorted(dataset['season'].unique())[-2]}, test {sorted(dataset['season'].unique())[-1]}")
    for result in results:
        print(
            f"{result['label']}: top1_accuracy={result['top1_accuracy']:.4f} "
            f"({result['top1_accuracy']:.2%}), log_loss={result['log_loss']:.4f}, "
            f"brier={result['brier_score']:.4f}, auc_pr={result['auc_pr']:.4f}, "
            f"features={result['features']}"
        )


if __name__ == "__main__":
    main()
