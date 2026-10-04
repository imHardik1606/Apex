import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

TARGET = "winner"
REMOVED_FEATURE = "driver_avg_quali_position_last_3"
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
FOLDS = [("A", [2022, 2023], 2024), ("B", [2022, 2023, 2024], 2025), ("C", [2022, 2023, 2024, 2025], 2026)]


def make_model(scale_pos_weight, colsample_bynode):
    return XGBClassifier(
        objective="binary:logistic", scale_pos_weight=scale_pos_weight,
        max_depth=3, n_estimators=300, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, colsample_bylevel=1.0,
        colsample_bynode=colsample_bynode, min_child_weight=5,
        gamma=1, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="aucpr", random_state=42, verbosity=0,
    )


def normalize_per_race(meta, probabilities):
    values = pd.Series(probabilities, index=meta.index, name="probability")
    return values.groupby([meta["season"], meta["round"]]).transform(
        lambda group: group / group.sum() if group.sum() > 0 else group
    ).to_numpy()


def evaluate(model, X_test, test):
    raw = model.predict_proba(X_test)[:, 1]
    meta = test[["season", "round", TARGET]].copy()
    normalized = normalize_per_race(meta, raw)
    ranking = meta.copy()
    ranking["probability"] = normalized
    top1 = ranking.groupby(["season", "round"], group_keys=False).apply(
        lambda group: group.loc[group["probability"].idxmax(), TARGET]
    ).mean()
    return {
        "log_loss": float(log_loss(test[TARGET], raw)),
        "brier": float(brier_score_loss(test[TARGET], raw)),
        "auc_pr": float(average_precision_score(test[TARGET], raw)),
        "top1": float(top1),
    }


def run_fold(dataset, train_seasons, test_season, feature_columns, colsample_bynode):
    train = dataset[dataset["season"].isin(train_seasons)].copy()
    test = dataset[dataset["season"] == test_season].copy()
    available = [column for column in feature_columns if column in dataset.columns]
    X_train = train[available].copy()
    X_test = test[available].copy()
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    scale_pos_weight = round((train[TARGET] == 0).sum() / max((train[TARGET] == 1).sum(), 1))
    model = make_model(scale_pos_weight, colsample_bynode)
    model.fit(X_train, train[TARGET], verbose=False)
    result = evaluate(model, X_test, test)
    result["features"] = len(available)
    return result


def print_table(title, rows):
    print(f"\n{title}")
    print("fold/test   log_loss   brier     auc_pr    top1")
    for row in rows:
        print(
            f"{row['fold']}/{row['test_season']}       "
            f"{row['log_loss']:.4f}     {row['brier']:.4f}    "
            f"{row['auc_pr']:.4f}    {row['top1']:.4f}"
        )


def main():
    dataset = pd.read_csv("data/processed/post_qualifying_dataset.csv")
    available = [column for column in FEATURES if column in dataset.columns]

    print("WALK-FORWARD VALIDATION")
    print("No calibration season is introduced: each requested fold trains only on its listed training seasons and evaluates on its listed test season.")
    print("Model config: colsample_bytree=0.8, colsample_bylevel=1.0, production colsample_bynode=1.0")

    removal_rows = []
    for fold, train_seasons, test_season in FOLDS:
        result = run_fold(
            dataset, train_seasons, test_season,
            [column for column in available if column != REMOVED_FEATURE],
            colsample_bynode=1.0,
        )
        result.update({"fold": fold, "test_season": test_season, "variant": "without_last_3"})
        removal_rows.append(result)
        result = run_fold(dataset, train_seasons, test_season, available, colsample_bynode=1.0)
        result.update({"fold": fold, "test_season": test_season, "variant": "with_last_3"})
        removal_rows.append(result)

    print_table("STEP 1: LAST-3 FEATURE WALK-FORWARD", [row for row in removal_rows if row["variant"] == "with_last_3"])
    print_table("STEP 1: LAST-3 REMOVED WALK-FORWARD", [row for row in removal_rows if row["variant"] == "without_last_3"])

    sampling_rows = []
    for colsample_bynode in [1.0, 0.8, 0.6]:
        for fold, train_seasons, test_season in FOLDS:
            result = run_fold(dataset, train_seasons, test_season, available, colsample_bynode)
            result.update({"fold": fold, "test_season": test_season, "colsample_bynode": colsample_bynode})
            sampling_rows.append(result)

    for colsample_bynode in [1.0, 0.8, 0.6]:
        print_table(
            f"STEP 2: colsample_bynode={colsample_bynode}",
            [row for row in sampling_rows if row["colsample_bynode"] == colsample_bynode],
        )

    print("\nSTEP 1 DELTAS: without_last_3 minus with_last_3")
    for fold in ["A", "B", "C"]:
        with_feature = next(row for row in removal_rows if row["fold"] == fold and row["variant"] == "with_last_3")
        without_feature = next(row for row in removal_rows if row["fold"] == fold and row["variant"] == "without_last_3")
        print(
            f"Fold {fold}: log_loss {without_feature['log_loss'] - with_feature['log_loss']:+.4f}, "
            f"brier {without_feature['brier'] - with_feature['brier']:+.4f}, "
            f"auc_pr {without_feature['auc_pr'] - with_feature['auc_pr']:+.4f}, "
            f"top1 {without_feature['top1'] - with_feature['top1']:+.4f}"
        )

    print("\nSTEP 2 DELTAS: each colsample_bynode value minus 1.0")
    for value in [0.8, 0.6]:
        print(f"colsample_bynode={value}")
        for fold in ["A", "B", "C"]:
            baseline = next(row for row in sampling_rows if row["fold"] == fold and row["colsample_bynode"] == 1.0)
            candidate = next(row for row in sampling_rows if row["fold"] == fold and row["colsample_bynode"] == value)
            print(
                f"  Fold {fold}: log_loss {candidate['log_loss'] - baseline['log_loss']:+.4f}, "
                f"brier {candidate['brier'] - baseline['brier']:+.4f}, "
                f"auc_pr {candidate['auc_pr'] - baseline['auc_pr']:+.4f}, "
                f"top1 {candidate['top1'] - baseline['top1']:+.4f}"
            )


if __name__ == "__main__":
    main()
