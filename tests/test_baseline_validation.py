import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation import runner


PREDICTION_COLUMNS = ["scenario", "fold", "ts_epoch", "y_true", "y_pred", "score"]


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_ROOT", str(tmp_path))
    config = {
        "data": {"timestamp_column": "window_start", "features": ["f_n_logs"]},
        "leakage_prevention": {"buffer_gap_mins": 60, "clean_train_only": True},
        "scenarios": {
            "holdout": {"strategy": "holdout", "train_ratio": 0.6},
            "expanding": {"strategy": "expanding_cv", "n_splits": 3},
        },
        "output": {"save_predictions": True},
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    rows = np.arange(180)
    df = pd.DataFrame(
        {
            "window_start": 1_700_000_000 + rows * 300,
            "y": (rows % 7 == 0).astype(int),
            "f_n_logs": np.where(rows % 5 == 0, 10_000, 1 + rows % 4),
            "f_sev_FATAL": (rows % 11 == 0).astype(int),
        }
    ).sample(frac=1, random_state=42)
    reports_dir = tmp_path / "reports" / "evaluation"
    return runner.EvaluationRunner(config_path), df, reports_dir


def test_baseline_predictions_match_each_scenario_and_fold(evaluation):
    evaluator, df, reports_dir = evaluation
    raw, summary = evaluator.run_all(df)
    actual = pd.read_csv(reports_dir / "baseline_predictions.csv")

    assert actual.columns.tolist() == PREDICTION_COLUMNS
    assert set(raw["detector"]) == {"Baseline_ZScore", "Rule_FATAL", "Isolation_Forest"}
    baseline_metrics = raw.loc[raw["detector"] == "Baseline_ZScore"]
    assert len(actual) == baseline_metrics["test_size"].sum()
    pd.testing.assert_frame_equal(pd.read_csv(reports_dir / "evaluation_folds_raw.csv"), raw)
    pd.testing.assert_frame_equal(
        pd.read_csv(reports_dir / "evaluation_mean_std_summary.csv"), summary
    )

    expected_groups = set()
    for scenario, config in evaluator.cfg["scenarios"].items():
        for fold in evaluator.splitter.split_by_scenario(df, scenario, config):
            expected_groups.add((scenario, fold.fold_idx))
            detector = runner.ZScoreBaselineDetector().fit(fold.train_df)
            expected_pred, expected_scores = detector.predict(fold.test_df)
            group = actual[(actual["scenario"] == scenario) & (actual["fold"] == fold.fold_idx)]
            assert len(group) == len(fold.test_df)
            np.testing.assert_array_equal(group["ts_epoch"], fold.test_df["window_start"])
            np.testing.assert_array_equal(group["y_true"], fold.test_df["y"])
            np.testing.assert_array_equal(group["y_pred"], expected_pred)
            np.testing.assert_allclose(group["score"], expected_scores)

    assert set(zip(actual["scenario"], actual["fold"])) == expected_groups
    assert len(actual.loc[actual["scenario"] == "expanding", "fold"].unique()) == 3
    assert set(actual["y_pred"]) == {0, 1}


def test_rerun_replaces_previous_predictions(evaluation):
    evaluator, df, reports_dir = evaluation
    prediction_path = reports_dir / "baseline_predictions.csv"
    evaluator.run_all(df)
    first = pd.read_csv(prediction_path)

    later = df.assign(window_start=df["window_start"] + 1_000_000)
    raw, _ = evaluator.run_all(later)
    second = pd.read_csv(prediction_path)

    assert len(second) == raw.loc[raw["detector"] == "Baseline_ZScore", "test_size"].sum()
    pd.testing.assert_frame_equal(second, first.assign(ts_epoch=first["ts_epoch"] + 1_000_000))


def test_save_predictions_false_skips_export(evaluation):
    evaluator, df, reports_dir = evaluation
    evaluator.cfg["output"]["save_predictions"] = False

    evaluator.run_all(df)

    assert not (reports_dir / "baseline_predictions.csv").exists()
    assert (reports_dir / "evaluation_folds_raw.csv").is_file()
    assert (reports_dir / "evaluation_mean_std_summary.csv").is_file()
