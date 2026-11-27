import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation import runner


PREDICTION_COLUMNS = ["scenario", "fold", "detector", "ts_epoch", "y_true", "y_pred", "score"]
BASELINE_DETECTORS = {"Rule_FATAL", "Baseline_ZScore"}


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
    assert set(actual["detector"]) == BASELINE_DETECTORS
    baseline_metrics = raw.loc[raw["detector"].isin(BASELINE_DETECTORS)]
    assert len(actual) == baseline_metrics["test_size"].sum()
    pd.testing.assert_frame_equal(pd.read_csv(reports_dir / "evaluation_folds_raw.csv"), raw)
    pd.testing.assert_frame_equal(
        pd.read_csv(reports_dir / "evaluation_mean_std_summary.csv"), summary
    )

    expected_groups = set()
    for scenario, config in evaluator.cfg["scenarios"].items():
        for fold in evaluator.splitter.split_by_scenario(df, scenario, config):
            detectors = evaluator.get_registered_detectors()
            for name in BASELINE_DETECTORS:
                expected_groups.add((scenario, fold.fold_idx, name))
                detector = detectors[name].fit(fold.train_df)
                expected_pred, expected_scores = detector.predict(fold.test_df)
                group = actual[
                    (actual["scenario"] == scenario) & (actual["fold"] == fold.fold_idx)
                    & (actual["detector"] == name)
                ]
                assert len(group) == len(fold.test_df)
                np.testing.assert_array_equal(group["ts_epoch"], fold.test_df["window_start"])
                np.testing.assert_array_equal(group["y_true"], fold.test_df["y"])
                np.testing.assert_array_equal(group["y_pred"], expected_pred)
                np.testing.assert_allclose(group["score"], expected_scores)

    assert set(zip(actual["scenario"], actual["fold"], actual["detector"])) == expected_groups
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

    assert len(second) == raw.loc[raw["detector"].isin(BASELINE_DETECTORS), "test_size"].sum()
    pd.testing.assert_frame_equal(second, first.assign(ts_epoch=first["ts_epoch"] + 1_000_000))


def test_save_predictions_false_skips_export(evaluation):
    evaluator, df, reports_dir = evaluation
    evaluator.cfg["output"]["save_predictions"] = False

    evaluator.run_all(df)

    assert not (reports_dir / "baseline_predictions.csv").exists()
    assert (reports_dir / "evaluation_folds_raw.csv").is_file()
    assert (reports_dir / "evaluation_mean_std_summary.csv").is_file()


@pytest.mark.parametrize("selected", [["Rule_FATAL"], ["Isolation_Forest"], []])
def test_prediction_detector_selection_only_filters_export(evaluation, selected):
    evaluator, df, reports_dir = evaluation
    evaluator.cfg["output"]["prediction_detectors"] = selected
    raw, _ = evaluator.run_all(df)
    actual = pd.read_csv(reports_dir / "baseline_predictions.csv")

    assert actual.columns.tolist() == PREDICTION_COLUMNS
    assert set(actual["detector"]) == set(selected)
    assert set(raw["detector"]) == BASELINE_DETECTORS | {"Isolation_Forest"}
    assert len(actual) == raw.loc[raw["detector"].isin(selected), "test_size"].sum()


@pytest.mark.parametrize("selected", ["Rule_FATAL", ["Unknown"]])
def test_invalid_prediction_detector_selection_fails(evaluation, selected):
    evaluator, df, reports_dir = evaluation
    evaluator.cfg["output"]["prediction_detectors"] = selected
    with pytest.raises(ValueError, match=r"output\.prediction_detectors"):
        evaluator.run_all(df)
    assert not (reports_dir / "baseline_predictions.csv").exists()


def test_default_detectors_get_fresh_state_and_clean_train_per_fold(evaluation, monkeypatch):
    evaluator, df, _ = evaluation
    create_detectors = evaluator.get_registered_detectors
    seen = []

    def track_detectors():
        detectors = create_detectors()
        detector = detectors["Baseline_ZScore"]
        fit = detector.fit
        train_inputs = []

        def track_fit(train):
            train_inputs.append(train)
            return fit(train)

        monkeypatch.setattr(detector, "fit", track_fit)
        seen.append((detector, train_inputs))
        return detectors

    monkeypatch.setattr(evaluator, "get_registered_detectors", track_detectors)
    evaluator.run_all(df)
    folds = [
        fold
        for name, config in evaluator.cfg["scenarios"].items()
        for fold in evaluator.splitter.split_by_scenario(df, name, config)
    ]
    assert len(seen) == len(folds)
    assert len({id(detector) for detector, _ in seen}) == len(folds)
    for (_, train_inputs), fold in zip(seen, folds):
        train, = train_inputs
        pd.testing.assert_frame_equal(train, fold.train_df)
        assert train["y"].eq(0).all()
        assert fold.test_df["window_start"].min() - train["window_start"].max() >= 3600
