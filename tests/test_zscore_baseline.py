import math

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation import runner
from src.evaluation.runner import EvaluationRunner, ZScoreBaselineDetector


def test_robust_scores_ignore_zero_training_windows_and_clip_below_median():
    detector = ZScoreBaselineDetector()
    train = pd.DataFrame({"f_n_logs": [0, 0, 1, 3, 7]})
    test = pd.DataFrame({"f_n_logs": [0, 1, 3, 7, 127]})

    assert detector.fit(train) is detector
    before = (detector.median_, detector.mad_)
    assert before == pytest.approx((math.log(4), math.log(2)))

    predictions, scores = detector.predict(test)

    np.testing.assert_allclose(scores, np.array([0, 0, 0, 1, 5]) / 1.4826)
    np.testing.assert_array_equal(predictions, [0, 0, 0, 0, 1])
    assert (detector.median_, detector.mad_) == before


def test_threshold_is_strict_and_feature_column_is_configurable():
    detector = ZScoreBaselineDetector(feature_col="count", threshold=0.0)
    detector.fit(pd.DataFrame({"count": [3, 3]}))

    predictions, scores = detector.predict(pd.DataFrame({"count": [3, 4]}))

    assert scores[0] == 0.0
    assert scores[1] > 0.0
    np.testing.assert_array_equal(predictions, [0, 1])


@pytest.mark.parametrize(
    ("counts", "expected_center", "expected_spread", "expected_scale"),
    [
        pytest.param([5, 5, 10], math.log(6), math.sqrt(2) * math.log(11 / 6) / 3,
                     1.4826 * math.sqrt(2) * math.log(11 / 6) / 3, id="std"),
        pytest.param([5, 5], math.log(6), 1.0, 1.4826, id="constant"),
        pytest.param([0, 0], 0.0, 1.0, 1.4826, id="all-zero"),
        pytest.param([1_000_000, 1_000_000, 1_000_001], math.log(1_000_001),
                     math.sqrt(2) * math.log1p(1 / 1_000_001) / 3, 1.0, id="tiny-std"),
    ],
)
def test_degenerate_training_spread(counts, expected_center, expected_spread, expected_scale):
    detector = ZScoreBaselineDetector()
    detector.fit(pd.DataFrame({"f_n_logs": [1, 3, 7]}))
    detector.fit(pd.DataFrame({"f_n_logs": counts}))

    predictions, scores = detector.predict(pd.DataFrame({"f_n_logs": [0, 2_000_000]}))

    assert detector.median_ == pytest.approx(expected_center)
    assert detector.mad_ == pytest.approx(expected_spread, rel=1e-6, abs=0)
    np.testing.assert_allclose(
        scores, [0.0, (math.log1p(2_000_000) - expected_center) / expected_scale]
    )
    np.testing.assert_array_equal(predictions, [0, int(scores[1] > 3.0)])


def test_runner_refits_registered_zscore_on_two_clean_rolling_folds(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_ROOT", str(tmp_path))
    config = tmp_path / "experiment.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {"timestamp_column": "ts_epoch"},
                "leakage_prevention": {"buffer_gap_mins": 60, "clean_train_only": True},
            }
        ),
        encoding="utf-8",
    )
    evaluation = EvaluationRunner(str(config))
    name = "Baseline_ZScore"
    detector = evaluation.get_registered_detectors()[name]
    windows = pd.DataFrame(
        [
            (0, 5, 0), (1, 5, 0), (2, 5, 0), (3, 1_000_000, 1),
            (30, 1_000_000, 1), (40, 1_000, 1), (41, 5, 0),
            (55, 100, 0), (56, 100, 0), (57, 100, 0), (58, 1_000_000, 1),
            (80, 1_000_000, 1), (90, 1_000, 0), (91, 100_000, 1), (100, 100, 0),
        ],
        columns=["ts_epoch", "f_n_logs", "y"],
    )
    windows["ts_epoch"] *= 300
    scenario = {
        "strategy": "rolling_cv",
        "unit": "time",
        "n_splits": 2,
        "train_window_ratio": 0.25,
        "test_window_ratio": 0.25,
    }

    results = evaluation.run_scenario(windows, "rolling_cv", scenario, {name: detector})

    columns = ["scenario", "fold", "detector", "train_size", "test_size", "gap_size",
               "TP", "FP", "FN", "TN", "F1"]
    assert list(results[columns].itertuples(index=False, name=None)) == [
        ("rolling_cv", 0, name, 3, 6, 1, 2, 0, 0, 4, 1.0),
        ("rolling_cv", 1, name, 3, 3, 1, 1, 0, 0, 2, 1.0),
    ]
    assert detector.median_ == pytest.approx(math.log1p(100))