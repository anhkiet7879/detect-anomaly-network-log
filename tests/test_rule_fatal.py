from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation.runner import EvaluationRunner, RuleFatalDetector
from src.features.window_features import build_set_a


@pytest.mark.parametrize("dtype", ["int8", "int64", "uint64"])
def test_predict_returns_labels_and_float_counts_in_row_order(dtype, monkeypatch):
    counts = pd.Series([0, 1, 3], index=[8, 2, 5], dtype=dtype)
    windows = pd.DataFrame({"f_sev_FATAL": counts})

    def reject_scalar_iteration(self):
        pytest.fail("Standard integer counts should use the vectorized path")

    monkeypatch.setattr(pd.Series, "__iter__", reject_scalar_iteration)

    labels, scores = RuleFatalDetector().predict(windows)

    np.testing.assert_array_equal(labels, [0, 1, 1])
    np.testing.assert_array_equal(scores, [0.0, 1.0, 3.0])
    assert np.issubdtype(labels.dtype, np.integer)
    assert scores.dtype == np.dtype(float)


def test_predict_accepts_empty_input_with_expected_column():
    labels, scores = RuleFatalDetector().predict(pd.DataFrame(columns=["f_sev_FATAL"]))

    assert labels.shape == scores.shape == (0,)
    assert np.issubdtype(labels.dtype, np.integer)
    assert scores.dtype == np.dtype(float)


def test_predict_uses_configured_fatal_column():
    windows = pd.DataFrame({"f_sev_FATAL": [0, 9], "n_fatal": [2, 0]})

    labels, scores = RuleFatalDetector(fatal_col="n_fatal").predict(windows)

    np.testing.assert_array_equal(labels, [1, 0])
    np.testing.assert_array_equal(scores, [2.0, 0.0])


def test_predict_rejects_missing_column():
    with pytest.raises(KeyError, match="f_sev_FATAL"):
        RuleFatalDetector().predict(pd.DataFrame({"f_sev_SEVERE": [2]}))


def test_predict_rejects_duplicate_count_columns():
    windows = pd.DataFrame([[0, 1]], columns=["f_sev_FATAL", "f_sev_FATAL"])

    with pytest.raises(ValueError, match="f_sev_FATAL"):
        RuleFatalDetector().predict(windows)


@pytest.mark.parametrize(
    "invalid_count",
    [np.nan, pd.NA, np.inf, -np.inf, -1, 0.5, "invalid", 1 + 2j, pd.Timestamp("2020-01-01")],
)
def test_predict_rejects_invalid_count(invalid_count):
    windows = pd.DataFrame({"f_sev_FATAL": [invalid_count]})

    with pytest.raises(ValueError, match="f_sev_FATAL"):
        RuleFatalDetector().predict(windows)


@pytest.mark.parametrize(
    "invalid_count",
    [
        np.complex128(1 + 2j),
        complex(1, 0),
        Decimal("1.00000000000000001"),
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-1"),
        "2",
        True,
    ],
)
def test_predict_validates_original_values_in_object_column(invalid_count):
    windows = pd.DataFrame({"f_sev_FATAL": pd.Series([0, invalid_count], dtype=object)})

    with pytest.raises(ValueError, match="finite non-negative integer counts"):
        RuleFatalDetector().predict(windows)


def test_predict_accepts_exact_integer_values_in_object_column():
    counts = pd.Series([0, Decimal("2.0"), 3, 4.0, np.int64(5), np.float32(6)], dtype=object)
    windows = pd.DataFrame({"f_sev_FATAL": counts})

    labels, scores = RuleFatalDetector().predict(windows)

    np.testing.assert_array_equal(labels, [0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(scores, [0.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert scores.dtype == np.dtype(float)


@pytest.mark.parametrize("count", [2**53 + 1, 10**400])
def test_predict_rejects_counts_not_exactly_representable_as_float(count):
    windows = pd.DataFrame({"f_sev_FATAL": pd.Series([count], dtype=object)})

    with pytest.raises(ValueError, match="not exactly representable as float"):
        RuleFatalDetector().predict(windows)


def test_predict_accepts_large_count_exactly_representable_as_float():
    count = 2**53 + 2
    windows = pd.DataFrame({"f_sev_FATAL": [count]})

    labels, scores = RuleFatalDetector().predict(windows)

    assert labels.tolist() == [1]
    assert int(scores[0]) == count


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_predict_checks_large_integer_counts_without_rounding(dtype):
    counts = pd.Series([2**53, 2**53 + 1], dtype=dtype)
    with pytest.raises(ValueError, match="not exactly representable as float"):
        RuleFatalDetector().predict(pd.DataFrame({"f_sev_FATAL": counts}))


def test_predict_rejects_missing_nullable_integer_count():
    counts = pd.Series([1, pd.NA], dtype="Int64")
    with pytest.raises(ValueError, match="finite non-negative integer counts"):
        RuleFatalDetector().predict(pd.DataFrame({"f_sev_FATAL": counts}))


def test_fit_is_stateless_and_does_not_modify_input():
    detector = RuleFatalDetector()
    train = pd.DataFrame({"f_sev_FATAL": [50, 0], "y": [1, 0]})
    windows = pd.DataFrame({"f_sev_FATAL": [0.0, 2.0], "y": [1, 0]})
    original_train = train.copy(deep=True)
    original_windows = windows.copy(deep=True)

    assert detector.fit(train) is detector
    first_labels, first_scores = detector.predict(windows)
    assert detector.fit(train.iloc[0:0]) is detector
    next_labels, next_scores = detector.predict(windows)

    np.testing.assert_array_equal(first_labels, [0, 1])
    np.testing.assert_array_equal(next_labels, first_labels)
    np.testing.assert_array_equal(next_scores, first_scores)
    next_scores[0] = 99.0
    pd.testing.assert_frame_equal(train, original_train)
    pd.testing.assert_frame_equal(windows, original_windows)


def test_window_counts_feed_registered_rule_in_scenario(tmp_path):
    logs = pd.DataFrame(
        {
            "ts_epoch": [0, 8100, 8101, 8400, 9000, 9001],
            "node": ["synthetic-node"] * 6,
            "component": ["synthetic-component"] * 6,
            "severity_raw": ["INFO", "FATAL", "FATAL", "FATAL", "SEVERE", "FAILURE"],
            "message": ["", "", "", "", "FATAL in message only", ""],
            "gt_is_alert": [0, 1, 0, 0, 1, 0],
            "gt_alert_type": ["NORMAL", "ALERT", "NORMAL", "NORMAL", "ALERT", "NORMAL"],
        }
    )
    windows = build_set_a(logs, win_sec=300, components=[])
    test_windows = windows.set_index("ts_epoch").loc[[8100, 8400, 8700, 9000]]
    assert test_windows["f_sev_FATAL"].tolist() == [2, 1, 0, 0]
    assert test_windows.loc[8700, "f_n_logs"] == 0

    config = {
        "data": {"timestamp_column": "ts_epoch", "target_column": "y"},
        "leakage_prevention": {"buffer_gap_mins": 60, "clean_train_only": True},
        "output": {"reports_dir": str(tmp_path / "reports")},
    }
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    runner = EvaluationRunner(str(config_file))
    detectors = runner.get_registered_detectors()

    results = runner.run_scenario(
        windows,
        "holdout_rule_fatal",
        {"strategy": "holdout", "train_ratio": 0.5},
        detectors={"Rule_FATAL": detectors["Rule_FATAL"]},
    )

    assert len(results) == 1
    assert results.loc[0, "detector"] == "Rule_FATAL"
    assert results.loc[0, "test_size"] == 4
    assert results.loc[0, ["TP", "FP", "FN", "TN"]].tolist() == [1, 1, 1, 1]
