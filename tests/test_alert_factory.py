"""
test_alert_factory.py — Kiểm thử đơn vị cho [K5] AlertFactory & Alert Schema 16 trường (K3).
Chạy: python -m pytest tests/test_alert_factory.py -v
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.alert.alert_factory import (  # noqa: E402
    ALERT_SCHEMA_FIELDS,
    AlertFactory,
    AlertRecord,
    build_dedup_key,
    epoch_to_iso_utc,
    validate_alert,
)


@pytest.fixture
def factory():
    return AlertFactory()


@pytest.fixture
def sample_windows_df():
    return pd.DataFrame(
        [
            {
                "ts_epoch": 1118764800,
                "y": 0,
                "gt_alert_lines": 0,
                "f_n_logs": 120,
                "f_n_nodes": 12,
                "f_n_components": 3,
                "f_sev_INFO": 120,
                "f_sev_ERROR": 0,
                "f_sev_FATAL": 0,
                "f_n_templates": 4,
                "f_n_new_tpl_lines": 0,
                "f_n_rare_tpl_lines": 0,
            },
            {
                "ts_epoch": 1118765100,
                "y": 1,
                "gt_alert_lines": 16384,
                "f_n_logs": 16422,
                "f_n_nodes": 512,
                "f_n_components": 1,
                "f_sev_INFO": 38,
                "f_sev_ERROR": 0,
                "f_sev_FATAL": 16384,
                "f_n_templates": 2,
                "f_n_new_tpl_lines": 0,
                "f_n_rare_tpl_lines": 38,
            },
            {
                "ts_epoch": 1118765400,
                "y": 1,
                "gt_alert_lines": 50,
                "f_n_logs": 500,
                "f_n_nodes": 25,
                "f_n_components": 2,
                "f_sev_INFO": 450,
                "f_sev_ERROR": 50,
                "f_sev_FATAL": 0,
                "f_n_templates": 8,
                "f_n_new_tpl_lines": 60,
                "f_n_rare_tpl_lines": 12,
            },
        ]
    )


def test_create_alert_has_exact_16_schema_fields(factory):
    alert = factory.create_alert(
        ts_epoch=1118765100,
        source="bgl",
        detector="Isolation_Forest",
        alert_type="IF_MULTIVARIATE",
        score=0.892,
        threshold=0.634,
        features={"f_n_logs": 16422, "f_n_nodes": 512, "f_sev_FATAL": 16384},
        detector_detail={
            "contamination": 0.01,
            "top_features": [{"name": "f_sev_FATAL", "value": 16384, "contribution": 0.72}],
        },
        context={"top_components": ["KERNEL"]},
        total_alerts_gt=16384,
    )

    data = alert.to_dict()
    assert tuple(data.keys()) == ALERT_SCHEMA_FIELDS
    assert len(data) == 16

    is_valid, errors = validate_alert(data)
    assert is_valid, f"Alert không hợp lệ: {errors}"

    # Kiểm tra các giá trị cốt lõi khớp K3
    assert data["ts_utc"] == "2005-06-14T16:05:00Z"
    assert data["severity"] == "CRITICAL"  # IF + FATAL -> leo thang CRITICAL
    assert data["severity_score"] == 4
    assert data["playbook_id"] == "PB-004"
    assert data["dedup_key"] == "bgl:Isolation_Forest:IF_MULTIVARIATE:1118765100"

    # Kiểm tra JSON & DB serialization
    parsed = json.loads(alert.to_json())
    assert parsed["alert_id"] == data["alert_id"]

    db_row = alert.to_db_row()
    assert db_row["status"] == "OPEN"
    assert isinstance(db_row["evidence"], str)
    assert json.loads(db_row["evidence"])["window"]["total_logs"] == 16422


def test_playbook_auto_mapping_for_all_detectors(factory):
    cases = [
        ("RULE_FATAL_LOG", "PB-001"),
        ("RULE_ERROR_SPIKE", "PB-001"),
        ("ZSCORE_LOG_VOLUME", "PB-002"),
        ("ZSCORE_SEVERITY_SPIKE", "PB-002"),
        ("NEW_TEMPLATE", "PB-003"),
        ("RARE_TEMPLATE_BURST", "PB-003"),
        ("IF_MULTIVARIATE", "PB-004"),
        ("IF_TEMPLATE_DRIFT", "PB-004"),
    ]
    for atype, expected_pb in cases:
        a = factory.create_alert(
            ts_epoch=1118765100,
            source="bgl",
            detector="TestDetector",
            alert_type=atype,
            score=4.0,
            threshold=3.0,
        )
        assert a.playbook_id == expected_pb
        ok, errs = validate_alert(a)
        assert ok, errs


def test_from_detector_predictions_batch_conversion(factory, sample_windows_df):
    y_pred = np.array([0, 1, 1])
    scores = np.array([0.5, 6.4, 3.8])

    alerts = factory.from_detector_predictions(
        windows_df=sample_windows_df,
        y_pred=y_pred,
        scores=scores,
        detector="Baseline_ZScore",
        alert_type="ZSCORE_LOG_VOLUME",
        threshold=3.0,
        source="bgl",
    )

    assert len(alerts) == 2
    assert alerts[0].ts_epoch == 1118765100
    assert alerts[0].severity == "HIGH"  # Z=6.4 > 5
    assert alerts[0].evidence["context"]["prev_window_alert"] is False
    assert alerts[0].evidence["context"]["next_window_alert"] is True
    assert alerts[0].evidence["window"]["total_alerts_gt"] == 16384

    assert alerts[1].ts_epoch == 1118765400
    assert alerts[1].severity == "MEDIUM"  # Z=3.8 in (3, 5]
    assert alerts[1].evidence["context"]["prev_window_alert"] is True
    assert alerts[1].evidence["context"]["next_window_alert"] is False

    for a in alerts:
        ok, errs = validate_alert(a)
        assert ok, errs

    df_out = factory.to_dataframe(alerts)
    assert list(df_out.columns) == list(ALERT_SCHEMA_FIELDS)
    assert len(df_out) == 2


def test_from_detector_predictions_length_mismatch_raises(factory, sample_windows_df):
    with pytest.raises(ValueError):
        factory.from_detector_predictions(
            windows_df=sample_windows_df,
            y_pred=[0, 1],
            scores=[0.1, 0.9],
            detector="Rule_FATAL",
            alert_type="RULE_FATAL_LOG",
            threshold=0.0,
        )


def test_validate_alert_detects_schema_violations(factory):
    valid = factory.create_alert(
        ts_epoch=1118765100,
        source="bgl",
        detector="Rule_FATAL",
        alert_type="RULE_FATAL_LOG",
        score=1.0,
        threshold=0.0,
    ).to_dict()

    # Mismatch severity vs severity_score
    bad1 = dict(valid, severity="CRITICAL", severity_score=1)
    ok1, errs1 = validate_alert(bad1)
    assert not ok1
    assert any("severity_score" in e for e in errs1)

    # Sai dedup_key
    bad2 = dict(valid, dedup_key="wrong:key")
    ok2, errs2 = validate_alert(bad2)
    assert not ok2
    assert any("dedup_key" in e for e in errs2)

    # Thiếu trường bắt buộc
    bad3 = dict(valid)
    bad3.pop("evidence")
    ok3, errs3 = validate_alert(bad3)
    assert not ok3
    assert any("evidence" in e for e in errs3)


def test_helpers_epoch_and_dedup_key():
    assert epoch_to_iso_utc(1117869872) == "2005-06-04T07:24:32Z"
    assert (
        build_dedup_key("bgl", "Rule_FATAL", "RULE_FATAL_LOG", 1118765100)
        == "bgl:Rule_FATAL:RULE_FATAL_LOG:1118765100"
    )
    rec = AlertRecord.from_dict(
        AlertFactory()
        .create_alert(
            ts_epoch=1118765100,
            source="openssh",
            detector="Rule_SSH",
            alert_type="RULE_ERROR_SPIKE",
            score=50.0,
            threshold=20.0,
        )
        .to_dict()
    )
    assert rec.source == "openssh"
    assert rec.severity == "HIGH"
