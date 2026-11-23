"""
test_severity.py — Kiểm thử đơn vị cho [K6] SeverityMapper & config/severity_rules.yaml.
Chạy: python -m pytest tests/test_severity.py -v
"""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.alert.severity import (  # noqa: E402
    DEFAULT_SEVERITY_CONFIG_PATH,
    SeverityLevel,
    SeverityMapper,
    max_severity,
)


@pytest.fixture
def mapper():
    return SeverityMapper.from_yaml(DEFAULT_SEVERITY_CONFIG_PATH)


def test_severity_level_enum_values():
    assert int(SeverityLevel.LOW) == 1
    assert int(SeverityLevel.MEDIUM) == 2
    assert int(SeverityLevel.HIGH) == 3
    assert int(SeverityLevel.CRITICAL) == 4
    assert SeverityLevel.from_name("high") == SeverityLevel.HIGH
    assert SeverityLevel.from_score(4) == SeverityLevel.CRITICAL
    with pytest.raises(ValueError):
        SeverityLevel.from_name("UNKNOWN")
    with pytest.raises(ValueError):
        SeverityLevel.from_score(99)


def test_max_severity():
    assert max_severity(["LOW", "MEDIUM", "HIGH"]) == SeverityLevel.HIGH
    assert max_severity(["MEDIUM", "CRITICAL", "LOW"]) == SeverityLevel.CRITICAL
    assert max_severity([]) == SeverityLevel.LOW


def test_rule_fatal_mapping(mapper):
    # Rule FATAL đơn lẻ -> MEDIUM (K3/K6)
    d_single = mapper.map_severity("RULE_FATAL_LOG", score=1, threshold=0, features={"f_sev_FATAL": 1})
    assert d_single.severity == "MEDIUM"
    assert d_single.severity_score == 2

    # FATAL mật độ cao -> HIGH
    d_high = mapper.map_severity("RULE_FATAL_LOG", score=150, threshold=0, features={"f_sev_FATAL": 150})
    assert d_high.severity == "HIGH"
    assert d_high.severity_score == 3

    # Bão FATAL cực đoan -> CRITICAL
    d_crit = mapper.map_severity("RULE_FATAL_LOG", score=16384, threshold=0, features={"f_sev_FATAL": 16384})
    assert d_crit.severity == "CRITICAL"
    assert d_crit.severity_score == 4


def test_zscore_mapping_boundaries(mapper):
    # Z-score 2..3 -> LOW
    assert mapper.map_severity("ZSCORE_LOG_VOLUME", score=2.5, threshold=2.0).severity == "LOW"
    # Z-score > 3..5 -> MEDIUM
    assert mapper.map_severity("ZSCORE_LOG_VOLUME", score=3.8, threshold=3.0).severity == "MEDIUM"
    # Z-score > 5..<10 -> HIGH
    assert mapper.map_severity("ZSCORE_LOG_VOLUME", score=6.2, threshold=3.0).severity == "HIGH"
    # Z-score >= 10 -> CRITICAL
    assert mapper.map_severity("ZSCORE_LOG_VOLUME", score=11.0, threshold=3.0).severity == "CRITICAL"


def test_isolation_forest_mapping(mapper):
    # IF vượt ngưỡng phân vị 99% (không có FATAL) -> HIGH
    d_if = mapper.map_severity(
        "IF_MULTIVARIATE",
        score=0.75,
        threshold=0.634,
        features={"f_sev_FATAL": 0, "f_n_logs": 200},
    )
    assert d_if.severity == "HIGH"
    assert d_if.severity_score == 3


def test_new_template_mapping(mapper):
    # < 5 dòng -> LOW
    assert mapper.map_severity("NEW_TEMPLATE", score=3, features={"f_n_new_tpl_lines": 3}).severity == "LOW"
    # 5..50 dòng -> MEDIUM
    assert mapper.map_severity("NEW_TEMPLATE", score=20, features={"f_n_new_tpl_lines": 20}).severity == "MEDIUM"
    # > 50 dòng -> HIGH
    assert mapper.map_severity("NEW_TEMPLATE", score=51, features={"f_n_new_tpl_lines": 51}).severity == "HIGH"


def test_rare_template_burst_mapping(mapper):
    assert mapper.map_severity("RARE_TEMPLATE_BURST", score=12, features={"f_n_rare_tpl_lines": 12}).severity == "MEDIUM"
    assert mapper.map_severity("RARE_TEMPLATE_BURST", score=80, features={"f_n_rare_tpl_lines": 80}).severity == "HIGH"
    assert mapper.map_severity("RARE_TEMPLATE_BURST", score=600, features={"f_n_rare_tpl_lines": 600}).severity == "CRITICAL"


def test_escalation_fatal_plus_isolation_forest(mapper):
    # IF phát hiện trên cửa sổ có FATAL -> tự động leo thang CRITICAL
    d = mapper.map_severity(
        "IF_MULTIVARIATE",
        score=0.892,
        threshold=0.634,
        features={"f_sev_FATAL": 5, "f_n_logs": 500},
    )
    assert d.severity == "CRITICAL"
    assert d.matched_rule == "ESCALATION_FATAL_PLUS_IF"

    # Rule FATAL + đồng thời có IF_MULTIVARIATE -> CRITICAL
    d2 = mapper.map_severity(
        "RULE_FATAL_LOG",
        score=2,
        threshold=0,
        features={"f_sev_FATAL": 2},
        concurrent_alert_types=["IF_MULTIVARIATE"],
    )
    assert d2.severity == "CRITICAL"
    assert d2.matched_rule == "ESCALATION_FATAL_PLUS_IF"


def test_escalation_prolonged_incident_over_10_mins(mapper):
    # Sự cố kéo dài <= 600s chưa leo thang
    d_short = mapper.map_severity("ZSCORE_LOG_VOLUME", score=3.5, threshold=3.0, incident_duration_sec=600)
    assert d_short.severity == "MEDIUM"

    # Sự cố kéo dài > 600s (> 10 phút) -> CRITICAL
    d_long = mapper.map_severity("ZSCORE_LOG_VOLUME", score=3.5, threshold=3.0, incident_duration_sec=900)
    assert d_long.severity == "CRITICAL"
    assert d_long.matched_rule == "ESCALATION_PROLONGED_INCIDENT"

    # Hoặc liên tiếp >= 3 cửa sổ (15 phút) -> CRITICAL (Zero Data Leakage)
    d_windows = mapper.map_severity("ZSCORE_LOG_VOLUME", score=3.5, threshold=3.0, consecutive_windows=3)
    assert d_windows.severity == "CRITICAL"
    assert d_windows.matched_rule == "ESCALATION_PROLONGED_INCIDENT"


def test_escalation_three_concurrent_detectors(mapper):
    d = mapper.map_severity(
        "ZSCORE_LOG_VOLUME",
        score=3.2,
        threshold=3.0,
        concurrent_detectors=["Rule_FATAL", "Baseline_ZScore", "Isolation_Forest"],
    )
    assert d.severity == "CRITICAL"
    assert d.matched_rule == "ESCALATION_MULTI_DETECTOR"
