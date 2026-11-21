"""
test_dedup.py — Kiểm thử đơn vị & tích hợp cho [K7] AlertDeduplicator (Chống bão cảnh báo).
Chạy: python -m pytest tests/test_dedup.py -v
"""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.alert.alert_factory import AlertFactory, validate_alert  # noqa: E402
from src.alert.dedup import AlertDeduplicator  # noqa: E402


@pytest.fixture
def factory():
    return AlertFactory()


def _make_alert(
    factory: AlertFactory,
    ts_epoch: int,
    source: str = "bgl",
    detector: str = "Baseline_ZScore",
    alert_type: str = "ZSCORE_LOG_VOLUME",
    score: float = 3.5,
    threshold: float = 3.0,
    features: dict | None = None,
):
    return factory.create_alert(
        ts_epoch=ts_epoch,
        source=source,
        detector=detector,
        alert_type=alert_type,
        score=score,
        threshold=threshold,
        features=features or {"f_n_logs": 300, "f_sev_FATAL": 0},
    )


def test_exact_duplicate_dedup_key_merged(factory):
    """Cùng source, detector, alert_type và ts_epoch -> gộp thành 1, giữ score & severity cao hơn."""
    ts = 1118765100
    a1 = _make_alert(factory, ts_epoch=ts, score=3.4)  # MEDIUM
    a2 = _make_alert(factory, ts_epoch=ts, score=6.1)  # HIGH

    dedup = AlertDeduplicator(window_mins=15)
    result = dedup.deduplicate([a1, a2])

    assert len(result) == 1
    merged = result[0]
    assert merged.score == pytest.approx(6.1)
    assert merged.severity == "HIGH"
    assert merged.severity_score == 3
    assert merged.evidence["deduplication"]["occurrence_count"] == 1
    assert merged.evidence["deduplication"]["exact_duplicate_count"] == 1
    assert merged.evidence["deduplication"]["duration_sec"] == 0
    assert dedup.get_stats().exact_duplicates_merged == 1
    assert dedup.get_stats().suppression_rate == pytest.approx(0.5)

    ok, errs = validate_alert(merged)
    assert ok, errs


def test_sliding_window_merges_consecutive_storm_and_escalates(factory):
    """
    4 cửa sổ 5 phút liên tiếp (0m, 5m, 10m, 15m -> duration = 900s > 600s):
      - Gộp thành 1 sự cố duy nhất (giảm 75% bão cảnh báo)
      - Tự động leo thang từ MEDIUM lên CRITICAL theo luật sự cố kéo dài > 10 phút (K6)
    """
    base_ts = 1118765100
    alerts = [
        _make_alert(factory, ts_epoch=base_ts + i * 300, score=3.5 + i * 0.2)
        for i in range(4)
    ]

    dedup = AlertDeduplicator(window_mins=15)
    result = dedup.deduplicate(alerts)

    assert len(result) == 1
    inc = result[0]
    meta = inc.evidence["deduplication"]

    assert meta["occurrence_count"] == 4
    assert meta["first_ts_epoch"] == base_ts
    assert meta["last_ts_epoch"] == base_ts + 900
    assert meta["duration_sec"] == 900
    assert len(meta["merged_alert_ids"]) == 3
    assert len(meta["window_timestamps"]) == 4
    assert inc.score == pytest.approx(4.1)

    # Leo thang CRITICAL do kéo dài 900s > 600s
    assert inc.severity == "CRITICAL"
    assert inc.severity_score == 4
    assert inc.title.startswith("[CRITICAL]")
    assert inc.evidence["detector_detail"]["escalation_rule"] == "ESCALATION_PROLONGED_INCIDENT"

    stats = dedup.get_stats()
    assert stats.input_count == 4
    assert stats.output_count == 1
    assert stats.window_merged == 3
    assert stats.suppression_rate == pytest.approx(0.75)
    assert stats.escalated_to_critical == 1

    ok, errs = validate_alert(inc)
    assert ok, errs


def test_gap_exceeding_window_creates_two_separate_incidents(factory):
    """2 cảnh báo cách nhau 20 phút (> window_mins=15) phải tách thành 2 sự cố độc lập."""
    base_ts = 1118765100
    a1 = _make_alert(factory, ts_epoch=base_ts, score=3.5)
    a2 = _make_alert(factory, ts_epoch=base_ts + 300, score=3.8)        # +5 phút -> gộp với a1
    a3 = _make_alert(factory, ts_epoch=base_ts + 300 + 1200, score=4.0) # +20 phút từ a2 -> sự cố mới

    dedup = AlertDeduplicator(window_mins=15)
    result = dedup.deduplicate([a1, a2, a3])

    assert len(result) == 2
    assert result[0].evidence["deduplication"]["occurrence_count"] == 2
    assert result[0].evidence["deduplication"]["duration_sec"] == 300
    assert result[0].severity == "MEDIUM"  # 300s <= 600s chưa leo thang

    assert result[1].evidence["deduplication"]["occurrence_count"] == 1
    assert result[1].ts_epoch == base_ts + 1500


def test_different_sources_detectors_or_types_not_merged(factory):
    """Không gộp nhầm giữa các source, detector, hoặc alert_type khác nhau."""
    ts = 1118765100
    a_bgl_rule = _make_alert(
        factory, ts_epoch=ts, source="bgl", detector="Rule_FATAL", alert_type="RULE_FATAL_LOG", score=2, threshold=0
    )
    a_bgl_z = _make_alert(
        factory, ts_epoch=ts, source="bgl", detector="Baseline_ZScore", alert_type="ZSCORE_LOG_VOLUME", score=4.0
    )
    a_ssh_rule = _make_alert(
        factory, ts_epoch=ts, source="openssh", detector="Rule_FATAL", alert_type="RULE_FATAL_LOG", score=2, threshold=0
    )

    dedup = AlertDeduplicator(window_mins=15)
    result = dedup.deduplicate([a_bgl_rule, a_bgl_z, a_ssh_rule])

    assert len(result) == 3
    keys = {(r.source, r.detector, r.alert_type) for r in result}
    assert keys == {
        ("bgl", "Rule_FATAL", "RULE_FATAL_LOG"),
        ("bgl", "Baseline_ZScore", "ZSCORE_LOG_VOLUME"),
        ("openssh", "Rule_FATAL", "RULE_FATAL_LOG"),
    }


def test_out_of_order_input_handled_consistently(factory):
    """Batch deduplicate phải sắp xếp theo ts_epoch để xử lý đúng ngay cả khi input đảo thứ tự."""
    base_ts = 1118765100
    a1 = _make_alert(factory, ts_epoch=base_ts, score=3.2)
    a2 = _make_alert(factory, ts_epoch=base_ts + 300, score=4.5)
    a3 = _make_alert(factory, ts_epoch=base_ts + 600, score=3.9)

    dedup = AlertDeduplicator(window_mins=15)
    result = dedup.deduplicate([a3, a1, a2])

    assert len(result) == 1
    assert result[0].ts_epoch == base_ts
    assert result[0].evidence["deduplication"]["occurrence_count"] == 3
    assert result[0].evidence["deduplication"]["duration_sec"] == 600
    assert result[0].score == pytest.approx(4.5)


def test_cross_detector_escalation_fatal_plus_if_same_window(factory):
    """Rule_FATAL (MEDIUM) + Isolation_Forest (HIGH) cùng cửa sổ -> cả hai leo thang CRITICAL."""
    ts = 1118765100
    a_rule = _make_alert(
        factory,
        ts_epoch=ts,
        detector="Rule_FATAL",
        alert_type="RULE_FATAL_LOG",
        score=2.0,
        threshold=0.0,
        features={"f_n_logs": 200, "f_sev_FATAL": 2},
    )
    a_if = _make_alert(
        factory,
        ts_epoch=ts,
        detector="Isolation_Forest",
        alert_type="IF_MULTIVARIATE",
        score=0.78,
        threshold=0.63,
        features={"f_n_logs": 200, "f_sev_FATAL": 0},
    )

    assert a_rule.severity == "MEDIUM"
    assert a_if.severity == "HIGH"

    dedup = AlertDeduplicator(window_mins=15, enable_cross_detector_escalation=True)
    result = dedup.deduplicate([a_rule, a_if])

    assert len(result) == 2
    for rec in result:
        assert rec.severity == "CRITICAL"
        assert rec.severity_score == 4
        assert len(rec.evidence["context"]["corroborating_detectors"]) == 1


def test_consolidate_by_window_picks_highest_severity(factory):
    """Hợp nhất đa detector cùng 1 cửa sổ thành 1 đại diện duy nhất giữ mức cao nhất."""
    ts = 1118765100
    a_low = _make_alert(factory, ts_epoch=ts, detector="Baseline_ZScore", alert_type="ZSCORE_LOG_VOLUME", score=2.4, threshold=2.0)
    a_high = _make_alert(factory, ts_epoch=ts, detector="Isolation_Forest", alert_type="IF_MULTIVARIATE", score=0.81, threshold=0.63)

    dedup = AlertDeduplicator(window_mins=15)
    consolidated = dedup.consolidate_by_window([a_low, a_high])

    assert len(consolidated) == 1
    assert consolidated[0].detector == "Isolation_Forest"
    assert consolidated[0].severity == "HIGH"
    assert len(consolidated[0].evidence["context"]["consolidated_alerts"]) == 1


def test_streaming_process_one_and_reset(factory):
    """Kiểm thử chế độ streaming `process_one()` và `reset()`."""
    base_ts = 1118765100
    dedup = AlertDeduplicator(window_mins=15)

    act1, _ = dedup.process_one(_make_alert(factory, ts_epoch=base_ts, score=3.4))
    act2, _ = dedup.process_one(_make_alert(factory, ts_epoch=base_ts, score=3.9))
    act3, _ = dedup.process_one(_make_alert(factory, ts_epoch=base_ts + 300, score=4.2))
    act4, _ = dedup.process_one(_make_alert(factory, ts_epoch=base_ts + 1800, score=3.6))

    assert act1 == "NEW"
    assert act2 == "MERGED_EXACT"
    assert act3 == "MERGED_WINDOW"
    assert act4 == "NEW"
    assert len(dedup.flush()) == 2

    dedup.reset()
    assert dedup.flush() == []
    assert dedup.get_stats().input_count == 0


def test_empty_input_and_invalid_window_mins():
    dedup = AlertDeduplicator(window_mins=15)
    assert dedup.deduplicate([]) == []
    assert dedup.consolidate_by_window([]) == []
    assert dedup.get_stats().suppression_rate == 0.0

    with pytest.raises(ValueError):
        AlertDeduplicator(window_mins=-5)
