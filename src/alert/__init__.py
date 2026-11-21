"""
Package `src.alert` — Alert Engine cốt lõi của đề tài N3.10 (Tasks K5, K6, K7).
  - severity.py      (K6): Ánh xạ & leo thang mức độ nghiêm trọng (LOW..CRITICAL)
  - alert_factory.py (K5): Khởi tạo & chuẩn hóa Alert Schema 16 trường
  - dedup.py         (K7): Gộp cảnh báo lặp & chống bão cảnh báo (Sliding Window)
"""

from src.alert.alert_factory import (
    ALERT_SCHEMA_FIELDS,
    ALERT_TYPE_TO_PLAYBOOK,
    AlertFactory,
    AlertRecord,
    build_dedup_key,
    epoch_to_iso_utc,
    validate_alert,
)
from src.alert.dedup import AlertDeduplicator, DedupStats, build_incident_key
from src.alert.severity import (
    SEVERITY_SCORES,
    SeverityDecision,
    SeverityLevel,
    SeverityMapper,
    max_severity,
)

__all__ = [
    "ALERT_SCHEMA_FIELDS",
    "ALERT_TYPE_TO_PLAYBOOK",
    "AlertFactory",
    "AlertRecord",
    "build_dedup_key",
    "epoch_to_iso_utc",
    "validate_alert",
    "AlertDeduplicator",
    "DedupStats",
    "build_incident_key",
    "SEVERITY_SCORES",
    "SeverityDecision",
    "SeverityLevel",
    "SeverityMapper",
    "max_severity",
]
