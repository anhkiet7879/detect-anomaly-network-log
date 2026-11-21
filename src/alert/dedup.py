"""
dedup.py — [K7] Module Gộp cảnh báo lặp & Chống bão cảnh báo (Alert Deduplication)
Đề tài N3.10: Phát hiện bất thường trong log và dữ liệu vận hành mạng

Tuân thủ ADR-04 (docs/architecture.md), Alert Schema 16 trường (K3) và Severity Rules (K6):
  1. Exact Deduplication:
     Gộp các cảnh báo trùng chính xác `dedup_key = "{source}:{detector}:{alert_type}:{ts_epoch}"`
     (ví dụ khi nạp lại file log hoặc chạy lặp pipeline).
  2. Sliding Time-Window Deduplication (Chống bão cảnh báo):
     Gom các cảnh báo lặp liên tiếp có cùng khóa sự cố `incident_key = "{source}:{detector}:{alert_type}"`
     trong khoảng cách thời gian `<= window_mins` (mặc định 15 phút = 900 giây, khớp quy tắc gom
     episode của BGL).
     - Giữ bản ghi gốc hợp lệ 100% theo Alert Schema 16 trường.
     - Ghi chi tiết quá trình gộp vào `evidence["deduplication"]`:
       occurrence_count, first_ts_epoch, last_ts_epoch, duration_sec, peak_score,
       merged_alert_ids, window_timestamps.
     - Tự động nâng mức độ nghiêm trọng lên CRITICAL (4) nếu sự cố kéo dài > 10 phút
       (theo luật leo thang K6).
  3. Cross-Detector Escalation (Tương quan đa tầng trên cùng cửa sổ):
     Nếu trên cùng một cửa sổ `(source, ts_epoch)` vừa có log FATAL vừa có Isolation Forest
     (hoặc >= 3 detector độc lập cùng báo động), tự động leo thang lên CRITICAL (4) và ghi nhận
     `corroborating_detectors` vào `evidence["context"]`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.alert.alert_factory import AlertRecord, epoch_to_iso_utc
from src.alert.severity import SeverityLevel, SeverityMapper, max_severity


@dataclass
class DedupStats:
    """Thống kê hiệu quả chống bão cảnh báo của AlertDeduplicator."""

    input_count: int = 0
    output_count: int = 0
    exact_duplicates_merged: int = 0
    window_merged: int = 0
    escalated_to_critical: int = 0

    @property
    def total_suppressed(self) -> int:
        return self.exact_duplicates_merged + self.window_merged

    @property
    def suppression_rate(self) -> float:
        if self.input_count == 0:
            return 0.0
        return self.total_suppressed / self.input_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "exact_duplicates_merged": self.exact_duplicates_merged,
            "window_merged": self.window_merged,
            "total_suppressed": self.total_suppressed,
            "suppression_rate": round(self.suppression_rate, 4),
            "escalated_to_critical": self.escalated_to_critical,
        }


@dataclass
class _ActiveIncident:
    """Trạng thái nội bộ của một đợt sự cố đang mở trong sliding window."""

    record: AlertRecord
    first_ts_epoch: int
    last_ts_epoch: int
    occurrence_count: int = 1
    exact_dup_count: int = 0
    peak_score: float = 0.0
    merged_alert_ids: List[str] = field(default_factory=list)
    window_timestamps: List[int] = field(default_factory=list)
    seen_dedup_keys: set = field(default_factory=set)


def build_incident_key(
    record: AlertRecord,
    group_by: Sequence[str] = ("source", "detector", "alert_type"),
) -> str:
    """Tạo khóa gom sự cố theo các trường cấu hình (mặc định: source:detector:alert_type)."""
    parts = [str(getattr(record, attr, "")).strip() for attr in group_by]
    return ":".join(parts)


def _normalize_record(alert: AlertRecord | Mapping[str, Any]) -> AlertRecord:
    """Sao chép sâu thành AlertRecord độc lập để không làm biến đổi dữ liệu đầu vào."""
    if isinstance(alert, AlertRecord):
        return AlertRecord.from_dict(copy.deepcopy(alert.to_dict()))
    return AlertRecord.from_dict(copy.deepcopy(dict(alert)))


def _update_title_severity(title: str, old_sev: str, new_sev: str) -> str:
    """Cập nhật nhãn [SEVERITY] ở đầu tiêu đề khi cảnh báo được nâng cấp mức."""
    old_prefix = f"[{old_sev}]"
    new_prefix = f"[{new_sev}]"
    if title.startswith(old_prefix):
        return new_prefix + title[len(old_prefix) :]
    return title


class AlertDeduplicator:
    """
    Bộ gộp cảnh báo lặp & chống bão cảnh báo (ADR-04, Task K7).
    Hỗ trợ cả chế độ Batch (`deduplicate`) và Streaming (`process_one`).
    """

    def __init__(
        self,
        window_mins: int = 15,
        group_by: Sequence[str] = ("source", "detector", "alert_type"),
        severity_mapper: Optional[SeverityMapper] = None,
        enable_prolonged_escalation: bool = True,
        enable_cross_detector_escalation: bool = True,
    ):
        if window_mins < 0:
            raise ValueError("window_mins phải >= 0")
        self.window_mins = int(window_mins)
        self.window_sec = self.window_mins * 60
        self.group_by = tuple(group_by)
        self.severity_mapper = severity_mapper or SeverityMapper()
        self.enable_prolonged_escalation = bool(enable_prolonged_escalation)
        self.enable_cross_detector_escalation = bool(enable_cross_detector_escalation)

        self._active: Dict[str, _ActiveIncident] = {}
        self._closed: List[AlertRecord] = []
        self._stats = DedupStats()

    def reset(self) -> None:
        """Xóa sạch toàn bộ trạng thái nội bộ (phục vụ đánh giá độc lập từng fold)."""
        self._active.clear()
        self._closed.clear()
        self._stats = DedupStats()

    def get_stats(self) -> DedupStats:
        return self._stats

    def _sync_dedup_evidence(self, inc: _ActiveIncident) -> None:
        """Cập nhật khối `evidence['deduplication']` trên bản ghi đại diện."""
        duration_sec = max(0, inc.last_ts_epoch - inc.first_ts_epoch)
        dedup_meta = {
            "incident_key": build_incident_key(inc.record, self.group_by),
            "window_mins": self.window_mins,
            "occurrence_count": inc.occurrence_count,
            "exact_duplicate_count": inc.exact_dup_count,
            "first_ts_epoch": inc.first_ts_epoch,
            "first_ts_utc": epoch_to_iso_utc(inc.first_ts_epoch),
            "last_ts_epoch": inc.last_ts_epoch,
            "last_ts_utc": epoch_to_iso_utc(inc.last_ts_epoch),
            "duration_sec": duration_sec,
            "peak_score": float(inc.peak_score),
            "merged_alert_ids": list(inc.merged_alert_ids),
            "window_timestamps": sorted(inc.window_timestamps),
        }
        inc.record.evidence["deduplication"] = dedup_meta

    def _apply_prolonged_escalation(self, inc: _ActiveIncident) -> None:
        """Nếu sự cố kéo dài vượt ngưỡng (> 10 phút), leo thang lên CRITICAL (K6)."""
        if not self.enable_prolonged_escalation:
            return
        duration_sec = max(0, inc.last_ts_epoch - inc.first_ts_epoch)
        if duration_sec <= 0:
            return

        current_lvl = SeverityLevel.from_name(inc.record.severity)
        if current_lvl >= SeverityLevel.CRITICAL:
            return

        decision = self.severity_mapper.map_severity(
            alert_type=inc.record.alert_type,
            score=inc.peak_score,
            threshold=inc.record.threshold,
            features=inc.record.evidence.get("features", {}),
            incident_duration_sec=duration_sec,
        )
        if decision.severity_score > inc.record.severity_score:
            old_sev = inc.record.severity
            inc.record.severity = decision.severity
            inc.record.severity_score = decision.severity_score
            inc.record.title = _update_title_severity(inc.record.title, old_sev, decision.severity)
            detail = inc.record.evidence.setdefault("detector_detail", {})
            detail["escalated_from"] = old_sev
            detail["escalation_rule"] = decision.matched_rule
            detail["escalation_reason"] = decision.reason
            if decision.severity == "CRITICAL":
                self._stats.escalated_to_critical += 1

    def _merge_into_incident(
        self,
        inc: _ActiveIncident,
        incoming: AlertRecord,
        is_exact: bool,
    ) -> None:
        """Gộp một cảnh báo mới vào đợt sự cố đang mở."""
        if is_exact:
            inc.exact_dup_count += 1
            self._stats.exact_duplicates_merged += 1
        else:
            inc.occurrence_count += 1
            self._stats.window_merged += 1
            if incoming.ts_epoch not in inc.window_timestamps:
                inc.window_timestamps.append(incoming.ts_epoch)

        if incoming.alert_id != inc.record.alert_id and incoming.alert_id not in inc.merged_alert_ids:
            inc.merged_alert_ids.append(incoming.alert_id)

        inc.seen_dedup_keys.add(incoming.dedup_key)
        inc.first_ts_epoch = min(inc.first_ts_epoch, incoming.ts_epoch)
        inc.last_ts_epoch = max(inc.last_ts_epoch, incoming.ts_epoch)

        # Cập nhật điểm đỉnh (peak_score)
        if incoming.score > inc.peak_score:
            inc.peak_score = incoming.score
            inc.record.score = incoming.score

        # Giữ mức severity cao nhất giữa các lần xuất hiện
        best_lvl = max_severity([inc.record.severity, incoming.severity])
        if int(best_lvl.value) > inc.record.severity_score:
            old_sev = inc.record.severity
            inc.record.severity = best_lvl.name
            inc.record.severity_score = int(best_lvl.value)
            inc.record.title = _update_title_severity(inc.record.title, old_sev, best_lvl.name)

        # Gộp thông tin context nếu có (VD: corroborating_detectors)
        inc_ctx = inc.record.evidence.setdefault("context", {})
        in_ctx = incoming.evidence.get("context", {})
        if "corroborating_detectors" in in_ctx:
            merged_dets = sorted(
                set(inc_ctx.get("corroborating_detectors", []))
                | set(in_ctx.get("corroborating_detectors", []))
            )
            inc_ctx["corroborating_detectors"] = merged_dets

        self._apply_prolonged_escalation(inc)
        self._sync_dedup_evidence(inc)

    def process_one(self, alert: AlertRecord | Mapping[str, Any]) -> Tuple[str, AlertRecord]:
        """
        Xử lý từng cảnh báo theo chế độ Streaming (thời gian thực).
        Lưu ý: Luồng streaming giả định cảnh báo đến theo thứ tự thời gian tăng dần.

        Trả về: (action, alert_record)
          - "NEW": Mở một đợt sự cố mới.
          - "MERGED_EXACT": Gộp vì trùng chính xác dedup_key.
          - "MERGED_WINDOW": Gộp vì nằm trong cửa sổ trượt <= window_mins.
        """
        record = _normalize_record(alert)
        self._stats.input_count += 1
        key = build_incident_key(record, self.group_by)

        active = self._active.get(key)
        if active is not None:
            # Kiểm tra trùng chính xác dedup_key
            if record.dedup_key in active.seen_dedup_keys:
                self._merge_into_incident(active, record, is_exact=True)
                return "MERGED_EXACT", active.record

            # Kiểm tra khoảng cách thời gian so với mốc cuối của đợt sự cố đang mở
            gap_sec = record.ts_epoch - active.last_ts_epoch
            if 0 <= gap_sec <= self.window_sec:
                self._merge_into_incident(active, record, is_exact=False)
                return "MERGED_WINDOW", active.record

            # Đã quá cửa sổ trượt -> đóng đợt sự cố cũ, lưu vào _closed
            self._closed.append(active.record)

        # Khởi tạo đợt sự cố mới
        inc = _ActiveIncident(
            record=record,
            first_ts_epoch=record.ts_epoch,
            last_ts_epoch=record.ts_epoch,
            occurrence_count=1,
            exact_dup_count=0,
            peak_score=record.score,
            merged_alert_ids=[],
            window_timestamps=[record.ts_epoch],
            seen_dedup_keys={record.dedup_key},
        )
        self._sync_dedup_evidence(inc)
        self._active[key] = inc
        self._stats.output_count += 1
        return "NEW", inc.record

    def flush(self) -> List[AlertRecord]:
        """Lấy toàn bộ danh sách cảnh báo đã gộp (bao gồm cả đợt đã đóng và đang mở)."""
        result = list(self._closed) + [inc.record for inc in self._active.values()]
        result.sort(key=lambda r: (r.ts_epoch, -r.severity_score, r.detector, r.alert_type))
        return result

    def _apply_cross_detector_escalation(self, records: List[AlertRecord]) -> List[AlertRecord]:
        """
        Kiểm tra tương quan đa detector trên cùng cửa sổ `(source, ts_epoch)`:
        Nếu vừa có FATAL vừa có Isolation Forest (hoặc >= 3 detector độc lập),
        nâng mức lên CRITICAL theo quy tắc K3 & K6.
        """
        if not self.enable_cross_detector_escalation or len(records) < 2:
            return records

        by_window: Dict[Tuple[str, int], List[AlertRecord]] = {}
        for rec in records:
            by_window.setdefault((rec.source, rec.ts_epoch), []).append(rec)

        for (_src, _ts), group in by_window.items():
            if len(group) < 2:
                continue
            detectors = sorted({r.detector for r in group})
            alert_types = sorted({r.alert_type for r in group})

            for rec in group:
                ctx = rec.evidence.setdefault("context", {})
                ctx["corroborating_detectors"] = [d for d in detectors if d != rec.detector]
                ctx["corroborating_alert_types"] = [t for t in alert_types if t != rec.alert_type]

                if rec.severity_score >= int(SeverityLevel.CRITICAL):
                    continue

                decision = self.severity_mapper.map_severity(
                    alert_type=rec.alert_type,
                    score=rec.score,
                    threshold=rec.threshold,
                    features=rec.evidence.get("features", {}),
                    concurrent_alert_types=alert_types,
                    concurrent_detectors=detectors,
                )
                if decision.severity_score > rec.severity_score:
                    old_sev = rec.severity
                    rec.severity = decision.severity
                    rec.severity_score = decision.severity_score
                    rec.title = _update_title_severity(rec.title, old_sev, decision.severity)
                    detail = rec.evidence.setdefault("detector_detail", {})
                    detail["escalated_from"] = old_sev
                    detail["escalation_rule"] = decision.matched_rule
                    detail["escalation_reason"] = decision.reason
                    if decision.severity == "CRITICAL":
                        self._stats.escalated_to_critical += 1

        return records

    def deduplicate(
        self,
        alerts: Iterable[AlertRecord | Mapping[str, Any]],
        as_dict: bool = False,
    ) -> List[AlertRecord] | List[Dict[str, Any]]:
        """
        Xử lý gộp cảnh báo theo chế độ Batch (tự sắp xếp theo ts_epoch để xử lý chính xác
        cả khi đầu vào chưa theo thứ tự thời gian).
        """
        self.reset()
        normalized = [_normalize_record(a) for a in alerts]
        if not normalized:
            return []

        # Bước 1: Leo thang chéo giữa nhiều detector trên cùng cửa sổ (nếu bật)
        normalized = self._apply_cross_detector_escalation(normalized)

        # Bước 2: Sắp xếp ổn định theo thời gian tăng dần để gom sliding window chính xác
        normalized.sort(key=lambda r: (r.ts_epoch, r.source, r.detector, r.alert_type))

        # Bước 3: Chạy gom nhóm qua sliding window & exact dedup
        for rec in normalized:
            self.process_one(rec)

        final_records = self.flush()
        if as_dict:
            return [r.to_dict() for r in final_records]
        return final_records

    def consolidate_by_window(
        self,
        alerts: Iterable[AlertRecord | Mapping[str, Any]],
    ) -> List[AlertRecord]:
        """
        (Tùy chọn hiển thị Dashboard) Hợp nhất nhiều detector cùng kích hoạt trên cùng
        một cửa sổ `(source, ts_epoch)` thành 1 cảnh báo đại diện duy nhất có mức severity
        cao nhất theo lưu ý mục 4 của docs/alert_schema.md (K3).
        """
        normalized = [_normalize_record(a) for a in alerts]
        if not normalized:
            return []

        normalized = self._apply_cross_detector_escalation(normalized)

        by_window: Dict[Tuple[str, int], List[AlertRecord]] = {}
        for rec in normalized:
            by_window.setdefault((rec.source, rec.ts_epoch), []).append(rec)

        consolidated: List[AlertRecord] = []
        for (_src, _ts), group in sorted(by_window.items(), key=lambda x: x[0][1]):
            # Chọn cảnh báo có severity_score cao nhất, hòa thì chọn score cao hơn
            group.sort(key=lambda r: (-r.severity_score, -r.score, r.detector))
            primary = group[0]
            if len(group) > 1:
                primary.evidence.setdefault("context", {})["consolidated_alerts"] = [
                    {
                        "alert_id": r.alert_id,
                        "detector": r.detector,
                        "alert_type": r.alert_type,
                        "severity": r.severity,
                        "score": r.score,
                        "playbook_id": r.playbook_id,
                    }
                    for r in group[1:]
                ]
            consolidated.append(primary)

        return consolidated
