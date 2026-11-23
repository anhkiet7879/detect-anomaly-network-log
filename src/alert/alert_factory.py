"""
alert_factory.py — [K5] Bộ khởi tạo & chuẩn hóa cảnh báo (Alert Factory)
Đề tài N3.10: Phát hiện bất thường trong log và dữ liệu vận hành mạng

Chức năng chính:
  1. Chuyển đổi đầu ra thô của cả 3 tầng detector (Rule, Baseline Z-score, Isolation Forest)
     thành bản ghi Alert chuẩn 16 trường theo docs/alert_schema.md (K3).
  2. Tích hợp SeverityMapper (K6) để tự động tính severity (LOW..CRITICAL) và severity_score (1..4).
  3. Đóng gói khối Evidence JSON chuẩn 4 phần: window, features, detector_detail, context.
  4. Tự động gắn playbook_id (PB-001 .. PB-004) và dedup_key phục vụ module Deduplication (K7).
  5. Cung cấp hàm validate_alert() kiểm tra hợp đồng giao tiếp Alert Schema (Interface Contract 5).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.alert.severity import SeverityDecision, SeverityLevel, SeverityMapper

# Danh sách 16 trường bắt buộc khớp 100% docs/alert_schema.md (K3)
ALERT_SCHEMA_FIELDS: Tuple[str, ...] = (
    "alert_id",
    "ts_epoch",
    "ts_utc",
    "source",
    "detector",
    "alert_type",
    "severity",
    "severity_score",
    "score",
    "threshold",
    "title",
    "description",
    "evidence",
    "playbook_id",
    "dedup_key",
    "created_at",
)

# Bảng ánh xạ mã loại cảnh báo -> Playbook (K3 & K10)
ALERT_TYPE_TO_PLAYBOOK: Dict[str, str] = {
    "RULE_FATAL_LOG": "PB-001",
    "RULE_ERROR_SPIKE": "PB-001",
    "ZSCORE_LOG_VOLUME": "PB-002",
    "ZSCORE_SEVERITY_SPIKE": "PB-002",
    "NEW_TEMPLATE": "PB-003",
    "RARE_TEMPLATE_BURST": "PB-003",
    "IF_MULTIVARIATE": "PB-004",
    "IF_TEMPLATE_DRIFT": "PB-004",
}

# Tên mô tả ngắn theo loại cảnh báo để sinh title tự động (~80 ký tự)
ALERT_TYPE_LABELS: Dict[str, str] = {
    "RULE_FATAL_LOG": "Phát hiện log FATAL",
    "RULE_ERROR_SPIKE": "Tăng đột biến log ERROR",
    "ZSCORE_LOG_VOLUME": "Đột biến lưu lượng log (Z-score)",
    "ZSCORE_SEVERITY_SPIKE": "Đột biến lỗi nghiêm trọng (Z-score)",
    "NEW_TEMPLATE": "Xuất hiện mẫu log mới (Drain3)",
    "RARE_TEMPLATE_BURST": "Tăng đột biến mẫu log hiếm",
    "IF_MULTIVARIATE": "Bất thường đa chiều (Isolation Forest)",
    "IF_TEMPLATE_DRIFT": "Dịch chuyển phân bố mẫu log (IF)",
}

# Các đặc trưng cửa sổ cốt lõi được trích vào evidence["features"]
SNAPSHOT_FEATURE_KEYS: Tuple[str, ...] = (
    "f_n_logs",
    "f_n_nodes",
    "f_n_components",
    "f_sev_INFO",
    "f_sev_WARNING",
    "f_sev_SEVERE",
    "f_sev_ERROR",
    "f_sev_FATAL",
    "f_sev_FAILURE",
    "f_n_templates",
    "f_n_new_tpl_lines",
    "f_n_rare_tpl_lines",
)


def epoch_to_iso_utc(ts_epoch: int | float) -> str:
    """Chuyển đổi epoch giây sang chuỗi ISO 8601 UTC có hậu tố Z."""
    return datetime.fromtimestamp(int(ts_epoch), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_iso_utc() -> str:
    """Thời điểm hiện tại theo chuẩn ISO 8601 UTC có hậu tố Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_dedup_key(
    source: str, detector: str, alert_type: str, ts_epoch: Optional[int] = None
) -> str:
    """
    Sinh khóa chống trùng lặp theo đúng ADR-04 & K3 mới: {source}:{detector}:{alert_type}.
    Tham số ts_epoch giữ lại để tương thích ngược nếu truyền vào nhưng không đưa vào key.
    """
    return f"{source}:{detector}:{alert_type}"


def _to_native(value: Any) -> Any:
    """Chuyển kiểu numpy/pandas sang kiểu Python thuần để serialize JSON an toàn."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    return value


@dataclass
class AlertRecord:
    """Bản ghi cảnh báo chuẩn 16 trường theo docs/alert_schema.md (K3)."""

    alert_id: str
    ts_epoch: int
    ts_utc: str
    source: str
    detector: str
    alert_type: str
    severity: str
    severity_score: int
    score: float
    threshold: float
    title: str
    description: str
    evidence: Dict[str, Any]
    playbook_id: str
    dedup_key: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Xuất đúng dictionary 16 trường chuẩn K3 (JSON-serializable)."""
        return {
            "alert_id": self.alert_id,
            "ts_epoch": int(self.ts_epoch),
            "ts_utc": self.ts_utc,
            "source": self.source,
            "detector": self.detector,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "severity_score": int(self.severity_score),
            "score": float(self.score),
            "threshold": float(self.threshold),
            "title": self.title,
            "description": self.description,
            "evidence": _to_native(self.evidence),
            "playbook_id": self.playbook_id,
            "dedup_key": self.dedup_key,
            "created_at": self.created_at,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Xuất chuỗi JSON UTF-8 chuẩn."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_db_row(self, status: str = "OPEN") -> Dict[str, Any]:
        """
        Xuất bản ghi sẵn sàng nạp vào bảng PostgreSQL `alerts` (Phước P5).
        Trường `evidence` được chuyển thành chuỗi JSON để ghi vào cột JSONB.
        """
        row = self.to_dict()
        row["evidence"] = json.dumps(row["evidence"], ensure_ascii=False)
        row["status"] = status
        return row

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AlertRecord":
        return cls(
            alert_id=str(data["alert_id"]),
            ts_epoch=int(data["ts_epoch"]),
            ts_utc=str(data["ts_utc"]),
            source=str(data["source"]),
            detector=str(data["detector"]),
            alert_type=str(data["alert_type"]),
            severity=str(data["severity"]),
            severity_score=int(data["severity_score"]),
            score=float(data["score"]),
            threshold=float(data["threshold"]),
            title=str(data["title"]),
            description=str(data["description"]),
            evidence=dict(data["evidence"]) if isinstance(data["evidence"], Mapping) else {},
            playbook_id=str(data.get("playbook_id", "")),
            dedup_key=str(data["dedup_key"]),
            created_at=str(data["created_at"]),
        )


def validate_alert(alert: AlertRecord | Mapping[str, Any]) -> Tuple[bool, List[str]]:
    """
    Kiểm tra tính hợp lệ của một Alert theo đúng hợp đồng 16 trường (K3).
    Trả về: (is_valid: bool, errors: List[str]).
    """
    data = alert.to_dict() if isinstance(alert, AlertRecord) else dict(alert)
    errors: List[str] = []

    # 1. Kiểm tra đủ 16 trường bắt buộc
    for field in ALERT_SCHEMA_FIELDS:
        if field not in data:
            errors.append(f"Thiếu trường bắt buộc: '{field}'")

    if errors:
        return False, errors

    # 2. Kiểm tra kiểu và giá trị
    if not isinstance(data["alert_id"], str) or not data["alert_id"].strip():
        errors.append("'alert_id' phải là chuỗi không rỗng")

    if not isinstance(data["ts_epoch"], int) or isinstance(data["ts_epoch"], bool):
        errors.append("'ts_epoch' phải là số nguyên (int64)")

    if not isinstance(data["ts_utc"], str) or not data["ts_utc"].endswith("Z"):
        errors.append("'ts_utc' phải là chuỗi ISO 8601 UTC kết thúc bằng 'Z'")

    if not isinstance(data["created_at"], str) or not data["created_at"].endswith("Z"):
        errors.append("'created_at' phải là chuỗi ISO 8601 UTC kết thúc bằng 'Z'")

    if not isinstance(data["source"], str) or not data["source"].strip():
        errors.append("'source' phải là chuỗi không rỗng")

    if not isinstance(data["detector"], str) or not data["detector"].strip():
        errors.append("'detector' phải là chuỗi không rỗng")

    if not isinstance(data["alert_type"], str) or not data["alert_type"].strip():
        errors.append("'alert_type' phải là chuỗi không rỗng")

    try:
        lvl = SeverityLevel.from_name(data["severity"])
        if int(data["severity_score"]) != int(lvl.value):
            errors.append(
                f"'severity_score' ({data['severity_score']}) không khớp với 'severity' ({data['severity']}={lvl.value})"
            )
    except ValueError as exc:
        errors.append(str(exc))

    if not isinstance(data["score"], (int, float)) or isinstance(data["score"], bool):
        errors.append("'score' phải là số thực")
    if not isinstance(data["threshold"], (int, float)) or isinstance(data["threshold"], bool):
        errors.append("'threshold' phải là số thực")

    if not isinstance(data["title"], str) or not data["title"].strip():
        errors.append("'title' phải là chuỗi không rỗng")

    if not isinstance(data["description"], str) or not data["description"].strip():
        errors.append("'description' phải là chuỗi không rỗng")

    # 3. Kiểm tra khối evidence bắt buộc (window, features, detector_detail)
    ev = data["evidence"]
    if not isinstance(ev, dict):
        errors.append("'evidence' phải là object/dict JSON")
    else:
        for required_block in ("window", "features", "detector_detail"):
            if required_block not in ev or not isinstance(ev[required_block], dict):
                errors.append(f"'evidence' thiếu khối bắt buộc '{required_block}'")

    # 4. Kiểm tra cấu trúc dedup_key theo chuẩn ADR-04 (source:detector:alert_type)
    expected_dedup = build_dedup_key(
        data["source"], data["detector"], data["alert_type"]
    )
    if data["dedup_key"] != expected_dedup:
        errors.append(
            f"'dedup_key' ('{data['dedup_key']}') không khớp chuẩn '{expected_dedup}'"
        )

    return len(errors) == 0, errors


class AlertFactory:
    """
    Factory chuyển đổi đầu ra thô từ mọi detector (Rule, Z-score, Isolation Forest)
    thành AlertRecord chuẩn 16 trường với đầy đủ Evidence, Severity và Playbook ID.
    """

    def __init__(
        self,
        severity_mapper: Optional[SeverityMapper] = None,
        playbook_map: Optional[Mapping[str, str]] = None,
        window_duration_sec: int = 300,
    ):
        self.severity_mapper = severity_mapper or SeverityMapper()
        self.playbook_map: Dict[str, str] = dict(ALERT_TYPE_TO_PLAYBOOK)
        if playbook_map:
            self.playbook_map.update(playbook_map)
        self.window_duration_sec = int(window_duration_sec)

    def resolve_playbook_id(self, alert_type: str, override: Optional[str] = None) -> str:
        if override is not None:
            return str(override)
        return self.playbook_map.get(str(alert_type).strip().upper(), "")

    @staticmethod
    def extract_feature_snapshot(row: Mapping[str, Any]) -> Dict[str, Any]:
        """Trích xuất các trường đặc trưng chính của cửa sổ từ row/dict."""
        snapshot: Dict[str, Any] = {}
        for key in SNAPSHOT_FEATURE_KEYS:
            if key in row and row[key] is not None:
                snapshot[key] = _to_native(row[key])
        # Nếu row có các trường f_ khác do người dùng truyền thêm
        if not snapshot:
            for key, val in row.items():
                if str(key).startswith("f_") and val is not None:
                    snapshot[str(key)] = _to_native(val)
        return snapshot

    def _build_title(
        self,
        severity: str,
        source: str,
        ts_utc: str,
        alert_type: str,
        score: float,
        threshold: float,
    ) -> str:
        hhmm = ts_utc[11:16] if len(ts_utc) >= 16 else ts_utc
        label = ALERT_TYPE_LABELS.get(alert_type, alert_type)
        src_upper = source.upper()
        title = f"[{severity}] {src_upper} {hhmm} — {label} ({score:.2f} > {threshold:.2f})"
        return title[:120]

    def _build_description(
        self,
        detector: str,
        source: str,
        ts_utc: str,
        alert_type: str,
        score: float,
        threshold: float,
        features: Mapping[str, Any],
        severity_decision: SeverityDecision,
    ) -> str:
        win_mins = max(1, self.window_duration_sec // 60)
        parts: List[str] = [
            f"Bộ phát hiện {detector} ({alert_type}) ghi nhận bất thường trên nguồn {source.upper()} "
            f"tại cửa sổ {win_mins} phút bắt đầu lúc {ts_utc}."
        ]
        n_logs = features.get("f_n_logs")
        n_fatal = features.get("f_sev_FATAL")
        n_nodes = features.get("f_n_nodes")
        n_new_tpl = features.get("f_n_new_tpl_lines")

        stats: List[str] = []
        if n_logs is not None:
            stats.append(f"tổng {int(n_logs):,} dòng log")
        if n_fatal is not None and int(n_fatal) > 0:
            stats.append(f"{int(n_fatal):,} dòng FATAL")
        if n_nodes is not None and int(n_nodes) > 0:
            stats.append(f"{int(n_nodes):,} node")
        if n_new_tpl is not None and int(n_new_tpl) > 0:
            stats.append(f"{int(n_new_tpl):,} dòng mẫu mới")

        if stats:
            parts.append("Bối cảnh cửa sổ: " + ", ".join(stats) + ".")

        parts.append(
            f"Điểm bất thường {score:.4f} vượt ngưỡng {threshold:.4f}. "
            f"Đánh giá mức {severity_decision.severity}: {severity_decision.reason}."
        )
        return " ".join(parts)

    def create_alert(
        self,
        ts_epoch: int,
        source: str,
        detector: str,
        alert_type: str,
        score: float,
        threshold: float,
        features: Optional[Mapping[str, Any]] = None,
        detector_detail: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
        total_alerts_gt: Optional[int] = None,
        concurrent_alert_types: Optional[Sequence[str]] = None,
        concurrent_detectors: Optional[Sequence[str]] = None,
        incident_duration_sec: float = 0.0,
        severity_override: Optional[str] = None,
        playbook_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        alert_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> AlertRecord:
        """
        Tạo một bản ghi AlertRecord chuẩn 16 trường từ kết quả phát hiện của 1 cửa sổ.
        """
        ts_int = int(ts_epoch)
        ts_utc = epoch_to_iso_utc(ts_int)
        src_clean = str(source).strip().lower()
        det_clean = str(detector).strip()
        atype_clean = str(alert_type).strip().upper()
        score_f = float(score)
        threshold_f = float(threshold)
        feat_snapshot = _to_native(dict(features or {}))

        # 1. Tính Severity theo quy tắc K6
        if severity_override is not None:
            lvl = SeverityLevel.from_name(severity_override)
            sev_decision = SeverityDecision(
                severity=lvl.name,
                severity_score=int(lvl.value),
                matched_rule="MANUAL_OVERRIDE",
                reason=f"Chỉ định thủ công mức {lvl.name}",
            )
        else:
            sev_decision = self.severity_mapper.map_severity(
                alert_type=atype_clean,
                score=score_f,
                threshold=threshold_f,
                features=feat_snapshot,
                concurrent_alert_types=concurrent_alert_types,
                concurrent_detectors=concurrent_detectors,
                incident_duration_sec=incident_duration_sec,
            )

        # 2. Gắn Playbook ID (K3 & K10)
        pb_id = self.resolve_playbook_id(atype_clean, override=playbook_id)

        # 3. Sinh dedup_key theo chuẩn ADR-04 (source:detector:alert_type)
        dedup_key = build_dedup_key(src_clean, det_clean, atype_clean)

        # 4. Đóng gói khối Evidence JSON chuẩn 4 phần (K3 mục 5)
        window_block: Dict[str, Any] = {
            "ts_epoch": ts_int,
            "ts_utc": ts_utc,
            "duration_sec": self.window_duration_sec,
            "total_logs": int(feat_snapshot.get("f_n_logs", 0)),
        }
        if total_alerts_gt is not None:
            window_block["total_alerts_gt"] = int(total_alerts_gt)

        detail_block: Dict[str, Any] = {
            "method": det_clean.lower(),
            "score": score_f,
            "threshold": threshold_f,
            "severity_rule": sev_decision.matched_rule,
        }
        if detector_detail:
            detail_block.update(_to_native(dict(detector_detail)))

        evidence: Dict[str, Any] = {
            "window": window_block,
            "features": feat_snapshot,
            "detector_detail": detail_block,
            "context": _to_native(dict(context)) if context else {},
        }

        # 5. Sinh title & description
        final_title = title or self._build_title(
            sev_decision.severity, src_clean, ts_utc, atype_clean, score_f, threshold_f
        )
        final_desc = description or self._build_description(
            det_clean, src_clean, ts_utc, atype_clean, score_f, threshold_f, feat_snapshot, sev_decision
        )

        return AlertRecord(
            alert_id=alert_id or str(uuid.uuid4()),
            ts_epoch=ts_int,
            ts_utc=ts_utc,
            source=src_clean,
            detector=det_clean,
            alert_type=atype_clean,
            severity=sev_decision.severity,
            severity_score=sev_decision.severity_score,
            score=score_f,
            threshold=threshold_f,
            title=final_title,
            description=final_desc,
            evidence=evidence,
            playbook_id=pb_id,
            dedup_key=dedup_key,
            created_at=created_at or now_iso_utc(),
        )

    def from_detector_predictions(
        self,
        windows_df: pd.DataFrame,
        y_pred: Sequence[int] | np.ndarray,
        scores: Sequence[float] | np.ndarray,
        detector: str,
        alert_type: str,
        threshold: float,
        source: str = "bgl",
        timestamp_col: str = "ts_epoch",
        extra_details_per_row: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
        extra_context_per_row: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
    ) -> List[AlertRecord]:
        """
        Chuyển đổi trực tiếp kết quả dự đoán hàng loạt (y_pred, scores) từ bất kỳ
        BaseDetector nào (Rule, Z-score, Isolation Forest) kết hợp với windows_df
        (từ window_features.py) thành danh sách AlertRecord chuẩn 16 trường.
        """
        n_rows = len(windows_df)
        y_arr = np.asarray(y_pred)
        s_arr = np.asarray(scores, dtype=float)

        if len(y_arr) != n_rows or len(s_arr) != n_rows:
            raise ValueError(
                f"Độ dài không khớp: windows_df={n_rows}, y_pred={len(y_arr)}, scores={len(s_arr)}"
            )
        if timestamp_col not in windows_df.columns:
            raise KeyError(f"Thiếu cột thời gian '{timestamp_col}' trong windows_df")

        alerts: List[AlertRecord] = []
        # Reset index vị trí để duyệt nhanh, an toàn với DataFrame đã lọc
        for pos in range(n_rows):
            if int(y_arr[pos]) != 1:
                continue

            row = windows_df.iloc[pos]
            ts_val = int(row[timestamp_col])
            feat_snapshot = self.extract_feature_snapshot(row.to_dict())

            gt_alerts = None
            if "gt_alert_lines" in windows_df.columns and pd.notna(row["gt_alert_lines"]):
                gt_alerts = int(row["gt_alert_lines"])

            # Ghi nhận bối cảnh cửa sổ liền kề (prev/next window alert) theo K3 mục 5
            ctx: Dict[str, Any] = {
                "prev_window_alert": bool(pos > 0 and int(y_arr[pos - 1]) == 1),
                "next_window_alert": bool(pos + 1 < n_rows and int(y_arr[pos + 1]) == 1),
            }
            if extra_context_per_row is not None and extra_context_per_row[pos]:
                ctx.update(dict(extra_context_per_row[pos]))  # type: ignore[arg-type]

            det_detail: Dict[str, Any] = {}
            if extra_details_per_row is not None and extra_details_per_row[pos]:
                det_detail.update(dict(extra_details_per_row[pos]))  # type: ignore[arg-type]

            record = self.create_alert(
                ts_epoch=ts_val,
                source=source,
                detector=detector,
                alert_type=alert_type,
                score=float(s_arr[pos]),
                threshold=float(threshold),
                features=feat_snapshot,
                detector_detail=det_detail,
                context=ctx,
                total_alerts_gt=gt_alerts,
            )
            alerts.append(record)

        return alerts

    @staticmethod
    def to_dataframe(alerts: Iterable[AlertRecord | Mapping[str, Any]]) -> pd.DataFrame:
        """Chuyển danh sách Alert thành DataFrame theo đúng thứ tự 16 cột chuẩn K3."""
        rows = [a.to_dict() if isinstance(a, AlertRecord) else dict(a) for a in alerts]
        if not rows:
            return pd.DataFrame(columns=list(ALERT_SCHEMA_FIELDS))
        df = pd.DataFrame(rows)
        ordered = [c for c in ALERT_SCHEMA_FIELDS if c in df.columns]
        extra = [c for c in df.columns if c not in ordered]
        return df[ordered + extra]
