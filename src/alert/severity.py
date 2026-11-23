"""
severity.py — [K6] Bộ ánh xạ & leo thang mức độ nghiêm trọng cảnh báo (Severity Mapping)
Đề tài N3.10: Phát hiện bất thường trong log và dữ liệu vận hành mạng

Tuân thủ tuyệt đối bảng ánh xạ Severity trong docs/alert_schema.md (K3) và config/severity_rules.yaml (K6):
  - CRITICAL (4): Có log FATAL + Isolation Forest phát hiện đồng thời, HOẶC sự cố kéo dài > 10 phút,
                  HOẶC bão FATAL cực đoan / >= 3 detector độc lập cùng kích hoạt.
  - HIGH     (3): Isolation Forest vượt ngưỡng (phân vị 99%), HOẶC z-score > 5,
                  HOẶC template mới với > 50 dòng.
  - MEDIUM   (2): Rule FATAL đơn lẻ, HOẶC z-score > 3, HOẶC template hiếm tăng đột biến,
                  HOẶC template mới 5..50 dòng.
  - LOW      (1): Z-score vùng 2–3, HOẶC template mới ít dòng (< 5).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import yaml

# Đường dẫn mặc định tới file cấu hình severity_rules.yaml
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
DEFAULT_SEVERITY_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "severity_rules.yaml")

IF_ALERT_TYPES = frozenset({"IF_MULTIVARIATE", "IF_TEMPLATE_DRIFT"})
FATAL_ALERT_TYPES = frozenset({"RULE_FATAL_LOG"})
ZSCORE_ALERT_TYPES = frozenset({"ZSCORE_LOG_VOLUME", "ZSCORE_SEVERITY_SPIKE"})


class SeverityLevel(IntEnum):
    """Thang đo 4 mức độ nghiêm trọng chuẩn Alert Schema (K3)."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def from_name(cls, name: str) -> "SeverityLevel":
        if isinstance(name, cls):
            return name
        key = str(name).strip().upper()
        if key not in cls.__members__:
            raise ValueError(
                f"Mức severity không hợp lệ: '{name}'. Chỉ chấp nhận: {list(cls.__members__.keys())}"
            )
        return cls[key]

    @classmethod
    def from_score(cls, score: int) -> "SeverityLevel":
        try:
            return cls(int(score))
        except ValueError as exc:
            raise ValueError(
                f"Giá trị severity_score không hợp lệ: {score}. Chỉ chấp nhận 1..4."
            ) from exc


SEVERITY_SCORES: Dict[str, int] = {lvl.name: int(lvl.value) for lvl in SeverityLevel}


@dataclass(frozen=True)
class SeverityDecision:
    """Kết quả đánh giá mức độ nghiêm trọng kèm lý do giải thích."""

    severity: str
    severity_score: int
    matched_rule: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "severity_score": self.severity_score,
            "matched_rule": self.matched_rule,
            "reason": self.reason,
        }


DEFAULT_RULES_CONFIG: Dict[str, Any] = {
    "version": "1.0.0",
    "levels": {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4},
    "escalation": {
        "prolonged_incident_sec": 600,
        "fatal_plus_if_critical": True,
        "extreme_fatal_count": 1000,
        "min_detectors_for_critical": 3,
    },
    "rules": {
        "RULE_FATAL_LOG": {
            "default_severity": "MEDIUM",
            "high_fatal_count": 100,
            "critical_fatal_count": 1000,
        },
        "RULE_ERROR_SPIKE": {
            "default_severity": "MEDIUM",
            "high_ratio_multiplier": 2.0,
            "critical_ratio_multiplier": 5.0,
        },
        "ZSCORE_LOG_VOLUME": {
            "low_min": 2.0,
            "medium_min": 3.0,
            "high_min": 5.0,
            "critical_min": 10.0,
        },
        "ZSCORE_SEVERITY_SPIKE": {
            "low_min": 2.0,
            "medium_min": 3.0,
            "high_min": 5.0,
            "critical_min": 8.0,
        },
        "IF_MULTIVARIATE": {
            "default_severity": "HIGH",
            "low_fallback": "MEDIUM",
        },
        "IF_TEMPLATE_DRIFT": {
            "default_severity": "HIGH",
            "critical_new_tpl_lines": 200,
        },
        "NEW_TEMPLATE": {
            "low_max_lines": 4,
            "medium_max_lines": 50,
            "high_min_lines": 51,
        },
        "RARE_TEMPLATE_BURST": {
            "default_severity": "MEDIUM",
            "high_min_lines": 50,
            "critical_min_lines": 500,
        },
    },
}


def max_severity(severities: Iterable[str | SeverityLevel]) -> SeverityLevel:
    """Trả về mức nghiêm trọng cao nhất trong danh sách (theo quy tắc K3)."""
    levels = [SeverityLevel.from_name(s) for s in severities]
    if not levels:
        return SeverityLevel.LOW
    return max(levels)


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


class SeverityMapper:
    """
    Bộ ánh xạ quy tắc Severity (LOW -> CRITICAL) đọc từ config/severity_rules.yaml.
    Tách bạch giữa:
      1. Đánh giá cơ sở (base severity) theo từng alert_type & score/features.
      2. Đánh giá leo thang (escalation) khi đa detector trùng khớp hoặc sự cố kéo dài.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config if config is not None else self._load_default_config()
        self.escalation_cfg: Dict[str, Any] = self.config.get("escalation", {})
        self.rules_cfg: Dict[str, Any] = self.config.get("rules", {})

    @classmethod
    def from_yaml(cls, config_path: str = DEFAULT_SEVERITY_CONFIG_PATH) -> "SeverityMapper":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cls(config=cfg)

    @staticmethod
    def _load_default_config() -> Dict[str, Any]:
        if os.path.exists(DEFAULT_SEVERITY_CONFIG_PATH):
            with open(DEFAULT_SEVERITY_CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    return loaded
        return DEFAULT_RULES_CONFIG

    def _evaluate_base(
        self,
        alert_type: str,
        score: float,
        threshold: Optional[float],
        features: Mapping[str, Any],
    ) -> SeverityDecision:
        atype = str(alert_type).strip().upper()
        rule = self.rules_cfg.get(atype, {})

        # 1. RULE_FATAL_LOG
        if atype == "RULE_FATAL_LOG":
            fatal_count = max(
                _safe_float(features.get("f_sev_FATAL"), 0.0),
                _safe_float(score, 0.0),
            )
            crit_cnt = _safe_float(rule.get("critical_fatal_count", 1000))
            high_cnt = _safe_float(rule.get("high_fatal_count", 100))
            if fatal_count >= crit_cnt:
                return SeverityDecision(
                    "CRITICAL",
                    int(SeverityLevel.CRITICAL),
                    "RULE_FATAL_EXTREME",
                    f"Bão log FATAL cực lớn ({int(fatal_count)} >= {int(crit_cnt)} dòng/5p)",
                )
            if fatal_count >= high_cnt:
                return SeverityDecision(
                    "HIGH",
                    int(SeverityLevel.HIGH),
                    "RULE_FATAL_HIGH_VOLUME",
                    f"Số lượng log FATAL cao ({int(fatal_count)} >= {int(high_cnt)} dòng/5p)",
                )
            return SeverityDecision(
                "MEDIUM",
                int(SeverityLevel.MEDIUM),
                "RULE_FATAL_SINGLE",
                f"Phát hiện log mức FATAL ({int(fatal_count)} dòng trong cửa sổ 5 phút)",
            )

        # 2. RULE_ERROR_SPIKE
        if atype == "RULE_ERROR_SPIKE":
            th = _safe_float(threshold, 1.0)
            ratio = (score / th) if th > 0 else score
            crit_mult = _safe_float(rule.get("critical_ratio_multiplier", 5.0))
            high_mult = _safe_float(rule.get("high_ratio_multiplier", 2.0))
            if ratio >= crit_mult:
                return SeverityDecision(
                    "CRITICAL",
                    int(SeverityLevel.CRITICAL),
                    "RULE_ERROR_SPIKE_CRITICAL",
                    f"Lỗi ERROR vượt gấp {ratio:.1f}x ngưỡng ({score:.0f} vs {th:.0f})",
                )
            if ratio >= high_mult:
                return SeverityDecision(
                    "HIGH",
                    int(SeverityLevel.HIGH),
                    "RULE_ERROR_SPIKE_HIGH",
                    f"Lỗi ERROR vượt gấp {ratio:.1f}x ngưỡng ({score:.0f} vs {th:.0f})",
                )
            return SeverityDecision(
                "MEDIUM",
                int(SeverityLevel.MEDIUM),
                "RULE_ERROR_SPIKE_BASE",
                f"Số lượng log ERROR vượt ngưỡng ({score:.0f} > {th:.0f})",
            )

        # 3. ZSCORE_LOG_VOLUME & ZSCORE_SEVERITY_SPIKE
        if atype in ZSCORE_ALERT_TYPES:
            crit_min = _safe_float(rule.get("critical_min", 10.0))
            high_min = _safe_float(rule.get("high_min", 5.0))
            med_min = _safe_float(rule.get("medium_min", 3.0))
            if score >= crit_min:
                return SeverityDecision(
                    "CRITICAL",
                    int(SeverityLevel.CRITICAL),
                    f"{atype}_EXTREME",
                    f"Đột biến cực đoan với Z-score = {score:.2f} (>= {crit_min:.1f})",
                )
            if score > high_min:
                return SeverityDecision(
                    "HIGH",
                    int(SeverityLevel.HIGH),
                    f"{atype}_HIGH",
                    f"Tăng đột biến mạnh với Z-score = {score:.2f} (> {high_min:.1f})",
                )
            if score > med_min:
                return SeverityDecision(
                    "MEDIUM",
                    int(SeverityLevel.MEDIUM),
                    f"{atype}_MEDIUM",
                    f"Lưu lượng bất thường với Z-score = {score:.2f} (> {med_min:.1f})",
                )
            return SeverityDecision(
                "LOW",
                int(SeverityLevel.LOW),
                f"{atype}_LOW",
                f"Cảnh báo sớm Z-score = {score:.2f} (vùng 2–3)",
            )

        # 4. IF_MULTIVARIATE & IF_TEMPLATE_DRIFT
        if atype in IF_ALERT_TYPES:
            if atype == "IF_TEMPLATE_DRIFT":
                new_lines = _safe_float(features.get("f_n_new_tpl_lines"), 0.0)
                crit_new = _safe_float(rule.get("critical_new_tpl_lines", 200))
                if new_lines >= crit_new:
                    return SeverityDecision(
                        "CRITICAL",
                        int(SeverityLevel.CRITICAL),
                        "IF_TEMPLATE_DRIFT_CRITICAL",
                        f"Isolation Forest + bùng phát mẫu log mới ({int(new_lines)} dòng)",
                    )
            if threshold is None or score >= threshold:
                return SeverityDecision(
                    "HIGH",
                    int(SeverityLevel.HIGH),
                    f"{atype}_THRESHOLD_EXCEEDED",
                    f"Isolation Forest vượt ngưỡng phân vị ({score:.3f} >= {_safe_float(threshold):.3f})",
                )
            fallback = SeverityLevel.from_name(rule.get("low_fallback", "MEDIUM"))
            return SeverityDecision(
                fallback.name,
                int(fallback.value),
                f"{atype}_BELOW_THRESHOLD",
                f"Isolation Forest ghi nhận độ lệch nhẹ ({score:.3f} < {_safe_float(threshold):.3f})",
            )

        # 5. NEW_TEMPLATE
        if atype == "NEW_TEMPLATE":
            new_lines = max(
                _safe_float(features.get("f_n_new_tpl_lines"), 0.0),
                _safe_float(score, 0.0),
            )
            high_min = _safe_float(rule.get("high_min_lines", 51))
            low_max = _safe_float(rule.get("low_max_lines", 4))
            if new_lines >= high_min:
                return SeverityDecision(
                    "HIGH",
                    int(SeverityLevel.HIGH),
                    "NEW_TEMPLATE_BURST",
                    f"Xuất hiện mẫu log mới với mật độ lớn ({int(new_lines)} > 50 dòng)",
                )
            if new_lines > low_max:
                return SeverityDecision(
                    "MEDIUM",
                    int(SeverityLevel.MEDIUM),
                    "NEW_TEMPLATE_MODERATE",
                    f"Xuất hiện mẫu log chưa từng thấy ({int(new_lines)} dòng)",
                )
            return SeverityDecision(
                "LOW",
                int(SeverityLevel.LOW),
                "NEW_TEMPLATE_MINOR",
                f"Xuất hiện mẫu log mới với số lượng nhỏ ({int(new_lines)} < 5 dòng)",
            )

        # 6. RARE_TEMPLATE_BURST
        if atype == "RARE_TEMPLATE_BURST":
            rare_lines = max(
                _safe_float(features.get("f_n_rare_tpl_lines"), 0.0),
                _safe_float(score, 0.0),
            )
            crit_min = _safe_float(rule.get("critical_min_lines", 500))
            high_min = _safe_float(rule.get("high_min_lines", 50))
            if rare_lines >= crit_min:
                return SeverityDecision(
                    "CRITICAL",
                    int(SeverityLevel.CRITICAL),
                    "RARE_TEMPLATE_CRITICAL",
                    f"Bùng phát mẫu log hiếm cực mạnh ({int(rare_lines)} >= {int(crit_min)} dòng)",
                )
            if rare_lines >= high_min:
                return SeverityDecision(
                    "HIGH",
                    int(SeverityLevel.HIGH),
                    "RARE_TEMPLATE_HIGH",
                    f"Mẫu log hiếm tăng mạnh ({int(rare_lines)} >= {int(high_min)} dòng)",
                )
            return SeverityDecision(
                "MEDIUM",
                int(SeverityLevel.MEDIUM),
                "RARE_TEMPLATE_BASE",
                f"Phát hiện mẫu log hiếm tăng bất thường ({int(rare_lines)} dòng)",
            )

        # Fallback cho loại cảnh báo mở rộng (OpenSSH / TelecomTS sau này)
        default_lvl = SeverityLevel.from_name(rule.get("default_severity", "LOW"))
        return SeverityDecision(
            default_lvl.name,
            int(default_lvl.value),
            "DEFAULT_FALLBACK",
            f"Áp dụng mức mặc định cho loại cảnh báo '{atype}'",
        )

    def evaluate_escalation(
        self,
        base_decision: SeverityDecision,
        alert_type: str,
        features: Optional[Mapping[str, Any]] = None,
        concurrent_alert_types: Optional[Sequence[str]] = None,
        concurrent_detectors: Optional[Sequence[str]] = None,
        incident_duration_sec: float = 0.0,
        consecutive_windows: int = 1,
    ) -> SeverityDecision:
        """
        Kiểm tra các quy tắc leo thang mức nghiêm trọng lên CRITICAL (K3 & K6):
          1. Có log FATAL + Isolation Forest phát hiện đồng thời.
          2. Detector cảnh báo liên tiếp >= 3 cửa sổ (15 phút) HOẶC sự cố kéo dài > 10 phút (Zero Data Leakage).
          3. Có >= 3 detector độc lập cùng báo động trên cùng cửa sổ.
        """
        feats = features or {}
        atype = str(alert_type).strip().upper()
        all_types = {atype} | {str(t).strip().upper() for t in (concurrent_alert_types or ())}
        det_set = {str(d).strip() for d in (concurrent_detectors or ()) if d}

        fatal_count = _safe_float(feats.get("f_sev_FATAL"), 0.0)
        has_fatal = fatal_count > 0 or bool(all_types & FATAL_ALERT_TYPES)
        has_if = bool(all_types & IF_ALERT_TYPES)

        # Luật leo thang 1: FATAL + Isolation Forest đồng thời
        if self.escalation_cfg.get("fatal_plus_if_critical", True) and has_fatal and has_if:
            return SeverityDecision(
                "CRITICAL",
                int(SeverityLevel.CRITICAL),
                "ESCALATION_FATAL_PLUS_IF",
                "Leo thang CRITICAL: Phát hiện đồng thời log FATAL và bất thường Isolation Forest",
            )

        # Luật leo thang 2: Detector cảnh báo liên tiếp >= 3 cửa sổ (15 phút) HOẶC sự cố kéo dài > 600s
        min_consec = int(self.escalation_cfg.get("min_consecutive_windows", 3))
        prolonged_sec = _safe_float(self.escalation_cfg.get("prolonged_incident_sec", 600))
        if (consecutive_windows >= min_consec and min_consec > 1) or incident_duration_sec > prolonged_sec:
            reason = (
                f"Leo thang CRITICAL: Detector cảnh báo liên tiếp {consecutive_windows} cửa sổ (>= {min_consec})"
                if consecutive_windows >= min_consec
                else f"Leo thang CRITICAL: Đợt sự cố kéo dài {int(incident_duration_sec)}s (> {int(prolonged_sec)}s)"
            )
            return SeverityDecision(
                "CRITICAL",
                int(SeverityLevel.CRITICAL),
                "ESCALATION_PROLONGED_INCIDENT",
                reason,
            )

        # Luật leo thang 3: Nhiều detector độc lập cùng xác nhận (>= 3)
        min_dets = int(self.escalation_cfg.get("min_detectors_for_critical", 3))
        if len(det_set) >= min_dets:
            return SeverityDecision(
                "CRITICAL",
                int(SeverityLevel.CRITICAL),
                "ESCALATION_MULTI_DETECTOR",
                f"Leo thang CRITICAL: {len(det_set)} detector độc lập cùng phát hiện bất thường",
            )

        return base_decision

    def map_severity(
        self,
        alert_type: str,
        score: float,
        threshold: Optional[float] = None,
        features: Optional[Mapping[str, Any]] = None,
        concurrent_alert_types: Optional[Sequence[str]] = None,
        concurrent_detectors: Optional[Sequence[str]] = None,
        incident_duration_sec: float = 0.0,
        consecutive_windows: int = 1,
    ) -> SeverityDecision:
        """
        Điểm vào chính (Single Entry Point) để tính Severity cho một cảnh báo.
        Kết hợp cả luật cơ sở theo detector và luật leo thang bối cảnh.
        """
        feats = features or {}
        base = self._evaluate_base(alert_type, float(score), threshold, feats)
        final = self.evaluate_escalation(
            base_decision=base,
            alert_type=alert_type,
            features=feats,
            concurrent_alert_types=concurrent_alert_types,
            concurrent_detectors=concurrent_detectors,
            incident_duration_sec=incident_duration_sec,
            consecutive_windows=consecutive_windows,
        )
        # Luôn giữ mức cao hơn giữa base và escalated
        if final.severity_score >= base.severity_score:
            return final
        return base
