"""
test_window_features.py — kiểm tra [H5] bộ A và [H6] xuất/nạp Parquet trên dữ liệu giả lập nhỏ (biết trước đáp án).
Chạy (trong thư mục N3.10):   python -m pytest tests\\test_window_features.py -v
"""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from features.window_features import build_set_a, feature_columns, load_windows, save_windows  # noqa: E402

T0 = 1117838400  # chia hết cho 300 -> đầu một cửa sổ


@pytest.fixture
def lines():
    # cửa sổ 0: 3 dòng (1 alert) | cửa sổ 1: trống | cửa sổ 2: 2 dòng
    return pd.DataFrame({
        "ts_epoch":      [T0, T0 + 10, T0 + 299, T0 + 600, T0 + 899],
        "node":          ["n1", "n2", "n1", "n3", "n3"],
        "component":     ["KERNEL", "KERNEL", "APP", "KERNEL", "MMCS"],
        "severity_raw":  ["INFO", "FATAL", "FATAL", "INFO", "WEIRD"],
        "gt_alert_type": ["NORMAL", "KERNDTLB", "NORMAL", "NORMAL", "NORMAL"],
        "gt_is_alert":   [0, 1, 0, 0, 0],
    })


def test_counts_and_labels(lines):
    w = build_set_a(lines)
    assert list(w.ts_epoch) == [T0, T0 + 300, T0 + 600]          # có cửa sổ trống ở giữa
    assert list(w.f_n_logs) == [3, 0, 2]
    assert list(w.f_n_nodes) == [2, 0, 1]
    assert list(w.f_sev_FATAL) == [2, 0, 0]
    assert list(w.f_sev_other) == [0, 0, 1]                      # mức lạ "WEIRD" vào other
    assert list(w.y) == [1, 0, 0]
    assert list(w.gt_alert_types) == ["KERNDTLB", "", ""]
    assert (w.source == "bgl").all() and w.ts_utc.iloc[0] == "2005-06-03T22:40:00Z"


def test_no_fill_empty(lines):
    assert list(build_set_a(lines, fill_empty=False).ts_epoch) == [T0, T0 + 600]


def test_feature_columns_exclude_labels(lines):
    feats = feature_columns(build_set_a(lines), ("A",))
    assert feats and not any(c.startswith("gt_") or c in ("y", "ts_epoch", "ts_utc", "source") for c in feats)


@pytest.mark.parametrize("ext", ["parquet", "csv"])
def test_save_load_roundtrip(lines, tmp_path, ext):
    w = build_set_a(lines)
    path = tmp_path / f"w.{ext}"
    save_windows(w, str(path))
    back = load_windows(str(path))
    assert list(back.gt_alert_types) == ["KERNDTLB", "", ""]    # chuỗi rỗng KHÔNG thành NaN
    if ext == "parquet":
        assert back.equals(w)
