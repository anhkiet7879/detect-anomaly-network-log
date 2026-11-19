"""
window_features.py — [H5] Đặc trưng theo cửa sổ 5 phút (tumbling window).

Mỗi hàng đầu ra = 1 cửa sổ 5 phút. Cột:
  ts_epoch                 mốc ĐẦU cửa sổ (epoch giây UTC) — trùng tên timestamp_column của experiment.yaml
  y                        nhãn cửa sổ: 1 nếu có ít nhất 1 dòng alert (CHỈ để chấm điểm)
  gt_alert_lines           số dòng alert trong cửa sổ          (CHỈ để chấm điểm)
  gt_alert_types           các loại alert, nối bằng "|"        (CHỈ để chấm điểm)

  BỘ A — không dùng message (giữ nguyên tên như mvp_step2_windows.py):
  f_n_logs                 tổng số dòng log
  f_n_nodes                số node khác nhau
  f_n_components           số component khác nhau
  f_sev_<LEVEL>            số dòng theo severity_raw (INFO, WARNING, SEVERE, ERROR, FATAL, FAILURE, other)
  f_comp_<COMPONENT>       số dòng theo component (danh sách component lấy từ Train)

  BỘ B — dựa trên template Drain3 (cần template_id, do LogTemplateExtractor gán):
  f_n_templates            số template_id khác nhau trong cửa sổ
  f_n_new_tpl_lines        số dòng có template MỚI (template_id = -1, chưa có khi học)  -> luôn 0 trên Train
  f_n_rare_tpl_lines       số dòng thuộc template HIẾM (số dòng trên Train < rare_max)
  f_tpl_<id>               số dòng của từng template đã học trên Train (vector tần suất)
  f_tpl_new                số dòng template mới (= f_n_new_tpl_lines, đặt cạnh vector cho đủ)

QUY TẮC CHỐNG RÒ RỈ
  - Danh sách component, danh sách template và số dòng/template (để xét "hiếm") đều lấy từ TRAIN.
    Train và Test vì vậy luôn có CÙNG bộ cột.
  - Dùng build_fold_features() ở MỖI fold: hàm này reset Drain3, học trên Train, khóa, rồi mới gán Test.

CỬA SỔ TRỐNG
  fill_empty=True (mặc định, giống MVP): thêm cửa sổ không có log với mọi đếm = 0 để chuỗi liên tục.
  Lưu ý: BGL có >1.000 khoảng trống > 1 giờ (có thể là mất log chứ không phải "không có hoạt động");
  cửa sổ trống chiếm đa số -> bộ phát hiện nên được thử cả khi bỏ cửa sổ f_n_logs = 0.
"""
import numpy as np
import pandas as pd

WINDOW_SEC = 300
LEVELS = ["INFO", "WARNING", "SEVERE", "ERROR", "FATAL", "FAILURE"]
LINE_COLUMNS = ["line_no", "ts_epoch", "node", "component", "severity_raw", "message",
                "gt_alert_type", "gt_is_alert"]


def add_window_column(lines, win_sec=WINDOW_SEC):
    lines = lines.copy()
    lines["win"] = lines["ts_epoch"] - lines["ts_epoch"] % win_sec
    return lines


def window_index(lines, win_sec=WINDOW_SEC, fill_empty=True):
    """Danh sách mốc đầu cửa sổ: liên tục từ cửa sổ đầu đến cuối (fill_empty) hoặc chỉ cửa sổ có log."""
    if fill_empty:
        return np.arange(lines["win"].min(), lines["win"].max() + win_sec, win_sec)
    return np.sort(lines["win"].unique())


def base_features(lines, idx, components):
    """Bộ A + cột nhãn. lines phải có cột 'win'."""
    g = lines.groupby("win")
    out = pd.DataFrame({
        "gt_alert_lines": g["gt_is_alert"].sum(),
        "f_n_logs": g.size(),
        "f_n_nodes": g["node"].nunique(),
        "f_n_components": g["component"].nunique(),
    })
    sev = pd.crosstab(lines["win"], lines["severity_raw"])
    for lv in LEVELS:
        out[f"f_sev_{lv}"] = sev[lv] if lv in sev.columns else 0
    other = [c for c in sev.columns if c not in LEVELS]
    out["f_sev_other"] = sev[other].sum(axis=1) if other else 0
    comp = pd.crosstab(lines["win"], lines["component"])
    for c in components:
        out[f"f_comp_{c}"] = comp[c] if c in comp.columns else 0

    out = out.reindex(idx).fillna(0).astype("int64")
    types = (lines[lines["gt_is_alert"] == 1].groupby("win")["gt_alert_type"]
             .agg(lambda s: "|".join(sorted(set(s)))))
    out["gt_alert_types"] = types.reindex(idx).fillna("")
    out["y"] = (out["gt_alert_lines"] > 0).astype("int8")
    return out


def template_features(lines, idx, train_counts, rare_max=10):
    """Bộ B. lines phải có 'win' và 'template_id'. train_counts: Series template_id -> số dòng trên Train."""
    tid = lines["template_id"]
    is_new = tid == -1
    n_train = tid.map(train_counts).fillna(0)
    is_rare = (~is_new) & (n_train < rare_max)

    w = lines["win"]
    out = pd.DataFrame({
        "f_n_templates": lines.groupby("win")["template_id"].nunique(),
        "f_n_new_tpl_lines": is_new.groupby(w).sum(),
        "f_n_rare_tpl_lines": is_rare.groupby(w).sum(),
    })

    # Vector tần suất: cột cố định theo từ điển Train (+ cột cho template mới)
    known_ids = sorted(int(i) for i in train_counts.index)
    mat = pd.crosstab(w, tid).reindex(columns=known_ids + [-1], fill_value=0)
    mat.columns = [f"f_tpl_{i}" if i != -1 else "f_tpl_new" for i in mat.columns]

    out = out.join(mat).reindex(idx).fillna(0).astype("int64")
    return out


def build_windows(lines, components, train_counts, rare_max=10, win_sec=WINDOW_SEC, fill_empty=True):
    """Ghép bộ A + bộ B cho MỘT phần dữ liệu (Train hoặc Test). lines đã có template_id."""
    lines = add_window_column(lines, win_sec)
    idx = window_index(lines, win_sec, fill_empty)
    a = base_features(lines, idx, components)
    b = template_features(lines, idx, train_counts, rare_max)
    w = a.join(b)
    w.index.name = "ts_epoch"
    w = w.reset_index()
    front = ["ts_epoch", "y", "gt_alert_lines", "gt_alert_types"]
    return w[front + [c for c in w.columns if c not in front]]


def build_fold_features(train_lines, test_lines, extractor, rare_max=10, win_sec=WINDOW_SEC, fill_empty=True):
    """Dùng trong runner.py ở MỖI fold.
    1) reset + học Drain3 trên train_lines  2) khóa, gán template cho test_lines
    3) tạo cửa sổ cho cả hai với cùng bộ cột (lấy từ Train)."""
    train_lines = train_lines.copy()
    test_lines = test_lines.copy()
    train_lines["template_id"] = extractor.fit_transform(train_lines["message"])
    test_lines["template_id"], _ = extractor.transform(test_lines["message"])

    components = sorted(train_lines["component"].unique())
    train_counts = train_lines["template_id"].value_counts()
    w_train = build_windows(train_lines, components, train_counts, rare_max, win_sec, fill_empty)
    w_test = build_windows(test_lines, components, train_counts, rare_max, win_sec, fill_empty)
    return w_train, w_test


def feature_columns(windows, sets=("A", "B")):
    """Lấy danh sách cột đặc trưng theo bộ. Bộ A: f_ không thuộc template; Bộ B: cột template."""
    tpl = [c for c in windows.columns if c.startswith(("f_tpl_", "f_n_templates", "f_n_new_tpl", "f_n_rare_tpl"))]
    a = [c for c in windows.columns if c.startswith("f_") and c not in tpl]
    return (a if "A" in sets else []) + (tpl if "B" in sets else [])
