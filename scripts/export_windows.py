"""
export_windows.py — [H6] Xuất đặc trưng cửa sổ 5 phút (BỘ A) của toàn bộ dữ liệu ra Parquet để cả nhóm dùng chung.

Vì sao chỉ BỘ A?
  Bộ A chỉ ĐẾM (số log, severity, node, component) — không học gì từ dữ liệu — nên tính 1 lần trên cả file
  không gây rò rỉ. Bộ B (template Drain3) phải học lại trên Train của TỪNG fold -> không xuất ra file cố định;
  runner.py tính bộ B bằng build_fold_features() ở mỗi fold.

Dùng cho:
  - Phước: Rule FATAL (P1), Z-score (P3), chạy baseline (P4), nạp bảng log_windows (P5).
  - Kiệt:  runner.py — thay data/features/bgl_mvp_windows.csv (chỉ 20% đầu) bằng file toàn bộ BGL.

CÁCH CHẠY (Windows, trong thư mục N3.10)
    pip install pyarrow
    python scripts\\export_windows.py
    python scripts\\export_windows.py --input data\\processed\\bgl_first20.csv --output data\\features\\bgl_first20_windows_A.parquet

ĐẦU RA
    data\\features\\bgl_windows_A.parquet      (~vài MB, KHÔNG bị .gitignore chặn -> đưa lên Git để cả nhóm dùng,
                                               không cần file bgl_parsed.csv 673 MB)
    reports\\features\\h6_export_summary.txt    báo cáo + tự kiểm tra

CỘT: ts_epoch (đầu cửa sổ), ts_utc, source, y, gt_alert_lines, gt_alert_types, f_n_logs, f_n_nodes,
     f_n_components, f_sev_<LEVEL> (6 mức + other), f_comp_<COMPONENT>.  Cột gt_ và y CHỈ để chấm điểm.
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from features.window_features import build_set_a, feature_columns, load_windows, save_windows  # noqa: E402

USECOLS = ["ts_epoch", "node", "component", "severity_raw", "gt_alert_type", "gt_is_alert"]


def check(cond):
    return "ĐẠT" if cond else "KHÔNG ĐẠT"


def main():
    ap = argparse.ArgumentParser(description="[H6] Xuất bộ đặc trưng A (cửa sổ 5 phút) ra Parquet")
    ap.add_argument("--input", default=os.path.join("data", "processed", "bgl_parsed.csv"))
    ap.add_argument("--output", default=os.path.join("data", "features", "bgl_windows_A.parquet"))
    ap.add_argument("--source", default="bgl")
    ap.add_argument("--no-fill-empty", action="store_true", help="không thêm cửa sổ trống (mặc định: có thêm)")
    ap.add_argument("--report", default=os.path.join("reports", "features", "h6_export_summary.txt"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    t0 = time.perf_counter()
    print(f"Đọc {args.input} (chỉ {len(USECOLS)} cột, không đọc message) ...")
    lines = pd.read_csv(args.input, usecols=USECOLS, keep_default_na=False,
                        dtype={"ts_epoch": "int64", "gt_is_alert": "int8", "node": "category",
                               "component": "category", "severity_raw": "category", "gt_alert_type": "category"})
    t_read = time.perf_counter() - t0
    print(f"  {len(lines):,} dòng ({t_read:.1f} giây). Tạo cửa sổ ...")

    w = build_set_a(lines, fill_empty=not args.no_fill_empty, source=args.source)
    save_windows(w, args.output)
    t_all = time.perf_counter() - t0

    back = load_windows(args.output)
    sev_cols = [c for c in w.columns if c.startswith("f_sev_")]
    comp_cols = [c for c in w.columns if c.startswith("f_comp_")]
    feats = feature_columns(w, ("A",))
    utc = lambda t: pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%dT%H:%MZ")
    nonempty = w[w.f_n_logs > 0]

    L = []
    p = L.append
    p(f"===== [H6] XUẤT ĐẶC TRƯNG CỬA SỔ 5 PHÚT (BỘ A) — {os.path.basename(args.input)} =====")
    p(f"Đầu vào        : {args.input}  ({len(lines):,} dòng)")
    p(f"Đầu ra         : {args.output}  ({os.path.getsize(args.output) / 1e6:.1f} MB)")
    p(f"Thời gian      : đọc {t_read:.1f} giây, tổng {t_all:.1f} giây")
    p(f"Khoảng thời gian: {utc(w.ts_epoch.min())} -> {utc(w.ts_epoch.max())}")
    p(f"Cửa sổ trống   : {'KHÔNG thêm' if args.no_fill_empty else 'CÓ thêm (điền 0)'}")
    p("")
    p(f"Số cửa sổ      : {len(w):,}  (có log: {len(nonempty):,} = {len(nonempty) / len(w):.1%})")
    p(f"Cửa sổ alert   : {int(w.y.sum()):,}  ({w.y.mean():.2%} tổng số cửa sổ, "
      f"{nonempty.y.mean():.2%} số cửa sổ có log)")
    p(f"Số cột         : {w.shape[1]} (đặc trưng bộ A: {len(feats)}; f_sev_*: {len(sev_cols)}; f_comp_*: {len(comp_cols)})")
    p("")
    p("--- Tự kiểm tra ---")
    p(f"Tổng f_n_logs = số dòng ({int(w.f_n_logs.sum()):,})                 : {check(w.f_n_logs.sum() == len(lines))}")
    p(f"Tổng gt_alert_lines = số dòng alert ({int(lines.gt_is_alert.sum()):,})    : "
      f"{check(w.gt_alert_lines.sum() == lines.gt_is_alert.sum())}")
    p(f"Tổng f_sev_* = f_n_logs ở mọi cửa sổ                    : {check((w[sev_cols].sum(axis=1) == w.f_n_logs).all())}")
    p(f"Tổng f_comp_* = f_n_logs ở mọi cửa sổ                   : {check((w[comp_cols].sum(axis=1) == w.f_n_logs).all())}")
    p(f"ts_epoch tăng dần, cách đều 300 giây (khi có cửa sổ trống): "
      f"{check(w.ts_epoch.is_monotonic_increasing and (args.no_fill_empty or (w.ts_epoch.diff().dropna() == 300).all()))}")
    p(f"Đọc lại file Parquet giống hệt bảng đã ghi               : {check(back.equals(w))}")
    p(f"Không có cột gt_/y trong danh sách đặc trưng             : "
      f"{check(not any(c.startswith('gt_') or c == 'y' for c in feats))}")
    p("")
    p("--- Kiểu dữ liệu (để Phước/Kiệt đọc đúng) ---")
    p("  " + ", ".join(f"{c}:{t}" for c, t in back.dtypes.astype(str).items() if c in
                       ["ts_epoch", "ts_utc", "source", "y", "gt_alert_lines", "gt_alert_types", "f_n_logs"]))
    p("")
    p("--- Cách đọc ---")
    p("  import pandas as pd")
    p(f"  w = pd.read_parquet(r\"{args.output}\")")
    p("  # hoặc: from features.window_features import load_windows; w = load_windows(path)")
    p("  # chọn đặc trưng bộ A: from features.window_features import feature_columns; X = w[feature_columns(w, ('A',))]")

    text = "\n".join(L)
    print("\n" + text)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nĐã lưu báo cáo: {args.report}")


if __name__ == "__main__":
    main()
