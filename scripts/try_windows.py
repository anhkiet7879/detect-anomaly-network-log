"""
try_windows.py — [H5] Chạy thử đặc trưng cửa sổ 5 phút (bộ A + bộ B) với MỘT lần chia Train/Test.

Các bước:
  1. Đọc N dòng đầu của CSV đã parse.
  2. Chia theo thời gian: Train = trước mốc cắt (làm tròn về đầu cửa sổ 5 phút);
     bỏ khoảng đệm --buffer-mins phút sau mốc cắt (giống buffer_gap_mins của experiment.yaml); phần còn lại = Test.
  3. build_fold_features(): học Drain3 trên Train, khóa, gán Test, tạo cửa sổ cho cả hai.
  4. Tự kiểm tra: tổng f_n_logs = số dòng; tổng f_sev_* = f_n_logs; tổng f_tpl_* = f_n_logs;
     Train và Test cùng bộ cột; Train không có template mới.
  5. So sánh trung bình đặc trưng giữa cửa sổ có alert (y=1) và không (y=0) trên Test.

CÁCH CHẠY (Windows, trong thư mục N3.10)
    python scripts\\try_windows.py
    python scripts\\try_windows.py --nrows 949592 --rare-max 10 --no-fill-empty

ĐẦU RA
    data\\features\\h5_windows_train.csv, h5_windows_test.csv   (bị .gitignore chặn — không đưa lên Git)
    reports\\features\\h5_windows_summary.txt                     báo cáo
    reports\\features\\h5_templates.csv                           từ điển template_id -> template (để đọc cột f_tpl_<id>)
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from features.window_features import (LINE_COLUMNS, WINDOW_SEC, build_fold_features,  # noqa: E402
                                      feature_columns)
from templates.log_template_extractor import LogTemplateExtractor  # noqa: E402


def check(cond):
    return "ĐẠT" if cond else "KHÔNG ĐẠT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join("data", "processed", "bgl_first20.csv"))
    ap.add_argument("--nrows", type=int, default=949_592)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--buffer-mins", type=int, default=60)
    ap.add_argument("--rare-max", type=int, default=10, help="template có < rare_max dòng trên Train = hiếm")
    ap.add_argument("--no-fill-empty", action="store_true", help="không thêm cửa sổ trống")
    ap.add_argument("--config", default=os.path.join("config", "drain3.ini"))
    ap.add_argument("--feat-dir", default=os.path.join("data", "features"))
    ap.add_argument("--report-dir", default=os.path.join("reports", "features"))
    args = ap.parse_args()
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    fill_empty = not args.no_fill_empty

    df = pd.read_csv(args.input, nrows=args.nrows, keep_default_na=False, usecols=LINE_COLUMNS,
                     dtype={"ts_epoch": "int64", "gt_is_alert": "int8"})

    # ---- chia theo thời gian, cắt đúng ranh giới cửa sổ, có khoảng đệm
    cut_ts = int(df["ts_epoch"].iloc[int(len(df) * args.train_frac)])
    cut_ts -= cut_ts % WINDOW_SEC
    test_start = cut_ts + args.buffer_mins * 60
    train = df[df["ts_epoch"] < cut_ts]
    gap = df[(df["ts_epoch"] >= cut_ts) & (df["ts_epoch"] < test_start)]
    test = df[df["ts_epoch"] >= test_start]

    ext = LogTemplateExtractor(args.config)
    w_tr, w_te = build_fold_features(train, test, ext, rare_max=args.rare_max, fill_empty=fill_empty)

    w_tr.to_csv(os.path.join(args.feat_dir, "h5_windows_train.csv"), index=False)
    w_te.to_csv(os.path.join(args.feat_dir, "h5_windows_test.csv"), index=False)
    ext.templates().to_csv(os.path.join(args.report_dir, "h5_templates.csv"), index=False, encoding="utf-8")

    cols_a = feature_columns(w_tr, ("A",))
    cols_b = feature_columns(w_tr, ("B",))
    sev_cols = [c for c in w_tr.columns if c.startswith("f_sev_")]
    tpl_cols = [c for c in w_tr.columns if c.startswith("f_tpl_")]
    utc = lambda t: pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%dT%H:%MZ")

    L = []
    p = L.append
    p(f"===== [H5] ĐẶC TRƯNG CỬA SỔ 5 PHÚT — {os.path.basename(args.input)}, {len(df):,} dòng =====")
    p(f"Mốc cắt        : {utc(cut_ts)}  | khoảng đệm {args.buffer_mins} phút ({len(gap):,} dòng bị bỏ)")
    p(f"Cửa sổ trống   : {'CÓ thêm (fill_empty)' if fill_empty else 'KHÔNG thêm'}   | template hiếm: < {args.rare_max} dòng trên Train")
    p(f"Số đặc trưng   : bộ A {len(cols_a)} cột, bộ B {len(cols_b)} cột ({len(tpl_cols)} cột f_tpl_*)")
    p("")
    p(f"{'phần':<6} {'từ':<18} {'đến':<18} {'dòng log':>10} {'cửa sổ':>8} {'có log':>7} {'alert':>6} {'%alert':>7}")
    for name, lines, w in [("train", train, w_tr), ("test", test, w_te)]:
        p(f"{name:<6} {utc(w.ts_epoch.min()):<18} {utc(w.ts_epoch.max()):<18} {len(lines):>10,} "
          f"{len(w):>8,} {(w.f_n_logs > 0).sum():>7,} {w.y.sum():>6,} {w.y.mean():>7.2%}")
    p("")
    p("--- Tự kiểm tra ---")
    for name, lines, w in [("train", train, w_tr), ("test", test, w_te)]:
        p(f"[{name}] tổng f_n_logs = số dòng ({w.f_n_logs.sum():,})        : {check(w.f_n_logs.sum() == len(lines))}")
        p(f"[{name}] tổng f_sev_* = f_n_logs ở mọi cửa sổ               : "
          f"{check((w[sev_cols].sum(axis=1) == w.f_n_logs).all())}")
        p(f"[{name}] tổng f_tpl_* = f_n_logs ở mọi cửa sổ               : "
          f"{check((w[tpl_cols].sum(axis=1) == w.f_n_logs).all())}")
        p(f"[{name}] tổng gt_alert_lines = số dòng alert ({lines.gt_is_alert.sum():,}) : "
          f"{check(w.gt_alert_lines.sum() == lines.gt_is_alert.sum())}")
    p(f"Train và Test cùng bộ cột                                : {check(list(w_tr.columns) == list(w_te.columns))}")
    p(f"Train không có dòng template mới                         : {check(w_tr.f_n_new_tpl_lines.sum() == 0)}")
    p(f"Không có cột gt_/y trong danh sách đặc trưng             : "
      f"{check(not any(c.startswith('gt_') or c == 'y' for c in cols_a + cols_b))}")
    p("")
    p("--- Test: trung bình đặc trưng ở cửa sổ alert (y=1) vs bình thường (y=0), chỉ cửa sổ có log ---")
    t = w_te[w_te.f_n_logs > 0]
    show = ["f_n_logs", "f_n_nodes", "f_sev_FATAL", "f_sev_FAILURE", "f_sev_ERROR",
            "f_n_templates", "f_n_new_tpl_lines", "f_n_rare_tpl_lines"]
    p(f"  {'đặc trưng':<20} {'y=0':>12} {'y=1':>12}   cửa sổ: y=0 {int((t.y == 0).sum()):,} | y=1 {int(t.y.sum()):,}")
    for c in show:
        m0, m1 = t.loc[t.y == 0, c].mean(), t.loc[t.y == 1, c].mean()
        p(f"  {c:<20} {m0:>12,.2f} {m1:>12,.2f}")
    p("")
    p("--- Test: cửa sổ có ít nhất 1 dòng template mới ---")
    has_new = t.f_n_new_tpl_lines > 0
    p(f"  {int(has_new.sum()):,} cửa sổ, trong đó {int(t.loc[has_new, 'y'].sum()):,} cửa sổ có alert "
      f"(tổng cửa sổ alert trên Test: {int(t.y.sum()):,})")

    text = "\n".join(L)
    print(text)
    with open(os.path.join(args.report_dir, "h5_windows_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\nĐã lưu: " + os.path.join(args.feat_dir, "h5_windows_train.csv") + ", h5_windows_test.csv")
    print("        " + os.path.join(args.report_dir, "h5_windows_summary.txt") + ", h5_templates.csv")


if __name__ == "__main__":
    main()
