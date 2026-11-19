"""
try_drain3.py — [H3/H4] Thử nghiệm Drain3 trên N dòng đầu của CSV đã parse.

Làm 3 việc:
  1. Học template trên 80% đầu của N dòng (giả lập Train), đo thời gian, đếm số template.
  2. Chạy transform() trên 20% sau (giả lập Test): bao nhiêu dòng rơi vào template mới (-1)?
     Kiểm tra từ điển KHÔNG đổi sau transform (điều kiện chống rò rỉ, test T4 của Khang).
  3. Kiểm tra reset(): học lại lần 2 trên cùng dữ liệu -> phải ra đúng số template như lần 1.

CÁCH CHẠY (Windows, trong thư mục N3.10)
    python scripts\\try_drain3.py
    python scripts\\try_drain3.py --input data\\processed\\bgl_first20.csv --nrows 200000 --sim-th 0.4

ĐẦU RA
    reports\\drain3\\drain3_trial_<nrows>.txt          báo cáo
    reports\\drain3\\drain3_templates_<nrows>.csv      từ điển template (id, số dòng train, template)
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from templates.log_template_extractor import LogTemplateExtractor, NEW_ID  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join("data", "processed", "bgl_first20.csv"))
    ap.add_argument("--nrows", type=int, default=200_000)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--config", default=os.path.join("config", "drain3.ini"))
    ap.add_argument("--sim-th", type=float, default=None, help="ghi đè sim_th trong drain3.ini để thử")
    ap.add_argument("--out-dir", default=os.path.join("reports", "drain3"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input, nrows=args.nrows, keep_default_na=False,
                     usecols=["line_no", "ts_utc", "severity_raw", "message", "gt_alert_type", "gt_is_alert"])
    cut = int(len(df) * args.train_frac)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    ext = LogTemplateExtractor(args.config)
    if args.sim_th is not None:
        ext.config.drain_sim_th = args.sim_th
        ext.reset()
    sim_th = ext.config.drain_sim_th

    # 1. Học trên Train
    train["template_id"] = ext.fit_transform(train["message"])
    n_tpl_1 = ext.n_templates
    fp_before = ext.fingerprint()

    # 2. Suy luận trên Test
    t0 = time.perf_counter()
    test["template_id"], test["is_new_template"] = ext.transform(test["message"])
    t_inf = time.perf_counter() - t0
    dict_unchanged = ext.fingerprint() == fp_before

    try:
        ext.learn("thử học thêm sau khi khóa")
        lock_ok = False
    except RuntimeError:
        lock_ok = True

    # 3. Reset rồi học lại
    tpl = ext.templates()
    ext2 = LogTemplateExtractor(args.config)
    ext2.config.drain_sim_th = sim_th
    ext2.reset()
    ext2.fit_transform(train["message"])
    reset_ok = ext2.n_templates == n_tpl_1

    # ---------------------------------------------------------- báo cáo
    new = test[test["is_new_template"] == 1]
    per_tpl = (train.groupby("template_id")
               .agg(alert_rate=("gt_is_alert", "mean"), sev=("severity_raw", lambda s: s.mode()[0])))
    tpl = tpl.join(per_tpl, on="template_id")

    L = []
    p = L.append
    p(f"===== THỬ NGHIỆM DRAIN3 — {os.path.basename(args.input)}, {len(df):,} dòng đầu =====")
    p(f"Cấu hình         : {args.config}  (sim_th = {sim_th}, depth = {ext.config.drain_depth})")
    p(f"Train / Test     : {len(train):,} dòng ({train.ts_utc.iloc[0]} -> {train.ts_utc.iloc[-1]})"
      f" / {len(test):,} dòng ({test.ts_utc.iloc[0]} -> {test.ts_utc.iloc[-1]})")
    p("")
    p("--- 1. Học trên Train ---")
    p(f"Số template học được : {n_tpl_1:,}  (+ <EMPTY> nếu có: {ext.n_empty_train:,} dòng rỗng)")
    p(f"Số message khác nhau : {train.message.nunique():,}")
    p(f"Thời gian học        : {ext.fit_seconds:.1f} giây  ({len(train) / ext.fit_seconds:,.0f} dòng/giây)")
    p(f"Template chiếm 90% số dòng: {int((tpl.train_count.cumsum() < 0.9 * len(train)).sum()) + 1}")
    p(f"Template chỉ có 1 dòng    : {(tpl.train_count == 1).sum():,}")
    p("")
    p("--- 2. Suy luận trên Test (từ điển đã khóa) ---")
    p(f"Thời gian            : {t_inf:.1f} giây")
    p(f"Dòng rơi vào template MỚI (-1): {len(new):,} / {len(test):,} ({len(new) / max(len(test), 1):.2%})")
    if len(new):
        p(f"  trong đó là alert : {new.gt_is_alert.sum():,}")
        p("  ví dụ message mới (tối đa 8 mẫu khác nhau):")
        for m, n in new.message.value_counts().head(8).items():
            p(f"    {n:>7,} × {m[:110]}")
    p(f"Từ điển KHÔNG đổi sau transform : {'ĐẠT' if dict_unchanged else 'KHÔNG ĐẠT'}")
    p(f"Gọi learn() sau khi khóa bị chặn: {'ĐẠT' if lock_ok else 'KHÔNG ĐẠT'}")
    p("")
    p("--- 3. Reset State ---")
    p(f"Học lại từ đầu ra cùng số template ({ext2.n_templates:,}): {'ĐẠT' if reset_ok else 'KHÔNG ĐẠT'}")
    p("")
    p("--- 4. 15 template nhiều dòng nhất trên Train ---")
    p(f"  {'id':>4} {'số dòng':>9} {'%alert':>7} {'sev':<8} template")
    for r in tpl.head(15).itertuples():
        p(f"  {r.template_id:>4} {r.train_count:>9,} {r.alert_rate:>7.1%} {r.sev:<8} {r.template[:90]}")
    p("")
    p("--- 5. Template có alert trên Train ---")
    for r in tpl[tpl.alert_rate > 0].itertuples():
        p(f"  {r.template_id:>4} {r.train_count:>9,} {r.alert_rate:>7.1%} {r.sev:<8} {r.template[:90]}")

    text = "\n".join(L)
    print(text)
    tag = f"{args.nrows // 1000}k" + (f"_sim{sim_th}" if args.sim_th is not None else "")
    with open(os.path.join(args.out_dir, f"drain3_trial_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    tpl.to_csv(os.path.join(args.out_dir, f"drain3_templates_{tag}.csv"), index=False, encoding="utf-8")
    print(f"\nĐã lưu báo cáo và từ điển template vào {args.out_dir}")


if __name__ == "__main__":
    main()
