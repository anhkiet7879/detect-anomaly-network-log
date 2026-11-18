"""
mvp_step2_windows.py — MVP Bước 2: gộp log thành cửa sổ 5 phút, tạo đặc trưng BỘ A
(không dùng message), gắn nhãn cửa sổ và chia train / val / test theo thời gian.

Đầu vào : data\\processed\\bgl_first20.csv  (do bgl_parse_v2.py --fraction 0.2 tạo, schema 12 cột)
Đầu ra  : data\\features\\bgl_mvp_windows.csv   mỗi hàng = 1 cửa sổ
          reports\\mvp\\step2_summary.txt         thống kê từng phần

Cách chia (cách B):
  - line_no <  SPLIT_LINE (10% đầu file)  -> phần HỌC
        + 80% thời gian đầu của phần học  -> train
        + 20% thời gian cuối của phần học -> val
  - line_no >= SPLIT_LINE (10–20%)        -> test

Cách chạy (trong thư mục N3.10):
  python mvp_step2_windows.py

QUY ƯỚC TÊN CỘT
  f_...    đặc trưng (feature)  -> mô hình ĐƯỢC dùng
  gt_...   thông tin từ nhãn    -> CHỈ dùng để chấm điểm / phân tích, KHÔNG đưa vào mô hình
  y        nhãn cửa sổ: 1 nếu có ít nhất 1 dòng alert
"""
import argparse
import os

import numpy as np
import pandas as pd

LEVELS = ["INFO", "WARNING", "SEVERE", "ERROR", "FATAL", "FAILURE"]


def build_windows(d, win_sec, components):
    """Gộp các dòng log của MỘT phần (học hoặc test) thành cửa sổ, kể cả cửa sổ trống."""
    d = d.copy()
    d["win"] = d["ts_epoch"] - d["ts_epoch"] % win_sec
    g = d.groupby("win")

    out = pd.DataFrame({
        "f_n_logs": g.size(),
        "f_n_nodes": g["node"].nunique(),
        "f_n_components": g["component"].nunique(),
        "gt_alert_lines": g["gt_is_alert"].sum(),
    })
    sev = pd.crosstab(d["win"], d["severity_raw"])
    for lv in LEVELS:
        out[f"f_sev_{lv}"] = sev[lv] if lv in sev.columns else 0
    out["f_sev_other"] = sev.drop(columns=[c for c in LEVELS if c in sev.columns]).sum(axis=1) \
        if len(sev.columns) else 0
    comp = pd.crosstab(d["win"], d["component"])
    for c in components:
        out[f"f_comp_{c}"] = comp[c] if c in comp.columns else 0

    types = (d[d["gt_is_alert"] == 1].groupby("win")["gt_alert_type"]
             .agg(lambda s: "|".join(sorted(set(s)))))
    out["gt_alert_types"] = types

    # Thêm các cửa sổ không có log (n_logs = 0) để chuỗi thời gian liên tục
    full = np.arange(out.index.min(), out.index.max() + win_sec, win_sec)
    out = out.reindex(full)
    num_cols = [c for c in out.columns if c != "gt_alert_types"]
    out[num_cols] = out[num_cols].fillna(0).astype("int64")
    out["gt_alert_types"] = out["gt_alert_types"].fillna("")
    out.index.name = "ts_epoch"
    return out.reset_index()


def types_of(s):
    t = set()
    for v in s:
        if v:
            t.update(v.split("|"))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join("data", "processed", "bgl_first20.csv"))
    ap.add_argument("--output", default=os.path.join("data", "features", "bgl_mvp_windows.csv"))
    ap.add_argument("--report", default=os.path.join("reports", "mvp", "step2_summary.txt"))
    ap.add_argument("--split-line", type=int, default=474_796,
                    help="line_no đầu tiên của phần test (mặc định = mốc 10%% của BGL)")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="tỷ lệ thời gian CUỐI của phần học dùng làm val")
    ap.add_argument("--window-min", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    win_sec = args.window_min * 60

    print(f"Đang đọc {args.input} ...")
    df = pd.read_csv(
        args.input, keep_default_na=False,            # giữ nguyên chữ "NULL"
        usecols=["line_no", "ts_epoch", "node", "component", "severity_raw",
                 "gt_alert_type", "gt_is_alert"],
        dtype={"line_no": "int64", "ts_epoch": "int64", "gt_is_alert": "int8",
               "node": "string", "component": "string", "severity_raw": "string",
               "gt_alert_type": "string"})
    print(f"  {len(df):,} dòng")

    components = sorted(df["component"].unique())
    learn = df[df["line_no"] < args.split_line]
    test = df[df["line_no"] >= args.split_line]
    if len(learn) == 0 or len(test) == 0:
        raise SystemExit("Phần học hoặc phần test rỗng: kiểm tra lại --split-line và file đầu vào.")

    w_learn = build_windows(learn, win_sec, components)
    w_test = build_windows(test, win_sec, components)

    # Chia phần học thành train / val theo THỜI GIAN
    t0, t1 = w_learn["ts_epoch"].min(), w_learn["ts_epoch"].max()
    cut = t0 + (t1 - t0) * (1 - args.val_frac)
    w_learn["split"] = np.where(w_learn["ts_epoch"] < cut, "train", "val")
    w_test["split"] = "test"

    w = pd.concat([w_learn, w_test], ignore_index=True)
    w["y"] = (w["gt_alert_lines"] > 0).astype("int8")
    w.insert(1, "ts_utc",
             pd.to_datetime(w["ts_epoch"], unit="s").dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    front = ["ts_epoch", "ts_utc", "split", "y"]
    w = w[front + [c for c in w.columns if c not in front]]
    w.to_csv(args.output, index=False)

    # ---------- Thống kê ----------
    out = []
    p = out.append
    p("===== MVP BƯỚC 2: CỬA SỔ 5 PHÚT + ĐẶC TRƯNG BỘ A =====")
    p(f"Đầu vào: {args.input}  ({len(df):,} dòng)")
    p(f"Mốc chia học/test: line_no {args.split_line:,}")
    p(f"Số đặc trưng (f_...): {sum(c.startswith('f_') for c in w.columns)}")
    p("")
    p(f"{'phần':<6} {'từ':<20} {'đến':<20} {'cửa sổ':>7} {'có log':>7} {'alert':>6} {'%alert':>7}")
    types = {}
    for s in ["train", "val", "test"]:
        x = w[w["split"] == s]
        types[s] = types_of(x["gt_alert_types"])
        p(f"{s:<6} {x.ts_utc.min():<20} {x.ts_utc.max():<20} "
          f"{len(x):>7,} {(x.f_n_logs > 0).sum():>7,} {x.y.sum():>6,} {x.y.mean():>7.2%}")
    p("")
    for s in ["train", "val", "test"]:
        p(f"Loại alert trong {s:<5}: {', '.join(sorted(types[s])) or '(không có)'}")
    seen = types["train"] | types["val"]
    p("")
    p(f"Loại CHỈ có trong test (chưa gặp khi học): {', '.join(sorted(types['test'] - seen)) or '(không có)'}")
    new_win = w[(w.split == "test") & (w.y == 1)]
    only_new = new_win["gt_alert_types"].apply(lambda v: set(v.split("|")).isdisjoint(seen))
    p(f"Cửa sổ test chỉ chứa loại chưa gặp: {only_new.sum()} / {len(new_win)} cửa sổ alert")
    p("")
    p(f"Đã ghi: {args.output}")

    text = "\n".join(out)
    print("\n" + text)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Đã lưu thống kê: {args.report}")


if __name__ == "__main__":
    main()
