"""
bgl_episodes.py — [H2] Trích xuất danh sách ĐỢT SỰ CỐ (episode) từ CSV đã parse.

Đợt sự cố = chuỗi các dòng alert (gt_is_alert = 1) liên tiếp mà dòng sau cách dòng
trước <= GAP giây (mặc định 900 giây = 15 phút, theo hợp đồng giao tiếp số 2).
Dòng alert cách dòng alert trước > GAP  ->  mở đợt mới.

Dùng cho: bộ chia dữ liệu của Kiệt (cắt theo episode, không cắt đôi một đợt)
và chấm điểm theo sự kiện của Khang.

------------------------------------------------------------------
CÁCH CHẠY (Windows, trong thư mục N3.10)
    python src\\parser\\bgl_episodes.py
    python src\\parser\\bgl_episodes.py --input data\\processed\\bgl_first20.csv --gap-mins 15

ĐẦU RA
    data\\processed\\bgl_episodes.csv               (hoặc <tên>_episodes.csv nếu input khác)
    reports\\parser\\bgl_episodes_quality.txt         thống kê phân bố đợt

CỘT CỦA FILE EPISODES
    4 cột bắt buộc theo hợp đồng:
      episode_id     số thứ tự đợt, đếm từ 0, tăng dần theo thời gian
      start_ts       epoch (giây UTC) của dòng alert đầu tiên trong đợt
      end_ts         epoch của dòng alert cuối cùng trong đợt
      anomaly_count  số dòng alert trong đợt
    Cột bổ sung (chỉ để đọc / báo cáo, bộ chia không cần):
      start_utc, end_utc, duration_s, n_alert_types, main_alert_type, alert_types
------------------------------------------------------------------
Bộ nhớ: chỉ đọc 3 cột (ts_epoch, gt_is_alert, gt_alert_type) theo khối 1 triệu dòng
và chỉ giữ lại các dòng alert (~348 nghìn dòng) -> vài chục MB RAM.
"""
import argparse
import os
from collections import Counter
from datetime import datetime, timezone

import pandas as pd


def iso_utc(epoch):
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_alert_rows(path, chunksize=1_000_000):
    """Đọc CSV theo khối, trả về DataFrame chỉ gồm các dòng alert (ts_epoch, gt_alert_type)."""
    parts, n_total = [], 0
    for chunk in pd.read_csv(path, usecols=["ts_epoch", "gt_is_alert", "gt_alert_type"],
                             keep_default_na=False, chunksize=chunksize,
                             dtype={"ts_epoch": "int64", "gt_is_alert": "int8", "gt_alert_type": "str"}):
        n_total += len(chunk)
        parts.append(chunk.loc[chunk["gt_is_alert"] == 1, ["ts_epoch", "gt_alert_type"]])
        print(f"  đã đọc {n_total:,} dòng...")
    alerts = pd.concat(parts, ignore_index=True)
    return alerts, n_total


def build_episodes(alerts, gap_s):
    """Gom dòng alert thành đợt. alerts phải sắp theo thời gian (CSV giữ thứ tự file gốc)."""
    if alerts.empty:
        return pd.DataFrame(columns=["episode_id", "start_ts", "end_ts", "anomaly_count"])
    # Kiểm tra an toàn: parser đã xác nhận epoch không lùi, nhưng input có thể là file khác.
    if not alerts["ts_epoch"].is_monotonic_increasing:
        print("  CẢNH BÁO: ts_epoch không tăng dần -> sắp xếp lại trước khi gom đợt")
        alerts = alerts.sort_values("ts_epoch", kind="stable").reset_index(drop=True)

    gap = alerts["ts_epoch"].diff()                       # dòng đầu: NaN
    alerts = alerts.assign(episode_id=(gap.isna() | (gap > gap_s)).cumsum() - 1)

    g = alerts.groupby("episode_id")
    ep = pd.DataFrame({
        "start_ts": g["ts_epoch"].min(),
        "end_ts": g["ts_epoch"].max(),
        "anomaly_count": g.size(),
    }).reset_index()

    types = g["gt_alert_type"].agg(lambda s: Counter(s).most_common())
    ep["start_utc"] = ep["start_ts"].map(iso_utc)
    ep["end_utc"] = ep["end_ts"].map(iso_utc)
    ep["duration_s"] = ep["end_ts"] - ep["start_ts"]
    ep["n_alert_types"] = types.map(len).values
    ep["main_alert_type"] = types.map(lambda t: t[0][0]).values
    ep["alert_types"] = types.map(lambda t: "|".join(f"{k}:{v}" for k, v in t)).values
    return ep


def quality_report(ep, n_total, n_alert, gap_s, src, out_csv):
    q = lambda s, x: s.quantile(x)
    lines = [
        f"===== BÁO CÁO ĐỢT SỰ CỐ (EPISODES) — {os.path.basename(src)} =====",
        f"Input             : {src}  ({n_total:,} dòng)",
        f"Output            : {out_csv}",
        f"Quy tắc gom       : 2 dòng alert liên tiếp cách nhau <= {gap_s} giây ({gap_s // 60} phút) -> cùng đợt",
        "",
        f"Dòng alert        : {n_alert:,}",
        f"Số đợt            : {len(ep):,}",
    ]
    if len(ep):
        d, c = ep["duration_s"], ep["anomaly_count"]
        big = c.nlargest(3)
        lines += [
            f"Kiểm tra tổng     : tổng anomaly_count = {c.sum():,}  "
            f"({'KHỚP' if c.sum() == n_alert else 'KHÔNG KHỚP'} số dòng alert)",
            f"Thời lượng (giây) : trung vị {q(d, .5):,.0f} | p90 {q(d, .9):,.0f} | max {d.max():,}",
            f"Số dòng / đợt     : trung vị {q(c, .5):,.0f} | p90 {q(c, .9):,.0f} | max {c.max():,}",
            f"Đợt chỉ có 1 dòng : {(c == 1).sum():,}",
            f"Đợt có >1 loại alert: {(ep['n_alert_types'] > 1).sum():,}",
            f"3 đợt lớn nhất chiếm {big.sum() / n_alert:.1%} số dòng alert:",
        ]
        for i in big.index:
            r = ep.loc[i]
            lines.append(f"    đợt {r.episode_id:>4}: {r.start_utc} -> {r.end_utc}, "
                         f"{r.anomaly_count:,} dòng, chính: {r.main_alert_type}")
        lines += ["", "Số đợt theo loại alert chính:"]
        for t, n in ep["main_alert_type"].value_counts().items():
            lines.append(f"    {t:<12} {n:>5}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="[H2] Trích xuất đợt sự cố BGL từ CSV đã parse")
    ap.add_argument("--input", default=os.path.join("data", "processed", "bgl_parsed.csv"))
    ap.add_argument("--gap-mins", type=float, default=15, help="khoảng cách tối đa giữa 2 dòng alert cùng đợt")
    ap.add_argument("--out-dir", default=os.path.join("data", "processed"))
    ap.add_argument("--report-dir", default=os.path.join("reports", "parser"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.input))[0]
    name = "bgl_episodes" if stem == "bgl_parsed" else f"{stem}_episodes"
    out_csv = os.path.join(args.out_dir, f"{name}.csv")
    out_rep = os.path.join(args.report_dir, f"{name}_quality.txt")
    gap_s = int(args.gap_mins * 60)

    print(f"Đọc {args.input} ...")
    alerts, n_total = load_alert_rows(args.input)
    ep = build_episodes(alerts, gap_s)
    ep.to_csv(out_csv, index=False)

    text = quality_report(ep, n_total, len(alerts), gap_s, args.input, out_csv)
    print("\n" + text)
    with open(out_rep, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nĐã lưu: {out_csv}\n        {out_rep}")


if __name__ == "__main__":
    main()
