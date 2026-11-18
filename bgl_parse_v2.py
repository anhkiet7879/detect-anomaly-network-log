"""
bgl_parse_v2.py — Parser BGL phiên bản 2: chuyển BGL.log (txt) sang CSV theo schema 12 cột
đã chốt ngày 17/11/2026 (xem tài liệu Project "claude/bgl-schema-quyet-dinh.md"),
kèm BÁO CÁO CHẤT LƯỢNG.

KHÔNG sửa file gốc. Dòng không khớp regex được đưa vào file cách ly.

------------------------------------------------------------------
CÁCH CHẠY (Windows, trong thư mục N3.10)
  Toàn bộ file (mặc định):
    python bgl_parse_v2.py --log data\\raw\\bgl\\BGL\\BGL.log
  Một đoạn, ví dụ 20% đầu:
    python bgl_parse_v2.py --log data\\raw\\bgl\\BGL\\BGL.log --fraction 0.2
  Đoạn 10–20%:
    python bgl_parse_v2.py --log data\\raw\\bgl\\BGL\\BGL.log --start 0.1 --fraction 0.1

ĐẦU RA
  data\\processed\\<tên>.csv             dữ liệu sạch, 12 cột
  data\\processed\\<tên>_quarantine.txt  dòng không parse được (số dòng <TAB> nội dung)
  reports\\parser\\<tên>_quality.txt      báo cáo chất lượng
  <tên> = bgl_parsed (toàn bộ) | bgl_first20 | bgl_10to20 ...

ĐỌC FILE CSV BẰNG PANDAS: luôn dùng keep_default_na=False (để giữ chữ "NULL").

------------------------------------------------------------------
SCHEMA 12 CỘT
  1  line_no        số thứ tự dòng trong file gốc (đếm từ 0)
  2  source         luôn "bgl"
  3  ts_epoch       giây UTC (trường 2 của log)
  4  ts_utc         ISO 8601 có Z, vd 2005-06-05T15:31:52Z
  5  node           vị trí phần cứng, nguyên văn (trường 4)
  6  component      bộ phận ghi log, nguyên văn (trường 8)
  7  log_type       RAS / NULL, nguyên văn (trường 7)
  8  severity_raw   mức gốc, nguyên văn (trường 9)
  9  severity_norm  thang syslog 0–7 (bảng SEVERITY_MAP); trống nếu mức lạ
  10 message        nội dung, nguyên văn, có thể rỗng (trường 10)
  11 gt_alert_type  tên loại alert, hoặc NORMAL  (ĐÁP ÁN — không dùng làm đặc trưng)
  12 gt_is_alert    1 = alert, 0 = bình thường   (ĐÁP ÁN — không dùng làm đặc trưng)
  Bỏ: Date (trường 3), Time (trường 5), NodeRepeat (trường 6).
"""
import argparse
import csv
import os
import re
from collections import Counter
from datetime import datetime, timezone

# ---------------------------------------------------------------- cấu hình
RX = re.compile(
    r"^(?P<label>\S+) (?P<epoch>\d{10}) (?P<date>\d{4}\.\d{2}\.\d{2}) (?P<node>\S+) "
    r"(?P<time>\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d{6}) "
    r"(?:(?P<node_repeat>\S+) )?(?P<type>RAS|NULL) (?P<component>[A-Z_]+) "
    r"(?P<level>[A-Z]+)(?: (?P<content>.*))?$")

COLUMNS = ["line_no", "source", "ts_epoch", "ts_utc", "node", "component", "log_type",
           "severity_raw", "severity_norm", "message", "gt_alert_type", "gt_is_alert"]

# Bảng ánh xạ BGL -> syslog (RFC 5424). SEVERE/FATAL/FAILURE là GIẢ ĐỊNH (không có tài liệu IBM).
SEVERITY_MAP = {"INFO": 6, "WARNING": 4, "ERROR": 3, "SEVERE": 2, "FATAL": 2, "FAILURE": 1}

UNKNOWN_NODES = {"NULL", "UNKNOWN_LOCATION", "-"}
RX_LINK = re.compile(r"-L\d")


def card_type(node):
    """Loại card suy từ node — CHỈ dùng cho báo cáo chất lượng, không lưu thành cột."""
    if node in UNKNOWN_NODES:
        return "unknown"
    if "-C:" in node:
        return "compute"
    if "-I:" in node:
        return "io"
    if RX_LINK.search(node):
        return "link"
    return "other"          # vd chỉ đến cấp node card: R33-M1-ND


def iso_utc(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_offset_hours(epoch, t):
    """Độ lệch (giờ) giữa giờ UTC từ epoch và giờ địa phương ở trường Time.
    t dạng 2005-06-05-08.31.52.392149 -> kỳ vọng 7 (PDT) hoặc 8 (PST)."""
    local = datetime(int(t[0:4]), int(t[5:7]), int(t[8:10]),
                     int(t[11:13]), int(t[14:16]), int(t[17:19]))
    utc = datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None)
    return (utc - local).total_seconds() / 3600


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


def table(counter_2d, rows, cols, title_row, width=11):
    """In bảng chéo từ Counter có khóa (row, col)."""
    head = f"  {title_row:<12}" + "".join(f"{c:>{width}}" for c in cols)
    out = [head]
    for r in rows:
        out.append(f"  {r:<12}" + "".join(f"{counter_2d[(r, c)]:>{width},}" for c in cols))
    return out


# ---------------------------------------------------------------- chính
def main():
    ap = argparse.ArgumentParser(description="Parser BGL v2 -> CSV 12 cột + báo cáo chất lượng")
    ap.add_argument("--log", required=True, help="đường dẫn BGL.log")
    ap.add_argument("--start", type=float, default=0.0, help="vị trí bắt đầu theo tỷ lệ (0.1 = mốc 10%%)")
    ap.add_argument("--fraction", type=float, default=1.0, help="tỷ lệ số dòng cần lấy (mặc định 1.0 = toàn bộ)")
    ap.add_argument("--out-dir", default=os.path.join("data", "processed"))
    ap.add_argument("--report-dir", default=os.path.join("reports", "parser"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    print("Đang đếm số dòng của file gốc...")
    total_file = count_lines(args.log)
    i_start = int(total_file * args.start)
    i_end = min(total_file, int(total_file * (args.start + args.fraction)))
    a, b = round(args.start * 100), round((args.start + args.fraction) * 100)
    name = "bgl_parsed" if (a == 0 and b >= 100) else (f"bgl_first{b}" if a == 0 else f"bgl_{a}to{b}")
    out_csv = os.path.join(args.out_dir, f"{name}.csv")
    out_q = os.path.join(args.out_dir, f"{name}_quarantine.txt")
    out_rep = os.path.join(args.report_dir, f"{name}_quality.txt")
    print(f"  file gốc {total_file:,} dòng -> lấy dòng {i_start:,} đến {i_end - 1:,} ({i_end - i_start:,} dòng)")

    # bộ đếm cho báo cáo chất lượng
    n_read = n_ok = n_bad = 0
    n_empty_msg = n_no_repeat = n_repeat_diff = 0
    n_backward = 0
    prev_epoch = None
    first_epoch = last_epoch = None
    offsets = Counter()
    n_offset_odd = 0
    sev = Counter()
    sev_alert = Counter()
    n_sev_unknown = 0
    alert_types = Counter()
    type_comp = Counter()
    comp_card = Counter()
    comps, types = Counter(), Counter()
    repeat_diff_examples = []

    with open(args.log, encoding="utf-8", errors="replace") as f, \
         open(out_csv, "w", newline="", encoding="utf-8") as fo, \
         open(out_q, "w", encoding="utf-8") as fq:
        w = csv.writer(fo)
        w.writerow(COLUMNS)
        for i, raw in enumerate(f):
            if i < i_start:
                continue
            if i >= i_end:
                break
            n_read += 1
            line = raw.rstrip("\r\n")
            m = RX.match(line)
            if not m:
                n_bad += 1
                fq.write(f"{i}\t{line}\n")
                continue
            n_ok += 1

            ep = int(m["epoch"])
            node = m["node"]
            comp = m["component"]
            ltype = m["type"]
            level = m["level"]
            msg = m["content"] or ""
            label = m["label"]
            is_alert = 0 if label == "-" else 1
            norm = SEVERITY_MAP.get(level, "")

            w.writerow([i, "bgl", ep, iso_utc(ep), node, comp, ltype, level, norm, msg,
                        label if is_alert else "NORMAL", is_alert])

            # ---- kiểm tra chất lượng
            if first_epoch is None:
                first_epoch = ep
            last_epoch = ep
            if prev_epoch is not None and ep < prev_epoch:
                n_backward += 1
            prev_epoch = ep
            if not msg:
                n_empty_msg += 1
            rep = m["node_repeat"]
            if rep is None:
                n_no_repeat += 1
            elif rep != node:
                n_repeat_diff += 1
                if len(repeat_diff_examples) < 5:
                    repeat_diff_examples.append(f"dòng {i}: node={node} | node_repeat={rep}")
            off = local_offset_hours(ep, m["time"])
            off_r = round(off)
            if abs(off - off_r) > 0.01:          # lệch không tròn giờ (> ~36 giây)
                n_offset_odd += 1
            offsets[off_r] += 1
            sev[level] += 1
            if norm == "":
                n_sev_unknown += 1
            if is_alert:
                sev_alert[level] += 1
                alert_types[label] += 1
            comps[comp] += 1
            types[ltype] += 1
            type_comp[(comp, ltype)] += 1
            comp_card[(comp, card_type(node))] += 1

            if n_read % 1_000_000 == 0:
                print(f"  đã đọc {n_read:,} dòng...")

    # ---------------------------------------------------------------- báo cáo
    total_alert = sum(alert_types.values())
    out = []
    p = out.append
    p(f"===== BÁO CÁO CHẤT LƯỢNG PARSER BGL v2 — {name} =====")
    p(f"File gốc        : {args.log}  ({total_file:,} dòng)")
    p(f"Đoạn lấy        : dòng {i_start:,} đến {i_end - 1:,}")
    p(f"Đầu ra          : {out_csv}")
    p("")
    p("--- 1. Đọc dữ liệu ---")
    p(f"Số dòng đọc       : {n_read:,}")
    p(f"Parse được        : {n_ok:,}  ({n_ok / max(n_read, 1):.4%})")
    p(f"Cách ly           : {n_bad:,}  -> {out_q}")
    p(f"Message rỗng      : {n_empty_msg:,}")
    p(f"Thiếu NodeRepeat  : {n_no_repeat:,}")
    p(f"NodeRepeat ≠ Node : {n_repeat_diff:,}")
    for e in repeat_diff_examples:
        p(f"    vd {e}")
    p("")
    p("--- 2. Thời gian ---")
    if first_epoch is not None:
        p(f"Dòng đầu / cuối   : {iso_utc(first_epoch)}  ->  {iso_utc(last_epoch)}")
    p(f"Epoch lùi lại     : {n_backward:,} dòng")
    p("Độ lệch Time–epoch (giờ, làm tròn) — kỳ vọng 7 (PDT, mùa hè) hoặc 8 (PST, mùa đông):")
    for k, v in sorted(offsets.items()):
        p(f"    {k:>3} giờ : {v:>11,} dòng")
    p(f"Lệch không tròn giờ (>36 giây): {n_offset_odd:,} dòng")
    p("")
    p("--- 3. Severity ---")
    p(f"  {'severity_raw':<12} {'norm':>5} {'số dòng':>11} {'là alert':>10}")
    for lv, c in sev.most_common():
        p(f"  {lv:<12} {str(SEVERITY_MAP.get(lv, '?')):>5} {c:>11,} {sev_alert[lv]:>10,}")
    p(f"Mức không có trong bảng ánh xạ (severity_norm trống): {n_sev_unknown:,} dòng")
    p("")
    p("--- 4. Nhãn ---")
    p(f"Dòng alert        : {total_alert:,}  ({total_alert / max(n_ok, 1):.2%})   [toàn file kỳ vọng 348.460]")
    p(f"Số loại alert     : {len(alert_types)}")
    for t, c in alert_types.most_common():
        p(f"    {t:<12} {c:>9,}")
    p("")
    p("--- 5. Bảng chéo component × log_type ---")
    out += table(type_comp, [c for c, _ in comps.most_common()], [t for t, _ in types.most_common()],
                 "component")
    p("")
    p("--- 6. Bảng chéo component × loại card (suy từ node, không lưu cột) ---")
    cards = ["compute", "io", "link", "other", "unknown"]
    out += table(comp_card, [c for c, _ in comps.most_common()], cards, "component")

    text = "\n".join(out)
    print("\n" + text)
    with open(out_rep, "w", encoding="utf-8") as fr:
        fr.write(text + "\n")
    print(f"\nĐã lưu báo cáo chất lượng: {out_rep}")


if __name__ == "__main__":
    main()
