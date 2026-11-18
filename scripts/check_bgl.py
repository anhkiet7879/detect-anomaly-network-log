"""
check_bgl.py - Danh gia bo log BGL (Loghub) truoc khi dua vao pipeline.

Cach chay (tai thu muc du an):
    python scripts\\check_bgl.py data\\raw\\bgl\\BGL\\BGL.log

Doc tung dong (streaming), khong nap ca file vao RAM. Mat khoang 2-4 phut.

Tra loi cac cau hoi:
  [1] Quy mo: so dong, dong loi dinh dang, dong khong doc duoc UTF-8.
  [2] Nhan: ty le bat thuong, cac loai nhan.
  [3] Level (severity goc) so voi nhan -> neu dung luat "FATAL la bat thuong" thi duoc gi?
  [4] Thoi gian: khoang thoi gian, ngay co du lieu, khoang trong lon, dong lech thu tu.
  [5] Mui gio: do lech giua cot Epoch va cot Time (chuoi ngay gio).
  [6] Nguon: so node, muc do tap trung cua bat thuong theo node.
  [7] Bat thuong theo "su kien": gom cac dong bat thuong gan nhau thanh dot (episode).
  [8] Theo cua so thoi gian 1 gio / 5 phut: bao nhieu % cua so co bat thuong.
  [9] Dong trung lap lien tiep (cung node + cung noi dung).

Dau ra:
  reports\\bgl_summary.txt  : ban tom tat in ra man hinh.
  reports\\bgl_daily.csv    : so dong / so dong bat thuong theo ngay.
  reports\\bgl_episodes.csv : danh sach cac dot bat thuong.
"""
import sys
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

EPISODE_GAP_S = 300  # 2 dong bat thuong cach nhau <= 5 phut -> cung 1 dot


def utc(t: int) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def main(path_str: str) -> None:
    path = Path(path_str)
    if not path.exists():
        sys.exit(f"Khong tim thay file: {path}")

    total = bad = non_utf8 = out_of_order = dup_consecutive = 0
    labels = Counter()
    level_x = defaultdict(lambda: [0, 0])      # level -> [binh thuong, bat thuong]
    daily = defaultdict(lambda: [0, 0])        # ngay -> [tong, bat thuong]
    node_total, node_anom = Counter(), Counter()
    tz_offsets = Counter()                     # do lech (gio) giua Epoch va cot Time
    hours_all, hours_anom = set(), set()
    m5_all, m5_anom = set(), set()
    t_min = t_max = prev_t = None
    prev_key = None
    gaps = []                                  # (do dai khoang trong, tu, den)

    episodes, ep = [], None                    # dot bat thuong dang mo

    with path.open("rb") as f:
        for raw in f:
            total += 1
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                non_utf8 += 1
                line = raw.decode("utf-8", errors="replace")
            parts = line.rstrip("\r\n").split(" ", 9)
            # Label Epoch Date Node Time NodeRepeat Type Component Level Content
            if len(parts) < 9 or not parts[1].isdigit():
                bad += 1
                continue
            label, t, node, tstr, level = parts[0], int(parts[1]), parts[3], parts[4], parts[8]
            content = parts[9] if len(parts) > 9 else ""
            is_anom = label != "-"

            labels[label] += 1
            level_x[level][is_anom] += 1
            node_total[node] += 1
            day = utc(t)[:10]
            daily[day][0] += 1
            hours_all.add(t // 3600)
            m5_all.add(t // 300)

            # Mui gio: so cot Epoch voi chuoi Time (vd 2005-06-03-15.42.50.675872)
            try:
                local = datetime.strptime(tstr[:19], "%Y-%m-%d-%H.%M.%S").replace(tzinfo=timezone.utc)
                tz_offsets[round((local.timestamp() - t) / 3600)] += 1
            except ValueError:
                tz_offsets["khong_doc_duoc"] += 1

            key = (node, content)
            if key == prev_key:
                dup_consecutive += 1
            prev_key = key

            if prev_t is not None:
                if t < prev_t:
                    out_of_order += 1
                elif t - prev_t > 3600:
                    gaps.append((t - prev_t, prev_t, t))
            prev_t = t
            t_min = t if t_min is None else min(t_min, t)
            t_max = t if t_max is None else max(t_max, t)

            if is_anom:
                daily[day][1] += 1
                node_anom[node] += 1
                hours_anom.add(t // 3600)
                m5_anom.add(t // 300)
                if ep and 0 <= t - ep["end"] <= EPISODE_GAP_S:
                    ep["end"] = t
                    ep["lines"] += 1
                    ep["nodes"].add(node)
                    ep["labels"][label] += 1
                else:
                    if ep:
                        episodes.append(ep)
                    ep = {"start": t, "end": t, "lines": 1, "nodes": {node}, "labels": Counter({label: 1})}
    if ep:
        episodes.append(ep)

    valid = total - bad
    n_anom = valid - labels["-"]
    out = []
    p = out.append

    p("[1] QUY MO")
    p(f"  Tong so dong              : {total:,}")
    p(f"  Dong loi dinh dang        : {bad:,}")
    p(f"  Dong co byte khong UTF-8  : {non_utf8:,}")

    p("\n[2] NHAN")
    p(f"  Dong bat thuong           : {n_anom:,} ({n_anom / valid:.2%})")
    p(f"  So loai nhan bat thuong   : {len(labels) - 1}")
    for lab, c in labels.most_common(16):
        if lab != "-":
            p(f"    {lab:<12}{c:>12,}")

    p("\n[3] LEVEL (severity goc) x NHAN")
    p(f"  {'Level':<10}{'Binh thuong':>14}{'Bat thuong':>14}{'% bat thuong':>14}")
    for lvl, (n, a) in sorted(level_x.items(), key=lambda x: -sum(x[1])):
        p(f"  {lvl:<10}{n:>14,}{a:>14,}{a / (n + a):>14.1%}")
    tp = level_x.get("FATAL", [0, 0])[1]
    fp = level_x.get("FATAL", [0, 0])[0]
    if tp + fp:
        p(f"  -> Luat 'Level=FATAL la bat thuong' (theo dong): "
          f"Precision={tp / (tp + fp):.2%}, Recall={tp / max(n_anom, 1):.2%}")

    p("\n[4] THOI GIAN (theo cot Epoch, hien thi UTC)")
    p(f"  Tu {utc(t_min)} den {utc(t_max)} ({(t_max - t_min) / 86400:.1f} ngay)")
    span_days = (t_max - t_min) // 86400 + 1
    p(f"  So ngay co du lieu        : {len(daily)} / {span_days}")
    p(f"  Dong lech thu tu thoi gian: {out_of_order:,}")
    p(f"  Khoang trong > 1 gio      : {len(gaps):,}")
    for g, a, b in sorted(gaps, reverse=True)[:5]:
        p(f"    {g / 3600:8.1f} gio  tu {utc(a)} den {utc(b)}")

    p("\n[5] MUI GIO: do lech (gio) = cot Time - cot Epoch")
    for k, c in tz_offsets.most_common(5):
        p(f"    {str(k):>14} gio : {c:,} dong")

    p("\n[6] NGUON (node)")
    p(f"  So node                   : {len(node_total):,}")
    p(f"  So node co bat thuong     : {len(node_anom):,}")
    top10 = sum(c for _, c in node_anom.most_common(10))
    p(f"  10 node nhieu bat thuong nhat chiem {top10 / max(n_anom, 1):.1%} so dong bat thuong:")
    for nd, c in node_anom.most_common(5):
        p(f"    {nd:<28}{c:>10,}")

    p(f"\n[7] DOT BAT THUONG (gom cac dong bat thuong cach nhau <= {EPISODE_GAP_S // 60} phut)")
    durs = sorted(e["end"] - e["start"] for e in episodes)
    sizes = sorted(e["lines"] for e in episodes)
    q = lambda v, x: v[min(int(len(v) * x), len(v) - 1)]
    p(f"  So dot                    : {len(episodes):,}")
    if episodes:
        p(f"  Thoi luong (giay)  median : {q(durs, .5):,}   p90: {q(durs, .9):,}   max: {durs[-1]:,}")
        p(f"  So dong / dot      median : {q(sizes, .5):,}   p90: {q(sizes, .9):,}   max: {sizes[-1]:,}")
        p(f"  Dot chi co 1 dong         : {sum(s == 1 for s in sizes):,}")
        big = sorted(episodes, key=lambda e: -e["lines"])[:3]
        p("  3 dot lon nhat chiem "
          f"{sum(e['lines'] for e in big) / n_anom:.1%} so dong bat thuong")

    p("\n[8] THEO CUA SO THOI GIAN (chi tinh cua so co it nhat 1 dong log)")
    p(f"  Cua so 1 gio  : {len(hours_all):,}, co bat thuong {len(hours_anom):,} ({len(hours_anom) / len(hours_all):.1%})")
    p(f"  Cua so 5 phut : {len(m5_all):,}, co bat thuong {len(m5_anom):,} ({len(m5_anom) / len(m5_all):.1%})")

    p("\n[9] TRUNG LAP")
    p(f"  Dong lap lai lien tiep (cung node + noi dung): {dup_consecutive:,} ({dup_consecutive / valid:.1%})")

    report = "\n".join(out)
    print(report)

    od = Path("reports")
    od.mkdir(exist_ok=True)
    (od / "bgl_summary.txt").write_text(report, encoding="utf-8")
    with (od / "bgl_daily.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "total_lines", "anomalous_lines"])
        for d in sorted(daily):
            w.writerow([d, *daily[d]])
    with (od / "bgl_episodes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["start_utc", "end_utc", "duration_s", "lines", "n_nodes", "main_label"])
        for e in episodes:
            w.writerow([utc(e["start"]), utc(e["end"]), e["end"] - e["start"], e["lines"],
                        len(e["nodes"]), e["labels"].most_common(1)[0][0]])
    print(f"\nDa ghi: reports\\bgl_summary.txt, reports\\bgl_daily.csv, reports\\bgl_episodes.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Cach dung: python scripts\\check_bgl.py <duong_dan_BGL.log>")
    main(sys.argv[1])
