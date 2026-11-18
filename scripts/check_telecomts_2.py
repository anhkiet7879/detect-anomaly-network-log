"""
check_telecomts_2.py - Kiem tra bo sung cho TelecomTS (chay SAU check_telecomts.py).

Tra loi 3 cau hoi:
  [A] Cac cua so co chong lan nhau khong? (buoc truot giua 2 cua so lien tiep)
  [B] Cot nao bi thieu du lieu, va thieu theo ung dung nao?
  [C] Cua so bat thuong co khop vao thoi gian cua file tho khong?
      Neu co: bao nhieu % so dong trong file tho bat thuong la bat thuong that?

Cach chay (tai thu muc du an):
    python scripts\\check_telecomts_2.py data\\raw\\telecomts
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

STEP = pd.Timedelta(milliseconds=100)


def main(root_str: str) -> None:
    root = Path(root_str)
    w_path = Path("reports") / "telecomts_windows.csv"
    if not w_path.exists():
        sys.exit("Chua co reports\\telecomts_windows.csv -> hay chay check_telecomts.py truoc.")
    w = pd.read_csv(w_path, parse_dates=["start_time"])

    # [A] Buoc truot giua cac cua so lien tiep trong cung kich ban
    print("[A] Buoc truot giua 2 cua so lien tiep (don vi: so buoc 100 ms)")
    rows = []
    for sc, g in w.groupby("scenario"):
        d = g["start_time"].sort_values().diff().dropna() / STEP
        rows.append({"scenario": sc, "median_stride": d.median(), "min_stride": d.min(),
                     "pct_overlap": round((d < 128).mean() * 100, 1)})
    a = pd.DataFrame(rows)
    print(a.groupby(a["scenario"].str.split("/").str[0])[["median_stride", "pct_overlap"]].describe().round(1).to_string())
    print("  -> stride < 128 nghia la 2 cua so lien tiep dung chung du lieu.\n")

    # [B] Cot bi thieu theo ung dung
    print("[B] Ty le o trong (%) theo cot, gop theo ung dung (chi hien cot co thieu)")
    miss = []
    for f in sorted(root.rglob("metrics.csv")):
        df = pd.read_csv(f)
        app = f.parent.parent.name
        miss.append(df.drop(columns="timestamp").isna().mean().mul(100).rename(app))
    m = pd.concat(miss, axis=1).T.groupby(level=0).mean().round(1)
    print(m.loc[:, (m > 0.5).any()].to_string(), "\n")

    # [C] Doi chieu cua so bat thuong voi file tho
    print("[C] Doi chieu cua so bat thuong voi file tho metrics.csv")
    res = []
    for sc, g in w[w["folder"] == "anomalous"].groupby("scenario"):
        raw = pd.read_csv(root / sc / "raw" / "metrics.csv", usecols=["timestamp"])
        ts = pd.to_datetime(raw["timestamp"])
        label = np.zeros(len(ts), dtype=bool)
        inside = 0
        for _, r in g.iterrows():
            s = r["start_time"] + int(r["a_start"]) * STEP
            e = r["start_time"] + int(r["a_end"]) * STEP
            if ts.iloc[0] <= r["start_time"] <= ts.iloc[-1]:
                inside += 1
            label |= ((ts >= s) & (ts <= e)).to_numpy()
        res.append({"scenario": sc, "windows": len(g),
                    "win_start_in_raw_range": f"{inside}/{len(g)}",
                    "raw_t_start": ts.iloc[0], "win_t_start_min": g["start_time"].min(),
                    "pct_raw_rows_anomalous": round(label.mean() * 100, 1)})
    print(pd.DataFrame(res).to_string(index=False))
    print("  -> Neu win_start_in_raw_range = N/N va pct < 100: file tho co ca doan binh thuong,")
    print("     co the gan nhan tung thoi diem cho chuoi lien tuc.")
    print("  -> Neu 0/N: thoi gian cua so va file tho khong cung he quy chieu, KHONG ghep duoc truc tiep.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Cach dung: python scripts\\check_telecomts_2.py <thu_muc_telecomts>")
    main(sys.argv[1])
