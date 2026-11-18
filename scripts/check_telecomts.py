"""
check_telecomts.py - Danh gia nhanh bo TelecomTS truoc khi dua vao pipeline.

Cach chay (tai thu muc du an):
    python scripts\\check_telecomts.py data\\raw\\telecomts

Dau ra:
    - In ra man hinh 7 bang thong ke.
    - reports\\telecomts_windows.csv : 1 dong / 1 cua so (chi giu metadata va nhan,
      bo phan mo ta va hoi-dap), dung lai cho cac buoc sau.
    - reports\\telecomts_raw_files.csv: thong ke chat luong tung file metrics.csv.
"""
import sys
import json
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)


def read_windows(root: Path) -> pd.DataFrame:
    """Doc tat ca chunked.jsonl, moi dong JSON -> 1 dong metadata."""
    rows = []
    files = sorted(root.rglob("chunked.jsonl"))
    if not files:
        sys.exit(f"Khong tim thay chunked.jsonl nao trong {root}")
    for f in files:
        scenario = f.parent.parent.relative_to(root).as_posix()  # vd: normal/mobile/File
        folder = scenario.split("/")[0]                           # normal | anomalous
        with f.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  [Canh bao] Dong JSON loi: {f} dong {i + 1}")
                    continue
                kpis = r.get("KPIs") or {}
                lab = r.get("labels") or {}
                a = r.get("anomalies") or {}
                d = a.get("anomaly_duration") or {}
                rows.append({
                    "scenario": scenario,
                    "folder": folder,
                    "start_time": r.get("start_time"),
                    "end_time": r.get("end_time"),
                    "n_steps": len(next(iter(kpis.values()), [])),
                    "n_kpis": len(kpis),
                    "anomaly_present": lab.get("anomaly_present"),
                    "anomaly_exists": a.get("exists"),
                    "anomaly_type": a.get("type"),
                    "a_start": d.get("start"),
                    "a_end": d.get("end"),
                    "affected_kpis": "|".join(a.get("affected_kpis") or []),
                    "zone": lab.get("zone"),
                    "application": lab.get("application"),
                    "mobility": lab.get("mobility"),
                    "congestion": lab.get("congestion"),
                })
    return pd.DataFrame(rows)


def check_raw(root: Path) -> pd.DataFrame:
    """Kiem tra chat luong tung file metrics.csv (du lieu tho lien tuc)."""
    out = []
    for f in sorted(root.rglob("metrics.csv")):
        df = pd.read_csv(f)
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        step_ms = ts.diff().dt.total_seconds().mul(1000).round()
        out.append({
            "scenario": f.parent.parent.relative_to(root).as_posix(),
            "rows": len(df),
            "t_start": ts.min(),
            "t_end": ts.max(),
            "minutes": round((ts.max() - ts.min()).total_seconds() / 60, 1),
            "bad_timestamp": int(ts.isna().sum()),
            "dup_timestamp": int(ts.duplicated().sum()),
            "step_median_ms": step_ms.median(),
            "pct_step_not_100ms": round((step_ms.dropna() != 100).mean() * 100, 2),
            "max_gap_s": round(step_ms.max() / 1000, 1) if step_ms.notna().any() else None,
            "missing_cells": int(df.drop(columns="timestamp").isna().sum().sum()),
        })
    return pd.DataFrame(out)


def main(root_str: str) -> None:
    root = Path(root_str)
    if not root.exists():
        sys.exit(f"Khong tim thay thu muc: {root}")

    print("Dang doc chunked.jsonl (co the mat 1-3 phut)...")
    w = read_windows(root)
    w["a_len"] = pd.to_numeric(w["a_end"], errors="coerce") - pd.to_numeric(w["a_start"], errors="coerce") + 1
    w["full_window"] = w["a_len"] == w["n_steps"]

    print("\n[1] So cua so theo nhom thu muc")
    c = w["folder"].value_counts()
    print(c.to_string())
    print(f"Ty le cua so trong thu muc 'anomalous': {c.get('anomalous', 0) / len(w):.2%}")

    print("\n[2] Kiem tra nhat quan nhan: thu muc x anomaly_present x anomaly_exists")
    print(pd.crosstab([w["folder"], w["anomaly_present"]], w["anomaly_exists"].astype(str)).to_string())

    print("\n[3] Cac loai bat thuong (chi tinh cua so co anomaly_exists = True)")
    an = w[w["anomaly_exists"] == True]
    print(an["anomaly_type"].value_counts().to_string())

    print("\n[4] Do dai doan bat thuong trong cua so")
    print(f"  So cua so bat thuong            : {len(an):,}")
    print(f"  Bat thuong chiem TOAN BO cua so : {an['full_window'].sum():,} ({an['full_window'].mean():.2%})")
    print(an["a_len"].describe().round(1).to_string())

    print("\n[5] Do dai cua so (so buoc) va so KPI")
    print(w[["n_steps", "n_kpis"]].value_counts().to_string())

    print("\n[6] So cua so theo tung kich ban")
    print(w.groupby(["folder", "scenario"]).size().to_string())

    print("\nDang kiem tra metrics.csv...")
    raw = check_raw(root)
    print("\n[7] Chat luong du lieu tho metrics.csv")
    print(raw.drop(columns=["t_start", "t_end"]).to_string(index=False))

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    w.to_csv(out_dir / "telecomts_windows.csv", index=False, encoding="utf-8")
    raw.to_csv(out_dir / "telecomts_raw_files.csv", index=False, encoding="utf-8")
    print(f"\nDa ghi: {out_dir / 'telecomts_windows.csv'} va {out_dir / 'telecomts_raw_files.csv'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Cach dung: python scripts\\check_telecomts.py <thu_muc_telecomts>")
    main(sys.argv[1])
