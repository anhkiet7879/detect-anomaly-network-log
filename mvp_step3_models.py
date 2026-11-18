"""
mvp_step3_models.py — MVP Bước 3: baseline + mô hình + đánh giá trên cửa sổ 5 phút.

Đầu vào : data\\features\\bgl_mvp_windows.csv (do mvp_step2_windows.py tạo)
Đầu ra  : reports\\mvp\\step3_summary.txt          bảng kết quả
          reports\\mvp\\step3_predictions_test.csv  dự đoán từng cửa sổ test (để xem lại / dashboard)

QUY TRÌNH (giống nhau cho mọi phương pháp)
  (a) Học trên TRAIN  -> đo trên VAL   : chỉ để kiểm tra sơ bộ (val chỉ có ít alert)
  (b) Học lại trên TRAIN + VAL -> đo trên TEST : con số chính thức, đo MỘT lần

5 PHƯƠNG PHÁP
  1. rule_fatal   : cảnh báo nếu cửa sổ có >= 1 dòng FATAL.             (luật, không học)
  2. mad_logs     : điểm = (log(1+n_logs) - trung vị) / (1.4826*MAD),     (baseline thống kê,
                    học trên cửa sổ BÌNH THƯỜNG có log; cảnh báo nếu điểm > 3)   không giám sát)
  3. iforest      : Isolation Forest học trên cửa sổ BÌNH THƯỜNG;          (không giám sát)
                    ngưỡng = phân vị 99% điểm bất thường của chính dữ liệu học
  4. logreg       : Logistic Regression, class_weight="balanced"          (có giám sát)
  5. rforest      : Random Forest, class_weight="balanced"                (có giám sát)

  "Cửa sổ bình thường" = y == 0 (mô phỏng: kỹ sư đã loại các đoạn sự cố đã xác nhận).
  Mô hình chỉ dùng các cột f_...; các cột gt_... và y không bao giờ là đặc trưng.

Cách chạy (trong thư mục N3.10):
  python mvp_step3_models.py
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


# ---------------------------------------------------------------- phương pháp
def fit_predict(method, tr, te, feats):
    """Học trên tr, trả về (dự đoán 0/1, điểm) cho te."""
    Xtr = np.log1p(tr[feats].to_numpy(float))   # log(1+x): giảm ảnh hưởng của các cửa sổ khổng lồ
    Xte = np.log1p(te[feats].to_numpy(float))
    ytr = tr["y"].to_numpy()

    if method == "rule_fatal":
        score = te["f_sev_FATAL"].to_numpy(float)
        return (score > 0).astype(int), score

    if method == "mad_logs":
        base = np.log1p(tr.loc[(tr.y == 0) & (tr.f_n_logs > 0), "f_n_logs"].to_numpy(float))
        med = np.median(base)
        mad = np.median(np.abs(base - med)) * 1.4826
        mad = mad if mad > 0 else 1e-9
        score = (np.log1p(te["f_n_logs"].to_numpy(float)) - med) / mad
        return (score > 3).astype(int), score

    if method == "iforest":
        normal = Xtr[ytr == 0]
        m = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(normal)
        thr = np.quantile(-m.score_samples(normal), 0.99)
        score = -m.score_samples(Xte)            # càng lớn càng bất thường
        return (score > thr).astype(int), score

    if method == "logreg":
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=2000))
        m.fit(Xtr, ytr)
        score = m.predict_proba(Xte)[:, 1]
        return (score >= 0.5).astype(int), score

    if method == "rforest":
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   min_samples_leaf=2, random_state=SEED, n_jobs=-1)
        m.fit(Xtr, ytr)
        score = m.predict_proba(Xte)[:, 1]
        fit_predict.rf_importance = pd.Series(m.feature_importances_, index=feats)
        return (score >= 0.5).astype(int), score

    raise ValueError(method)


METHODS = ["rule_fatal", "mad_logs", "iforest", "logreg", "rforest"]
GROUP = {"rule_fatal": "luật", "mad_logs": "không GS", "iforest": "không GS",
         "logreg": "có GS", "rforest": "có GS"}


# ---------------------------------------------------------------- chấm điểm
def metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, Precision=prec, Recall=rec, F1=f1, FPR=fpr)


def table(rows):
    head = f"  {'phương pháp':<11} {'nhóm':<9} {'TP':>4} {'FP':>5} {'FN':>4} " \
           f"{'Precision':>9} {'Recall':>7} {'F1':>6} {'FPR':>7}"
    lines = [head, "  " + "-" * (len(head) - 2)]
    for name, r in rows:
        lines.append(f"  {name:<11} {GROUP[name]:<9} {r['TP']:>4} {r['FP']:>5} {r['FN']:>4} "
                     f"{r['Precision']:>9.3f} {r['Recall']:>7.3f} {r['F1']:>6.3f} {r['FPR']:>7.4f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join("data", "features", "bgl_mvp_windows.csv"))
    ap.add_argument("--report-dir", default=os.path.join("reports", "mvp"))
    args = ap.parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    w = pd.read_csv(args.input, keep_default_na=False)
    feats = [c for c in w.columns if c.startswith("f_")]
    tr, va, te = (w[w.split == s].reset_index(drop=True) for s in ["train", "val", "test"])
    trva = pd.concat([tr, va], ignore_index=True)

    out = []
    p = out.append
    p("===== MVP BƯỚC 3: BASELINE + MÔ HÌNH (cửa sổ 5 phút, đặc trưng bộ A) =====")
    p(f"Số đặc trưng: {len(feats)}  |  train {len(tr):,} (alert {tr.y.sum()})  "
      f"val {len(va):,} (alert {va.y.sum()})  test {len(te):,} (alert {te.y.sum()})")
    p("")

    # (a) train -> val
    p("(a) Học trên TRAIN, đo trên VAL  — chỉ để kiểm tra sơ bộ, KHÔNG dùng để chọn mô hình")
    rows = []
    for mth in METHODS:
        pred, _ = fit_predict(mth, tr, va, feats)
        rows.append((mth, metrics(va.y.to_numpy(), pred)))
    out += table(rows)
    p("")

    # (b) train+val -> test
    p("(b) Học lại trên TRAIN + VAL, đo trên TEST  — KẾT QUẢ CHÍNH THỨC")
    rows = []
    preds = te[["ts_epoch", "ts_utc", "y", "f_n_logs", "f_sev_FATAL",
                "gt_alert_lines", "gt_alert_types"]].copy()
    for mth in METHODS:
        pred, score = fit_predict(mth, trva, te, feats)
        rows.append((mth, metrics(te.y.to_numpy(), pred)))
        preds[f"pred_{mth}"] = pred
        preds[f"score_{mth}"] = np.round(score, 4)
    out += table(rows)
    p("")
    p("  Precision = trong các cảnh báo, bao nhiêu % đúng  |  Recall = trong các cửa sổ alert, bắt được bao nhiêu %")
    p("  FPR = tỷ lệ cửa sổ bình thường bị báo nhầm")
    p("")

    # Recall theo loại alert (test)
    seen = set()
    for v in trva["gt_alert_types"]:
        if v:
            seen.update(v.split("|"))
    alert_te = preds[preds.y == 1]
    all_types = sorted({t for v in alert_te.gt_alert_types for t in v.split("|") if t})
    p("Recall theo LOẠI alert trên test (số cửa sổ chứa loại đó được cảnh báo / tổng):")
    p(f"  {'loại':<10} {'đã gặp?':<8} {'số cửa sổ':>9} " + " ".join(f"{m:>10}" for m in METHODS))
    for t in all_types:
        mask = alert_te.gt_alert_types.apply(lambda v: t in v.split("|"))
        n = int(mask.sum())
        cells = " ".join(f"{int(alert_te.loc[mask, f'pred_{m}'].sum()):>6}/{n:<3}" for m in METHODS)
        p(f"  {t:<10} {'có' if t in seen else 'CHƯA':<8} {n:>9} {cells}")
    only_new = alert_te.gt_alert_types.apply(lambda v: set(v.split("|")).isdisjoint(seen))
    n_new = int(only_new.sum())
    cells = " ".join(f"{int(alert_te.loc[only_new, f'pred_{m}'].sum()):>6}/{n_new:<3}" for m in METHODS)
    p(f"  {'CHỈ loại chưa gặp':<29} {cells}")
    p("")

    if hasattr(fit_predict, "rf_importance"):
        p("10 đặc trưng quan trọng nhất của Random Forest (học trên train + val):")
        for f, v in fit_predict.rf_importance.sort_values(ascending=False).head(10).items():
            p(f"  {f:<22} {v:.3f}")
        p("")

    pred_path = os.path.join(args.report_dir, "step3_predictions_test.csv")
    preds.to_csv(pred_path, index=False)
    p(f"Đã ghi dự đoán từng cửa sổ test: {pred_path}")

    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.report_dir, "step3_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Đã lưu: {os.path.join(args.report_dir, 'step3_summary.txt')}")


if __name__ == "__main__":
    main()
