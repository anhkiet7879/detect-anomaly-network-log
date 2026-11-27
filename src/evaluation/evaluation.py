

from pathlib import Path
import argparse
import json
import math

import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


# ============================================================
# 1. CAC CHI SO DANH GIA CO BAN
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Tinh TP, TN, FP, FN, Precision, Recall, F1 va MCC.
    Nhan hop le: 0 (binh thuong), 1 (bat thuong).
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true va y_pred phai cung do dai")

    if len(y_true) == 0:
        raise ValueError("Khong co du lieu de danh gia")

    valid_labels = {0, 1}

    if not set(y_true).issubset(valid_labels):
        raise ValueError("y_true chi duoc chua gia tri 0 hoac 1")

    if not set(y_pred).issubset(valid_labels):
        raise ValueError("y_pred chi duoc chua gia tri 0 hoac 1")

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    mcc = matthews_corrcoef(y_true, y_pred)

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Accuracy": float(accuracy),
        "MCC": float(mcc)
    }


# ============================================================
# 2. TINH METRIC TU CONFUSION MATRIX CUA MOI FOLD
# ============================================================

def calculate_metrics_from_counts(row):
    """
    Tinh lai metric tu TP, FP, FN, TN cua mot Fold.
    """

    tp = int(row["TP"])
    fp = int(row["FP"])
    fn = int(row["FN"])
    tn = int(row["TN"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    total = tp + fp + fn + tn

    accuracy = (tp + tn) / total if total > 0 else 0.0

    denominator = math.sqrt(
        (tp + fp) * (tp + fn) *
        (tn + fp) * (tn + fn)
    )

    mcc = (
        (tp * tn - fp * fn) / denominator
        if denominator > 0 else 0.0
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "MCC": mcc
    }


# ============================================================
# 3. KIEM TRA DU LIEU DAU VAO
# ============================================================

def validate_fold_data(df):
    """
    Kiem tra cau truc file evaluation_folds_raw.csv.
    """

    required_columns = {
        "scenario",
        "fold",
        "detector",
        "TP",
        "FP",
        "FN",
        "TN"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Thieu cac cot bat buoc: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("File CSV khong co du lieu")

    if df[list(required_columns)].isna().any().any():
        raise ValueError("Du lieu Fold khong duoc de trong")

    count_columns = ["TP", "FP", "FN", "TN"]

    for column in count_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

        if df[column].isna().any():
            raise ValueError(
                f"Cot {column} chua gia tri khong hop le"
            )

        if (df[column] < 0).any():
            raise ValueError(
                f"Cot {column} khong duoc co gia tri am"
            )

        if (df[column] % 1 != 0).any():
            raise ValueError(
                f"Cot {column} phai la so nguyen"
            )

    duplicate_columns = ["scenario", "fold", "detector"]

    if df.duplicated(subset=duplicate_columns).any():
        raise ValueError(
            "Phat hien Fold bi trung trong cung scenario va detector"
        )

    return df


# ============================================================
# 4. TONG HOP MEAN VA STD QUA CAC FOLD
# ============================================================

def summarize_folds(input_path):
    """
    Doc ket qua tung Fold va tong hop Mean +/- Std
    theo tung Scenario va Detector.

    Std su dung ddof=0 (population standard deviation).
    """

    df = pd.read_csv(input_path)

    df = validate_fold_data(df)

    metric_names = [
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
        "MCC"
    ]

    # Tinh lai metric cho tung Fold
    calculated_metrics = df.apply(
        calculate_metrics_from_counts,
        axis=1
    )

    for metric in metric_names:
        df[metric] = calculated_metrics.apply(
            lambda result: result[metric]
        )

    summary_records = []

    # Nhom theo Scenario va Detector
    grouped = df.groupby(
        ["scenario", "detector"],
        sort=True
    )

    for (scenario, detector), group in grouped:

        record = {
            "scenario": scenario,
            "detector": detector,
            "folds": int(group["fold"].nunique())
        }

        for metric in metric_names:

            values = group[metric]

            mean_value = values.mean()

            std_value = values.std(ddof=0)

            record[f"{metric}_mean"] = float(mean_value)

            record[f"{metric}_std"] = float(std_value)

            record[f"{metric} (Mean±Std)"] = (
                f"{mean_value:.4f} ± {std_value:.4f}"
            )

        summary_records.append(record)

    return pd.DataFrame(summary_records)


# ============================================================
# 5. DANH GIA FILE PREDICTIONS CU
# ============================================================

def evaluate_predictions_csv(input_path):
    """
    Duy tri chuc nang cu:
    Doc CSV co y_true va y_pred.
    """

    df = pd.read_csv(input_path)

    required_columns = {"y_true", "y_pred"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV phai co cot y_true va y_pred"
        )

    if df[["y_true", "y_pred"]].isna().any().any():
        raise ValueError("Du lieu nhan khong duoc de trong")

    for column in ["y_true", "y_pred"]:

        if not df[column].isin([0, 1]).all():
            raise ValueError(
                f"Cot {column} chi duoc chua 0 hoac 1"
            )

    return calculate_metrics(
        df["y_true"].astype(int).tolist(),
        df["y_pred"].astype(int).tolist()
    )


# ============================================================
# 6. LUU KET QUA
# ============================================================

def save_results(results, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if isinstance(results, pd.DataFrame):

        if output_path.suffix.lower() == ".json":

            results.to_json(
                output_path,
                orient="records",
                indent=4,
                force_ascii=False
            )

        else:

            results.to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig"
            )

    else:

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
                ensure_ascii=False
            )

    print(f"\nDa luu ket qua tai: {output_path}")


# ============================================================
# 7. CHUONG TRINH CHINH
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="HKDN Evaluation - Multi-scenario Mean Std"
    )

    parser.add_argument(
        "--input",
        default="reports/evaluation/evaluation_folds_raw.csv",
        help="File CSV dau vao"
    )

    parser.add_argument(
        "--output",
        default="reports/evaluation/evaluation_t3_summary.csv",
        help="File luu ket qua"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file dau vao: {input_path}"
        )

    # Doc header de xac dinh loai du lieu
    columns = set(
        pd.read_csv(input_path, nrows=0).columns
    )

    fold_columns = {
        "scenario",
        "fold",
        "detector",
        "TP",
        "FP",
        "FN",
        "TN"
    }

    prediction_columns = {"y_true", "y_pred"}

    if fold_columns.issubset(columns):

        print("\n===== HKDN MULTI-SCENARIO EVALUATION =====")

        print("Dang tong hop ket qua cac Fold...")

        results = summarize_folds(input_path)

        print("\n===== MEAN ± STD SUMMARY =====")

        display_columns = [
            "scenario",
            "detector",
            "folds",
            "F1 (Mean±Std)",
            "Precision (Mean±Std)",
            "Recall (Mean±Std)",
            "MCC (Mean±Std)"
        ]

        print(
            results[display_columns].to_string(index=False)
        )

        save_results(results, args.output)

    elif prediction_columns.issubset(columns):

        print("\n===== HKDN SINGLE DATASET EVALUATION =====")

        results = evaluate_predictions_csv(input_path)

        for key, value in results.items():

            if isinstance(value, float):
                print(f"{key}: {value:.4f}")

            else:
                print(f"{key}: {value}")

        save_results(results, args.output)

    else:

        raise ValueError(
            "Dinh dang CSV khong hop le. "
            "Can cac cot TP, FP, FN, TN, scenario, fold, detector "
            "hoac y_true, y_pred."
        )


if __name__ == "__main__":
    main()
	