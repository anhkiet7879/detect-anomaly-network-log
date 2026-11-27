
from pathlib import Path
import argparse
import json

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


def calculate_metrics(y_true, y_pred):
    """
    Tinh cac chi so danh gia:
    Precision, Recall, F1-score, MCC.

    y_true: Nhan thuc te (0 hoac 1).
    y_pred: Nhan du doan (0 hoac 1).
    """

    # Kiem tra du lieu dau vao
    if len(y_true) != len(y_pred):
        raise ValueError("y_true va y_pred phai cung do dai")

    if len(y_true) == 0:
        raise ValueError("Khong co du lieu de danh gia")

    # Kiem tra nhan hop le
    valid_labels = {0, 1}

    if not set(y_true).issubset(valid_labels):
        raise ValueError("y_true chi duoc chua gia tri 0 hoac 1")

    if not set(y_pred).issubset(valid_labels):
        raise ValueError("y_pred chi duoc chua gia tri 0 hoac 1")

    # Tinh confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # Tinh cac metrics
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

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "MCC": float(mcc)
    }


def evaluate_csv(input_path):
    """
    Doc CSV va danh gia ket qua du doan.
    """

    df = pd.read_csv(input_path)

    required_columns = {"y_true", "y_pred"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV phai co cot y_true va y_pred"
        )

    if df[["y_true", "y_pred"]].isna().any().any():
        raise ValueError("Du lieu nhan khong duoc de trong")

    # Kiem tra gia tri nhan
    for column in ["y_true", "y_pred"]:
        if not df[column].isin([0, 1]).all():
            raise ValueError(
                f"Cot {column} chi duoc chua 0 hoac 1"
            )

    results = calculate_metrics(
        df["y_true"].astype(int).tolist(),
        df["y_pred"].astype(int).tolist()
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="HKDN Model Evaluation"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Duong dan file predictions.csv"
    )

    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="File JSON luu ket qua"
    )

    args = parser.parse_args()

    results = evaluate_csv(args.input)

    print("\n===== HKDN EVALUATION RESULTS =====")

    for key, value in results.items():
        if key in ["Precision", "Recall", "F1", "MCC"]:
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nDa luu ket qua vao: {args.output}")


if __name__ == "__main__":
    main()
