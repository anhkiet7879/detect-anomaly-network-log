"""
runner.py — Khung thực thi đánh giá đa kịch bản (Scenario Runner)
Thuộc Task K1: Đề tài N3.10 AIOps / Observability

Nhiệm vụ cốt lõi:
1. Đọc kịch bản từ config/experiment.yaml (Holdout 80/20, 75/25, 70/30, 60/40; expanding_cv; rolling_cv).
2. Gọi splitter.py để sinh các fold (train_df, test_df) với nguyên tắc Zero Data Leakage:
   - buffer_gap_mins >= 60 phút
   - clean_train_only = True (loại trừ log lỗi khỏi tập Train)
3. Quy tắc Độc lập trạng thái (Reset State):
   - Ở mỗi fold/scenario, mô hình (Rule, Z-score, Isolation Forest, ...) được khởi tạo mới hoàn toàn từ đầu trên train_df.
   - Tuyệt đối không lưu lại trạng thái hay rò rỉ thông tin sang test_df.
4. Tính toán các chỉ số phân loại: Precision, Recall, F1-score, Accuracy, MCC cho từng fold.
5. Tổng hợp bảng so sánh đa kịch bản dạng Mean ± Std.
6. Xuất báo cáo ra định dạng Markdown, CSV và file dự đoán chi tiết.
"""

import argparse
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

# Thêm đường dẫn gốc vào sys.path để import dễ dàng
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.splitter import SplitFold, TimeSeriesDataSplitter
from src.features.window_features import load_windows


# ==============================================================================
# 1. GIAO DIỆN CHUẨN DETECTOR (INTERFACE CONTRACT 4)
# ==============================================================================
class BaseDetector(ABC):
    """
    Interface Contract bắt buộc cho mọi detector trong đề tài N3.10.
    Tất cả các mô hình của Huy (IF) và Phước (Rule, Z-score) đều phải kế thừa
    hoặc tuân theo chuẩn fit(train_df) và predict(test_df).
    """

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseDetector":
        """Khởi tạo và huấn luyện lại toàn bộ trạng thái từ đầu trên train_df."""
        pass

    @abstractmethod
    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dự đoán trên test_df.
        :return: (y_pred (0/1), scores (float độ bất thường))
        """
        pass

    def reset(self):
        """Reset trạng thái bên trong về ban đầu."""
        pass


# ==============================================================================
# 2. CÁC MÔ HÌNH THỰC THI CHUẨN INTERFACE
# ==============================================================================
class RuleFatalDetector(BaseDetector):
    """Bộ phát hiện luật: Cảnh báo nếu cửa sổ có log FATAL."""

    def __init__(self, fatal_col: str = "f_sev_FATAL"):
        self.fatal_col = fatal_col

    def fit(self, train_df: pd.DataFrame) -> "RuleFatalDetector":
        return self

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.fatal_col in test_df.columns:
            scores = test_df[self.fatal_col].to_numpy(float)
        else:
            scores = np.zeros(len(test_df), dtype=float)
        y_pred = (scores > 0).astype(int)
        return y_pred, scores


class ZScoreBaselineDetector(BaseDetector):
    """
    Bộ phát hiện Baseline thống kê Z-score trên số lượng log / 5 phút.
    Tính mean và std (hoặc Median & MAD) chỉ trên tập Train.
    """

    def __init__(self, feature_col: str = "f_n_logs", threshold: float = 3.0):
        self.feature_col = feature_col
        self.threshold = threshold
        self.median_ = 0.0
        self.mad_ = 1.0

    def fit(self, train_df: pd.DataFrame) -> "ZScoreBaselineDetector":
        vals = np.log1p(train_df[self.feature_col].to_numpy(float))
        # Chỉ tính trên các window có log > 0
        vals_nonzero = vals[vals > 0]
        if len(vals_nonzero) > 0:
            self.median_ = float(np.median(vals_nonzero))
            self.mad_ = float(np.median(np.abs(vals_nonzero - self.median_)))
            if self.mad_ < 1e-6:
                self.mad_ = float(np.std(vals_nonzero)) or 1.0
        else:
            self.median_ = 0.0
            self.mad_ = 1.0
        return self

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        vals = np.log1p(test_df[self.feature_col].to_numpy(float))
        scale = 1.4826 * self.mad_ if self.mad_ > 1e-6 else 1.0
        scores = (vals - self.median_) / scale
        scores = np.maximum(scores, 0.0)
        y_pred = (scores > self.threshold).astype(int)
        return y_pred, scores


class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest Detector chuẩn fit(train_df) & predict(test_df).
    Khởi tạo lại độc lập ở từng fold để chống rò rỉ.
    """

    def __init__(
        self,
        features: List[str],
        contamination: float = 0.01,
        random_state: int = 42,
    ):
        self.features = features
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.threshold_: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "IsolationForestDetector":
        # Khởi tạo mô hình mới hoàn toàn
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        X_train = np.log1p(train_df[self.features].to_numpy(float))
        self.model.fit(X_train)

        # Tính ngưỡng phân vị dựa trên chính dữ liệu học
        train_raw_scores = -self.model.score_samples(X_train)
        self.threshold_ = float(np.percentile(train_raw_scores, 100 * (1 - self.contamination)))
        return self

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Mô hình chưa được fit trước khi predict.")
        X_test = np.log1p(test_df[self.features].to_numpy(float))
        raw_scores = -self.model.score_samples(X_test)
        y_pred = (raw_scores >= self.threshold_).astype(int)
        return y_pred, raw_scores


# ==============================================================================
# 3. CLASS THỰC THI RUNNER
# ==============================================================================
class EvaluationRunner:
    def __init__(self, config_path: str = "config/experiment.yaml"):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.reports_dir = os.path.join(
            PROJECT_ROOT, self.cfg.get("output", {}).get("reports_dir", "reports/evaluation")
        )
        os.makedirs(self.reports_dir, exist_ok=True)

        # Đọc danh sách đợt sự cố nếu có
        eps_path = os.path.join(
            PROJECT_ROOT, self.cfg.get("data", {}).get("episodes_path", "reports/bgl_episodes.csv")
        )
        if not os.path.exists(eps_path):
            eps_path = os.path.join(PROJECT_ROOT, "reports/bgl_episodes.csv")
        
        self.episodes_df = pd.read_csv(eps_path) if os.path.exists(eps_path) else None

        self.splitter = TimeSeriesDataSplitter.from_yaml(config_path, episodes_df=self.episodes_df)
        self.features = self.cfg.get("data", {}).get("features", [])
        self.target_col = self.cfg.get("data", {}).get("target_column", "y")

    def _calc_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Tính toán toàn diện các metrics phân loại theo quy ước."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

        return {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "Precision": float(p),
            "Recall": float(r),
            "F1": float(f1),
            "Accuracy": float(acc),
            "MCC": float(mcc),
        }

    def get_registered_detectors(self) -> Dict[str, BaseDetector]:
        """Danh sách các detector tham gia đánh giá đối sánh."""
        return {
            "Rule_FATAL": RuleFatalDetector(fatal_col="f_sev_FATAL"),
            "Baseline_ZScore": ZScoreBaselineDetector(feature_col="f_n_logs", threshold=3.0),
            "Isolation_Forest": IsolationForestDetector(
                features=self.features,
                contamination=0.01,
                random_state=self.cfg.get("experiment", {}).get("random_seed", 42),
            ),
        }

    def run_scenario(
        self,
        df: pd.DataFrame,
        scenario_name: str,
        scenario_cfg: Dict,
        detectors: Optional[Dict[str, BaseDetector]] = None,
    ) -> pd.DataFrame:
        """Thực thi một kịch bản chia cụ thể trên toàn bộ detectors."""
        if detectors is None:
            detectors = self.get_registered_detectors()

        results = []
        splits = list(self.splitter.split_by_scenario(df, scenario_name, scenario_cfg))

        for fold in splits:
            y_test = fold.test_df[self.target_col].to_numpy(int)

            for det_name, det_model in detectors.items():
                # RESET STATE HOÀN TOÀN: fit trên train_df đã được clean
                det_model.fit(fold.train_df)
                y_pred, scores = det_model.predict(fold.test_df)

                m = self._calc_metrics(y_test, y_pred)
                record = {
                    "scenario": scenario_name,
                    "fold": fold.fold_idx,
                    "detector": det_name,
                    "train_size": len(fold.train_df),
                    "test_size": len(fold.test_df),
                    "gap_size": len(fold.gap_df) if fold.gap_df is not None else 0,
                    **m,
                }
                results.append(record)

        return pd.DataFrame(results)

    def run_all(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chạy toàn bộ các kịch bản trong experiment.yaml.
        Trả về: (df_raw_folds, df_aggregated_mean_std)
        """
        if df is None:
            features_path = os.path.join(
                PROJECT_ROOT, self.cfg.get("data", {}).get("features_path", "data/features/bgl_windows_A.parquet")
            )
            if not os.path.exists(features_path):
                raise FileNotFoundError(f"Không tìm thấy file đặc trưng tại {features_path}")
            df = load_windows(features_path)

        all_fold_results = []
        scenarios = self.cfg.get("scenarios", {})

        print(f"=== BẮT ĐẦU CHẠY ĐÁNH GIÁ ĐA KỊCH BẢN (TỔNG {len(scenarios)} SCENARIOS) ===")
        for sc_name, sc_cfg in scenarios.items():
            print(f"-> Đang thực thi kịch bản: {sc_name}...")
            df_sc = self.run_scenario(df, sc_name, sc_cfg)
            all_fold_results.append(df_sc)

        df_all = pd.concat(all_fold_results, ignore_index=True)

        # Tính tổng hợp Mean ± Std cho từng detector qua các kịch bản
        summary_rows = []
        metric_cols = ["Precision", "Recall", "F1", "Accuracy", "MCC"]

        for (sc_name, det_name), g in df_all.groupby(["scenario", "detector"]):
            row = {"scenario": sc_name, "detector": det_name, "folds": len(g)}
            for m in metric_cols:
                mean_val = g[m].mean()
                std_val = g[m].std(ddof=0) if len(g) > 1 else 0.0
                row[f"{m}_mean"] = mean_val
                row[f"{m}_std"] = std_val
                row[f"{m} (Mean±Std)"] = f"{mean_val:.4f} ± {std_val:.4f}"
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)

        # Lưu kết quả
        raw_csv_path = os.path.join(self.reports_dir, "evaluation_folds_raw.csv")
        summary_csv_path = os.path.join(self.reports_dir, "evaluation_mean_std_summary.csv")
        df_all.to_csv(raw_csv_path, index=False)
        df_summary.to_csv(summary_csv_path, index=False)

        print(f" Đã lưu kết quả chi tiết từng fold tại: {raw_csv_path}")
        print(f" Đã lưu kết quả tổng hợp Mean ± Std tại: {summary_csv_path}")

        return df_all, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy khung đánh giá K1 (Zero Data Leakage)")
    parser.add_argument("--config", default="config/experiment.yaml", help="Đường dẫn file cấu hình YAML")
    args = parser.parse_args()

    runner = EvaluationRunner(config_path=args.config)
    df_raw, df_sum = runner.run_all()
    print("\n--- BẢNG KẾT QUẢ TỔNG HỢP (MEAN ± STD) ---")
    cols_show = ["scenario", "detector", "F1 (Mean±Std)", "Precision (Mean±Std)", "Recall (Mean±Std)", "MCC (Mean±Std)"]
    print(df_sum[cols_show].to_string(index=False))
