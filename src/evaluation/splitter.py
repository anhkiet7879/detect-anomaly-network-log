"""
splitter.py — Bộ chia dữ liệu chuỗi thời gian & sự cố chống rò rỉ (Zero Data Leakage)
Thuộc Task K1: Đề tài N3.10 AIOps / Observability

Quy tắc cốt lõi:
1. Độc lập thời gian: Không bao giờ xáo trộn (shuffle=False), dữ liệu Train luôn đi trước Test theo thời gian.
2. Vùng đệm an toàn (Buffer Gap): Luôn giữ khoảng cách đệm (buffer_gap_mins >= 60 phút) giữa mốc kết thúc
   của Train và mốc bắt đầu của Test để tránh rò rỉ cửa sổ trượt và các đợt sự cố giao thoa.
3. Clean Train Only: Khi bật cờ clean_train_only=True, tự động loại bỏ các window/dòng có bất thường (is_anomaly == 1)
   khỏi tập Train để mô hình học máy (Isolation Forest, Z-score) chỉ học phân phối trạng thái bình thường.
4. Chia theo Đợt sự cố (Episodes): Nếu unit='episodes', không cắt đôi bất kỳ đợt sự cố nào.
"""

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml


@dataclass
class SplitFold:
    fold_idx: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    gap_df: Optional[pd.DataFrame] = None
    train_start_ts: Optional[float] = None
    train_end_ts: Optional[float] = None
    test_start_ts: Optional[float] = None
    test_end_ts: Optional[float] = None
    clean_train_applied: bool = True
    buffer_gap_seconds: float = 3600.0


class TimeSeriesDataSplitter:
    def __init__(
        self,
        buffer_gap_mins: int = 60,
        clean_train_only: bool = True,
        timestamp_col: str = "win",
        anomaly_col: str = "y",
        episodes_df: Optional[pd.DataFrame] = None,
    ):
        """
        Khởi tạo TimeSeriesDataSplitter
        :param buffer_gap_mins: Khoảng cách đệm an toàn giữa Train và Test (phút)
        :param clean_train_only: Nếu True, lọc sạch nhãn bất thường khỏi train_df
        :param timestamp_col: Cột mốc thời gian (epoch giây hoặc datetime)
        :param anomaly_col: Cột nhãn bất thường (0: bình thường, 1: bất thường)
        :param episodes_df: DataFrame danh sách episode (nếu chia theo episode)
        """
        self.buffer_gap_seconds = buffer_gap_mins * 60
        self.clean_train_only = clean_train_only
        self.timestamp_col = timestamp_col
        self.anomaly_col = anomaly_col
        self.episodes_df = episodes_df

    @classmethod
    def from_yaml(cls, config_path: str, episodes_df: Optional[pd.DataFrame] = None):
        """Tạo đối tượng splitter từ file cấu hình config/experiment.yaml"""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        leakage_cfg = cfg.get("leakage_prevention", {})
        data_cfg = cfg.get("data", {})
        
        return cls(
            buffer_gap_mins=leakage_cfg.get("buffer_gap_mins", 60),
            clean_train_only=leakage_cfg.get("clean_train_only", True),
            timestamp_col=data_cfg.get("timestamp_column", "win"),
            anomaly_col=data_cfg.get("target_column", "y"),
            episodes_df=episodes_df,
        )

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Đảm bảo dữ liệu được sắp xếp tăng dần theo thời gian."""
        if self.timestamp_col not in df.columns:
            raise KeyError(f"Cột thời gian '{self.timestamp_col}' không tồn tại trong dữ liệu.")
        return df.sort_values(by=self.timestamp_col).reset_index(drop=True)

    def _apply_clean_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Lọc bỏ toàn bộ mẫu bất thường khỏi tập Train nếu cờ clean_train_only=True."""
        if not self.clean_train_only:
            return train_df
        
        if self.anomaly_col in train_df.columns:
            return train_df[train_df[self.anomaly_col] == 0].copy()
        return train_df

    def split_holdout_by_time(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        test_ratio: Optional[float] = None,
    ) -> Generator[SplitFold, None, None]:
        """
        Chia dữ liệu theo chiến lược Holdout (theo trục thời gian).
        Áp dụng khoảng đệm buffer_gap_seconds ở giữa.
        """
        df = self._prepare_df(df)
        min_ts = df[self.timestamp_col].min()
        max_ts = df[self.timestamp_col].max()
        total_span = max_ts - min_ts

        train_cutoff_ts = min_ts + total_span * train_ratio
        test_start_ts = train_cutoff_ts + self.buffer_gap_seconds

        train_raw = df[df[self.timestamp_col] <= train_cutoff_ts].copy()
        gap_df = df[
            (df[self.timestamp_col] > train_cutoff_ts)
            & (df[self.timestamp_col] < test_start_ts)
        ].copy()
        
        if test_ratio is not None and (train_ratio + test_ratio < 1.0):
            test_end_ts = test_start_ts + total_span * test_ratio
            test_df = df[
                (df[self.timestamp_col] >= test_start_ts)
                & (df[self.timestamp_col] <= test_end_ts)
            ].copy()
        else:
            test_df = df[df[self.timestamp_col] >= test_start_ts].copy()

        train_clean = self._apply_clean_train(train_raw)

        yield SplitFold(
            fold_idx=0,
            train_df=train_clean,
            test_df=test_df,
            gap_df=gap_df,
            train_start_ts=train_raw[self.timestamp_col].min() if len(train_raw) else None,
            train_end_ts=train_cutoff_ts,
            test_start_ts=test_start_ts,
            test_end_ts=test_df[self.timestamp_col].max() if len(test_df) else None,
            clean_train_applied=self.clean_train_only,
            buffer_gap_seconds=self.buffer_gap_seconds,
        )

    def split_holdout_by_episodes(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
    ) -> Generator[SplitFold, None, None]:
        """
        Chia dữ liệu theo danh sách các đợt sự cố (episodes).
        Bảo toàn toàn bộ episode không bị chia cắt đôi giữa Train và Test.
        """
        df = self._prepare_df(df)
        if self.episodes_df is None or len(self.episodes_df) == 0:
            # Fallback nếu không có episode file: dùng chia theo time
            yield from self.split_holdout_by_time(df, train_ratio=train_ratio)
            return

        min_df_ts = df[self.timestamp_col].min()
        max_df_ts = df[self.timestamp_col].max()

        eps = self.episodes_df.copy()
        time_col = "start_ts" if "start_ts" in eps.columns else "start_utc"
        end_col = "end_ts" if "end_ts" in eps.columns else "end_utc"

        # Chuyển đổi timestamp sang epoch giây chuẩn (float)
        if pd.api.types.is_string_dtype(eps[time_col]):
            eps["start_epoch"] = pd.to_datetime(eps[time_col], utc=True).map(lambda x: x.timestamp())
            eps["end_epoch"] = pd.to_datetime(eps[end_col], utc=True).map(lambda x: x.timestamp())
        else:
            eps["start_epoch"] = eps[time_col].astype(float)
            eps["end_epoch"] = eps[end_col].astype(float)

        # Chỉ lọc các episodes nằm trong khoảng dữ liệu của df hiện tại
        eps_in_range = eps[
            (eps["start_epoch"] >= min_df_ts) & (eps["end_epoch"] <= max_df_ts)
        ].sort_values(by="start_epoch").reset_index(drop=True)

        if len(eps_in_range) < 2:
            # Nếu tập con không đủ episodes để chia, fallback sang split time
            yield from self.split_holdout_by_time(df, train_ratio=train_ratio)
            return

        split_idx = max(1, int(len(eps_in_range) * train_ratio))
        train_last_ep = eps_in_range.iloc[split_idx - 1]
        test_first_ep = eps_in_range.iloc[split_idx]

        train_end_ts = float(train_last_ep["end_epoch"])
        test_start_ts = float(test_first_ep["start_epoch"])

        # Đảm bảo test_start_ts cách train_end_ts ít nhất buffer_gap_seconds
        test_start_ts = max(test_start_ts, train_end_ts + self.buffer_gap_seconds)

        train_raw = df[df[self.timestamp_col] <= train_end_ts].copy()
        gap_df = df[
            (df[self.timestamp_col] > train_end_ts)
            & (df[self.timestamp_col] < test_start_ts)
        ].copy()
        test_df = df[df[self.timestamp_col] >= test_start_ts].copy()

        # Đảm bảo test_df không rỗng
        if len(test_df) == 0:
            yield from self.split_holdout_by_time(df, train_ratio=train_ratio)
            return

        train_clean = self._apply_clean_train(train_raw)

        yield SplitFold(
            fold_idx=0,
            train_df=train_clean,
            test_df=test_df,
            gap_df=gap_df,
            train_start_ts=train_raw[self.timestamp_col].min() if len(train_raw) else None,
            train_end_ts=train_end_ts,
            test_start_ts=test_start_ts,
            test_end_ts=test_df[self.timestamp_col].max() if len(test_df) else None,
            clean_train_applied=self.clean_train_only,
            buffer_gap_seconds=self.buffer_gap_seconds,
        )

    def split_expanding_cv(
        self,
        df: pd.DataFrame,
        n_splits: int = 4,
        min_train_ratio: float = 0.4,
    ) -> Generator[SplitFold, None, None]:
        """
        Expanding Window Cross-Validation (Walk-Forward).
        Kích thước Train mở rộng dần theo thời gian, luôn có đệm an toàn buffer_gap trước Test.
        """
        df = self._prepare_df(df)
        min_ts = df[self.timestamp_col].min()
        max_ts = df[self.timestamp_col].max()
        total_span = max_ts - min_ts

        # Khoảng thời gian dành cho test chia đều theo n_splits
        remaining_ratio = 1.0 - min_train_ratio
        test_ratio_per_fold = remaining_ratio / n_splits

        for fold in range(n_splits):
            train_ratio = min_train_ratio + fold * test_ratio_per_fold
            train_cutoff = min_ts + total_span * train_ratio
            test_start = train_cutoff + self.buffer_gap_seconds
            test_end = min_ts + total_span * (train_ratio + test_ratio_per_fold)

            if test_start >= test_end or test_start >= max_ts:
                continue

            train_raw = df[df[self.timestamp_col] <= train_cutoff].copy()
            gap_df = df[
                (df[self.timestamp_col] > train_cutoff)
                & (df[self.timestamp_col] < test_start)
            ].copy()
            test_df = df[
                (df[self.timestamp_col] >= test_start)
                & (df[self.timestamp_col] <= test_end)
            ].copy()

            train_clean = self._apply_clean_train(train_raw)

            yield SplitFold(
                fold_idx=fold,
                train_df=train_clean,
                test_df=test_df,
                gap_df=gap_df,
                train_start_ts=train_raw[self.timestamp_col].min() if len(train_raw) else None,
                train_end_ts=train_cutoff,
                test_start_ts=test_start,
                test_end_ts=test_end,
                clean_train_applied=self.clean_train_only,
                buffer_gap_seconds=self.buffer_gap_seconds,
            )

    def split_rolling_cv(
        self,
        df: pd.DataFrame,
        n_splits: int = 4,
        train_window_ratio: float = 0.4,
        test_window_ratio: float = 0.15,
    ) -> Generator[SplitFold, None, None]:
        """
        Rolling Window Cross-Validation (Sliding Window).
        Cửa sổ Train trượt theo thời gian với kích thước cố định.
        """
        df = self._prepare_df(df)
        min_ts = df[self.timestamp_col].min()
        max_ts = df[self.timestamp_col].max()
        total_span = max_ts - min_ts

        stride_ratio = (1.0 - train_window_ratio - test_window_ratio) / max(1, n_splits - 1)

        for fold in range(n_splits):
            start_ratio = fold * stride_ratio
            train_start = min_ts + total_span * start_ratio
            train_cutoff = train_start + total_span * train_window_ratio
            test_start = train_cutoff + self.buffer_gap_seconds
            test_end = test_start + total_span * test_window_ratio

            if test_start >= max_ts:
                break

            train_raw = df[
                (df[self.timestamp_col] >= train_start)
                & (df[self.timestamp_col] <= train_cutoff)
            ].copy()
            gap_df = df[
                (df[self.timestamp_col] > train_cutoff)
                & (df[self.timestamp_col] < test_start)
            ].copy()
            test_df = df[
                (df[self.timestamp_col] >= test_start)
                & (df[self.timestamp_col] <= min(test_end, max_ts))
            ].copy()

            train_clean = self._apply_clean_train(train_raw)

            yield SplitFold(
                fold_idx=fold,
                train_df=train_clean,
                test_df=test_df,
                gap_df=gap_df,
                train_start_ts=train_start,
                train_end_ts=train_cutoff,
                test_start_ts=test_start,
                test_end_ts=test_df[self.timestamp_col].max() if len(test_df) else None,
                clean_train_applied=self.clean_train_only,
                buffer_gap_seconds=self.buffer_gap_seconds,
            )

    def split_by_scenario(
        self,
        df: pd.DataFrame,
        scenario_name: str,
        scenario_cfg: Dict,
    ) -> Generator[SplitFold, None, None]:
        """Chia dữ liệu theo tên và cấu hình kịch bản từ YAML."""
        strategy = scenario_cfg.get("strategy", "holdout")
        unit = scenario_cfg.get("unit", "time")

        if strategy == "holdout":
            if unit == "episodes":
                yield from self.split_holdout_by_episodes(
                    df, train_ratio=scenario_cfg.get("train_ratio", 0.7)
                )
            else:
                yield from self.split_holdout_by_time(
                    df,
                    train_ratio=scenario_cfg.get("train_ratio", 0.8),
                    test_ratio=scenario_cfg.get("test_ratio", None),
                )
        elif strategy == "expanding_cv":
            yield from self.split_expanding_cv(
                df,
                n_splits=scenario_cfg.get("n_splits", 4),
                min_train_ratio=scenario_cfg.get("min_train_ratio", 0.4),
            )
        elif strategy == "rolling_cv":
            yield from self.split_rolling_cv(
                df,
                n_splits=scenario_cfg.get("n_splits", 4),
                train_window_ratio=scenario_cfg.get("train_window_ratio", 0.4),
                test_window_ratio=scenario_cfg.get("test_window_ratio", 0.15),
            )
        else:
            raise ValueError(f"Chiến lược chia không hợp lệ: {strategy}")
