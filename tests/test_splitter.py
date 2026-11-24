
import pandas as pd
import pytest

from src.evaluation.splitter import TimeSeriesDataSplitter


# =========================================================
# FIXTURE: Tao du lieu gia lap de kiem tra bo chia
# =========================================================

@pytest.fixture
def sample_data():
    """
    Tao du lieu log theo chu ky 5 phut trong 48 gio.
    Tao cac Episode rieng biet, moi Episode keo dai 30 phut.
    """

    interval = 300  # 5 phut
    total_rows = 576  # 48 gio

    start_time = 1_700_000_000

    timestamps = [
        start_time + i * interval
        for i in range(total_rows)
    ]

    df = pd.DataFrame({
        "ts_epoch": timestamps,
        "y": [0] * total_rows,
        "f_n_logs": [10] * total_rows,
        "f_n_nodes": [5] * total_rows,
    })

    # Tao Episode moi 2 gio, moi Episode dai 30 phut.
    episodes = []

    for i in range(24):
        episode_start = start_time + i * 7200 + 3600

        episodes.append({
            "episode_id": f"EP_{i + 1}",
            "start_ts": episode_start,
            "end_ts": episode_start + 1800,
        })

    episodes_df = pd.DataFrame(episodes)

    return df, episodes_df


# =========================================================
# HAM HO TRO
# =========================================================

def count_complete_episodes(df, episodes_df):
    """
    Dem so Episode nam tron trong khoang thoi gian cua DataFrame.
    """

    if df.empty:
        return 0

    data_start = df["ts_epoch"].min()
    data_end = df["ts_epoch"].max()

    complete_episodes = episodes_df[
        (episodes_df["start_ts"] >= data_start)
        & (episodes_df["end_ts"] <= data_end)
    ]

    return len(complete_episodes)


def check_no_time_leakage(fold, buffer_seconds=3600):
    """
    Kiem tra Train va Validation khong chong lan thoi gian
    va co khoang dem toi thieu 60 phut.
    """

    train_df = fold.train_df
    test_df = fold.test_df

    assert not train_df.empty, "Train khong duoc rong"
    assert not test_df.empty, "Validation khong duoc rong"

    train_times = set(train_df["ts_epoch"])
    test_times = set(test_df["ts_epoch"])

    # Khong co timestamp nao xuat hien o ca hai tap.
    assert train_times.isdisjoint(test_times), (
        "Phat hien timestamp bi trung giua Train va Validation"
    )

    train_end = train_df["ts_epoch"].max()
    test_start = test_df["ts_epoch"].min()

    # Validation phai nam sau Train.
    assert test_start > train_end, (
        "Validation khong nam sau Train"
    )

    # Khoang cach thoi gian phai >= 60 phut.
    actual_gap = test_start - train_end

    assert actual_gap >= buffer_seconds, (
        f"Khoang dem khong du 60 phut: {actual_gap} giay"
    )


# =========================================================
# TEST 1: EXPANDING WINDOW CV
# =========================================================

def test_expanding_cv_no_data_leakage_and_enough_episodes(sample_data):

    df, episodes_df = sample_data

    splitter = TimeSeriesDataSplitter(
        buffer_gap_mins=60,
        clean_train_only=True,
        timestamp_col="ts_epoch",
        anomaly_col="y",
        episodes_df=episodes_df,
    )

    folds = list(
        splitter.split_expanding_cv(
            df,
            n_splits=4,
            min_train_ratio=0.4,
        )
    )

    # Phai tao du 4 folds.
    assert len(folds) == 4, (
        f"Expanding CV phai co 4 folds, hien tai co {len(folds)}"
    )

    for fold in folds:

        # Kiem tra khong ro ri du lieu.
        check_no_time_leakage(fold)

        # Kiem tra moi tap co du Episode.
        train_episode_count = count_complete_episodes(
            fold.train_df,
            episodes_df,
        )

        test_episode_count = count_complete_episodes(
            fold.test_df,
            episodes_df,
        )

        assert train_episode_count >= 1, (
            f"Fold {fold.fold_idx}: Train khong co Episode hoan chinh"
        )

        assert test_episode_count >= 1, (
            f"Fold {fold.fold_idx}: Validation khong co Episode hoan chinh"
        )


# =========================================================
# TEST 2: ROLLING WINDOW CV
# =========================================================

def test_rolling_cv_no_data_leakage_and_enough_episodes(sample_data):

    df, episodes_df = sample_data

    splitter = TimeSeriesDataSplitter(
        buffer_gap_mins=60,
        clean_train_only=True,
        timestamp_col="ts_epoch",
        anomaly_col="y",
        episodes_df=episodes_df,
    )

    folds = list(
        splitter.split_rolling_cv(
            df,
            n_splits=4,
            train_window_ratio=0.4,
            test_window_ratio=0.15,
        )
    )

    # Phai tao du 4 folds.
    assert len(folds) == 4, (
        f"Rolling CV phai co 4 folds, hien tai co {len(folds)}"
    )

    for fold in folds:

        # Kiem tra khong ro ri du lieu.
        check_no_time_leakage(fold)

        # Kiem tra so Episode trong moi fold.
        train_episode_count = count_complete_episodes(
            fold.train_df,
            episodes_df,
        )

        test_episode_count = count_complete_episodes(
            fold.test_df,
            episodes_df,
        )

        assert train_episode_count >= 1, (
            f"Fold {fold.fold_idx}: Train khong co Episode hoan chinh"
        )

        assert test_episode_count >= 1, (
            f"Fold {fold.fold_idx}: Validation khong co Episode hoan chinh"
        )
