import pytest
import pandas as pd

from src.evaluation.splitter import TimeSeriesDataSplitter


@pytest.fixture
def sample_df():
    """Tao du lieu gia lap theo thoi gian."""
    base_ts = 1_700_000_000

    df = pd.DataFrame({
        "win": [base_ts + i * 600 for i in range(200)],
        "y": [0] * 200,
        "feature": range(200),
    })

    # Tao cac mau bat thuong
    df.loc[[2, 4, 180], "y"] = 1

    return df


def test_buffer_gap_must_be_at_least_60_minutes():
    """Kiem tra buffer gap toi thieu 60 phut."""

    # Hien tai splitter chua chan gia tri duoi 60.
    # Test nay se phat hien loi do.
    with pytest.raises(ValueError):
        TimeSeriesDataSplitter(buffer_gap_mins=30)

    splitter = TimeSeriesDataSplitter(buffer_gap_mins=60)

    assert splitter.buffer_gap_seconds == 3600


def test_clean_train_only_removes_anomalies(sample_df):
    """Train khong duoc chua mau bat thuong khi clean_train_only=True."""

    splitter = TimeSeriesDataSplitter(
        buffer_gap_mins=60,
        clean_train_only=True,
        timestamp_col="win",
        anomaly_col="y",
    )

    fold = next(
        splitter.split_holdout_by_time(
            sample_df,
            train_ratio=0.5,
        )
    )

    assert not fold.train_df.empty
    assert (fold.train_df["y"] == 0).all()
    assert fold.clean_train_applied is True


def test_train_always_before_test(sample_df):
    """Train phai nam truoc Test theo thoi gian."""

    splitter = TimeSeriesDataSplitter(
        buffer_gap_mins=60,
        clean_train_only=False,
        timestamp_col="win",
    )

    fold = next(
        splitter.split_holdout_by_time(
            sample_df,
            train_ratio=0.5,
        )
    )

    assert not fold.train_df.empty
    assert not fold.test_df.empty

    assert fold.train_df["win"].max() < fold.test_df["win"].min()

    actual_gap = fold.test_start_ts - fold.train_end_ts

    assert actual_gap >= 3600


def test_episode_is_not_split_between_train_and_test():
    """Mot episode khong duoc chia vao ca Train va Test."""

    base_ts = 1_700_000_000

    rows = []
    episodes = []

    for episode_id in range(6):
        start_ts = base_ts + episode_id * 7200
        end_ts = start_ts + 1200

        episodes.append({
            "episode_id": episode_id,
            "start_ts": start_ts,
            "end_ts": end_ts,
        })

        for offset in [0, 600, 1200]:
            rows.append({
                "win": start_ts + offset,
                "y": 1,
                "episode_id": episode_id,
            })

    df = pd.DataFrame(rows)
    episodes_df = pd.DataFrame(episodes)

    splitter = TimeSeriesDataSplitter(
        buffer_gap_mins=60,
        clean_train_only=False,
        timestamp_col="win",
        anomaly_col="y",
        episodes_df=episodes_df,
    )

    fold = next(
        splitter.split_holdout_by_episodes(
            df,
            train_ratio=0.5,
        )
    )

    train_ids = set(fold.train_df["episode_id"])
    test_ids = set(fold.test_df["episode_id"])

    assert train_ids.isdisjoint(test_ids)

    for episode_id in range(6):
        in_train = (
            fold.train_df["episode_id"] == episode_id
        ).sum()

        in_test = (
            fold.test_df["episode_id"] == episode_id
        ).sum()

        assert (
            (in_train == 3 and in_test == 0)
            or (in_train == 0 and in_test == 3)
        )