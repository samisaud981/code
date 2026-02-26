"""Feature engineering utilities for predictive maintenance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

SENSORS = ["temperature", "vibration", "pressure", "humidity", "rpm"]


@dataclass
class FeatureConfig:
    rolling_window: int = 12
    lag_steps: int = 1


class FeatureEngineer:
    """Construct lagged and rolling time-series features per machine."""

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.sort_values(["machine_id", "timestamp"]).copy()

        for sensor in SENSORS:
            grouped = work.groupby("machine_id")[sensor]
            work[f"{sensor}_lag{self.config.lag_steps}"] = grouped.shift(self.config.lag_steps)
            work[f"{sensor}_rolling_mean"] = grouped.transform(
                lambda x: x.rolling(self.config.rolling_window, min_periods=1).mean()
            )
            work[f"{sensor}_rolling_std"] = grouped.transform(
                lambda x: x.rolling(self.config.rolling_window, min_periods=2).std()
            )
            work[f"{sensor}_roc"] = grouped.pct_change().replace([pd.NA], 0.0)

        work = work.fillna(method="bfill").fillna(method="ffill").fillna(0.0)
        return work


def build_training_matrices(df: pd.DataFrame, target_col: str = "failure") -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_cols = [
        c
        for c in df.columns
        if c not in {"timestamp", "machine_id", target_col, "is_anomaly"}
    ]
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    return X, y, feature_cols


def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = df.sort_values("timestamp").copy()
    split_idx = int(len(sorted_df) * train_ratio)
    train_df = sorted_df.iloc[:split_idx]
    test_df = sorted_df.iloc[split_idx:]
    return train_df, test_df
