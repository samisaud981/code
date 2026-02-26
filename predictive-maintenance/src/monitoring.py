"""Monitoring utilities for drift and prediction logging."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class MonitoringConfig:
    stats_path: str = "predictive-maintenance/models/training_stats.json"
    log_path: str = "predictive-maintenance/models/prediction_logs.jsonl"
    drift_threshold_z: float = 3.0


class Monitor:
    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        stats = json.loads(Path(config.stats_path).read_text())
        self.reference_means: Dict[str, float] = stats["reference_means"]
        self.reference_stds: Dict[str, float] = stats["reference_stds"]
        Path(config.log_path).parent.mkdir(parents=True, exist_ok=True)

    def detect_drift(self, feature_frame: pd.DataFrame) -> Dict[str, float]:
        drift_scores: Dict[str, float] = {}
        for col, ref_mean in self.reference_means.items():
            if col not in feature_frame.columns:
                continue
            current_mean = float(feature_frame[col].mean())
            std = float(self.reference_stds.get(col, 1.0))
            z = abs(current_mean - ref_mean) / max(std, 1e-6)
            drift_scores[col] = float(z)
        return drift_scores

    def is_drifted(self, drift_scores: Dict[str, float]) -> bool:
        if not drift_scores:
            return False
        return bool(np.max(list(drift_scores.values())) > self.config.drift_threshold_z)

    def log_prediction(self, payload: Dict, result: Dict, drift_scores: Dict[str, float]) -> None:
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input": payload,
            "result": result,
            "drift_scores": drift_scores,
            "drift_detected": self.is_drifted(drift_scores),
        }
        with open(self.config.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
