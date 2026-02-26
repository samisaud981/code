"""Inference utilities for batch/online prediction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from monitoring import Monitor, MonitoringConfig

SENSORS: List[str] = ["temperature", "vibration", "pressure", "humidity", "rpm"]
MODEL_PATH = Path("predictive-maintenance/models/model.joblib")
META_PATH = Path("predictive-maintenance/models/model_metadata.json")


class InferenceService:
    def __init__(self) -> None:
        self.model = joblib.load(MODEL_PATH)
        self.meta = json.loads(META_PATH.read_text())
        self.feature_columns: List[str] = self.meta["feature_columns"]
        self.threshold: float = float(self.meta.get("threshold", 0.5))
        self.monitor = Monitor(MonitoringConfig())

    @staticmethod
    def _build_features(payload: Dict[str, float]) -> pd.DataFrame:
        feat: Dict[str, float] = {
            "degradation": float(payload.get("degradation", 0.5)),
        }

        for sensor in SENSORS:
            val = float(payload[sensor])
            feat[sensor] = val
            feat[f"{sensor}_lag1"] = float(payload.get(f"{sensor}_lag1", val))
            feat[f"{sensor}_rolling_mean"] = float(payload.get(f"{sensor}_rolling_mean", val))
            feat[f"{sensor}_rolling_std"] = float(payload.get(f"{sensor}_rolling_std", 0.0))
            feat[f"{sensor}_roc"] = float(payload.get(f"{sensor}_roc", 0.0))

        return pd.DataFrame([feat])

    def predict(self, payload: Dict[str, float]) -> Dict[str, float | int | bool]:
        features = self._build_features(payload)
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0.0
        features = features[self.feature_columns]

        failure_prob = float(self.model.predict_proba(features)[0, 1])
        anomaly_score = float(np.mean(np.abs(features.values - features.values.mean())))
        binary_pred = int(failure_prob >= self.threshold)

        drift_scores = self.monitor.detect_drift(features)
        result = {
            "failure_probability": failure_prob,
            "anomaly_score": anomaly_score,
            "binary_prediction": binary_pred,
            "drift_detected": self.monitor.is_drifted(drift_scores),
        }
        self.monitor.log_prediction(payload=payload, result=result, drift_scores=drift_scores)
        return result
