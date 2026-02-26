"""Synthetic data generation for predictive maintenance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


SENSORS: List[str] = ["temperature", "vibration", "pressure", "humidity", "rpm"]


@dataclass
class SimulationConfig:
    n_machines: int = 20
    points_per_machine: int = 1500
    freq: str = "5min"
    seed: int = 42
    anomaly_rate: float = 0.01
    failure_threshold: float = 0.82
    output_path: str = "predictive-maintenance/data/sensor_data.csv"


def _base_signal(sensor: str, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 12 * np.pi, n)
    if sensor == "temperature":
        return 65 + 4 * np.sin(t / 2) + rng.normal(0, 0.8, n)
    if sensor == "vibration":
        return 0.9 + 0.15 * np.sin(t * 1.3) + rng.normal(0, 0.04, n)
    if sensor == "pressure":
        return 32 + 1.8 * np.cos(t / 3) + rng.normal(0, 0.5, n)
    if sensor == "humidity":
        return 45 + 7 * np.sin(t / 5) + rng.normal(0, 1.2, n)
    if sensor == "rpm":
        return 1420 + 60 * np.sin(t / 4) + rng.normal(0, 12, n)
    raise ValueError(f"Unknown sensor: {sensor}")


def simulate_data(config: SimulationConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    frames: List[pd.DataFrame] = []

    start = pd.Timestamp("2024-01-01 00:00:00")

    for machine_id in range(config.n_machines):
        ts = pd.date_range(start, periods=config.points_per_machine, freq=config.freq)
        n = len(ts)

        # Degradation gradually increases with random machine-specific dynamics.
        slope = rng.uniform(0.00025, 0.0009)
        degradation = np.clip(np.cumsum(rng.normal(slope, slope / 3, n)), 0, 1)

        data = {"timestamp": ts, "machine_id": machine_id, "degradation": degradation}
        for sensor in SENSORS:
            base = _base_signal(sensor, n, rng)
            if sensor == "temperature":
                data[sensor] = base + 22 * degradation
            elif sensor == "vibration":
                data[sensor] = base + 1.6 * degradation
            elif sensor == "pressure":
                data[sensor] = base - 6.5 * degradation
            elif sensor == "humidity":
                data[sensor] = base + 8 * degradation
            elif sensor == "rpm":
                data[sensor] = base - 190 * degradation

        df = pd.DataFrame(data)

        # Inject random anomalies.
        anomaly_mask = rng.random(n) < config.anomaly_rate
        anomaly_magnitude = rng.normal(0, 1, n)
        for sensor in SENSORS:
            scale = {
                "temperature": 15,
                "vibration": 1.2,
                "pressure": 8,
                "humidity": 15,
                "rpm": 250,
            }[sensor]
            df.loc[anomaly_mask, sensor] += anomaly_magnitude[anomaly_mask] * scale

        df["is_anomaly"] = anomaly_mask.astype(int)
        df["failure"] = ((df["degradation"] > config.failure_threshold) | anomaly_mask).astype(int)
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "machine_id"])
    return full_df


def main() -> None:
    config = SimulationConfig()
    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = simulate_data(config)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
