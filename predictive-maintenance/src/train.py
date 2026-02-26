"""Model training pipeline with MLflow and XGBoost."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import FeatureConfig, FeatureEngineer, build_training_matrices, time_split

DATA_PATH = Path("predictive-maintenance/data/sensor_data.csv")
MODEL_PATH = Path("predictive-maintenance/models/model.joblib")
META_PATH = Path("predictive-maintenance/models/model_metadata.json")
STATS_PATH = Path("predictive-maintenance/models/training_stats.json")


def evaluate(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run data_simulation.py first.")

    raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    fe = FeatureEngineer(FeatureConfig(rolling_window=12, lag_steps=1))
    engineered = fe.transform(raw)
    train_df, test_df = time_split(engineered, train_ratio=0.8)

    X_train, y_train, feature_cols = build_training_matrices(train_df)
    X_test, y_test, _ = build_training_matrices(test_df)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=2,
                ),
            ),
        ]
    )

    param_dist = {
        "model__n_estimators": [150, 250, 400],
        "model__max_depth": [3, 4, 6],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.8, 0.95, 1.0],
        "model__colsample_bytree": [0.8, 0.95, 1.0],
    }

    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=8,
        scoring="f1",
        cv=tscv,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )

    mlflow.set_tracking_uri("file:./predictive-maintenance/mlruns")
    mlflow.set_experiment("predictive-maintenance")

    with mlflow.start_run(run_name="xgboost_timeseries"):
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        probs = best_model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, probs)

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # Register best model in local MLflow registry.
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        registered = mlflow.register_model(model_uri=model_uri, name="predictive_maintenance_xgb")
        mlflow.set_tag("registered_model_version", registered.version)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        meta = {
            "feature_columns": feature_cols,
            "threshold": 0.5,
            "best_params": search.best_params_,
            "metrics": metrics,
        }
        META_PATH.write_text(json.dumps(meta, indent=2))

        stats = {
            "reference_means": X_train.mean().to_dict(),
            "reference_stds": X_train.std(ddof=0).replace(0, 1e-6).to_dict(),
        }
        STATS_PATH.write_text(json.dumps(stats, indent=2))

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
