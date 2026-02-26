"""FastAPI service for predictive maintenance inference."""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from inference import InferenceService  # noqa: E402


class SensorPayload(BaseModel):
    temperature: float = Field(..., description="Current temperature reading")
    vibration: float = Field(..., description="Current vibration reading")
    pressure: float = Field(..., description="Current pressure reading")
    humidity: float = Field(..., description="Current humidity reading")
    rpm: float = Field(..., description="Current motor rpm reading")
    degradation: Optional[float] = Field(0.5, ge=0.0, le=1.0)


app = FastAPI(title="Predictive Maintenance API", version="1.0.0")
service = InferenceService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: SensorPayload) -> dict:
    return service.predict(payload.model_dump())
