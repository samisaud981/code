# Predictive Maintenance ML System

Production-style end-to-end predictive maintenance project with data simulation, feature engineering, model training (XGBoost + MLflow), FastAPI inference, monitoring, and a dynamic Streamlit UI.

## Architecture

1. **Data simulation (`src/data_simulation.py`)**
   - Generates multivariate time-series sensor data (`temperature`, `vibration`, `pressure`, `humidity`, `rpm`).
   - Injects gradual degradation and random anomalies.
   - Creates binary failure labels.
2. **Feature pipeline (`src/features.py`)**
   - Rolling mean/std, lag features, rate-of-change.
   - Time-based train/test split.
3. **Training (`src/train.py`)**
   - XGBoost classifier in sklearn pipeline.
   - Hyperparameter tuning with `RandomizedSearchCV` + `TimeSeriesSplit`.
   - Logs experiments to MLflow and registers model version.
   - Saves model artifact and metadata.
4. **Inference + monitoring (`src/inference.py`, `src/monitoring.py`)**
   - Returns failure probability, anomaly score, binary prediction.
   - Drift detection against training reference statistics.
   - JSONL logging for all predictions.
5. **Serving (`api/main.py`)**
   - FastAPI `/predict` endpoint.
6. **Interactive app (`app/streamlit_app.py`)**
   - Dynamic controls for sensor values.
   - Calls API and visualizes prediction output.

## Project Structure

```text
predictive-maintenance/
├── app/
│   └── streamlit_app.py
├── api/
│   └── main.py
├── data/
├── mlruns/
├── models/
├── src/
│   ├── data_simulation.py
│   ├── features.py
│   ├── inference.py
│   ├── monitoring.py
│   └── train.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Setup

```bash
cd predictive-maintenance
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Training

```bash
python src/data_simulation.py
python src/train.py
```

Artifacts:
- Dataset: `data/sensor_data.csv`
- Model: `models/model.joblib`
- Metadata: `models/model_metadata.json`
- Monitoring stats: `models/training_stats.json`
- MLflow runs: `mlruns/`

## Start API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Start Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Open `http://localhost:8501`.

## Example `curl` Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 88.2,
    "vibration": 1.9,
    "pressure": 28.3,
    "humidity": 56.1,
    "rpm": 1240,
    "degradation": 0.73
  }'
```

## Docker

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`
