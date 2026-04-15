# Fraud Detection MLOps

An end-to-end machine learning system for real-time credit card fraud detection, built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset.

## Results

| Metric | Value |
|--------|-------|
| AUPRC | 0.75 |
| F2 Score | 0.71 |
| Best Threshold | 0.62 |
| Class imbalance | ~3.5% fraud |

F2 score is used as the primary metric because recall is prioritized over precision — missing fraud is more costly than a false alarm.

## Architecture

```
preprocess.py  →  train.py  →  MLflow Registry  →  serve.py  →  /predict
                                  (champion alias)
```

- **Preprocessing**: log transform, null indicator features, one-hot encoding (low-cardinality), target encoding (high-cardinality)
- **Training**: XGBoost with `scale_pos_weight` for class imbalance; F2-optimized threshold via precision-recall curve
- **Experiment tracking**: MLflow — params, metrics, encoder artifacts, and model all logged in the same run
- **Model registry**: champion alias pattern — new model auto-promoted only if F2 exceeds current champion
- **Serving**: FastAPI app loads champion model + artifacts from MLflow on startup, returns fraud probability and label

## Project Structure

```
├── src/
│   ├── preprocess.py     # feature engineering, OHE, target encoding
│   ├── train.py          # MLflow experiment tracking, champion promotion
│   ├── evaluate.py       # AUPRC, F2 threshold selection
│   ├── serve.py          # FastAPI prediction endpoint
│   ├── test.py           # integration test against live API
│   └── monitor.py        # drift and performance monitoring (WIP)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── notebooks/
│   └── eda.ipynb
└── requirements.txt
```

## Running Locally

**1. Install dependencies**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Start MLflow server** (from project root)
```bash
mlflow server --host 127.0.0.1 --port 5000
```

**3. Run the pipeline**
```bash
cd src
python preprocess.py   # saves artifacts to data/processed/
python train.py        # trains, logs to MLflow, promotes champion
```

**4. Start the API**
```bash
cd src
uvicorn serve:app --host 0.0.0.0 --port 8000
```

**5. Test**
```bash
python src/test.py
# or visit http://localhost:8000/docs
```

## Running with Docker

MLflow must be running on the host first (step 2 above), then:

```bash
docker compose -f docker/docker-compose.yml up --build
```

The container connects to MLflow via `host.docker.internal:5000` and downloads the champion model and encoder artifacts on startup.

## API

`POST /predict`

Accepts any subset of the transaction + identity fields. Returns:

```json
{
  "prediction_probability": 0.87,
  "label": "fraud"
}
```

## Tech Stack

- **Model**: XGBoost
- **Experiment tracking**: MLflow
- **Serving**: FastAPI + Uvicorn
- **Containerization**: Docker
- **Data**: IEEE-CIS Fraud Detection (Kaggle)
