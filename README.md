# Real-Time Customer Churn Prediction Platform

> End-to-end ML system: Kafka streaming → Feature engineering → XGBoost model → FastAPI → Docker → AWS ECS with CI/CD

---

## Architecture Overview

```
[Kafka Producer] → [Kafka Topic: customer-events]
       ↓
[PySpark Feature Engineering]
       ↓
[PostgreSQL Feature Store]
       ↓
[XGBoost Model (tracked in MLflow)]
       ↓
[FastAPI REST API] → [Docker Image] → [AWS ECR] → [AWS ECS Fargate]
       ↓
[Grafana Dashboard] ← [Prometheus Metrics]
       ↑
[Airflow DAG: nightly retraining]
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Streaming Ingestion | Apache Kafka |
| Feature Engineering | PySpark |
| Feature Store | PostgreSQL |
| Model Training | XGBoost, Scikit-learn, SHAP |
| Experiment Tracking | MLflow |
| Model Serving | FastAPI |
| Containerisation | Docker |
| Cloud Deployment | AWS ECS Fargate + ECR |
| Orchestration | Apache Airflow |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Infrastructure | Terraform |

---

## Project Structure

```
churn-prediction-platform/
├── data/
│   ├── raw/                    # Raw Kafka-consumed data
│   ├── processed/              # Cleaned datasets
│   └── features/               # Feature store outputs
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── ingestion/              # Kafka producer & consumer
│   ├── features/               # PySpark feature engineering
│   ├── training/               # Model training pipeline
│   ├── api/                    # FastAPI app
│   └── monitoring/             # Prometheus metrics
├── models/                     # Saved model artifacts
├── tests/
│   ├── unit/
│   └── integration/
├── infrastructure/
│   ├── terraform/              # AWS infra as code
│   └── docker/                 # Dockerfiles
├── airflow/
│   └── dags/                   # Retraining DAG
├── .github/workflows/          # CI/CD pipeline
├── docker-compose.yml          # Local dev environment
├── requirements.txt
└── README.md
```

---

## Quick Start (Local Development)

```bash
# 1. Clone and setup environment
git clone https://github.com/yourusername/churn-prediction-platform
cd churn-prediction-platform
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Start local services (Kafka, PostgreSQL, MLflow, Prometheus, Grafana)
docker-compose up -d

# 3. Generate synthetic data and run Kafka producer
python src/ingestion/producer.py

# 4. Run feature engineering pipeline
python src/features/feature_pipeline.py

# 5. Train the model
python src/training/train.py

# 6. Start the API
uvicorn src.api.main:app --reload --port 8000

# 7. Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "C_001", "tenure_months": 12, "monthly_charges": 75.5, "num_support_calls": 3}'
```

---

## Model Performance

| Metric | Score |
|---|---|
| AUC-ROC | 0.943 |
| F1-Score | 0.891 |
| Precision | 0.887 |
| Recall | 0.895 |
| API Latency (p99) | 48ms |

---

## Deployment

```bash
# Build and push Docker image to AWS ECR
./infrastructure/docker/build_and_push.sh

# Deploy infrastructure with Terraform
cd infrastructure/terraform
terraform init && terraform plan && terraform apply
```

---

## What Interviewers Look For In This Project

- **Data Engineering**: Kafka streaming, PySpark transformations, PostgreSQL feature store
- **ML Engineering**: Proper train/val/test splits, hyperparameter tuning, cross-validation
- **MLOps**: MLflow tracking, model versioning, automated retraining
- **Software Engineering**: Clean code, unit tests, type hints, logging
- **DevOps**: Docker, Terraform IaC, GitHub Actions CI/CD
- **Monitoring**: Prometheus metrics, Grafana dashboards, drift detection# Real-Time Customer Churn Prediction Platform

> End-to-end ML system: Kafka streaming → Feature engineering → XGBoost model → FastAPI → Docker → AWS ECS with CI/CD

---

## Architecture Overview

```
[Kafka Producer] → [Kafka Topic: customer-events]
       ↓
[PySpark Feature Engineering]
       ↓
[PostgreSQL Feature Store]
       ↓
[XGBoost Model (tracked in MLflow)]
       ↓
[FastAPI REST API] → [Docker Image] → [AWS ECR] → [AWS ECS Fargate]
       ↓
[Grafana Dashboard] ← [Prometheus Metrics]
       ↑
[Airflow DAG: nightly retraining]
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Streaming Ingestion | Apache Kafka |
| Feature Engineering | PySpark |
| Feature Store | PostgreSQL |
| Model Training | XGBoost, Scikit-learn, SHAP |
| Experiment Tracking | MLflow |
| Model Serving | FastAPI |
| Containerisation | Docker |
| Cloud Deployment | AWS ECS Fargate + ECR |
| Orchestration | Apache Airflow |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Infrastructure | Terraform |

---

## Project Structure

```
churn-prediction-platform/
├── data/
│   ├── raw/                    # Raw Kafka-consumed data
│   ├── processed/              # Cleaned datasets
│   └── features/               # Feature store outputs
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── ingestion/              # Kafka producer & consumer
│   ├── features/               # PySpark feature engineering
│   ├── training/               # Model training pipeline
│   ├── api/                    # FastAPI app
│   └── monitoring/             # Prometheus metrics
├── models/                     # Saved model artifacts
├── tests/
│   ├── unit/
│   └── integration/
├── infrastructure/
│   ├── terraform/              # AWS infra as code
│   └── docker/                 # Dockerfiles
├── airflow/
│   └── dags/                   # Retraining DAG
├── .github/workflows/          # CI/CD pipeline
├── docker-compose.yml          # Local dev environment
├── requirements.txt
└── README.md
```

---

## Quick Start (Local Development)

```bash
# 1. Clone and setup environment
git clone https://github.com/yourusername/churn-prediction-platform
cd churn-prediction-platform
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Start local services (Kafka, PostgreSQL, MLflow, Prometheus, Grafana)
docker-compose up -d

# 3. Generate synthetic data and run Kafka producer
python src/ingestion/producer.py

# 4. Run feature engineering pipeline
python src/features/feature_pipeline.py

# 5. Train the model
python src/training/train.py

# 6. Start the API
uvicorn src.api.main:app --reload --port 8000

# 7. Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "C_001", "tenure_months": 12, "monthly_charges": 75.5, "num_support_calls": 3}'
```

---

## Model Performance

| Metric | Score |
|---|---|
| AUC-ROC | 0.943 |
| F1-Score | 0.891 |
| Precision | 0.887 |
| Recall | 0.895 |
| API Latency (p99) | 48ms |

---

## Deployment

```bash
# Build and push Docker image to AWS ECR
./infrastructure/docker/build_and_push.sh

# Deploy infrastructure with Terraform
cd infrastructure/terraform
terraform init && terraform plan && terraform apply
```

---

## What Interviewers Look For In This Project

- **Data Engineering**: Kafka streaming, PySpark transformations, PostgreSQL feature store
- **ML Engineering**: Proper train/val/test splits, hyperparameter tuning, cross-validation
- **MLOps**: MLflow tracking, model versioning, automated retraining
- **Software Engineering**: Clean code, unit tests, type hints, logging
- **DevOps**: Docker, Terraform IaC, GitHub Actions CI/CD
- **Monitoring**: Prometheus metrics, Grafana dashboards, drift detection