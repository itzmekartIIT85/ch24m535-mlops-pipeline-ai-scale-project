“End Term Project: Robust MLOps Pipeline (ID5003W, IITM)”
#  Titanic Survival Prediction (MLOps Project)

This project demonstrates an **end-to-end ML pipeline** with:
- Data preprocessing (Spark)
- Model training & hyperparameter tuning (Spark MLlib)
- Experiment tracking & model registry (MLflow + SQLite backend)
- FastAPI service for inference
- Optional Docker containerization

---

## Project Structure
├── configs/ # Configuration files (optional)
├── data/
│ ├── raw/ # Raw datasets (titanic.csv, not in Git)
│ └── processed/ # Processed data + stats (ignored in Git)
├── notebooks/ # Jupyter notebooks for exploration
├── reports/ # Reports, plots, artifacts
├── src/
│ ├── data_pipeline.py # Spark preprocessing pipeline
│ ├── train.py # Model training + MLflow logging
│ ├── app.py # FastAPI serving
│ ├── test_script.py # Simple API test
│ └── ...
├── requirements.txt
└── README.md


---

## ⚙️ Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/titanic-mlops.git
   cd titanic-mlops
2. Create a conda/venv environment:
conda create -n titanic python=3.9
conda activate titanic
pip install -r requirements.txt
3.(Optional) Setup MLflow UI locally:

mlflow ui --backend-store-uri sqlite:///mlflow.db

# Data Pipeline

Run preprocessing (imputation, encoding, feature engineering):

python src/data_pipeline.py


This generates:

data/processed/titanic/ (parquet)

data/processed/impute_stats.json

models/preprocess_pipeline/

# Training

Train + track models with MLflow:

python src/train.py


Artifacts logged to MLflow:

Metrics: AUC, Accuracy, Precision, Recall, F1

Confusion matrix

Feature importances / coefficients

Registered model: TitanicClassifier

# Serving (FastAPI)

Run the API:

uvicorn src.app:app --reload


Endpoints:

GET / → Healthcheck

POST /predict → Predict survival from passenger features

Example request:

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Pclass": 3, "Sex": "male", "Age": 22, "Fare": 7.25, "Embarked": "S"}'


Response:

{
  "survived": 0,
  "probability_survive": 0.0532
}

# Docker (Optional)

Build image:

docker build -t titanic-api .


Run container:

docker run -p 8000:8000 titanic-api
