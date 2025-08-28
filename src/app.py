# app.py — FastAPI serving Titanic model with preprocessing pipeline
import json
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from mlflow.tracking import MlflowClient

# -------------------
# Config paths
# -------------------
PREPROCESS_PATH = "/home/karthik/mlops-pipeline-ch24m535/models/preprocess_pipeline"
STATS_PATH = "/home/karthik/mlops-pipeline-ch24m535/data/processed/impute_stats.json"

# -------------------
# MLflow model loading (latest version of TitanicClassifier)
# -------------------
EXPERIMENT_NAME = "Titanic_Classification"
MODEL_NAME = "TitanicClassifier"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Get latest version of model (any stage, highest version number)
latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging", "Production"])
print(f"Found the {len(latest_versions)} registered versions for model: {MODEL_NAME}")
if not latest_versions:
    raise RuntimeError(f"No registered versions found for model: {MODEL_NAME}")

# Sort by version and pick the highest
latest_model = sorted(latest_versions, key=lambda m: int(m.version))[-1]
MODEL_URI = f"models:/{MODEL_NAME}/{latest_model.version}"

print(f"✅ Loading latest model: {MODEL_NAME} v{latest_model.version} (stage={latest_model.current_stage})")


# -------------------
# FastAPI app
# -------------------
app = FastAPI(title="Titanic Survival Prediction API")

# Spark session (reused for preprocessing)
spark = SparkSession.builder.appName("TitanicAPI").getOrCreate()

# Load preprocessing pipeline + imputation stats
preprocess_model = PipelineModel.load(PREPROCESS_PATH)
with open(STATS_PATH, "r") as f:
    stats = json.load(f)

# Load trained MLflow model
model = mlflow.spark.load_model(MODEL_URI)

# -------------------
# Input Schema (raw features, not encoded)
# -------------------
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    Embarked: str 

# -------------------
# Helper: preprocessing
# -------------------
def preprocess_input(passenger: Passenger):
    """Convert raw JSON → Spark DataFrame → imputation → pipeline transform"""
    data_dict = passenger.dict()

    # Apply imputation defaults
    if data_dict.get("Age") is None:
        data_dict["Age"] = stats["Age_mean"]
    if data_dict.get("Embarked") is None:
        data_dict["Embarked"] = stats["Embarked_mode"]

    # Convert to Spark DataFrame
    df = spark.createDataFrame([data_dict])

    # Apply saved preprocessing pipeline (StringIndexer, Bucketizer, VectorAssembler)
    df = preprocess_model.transform(df)
    return df


# -------------------
# Routes
# -------------------
@app.post("/predict")
def predict(passenger: Passenger):
    try:
        # Preprocess
        df = preprocess_input(passenger)

        # Run prediction
        preds = model.transform(df).select("prediction", "probability").collect()[0]
        prediction = int(preds["prediction"])
        prob_survive = float(preds["probability"][1])  # probability of class 1

        return {
            "survived": prediction,
            "probability_survive": round(prob_survive, 4),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def healthcheck():
    return {"status": "ok", "message": "Titanic Survival API running!"}