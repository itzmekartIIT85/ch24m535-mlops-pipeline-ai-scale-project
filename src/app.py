"""
FastAPI Service for Titanic Survival Prediction
----------------------------------------------
1. Loads trained Spark Logistic Regression model.
2. Provides a REST API endpoint (/predict) for predictions.
3. Accepts passenger features as JSON and returns survival prediction.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import StringIndexerModel, VectorAssembler
from pyspark.ml import PipelineModel
from pyspark.sql import Row

# ---------------------------
# Define request schema
# ---------------------------
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    Embarked: str

# ---------------------------
# Initialize FastAPI
# ---------------------------
app = FastAPI()

# ---------------------------
# Initialize Spark
# ---------------------------
spark = SparkSession.builder.appName("TitanicAPI").getOrCreate()

# Load trained model
model = LogisticRegressionModel.load("/home/karthik/mlops-pipeline-ch24m535/models/titanic_lr")

# NOTE: You should also reload preprocessing pipeline if saved separately
# For now, we will manually encode categorical features inside API

# ---------------------------
# Utility: Manual preprocessing
# (should ideally come from saved preprocessing pipeline)
# ---------------------------
def preprocess(passenger: Passenger):
    # Simple manual encodings (must match training phase)
    sex_map = {"male": 1.0, "female": 0.0}
    embarked_map = {"S": 0.0, "C": 1.0, "Q": 2.0}

    return Row(
        Pclass=passenger.Pclass,
        Age=passenger.Age,
        Fare=passenger.Fare,
        SexIndexed=sex_map.get(passenger.Sex.lower(), 0.0),
        EmbarkedIndexed=embarked_map.get(passenger.Embarked.upper(), 0.0),
    )

# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/predict")
def predict(passenger: Passenger):
    # Convert input into Spark DataFrame
    row = preprocess(passenger)
    df = spark.createDataFrame([row])

    # Assemble features
    assembler = VectorAssembler(
        inputCols=["Pclass", "Age", "Fare", "SexIndexed", "EmbarkedIndexed"],
        outputCol="features"
    )
    df = assembler.transform(df)

    # Predict
    prediction = model.transform(df).collect()[0]
    survival = int(prediction.prediction)

    return {"Survived": survival, "Probability": float(prediction.probability[1])}