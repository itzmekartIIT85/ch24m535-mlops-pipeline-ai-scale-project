"""
Titanic Model Training with Spark + MLflow
------------------------------------------
This script:
1. Loads processed Titanic dataset.
2. Trains a Logistic Regression classifier (binary classification).
3. Logs parameters, metrics, and model to MLflow.
4. Saves trained model locally for deployment.
"""

import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def main():
    # Start Spark session
    spark = SparkSession.builder.appName("TitanicTraining").getOrCreate()

    # Load processed dataset
    df = spark.read.parquet("/home/karthik/mlops-pipeline-ch24m535/data/processed/titanic")

    # Train-test split
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # Define Logistic Regression model
    lr = LogisticRegression(labelCol="Survived", featuresCol="features", maxIter=20)

    # Setup MLflow tracking
    mlflow.set_experiment("Titanic_Classification")

    with mlflow.start_run():
        # Train model
        lr_model = lr.fit(train)

        # Evaluate on test set
        predictions = lr_model.transform(test)
        evaluator = BinaryClassificationEvaluator(
            labelCol="Survived", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
        )
        auc = evaluator.evaluate(predictions)

        # Log parameters and metrics
        mlflow.log_param("maxIter", 20)
        mlflow.log_metric("AUC", auc)

        # Log Spark model
        mlflow.spark.log_model(lr_model, "model")

        print(f"✅ Model trained and logged to MLflow with AUC = {auc:.4f}")

        # Save locally as well (for FastAPI serving later)
        lr_model.write().overwrite().save("/home/karthik/mlops-pipeline-ch24m535/models/titanic_lr")

    spark.stop()

if __name__ == "__main__":
    main()
