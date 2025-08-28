"""
train.py — distributed training (Spark ML) + HPO + MLflow tracking + model registry
- Loads processed Titanic parquet (must include 'features' column)
- Grid-search over candidate estimators (LogReg, RandomForest)
- Logs params, AUC, confusion matrix, feature importance/coefficients to MLflow
- Registers best run to MLflow Model Registry (fallback: local save)
"""

import os
import itertools
import traceback
import json
import tempfile

import mlflow
import mlflow.spark
from pyspark.sql import SparkSession

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# extra libs for artifacts
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------
# Helpers
# ----------------------
def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_feature_importances(feat_names, importances, out_path, title="Feature Importances"):
    df = pd.DataFrame({"feature": feat_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(50)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(df))))
    ax.barh(df["feature"][::-1], df["importance"][::-1])
    ax.set_title(title)
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def save_feature_table(feat_names, values, csv_path):
    df = pd.DataFrame({"feature": feat_names, "value": list(values)})
    df.to_csv(csv_path, index=False)

# ----------------------
# Main
# ----------------------
def main():
    spark = None
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Titanic_Classification")

        spark = SparkSession.builder.appName("TitanicTraining").getOrCreate()

        # load processed data (must have 'features' col)
        processed_path = "/home/karthik/mlops-pipeline-ch24m535/data/processed/titanic"
        try:
            df = spark.read.parquet(processed_path)
            print(f" Loaded processed data: {df.count()} rows")
        except Exception as e:
            print(f" Error loading processed data at {processed_path}: {e}")
            return

        # Try to load exact feature names produced by preprocessing.
        # preprocess.py can write a feature_names.json file listing assembler inputCols in order.
        feature_names_file = os.path.join(os.path.dirname(processed_path), "feature_names.json")
        feature_names = None
        if os.path.exists(feature_names_file):
            try:
                feature_names = json.load(open(feature_names_file, "r"))
                print("Loaded feature names from feature_names.json")
            except Exception as e:
                print("Could not load feature_names.json:", e)
                feature_names = None

        # If not available, infer heuristically (best-effort)
        inferred_feat_names = ["Pclass", "SexIndexed", "Age", "Fare", "EmbarkedIndexed"]
        if feature_names is None:
            feature_names = inferred_feat_names
            print("Inferred feature names (heuristic):", feature_names)
        else:
            print("Feature names (from file):", feature_names)

        # train/test split
        train, test = df.randomSplit([0.8, 0.2], seed=42)

        evaluator = BinaryClassificationEvaluator(
            labelCol="Survived",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        mlflow.set_experiment("Titanic_Classification")

        # ---------------------------
        # Candidate models + params
        # ---------------------------
        model_candidates = {
            "LogisticRegression": {
                "estimator": LogisticRegression(labelCol="Survived", featuresCol="features"),
                "params": {
                    "maxIter": [10, 20],
                    "regParam": [0.0, 0.1]
                }
            },
            "RandomForest": {
                "estimator": RandomForestClassifier(labelCol="Survived", featuresCol="features"),
                "params": {
                    "numTrees": [50, 100],
                    "maxDepth": [5, 10]
                }
            }
        }

        best_auc = -1.0
        best_model = None
        best_run_id = None

        # ---------------------------
        # Grid Search (apply params via setParams)
        # ---------------------------
        for model_name, cfg in model_candidates.items():
            print(f"\n Training {model_name}...")
            base_est = cfg["estimator"]
            param_grid = cfg["params"]
            # Cartesian product of params
            keys, values = zip(*param_grid.items())
            for combo in itertools.product(*values):
                params = dict(zip(keys, combo))

                try:
                    with mlflow.start_run() as run:
                        # fresh estimator instance
                        estimator = base_est.copy()

                        # Apply params using setParams (string-name to value)
                        for k, v in params.items():
                            try:
                                # setParams expects keyword args like maxIter=10
                                estimator = estimator.setParams(**{k: v})
                            except Exception as e:
                                # Some params may have different names; warn and skip
                                print(f" Could not set param {k} for {model_name}: {e}")

                        # Train (data already has 'features' column from preprocess)
                        fitted = estimator.fit(train)

                        # Evaluate
                        preds = fitted.transform(test)
                        # save confusion matrix artifact
                        y_true = [int(r["Survived"]) for r in preds.select("Survived").collect()]
                        y_pred = [int(r["prediction"]) for r in preds.select("prediction").collect()]
                        # AUC
                        auc = evaluator.evaluate(preds)
                        acc = accuracy_score(y_true, y_pred)
                        prec = precision_score(y_true, y_pred, zero_division=0)
                        rec = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)

                        # Log params/metrics
                        mlflow.log_param("model_type", model_name)
                        for k, v in params.items():
                            mlflow.log_param(k, v)
                        mlflow.log_metric("AUC", float(auc))
                        mlflow.log_metric("Accuracy", float(acc))
                        mlflow.log_metric("Precision", float(prec))
                        mlflow.log_metric("Recall", float(rec))
                        mlflow.log_metric("F1", float(f1))


                        with tempfile.TemporaryDirectory() as td:
                            cm_path = os.path.join(td, f"confusion_{model_name}.png")
                            plot_confusion_matrix(y_true, y_pred, cm_path)
                            mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")

                        # Log model (estimator only). To make serving easier, consider logging the full preprocessing+estimator pipeline.
                        # Here we log the fitted estimator (it expects dataframe with 'features' column).
                        mlflow.spark.log_model(fitted, "model")

                        print(f" {model_name} {params} → AUC={auc:.4f}")

                        # Feature importances / coefficients
                        try:
                            feat_vals = None
                            art_prefix = "feature_importance"
                            last_stage = fitted  # fitted estimator
                            # RandomForest -> featureImportances
                            if hasattr(last_stage, "featureImportances"):
                                fi = last_stage.featureImportances.toArray()
                                # align length
                                if len(fi) == len(feature_names):
                                    names = feature_names
                                else:
                                    # fallback: use last N feature names or generic names
                                    if len(feature_names) >= len(fi):
                                        names = feature_names[: len(fi)]
                                    else:
                                        names = [f"f{i}" for i in range(len(fi))]
                                with tempfile.TemporaryDirectory() as td:
                                    csv_path = os.path.join(td, f"{model_name}_feature_importance.csv")
                                    png_path = os.path.join(td, f"{model_name}_feature_importance.png")
                                    save_feature_table(names, fi, csv_path)
                                    plot_feature_importances(names, fi, png_path, title=f"{model_name} FeatureImportances")
                                    mlflow.log_artifact(csv_path, artifact_path=art_prefix)
                                    mlflow.log_artifact(png_path, artifact_path=art_prefix)

                            # LogisticRegression -> coefficients
                            elif hasattr(last_stage, "coefficients"):
                                coeffs = np.array(last_stage.coefficients)
                                if len(coeffs) == len(feature_names):
                                    names = feature_names
                                else:
                                    names = [f"f{i}" for i in range(len(coeffs))]
                                with tempfile.TemporaryDirectory() as td:
                                    csv_path = os.path.join(td, f"{model_name}_coefficients.csv")
                                    png_path = os.path.join(td, f"{model_name}_coefficients.png")
                                    save_feature_table(names, coeffs, csv_path)
                                    plot_feature_importances(names, np.abs(coeffs), png_path, title=f"{model_name} Coefficients(abs)")
                                    mlflow.log_artifact(csv_path, artifact_path="coefficients")
                                    mlflow.log_artifact(png_path, artifact_path="coefficients")
                            else:
                                print(" Estimator has no featureImportances/coefficients attribute")
                        except Exception as ex:
                            print(" Could not log feature importance/coefficients:", ex)

                        # Track best
                        if auc > best_auc:
                            best_auc = auc
                            best_model = fitted
                            best_run_id = run.info.run_id

                except Exception as e:
                    print(f" Error training {model_name} with {params}: {e}")
                    traceback.print_exc()
                    continue

        # ---------------------------
        # Save/register best model
        # ---------------------------
        if best_model and best_run_id:
            try:
                try:
                    result = mlflow.register_model(f"runs:/{best_run_id}/model", "TitanicClassifier")
                    print(f" Best model registered: AUC={best_auc:.4f} (run {best_run_id})")
                    print(f"   Model version: {result.version}")
                except Exception as registry_error:
                    print(f" Model Registry not available: {registry_error}")
                    # fallback: save locally
                    model_path = "/home/karthik/mlops-pipeline-ch24m535/models/best_titanic_model"
                    best_model.write().overwrite().save(model_path)
                    with open(f"{model_path}_info.txt", "w") as f:
                        f.write(f"AUC: {best_auc:.4f}\n")
                        f.write(f"MLflow Run ID: {best_run_id}\n")
                        f.write(f"Model Path: {model_path}\n")
                    print(f" Saved best model locally: {model_path}")
            except Exception as e:
                print(" Error saving/registering best model:", e)
                traceback.print_exc()
        else:
            print(" No successful models to save/register")

    except Exception as e:
        print(" Fatal error:", e)
        traceback.print_exc()
    finally:
        if spark:
            spark.stop()
            print(" Spark session stopped")

if __name__ == "__main__":
    main()
