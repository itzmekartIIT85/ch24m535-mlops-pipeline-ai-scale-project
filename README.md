“End Term Project: Robust MLOps Pipeline (ID5003W, IITM)”
#  Titanic Survival Prediction (MLOps Project)
This project demonstrates an end-to-end ML pipeline for Titanic survival prediction using Apache Spark, MLflow, DVC, FastAPI, and Docker. The pipeline is automated, reproducible, and production-ready, including drift detection and model versioning.

Directory Structure
mlops-pipeline-ch24m535/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── app.py
│   ├── utils.py
│   └── config.py
├── tests/
│   ├── test_script.py
│   └── test.csv
├── data/
│   ├── raw/           # Large datasets (DVC tracked)
│   └── processed/     # Processed datasets (DVC tracked)
├── Dockerfile
├── Makefile
├── requirements.txt
├── README.md
├── .gitignore
└── report/
    └── Titanic_ML_Pipeline.pdf

Setup

Clone the repository:

git clone <your_repo_url>
cd mlops-pipeline-ch24m535


Create and activate Python environment:

conda create -n bigdl_class python=3.7 -y
conda activate bigdl_class
pip install -r requirements.txt


Initialize DVC (if not already):

dvc pull


This will download the raw and processed data tracked by DVC.

Makefile Commands
Command	Description
make preprocess	Run data preprocessing (src/preprocess.py)
make train	Train models and log to MLflow (src/train.py)
make serve	Launch FastAPI app (src/app.py) on port 8000
make test	Run tests/test_script.py on sample test.csv

Example:

make preprocess
make train
make serve
make test

FastAPI App Usage

Launch API:

uvicorn src.app:app --reload --port 8000


Test API using test_script.py:

python tests/test_script.py --input tests/test.csv


Response format:

{
  "survived": 1,
  "probability_survive": 0.834,
  "drift_alerts": "No drift detected"
}


Note: test_script.py can take any CSV of passenger data as input.

MLflow

Launch MLflow UI:

mlflow ui


Track experiment metrics, registered model versions, and artifacts like ROC curves and confusion matrices.

Docker Usage

Build Docker image:

docker build -t titanic-ml-pipeline .


Run container:

docker run -p 8000:8000 titanic-ml-pipeline


Access API at:

http://127.0.0.1:8000/predict

Testing Procedure

Use tests/test.csv to validate your pipeline after deployment.

tests/test_script.py selects random rows and sends POST requests to /predict.

Can be extended for automated regression testing or CI/CD pipelines.

Future Work

Add CI/CD for retraining and deployment.

Advanced drift detection (KL divergence, adaptive thresholds).

Multi-model ensemble learning.

References

MLflow

DVC

FastAPI

PySpark
