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

