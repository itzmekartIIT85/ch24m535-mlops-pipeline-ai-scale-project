# ===========================
# MLOps Pipeline Makefile
# ===========================

# Default paths
RAW_DATA=data/raw/titanic.csv
PROCESSED_DATA=data/processed/
PORT=8000


#=============================
# Environment Variable
# ============================
export PROCESSED_DATA

.PHONY: all preprocess train serve pipeline clean

# ---------------------------
# End-to-end pipeline
# ---------------------------
all: pipeline

pipeline: preprocess train serve

# ---------------------------
# Step 1: Preprocessing
# ---------------------------
preprocess:
	@echo "Running preprocessing..."
	python src/data_pipeline.py --input_path $(RAW_DATA) --output_dir $(PROCESSED_DATA)
	@echo "Preprocessing complete. Processed data saved in $(PROCESSED_DATA)"

# ---------------------------
# Step 2: Training
# ---------------------------
train:
	@echo "Starting model training..."
	python src/train.py --processed_dir $(PROCESSED_DATA)
	@echo "Training complete. Check MLflow UI for metrics and registered models"

# ---------------------------
# Step 3: Serve API
# ---------------------------
serve:
	@echo "Starting FastAPI server..."
	uvicorn src.app:app --reload --port $(PORT)

# ---------------------------
# Utility: Test predictions
# ---------------------------
test:
	@echo "Running test script..."
	python src/test_script.py

# ---------------------------
# Clean artifacts
# ---------------------------
clean:
	@echo "Cleaning processed data, models, and temporary files..."
	rm -rf $(PROCESSED_DATA)
	rm -rf data/processed/*.json
	@echo "Clean complete."

