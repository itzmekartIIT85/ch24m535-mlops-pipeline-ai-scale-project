import pandas as pd
import requests
import math
import argparse

# -------------------
# Config
# -------------------
url = "http://127.0.0.1:8000/predict"
test_csv_path = "data/raw/test.csv"  # path to your test dataset

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="CSV file to test the API")
args = parser.parse_args()
# -------------------
# Load test data
# -------------------
df_test = pd.read_csv(args.input_file)

# Ensure columns match API input
required_cols = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
missing_cols = [c for c in required_cols if c not in df_test.columns]
if missing_cols:
    raise ValueError(f"Missing columns in test CSV: {missing_cols}")
# use only random 50 rows for quick testing
df_test = df_test[required_cols].sample(10, random_state=42).reset_index(drop=True)

# -------------------
# Replace missing/NaN or infinite values
# -------------------
for col in ["Age", "Fare"]:
    df_test[col] = pd.to_numeric(df_test[col], errors='coerce')  # convert invalids to NaN
    df_test[col] = df_test[col].fillna(df_test[col].mean())      # fill NaN with column mean
    df_test[col] = df_test[col].apply(lambda x: float(x) if math.isfinite(x) else df_test[col].mean())

for col in ["Pclass"]:
    df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(3).astype(int)

for col in ["Sex", "Embarked"]:
    df_test[col] = df_test[col].fillna("S")

# -------------------
# Send requests and collect predictions
# -------------------
predictions = []

for idx, row in df_test.iterrows():
    payload = row[required_cols].to_dict()
    response = requests.post(url, json=payload)
    #print(f"Request payload: {payload} => Response: {response.status_code}, {response.text}")
    if response.status_code == 200:
        res_json = response.json()
        #print(f"Prediction: {res_json}")
        predictions.append({
            "Pclass": row["Pclass"],
            "Sex": row["Sex"],
            "Age": row["Age"],
            "Fare": row["Fare"],
            "Embarked": row["Embarked"],
            "Predicted_Survived": res_json["survived"],
            "Prob_Survive": res_json["probability_survive"],
            "Drift_Alerts": res_json["drift_alerts"]
        })
    else:
        print(f"Error for row {idx}: {response.text}")

# -------------------
# Convert to DataFrame and print
# -------------------
pred_df = pd.DataFrame(predictions)

print("✅ Predictions for test dataset:")
print(pred_df)

# Optional: Save to CSV as well
pred_df.to_csv("data/processed/titanic/test_predictions.csv", index=False)
print("✅ Predictions also saved to data/processed/titanic/test_predictions.csv")
