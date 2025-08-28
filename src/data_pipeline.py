"""
Titanic Data Preprocessing with Spark (Generic + Auto High-Cardinality Filter)

Usage:
    python data_pipeline.py --input_path /path/to/titanic.csv --output_dir /path/to/processed/
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
import json
import argparse
import os

def impute_missing(df):
    stats = {}
    for c, dtype in df.dtypes:
        if c == "Survived":
            continue
        if dtype in ("double", "int", "float", "bigint"):
            mean_val = df.select(mean(col(c))).first()[0]
            if mean_val is not None:
                df = df.fillna({c: mean_val})
                stats[c + "_mean"] = float(mean_val)
        else:
            mode_row = df.groupBy(c).count().orderBy("count", ascending=False).first()
            if mode_row and mode_row[0] is not None:
                df = df.fillna({c: mode_row[0]})
                stats[c + "_mode"] = mode_row[0]
            else:
                df = df.fillna({c: "Unknown"})
                stats[c + "_mode"] = "Unknown"
    return df, stats

def main(input_path: str, output_dir: str):
    spark = SparkSession.builder.appName("TitanicPreprocessing").getOrCreate()

    # Load dataset
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    print("Schema before preprocessing:")
    df.printSchema()

    # Impute missing values
    df, stats = impute_missing(df)
    df = df.dropna(subset=["Survived"])
    df = df.fillna({"Age": int(df.agg({"Age": "mean"}).first()[0]), "Embarked": "S"})

    features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
    df = df.select(["Survived"] + features)

    # Encode categorical features
    sex_indexer = StringIndexer(inputCol="Sex", outputCol="SexIndexed")
    embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkedIndexed")
    feature_cols = ["Pclass", "SexIndexed", "Age", "Fare", "EmbarkedIndexed"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    pipeline = Pipeline(stages=[sex_indexer, embarked_indexer, assembler])
    model = pipeline.fit(df)
    #Save the model
    model.write().overwrite().save(os.path.join(output_dir, "preprocess_pipeline"))
    processed = model.transform(df)

    os.makedirs(output_dir, exist_ok=True)
    processed.write.mode("overwrite").parquet(os.path.join(output_dir, "titanic"))

    # Save feature names and imputation stats
    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(output_dir, "impute_stats.json"), "w") as f:
        json.dump(stats, f)

    print(f"\nPreprocessing complete. Data saved to: {output_dir}")
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data")
    args = parser.parse_args()
    main(args.input_path, args.output_dir)
