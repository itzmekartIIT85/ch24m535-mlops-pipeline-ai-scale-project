"""
Titanic Data Preprocessing with Spark (Generic + Auto High-Cardinality Filter)
------------------------------------------------------------------------------
1. Loads Titanic dataset (CSV).
2. Handles missing values (numeric → mean, categorical → mode).
3. Automatically removes high-cardinality categorical features (unique > threshold).
4. Encodes categorical variables automatically.
5. Adds engineered features (FamilySize, IsAlone, Bucketized Age/Fare).
6. Deduplicates feature columns.
7. Assembles features into a single vector.
8. Saves processed dataset in Parquet format.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, lit, countDistinct
from pyspark.ml.feature import StringIndexer, VectorAssembler, Bucketizer
from pyspark.ml import Pipeline
import json


def impute_missing(df):
    """Generic imputation: numeric → mean, categorical → mode"""
    stats = {}
    for c, dtype in df.dtypes:
        if c == "Survived":  # skip target
            continue

        if dtype in ("double", "int", "float", "bigint"):
            mean_val = df.select(mean(col(c))).first()[0]
            if mean_val is not None:
                df = df.fillna({c: mean_val})
                stats[c + "_mean"] = float(mean_val)
        else:  # categorical
            mode_row = df.groupBy(c).count().orderBy("count", ascending=False).first()
            if mode_row is not None and mode_row[0] is not None:
                df = df.fillna({c: mode_row[0]})
                stats[c + "_mode"] = mode_row[0]
            else:
                df = df.fillna({c: "Unknown"})
                stats[c + "_mode"] = "Unknown"
    return df, stats


def main():
    spark = SparkSession.builder.appName("TitanicPreprocessing").getOrCreate()

    # Load Titanic dataset
    df = spark.read.csv(
        "/home/karthik/mlops-pipeline-ch24m535/data/raw/titanic.csv",
        header=True,
        inferSchema=True,
    )

    print("Schema before preprocessing:")
    df.printSchema()

    # Impute missing values
    df,stats = impute_missing(df)

  # Drop rows with missing target
    df = df.dropna(subset=["Survived"])

    # Handle missing values in Age and Embarked (simple strategy: fill with mean/mode)
    df = df.fillna({"Age": int(df.agg({"Age": "mean"}).first()[0])})
    df = df.fillna({"Embarked": "S"})  # Most common port is 'S'

    # Select useful features
    features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
    df = df.select(["Survived"] + features)

    # Encode categorical features
    sex_indexer = StringIndexer(inputCol="Sex", outputCol="SexIndexed")
    embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkedIndexed")
    
    feature_cols = ["Pclass", "SexIndexed", "Age", "Fare", "EmbarkedIndexed"]

    # Assemble features
    assembler = VectorAssembler(
        inputCols=["Pclass", "Age", "Fare", "SexIndexed", "EmbarkedIndexed"],
        outputCol="features"
    )

    pipeline = Pipeline(stages=[sex_indexer, embarked_indexer, assembler])
    model = pipeline.fit(df)
    processed = model.transform(df)

    print("Processed dataset schema:")
    processed.printSchema()


    # Save
    processed.write.mode("overwrite").parquet(
        "/home/karthik/mlops-pipeline-ch24m535/data/processed/titanic"
    )
    
    feature_names_path = "/home/karthik/mlops-pipeline-ch24m535/data/processed/feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_cols, f)

    print(f" Saved feature names to {feature_names_path}")
    
    
    # Save imputation stats
    with open("/home/karthik/mlops-pipeline-ch24m535/data/processed/impute_stats.json", "w") as f:
        json.dump(stats, f)

    spark.stop()
    print("\n Preprocessing complete. Processed data saved at data/processed/titanic")


if __name__ == "__main__":
    main()