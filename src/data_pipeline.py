"""
Titanic Data Preprocessing with Spark
-------------------------------------
This script:
1. Loads Titanic dataset (CSV).
2. Handles missing values.
3. Encodes categorical variables (Sex, Embarked).
4. Assembles features into a single vector.
5. Saves processed dataset in Parquet format.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

def main():
    # Start Spark session
    spark = SparkSession.builder.appName("TitanicPreprocessing").getOrCreate()

    # Load Titanic dataset
    df = spark.read.csv("/home/karthik/mlops-pipeline-ch24m535/data/raw/titanic.csv", header=True, inferSchema=True)

    print("Schema before preprocessing:")
    df.printSchema()

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

    # Save processed dataset in Parquet format
    processed.write.mode("overwrite").parquet("/home/karthik/mlops-pipeline-ch24m535/data/processed/titanic")

    spark.stop()
    print("✅ Preprocessing complete. Processed data saved at data/processed/titanic")

if __name__ == "__main__":
    main()
