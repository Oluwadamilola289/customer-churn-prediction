import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def main():

    print("=" * 60)
    print("TRAINING CUSTOMER CHURN PIPELINE")
    print("=" * 60)

    # Load the CLEAN dataset (not the encoded one)
    df = pd.read_csv(
        "data/processed/clean_telco_customer_churn.csv"
    )

    # Remove ID column
    df = df.drop(columns=["customerID"])

    # Convert target to binary
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Separate categorical and numeric columns
    categorical_features = X.select_dtypes(include=["object"]).columns
    numeric_features = X.select_dtypes(exclude=["object"]).columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "num",
                "passthrough",
                numeric_features
            )
        ]
    )

    # Full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=5000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}\n")

    print(classification_report(
        y_test,
        predictions
    ))

    # Save the entire pipeline
    joblib.dump(
        pipeline,
        "models/churn_prediction_pipeline.pkl"
    )

    print("\nPipeline saved successfully!")


if __name__ == "__main__":
    main()