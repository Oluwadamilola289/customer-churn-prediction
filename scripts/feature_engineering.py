import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df = pd.read_csv(
        "data/processed/clean_telco_customer_churn.csv"
    )

    # Remove CustomerID
    df.drop(columns=["customerID"], inplace=True)

    # Encode target
    df["Churn"] = LabelEncoder().fit_transform(df["Churn"])

    # Encode categorical variables
    categorical = df.select_dtypes(include="object").columns

    encoder = LabelEncoder()

    for col in categorical:
        df[col] = encoder.fit_transform(df[col])

    print("\nDataset Ready For Training")

    print(df.head())

    print("\nShape")

    print(df.shape)

    df.to_csv(
        "data/processed/featured_telco_customer_churn.csv",
        index=False
    )

    print("\nFeature engineered dataset saved!")


if __name__ == "__main__":
    main()