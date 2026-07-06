import pandas as pd


def main():

    print("=" * 60)
    print("TELCO CUSTOMER CHURN PREPROCESSING")
    print("=" * 60)

    df = pd.read_csv("data/raw/telco_customer_churn.csv")

    print(f"\nRows Loaded: {len(df)}")

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    print("\nMissing Values")
    print(df.isnull().sum())

    # Fill missing TotalCharges with MonthlyCharges
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    print("\nMissing Values After Cleaning")
    print(df.isnull().sum())

    # Save cleaned dataset
    df.to_csv(
        "data/processed/clean_telco_customer_churn.csv",
        index=False
    )

    print("\nClean dataset saved successfully!")
    print(f"Final Shape: {df.shape}")


if __name__ == "__main__":
    main()