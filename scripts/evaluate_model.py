import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score


def evaluate(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return accuracy


def main():

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    df = pd.read_csv(
        "data/processed/featured_telco_customer_churn.csv"
    )

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []

    print()

    for name, model in models.items():

        accuracy = evaluate(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print(f"{name:<25} {accuracy:.4f}")

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy
            }
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "reports/model_comparison.csv",
        index=False
    )

    best = results_df.sort_values(
        "Accuracy",
        ascending=False
    ).iloc[0]

    print("\n" + "=" * 60)

    print("BEST MODEL")

    print(best)

    print("=" * 60)


if __name__ == "__main__":
    main()