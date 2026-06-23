"""
Project: Transaction Fraud Detection
File: train_model.py

Purpose:
Train baseline fraud detection models and compare their performance.

Business context:
Fraud detection is usually an imbalanced classification problem.
Accuracy alone is not enough, so this script evaluates precision, recall, F1-score,
ROC-AUC, and PR-AUC.
"""

from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

INPUT_FILE = DATA_DIR / "fraud_features.csv"
METRICS_FILE = OUTPUT_DIR / "metrics_summary.csv"


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def evaluate_model(model_name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }

    print(f"\n===== {model_name} =====")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. "
            "Run make_features.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    target = "fraud_flag"

    numeric_features = [
        "amount",
        "customer_tenure_days",
        "prior_30d_txn_count",
        "prior_30d_avg_amount",
        "transaction_hour",
        "transaction_dayofweek",
        "transaction_month",
        "amount_to_prior_avg",
        "combined_risk_signal_count",
    ]

    categorical_features = [
        "channel",
        "merchant_category",
    ]

    binary_features = [
        "device_changed",
        "new_payee_flag",
        "country_mismatch",
        "is_night_transaction",
        "high_amount_vs_history",
        "high_velocity_customer",
        "new_customer_flag",
        "high_risk_channel_flag",
        "high_risk_merchant_flag",
    ]

    feature_columns = numeric_features + categorical_features + binary_features

    X = df[feature_columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(
        numeric_features=numeric_features + binary_features,
        categorical_features=categorical_features,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(model_name, pipeline, X_test, y_test)
        results.append(metrics)

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(METRICS_FILE, index=False)

    print("\nMetrics summary saved successfully.")
    print(f"Output file: {METRICS_FILE}")
    print(metrics_df)


if __name__ == "__main__":
    main()