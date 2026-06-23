"""
Project: Transaction Fraud Detection
File: evaluate_thresholds.py

Purpose:
Evaluate how different probability thresholds affect fraud detection performance.

Business context:
In fraud detection, the default 0.50 threshold is often not ideal.
Lower thresholds can capture more fraud but may increase false positives and manual review volume.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

INPUT_FILE = DATA_DIR / "fraud_features.csv"
THRESHOLD_OUTPUT_FILE = OUTPUT_DIR / "threshold_analysis.csv"


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_model_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
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

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def calculate_threshold_metrics(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_transactions = len(y_true)
    actual_fraud = tp + fn
    flagged_transactions = tp + fp

    precision = tp / flagged_transactions if flagged_transactions > 0 else 0
    recall = tp / actual_fraud if actual_fraud > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    review_rate = flagged_transactions / total_transactions

    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "fraud_capture_rate": recall,
        "false_positive_rate": false_positive_rate,
        "review_rate": review_rate,
    }


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

    categorical_features = [
        "channel",
        "merchant_category",
    ]

    feature_columns = numeric_features + categorical_features

    X = df[feature_columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = build_model_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]

    results = [
        calculate_threshold_metrics(y_test, y_proba, threshold)
        for threshold in thresholds
    ]

    threshold_df = pd.DataFrame(results)
    threshold_df.to_csv(THRESHOLD_OUTPUT_FILE, index=False)

    print("Threshold analysis saved successfully.")
    print(f"Output file: {THRESHOLD_OUTPUT_FILE}")
    print(threshold_df)

    feasible = threshold_df[threshold_df["recall"] >= 0.75]

    if not feasible.empty:
        recommended = feasible.sort_values(
            ["precision", "false_positive_rate"],
            ascending=[False, True],
        ).iloc[0]

        print("\nRecommended threshold candidate:")
        print(recommended)
    else:
        print("\nNo threshold reached the target recall of 0.75.")


if __name__ == "__main__":
    main()