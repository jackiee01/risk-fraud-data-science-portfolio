"""
Project: Transaction Fraud Detection
File: make_features.py

Purpose:
Create fraud risk features from transaction-level data.

Business context:
Fraud models often rely on behavioural and transactional indicators such as:
- transaction amount relative to past behaviour
- night-time transactions
- device change
- new payee activity
- risky transaction channels
"""

from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "synthetic_transactions.csv"
OUTPUT_FILE = DATA_DIR / "fraud_features.csv"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"])

    df["transaction_hour"] = df["transaction_datetime"].dt.hour
    df["transaction_dayofweek"] = df["transaction_datetime"].dt.dayofweek
    df["transaction_month"] = df["transaction_datetime"].dt.month

    df["is_night_transaction"] = (
        (df["transaction_hour"] < 6) | (df["transaction_hour"] >= 23)
    ).astype(int)

    df["amount_to_prior_avg"] = df["amount"] / df["prior_30d_avg_amount"].clip(lower=1)

    df["high_amount_vs_history"] = (df["amount_to_prior_avg"] >= 3).astype(int)
    df["high_velocity_customer"] = (df["prior_30d_txn_count"] >= 25).astype(int)
    df["new_customer_flag"] = (df["customer_tenure_days"] < 90).astype(int)

    df["high_risk_channel_flag"] = df["channel"].isin(
        ["web", "card_not_present"]
    ).astype(int)

    df["high_risk_merchant_flag"] = df["merchant_category"].isin(
        ["electronics", "gaming", "cash_transfer", "luxury"]
    ).astype(int)

    df["combined_risk_signal_count"] = (
        df[
            [
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
        ].sum(axis=1)
    )

    return df


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. "
            "Run generate_sample_data.py first."
        )

    df = pd.read_csv(INPUT_FILE)
    feature_df = create_features(df)

    feature_df.to_csv(OUTPUT_FILE, index=False)

    print("Feature dataset created successfully.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows: {len(feature_df):,}")
    print(feature_df.head())


if __name__ == "__main__":
    main()