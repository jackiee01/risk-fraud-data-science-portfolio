

---



# 4. `src/generate_sample_data.py`



```python

"""

Project: Transaction Fraud Detection

File: generate_sample_data.py



Purpose:

Generate synthetic transaction data for a fraud detection workflow.



Business context:

Real banking transaction data is sensitive and cannot be used in a public portfolio.

This script creates realistic synthetic transaction data with common fraud risk patterns.

"""



from pathlib import Path

import numpy as np

import pandas as pd





PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"

OUTPUT_FILE = DATA_DIR / "synthetic_transactions.csv"





def generate_transactions(n_transactions: int = 25000, random_state: int = 42) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)



    DATA_DIR.mkdir(parents=True, exist_ok=True)



    customer_ids = rng.integers(10000, 13000, size=n_transactions)

    transaction_ids = np.arange(1, n_transactions + 1)



    start_date = pd.Timestamp("2024-01-01")

    random_minutes = rng.integers(0, 90 * 24 * 60, size=n_transactions)

    transaction_datetimes = start_date + pd.to_timedelta(random_minutes, unit="m")



    channels = rng.choice(

        ["mobile_app", "web", "atm", "branch", "card_present", "card_not_present"],

        size=n_transactions,

        p=[0.30, 0.22, 0.12, 0.08, 0.16, 0.12],

    )



    merchant_categories = rng.choice(

        ["grocery", "electronics", "travel", "gaming", "cash_transfer", "luxury", "utilities"],

        size=n_transactions,

        p=[0.25, 0.14, 0.10, 0.12, 0.13, 0.08, 0.18],

    )



    customer_tenure_days = rng.integers(1, 2500, size=n_transactions)

    prior_30d_txn_count = rng.poisson(lam=12, size=n_transactions)

    prior_30d_avg_amount = rng.lognormal(mean=3.6, sigma=0.7, size=n_transactions).round(2)



    base_amount = rng.lognormal(mean=3.8, sigma=0.9, size=n_transactions)

    high_risk_amount_boost = rng.choice([1, 2.5, 5], size=n_transactions, p=[0.90, 0.08, 0.02])

    amount = (base_amount * high_risk_amount_boost).round(2)



    device_changed = rng.binomial(1, 0.12, size=n_transactions)

    new_payee_flag = rng.binomial(1, 0.10, size=n_transactions)

    country_mismatch = rng.binomial(1, 0.06, size=n_transactions)



    hour = pd.Series(transaction_datetimes).dt.hour.to_numpy()

    is_night = ((hour < 6) | (hour >= 23)).astype(int)



    amount_to_prior_avg = amount / np.maximum(prior_30d_avg_amount, 1)



    risky_channel = np.isin(channels, ["web", "card_not_present"]).astype(int)

    risky_merchant = np.isin(merchant_categories, ["electronics", "gaming", "cash_transfer", "luxury"]).astype(int)



    fraud_score = (

        -4.8

        + 0.85 * device_changed

        + 0.95 * new_payee_flag

        + 1.15 * country_mismatch

        + 0.55 * is_night

        + 0.45 * risky_channel

        + 0.50 * risky_merchant

        + 0.35 * (amount_to_prior_avg > 3).astype(int)

        + 0.40 * (customer_tenure_days < 90).astype(int)

        + 0.25 * (prior_30d_txn_count > 25).astype(int)

    )



    fraud_probability = 1 / (1 + np.exp(-fraud_score))

    fraud_flag = rng.binomial(1, fraud_probability)



    df = pd.DataFrame(

        {

            "transaction_id": transaction_ids,

            "customer_id": customer_ids,

            "transaction_datetime": transaction_datetimes,

            "amount": amount,

            "channel": channels,

            "merchant_category": merchant_categories,

            "customer_tenure_days": customer_tenure_days,

            "prior_30d_txn_count": prior_30d_txn_count,

            "prior_30d_avg_amount": prior_30d_avg_amount,

            "device_changed": device_changed,

            "new_payee_flag": new_payee_flag,

            "country_mismatch": country_mismatch,

            "fraud_flag": fraud_flag,

        }

    )



    df = df.sort_values(["customer_id", "transaction_datetime"]).reset_index(drop=True)

    return df





def main() -> None:

    df = generate_transactions()

    df.to_csv(OUTPUT_FILE, index=False)



    fraud_rate = df["fraud_flag"].mean()



    print("Synthetic transaction data generated successfully.")

    print(f"Output file: {OUTPUT_FILE}")

    print(f"Rows: {len(df):,}")

    print(f"Fraud rate: {fraud_rate:.2%}")

    print(df.head())





if __name__ == "__main__":

    main()