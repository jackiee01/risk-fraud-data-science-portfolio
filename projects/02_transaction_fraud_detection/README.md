# Transaction Fraud Detection

## Overview

This project demonstrates an end-to-end fraud analytics workflow using synthetic transaction data.

The goal is to identify potentially fraudulent transactions while balancing fraud detection performance with false positive risk.

## Business Problem

Financial institutions need to detect fraudulent transactions quickly, but overly aggressive detection rules can incorrectly flag legitimate customers.

This project simulates a fraud detection workflow where transaction-level data is used to:

- Generate fraud risk indicators
- Build classification models
- Evaluate model performance
- Compare thresholds for fraud capture and false positive trade-offs

## Domain

Fraud / Risk Analytics

## Problem Type

Binary Classification

Target variable:

- `fraud_flag`: 1 if the transaction is fraudulent, 0 otherwise

## Dataset

This project uses synthetic transaction data generated locally.

The synthetic dataset is designed to include common fraud analytics patterns such as:

- Unusual transaction amounts
- Device changes
- New payee activity
- Country mismatch
- High-risk transaction channels
- Customer transaction history indicators

Raw generated data is stored locally and is not uploaded to GitHub.

## Project Structure

| Folder / File | Purpose |
|---|---|
| `src/generate_sample_data.py` | Generates synthetic transaction data |
| `src/make_features.py` | Creates fraud risk features |
| `src/train_model.py` | Trains and evaluates classification models |
| `src/evaluate_thresholds.py` | Analyzes threshold trade-offs |
| `sql/fraud_monitoring_queries.sql` | Example SQL queries for fraud monitoring |
| `outputs/` | Stores model metrics and threshold analysis outputs |
| `data/` | Local generated dataset storage, not uploaded to GitHub |


## Workflow

1. Generate synthetic transaction data
2. Create fraud risk features
3. Train baseline fraud detection models
4. Evaluate model performance
5. Analyze probability thresholds
6. Interpret results from a fraud risk perspective

## Models

- Logistic Regression
- Random Forest

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- False Positive Rate
- Fraud Capture Rate
- Review Rate

## How to Run

Run the following commands from the repository root:

```powershell
py projects\02_transaction_fraud_detection\src\generate_sample_data.py
py projects\02_transaction_fraud_detection\src\make_features.py
py projects\02_transaction_fraud_detection\src\train_model.py
py projects\02_transaction_fraud_detection\src\evaluate_thresholds.py

Business Interpretation

This project focuses on the trade-off between detecting fraud and minimizing friction for legitimate customers.

In fraud detection, recall is important because missed fraud cases may lead to financial losses. However, precision and false positive rate are also important because incorrectly flagged transactions can create customer friction and operational review costs.

Status

In Progress

