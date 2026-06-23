/*
Project: Transaction Fraud Detection
File: fraud_monitoring_queries.sql

Purpose:
Example SQL queries for fraud monitoring and risk analysis.

Assumed table:
transactions

Expected columns:
- transaction_id
- customer_id
- transaction_datetime
- amount
- channel
- merchant_category
- device_changed
- new_payee_flag
- country_mismatch
- fraud_flag
*/

-- 1. Overall fraud rate and loss monitoring
SELECT
    COUNT(*) AS total_transactions,
    SUM(fraud_flag) AS fraud_cases,
    1.0 * SUM(fraud_flag) / COUNT(*) AS fraud_rate,
    SUM(CASE WHEN fraud_flag = 1 THEN amount ELSE 0 END) AS fraud_loss
FROM transactions;


-- 2. Fraud rate by channel
SELECT
    channel,
    COUNT(*) AS total_transactions,
    SUM(fraud_flag) AS fraud_cases,
    1.0 * SUM(fraud_flag) / COUNT(*) AS fraud_rate,
    SUM(CASE WHEN fraud_flag = 1 THEN amount ELSE 0 END) AS fraud_loss
FROM transactions
GROUP BY channel
ORDER BY fraud_rate DESC;


-- 3. Fraud rate by merchant category
SELECT
    merchant_category,
    COUNT(*) AS total_transactions,
    SUM(fraud_flag) AS fraud_cases,
    1.0 * SUM(fraud_flag) / COUNT(*) AS fraud_rate,
    SUM(CASE WHEN fraud_flag = 1 THEN amount ELSE 0 END) AS fraud_loss
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_rate DESC;


-- 4. Daily fraud trend
SELECT
    CAST(transaction_datetime AS DATE) AS transaction_date,
    COUNT(*) AS total_transactions,
    SUM(fraud_flag) AS fraud_cases,
    1.0 * SUM(fraud_flag) / COUNT(*) AS fraud_rate,
    SUM(CASE WHEN fraud_flag = 1 THEN amount ELSE 0 END) AS fraud_loss
FROM transactions
GROUP BY CAST(transaction_datetime AS DATE)
ORDER BY transaction_date;


-- 5. High-risk customer summary
SELECT
    customer_id,
    COUNT(*) AS total_transactions,
    SUM(fraud_flag) AS fraud_cases,
    1.0 * SUM(fraud_flag) / COUNT(*) AS customer_fraud_rate,
    SUM(amount) AS total_transaction_amount,
    SUM(CASE WHEN fraud_flag = 1 THEN amount ELSE 0 END) AS fraud_loss,
    SUM(device_changed) AS device_change_count,
    SUM(new_payee_flag) AS new_payee_count,
    SUM(country_mismatch) AS country_mismatch_count
FROM transactions
GROUP BY customer_id
HAVING COUNT(*) >= 3
ORDER BY fraud_loss DESC, customer_fraud_rate DESC;


-- 6. Previous transaction amount using window function
SELECT
    transaction_id,
    customer_id,
    transaction_datetime,
    amount,
    LAG(amount) OVER (
        PARTITION BY customer_id
        ORDER BY transaction_datetime
    ) AS previous_amount,
    amount - LAG(amount) OVER (
        PARTITION BY customer_id
        ORDER BY transaction_datetime
    ) AS amount_change
FROM transactions
ORDER BY customer_id, transaction_datetime;


-- 7. Rule-based high-risk transaction flags
SELECT
    transaction_id,
    customer_id,
    transaction_datetime,
    amount,
    channel,
    merchant_category,
    fraud_flag,
    CASE
        WHEN amount >= 500
          AND device_changed = 1
          AND new_payee_flag = 1
        THEN 1
        ELSE 0
    END AS high_risk_rule_flag
FROM transactions;