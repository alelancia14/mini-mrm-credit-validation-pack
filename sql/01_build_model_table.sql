WITH source_data AS (
    SELECT *
    FROM read_csv_auto('data/raw/credit_g.csv', header = TRUE)
),
typed_base AS (
    SELECT
        CAST(checking_status AS VARCHAR) AS checking_status,
        CAST(duration AS INTEGER) AS duration,
        CAST(credit_history AS VARCHAR) AS credit_history,
        CAST(purpose AS VARCHAR) AS purpose,
        CAST(credit_amount AS INTEGER) AS credit_amount,
        CAST(savings_status AS VARCHAR) AS savings_status,
        CAST(employment AS VARCHAR) AS employment,
        CAST(installment_commitment AS VARCHAR) AS installment_commitment,
        CAST(personal_status AS VARCHAR) AS personal_status,
        CAST(other_parties AS VARCHAR) AS other_parties,
        CAST(residence_since AS VARCHAR) AS residence_since,
        CAST(property_magnitude AS VARCHAR) AS property_magnitude,
        CAST(age AS INTEGER) AS age,
        CAST(other_payment_plans AS VARCHAR) AS other_payment_plans,
        CAST(housing AS VARCHAR) AS housing,
        CAST(existing_credits AS VARCHAR) AS existing_credits,
        CAST(job AS VARCHAR) AS job,
        CAST(num_dependents AS VARCHAR) AS num_dependents,
        CAST(own_telephone AS VARCHAR) AS own_telephone,
        CAST(foreign_worker AS VARCHAR) AS foreign_worker,
        CAST("default" AS INTEGER) AS "default"
    FROM source_data
),
model_table AS (
    SELECT
        *,
        ln(credit_amount + 1) AS credit_amount_log,
        credit_amount::DOUBLE / NULLIF(duration, 0) AS credit_per_month
    FROM typed_base
)
SELECT *
FROM model_table;
