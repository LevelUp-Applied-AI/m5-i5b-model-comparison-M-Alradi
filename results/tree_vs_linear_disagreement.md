# Tree vs. Linear Disagreement Analysis

## Sample Details

- **Test-set index:** 4060
- **True label:** 0
- **RF predicted P(churn=1):** 0.5998
- **LR predicted P(churn=1):** 0.1700
- **Probability difference:** 0.4299

## Feature Values

- **tenure:** 36.0
- **monthly_charges:** 20.0
- **total_charges:** 1077.33
- **num_support_calls:** 2.0
- **senior_citizen:** 0.0
- **has_partner:** 0.0
- **has_dependents:** 0.0
- **contract_months:** 1.0

## Structural Explanation

The Random Forest likely picked up a non-linear interaction: even though tenure is moderate (36 months) and charges are low, the month-to-month contract (contract_months = 1) combined with some support activity (2 calls) signals higher churn risk in certain branches.

The Logistic Regression, being linear, averages these effects and is “pulled down” by the relatively stable signals (longer tenure and low monthly charges), so it underestimates churn for this specific pattern.
