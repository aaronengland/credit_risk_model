**Credit Risk Model - Braviant**

**[EDA](https://github.com/aaronengland/credit_risk_model_braviant/blob/main/01_eda/notebook.ipynb)**

The dataset includes 25,308 funded
unsecured installment loans originated between January 2022 and December
2024, with a 12-month default rate of \~18.7%. Data quality is strong,
with minimal missingness (\~0.7%) and stable origination volume and
default rates over time. The portfolio resembles a near-prime
population, with an average bureau score of \~630, median stated income
of \~\$4,965, and average utilization near 37%. Univariate analysis
shows APR, inquiries, delinquencies, public records, and utilization are
positively associated with default, while bureau score, income, open
trades, and term are protective. Channel and state variables show
limited standalone predictive value, reinforcing that traditional bureau
and affordability metrics are the primary drivers of risk.

**[Data Split](https://github.com/aaronengland/credit_risk_model_braviant/blob/main/02_split_data/notebook.ipynb)**

An out-of-time split was used to reflect real deployment. Loans were sorted
chronologically, with 50% assigned to training, 25% to validation, and
25% to test. Default rates remain stable across splits---19.0% (train),
18.7% (valid), and 18.1% (test)---indicating minimal time-based drift
and supporting reliable performance evaluation.

**[XGBoost Model](https://github.com/aaronengland/credit_risk_model_braviant/blob/main/03_xgboost/notebook.ipynb)**

An XGBoost model was developed to capture nonlinear
relationships and interactions using only underwriting-available
variables. Post-origination fields (e.g., charged_off_amount,
paid_interest_amount, apr), state, and engineered age were excluded for
compliance and defendability reasons. Preprocessing was fit on training
data only and included conservative imputation and engineered features
such as log income and loan-to-income. Monotone constraints aligned
model behavior with credit intuition, and class imbalance was handled
using scale_pos_weight (\~4.26). The final model achieved ROC AUC ≈ 0.80
and PR AUC ≈ 0.49--0.50 on validation and test. SHAP analysis confirms
bureau score, loan-to-income, utilization, delinquencies, income, and
inquiries as the primary risk drivers. The model was simplified from 15
to 11 features with no material loss in PR AUC.

**[Logistic Regression Model](https://github.com/aaronengland/credit_risk_model_braviant/blob/main/04_logistic_regression/notebook.ipynb)**

A logistic regression baseline was built using
the same features and preprocessing pipeline, with continuous variables
standardized. VIF values (≈1.0--1.8) indicate minimal multicollinearity.
The model achieved ROC AUC ≈ 0.76 and PR AUC ≈ 0.40 on validation and
test. Coefficients align with credit intuition, and performance remained
stable after reducing to \~11 core features.

**[Model Comparison and Conclusion](https://github.com/aaronengland/credit_risk_model_braviant/blob/main/05_comparison/notebook.ipynb)**

XGBoost outperforms logistic regression
across all metrics, delivering a \~5--6% relative lift in ROC AUC and a
\~24--26% lift in PR AUC. The improvement is especially meaningful given
the \~19% default rate, as it reflects materially better identification
of higher-risk borrowers. These gains are consistent from validation to
test, suggesting robust generalization. The recommended approach is a
balanced, risk-tiered approval strategy: auto-approve low-risk
borrowers, apply risk-based pricing or exposure limits to mid-risk
applicants, and decline the highest-risk segment. This supports either
lower losses at a fixed approval rate or higher approvals at the same
risk target. Ongoing governance should include monitoring for
performance stability, data drift, score distribution shifts, and fair
lending considerations.
