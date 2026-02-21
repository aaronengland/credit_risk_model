# Credit Risk Model --- Braviant

## Executive Summary

This project develops and evaluates machine learning models to predict
12-month default risk for unsecured installment loans. The objective is
to build a deployable, compliant, and interpretable underwriting model
using only application-time variables.

The dataset includes **25,308 funded unsecured installment loans**
originated between **January 2022 and December 2024**, with an overall
**12-month default rate of \~18.7%**. The portfolio represents a
near-prime borrower population and exhibits strong data quality and
temporal stability.

------------------------------------------------------------------------

## Dataset Overview

-   **Total loans:** 25,308
-   **Default rate:** \~18.7%
-   **Missingness:** \~0.7%
-   **Average bureau score:** \~630
-   **Median stated income:** \~\$4,965
-   **Average utilization:** \~37%

### Key Risk Indicators

**Positively Associated with Default** - APR
- Credit inquiries
- Delinquencies
- Public records
- Utilization

**Protective Factors** - Bureau score
- Income
- Open trades
- Loan term

Channel and state variables demonstrated limited standalone predictive
value, reinforcing that traditional bureau and affordability metrics are
the primary drivers of risk.

------------------------------------------------------------------------

## Data Splitting Strategy

An out-of-time split was used to reflect real-world deployment
conditions.

-   **Training:** 50% (Default rate: 19.0%)
-   **Validation:** 25% (Default rate: 18.7%)
-   **Test:** 25% (Default rate: 18.1%)

Stable default rates across splits indicate minimal temporal drift and
support reliable model evaluation.

------------------------------------------------------------------------

# Modeling Approach

## 1. XGBoost Model

### Design Principles

-   Used only underwriting-available variables
-   Excluded post-origination fields (e.g., `charged_off_amount`,
    `paid_interest_amount`, `apr`)
-   Excluded state and engineered age for compliance and defendability
-   Preprocessing fit on training data only
-   Conservative imputation
-   Feature engineering: log income, loan-to-income
-   Monotonic constraints aligned with credit intuition
-   Class imbalance handled using `scale_pos_weight ≈ 4.26`

### Performance

  Metric    Validation     Test
  --------- -------------- --------------
  ROC AUC   ≈ 0.80         ≈ 0.80
  PR AUC    ≈ 0.49--0.50   ≈ 0.49--0.50

### Interpretability

SHAP analysis identified the primary drivers of risk:

-   Bureau score
-   Loan-to-income
-   Utilization
-   Delinquencies
-   Income
-   Inquiries

The model was simplified from 15 to 11 features with no material
degradation in PR AUC.

------------------------------------------------------------------------

## 2. Logistic Regression (Baseline)

A logistic regression model was built using the same features and
preprocessing pipeline, with standardized continuous variables.

### Diagnostics

-   VIF range: ≈ 1.0--1.8 (minimal multicollinearity)

### Performance

  Metric    Validation   Test
  --------- ------------ --------
  ROC AUC   ≈ 0.76       ≈ 0.76
  PR AUC    ≈ 0.40       ≈ 0.40

Model coefficients align with established credit risk intuition.
Performance remained stable after reducing to \~11 core features.

------------------------------------------------------------------------

# Model Comparison

  Metric    XGBoost        Logistic Regression   Relative Lift
  --------- -------------- --------------------- ---------------
  ROC AUC   \~0.80         \~0.76                \~5--6%
  PR AUC    \~0.49--0.50   \~0.40                \~24--26%

The improvement is particularly meaningful given the \~19% default rate,
reflecting materially better identification of high-risk borrowers.
Performance gains were consistent between validation and test sets,
indicating strong generalization.

------------------------------------------------------------------------

# Deployment Recommendation

A balanced, risk-tiered approval framework is recommended:

-   **Auto-approve** low-risk borrowers
-   **Risk-based pricing or exposure limits** for mid-risk applicants
-   **Decline** highest-risk segment

This strategy enables either:

-   Lower losses at a fixed approval rate
-   **or**
-   Higher approvals at a fixed risk target

------------------------------------------------------------------------

# Governance & Monitoring

Ongoing governance should include:

-   Performance stability monitoring
-   Data drift detection
-   Score distribution tracking
-   Fair lending analysis

------------------------------------------------------------------------

## Author

**Aaron England, PhD**\
Machine Learning Engineer | Credit Risk Modeling | Production ML
Systems
