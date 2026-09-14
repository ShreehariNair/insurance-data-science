# Responsible AI Evaluation & Governance Report

## 1. Model Overview & Purpose
* **Model Type**: Scikit-Learn Pipeline (`ColumnTransformer` + Regression Estimator)
* **Target Variable**: `premium` (Annual Insurance Premium in ₹)
* **Primary Objective**: Provide transparent, accurate, and fair health insurance premium estimations.
* **Sensitive Attributes Tracked**: `gender` (Male, Female)

---

## 2. Fairness & Performance Disparity Audit

### Audit Methodology
Fairness is evaluated using the `fairlearn` toolkit across sensitive demographic groups (`gender`). Metrics assess both raw regression accuracy disparities and binarized selection rates (categorized by the median premium threshold).

### Evaluated Fairness Metrics
* **Demographic Parity Difference**: Measures the difference in high-cost prediction rates between gender groups. Target threshold: < 0.10.
* **Equalized Odds Difference**: Measures true positive rate and false positive rate disparities for binarized premium assignments. Target threshold: < 0.10.
* **Regression Accuracy Disparity**: Evaluates Mean Absolute Error (MAE in ₹) and Root Mean Squared Error (RMSE in ₹) separately for Male and Female cohorts to ensure equivalent model accuracy across groups.

### Audit Execution
Run the automated fairness audit script post-training:
```bash
python audit_fairness.py

```

---

## 3. Privacy, Security & Data Protection Protocols

* **PII Anonymization**: All Direct Personally Identifiable Information (Names, SSNs, Aadhaar Numbers, Contact Details, Addresses) is completely stripped prior to model ingestion and dataset cleaning.
* **Demographic Feature Masking**: Geographic attributes are generalized into broad categories (`city`: Tier 1, Tier 2, Tier 3) to prevent localized re-identification risks.
* **Artifact Serialization Integrity**: Model binaries (`data/best_insurance_model.pkl`) are serialized using strict environment dependency pinning (`scikit-learn`, `joblib`). Signature validation prevents unauthenticated code execution.

---

## 4. User Consent & Operational Transparency

* **Explicit Data Collection Consent**: Users must explicitly opt-in to risk profiling before submitting personal health and financial attributes (`income_lpa`, `smoker`, `reimbursement`).
* **Explainability Interface**: Every policyholder receives model transparency through:
* **Global SHAP Beeswarm Plots**: Showing overall dataset feature impacts.
* **Local LIME Feature Explanations**: Highlighting specific factors driving their individual premium calculation in ₹.



---

## 5. Algorithmic Recourse & Actionable Guidance

Policyholders assigned premiums above historical averages are provided transparent recourse options based on actionable inputs:

* **Controllable Attributes**:
* **Smoker Status (`smoker`)**: Highlights premium reductions associated with smoking cessation programs.
* **BMI Category (`bmi_category`)**: Demonstrates prospective savings tied to weight management programs.


* **Immutable / Structural Attributes**: Attributes such as `gender` or `age_group` are strictly monitored to ensure they do not act as punitive proxies in premium determination.

---

## 6. Model Monitoring & Drift Mitigation

* **Runtime Drift Detection**: Integrated Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI) calculations flag statistical distribution shifts in continuous features (e.g., `income_lpa`).
* **Drift Thresholds**:
* **PSI < 0.10**: Stable (No action required).
* **0.10 <= PSI <= 0.20**: Moderate shift (Warning issued).
* **PSI > 0.20**: High drift (Triggers automated pipeline retraining).
