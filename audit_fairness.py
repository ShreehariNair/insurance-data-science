import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load trained pipeline and dataset
pipeline = joblib.load("data/best_insurance_model.pkl")
df = pd.read_csv("data/dataset_clean.csv")

feature_order = [
    "income_lpa", "tenure_years", "children", "reimbursement", "smoker",
    "age_group", "bmi_category", "gender", "city", "occupation", 
    "type_policy", "type_product"
]

X = df[feature_order]
y = df["premium"]
sensitive_features = df["gender"]

# 2. Model Predictions
y_pred = pipeline.predict(X)

# 3. Performance Disparity Audit Across Sensitive Groups (Gender)
metrics = {
    'Mean Predicted Premium (₹)': lambda y_true, y_p: np.mean(y_p),
    'MAE (₹)': mean_absolute_error,
    'RMSE (₹)': lambda y_true, y_p: np.sqrt(mean_squared_error(y_true, y_p))
}

metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

print("=== FAIRNESS AUDIT REPORT (REGRESSION METRICS BY GROUP) ===")
print(metric_frame.by_group)

# 4. Binarized Selection Audit for Demographic Parity & Equalized Odds
median_premium = y.median()
y_binary = (y > median_premium).astype(int)
y_pred_binary = (y_pred > median_premium).astype(int)

dp_diff = demographic_parity_difference(y_true=y_binary, y_pred=y_pred_binary, sensitive_features=sensitive_features)
eo_diff = equalized_odds_difference(y_true=y_binary, y_pred=y_pred_binary, sensitive_features=sensitive_features)

print("\n=== BINARIZED SELECTION FAIRNESS METRICS ===")
print(f"Demographic Parity Difference : {dp_diff:.4f}")
print(f"Equalized Odds Difference     : {eo_diff:.4f}")

# 5. Save Disparity Visualization
metric_frame.by_group[['Mean Predicted Premium (₹)', 'MAE (₹)']].plot(
    kind='bar', figsize=(8, 5), title='Prediction & MAE Disparity Across Gender Groups'
)
plt.ylabel('Amount (₹)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("fairness_disparity_plot.png")
print("\nFairness audit plot saved as 'fairness_disparity_plot.png'")