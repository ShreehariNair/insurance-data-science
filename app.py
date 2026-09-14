import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Insurance XAI Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("data/best_insurance_model.pkl")
    data = pd.read_csv("data/dataset_clean.csv")
    return model, data

try:
    model, df_clean = load_assets()
except Exception as e:
    st.error(f"Error loading assets from data/ folder: {e}")
    st.stop()

def calculate_psi(baseline, target, buckets=10):
    min_val = min(baseline.min(), target.min())
    max_val = max(baseline.max(), target.max())
    b_scaled = (baseline - min_val) / (max_val - min_val + 1e-6)
    t_scaled = (target - min_val) / (max_val - min_val + 1e-6)
    
    counts_b, bin_edges = np.histogram(b_scaled, bins=buckets, range=(0, 1))
    counts_t, _ = np.histogram(t_scaled, bins=bin_edges)
    
    perc_b = np.where(counts_b == 0, 0.0001, counts_b) / len(baseline)
    perc_t = np.where(counts_t == 0, 0.0001, counts_t) / len(target)
    return np.sum((perc_t - perc_b) * np.log(perc_t / perc_b))

st.title("Insurance Premium Prediction & XAI System")

# Sidebar - User Inputs
st.sidebar.header("Customer Profile Input")

income_lpa = st.sidebar.number_input("Income (LPA)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
tenure_years = st.sidebar.slider("Tenure (Years)", 0, 40, 5)
children = st.sidebar.slider("Children", 0, 5, 0)
reimbursement = st.sidebar.number_input("Prior Reimbursement Amount (₹)", min_value=0.0, value=0.0, step=1000.0)
smoker_input = st.sidebar.selectbox("Smoker", ["No", "Yes"])
smoker = 1 if smoker_input == "Yes" else 0

age_group = st.sidebar.selectbox("Age Group", ["Young Adult", "Adult", "Middle-Aged"])
bmi_category = st.sidebar.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
city = st.sidebar.selectbox("City Category", ["Tier 1", "Tier 2", "Tier 3"])
occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Self-Employed", "Student", "Retired"])
type_policy = st.sidebar.selectbox("Policy Type", ["Individual", "Family Floating", "Comprehensive"])
type_product = st.sidebar.selectbox("Product Type", ["Standard", "Premium", "Basic"])

feature_order = [
    "income_lpa", "tenure_years", "children", "reimbursement", "smoker",
    "age_group", "bmi_category", "gender", "city", "occupation", 
    "type_policy", "type_product"
]

input_dict = {
    "income_lpa": income_lpa,
    "tenure_years": tenure_years,
    "children": children,
    "reimbursement": reimbursement,
    "smoker": smoker,
    "age_group": age_group,
    "bmi_category": bmi_category,
    "gender": gender,
    "city": city,
    "occupation": occupation,
    "type_policy": type_policy,
    "type_product": type_product
}

input_df = pd.DataFrame([input_dict])[feature_order]

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Global & Local XAI", "Model Diagnostics", "Drift Detection"])

with tab1:
    st.subheader("Real-Time Premium Estimation")
    prediction = model.predict(input_df)[0]
    st.metric("Predicted Annual Premium", f"₹{prediction:,.2f}")

with tab2:
    st.subheader("Model Explainability (SHAP & LIME)")
    
    try:
        preprocessor = model.named_steps['preprocessor']
        regressor = model.named_steps['regressor']
        
        X_baseline = df_clean[feature_order]
        
        # Transform baseline and single input
        X_trans_raw = preprocessor.transform(X_baseline)
        input_trans_raw = preprocessor.transform(input_df)
        
        # Convert sparse matrices to dense numpy arrays if required
        if hasattr(X_trans_raw, "toarray"):
            X_trans_raw = X_trans_raw.toarray()
        if hasattr(input_trans_raw, "toarray"):
            input_trans_raw = input_trans_raw.toarray()
            
        feature_names = preprocessor.get_feature_names_out()
        
        # Wrap into DataFrames to provide explicit 2D shape metadata to SHAP maskers
        X_transformed = pd.DataFrame(X_trans_raw, columns=feature_names)
        input_transformed = pd.DataFrame(input_trans_raw, columns=feature_names)
        
        # --- 1. SHAP Explainer Initialization ---
        try:
            explainer = shap.TreeExplainer(regressor)
            shap_values = explainer(X_transformed)
        except Exception:
            # Subsample background data for speed & stability
            background_data = shap.sample(X_transformed, 50, random_state=42)
            explainer = shap.Explainer(regressor.predict, background_data)
            shap_values = explainer(X_transformed)
            
        shap_matrix = shap_values.values if hasattr(shap_values, "values") else shap_values

        col1, col2 = st.columns(2)
        
        # --- 2. SHAP Beeswarm Summary Plot ---
        with col1:
            st.write("**SHAP Global Feature Importance (Beeswarm)**")
            fig_shap, ax_shap = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_matrix, X_transformed, show=False)
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.clf()

        # --- 3. SHAP Dependence Plot for Top Feature ---
        with col2:
            mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
            top_feature_idx = np.argmax(mean_abs_shap)
            top_feature_name = feature_names[top_feature_idx]
            
            st.write(f"**SHAP Dependence Plot (Top Feature: '{top_feature_name}')**")
            fig_dep, ax_dep = plt.subplots(figsize=(8, 6))
            shap.dependence_plot(top_feature_idx, shap_matrix, X_transformed, ax=ax_dep, show=False)
            plt.tight_layout()
            st.pyplot(fig_dep)
            plt.clf()

        st.divider()

        # --- 4. LIME Local Explanation for Selected Customer Profile ---
        st.write("**LIME Local Feature Contribution for Current Input Profile**")
        
        lime_explainer = LimeTabularExplainer(
            training_data=X_transformed.values,
            feature_names=list(feature_names),
            mode='regression',
            random_state=42
        )
        
        # Pass 1D dense vector for the single input row
        exp = lime_explainer.explain_instance(
            data_row=input_transformed.values[0],
            predict_fn=regressor.predict,
            num_features=8
        )
        
        fig_lime = exp.as_pyplot_figure()
        plt.title("LIME Local Explanation (Selected Profile)", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig_lime)
        plt.clf()

    except Exception as e:
        st.error(f"Error computing explainability modules: {e}")
with tab3:
    st.subheader("Performance Metrics & Residuals")
    if 'premium' in df_clean.columns:
        y_true = df_clean['premium']
        X_data = df_clean[feature_order]
        y_pred = model.predict(X_data)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{r2_score(y_true, y_pred):.4f}")
        c2.metric("MAE", f"₹{mean_absolute_error(y_true, y_pred):,.2f}")
        c3.metric("RMSE", f"₹{np.sqrt(mean_squared_error(y_true, y_pred)):,.2f}")
        
        fig_res, ax_res = plt.subplots(figsize=(8, 3))
        residuals = y_true - y_pred
        ax_res.scatter(y_pred, residuals, alpha=0.5)
        ax_res.axhline(0, color='red', linestyle='--')
        ax_res.set_xlabel("Predicted Premium (₹)")
        ax_res.set_ylabel("Residuals (₹)")
        st.pyplot(fig_res)
        plt.clf()

with tab4:
    st.subheader("Data Drift Check")
    if 'income_lpa' in df_clean.columns:
        simulated_runtime_data = df_clean['income_lpa'] + np.random.normal(0, 0.5, len(df_clean))
        ks_stat, p_val = ks_2samp(df_clean['income_lpa'], simulated_runtime_data)
        psi_val = calculate_psi(df_clean['income_lpa'], simulated_runtime_data)
        
        d1, d2 = st.columns(2)
        d1.metric("KS Test P-Value (Income)", f"{p_val:.4f}", delta="Drift" if p_val < 0.05 else "Stable")
        d2.metric("PSI (Income)", f"{psi_val:.4f}", delta="High Drift" if psi_val > 0.2 else "Stable")