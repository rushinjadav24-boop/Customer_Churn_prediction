import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
)

# ── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

clf = load_model()

# ── Encoding Maps (match training) ──────────────────────────
FREQUENT_FLYER_MAP  = {"No": 0, "No Record": 1, "Yes": 2}
INCOME_MAP          = {"High Income": 0, "Low Income": 1, "Middle Income": 2}
SOCIAL_MAP          = {"No": 0, "Yes": 1}
HOTEL_MAP           = {"No": 0, "Yes": 1}

# ── Header ───────────────────────────────────────────────────
st.title("✈️ Customer Churn Prediction")
st.markdown(
    "**Predict whether a travel customer is likely to churn** using a "
    "Random Forest model trained on demographic and service-usage data."
)
st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        "This app predicts customer churn for a travel company.\n\n"
        "**Model:** Random Forest Classifier  \n"
        "**Accuracy:** 87.43%  \n"
        "**AUC-ROC:** 0.9473"
    )
    st.divider()
    st.caption("B.Tech – Gen AI | 2nd Semester | Final Project")

# ── Input Form ───────────────────────────────────────────────
st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", min_value=18, max_value=80, value=35, step=1)
    frequent_flyer = st.selectbox("Frequent Flyer", ["No", "Yes", "No Record"])

with col2:
    annual_income = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"])
    services_opted = st.slider("Services Opted", min_value=1, max_value=9, value=4, step=1)

with col3:
    social_sync = st.selectbox("Account Synced to Social Media", ["No", "Yes"])
    hotel_booked = st.selectbox("Booked Hotel", ["No", "Yes"])

# ── Predict ──────────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔍 Predict Churn", type="primary", use_container_width=True)

if predict_btn:
    input_data = pd.DataFrame([{
        "Age":                       age,
        "FrequentFlyer":             FREQUENT_FLYER_MAP[frequent_flyer],
        "AnnualIncomeClass":         INCOME_MAP[annual_income],
        "ServicesOpted":             services_opted,
        "AccountSyncedToSocialMedia": SOCIAL_MAP[social_sync],
        "BookedHotelOrNot":          HOTEL_MAP[hotel_booked],
    }])

    prediction = clf.predict(input_data)[0]
    probability = clf.predict_proba(input_data)[0]

    st.divider()
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        if prediction == 1:
            st.error("⚠️ **Prediction: Customer is likely to CHURN**")
        else:
            st.success("✅ **Prediction: Customer is NOT likely to churn**")

    with col_res2:
        st.metric("Churn Probability",    f"{probability[1]*100:.1f}%")
        st.metric("Retention Probability", f"{probability[0]*100:.1f}%")

    # Feature importance breakdown
    st.divider()
    st.subheader("📊 Feature Importance")
    feat_names  = ["Age", "FrequentFlyer", "AnnualIncomeClass",
                   "ServicesOpted", "AccountSyncedToSocialMedia", "BookedHotelOrNot"]
    feat_imp    = clf.feature_importances_
    imp_df      = pd.DataFrame({"Feature": feat_names, "Importance": feat_imp})
    imp_df      = imp_df.sort_values("Importance", ascending=False)
    st.bar_chart(imp_df.set_index("Feature"))
