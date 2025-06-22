import streamlit as st
import numpy as np
import joblib
import os

# --- Page Configuration ---
st.set_page_config(page_title="Credit Score Predictor", layout="wide")
st.title("💳 Credit Score Prediction App")
st.markdown("Enter the applicant's financial details to estimate their credit score category.")

# --- Define dynamic file paths ---
CURRENT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(CURRENT_DIR, "models", "scaler.pkl")


# --- Load model and scaler ---
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model or scaler: {e}")
        return None, None

model, scaler = load_model_and_scaler()
if model is None or scaler is None:
    st.stop()

# --- Sidebar info ---
st.sidebar.title("🧾 Feature Guidance")
st.sidebar.markdown("""
- **Outstanding Debt**: Total debt in your account  
- **Interest Rate**: Annual percentage rate on loans  
- **Credit Mix**: Credit diversity (Poor = 0, Standard = 1, Good = 2)  
- **Delay from Due Date**: Average late days  
- **Changed Credit Limit**: Amount your credit limit changed  
- **Credit History Age**: Total months since credit inception  
- **Num Credit Inquiries**: Recent hard pulls  
- **Monthly Balance**: End-of-month account balance  
- **Amount Invested Monthly**: Monthly savings or investments  
- **Number of Credit Cards**: Active cards on file  
""")

# --- Input form ---
st.subheader("📋 Enter Applicant Details")
col1, col2, col3 = st.columns(3)

with col1:
    outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, step=100.0)
    credit_mix_label = st.selectbox("Credit Mix", ["Poor", "Standard", "Good"])
    credit_mix = {"Poor": 0, "Standard": 1, "Good": 2}[credit_mix_label]
    credit_history = st.number_input("Credit History Age (months)", min_value=0, step=1, value=36)

with col2:
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.5)
    delay_days = st.number_input("Avg. Delay from Due Date (days)", min_value=0, step=1)
    credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, step=1)

with col3:
    credit_limit_change = st.number_input("Changed Credit Limit", step=100.0)
    monthly_balance = st.number_input("Monthly Balance", step=100.0)
    monthly_investment = st.number_input("Amount Invested Monthly", step=100.0)
    num_credit_card = st.number_input("Number of Credit Cards", min_value=0, step=1)

# --- Format input for prediction ---
input_features = np.array([[
    outstanding_debt,
    interest_rate,
    credit_mix,
    delay_days,
    credit_limit_change,
    credit_history,
    credit_inquiries,
    monthly_balance,
    monthly_investment,
    num_credit_card
]])

# --- Prediction ---
if st.button("🔍 Predict Credit Score"):
    try:
        input_scaled = scaler.transform(input_features)
        prediction = model.predict(input_scaled)[0]

        label_map = {0: "Poor", 1: "Standard", 2: "Good"}
        prediction_label = label_map.get(prediction, "Unknown")

        st.markdown("---")
        st.success(f"🏷️ Predicted Credit Score Category: **{prediction_label}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Developed by Ifeoma Adigwe • Powered by Streamlit & scikit-learn")
