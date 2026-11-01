"""Streamlit app - Wellness Tourism Package purchase predictor.

Loads the model that the GitHub Actions pipeline committed next to this file,
collects the customer details into a DataFrame and shows the prediction.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_tourism_model_v1.joblib"
METRICS_PATH = APP_DIR / "model_metrics.json"

st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="🧳", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_threshold() -> float:
    if METRICS_PATH.exists():
        return float(json.loads(METRICS_PATH.read_text()).get("threshold", 0.45))
    return 0.45


st.title("🧳 Wellness Tourism Package Predictor")
st.write("Enter the customer details below to estimate whether the customer is likely "
         "to purchase the **Wellness Tourism Package** before the sales team makes contact.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run the GitHub Actions pipeline so it commits "
             "`best_tourism_model_v1.joblib` to `tourism_project/deployment/`.")
    st.stop()

model = load_model()
threshold = load_threshold()

st.subheader("Customer details")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly income", min_value=1000, max_value=200000, value=22000, step=500)
    city_tier = st.selectbox("City tier", [1, 2, 3])
    passport = st.selectbox("Holds a valid passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a car?", ["Yes", "No"])
with c2:
    persons = st.number_input("Number of persons visiting", min_value=1, max_value=10, value=3)
    children = st.number_input("Children under 5 visiting", min_value=0, max_value=5, value=1)
    trips = st.number_input("Trips per year", min_value=0, max_value=25, value=3)
    star = st.selectbox("Preferred property star rating", [3.0, 4.0, 5.0])
    contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)
    duration = st.number_input("Duration of pitch (minutes)", min_value=1, max_value=150, value=15)
    satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)

# Collect the inputs into a DataFrame with the same columns used in training
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups,
    "ProductPitched": product,
    "PreferredPropertyStar": star,
    "MaritalStatus": marital,
    "NumberOfTrips": trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children,
    "Designation": designation,
    "MonthlyIncome": income,
}])

with st.expander("Model input (DataFrame)"):
    st.dataframe(input_df)

if st.button("Predict", type="primary"):
    probability = float(model.predict_proba(input_df)[0, 1])
    st.metric("Purchase probability", f"{probability:.1%}")
    if probability >= threshold:
        st.success("✅ Likely to purchase the Wellness Tourism Package - prioritise this customer.")
    else:
        st.warning("⚠️ Unlikely to purchase the package at this time.")
    st.caption(f"Decision threshold: {threshold:.2f}")
