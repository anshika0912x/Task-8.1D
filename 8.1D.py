"""
Sydney Housing Price Prediction App
SIT307 8.1 Distinction Task

"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "log_linear_model.pkl")

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 Sydney Housing Price Predictor")
st.write(
    "Estimate a property's sale price in **Mosman**, **Marrickville**, or **Blacktown** "
    "using a log-target linear regression model trained on 106 recent sales."
)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


try:
    model = load_model(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error(
        f"Could not find `log_linear_model.pkl` at:\n\n`{MODEL_PATH}`\n\n"
        "Run the notebook's Part 5 cell "
        "(`joblib.dump(log_linear_model, 'log_linear_model.pkl')`) and confirm the "
        "resulting file is saved in this exact folder."
    )
except Exception as e:
    model_loaded = False
    st.error(
        f"Found `log_linear_model.pkl` but couldn't load it ({type(e).__name__}: {e}). "
        "This usually means the scikit-learn version used to train the model doesn't "
        "match the one installed here — try `pip install --upgrade scikit-learn joblib` "
        "or re-run the notebook's Part 5 cell to regenerate the file."
    )

st.markdown(
    """
    <style>
    div[class*="st-key-counter_"] div[data-testid="stHorizontalBlock"] {
        gap: 0.3rem;
        max-width: 170px;
    }
    div[class*="st-key-counter_"] button {
        width: 2.4rem;
        padding: 0.25rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def counter_input(label, key, min_value, max_value, default, step=1):
    if key not in st.session_state:
        st.session_state[key] = default

    st.write(label)
    with st.container(key=f"counter_{key}"):
        minus_col, value_col, plus_col = st.columns([1, 1, 1])

        with minus_col:
            if st.button("−", key=f"{key}_minus"):
                st.session_state[key] = max(min_value, st.session_state[key] - step)

        with value_col:
            st.markdown(
                f"<div style='text-align:center; font-size:1.1em; font-weight:600; "
                f"padding-top:0.45rem;'>{st.session_state[key]}</div>",
                unsafe_allow_html=True,
            )

        with plus_col:
            if st.button("+", key=f"{key}_plus"):
                st.session_state[key] = min(max_value, st.session_state[key] + step)

    return st.session_state[key]


st.header("Property details")

col1, col2 = st.columns(2)

with col1:
    suburb = st.selectbox("Suburb", ["Mosman", "Marrickville", "Blacktown"])
    property_type = st.selectbox("Property type", ["House", "Apartment", "Townhouse"])
    bedrooms = counter_input("Bedrooms", "bedrooms", min_value=1, max_value=10, default=3)
    bathrooms = counter_input("Bathrooms", "bathrooms", min_value=1, max_value=8, default=2)

with col2:
    parking_spaces = counter_input("Parking spaces", "parking_spaces", min_value=0, max_value=10, default=1)
    area_m2 = counter_input("Area (m²)", "area_m2", min_value=10, max_value=2000, default=250, step=5)
    sale_month = st.slider("Month of sale (1-12)", min_value=1, max_value=12, value=9)

st.divider()

if st.button("Predict sale price", type="primary", disabled=not model_loaded):
    input_df = pd.DataFrame({
        "Suburb": [suburb],
        "PropertyType": [property_type],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "ParkingSpaces": [parking_spaces],
        "Area_m2": [area_m2],
        "SaleMonth": [sale_month],
    })

    predicted_log_price = model.predict(input_df)
    predicted_price = np.expm1(predicted_log_price)[0]

    st.success(f"### Estimated sale price: ${predicted_price:,.0f}")
    st.caption(
        "This is an approximate estimate based on a model trained on 106 sales across three "
        "suburbs. It does not account for property condition, renovations, views, school "
        "catchments, or exact street location. Treat it as a starting point, not a valuation."
    )

    with st.expander("See the input used for this prediction"):
        st.dataframe(input_df, hide_index=True)

st.divider()
st.caption(
    "SIT307 Machine Learning Mini Project · Model: log-target linear regression · "
    "Test R² = 0.770, Test MAE ≈ $417,310"
)