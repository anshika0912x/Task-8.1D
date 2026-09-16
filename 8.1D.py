"""
Sydney Housing Price Prediction App
SIT307 Machine Learing - 8.1 Distinction Task
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ---------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "log_linear_model.pkl")

st.set_page_config(
    page_title="Sydney Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# ---------------------------------------------------------
# APP TITLE
# ---------------------------------------------------------

st.title("🏠 Sydney Housing Price Predictor")

st.write(
    "Estimate a property's sale price in **Mosman**, **Marrickville**, or "
    "**Blacktown** using a log-target linear regression model trained on "
    "106 recent property sales."
)


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model(path):
    return joblib.load(path)


try:
    model = load_model(MODEL_PATH)
    model_loaded = True

except FileNotFoundError:
    model_loaded = False

    st.error(
        f"Could not find `log_linear_model.pkl` at:\n\n"
        f"`{MODEL_PATH}`\n\n"
        "Run the model export cell in the notebook and make sure "
        "`log_linear_model.pkl` is saved in the same folder as this app."
    )

except Exception as e:
    model_loaded = False

    st.error(
        f"Found `log_linear_model.pkl` but could not load it.\n\n"
        f"Error: {type(e).__name__}: {e}\n\n"
        "This may be caused by a difference between the scikit-learn "
        "version used to train the model and the version installed locally."
    )


# ---------------------------------------------------------
# APP STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* Slightly improve button appearance */
    div[data-testid="stButton"] button {
        min-height: 2.4rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# CUSTOM COUNTER INPUT
# ---------------------------------------------------------

def update_counter(key, amount, min_value, max_value):
    """Increase or decrease a counter while keeping it inside its limits."""

    new_value = st.session_state[key] + amount

    st.session_state[key] = max(
        min_value,
        min(max_value, new_value)
    )


def counter_input(
    label,
    key,
    min_value,
    max_value,
    default,
    step=1
):
    """Create a simple minus/value/plus counter."""

    if key not in st.session_state:
        st.session_state[key] = default

    st.write(label)

    minus_col, value_col, plus_col = st.columns([1, 1.2, 1])

    with minus_col:
        st.button(
            "−",
            key=f"{key}_minus",
            on_click=update_counter,
            args=(key, -step, min_value, max_value),
            use_container_width=True
        )

    with value_col:
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 1.2rem;
                font-weight: 600;
                padding-top: 0.45rem;
            ">
                {st.session_state[key]}
            </div>
            """,
            unsafe_allow_html=True
        )

    with plus_col:
        st.button(
            "+",
            key=f"{key}_plus",
            on_click=update_counter,
            args=(key, step, min_value, max_value),
            use_container_width=True
        )

    return st.session_state[key]


# ---------------------------------------------------------
# PROPERTY INPUTS
# ---------------------------------------------------------

st.header("Property details")

col1, col2 = st.columns(2)


# LEFT COLUMN
with col1:

    suburb = st.selectbox(
        "Suburb",
        [
            "Mosman",
            "Marrickville",
            "Blacktown"
        ]
    )

    property_type = st.selectbox(
        "Property type",
        [
            "House",
            "Apartment",
            "Townhouse"
        ]
    )

    bedrooms = counter_input(
        "Bedrooms",
        "bedrooms",
        min_value=1,
        max_value=10,
        default=3
    )

    bathrooms = counter_input(
        "Bathrooms",
        "bathrooms",
        min_value=1,
        max_value=8,
        default=2
    )


# RIGHT COLUMN
with col2:

    parking_spaces = counter_input(
        "Parking spaces",
        "parking_spaces",
        min_value=0,
        max_value=10,
        default=1
    )

    area_m2 = counter_input(
        "Area (m²)",
        "area_m2",
        min_value=10,
        max_value=2000,
        default=250,
        step=5
    )

    sale_month = st.slider(
        "Month of sale",
        min_value=1,
        max_value=12,
        value=9
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

st.divider()

if st.button(
    "Predict sale price",
    type="primary",
    disabled=not model_loaded,
    use_container_width=True
):

    # Create dataframe using the same feature names
    # used when training the machine learning model.
    input_df = pd.DataFrame(
        {
            "Suburb": [suburb],
            "PropertyType": [property_type],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "ParkingSpaces": [parking_spaces],
            "Area_m2": [area_m2],
            "SaleMonth": [sale_month],
        }
    )

    try:

        # Model predicts log-transformed sale price
        predicted_log_price = model.predict(input_df)

        # Convert prediction back to dollars
        predicted_price = np.expm1(
            predicted_log_price
        )[0]

        st.success(
            f"### Estimated sale price: "
            f"${predicted_price:,.0f}"
        )

        st.caption(
            "This is an approximate estimate based on a model trained "
            "on 106 property sales across three Sydney suburbs. "
            "The model does not account for factors such as property "
            "condition, renovations, views, school catchments, or exact "
            "street location. The prediction should therefore be treated "
            "as an estimate rather than a professional property valuation."
        )

        with st.expander(
            "See the input used for this prediction"
        ):

            st.dataframe(
                input_df,
                hide_index=True,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Prediction could not be completed.\n\n"
            f"Error: {type(e).__name__}: {e}"
        )


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

st.divider()

st.caption(
    "SIT307 Machine Learning Mini Project · "
    "Model: Log-Target Linear Regression · "
    "Test R² = 0.770 · "
    "Test MAE ≈ $417,310"
)
