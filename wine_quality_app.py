"""Wine-quality regression demonstration built on a tiny toy dataset.

Author: Moe Htet Ar Kar (Phoe Cho)

Only ten hand-entered records are included. Metrics are highly unstable and
must not be used to claim that this model predicts actual wine quality.
"""

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title = "Wine Quality Lab", page_icon = "🍷", layout ="wide")
st.markdown("<style>" + """.stApp{background:linear-gradient(145deg,#21121a 0%,#321a28 55%,#1f121a 100%);color:#fff0f5;} [data-testid='stSidebar']{background:#301725;} 
[data-testid='stMetric']{background:#3b202f;border:1px solid #82415e;padding:14px;border-radius:12px;} .stButton>button{background:#e78cae;color:#2c1420;border:0;
border-radius:9px;font-weight:700;} h1,h2,h3{color:#fff0f5;}""" + "</style>", unsafe_allow_html = True)

st.title("🍷 Wine Quality Regression Lab")
st.caption("Random Forest Regression · Small-data learning example")
st.warning("Only 10 manually specified training rows are used. This is a toy example; its metrics and predictions are not reliable quality assessments.")

FEATURES = ["Alcohol", "Acidity", "Sugar"]


@st.cache_resource
def train_wine_model():
    """Fit a regressor and return metrics from a very small holdout split."""
    toy_data = pd.DataFrame(
        {
            "Alcohol": [9.4, 9.8, 10.1, 9.1, 10.5, 11.2, 9.5, 10.8, 11.5, 9.9],
            "Acidity": [0.7, 0.88, 0.76, 0.38, 0.5, 0.45, 0.65, 0.55, 0.42, 0.60],
            "Sugar": [1.9, 2.6, 2.3, 1.6, 2.0, 2.2, 1.8, 2.4, 2.1, 1.9],
            "Quality": [5, 5, 5, 6, 6, 7, 5, 6, 7, 6],
        }
    )
    x_train, x_test, y_train, y_test = train_test_split(
        toy_data[FEATURES], toy_data["Quality"], test_size = 0.2, random_state = 42
    )
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)
    regressor.fit(x_train, y_train)
    predictions = regressor.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return regressor, float(mse), float(r2), toy_data


model, mse, r2, training_data = train_wine_model()
st.sidebar.header("Chemical composition inputs")
alcohol = st.sidebar.slider("Alcohol content (%)", 8.0, 15.0, 10.2)
acidity = st.sidebar.slider("Volatile acidity (g/dm³)", 0.2, 1.5, 0.6)
sugar = st.sidebar.slider("Residual sugar (g/dm³)", 1.0, 15.0, 2.0)

left, right = st.columns([1, 1])
with left:
    st.subheader("Input profile")
    st.metric("Alcohol", f"{alcohol:.1f}%")
    st.metric("Volatile acidity", f"{acidity:.2f} g/dm³")
    st.metric("Residual sugar", f"{sugar:.1f} g/dm³")
    st.metric("Training records", f"{len(training_data)}")
with right:
    st.subheader("Prediction report")
    if st.button("Estimate toy quality score", use_container_width = True):
        row = pd.DataFrame([[alcohol, acidity, sugar]], columns = FEATURES)
        estimated_score = float(model.predict(row)[0])
        st.metric("Estimated score (model output)", f"{estimated_score:.2f} / 10")
        st.caption("The model predicts only within the narrow patterns represented by ten toy rows; this is not a verified rating.")
    else:
        st.info("Set the inputs and click **Estimate toy quality score**.")

st.subheader("Holdout metrics — interpret with caution")
metric_col1, metric_col2 = st.columns(2)
metric_col1.metric("Mean squared error", f"{mse:.4f}")
metric_col2.metric("R² score", f"{r2:.4f}")
st.caption("The test split contains only two records. These values are unstable and must not be presented as evidence of real-world accuracy.")

st.divider()
st.caption("Educational regression project · 10-row toy dataset · Not a validated wine-quality model")
