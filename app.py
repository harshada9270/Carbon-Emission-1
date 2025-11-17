import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Carbon Emissions Predictor", layout="wide")

# Dark theme styling for Streamlit + Matplotlib
plt.style.use("dark_background")
matplotlib.rcParams.update({
    'axes.facecolor': '#0e1117',
    'figure.facecolor': '#0e1117',
    'savefig.facecolor': '#0e1117',
    'grid.color': '#3a3f44',
    'text.color': '#e8eaed',
    'axes.labelcolor': '#e8eaed',
    'xtick.color': '#e8eaed',
    'ytick.color': '#e8eaed',
    'axes.edgecolor': '#5f6368'
})

# Force dark background for the app shell
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp, html, body { background-color: #0e1117; color: #e8eaed; }
    .stMetric label, .st-emotion-cache-10trblm { color: #e8eaed !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_artifacts():
    with open("carbon_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data

artifacts = load_artifacts()
model = artifacts["model"]
scaler_X = artifacts["scaler_X"]
scaler_y = artifacts["scaler_y"]
countries = artifacts["countries"]
feature_cols = artifacts["feature_columns"]
historical_df = artifacts["historical_df"]

st.title("🌍 Carbon Emissions Prediction System")
st.markdown("### Predict future carbon emissions using machine learning")
st.write("Select a country and target year to see AI-powered predictions with trend analysis.")

col1, col2 = st.columns(2)
with col1:
    country = st.selectbox("🏳️ Select Country", countries, index=countries.index('China') if 'China' in countries else 0)
with col2:
    year = st.slider("📅 Target Year", min_value=2023, max_value=2035, value=2030)

def make_input(country_name, target_year):
    base = {col: 0 for col in feature_cols}
    base["Year"] = target_year
    country_col = f"Country_{country_name}"
    if country_col in base:
        base[country_col] = 1
    return pd.DataFrame([base], columns=feature_cols)

def predict(country_name, target_year):
    input_df = make_input(country_name, target_year)
    input_scaled = scaler_X.transform(input_df)
    pred_scaled = model.predict(input_scaled, verbose=0)
    return scaler_y.inverse_transform(pred_scaled)[0][0]

def get_trend_label(history, future_value):
    last_actual = history["Emission"].iloc[-1]
    diff = future_value - last_actual
    if diff > 200:  # increase threshold
        return "high", "Try investing in renewable energy and efficiency.implementing carbon removal strategies.reducing food waste and improving land use efficiency"
    if diff < -200:  # decrease threshold
        return "good", "Keep up the policies that reduce emissions.investing in carbon capture and storage technologies.promoting sustainable agriculture and forestry practices.supporting policies that encourage low-carbon transportation."
    return "stable", "Maintain current efforts; monitor for changes.investing in renewable energy and energy efficiency.supporting policies that encourage low-carbon transportation.promoting sustainable agriculture and forestry practices."

# Historical data for selected country
country_hist = historical_df[historical_df["Country"] == country].sort_values("Year")

prediction = predict(country, year)
label, suggestion = get_trend_label(country_hist, prediction)

st.subheader(" Prediction Results")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric(label="Predicted Emission (metric tons CO₂)", value=f"{prediction:,.2f}")
with col2:
    if label == "high":
        st.error(f" **INCREASING TREND** — {suggestion}")
    elif label == "good":
        st.success(f" **DECREASING TREND** — {suggestion}")
    else:
        st.info(f" **STABLE TREND** — {suggestion}")

# Plot historical vs future
future_years = range(int(country_hist["Year"].max()) + 1, year + 1)
future_preds = [predict(country, y) for y in future_years]

st.subheader("Emissions Trend Analysis")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(country_hist["Year"], country_hist["Emission"], marker="o", linewidth=2.5, markersize=5, label="Historical Data", color="#1f77b4")
if future_preds:
    ax.plot(list(future_years), future_preds, marker="s", linestyle="--", linewidth=2.5, markersize=6, color="#ff7f0e", label="AI Predictions")
    ax.scatter(year, future_preds[-1], color="#d62728", s=120, label=f"{year} Target", zorder=5, edgecolors='white', linewidth=2)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Carbon Emissions (metric tons CO₂)", fontsize=12)
ax.set_title(f"{country} - Carbon Emissions Forecast", fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

