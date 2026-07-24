import os
import sys
import yaml
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="XAU/USD Volatility Regime Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("🥇 XAU/USD Market Regime & Volatility Dashboard")
st.markdown("Visualizing historical gold price trends alongside XGBoost **Volatility Regime** predictions.")

# Load configuration and model
@st.cache_data
def load_data(config):
    processed_path = config["data"]["processed_data_path"]
    if not os.path.exists(processed_path):
        st.error(f"Processed data file missing at {processed_path}. Run features pipeline first!")
        st.stop()
    return pd.read_parquet(processed_path)

@st.cache_resource
def load_model(config):
    model_path = config["model"]["model_save_path"]
    if not os.path.exists(model_path):
        st.error(f"Model file missing at {model_path}. Train model first!")
        st.stop()
    return joblib.load(model_path)

# Parse YAML config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load data and artifacts
df = load_data(config)
model = load_model(config)

features = config["model"]["features"]

# Generate full-dataset predictions
df["pred_regime"] = model.predict(df[features])
df["prob_high_vol"] = model.predict_proba(df[features])[:, 1]

# Sidebar filters
st.sidebar.header("📊 Filter Options")
min_date = pd.to_datetime(df["date"]).min().date()
max_date = pd.to_datetime(df["date"]).max().date()

selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_d, end_d = selected_dates
    mask = (pd.to_datetime(df["date"]).dt.date >= start_d) & (pd.to_datetime(df["date"]).dt.date <= end_d)
    filtered_df = df[mask].copy()
else:
    filtered_df = df.copy()

# Summary Metric Cards
col1, col2, col3, col4 = st.columns(4)

latest_row = filtered_df.iloc[-1] if len(filtered_df) > 0 else df.iloc[-1]
current_regime = "🔴 High Volatility" if latest_row["pred_regime"] == 1 else "🟢 Normal Volatility"

col1.metric("Latest Close Price", f"${latest_row['close']:,.2f}")
col2.metric("5-Day Rolling Volatility", f"{latest_row['volatility_5d']*100:.2f}%")
col3.metric("Current Market Regime", current_regime)
col4.metric("High Vol Probability", f"{latest_row['prob_high_vol']*100:.1f}%")

st.divider()

# Plotly Dual-Panel Subplot Chart
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.08, 
    row_heights=[0.7, 0.3],
    subplot_titles=("Gold Close Price (USD) with High Volatility Overlay", "Model Predicted Probability of High Volatility")
)

# Gold Price Line
fig.add_trace(
    go.Scatter(x=filtered_df["date"], y=filtered_df["close"], name="Gold Price", line=dict(color="#FFD700", width=2)),
    row=1, col=1
)

# Highlight High Volatility Regimes on Price Panel
high_vol_df = filtered_df[filtered_df["pred_regime"] == 1]
fig.add_trace(
    go.Scatter(
        x=high_vol_df["date"], 
        y=high_vol_df["close"], 
        mode="markers", 
        name="High Vol Regime", 
        marker=dict(color="#FF4136", size=4)
    ),
    row=1, col=1
)

# Probability Area Trace
fig.add_trace(
    go.Scatter(
        x=filtered_df["date"], 
        y=filtered_df["prob_high_vol"], 
        name="High Vol Probability", 
        fill="tozeroy",
        line=dict(color="#FF4136", width=1)
    ),
    row=2, col=1
)

fig.update_layout(height=650, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Probability", range=[0, 1], row=2, col=1)

st.plotly_chart(fig, use_container_width=True)
