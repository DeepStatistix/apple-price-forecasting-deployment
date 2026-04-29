"""
Apple Price Forecasting - NHiTS Streamlit Dashboard
======================================================
Loads pre-trained NHiTS models from experiment folders and forecasts
for 2026 sale-period dates only.

Markets: Azadpur, Shopian, Sopore
Varieties: American, Delicious
Grades: A, B
Horizons: 7 / 15 / 30 sale-period days

Usage:
    cd "d:/ML Repositories/marketdata"
    streamlit run deployment/streamlit_app.py
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deployment.experiment_registry import ExperimentRegistry
from deployment.inference_engine import NHiTSEngine

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Apple Price Forecasting Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for publication-grade look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.4rem;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #4a4a6a;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            border-left: 4px solid #4e73df;
        }
        .up-trend { color: #1cc88a; font-weight: 700; }
        .down-trend { color: #e74a3b; font-weight: 700; }
        .neutral-trend { color: #858796; font-weight: 700; }
        .stDataFrame { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cache registry
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data registry...")
def get_registry():
    return ExperimentRegistry()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
def render_sidebar(registry):
    st.sidebar.markdown("## 🍎 Forecast Configuration")
    st.sidebar.markdown("---")

    markets = registry.list_markets()
    if not markets:
        st.sidebar.error("No data files found in data/raw/processed/ folder.")
        st.stop()

    market = st.sidebar.selectbox("Select Market", markets)

    varieties = registry.list_varieties(market)
    if not varieties:
        st.sidebar.error(f"No varieties found for {market}.")
        st.stop()
    variety = st.sidebar.selectbox("Select Variety", varieties)

    grades = registry.list_grades(market, variety)
    if not grades:
        st.sidebar.error(f"No grades found for {market} / {variety}.")
        st.stop()
    grade = st.sidebar.selectbox("Select Grade", grades)

    # Fixed forecast horizons
    horizon = st.sidebar.selectbox("Forecast Horizon (sale days)", [7, 15, 30])

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "NHiTS model is loaded from the experiments folder. "
        "Forecasts are generated for 2026 sale-period dates only."
    )
    return market, variety, grade, horizon


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-header">Apple Price Forecasting Dashboard</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">NHiTS-based forecasting for 2026 sale-period dates '
            '(historical data up to 2025)</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.image(
            "https://img.icons8.com/color/96/apple.png",
            width=80,
        )
    st.markdown("---")


# ---------------------------------------------------------------------------
# Metadata panel
# ---------------------------------------------------------------------------
def render_metadata(entry, forecast_days):
    st.markdown("### Model Metadata")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Model Used", "NHiTS")
    with cols[1]:
        st.metric("Selected Horizon", f"{forecast_days} sale days")
    with cols[2]:
        st.metric("Forecast Year", "2026")
    with cols[3]:
        model_status = "✅ Loaded" if entry.nhits_model_path else "❌ Not Found"
        st.metric("Model Status", model_status)

    with st.expander("Detailed info"):
        st.json({
            "market": entry.market,
            "variety": entry.variety,
            "grade": entry.grade,
            "model_path": entry.nhits_model_path,
        })
    st.markdown("---")


# ---------------------------------------------------------------------------
# Directional indicator
# ---------------------------------------------------------------------------
def directional_badge(current, future):
    if current == 0:
        return "<span class='neutral-trend'>→ No change</span>"
    pct = (future - current) / current * 100
    if pct > 1:
        return f"<span class='up-trend'>▲ Up {pct:.1f}%</span>"
    elif pct < -1:
        return f"<span class='down-trend'>▼ Down {abs(pct):.1f}%</span>"
    else:
        return "<span class='neutral-trend'>→ Stable</span>"


# ---------------------------------------------------------------------------
# Forecast chart (gap-free)
# ---------------------------------------------------------------------------
def render_forecast_chart(forecast_df, hist_df, entry):
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(f"Price Trajectory — {entry.market} | {entry.variety} | Grade {entry.grade}",),
    )

    # Historical trace
    if not hist_df.empty:
        fig.add_trace(
            go.Scatter(
                x=hist_df["ds"],
                y=hist_df["y"],
                mode="lines+markers",
                name="Historical Price (≤2025)",
                line=dict(color="#4e73df", width=2),
                marker=dict(size=4),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f} Rs/kg<extra></extra>",
            )
        )

    # Gap-free forecast: prepend the last historical point so the line connects
    if not forecast_df.empty and not hist_df.empty:
        last_hist_date = hist_df["ds"].iloc[-1]
        last_hist_price = hist_df["y"].iloc[-1]

        gap_free_x = [last_hist_date] + forecast_df["ds"].tolist()
        gap_free_y = [last_hist_price] + forecast_df["forecast"].tolist()

        fig.add_trace(
            go.Scatter(
                x=gap_free_x,
                y=gap_free_y,
                mode="lines+markers",
                name="Forecasted Price (2026)",
                line=dict(color="#1cc88a", width=2, dash="dash"),
                marker=dict(size=5, symbol="diamond"),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: %{y:.2f} Rs/kg<extra></extra>",
            )
        )
    elif not forecast_df.empty:
        # Fallback if no historical data
        fig.add_trace(
            go.Scatter(
                x=forecast_df["ds"],
                y=forecast_df["forecast"],
                mode="lines+markers",
                name="Forecasted Price (2026)",
                line=dict(color="#1cc88a", width=2, dash="dash"),
                marker=dict(size=5, symbol="diamond"),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: %{y:.2f} Rs/kg<extra></extra>",
            )
        )

    # Vertical line at forecast start
    if not forecast_df.empty:
        x0 = forecast_df["ds"].iloc[0]
        fig.add_shape(
            type="line",
            x0=x0,
            x1=x0,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=x0,
            y=1,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            font=dict(color="gray", size=10),
            xanchor="left",
            yanchor="bottom",
        )

    fig.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Price (Rs / kg)",
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def render_summary(forecast_df, hist_df, entry):
    st.markdown("### Forecast Summary")
    c1, c2, c3, c4 = st.columns(4)

    last_hist_price = hist_df["y"].iloc[-1] if not hist_df.empty else np.nan
    avg_forecast = forecast_df["forecast"].mean()
    max_forecast = forecast_df["forecast"].max()
    min_forecast = forecast_df["forecast"].min()

    with c1:
        st.metric("Latest Historical Price", f"{last_hist_price:.2f} Rs/kg" if not np.isnan(last_hist_price) else "N/A")
    with c2:
        st.metric("Avg Forecast Price", f"{avg_forecast:.2f} Rs/kg")
    with c3:
        st.metric("Max Forecast Price", f"{max_forecast:.2f} Rs/kg")
    with c4:
        st.metric("Min Forecast Price", f"{min_forecast:.2f} Rs/kg")

    # Directional movement
    st.markdown("#### Directional Movement")
    start_price = forecast_df["forecast"].iloc[0] if not forecast_df.empty else np.nan
    end_price = forecast_df["forecast"].iloc[-1] if not forecast_df.empty else np.nan
    if not np.isnan(start_price) and not np.isnan(end_price):
        badge = directional_badge(start_price, end_price)
        st.markdown(f"Overall trend over forecast horizon: {badge}", unsafe_allow_html=True)

    # Day-by-day direction
    diffs = forecast_df["forecast"].diff().dropna()
    up_days = int((diffs > 0).sum())
    down_days = int((diffs < 0).sum())
    stable_days = int((diffs == 0).sum())
    st.caption(f"Sale-date breakdown: {up_days} up | {down_days} down | {stable_days} stable")
    st.markdown("---")


# ---------------------------------------------------------------------------
# Forecast table
# ---------------------------------------------------------------------------
def render_table(forecast_df):
    st.markdown("### Forecast Table (2026 Sale Dates)")
    display_df = forecast_df.copy()
    display_df["ds"] = display_df["ds"].dt.strftime("%Y-%m-%d")
    display_df = display_df.rename(columns={"ds": "Date", "forecast": "Forecasted Price (Rs/kg)"})
    display_df["Forecasted Price (Rs/kg)"] = display_df["Forecasted Price (Rs/kg)"].round(2)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    render_header()

    registry = get_registry()
    market, variety, grade, horizon = render_sidebar(registry)

    entry = registry.get_entry(market, variety, grade)
    if entry is None:
        st.error(f"No data file found for {market} / {variety} / Grade {grade}")
        st.stop()

    render_metadata(entry, horizon)

    # Warn if no NHITS model is available
    if not entry.nhits_model_path:
        st.warning(
            f"⚠️ No pre-trained NHiTS model found for {market} / {variety} / Grade {grade}. "
            f"Please train a model first."
        )
        st.stop()

    # Generate forecast only when user clicks the button
    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Loading NHiTS model and forecasting 2026/2027 sale dates..."):
            try:
                engine = NHiTSEngine(
                    entry.data_path,
                    model_path=entry.nhits_model_path,
                    market=entry.market,
                    variety=entry.variety,
                    grade=entry.grade,
                )
                forecast_df, hist_df = engine.forecast(horizon)
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                st.stop()

        if forecast_df.empty:
            st.warning("Forecast generation returned empty results.")
            st.stop()

        # Check for NaN forecasts
        nan_count = forecast_df["forecast"].isna().sum()
        if nan_count > 0:
            st.warning(f"Warning: {nan_count} out of {len(forecast_df)} forecast values are NaN.")

        render_forecast_chart(forecast_df, hist_df, entry)
        render_summary(forecast_df, hist_df, entry)
        render_table(forecast_df)

        # Footer
        st.markdown("---")
        st.caption(
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
            f"Model: NHiTS (pre-trained) | "
            f"Market: {entry.market} | Variety: {entry.variety} | Grade: {entry.grade}"
        )
    else:
        st.info("Click **Generate Forecast** to load the NHiTS model and view 2026 sale-period predictions.")


if __name__ == "__main__":
    main()

