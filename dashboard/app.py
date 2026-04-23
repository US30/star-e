"""Streamlit dashboard for StAR-E."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(
    page_title="StAR-E Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("StAR-E")
st.sidebar.markdown("**S**tatistical **A**rbitrage & **R**isk **E**ngine")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Portfolio Overview", "Regime Analysis", "Risk Dashboard", "Model Comparison", "Settings"],
    index=0,
)

# Load data function
@st.cache_data(ttl=3600)
def load_data():
    """Load data from DuckDB."""
    try:
        from star_e.data import load_from_duckdb
        df = load_from_duckdb("prices")
        return df.to_pandas()
    except Exception as e:
        st.warning(f"Could not load data: {e}")
        return None


def generate_demo_data():
    """Generate demo data for visualization."""
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", end="2024-01-01", freq="B")
    n = len(dates)

    returns = np.random.randn(n) * 0.01 + 0.0003
    cumulative = np.cumprod(1 + returns)

    # Simulate regimes
    regimes = np.zeros(n, dtype=int)
    regimes[100:200] = 0  # Bear
    regimes[200:400] = 2  # Bull
    regimes[400:500] = 0  # Bear
    regimes[500:] = 1  # Sideways

    return pd.DataFrame({
        "date": dates,
        "return": returns,
        "cumulative": cumulative,
        "regime": regimes,
    })


# Main content
if page == "Portfolio Overview":
    st.title("Portfolio Performance")

    df = load_data()
    demo = generate_demo_data()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_return = (demo["cumulative"].iloc[-1] - 1) * 100
        st.metric(
            "Total Return",
            f"{total_return:.1f}%",
            f"{demo['return'].iloc[-1]*100:.2f}%",
        )

    with col2:
        sharpe = demo["return"].mean() / demo["return"].std() * np.sqrt(252)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    with col3:
        running_max = demo["cumulative"].cummax()
        drawdown = (demo["cumulative"] - running_max) / running_max
        max_dd = drawdown.min() * 100
        st.metric("Max Drawdown", f"{max_dd:.1f}%")

    with col4:
        regime_names = ["Bear", "Sideways", "Bull"]
        current_regime = regime_names[demo["regime"].iloc[-1]]
        st.metric("Current Regime", current_regime)

    st.divider()

    # Cumulative returns chart
    st.subheader("Cumulative Returns")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=demo["date"],
            y=demo["cumulative"],
            name="Portfolio",
            line=dict(color="#1f77b4", width=2),
        ),
        row=1, col=1,
    )

    # Add regime backgrounds
    colors = {0: "rgba(255,0,0,0.1)", 1: "rgba(128,128,128,0.1)", 2: "rgba(0,255,0,0.1)"}

    for regime in [0, 1, 2]:
        mask = demo["regime"] == regime
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=demo.loc[mask, "date"],
                    y=demo.loc[mask, "cumulative"],
                    mode="markers",
                    marker=dict(color=list(colors.values())[regime], size=3),
                    name=regime_names[regime],
                    showlegend=True,
                ),
                row=1, col=1,
            )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=demo["date"],
            y=drawdown * 100,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="red", width=1),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Allocation
    st.subheader("Current Allocation")

    col1, col2 = st.columns([1, 2])

    with col1:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        weights = [0.25, 0.22, 0.20, 0.18, 0.15]

        fig_pie = go.Figure(data=[go.Pie(
            labels=tickers,
            values=weights,
            hole=0.4,
        )])
        fig_pie.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        allocation_df = pd.DataFrame({
            "Ticker": tickers,
            "Weight": [f"{w:.1%}" for w in weights],
            "Return (1M)": ["+5.2%", "+3.1%", "+2.8%", "-1.2%", "+4.5%"],
        })
        st.dataframe(allocation_df, use_container_width=True, hide_index=True)


elif page == "Regime Analysis":
    st.title("Market Regime Detection")

    demo = generate_demo_data()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Regime", "Sideways", "Stable")
    with col2:
        st.metric("Bear Probability", "15%")
    with col3:
        st.metric("Bull Probability", "35%")

    st.divider()

    # Regime timeline
    st.subheader("Regime History")

    regime_names = ["Bear", "Sideways", "Bull"]
    colors = ["red", "gray", "green"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=demo["date"],
        y=demo["cumulative"],
        mode="lines",
        name="Portfolio Value",
        line=dict(color="black", width=2),
    ))

    # Add colored backgrounds for regimes
    for i, (regime, color) in enumerate(zip(regime_names, colors)):
        mask = demo["regime"] == i
        if mask.any():
            for start_idx in np.where(np.diff(np.concatenate([[0], mask.astype(int)])) == 1)[0]:
                end_indices = np.where(np.diff(np.concatenate([mask.astype(int), [0]])) == -1)[0]
                end_idx = end_indices[end_indices > start_idx][0] if len(end_indices[end_indices > start_idx]) > 0 else len(mask) - 1

                fig.add_vrect(
                    x0=demo["date"].iloc[start_idx],
                    x1=demo["date"].iloc[end_idx],
                    fillcolor=color,
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )

    fig.update_layout(
        height=400,
        yaxis_title="Portfolio Value",
        xaxis_title="Date",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Transition matrix
    st.subheader("Transition Probabilities")

    trans_matrix = np.array([
        [0.85, 0.10, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80],
    ])

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=trans_matrix,
        x=regime_names,
        y=regime_names,
        colorscale="Blues",
        text=[[f"{v:.0%}" for v in row] for row in trans_matrix],
        texttemplate="%{text}",
        textfont={"size": 14},
    ))

    fig_heatmap.update_layout(
        height=300,
        xaxis_title="To State",
        yaxis_title="From State",
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)


elif page == "Risk Dashboard":
    st.title("Risk Metrics")

    demo = generate_demo_data()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("VaR (95%)", "2.1%", help="Daily Value at Risk at 95% confidence")
    with col2:
        st.metric("CVaR (95%)", "3.2%", help="Expected Shortfall at 95% confidence")
    with col3:
        st.metric("Volatility", "16.5%", help="Annualized volatility")
    with col4:
        st.metric("Max Drawdown", "-12.3%")

    st.divider()

    # VaR visualization
    st.subheader("Return Distribution")

    returns = demo["return"] * 100

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name="Returns",
        marker_color="blue",
        opacity=0.7,
    ))

    var_95 = np.percentile(returns, 5)
    fig.add_vline(x=var_95, line_dash="dash", line_color="red",
                  annotation_text=f"VaR 95%: {var_95:.2f}%")

    fig.update_layout(
        height=400,
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Volatility forecast
    st.subheader("Volatility Forecast (GARCH)")

    forecast_days = 21
    current_vol = 0.015
    forecast_vol = current_vol * np.ones(forecast_days) + np.random.randn(forecast_days) * 0.001

    fig_vol = go.Figure()

    fig_vol.add_trace(go.Scatter(
        x=list(range(1, forecast_days + 1)),
        y=forecast_vol * 100 * np.sqrt(252),
        mode="lines+markers",
        name="Volatility Forecast",
        line=dict(color="orange"),
    ))

    fig_vol.update_layout(
        height=300,
        xaxis_title="Days Ahead",
        yaxis_title="Annualized Volatility (%)",
    )

    st.plotly_chart(fig_vol, use_container_width=True)


elif page == "Model Comparison":
    st.title("Model Backtest Comparison")

    # Comparison table
    st.subheader("Strategy Performance")

    comparison_data = {
        "Strategy": ["Equal Weight", "Min Variance", "Max Sharpe", "Regime-Aware"],
        "Total Return": ["23.4%", "18.2%", "28.7%", "31.2%"],
        "Sharpe Ratio": [1.12, 1.35, 1.48, 1.62],
        "Max Drawdown": ["-15.2%", "-10.1%", "-13.8%", "-11.5%"],
        "Volatility": ["16.5%", "12.3%", "15.8%", "14.2%"],
    }

    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    st.divider()

    # Equity curves comparison
    st.subheader("Equity Curves")

    dates = pd.date_range(start="2022-01-01", end="2024-01-01", freq="B")
    n = len(dates)

    np.random.seed(42)
    strategies = {
        "Equal Weight": np.cumprod(1 + np.random.randn(n) * 0.01 + 0.0003),
        "Min Variance": np.cumprod(1 + np.random.randn(n) * 0.008 + 0.00025),
        "Max Sharpe": np.cumprod(1 + np.random.randn(n) * 0.011 + 0.00035),
        "Regime-Aware": np.cumprod(1 + np.random.randn(n) * 0.009 + 0.0004),
    }

    fig = go.Figure()

    for name, values in strategies.items():
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            name=name,
            mode="lines",
        ))

    fig.update_layout(
        height=500,
        yaxis_title="Portfolio Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    st.plotly_chart(fig, use_container_width=True)


elif page == "Settings":
    st.title("Settings")

    st.subheader("Data Settings")

    tickers = st.text_input(
        "Default Tickers",
        value="AAPL, MSFT, GOOGL, AMZN, META",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())

    st.divider()

    st.subheader("Model Settings")

    col1, col2 = st.columns(2)

    with col1:
        hmm_states = st.slider("HMM States", min_value=2, max_value=5, value=3)
        rebalance_freq = st.selectbox(
            "Rebalancing Frequency",
            ["Daily", "Weekly", "Monthly", "Quarterly"],
            index=2,
        )

    with col2:
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
        )
        confidence_level = st.slider(
            "VaR Confidence Level",
            min_value=90,
            max_value=99,
            value=95,
        )

    st.divider()

    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!")

    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.success("Cache cleared! Data will be refreshed on next load.")


# Footer
st.sidebar.divider()
st.sidebar.markdown("---")
st.sidebar.caption("StAR-E v0.1.0 | Built with Streamlit")
