import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

from factors import calculate_factors
from portfolio import build_factor_score, select_stocks
from backtest import backtest

# --- App title ---
st.title("📈 Factor Investing Simulator")

# --- User inputs ---
tickers = st.text_input("Enter tickers (comma-separated):", "AAPL,MSFT,GOOGL,AMZN,TSLA,META")
n_stocks = st.slider("Number of stocks to select:", 3, 20, 5)

if st.button("Run Simulation"):
    tickers = [t.strip() for t in tickers.split(",")]

    st.write("Loading data...")
    price_df = yf.download(tickers, period="5y")["Adj Close"].dropna()

    # Fake market caps (for demo)
    market_caps = pd.Series({t: yf.Ticker(t).info.get("marketCap", 1e10) for t in tickers})

    # Calculate factors
    st.write("Calculating factors...")
    factors = calculate_factors(price_df, market_caps)
    st.dataframe(factors)

    # Build scores
    scores = build_factor_score(factors)

    # Select portfolio
    selected = select_stocks(scores, n=n_stocks)
    st.subheader("Selected Portfolio:")
    st.write(selected)

    # Backtest
    st.write("Running backtest...")
    ret, cumulative = backtest(price_df, selected)

    # Plot cumulative return
    fig = px.line(cumulative, title="Cumulative Returns")
    st.plotly_chart(fig)

    st.subheader("Performance Stats")
    st.write(f"Total return: {cumulative.iloc[-1] - 1:.2%}")
    st.write(f"Annual volatility: {ret.std() * (12 ** 0.5):.2%}")
    st.write(f"Sharpe ratio (approx): {ret.mean() / ret.std() * (12 ** 0.5):.2f}")
