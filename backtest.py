import pandas as pd

def backtest(price_df, stocks, rebalance_freq="M"):
    """
    Simple equal-weight backtest.
    price_df: dataframe of close prices
    stocks: list of chosen tickers
    """
    prices = price_df[stocks]

    # Monthly returns
    monthly_returns = prices.resample("M").last().pct_change().dropna()

    # Equal-weight portfolio
    portfolio_returns = monthly_returns.mean(axis=1)

    # Create cumulative returns
    cumulative = (1 + portfolio_returns).cumprod()

    return portfolio_returns, cumulative
