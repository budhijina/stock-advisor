"""
Stock Forecaster — next-day momentum/trend prediction engine for day trading.

Modules
-------
data        : market data ingestion (yfinance) with caching + synthetic fallback
indicators  : 25+ industry-standard technical indicators (pure pandas)
features    : feature engineering + next-day return targets
news        : news collection (Google News RSS) + financial sentiment scoring
models      : statistical signals + walk-forward machine-learning models
ranker      : ensemble scoring -> top picks with expected % move
backtest    : walk-forward backtest engine + Monte Carlo analysis
risk        : ATR stops, position sizing, Kelly, portfolio construction
workflow    : end-to-end "experienced investor" pipeline
cli         : command-line interface (forecast / backtest)

Disclaimer: educational/research tool. Not financial advice.
"""

__version__ = "1.0.0"
