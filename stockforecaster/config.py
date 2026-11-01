"""Central configuration: universe, feature/model weights, risk thresholds.

Every tunable in the application lives here so the strategy can be adjusted
in one place (and the Methodology tab can surface it to the user).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Universe of liquid, actively-traded US stocks (S&P 100 core + active
# day-trading names). Users can override via CLI / UI.
# ---------------------------------------------------------------------------
DEFAULT_UNIVERSE: list[str] = [
    # Mega-cap tech / communication
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "NFLX",
    "AVGO", "ORCL", "CRM", "AMD", "INTC", "QCOM", "MU", "ARM",
    # High-beta / active day-trading names
    "PLTR", "COIN", "MSTR", "SMCI", "SNOW", "DDOG", "CRWD", "PANW",
    "HOOD", "SOFI", "RIVN", "LCID", "AFRM", "UPST", "DKNG", "CVNA",
    # Financials / energy / healthcare / staples
    "JPM", "BAC", "GS", "XOM", "CVX", "JNJ", "UNH", "LLY", "WMT",
    "KO", "PEP", "PG", "DIS", "BA", "CAT", "MCD", "NKE",
]

# ---------------------------------------------------------------------------
# Universe screening filters (what an experienced investor screens for)
# ---------------------------------------------------------------------------
MIN_PRICE = 5.0                   # avoid penny stocks
MIN_DOLLAR_VOLUME = 20_000_000.0  # avg daily $ traded (liquidity)
MIN_AVG_VOLUME = 500_000          # avg daily shares

# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------
HISTORY_PERIOD = "1y"             # yfinance window used for features + training
TRAIN_DAYS = 240                  # minimum history rows required to process
CACHE_TTL_MINUTES = 60            # re-fetch market data after 60 min
NEWS_LOOKBACK_HOURS = 36          # consider news from the last 36 hours
NEWS_PER_TICKER = 20              # max headlines fetched per ticker
NEWS_RECENCY_HALF_LIFE_H = 12     # exponential decay half-life for headlines

# ---------------------------------------------------------------------------
# Composite score weights (technical / statistical / machine-learning / news)
# These are the ensemble weights of the final ranking.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "technical": 0.30,     # trend + momentum indicators
    "statistical": 0.20,   # return distribution, mean reversion, volatility
    "machine": 0.30,       # walk-forward ML prediction (RF + GBM ensemble)
    "news": 0.20,          # headline sentiment + impact
}

# ---------------------------------------------------------------------------
# Signal thresholds
# ---------------------------------------------------------------------------
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
ADX_TREND_STRENGTH = 25.0           # ADX >= 25 => trending market
ATR_PERIOD = 14
MIN_ATR_PCT = 0.01                  # skip dead names with tiny ATR%
MAX_ATR_PCT = 0.18                  # skip lottery-ticket volatility
MIN_NEWS_SCORE = -1.0               # sentiment floor (items are dropped below)
NEWS_IMPACT_WEIGHT = 0.5            # how much a single strong headline moves score

# ---------------------------------------------------------------------------
# Machine-learning settings
# ---------------------------------------------------------------------------
ML = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 20,
    "test_size": 0.25,             # walk-forward: last 25% is out-of-sample
    "min_train_rows": 200,
    "random_state": 42,
    "clip_prediction": 0.12,       # clamp predicted next-day move to +/-12%
}

# ---------------------------------------------------------------------------
# Risk management (ATR stops, sizing, portfolio limits)
# ---------------------------------------------------------------------------
RISK = {
    "stop_atr_mult": 2.0,          # stop = 2 x ATR from entry
    "target_atr_mult": 3.0,        # take-profit = 3 x ATR
    "risk_per_trade_pct": 1.0,     # risk 1% of capital per position
    "max_position_pct": 10.0,      # no single position > 10% of capital
    "max_positions": 8,            # max concurrent positions
    "kelly_fraction": 0.5,         # half-Kelly for drawdown control
    "kelly_cap": 0.10,             # cap Kelly allocation at 10%
    "max_sector_exposure": 0.35,   # max 35% of book in one sector
}

# ---------------------------------------------------------------------------
# Ranking / output controls
# ---------------------------------------------------------------------------
TOP_N_DEFAULT = 10
MIN_SCORE_TO_SHOW = 40.0
CONFIDENCE_LEVELS = {              # prob_up interpretation
    "Strong": 0.65,
    "Moderate": 0.58,
    "Weak": 0.52,
}

# Map each ticker to a sector (used for diversification checks)
SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Communication", "META": "Communication", "AMZN": "Consumer",
    "TSLA": "Consumer", "NFLX": "Communication", "AVGO": "Technology",
    "ORCL": "Technology", "CRM": "Technology", "AMD": "Technology",
    "INTC": "Technology", "QCOM": "Technology", "MU": "Technology",
    "ARM": "Technology", "PLTR": "Technology", "COIN": "Financials",
    "MSTR": "Technology", "SMCI": "Technology", "SNOW": "Technology",
    "DDOG": "Technology", "CRWD": "Technology", "PANW": "Technology",
    "HOOD": "Financials", "SOFI": "Financials", "RIVN": "Consumer",
    "LCID": "Consumer", "AFRM": "Financials", "UPST": "Financials",
    "DKNG": "Consumer", "CVNA": "Consumer", "JPM": "Financials",
    "BAC": "Financials", "GS": "Financials", "XOM": "Energy",
    "CVX": "Energy", "JNJ": "Healthcare", "UNH": "Healthcare",
    "LLY": "Healthcare", "WMT": "Staples", "KO": "Staples",
    "PEP": "Staples", "PG": "Staples", "DIS": "Communication",
    "BA": "Industrials", "CAT": "Industrials", "MCD": "Consumer",
    "NKE": "Consumer",
}

BENCHMARK = "SPY"
RISK_FREE_RATE = 0.0   # daily risk-free approx (0% for short horizon)


