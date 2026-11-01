"""Feature engineering for the machine-learning layer.

Creates the supervised dataset used by the walk-forward models:
  - features derived from the technical-indicator table (confined to each ticker)
  - target: next-day simple return (regression) and up/down (classification)
  - row identity: (ticker, date)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_all_indicators

# Final feature ordering (cross-sectionally comparable, normalized)
FEATURE_COLUMNS = [
    # Trend
    "dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct",
    "macd_hist_pct", "adx", "plus_di_minus_minus", "aroon_osc",
    "tenkan_kijun", "close_vs_senkou_a",
    # Momentum
    "rsi14", "rsi7", "stoch_k", "stoch_k_minus_d", "williams_r14",
    "cci20_norm", "mfi14", "uo",
    # Volatility / mean reversion
    "bb_pct_b", "bb_width", "atr_pct", "hv20", "zscore_20",
    "dist_bb_mid_pct", "vol_ratio",
    # Volume / money flow
    "cmf20", "vzo14", "obv_slope_rel", "force13_rel", "vwap_dist_pct",
    # Price action
    "gap_pct", "hl_range_pct", "intraday_pos", "near_high_20",
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "vol20", "candle_score",
]


def _candle_score(row: pd.Series) -> float:
    s = 0.0
    if bool(row.get("bullish_engulf", False)):
        s += 2.0
    if bool(row.get("hammer", False)):
        s += 1.0
    if bool(row.get("bearish_engulf", False)):
        s -= 2.0
    if bool(row.get("shooting_star", False)):
        s -= 1.0
    return s


def _candle_score_vec(ind: pd.DataFrame) -> pd.Series:
    """Vectorized candle pattern score (much faster than row-wise apply)."""
    return (ind["bullish_engulf"].astype(int) * 2.0
            + ind["hammer"].astype(int) * 1.0
            - ind["bearish_engulf"].astype(int) * 2.0
            - ind["shooting_star"].astype(int) * 1.0)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert one ticker's OHLCV history into a row-per-day feature frame.

    Returns a DataFrame indexed by date with all FEATURE_COLUMNS + target
    columns: `target_ret` (next-day return), `target_up` (0/1).
    """
    ind = add_all_indicators(df)

    # Normalize scale-dependent indicators to percentages
    ind["macd_hist_pct"] = ind["macd_hist"] / ind["Close"] * 100
    ind["plus_di_minus_minus"] = ind["plus_di"] - ind["minus_di"]
    ind["tenkan_kijun"] = (ind["tenkan"] - ind["kijun"]) / ind["Close"] * 100
    ind["close_vs_senkou_a"] = (ind["Close"] - ind["senkou_a"]) / ind["Close"] * 100
    ind["cci20_norm"] = (ind["cci20"] / 100.0).clip(-3, 3)
    ind["stoch_k_minus_d"] = ind["stoch_k"] - ind["stoch_d"]
    ind["dist_bb_mid_pct"] = (ind["Close"] - ind["bb_mid"]) / ind["bb_mid"] * 100
    ind["obv_slope_rel"] = ind["obv_slope20"] / ind["vol_ma20"].replace(0, np.nan)
    ind["force13_rel"] = ind["force13"] / (ind["vol_ma20"] * ind["Close"]).replace(0, np.nan)
    ind["vwap_dist_pct"] = (ind["Close"] - ind["vwap14"]) / ind["vwap14"] * 100
    ind["candle_score"] = _candle_score_vec(ind)

    out = ind[FEATURE_COLUMNS].copy()
    out["target_ret"] = ind["Close"].shift(-1) / ind["Close"] - 1.0
    out["target_up"] = (out["target_ret"] > 0).astype(int)
    out["Close"] = ind["Close"]  # raw price for risk planning / display
    out.attrs["ticker"] = getattr(df, "attrs", {}).get("ticker", "?")
    return out


def build_dataset(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack all ticker feature frames into one supervised dataset.

    Columns: ticker, date, features..., target_ret, target_up.
    Features are winsorized per column to +-4 sigma for ML robustness.
    """
    frames = []
    for ticker, df in histories.items():
        ff = build_feature_frame(df.copy())
        ff["ticker"] = ticker
        ff["date"] = ff.index
        frames.append(ff.reset_index(drop=True))

    if not frames:
        return pd.DataFrame(columns=["ticker", "date"] + FEATURE_COLUMNS +
                                     ["target_ret", "target_up"])
    ds = pd.concat(frames, ignore_index=True)

    # Winsorize per column (pooled across tickers) to dampen outliers
    for col in FEATURE_COLUMNS:
        s = ds[col]
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        ds[col] = s.clip(lo, hi)
    ds = ds.sort_values(["date", "ticker"]).reset_index(drop=True)
    return ds


def latest_rows(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per ticker (the most recent day) for scoring today."""
    rows = []
    for ticker, df in histories.items():
        ff = build_feature_frame(df.copy())
        last = ff.iloc[-1].copy()
        last["ticker"] = ticker
        last["date"] = ff.index[-1]
        rows.append(last)
    if not rows:
        return pd.DataFrame(columns=["ticker", "date"] + FEATURE_COLUMNS)
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out
