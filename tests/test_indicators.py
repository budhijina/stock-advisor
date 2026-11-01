"""Tests for the technical-indicator library (pure pandas)."""
import numpy as np
import pandas as pd
import pytest

from stockforecaster import indicators as ta
from stockforecaster.data import synthetic_history


@pytest.fixture(scope="module")
def df():
    return synthetic_history("TEST", "1y")


def test_rsi_bounds(df):
    r = ta.rsi(df["Close"], 14)
    assert r.between(0, 100).all()
    assert 0 < r.iloc[-1] < 100


def test_macd_structure(df):
    m = ta.macd(df["Close"])
    assert list(m.columns) == ["macd", "macd_signal", "macd_hist"]
    assert m["macd_hist"].notna().any()
    # hist = macd - signal
    assert np.allclose(m["macd_hist"].dropna(), (m["macd"] - m["macd_signal"]).dropna(), atol=1e-9)


def test_atr_positive(df):
    a = ta.atr(df["High"], df["Low"], df["Close"], 14)
    assert (a.dropna() > 0).all()


def test_bollinger(df):
    b = ta.bollinger(df["Close"])
    assert (b["bb_upper"].dropna() >= b["bb_mid"].dropna()).all()
    assert (b["bb_mid"].dropna() >= b["bb_lower"].dropna()).all()
    assert b["bb_pct_b"].dropna().between(-1.0, 2.0).all()


def test_stochastic_range(df):
    s = ta.stochastic(df["High"], df["Low"], df["Close"])
    assert s["stoch_k"].dropna().between(0, 100).all()
    assert s["stoch_d"].dropna().between(0, 100).all()


def test_adx_positive(df):
    a = ta.adx(df["High"], df["Low"], df["Close"], 14)
    assert (a["adx"].dropna() >= 0).all()


def test_obv_signs(df):
    o = ta.obv(df["Close"], df["Volume"])
    assert len(o) == len(df)
    obv_diff = o.diff().dropna()
    close_diff = df["Close"].diff().loc[obv_diff.index]
    assert (np.sign(obv_diff) == np.sign(close_diff)).all()


def test_add_all_indicators_columns(df):
    out = ta.add_all_indicators(df)
    for col in ["rsi14", "macd_hist", "adx", "bb_pct_b", "atr_pct", "obv",
                "gap_pct", "zscore_20", "near_high_20", "bullish_engulf"]:
        assert col in out.columns, f"missing {col}"
    assert len(out) == len(df)


def test_gap_definition(df):
    g = ta.gap_pct(df["Open"], df["Close"])
    expected = (df["Open"].iloc[1] / df["Close"].iloc[0] - 1) * 100
    assert abs(g.iloc[1] - expected) < 1e-9
