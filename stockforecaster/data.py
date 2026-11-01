"""Market data layer.

Primary source : yfinance (free daily OHLCV).
Secondary      : Stooq free CSV endpoint (no key) when Yahoo is blocked.
Last resort    : deterministic synthetic price generator (GBM with regimes)
                 so the app keeps working fully offline.
Also computes the market-regime context (SPY trend vs 50/200 DMA, VIX level).
"""
from __future__ import annotations

import io
import os
import time
import warnings
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from . import config

warnings.filterwarnings("ignore")

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "cache")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(key: str) -> str:
    _ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"{key}.csv")


def _fresh(path: str, ttl_minutes: int = config.CACHE_TTL_MINUTES) -> bool:
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_minutes * 60


def load_cache(key: str) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if _fresh(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df
        except Exception:
            return None
    return None


def save_cache(key: str, df: pd.DataFrame) -> None:
    try:
        df.to_csv(_cache_path(key))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Synthetic fallback (deterministic per ticker, so results are reproducible)
# ---------------------------------------------------------------------------
def _hash_seed(text: str) -> int:
    return abs(hash(text)) % (2 ** 31)


def synthetic_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Generate a realistic OHLCV series with regimes, momentum, volume spikes.

    Deterministic: same ticker -> same series (seeded by ticker). Volatility and
    drift follow cyclical regimes; volume correlates with |return| (volume-price
    relationship typical of US equities).
    """
    rng = np.random.default_rng(_hash_seed(ticker))
    ndays = {"1y": 252, "6mo": 126, "3mo": 63, "1mo": 21, "5d": 5}.get(period, 252)
    price = float(rng.uniform(15, 400))
    base_vol = float(rng.uniform(0.012, 0.035))
    rows = []
    # Drift regime: a few trend cycles over the year
    t = np.arange(ndays)
    regime = np.sin(2 * np.pi * t / 60 + rng.uniform(0, 6.28))
    drift = 0.0015 * regime + rng.normal(0, 0.0004, ndays)

    for i in range(ndays):
        vol = base_vol * (1.0 + 0.6 * abs(np.sin(2 * np.pi * i / 21)))
        ret = drift[i] + rng.normal(0, vol)
        gap = rng.normal(0, vol * 0.35)
        o = price * (1 + gap)
        c = o * (1 + ret)
        hi = max(o, c) * (1 + abs(rng.normal(0, vol * 0.5)))
        lo = min(o, c) * (1 - abs(rng.normal(0, vol * 0.5)))
        volume = float(abs(rng.normal(1, 0.5)) * (1 + 6 * abs(ret)) * 1e6)
        rows.append((o, hi, lo, c, volume))
        price = c

    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])
    dates = pd.bdate_range(end=datetime.now().date(), periods=ndays)
    df.index = dates
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Real data (yfinance primary, Stooq free-CSV secondary)
# ---------------------------------------------------------------------------
def _fetch_yf(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        # yfinance inserts its own DeprecationWarning filter at the front of the
        # warnings stack; re-insert ours (after import) so its pandas-3 noise is silent.
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^yfinance")
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True,
                         progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        return df if len(df) >= 30 else None
    except Exception:
        return None


def _fetch_stooq(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Free fallback provider: Stooq daily CSV (https://stooq.com) — no API key.

    Symbol convention: `aapl.us`, `spy.us`, ... Indexes (^VIX) are not covered.
    NOTE: Stooq serves a JavaScript anti-bot challenge on some networks; on those
    it simply returns None and the next fallback (synthetic) is used.
    """
    try:
        if ticker.startswith("^"):
            return None
        symbol = ticker.lower().replace(".", "-") + ".us"
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        resp = requests.get(url, timeout=12,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200 or not resp.text.strip():
            return None
        df = pd.read_csv(io.StringIO(resp.text))
        if df is None or df.empty or "Close" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index.name = "Date"
        ndays = {"1y": 260, "6mo": 130, "3mo": 65, "1mo": 22, "5d": 6}.get(period, 260)
        df = df.tail(ndays)
        return df if len(df) >= 30 else None
    except Exception:
        return None


def get_history(ticker: str, period: str = "1y", allow_cache: bool = True,
                synthetic_ok: bool = True) -> pd.DataFrame:
    """Return OHLCV history for one ticker.

    Order: disk cache -> yfinance -> Stooq (free CSV) -> synthetic fallback.
    Real data is cached; synthetic fallback data is never cached so demo series
    can never mask live prices later.
    """
    key = f"hist_{ticker}_{period}"
    if allow_cache:
        cached = load_cache(key)
        if cached is not None and len(cached) >= 30:
            return cached

    want_live = live_available() or not synthetic_ok
    df = None
    if want_live:
        df = _fetch_yf(ticker, period)
        if df is None:
            df = _fetch_stooq(ticker, period)
    if df is None and synthetic_ok:
        df = synthetic_history(ticker, period)
    if df is not None and allow_cache and get_provider() is not None:
        save_cache(key, df)
    return df


def get_histories(tickers: list[str], period: str = "1y", synthetic_ok: bool = True,
                  progress_cb=None) -> dict[str, pd.DataFrame]:
    """Fetch history for many tickers, returning {ticker: DataFrame}.

    Returns only tickers with enough data. `synthetic_ok=False` requires real data.
    """
    out: dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers):
        df = get_history(t, period=period, synthetic_ok=synthetic_ok)
        if df is not None and len(df) >= min(config.TRAIN_DAYS, 60):
            out[t] = df
        if progress_cb:
            progress_cb(i + 1, len(tickers), t)
    return out


# ---------------------------------------------------------------------------
# Market regime
# ---------------------------------------------------------------------------
def market_regime(benchmark_df: Optional[pd.DataFrame] = None,
                  vix_df: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """Classify the market regime: trend direction, volatility level, risk appetite."""
    if benchmark_df is None or len(benchmark_df) < 200:
        return {"regime": "UNKNOWN", "trend": "flat", "volatility": "unknown",
                "risk_on": 0.5, "sma50_vs_sma200": 0.0}

    close = benchmark_df["Close"]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    last = close.iloc[-1]
    ret20 = close.pct_change(20).iloc[-1]
    vol20 = close.pct_change().rolling(20).std().iloc[-1]

    trend = "bullish" if last > sma50 > sma200 else (
        "bearish" if last < sma50 < sma200 else "mixed")

    if vix_df is not None and len(vix_df):
        vix = float(vix_df["Close"].iloc[-1])
        vol_level = "calm" if vix < 15 else ("normal" if vix < 22 else "elevated" if vix < 30 else "stress")
    else:
        vix = float("nan")
        vol_level = "normal" if vol20 > 0 else "unknown"

    # Risk-on proxy: positive 20d momentum of the benchmark and vol not elevated
    risk_on = 0.5 + (0.3 if ret20 > 0 else -0.3) + (0.2 if vol_level in ("calm", "normal") else -0.2)
    risk_on = float(np.clip(risk_on, 0.0, 1.0))

    return {
        "regime": f"{trend.upper()} / {vol_level.upper()}",
        "trend": trend,
        "volatility": vol_level,
        "vix": vix,
        "risk_on": risk_on,
        "sma50_vs_sma200": float(sma50 / sma200 - 1.0),
        "benchmark_return_20d": float(ret20),
        "benchmark_last": float(last),
    }


# ---------------------------------------------------------------------------
# Screeners
# ---------------------------------------------------------------------------
def screen_universe(histories: dict[str, pd.DataFrame]) -> tuple[list[str], pd.DataFrame]:
    """Apply liquidity/price filters -> (passing tickers, screening table)."""
    rows = []
    passing = []
    for t, df in histories.items():
        last = df.iloc[-1]
        price = float(last["Close"])
        avg_vol = float(df["Volume"].tail(20).mean())
        dollar_vol = avg_vol * price
        ok = (price >= config.MIN_PRICE and
              dollar_vol >= config.MIN_DOLLAR_VOLUME and
              avg_vol >= config.MIN_AVG_VOLUME)
        rows.append({"ticker": t, "price": price, "avg_dollar_volume": dollar_vol,
                     "avg_volume": avg_vol, "passes": ok})
        if ok:
            passing.append(t)
    screen = pd.DataFrame(rows).set_index("ticker").sort_values("avg_dollar_volume", ascending=False)
    return passing, screen


_PROBE = {"ts": 0.0, "provider": None}
_PROBE_TTL = 600.0


def _probe_provider(attempts: int = 2) -> Optional[str]:
    """Determine which real data provider is reachable (yfinance -> stooq).

    Uses a 3-month window so the probe returns enough rows to validate.
    """
    for _ in range(attempts):
        if _fetch_yf(config.BENCHMARK, "3mo") is not None:
            return "yfinance"
        if _fetch_stooq(config.BENCHMARK, "3mo") is not None:
            return "stooq"
        time.sleep(1.0)
    return None


def reset_live_probe() -> None:
    """Force a fresh provider check (used by the UI 'Retry live data' button)."""
    _PROBE.update(ts=0.0, provider=None)


def live_available(ttl: float = _PROBE_TTL) -> bool:
    return get_provider(ttl=ttl) is not None


def get_provider(ttl: float = _PROBE_TTL) -> Optional[str]:
    """Return the active real-data provider: 'yfinance' | 'stooq' | None (synthetic)."""
    now = time.time()
    if _PROBE["provider"] is not None and (now - _PROBE["ts"]) < ttl:
        return _PROBE["provider"]
    provider = _probe_provider()
    _PROBE.update(ts=now, provider=provider)
    return provider


