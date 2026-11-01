"""Technical indicators — pure pandas implementations (no TA-Lib dependency).

Implements the industry-standard catalogue used by technical analysts:

  Moving averages      : SMA, EMA, WMA, VWMA
  Trend / momentum     : MACD (12/26/9), ADX/DMI (14), Aroon, Ichimoku, ROC,
                         TRIX, TSI, linear-regression slope
  Oscillators          : RSI (Wilder), Stochastic %K/%D, Williams %R, CCI,
                         MFI (Money Flow Index), Ultimate Oscillator
  Volatility           : ATR, Bollinger Bands, Keltner Channels, historical vol
  Volume               : OBV, VWAP, Chaikin Money Flow, Force Index, VZO
  Price action         : Donchian channels, pivot points, gap detection,
                         candle patterns (engulfing, doji, hammer)

Every function returns a pandas Series/DataFrame aligned to the input index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1, dtype=float)
    return s.rolling(n, min_periods=n).apply(
        lambda x: float(np.dot(x, weights) / weights.sum()), raw=True)


def vwma(s: pd.Series, volume: pd.Series, n: int) -> pd.Series:
    """Volume-weighted moving average (rolling VWAP)."""
    return (s * volume).rolling(n, min_periods=n).sum() / volume.rolling(n, min_periods=n).sum()


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing). >70 overbought, <30 oversold."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_len: int = 14, d_len: int = 3, smooth: int = 3) -> pd.DataFrame:
    """Stochastic oscillator %K/%D (0-100)."""
    ll = low.rolling(k_len, min_periods=k_len).min()
    hh = high.rolling(k_len, min_periods=k_len).max()
    raw = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    k = raw.rolling(smooth, min_periods=1).mean()
    d = k.rolling(d_len, min_periods=1).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Williams %R (inverted stochastic, -100..0)."""
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    """Commodity Channel Index: >+100 overbought, <-100 oversold."""
    tp = (high + low + close) / 3
    ma = tp.rolling(n, min_periods=n).mean()
    md = tp.rolling(n, min_periods=n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
        n: int = 14) -> pd.Series:
    """Money Flow Index: volume-weighted RSI, 0-100."""
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    pos = raw_mf.where(tp > tp.shift(1), 0.0)
    neg = raw_mf.where(tp < tp.shift(1), 0.0)
    pos_s = pos.rolling(n, min_periods=n).sum()
    neg_s = neg.rolling(n, min_periods=n).sum()
    ratio = pos_s / neg_s.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50.0)


def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                        n1: int = 7, n2: int = 14, n3: int = 28) -> pd.Series:
    """Ultimate Oscillator (0-100); >70 overbought, <30 oversold."""
    pc = close.shift(1)
    bp = close - pd.concat([low, pc], axis=1).min(axis=1)
    tr = pd.concat([high, pc], axis=1).max(axis=1) - pd.concat([low, pc], axis=1).min(axis=1)
    def _avg(n):
        return bp.rolling(n, min_periods=n).sum() / tr.rolling(n, min_periods=n).sum().replace(0, np.nan)
    return 100 * (4 * _avg(n1) + 2 * _avg(n2) + _avg(n3)) / 7


# ---------------------------------------------------------------------------
# Trend strength
# ---------------------------------------------------------------------------
def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    """Average Directional Index (+DI/-DI/ADX). ADX>25 => trending, ADX<20 => ranging."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_})


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def aroon(high: pd.Series, low: pd.Series, n: int = 25) -> pd.DataFrame:
    """Aroon up/down/oscillator (0-100); oscillator >0 => uptrend."""
    up = 100 * high.rolling(n + 1, min_periods=n + 1).apply(lambda x: float(np.argmax(x)) / n, raw=True)
    down = 100 * low.rolling(n + 1, min_periods=n + 1).apply(lambda x: float(np.argmin(x)) / n, raw=True)
    return pd.DataFrame({"aroon_up": up, "aroon_down": down, "aroon_osc": up - down})


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
             tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> pd.DataFrame:
    """Ichimoku cloud components."""
    tenkan_sen = (high.rolling(tenkan, min_periods=tenkan).max() +
                  low.rolling(tenkan, min_periods=tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun, min_periods=kijun).max() +
                 low.rolling(kijun, min_periods=kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b = ((high.rolling(senkou_b, min_periods=senkou_b).max() +
                 low.rolling(senkou_b, min_periods=senkou_b).min()) / 2).shift(kijun)
    chikou = close.shift(-kijun)
    return pd.DataFrame({"tenkan": tenkan_sen, "kijun": kijun_sen,
                         "senkou_a": senkou_a, "senkou_b": senkou_b, "chikou": chikou})


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    return true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands + %B + bandwidth."""
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
                         "bb_pct_b": pct_b, "bb_width": bandwidth})


def keltner(high: pd.Series, low: pd.Series, close: pd.Series,
            n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Keltner Channels (EMA + ATR bands)."""
    mid = ema(close, n)
    atr_ = atr(high, low, close, 10)
    return pd.DataFrame({"kc_mid": mid, "kc_upper": mid + k * atr_, "kc_lower": mid - k * atr_})


def historical_volatility(close: pd.Series, n: int = 20, annualize: bool = True) -> pd.Series:
    ret = np.log(close / close.shift(1))
    vol = ret.rolling(n, min_periods=n).std(ddof=0)
    return vol * np.sqrt(252) if annualize else vol


# ---------------------------------------------------------------------------
# Volume indicators
# ---------------------------------------------------------------------------
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (cumulative volume signed by price direction)."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
         n: int = 14) -> pd.Series:
    """Rolling VWAP (approximation of session anchor on daily resolution)."""
    tp = (high + low + close) / 3
    return (tp * volume).rolling(n, min_periods=n).sum() / volume.rolling(n, min_periods=n).sum().replace(0, np.nan)


def chaikin_mf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
               n: int = 20) -> pd.Series:
    """Chaikin Money Flow. >0 accumulation, <0 distribution."""
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * volume
    return mfv.rolling(n, min_periods=n).sum() / volume.rolling(n, min_periods=n).sum().replace(0, np.nan)


def force_index(close: pd.Series, volume: pd.Series, n: int = 13) -> pd.Series:
    """Force Index: volume-weighted momentum, smoothed with EMA."""
    return (close.diff() * volume).ewm(span=n, adjust=False, min_periods=n).mean()


def volume_zone(close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
    """Volume Zone Oscillator (-100..100): confirms volume in up vs down days."""
    up_vol = volume.where(close.diff() > 0, 0.0).rolling(n, min_periods=n).sum()
    dn_vol = volume.where(close.diff() < 0, 0.0).rolling(n, min_periods=n).sum()
    return 100 * (up_vol - dn_vol) / (up_vol + dn_vol).replace(0, np.nan)


# ---------------------------------------------------------------------------
# Price action
# ---------------------------------------------------------------------------
def donchian(high: pd.Series, low: pd.Series, n: int = 20) -> pd.DataFrame:
    """Donchian channels (highest high / lowest low / mid)."""
    upper = high.rolling(n, min_periods=n).max()
    lower = low.rolling(n, min_periods=n).min()
    return pd.DataFrame({"dc_upper": upper, "dc_lower": lower, "dc_mid": (upper + lower) / 2})


def pivots(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Classic daily pivot points (P/S1/S2/R1/R2)."""
    p = (high + low + close) / 3
    return pd.DataFrame({
        "pivot_p": p,
        "pivot_s1": 2 * p - high,
        "pivot_s2": p - (high - low),
        "pivot_r1": 2 * p - low,
        "pivot_r2": p + (high - low),
    })


def gap_pct(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Overnight gap % (open vs prior close)."""
    return (open_ / close.shift(1) - 1.0) * 100


def candle_patterns(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Boolean flags for common candle patterns."""
    body = (close - open_).abs()
    rng = (high - low).replace(0, np.nan)
    upper = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower = pd.concat([open_, close], axis=1).min(axis=1) - low

    prev_o, prev_c = open_.shift(1), close.shift(1)
    bullish_engulf = (close > open_) & (prev_c < prev_o) & \
                     (open_ <= prev_c) & (close >= prev_o)
    bearish_engulf = (close < open_) & (prev_c > prev_o) & \
                     (open_ >= prev_c) & (close <= prev_o)
    doji = (body / rng < 0.1) & (rng > 0)
    hammer = (lower >= 2 * body) & (upper <= body)
    shooting_star = (upper >= 2 * body) & (lower <= body)
    return pd.DataFrame({
        "bullish_engulf": bullish_engulf, "bearish_engulf": bearish_engulf,
        "doji": doji, "hammer": hammer, "shooting_star": shooting_star,
    })


# ---------------------------------------------------------------------------
# Composite: compute the full indicator table for one ticker
# ---------------------------------------------------------------------------
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach every indicator column to an OHLCV DataFrame (in place copy)."""
    out = df.copy()
    c, h, l, o_, v = out["Close"], out["High"], out["Low"], out["Open"], out["Volume"]

    # Trend
    out["sma10"], out["sma20"], out["sma50"], out["sma200"] = (
        sma(c, 10), sma(c, 20), sma(c, 50), sma(c, 200))
    out["ema12"], out["ema26"] = ema(c, 12), ema(c, 26)
    out["sar_like_hl2"] = (h + l + c) / 3  # typical price

    macd_df = macd(c)
    out = out.join(macd_df)
    adx_df = adx(h, l, c)
    out = out.join(adx_df)
    aroon_df = aroon(h, l)
    out = out.join(aroon_df)
    ichi = ichimoku(h, l, c)
    out = out.join(ichi)

    # Oscillators
    out["rsi14"] = rsi(c, 14)
    out["rsi7"] = rsi(c, 7)
    stoch_df = stochastic(h, l, c)
    out = out.join(stoch_df)
    out["williams_r14"] = williams_r(h, l, c, 14)
    out["cci20"] = cci(h, l, c, 20)
    out["mfi14"] = mfi(h, l, c, v, 14)
    out["uo"] = ultimate_oscillator(h, l, c)

    # Volatility
    out["atr14"] = atr(h, l, c, 14)
    out["atr_pct"] = out["atr14"] / c * 100
    bb = bollinger(c)
    out = out.join(bb)
    kc = keltner(h, l, c)
    out = out.join(kc)
    out["hv20"] = historical_volatility(c, 20)
    out["vol20"] = c.pct_change().rolling(20, min_periods=20).std(ddof=0)

    # Volume
    out["obv"] = obv(c, v)
    out["obv_slope20"] = out["obv"].diff(20)
    out["vwap14"] = vwap(h, l, c, v, 14)
    out["cmf20"] = chaikin_mf(h, l, c, v, 20)
    out["force13"] = force_index(c, v, 13)
    out["vzo14"] = volume_zone(c, v, 14)
    out["vol_ratio"] = v / v.rolling(20, min_periods=20).mean()
    out["vol_ma20"] = v.rolling(20, min_periods=20).mean()

    # Price action
    dc = donchian(h, l, 20)
    out = out.join(dc)
    pv = pivots(h, l, c)
    out = out.join(pv)
    out["gap_pct"] = gap_pct(o_, c)
    pat = candle_patterns(o_, h, l, c)
    out = out.join(pat)

    # Returns / statistical
    out["ret_1d"] = c.pct_change()
    out["ret_5d"] = c.pct_change(5)
    out["ret_10d"] = c.pct_change(10)
    out["ret_20d"] = c.pct_change(20)
    out["ret_60d"] = c.pct_change(60)
    out["log_ret"] = np.log(c / c.shift(1))
    out["zscore_20"] = (c - sma(c, 20)) / c.rolling(20, min_periods=20).std(ddof=0)
    out["hl_range_pct"] = (h - l) / c * 100
    out["intraday_pos"] = (c - l) / (h - l).replace(0, np.nan)
    out["dist_sma20_pct"] = (c / out["sma20"] - 1) * 100
    out["dist_sma50_pct"] = (c / out["sma50"] - 1) * 100
    out["dist_sma200_pct"] = (c / out["sma200"] - 1) * 100
    out["near_high_20"] = c / dc["dc_upper"]  # 1.0 => at 20d high

    return out


__all__ = [
    "sma", "ema", "wma", "vwma", "rsi", "stochastic", "williams_r", "cci",
    "mfi", "ultimate_oscillator", "adx", "macd", "aroon", "ichimoku",
    "true_range", "atr", "bollinger", "keltner", "historical_volatility",
    "obv", "vwap", "chaikin_mf", "force_index", "volume_zone",
    "donchian", "pivots", "gap_pct", "candle_patterns", "add_all_indicators",
]





