"""Composite ranker: turns signals into the "stocks likely to rise tomorrow" list.

Blends four industry-standard signal families into one 0-100 score using the
config.WEIGHTS ensemble, then converts the ML expectation (+ historical
calibration + news impact) into an expected next-day % move and a probability
of an up day, with explicit reasons and risk levels for every pick.

This is the module that answers: "which top stocks will go up next day — and by
what percentage?"
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from . import config
from .features import latest_rows
from .models import ModelEnsemble, historical_rates, statistical_signals
from .risk import plan_trade


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_z(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(50.0, index=series.index)
    return 50 + 30 * (series - series.mean()) / sd


def rank_universe(histories: dict[str, pd.DataFrame],
                  news_map: dict[str, dict[str, Any]] | None = None,
                  model: Optional[ModelEnsemble] = None,
                  dataset: Optional[pd.DataFrame] = None,
                  regime: Optional[dict] = None,
                  capital: float = 100_000.0) -> pd.DataFrame:
    """Rank every ticker in `histories` and return the full pick table.

    Columns: ticker, score, expected_pct, prob_up, confidence, price, reasons,
    flags, sector, component scores, news summary and risk plan.
    """
    news_map = news_map or {}
    regime = regime or {}
    latest = latest_rows(histories)
    if latest.empty:
        return pd.DataFrame()

    # Machine predictions (cross-sectionally normalized)
    ml: Optional[pd.DataFrame] = None
    base_rates = historical_rates(dataset) if dataset is not None else {
        "base_rate_up": 0.52, "up_after_up5": 0.52, "up_after_down5": 0.50}
    if model is not None and model.is_fitted:
        ml = model.predict_latest(latest)
        ml = ml.merge(latest[["ticker"]], on="ticker", how="right")
        ml["ml_norm"] = _norm_z(ml["pred_ret"].fillna(0.0))
    else:
        ml = pd.DataFrame({"ticker": latest["ticker"], "pred_ret": np.nan,
                           "pred_up": base_rates["base_rate_up"], "ml_norm": 50.0})

    # Out-of-sample top-quartile expected return (calibration anchor)
    oos_anchor = model._calibration.get("top_quartile_expected_ret", 0.0) \
        if model is not None else 0.0

    rows = []
    for _, row in latest.iterrows():
        t = row["ticker"]
        mrow = ml[ml["ticker"] == t].iloc[0]
        sig = statistical_signals(row)
        sc = sig["scores"]
        reasons = list(sig["reasons"])

        technical = np.mean([sc.get("momentum", 50), sc.get("trend", 50), sc.get("volume", 50)])
        statistical = np.mean(list(sc.values()))
        machine = float(mrow["ml_norm"])

        news = news_map.get(t, {})
        sent = float(news.get("score", 0.0))
        news_score = _clip(50 + 35 * sent, 0, 100)

        w = config.WEIGHTS
        composite = (w["technical"] * technical +
                     w["statistical"] * statistical +
                     w["machine"] * machine +
                     w["news"] * news_score)

        # ---- expected next-day move ------------------------------------------
        ml_ret = float(mrow["pred_ret"]) if pd.notna(mrow["pred_ret"]) else 0.0
        exp_move = 0.5 * ml_ret + 0.3 * oos_anchor + 0.2 * (sent * row["atr_pct"] / 100.0)
        exp_move = _clip(exp_move, -config.ML["clip_prediction"], config.ML["clip_prediction"])

        # ---- probability of an up day ----------------------------------------
        ml_prob = float(mrow["pred_up"]) if pd.notna(mrow["pred_up"]) else 0.52
        prob = 0.65 * ml_prob + 0.25 * base_rates.get("up_after_up5", 0.52) + 0.10 * (0.5 + sent * 0.5)
        risk_on = float(regime.get("risk_on", 0.5))
        prob = _clip(prob + (risk_on - 0.5) * 0.08, 0.25, 0.85)

        if prob >= config.CONFIDENCE_LEVELS["Strong"]:
            conf = "Strong"
        elif prob >= config.CONFIDENCE_LEVELS["Moderate"]:
            conf = "Moderate"
        elif prob >= config.CONFIDENCE_LEVELS["Weak"]:
            conf = "Weak"
        else:
            conf = "Low"

        plan_row = row.copy()
        plan_row["pred_up"] = prob
        plan = plan_trade(plan_row, capital=capital)

        flags = []
        if row["atr_pct"] > config.MAX_ATR_PCT * 100:
            flags.append("High volatility (>18% ATR) — size down")
        if sent < -0.3:
            flags.append("Negative news pressure")
        if row["vol_ratio"] > 4:
            flags.append("Volume spike (possible event) — confirm source")
        if row["near_high_20"] > 0.995:
            flags.append("Near 20-day high — breakout vs exhaustion watch")
        if row["gap_pct"] > 3:
            flags.append(f"Large overnight gap ({row['gap_pct']:.1f}%) — wait for open")
        if sc.get("mean_reversion", 0) >= 70 and sc.get("trend", 0) < 45:
            reasons.append("Counter-trend bounce setup (higher risk)")

        # ---- explicit recommendation (personal advisor mode) ----------------
        red_flags = any(("High volatility" in f) or ("Negative news" in f) for f in flags)
        if prob >= 0.60 and composite >= 65 and not red_flags:
            rec = "BUY"
        elif prob >= 0.54 and composite >= 55:
            rec = "WATCH"
        else:
            rec = "AVOID"

        if news.get("n_items", 0):
            reasons.append(f"News: {news['sentiment_label']} "
                           f"({news['n_items']} headlines, score {sent:+.2f})")
            if news.get("top_positive"):
                reasons.append(f"Top positive: {news['top_positive'][:80]}...")
            if news.get("top_negative"):
                reasons.append(f"Top negative: {news['top_negative'][:80]}...")
        else:
            reasons.append("No significant news in last 36h")
        if flags:
            reasons.append("WARN: " + "; ".join(flags))

        rows.append({
            "ticker": t,
            "Close": round(float(row["Close"]), 4),  # raw price (risk planner)
            "predict_price": round(float(row["Close"]) * (1 + exp_move), 2),  # next-day target price
            "pred_ret": exp_move,                     # decimal (portfolio stats)
            "pred_up": prob,                          # decimal (portfolio stats)
            "score": round(composite, 1),
            "expected_pct": round(exp_move * 100, 2),
            "prob_up": round(prob, 3),
            "confidence": conf,
            "price": round(float(row["Close"]), 2),
            "atr_pct": round(float(row["atr_pct"]), 2),
            "rsi14": round(float(row["rsi14"]), 1),
            "adx": round(float(row["adx"]), 1),
            "vol_ratio": round(float(row["vol_ratio"]), 2),
            "gap_pct": round(float(row["gap_pct"]), 2),
            "sector": config.SECTORS.get(t, "Other"),
            "tech_score": round(technical, 1),
            "stat_score": round(statistical, 1),
            "ml_score": round(machine, 1),
            "news_score": round(news_score, 1),
            "news_sentiment": round(sent, 3),
            "n_news": int(news.get("n_items", 0)),
            "news_label": news.get("sentiment_label", "no news"),
            "recommendation": rec,
            "stop_pct": round(plan["stop_pct"], 2),
            "target_pct": round(plan["target_pct"], 2),
            "risk_reward": plan["risk_reward"],
            "shares": plan["shares"],
            "flag": "; ".join(flags) if flags else "",
            "reasons": " | ".join(reasons[:6]),
        })

    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def top_picks(ranked: pd.DataFrame, n: int = config.TOP_N_DEFAULT,
              min_score: float = config.MIN_SCORE_TO_SHOW) -> pd.DataFrame:
    """Filter the ranked table to the actionable long candidates."""
    if ranked.empty:
        return ranked
    return ranked[(ranked["score"] >= min_score) & (ranked["expected_pct"] > 0)].head(n)

