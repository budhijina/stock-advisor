"""Risk management for day-trading positions.

Implements the discipline an experienced trader applies after the signal:
  - ATR-based stop-loss and take-profit levels
  - Risk-per-trade position sizing (volatility-scaled)
  - Half-Kelly optimal sizing (capped) using the model's win-rate & payoff
  - Portfolio construction: position count cap, 10% single-name cap,
    sector exposure cap
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from . import config


def plan_trade(row: pd.Series, capital: float = 100_000.0,
               risk_pct: Optional[float] = None,
               atr_value: Optional[float] = None) -> dict[str, Any]:
    """Build an execution plan for one ticker from its latest features.

    `row` should contain at least: Close, atr14, pred_ret, pred_up, ticker.
    """
    cfg = config.RISK
    risk_pct = cfg["risk_per_trade_pct"] if risk_pct is None else risk_pct
    entry = float(row["Close"])
    atr_val = float(atr_value) if atr_value is not None else (
        float(row.get("atr14", np.nan)) if pd.notna(row.get("atr14", np.nan))
        else entry * float(row.get("atr_pct", 3.0)) / 100.0)
    if not atr_val or atr_val <= 0:
        atr_val = entry * 0.03

    stop_dist = cfg["stop_atr_mult"] * atr_val                # dollars
    target_dist = cfg["target_atr_mult"] * atr_val
    stop = entry - stop_dist
    target = entry + target_dist

    risk_usd = capital * (risk_pct / 100.0)
    shares = risk_usd / stop_dist if stop_dist > 0 else 0.0
    shares = min(shares, (capital * cfg["max_position_pct"] / 100.0) / entry)
    notional = shares * entry

    # Half-Kelly sizing: f* = p - q/b ; use historical p from model, payoff b = target/stop
    p = float(row.get("pred_up", 0.5))
    b = target_dist / stop_dist if stop_dist > 0 else 1.5
    kelly = max(0.0, p - (1 - p) / b) * cfg["kelly_fraction"]
    kelly = min(kelly, cfg["kelly_cap"])

    rr = round(target_dist / stop_dist, 2) if stop_dist else float("nan")
    return {
        "ticker": row.get("ticker", "?"),
        "entry": entry,
        "stop": stop,
        "target": target,
        "stop_pct": stop_dist / entry * 100,
        "target_pct": target_dist / entry * 100,
        "risk_per_share": stop_dist,
        "reward_per_share": target_dist,
        "risk_reward": rr,
        "shares": int(shares),
        "notional": notional,
        "risked_usd": risk_usd,
        "kelly_fraction": kelly,
        "suggested_weight_pct": min(notional / capital * 100 if capital else 0.0, cfg["max_position_pct"]),
    }


def build_portfolio(picks: pd.DataFrame, capital: float = 100_000.0,
                    max_positions: Optional[int] = None) -> pd.DataFrame:
    """Diversify a ranked pick-list into a final portfolio of trades.

    Constraints (config.RISK):
      - max N concurrent positions (default 8)
      - single-name weight <= max_position_pct
      - sector exposure <= max_sector_exposure (deduplicate names same sector)
    """
    cfg = config.RISK
    mx = max_positions or cfg["max_positions"]
    rows = []
    sector_w: dict[str, float] = {}
    taken = 0
    for _, r in picks.iterrows():
        if taken >= mx:
            break
        ticker = r["ticker"]
        sector = config.SECTORS.get(ticker, "Other")
        plan = plan_trade(r, capital=capital)
        w_pct = plan["suggested_weight_pct"]
        if w_pct <= 0:
            continue
        if sector_w.get(sector, 0.0) + w_pct > cfg["max_sector_exposure"] * 100:
            continue
        sector_w[sector] = sector_w.get(sector, 0.0) + w_pct
        rows.append({**r.to_dict(), **plan, "sector": sector})
        taken += 1
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def expected_portfolio_stats(portfolio: pd.DataFrame) -> dict[str, float]:
    """Aggregate expected risk/reward of the constructed portfolio."""
    if portfolio.empty:
        return {"expected_return_pct": 0.0, "avg_prob_up": 0.0,
                "total_weight_pct": 0.0, "n_positions": 0}
    avg_ret = float(portfolio["pred_ret"].mean())
    avg_prob = float(portfolio["pred_up"].mean())
    total_w = float(portfolio["notional"].sum())
    return {
        "expected_return_pct": avg_ret * 100,
        "avg_prob_up": avg_prob,
        "total_weight_pct": total_w,  # capital-normalized later by caller
        "n_positions": int(len(portfolio)),
        "avg_rr": float(portfolio["risk_reward"].mean()),
        "avg_stop_pct": float(portfolio["stop_pct"].mean()),
        "avg_target_pct": float(portfolio["target_pct"].mean()),
    }
