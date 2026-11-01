"""End-to-end workflow: everything an experienced investor does, automated.

Step 0  Universe definition + data collection (yfinance live, fallback synthetic)
Step 1  Liquidity / price screening ("don't trade illiquid junk")
Step 2  Market regime (SPY trend vs 50/200 DMA, VIX, risk-on gauge)
Step 3  Technical analysis (30+ indicators per ticker)
Step 4  Statistical signal scoring (momentum/trend/volume/mean-reversion/vol)
Step 5  News & sentiment scan (Google News RSS + financial lexicon, recency-weighted)
Step 6  Machine-learning prediction (walk-forward RF + GBM + classifier, calibrated)
Step 7  Ensemble ranking -> expected next-day % move + probability
Step 8  Risk plan (ATR stop/target, position size, Kelly) + portfolio construction
Step 9  Evidence: backtest (walk-forward) + Monte Carlo + base rates

`run()` returns a structured result dict consumed by both CLI and Streamlit.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from . import config
from . import data as market_data
from . import news as news_mod
from .backtest import walk_forward_backtest
from .models import ModelEnsemble, fit_ensemble, historical_rates
from .ranker import rank_universe, top_picks
from .risk import build_portfolio, expected_portfolio_stats


def run(tickers: Optional[list[str]] = None,
        top_n: int = config.TOP_N_DEFAULT,
        capital: float = 100_000.0,
        use_news: bool = True,
        demo_mode: bool = False,
        run_backtest: bool = True,
        progress_cb=None) -> dict[str, Any]:
    """Run the complete daily workflow. Returns a structured result dict."""
    tickers = tickers or config.DEFAULT_UNIVERSE

    # Step 1: data
    def _progress(i: int, n: int, t: str) -> None:
        if progress_cb:
            progress_cb(f"Fetching market data {i}/{n}: {t}")

    histories = market_data.get_histories(tickers, synthetic_ok=True)
    if progress_cb:
        progress_cb("Screening universe for liquidity...")
    passing, screen = market_data.screen_universe(histories)
    # Always include benchmark for regime/backtest reference
    bench = market_data.get_history(config.BENCHMARK, synthetic_ok=True)
    histories[config.BENCHMARK] = bench
    vix = market_data.get_history("^VIX", synthetic_ok=False)
    if vix is not None:
        histories["^VIX"] = vix
    regime = market_data.market_regime(bench, vix)

    # Step 2: features + ML
    if progress_cb:
        progress_cb("Building features and training walk-forward models...")
    model, dataset = fit_ensemble(histories)
    rates = historical_rates(dataset)

    # Step 3: news
    if use_news:
        if progress_cb:
            progress_cb(f"Scanning news (Google News RSS) for {len(passing)} names...")
        news_map = news_mod.news_signals(passing, synthetic=demo_mode)
    else:
        news_map = {}

    # Step 4-7: rank (benchmark & indices are never candidates)
    if progress_cb:
        progress_cb("Ranking universe (technical + statistical + ML + news)...")
    candidates = {t: d for t, d in histories.items()
                  if not t.startswith("^") and t != config.BENCHMARK}
    ranked = rank_universe(candidates, news_map, model, dataset, regime, capital)
    picks = top_picks(ranked, n=top_n)

    # Step 8: risk / portfolio
    portfolio = build_portfolio(picks, capital=capital) if not picks.empty else pd.DataFrame()
    port_stats = expected_portfolio_stats(portfolio)

    # Step 9: evidence / backtest
    bt = None
    if run_backtest:
        if progress_cb:
            progress_cb("Running walk-forward backtest + Monte Carlo...")
        bt = walk_forward_backtest(histories, top_n=top_n, oos_days=60)

    return {
        "tickers": tickers,
        "histories": histories,
        "screen": screen,
        "passing": passing,
        "regime": regime,
        "model": model,
        "dataset": dataset,
        "base_rates": rates,
        "news_map": news_map,
        "ranked": ranked,
        "picks": picks,
        "portfolio": portfolio,
        "portfolio_stats": port_stats,
        "backtest": bt,
        "source": market_data.get_provider(),  # 'yfinance' | 'stooq' | None
        "generated_at": pd.Timestamp.now(),
    }


def pick_explanations(picks: pd.DataFrame) -> list[dict[str, Any]]:
    """Human-readable explanation for each pick (for CLI/UI cards)."""
    out = []
    for _, r in picks.iterrows():
        out.append({
            "rank": int(r["rank"]),
            "ticker": r["ticker"],
            "score": r["score"],
            "expected_pct": r["expected_pct"],
            "prob_up": r["prob_up"],
            "confidence": r["confidence"],
            "price": r["price"],
            "stop_pct": r["stop_pct"],
            "target_pct": r["target_pct"],
            "risk_reward": r["risk_reward"],
            "reasons": r["reasons"],
        })
    return out
