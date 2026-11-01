"""Walk-forward backtest engine.

Simulates the exact workflow an investor would have run day by day:
  1. On trading day t (close), use only data available up to and including t.
  2. Compute statistical signals + (periodically re-trained) ML predictions.
  3. Rank the universe, select the top-N names.
  4. Realize the next-day return (close t -> close t+1).

No look-ahead: features at date t contain only information up to t; the target
(target_ret) is strictly the NEXT day's return and is never used in ranking.
News is not simulated historically (free headline archives are not complete),
so the backtest covers the technical+statistical+ML layers only — this is
documented and surfaced in the UI.

Outputs: metrics (hit rate, expectancy, profit factor, Sharpe, max drawdown),
daily return series, benchmark comparison, and a Monte-Carlo bootstrap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_COLUMNS, build_dataset
from .models import ModelEnsemble, statistical_signals


def _rank_day(rows: pd.DataFrame) -> pd.DataFrame:
    """Score all tickers present on a single date and return the ranked table."""
    items = []
    for _, r in rows.iterrows():
        sig = statistical_signals(r)
        stats_score = np.mean(list(sig["scores"].values()))
        ml_score = float(r.get("pred_ret", 0.0))
        if pd.notna(ml_score) and len(rows) > 1:
            mu, sd = rows["pred_ret"].mean(), rows["pred_ret"].std(ddof=0)
            ml_norm = 50 + 30 * (ml_score - mu) / (sd + 1e-9) if sd > 0 else 50
        else:
            ml_norm = 50.0
        tech = np.mean([sig["scores"].get(k, 50) for k in
                        ("momentum", "trend", "volume", "mean_reversion")])
        # Backtest composite: technical + statistical + ML (news excluded)
        composite = 0.45 * tech + 0.20 * stats_score + 0.35 * ml_norm
        items.append({
            "ticker": r["ticker"], "date": r["date"],
            "tech_score": tech, "stats_score": stats_score,
            "ml_score": ml_score, "composite": composite,
            "target_ret": r["target_ret"],
        })
    return pd.DataFrame(items)


def walk_forward_backtest(histories: dict[str, pd.DataFrame],
                          top_n: int = config.TOP_N_DEFAULT,
                          oos_days: int = 60,
                          retrain_every: int = 20) -> dict:
    """Run the walk-forward backtest and return metrics + series + diagnostics."""
    ds = build_dataset(histories)
    ds = ds.dropna(subset=FEATURE_COLUMNS + ["target_ret"]).sort_values("date").reset_index(drop=True)
    if len(ds) < 200:
        return {"error": "not enough data for backtest"}
    dates = sorted(ds["date"].unique())
    if len(dates) < oos_days + 60:
        oos_days = max(10, len(dates) // 3)
    oos_dates = dates[-oos_days:]

    daily_returns: list[dict] = []
    model = ModelEnsemble()
    for i, d in enumerate(oos_dates):
        # Periodically retrain using ONLY rows with date < d (no look-ahead)
        if not model.is_fitted or (i % retrain_every) == 0:
            past = ds[ds["date"] < d]
            if len(past) >= config.ML["min_train_rows"]:
                model = ModelEnsemble().fit(past)
        day_rows = ds[ds["date"] == d].copy()
        if model.is_fitted:
            X = day_rows[FEATURE_COLUMNS].dropna()
            if len(X):
                rf, gb = model.regressors
                pred = np.clip((rf.predict(X) + gb.predict(X)) / 2,
                               -config.ML["clip_prediction"], config.ML["clip_prediction"])
                day_rows.loc[X.index, "pred_ret"] = pred
        day_rows = day_rows.dropna(subset=["pred_ret"]) if model.is_fitted else day_rows
        if len(day_rows) < 5:
            continue
        ranked = _rank_day(day_rows).sort_values("composite", ascending=False)
        picks = ranked.head(top_n)
        if not len(picks):
            continue
        daily_returns.append({
            "date": d,
            "n_picks": len(picks),
            "picks": list(picks["ticker"]),
            "return": float(picks["target_ret"].mean()),
            "hit": bool(picks["target_ret"].mean() > 0),
            "best": float(picks["target_ret"].max()),
            "worst": float(picks["target_ret"].min()),
        })

    if not daily_returns:
        return {"error": "walk-forward produced no days"}
    series = pd.DataFrame(daily_returns).set_index("date")
    return _summarize(series, histories, oos_days)


def _summarize(series: pd.DataFrame, histories: dict[str, pd.DataFrame],
               oos_days: int) -> dict:
    rets = series["return"].values
    n = len(rets)
    hit_rate = float((rets > 0).mean())
    avg_ret = float(rets.mean())
    total = float(np.prod(1 + rets) - 1)
    vol_d = float(rets.std(ddof=0))
    sharpe = float(avg_ret / vol_d * np.sqrt(252)) if vol_d > 0 else 0.0
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    max_dd = float((cum / peak - 1).min())
    wins = rets[rets > 0].sum()
    losses = -rets[rets < 0].sum()
    profit_factor = float(wins / losses) if losses > 0 else float("inf")
    avg_win = float(rets[rets > 0].mean()) if (rets > 0).any() else 0.0
    avg_loss = float(rets[rets < 0].mean()) if (rets < 0).any() else 0.0

    # Benchmark: buy & hold of the index over the same window
    bench = None
    bench_next = None
    if config.BENCHMARK in histories:
        bh = histories[config.BENCHMARK].loc[series.index[0]:series.index[-1]]
        if len(bh) > 1:
            bench = float(bh["Close"].iloc[-1] / bh["Close"].iloc[0] - 1)
            bench_next = float(bh["Close"].pct_change().iloc[1:].mean())

    # Monte Carlo bootstrap of the daily-return distribution
    rng = np.random.default_rng(42)
    n_sims = 5000
    draws = rng.choice(rets, size=(n_sims, n), replace=True)
    sim_totals = np.prod(1 + draws, axis=1) - 1
    mc = {
        "p50": float(np.quantile(sim_totals, 0.5)),
        "p05": float(np.quantile(sim_totals, 0.05)),
        "p95": float(np.quantile(sim_totals, 0.95)),
        "prob_positive": float((sim_totals > 0).mean()),
        "prob_beats_bench": float((sim_totals > (bench or 0)).mean()) if bench is not None else None,
    }

    return {
        "n_days": n,
        "n_picks_per_day": int(series["n_picks"].iloc[0]),
        "avg_return_per_day_pct": avg_ret * 100,
        "total_return_pct": total * 100,
        "hit_rate": hit_rate,
        "avg_win_pct": avg_win * 100,
        "avg_loss_pct": avg_loss * 100,
        "profit_factor": profit_factor,
        "expectancy_pct": avg_ret * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "benchmark_return_pct": (bench * 100) if bench is not None else None,
        "benchmark_avg_next_day_pct": (bench_next * 100) if bench_next is not None else None,
        "monte_carlo": mc,
        "series": series,
        "top_picks_last": list(series.iloc[-1]["picks"]),
    }


def monte_carlo_summary(daily_returns: np.ndarray, n_sims: int = 5000,
                        seed: int = 42) -> dict[str, float]:
    """Standalone bootstrap helper (used by the UI on chosen samples)."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(daily_returns, size=(n_sims, len(daily_returns)), replace=True)
    totals = np.prod(1 + draws, axis=1) - 1
    return {"p05": float(np.quantile(totals, 0.05)), "p50": float(np.quantile(totals, 0.5)),
            "p95": float(np.quantile(totals, 0.95)),
            "prob_positive": float((totals > 0).mean())}

