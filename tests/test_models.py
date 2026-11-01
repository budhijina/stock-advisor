"""Tests for the models + ranker + backtest + risk pipeline (synthetic data)."""
import numpy as np
import pandas as pd
import pytest

from stockforecaster import data, risk
from stockforecaster.backtest import walk_forward_backtest
from stockforecaster.features import FEATURE_COLUMNS, build_dataset, latest_rows
from stockforecaster.models import ModelEnsemble, historical_rates, statistical_signals
from stockforecaster.ranker import rank_universe, top_picks

TICKERS = ["AAPL", "NVDA", "TSLA", "AMD", "PLTR", "COIN", "JPM", "XOM"]


@pytest.fixture(scope="module")
def histories():
    return data.get_histories(TICKERS)


def test_dataset_shape(histories):
    ds = build_dataset(histories)
    assert {"ticker", "date", "target_ret", "target_up", *FEATURE_COLUMNS} <= set(ds.columns)
    assert len(ds) > 1000
    assert ds["target_up"].isin([0, 1]).all()


def test_model_fit_and_metrics(histories):
    ds = build_dataset(histories)
    model = ModelEnsemble().fit(ds)
    assert model.is_fitted
    assert model.metrics["n_test"] > 0
    assert 0 <= model.metrics["accuracy"] <= 1
    assert model.feature_importances


def test_predict_latest(histories):
    model = ModelEnsemble().fit(build_dataset(histories))
    latest = latest_rows(histories)
    pred = model.predict_latest(latest)
    assert len(pred) == len(latest)
    assert {"ticker", "pred_ret", "pred_up"} <= set(pred.columns)
    assert pred["pred_up"].between(0, 1).all()


def test_statistical_signals_scores():
    histories = data.get_histories(["AAPL"])
    lr = latest_rows(histories)
    sig = statistical_signals(lr.iloc[0])
    for k in ["momentum", "trend", "volume", "mean_reversion", "volatility"]:
        assert 0 <= sig["scores"][k] <= 100


def test_rank_universe_and_top_picks(histories):
    model = ModelEnsemble().fit(build_dataset(histories))
    ranked = rank_universe(histories, {}, model, build_dataset(histories))
    assert ranked["score"].is_monotonic_decreasing
    assert {"ticker", "expected_pct", "prob_up", "confidence", "stop_pct",
            "target_pct", "reasons", "recommendation", "predict_price"} <= set(ranked.columns)
    assert set(ranked["recommendation"]) <= {"BUY", "WATCH", "AVOID"}
    assert (ranked["predict_price"] > 0).all()
    picks = top_picks(ranked, n=5)
    assert len(picks) <= 5
    # ranks are 1..N
    assert list(ranked["rank"]) == list(range(1, len(ranked) + 1))


def test_backtest_runs(histories):
    bt = walk_forward_backtest(histories, top_n=4, oos_days=40)
    assert "error" not in bt
    assert bt["n_days"] > 5
    assert 0 <= bt["hit_rate"] <= 1
    assert "series" in bt and len(bt["series"]) == bt["n_days"]
    assert bt["monte_carlo"]["p05"] <= bt["monte_carlo"]["p50"] <= bt["monte_carlo"]["p95"]


def test_risk_plan():
    row = pd.Series({"Close": 100.0, "atr_pct": 2.5, "pred_up": 0.58, "ticker": "TEST"})
    plan = risk.plan_trade(row, capital=100_000.0)
    assert plan["stop"] < 100.0 < plan["target"]
    assert plan["risk_reward"] == pytest.approx(1.5, abs=0.01)
    assert plan["shares"] > 0
    assert plan["notional"] <= 100_000.0 * 0.10 + 1e-6


def test_portfolio_diversification():
    # 3 Tech + 4 Financials: the Financials cap (35% weight => max 3 names at 10%)
    # must drop the 4th Financial; Tech holds 3 names (30% <= 35%).
    from collections import Counter

    rows = []
    for i, t in enumerate(["AAPL", "MSFT", "NVDA", "JPM", "BAC", "GS", "COIN"]):
        rows.append(pd.Series({"Close": 100 + i, "atr_pct": 2.0,
                               "pred_up": 0.6, "pred_ret": 0.01, "score": 90 - i,
                               "ticker": t}))
    pf = risk.build_portfolio(pd.DataFrame(rows), capital=100_000.0)
    sectors = Counter(pf["sector"].values)
    assert sectors["Financials"] == 3        # 3 x 10% = 30% <= 35%
    assert sectors["Technology"] == 3
    assert len(pf) == 6
    assert pf["notional"].sum() <= 100_000.0 * 0.65 + 1e-6

