"""Stock Forecaster — Streamlit dashboard.

Run:  .venv/bin/streamlit run app.py
Tabs: Top Picks | Market Overview | Deep Dive | Screener | Backtest | News | Methodology

The whole experience-layer wraps `stockforecaster.workflow.run()` so the CLI and
the dashboard always use the identical pipeline.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stockforecaster import config                      # noqa: E402
from stockforecaster import data as market_data          # noqa: E402
from stockforecaster import news as news_mod             # noqa: E402
from stockforecaster.backtest import walk_forward_backtest  # noqa: E402
from stockforecaster.indicators import add_all_indicators  # noqa: E402
from stockforecaster.workflow import run                 # noqa: E402

st.set_page_config(page_title="Personal Stock Advisor — Next-Day Signals",
                   page_icon="🎯", layout="wide")

GREEN, RED, GRAY = "#26a69a", "#ef5350", "#78909c"


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def candlestick_chart(df: pd.DataFrame, title: str) -> go.Figure:
    ind = add_all_indicators(df)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25],
                        vertical_spacing=0.05, subplot_titles=(title, "RSI (14)", "Volume"))
    fig.add_trace(go.Candlestick(x=ind.index, open=ind["Open"], high=ind["High"],
                                 low=ind["Low"], close=ind["Close"], name="OHLC",
                                 increasing_line_color=GREEN, decreasing_line_color=RED), row=1, col=1)
    for col, name, color in [("sma20", "SMA20", "#ff9800"), ("sma50", "SMA50", "#2196f3"),
                             ("sma200", "SMA200", "#9c27b0")]:
        fig.add_trace(go.Scatter(x=ind.index, y=ind[col], name=name, mode="lines",
                                 line=dict(width=1.2, color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["bb_upper"], name="BB upper",
                             line=dict(width=0.8, color=GRAY, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["bb_lower"], name="BB lower",
                             line=dict(width=0.8, color=GRAY, dash="dot"), fill="tonexty",
                             fillcolor="rgba(120,144,156,0.08)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["rsi14"], name="RSI", line=dict(color="#ff5722")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=RED, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=GREEN, row=2, col=1)
    fig.add_trace(go.Bar(x=ind.index, y=ind["Volume"], name="Volume",
                         marker_color=np.where(ind["Close"] >= ind["Open"], GREEN, RED)), row=3, col=1)
    fig.update_layout(height=720, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02), margin=dict(l=40, r=20, t=60, b=20))
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def stock_vs_bench(df: pd.DataFrame, bench: pd.DataFrame, title: str) -> go.Figure:
    a = df["Close"].pct_change().dropna()
    b = bench["Close"].pct_change().dropna()
    idx = a.index.intersection(b.index)
    ca = (1 + a.loc[idx]).cumprod() * 100
    cb = (1 + b.loc[idx]).cumprod() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ca.index, y=ca, name=title, line=dict(color="#2196f3")))
    fig.add_trace(go.Scatter(x=cb.index, y=cb, name="SPY", line=dict(color="#ff9800")))
    fig.update_layout(title="Relative performance (rebased to 100)", height=380,
                      margin=dict(l=40, r=20, t=50, b=20))
    return fig


def gauge(value: float, title: str, lo: float = 0, hi: float = 100) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value,
                                 title={"text": title},
                                 gauge={"axis": {"range": [lo, hi]},
                                        "bar": {"color": "#2196f3"},
                                        "steps": [{"range": [lo, hi * 0.4], "color": "#ef5350"},
                                                  {"range": [hi * 0.4, hi * 0.6], "color": "#ffecb3"},
                                                  {"range": [hi * 0.6, hi], "color": "#c8e6c9"}]}))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, val) in zip(cols, items):
        col.metric(label, val)


# ---------------------------------------------------------------------------
# Lightweight table heat-coloring (no matplotlib dependency)
# ---------------------------------------------------------------------------
def _blend(v: float, lo: float, hi: float,
           lo_color: str = "#ef5350", hi_color: str = "#26a69a") -> str:
    t = 0.0 if hi == lo else (float(v) - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    lr, lg, lb = (int(lo_color[i:i + 2], 16) for i in (1, 3, 5))
    hr, hg, hb = (int(hi_color[i:i + 2], 16) for i in (1, 3, 5))
    return (f"#{int(lr + (hr - lr) * t):02x}{int(lg + (hg - lg) * t):02x}"
            f"{int(lb + (hb - lb) * t):02x}")


def heat_styler(df: pd.DataFrame, cols: list[str], fmt: dict | None = None,
                arrow_cols: list[str] | None = None) -> "Styler":
    """Number heat-coloring + optional green/red arrow coloring for text columns."""
    st = df.style.format(fmt or {})
    for c in cols:
        s = df[c].astype(float)
        lo, hi = float(s.min()), float(s.max())
        st = st.map(lambda v, lo=lo, hi=hi: f"color: {_blend(v, lo, hi)}", subset=[c])
    for c in (arrow_cols or []):
        st = st.map(lambda v: ("color: #26a69a" if str(v).startswith("▲")
                               else "color: #ef5350" if str(v).startswith("▼")
                               else f"color: {GRAY}"), subset=[c])
    return st


# ---------------------------------------------------------------------------
# Sidebar / data loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def _cached_run(tickers: tuple[str, ...], top_n: int, capital: float,
                use_news: bool, demo: bool) -> dict:
    return run(list(tickers), top_n=top_n, capital=capital,
               use_news=use_news, demo_mode=demo, run_backtest=True)


st.sidebar.title("🎯 Personal Stock Advisor")
st.sidebar.caption("Next-day signals, risk plans & evidence — for my own trading")

universe_default = config.DEFAULT_UNIVERSE
tickers_sel = st.sidebar.multiselect(
    "Universe (tickers)", sorted(set(universe_default)),
    default=universe_default[:24], key="universe_sel")
top_n = st.sidebar.slider("Number of top picks", 3, 25, config.TOP_N_DEFAULT)
capital = st.sidebar.number_input("Trading capital ($)", 10_000.0, 5_000_000.0,
                                  100_000.0, step=10_000.0)
use_news = st.sidebar.checkbox("Include news sentiment", value=True)
demo = st.sidebar.checkbox("Demo news (offline)", value=False,
                           help="Use synthetic headlines instead of Google News RSS")
run_clicked = st.sidebar.button("🚀 Run forecast", type="primary", use_container_width=True)
retry_live = st.sidebar.button("🔄 Retry live data", use_container_width=True,
                               help="Re-check yfinance / Stooq and re-run with real data "
                                    "if they are reachable")

# Defensive provider lookup: if the running process has a stale older data module
# in memory (Streamlit reloads app.py but not already-imported modules), fall back
# gracefully instead of crashing with AttributeError.
def _provider() -> str | None:
    fn = getattr(market_data, "get_provider", None)
    if callable(fn):
        return fn()
    try:
        return "yfinance" if market_data.live_available() else None
    except Exception:
        return None


def _reset_probe() -> None:
    fn = getattr(market_data, "reset_live_probe", None)
    if callable(fn):
        fn()


provider = _provider()
st.sidebar.divider()
if provider:
    st.sidebar.markdown(f"**Data source:** 🟢 **{provider}** live")
    st.sidebar.caption("Real market data. Prices are adjusted for splits/dividends.")
else:
    st.sidebar.markdown("**Data source:** 🟠 synthetic fallback")
    st.sidebar.caption("Yahoo Finance and Stooq are currently unreachable from this "
                       "network — deterministic synthetic demo data is used so you can "
                       "evaluate the full pipeline. Live data activates automatically "
                       "once either API responds (press 'Retry live data').")
st.sidebar.caption("Personal decision tool — every signal comes with a stop and a size.")

if retry_live:
    _reset_probe()
    _cached_run.clear()
    st.session_state.pop("result", None)
    if _provider():
        st.sidebar.success("Live data available ✔ — press Run forecast.")
    else:
        st.sidebar.warning("Still no live provider reachable — using demo data.")

if "result" not in st.session_state or run_clicked:
    if not tickers_sel:
        st.warning("Select at least one ticker in the sidebar.")
        st.stop()
    kwargs = (tuple(tickers_sel), top_n, capital, use_news, demo)
    with st.status("Running the full investor workflow...", expanded=True) as status:
        st.write("1. Fetching historical data (1y OHLCV)")
        st.write("2. Screening universe for liquidity/price filters")
        st.write("3. Computing 30+ technical indicators")
        st.write("4. Training walk-forward ML ensemble (RF + GBM + classifier)")
        st.write("5. Scanning news & sentiment")
        st.write("6. Ranking & risk planning")
        st.write("7. Walk-forward backtest + Monte Carlo")
        res = _cached_run(*kwargs)
        status.update(label="Workflow complete ✔", state="complete", expanded=False)
    st.session_state["result"] = res
    st.session_state["run_kwargs"] = kwargs

res = st.session_state["result"]
histories = res["histories"]
ranked = res["ranked"]
picks = res["picks"]


# Compatibility guard: if this process holds an older in-memory ranker/risk module
# (Streamlit reloads app.py but not already-imported modules), cached results may
# lack newer columns — rebuild them so the UI can never crash on a KeyError.
def _normalize_output(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    if "predict_price" not in df.columns:
        df["predict_price"] = (df["price"] * (1 + df["expected_pct"] / 100)).round(2)
    if "recommendation" not in df.columns:
        df["recommendation"] = np.where(
            (df["prob_up"] >= 0.60) & (df["score"] >= 65), "BUY",
            np.where((df["prob_up"] >= 0.54) & (df["score"] >= 55), "WATCH", "AVOID"))
    return df


picks = _normalize_output(picks)
ranked = _normalize_output(ranked)
pf = res.get("portfolio")
if pf is not None and not pf.empty and "recommendation" not in pf.columns:
    pf = pf.copy()
    pf["recommendation"] = picks.set_index("ticker")["recommendation"].reindex(pf["ticker"]).values
    res["portfolio"] = pf
regime = res["regime"]

st.title("🎯 Personal Stock Advisor")
st.caption(f"Generated {res['generated_at'].strftime('%Y-%m-%d %H:%M')} | "
           f"Universe screened {len(res['passing'])}/{len(res['tickers'])} | "
           f"Data {'live (' + res['source'] + ')' if res['source'] else 'synthetic (offline demo)'}")

tab_top, tab_market, tab_deep, tab_screen, tab_bt, tab_news, tab_method = st.tabs(
    ["🏠 Top Picks", "📊 Market Overview", "🔬 Deep Dive", "🔍 Screener",
     "🧪 Backtest", "📰 News", "⚙️ Methodology"])


# ---------------------------------------------------------------------------
# Tab 1 — Top Picks
# ---------------------------------------------------------------------------
with tab_top:
    st.subheader("Your next-day trade list (BUY / WATCH / AVOID)")
    st.caption("Each name is ranked by technical + statistical + ML + news signals. "
               "BUY = high score, high P(up), no red flags; WATCH = valid setup, "
               "needs a green open; AVOID = insufficient edge or conflicting signals.")

    if picks.empty:
        st.warning("No names cleared the minimum score today. Try expanding the universe "
                   "or lowering the score threshold in config.MIN_SCORE_TO_SHOW.")
    else:
        metric_row([
            ("Regime", regime.get("regime", "?")),
            ("Top pick", f"{picks.iloc[0]['ticker']} "
                         f"{'▲' if picks.iloc[0]['expected_pct'] >= 0 else '▼'}"
                         f"${picks.iloc[0]['predict_price']:,.2f}"),
            ("Avg expected move", f"{picks['expected_pct'].mean():+.2f}%"),
            ("Avg P(up)", f"{picks['prob_up'].mean() * 100:.0f}%"),
            ("BUY signals", f"{(picks['recommendation'] == 'BUY').sum()}/{len(picks)}"),
        ])

        disp = picks[["rank", "recommendation", "ticker", "price", "expected_pct", "prob_up",
                      "confidence", "score", "sector", "news_label", "atr_pct", "stop_pct",
                      "target_pct", "risk_reward"]].copy()
        disp["pred_fmt"] = disp.apply(
            lambda r: f"{'▲' if r['expected_pct'] >= 0 else '▼'} "
                      f"${r['price'] * (1 + r['expected_pct'] / 100):,.2f}", axis=1)
        disp = disp[["rank", "recommendation", "ticker", "price", "pred_fmt", "expected_pct",
                     "prob_up", "confidence", "score", "sector", "news_label", "atr_pct",
                     "stop_pct", "target_pct", "risk_reward"]]
        disp.columns = ["#", "Call", "Ticker", "Price", "Predicted", "Exp. next-day %",
                        "P(up)", "Confidence", "Score", "Sector", "News", "ATR %",
                        "Stop %", "Target %", "R:R"]
        st.dataframe(heat_styler(disp, cols=["Exp. next-day %", "P(up)", "Score"],
                                 arrow_cols=["Predicted"],
                                 fmt={"Price": "${:,.2f}", "Exp. next-day %": "{:+.2f}",
                                      "P(up)": "{:.1%}", "ATR %": "{:.2f}", "Stop %": "{:.1f}",
                                      "Target %": "{:.1f}", "R:R": "{:.1f}"}),
                     use_container_width=True, hide_index=True)

        st.markdown("#### 🗓️ Today's trade plan — exact orders")
        for _, r in picks.head(4).iterrows():
            stop_px = r["price"] * (1 - r["stop_pct"] / 100)
            tgt_px = r["price"] * (1 + r["target_pct"] / 100)
            st.markdown(f"- **{r['recommendation']} {r['ticker']}** — buy near **${r['price']:.2f}**, "
                        f"stop **${stop_px:.2f}** (−{r['stop_pct']:.1f}%), target **${tgt_px:.2f}** "
                        f"(+{r['target_pct']:.1f}%), size ≈ **{r['shares']} sh** "
                        f"({r['risk_reward']:.1f}:1 R:R). *{r['reasons'].split(' | ')[0]}*")
        st.caption("Rule that pays: never average down below the stop. If the stop is hit, "
                   "the setup is wrong — take the small loss and move on.")

        st.markdown("#### Why these picks — full reasoning")
        for _, r in picks.iterrows():
            with st.expander(f"#{int(r['rank'])} {r['ticker']} — {r['recommendation']} | "
                             f"score {r['score']:.0f} | exp {r['expected_pct']:+.2f}% | "
                             f"P(up) {r['prob_up'] * 100:.0f}%"):
                st.write(r["reasons"])
                st.caption(f"Sub-scores — technical {r['tech_score']:.0f} | statistical "
                           f"{r['stat_score']:.0f} | ML {r['ml_score']:.0f} | "
                           f"news {r['news_score']:.0f} | flags: {r['flag'] or 'none'}")

        if not res["portfolio"].empty:
            st.markdown("#### Portfolio plan (ATR stops, risk sizing, diversification)")
            pf = res["portfolio"][["recommendation", "ticker", "entry", "stop", "target",
                                   "shares", "notional", "risk_reward", "sector"]].copy()
            st.dataframe(heat_styler(pf, cols=["risk_reward"],
                                     fmt={"entry": "${:.2f}", "stop": "${:.2f}",
                                          "target": "${:.2f}", "notional": "${:,.0f}",
                                          "risk_reward": "{:.1f}"}),
                         use_container_width=True, hide_index=True)
            ps = res["portfolio_stats"]
            st.caption(f"Avg stop {ps['avg_stop_pct']:.1f}% / target {ps['avg_target_pct']:.1f}% "
                       f"| avg R:R 1:{ps['avg_rr']:.1f} | {ps['n_positions']} positions "
                       f"(max {config.RISK['max_positions']})")

        bt = res.get("backtest")
        if bt and "error" not in bt:
            st.success(f"**Track record to trust:** walk-forward OOS — hit rate "
                       f"{bt['hit_rate'] * 100:.0f}%, avg {bt['avg_return_per_day_pct']:+.2f}%/day, "
                       f"vs SPY {bt['benchmark_return_pct']:+.1f}% over {bt['n_days']} days. "
                       f"Today's picks are drawn from the same engine — check the Backtest tab "
                       f"before sizing up.")

    st.info("⚠️ **Not investment advice.** Predicted moves are statistical estimates "
            "with wide error bars; always use the suggested stops and size positions "
            "so no single day can materially hurt your account.")


# ---------------------------------------------------------------------------
# Tab 2 — Market Overview
# ---------------------------------------------------------------------------
with tab_market:
    st.subheader("Market regime & breadth")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Regime", regime.get("regime", "?"))
    m2.metric("Risk-on gauge", f"{regime.get('risk_on', 0.5) * 100:.0f}%",
              help="Composite of 20d index momentum + volatility")
    m3.metric("Bench 20-day return",
              f"{(regime.get('benchmark_return_20d') or 0) * 100:+.2f}%")
    m4.metric("SMA50 vs SMA200",
              f"{(regime.get('sma50_vs_sma200') or 0) * 100:+.1f}%")

    c1, c2 = st.columns([2, 1])
    with c1:
        bench = histories.get(config.BENCHMARK)
        if bench is not None:
            ind = add_all_indicators(bench)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], name="SPY close",
                                     line=dict(color="#2196f3", width=1.6)))
            fig.add_trace(go.Scatter(x=ind.index, y=ind["sma50"], name="SMA50",
                                     line=dict(color="#ff9800", width=1.1)))
            fig.add_trace(go.Scatter(x=ind.index, y=ind["sma200"], name="SMA200",
                                     line=dict(color="#9c27b0", width=1.1)))
            fig.update_layout(title=f"{config.BENCHMARK} trend vs 50/200 DMA", height=420,
                              margin=dict(l=40, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Benchmark series unavailable.")
    with c2:
        st.plotly_chart(gauge(regime.get("risk_on", 0.5) * 100, "Risk-on gauge"),
                        use_container_width=True)

    st.markdown("#### Universe breadth (today)")
    breadth_rows = []
    for t, df in histories.items():
        if t.startswith("^") or t == config.BENCHMARK:
            continue
        r = df["Close"].pct_change().iloc[-1]
        breadth_rows.append({"ticker": t, "chg_pct": float(r) * 100,
                             "above_sma50": float(df["Close"].iloc[-1] > df["Close"].rolling(50).mean().iloc[-1]),
                             "above_sma200": float(df["Close"].iloc[-1] > df["Close"].rolling(200).mean().iloc[-1])})
    br = pd.DataFrame(breadth_rows)
    if not br.empty:
        m = st.columns(4)
        m[0].metric("Advancing", f"{(br['chg_pct'] > 0).sum()}/{len(br)}")
        m[1].metric("Declining", f"{(br['chg_pct'] < 0).sum()}/{len(br)}")
        m[2].metric("Above SMA50", f"{int(br['above_sma50'].sum())}/{len(br)}")
        m[3].metric("Above SMA200", f"{int(br['above_sma200'].sum())}/{len(br)}")
        st.dataframe(heat_styler(br.sort_values("chg_pct", ascending=False),
                                 cols=["chg_pct"], fmt={"chg_pct": "{:+.2f}%"}),
                     use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3 — Deep Dive
# ---------------------------------------------------------------------------
with tab_deep:
    st.subheader("Single-stock deep dive")
    ticker = st.selectbox("Ticker", sorted([t for t in histories
                                            if not t.startswith("^") and t != config.BENCHMARK]),
                          key="deep_ticker")
    df = histories[ticker]
    st.plotly_chart(candlestick_chart(df.tail(180), f"{ticker} — 180 trading days"),
                    use_container_width=True)

    rrow = ranked[ranked["ticker"] == ticker]
    if not rrow.empty:
        r = rrow.iloc[0]
        st.markdown(f"**Score {r['score']:.0f}** | expected {r['expected_pct']:+.2f}% | "
                    f"P(up) {r['prob_up'] * 100:.0f}% | {r['confidence']} confidence")
        st.write(r["reasons"])

    st.markdown("#### Latest headlines (with sentiment)")
    items = news_mod.analyze_ticker_news(news_mod.fetch_news(ticker))["headlines"]
    if not items:
        items = news_mod._synthetic_news(ticker)
    for it in items[:8]:
        s = it["sentiment"]
        icon = "🟢" if s > 0.2 else ("🔴" if s < -0.2 else "⚪")
        st.markdown(f"{icon} **{s:+.2f}** · {it['title']}  \n<small>{it['source']} · "
                    f"{it['published'].strftime('%m-%d %H:%M')} UTC</small>",
                    unsafe_allow_html=True)

    st.markdown("#### Model feature importance (top 12)")
    if res["model"].feature_importances:
        fi = pd.Series(res["model"].feature_importances).sort_values(ascending=False).head(12)
        fig = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h",
                               marker_color="#2196f3"))
        fig.update_layout(height=360, title="Which indicators drive the ML prediction",
                          margin=dict(l=10, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        m = res["model"].metrics
        st.caption(f"OOS walk-forward: MAE {m['mae'] * 100:.2f}% | accuracy {m['accuracy'] * 100:.0f}% "
                   f"| base up-rate {m['base_rate_up'] * 100:.0f}% | {m['n_train']} train / "
                   f"{m['n_test']} test rows")


# ---------------------------------------------------------------------------
# Tab 4 — Screener
# ---------------------------------------------------------------------------
with tab_screen:
    st.subheader("Full screened universe (all ranked)")
    colf1, colf2, colf3, colf4 = st.columns(4)
    min_score_f = colf1.slider("Min composite score", 0, 90, 40)
    only_positive = colf2.checkbox("Only positive expected move", value=False)
    min_prob = colf3.slider("Min P(up)", 0.40, 0.80, 0.45, 0.01)
    sector_f = colf4.multiselect("Sectors", sorted(ranked["sector"].unique()),
                                 default=sorted(ranked["sector"].unique()))
    flt = ranked[(ranked["score"] >= min_score_f) &
                 (ranked["sector"].isin(sector_f)) &
                 (ranked["prob_up"] >= min_prob)].copy()
    if only_positive:
        flt = flt[flt["expected_pct"] > 0]
    show = flt[["rank", "recommendation", "ticker", "score", "expected_pct", "prob_up",
                "confidence", "price", "tech_score", "stat_score", "ml_score",
                "news_score", "sector", "news_label", "atr_pct", "flag"]].copy()
    show["pred_fmt"] = show.apply(
        lambda r: f"{'▲' if r['expected_pct'] >= 0 else '▼'} "
                  f"${r['price'] * (1 + r['expected_pct'] / 100):,.2f}", axis=1)
    show = show[["rank", "recommendation", "ticker", "price", "pred_fmt", "score",
                 "expected_pct", "prob_up", "confidence", "tech_score", "stat_score",
                 "ml_score", "news_score", "sector", "news_label", "atr_pct", "flag"]]
    show.columns = ["#", "Call", "Ticker", "Price", "Predicted", "Score", "Exp. next-day %",
                    "P(up)", "Confidence", "Tech", "Stat", "ML", "News", "Sector",
                    "News label", "ATR %", "Flags"]
    st.dataframe(heat_styler(show, cols=["Score", "Exp. next-day %", "P(up)"],
                             arrow_cols=["Predicted"],
                             fmt={"Price": "${:,.2f}", "Exp. next-day %": "{:+.2f}",
                                  "P(up)": "{:.1%}", "ATR %": "{:.2f}"}),
                 use_container_width=True, hide_index=True)
    st.caption("Sub-scores: technical = momentum+trend+volume | statistical = incl. "
               "mean-reversion & volatility | ML = walk-forward ensemble | news = sentiment.")


# ---------------------------------------------------------------------------
# Tab 5 — Backtest
# ---------------------------------------------------------------------------
with tab_bt:
    st.subheader("Walk-forward backtest (honest, no look-ahead)")
    st.caption("Every simulated day: rank with data available at that day's close only, "
               "take the top-N, realize the next-day equal-weight return. Models are "
               "re-trained every 20 days. News is **not** backtested (no free historical "
               "headline archive) — the backtest covers technical + statistical + ML layers.")

    bt = res.get("backtest")
    if not bt or "error" in bt:
        st.warning("Backtest not available for this run (insufficient history).")
    else:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Hit rate", f"{bt['hit_rate'] * 100:.1f}%")
        b2.metric("Avg/day", f"{bt['avg_return_per_day_pct']:+.2f}%",
                  delta=f"vs SPY {bt['benchmark_avg_next_day_pct'] if bt['benchmark_avg_next_day_pct'] is not None else 0:+.2f}%/day")
        b3.metric("Total (OOS period)", f"{bt['total_return_pct']:+.2f}%",
                  delta=f"SPY {bt['benchmark_return_pct'] if bt['benchmark_return_pct'] is not None else 0:+.2f}%")
        b4.metric("Sharpe / MaxDD", f"{bt['sharpe']:.2f} / {bt['max_drawdown_pct']:.1f}%")

        series = bt["series"]
        cum = (1 + series["return"]).cumprod() * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum.index, y=cum, name="Strategy (top-N, equal weight)",
                                 line=dict(color="#2196f3", width=1.8)))
        if bt["benchmark_return_pct"] is not None and config.BENCHMARK in histories:
            bh = histories[config.BENCHMARK].loc[series.index[0]:series.index[-1]]
            bench_cum = bh["Close"] / bh["Close"].iloc[0] * 100
            fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name="SPY B&H",
                                     line=dict(color="#ff9800", width=1.8, dash="dot")))
        fig.update_layout(title="OOS equity curve (rebased to 100)", height=420,
                          margin=dict(l=40, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        mc = bt["monte_carlo"]
        st.markdown("#### Monte-Carlo bootstrap (5,000 resamples of daily returns)")
        mcol = st.columns(5)
        mcol[0].metric("5th pct", f"{mc['p05'] * 100:+.2f}%")
        mcol[1].metric("Median", f"{mc['p50'] * 100:+.2f}%")
        mcol[2].metric("95th pct", f"{mc['p95'] * 100:+.2f}%")
        mcol[3].metric("P(profitable)", f"{mc['prob_positive'] * 100:.0f}%")
        mcol[4].metric("P(beat SPY)", f"{(mc.get('prob_beats_bench') or 0) * 100:.0f}%")

        st.markdown("#### Daily detail")
        st.dataframe(series.tail(30).style.format({"return": "{:+.2%}", "best": "{:+.2%}",
                                                   "worst": "{:+.2%}"}),
                     use_container_width=True)
        st.caption("⚠️ Backtest results on synthetic demo data prove the pipeline works; on "
                   "live data they estimate real performance. Past performance ≠ future results.")
# ---------------------------------------------------------------------------
# Tab 6 — News board
# ---------------------------------------------------------------------------
with tab_news:
    st.subheader("News sentiment board (last 36h, recency-weighted)")
    st.caption("Sources: Google News RSS per ticker (no API key). Each headline is scored "
               "with a financial lexicon (beats/misses, upgrades, probes, deals...) "
               "weighted by an exponential half-life of 12h.")

    news_map = res["news_map"]
    rows_n = []
    for t, nf in news_map.items():
        rows_n.append({"ticker": t, "score": nf["score"], "label": nf["sentiment_label"],
                       "n": nf["n_items"], "impact": nf["impact"],
                       "top": nf["top_positive"] or nf["top_negative"] or ""})
    nb = pd.DataFrame(rows_n).sort_values("score", ascending=False)
    st.dataframe(heat_styler(nb, cols=["score"],
                             fmt={"score": "{:+.2f}", "impact": "{:.2f}"}),
                 use_container_width=True, hide_index=True)

    sel = st.selectbox("Inspect headlines for:", nb["ticker"].tolist(), key="news_sel")
    detail = res["news_map"][sel]
    for it in detail["headlines"][:10]:
        s = it["sentiment"]
        icon = "🟢" if s > 0.2 else ("🔴" if s < -0.2 else "⚪")
        st.markdown(f"{icon} **{s:+.2f}** (w={it['weight']:.2f}) · {it['title']}  \n"
                    f"<small>{it['source']} · {it['published'].strftime('%m-%d %H:%M')} UTC</small>",
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 7 — Methodology
# ---------------------------------------------------------------------------
with tab_method:
    st.subheader("How the forecast is built (industry-standard pipeline)")

    st.markdown("""
    **1. Universe & screening** — liquid US names screened for price ≥ $5, average daily
    dollar volume ≥ $20M (config).
    **2. Market regime** — benchmark trend vs SMA50/SMA200, VIX-type volatility and
    20-day momentum produce a risk-on/risk-off gauge; it dampens confidence in risk-off regimes.
    **3. Technical analysis** — 30+ indicators in pure pandas:
    SMA/EMA, MACD (12/26/9), RSI (Wilder), Stochastic, Williams %R, CCI, MFI, Ultimate
    Oscillator, ADX/DMI, Aroon, Ichimoku, Bollinger (%B/bandwidth), Keltner, ATR,
    historical vol, OBV, VWAP, Chaikin MF, Force Index, VZO, Donchian, pivots, gaps,
    candle patterns.
    **4. Statistical signals** — rule-based 0-100 scores for momentum, trend, volume,
    mean-reversion and volatility-suitability, each with human-readable reasons.
    **5. News & sentiment** — per-ticker headlines from Google News RSS; scored with a
    finance-specific lexicon + phrase rules (earnings beats, upgrades, probes, buybacks...)
    + negation/intensifiers/emphasis; recency decayed (12h half-life) into a ticker signal.
    **6. Machine learning** — RandomForest + GradientBoosting regressors predict the exact
    next-day return; a RandomForest classifier estimates P(up). Trained on pooled
    cross-sectional features with a **temporal walk-forward split** so out-of-sample metrics
    (MAE, accuracy, top-quartile hit rate) are honest. Predictions are calibrated against
    their own historical hit rate.
    **7. Ensemble ranking** — weighted blend (config.WEIGHTS):
    technical 30% + statistical 20% + ML 30% + news 20% → composite 0-100 score,
    expected next-day move % and P(up).
    **8. Risk management** — ATR-based stop (2×ATR) & target (3×ATR), 1% risk per trade,
    half-Kelly sizing (capped), max 8 positions, 10% single-name cap, 35% sector cap.
    **9. Validation** — walk-forward backtest (top-N equal-weight, re-trained every 20d,
    no look-ahead) plus a 5,000-path Monte-Carlo bootstrap.
    """)

    st.markdown("#### Composite weights & model stats")
    w1, w2 = st.columns(2)
    with w1:
        st.write(pd.DataFrame({"component": list(config.WEIGHTS.keys()),
                               "weight": list(config.WEIGHTS.values())}).set_index("component")
                 .style.format({"weight": "{:.0%}"}))
    with w2:
        m = res["model"].metrics
        st.write(pd.DataFrame({"metric": ["OOS MAE", "OOS accuracy", "Base up-rate",
                                          "Train rows", "Test rows"],
                               "value": [f"{m['mae'] * 100:.2f}%", f"{m['accuracy'] * 100:.0f}%",
                                         f"{m['base_rate_up'] * 100:.0f}%",
                                         f"{m['n_train']:,}", f"{m['n_test']:,}"]})
                 .set_index("metric"))

    st.markdown("#### What you can expect from the model (honest capabilities)")
    st.warning("""
    - **How much to trust it:** the Backtest tab shows the engine's own walk-forward
      numbers (hit rate, avg daily move, vs SPY). Today's BUY/WATCH calls come from the
      same engine — assume today's picks perform like those numbers, not better.
    - No model nails next-day direction every time. Daily returns are noise-dominated;
      the edge is small and regime-dependent. That's why every signal ships with a stop.
    - News scoring is lexicon-based (not a fine-tuned LLM); an ambiguous headline can be
      misread. Skim the News tab before acting on a news-driven pick.
    - The backtest assumes zero costs and perfect fills — real fills will be a bit worse.
    - **Do not trade synthetic (demo) data** — it only demonstrates the pipeline.
      Always confirm 🟢 live data in the sidebar first.
    """)

