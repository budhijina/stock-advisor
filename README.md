# 🎯 Personal Stock Advisor — Next-Day Stock Signals

A day-trading **personal advisor** that automates the entire workflow you would run
before buying a stock, then answers the core question:

> **Which top stocks should I buy today, at what entry — and with what stop, target and size?**

It combines **industry-standard technical analysis**, **statistical signal scoring**,
**walk-forward machine learning**, **news sentiment**, **risk management** and
**backtest validation** into explicit **BUY / WATCH / AVOID** calls with an exact
order plan, surfaced through a Streamlit dashboard **and** a CLI.

> 💡 **Trust it like this:** every call ships with a stop and a size, and the Backtest tab
> shows the engine's own walk-forward hit rate / average move / SPY comparison. Today's
> calls come from that same engine — assume they perform like those numbers.

---

## Quick start

```bash
# 1) Create a virtualenv and install
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2) Run the dashboard
.venv/bin/streamlit run app.py

# 3) Or run from the terminal
.venv/bin/python -m stockforecaster.cli forecast --top 10
.venv/bin/python -m stockforecaster.cli forecast --tickers AAPL,NVDA,TSLA --json
.venv/bin/python -m stockforecaster.cli backtest --days 60
```

**Data sources — no paid API keys:**
- Historical OHLCV: [yfinance](https://github.com/ranaroussi/yfinance) (free) with a
  **Stooq** free-CSV fallback; the sidebar shows the active provider (🟢 yfinance / 🟢 stooq)
  and has a **🔄 Retry live data** button.
- News: Google News RSS search per ticker (free, no key).
- If no real provider is reachable, the app transparently falls back to a *deterministic
  synthetic* dataset so you can still exercise the whole pipeline offline (demo only —
  it has no predictive meaning).

---

## The automated "investor workflow"

| Step | What an investor does | What the app does |
|---|---|---|
| 0 | Define an investable universe | ~50 liquid megacap + high-beta names (editable in `config.py`) |
| 1 | Don't trade illiquid junk | Screens price ≥ $5, avg $ volume ≥ $20M |
| 2 | Is the market risk-on or risk-off? | Regime engine: SPY vs SMA50/200, volatility, 20d momentum → risk-on gauge |
| 3 | Study the chart | 30+ technical indicators — SMA/EMA, MACD, RSI (Wilder), Stochastic, Williams %R, CCI, MFI, Ultimate Oscillator, ADX/DMI, Aroon, Ichimoku, Bollinger, Keltner, ATR, historical vol, OBV, VWAP, Chaikin MF, Force Index, VZO, Donchian, pivots, gaps, candle patterns |
| 4 | Judge momentum & mean reversion | 0–100 statistical scores: momentum, trend, volume, mean-reversion, volatility-suitability — each with human-readable reasons |
| 5 | Read the news | Per-ticker headlines (36h), finance-lexicon sentiment (beats/misses, upgrades, probes, buybacks…), negation & emphasis handling, 12h recency decay |
| 6 | Apply experience/ML | RandomForest + GradientBoosting regressors (exact next-day return) + RF classifier (P(up)); **temporal walk-forward split** so accuracy is honest; calibrated to own historical hit-rate |
| 7 | Decide | **Composite score (0–100)** = 30% technical + 20% statistical + 30% ML + 20% news; **expected next-day % move** and **P(up)** with confidence label |
| 8 | Manage risk | ATR stops (2×ATR) / targets (3×ATR), 1% risk/trade, half-Kelly sizing (capped), max 8 positions, 10% single-name cap, 35% sector cap |
| 9 | Prove it works | Walk-forward backtest (top-N equal weight, re-trained every 20d, zero look-ahead) + 5,000-path Monte Carlo + SPY benchmark |

## Dashboard tabs

- **🏠 Top Picks** — ranked list with expected %, probability, confidence, reasons, portfolio plan.
- **📊 Market Overview** — regime, risk-on gauge, SPY trend, universe breadth.
- **🔬 Deep Dive** — candlestick + SMA/BB, RSI, volume, per-headline sentiment, model feature importances.
- **🔍 Screener** — filterable full-universe table (score / P(up) / sector).
- **🧪 Backtest** — OOS equity curve vs SPY, hit rate, Sharpe, max DD, Monte-Carlo percentiles.
- **📰 News** — sentiment board + headline inspector.
- **⚙️ Methodology** — full transparency of every algorithm, weights, model stats and limitations.

## Project layout

```
stock-forecaster/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── stockforecaster/
│   ├── config.py           # universe, weights, thresholds (single tuning point)
│   ├── data.py             # yfinance + cache + synthetic fallback + regimes
│   ├── indicators.py       # 30+ pure-pandas TA indicators
│   ├── features.py         # feature engineering + next-day targets
│   ├── news.py             # Google News RSS + financial-lexicon sentiment
│   ├── models.py           # statistical scorers + RF/GBM ensemble (walk-forward)
│   ├── ranker.py           # composite scoring → top picks with % & probability
│   ├── backtest.py         # walk-forward engine + Monte Carlo
│   ├── risk.py             # ATR stops, sizing, Kelly, portfolio construction
│   ├── workflow.py         # end-to-end pipeline (shared by CLI and UI)
│   └── cli.py              # `forecast` / `backtest`
└── tests/                  # 26 pytest tests incl. a Streamlit AppTest
```

## Testing

```bash
.venv/bin/python -m pytest
```

## Tuning

Everything is configurable in `stockforecaster/config.py`:
`WEIGHTS` (ensemble mix), `RISK` (stops/sizing/caps), `ML` (model params),
universe, screening filters, sentiment half-life…

## Known limitations (surfaced in-app too)

- Next-day returns are mostly noise; the honest OOS metrics are always shown — do not
  expect reliable daily direction from any indicator.
- News sentiment is lexicon-based, not a modern LLM; misclassification is possible.
- Backtest excludes news and assumes zero costs/slippage.
- Synthetic (offline) data exercises the pipeline but has **no predictive meaning**.
