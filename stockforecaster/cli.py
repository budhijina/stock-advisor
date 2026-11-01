"""Command-line interface: `python -m stockforecaster.cli forecast [options]`.

Examples
--------
  .venv/bin/python -m stockforecaster.cli forecast                 # default universe
  .venv/bin/python -m stockforecaster.cli forecast --top 15 --json
  .venv/bin/python -m stockforecaster.cli forecast --tickers AAPL NVDA TSLA
  .venv/bin/python -m stockforecaster.cli backtest --days 60
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from .workflow import run, pick_explanations


def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    def line_():
        return "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    out = ["", line_(), sep]
    for r in rows:
        out.append("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def cmd_forecast(args: argparse.Namespace) -> int:
    tickers = args.tickers.split(",") if args.tickers else None
    n = len(tickers) if tickers else len(config.DEFAULT_UNIVERSE)
    print(f"Running forecast for {n} tickers...", file=sys.stderr)
    res = run(tickers=tickers, top_n=args.top, capital=args.capital,
              use_news=not args.no_news, demo_mode=args.demo,
              run_backtest=not args.no_backtest,
              progress_cb=lambda msg: print(f"  » {msg}", file=sys.stderr))


    picks = res["picks"]
    if picks.empty:
        print("No picks cleared the minimum score today.")
        return 0

    regime = res["regime"]
    print(f"\nMARKET REGIME: {regime.get('regime', 'n/a')} | "
          f"risk-on {regime.get('risk_on', 0.5):.2f} | "
          f"bench 20d {(regime.get('benchmark_return_20d', 0) or 0) * 100:+.2f}% | "
          f"SMA50/200 {(regime.get('sma50_vs_sma200', 0) or 0) * 100:+.2f}%")
    print(f"Universe screened: {len(res['passing'])}/{len(res['tickers'])} pass filters")

    rows = [[str(r["rank"]), r["recommendation"], r["ticker"], f"${r['price']:.2f}",
             f"{'▲' if r['expected_pct'] >= 0 else '▼'}${r['price'] * (1 + r['expected_pct'] / 100):.2f}",
             f"{r['expected_pct']:+.2f}%", f"{r['prob_up'] * 100:.1f}%",
             r["confidence"], f"{r['score']:.1f}",
             f"-{r['stop_pct']:.1f}%/+{r['target_pct']:.1f}%",
             f"1:{r['risk_reward']:.1f}"] for _, r in picks.iterrows()]
    print(_fmt_table(["#", "CALL", "TICKER", "PRICE", "PREDICTED", "EXPECTED NEXT-DAY",
                      "PROB UP", "CONF", "SCORE", "STOP/TARGET", "R:R"], rows))

    print("\nTODAY'S TRADE PLAN (exact orders):")
    for _, r in picks.head(4).iterrows():
        stop_px = r["price"] * (1 - r["stop_pct"] / 100)
        tgt_px = r["price"] * (1 + r["target_pct"] / 100)
        print(f"  {r['recommendation']:6s} {r['ticker']:5s} buy ≈ ${r['price']:.2f} "
              f"({'▲' if r['expected_pct'] >= 0 else '▼'} ${r['price'] * (1 + r['expected_pct'] / 100):.2f} pred.) | "
              f"stop ${stop_px:.2f} (-{r['stop_pct']:.1f}%) | target ${tgt_px:.2f} "
              f"(+{r['target_pct']:.1f}%) | {r['shares']} sh | R:R 1:{r['risk_reward']:.1f}")
    print("  Rule: never average down below the stop — if the stop hits, the setup was wrong.")

    print("\nWHY THESE PICKS:")
    for e in pick_explanations(picks):
        print(f"\n  #{e['rank']} {e['ticker']} (score {e['score']})")
        print(f"     Expected {e['expected_pct']:+.2f}% | P(up) {e['prob_up'] * 100:.0f}% | "
              f"stop -{e['stop_pct']:.1f}% / target +{e['target_pct']:.1f}% | R:R 1:{e['risk_reward']:.1f}")
        print(f"     {e['reasons']}")

    if res.get("portfolio") is not None and not res["portfolio"].empty:
        pf = res["portfolio"]
        print(f"\nPORTFOLIO PLAN (${args.capital:,.0f} capital):")
        rows_p = [[r["ticker"], f"${r['entry']:.2f}", f"${r['stop']:.2f}", f"${r['target']:.2f}",
                   str(r["shares"]), f"${r['notional']:,.0f}",
                   f"{r['risk_reward']:.1f}"] for _, r in pf.iterrows()]
        print(_fmt_table(["TICKER", "ENTRY", "STOP", "TARGET", "SHARES", "NOTIONAL", "R:R"], rows_p))

    if res.get("backtest") and "error" not in res["backtest"]:
        bt = res["backtest"]
        print(f"\nWALK-FORWARD EVIDENCE (last {bt['n_days']} days, top-{bt['n_picks_per_day']} equal-weight):")
        print(f"  Hit rate {bt['hit_rate'] * 100:.1f}% | avg/day {bt['avg_return_per_day_pct']:+.2f}% | "
              f"total {bt['total_return_pct']:+.2f}% | expectancy {bt['expectancy_pct']:+.2f}%")
        print(f"  Profit factor {bt['profit_factor']:.2f} | Sharpe {bt['sharpe']:.2f} | "
              f"max DD {bt['max_drawdown_pct']:.2f}% | "
              f"vs SPY {bt['benchmark_return_pct']:.2f}%")
        mc = bt["monte_carlo"]
        print(f"  Monte-Carlo 5000 sims: 5% {mc['p05'] * 100:+.1f}% | med {mc['p50'] * 100:+.1f}% | "
              f"95% {mc['p95'] * 100:+.1f}% | P(positive) {mc['prob_positive'] * 100:.0f}% | "
              f"P(beat SPY) {(mc.get('prob_beats_bench') or 0) * 100:.0f}%")

    if args.json:
        print("\nJSON:")
        print(json.dumps({
            "regime": regime, "picks": picks.to_dict("records"),
            "base_rates": res["base_rates"],
            "backtest": {k: v for k, v in (res["backtest"] or {}).items()
                         if k != "series"} if res.get("backtest") else None,
        }, indent=2, default=str))

    print("\nPersonal advisor mode — signals for your own trading decisions. "
          "Every order includes a stop; the Backtest numbers are the honest expectation.", file=sys.stderr)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from .backtest import walk_forward_backtest
    from . import data as market_data
    tickers = args.tickers.split(",") if args.tickers else config.DEFAULT_UNIVERSE
    print("Collecting data for backtest...", file=sys.stderr)
    histories = market_data.get_histories(tickers, synthetic_ok=True)
    bt = walk_forward_backtest(histories, top_n=args.top, oos_days=args.days)
    if "error" in bt:
        print(f"Backtest failed: {bt['error']}")
        return 1
    print(f"WALK-FORWARD BACKTEST — {bt['n_days']} days, top-{bt['n_picks_per_day']} equal weight")
    for k in ["avg_return_per_day_pct", "total_return_pct", "hit_rate", "avg_win_pct",
              "avg_loss_pct", "profit_factor", "expectancy_pct", "sharpe",
              "max_drawdown_pct", "benchmark_return_pct"]:
        v = bt.get(k)
        print(f"  {k:28s}: {f'{v:.2f}%' if 'pct' in k else f'{v:.3f}'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Stock Forecaster CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("forecast", help="Generate today's top-pick forecast")
    f.add_argument("--tickers", default=None, help="Comma-separated ticker list")
    f.add_argument("--top", type=int, default=config.TOP_N_DEFAULT)
    f.add_argument("--capital", type=float, default=100_000.0)
    f.add_argument("--no-news", action="store_true", help="Skip news scan")
    f.add_argument("--demo", action="store_true", help="Force synthetic news")
    f.add_argument("--no-backtest", action="store_true", help="Skip backtest")
    f.add_argument("--json", action="store_true", help="Emit JSON summary")
    f.set_defaults(func=cmd_forecast)

    b = sub.add_parser("backtest", help="Run walk-forward backtest")
    b.add_argument("--tickers", default=None)
    b.add_argument("--top", type=int, default=config.TOP_N_DEFAULT)
    b.add_argument("--days", type=int, default=60)
    b.set_defaults(func=cmd_backtest)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

