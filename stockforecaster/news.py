"""News intelligence module.

Collection: Google News RSS (https://news.google.com/rss/search?q=...) — free,
no API key required. Fetches stock-specific headlines for each ticker.
Sentiment:  a financial-domain lexicon scorer (built in — no downloads) that
            handles negation, intensifiers, and finance-specific terms, plus
            case/exclamation emphasis. Optionally boosted by pattern rules for
            earnings beats / upgrades / downgrades / probes etc.
Scoring:    per-headline score in [-1, +1], recency-weighted with exponential
            decay; aggregated into a ticker-level signal in [-1, +1].

Offline mode: deterministic synthetic headlines so the demo never breaks.
"""
from __future__ import annotations

import html
import random
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

from . import config

# ---------------------------------------------------------------------------
# Financial sentiment lexicon (hand-curated extension of the classic
# Loughran-McDonald finance word lists: dependency-free).
# ---------------------------------------------------------------------------
POSITIVE_WORDS: dict[str, float] = {
    "surge": 1.5, "soar": 1.5, "jump": 1.2, "rally": 1.2, "climb": 1.0,
    "beat": 1.5, "beats": 1.5, "record": 1.3, "record-high": 1.5,
    "upgrade": 1.5, "upgraded": 1.5, "outperform": 1.3, "buy": 1.2,
    "strong": 1.0, "growth": 1.0, "profit": 1.0, "earnings-beat": 1.5,
    "expand": 1.0, "expansion": 1.0, "gain": 1.0, "gains": 1.0,
    "win": 1.2, "wins": 1.2, "approval": 1.4, "approved": 1.4,
    "positive": 1.2, "boost": 1.3, "bullish": 1.4, "breakout": 1.3,
    "deal": 1.2, "acquisition": 1.1, "merger": 1.1, "partnership": 1.0,
    "success": 1.3, "successful": 1.3, "outlook": 0.8,
    "raise": 1.0, "raises": 1.0, "dividend": 0.8, "buyback": 1.2,
    "repurchase": 1.2, "stellar": 1.5, "soars": 1.5, "tops": 1.2,
    "exceeds": 1.3, "exceed": 1.3, "accelerate": 1.1, "momentum": 0.9,
    "upside": 1.2, "recommend": 1.0, "add": 0.8, "overweight": 1.3,
    "catalyst": 0.9, "breakthrough": 1.4, "launch": 0.7, "unveil": 0.7,
    "unveils": 0.7, "resume": 0.8, "settle": 0.5, "settles": 0.5,
    "recover": 0.9, "recovery": 0.9, "rebound": 1.0, "rebounds": 1.0,
    "optimistic": 1.0, "booming": 1.3, "boom": 1.1, "best": 1.1,
    "favorable": 1.0, "favor": 0.9, "innovate": 1.0, "innovation": 1.0,
    "collaboration": 1.0, "partnership": 1.0, "momentum": 0.9,
}

NEGATIVE_WORDS: dict[str, float] = {
    "plunge": -1.5, "plummet": -1.5, "crash": -1.6, "tumble": -1.3,
    "sink": -1.2, "slump": -1.2, "fall": -1.0, "drops": -1.0, "drop": -0.9,
    "miss": -1.5, "misses": -1.5, "downgrade": -1.5, "downgraded": -1.5,
    "underperform": -1.3, "sell": -1.1, "weak": -1.0, "weakness": -1.0,
    "loss": -1.2, "losses": -1.2, "fear": -1.2,
    "probe": -1.3, "investigation": -1.3, "lawsuit": -1.4, "sue": -1.3,
    "sues": -1.3, "subpoena": -1.4, "fraud": -1.7, "scandal": -1.6,
    "recall": -1.4, "recalls": -1.4, "bankruptcy": -1.8, "insolvency": -1.8,
    "layoff": -1.2, "layoffs": -1.2, "cut": -1.0, "cuts": -1.0,
    "warning": -1.3, "warns": -1.3,
    "negative": -1.2, "bearish": -1.3, "short": -0.7, "shorts": -0.7,
    "decline": -1.0, "declines": -1.0, "slowdown": -1.0, "contraction": -1.1,
    "recession": -1.4, "inflation": -0.6, "rate-hike": -0.8, "tariff": -0.7,
    "tariffs": -0.7, "ban": -1.3, "bans": -1.3, "block": -1.0, "blocks": -1.0,
    "regulatory": -0.8, "fine": -1.2, "fined": -1.2, "penalty": -1.2,
    "halt": -1.2, "halts": -1.2, "halted": -1.2, "delay": -0.9,
    "delays": -0.9, "cancel": -1.1, "cancels": -1.1, "withdraw": -1.1,
    "withdraws": -1.1, "fail": -1.3, "fails": -1.3, "failed": -1.3,
    "default": -1.5, "crisis": -1.5, "trouble": -1.2, "troubles": -1.2,
    "struggle": -1.1, "struggles": -1.1, "risky": -0.9, "risk": -0.5,
    "caution": -0.8, "concern": -0.8, "concerns": -0.8, "worries": -0.9,
    "worry": -0.9, "selloff": -1.4, "sell-off": -1.4,
}

INTENSIFIERS = {"very": 1.4, "sharply": 1.4, "significantly": 1.3,
                "huge": 1.4, "massive": 1.4, "deeply": 1.3, "big": 1.3,
                "much": 1.2, "way": 1.2, "far": 1.2, "slightly": 0.7,
                "barely": 0.6}

NEGATIONS = {"not", "no", "never", "without", "hardly", "barely"}

# Frequent finance phrases -> direct score (highest priority rules)
PHRASE_RULES: list[tuple[str, float]] = [
    (r"earnings?\s+(beat|tops|exceeds)\b", 1.8),
    (r"beats?\s+expectations", 1.8),
    (r"tops?\s+estimates", 1.6),
    (r"raises?\s+(full[- ]year\s+)?guidance", 1.5),
    (r"guidance\s+raise", 1.5),
    (r"upgrade[sd]?\s+to\s+buy", 1.6),
    (r"upgraded?\s+to\s*overweight", 1.5),
    (r"price[- ]target\s+raised", 1.4),
    (r"record\s+(profit|revenue|quarter|sales|high)", 1.5),
    (r"buyback\s+program", 1.3),
    (r"dividend\s+(increase|boost|raised)", 1.2),
    (r"fda\s+(approval|approves|clears)", 1.7),
    (r"(approval|approved|cleared)", 1.4),
    (r"analysts?\s+(say|see)\s+upside", 1.0),
    (r"misses\s+(expectations|estimates|targets)", -1.8),
    (r"misses?\s+earnings", -1.8),
    (r"cuts?\s+(full[- ]year\s+)?guidance", -1.6),
    (r"guidance\s+cut", -1.6),
    (r"downgrad[ed]?\s+to\s+(sell|underperform|neutral)", -1.6),
    (r"price[- ]target\s+cut|cut[s]?\s+price[- ]target", -1.4),
    (r"securit(ies|y)?\s+(investigation|probe|fraud)", -1.7),
    (r"lawsuit\s+filed", -1.6),
    (r"class[- ]action", -1.4),
    (r"delisted|delisting", -1.8),
    (r"bankrupt|insolven", -1.9),
    (r"warning|warns", -1.3),
    (r"recall[s]?", -1.4),
    (r"layoff[s]?", -1.3),
    (r"short[- ]seller", -1.3),
    (r"sell[- ]off|selloff", -1.4),
    (r"plunge|plummet|crash|tumble", -1.6),
    (r"surge|soar|rally|jump", 1.5),
    (r"(partner|deal|acquisit|merger)\b", 1.1),
]


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9'\- ]", " ", text.lower())
    return text.split()


def score_text(text: str) -> float:
    """Score a headline/text in [-1.0, 1.0] (rules + lexicon + emphasis)."""
    if not text:
        return 0.0
    lower = text.lower()

    # 1) Phrase rules (highest priority; capture multi-word finance events)
    phrase_score = 0.0
    phrase_count = 0
    for pat, val in PHRASE_RULES:
        if re.search(pat, lower):
            phrase_score += val
            phrase_count += 1
    if phrase_count:
        phrase_score = max(-2.0, min(2.0, phrase_score))

    # 2) Lexicon with negation + intensifiers
    lex_score = 0.0
    tokens = _tokenize(text)
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        negated = prev in NEGATIONS or tok.startswith("un") or tok.startswith("non")
        if tok in POSITIVE_WORDS:
            val = POSITIVE_WORDS[tok]
        elif tok in NEGATIVE_WORDS:
            val = NEGATIVE_WORDS[tok]
        else:
            continue
        if prev in INTENSIFIERS:
            val *= INTENSIFIERS[prev]
        if negated:
            val *= -0.8
        lex_score += val
    lex_score = max(-2.0, min(2.0, lex_score))

    # 3) Emphasis: ALL-CAPS words and exclamation marks
    caps = len(re.findall(r"\b[A-Z]{3,}\b", text))
    bangs = text.count("!")
    emphasis = 0.15 * caps + 0.10 * bangs
    if phrase_score < 0 or lex_score < 0:
        emphasis *= -1.0

    raw = 0.6 * lex_score + 0.9 * phrase_score + emphasis
    return max(-1.0, min(1.0, raw / 3.0))


# ---------------------------------------------------------------------------
# News fetching (Google News RSS, no API key)
# ---------------------------------------------------------------------------
def _parse_rss(xml_text: str, max_items: int) -> list[dict[str, Any]]:
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items
    for item in root.iter("item"):
        if len(items) >= max_items:
            break
        title = html.unescape((item.findtext("title") or "")).strip()
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate") or ""
        source = item.findtext("source") or ""
        try:
            dt = datetime.strptime(pub.strip(), "%a, %d %b %Y %H:%M:%S %Z")
            dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            dt = datetime.now(timezone.utc)
        items.append({"title": title, "link": link, "source": source,
                      "published": dt})
    return items


def fetch_news(ticker: str, query: Optional[str] = None,
               lookback_hours: int = config.NEWS_LOOKBACK_HOURS,
               max_items: int = config.NEWS_PER_TICKER,
               timeout: int = 10) -> list[dict[str, Any]]:
    """Fetch headlines for a ticker from Google News RSS."""
    q = query or f"{ticker} stock"
    try:
        url = ("https://news.google.com/rss/search?q=" +
               requests.utils.quote(q) + "&hl=en-US&gl=US&ceid=US:en")
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200 or not resp.text.strip():
            return []
        return _parse_rss(resp.text, max_items)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Ticker-level news signal
# ---------------------------------------------------------------------------
def _recency_weight(published: datetime, now: datetime,
                    half_life_h: float = config.NEWS_RECENCY_HALF_LIFE_H) -> float:
    h = (now - published).total_seconds() / 3600.0
    return float(0.5 ** (max(h, 0.0) / half_life_h))


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def analyze_ticker_news(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate headlines into a ticker news signal.

    Returns dict with:
      score           : recency-weighted mean sentiment in [-1, 1]
      impact          : scaled magnitude of notable, recent headlines
      n_items / headlines / top_positive / top_negative / sentiment_label
    """
    now = datetime.now(timezone.utc)
    scored = []
    for it in items:
        s = score_text(it["title"])
        w = _recency_weight(it["published"], now)
        scored.append({**it, "sentiment": round(s, 3), "weight": round(w, 3)})
    if not scored:
        return {"score": 0.0, "impact": 0.0, "n_items": 0, "headlines": [],
                "top_positive": None, "top_negative": None,
                "sentiment_label": "no news"}

    total_w = sum(x["weight"] for x in scored) or 1.0
    score = sum(x["sentiment"] * x["weight"] for x in scored) / total_w
    impact = sum(x["weight"] * min(abs(x["sentiment"]), 1.0) for x in scored)
    if score > 0.2:
        label = "positive"
    elif score < -0.2:
        label = "negative"
    else:
        label = "neutral"
    top_pos = max(scored, key=lambda x: x["sentiment"], default=None)
    top_neg = min(scored, key=lambda x: x["sentiment"], default=None)
    return {"score": _clip(score, -1.0, 1.0),
            "impact": _clip(impact, 0.0, 5.0),
            "n_items": len(scored),
            "headlines": sorted(scored, key=lambda x: -x["weight"]),
            "top_positive": top_pos["title"] if top_pos and top_pos["sentiment"] > 0 else None,
            "top_negative": top_neg["title"] if top_neg and top_neg["sentiment"] < 0 else None,
            "sentiment_label": label}


def news_signals(tickers: list[str], synthetic: bool = False) -> dict[str, dict[str, Any]]:
    """Compute the news signal for every ticker.

    `synthetic=True` -> deterministic demo headlines (offline mode).
    """
    out: dict[str, dict[str, Any]] = {}
    for t in tickers:
        if synthetic:
            items = _synthetic_news(t)
        else:
            items = fetch_news(t)
        out[t] = analyze_ticker_news(items)
    return out


# ---------------------------------------------------------------------------
# Synthetic news (offline / demo fallback) — deterministic per ticker
# ---------------------------------------------------------------------------
_SYNTH_POS = [
    "{} beats Q3 earnings estimates, raises full-year guidance",
    "{} shares surge as analysts upgrade to Buy with raised price target",
    "{} announces record revenue growth and a $5B buyback program",
    "{} wins major enterprise deal, stock climbs on strong demand",
    "{} hits all-time high after positive earnings momentum and growth outlook",
    "{} unveils breakthrough product; boosted by strong order book",
]
_SYNTH_NEG = [
    "{} misses earnings expectations, cuts guidance; shares fall",
    "{} faces regulatory probe after accounting questions",
    "{} shares plunge on weak revenue outlook and layoff plans",
    "{} downgraded to Underperform; price target cut by analysts",
]


def _synthetic_news(ticker: str) -> list[dict[str, Any]]:
    rng = random.Random(abs(hash(ticker)) % (2 ** 31))
    n = rng.randint(0, 6)
    items = []
    now = datetime.now(timezone.utc)
    pool = _SYNTH_POS + _SYNTH_NEG + [
        f"{ticker} holds annual shareholder meeting", f"{ticker} announces new analyst day"]
    for _ in range(n):
        title = rng.choice(pool).format(ticker)
        pub = now - timedelta(hours=rng.uniform(1, config.NEWS_LOOKBACK_HOURS))
        items.append({"title": title, "link": "https://example.com/demo",
                      "source": "Demo", "published": pub})
    return items


def latest_headlines(ticker: str) -> list[dict[str, Any]]:
    """Convenience helper for the UI: headlines for one ticker (live or demo)."""
    items = fetch_news(ticker)
    if not items:
        items = _synthetic_news(ticker)
    return analyze_ticker_news(items)["headlines"]


