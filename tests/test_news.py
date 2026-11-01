"""Tests for news fetching/parsing and sentiment scoring."""
from datetime import datetime, timedelta, timezone

from stockforecaster import news


def test_score_text_positive():
    t = "NVDA beats earnings expectations, raises full-year guidance, shares surge"
    assert news.score_text(t) > 0.5


def test_score_text_negative():
    t = "Company misses estimates badly, cuts guidance, shares plunge on probe"
    assert news.score_text(t) < -0.5


def test_score_text_neutral():
    assert abs(news.score_text("Company announces date of annual shareholder meeting")) < 0.3


def test_score_text_negation():
    pos = news.score_text("The stock is strong and profitable")
    neg = news.score_text("The stock is not strong and not profitable")
    assert neg < pos


def test_rss_parse():
    xml = """<?xml version="1.0"?><rss><channel><item>
      <title>Apple hits all-time high on strong iPhone demand</title>
      <link>http://example.com/1</link>
      <pubDate>Mon, 27 Oct 2026 14:30:00 GMT</pubDate>
      <source>Reuters</source></item><item>
      <title>Apple misses estimates, shares fall</title>
      <link>http://example.com/2</link>
      <pubDate>Mon, 27 Oct 2026 15:45:00 GMT</pubDate>
      <source>Bloomberg</source></item></channel></rss>"""
    items = news._parse_rss(xml, 10)
    assert len(items) == 2
    assert "iPhone" in items[0]["title"]
    assert items[0]["published"].tzinfo is not None


def test_synthetic_news_analysis():
    items = news._synthetic_news("TEST")
    res = news.analyze_ticker_news(items)
    assert -1.0 <= res["score"] <= 1.0
    assert res["n_items"] == len(items)
    for h in res["headlines"]:
        assert -1.0 <= h["sentiment"] <= 1.0
        assert 0 < h["weight"] <= 1.0


def test_recency_decay():
    now = datetime.now(timezone.utc)
    old = now - timedelta(hours=24)
    fresh = now - timedelta(hours=1)
    assert news._recency_weight(fresh, now) > news._recency_weight(old, now)


def test_no_news_analysis():
    res = news.analyze_ticker_news([])
    assert res["n_items"] == 0 and res["score"] == 0.0
    assert res["sentiment_label"] == "no news"
