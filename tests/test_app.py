"""Validate the Streamlit app boots and renders without exceptions (AppTest)."""
import os

from streamlit.testing.v1 import AppTest


def test_app_renders():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=600)
    at.run()
    assert not at.exception, f"App raised: {at.exception}"
    assert at.title[0].value == "🎯 Personal Stock Advisor"
    # Sidebar controls exist
    labels = [x.label for x in at.sidebar.multiselect] + [b.label for b in at.sidebar.button]
    assert any("Universe" in l for l in labels)
    assert any("Run forecast" in l for l in labels)
    # Tabs rendered
    tabs = [t.label for t in at.tabs]
    assert any("Top Picks" in t for t in tabs)
    assert any("Backtest" in t for t in tabs)
