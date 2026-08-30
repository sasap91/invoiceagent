"""Render-level checks run in CI without opening a browser."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_all_fixture_routes_without_exceptions():
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "$500 → $200 → $0",
        "Routing replay",
        "Invoice & payment ledger",
        "Structured fallback",
        "Live local model",
        "How routing works",
    ]
    for scenario in ("easy-local", "model-accept", "ambiguous-escalate"):
        app.radio(key="scenario-selector").set_value(scenario).run()
        assert not app.exception


def test_payment_story_reaches_paid_state_without_model_call():
    app = AppTest.from_file(APP, default_timeout=20).run()
    app.radio(key="payment-story-stage").set_value(2).run()
    assert not app.exception
    assert any("PAY-799" in markdown.value for markdown in app.markdown)
