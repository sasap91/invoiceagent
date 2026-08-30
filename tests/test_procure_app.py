"""Render-level acceptance checks for ProcureAgent's controlled P0 UI."""

from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from demo.procure_scenarios import (
    INVOICE_FIXTURES,
    PRIMARY_SCENARIO,
    UNKNOWNCO_ADVERSARIAL,
    format_minor,
)
from procureagent.ui_adapters import _reset_cached_ryan_adapter_for_tests


APP = str((Path(__file__).resolve().parents[1] / "procure_app.py").resolve())


def boot() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception
    return app


def values(elements) -> str:
    return "\n".join(str(element.value) for element in elements)


def page_text(app: AppTest) -> str:
    return values(
        [
            *app.markdown,
            *app.caption,
            *app.info,
            *app.warning,
            *app.success,
            *app.error,
        ]
    )


def metric_values(app: AppTest, label: str) -> list[str]:
    return [str(metric.value) for metric in app.metric if metric.label == label]


def dollars(value: str) -> Decimal:
    return Decimal(value.replace("$", "").replace(",", ""))


def test_locked_presentation_fixture_preserves_headline_contract() -> None:
    assert len(INVOICE_FIXTURES) == 4
    assert sum(invoice["amount_minor"] for invoice in INVOICE_FIXTURES) == 620_000
    assert PRIMARY_SCENARIO["cash_minor"] == 500_000
    assert PRIMARY_SCENARIO["obligations_minor"] == 620_000
    assert UNKNOWNCO_ADVERSARIAL["lookup_permitted"] is False
    assert UNKNOWNCO_ADVERSARIAL["included_in_obligations"] is False
    assert format_minor(150_000) == "$1,500.00"


def test_three_route_shell_defaults_to_progressive_guided_demo() -> None:
    app = boot()

    assert not app.tabs
    route = app.radio(key="top-route")
    assert route.value == "Guided demo"
    assert list(route.options) == ["Guided demo", "Overview", "Evidence & methods"]
    assert '<ol class="pa-progress"' in page_text(app)
    assert 'class="current" aria-current="step"' in page_text(app)
    assert "Read the invoice" in page_text(app)
    assert "Simulation only" in page_text(app)
    assert "Human approval required" in page_text(app)
    assert "No bank or ERP connected" in page_text(app)

    # Progressive disclosure: only the current step's action exists.
    button_keys = {button.key for button in app.button}
    assert "eval-run-document-adapter" in button_keys
    assert "eval-record-human-review" not in button_keys
    assert "eval-approve-batch" not in button_keys
    assert "eval-run-receipt-adapter" not in button_keys
    assert "eval-confirm-payment" not in button_keys


def test_overview_route_preserves_locked_business_contract() -> None:
    app = boot()
    app.radio(key="top-route").set_value("Overview").run()
    assert not app.exception

    assert dollars(metric_values(app, "Cash available")[0]) == Decimal("5000.00")
    assert dollars(metric_values(app, "Supplier obligations")[0]) == Decimal("6200.00")
    assert metric_values(app, "State version") == ["1"]

    grid = next(
        str(item.value)
        for item in app.markdown
        if '<div class="pa-grid">' in str(item.value)
    )
    cards = grid.split('<article class="pa-card">')[1:]
    assert len(cards) == 4
    expected = (
        ("Fresh Farms", "FF-10482", "$1,500.00", "PAY"),
        ("Prime Foods", "PF-25031", "$2,500.00", "PAY"),
        ("PackRight", "PR-15007", "$1,500.00", "DEFER"),
        ("CleanPro", "CP-70019", "$700.00", "VERIFY"),
    )
    for supplier, invoice_number, amount, action in expected:
        card = next(card for card in cards if f"<h3>{supplier}</h3>" in card)
        assert invoice_number in card
        assert amount in card
        assert f"ACTUAL POLICY · {action}" in card
        assert "not OCR extraction" in card

    text = page_text(app)
    assert "Criticality-Aware Greedy v1 and the batch verifier ran" in text
    assert "Operator approval: NOT RECORDED" in text
    decision = app.radio(key="operator-decision-preview")
    assert decision.disabled
    assert list(decision.options) == ["APPROVE", "MODIFY", "REJECT"]
    assert app.button(key="run-verifier").disabled
    assert app.button(key="commit-procuregym").disabled


def test_unknownco_fails_closed_without_changing_obligations() -> None:
    app = boot()
    app.radio(key="top-route").set_value("Evidence & methods").run()
    app.selectbox(key="document-fixture-selector").set_value("UnknownCo").run()
    assert not app.exception

    failure = next(str(item.value) for item in app.error if "UnknownCo" in str(item.value))
    for phrase in (
        "never reaches canonical lookup",
        "activates no payable",
        "excluded from the $6,200 obligations",
        "did not run C2",
    ):
        assert phrase in failure
    assert "BLOCKED BY DESIGN — no canonical lookup" in page_text(app)


def test_guided_defaults_are_lazy_and_require_supplier_then_human_choice() -> None:
    _reset_cached_ryan_adapter_for_tests()
    app = boot()
    import procureagent.ui_adapters as adapters

    assert adapters._CACHED_RYAN_ADAPTER is None
    assert app.selectbox(key="eval-supplier").value is None
    assert app.button(key="eval-run-document-adapter").disabled
    app.selectbox(key="eval-supplier").set_value("Fresh Farms").run()
    assert not app.button(key="eval-run-document-adapter").disabled
    assert adapters._CACHED_RYAN_ADAPTER is None

    text = page_text(app)
    for phrase in (
        "Nothing runs until you ask it to",
        "Accounts Payable is money the business owes a supplier",
        "NOT RUN · click required",
        "The evaluation answer key is hidden from this workflow",
    ):
        assert phrase in text
    assert "Operator expected invoice number" not in text
    assert not app.text_input

    for stale in (
        "C4 is not live",
        "No generated receipt is bundled yet",
        "does not include receipt matching",
        "actual run pending",
        "PRD HYPOTHESIS",
    ):
        assert stale not in text


def test_mocked_guided_flow_keeps_every_explicit_gate_and_no_second_cash_hit() -> None:
    """Exercise each progressive screen with the bundled synthetic assets."""

    from tests.test_procure_ui_adapters import ReceiptOcr, analyzed_document
    import procureagent.ui_adapters as adapters

    analysis = analyzed_document()
    real_receipt_adapter = adapters.analyze_receipt_upload

    def fake_receipt_adapter(simulation, image_bytes, **kwargs):
        return real_receipt_adapter(
            simulation,
            image_bytes,
            ocr_engine=ReceiptOcr(),
            **kwargs,
        )

    with (
        patch.object(adapters, "analyze_invoice_upload", return_value=analysis),
        patch.object(adapters, "analyze_receipt_upload", side_effect=fake_receipt_adapter),
    ):
        # Step 1: supplier confirmation is required before perception can run.
        app = boot()
        app.selectbox(key="eval-supplier").set_value("Fresh Farms").run()
        app.button(key="eval-run-document-adapter").click().run()
        assert not app.exception
        assert "REVIEW_REQUIRED · LOW_MODEL_CONFIDENCE" in page_text(app)
        assert "Invoice number · invoice rule and LayoutLMv3 model" in page_text(app)
        assert "Amount · invoice amount rule" in page_text(app)
        assert app.radio(key="eval-document-review-choice").value is None
        assert app.button(key="eval-record-human-review").disabled
        assert "Strict exact" not in page_text(app)

        # Use a clean AppTest tree per screen because Streamlit 1.32 retains
        # removed widget nodes after an internal rerun. Session values are the
        # actual immutable objects produced by the preceding screen.
        review = AppTest.from_file(APP, default_timeout=20)
        review.session_state["eval-document-analysis"] = analysis
        review.run()
        review.radio(key="eval-document-review-choice").set_value(
            "Confirm the displayed invoice number"
        ).run()
        assert not review.button(key="eval-record-human-review").disabled
        review.button(key="eval-record-human-review").click().run()
        assert not review.exception
        prepared = review.session_state["eval-prepared"]
        human = review.session_state["eval-human-decision"]
        assert "Exact invoice found" in page_text(review)
        assert "Accounts Payable" in page_text(review)

        approve = AppTest.from_file(APP, default_timeout=20)
        approve.session_state["eval-document-analysis"] = analysis
        approve.session_state["eval-human-decision"] = human
        approve.session_state["eval-prepared"] = prepared
        approve.run()
        assert not approve.exception
        assert approve.checkbox(key="eval-operator-confirmation").value is False
        assert approve.button(key="eval-approve-batch").disabled
        approve.checkbox(key="eval-operator-confirmation").check().run()
        assert not approve.button(key="eval-approve-batch").disabled
        approve.button(key="eval-approve-batch").click().run()
        assert not approve.exception
        simulation = approve.session_state["eval-simulation"]
        assert "Debit" in page_text(approve)
        assert "Credit" in page_text(approve)
        assert "does not post this entry or deduct cash a second time" in page_text(approve)

        proof = AppTest.from_file(APP, default_timeout=20)
        proof.session_state["eval-document-analysis"] = analysis
        proof.session_state["eval-human-decision"] = human
        proof.session_state["eval-prepared"] = prepared
        proof.session_state["eval-simulation"] = simulation
        proof.run()
        assert not proof.exception
        proof.button(key="eval-run-receipt-adapter").click().run()
        assert not proof.exception
        assert "Exact payment-proof check" in page_text(proof)
        assert "Amount · receipt field rule" in page_text(proof)
        assert proof.checkbox(key="eval-proof-confirmation").value is False
        assert proof.button(key="eval-confirm-payment").disabled
        proof.checkbox(key="eval-proof-confirmation").check().run()
        assert not proof.button(key="eval-confirm-payment").disabled
        proof.button(key="eval-confirm-payment").click().run()
        assert not proof.exception
        assert "PAID_CONFIRMED in the simulated AP ledger" in page_text(proof)
        assert metric_values(proof, "Second cash deduction") == ["$0.00"]
        assert proof.button(key="eval-confirm-payment").disabled
        assert proof.button(key="eval-run-receipt-adapter").disabled
        assert proof.radio(key="eval-receipt-source").disabled
