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


EPISODE_KEY_FOR_TESTS = "eval-episode"


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


def test_app_renders_actual_overview_without_claiming_live_perception() -> None:
    app = boot()
    assert [tab.label for tab in app.tabs] == [
        "1 · Restaurant state",
        "2 · Document evidence",
        "3 · Daily batch",
        "4 · ProcureGym",
        "5 · /eval recording",
        "6 · Task status",
    ]

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
    for phrase in (
        "OCR and the local model run only after an explicit /eval click",
        "Criticality-Aware Greedy v1 and the batch verifier ran",
        "Operator approval: NOT RECORDED",
        "Actual deterministic benchmark complete",
        "Three axes stay separate",
        "7 hand-authored synthetic development rows",
        "all 7 context bins also appeared in training",
        "no frozen test or generalization claim",
        "David / @cheezburgerz",
    ):
        assert phrase in text

    decision = app.radio(key="operator-decision-preview")
    assert decision.disabled
    assert list(decision.options) == ["APPROVE", "MODIFY", "REJECT"]
    assert app.button(key="run-verifier").disabled
    assert app.button(key="commit-procuregym").disabled
    assert metric_values(app, "Criticality-aware reward") == ["-12.400"]
    assert metric_values(app, "EDF reward") == ["-68.800"]
    assert metric_values(app, "Oracle reward") == ["-12.400"]


def test_unknownco_fails_closed_without_changing_obligations() -> None:
    app = boot()
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
    assert dollars(metric_values(app, "Supplier obligations")[0]) == Decimal("6200.00")

    grid = next(
        str(item.value)
        for item in app.markdown
        if '<div class="pa-grid">' in str(item.value)
    )
    assert "UnknownCo" not in grid


def test_eval_defaults_are_lazy_and_every_mutation_gate_is_locked() -> None:
    _reset_cached_ryan_adapter_for_tests()
    app = boot()
    import procureagent.ui_adapters as adapters

    assert adapters._CACHED_RYAN_ADAPTER is None
    assert not app.button(key="eval-run-document-adapter").disabled
    assert app.button(key="eval-record-human-review").disabled
    assert app.button(key="eval-approve-batch").disabled
    assert app.button(key="eval-run-receipt-adapter").disabled
    assert app.button(key="eval-confirm-payment").disabled

    text = page_text(app)
    for phrase in (
        "Nothing heavy runs when this page renders",
        "AP in plain English",
        "NOT RUN · click required",
        "lookup blocked",
        "restaurant state unchanged",
        "Receipt proof is locked until operator-approved ProcureGym",
        "No bank is connected",
    ):
        assert phrase in text
    assert app.radio(key="eval-receipt-source").options[0] == (
        "Bundled deterministic receipt PNG"
    )

    for stale in (
        "C4 is not live",
        "No generated receipt is bundled yet",
        "does not include receipt matching",
        "actual run pending",
        "PRD HYPOTHESIS",
    ):
        assert stale not in text


def test_eval_mocked_perception_records_all_human_and_simulation_gates() -> None:
    """AppTest 1.32 cannot upload bytes, so exercise the bundled asset route."""

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
        app = boot()
        app.button(key="eval-run-document-adapter").click().run()
        assert not app.exception
        assert "REVIEW_REQUIRED · LOW_MODEL_CONFIDENCE" in page_text(app)
        assert "Strict exact: YES" in page_text(app)
        assert not app.button(key="eval-record-human-review").disabled
        assert app.button(key="eval-approve-batch").disabled

        app.button(key="eval-record-human-review").click().run()
        assert not app.exception
        assert "Composite lookup activated" in page_text(app)
        assert "Prime Foods, PackRight, and CleanPro are locked fixture/replay identities" in page_text(app)
        assert not app.button(key="eval-approve-batch").disabled
        assert app.button(key="eval-run-receipt-adapter").disabled

        app.button(key="eval-approve-batch").click().run()
        assert not app.exception
        assert "simulation_only=True" in page_text(app)
        assert "no real money moved" in page_text(app)
        assert not app.button(key="eval-run-receipt-adapter").disabled

        app.button(key="eval-run-receipt-adapter").click().run()
        assert not app.exception
        assert "Full-payment proof gate" in page_text(app)
        assert "Verified full proof is ready" in page_text(app)
        assert not app.button(key="eval-confirm-payment").disabled

        app.button(key="eval-confirm-payment").click().run()
        assert not app.exception
        assert "PAID_CONFIRMED in the simulated AP ledger" in page_text(app)
        app.run()
        assert app.button(key="eval-confirm-payment").disabled
        assert app.button(key="eval-run-receipt-adapter").disabled
        assert app.radio(key="eval-receipt-source").disabled

        # Even a test harness forcing events on disabled widgets cannot clear
        # the immutable confirmed result or re-enter the proof adapter.
        app.radio(key="eval-receipt-source").set_value("Upload PNG/JPEG").run()
        app.button(key="eval-run-receipt-adapter").click().run()
        assert not app.exception
        assert not any("Receipt pipeline failed closed" in str(item.value) for item in app.error)
        assert "PAID_CONFIRMED in the simulated AP ledger" in page_text(app)
        assert app.button(key="eval-run-receipt-adapter").disabled


def test_eval_day_console_is_operator_gated_across_the_whole_episode() -> None:
    """REJECT must not advance time, and only APPROVE may reach the horizon."""

    from tests.test_procure_ui_adapters import analyzed_document
    import procureagent.ui_adapters as adapters

    with patch.object(
        adapters, "analyze_invoice_upload", return_value=analyzed_document()
    ):
        app = boot()
        assert "Day 0 proposal" in page_text(app)
        assert app.button(key="eval-approve-batch").disabled

        app.button(key="eval-run-document-adapter").click().run()
        app.button(key="eval-record-human-review").click().run()
        assert not app.exception
        assert "Proposal binds state version" in page_text(app)
        assert not app.button(key="eval-approve-batch").disabled

        # REJECT changes nothing and does not advance the day.
        app.radio(key="eval-operator-decision").set_value("REJECT").run()
        app.button(key="eval-reject-batch").click().run()
        assert not app.exception
        text = page_text(app)
        assert "ProcureGym.step was never called" in text
        assert "(unchanged)" in text

        # APPROVE is the only thing that advances the simulated day.
        app.radio(key="eval-operator-decision").set_value("APPROVE").run()
        approvals = 0
        while not app.button(key="eval-approve-batch").disabled and approvals < 10:
            app.button(key="eval-approve-batch").click().run()
            assert not app.exception
            approvals += 1
            if "Episode complete" in page_text(app):
                break

        assert approvals == 7, "seven explicit approvals should reach the horizon"
        final = page_text(app)
        assert "Episode complete" in final
        assert "TRUNCATED" in final


def test_eval_modify_into_an_over_budget_pay_is_blocked_and_cannot_step() -> None:
    """AC-09 on stage: a modified batch must clear the verifier again."""

    from tests.test_procure_ui_adapters import analyzed_document
    import procureagent.ui_adapters as adapters

    with patch.object(
        adapters, "analyze_invoice_upload", return_value=analyzed_document()
    ):
        app = boot()
        app.button(key="eval-run-document-adapter").click().run()
        app.button(key="eval-record-human-review").click().run()
        app.button(key="eval-approve-batch").click().run()
        assert not app.exception

        # Day 1: PackRight is $1,500 against $1,000 of cash.
        app.radio(key="eval-operator-decision").set_value("MODIFY").run()
        app.selectbox(key="eval-modify-packright").set_value("PAY").run()
        app.button(key="eval-apply-modify").click().run()
        assert not app.exception

        text = page_text(app)
        assert "BLOCKED" in text
        assert "OVER_BUDGET" in text
        assert "OPERATOR_MODIFIED" in text

        app.radio(key="eval-operator-decision").set_value("APPROVE").run()
        assert app.button(key="eval-approve-batch").disabled
        assert "Proposal binds state version" in page_text(app)


def test_reset_recording_flow_does_not_rewind_the_episode() -> None:
    """Clearing the document steps must not silently rewind the restaurant."""

    from tests.test_procure_ui_adapters import analyzed_document
    import procureagent.ui_adapters as adapters

    with patch.object(
        adapters, "analyze_invoice_upload", return_value=analyzed_document()
    ):
        app = boot()
        app.button(key="eval-run-document-adapter").click().run()
        app.button(key="eval-record-human-review").click().run()
        app.button(key="eval-approve-batch").click().run()
        assert not app.exception

        app.button(key="eval-reset-flow").click().run()
        assert not app.exception
        assert "Proposal binds state version" in page_text(app)

        app.button(key="eval-restart-episode").click().run()
        assert not app.exception
        assert "Day 0 proposal" in page_text(app)


def test_eval_cashflow_scenario_lets_the_agent_choose_a_payment_day() -> None:
    """The 'when' decision must be reachable from the UI, not just headless."""

    from tests.test_procure_ui_adapters import analyzed_document
    import procureagent.ui_adapters as adapters

    with patch.object(
        adapters, "analyze_invoice_upload", return_value=analyzed_document()
    ):
        app = boot()
        app.radio(key="eval-episode-scenario-choice").set_value(
            "Cash-flow · restaurant_cashflow_v1"
        ).run()
        assert not app.exception
        assert "restaurant_cashflow_v1" in page_text(app)

        app.button(key="eval-run-document-adapter").click().run()
        app.button(key="eval-record-human-review").click().run()
        assert not app.exception

        # Day 0 pays the two critical suppliers; PackRight cannot be afforded.
        app.button(key="eval-approve-batch").click().run()
        # Day 1 still cannot afford it.
        app.button(key="eval-approve-batch").click().run()
        assert not app.exception
        assert "PackRight" in page_text(app)

        # Day 2: revenue has accumulated and the agent now proposes paying it.
        text = page_text(app)
        assert "day 2" in text.lower()
        app.button(key="eval-approve-batch").click().run()
        assert not app.exception

        episode = app.session_state[EPISODE_KEY_FOR_TESTS]
        packright = next(
            invoice
            for invoice in episode.environment.state.invoices
            if invoice.supplier_id == "packright"
        )
        assert packright.payment_status.value == "simulated_payment_approved"
