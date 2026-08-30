"""Render-level acceptance checks for InvoiceAgent's controlled P0 UI."""

import hashlib
from dataclasses import replace
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
SESSION_SCHEMA = "invoiceagent-guided-v3-layoutlm-tokens-20260830"


def boot() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception
    return app


def boot_with_flow(**values) -> AppTest:
    """Seed current-schema workflow objects without the hot-reload guard clearing them."""

    app = AppTest.from_file(APP, default_timeout=20)
    app.session_state["invoiceagent-session-schema"] = SESSION_SCHEMA
    for key, value in values.items():
        app.session_state[key] = value
    app.run()
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
    assert "InvoiceAgent" in page_text(app)
    assert "Paper invoice in. Paid proof out." in page_text(app)
    assert "Sugar &amp; Spice Thai Restaurant" in page_text(app)
    assert "Scan the vendor invoice" in page_text(app)
    assert "Synthetic documents" in page_text(app)
    assert "No affiliation" in page_text(app)
    assert "Simulation only" in page_text(app)
    assert "Human controlled" in page_text(app)
    assert "No bank or ERP is connected" in page_text(app)

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
        "Accounts Payable: money the restaurant owes its vendor",
            "LayoutLMv3 + LoRA · adapted by Ryan",
        "The supervised specialist identifies the invoice number only",
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


def test_hot_reload_schema_guard_discards_only_stale_flow_objects() -> None:
    app = AppTest.from_file(APP, default_timeout=20)
    app.session_state["invoiceagent-session-schema"] = "old-contract-build"
    app.session_state["eval-document-analysis"] = object()
    app.session_state["top-route"] = "Guided demo"
    app.run()

    assert not app.exception
    assert "Scan the vendor invoice" in page_text(app)
    assert app.radio(key="top-route").value == "Guided demo"
    assert "eval-document-review-choice" not in {item.key for item in app.radio}


def test_receipt_in_invoice_slot_stops_with_friendly_message() -> None:
    from tests.test_procure_ui_adapters import analyzed_document
    import procureagent.ui_adapters as adapters
    from procureagent.ocr import TesseractOCR, ingest_image

    invoice_analysis = analyzed_document()
    receipt_path = Path(__file__).resolve().parents[1] / (
        "data/procureagent/assets/fresh_farms_payment_receipt.png"
    )
    receipt_image = ingest_image(
        receipt_path.read_bytes(), original_filename=receipt_path.name
    )
    receipt_ocr = TesseractOCR().run(receipt_image)
    assert "PAYMENT RECEIPT" in receipt_ocr.raw_text.upper()
    assert "PAID IN FULL" in receipt_ocr.raw_text.upper()
    receipt_model_run = replace(
        invoice_analysis.model_run,
        document_id=receipt_image.document_id,
        token_predictions=(),
    )
    receipt_analysis = replace(
        invoice_analysis,
        image=receipt_image,
        ocr=receipt_ocr,
        model_run=receipt_model_run,
    )
    with patch.object(adapters, "analyze_invoice_upload", return_value=receipt_analysis):
        app = boot()
        app.selectbox(key="eval-supplier").set_value("Fresh Farms").run()
        app.button(key="eval-run-document-adapter").click().run()

    assert not app.exception
    assert (
        "This appears to be a payment receipt—upload the supplier invoice first."
        in page_text(app)
    )
    assert "STOPPED SAFELY · receipt detected in invoice step" in page_text(app)
    assert "eval-document-review-choice" not in {item.key for item in app.radio}
    assert app.button(key="eval-run-document-adapter")


def test_bundled_invoice_is_not_misclassified_as_receipt() -> None:
    from tests.test_procure_ui_adapters import analyzed_document
    import procureagent.ui_adapters as adapters
    from procureagent.ocr import TesseractOCR, ingest_image

    invoice_path = Path(__file__).resolve().parents[1] / (
        "data/procureagent/assets/fresh_farms_invoice.png"
    )
    invoice_image = ingest_image(invoice_path.read_bytes(), original_filename=invoice_path.name)
    invoice_ocr = TesseractOCR().run(invoice_image)
    mocked = analyzed_document()
    analysis = replace(
        mocked,
        ocr=invoice_ocr,
        model_run=replace(mocked.model_run, token_predictions=()),
    )

    with patch.object(adapters, "analyze_invoice_upload", return_value=analysis):
        app = boot()
        app.selectbox(key="eval-supplier").set_value("Fresh Farms").run()
        app.button(key="eval-run-document-adapter").click().run()

    assert not app.exception
    assert "This appears to be a payment receipt" not in page_text(app)
    assert app.radio(key="eval-document-review-choice")


def test_rule_only_suggestion_cannot_be_confirmed_as_model_output() -> None:
    from tests.test_procure_ui_adapters import InvoiceOcr, NoCandidateModel
    import procureagent.ui_adapters as adapters

    analysis = adapters.analyze_invoice_upload(
        Path(__file__).resolve().parents[1]
        .joinpath("data/procureagent/assets/fresh_farms_invoice.png")
        .read_bytes(),
        filename="fresh_farms_invoice.png",
        ocr_engine=InvoiceOcr(),
        model_adapter=NoCandidateModel(),
    )
    review = boot_with_flow(**{"eval-document-analysis": analysis})

    assert not review.exception
    options = list(review.radio(key="eval-document-review-choice").options)
    assert options == ["Correct the invoice number", "Reject this document"]
    assert "Confirm the displayed invoice number" not in options
    assert review.button(key="eval-record-human-review").disabled
    assert "Confirm is unavailable because LayoutLMv3 found no candidate" in page_text(review)
    assert "use **Correct**" in page_text(review)


def test_unknown_correction_shows_friendly_fail_closed_message() -> None:
    from tests.test_procure_ui_adapters import analyzed_document

    review = boot_with_flow(**{"eval-document-analysis": analyzed_document()})
    review.radio(key="eval-document-review-choice").set_value(
        "Correct the invoice number"
    ).run()
    review.text_input(key="eval-corrected-reference").set_value("ZZ-99999").run()
    assert not review.button(key="eval-record-human-review").disabled
    review.button(key="eval-record-human-review").click().run()

    assert not review.exception
    assert "not in this demo's locked Accounts Payable records" in page_text(review)
    assert "eval-human-decision" not in review.session_state
    assert "eval-prepared" not in review.session_state


def test_receipt_id_only_view_is_clean_and_cannot_false_close_ap() -> None:
    from tests.test_procure_ui_adapters import receipt_id_only_analysis

    analysis, human, prepared, simulation, receipt = receipt_id_only_analysis()
    receipt_path = Path(__file__).resolve().parents[1] / (
        "data/procureagent/assets/fresh_farms_payment_receipt.png"
    )
    proof = boot_with_flow(
        **{
            "eval-document-analysis": analysis,
            "eval-human-decision": human,
            "eval-prepared": prepared,
            "eval-simulation": simulation,
            "eval-receipt-analysis": receipt,
            "eval-receipt-input-key": hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
        }
    )

    assert not proof.exception
    text = page_text(proof)
    assert "Receipt ID captured" in text
    assert "19729058 · grounded in the uploaded OCR evidence" in text
    assert "Manual matching details · why this ID cannot close AP yet" in [
        item.label for item in proof.expander
    ]
    assert "Payment proof incomplete · 5 required fields missing" in text
    assert "SIMULATED_PAYMENT_APPROVED" in text
    assert "no second cash entry or PAID_CONFIRMED status was created" in text
    assert "SAFE_REVIEW · reward -1.0" in text
    assert "VERIFIED_FULL_MATCH · reward 10.0" not in text
    assert metric_values(proof, "Receipt ID") == ["19729058"]
    assert metric_values(proof, "Grounding") == ["OCR MATCH"]
    assert proof.button(key="eval-confirm-payment").disabled
    assert "pa-token muted" in text
    assert "pa-token target receipt-id" in text


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
        assert "NOT_EVALUATED means the tokenizer truncated that word" in page_text(app)
        assert any(
            item.label == "LayoutLMv3 token evidence · 1/9 OCR words evaluated"
            for item in app.expander
        )
        assert app.radio(key="eval-document-review-choice").value is None
        assert app.button(key="eval-record-human-review").disabled
        assert "Strict exact" not in page_text(app)

        # Use a clean AppTest tree per screen because Streamlit 1.32 retains
        # removed widget nodes after an internal rerun. Session values are the
        # actual immutable objects produced by the preceding screen.
        review = boot_with_flow(**{"eval-document-analysis": analysis})
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

        approve = boot_with_flow(
            **{
                "eval-document-analysis": analysis,
                "eval-human-decision": human,
                "eval-prepared": prepared,
            }
        )
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

        proof = boot_with_flow(
            **{
                "eval-document-analysis": analysis,
                "eval-human-decision": human,
                "eval-prepared": prepared,
                "eval-simulation": simulation,
            }
        )
        assert not proof.exception
        proof.button(key="eval-run-receipt-adapter").click().run()
        assert not proof.exception
        assert "Exact payment-proof check" in page_text(proof)
        assert "Amount · receipt field rule" in page_text(proof)
        assert "RL-ready evaluation signal · no policy/model was trained" in page_text(proof)
        assert "VERIFIED_FULL_MATCH · reward 10.0 · action ACCEPT_MATCH" in page_text(proof)
        assert proof.checkbox(key="eval-proof-confirmation").value is False
        assert proof.button(key="eval-confirm-payment").disabled
        assert "eval-view-ap-history" not in {button.key for button in proof.button}
        proof.checkbox(key="eval-proof-confirmation").check().run()
        assert not proof.button(key="eval-confirm-payment").disabled
        proof.button(key="eval-confirm-payment").click().run()
        assert not proof.exception
        assert "PAID_CONFIRMED in the simulated ledger" in page_text(proof)
        assert metric_values(proof, "Second cash deduction") == ["$0.00"]
        assert proof.button(key="eval-confirm-payment").disabled
        assert proof.button(key="eval-run-receipt-adapter").disabled
        assert proof.radio(key="eval-receipt-source").disabled
        assert not proof.tabs
        history_button = proof.button(key="eval-view-ap-history")
        assert history_button.label == "Done — view AP history"

        history_button.click().run()
        assert not proof.exception
        assert len(proof.tabs) == 3
        assert [tab.label for tab in proof.tabs] == [
            "Open invoices (2)",
            "Paid · awaiting proof (1)",
            "Completed (1)",
        ]

        history_text = page_text(proof)
        for phrase in (
            "PackRight",
            "PR-15007",
            "OPEN · DEFER",
            "CleanPro",
            "CP-70019",
            "OPEN · VERIFY",
            "Prime Foods",
            "PF-25031",
            "PAID · AWAITING PROOF",
            "simulated_payment_approved",
            "Fresh Farms",
            "FF-10482",
            "COMPLETED · PROOF MATCHED",
            "paid_confirmed",
            "Auditable Fresh Farms journal component",
            "Accounts Payable — Fresh Farms",
            "RCPT-FF-10482",
            "2026-08-30",
            "synthetic_fixture_replay",
            "Receipt confirmation cash impact: $0.00",
            "Second cash hit: NO",
            "not full Net Working Capital (NWC)",
            "Accounts Receivable",
            "inventory valuation",
            "Accounting interpretation of the isolated simulated transition",
            "Accounts Payable — Prime Foods",
            "balanced batch $4,000.00",
        ):
            assert phrase in history_text
        assert metric_values(proof, "Cash after batch") == ["$1,000.00"]
        assert metric_values(proof, "Remaining open AP") == ["$2,200.00"]
        assert metric_values(proof, "Paid · awaiting proof") == ["$2,500.00"]
        assert metric_values(proof, "Completed invoices") == ["1"]


def test_source_citations_point_at_the_code_they_claim() -> None:
    """The evidence panel renders source by hardcoded line number.

    Editing an adapter shifts those ranges silently, so the panel would show the
    wrong function under a confident label. That is a credibility bug in the one
    panel built to prove "this is the real source", so pin it.
    """

    import re

    repo_root = Path(__file__).resolve().parents[1]
    app_text = (repo_root / "procure_app.py").read_text(encoding="utf-8")
    citations = re.findall(
        r'\("([^"]+)",\s*"(src/[^"]+\.py)",\s*(\d+),\s*(\d+)\)', app_text
    )
    assert citations, "expected the evidence panel to cite source ranges"

    for label, relative, start, end in citations:
        lines = (repo_root / relative).read_text(encoding="utf-8").splitlines()
        start, end = int(start), int(end)
        assert 0 < start <= end <= len(lines), f"{label}: {relative}:{start}-{end} out of range"
        head = lines[start - 1].strip()
        assert head.startswith(("def ", "class ", "@")), (
            f"{label} cites {relative}:{start}, which is {head!r} rather than a "
            "definition; the range has drifted"
        )


def _flow_through_receipt():
    """Drive the real adapters to a confirmed payment, returning session objects."""

    from tests.test_procure_ui_adapters import ReceiptOcr, analyzed_document
    from procureagent.contracts import DocumentReviewDecision, PaymentProofSource
    from procureagent.ui_adapters import (
        analyze_receipt_upload,
        approve_and_simulate,
        confirm_verified_payment,
        prepare_procurement,
        record_human_identity_decision,
    )

    analysis = analyzed_document()
    human = record_human_identity_decision(analysis, DocumentReviewDecision.CONFIRM)
    prepared = prepare_procurement(human)
    simulation = approve_and_simulate(prepared)
    receipt = analyze_receipt_upload(
        simulation,
        (Path(__file__).resolve().parents[1] / RECEIPT_REL).read_bytes(),
        filename="receipt.png",
        source=PaymentProofSource.SYNTHETIC_FIXTURE_REPLAY,
        provenance="test",
        ocr_engine=ReceiptOcr(),
    )
    confirmed = confirm_verified_payment(receipt)
    return {
        "eval-document-analysis": analysis,
        "eval-human-decision": human,
        "eval-prepared": prepared,
        "eval-simulation": simulation,
        "eval-receipt-analysis": receipt,
        "eval-confirmed-payment": confirmed,
    }


RECEIPT_REL = "data/procureagent/assets/fresh_farms_payment_receipt.png"


def test_step_five_continues_the_week_and_reject_does_not_advance_time() -> None:
    """The week keeps running after the first payment closes, still operator-gated."""

    week = boot_with_flow(**_flow_through_receipt())
    text = page_text(week)
    assert "Continue the week" in text
    assert "Proposal binds state version" in text

    week.radio(key="eval-operator-decision").set_value("REJECT").run()
    week.button(key="eval-reject-day").click().run()
    assert not week.exception
    after = page_text(week)
    assert "ProcureGym.step was never called" in after
    assert "(unchanged)" in after


def test_step_five_modify_into_an_over_budget_pay_is_blocked() -> None:
    """AC-09 on stage: a modified batch must clear the verifier again."""

    week = boot_with_flow(**_flow_through_receipt())
    week.radio(key="eval-operator-decision").set_value("MODIFY").run()
    week.selectbox(key="eval-modify-packright").set_value("PAY").run()
    week.button(key="eval-apply-modify").click().run()
    assert not week.exception

    text = page_text(week)
    assert "BLOCKED" in text
    assert "OVER_BUDGET" in text
    week.radio(key="eval-operator-decision").set_value("APPROVE").run()
    assert week.button(key="eval-approve-day").disabled


def test_step_five_offers_the_revenue_week_because_the_locked_one_is_decided() -> None:
    """The frozen scenario has no timing decision left; say so and offer one."""

    week = boot_with_flow(**_flow_through_receipt())
    text = page_text(week)
    assert "PackRight's $1,500 never" in text or "never becomes affordable" in text

    week.button(key="eval-week-cashflow").click().run()
    assert not week.exception
    episode = week.session_state["eval-episode"]
    assert episode.environment.scenario.scenario_id == "restaurant_cashflow_v1"
    assert episode.environment.scenario.daily_cash_inflow_minor == 25_000
    assert "Payment timing · operator versus bounded oracle" in page_text(week)
