"""Focused tests for the UI orchestration safety gates."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

from invoiceagent.extraction import TokenPrediction
from procureagent.contracts import (
    BoundingBox,
    DocumentReviewDecision,
    InvoiceNumberCandidate,
    InvoicePaymentStatus,
    PaymentProofSource,
    VerifierResult,
)
from procureagent.document import (
    InvoiceModelRun,
    InvoiceModelRunStatus,
    ModelInvoiceCandidate,
    RyanInvoiceAdapter,
)
from procureagent.ocr import OcrResult, OcrStatus, OcrWord, PixelBox
from procureagent.ui_adapters import (
    UiFlowError,
    _reset_cached_ryan_adapter_for_tests,
    analyze_invoice_upload,
    analyze_receipt_upload,
    approve_and_simulate,
    confirm_verified_payment,
    get_cached_ryan_adapter,
    prepare_procurement,
    record_human_identity_decision,
)


ROOT = Path(__file__).resolve().parents[1]
INVOICE_ASSET = ROOT / "data/procureagent/assets/fresh_farms_invoice.png"
RECEIPT_ASSET = ROOT / "data/procureagent/assets/fresh_farms_payment_receipt.png"


def make_ocr(image, lines, *, status=OcrStatus.SUCCESS):
    if status is not OcrStatus.SUCCESS:
        return OcrResult(
            document_id=image.document_id,
            status=status,
            words=(),
            raw_text="",
            language="eng",
            engine="fake_tesseract",
            engine_version="missing",
            runtime_ms=Decimal("0.1"),
            error_code="TESSERACT_NOT_FOUND",
            error_message="not installed",
        )
    words = []
    for line_number, line in enumerate(lines, start=1):
        for word_number, token in enumerate(line, start=1):
            sequence = len(words)
            x0 = 10 + (word_number - 1) * 110
            y0 = 10 + (line_number - 1) * 45
            pixel = PixelBox(x0, y0, x0 + 90, y0 + 30)
            normalized = BoundingBox(
                1000 * pixel.x0 // image.width,
                1000 * pixel.y0 // image.height,
                (1000 * pixel.x1 + image.width - 1) // image.width,
                (1000 * pixel.y1 + image.height - 1) // image.height,
            )
            words.append(
                OcrWord(
                    sequence=sequence,
                    text=token,
                    confidence=Decimal("0.95"),
                    pixel_box=pixel,
                    normalized_box=normalized,
                    page=1,
                    block=1,
                    paragraph=1,
                    line=line_number,
                    word=word_number,
                )
            )
    return OcrResult(
        document_id=image.document_id,
        status=OcrStatus.SUCCESS,
        words=tuple(words),
        raw_text="\n".join(" ".join(line) for line in lines),
        language="eng",
        engine="fake_tesseract",
        engine_version="5.test",
        runtime_ms=Decimal("3.2"),
    )


class InvoiceOcr:
    def run(self, image):
        return make_ocr(
            image,
            (
                ("Invoice", "No:", "FF-10482"),
                ("Supplier", "ID:", "fresh_farms"),
                ("Total", "USD", "$1,500.00"),
            ),
        )


class LowScoreExactModel:
    def run(self, image, ocr):
        word = ocr.words[2]
        evidence = ModelInvoiceCandidate(
            candidate=InvoiceNumberCandidate(
                invoice_number="FF-10482",
                entity_confidence=Decimal("0.6467026472091675"),
                grounded_in_ocr=True,
                evidence_tokens=(word.text,),
                evidence_boxes=(word.normalized_box,),
            ),
            word_indices=(2,),
            minimum_confidence=Decimal("0.6467026472091675"),
            mean_confidence=Decimal("0.6467026472091675"),
            mean_margin=Decimal("0.3744482994079590"),
        )
        return InvoiceModelRun(
            document_id=image.document_id,
            status=InvoiceModelRunStatus.SUCCESS,
            candidates=(evidence,),
            model_version="ryanznie/test-fixture on cpu",
            latency_ms=Decimal("12.3"),
            token_predictions=(
                TokenPrediction(
                    word_index=2,
                    word=word.text,
                    box=(
                        word.normalized_box.x0,
                        word.normalized_box.y0,
                        word.normalized_box.x1,
                        word.normalized_box.y1,
                    ),
                    label="B-INVOICE_ID",
                    confidence=Decimal("0.6467026472091675"),
                    margin=Decimal("0.3744482994079590"),
                ),
            ),
        )


class NoCandidateModel:
    def run(self, image, _ocr):
        return InvoiceModelRun(
            document_id=image.document_id,
            status=InvoiceModelRunStatus.NO_CANDIDATE,
            candidates=(),
            model_version="ryanznie/test-fixture on cpu",
            latency_ms=Decimal("4.2"),
        )


class ReceiptOcr:
    def run(self, image):
        return make_ocr(
            image,
            (
                ("Receipt", "ID:", "RCPT-FF-10482"),
                ("Supplier:", "Fresh", "Farms"),
                ("Invoice", "Number:", "FF-10482"),
                ("Paid", "Date:", "2026-08-30"),
                ("Currency:", "USD"),
                ("Amount", "Paid:", "$1,500.00"),
            ),
        )


class ReceiptIdOnlyOcr:
    def run(self, image):
        return make_ocr(
            image,
            (
                ("RECEIPT", "Receipt", "No:", "19729058"),
                ("Date", ":", "01/02/2018"),
                ("Amount", "(RM)", "12.40"),
            ),
        )


def analyzed_document():
    return analyze_invoice_upload(
        INVOICE_ASSET.read_bytes(),
        filename=INVOICE_ASSET.name,
        ocr_engine=InvoiceOcr(),
        model_adapter=LowScoreExactModel(),
    )


def receipt_id_only_analysis():
    analysis = analyzed_document()
    human = record_human_identity_decision(analysis, DocumentReviewDecision.CONFIRM)
    prepared = prepare_procurement(human)
    simulation = approve_and_simulate(prepared)
    receipt = analyze_receipt_upload(
        simulation,
        RECEIPT_ASSET.read_bytes(),
        filename="external-receipt.png",
        source=PaymentProofSource.OPERATOR_UPLOAD,
        provenance="operator_upload:test_receipt_id_only.png",
        ocr_engine=ReceiptIdOnlyOcr(),
    )
    return analysis, human, prepared, simulation, receipt


def test_low_score_exact_document_still_requires_explicit_human_review():
    analysis = analyzed_document()
    assert analysis.strict_exact is True
    assert analysis.rule_candidates[0].invoice_number == "FF-10482"
    assert analysis.gate.status.value == "REVIEW_REQUIRED"
    assert analysis.gate.reason_codes == ("LOW_MODEL_CONFIDENCE",)
    assert analysis.gate.verified_identity is None

    rejected = record_human_identity_decision(
        analysis, DocumentReviewDecision.REJECT
    )
    assert rejected.may_activate_lookup is False
    with pytest.raises(UiFlowError, match="explicit human"):
        prepare_procurement(rejected)


def test_unknown_human_correction_fails_before_lookup_or_placeholder_creation():
    analysis = analyzed_document()
    with pytest.raises(UiFlowError, match="absent from locked lookup"):
        record_human_identity_decision(
            analysis,
            DocumentReviewDecision.CORRECT,
            corrected_invoice_number="ZZ-99999",
        )


def test_rule_only_result_cannot_be_confirmed_as_model_evidence():
    analysis = analyze_invoice_upload(
        INVOICE_ASSET.read_bytes(),
        filename=INVOICE_ASSET.name,
        ocr_engine=InvoiceOcr(),
        model_adapter=NoCandidateModel(),
    )
    assert analysis.rule_candidates[0].invoice_number == "FF-10482"
    assert analysis.selected_model_candidate is None
    assert "MODEL_CANDIDATE_MISSING" in analysis.gate.reason_codes
    with pytest.raises(UiFlowError, match="displayed model candidate"):
        record_human_identity_decision(analysis, DocumentReviewDecision.CONFIRM)


def test_full_controlled_flow_needs_human_then_operator_then_verified_proof():
    analysis = analyzed_document()
    human = record_human_identity_decision(
        analysis, DocumentReviewDecision.CONFIRM
    )
    prepared = prepare_procurement(human)
    assert prepared.looked_up_invoice.invoice_number == "FF-10482"
    assert prepared.verification.result is VerifierResult.REQUIRES_OPERATOR
    assert prepared.scenario.initial_state.cash_minor == 500_000
    assert prepared.scenario.initial_state.invoices[0].payment_status is (
        InvoicePaymentStatus.UNPAID
    )

    simulation = approve_and_simulate(prepared)
    assert simulation.state_after.cash_minor == 100_000
    assert simulation.info["simulation_only"] is True
    assert simulation.state_after.invoices[0].payment_status is (
        InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED
    )

    receipt = analyze_receipt_upload(
        simulation,
        RECEIPT_ASSET.read_bytes(),
        filename=RECEIPT_ASSET.name,
        source=PaymentProofSource.SYNTHETIC_FIXTURE_REPLAY,
        provenance="bundled_deterministic_svg_fixture",
        ocr_engine=ReceiptOcr(),
    )
    assert receipt.parsed.status.value == "READY_FOR_PROOF"
    assert receipt.proof_gate.closes_obligation is True
    assert receipt.simulation.environment.state.invoices[0].payment_status is (
        InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED
    )

    confirmed = confirm_verified_payment(receipt)
    assert confirmed.payment_status is InvoicePaymentStatus.PAID_CONFIRMED
    assert confirmed.state_after.cash_minor == 100_000
    assert confirmed.state_after.day == 1


def test_receipt_cannot_run_before_operator_approved_simulation():
    with pytest.raises(UiFlowError, match="approved simulated step"):
        analyze_receipt_upload(  # type: ignore[arg-type]
            object(),
            RECEIPT_ASSET.read_bytes(),
            filename=RECEIPT_ASSET.name,
            source=PaymentProofSource.OPERATOR_UPLOAD,
            provenance="test",
            ocr_engine=ReceiptOcr(),
        )


def test_receipt_id_only_is_captured_but_cannot_close_accounts_payable():
    _analysis, _human, _prepared, simulation, receipt = receipt_id_only_analysis()

    assert receipt.parsed.receipt_id == "19729058"
    assert receipt.parsed.status.value == "REVIEW_REQUIRED"
    assert "UNUSED_RECEIPT_ID" in receipt.proof_gate.checks_passed
    assert "MISSING_SUPPLIER" in receipt.proof_gate.reason_codes
    assert "MISSING_INVOICE_NUMBER" in receipt.proof_gate.reason_codes
    assert "MISSING_AMOUNT_MINOR" in receipt.proof_gate.reason_codes
    assert not receipt.proof_gate.closes_obligation
    assert simulation.environment.state.cash_minor == 100_000
    assert simulation.environment.state.invoices[0].payment_status is (
        InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED
    )
    with pytest.raises(UiFlowError, match="verified full payment proof"):
        confirm_verified_payment(receipt)


def test_failed_ocr_does_not_initialize_the_lazy_model():
    calls = []

    def factory():
        calls.append(True)
        raise AssertionError("model factory must not run after failed OCR")

    adapter = RyanInvoiceAdapter(extractor_factory=factory)

    class MissingOcr:
        def run(self, image):
            return make_ocr(image, (), status=OcrStatus.UNAVAILABLE)

    result = analyze_invoice_upload(
        INVOICE_ASSET.read_bytes(),
        filename=INVOICE_ASSET.name,
        ocr_engine=MissingOcr(),
        model_adapter=adapter,
    )
    assert calls == []
    assert result.ocr.status is OcrStatus.UNAVAILABLE
    assert result.model_run.status is InvoiceModelRunStatus.FAILED
    assert result.gate.status.value == "REVIEW_REQUIRED"


def test_cached_ryan_adapter_is_singleton_under_concurrent_access(monkeypatch):
    import procureagent.ui_adapters as adapters

    created = []

    class LightweightAdapter:
        pass

    def factory():
        created.append(True)
        return LightweightAdapter()

    _reset_cached_ryan_adapter_for_tests()
    monkeypatch.setattr(adapters, "RyanInvoiceAdapter", factory)
    with ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(lambda _: get_cached_ryan_adapter(), range(24)))
    assert len(created) == 1
    assert len({id(item) for item in instances}) == 1
    _reset_cached_ryan_adapter_for_tests()
