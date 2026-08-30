from dataclasses import replace
from decimal import Decimal
import struct
from types import SimpleNamespace
import zlib

import pytest

from invoiceagent.extraction import EntitySpan, InvoiceNumberResult, TokenPrediction
from procureagent.contracts import (
    BoundingBox,
    ContractValidationError,
    DocumentStatus,
    InvoiceIdentity,
    InvoiceNumberCandidate,
)
from procureagent.document import (
    AnchoredInvoiceCandidate,
    InvoiceModelRun,
    InvoiceModelRunStatus,
    ModelInvoiceCandidate,
    RyanInvoiceAdapter,
    align_model_token_predictions,
    anchored_invoice_candidates,
    gate_document_identity,
    to_ryan_ocr,
)
from procureagent.ocr import (
    OcrResult,
    OcrStatus,
    OcrWord,
    PixelBox,
    ingest_image,
    normalize_pixel_box,
)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _png() -> bytes:
    width = height = 100
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(
        b"IDAT", zlib.compress(scanlines)
    ) + _chunk(b"IEND", b"")


def make_ocr(tokens=("Invoice", "No", "INV", "-", "204", "Total", "500.00")):
    image = ingest_image(_png())
    words = []
    for index, token in enumerate(tokens):
        pixel = PixelBox(index * 10, 10, index * 10 + 8, 20)
        words.append(
            OcrWord(
                sequence=index,
                text=token,
                confidence=Decimal("0.95"),
                pixel_box=pixel,
                normalized_box=normalize_pixel_box(pixel, 100, 100),
                page=1,
                block=1,
                paragraph=1,
                line=1,
                word=index + 1,
            )
        )
    return image, OcrResult(
        document_id=image.document_id,
        status=OcrStatus.SUCCESS,
        words=tuple(words),
        raw_text=" ".join(tokens),
        language="eng",
        engine="tesseract_local",
        engine_version="tesseract test",
        runtime_ms=Decimal("2.0"),
    )


def test_anchored_rule_returns_grounded_token_and_box_evidence():
    _, ocr = make_ocr()
    candidates = anchored_invoice_candidates(ocr)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.invoice_number == "INV-204"
    assert candidate.word_indices == (2, 3, 4)
    assert candidate.evidence_tokens == ("INV", "-", "204")
    assert candidate.evidence_boxes == tuple(
        word.normalized_box for word in ocr.words[2:5]
    )
    assert candidate.grounded_in_ocr


def test_anchored_rule_preserves_distinct_candidates_for_gate_ambiguity():
    _, ocr = make_ocr(("Invoice", ":", "INV-1", "Invoice", ":", "INV-2"))
    assert tuple(item.invoice_number for item in anchored_invoice_candidates(ocr)) == (
        "INV-1",
        "INV-2",
    )


def test_procure_ocr_translation_matches_ryan_adapter_contract():
    _, ocr = make_ocr()
    translated = to_ryan_ocr(ocr)
    assert translated.words == ocr.ordered_text
    assert translated.boxes == tuple(
        (
            word.normalized_box.x0,
            word.normalized_box.y0,
            word.normalized_box.x1,
            word.normalized_box.y1,
        )
        for word in ocr.words
    )
    assert translated.quality == Decimal("0.95")
    assert translated.engine == "tesseract_local:tesseract test"


class FakeExtractor:
    adapter_model = "ryanznie/test-adapter"
    device = "cpu"

    def __init__(self, ocr, *, value="INV-204", fail_load=False):
        self.ocr = ocr
        self.value = value
        self.fail_load = fail_load
        self.load_calls = 0
        self.predict_calls = 0

    def load(self):
        self.load_calls += 1
        if self.fail_load:
            raise RuntimeError("offline model unavailable")

    def predict(self, image, translated):
        self.predict_calls += 1
        assert image == "decoded-image"
        assert translated.words == self.ocr.ordered_text
        boxes = tuple(
            (
                word.normalized_box.x0,
                word.normalized_box.y0,
                word.normalized_box.x1,
                word.normalized_box.y1,
            )
            for word in self.ocr.words[2:5]
        )
        span = EntitySpan(
            value=self.value,
            word_indices=(2, 3, 4),
            boxes=boxes,
            token_confidences=(Decimal("0.96"), Decimal("0.94"), Decimal("0.95")),
            token_margins=(Decimal("0.60"), Decimal("0.50"), Decimal("0.55")),
        )
        return InvoiceNumberResult((span,), "ryanznie/test-adapter on cpu", Decimal("12.3"))


def test_ryan_adapter_loads_lazily_once_and_preserves_actual_evidence():
    image, ocr = make_ocr()
    fake = FakeExtractor(ocr)
    adapter = RyanInvoiceAdapter(
        extractor=fake, image_decoder=lambda image: "decoded-image"
    )
    assert not adapter.loaded
    first = adapter.run(image, ocr)
    second = adapter.run(image, ocr)
    assert adapter.loaded
    assert fake.load_calls == 1
    assert fake.predict_calls == 2
    assert first.status is InvoiceModelRunStatus.SUCCESS
    assert first.model_version == "ryanznie/test-adapter on cpu"
    assert first.latency_ms == Decimal("12.3")
    evidence = first.candidates[0]
    assert evidence.candidate.invoice_number == "INV-204"
    assert evidence.candidate.grounded_in_ocr
    assert evidence.candidate.evidence_tokens == ("INV", "-", "204")
    assert evidence.minimum_confidence == Decimal("0.94")
    assert evidence.mean_margin == Decimal("0.55")
    assert second.candidates == first.candidates


def test_pinned_model_provenance_fits_the_public_proposal_contract():
    _, ocr = make_ocr()
    run = model_run(ocr)
    pinned = replace(
        run,
        model_version=(
            "ryanznie/layoutlmv3-lora-invoice-number@"
            "7dc28f5a3b14aa100ba432ee1b0a6cac6c7b2c5c "
            "(base microsoft/layoutlmv3-base@"
            "cfbbbff0762e6aab37086fdd4739ad14fe7d5db4) on mps"
        ),
    )

    proposal = pinned.to_proposal(
        supplier_id="fresh_farms",
        supplier_confirmed=True,
    )

    assert proposal.model_version == pinned.model_version


def test_token_predictions_align_by_explicit_ocr_index_text_and_box():
    _, ocr = make_ocr()
    word = ocr.words[2]
    prediction = TokenPrediction(
        word_index=2,
        word=word.text,
        box=(
            word.normalized_box.x0,
            word.normalized_box.y0,
            word.normalized_box.x1,
            word.normalized_box.y1,
        ),
        label="B-INVOICE_ID",
        confidence=Decimal("0.96"),
        margin=Decimal("0.60"),
    )
    run = replace(model_run(ocr), token_predictions=(prediction,))

    aligned = align_model_token_predictions(run, ocr)

    assert len(aligned) == len(ocr.words)
    assert aligned[2] == prediction
    assert all(item is None for index, item in enumerate(aligned) if index != 2)


@pytest.mark.parametrize("changed_field", ("index", "text", "box"))
def test_token_prediction_alignment_fails_closed(changed_field):
    _, ocr = make_ocr()
    word = ocr.words[2]
    values = {
        "word_index": 2,
        "word": word.text,
        "box": (
            word.normalized_box.x0,
            word.normalized_box.y0,
            word.normalized_box.x1,
            word.normalized_box.y1,
        ),
        "label": "B-INVOICE_ID",
        "confidence": Decimal("0.96"),
        "margin": Decimal("0.60"),
    }
    if changed_field == "index":
        values["word_index"] = len(ocr.words)
    elif changed_field == "text":
        values["word"] = "NOT-IN-OCR"
    else:
        values["box"] = (0, 0, 1, 1)
    prediction = TokenPrediction(**values)
    run = replace(model_run(ocr), token_predictions=(prediction,))

    with pytest.raises(ContractValidationError, match="token prediction"):
        align_model_token_predictions(run, ocr)


def test_ryan_adapter_records_ungrounded_model_output_instead_of_trusting_indices():
    image, ocr = make_ocr()
    result = RyanInvoiceAdapter(
        extractor=FakeExtractor(ocr, value="FAKE-999"),
        image_decoder=lambda image: "decoded-image",
    ).run(image, ocr)
    assert result.status is InvoiceModelRunStatus.SUCCESS
    assert not result.candidates[0].candidate.grounded_in_ocr


def test_ryan_adapter_reports_load_failure_and_does_not_download_at_construction():
    image, ocr = make_ocr()
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return FakeExtractor(ocr, fail_load=True)

    adapter = RyanInvoiceAdapter(
        extractor_factory=factory, image_decoder=lambda image: "decoded-image"
    )
    assert factory_calls == []
    result = adapter.run(image, ocr)
    assert factory_calls == [True]
    assert result.status is InvoiceModelRunStatus.FAILED
    assert result.error_code == "MODEL_RUN_FAILED"
    assert "RuntimeError" in result.error_message
    assert not adapter.loaded


def model_run(ocr, *, value="INV-204", grounded=True, confidence="0.94", margin="0.55"):
    candidate = InvoiceNumberCandidate(
        invoice_number=value,
        entity_confidence=confidence,
        grounded_in_ocr=grounded,
        evidence_tokens=("INV", "-", "204"),
        evidence_boxes=tuple(word.normalized_box for word in ocr.words[2:5]),
    )
    evidence = ModelInvoiceCandidate(
        candidate=candidate,
        word_indices=(2, 3, 4),
        minimum_confidence=Decimal(confidence),
        mean_confidence=Decimal(confidence),
        mean_margin=Decimal(margin),
    )
    return InvoiceModelRun(
        document_id=ocr.document_id,
        status=InvoiceModelRunStatus.SUCCESS,
        candidates=(evidence,),
        model_version="ryanznie/test on cpu",
        latency_ms=Decimal("12.3"),
    )


def gate(ocr, rules, run, **overrides):
    values = dict(
        document_id=ocr.document_id,
        supplier_id="fresh_farms",
        supplier_confirmed=True,
        known_supplier_ids=("fresh_farms", "prime_foods"),
        ocr=ocr,
        rule_candidates=rules,
        model_run=run,
    )
    values.update(overrides)
    return gate_document_identity(**values)


def test_document_gate_confirms_one_grounded_rule_model_agreement():
    _, ocr = make_ocr()
    rules = anchored_invoice_candidates(ocr)
    result = gate(ocr, rules, model_run(ocr))
    assert result.status is DocumentStatus.CONFIRMED
    assert result.reason_codes == ()
    assert result.may_activate_lookup
    assert result.verified_identity.identity == InvoiceIdentity("fresh_farms", "INV-204")


@pytest.mark.parametrize(
    ("case", "reason"),
    (
        ("missing_rule", "RULE_CANDIDATE_MISSING"),
        ("ambiguous_rule", "AMBIGUOUS_RULE_CANDIDATES"),
        ("missing_model", "MODEL_RESULT_MISSING"),
        ("ungrounded_model", "UNGROUNDED_MODEL_CANDIDATE"),
        ("disagreement", "RULE_MODEL_DISAGREEMENT"),
        ("low_confidence", "LOW_MODEL_CONFIDENCE"),
        ("low_margin", "LOW_MODEL_MARGIN"),
        ("unknown_supplier", "UNKNOWN_SUPPLIER"),
        ("unconfirmed_supplier", "SUPPLIER_NOT_CONFIRMED"),
        ("duplicate_invoice", "UNEXPECTED_DUPLICATE_INVOICE"),
    ),
)
def test_document_gate_fail_closed_matrix(case, reason):
    _, ocr = make_ocr()
    rules = anchored_invoice_candidates(ocr)
    run = model_run(ocr)
    overrides = {}
    if case == "missing_rule":
        rules = ()
    elif case == "ambiguous_rule":
        rules = (rules[0], AnchoredInvoiceCandidate("INV-999", (4,), ("204",), (ocr.words[4].normalized_box,)))
    elif case == "missing_model":
        run = None
    elif case == "ungrounded_model":
        run = model_run(ocr, grounded=False)
    elif case == "disagreement":
        run = model_run(ocr, value="INV-999")
    elif case == "low_confidence":
        run = model_run(ocr, confidence="0.70")
    elif case == "low_margin":
        run = model_run(ocr, margin="0.05")
    elif case == "unknown_supplier":
        overrides["supplier_id"] = "unknownco"
    elif case == "unconfirmed_supplier":
        overrides["supplier_confirmed"] = False
    elif case == "duplicate_invoice":
        overrides["active_identities"] = (InvoiceIdentity("fresh_farms", "INV-204"),)
    result = gate(ocr, rules, run, **overrides)
    assert result.status is DocumentStatus.REVIEW_REQUIRED
    assert reason in result.reason_codes
    assert result.verified_identity is None
    assert not result.may_activate_lookup


def test_document_gate_blocks_ocr_and_model_failures():
    _, ocr = make_ocr()
    failed_ocr = OcrResult(
        document_id=ocr.document_id,
        status=OcrStatus.TIMEOUT,
        words=(),
        raw_text="",
        language="eng",
        engine="tesseract_local",
        engine_version="unknown",
        runtime_ms=Decimal("1000"),
        error_code="TESSERACT_TIMEOUT",
        error_message="timeout",
    )
    failed_model = InvoiceModelRun(
        document_id=ocr.document_id,
        status=InvoiceModelRunStatus.FAILED,
        candidates=(),
        model_version="ryanznie/test",
        latency_ms=Decimal("1"),
        error_code="MODEL_RUN_FAILED",
        error_message="failed",
    )
    result = gate(failed_ocr, (), failed_model)
    assert "OCR_TIMEOUT" in result.reason_codes
    assert "MODEL_PROCESSING_FAILED" in result.reason_codes
    assert not result.may_activate_lookup
