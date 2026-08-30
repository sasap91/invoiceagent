"""Evidence-grounded OCR token labels for the guided invoice/receipt demo."""

from dataclasses import replace
from decimal import Decimal
import os
from pathlib import Path
import shutil

import pytest

from procureagent.contracts import (
    BoundingBox,
    ContractValidationError,
    InvoiceNumberCandidate,
    load_locked_scenario,
)
from procureagent.document import (
    InvoiceModelRun,
    InvoiceModelRunStatus,
    ModelInvoiceCandidate,
    anchored_invoice_candidates,
    gate_document_identity,
)
from procureagent.ocr import (
    OcrResult,
    OcrStatus,
    OcrWord,
    PixelBox,
    TesseractOCR,
    ingest_image,
    normalize_pixel_box,
)
from procureagent.receipt import ReceiptFieldEvidence, parse_receipt
from procureagent.token_labels import (
    LabeledToken,
    TokenLabel,
    TokenSource,
    label_invoice_tokens,
    label_receipt_tokens,
    summarize_token_labels,
)


DOCUMENT_ID = "doc_" + "c" * 64
ROOT = Path(__file__).resolve().parents[1]
INVOICE_ASSET = ROOT / "data/procureagent/assets/fresh_farms_invoice.png"
RECEIPT_ASSET = ROOT / "data/procureagent/assets/fresh_farms_payment_receipt.png"


def make_ocr(lines, *, document_id=DOCUMENT_ID):
    words = []
    for line_number, line in enumerate(lines, start=1):
        for word_number, text in enumerate(line, start=1):
            sequence = len(words)
            x0 = 20 + (word_number - 1) * 140
            y0 = 20 + (line_number - 1) * 70
            pixel = PixelBox(x0, y0, x0 + 120, y0 + 45)
            words.append(
                OcrWord(
                    sequence=sequence,
                    text=text,
                    confidence=Decimal("0.95"),
                    pixel_box=pixel,
                    normalized_box=normalize_pixel_box(pixel, 2000, 1000),
                    page=1,
                    block=1,
                    paragraph=1,
                    line=line_number,
                    word=word_number,
                )
            )
    return OcrResult(
        document_id=document_id,
        status=OcrStatus.SUCCESS,
        words=tuple(words),
        raw_text="\n".join(" ".join(line) for line in lines),
        language="eng",
        engine="test_tesseract",
        engine_version="5.test",
        runtime_ms=Decimal("1.2"),
    )


def model_run(
    ocr,
    indices,
    *,
    value=None,
    grounded=True,
    confidence="0.94",
    margin="0.40",
):
    evidence_words = tuple(ocr.words[index] for index in indices)
    candidate_value = value or "".join(word.text for word in evidence_words)
    return InvoiceModelRun(
        document_id=ocr.document_id,
        status=InvoiceModelRunStatus.SUCCESS,
        candidates=(
            ModelInvoiceCandidate(
                candidate=InvoiceNumberCandidate(
                    invoice_number=candidate_value,
                    entity_confidence=Decimal(confidence),
                    grounded_in_ocr=grounded,
                    evidence_tokens=tuple(word.text for word in evidence_words),
                    evidence_boxes=tuple(word.normalized_box for word in evidence_words),
                ),
                word_indices=tuple(indices),
                minimum_confidence=Decimal(confidence),
                mean_confidence=Decimal(confidence),
                mean_margin=Decimal(margin),
            ),
        ),
        model_version="ryanznie/layoutlmv3-test",
        latency_ms=Decimal("2.1"),
    )


def by_text(tokens, text):
    return tuple(token for token in tokens if token.text == text)


def test_invoice_labels_every_word_and_keeps_model_rule_amount_provenance_separate():
    ocr = make_ocr(
        (
            ("Invoice", "No:", "INV", "-", "204"),
            ("TOTAL", "USD", "$1,500.00"),
            ("Memo", "<script>alert(1)</script>"),
        )
    )
    rules = anchored_invoice_candidates(ocr)
    run = model_run(ocr, (2, 3, 4), value="INV-204")
    gate = gate_document_identity(
        document_id=ocr.document_id,
        supplier_id="fresh_farms",
        supplier_confirmed=True,
        known_supplier_ids=("fresh_farms",),
        ocr=ocr,
        rule_candidates=rules,
        model_run=run,
    )

    labeled = label_invoice_tokens(ocr, gate=gate)

    assert len(labeled) == len(ocr.words)
    assert tuple(token.index for token in labeled) == tuple(range(len(ocr.words)))
    assert tuple(token.text for token in labeled) == ocr.ordered_text
    assert {token.label for token in labeled[2:5]} == {TokenLabel.INVOICE_NUMBER}
    assert {token.source for token in labeled[2:5]} == {
        TokenSource.INVOICE_RULE_AND_RYAN_MODEL
    }
    amount = by_text(labeled, "$1,500.00")
    assert len(amount) == 1
    assert amount[0].label is TokenLabel.AMOUNT
    assert amount[0].source is TokenSource.INVOICE_AMOUNT_RULE
    assert by_text(labeled, "USD")[0].label is TokenLabel.OTHER
    # OCR is untrusted data.  The backend preserves it and does not emit markup.
    unsafe = by_text(labeled, "<script>alert(1)</script>")[0]
    assert unsafe.text == "<script>alert(1)</script>"
    assert unsafe.label is TokenLabel.OTHER
    assert unsafe.source is TokenSource.OCR_ONLY
    assert all(isinstance(token, LabeledToken) for token in labeled)
    counts = summarize_token_labels(labeled)
    assert counts[TokenLabel.INVOICE_NUMBER] == 3
    assert counts[TokenLabel.AMOUNT] == 1
    assert counts[TokenLabel.OTHER] == len(labeled) - 4
    assert counts[TokenLabel.RECEIPT_ID] == 0


@pytest.mark.parametrize(
    ("include_rule", "include_model", "expected_source"),
    (
        (True, False, TokenSource.INVOICE_ANCHORED_RULE),
        (False, True, TokenSource.RYAN_INVOICE_NUMBER_MODEL),
        (True, True, TokenSource.INVOICE_RULE_AND_RYAN_MODEL),
    ),
)
def test_invoice_number_source_is_visible_for_each_supported_context(
    include_rule, include_model, expected_source
):
    ocr = make_ocr((("Invoice", "No:", "FF-10482"),))
    rules = anchored_invoice_candidates(ocr) if include_rule else ()
    run = model_run(ocr, (2,), value="FF-10482") if include_model else None

    labeled = label_invoice_tokens(
        ocr,
        rule_candidates=rules,
        model_run=run,
    )

    selected = by_text(labeled, "FF-10482")[0]
    assert selected.label is TokenLabel.INVOICE_NUMBER
    assert selected.source is expected_source


def test_low_confidence_model_evidence_stays_visible_while_gate_owns_review():
    ocr = make_ocr((("Invoice", "No:", "FF-10482"),))
    rules = anchored_invoice_candidates(ocr)
    run = model_run(
        ocr,
        (2,),
        value="FF-10482",
        confidence="0.63",
        margin="0.35",
    )
    gate = gate_document_identity(
        document_id=ocr.document_id,
        supplier_id="fresh_farms",
        supplier_confirmed=True,
        known_supplier_ids=("fresh_farms",),
        ocr=ocr,
        rule_candidates=rules,
        model_run=run,
    )
    assert gate.reason_codes == ("LOW_MODEL_CONFIDENCE",)

    selected = by_text(label_invoice_tokens(ocr, gate=gate), "FF-10482")[0]

    assert selected.label is TokenLabel.INVOICE_NUMBER
    assert selected.source is TokenSource.INVOICE_RULE_AND_RYAN_MODEL


def test_ungrounded_model_output_is_not_attributed_to_ryan():
    ocr = make_ocr((("Invoice", "No:", "FF-10482"),))
    run = model_run(ocr, (2,), value="FF-10482", grounded=False)

    selected = by_text(
        label_invoice_tokens(
            ocr,
            rule_candidates=anchored_invoice_candidates(ocr),
            model_run=run,
        ),
        "FF-10482",
    )[0]

    assert selected.label is TokenLabel.INVOICE_NUMBER
    assert selected.source is TokenSource.INVOICE_ANCHORED_RULE


def test_rule_model_disagreement_fails_closed_without_hiding_unique_amount():
    ocr = make_ocr(
        (
            ("Invoice", "No:", "FF-10482"),
            ("Reference", "ALT-999"),
            ("TOTAL", "$10.00"),
        )
    )
    rules = anchored_invoice_candidates(ocr)
    run = model_run(ocr, (4,), value="ALT-999")

    labeled = label_invoice_tokens(ocr, rule_candidates=rules, model_run=run)

    assert by_text(labeled, "FF-10482")[0].label is TokenLabel.OTHER
    assert by_text(labeled, "ALT-999")[0].label is TokenLabel.OTHER
    amount = by_text(labeled, "$10.00")[0]
    assert amount.label is TokenLabel.AMOUNT
    assert amount.source is TokenSource.INVOICE_AMOUNT_RULE


def test_ambiguous_invoice_number_or_total_is_not_selected():
    ocr = make_ocr(
        (
            ("Invoice", ":", "INV-1"),
            ("Invoice", ":", "INV-2"),
            ("TOTAL", "$10.00"),
            ("AMOUNT", "DUE", "$11.00"),
        )
    )

    labeled = label_invoice_tokens(
        ocr,
        rule_candidates=anchored_invoice_candidates(ocr),
    )

    assert by_text(labeled, "INV-1")[0].label is TokenLabel.OTHER
    assert by_text(labeled, "INV-2")[0].label is TokenLabel.OTHER
    assert by_text(labeled, "$10.00")[0].label is TokenLabel.OTHER
    assert by_text(labeled, "$11.00")[0].label is TokenLabel.OTHER


def test_invoice_money_rule_uses_exact_decimal_syntax_and_rejects_exponents():
    exact = make_ocr((("TOTAL", "$0.10"),))
    unsafe = make_ocr((("TOTAL", "1e3"),))

    exact_amount = by_text(label_invoice_tokens(exact), "$0.10")[0]
    exponent = by_text(label_invoice_tokens(unsafe), "1e3")[0]

    assert exact_amount.label is TokenLabel.AMOUNT
    assert exact_amount.source is TokenSource.INVOICE_AMOUNT_RULE
    assert exponent.label is TokenLabel.OTHER


def test_repeated_total_value_occurrences_are_ambiguous():
    ocr = make_ocr((("TOTAL", "$10.00", "$10.00"),))

    labeled = label_invoice_tokens(ocr)

    assert all(token.label is TokenLabel.OTHER for token in labeled)


def test_invalid_or_mismatched_invoice_context_fails_closed():
    ocr = make_ocr((("Invoice", "No:", "FF-10482"),))
    rules = anchored_invoice_candidates(ocr)
    run = model_run(ocr, (2,), value="FF-10482")
    gate = gate_document_identity(
        document_id=ocr.document_id,
        supplier_id="fresh_farms",
        supplier_confirmed=True,
        known_supplier_ids=("fresh_farms",),
        ocr=ocr,
        rule_candidates=rules,
        model_run=run,
    )
    mismatched = replace(gate, document_id="doc_" + "d" * 64)

    assert all(
        token.label is TokenLabel.OTHER
        for token in label_invoice_tokens(ocr, gate=mismatched)
    )
    with pytest.raises(ContractValidationError, match="cannot be combined"):
        label_invoice_tokens(ocr, gate=gate, rule_candidates=rules)


def receipt_ocr(*, extra_lines=()):
    return make_ocr(
        (
            ("Receipt", "ID:", "RCPT-FF-10482"),
            ("Supplier:", "Fresh", "Farms"),
            ("Invoice", "Number:", "FF-10482"),
            ("Paid", "Date:", "2026-08-30"),
            ("Currency:", "USD"),
            ("Amount", "Paid:", "$1,500.00"),
            *extra_lines,
        )
    )


def parsed_receipt(ocr):
    return parse_receipt(ocr, known_suppliers=load_locked_scenario().suppliers)


def test_receipt_labels_only_value_spans_from_grounded_rule_evidence():
    ocr = receipt_ocr()
    parsed = parsed_receipt(ocr)

    labeled = label_receipt_tokens(ocr, parsed)

    assert len(labeled) == len(ocr.words)
    assert by_text(labeled, "RCPT-FF-10482")[0].label is TokenLabel.RECEIPT_ID
    assert {token.label for token in by_text(labeled, "Fresh")} == {
        TokenLabel.SUPPLIER
    }
    assert {token.label for token in by_text(labeled, "Farms")} == {
        TokenLabel.SUPPLIER
    }
    assert by_text(labeled, "FF-10482")[0].label is TokenLabel.INVOICE_NUMBER
    assert by_text(labeled, "2026-08-30")[0].label is TokenLabel.DATE
    assert by_text(labeled, "USD")[0].label is TokenLabel.CURRENCY
    assert by_text(labeled, "$1,500.00")[0].label is TokenLabel.AMOUNT
    selected = tuple(token for token in labeled if token.label is not TokenLabel.OTHER)
    assert selected
    assert {token.source for token in selected} == {TokenSource.RECEIPT_FIELD_RULE}
    assert all(
        token.label is TokenLabel.OTHER
        for token in labeled
        if token.text
        in {
            "Receipt",
            "ID:",
            "Supplier:",
            "Invoice",
            "Number:",
            "Paid",
            "Date:",
            "Currency:",
            "Amount",
        }
    )
    counts = summarize_token_labels(labeled)
    assert counts == {
        TokenLabel.RECEIPT_ID: 1,
        TokenLabel.INVOICE_NUMBER: 1,
        TokenLabel.AMOUNT: 1,
        TokenLabel.CURRENCY: 1,
        TokenLabel.DATE: 1,
        TokenLabel.SUPPLIER: 2,
        TokenLabel.OTHER: len(labeled) - 7,
    }


def test_ambiguous_receipt_amount_is_not_labeled_but_unique_fields_remain_grounded():
    ocr = receipt_ocr(extra_lines=(("Amount", "Paid:", "$1,499.00"),))
    parsed = parsed_receipt(ocr)
    assert parsed.amount_minor is None
    assert "AMBIGUOUS_AMOUNT_MINOR" in parsed.reason_codes

    labeled = label_receipt_tokens(ocr, parsed)

    assert all(
        token.label is TokenLabel.OTHER
        for token in labeled
        if token.text in {"$1,500.00", "$1,499.00"}
    )
    assert by_text(labeled, "RCPT-FF-10482")[0].label is TokenLabel.RECEIPT_ID


def test_forged_receipt_evidence_and_document_mismatch_fail_closed():
    ocr = receipt_ocr()
    parsed = parsed_receipt(ocr)
    invoice_evidence = next(
        item for item in parsed.evidence if item.field_name == "invoice_number"
    )
    forged_item = ReceiptFieldEvidence(
        field_name="invoice_number",
        value=invoice_evidence.value,
        word_indices=(0,),
        evidence_tokens=("FF-10482",),
    )
    forged = replace(
        parsed,
        evidence=tuple(
            forged_item if item.field_name == "invoice_number" else item
            for item in parsed.evidence
        ),
    )

    forged_labels = label_receipt_tokens(ocr, forged)
    mismatch_labels = label_receipt_tokens(
        ocr, replace(parsed, document_id="doc_" + "e" * 64)
    )

    assert by_text(forged_labels, "FF-10482")[0].label is TokenLabel.OTHER
    assert all(token.label is TokenLabel.OTHER for token in mismatch_labels)


def test_summarizer_rejects_untyped_values():
    with pytest.raises(ContractValidationError, match="LabeledToken"):
        summarize_token_labels((object(),))  # type: ignore[arg-type]


@pytest.mark.skipif(
    os.environ.get("RUN_TESSERACT_SMOKE") != "1" or shutil.which("tesseract") is None,
    reason="set RUN_TESSERACT_SMOKE=1 with local tesseract to run real fixtures",
)
def test_bundled_real_invoice_and_receipt_ocr_labels_are_grounded():
    invoice_image = ingest_image(
        INVOICE_ASSET.read_bytes(), original_filename=INVOICE_ASSET.name
    )
    invoice_ocr = TesseractOCR(timeout_seconds=10).run(invoice_image)
    rules = anchored_invoice_candidates(invoice_ocr)
    invoice_index = next(
        word.sequence for word in invoice_ocr.words if word.text == "FF-10482"
    )
    run = model_run(invoice_ocr, (invoice_index,), value="FF-10482")
    invoice_labels = label_invoice_tokens(
        invoice_ocr, rule_candidates=rules, model_run=run
    )

    assert by_text(invoice_labels, "FF-10482")[0].source is (
        TokenSource.INVOICE_RULE_AND_RYAN_MODEL
    )
    totals = by_text(invoice_labels, "$1,500.00")
    assert sum(token.label is TokenLabel.AMOUNT for token in totals) == 1
    assert next(token for token in totals if token.label is TokenLabel.AMOUNT).source is (
        TokenSource.INVOICE_AMOUNT_RULE
    )

    receipt_image = ingest_image(
        RECEIPT_ASSET.read_bytes(), original_filename=RECEIPT_ASSET.name
    )
    receipt_result = TesseractOCR(timeout_seconds=10).run(receipt_image)
    parsed = parse_receipt(
        receipt_result, known_suppliers=load_locked_scenario().suppliers
    )
    receipt_labels = label_receipt_tokens(receipt_result, parsed)
    selected = {
        token.text: token.label
        for token in receipt_labels
        if token.label is not TokenLabel.OTHER
    }
    assert selected["RCPT-FF-10482"] is TokenLabel.RECEIPT_ID
    assert selected["FF-10482"] is TokenLabel.INVOICE_NUMBER
    assert selected["2026-08-30"] is TokenLabel.DATE
    assert selected["USD"] is TokenLabel.CURRENCY
    assert selected["$1,500.00"] is TokenLabel.AMOUNT
