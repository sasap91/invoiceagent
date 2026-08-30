from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from procureagent.contracts import (
    ContractValidationError,
    InvoiceIdentity,
    InvoicePaymentStatus,
    PaymentProofSource,
    PaymentProofStatus,
    load_locked_scenario,
)
from procureagent.ocr import OcrResult, OcrStatus, OcrWord, PixelBox, normalize_pixel_box
from procureagent.receipt import (
    ReceiptParseStatus,
    build_payment_proof,
    parse_money_to_minor_units,
    parse_receipt,
)


DOCUMENT_ID = "doc_" + "a" * 64


def receipt_ocr(
    *,
    receipt_id="receipt_fresh_farms_10482",
    supplier="Fresh Farms",
    invoice_number="FF-10482",
    amount="USD $1,500.00",
    paid_date="2026-08-30",
    extra_lines=(),
):
    lines = [
        ("Receipt", "ID", ":", receipt_id),
        ("Supplier", ":", *supplier.split()),
        ("Invoice", "#", ":", invoice_number),
        ("Amount", "Paid", ":", *amount.split()),
        ("Paid", "Date", ":", paid_date),
        *extra_lines,
    ]
    words = []
    for line_number, line in enumerate(lines, start=1):
        for word_number, token in enumerate(line, start=1):
            sequence = len(words)
            x0 = min((word_number - 1) * 12, 90)
            pixel_box = PixelBox(x0, line_number * 10, min(x0 + 10, 100), line_number * 10 + 8)
            words.append(
                OcrWord(
                    sequence=sequence,
                    text=token,
                    confidence=Decimal("0.96"),
                    pixel_box=pixel_box,
                    normalized_box=normalize_pixel_box(pixel_box, 100, 100),
                    page=1,
                    block=1,
                    paragraph=1,
                    line=line_number,
                    word=word_number,
                )
            )
    return OcrResult(
        document_id=DOCUMENT_ID,
        status=OcrStatus.SUCCESS,
        words=tuple(words),
        raw_text="\n".join(" ".join(line) for line in lines),
        language="eng",
        engine="tesseract_local",
        engine_version="tesseract test",
        runtime_ms=Decimal("3.2"),
    )


@pytest.fixture(scope="module")
def scenario():
    return load_locked_scenario()


def parse(ocr, scenario):
    return parse_receipt(ocr, known_suppliers=scenario.suppliers)


def test_receipt_parser_extracts_all_fields_with_grounded_rule_evidence(scenario):
    parsed = parse(receipt_ocr(), scenario)
    assert parsed.status is ReceiptParseStatus.READY_FOR_PROOF
    assert parsed.extraction_method == "ocr_plus_deterministic_rules"
    assert parsed.receipt_id == "receipt_fresh_farms_10482"
    assert parsed.supplier_name == "Fresh Farms"
    assert parsed.supplier_id == "fresh_farms"
    assert parsed.invoice_number == "FF-10482"
    assert parsed.amount_minor == 150_000
    assert parsed.currency == "USD"
    assert parsed.paid_date.isoformat() == "2026-08-30"
    assert parsed.reason_codes == ()
    assert {item.field_name for item in parsed.evidence} == {
        "receipt_id",
        "supplier",
        "invoice_number",
        "amount_minor",
        "currency",
        "paid_date",
    }
    assert all(item.word_indices and item.evidence_tokens for item in parsed.evidence)


@pytest.mark.parametrize(
    ("text", "minor"),
    (
        ("1500", 150_000),
        ("1500.0", 150_000),
        ("1500.00", 150_000),
        ("1,500.00", 150_000),
        ("USD 1,500.00", 150_000),
        ("$1,500.00", 150_000),
    ),
)
def test_money_parser_uses_exact_minor_units(text, minor):
    assert parse_money_to_minor_units(text) == minor


@pytest.mark.parametrize(
    "value",
    (1500.0, "1,50.00", "1.234", "1e3", "-1.00", "0.00", " $1.00 "),
)
def test_money_parser_rejects_binary_float_or_ambiguous_money(value):
    with pytest.raises(ContractValidationError):
        parse_money_to_minor_units(value)


def test_receipt_parser_requires_explicit_currency_evidence(scenario):
    parsed = parse(receipt_ocr(amount="$1,500.00"), scenario)
    assert parsed.status is ReceiptParseStatus.REVIEW_REQUIRED
    assert "MISSING_CURRENCY" in parsed.reason_codes


def test_receipt_parser_routes_unknown_supplier_to_review(scenario):
    parsed = parse(receipt_ocr(supplier="UnknownCo"), scenario)
    assert parsed.status is ReceiptParseStatus.REVIEW_REQUIRED
    assert parsed.supplier_id is None
    assert "UNKNOWN_SUPPLIER" in parsed.reason_codes
    assert "MISSING_SUPPLIER" in parsed.reason_codes


def test_receipt_parser_routes_missing_and_ambiguous_fields_to_review(scenario):
    ambiguous = parse(
        receipt_ocr(extra_lines=(("Amount", "Paid", ":", "USD", "$1,499.00"),)),
        scenario,
    )
    assert ambiguous.status is ReceiptParseStatus.REVIEW_REQUIRED
    assert ambiguous.amount_minor is None
    assert "AMBIGUOUS_AMOUNT_MINOR" in ambiguous.reason_codes
    missing = parse(receipt_ocr(receipt_id="missing"), scenario)
    assert missing.status is ReceiptParseStatus.READY_FOR_PROOF
    # Removing the anchored qualifier entirely means no receipt ID may be invented.
    words = tuple(word for word in receipt_ocr().words if word.line != 1)
    reindexed = tuple(replace(word, sequence=index) for index, word in enumerate(words))
    no_id_ocr = replace(receipt_ocr(), words=reindexed)
    missing = parse(no_id_ocr, scenario)
    assert missing.status is ReceiptParseStatus.REVIEW_REQUIRED
    assert "MISSING_RECEIPT_ID" in missing.reason_codes


def test_receipt_parser_propagates_ocr_failure_without_inventing_fields(scenario):
    failed = OcrResult(
        document_id=DOCUMENT_ID,
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
    parsed = parse(failed, scenario)
    assert parsed.status is ReceiptParseStatus.REVIEW_REQUIRED
    assert parsed.reason_codes == ("OCR_TIMEOUT",)
    assert parsed.evidence == ()
    assert parsed.receipt_id is None


def test_exact_verified_receipt_builds_full_payment_proof(scenario):
    ocr = receipt_ocr()
    parsed = parse(ocr, scenario)
    invoice = replace(
        scenario.initial_state.invoices[0],
        payment_status=InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED,
    )
    result = build_payment_proof(
        parsed,
        invoice,
        ocr=ocr,
        known_suppliers=scenario.suppliers,
        source=PaymentProofSource.OPERATOR_UPLOAD,
        provenance="live_local_tesseract_upload",
    )
    assert result.status is PaymentProofStatus.VERIFIED
    assert result.closes_obligation
    assert result.reason_codes == ()
    assert result.field_match_warnings == ()
    assert result.proof.identity == InvoiceIdentity("fresh_farms", "FF-10482")
    assert result.proof.amount_minor == 150_000
    assert result.proof.source is PaymentProofSource.OPERATOR_UPLOAD
    assert {
        "RECEIPT_ID_GROUNDED_IN_OCR",
        "SIMULATED_PAYMENT_APPROVED",
        "UNUSED_RECEIPT_ID",
        "SUPPLIER_MATCH",
        "INVOICE_MATCH",
        "FULL_AMOUNT_MATCH",
        "CURRENCY_MATCH",
    }.issubset(result.checks_passed)


def test_forged_parsed_evidence_cannot_cross_the_ocr_bound_proof_gate(scenario):
    ocr = receipt_ocr()
    parsed = parse(ocr, scenario)
    forged = replace(
        parsed,
        evidence=tuple(
            replace(
                item,
                evidence_tokens=tuple("unrelated-token" for _ in item.evidence_tokens),
            )
            for item in parsed.evidence
        ),
    )
    invoice = replace(
        scenario.initial_state.invoices[0],
        payment_status=InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED,
    )

    result = build_payment_proof(
        forged,
        invoice,
        ocr=ocr,
        known_suppliers=scenario.suppliers,
        source=PaymentProofSource.OPERATOR_UPLOAD,
        provenance="adversarial_forged_value_object",
    )

    assert result.status is PaymentProofStatus.REVIEW_REQUIRED
    assert "RECEIPT_OCR_BINDING_MISMATCH" in result.reason_codes
    assert "PARSED_RECEIPT_BOUND_TO_OCR" not in result.checks_passed
    assert result.proof is None


@pytest.mark.parametrize(
    ("ocr", "warning"),
    (
        (receipt_ocr(supplier="Prime Foods"), "SUPPLIER_MISMATCH"),
        (receipt_ocr(invoice_number="FF-99999"), "INVOICE_MISMATCH"),
        (receipt_ocr(amount="USD $1,499.99"), "AMOUNT_MISMATCH"),
        (receipt_ocr(amount="CAD $1,500.00"), "CURRENCY_MISMATCH"),
    ),
)
def test_mismatched_fields_are_disclosed_but_do_not_block_closure(scenario, ocr, warning):
    """Only the receipt ID gates closure; every other mismatch is a disclosed warning.

    The resulting proof always carries the real invoice's own identity,
    amount, and currency — never the receipt's mismatched reading — so the
    deeper validate_full_payment_proof contract still holds exactly.
    """

    parsed = parse(ocr, scenario)
    invoice = replace(
        scenario.initial_state.invoices[0],
        payment_status=InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED,
    )
    result = build_payment_proof(
        parsed,
        invoice,
        ocr=ocr,
        known_suppliers=scenario.suppliers,
        source=PaymentProofSource.OPERATOR_UPLOAD,
        provenance="test_upload",
    )
    assert result.status is PaymentProofStatus.VERIFIED
    assert result.closes_obligation
    assert result.reason_codes == ()
    assert warning in result.field_match_warnings
    assert result.proof.identity == InvoiceIdentity("fresh_farms", "FF-10482")
    assert result.proof.amount_minor == 150_000
    assert result.proof.currency == "USD"


def test_partial_and_excess_amounts_are_disclosed_but_do_not_block_closure(scenario):
    invoice = replace(
        scenario.initial_state.invoices[0],
        payment_status=InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED,
    )
    for amount in ("USD $1,000.00", "USD $1,500.01"):
        ocr = receipt_ocr(amount=amount)
        result = build_payment_proof(
            parse(ocr, scenario),
            invoice,
            ocr=ocr,
            known_suppliers=scenario.suppliers,
            source=PaymentProofSource.OPERATOR_UPLOAD,
            provenance="test_upload",
        )
        assert result.status is PaymentProofStatus.VERIFIED
        assert "AMOUNT_MISMATCH" in result.field_match_warnings
        # The obligation still closes for exactly its real, full amount.
        assert result.proof.amount_minor == 150_000


def _receipt_id_only_ocr(receipt_id="19729058"):
    """A receipt where only a receipt ID line is recognizable — no supplier,
    invoice number, amount, currency, or paid date — matching the reported
    real-world case of an off-template (SROIE-style) receipt upload."""

    lines = [("Receipt", "ID", ":", receipt_id)]
    words = []
    for line_number, line in enumerate(lines, start=1):
        for word_number, token in enumerate(line, start=1):
            sequence = len(words)
            x0 = min((word_number - 1) * 12, 90)
            pixel_box = PixelBox(x0, line_number * 10, min(x0 + 10, 100), line_number * 10 + 8)
            words.append(
                OcrWord(
                    sequence=sequence,
                    text=token,
                    confidence=Decimal("0.9"),
                    pixel_box=pixel_box,
                    normalized_box=normalize_pixel_box(pixel_box, 100, 100),
                    page=1,
                    block=1,
                    paragraph=1,
                    line=line_number,
                    word=word_number,
                )
            )
    return OcrResult(
        document_id=DOCUMENT_ID,
        status=OcrStatus.SUCCESS,
        words=tuple(words),
        raw_text="\n".join(" ".join(line) for line in lines),
        language="eng",
        engine="tesseract_local",
        engine_version="tesseract test",
        runtime_ms=Decimal("3.2"),
    )


def test_receipt_id_alone_is_sufficient_to_close_the_obligation(scenario):
    """Regression test for the reported case: an off-template receipt where
    only the receipt ID parses. Every other field must be disclosed as
    unconfirmed, never block closure, and the proof must still carry the
    real invoice's own identity/amount/currency."""

    ocr = _receipt_id_only_ocr()
    parsed = parse(ocr, scenario)
    assert parsed.receipt_id == "19729058"
    assert parsed.supplier_id is None
    assert parsed.invoice_number is None
    assert parsed.amount_minor is None
    assert parsed.currency is None
    assert parsed.paid_date is None
    assert parsed.status is ReceiptParseStatus.REVIEW_REQUIRED

    invoice = replace(
        scenario.initial_state.invoices[0],
        payment_status=InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED,
    )
    result = build_payment_proof(
        parsed,
        invoice,
        ocr=ocr,
        known_suppliers=scenario.suppliers,
        source=PaymentProofSource.OPERATOR_UPLOAD,
        provenance="test_X51005230605",
    )
    assert result.status is PaymentProofStatus.VERIFIED
    assert result.closes_obligation
    assert result.reason_codes == ()
    assert set(result.field_match_warnings) == {
        "SUPPLIER_NOT_CONFIRMED_BY_RECEIPT",
        "INVOICE_NUMBER_NOT_CONFIRMED_BY_RECEIPT",
        "AMOUNT_NOT_CONFIRMED_BY_RECEIPT",
        "CURRENCY_NOT_CONFIRMED_BY_RECEIPT",
        "PAID_DATE_NOT_CONFIRMED_BY_RECEIPT",
    }
    assert result.proof.receipt_id == "19729058"
    assert result.proof.identity == InvoiceIdentity("fresh_farms", "FF-10482")
    assert result.proof.amount_minor == 150_000
    assert result.proof.currency == "USD"


def test_ambiguous_or_missing_receipt_id_still_blocks_closure(scenario):
    ocr = receipt_ocr(
        extra_lines=(("Receipt", "ID", ":", "ANOTHER-ID"),)
    )
    parsed = parse(ocr, scenario)
    assert parsed.receipt_id is None
    assert "AMBIGUOUS_RECEIPT_ID" in parsed.reason_codes
    invoice = replace(
        scenario.initial_state.invoices[0],
        payment_status=InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED,
    )
    result = build_payment_proof(
        parsed,
        invoice,
        ocr=ocr,
        known_suppliers=scenario.suppliers,
        source=PaymentProofSource.OPERATOR_UPLOAD,
        provenance="test_upload",
    )
    assert result.status is PaymentProofStatus.REVIEW_REQUIRED
    assert "AMBIGUOUS_RECEIPT_ID" in result.reason_codes
    assert result.proof is None


def test_duplicate_and_consumed_receipt_ids_are_blocked(scenario):
    ocr = receipt_ocr()
    parsed = parse(ocr, scenario)
    invoice = replace(
        scenario.initial_state.invoices[0],
        payment_status=InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED,
    )
    duplicate = build_payment_proof(
        parsed,
        invoice,
        ocr=ocr,
        known_suppliers=scenario.suppliers,
        source=PaymentProofSource.OPERATOR_UPLOAD,
        provenance="test_upload",
        seen_receipt_ids=(parsed.receipt_id,),
    )
    assert "DUPLICATE_RECEIPT_ID" in duplicate.reason_codes
    consumed = build_payment_proof(
        parsed,
        invoice,
        ocr=ocr,
        known_suppliers=scenario.suppliers,
        source=PaymentProofSource.OPERATOR_UPLOAD,
        provenance="test_upload",
        consumed_receipt_ids={
            parsed.receipt_id: InvoiceIdentity("prime_foods", "PF-25031")
        },
    )
    assert "RECEIPT_ALREADY_CONSUMED" in consumed.reason_codes
    assert consumed.proof is None


def test_unapproved_simulated_invoice_cannot_be_closed(scenario):
    ocr = receipt_ocr()
    result = build_payment_proof(
        parse(ocr, scenario),
        scenario.initial_state.invoices[0],
        ocr=ocr,
        known_suppliers=scenario.suppliers,
        source=PaymentProofSource.OPERATOR_UPLOAD,
        provenance="test_upload",
    )
    assert result.status is PaymentProofStatus.REVIEW_REQUIRED
    assert "INVOICE_NOT_SIMULATED_PAYMENT_APPROVED" in result.reason_codes
    assert result.proof is None


def test_receipt_module_has_no_invoice_model_dependency():
    source = (Path(__file__).parents[1] / "src" / "procureagent" / "receipt.py").read_text(
        encoding="utf-8"
    )
    assert "from invoiceagent" not in source
    assert "LayoutLMv3InvoiceExtractor" not in source
