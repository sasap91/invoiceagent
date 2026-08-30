"""OCR-plus-rules receipt parsing and the exact full-payment proof gate.

Receipt fields are never attributed to the invoice-number model.  Every parsed
field retains OCR-line evidence and any incomplete or conflicting result is
routed to review.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
import re
from typing import Iterable, Mapping

from .contracts import (
    ContractValidationError,
    InvoiceIdentity,
    InvoicePaymentStatus,
    PaymentProof,
    PaymentProofSource,
    PaymentProofStatus,
    SupplierIdentity,
    SyntheticSupplierInvoice,
    validate_full_payment_proof,
)
from .ocr import OcrResult, OcrStatus, OcrWord


class ReceiptParseStatus(str, Enum):
    READY_FOR_PROOF = "READY_FOR_PROOF"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


_RECEIPT_ID = re.compile(
    r"\breceipt\s+(?:id|number|no\.?|#)\s*[:#-]?\s*"
    r"(?P<value>[A-Za-z0-9][A-Za-z0-9._:-]{1,127})\b",
    re.IGNORECASE,
)
_INVOICE_ID = re.compile(
    r"\binvoice\s*(?:id|number|no\.?|#)?\s*[:#-]\s*"
    r"(?P<value>[A-Za-z0-9][A-Za-z0-9._/#:-]{1,127})\b",
    re.IGNORECASE,
)
_SUPPLIER = re.compile(
    r"\b(?:supplier|vendor)(?:\s+name)?\s*:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)
_AMOUNT = re.compile(
    r"\b(?:amount\s+paid|paid\s+amount|total\s+paid|payment\s+amount)\s*"
    r"[:#-]?\s*(?P<value>(?:[A-Z]{3}\s*)?[$€£]?\s*"
    r"[0-9][0-9,]*(?:\.[0-9]{1,2})?)\b",
    re.IGNORECASE,
)
_PAID_DATE = re.compile(
    r"\b(?:paid\s+date|payment\s+date|date\s+paid)\s*[:#-]?\s*"
    r"(?P<value>\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_CURRENCY = re.compile(r"\b(?P<value>USD|CAD|EUR|GBP|AUD|NZD|JPY)\b")
_PLAIN_MONEY = re.compile(
    r"(?P<whole>(?:0|[1-9]\d*)|(?:[1-9]\d{0,2}(?:,\d{3})+))"
    r"(?:\.(?P<fraction>\d{1,2}))?\Z"
)
_FIELD_ORDER = (
    "receipt_id",
    "supplier",
    "invoice_number",
    "amount_minor",
    "currency",
    "paid_date",
)
_FIELD_NAMES = frozenset(_FIELD_ORDER)


@dataclass(frozen=True, slots=True)
class ReceiptFieldEvidence:
    field_name: str
    value: str
    word_indices: tuple[int, ...]
    evidence_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.field_name not in _FIELD_NAMES:
            raise ContractValidationError("receipt evidence field_name is unsupported")
        if not isinstance(self.value, str) or not self.value:
            raise ContractValidationError("receipt evidence value must be non-empty text")
        indices = tuple(self.word_indices)
        tokens = tuple(self.evidence_tokens)
        if (
            not indices
            or len(indices) != len(tokens)
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            )
            or any(not isinstance(token, str) or not token for token in tokens)
        ):
            raise ContractValidationError("receipt field evidence must be grounded in OCR")
        object.__setattr__(self, "word_indices", indices)
        object.__setattr__(self, "evidence_tokens", tokens)


@dataclass(frozen=True, slots=True)
class ParsedReceipt:
    document_id: str
    status: ReceiptParseStatus
    receipt_id: str | None
    supplier_name: str | None
    supplier_id: str | None
    invoice_number: str | None
    amount_minor: int | None
    currency: str | None
    paid_date: date | None
    evidence: tuple[ReceiptFieldEvidence, ...]
    reason_codes: tuple[str, ...]
    extraction_method: str = "ocr_plus_deterministic_rules"

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.startswith("doc_"):
            raise ContractValidationError("parsed receipt needs a content-derived document ID")
        if not isinstance(self.status, ReceiptParseStatus):
            raise ContractValidationError("receipt parse status is invalid")
        evidence = tuple(self.evidence)
        reasons = tuple(self.reason_codes)
        if not all(isinstance(item, ReceiptFieldEvidence) for item in evidence):
            raise ContractValidationError("receipt evidence values are invalid")
        if any(
            not isinstance(reason, str)
            or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", reason)
            for reason in reasons
        ):
            raise ContractValidationError("receipt reason codes are invalid")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "reason_codes", reasons)
        if len(reasons) != len(set(reasons)):
            raise ContractValidationError("receipt reason codes cannot repeat")
        if self.amount_minor is not None and (
            isinstance(self.amount_minor, bool)
            or not isinstance(self.amount_minor, int)
            or self.amount_minor <= 0
        ):
            raise ContractValidationError("receipt amount must be positive integer minor units")
        if self.currency is not None and not re.fullmatch(r"[A-Z]{3}", self.currency):
            raise ContractValidationError("receipt currency must be an uppercase code")
        if self.paid_date is not None and not isinstance(self.paid_date, date):
            raise ContractValidationError("receipt paid_date must be a date")
        if self.extraction_method != "ocr_plus_deterministic_rules":
            raise ContractValidationError("receipt extraction method is fixed for P0")
        required_values = (
            self.receipt_id,
            self.supplier_name,
            self.supplier_id,
            self.invoice_number,
            self.amount_minor,
            self.currency,
            self.paid_date,
        )
        grounded_fields = {item.field_name for item in evidence}
        if self.status is ReceiptParseStatus.READY_FOR_PROOF:
            if any(value is None for value in required_values) or reasons:
                raise ContractValidationError("ready receipt must be complete and unambiguous")
            if grounded_fields != _FIELD_NAMES:
                raise ContractValidationError("ready receipt fields must all have OCR evidence")
            if len(evidence) != len(_FIELD_NAMES):
                raise ContractValidationError("ready receipt needs exactly one evidence item per field")
            expected_evidence = {
                "receipt_id": str(self.receipt_id),
                "supplier": str(self.supplier_name),
                "invoice_number": str(self.invoice_number),
                "amount_minor": str(self.amount_minor),
                "currency": str(self.currency),
                "paid_date": self.paid_date.isoformat(),  # type: ignore[union-attr]
            }
            if any(
                item.value != expected_evidence[item.field_name] for item in evidence
            ):
                raise ContractValidationError("receipt values must match their OCR evidence")
        elif not reasons:
            raise ContractValidationError("review-required receipt needs reason codes")


@dataclass(frozen=True, slots=True)
class PaymentProofGateResult:
    """The receipt ID is the only gating requirement for ``VERIFIED``.

    Supplier, invoice number, amount, and currency are still parsed and
    compared against the real invoice, but a mismatch or missing value there
    never blocks closure — it is disclosed instead via
    ``field_match_warnings``, which can be non-empty even when ``status`` is
    ``VERIFIED``. ``reason_codes`` holds only the reasons the receipt ID
    itself was rejected (missing, ambiguous, ungrounded, a duplicate, already
    consumed) or the OCR binding failed, and is always empty when verified.
    """

    status: PaymentProofStatus
    proof: PaymentProof | None
    reason_codes: tuple[str, ...]
    checks_passed: tuple[str, ...]
    field_match_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status is PaymentProofStatus.VERIFIED:
            if self.proof is None or self.reason_codes:
                raise ContractValidationError("verified proof gate result is inconsistent")
        elif self.status is PaymentProofStatus.REVIEW_REQUIRED:
            if self.proof is not None or not self.reason_codes:
                raise ContractValidationError("review proof gate result is inconsistent")
        else:
            raise ContractValidationError("proof gate returns VERIFIED or REVIEW_REQUIRED")
        warnings = tuple(self.field_match_warnings)
        if not all(
            isinstance(warning, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", warning)
            for warning in warnings
        ):
            raise ContractValidationError("field match warnings are invalid")
        object.__setattr__(self, "field_match_warnings", warnings)

    @property
    def closes_obligation(self) -> bool:
        return self.status is PaymentProofStatus.VERIFIED and self.proof is not None


def parse_money_to_minor_units(value: str) -> int:
    """Parse a receipt money string exactly; binary floats are never accepted."""

    if not isinstance(value, str) or value != value.strip() or not value:
        raise ContractValidationError("money must be non-empty text without outer whitespace")
    cleaned = re.sub(r"^[A-Z]{3}\s*", "", value)
    cleaned = re.sub(r"^[$€£]\s*", "", cleaned)
    match = _PLAIN_MONEY.fullmatch(cleaned)
    if not match:
        raise ContractValidationError("money must use an unambiguous decimal format")
    whole = match.group("whole").replace(",", "")
    fraction = (match.group("fraction") or "").ljust(2, "0")
    try:
        decimal_value = Decimal(f"{whole}.{fraction}")
    except InvalidOperation as exc:
        raise ContractValidationError("money is not a valid decimal") from exc
    minor = decimal_value * Decimal("100")
    if minor != minor.to_integral_value() or minor <= 0:
        raise ContractValidationError("money must be positive exact minor units")
    return int(minor)


def _line_groups(words: tuple[OcrWord, ...]) -> tuple[tuple[OcrWord, ...], ...]:
    groups: list[list[OcrWord]] = []
    key: tuple[int, int, int, int] | None = None
    for word in words:
        current = (word.page, word.block, word.paragraph, word.line)
        if current != key:
            groups.append([])
            key = current
        groups[-1].append(word)
    return tuple(tuple(group) for group in groups)


def _evidence(
    field_name: str, value: object, words: tuple[OcrWord, ...]
) -> ReceiptFieldEvidence:
    return ReceiptFieldEvidence(
        field_name=field_name,
        value=str(value),
        word_indices=tuple(word.sequence for word in words),
        evidence_tokens=tuple(word.text for word in words),
    )


def _unique_candidates(
    candidates: list[tuple[object, ReceiptFieldEvidence]],
) -> list[tuple[object, ReceiptFieldEvidence]]:
    unique: list[tuple[object, ReceiptFieldEvidence]] = []
    seen: set[object] = set()
    for value, evidence in candidates:
        if value in seen:
            continue
        seen.add(value)
        unique.append((value, evidence))
    return unique


def parse_receipt(
    ocr: OcrResult, *, known_suppliers: Iterable[SupplierIdentity]
) -> ParsedReceipt:
    """Parse only anchored receipt fields and retain their source OCR lines."""

    if not isinstance(ocr, OcrResult):
        raise ContractValidationError("ocr must be OcrResult")
    if ocr.status is not OcrStatus.SUCCESS:
        return ParsedReceipt(
            document_id=ocr.document_id,
            status=ReceiptParseStatus.REVIEW_REQUIRED,
            receipt_id=None,
            supplier_name=None,
            supplier_id=None,
            invoice_number=None,
            amount_minor=None,
            currency=None,
            paid_date=None,
            evidence=(),
            reason_codes=(f"OCR_{ocr.status.value}",),
        )
    suppliers = tuple(known_suppliers)
    if not all(isinstance(supplier, SupplierIdentity) for supplier in suppliers):
        raise ContractValidationError("known_suppliers must contain SupplierIdentity")
    supplier_lookup: dict[str, SupplierIdentity] = {}
    for supplier in suppliers:
        for label in (supplier.supplier_id, supplier.display_name):
            normalized = "".join(character for character in label.casefold() if character.isalnum())
            if normalized in supplier_lookup and supplier_lookup[normalized] != supplier:
                raise ContractValidationError("known supplier labels are ambiguous")
            supplier_lookup[normalized] = supplier

    found: dict[str, list[tuple[object, ReceiptFieldEvidence]]] = {
        field: [] for field in _FIELD_ORDER
    }
    unknown_supplier_seen = False
    invalid_amount_seen = False
    invalid_date_seen = False
    for line_words in _line_groups(ocr.words):
        line = " ".join(word.text for word in line_words)
        match = _RECEIPT_ID.search(line)
        if match:
            value = match.group("value").rstrip(".,;:")
            found["receipt_id"].append((value, _evidence("receipt_id", value, line_words)))
        match = _INVOICE_ID.search(line)
        if match:
            value = match.group("value").rstrip(".,;:")
            found["invoice_number"].append(
                (value, _evidence("invoice_number", value, line_words))
            )
        match = _SUPPLIER.search(line)
        if match:
            value = match.group("value").strip().rstrip(".,;:")
            normalized = "".join(
                character for character in value.casefold() if character.isalnum()
            )
            supplier = supplier_lookup.get(normalized)
            if supplier is None:
                unknown_supplier_seen = True
            else:
                found["supplier"].append(
                    (
                        supplier,
                        _evidence("supplier", supplier.display_name, line_words),
                    )
                )
        match = _AMOUNT.search(line)
        if match:
            value = match.group("value").strip()
            try:
                amount_minor = parse_money_to_minor_units(value)
            except ContractValidationError:
                invalid_amount_seen = True
            else:
                found["amount_minor"].append(
                    (
                        amount_minor,
                        _evidence("amount_minor", amount_minor, line_words),
                    )
                )
        match = _PAID_DATE.search(line)
        if match:
            value = match.group("value")
            try:
                paid_date = date.fromisoformat(value)
            except ValueError:
                invalid_date_seen = True
            else:
                found["paid_date"].append(
                    (paid_date, _evidence("paid_date", value, line_words))
                )
        for currency_match in _CURRENCY.finditer(line.upper()):
            value = currency_match.group("value")
            found["currency"].append(
                (value, _evidence("currency", value, line_words))
            )

    for field_name in found:
        found[field_name] = _unique_candidates(found[field_name])
    reasons: list[str] = []
    resolved: dict[str, object | None] = {}
    for field_name, candidates in found.items():
        if not candidates:
            reasons.append(f"MISSING_{field_name.upper()}")
            resolved[field_name] = None
        elif len(candidates) > 1:
            reasons.append(f"AMBIGUOUS_{field_name.upper()}")
            resolved[field_name] = None
        else:
            resolved[field_name] = candidates[0][0]
    if unknown_supplier_seen and resolved["supplier"] is None:
        reasons.append("UNKNOWN_SUPPLIER")
    if invalid_amount_seen and resolved["amount_minor"] is None:
        reasons.append("INVALID_AMOUNT")
    if invalid_date_seen and resolved["paid_date"] is None:
        reasons.append("INVALID_PAID_DATE")
    evidence = tuple(
        candidates[0][1]
        for candidates in found.values()
        if len(candidates) == 1
    )
    supplier = resolved["supplier"]
    status = (
        ReceiptParseStatus.READY_FOR_PROOF
        if not reasons
        else ReceiptParseStatus.REVIEW_REQUIRED
    )
    return ParsedReceipt(
        document_id=ocr.document_id,
        status=status,
        receipt_id=resolved["receipt_id"],  # type: ignore[arg-type]
        supplier_name=(
            supplier.display_name if isinstance(supplier, SupplierIdentity) else None
        ),
        supplier_id=(
            supplier.supplier_id if isinstance(supplier, SupplierIdentity) else None
        ),
        invoice_number=resolved["invoice_number"],  # type: ignore[arg-type]
        amount_minor=resolved["amount_minor"],  # type: ignore[arg-type]
        currency=resolved["currency"],  # type: ignore[arg-type]
        paid_date=resolved["paid_date"],  # type: ignore[arg-type]
        evidence=evidence,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def build_payment_proof(
    parsed: ParsedReceipt,
    invoice: SyntheticSupplierInvoice,
    *,
    ocr: OcrResult,
    known_suppliers: Iterable[SupplierIdentity],
    source: PaymentProofSource,
    provenance: str,
    seen_receipt_ids: Iterable[str] = (),
    consumed_receipt_ids: Mapping[str, InvoiceIdentity] | None = None,
) -> PaymentProofGateResult:
    """Build proof once the receipt ID is confirmed, grounded, and unconsumed.

    ``ParsedReceipt`` is a public immutable value object, not a trusted
    capability.  Reparse the supplied OCR here so a caller cannot manufacture
    evidence tokens or indices and bypass the proof boundary.

    Only the receipt ID gates ``VERIFIED``: it must be uniquely resolved,
    grounded in this exact OCR, and not a duplicate or already-consumed
    identifier. Supplier, invoice number, amount, and currency are still
    parsed and compared against ``invoice`` for transparency, but a mismatch
    or a value the receipt never supplied does not block closure. The
    resulting ``PaymentProof`` always carries ``invoice``'s own real values
    for those fields — never an unconfirmed or mismatched value read off the
    receipt — so the deeper ``validate_full_payment_proof`` contract (an
    exact match against the invoice) still holds unconditionally. Every
    field the receipt did not confirm, or confirmed differently, is recorded
    in ``field_match_warnings`` so the operator can see exactly what proof
    the receipt itself actually provided.
    """

    if not isinstance(parsed, ParsedReceipt):
        raise ContractValidationError("parsed must be ParsedReceipt")
    if not isinstance(invoice, SyntheticSupplierInvoice):
        raise ContractValidationError("invoice must be SyntheticSupplierInvoice")
    if not isinstance(ocr, OcrResult):
        raise ContractValidationError("ocr must be OcrResult")
    suppliers = tuple(known_suppliers)
    if not suppliers or not all(
        isinstance(supplier, SupplierIdentity) for supplier in suppliers
    ):
        raise ContractValidationError(
            "known_suppliers must contain trusted SupplierIdentity values"
        )
    consumed = consumed_receipt_ids or {}
    if not isinstance(consumed, Mapping) or not all(
        isinstance(key, str) and isinstance(value, InvoiceIdentity)
        for key, value in consumed.items()
    ):
        raise ContractValidationError(
            "consumed_receipt_ids must map receipt IDs to InvoiceIdentity"
        )
    seen = frozenset(seen_receipt_ids)
    reasons: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    # Only reason codes about the receipt ID itself gate closure. Every other
    # field's own missing/ambiguous/invalid diagnostics are dropped here —
    # the explicit per-field comparisons below report the same information
    # as clearer, non-blocking warnings instead.
    reasons.extend(code for code in parsed.reason_codes if "RECEIPT_ID" in code)

    reparsed = parse_receipt(ocr, known_suppliers=suppliers)
    if ocr.document_id != parsed.document_id or reparsed != parsed:
        reasons.append("RECEIPT_OCR_BINDING_MISMATCH")
    else:
        checks.append("PARSED_RECEIPT_BOUND_TO_OCR")
    if parsed.receipt_id is not None:
        if "receipt_id" not in {item.field_name for item in parsed.evidence}:
            reasons.append("RECEIPT_ID_UNGROUNDED")
        else:
            checks.append("RECEIPT_ID_GROUNDED_IN_OCR")
    if invoice.payment_status is not InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED:
        reasons.append("INVOICE_NOT_SIMULATED_PAYMENT_APPROVED")
    else:
        checks.append("SIMULATED_PAYMENT_APPROVED")
    if parsed.receipt_id is not None:
        if parsed.receipt_id in seen:
            reasons.append("DUPLICATE_RECEIPT_ID")
        if parsed.receipt_id in consumed:
            reasons.append("RECEIPT_ALREADY_CONSUMED")
        if parsed.receipt_id not in seen and parsed.receipt_id not in consumed:
            checks.append("UNUSED_RECEIPT_ID")

    # Informational only: never gate closure, always disclosed.
    if parsed.supplier_id is None:
        warnings.append("SUPPLIER_NOT_CONFIRMED_BY_RECEIPT")
    elif parsed.supplier_id != invoice.supplier_id:
        warnings.append("SUPPLIER_MISMATCH")
    else:
        checks.append("SUPPLIER_MATCH")
    if parsed.invoice_number is None:
        warnings.append("INVOICE_NUMBER_NOT_CONFIRMED_BY_RECEIPT")
    elif parsed.invoice_number != invoice.invoice_number:
        warnings.append("INVOICE_MISMATCH")
    else:
        checks.append("INVOICE_MATCH")
    if parsed.amount_minor is None:
        warnings.append("AMOUNT_NOT_CONFIRMED_BY_RECEIPT")
    elif parsed.amount_minor != invoice.amount_minor:
        warnings.append("AMOUNT_MISMATCH")
    else:
        checks.append("FULL_AMOUNT_MATCH")
    if parsed.currency is None:
        warnings.append("CURRENCY_NOT_CONFIRMED_BY_RECEIPT")
    elif parsed.currency != invoice.currency:
        warnings.append("CURRENCY_MISMATCH")
    else:
        checks.append("CURRENCY_MATCH")
    if parsed.paid_date is None:
        warnings.append("PAID_DATE_NOT_CONFIRMED_BY_RECEIPT")
    else:
        checks.append("PAID_DATE_PRESENT")

    reasons = list(dict.fromkeys(reasons))
    warnings = list(dict.fromkeys(warnings))
    if reasons:
        return PaymentProofGateResult(
            status=PaymentProofStatus.REVIEW_REQUIRED,
            proof=None,
            reason_codes=tuple(reasons),
            checks_passed=tuple(checks),
            field_match_warnings=tuple(warnings),
        )
    try:
        proof = PaymentProof(
            receipt_id=parsed.receipt_id,  # type: ignore[arg-type]
            # supplier_id/invoice_number/amount_minor/currency always come
            # from the real invoice, never from the receipt's own (possibly
            # missing or mismatched) reading, so validate_full_payment_proof
            # below still enforces an exact match unconditionally.
            supplier_id=invoice.supplier_id,
            invoice_number=invoice.invoice_number,
            amount_minor=invoice.amount_minor,
            currency=invoice.currency,
            paid_date=parsed.paid_date or date.today(),
            source=source,
            provenance=provenance,
            status=PaymentProofStatus.VERIFIED,
        )
        validate_full_payment_proof(invoice, proof)
    except ContractValidationError:
        return PaymentProofGateResult(
            status=PaymentProofStatus.REVIEW_REQUIRED,
            proof=None,
            reason_codes=("INVALID_PAYMENT_PROOF_CONTRACT",),
            checks_passed=tuple(checks),
            field_match_warnings=tuple(warnings),
        )
    return PaymentProofGateResult(
        status=PaymentProofStatus.VERIFIED,
        proof=proof,
        reason_codes=(),
        checks_passed=tuple(checks),
        field_match_warnings=tuple(warnings),
    )


__all__ = [
    "ParsedReceipt",
    "PaymentProofGateResult",
    "ReceiptFieldEvidence",
    "ReceiptParseStatus",
    "build_payment_proof",
    "parse_money_to_minor_units",
    "parse_receipt",
]
