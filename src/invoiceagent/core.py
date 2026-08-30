"""Core accounting and routing primitives for the InvoiceAgent MVP.

This module deliberately has no web-framework or model-runtime dependency.  It
records where extracted data came from, reconciles conservatively, and never
silently turns an ambiguous payment into an accounting fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from enum import Enum
import re
from typing import Iterable, Sequence


ZERO = Decimal("0.00")
CENT = Decimal("0.01")
_MONEY_PATTERN = re.compile(r"(?:0|[1-9]\d*)(?:\.\d{1,2})?\Z")
_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_CURRENCY_PATTERN = re.compile(r"[A-Z]{3}\Z")


class ValidationError(ValueError):
    """Raised when financial input is not explicit and safe to process."""


class LedgerSide(str, Enum):
    """The small business's relationship to an invoice."""

    ACCOUNTS_PAYABLE = "accounts_payable"
    ACCOUNTS_RECEIVABLE = "accounts_receivable"


class ExtractionSource(str, Enum):
    """An honest record of how fields entered the system."""

    UNKNOWN = "unknown"
    MANUAL = "manual"
    HEURISTIC = "heuristic"
    SMALL_MODEL = "small_model"
    LARGER_MODEL = "larger_model"


class InvoiceStatus(str, Enum):
    OPEN = "open"
    PARTIALLY_PAID = "partially_paid"
    PAID = "paid"


class ReceiptStatus(str, Enum):
    MATCHED = "matched"
    PARTIALLY_MATCHED = "partially_matched"
    NEEDS_REVIEW = "needs_review"


class MatchMethod(str, Enum):
    EXPLICIT_REFERENCE = "explicit_reference"
    WEIGHTED_FALLBACK = "weighted_fallback"


class RoutingAction(str, Enum):
    ACCEPT = "accept"
    ESCALATE = "escalate"
    HUMAN_REVIEW = "human_review"


def _enum_value(value: object, enum_type: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            raise ValidationError(f"{field_name} is not a supported value") from exc
    raise ValidationError(f"{field_name} must be a {enum_type.__name__}")


def _required_text(value: object, field_name: str, *, limit: int = 256) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be text")
    cleaned = value.strip()
    if not cleaned:
        raise ValidationError(f"{field_name} cannot be empty")
    if len(cleaned) > limit:
        raise ValidationError(f"{field_name} is longer than {limit} characters")
    return cleaned


def parse_money(value: Decimal | str, field_name: str = "amount") -> Decimal:
    """Parse a non-negative money value without accepting binary floats.

    Accepted inputs are ``Decimal`` or a plain decimal string with at most two
    fractional digits. Currency symbols, commas, exponent notation, floats,
    negative values, NaN, and infinity are intentionally rejected.
    """

    if isinstance(value, bool) or isinstance(value, (float, int)):
        raise ValidationError(
            f"{field_name} must be Decimal or a plain decimal string, not a float/int"
        )
    if isinstance(value, str):
        if not _MONEY_PATTERN.fullmatch(value):
            raise ValidationError(
                f"{field_name} must be a non-negative decimal with at most two places"
            )
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:  # defensive; the regular expression is strict
            raise ValidationError(f"{field_name} is not valid money") from exc
    elif isinstance(value, Decimal):
        parsed = value
        if not parsed.is_finite() or parsed < ZERO or parsed.as_tuple().exponent < -2:
            raise ValidationError(
                f"{field_name} must be finite, non-negative, and have at most two places"
            )
    else:
        raise ValidationError(f"{field_name} must be Decimal or a plain decimal string")
    return parsed.quantize(CENT)


def _positive_money(value: Decimal | str, field_name: str) -> Decimal:
    parsed = parse_money(value, field_name)
    if parsed <= ZERO:
        raise ValidationError(f"{field_name} must be greater than zero")
    return parsed


def parse_iso_date(value: date | str, field_name: str = "date") -> date:
    """Accept a ``date`` or an exact ISO ``YYYY-MM-DD`` string."""

    if isinstance(value, datetime):
        raise ValidationError(f"{field_name} must be a date, not a datetime")
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not _DATE_PATTERN.fullmatch(value):
        raise ValidationError(f"{field_name} must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError(f"{field_name} is not a real calendar date") from exc


def _currency(value: object) -> str:
    if not isinstance(value, str) or not _CURRENCY_PATTERN.fullmatch(value):
        raise ValidationError("currency must be a three-letter uppercase code such as USD")
    return value


def _probability(value: Decimal | str, field_name: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, (float, int)):
        raise ValidationError(f"{field_name} must be Decimal or a decimal string")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValidationError(f"{field_name} is not a decimal probability") from exc
    if not parsed.is_finite() or parsed < Decimal("0") or parsed > Decimal("1"):
        raise ValidationError(f"{field_name} must be between 0 and 1")
    return parsed


def normalize_identifier(value: str) -> str:
    """Normalize a reference for comparison while preserving its stored display form."""

    return "".join(character for character in value.casefold() if character.isalnum())


def _party_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _counterparty_similarity(left: str, right: str) -> Decimal:
    left_tokens = _party_tokens(left)
    right_tokens = _party_tokens(right)
    left_text = " ".join(left_tokens)
    right_text = " ".join(right_tokens)
    if not left_text or not right_text:
        return Decimal("0")
    sequence_score = SequenceMatcher(None, left_text, right_text).ratio()
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    union = left_set | right_set
    token_score = len(left_set & right_set) / len(union) if union else 0.0
    return Decimal(str(round(max(sequence_score, token_score), 6)))


@dataclass(frozen=True, slots=True)
class ExtractionMetadata:
    """Provenance for extracted fields; it never implies that a model ran."""

    source: ExtractionSource = ExtractionSource.UNKNOWN
    confidence: Decimal | str | None = None
    grounded: bool = False
    model_name: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        source = _enum_value(self.source, ExtractionSource, "source")
        object.__setattr__(self, "source", source)
        if not isinstance(self.grounded, bool):
            raise ValidationError("grounded must be true or false")

        model_sources = {ExtractionSource.SMALL_MODEL, ExtractionSource.LARGER_MODEL}
        scored_sources = model_sources | {ExtractionSource.HEURISTIC}
        if source in scored_sources:
            if self.confidence is None:
                raise ValidationError(f"{source.value} extraction requires a confidence")
            object.__setattr__(
                self, "confidence", _probability(self.confidence, "confidence")
            )
        elif self.confidence is not None:
            raise ValidationError(
                "confidence may only be recorded for a heuristic or a named model"
            )

        if source in model_sources:
            object.__setattr__(
                self, "model_name", _required_text(self.model_name, "model_name", limit=128)
            )
        elif self.model_name is not None:
            raise ValidationError("model_name is only valid when a model was used")
        if not isinstance(self.note, str):
            raise ValidationError("note must be text")


@dataclass(frozen=True, slots=True)
class Invoice:
    invoice_number: str
    counterparty: str
    amount: Decimal | str
    issue_date: date | str
    due_date: date | str | None
    side: LedgerSide
    currency: str = "USD"
    extraction: ExtractionMetadata = field(default_factory=ExtractionMetadata)
    approved: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "invoice_number", _required_text(self.invoice_number, "invoice_number", limit=128)
        )
        object.__setattr__(
            self, "counterparty", _required_text(self.counterparty, "counterparty")
        )
        object.__setattr__(self, "amount", _positive_money(self.amount, "amount"))
        object.__setattr__(self, "issue_date", parse_iso_date(self.issue_date, "issue_date"))
        if self.due_date is not None:
            object.__setattr__(self, "due_date", parse_iso_date(self.due_date, "due_date"))
        object.__setattr__(self, "side", _enum_value(self.side, LedgerSide, "side"))
        object.__setattr__(self, "currency", _currency(self.currency))
        if self.due_date is not None and self.due_date < self.issue_date:
            raise ValidationError("due_date cannot be before issue_date")
        if not isinstance(self.extraction, ExtractionMetadata):
            raise ValidationError("extraction must be ExtractionMetadata")
        if not isinstance(self.approved, bool):
            raise ValidationError("approved must be true or false")

    @property
    def key(self) -> tuple[str, str, LedgerSide, str]:
        return (
            normalize_identifier(self.invoice_number),
            normalize_identifier(self.counterparty),
            self.side,
            self.currency,
        )


@dataclass(frozen=True, slots=True)
class PaymentReceipt:
    receipt_number: str
    counterparty: str
    amount: Decimal | str
    payment_date: date | str
    side: LedgerSide
    invoice_reference: str | None = None
    currency: str = "USD"
    extraction: ExtractionMetadata = field(default_factory=ExtractionMetadata)
    approved: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "receipt_number", _required_text(self.receipt_number, "receipt_number", limit=128)
        )
        object.__setattr__(
            self, "counterparty", _required_text(self.counterparty, "counterparty")
        )
        object.__setattr__(self, "amount", _positive_money(self.amount, "amount"))
        object.__setattr__(
            self, "payment_date", parse_iso_date(self.payment_date, "payment_date")
        )
        object.__setattr__(self, "side", _enum_value(self.side, LedgerSide, "side"))
        object.__setattr__(self, "currency", _currency(self.currency))
        if self.invoice_reference is not None:
            object.__setattr__(
                self,
                "invoice_reference",
                _required_text(self.invoice_reference, "invoice_reference", limit=128),
            )
        if not isinstance(self.extraction, ExtractionMetadata):
            raise ValidationError("extraction must be ExtractionMetadata")
        if not isinstance(self.approved, bool):
            raise ValidationError("approved must be true or false")


@dataclass(frozen=True, slots=True)
class Allocation:
    invoice_number: str
    receipt_number: str
    amount: Decimal
    method: MatchMethod
    score: Decimal
    reason: str


@dataclass(frozen=True, slots=True)
class InvoiceReconciliation:
    invoice: Invoice
    status: InvoiceStatus
    paid_amount: Decimal
    outstanding_amount: Decimal
    allocations: tuple[Allocation, ...]


@dataclass(frozen=True, slots=True)
class ReceiptReconciliation:
    receipt: PaymentReceipt
    status: ReceiptStatus
    allocated_amount: Decimal
    unallocated_amount: Decimal
    allocations: tuple[Allocation, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    invoices: tuple[InvoiceReconciliation, ...]
    receipts: tuple[ReceiptReconciliation, ...]
    allocations: tuple[Allocation, ...]

    @property
    def needs_review(self) -> tuple[ReceiptReconciliation, ...]:
        return tuple(
            receipt
            for receipt in self.receipts
            if receipt.status in {ReceiptStatus.PARTIALLY_MATCHED, ReceiptStatus.NEEDS_REVIEW}
        )


def _amount_fit(payment: Decimal, outstanding: Decimal) -> Decimal:
    if payment == outstanding:
        return Decimal("1")
    if payment < outstanding:
        ratio = payment / outstanding
        return Decimal("0.75") + Decimal("0.25") * ratio
    return Decimal("0.50") * (outstanding / payment)


def _date_fit(payment_date: date, invoice: Invoice) -> Decimal:
    if payment_date < invoice.issue_date:
        return Decimal("0")
    age = (payment_date - invoice.issue_date).days
    if age <= 366:
        return Decimal("1")
    if age <= 730:
        return Decimal("0.70")
    return Decimal("0.30")


def _fallback_score(
    receipt: PaymentReceipt, invoice: Invoice, outstanding: Decimal
) -> Decimal:
    party = _counterparty_similarity(receipt.counterparty, invoice.counterparty)
    amount = _amount_fit(receipt.amount, outstanding)
    timing = _date_fit(receipt.payment_date, invoice)
    score = Decimal("0.45") * party + Decimal("0.40") * amount + Decimal("0.15") * timing
    return score.quantize(Decimal("0.0001"))


def reconcile(
    invoices: Sequence[Invoice] | Iterable[Invoice],
    receipts: Sequence[PaymentReceipt] | Iterable[PaymentReceipt],
    *,
    minimum_fallback_score: Decimal | str = Decimal("0.85"),
    minimum_score_margin: Decimal | str = Decimal("0.10"),
) -> ReconciliationReport:
    """Reconcile payments without guessing when evidence is weak or ambiguous.

    Explicit invoice references are processed before unreferenced receipts so a
    weak fallback can never consume the balance intended by a later explicit
    payment. A receipt is allocated to at most one invoice; batch payments and
    any overpayment remainder are sent to review.
    """

    invoice_list = tuple(invoices)
    receipt_list = tuple(receipts)
    if any(not isinstance(item, Invoice) for item in invoice_list):
        raise ValidationError("invoices must contain only Invoice records")
    if any(not isinstance(item, PaymentReceipt) for item in receipt_list):
        raise ValidationError("receipts must contain only PaymentReceipt records")
    if any(
        not item.approved or item.extraction.source == ExtractionSource.UNKNOWN
        for item in invoice_list
    ):
        raise ValidationError(
            "every invoice must be accepted by the quality gate or confirmed manually"
        )
    if any(
        not item.approved or item.extraction.source == ExtractionSource.UNKNOWN
        for item in receipt_list
    ):
        raise ValidationError(
            "every receipt must be accepted by the quality gate or confirmed manually"
        )

    threshold = _probability(minimum_fallback_score, "minimum_fallback_score")
    margin = _probability(minimum_score_margin, "minimum_score_margin")

    seen_invoices: set[tuple[str, str, LedgerSide, str]] = set()
    for invoice in invoice_list:
        if invoice.key in seen_invoices:
            raise ValidationError(
                "duplicate invoice identity; add a distinct number or counterparty"
            )
        seen_invoices.add(invoice.key)

    seen_receipts: set[tuple[str, LedgerSide, str]] = set()
    for receipt in receipt_list:
        receipt_key = (
            normalize_identifier(receipt.receipt_number),
            receipt.side,
            receipt.currency,
        )
        if receipt_key in seen_receipts:
            raise ValidationError("duplicate receipt identity would double-count payment")
        seen_receipts.add(receipt_key)

    outstanding = [invoice.amount for invoice in invoice_list]
    invoice_allocations: list[list[Allocation]] = [[] for _ in invoice_list]
    receipt_results: list[ReceiptReconciliation | None] = [None for _ in receipt_list]
    all_allocations: list[Allocation] = []

    def record_allocation(
        receipt_index: int,
        invoice_index: int,
        method: MatchMethod,
        score: Decimal,
        reason: str,
    ) -> None:
        receipt = receipt_list[receipt_index]
        invoice = invoice_list[invoice_index]
        allocated = min(receipt.amount, outstanding[invoice_index])
        outstanding[invoice_index] -= allocated
        allocation = Allocation(
            invoice_number=invoice.invoice_number,
            receipt_number=receipt.receipt_number,
            amount=allocated,
            method=method,
            score=score,
            reason=reason,
        )
        invoice_allocations[invoice_index].append(allocation)
        all_allocations.append(allocation)
        unallocated = receipt.amount - allocated
        if unallocated == ZERO:
            status = ReceiptStatus.MATCHED
            result_reason = reason
        else:
            status = ReceiptStatus.PARTIALLY_MATCHED
            result_reason = (
                f"{reason}; {unallocated:.2f} remains unallocated and needs review"
            )
        receipt_results[receipt_index] = ReceiptReconciliation(
            receipt=receipt,
            status=status,
            allocated_amount=allocated,
            unallocated_amount=unallocated,
            allocations=(allocation,),
            reason=result_reason,
        )

    def mark_review(receipt_index: int, reason: str) -> None:
        receipt = receipt_list[receipt_index]
        receipt_results[receipt_index] = ReceiptReconciliation(
            receipt=receipt,
            status=ReceiptStatus.NEEDS_REVIEW,
            allocated_amount=ZERO,
            unallocated_amount=receipt.amount,
            allocations=(),
            reason=reason,
        )

    explicit_indices = [
        index for index, receipt in enumerate(receipt_list) if receipt.invoice_reference
    ]
    fallback_indices = [
        index for index, receipt in enumerate(receipt_list) if not receipt.invoice_reference
    ]

    for receipt_index in explicit_indices:
        receipt = receipt_list[receipt_index]
        reference = normalize_identifier(receipt.invoice_reference or "")
        same_reference = [
            index
            for index, invoice in enumerate(invoice_list)
            if invoice.side == receipt.side
            and invoice.currency == receipt.currency
            and normalize_identifier(invoice.invoice_number) == reference
        ]
        candidates = [index for index in same_reference if outstanding[index] > ZERO]
        if not same_reference:
            mark_review(receipt_index, "explicit invoice reference was not found")
            continue
        if not candidates:
            mark_review(receipt_index, "explicitly referenced invoice is already paid")
            continue

        exact_party = [
            index
            for index in candidates
            if normalize_identifier(invoice_list[index].counterparty)
            == normalize_identifier(receipt.counterparty)
        ]
        if len(exact_party) == 1:
            candidates = exact_party
        elif len(exact_party) > 1:
            mark_review(receipt_index, "invoice reference matches multiple open invoices")
            continue
        elif len(candidates) != 1:
            mark_review(
                receipt_index,
                "invoice reference is ambiguous and counterparty did not disambiguate it",
            )
            continue

        invoice_index = candidates[0]
        if receipt.payment_date < invoice_list[invoice_index].issue_date:
            mark_review(
                receipt_index,
                "payment date is before the referenced invoice was issued",
            )
            continue
        party_similarity = _counterparty_similarity(
            receipt.counterparty, invoice_list[invoice_index].counterparty
        )
        if party_similarity < Decimal("0.80"):
            mark_review(
                receipt_index,
                "invoice reference matched but counterparty conflicts; review for safety",
            )
            continue
        if receipt.amount > outstanding[invoice_index]:
            mark_review(
                receipt_index,
                "payment exceeds the outstanding balance; no amount was allocated",
            )
            continue
        record_allocation(
            receipt_index,
            invoice_index,
            MatchMethod.EXPLICIT_REFERENCE,
            Decimal("1.0000"),
            "matched the explicit invoice reference and counterparty",
        )

    for receipt_index in fallback_indices:
        receipt = receipt_list[receipt_index]
        candidate_scores: list[tuple[Decimal, int]] = []
        for invoice_index, invoice in enumerate(invoice_list):
            if (
                invoice.side != receipt.side
                or invoice.currency != receipt.currency
                or outstanding[invoice_index] <= ZERO
                or receipt.payment_date < invoice.issue_date
                or receipt.amount != outstanding[invoice_index]
            ):
                continue
            score = _fallback_score(receipt, invoice, outstanding[invoice_index])
            candidate_scores.append((score, invoice_index))

        if not candidate_scores:
            mark_review(receipt_index, "no open invoice exists on the same ledger side and currency")
            continue
        candidate_scores.sort(
            key=lambda item: (-item[0], invoice_list[item[1]].key)
        )
        best_score, best_index = candidate_scores[0]
        if best_score < threshold:
            mark_review(
                receipt_index,
                f"best weighted match {best_score} is below the safe threshold {threshold}",
            )
            continue
        if len(candidate_scores) > 1:
            second_score = candidate_scores[1][0]
            if best_score - second_score < margin:
                mark_review(
                    receipt_index,
                    "weighted candidates are too close; ambiguity fails closed",
                )
                continue
        if receipt.amount > outstanding[best_index]:
            mark_review(
                receipt_index,
                "payment exceeds the outstanding balance; no amount was allocated",
            )
            continue
        record_allocation(
            receipt_index,
            best_index,
            MatchMethod.WEIGHTED_FALLBACK,
            best_score,
            "matched by counterparty, outstanding amount, and payment timing",
        )

    invoice_results: list[InvoiceReconciliation] = []
    for index, invoice in enumerate(invoice_list):
        balance = outstanding[index]
        paid = invoice.amount - balance
        if paid == ZERO:
            status = InvoiceStatus.OPEN
        elif balance == ZERO:
            status = InvoiceStatus.PAID
        else:
            status = InvoiceStatus.PARTIALLY_PAID
        invoice_results.append(
            InvoiceReconciliation(
                invoice=invoice,
                status=status,
                paid_amount=paid,
                outstanding_amount=balance,
                allocations=tuple(invoice_allocations[index]),
            )
        )

    # Every receipt is assigned in exactly one of the two processing passes.
    finalized_receipts = tuple(result for result in receipt_results if result is not None)
    if len(finalized_receipts) != len(receipt_list):  # pragma: no cover - invariant guard
        raise RuntimeError("internal reconciliation invariant failed")
    return ReconciliationReport(
        invoices=tuple(invoice_results),
        receipts=tuple(receipt_results),  # type: ignore[arg-type]
        allocations=tuple(all_allocations),
    )


@dataclass(frozen=True, slots=True)
class CashFlowSummary:
    currency: str
    as_of: date
    total_payables: Decimal
    total_receivables: Decimal
    cash_paid: Decimal
    cash_received: Decimal
    outstanding_payables: Decimal
    outstanding_receivables: Decimal
    overdue_payables: Decimal
    overdue_receivables: Decimal
    payables_due_next_30_days: Decimal
    receivables_due_next_30_days: Decimal
    unallocated_receipts: Decimal
    net_cash_movement: Decimal
    receipts_needing_review: int


def summarize_cash_flow(
    report: ReconciliationReport,
    *,
    as_of: date | str | None = None,
    currency: str = "USD",
) -> CashFlowSummary:
    """Summarize AP cash out, AR cash in, outstanding balances, and near-term risk."""

    if not isinstance(report, ReconciliationReport):
        raise ValidationError("report must be a ReconciliationReport")
    summary_date = date.today() if as_of is None else parse_iso_date(as_of, "as_of")
    summary_currency = _currency(currency)
    values = {
        "total_payables": ZERO,
        "total_receivables": ZERO,
        "cash_paid": ZERO,
        "cash_received": ZERO,
        "outstanding_payables": ZERO,
        "outstanding_receivables": ZERO,
        "overdue_payables": ZERO,
        "overdue_receivables": ZERO,
        "payables_due_next_30_days": ZERO,
        "receivables_due_next_30_days": ZERO,
    }
    horizon = summary_date + timedelta(days=30)
    receipt_dates = {
        (
            normalize_identifier(item.receipt.receipt_number),
            item.receipt.side,
            item.receipt.currency,
        ): item.receipt.payment_date
        for item in report.receipts
    }
    for item in report.invoices:
        invoice = item.invoice
        if invoice.currency != summary_currency or invoice.issue_date > summary_date:
            continue
        paid_as_of = sum(
            (
                allocation.amount
                for allocation in item.allocations
                if receipt_dates.get(
                    (
                        normalize_identifier(allocation.receipt_number),
                        invoice.side,
                        invoice.currency,
                    )
                )
                is not None
                and receipt_dates[
                    (
                        normalize_identifier(allocation.receipt_number),
                        invoice.side,
                        invoice.currency,
                    )
                ]
                <= summary_date
            ),
            ZERO,
        )
        outstanding_as_of = invoice.amount - paid_as_of
        if invoice.side == LedgerSide.ACCOUNTS_PAYABLE:
            values["total_payables"] += invoice.amount
            values["cash_paid"] += paid_as_of
            values["outstanding_payables"] += outstanding_as_of
            if invoice.due_date is not None and invoice.due_date < summary_date:
                values["overdue_payables"] += outstanding_as_of
            elif invoice.due_date is not None and invoice.due_date <= horizon:
                values["payables_due_next_30_days"] += outstanding_as_of
        else:
            values["total_receivables"] += invoice.amount
            values["cash_received"] += paid_as_of
            values["outstanding_receivables"] += outstanding_as_of
            if invoice.due_date is not None and invoice.due_date < summary_date:
                values["overdue_receivables"] += outstanding_as_of
            elif invoice.due_date is not None and invoice.due_date <= horizon:
                values["receivables_due_next_30_days"] += outstanding_as_of

    review_receipts = [
        item
        for item in report.receipts
        if item.receipt.currency == summary_currency
        and item.receipt.payment_date <= summary_date
        and item.status in {ReceiptStatus.PARTIALLY_MATCHED, ReceiptStatus.NEEDS_REVIEW}
    ]
    unallocated = sum((item.unallocated_amount for item in review_receipts), ZERO)
    return CashFlowSummary(
        currency=summary_currency,
        as_of=summary_date,
        total_payables=values["total_payables"],
        total_receivables=values["total_receivables"],
        cash_paid=values["cash_paid"],
        cash_received=values["cash_received"],
        outstanding_payables=values["outstanding_payables"],
        outstanding_receivables=values["outstanding_receivables"],
        overdue_payables=values["overdue_payables"],
        overdue_receivables=values["overdue_receivables"],
        payables_due_next_30_days=values["payables_due_next_30_days"],
        receivables_due_next_30_days=values["receivables_due_next_30_days"],
        unallocated_receipts=unallocated,
        net_cash_movement=values["cash_received"] - values["cash_paid"],
        receipts_needing_review=len(review_receipts),
    )


@dataclass(frozen=True, slots=True)
class RoutingSignals:
    extraction: ExtractionMetadata
    ocr_quality: Decimal | str | None
    validation_passed: bool
    heuristic_model_agreement: bool | None
    ambiguity_detected: bool = False
    escalation_available: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.extraction, ExtractionMetadata):
            raise ValidationError("extraction must be ExtractionMetadata")
        for field_name in (
            "validation_passed",
            "ambiguity_detected",
            "escalation_available",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValidationError(f"{field_name} must be true or false")
        if self.heuristic_model_agreement not in {True, False, None}:
            raise ValidationError("heuristic_model_agreement must be true, false, or unknown")
        if self.ocr_quality is not None:
            object.__setattr__(
                self, "ocr_quality", _probability(self.ocr_quality, "ocr_quality")
            )


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    action: RoutingAction
    reasons: tuple[str, ...]
    accepted_source: ExtractionSource | None
    stayed_local: bool


def decide_small_first_route(
    signals: RoutingSignals,
    *,
    minimum_confidence: Decimal | str = Decimal("0.90"),
    minimum_ocr_quality: Decimal | str = Decimal("0.75"),
    uncorroborated_confidence: Decimal | str = Decimal("0.97"),
) -> RoutingDecision:
    """Accept strong local evidence, escalate uncertainty, and fail closed.

    Agreement is useful when both a heuristic and model produced a candidate.
    If the heuristic had no opinion, a higher model-confidence threshold is
    required. This decision records provenance; it does not run any model.
    """

    if not isinstance(signals, RoutingSignals):
        raise ValidationError("signals must be RoutingSignals")
    confidence_floor = _probability(minimum_confidence, "minimum_confidence")
    ocr_floor = _probability(minimum_ocr_quality, "minimum_ocr_quality")
    solo_floor = _probability(uncorroborated_confidence, "uncorroborated_confidence")
    metadata = signals.extraction

    def uncertain(*reasons: str) -> RoutingDecision:
        can_escalate = (
            signals.escalation_available
            and metadata.source
            not in {ExtractionSource.UNKNOWN, ExtractionSource.LARGER_MODEL}
        )
        return RoutingDecision(
            action=RoutingAction.ESCALATE if can_escalate else RoutingAction.HUMAN_REVIEW,
            reasons=tuple(reasons),
            accepted_source=None,
            stayed_local=False,
        )

    if metadata.source == ExtractionSource.UNKNOWN:
        return uncertain("extraction source is unknown")
    if signals.ambiguity_detected:
        return uncertain("multiple plausible values were detected")
    if not signals.validation_passed:
        return uncertain("candidate failed deterministic validation")

    if metadata.source == ExtractionSource.MANUAL:
        return RoutingDecision(
            action=RoutingAction.ACCEPT,
            reasons=("manually entered value passed deterministic validation",),
            accepted_source=metadata.source,
            stayed_local=True,
        )

    if not metadata.grounded:
        return uncertain("candidate is not grounded in OCR evidence")
    if metadata.confidence is None:  # guarded by ExtractionMetadata; defensive only
        return uncertain("extractor confidence is missing")
    if signals.ocr_quality is None:
        return uncertain("OCR quality was not measured")
    if signals.ocr_quality < ocr_floor:
        return uncertain(
            f"OCR quality {signals.ocr_quality} is below {ocr_floor}"
        )
    if signals.heuristic_model_agreement is False:
        return uncertain("heuristic and model disagree")
    if metadata.confidence < confidence_floor:
        return uncertain(
            f"extractor confidence {metadata.confidence} is below {confidence_floor}"
        )
    if (
        signals.heuristic_model_agreement is None
        and metadata.confidence < solo_floor
    ):
        return uncertain(
            "no independent heuristic agreed and confidence is below the stricter solo threshold"
        )

    stayed_local = metadata.source in {
        ExtractionSource.HEURISTIC,
        ExtractionSource.SMALL_MODEL,
    }
    return RoutingDecision(
        action=RoutingAction.ACCEPT,
        reasons=("grounded candidate passed confidence, OCR, agreement, and validation gates",),
        accepted_source=metadata.source,
        stayed_local=stayed_local,
    )
