"""Frozen, dependency-free contracts for the ProcureAgent MVP.

The module is intentionally limited to data contracts and the locked demo
fixture loader.  Policy, lookup, verification, and simulation behavior belong
to their respective components and consume these immutable values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, TypeVar


class ContractValidationError(ValueError):
    """Raised when data does not satisfy the frozen ProcureAgent contract."""


class RecordSource(str, Enum):
    SYNTHETIC_FIXTURE_LOOKUP = "synthetic_fixture_lookup"


class SupplierSource(str, Enum):
    OPERATOR_SELECTED = "operator_selected"
    MODEL_HINT = "model_hint"
    UNKNOWN = "unknown"


class DocumentMethod(str, Enum):
    ANCHORED_RULE = "anchored_rule"
    LAYOUTLMV3_LOCAL = "layoutlmv3_local"
    COMBINED_EVIDENCE = "combined_evidence"
    HUMAN_REVIEW = "human_review"
    REPLAY = "replay"


class DocumentStatus(str, Enum):
    PROPOSED = "PROPOSED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    CONFIRMED = "CONFIRMED"
    CORRECTED = "CORRECTED"
    REJECTED = "REJECTED"


class DocumentReviewDecision(str, Enum):
    CONFIRM = "CONFIRM"
    CORRECT = "CORRECT"
    REJECT = "REJECT"


class SupplierCriticality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SupplierStatus(str, Enum):
    ACTIVE = "active"
    AT_RISK = "at_risk"
    ON_HOLD = "on_hold"
    DISRUPTED = "disrupted"
    UNKNOWN = "unknown"


class InvoicePaymentStatus(str, Enum):
    UNPAID = "unpaid"
    SIMULATED_PAYMENT_APPROVED = "simulated_payment_approved"
    PAID_CONFIRMED = "paid_confirmed"


class PaymentProofSource(str, Enum):
    OPERATOR_UPLOAD = "operator_upload"
    SYNTHETIC_FIXTURE_REPLAY = "synthetic_fixture_replay"


class PaymentProofStatus(str, Enum):
    PROPOSED = "PROPOSED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"


class ProcurementAction(str, Enum):
    PAY = "PAY"
    DEFER = "DEFER"
    VERIFY = "VERIFY"


class PolicyType(str, Enum):
    DETERMINISTIC_RULES = "deterministic_rules"
    MODEL_DRIVEN = "model_driven"


class VerifierResult(str, Enum):
    BLOCKED = "BLOCKED"
    REQUIRES_OPERATOR = "REQUIRES_OPERATOR"
    VERIFIED = "VERIFIED"


class OperatorDecisionType(str, Enum):
    APPROVE = "APPROVE"
    MODIFY = "MODIFY"
    REJECT = "REJECT"


_E = TypeVar("_E", bound=Enum)
_AUDIT_ID_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,126}[A-Za-z0-9])?\Z")
_INVOICE_NUMBER_PATTERN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._/:-]{0,126}[A-Za-z0-9])?\Z"
)
_CURRENCY_PATTERN = re.compile(r"[A-Z]{3}\Z")
_REASON_CODE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
_ISO_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_CONTRACT_VERSION = "procureagent-contract-v1"


def _text(value: object, field_name: str, *, limit: int = 128) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{field_name} must be text")
    if not value or value != value.strip():
        raise ContractValidationError(
            f"{field_name} must be non-empty text without surrounding whitespace"
        )
    if len(value) > limit:
        raise ContractValidationError(
            f"{field_name} must be at most {limit} characters"
        )
    return value


def _audit_id(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    if not _AUDIT_ID_PATTERN.fullmatch(text) or ".." in text:
        raise ContractValidationError(
            f"{field_name} must be an audit-friendly identifier"
        )
    return text


def _invoice_number(value: object) -> str:
    text = _text(value, "invoice_number")
    if not _INVOICE_NUMBER_PATTERN.fullmatch(text) or ".." in text:
        raise ContractValidationError("invoice_number contains unsupported characters")
    return text


def _currency(value: object) -> str:
    if not isinstance(value, str) or not _CURRENCY_PATTERN.fullmatch(value):
        raise ContractValidationError(
            "currency must be a three-letter uppercase code such as USD"
        )
    return value


def _date(value: object, field_name: str) -> date:
    if isinstance(value, datetime):
        raise ContractValidationError(f"{field_name} must be a date, not a datetime")
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not _ISO_DATE_PATTERN.fullmatch(value):
        raise ContractValidationError(f"{field_name} must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ContractValidationError(f"{field_name} must be a real calendar date") from exc


def _integer(
    value: object,
    field_name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractValidationError(f"{field_name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ContractValidationError(f"{field_name} must be at most {maximum}")
    return value


def _minor_units(value: object, field_name: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    return _integer(value, field_name, minimum=minimum)


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"{field_name} must be true or false")
    return value


def _enum(value: object, enum_type: type[_E], field_name: str) -> _E:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            raise ContractValidationError(
                f"{field_name} is not a supported {enum_type.__name__}"
            ) from exc
    raise ContractValidationError(f"{field_name} must be a {enum_type.__name__}")


def _confidence(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, (int, float)):
        raise ContractValidationError(
            f"{field_name} must be Decimal or a plain decimal string"
        )
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} must be a decimal score") from exc
    if not result.is_finite() or result < 0 or result > 1:
        raise ContractValidationError(f"{field_name} must be between 0 and 1")
    return result


def _objects(
    value: Iterable[object], expected_type: type[Any], field_name: str
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ContractValidationError(f"{field_name} must be a sequence")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ContractValidationError(f"{field_name} must be a sequence") from exc
    if not all(isinstance(item, expected_type) for item in items):
        raise ContractValidationError(
            f"every {field_name} item must be {expected_type.__name__}"
        )
    return items


def _texts(
    value: Iterable[object],
    field_name: str,
    *,
    require_nonempty: bool = False,
    reason_codes: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ContractValidationError(f"{field_name} must be a sequence of text values")
    try:
        raw_items = tuple(value)
    except TypeError as exc:
        raise ContractValidationError(
            f"{field_name} must be a sequence of text values"
        ) from exc
    items = tuple(_text(item, field_name, limit=256) for item in raw_items)
    if require_nonempty and not items:
        raise ContractValidationError(f"{field_name} cannot be empty")
    if len(items) != len(set(items)):
        raise ContractValidationError(f"{field_name} cannot contain duplicates")
    if reason_codes and any(not _REASON_CODE_PATTERN.fullmatch(item) for item in items):
        raise ContractValidationError(
            f"{field_name} values must be uppercase machine-readable reason codes"
        )
    return items


def make_audit_id(namespace: str, *parts: object) -> str:
    """Build a stable, human-readable ID from already-known audit fields."""

    prefix = _audit_id(namespace, "namespace").lower()
    if not parts:
        raise ContractValidationError("an audit ID needs at least one identifying part")
    normalized: list[str] = []
    for index, part in enumerate(parts):
        text = _text(str(part), f"parts[{index}]").strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
        if not slug:
            raise ContractValidationError(f"parts[{index}] has no identifier characters")
        normalized.append(slug)
    return _audit_id("-".join((prefix, *normalized)), "audit_id")


@dataclass(frozen=True, slots=True, order=True)
class SupplierIdentity:
    supplier_id: str
    display_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "supplier_id", _audit_id(self.supplier_id, "supplier_id"))
        object.__setattr__(self, "display_name", _text(self.display_name, "display_name"))


@dataclass(frozen=True, slots=True, order=True)
class InvoiceIdentity:
    supplier_id: str
    invoice_number: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "supplier_id", _audit_id(self.supplier_id, "supplier_id"))
        object.__setattr__(self, "invoice_number", _invoice_number(self.invoice_number))


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """One normalized OCR box in the inclusive 0..1000 LayoutLM coordinate space."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        for name in ("x0", "y0", "x1", "y1"):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=0, maximum=1000)
            )
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ContractValidationError("bounding box coordinates are reversed")


@dataclass(frozen=True, slots=True)
class InvoiceNumberCandidate:
    invoice_number: str
    entity_confidence: Decimal | str
    grounded_in_ocr: bool
    evidence_tokens: tuple[str, ...]
    evidence_boxes: tuple[BoundingBox, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "invoice_number", _invoice_number(self.invoice_number))
        object.__setattr__(
            self,
            "entity_confidence",
            _confidence(self.entity_confidence, "entity_confidence"),
        )
        object.__setattr__(
            self, "grounded_in_ocr", _boolean(self.grounded_in_ocr, "grounded_in_ocr")
        )
        tokens = _texts(self.evidence_tokens, "evidence_tokens")
        boxes = _objects(self.evidence_boxes, BoundingBox, "evidence_boxes")
        object.__setattr__(self, "evidence_tokens", tokens)
        object.__setattr__(self, "evidence_boxes", boxes)
        if len(tokens) != len(boxes):
            raise ContractValidationError(
                "evidence_tokens and evidence_boxes must have the same length"
            )
        if self.grounded_in_ocr and not tokens:
            raise ContractValidationError("grounded candidates require OCR evidence")


@dataclass(frozen=True, slots=True)
class DocumentIdentityProposal:
    document_id: str
    supplier_id: str
    supplier_source: SupplierSource
    supplier_confirmed: bool
    candidate_spans: tuple[InvoiceNumberCandidate, ...]
    method: DocumentMethod
    status: DocumentStatus = DocumentStatus.PROPOSED
    model_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _audit_id(self.document_id, "document_id"))
        object.__setattr__(self, "supplier_id", _audit_id(self.supplier_id, "supplier_id"))
        object.__setattr__(
            self,
            "supplier_source",
            _enum(self.supplier_source, SupplierSource, "supplier_source"),
        )
        object.__setattr__(
            self,
            "supplier_confirmed",
            _boolean(self.supplier_confirmed, "supplier_confirmed"),
        )
        candidates = _objects(
            self.candidate_spans, InvoiceNumberCandidate, "candidate_spans"
        )
        object.__setattr__(self, "candidate_spans", candidates)
        method = _enum(self.method, DocumentMethod, "method")
        status = _enum(self.status, DocumentStatus, "status")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "status", status)
        if len({candidate.invoice_number for candidate in candidates}) != len(candidates):
            raise ContractValidationError("candidate invoice numbers must be unique")
        if method is DocumentMethod.LAYOUTLMV3_LOCAL:
            object.__setattr__(
                self,
                "model_version",
                _text(self.model_version, "model_version", limit=512),
            )
        elif self.model_version is not None:
            object.__setattr__(
                self,
                "model_version",
                _text(self.model_version, "model_version", limit=512),
            )
        if status in {DocumentStatus.CONFIRMED, DocumentStatus.CORRECTED}:
            if not self.supplier_confirmed:
                raise ContractValidationError("confirmed identity requires confirmed supplier")
            if len(candidates) != 1 or not candidates[0].grounded_in_ocr:
                raise ContractValidationError(
                    "confirmed identity requires exactly one grounded invoice candidate"
                )

    @property
    def confirmed_identity(self) -> InvoiceIdentity | None:
        if self.status not in {DocumentStatus.CONFIRMED, DocumentStatus.CORRECTED}:
            return None
        return InvoiceIdentity(self.supplier_id, self.candidate_spans[0].invoice_number)


@dataclass(frozen=True, slots=True)
class VerifiedInvoiceIdentity:
    document_id: str
    supplier_id: str
    invoice_number: str
    status: DocumentStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _audit_id(self.document_id, "document_id"))
        identity = InvoiceIdentity(self.supplier_id, self.invoice_number)
        object.__setattr__(self, "supplier_id", identity.supplier_id)
        object.__setattr__(self, "invoice_number", identity.invoice_number)
        status = _enum(self.status, DocumentStatus, "status")
        if status not in {DocumentStatus.CONFIRMED, DocumentStatus.CORRECTED}:
            raise ContractValidationError(
                "verified identity status must be CONFIRMED or CORRECTED"
            )
        object.__setattr__(self, "status", status)

    @property
    def identity(self) -> InvoiceIdentity:
        return InvoiceIdentity(self.supplier_id, self.invoice_number)


@dataclass(frozen=True, slots=True)
class SyntheticSupplierInvoice:
    supplier_id: str
    invoice_number: str
    category: str
    amount_minor: int
    currency: str
    due_in_days: int
    inventory_days_remaining: int
    delivery_lead_days: int
    payment_unlocks_delivery: bool
    delivery_inventory_days: int
    supplier_criticality: SupplierCriticality
    supplier_status: SupplierStatus
    payment_status: InvoicePaymentStatus
    state_version: int
    late_fee_minor_per_day: int
    disruption_after_days_overdue: int | None
    context_conflict_codes: tuple[str, ...] = ()
    record_source: RecordSource = RecordSource.SYNTHETIC_FIXTURE_LOOKUP

    def __post_init__(self) -> None:
        identity = InvoiceIdentity(self.supplier_id, self.invoice_number)
        object.__setattr__(self, "supplier_id", identity.supplier_id)
        object.__setattr__(self, "invoice_number", identity.invoice_number)
        object.__setattr__(self, "category", _audit_id(self.category, "category"))
        object.__setattr__(
            self, "amount_minor", _minor_units(self.amount_minor, "amount_minor", positive=True)
        )
        object.__setattr__(self, "currency", _currency(self.currency))
        object.__setattr__(
            self,
            "due_in_days",
            _integer(self.due_in_days, "due_in_days", minimum=-3650, maximum=3650),
        )
        object.__setattr__(
            self,
            "inventory_days_remaining",
            _integer(
                self.inventory_days_remaining,
                "inventory_days_remaining",
                minimum=0,
                maximum=3650,
            ),
        )
        object.__setattr__(
            self,
            "delivery_lead_days",
            _integer(
                self.delivery_lead_days,
                "delivery_lead_days",
                minimum=0,
                maximum=3650,
            ),
        )
        object.__setattr__(
            self,
            "payment_unlocks_delivery",
            _boolean(self.payment_unlocks_delivery, "payment_unlocks_delivery"),
        )
        object.__setattr__(
            self,
            "delivery_inventory_days",
            _integer(
                self.delivery_inventory_days,
                "delivery_inventory_days",
                minimum=0,
                maximum=3650,
            ),
        )
        if self.payment_unlocks_delivery and self.delivery_inventory_days == 0:
            raise ContractValidationError(
                "a payment-unlocked delivery requires positive inventory coverage"
            )
        if not self.payment_unlocks_delivery and self.delivery_inventory_days != 0:
            raise ContractValidationError(
                "delivery_inventory_days must be zero when payment unlocks no delivery"
            )
        object.__setattr__(
            self,
            "supplier_criticality",
            _enum(
                self.supplier_criticality,
                SupplierCriticality,
                "supplier_criticality",
            ),
        )
        object.__setattr__(
            self,
            "supplier_status",
            _enum(self.supplier_status, SupplierStatus, "supplier_status"),
        )
        object.__setattr__(
            self,
            "payment_status",
            _enum(self.payment_status, InvoicePaymentStatus, "payment_status"),
        )
        object.__setattr__(
            self,
            "state_version",
            _integer(self.state_version, "state_version", minimum=1),
        )
        object.__setattr__(
            self,
            "late_fee_minor_per_day",
            _minor_units(self.late_fee_minor_per_day, "late_fee_minor_per_day"),
        )
        if self.disruption_after_days_overdue is not None:
            object.__setattr__(
                self,
                "disruption_after_days_overdue",
                _integer(
                    self.disruption_after_days_overdue,
                    "disruption_after_days_overdue",
                    minimum=1,
                    maximum=3650,
                ),
            )
        object.__setattr__(
            self,
            "context_conflict_codes",
            _texts(
                self.context_conflict_codes,
                "context_conflict_codes",
                reason_codes=True,
            ),
        )
        object.__setattr__(
            self,
            "record_source",
            _enum(self.record_source, RecordSource, "record_source"),
        )

    @property
    def identity(self) -> InvoiceIdentity:
        return InvoiceIdentity(self.supplier_id, self.invoice_number)


@dataclass(frozen=True, slots=True)
class PaymentProof:
    """Evidence for confirming one previously approved simulated full payment."""

    receipt_id: str
    supplier_id: str
    invoice_number: str
    amount_minor: int
    currency: str
    paid_date: date | str
    source: PaymentProofSource
    provenance: str
    status: PaymentProofStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_id", _audit_id(self.receipt_id, "receipt_id"))
        identity = InvoiceIdentity(self.supplier_id, self.invoice_number)
        object.__setattr__(self, "supplier_id", identity.supplier_id)
        object.__setattr__(self, "invoice_number", identity.invoice_number)
        object.__setattr__(
            self, "amount_minor", _minor_units(self.amount_minor, "amount_minor", positive=True)
        )
        object.__setattr__(self, "currency", _currency(self.currency))
        object.__setattr__(self, "paid_date", _date(self.paid_date, "paid_date"))
        object.__setattr__(
            self, "source", _enum(self.source, PaymentProofSource, "source")
        )
        object.__setattr__(
            self, "provenance", _text(self.provenance, "provenance", limit=256)
        )
        object.__setattr__(
            self, "status", _enum(self.status, PaymentProofStatus, "status")
        )

    @property
    def identity(self) -> InvoiceIdentity:
        return InvoiceIdentity(self.supplier_id, self.invoice_number)


# A descriptive alias for callers that prefer the product term "payment receipt".
PaymentReceipt = PaymentProof


def validate_full_payment_proof(
    invoice: SyntheticSupplierInvoice, proof: PaymentProof
) -> None:
    """Fail closed unless proof closes exactly one approved full invoice amount."""

    if not isinstance(invoice, SyntheticSupplierInvoice):
        raise ContractValidationError("invoice must be SyntheticSupplierInvoice")
    if not isinstance(proof, PaymentProof):
        raise ContractValidationError("proof must be PaymentProof")
    if invoice.payment_status is not InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED:
        raise ContractValidationError(
            "payment proof can close only a simulated-payment-approved invoice"
        )
    if proof.status is not PaymentProofStatus.VERIFIED:
        raise ContractValidationError("payment proof must be VERIFIED")
    if proof.identity != invoice.identity:
        raise ContractValidationError("payment proof composite invoice identity does not match")
    if proof.amount_minor != invoice.amount_minor:
        raise ContractValidationError("payment proof must match the full invoice amount")
    if proof.currency != invoice.currency:
        raise ContractValidationError("payment proof currency does not match")


@dataclass(frozen=True, slots=True)
class RestaurantState:
    restaurant_id: str
    scenario_id: str
    day: int
    state_version: int
    cash_minor: int
    currency: str
    invoices: tuple[SyntheticSupplierInvoice, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "restaurant_id", _audit_id(self.restaurant_id, "restaurant_id")
        )
        object.__setattr__(self, "scenario_id", _audit_id(self.scenario_id, "scenario_id"))
        object.__setattr__(self, "day", _integer(self.day, "day", minimum=0))
        object.__setattr__(
            self,
            "state_version",
            _integer(self.state_version, "state_version", minimum=1),
        )
        object.__setattr__(self, "cash_minor", _minor_units(self.cash_minor, "cash_minor"))
        object.__setattr__(self, "currency", _currency(self.currency))
        invoices = _objects(self.invoices, SyntheticSupplierInvoice, "invoices")
        object.__setattr__(self, "invoices", invoices)
        identities = tuple(invoice.identity for invoice in invoices)
        if len(identities) != len(set(identities)):
            raise ContractValidationError("restaurant invoices must have unique identities")
        if any(invoice.currency != self.currency for invoice in invoices):
            raise ContractValidationError("every invoice must use the restaurant currency")
        if any(invoice.state_version != self.state_version for invoice in invoices):
            raise ContractValidationError("invoice and restaurant state versions must match")

    @property
    def active_invoices(self) -> tuple[SyntheticSupplierInvoice, ...]:
        return tuple(
            invoice
            for invoice in self.invoices
            if invoice.payment_status is InvoicePaymentStatus.UNPAID
        )

    @property
    def total_obligations_minor(self) -> int:
        return sum(invoice.amount_minor for invoice in self.active_invoices)


@dataclass(frozen=True, slots=True)
class Recommendation:
    supplier_id: str
    invoice_number: str
    action: ProcurementAction
    amount_minor: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        identity = InvoiceIdentity(self.supplier_id, self.invoice_number)
        object.__setattr__(self, "supplier_id", identity.supplier_id)
        object.__setattr__(self, "invoice_number", identity.invoice_number)
        object.__setattr__(
            self, "action", _enum(self.action, ProcurementAction, "action")
        )
        object.__setattr__(
            self, "amount_minor", _minor_units(self.amount_minor, "amount_minor", positive=True)
        )
        object.__setattr__(
            self,
            "reason_codes",
            _texts(
                self.reason_codes,
                "reason_codes",
                require_nonempty=True,
                reason_codes=True,
            ),
        )

    @property
    def identity(self) -> InvoiceIdentity:
        return InvoiceIdentity(self.supplier_id, self.invoice_number)


@dataclass(frozen=True, slots=True)
class DailyRecommendationBatch:
    batch_id: str
    state_version: int
    policy_name: str
    policy_version: str
    policy_type: PolicyType
    recommendations: tuple[Recommendation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "batch_id", _audit_id(self.batch_id, "batch_id"))
        object.__setattr__(
            self,
            "state_version",
            _integer(self.state_version, "state_version", minimum=1),
        )
        object.__setattr__(self, "policy_name", _audit_id(self.policy_name, "policy_name"))
        object.__setattr__(
            self, "policy_version", _audit_id(self.policy_version, "policy_version")
        )
        object.__setattr__(
            self, "policy_type", _enum(self.policy_type, PolicyType, "policy_type")
        )
        recommendations = _objects(
            self.recommendations, Recommendation, "recommendations"
        )
        object.__setattr__(self, "recommendations", recommendations)
        if not recommendations:
            raise ContractValidationError("daily batch must contain recommendations")
        identities = tuple(item.identity for item in recommendations)
        if len(identities) != len(set(identities)):
            raise ContractValidationError(
                "daily batch cannot recommend the same invoice more than once"
            )


@dataclass(frozen=True, slots=True)
class VerifierDecision:
    verification_id: str
    batch_id: str
    result: VerifierResult
    reason_codes: tuple[str, ...]
    checks_passed: tuple[str, ...]
    verified_batch_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "verification_id", _audit_id(self.verification_id, "verification_id")
        )
        object.__setattr__(self, "batch_id", _audit_id(self.batch_id, "batch_id"))
        result = _enum(self.result, VerifierResult, "result")
        object.__setattr__(self, "result", result)
        object.__setattr__(
            self,
            "reason_codes",
            _texts(self.reason_codes, "reason_codes", reason_codes=True),
        )
        object.__setattr__(
            self,
            "checks_passed",
            _texts(self.checks_passed, "checks_passed", reason_codes=True),
        )
        if self.verified_batch_id is not None:
            object.__setattr__(
                self,
                "verified_batch_id",
                _audit_id(self.verified_batch_id, "verified_batch_id"),
            )
        if result is VerifierResult.BLOCKED:
            if self.verified_batch_id is not None:
                raise ContractValidationError("blocked verifier result cannot verify a batch")
            if not self.reason_codes:
                raise ContractValidationError("blocked verifier result needs a reason code")
        elif self.verified_batch_id != self.batch_id:
            raise ContractValidationError(
                "non-blocked verifier result must identify the reviewed batch"
            )


@dataclass(frozen=True, slots=True)
class OperatorDecision:
    decision_id: str
    reviewed_batch_id: str
    decision: OperatorDecisionType
    approved_batch_id: str | None = None
    replacement_batch_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", _audit_id(self.decision_id, "decision_id"))
        object.__setattr__(
            self,
            "reviewed_batch_id",
            _audit_id(self.reviewed_batch_id, "reviewed_batch_id"),
        )
        decision = _enum(self.decision, OperatorDecisionType, "decision")
        object.__setattr__(self, "decision", decision)
        if self.approved_batch_id is not None:
            object.__setattr__(
                self,
                "approved_batch_id",
                _audit_id(self.approved_batch_id, "approved_batch_id"),
            )
        if self.replacement_batch_id is not None:
            object.__setattr__(
                self,
                "replacement_batch_id",
                _audit_id(self.replacement_batch_id, "replacement_batch_id"),
            )
        if decision is OperatorDecisionType.APPROVE:
            if self.approved_batch_id != self.reviewed_batch_id:
                raise ContractValidationError("APPROVE must approve the reviewed batch")
            if self.replacement_batch_id is not None:
                raise ContractValidationError("APPROVE cannot create a replacement batch")
        elif decision is OperatorDecisionType.MODIFY:
            if self.approved_batch_id is not None:
                raise ContractValidationError("MODIFY is not approval")
            if (
                self.replacement_batch_id is None
                or self.replacement_batch_id == self.reviewed_batch_id
            ):
                raise ContractValidationError("MODIFY must create a new replacement batch ID")
        elif self.approved_batch_id is not None or self.replacement_batch_id is not None:
            raise ContractValidationError("REJECT cannot approve or replace a batch")


@dataclass(frozen=True, slots=True)
class AdversarialDocumentMetadata:
    document_id: str
    supplier_label: str
    candidate_invoice_numbers: tuple[str, ...]
    expected_status: DocumentStatus
    reason_codes: tuple[str, ...]
    expected_lookup_activated: bool
    included_in_obligations: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _audit_id(self.document_id, "document_id"))
        object.__setattr__(
            self, "supplier_label", _text(self.supplier_label, "supplier_label")
        )
        candidates = tuple(
            _invoice_number(value) for value in self.candidate_invoice_numbers
        )
        if len(candidates) < 2 or len(candidates) != len(set(candidates)):
            raise ContractValidationError(
                "adversarial ambiguous metadata needs at least two unique candidates"
            )
        object.__setattr__(self, "candidate_invoice_numbers", candidates)
        status = _enum(self.expected_status, DocumentStatus, "expected_status")
        object.__setattr__(self, "expected_status", status)
        object.__setattr__(
            self,
            "reason_codes",
            _texts(
                self.reason_codes,
                "reason_codes",
                require_nonempty=True,
                reason_codes=True,
            ),
        )
        object.__setattr__(
            self,
            "expected_lookup_activated",
            _boolean(self.expected_lookup_activated, "expected_lookup_activated"),
        )
        object.__setattr__(
            self,
            "included_in_obligations",
            _boolean(self.included_in_obligations, "included_in_obligations"),
        )
        if status is not DocumentStatus.REVIEW_REQUIRED:
            raise ContractValidationError(
                "ambiguous adversarial identity must remain REVIEW_REQUIRED"
            )
        if self.expected_lookup_activated or self.included_in_obligations:
            raise ContractValidationError(
                "unverified adversarial identity cannot activate lookup or obligations"
            )


@dataclass(frozen=True, slots=True)
class ProcureScenario:
    contract_version: str
    scenario_id: str
    seed: int
    horizon_days: int
    suppliers: tuple[SupplierIdentity, ...]
    initial_state: RestaurantState
    payment_proofs: tuple[PaymentProof, ...]
    adversarial_documents: tuple[AdversarialDocumentMetadata, ...]
    # Simulated daily restaurant revenue, credited at the end of each committed
    # day. Defaults to zero so restaurant_demo_v1 -- whose bytes are SHA-256
    # pinned and which therefore cannot carry this key -- behaves identically.
    daily_cash_inflow_minor: int = 0

    def __post_init__(self) -> None:
        version = _audit_id(self.contract_version, "contract_version")
        object.__setattr__(self, "contract_version", version)
        object.__setattr__(self, "scenario_id", _audit_id(self.scenario_id, "scenario_id"))
        object.__setattr__(self, "seed", _integer(self.seed, "seed", minimum=0))
        object.__setattr__(
            self,
            "horizon_days",
            _integer(self.horizon_days, "horizon_days", minimum=1, maximum=365),
        )
        object.__setattr__(
            self,
            "daily_cash_inflow_minor",
            _integer(self.daily_cash_inflow_minor, "daily_cash_inflow_minor", minimum=0),
        )
        suppliers = _objects(self.suppliers, SupplierIdentity, "suppliers")
        payment_proofs = _objects(self.payment_proofs, PaymentProof, "payment_proofs")
        adversarial = _objects(
            self.adversarial_documents,
            AdversarialDocumentMetadata,
            "adversarial_documents",
        )
        object.__setattr__(self, "suppliers", suppliers)
        object.__setattr__(self, "payment_proofs", payment_proofs)
        object.__setattr__(self, "adversarial_documents", adversarial)
        if not isinstance(self.initial_state, RestaurantState):
            raise ContractValidationError("initial_state must be RestaurantState")
        if self.initial_state.scenario_id != self.scenario_id:
            raise ContractValidationError("scenario and initial state IDs must match")
        supplier_ids = tuple(supplier.supplier_id for supplier in suppliers)
        if not supplier_ids or len(supplier_ids) != len(set(supplier_ids)):
            raise ContractValidationError("scenario suppliers must be non-empty and unique")
        unknown_invoice_suppliers = {
            invoice.supplier_id for invoice in self.initial_state.invoices
        } - set(supplier_ids)
        if unknown_invoice_suppliers:
            raise ContractValidationError(
                "every active invoice must belong to a known scenario supplier"
            )
        invoices_by_identity = {
            invoice.identity: invoice for invoice in self.initial_state.invoices
        }
        receipt_ids = tuple(proof.receipt_id for proof in payment_proofs)
        if len(receipt_ids) != len(set(receipt_ids)):
            raise ContractValidationError("payment proof receipt IDs must be unique")
        for proof in payment_proofs:
            invoice = invoices_by_identity.get(proof.identity)
            if invoice is None:
                raise ContractValidationError(
                    "every payment proof must reference a known composite invoice identity"
                )
            if (
                proof.amount_minor != invoice.amount_minor
                or proof.currency != invoice.currency
            ):
                raise ContractValidationError(
                    "fixture payment proof must match the full invoice amount and currency"
                )


LOCKED_SCENARIO_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "procureagent" / "scenario_v1.json"
)
LOCKED_SCENARIO_SHA256 = (
    "e13db4d2767967826147cf8b538e8688edd46072369f4a30e8823e14579c4792"
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ContractValidationError(f"non-finite JSON number is forbidden: {value}")


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{path} must be a JSON object")
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractValidationError(f"{path} must be a JSON array")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    path: str,
    *,
    optional: frozenset[str] = frozenset(),
) -> None:
    """Reject missing and unexpected keys.

    ``optional`` names keys that may be present or absent. It exists so a newer
    contract field can be added without invalidating an already hash-pinned
    fixture that predates it; every other call site keeps exact-key behaviour.
    """

    missing = expected - set(value)
    extra = set(value) - expected - optional
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise ContractValidationError(f"{path} has " + " and ".join(details))


def _load_supplier(value: object, index: int) -> SupplierIdentity:
    path = f"suppliers[{index}]"
    item = _mapping(value, path)
    _exact_keys(item, {"supplier_id", "display_name"}, path)
    return SupplierIdentity(item["supplier_id"], item["display_name"])


def _load_invoice(value: object, index: int) -> SyntheticSupplierInvoice:
    path = f"invoices[{index}]"
    item = _mapping(value, path)
    expected = {
        "record_source",
        "supplier_id",
        "invoice_number",
        "category",
        "amount_minor",
        "currency",
        "due_in_days",
        "inventory_days_remaining",
        "delivery_lead_days",
        "payment_unlocks_delivery",
        "delivery_inventory_days",
        "supplier_criticality",
        "supplier_status",
        "payment_status",
        "state_version",
        "late_fee_minor_per_day",
        "disruption_after_days_overdue",
        "context_conflict_codes",
    }
    _exact_keys(item, expected, path)
    return SyntheticSupplierInvoice(
        supplier_id=item["supplier_id"],
        invoice_number=item["invoice_number"],
        category=item["category"],
        amount_minor=item["amount_minor"],
        currency=item["currency"],
        due_in_days=item["due_in_days"],
        inventory_days_remaining=item["inventory_days_remaining"],
        delivery_lead_days=item["delivery_lead_days"],
        payment_unlocks_delivery=item["payment_unlocks_delivery"],
        delivery_inventory_days=item["delivery_inventory_days"],
        supplier_criticality=item["supplier_criticality"],
        supplier_status=item["supplier_status"],
        payment_status=item["payment_status"],
        state_version=item["state_version"],
        late_fee_minor_per_day=item["late_fee_minor_per_day"],
        disruption_after_days_overdue=item["disruption_after_days_overdue"],
        context_conflict_codes=tuple(
            _list(item["context_conflict_codes"], f"{path}.context_conflict_codes")
        ),
        record_source=item["record_source"],
    )


def _load_adversarial(value: object, index: int) -> AdversarialDocumentMetadata:
    path = f"adversarial_documents[{index}]"
    item = _mapping(value, path)
    expected = {
        "document_id",
        "supplier_label",
        "candidate_invoice_numbers",
        "expected_status",
        "reason_codes",
        "expected_lookup_activated",
        "included_in_obligations",
    }
    _exact_keys(item, expected, path)
    return AdversarialDocumentMetadata(
        document_id=item["document_id"],
        supplier_label=item["supplier_label"],
        candidate_invoice_numbers=tuple(
            _list(
                item["candidate_invoice_numbers"],
                f"{path}.candidate_invoice_numbers",
            )
        ),
        expected_status=item["expected_status"],
        reason_codes=tuple(_list(item["reason_codes"], f"{path}.reason_codes")),
        expected_lookup_activated=item["expected_lookup_activated"],
        included_in_obligations=item["included_in_obligations"],
    )


def _load_payment_proof(value: object, index: int) -> PaymentProof:
    path = f"payment_proofs[{index}]"
    item = _mapping(value, path)
    expected = {
        "receipt_id",
        "supplier_id",
        "invoice_number",
        "amount_minor",
        "currency",
        "paid_date",
        "source",
        "provenance",
        "status",
    }
    _exact_keys(item, expected, path)
    return PaymentProof(
        receipt_id=item["receipt_id"],
        supplier_id=item["supplier_id"],
        invoice_number=item["invoice_number"],
        amount_minor=item["amount_minor"],
        currency=item["currency"],
        paid_date=item["paid_date"],
        source=item["source"],
        provenance=item["provenance"],
        status=item["status"],
    )


def load_scenario(path: str | Path) -> ProcureScenario:
    """Load one strictly validated ProcureAgent scenario JSON document."""

    fixture_path = Path(path)
    try:
        with fixture_path.open("r", encoding="utf-8") as handle:
            raw = json.load(
                handle,
                parse_float=Decimal,
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
    except ContractValidationError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractValidationError(
            f"unable to load scenario fixture {fixture_path}: {exc}"
        ) from exc

    root = _mapping(raw, "scenario")
    expected_root = {
        "contract_version",
        "scenario_id",
        "seed",
        "horizon_days",
        "restaurant",
        "suppliers",
        "invoices",
        "payment_proofs",
        "adversarial_documents",
    }
    _exact_keys(
        root, expected_root, "scenario", optional=frozenset({"daily_cash_inflow_minor"})
    )
    restaurant = _mapping(root["restaurant"], "restaurant")
    _exact_keys(
        restaurant,
        {"restaurant_id", "day", "state_version", "cash_minor", "currency"},
        "restaurant",
    )
    suppliers = tuple(
        _load_supplier(value, index)
        for index, value in enumerate(_list(root["suppliers"], "suppliers"))
    )
    invoices = tuple(
        _load_invoice(value, index)
        for index, value in enumerate(_list(root["invoices"], "invoices"))
    )
    adversarial = tuple(
        _load_adversarial(value, index)
        for index, value in enumerate(
            _list(root["adversarial_documents"], "adversarial_documents")
        )
    )
    payment_proofs = tuple(
        _load_payment_proof(value, index)
        for index, value in enumerate(
            _list(root["payment_proofs"], "payment_proofs")
        )
    )
    state = RestaurantState(
        restaurant_id=restaurant["restaurant_id"],
        scenario_id=root["scenario_id"],
        day=restaurant["day"],
        state_version=restaurant["state_version"],
        cash_minor=restaurant["cash_minor"],
        currency=restaurant["currency"],
        invoices=invoices,
    )
    return ProcureScenario(
        contract_version=root["contract_version"],
        scenario_id=root["scenario_id"],
        seed=root["seed"],
        horizon_days=root["horizon_days"],
        suppliers=suppliers,
        initial_state=state,
        payment_proofs=payment_proofs,
        adversarial_documents=adversarial,
        daily_cash_inflow_minor=root.get("daily_cash_inflow_minor", 0),
    )


def load_locked_scenario(path: str | Path | None = None) -> ProcureScenario:
    """Load and verify the immutable four-invoice ``restaurant_demo_v1`` fixture."""

    fixture_path = Path(path or LOCKED_SCENARIO_PATH)
    try:
        fixture_digest = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ContractValidationError(
            f"unable to hash locked scenario fixture {fixture_path}: {exc}"
        ) from exc
    if fixture_digest != LOCKED_SCENARIO_SHA256:
        raise ContractValidationError(
            "fixture bytes do not match the locked restaurant_demo_v1 SHA-256"
        )

    scenario = load_scenario(fixture_path)
    expected_amounts = {
        InvoiceIdentity("fresh_farms", "FF-10482"): 150_000,
        InvoiceIdentity("prime_foods", "PF-25031"): 250_000,
        InvoiceIdentity("packright", "PR-15007"): 150_000,
        InvoiceIdentity("cleanpro", "CP-70019"): 70_000,
    }
    actual_amounts = {
        invoice.identity: invoice.amount_minor
        for invoice in scenario.initial_state.active_invoices
    }
    locked_checks = (
        scenario.contract_version == _CONTRACT_VERSION,
        scenario.scenario_id == "restaurant_demo_v1",
        scenario.seed == 138,
        scenario.horizon_days == 7,
        scenario.initial_state.restaurant_id
        == "sugar_and_spice_thai_restaurant",
        scenario.initial_state.day == 0,
        scenario.initial_state.state_version == 1,
        scenario.initial_state.currency == "USD",
        scenario.initial_state.cash_minor == 500_000,
        scenario.initial_state.total_obligations_minor == 620_000,
        actual_amounts == expected_amounts,
        len(scenario.adversarial_documents) == 1,
        scenario.adversarial_documents[0].supplier_label == "UnknownCo",
        len(scenario.payment_proofs) == 1,
    )
    if not all(locked_checks):
        raise ContractValidationError(
            "fixture does not match the locked restaurant_demo_v1 scenario"
        )
    return scenario


__all__ = [
    "AdversarialDocumentMetadata",
    "BoundingBox",
    "ContractValidationError",
    "DailyRecommendationBatch",
    "DocumentIdentityProposal",
    "DocumentMethod",
    "DocumentReviewDecision",
    "DocumentStatus",
    "InvoiceIdentity",
    "InvoiceNumberCandidate",
    "InvoicePaymentStatus",
    "LOCKED_SCENARIO_PATH",
    "LOCKED_SCENARIO_SHA256",
    "OperatorDecision",
    "OperatorDecisionType",
    "PaymentProof",
    "PaymentProofSource",
    "PaymentProofStatus",
    "PaymentReceipt",
    "PolicyType",
    "ProcureScenario",
    "ProcurementAction",
    "Recommendation",
    "RecordSource",
    "RestaurantState",
    "SupplierCriticality",
    "SupplierIdentity",
    "SupplierSource",
    "SupplierStatus",
    "SyntheticSupplierInvoice",
    "VerifiedInvoiceIdentity",
    "VerifierDecision",
    "VerifierResult",
    "load_locked_scenario",
    "load_scenario",
    "make_audit_id",
    "validate_full_payment_proof",
]
