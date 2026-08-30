"""Deterministic, framework-neutral labels for displaying OCR tokens.

This module turns the evidence already produced by the document and receipt
pipelines into one label per OCR word.  It deliberately contains no HTML or
other rendering logic: callers must escape untrusted OCR text at the UI
boundary.

Invoice-number labels retain whether the evidence came from the anchored rule,
Ryan's LayoutLMv3 adapter, or both.  Invoice amounts are a separate,
deterministic OCR rule and are never attributed to the model.  Ambiguous or
ungrounded fields remain ``OTHER``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import re
from typing import Iterable

from invoiceagent.core import normalize_identifier

from .contracts import BoundingBox, ContractValidationError
from .document import (
    AnchoredInvoiceCandidate,
    DocumentGateResult,
    InvoiceModelRun,
    InvoiceModelRunStatus,
    ModelInvoiceCandidate,
)
from .ocr import OcrResult, OcrStatus, OcrWord
from .receipt import ParsedReceipt, ReceiptFieldEvidence, parse_money_to_minor_units


class TokenLabel(str, Enum):
    """Business field assigned to one OCR word."""

    RECEIPT_ID = "RECEIPT_ID"
    INVOICE_NUMBER = "INVOICE_NUMBER"
    AMOUNT = "AMOUNT"
    CURRENCY = "CURRENCY"
    DATE = "DATE"
    SUPPLIER = "SUPPLIER"
    OTHER = "OTHER"


class TokenSource(str, Enum):
    """Auditable source of a token label."""

    OCR_ONLY = "OCR_ONLY"
    INVOICE_ANCHORED_RULE = "INVOICE_ANCHORED_RULE"
    RYAN_INVOICE_NUMBER_MODEL = "RYAN_INVOICE_NUMBER_MODEL"
    INVOICE_RULE_AND_RYAN_MODEL = "INVOICE_RULE_AND_RYAN_MODEL"
    INVOICE_AMOUNT_RULE = "INVOICE_AMOUNT_RULE"
    RECEIPT_FIELD_RULE = "RECEIPT_FIELD_RULE"


_ALLOWED_SOURCES = {
    TokenLabel.OTHER: frozenset({TokenSource.OCR_ONLY}),
    TokenLabel.INVOICE_NUMBER: frozenset(
        {
            TokenSource.INVOICE_ANCHORED_RULE,
            TokenSource.RYAN_INVOICE_NUMBER_MODEL,
            TokenSource.INVOICE_RULE_AND_RYAN_MODEL,
            TokenSource.RECEIPT_FIELD_RULE,
        }
    ),
    TokenLabel.AMOUNT: frozenset(
        {TokenSource.INVOICE_AMOUNT_RULE, TokenSource.RECEIPT_FIELD_RULE}
    ),
    TokenLabel.RECEIPT_ID: frozenset({TokenSource.RECEIPT_FIELD_RULE}),
    TokenLabel.CURRENCY: frozenset({TokenSource.RECEIPT_FIELD_RULE}),
    TokenLabel.DATE: frozenset({TokenSource.RECEIPT_FIELD_RULE}),
    TokenLabel.SUPPLIER: frozenset({TokenSource.RECEIPT_FIELD_RULE}),
}


@dataclass(frozen=True, slots=True)
class LabeledToken:
    """A UI-facing data contract for one word; ``text`` is untrusted OCR."""

    index: int
    text: str
    label: TokenLabel
    source: TokenSource
    ocr_confidence: Decimal
    normalized_box: BoundingBox

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ContractValidationError("labeled-token index must be non-negative")
        if not isinstance(self.text, str) or not self.text:
            raise ContractValidationError("labeled-token text must be non-empty")
        if not isinstance(self.label, TokenLabel):
            raise ContractValidationError("labeled-token label is invalid")
        if not isinstance(self.source, TokenSource):
            raise ContractValidationError("labeled-token source is invalid")
        if self.source not in _ALLOWED_SOURCES[self.label]:
            raise ContractValidationError(
                "labeled-token label/source combination is invalid"
            )
        if (
            not isinstance(self.ocr_confidence, Decimal)
            or not self.ocr_confidence.is_finite()
            or not Decimal("0") <= self.ocr_confidence <= Decimal("1")
        ):
            raise ContractValidationError(
                "labeled-token OCR confidence must be Decimal in 0..1"
            )
        if not isinstance(self.normalized_box, BoundingBox):
            raise ContractValidationError("labeled-token box must be normalized")


Assignment = tuple[TokenLabel, TokenSource]


def _assign(
    assignments: dict[int, Assignment | None],
    indices: Iterable[int],
    assignment: Assignment,
) -> None:
    """Add an assignment while failing closed on conflicting field labels."""

    for index in indices:
        existing = assignments.get(index)
        if existing is None and index in assignments:
            continue
        if existing is None or existing == assignment:
            assignments[index] = assignment
        else:
            assignments[index] = None


def _labeled_tokens(
    ocr: OcrResult, assignments: dict[int, Assignment | None]
) -> tuple[LabeledToken, ...]:
    result: list[LabeledToken] = []
    for word in ocr.words:
        assignment = assignments.get(word.sequence)
        label, source = assignment or (TokenLabel.OTHER, TokenSource.OCR_ONLY)
        result.append(
            LabeledToken(
                index=word.sequence,
                text=word.text,
                label=label,
                source=source,
                ocr_confidence=word.confidence,
                normalized_box=word.normalized_box,
            )
        )
    return tuple(result)


def _valid_indices(indices: Iterable[int], ocr: OcrResult) -> tuple[int, ...]:
    candidate = tuple(indices)
    if (
        not candidate
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(ocr.words)
            for index in candidate
        )
        or tuple(sorted(set(candidate))) != candidate
    ):
        return ()
    return candidate


def _rule_indices(
    candidate: AnchoredInvoiceCandidate, ocr: OcrResult
) -> tuple[int, ...]:
    indices = _valid_indices(candidate.word_indices, ocr)
    if not indices:
        return ()
    words = tuple(ocr.words[index] for index in indices)
    if (
        candidate.evidence_tokens != tuple(word.text for word in words)
        or candidate.evidence_boxes != tuple(word.normalized_box for word in words)
        or normalize_identifier(" ".join(candidate.evidence_tokens))
        != normalize_identifier(candidate.invoice_number)
    ):
        return ()
    return indices


def _model_indices(
    candidate: ModelInvoiceCandidate, ocr: OcrResult
) -> tuple[int, ...]:
    indices = _valid_indices(candidate.word_indices, ocr)
    if not indices:
        return ()
    words = tuple(ocr.words[index] for index in indices)
    evidence = candidate.candidate
    if (
        not evidence.grounded_in_ocr
        or evidence.evidence_tokens != tuple(word.text for word in words)
        or evidence.evidence_boxes != tuple(word.normalized_box for word in words)
        or normalize_identifier(" ".join(evidence.evidence_tokens))
        != normalize_identifier(evidence.invoice_number)
    ):
        return ()
    return indices


def _line_groups(words: tuple[OcrWord, ...]) -> tuple[tuple[OcrWord, ...], ...]:
    groups: list[list[OcrWord]] = []
    previous: tuple[int, int, int, int] | None = None
    for word in words:
        key = (word.page, word.block, word.paragraph, word.line)
        if key != previous:
            groups.append([])
            previous = key
        groups[-1].append(word)
    return tuple(tuple(group) for group in groups)


def _normalized_anchor_word(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


_TOTAL_ANCHORS = (
    ("invoice", "total"),
    ("grand", "total"),
    ("amount", "due"),
    ("balance", "due"),
    ("total", "due"),
    ("total",),
)


def _anchor_ends(line: tuple[OcrWord, ...]) -> tuple[int, ...]:
    normalized = tuple(_normalized_anchor_word(word.text) for word in line)
    ends: list[int] = []
    for start in range(len(line)):
        matches = [
            anchor
            for anchor in _TOTAL_ANCHORS
            if normalized[start : start + len(anchor)] == anchor
        ]
        if matches:
            longest = max(matches, key=len)
            ends.append(start + len(longest))
    return tuple(ends)


def _money_indices(ocr: OcrResult) -> tuple[int, ...]:
    """Resolve one positive total value from label-anchored OCR lines."""

    candidates: dict[int, set[tuple[int, ...]]] = {}
    for line in _line_groups(ocr.words):
        for anchor_end in _anchor_ends(line):
            after_anchor = line[anchor_end:]
            spans_for_value: dict[int, list[tuple[int, ...]]] = {}
            for start in range(len(after_anchor)):
                for end in range(start + 1, min(len(after_anchor), start + 3) + 1):
                    span = after_anchor[start:end]
                    try:
                        minor = parse_money_to_minor_units(
                            " ".join(word.text for word in span)
                        )
                    except ContractValidationError:
                        continue
                    spans_for_value.setdefault(minor, []).append(
                        tuple(word.sequence for word in span)
                    )
            for minor, spans in spans_for_value.items():
                shortest = min(len(span) for span in spans)
                for span in spans:
                    if len(span) == shortest:
                        candidates.setdefault(minor, set()).add(span)
    if len(candidates) != 1:
        return ()
    spans = next(iter(candidates.values()))
    if len(spans) != 1:
        return ()
    return next(iter(spans))


def label_invoice_tokens(
    ocr: OcrResult,
    *,
    gate: DocumentGateResult | None = None,
    rule_candidates: Iterable[AnchoredInvoiceCandidate] | None = None,
    model_run: InvoiceModelRun | None = None,
) -> tuple[LabeledToken, ...]:
    """Label every invoice OCR word from grounded, unambiguous evidence.

    Pass ``gate=analysis.gate`` for the normal UI flow.  The explicit
    ``rule_candidates`` and ``model_run`` parameters support callers that have
    not constructed a gate yet.  A gate cannot be mixed with those parameters.
    A low-confidence but grounded candidate remains visible as model evidence;
    the document gate continues to own the accept/review decision.
    """

    if not isinstance(ocr, OcrResult):
        raise ContractValidationError("ocr must be OcrResult")
    if gate is not None and (rule_candidates is not None or model_run is not None):
        raise ContractValidationError(
            "gate cannot be combined with explicit invoice-label context"
        )
    if gate is not None and not isinstance(gate, DocumentGateResult):
        raise ContractValidationError("gate must be DocumentGateResult")
    if gate is not None:
        rules = gate.rule_candidates
        run = gate.model_run
        context_matches = gate.document_id == ocr.document_id
    else:
        rules = tuple(rule_candidates or ())
        run = model_run
        context_matches = True
    if not all(isinstance(candidate, AnchoredInvoiceCandidate) for candidate in rules):
        raise ContractValidationError(
            "rule_candidates must contain AnchoredInvoiceCandidate values"
        )
    if run is not None and not isinstance(run, InvoiceModelRun):
        raise ContractValidationError("model_run must be InvoiceModelRun")

    assignments: dict[int, Assignment | None] = {}
    if ocr.status is not OcrStatus.SUCCESS:
        return _labeled_tokens(ocr, assignments)

    # Amount extraction is deliberately independent of the invoice-number
    # model and remains useful even when identity evidence needs review.
    _assign(
        assignments,
        _money_indices(ocr),
        (TokenLabel.AMOUNT, TokenSource.INVOICE_AMOUNT_RULE),
    )
    if not context_matches:
        return _labeled_tokens(ocr, assignments)

    # More than one candidate from either source is an ambiguity, not a
    # selection.  No invoice-number token is then highlighted.
    model_candidates = (
        run.candidates
        if run is not None
        and run.document_id == ocr.document_id
        and run.status is InvoiceModelRunStatus.SUCCESS
        else ()
    )
    if len(rules) > 1 or len(model_candidates) > 1:
        return _labeled_tokens(ocr, assignments)

    rule = rules[0] if len(rules) == 1 else None
    model = model_candidates[0] if len(model_candidates) == 1 else None
    rule_indices = _rule_indices(rule, ocr) if rule is not None else ()
    model_indices = _model_indices(model, ocr) if model is not None else ()

    if rule_indices and model_indices:
        if normalize_identifier(rule.invoice_number) != normalize_identifier(
            model.candidate.invoice_number
        ):
            return _labeled_tokens(ocr, assignments)
        for index in sorted(set(rule_indices) | set(model_indices)):
            if index in rule_indices and index in model_indices:
                source = TokenSource.INVOICE_RULE_AND_RYAN_MODEL
            elif index in rule_indices:
                source = TokenSource.INVOICE_ANCHORED_RULE
            else:
                source = TokenSource.RYAN_INVOICE_NUMBER_MODEL
            _assign(assignments, (index,), (TokenLabel.INVOICE_NUMBER, source))
    elif rule_indices:
        _assign(
            assignments,
            rule_indices,
            (TokenLabel.INVOICE_NUMBER, TokenSource.INVOICE_ANCHORED_RULE),
        )
    elif model_indices:
        _assign(
            assignments,
            model_indices,
            (TokenLabel.INVOICE_NUMBER, TokenSource.RYAN_INVOICE_NUMBER_MODEL),
        )
    return _labeled_tokens(ocr, assignments)


_RECEIPT_LABELS = {
    "receipt_id": TokenLabel.RECEIPT_ID,
    "supplier": TokenLabel.SUPPLIER,
    "invoice_number": TokenLabel.INVOICE_NUMBER,
    "amount_minor": TokenLabel.AMOUNT,
    "currency": TokenLabel.CURRENCY,
    "paid_date": TokenLabel.DATE,
}


def _receipt_value(parsed: ParsedReceipt, field_name: str) -> object | None:
    if field_name == "supplier":
        return parsed.supplier_name
    return getattr(parsed, field_name)


def _grounded_receipt_evidence(
    evidence: ReceiptFieldEvidence, ocr: OcrResult
) -> tuple[int, ...]:
    indices = _valid_indices(evidence.word_indices, ocr)
    if not indices:
        return ()
    words = tuple(ocr.words[index] for index in indices)
    if evidence.evidence_tokens != tuple(word.text for word in words):
        return ()
    if indices != tuple(range(indices[0], indices[-1] + 1)):
        return ()
    line_keys = {
        (word.page, word.block, word.paragraph, word.line) for word in words
    }
    if len(line_keys) != 1:
        return ()
    return indices


def _joined_tokens(words: tuple[OcrWord, ...]) -> str:
    value = " ".join(word.text for word in words).strip()
    return re.sub(r"\s*([./#:$€£-])\s*", r"\1", value)


def _matching_receipt_spans(
    field_name: str,
    canonical: object,
    evidence_indices: tuple[int, ...],
    ocr: OcrResult,
) -> tuple[tuple[int, ...], ...]:
    evidence_words = tuple(ocr.words[index] for index in evidence_indices)
    matches: set[tuple[int, ...]] = set()
    for start in range(len(evidence_words)):
        for end in range(start + 1, len(evidence_words) + 1):
            span = evidence_words[start:end]
            joined = _joined_tokens(span)
            matched = False
            if field_name == "amount_minor":
                try:
                    matched = parse_money_to_minor_units(joined) == canonical
                except ContractValidationError:
                    matched = False
            elif field_name in {"receipt_id", "invoice_number", "supplier"}:
                matched = normalize_identifier(joined) == normalize_identifier(
                    str(canonical)
                )
            elif field_name == "paid_date":
                matched = joined.strip(".,;:") == canonical.isoformat()
            elif field_name == "currency":
                matched = joined.strip(".,;:").upper() == canonical
            if matched:
                matches.add(tuple(word.sequence for word in span))
    # Ignore a wider match only when it strictly contains a more precise
    # matching span (for example ``USD $10.00`` around ``$10.00``).  Distinct
    # occurrences remain ambiguous even if one happens to use fewer tokens.
    minimal = {
        span
        for span in matches
        if not any(
            other != span
            and len(other) < len(span)
            and set(other).issubset(span)
            for other in matches
        )
    }
    return tuple(sorted(minimal))


def label_receipt_tokens(
    ocr: OcrResult, parsed: ParsedReceipt
) -> tuple[LabeledToken, ...]:
    """Label every receipt OCR word using only grounded parsed evidence."""

    if not isinstance(ocr, OcrResult):
        raise ContractValidationError("ocr must be OcrResult")
    if not isinstance(parsed, ParsedReceipt):
        raise ContractValidationError("parsed must be ParsedReceipt")
    assignments: dict[int, Assignment | None] = {}
    if (
        ocr.status is not OcrStatus.SUCCESS
        or parsed.document_id != ocr.document_id
    ):
        return _labeled_tokens(ocr, assignments)

    by_field: dict[str, list[ReceiptFieldEvidence]] = {
        field_name: [] for field_name in _RECEIPT_LABELS
    }
    for evidence in parsed.evidence:
        if evidence.field_name in by_field:
            by_field[evidence.field_name].append(evidence)
    for field_name, evidence_items in by_field.items():
        canonical = _receipt_value(parsed, field_name)
        # Missing values or duplicate evidence are unresolved fields.
        if canonical is None or len(evidence_items) != 1:
            continue
        evidence = evidence_items[0]
        expected_evidence_value = (
            canonical.isoformat() if field_name == "paid_date" else str(canonical)
        )
        if evidence.value != expected_evidence_value:
            continue
        indices = _grounded_receipt_evidence(evidence, ocr)
        if not indices:
            continue
        spans = _matching_receipt_spans(field_name, canonical, indices, ocr)
        # Multiple distinct value occurrences are ambiguous; none is selected.
        if len(spans) != 1:
            continue
        _assign(
            assignments,
            spans[0],
            (_RECEIPT_LABELS[field_name], TokenSource.RECEIPT_FIELD_RULE),
        )
    return _labeled_tokens(ocr, assignments)


def summarize_token_labels(
    tokens: Iterable[LabeledToken],
) -> dict[TokenLabel, int]:
    """Count labels in stable enum order, including zero-count labels."""

    items = tuple(tokens)
    if not all(isinstance(token, LabeledToken) for token in items):
        raise ContractValidationError("tokens must contain LabeledToken values")
    counts = {label: 0 for label in TokenLabel}
    for token in items:
        counts[token.label] += 1
    return counts


__all__ = [
    "LabeledToken",
    "TokenLabel",
    "TokenSource",
    "label_invoice_tokens",
    "label_receipt_tokens",
    "summarize_token_labels",
]
