"""Invoice-number rules, Ryan-model adaptation, and the document safety gate."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from io import BytesIO
import re
from time import perf_counter
from typing import Any, Callable, Iterable

from invoiceagent.extraction import (
    LayoutLMv3InvoiceExtractor,
    OcrDocument as RyanOcrDocument,
    is_valid_invoice_identifier,
)
from invoiceagent.core import normalize_identifier

from .contracts import (
    BoundingBox,
    ContractValidationError,
    DocumentIdentityProposal,
    DocumentMethod,
    DocumentStatus,
    InvoiceIdentity,
    InvoiceNumberCandidate,
    SupplierSource,
    VerifiedInvoiceIdentity,
)
from .ocr import IngestedImage, OcrResult, OcrStatus


_LABELS = {"invoice", "inv", "bill"}
_BOUNDARIES = {
    "amount",
    "currency",
    "date",
    "due",
    "subtotal",
    "supplier",
    "tax",
    "total",
}
_IDENTIFIER_TOKEN = re.compile(r"[A-Za-z0-9._/#:-]+\Z")


def _normalized_word(value: str) -> str:
    return value.casefold().strip().rstrip(".:")


def _is_qualifier(value: str) -> bool:
    raw = value.casefold().strip()
    return raw in {"#", ":"} or raw.rstrip(".:") in {"id", "no", "number"}


def _join_identifier_tokens(tokens: Iterable[str]) -> str:
    value = " ".join(tokens).strip()
    return re.sub(r"\s*([./#:-])\s*", r"\1", value).rstrip(".,;:")


@dataclass(frozen=True, slots=True)
class AnchoredInvoiceCandidate:
    invoice_number: str
    word_indices: tuple[int, ...]
    evidence_tokens: tuple[str, ...]
    evidence_boxes: tuple[BoundingBox, ...]

    def __post_init__(self) -> None:
        if not is_valid_invoice_identifier(self.invoice_number):
            raise ContractValidationError("anchored invoice candidate is invalid")
        indices = tuple(self.word_indices)
        tokens = tuple(self.evidence_tokens)
        boxes = tuple(self.evidence_boxes)
        if (
            not indices
            or len(indices) != len(tokens)
            or len(tokens) != len(boxes)
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            )
            or any(not isinstance(token, str) or not token for token in tokens)
            or any(not isinstance(box, BoundingBox) for box in boxes)
        ):
            raise ContractValidationError("anchored candidate evidence is invalid")
        object.__setattr__(self, "word_indices", indices)
        object.__setattr__(self, "evidence_tokens", tokens)
        object.__setattr__(self, "evidence_boxes", boxes)

    @property
    def grounded_in_ocr(self) -> bool:
        return bool(self.word_indices and self.evidence_tokens and self.evidence_boxes)


def anchored_invoice_candidates(ocr: OcrResult) -> tuple[AnchoredInvoiceCandidate, ...]:
    """Return every distinct label-anchored invoice identifier in OCR order."""

    if not isinstance(ocr, OcrResult):
        raise ContractValidationError("ocr must be OcrResult")
    if ocr.status is not OcrStatus.SUCCESS:
        return ()
    words = ocr.words
    found: list[AnchoredInvoiceCandidate] = []
    seen: set[str] = set()
    for anchor_index, anchor in enumerate(words):
        if _normalized_word(anchor.text) not in _LABELS:
            continue
        cursor = anchor_index + 1
        while cursor < len(words) and _is_qualifier(words[cursor].text):
            cursor += 1
        pieces: list[str] = []
        best_end: int | None = None
        best_value: str | None = None
        for candidate_index in range(cursor, min(len(words), cursor + 5)):
            token = words[candidate_index].text.strip()
            normalized_token = _normalized_word(token)
            if (
                normalized_token in _BOUNDARIES
                or (pieces and normalized_token in _LABELS)
                or not _IDENTIFIER_TOKEN.fullmatch(token)
            ):
                break
            pieces.append(token)
            candidate_value = _join_identifier_tokens(pieces)
            if is_valid_invoice_identifier(candidate_value):
                best_value = candidate_value
                best_end = candidate_index + 1
        if best_value is None or best_end is None:
            continue
        normalized = normalize_identifier(best_value)
        if normalized in seen:
            continue
        seen.add(normalized)
        indices = tuple(range(cursor, best_end))
        found.append(
            AnchoredInvoiceCandidate(
                invoice_number=best_value,
                word_indices=indices,
                evidence_tokens=tuple(words[index].text for index in indices),
                evidence_boxes=tuple(
                    words[index].normalized_box for index in indices
                ),
            )
        )
    return tuple(found)


class InvoiceModelRunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NO_CANDIDATE = "NO_CANDIDATE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ModelInvoiceCandidate:
    candidate: InvoiceNumberCandidate
    word_indices: tuple[int, ...]
    minimum_confidence: Decimal
    mean_confidence: Decimal
    mean_margin: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, InvoiceNumberCandidate):
            raise ContractValidationError("candidate must be InvoiceNumberCandidate")
        if not self.word_indices or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.word_indices
        ):
            raise ContractValidationError("model word indices must be non-empty integers")
        for name in ("minimum_confidence", "mean_confidence", "mean_margin"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or not 0 <= value <= 1:
                raise ContractValidationError(f"{name} must be Decimal in 0..1")


@dataclass(frozen=True, slots=True)
class InvoiceModelRun:
    document_id: str
    status: InvoiceModelRunStatus
    candidates: tuple[ModelInvoiceCandidate, ...]
    model_version: str
    latency_ms: Decimal
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.startswith("doc_"):
            raise ContractValidationError("model run needs a content-derived document ID")
        if not isinstance(self.status, InvoiceModelRunStatus):
            raise ContractValidationError("model run status is invalid")
        candidates = tuple(self.candidates)
        if not all(isinstance(item, ModelInvoiceCandidate) for item in candidates):
            raise ContractValidationError("model candidates are invalid")
        object.__setattr__(self, "candidates", candidates)
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ContractValidationError("model_version is required")
        if not isinstance(self.latency_ms, Decimal) or self.latency_ms < 0:
            raise ContractValidationError("model latency must be non-negative Decimal")
        if self.status is InvoiceModelRunStatus.SUCCESS and not candidates:
            raise ContractValidationError("successful model run needs candidates")
        if self.status is InvoiceModelRunStatus.NO_CANDIDATE and candidates:
            raise ContractValidationError("NO_CANDIDATE run cannot contain candidates")
        if self.status is InvoiceModelRunStatus.FAILED:
            if candidates or not self.error_code:
                raise ContractValidationError("failed model run needs an error and no candidates")
        elif self.error_code is not None or self.error_message is not None:
            raise ContractValidationError("non-failed model run cannot contain an error")

    def to_proposal(
        self,
        *,
        supplier_id: str,
        supplier_confirmed: bool,
        status: DocumentStatus = DocumentStatus.PROPOSED,
    ) -> DocumentIdentityProposal:
        return DocumentIdentityProposal(
            document_id=self.document_id,
            supplier_id=supplier_id,
            supplier_source=SupplierSource.OPERATOR_SELECTED,
            supplier_confirmed=supplier_confirmed,
            candidate_spans=tuple(item.candidate for item in self.candidates),
            method=DocumentMethod.LAYOUTLMV3_LOCAL,
            model_version=self.model_version,
            status=status,
        )


def to_ryan_ocr(ocr: OcrResult) -> RyanOcrDocument:
    """Translate successful ProcureAgent OCR into Ryan's existing adapter input."""

    if not isinstance(ocr, OcrResult) or ocr.status is not OcrStatus.SUCCESS:
        raise ContractValidationError("Ryan adapter requires successful OCR")
    return RyanOcrDocument(
        words=ocr.ordered_text,
        boxes=tuple(
            (
                word.normalized_box.x0,
                word.normalized_box.y0,
                word.normalized_box.x1,
                word.normalized_box.y1,
            )
            for word in ocr.words
        ),
        quality=ocr.quality,
        raw_text=ocr.raw_text,
        engine=f"{ocr.engine}:{ocr.engine_version}",
    )


def _default_image_decoder(image: IngestedImage) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required only for an actual LayoutLMv3 run; install model extras"
        ) from exc
    with Image.open(BytesIO(image.image_bytes)) as opened:
        opened.load()
        return opened.convert("RGB")


ExtractorFactory = Callable[[], Any]
ImageDecoder = Callable[[IngestedImage], Any]


class RyanInvoiceAdapter:
    """Lazy, offline-at-import adapter around ``LayoutLMv3InvoiceExtractor``."""

    def __init__(
        self,
        *,
        extractor: Any | None = None,
        extractor_factory: ExtractorFactory = LayoutLMv3InvoiceExtractor,
        image_decoder: ImageDecoder = _default_image_decoder,
    ) -> None:
        self._extractor = extractor
        self._extractor_factory = extractor_factory
        self._image_decoder = image_decoder
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _model_version(self) -> str:
        if self._extractor is None:
            return "ryanznie/layoutlmv3-lora-invoice-number:not-loaded"
        adapter = getattr(self._extractor, "adapter_model", None)
        revision = getattr(self._extractor, "adapter_revision", None)
        device = getattr(self._extractor, "device", None)
        if adapter:
            pinned = f"{adapter}@{revision}" if revision else adapter
            return f"{pinned} on {device or 'unresolved'}"
        return self._extractor.__class__.__name__

    @staticmethod
    def _latency(started: float) -> Decimal:
        return Decimal(str((perf_counter() - started) * 1000)).quantize(Decimal("0.1"))

    def run(self, image: IngestedImage, ocr: OcrResult) -> InvoiceModelRun:
        if not isinstance(image, IngestedImage):
            raise ContractValidationError("image must be IngestedImage")
        if not isinstance(ocr, OcrResult) or ocr.document_id != image.document_id:
            raise ContractValidationError("OCR must belong to the ingested image")
        if ocr.status is not OcrStatus.SUCCESS:
            return InvoiceModelRun(
                document_id=image.document_id,
                status=InvoiceModelRunStatus.FAILED,
                candidates=(),
                model_version=self._model_version(),
                latency_ms=Decimal("0"),
                error_code="OCR_NOT_SUCCESSFUL",
                error_message=f"OCR status is {ocr.status.value}",
            )
        started = perf_counter()
        try:
            if self._extractor is None:
                self._extractor = self._extractor_factory()
            if not self._loaded:
                self._extractor.load()
                self._loaded = True
            decoded_image = self._image_decoder(image)
            ryan_ocr = to_ryan_ocr(ocr)
            result = self._extractor.predict(decoded_image, ryan_ocr)
            mapped: list[ModelInvoiceCandidate] = []
            for span in result.spans:
                indices = tuple(span.word_indices)
                indices_valid = bool(indices) and all(
                    0 <= index < len(ocr.words) for index in indices
                )
                evidence_words = (
                    tuple(ocr.words[index] for index in indices) if indices_valid else ()
                )
                evidence_value = _join_identifier_tokens(
                    word.text for word in evidence_words
                )
                boxes_match = indices_valid and tuple(span.boxes) == tuple(
                    (
                        word.normalized_box.x0,
                        word.normalized_box.y0,
                        word.normalized_box.x1,
                        word.normalized_box.y1,
                    )
                    for word in evidence_words
                )
                grounded = (
                    indices_valid
                    and boxes_match
                    and normalize_identifier(evidence_value)
                    == normalize_identifier(span.value)
                )
                candidate = InvoiceNumberCandidate(
                    invoice_number=span.value,
                    entity_confidence=span.minimum_confidence,
                    grounded_in_ocr=grounded,
                    evidence_tokens=tuple(word.text for word in evidence_words),
                    evidence_boxes=tuple(
                        word.normalized_box for word in evidence_words
                    ),
                )
                mapped.append(
                    ModelInvoiceCandidate(
                        candidate=candidate,
                        word_indices=indices,
                        minimum_confidence=span.minimum_confidence,
                        mean_confidence=span.mean_confidence,
                        mean_margin=span.mean_margin,
                    )
                )
        except Exception as exc:
            return InvoiceModelRun(
                document_id=image.document_id,
                status=InvoiceModelRunStatus.FAILED,
                candidates=(),
                model_version=self._model_version(),
                latency_ms=self._latency(started),
                error_code="MODEL_RUN_FAILED",
                error_message=f"{exc.__class__.__name__}: {exc}"[:512],
            )
        model_version = getattr(result, "model_name", None) or self._model_version()
        status = (
            InvoiceModelRunStatus.SUCCESS
            if mapped
            else InvoiceModelRunStatus.NO_CANDIDATE
        )
        return InvoiceModelRun(
            document_id=image.document_id,
            status=status,
            candidates=tuple(mapped),
            model_version=model_version,
            latency_ms=result.latency_ms,
        )


@dataclass(frozen=True, slots=True)
class DocumentGateResult:
    document_id: str
    status: DocumentStatus
    verified_identity: VerifiedInvoiceIdentity | None
    reason_codes: tuple[str, ...]
    rule_candidates: tuple[AnchoredInvoiceCandidate, ...]
    model_run: InvoiceModelRun | None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.startswith("doc_"):
            raise ContractValidationError("document gate needs a content-derived ID")
        if self.status is DocumentStatus.CONFIRMED:
            if self.verified_identity is None or self.reason_codes:
                raise ContractValidationError("confirmed document gate result is inconsistent")
        elif self.status is DocumentStatus.REVIEW_REQUIRED:
            if self.verified_identity is not None or not self.reason_codes:
                raise ContractValidationError("review document gate result is inconsistent")
        else:
            raise ContractValidationError("document gate returns CONFIRMED or REVIEW_REQUIRED")

    @property
    def may_activate_lookup(self) -> bool:
        return self.verified_identity is not None and self.status in {
            DocumentStatus.CONFIRMED,
            DocumentStatus.CORRECTED,
        }


def gate_document_identity(
    *,
    document_id: str,
    supplier_id: str,
    supplier_confirmed: bool,
    known_supplier_ids: Iterable[str],
    ocr: OcrResult,
    rule_candidates: Iterable[AnchoredInvoiceCandidate],
    model_run: InvoiceModelRun | None,
    active_identities: Iterable[InvoiceIdentity] = (),
    minimum_entity_confidence: Decimal = Decimal("0.80"),
    minimum_entity_margin: Decimal = Decimal("0.10"),
) -> DocumentGateResult:
    """Confirm only one grounded rule/model agreement for a known supplier."""

    if not isinstance(minimum_entity_confidence, Decimal) or not (
        Decimal("0") <= minimum_entity_confidence <= Decimal("1")
    ):
        raise ContractValidationError("minimum_entity_confidence must be Decimal in 0..1")
    if not isinstance(minimum_entity_margin, Decimal) or not (
        Decimal("0") <= minimum_entity_margin <= Decimal("1")
    ):
        raise ContractValidationError("minimum_entity_margin must be Decimal in 0..1")
    rules = tuple(rule_candidates)
    if not all(isinstance(item, AnchoredInvoiceCandidate) for item in rules):
        raise ContractValidationError("rule_candidates are invalid")
    known = frozenset(known_supplier_ids)
    active = frozenset(active_identities)
    reasons: list[str] = []
    if not isinstance(ocr, OcrResult) or ocr.document_id != document_id:
        reasons.append("OCR_DOCUMENT_MISMATCH")
    elif ocr.status is not OcrStatus.SUCCESS:
        reasons.append(f"OCR_{ocr.status.value}")
    if not isinstance(supplier_confirmed, bool) or not supplier_confirmed:
        reasons.append("SUPPLIER_NOT_CONFIRMED")
    if supplier_id not in known:
        reasons.append("UNKNOWN_SUPPLIER")
    if not rules:
        reasons.append("RULE_CANDIDATE_MISSING")
    elif len(rules) > 1:
        reasons.append("AMBIGUOUS_RULE_CANDIDATES")
    elif not _rule_grounded(rules[0], ocr):
        reasons.append("UNGROUNDED_RULE_CANDIDATE")
    if model_run is None:
        reasons.append("MODEL_RESULT_MISSING")
        model_candidates: tuple[ModelInvoiceCandidate, ...] = ()
    elif model_run.document_id != document_id:
        reasons.append("MODEL_DOCUMENT_MISMATCH")
        model_candidates = ()
    elif model_run.status is InvoiceModelRunStatus.FAILED:
        reasons.append("MODEL_PROCESSING_FAILED")
        model_candidates = ()
    elif model_run.status is InvoiceModelRunStatus.NO_CANDIDATE:
        reasons.append("MODEL_CANDIDATE_MISSING")
        model_candidates = ()
    else:
        model_candidates = model_run.candidates
        if len(model_candidates) > 1:
            reasons.append("AMBIGUOUS_MODEL_CANDIDATES")
        elif len(model_candidates) == 1:
            model_candidate = model_candidates[0]
            if not _model_grounded(model_candidate, ocr):
                reasons.append("UNGROUNDED_MODEL_CANDIDATE")
            if model_candidate.minimum_confidence < minimum_entity_confidence:
                reasons.append("LOW_MODEL_CONFIDENCE")
            if model_candidate.mean_margin < minimum_entity_margin:
                reasons.append("LOW_MODEL_MARGIN")
    if len(rules) == 1 and len(model_candidates) == 1:
        if normalize_identifier(rules[0].invoice_number) != normalize_identifier(
            model_candidates[0].candidate.invoice_number
        ):
            reasons.append("RULE_MODEL_DISAGREEMENT")
        identity = InvoiceIdentity(
            supplier_id, model_candidates[0].candidate.invoice_number
        )
        if identity in active:
            reasons.append("UNEXPECTED_DUPLICATE_INVOICE")
    else:
        identity = None
    reasons = list(dict.fromkeys(reasons))
    if reasons or identity is None:
        return DocumentGateResult(
            document_id=document_id,
            status=DocumentStatus.REVIEW_REQUIRED,
            verified_identity=None,
            reason_codes=tuple(reasons or ("IDENTITY_NOT_CONFIRMED",)),
            rule_candidates=rules,
            model_run=model_run,
        )
    verified = VerifiedInvoiceIdentity(
        document_id=document_id,
        supplier_id=identity.supplier_id,
        invoice_number=identity.invoice_number,
        status=DocumentStatus.CONFIRMED,
    )
    return DocumentGateResult(
        document_id=document_id,
        status=DocumentStatus.CONFIRMED,
        verified_identity=verified,
        reason_codes=(),
        rule_candidates=rules,
        model_run=model_run,
    )


def _rule_grounded(candidate: AnchoredInvoiceCandidate, ocr: OcrResult) -> bool:
    indices = candidate.word_indices
    if not indices or any(index >= len(ocr.words) for index in indices):
        return False
    words = tuple(ocr.words[index] for index in indices)
    return (
        candidate.evidence_tokens == tuple(word.text for word in words)
        and candidate.evidence_boxes == tuple(word.normalized_box for word in words)
        and normalize_identifier(_join_identifier_tokens(candidate.evidence_tokens))
        == normalize_identifier(candidate.invoice_number)
    )


def _model_grounded(candidate: ModelInvoiceCandidate, ocr: OcrResult) -> bool:
    indices = candidate.word_indices
    if not indices or any(index >= len(ocr.words) for index in indices):
        return False
    words = tuple(ocr.words[index] for index in indices)
    evidence = candidate.candidate
    return (
        evidence.grounded_in_ocr
        and evidence.evidence_tokens == tuple(word.text for word in words)
        and evidence.evidence_boxes == tuple(word.normalized_box for word in words)
        and normalize_identifier(_join_identifier_tokens(evidence.evidence_tokens))
        == normalize_identifier(evidence.invoice_number)
    )


__all__ = [
    "AnchoredInvoiceCandidate",
    "DocumentGateResult",
    "InvoiceModelRun",
    "InvoiceModelRunStatus",
    "ModelInvoiceCandidate",
    "RyanInvoiceAdapter",
    "anchored_invoice_candidates",
    "gate_document_identity",
    "to_ryan_ocr",
]
