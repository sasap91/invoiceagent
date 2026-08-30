"""Invoice-number extraction adapters for the small-first pipeline.

The dependency-light helpers in this module are always available. Heavy model
libraries are imported only when ``LayoutLMv3InvoiceExtractor.load`` is called,
so reconciliation and fixture mode still work without PyTorch installed.

LayoutLMv3 does not perform OCR here. Callers must supply the image together
with OCR words and normalized bounding boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from time import perf_counter
from typing import Any, Iterable, Sequence

from .core import (
    ExtractionMetadata,
    ExtractionSource,
    RoutingSignals,
    ValidationError,
    normalize_identifier,
)


DEFAULT_ADAPTER_MODEL = "ryanznie/layoutlmv3-lora-invoice-number"
DEFAULT_BASE_MODEL = "microsoft/layoutlmv3-base"
DEFAULT_ADAPTER_REVISION = "7dc28f5a3b14aa100ba432ee1b0a6cac6c7b2c5c"
DEFAULT_BASE_REVISION = "cfbbbff0762e6aab37086fdd4739ad14fe7d5db4"

_DATE_LIKE = re.compile(r"^(?:\d{1,4}[-/.]){2}\d{1,4}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/#:-]{2,127}$")
_RULE_PATTERNS = (
    re.compile(
        r"\b(?:invoice|inv|bill)\s*(?:number|no\.?|#)?\s*[:#-]?\s*"
        r"([A-Za-z0-9][A-Za-z0-9._/#:-]{2,127})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:receipt|rcpt|transaction|trn)\s*(?:number|no\.?|#)?\s*[:#-]?\s*"
        r"([A-Za-z0-9][A-Za-z0-9._/#:-]{2,127})\b",
        re.IGNORECASE,
    ),
)


def _probability(value: Decimal | str, name: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, (float, int)):
        raise ValidationError(f"{name} must be Decimal or a decimal string")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValidationError(f"{name} is not a decimal probability") from exc
    if not parsed.is_finite() or parsed < 0 or parsed > 1:
        raise ValidationError(f"{name} must be between 0 and 1")
    return parsed


def _box(value: Sequence[int], index: int) -> tuple[int, int, int, int]:
    if len(value) != 4 or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValidationError(f"OCR box {index} must contain four integers")
    x0, y0, x1, y1 = value
    if not all(0 <= item <= 1000 for item in value):
        raise ValidationError(f"OCR box {index} must be normalized to 0..1000")
    if x0 >= x1 or y0 >= y1:
        raise ValidationError(f"OCR box {index} has invalid geometry")
    return x0, y0, x1, y1


@dataclass(frozen=True, slots=True)
class OcrDocument:
    """Words and boxes produced by an OCR system outside LayoutLMv3."""

    words: Sequence[str]
    boxes: Sequence[Sequence[int]]
    quality: Decimal | str
    raw_text: str = ""
    engine: str = "precomputed"

    def __post_init__(self) -> None:
        words = tuple(self.words)
        boxes = tuple(self.boxes)
        if not words or len(words) != len(boxes):
            raise ValidationError("OCR words and boxes must be non-empty and the same length")
        if any(not isinstance(word, str) or not word.strip() for word in words):
            raise ValidationError("every OCR word must be non-empty text")
        object.__setattr__(self, "words", tuple(word.strip() for word in words))
        object.__setattr__(
            self,
            "boxes",
            tuple(_box(value, index) for index, value in enumerate(boxes)),
        )
        object.__setattr__(self, "quality", _probability(self.quality, "OCR quality"))
        if not isinstance(self.raw_text, str) or not isinstance(self.engine, str):
            raise ValidationError("OCR raw_text and engine must be text")


def normalize_pixel_boxes(
    boxes: Iterable[Sequence[int]], image_width: int, image_height: int
) -> tuple[tuple[int, int, int, int], ...]:
    """Normalize pixel-space OCR boxes into LayoutLMv3's 0..1000 space."""

    if image_width <= 0 or image_height <= 0:
        raise ValidationError("image dimensions must be positive")
    normalized: list[tuple[int, int, int, int]] = []
    for index, value in enumerate(boxes):
        if len(value) != 4 or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValidationError(f"pixel box {index} must contain four integers")
        x0, y0, x1, y1 = value
        if x0 < 0 or y0 < 0 or x1 > image_width or y1 > image_height:
            raise ValidationError(f"pixel box {index} falls outside the image")
        if x0 >= x1 or y0 >= y1:
            raise ValidationError(f"pixel box {index} has invalid geometry")
        candidate = (
            round(1000 * x0 / image_width),
            round(1000 * y0 / image_height),
            round(1000 * x1 / image_width),
            round(1000 * y1 / image_height),
        )
        normalized.append(_box(candidate, index))
    return tuple(normalized)


def is_valid_invoice_identifier(value: str | None) -> bool:
    """Reject empty, date-like, amount-like, and structurally unsafe candidates."""

    if not value or not _IDENTIFIER.fullmatch(value):
        return False
    if _DATE_LIKE.fullmatch(value):
        return False
    if value.replace(".", "", 1).isdigit() and "." in value:
        return False
    return any(character.isdigit() for character in value)


def extract_anchored_identifier(words: Sequence[str]) -> str | None:
    """Return a conservative label-anchored identifier candidate from OCR text."""

    labels = {"invoice", "inv", "bill", "receipt", "rcpt", "transaction", "trn"}
    qualifiers = {"no", "number", "#", ":"}
    field_boundaries = {
        "amount",
        "currency",
        "date",
        "due",
        "subtotal",
        "tax",
        "total",
    }
    for index, word in enumerate(words):
        normalized_word = word.casefold().strip().rstrip(".:")
        if normalized_word not in labels:
            continue
        cursor = index + 1
        while cursor < len(words) and words[cursor].casefold().strip().rstrip(".") in qualifiers:
            cursor += 1
        pieces: list[str] = []
        best: str | None = None
        for candidate_word in words[cursor : cursor + 5]:
            cleaned = candidate_word.strip()
            if cleaned.casefold().rstrip(":") in field_boundaries:
                break
            if not re.fullmatch(r"[A-Za-z0-9._/#:-]+", cleaned):
                break
            pieces.append(cleaned)
            candidate = _join_identifier_tokens(pieces).rstrip(".,;:")
            if is_valid_invoice_identifier(candidate):
                best = candidate
        if best:
            return best

    text = " ".join(words)
    for pattern in _RULE_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = match.group(1).rstrip(".,;:")
            if is_valid_invoice_identifier(candidate):
                return candidate
    return None


def _label_kind(label: str) -> str:
    normalized = label.upper().replace("_", "-")
    if normalized in {
        "LABEL-1",
        "B",
        "B-INVOICE",
        "B-INVOICE-ID",
        "B-INVOICE-NUMBER",
        "B-INV",
    }:
        return "B"
    if normalized in {
        "LABEL-2",
        "I",
        "I-INVOICE",
        "I-INVOICE-ID",
        "I-INVOICE-NUMBER",
        "I-INV",
    }:
        return "I"
    return "O"


def _join_identifier_tokens(tokens: Sequence[str]) -> str:
    value = " ".join(tokens).strip()
    return re.sub(r"\s*([./#:-])\s*", r"\1", value)


@dataclass(frozen=True, slots=True)
class EntitySpan:
    value: str
    word_indices: tuple[int, ...]
    boxes: tuple[tuple[int, int, int, int], ...]
    token_confidences: tuple[Decimal, ...]
    token_margins: tuple[Decimal, ...]

    @property
    def minimum_confidence(self) -> Decimal:
        return min(self.token_confidences)

    @property
    def mean_confidence(self) -> Decimal:
        return sum(self.token_confidences, Decimal("0")) / len(self.token_confidences)

    @property
    def mean_margin(self) -> Decimal:
        return sum(self.token_margins, Decimal("0")) / len(self.token_margins)


def decode_bio_spans(
    words: Sequence[str],
    boxes: Sequence[Sequence[int]],
    labels: Sequence[str],
    confidences: Sequence[Decimal | str],
    margins: Sequence[Decimal | str],
) -> tuple[EntitySpan, ...]:
    """Decode contiguous invoice spans and preserve their evidence.

    Confidence is attached only to selected entity words. Background ``O``
    tokens never inflate the score.
    """

    size = len(words)
    if not size or not all(len(values) == size for values in (boxes, labels, confidences, margins)):
        raise ValidationError("BIO words, boxes, labels, confidences, and margins must align")
    checked_boxes = tuple(_box(value, index) for index, value in enumerate(boxes))
    checked_confidences = tuple(
        _probability(value, f"confidence {index}") for index, value in enumerate(confidences)
    )
    checked_margins = tuple(
        _probability(value, f"margin {index}") for index, value in enumerate(margins)
    )

    groups: list[list[int]] = []
    current: list[int] = []
    for index, label in enumerate(labels):
        kind = _label_kind(label)
        if kind == "B":
            if current:
                groups.append(current)
            current = [index]
        elif kind == "I":
            if current:
                current.append(index)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    spans: list[EntitySpan] = []
    for indices in groups:
        value = _join_identifier_tokens([words[index] for index in indices])
        spans.append(
            EntitySpan(
                value=value,
                word_indices=tuple(indices),
                boxes=tuple(checked_boxes[index] for index in indices),
                token_confidences=tuple(checked_confidences[index] for index in indices),
                token_margins=tuple(checked_margins[index] for index in indices),
            )
        )
    return tuple(spans)


@dataclass(frozen=True, slots=True)
class InvoiceNumberResult:
    """One model call, including every candidate span and its provenance."""

    spans: tuple[EntitySpan, ...]
    model_name: str
    latency_ms: Decimal

    @property
    def selected(self) -> EntitySpan | None:
        if not self.spans:
            return None
        return max(
            self.spans,
            key=lambda span: (
                span.minimum_confidence,
                span.mean_confidence,
                span.mean_margin,
                -span.word_indices[0],
            ),
        )

    @property
    def candidate(self) -> str | None:
        return self.selected.value if self.selected else None

    @property
    def ambiguous(self) -> bool:
        return len(self.spans) != 1

    def routing_signals(
        self,
        ocr: OcrDocument,
        *,
        heuristic_candidate: str | None,
        escalation_available: bool = True,
    ) -> RoutingSignals:
        selected = self.selected
        if selected is None:
            metadata = ExtractionMetadata(
                source=ExtractionSource.UNKNOWN,
                note=f"{self.model_name} produced no invoice-number span",
            )
            agreement = None
        else:
            indices_in_range = bool(selected.word_indices) and all(
                0 <= index < len(ocr.words) for index in selected.word_indices
            )
            expected_value = (
                _join_identifier_tokens(
                    [ocr.words[index] for index in selected.word_indices]
                )
                if indices_in_range
                else ""
            )
            grounded = (
                indices_in_range
                and normalize_identifier(expected_value)
                == normalize_identifier(selected.value)
                and selected.boxes
                == tuple(ocr.boxes[index] for index in selected.word_indices)
            )
            metadata = ExtractionMetadata(
                source=ExtractionSource.SMALL_MODEL,
                confidence=selected.minimum_confidence,
                grounded=grounded,
                model_name=self.model_name,
                note=(
                    f"entity mean={selected.mean_confidence:.4f}; "
                    f"mean margin={selected.mean_margin:.4f}; latency={self.latency_ms:.1f}ms"
                ),
            )
            agreement = (
                None
                if heuristic_candidate is None
                else normalize_identifier(heuristic_candidate)
                == normalize_identifier(selected.value)
            )
        return RoutingSignals(
            extraction=metadata,
            ocr_quality=ocr.quality,
            validation_passed=is_valid_invoice_identifier(self.candidate),
            heuristic_model_agreement=agreement,
            ambiguity_detected=self.ambiguous,
            escalation_available=escalation_available,
        )


class LayoutLMv3InvoiceExtractor:
    """Lazy wrapper for Ryan's LayoutLMv3 LoRA invoice-number adapter."""

    def __init__(
        self,
        *,
        adapter_model: str = DEFAULT_ADAPTER_MODEL,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_revision: str = DEFAULT_ADAPTER_REVISION,
        base_revision: str = DEFAULT_BASE_REVISION,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.adapter_model = adapter_model
        self.base_model = base_model
        self.adapter_revision = adapter_revision
        self.base_revision = base_revision
        self.requested_device = device
        self.device = "cpu"
        self.max_length = max_length
        self.processor: Any = None
        self.model: Any = None
        self._torch: Any = None

    def load(self) -> None:
        """Download/load the base and adapter. This is intentionally explicit."""

        try:
            import torch
            from peft import PeftModel
            from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
        except ImportError as exc:
            raise RuntimeError(
                "Model dependencies are missing; install with pip install -e '.[model]'"
            ) from exc

        if self.requested_device:
            self.device = self.requested_device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.processor = LayoutLMv3Processor.from_pretrained(
            self.adapter_model,
            revision=self.adapter_revision,
            apply_ocr=False,
        )
        base = LayoutLMv3ForTokenClassification.from_pretrained(
            self.base_model,
            revision=self.base_revision,
            num_labels=3,
        )
        self.model = PeftModel.from_pretrained(
            base,
            self.adapter_model,
            revision=self.adapter_revision,
        )
        self.model.to(self.device)
        self.model.eval()
        self._torch = torch

    def predict(self, image: Any, ocr: OcrDocument) -> InvoiceNumberResult:
        if self.model is None or self.processor is None or self._torch is None:
            raise RuntimeError("model is not loaded; call load() once before predict()")
        if not isinstance(ocr, OcrDocument):
            raise ValidationError("ocr must be an OcrDocument")

        started = perf_counter()
        encoding = self.processor(
            image,
            list(ocr.words),
            boxes=[list(box) for box in ocr.boxes],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(0)
        device_encoding = {key: value.to(self.device) for key, value in encoding.items()}
        with self._torch.no_grad():
            logits = self.model(**device_encoding).logits[0]
            probabilities = self._torch.softmax(logits, dim=-1)
            top_values, top_indices = probabilities.topk(k=2, dim=-1)

        per_word: dict[int, tuple[str, Decimal, Decimal]] = {}
        id2label = self.model.config.id2label
        for token_index, word_index in enumerate(word_ids):
            if word_index is None or word_index in per_word or word_index >= len(ocr.words):
                continue
            predicted_id = int(top_indices[token_index, 0].detach().cpu().item())
            label = id2label.get(predicted_id, f"LABEL_{predicted_id}")
            confidence = Decimal(str(float(top_values[token_index, 0].detach().cpu().item())))
            runner_up = Decimal(str(float(top_values[token_index, 1].detach().cpu().item())))
            per_word[word_index] = (label, confidence, confidence - runner_up)

        labels: list[str] = []
        confidences: list[Decimal] = []
        margins: list[Decimal] = []
        for word_index in range(len(ocr.words)):
            label, confidence, margin = per_word.get(
                word_index, ("O", Decimal("0"), Decimal("0"))
            )
            labels.append(label)
            confidences.append(confidence)
            margins.append(max(margin, Decimal("0")))

        spans = decode_bio_spans(ocr.words, ocr.boxes, labels, confidences, margins)
        latency = Decimal(str((perf_counter() - started) * 1000)).quantize(Decimal("0.1"))
        return InvoiceNumberResult(
            spans=spans,
            model_name=(
                f"{self.adapter_model}@{self.adapter_revision} "
                f"(base {self.base_model}@{self.base_revision}) on {self.device}"
            ),
            latency_ms=latency,
        )
