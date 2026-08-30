"""Safe image ingestion and an explicit local Tesseract TSV adapter.

Nothing executes at import time.  Tesseract is invoked only by ``run`` and is
always called with an argument vector and ``shell=False``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import io
from pathlib import Path
import re
import struct
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Callable, Sequence
import zlib

from .contracts import BoundingBox, ContractValidationError


DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000
DEFAULT_MAX_DIMENSION = 10_000
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_LANGUAGE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.-]{0,63}\Z")
_JPEG_START_OF_FRAME = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


class ImageFormat(str, Enum):
    PNG = "PNG"
    JPEG = "JPEG"


class OcrStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NO_TEXT = "NO_TEXT"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    UNAVAILABLE = "UNAVAILABLE"


def _strict_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractValidationError(f"{name} must be a positive integer")
    return value


def _validate_dimensions(
    width: int,
    height: int,
    *,
    max_pixels: int,
    max_dimension: int,
) -> tuple[int, int]:
    _strict_positive_int(width, "image width")
    _strict_positive_int(height, "image height")
    if width > max_dimension or height > max_dimension:
        raise ContractValidationError("image dimensions exceed the configured limit")
    if width * height > max_pixels:
        raise ContractValidationError("image pixel count exceeds the configured limit")
    return width, height


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if not data.startswith(_PNG_SIGNATURE):
        raise ContractValidationError("image is not a PNG")
    offset = len(_PNG_SIGNATURE)
    width: int | None = None
    height: int | None = None
    saw_iend = False
    chunk_index = 0
    while offset < len(data):
        if offset + 12 > len(data):
            raise ContractValidationError("PNG contains a truncated chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            raise ContractValidationError("PNG chunk length exceeds the image bytes")
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : chunk_end])[0]
        actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise ContractValidationError("PNG contains an invalid chunk checksum")
        if chunk_index == 0:
            if chunk_type != b"IHDR" or length != 13:
                raise ContractValidationError("PNG must begin with a valid IHDR chunk")
            width, height = struct.unpack(">II", payload[:8])
        if chunk_type == b"IEND":
            if length != 0 or chunk_end != len(data):
                raise ContractValidationError("PNG has invalid data after IEND")
            saw_iend = True
            break
        offset = chunk_end
        chunk_index += 1
    if width is None or height is None or not saw_iend:
        raise ContractValidationError("PNG is missing required structural chunks")
    return width, height


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ContractValidationError("image is not a complete JPEG")
    offset = 2
    while offset < len(data) - 2:
        if data[offset] != 0xFF:
            raise ContractValidationError("JPEG marker structure is invalid")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0x01, 0xD8, 0xD9}:
            continue
        if marker == 0xDA:
            break
        if offset + 2 > len(data):
            raise ContractValidationError("JPEG contains a truncated segment")
        segment_length = struct.unpack(">H", data[offset : offset + 2])[0]
        if segment_length < 2 or offset + segment_length > len(data):
            raise ContractValidationError("JPEG segment length is invalid")
        if marker in _JPEG_START_OF_FRAME:
            if segment_length < 7:
                raise ContractValidationError("JPEG frame header is too short")
            height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
            return width, height
        offset += segment_length
    raise ContractValidationError("JPEG has no supported frame header")


@dataclass(frozen=True, slots=True)
class IngestedImage:
    document_id: str
    sha256: str
    image_bytes: bytes
    image_format: ImageFormat
    media_type: str
    width: int
    height: int
    original_filename: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image_bytes, bytes) or not self.image_bytes:
            raise ContractValidationError("image_bytes must be non-empty immutable bytes")
        digest = hashlib.sha256(self.image_bytes).hexdigest()
        if self.sha256 != digest or self.document_id != f"doc_{digest}":
            raise ContractValidationError("document ID and SHA-256 must derive from image bytes")
        if self.image_format is ImageFormat.PNG:
            expected_media_type = "image/png"
        elif self.image_format is ImageFormat.JPEG:
            expected_media_type = "image/jpeg"
        else:
            raise ContractValidationError("image_format must be PNG or JPEG")
        if self.media_type != expected_media_type:
            raise ContractValidationError("media_type does not match the inspected bytes")
        _strict_positive_int(self.width, "width")
        _strict_positive_int(self.height, "height")
        if self.original_filename is not None:
            if not isinstance(self.original_filename, str):
                raise ContractValidationError("original_filename must be text")
            if (
                not self.original_filename
                or self.original_filename != Path(self.original_filename).name
                or "\x00" in self.original_filename
                or len(self.original_filename) > 255
            ):
                raise ContractValidationError("original_filename must be a safe basename")


def ingest_image(
    image_bytes: bytes,
    *,
    original_filename: str | None = None,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> IngestedImage:
    """Validate PNG/JPEG structure and derive an immutable content-addressed ID."""

    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ContractValidationError("image input must be non-empty bytes")
    _strict_positive_int(max_bytes, "max_bytes")
    _strict_positive_int(max_pixels, "max_pixels")
    _strict_positive_int(max_dimension, "max_dimension")
    if len(image_bytes) > max_bytes:
        raise ContractValidationError("image exceeds the configured byte limit")
    if image_bytes.startswith(_PNG_SIGNATURE):
        image_format = ImageFormat.PNG
        media_type = "image/png"
        width, height = _png_dimensions(image_bytes)
    elif image_bytes.startswith(b"\xff\xd8"):
        image_format = ImageFormat.JPEG
        media_type = "image/jpeg"
        width, height = _jpeg_dimensions(image_bytes)
    else:
        raise ContractValidationError("only inspected PNG and JPEG bytes are accepted")
    _validate_dimensions(
        width,
        height,
        max_pixels=max_pixels,
        max_dimension=max_dimension,
    )
    digest = hashlib.sha256(image_bytes).hexdigest()
    return IngestedImage(
        document_id=f"doc_{digest}",
        sha256=digest,
        image_bytes=image_bytes,
        image_format=image_format,
        media_type=media_type,
        width=width,
        height=height,
        original_filename=original_filename,
    )


@dataclass(frozen=True, slots=True)
class PixelBox:
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        values = (self.x0, self.y0, self.x1, self.y1)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ContractValidationError("pixel box coordinates must be integers")
        if self.x0 < 0 or self.y0 < 0 or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ContractValidationError("pixel box geometry is invalid")


def normalize_pixel_box(box: PixelBox, width: int, height: int) -> BoundingBox:
    if not isinstance(box, PixelBox):
        raise ContractValidationError("box must be PixelBox")
    _strict_positive_int(width, "width")
    _strict_positive_int(height, "height")
    if box.x1 > width or box.y1 > height:
        raise ContractValidationError("pixel box falls outside the image")
    x0 = 1000 * box.x0 // width
    y0 = 1000 * box.y0 // height
    x1 = min(1000, (1000 * box.x1 + width - 1) // width)
    y1 = min(1000, (1000 * box.y1 + height - 1) // height)
    return BoundingBox(x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))


@dataclass(frozen=True, slots=True)
class OcrWord:
    sequence: int
    text: str
    confidence: Decimal
    pixel_box: PixelBox
    normalized_box: BoundingBox
    page: int
    block: int
    paragraph: int
    line: int
    word: int

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ContractValidationError("OCR sequence must be a non-negative integer")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ContractValidationError("OCR word must contain non-empty text")
        object.__setattr__(self, "text", self.text.strip())
        if not isinstance(self.confidence, Decimal):
            raise ContractValidationError("OCR confidence must be Decimal")
        if not self.confidence.is_finite() or not Decimal("0") <= self.confidence <= Decimal("1"):
            raise ContractValidationError("OCR confidence must be between 0 and 1")
        if not isinstance(self.pixel_box, PixelBox) or not isinstance(
            self.normalized_box, BoundingBox
        ):
            raise ContractValidationError("OCR word boxes use PixelBox and BoundingBox")
        for name in ("page", "block", "paragraph", "line", "word"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractValidationError(f"OCR {name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class OcrResult:
    document_id: str
    status: OcrStatus
    words: tuple[OcrWord, ...]
    raw_text: str
    language: str
    engine: str
    engine_version: str
    runtime_ms: Decimal
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.startswith("doc_"):
            raise ContractValidationError("OCR document_id must be a content-derived ID")
        if not isinstance(self.status, OcrStatus):
            raise ContractValidationError("OCR status must be OcrStatus")
        words = tuple(self.words)
        if not all(isinstance(word, OcrWord) for word in words):
            raise ContractValidationError("OCR words must contain OcrWord values")
        if tuple(word.sequence for word in words) != tuple(range(len(words))):
            raise ContractValidationError("OCR words must preserve contiguous TSV order")
        object.__setattr__(self, "words", words)
        for name in ("raw_text", "language", "engine", "engine_version"):
            if not isinstance(getattr(self, name), str):
                raise ContractValidationError(f"OCR {name} must be text")
        if not _LANGUAGE_PATTERN.fullmatch(self.language):
            raise ContractValidationError("OCR language has unsupported characters")
        if not isinstance(self.runtime_ms, Decimal) or self.runtime_ms < 0:
            raise ContractValidationError("OCR runtime_ms must be non-negative Decimal")
        if self.status is OcrStatus.SUCCESS and not words:
            raise ContractValidationError("successful OCR must contain words")
        if self.status is OcrStatus.NO_TEXT and words:
            raise ContractValidationError("NO_TEXT OCR cannot contain words")
        if self.status in {OcrStatus.FAILED, OcrStatus.TIMEOUT, OcrStatus.UNAVAILABLE}:
            if words or not self.error_code:
                raise ContractValidationError("failed OCR needs an error code and no words")
        elif self.error_code is not None or self.error_message is not None:
            raise ContractValidationError("successful/no-text OCR cannot carry an error")

    @property
    def quality(self) -> Decimal:
        if not self.words:
            return Decimal("0")
        return sum((word.confidence for word in self.words), Decimal("0")) / len(
            self.words
        )

    @property
    def ordered_text(self) -> tuple[str, ...]:
        return tuple(word.text for word in self.words)


_TSV_COLUMNS = (
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
)


def _tsv_int(row: dict[str | None, str | list[str] | None], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, str):
        raise ContractValidationError(f"Tesseract TSV is missing {key}")
    try:
        return int(value)
    except ValueError as exc:
        raise ContractValidationError(f"Tesseract TSV {key} is not an integer") from exc


def parse_tesseract_tsv(tsv: str, image: IngestedImage) -> tuple[OcrWord, ...]:
    """Parse word-level TSV rows in emitted order and retain both box spaces."""

    if not isinstance(tsv, str):
        raise ContractValidationError("Tesseract TSV must be text")
    reader = csv.DictReader(io.StringIO(tsv), delimiter="\t")
    if tuple(reader.fieldnames or ()) != _TSV_COLUMNS:
        raise ContractValidationError("Tesseract TSV header is unsupported")
    words: list[OcrWord] = []
    for row in reader:
        if None in row:
            raise ContractValidationError("Tesseract TSV row has extra columns")
        text_value = row.get("text")
        if _tsv_int(row, "level") != 5 or not isinstance(text_value, str) or not text_value.strip():
            continue
        left = _tsv_int(row, "left")
        top = _tsv_int(row, "top")
        box_width = _tsv_int(row, "width")
        box_height = _tsv_int(row, "height")
        pixel_box = PixelBox(left, top, left + box_width, top + box_height)
        confidence_value = row.get("conf")
        if not isinstance(confidence_value, str):
            raise ContractValidationError("Tesseract TSV is missing confidence")
        try:
            raw_confidence = Decimal(confidence_value)
        except InvalidOperation as exc:
            raise ContractValidationError("Tesseract confidence is invalid") from exc
        if not raw_confidence.is_finite() or not Decimal("0") <= raw_confidence <= Decimal("100"):
            raise ContractValidationError("word confidence must be between 0 and 100")
        words.append(
            OcrWord(
                sequence=len(words),
                text=text_value,
                confidence=raw_confidence / Decimal("100"),
                pixel_box=pixel_box,
                normalized_box=normalize_pixel_box(
                    pixel_box, image.width, image.height
                ),
                page=_tsv_int(row, "page_num"),
                block=_tsv_int(row, "block_num"),
                paragraph=_tsv_int(row, "par_num"),
                line=_tsv_int(row, "line_num"),
                word=_tsv_int(row, "word_num"),
            )
        )
    return tuple(words)


def _raw_text(words: Sequence[OcrWord]) -> str:
    chunks: list[str] = []
    previous_line: tuple[int, int, int, int] | None = None
    for word in words:
        current_line = (word.page, word.block, word.paragraph, word.line)
        if chunks:
            chunks.append("\n" if current_line != previous_line else " ")
        chunks.append(word.text)
        previous_line = current_line
    return "".join(chunks)


Runner = Callable[..., Any]


class TesseractOCR:
    """A local-only Tesseract runner with explicit operational failure results."""

    def __init__(
        self,
        *,
        binary: str = "tesseract",
        language: str = "eng",
        page_segmentation_mode: int = 6,
        timeout_seconds: int = 20,
        runner: Runner = subprocess.run,
    ) -> None:
        if not isinstance(binary, str) or not binary or "\x00" in binary:
            raise ContractValidationError("Tesseract binary must be non-empty text")
        if not _LANGUAGE_PATTERN.fullmatch(language):
            raise ContractValidationError("Tesseract language has unsupported characters")
        if (
            isinstance(page_segmentation_mode, bool)
            or not isinstance(page_segmentation_mode, int)
            or not 0 <= page_segmentation_mode <= 13
        ):
            raise ContractValidationError("page segmentation mode must be 0..13")
        self.binary = binary
        self.language = language
        self.page_segmentation_mode = page_segmentation_mode
        self.timeout_seconds = _strict_positive_int(timeout_seconds, "timeout_seconds")
        self._runner = runner
        self._engine_version: str | None = None

    def _version(self) -> str:
        if self._engine_version is not None:
            return self._engine_version
        try:
            completed = self._runner(
                [self.binary, "--version"],
                capture_output=True,
                check=False,
                timeout=min(self.timeout_seconds, 5),
                shell=False,
            )
            output = completed.stdout
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            first_line = output.splitlines()[0].strip() if output else ""
            self._engine_version = first_line[:128] or "unknown"
        except (OSError, subprocess.SubprocessError):
            self._engine_version = "unknown"
        return self._engine_version

    @staticmethod
    def _runtime(started: float) -> Decimal:
        return Decimal(str((perf_counter() - started) * 1000)).quantize(Decimal("0.1"))

    def _failure(
        self,
        image: IngestedImage,
        status: OcrStatus,
        code: str,
        message: str,
        started: float,
    ) -> OcrResult:
        return OcrResult(
            document_id=image.document_id,
            status=status,
            words=(),
            raw_text="",
            language=self.language,
            engine="tesseract_local",
            engine_version=self._engine_version or "unknown",
            runtime_ms=self._runtime(started),
            error_code=code,
            error_message=message[:512],
        )

    def run(self, image: IngestedImage) -> OcrResult:
        if not isinstance(image, IngestedImage):
            raise ContractValidationError("image must be an IngestedImage")
        started = perf_counter()
        suffix = ".png" if image.image_format is ImageFormat.PNG else ".jpg"
        try:
            with tempfile.TemporaryDirectory(prefix="procureagent-ocr-") as directory:
                image_path = Path(directory) / f"input{suffix}"
                image_path.write_bytes(image.image_bytes)
                completed = self._runner(
                    [
                        self.binary,
                        str(image_path),
                        "stdout",
                        "-l",
                        self.language,
                        "--psm",
                        str(self.page_segmentation_mode),
                        "tsv",
                    ],
                    capture_output=True,
                    check=False,
                    timeout=self.timeout_seconds,
                    shell=False,
                )
        except FileNotFoundError:
            return self._failure(
                image,
                OcrStatus.UNAVAILABLE,
                "TESSERACT_NOT_FOUND",
                "local Tesseract executable was not found",
                started,
            )
        except subprocess.TimeoutExpired:
            return self._failure(
                image,
                OcrStatus.TIMEOUT,
                "TESSERACT_TIMEOUT",
                "local Tesseract exceeded the configured timeout",
                started,
            )
        except OSError as exc:
            return self._failure(
                image,
                OcrStatus.FAILED,
                "TESSERACT_OS_ERROR",
                str(exc),
                started,
            )

        stdout = completed.stdout
        stderr = completed.stderr
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            return self._failure(
                image,
                OcrStatus.FAILED,
                "TESSERACT_NONZERO_EXIT",
                (stderr or f"Tesseract exited with code {completed.returncode}"),
                started,
            )
        try:
            words = parse_tesseract_tsv(stdout, image)
        except ContractValidationError as exc:
            return self._failure(
                image,
                OcrStatus.FAILED,
                "MALFORMED_TESSERACT_TSV",
                str(exc),
                started,
            )
        version = self._version()
        status = OcrStatus.SUCCESS if words else OcrStatus.NO_TEXT
        return OcrResult(
            document_id=image.document_id,
            status=status,
            words=words,
            raw_text=_raw_text(words),
            language=self.language,
            engine="tesseract_local",
            engine_version=version,
            runtime_ms=self._runtime(started),
        )


__all__ = [
    "DEFAULT_MAX_DIMENSION",
    "DEFAULT_MAX_IMAGE_BYTES",
    "DEFAULT_MAX_IMAGE_PIXELS",
    "ImageFormat",
    "IngestedImage",
    "OcrResult",
    "OcrStatus",
    "OcrWord",
    "PixelBox",
    "TesseractOCR",
    "ingest_image",
    "normalize_pixel_box",
    "parse_tesseract_tsv",
]
