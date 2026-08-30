from dataclasses import FrozenInstanceError
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import struct
import subprocess
import zlib

import pytest

from procureagent.contracts import ContractValidationError
from procureagent.ocr import (
    ImageFormat,
    OcrStatus,
    PixelBox,
    TesseractOCR,
    ingest_image,
    normalize_pixel_box,
    parse_tesseract_tsv,
)


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_ASSET = ROOT / "data/procureagent/assets/fresh_farms_payment_receipt.png"
RECEIPT_PROVENANCE = ROOT / "data/procureagent/assets/receipt_provenance.json"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def png_bytes(width: int = 100, height: int = 100) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(
        b"IDAT", zlib.compress(scanlines)
    ) + _chunk(b"IEND", b"")


def jpeg_bytes(width: int = 80, height: int = 60) -> bytes:
    frame = (
        b"\xff\xc0"
        + struct.pack(">H", 17)
        + b"\x08"
        + struct.pack(">HH", height, width)
        + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    )
    return b"\xff\xd8" + frame + b"\xff\xd9"


def test_synthetic_receipt_provenance_is_hash_bound_and_disclaims_affiliation():
    provenance = json.loads(RECEIPT_PROVENANCE.read_text(encoding="utf-8"))

    assert provenance["sha256"] == hashlib.sha256(
        RECEIPT_ASSET.read_bytes()
    ).hexdigest()
    assert provenance["contains_real_payment_data"] is False
    assert provenance["demonstration_customer"] == "Sugar & Spice Thai Restaurant"
    assert provenance["real_restaurant_affiliation"] is False
    assert "SYNTHETIC DEMO · NO AFFILIATION" in provenance["restaurant_disclosure"]


TSV = "\n".join(
    (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
        "1\t1\t0\t0\t0\t0\t0\t0\t100\t100\t-1\t",
        "5\t1\t1\t1\t1\t1\t10\t20\t20\t10\t95.5\tInvoice",
        "5\t1\t1\t1\t1\t2\t35\t20\t25\t10\t88\tFF-10482",
        "5\t1\t1\t1\t2\t1\t10\t40\t20\t10\t90\tTotal",
    )
)


def test_png_ingestion_is_content_addressed_and_immutable():
    raw = png_bytes(120, 80)
    image = ingest_image(raw, original_filename="invoice.png")
    digest = hashlib.sha256(raw).hexdigest()
    assert image.document_id == f"doc_{digest}"
    assert image.sha256 == digest
    assert image.image_format is ImageFormat.PNG
    assert image.media_type == "image/png"
    assert (image.width, image.height) == (120, 80)
    with pytest.raises(FrozenInstanceError):
        image.width = 1


def test_jpeg_ingestion_uses_inspected_bytes_not_filename():
    image = ingest_image(jpeg_bytes(), original_filename="actually-a-jpeg.png")
    assert image.image_format is ImageFormat.JPEG
    assert image.media_type == "image/jpeg"
    assert (image.width, image.height) == (80, 60)


@pytest.mark.parametrize(
    "raw",
    (
        b"GIF89a",
        b"\x89PNG\r\n\x1a\ntruncated",
        png_bytes()[:-1] + b"x",
        b"\xff\xd8\xff\xd9",
    ),
)
def test_ingestion_rejects_unsupported_or_structurally_invalid_images(raw):
    with pytest.raises(ContractValidationError):
        ingest_image(raw)


def test_ingestion_enforces_limits_and_safe_filename():
    raw = png_bytes(10, 10)
    with pytest.raises(ContractValidationError, match="byte limit"):
        ingest_image(raw, max_bytes=8)
    with pytest.raises(ContractValidationError, match="pixel count"):
        ingest_image(raw, max_pixels=99)
    with pytest.raises(ContractValidationError, match="basename"):
        ingest_image(raw, original_filename="../invoice.png")


def test_pixel_box_normalization_preserves_tiny_nonzero_boxes():
    normalized = normalize_pixel_box(PixelBox(1, 1, 2, 2), 10_000, 10_000)
    assert normalized.x1 > normalized.x0
    assert normalized.y1 > normalized.y0


def test_tsv_parser_keeps_order_pixel_boxes_normalized_boxes_and_confidence():
    image = ingest_image(png_bytes())
    words = parse_tesseract_tsv(TSV, image)
    assert tuple(word.text for word in words) == ("Invoice", "FF-10482", "Total")
    assert words[0].sequence == 0
    assert words[0].pixel_box == PixelBox(10, 20, 30, 30)
    assert (
        words[0].normalized_box.x0,
        words[0].normalized_box.y0,
        words[0].normalized_box.x1,
        words[0].normalized_box.y1,
    ) == (100, 200, 300, 300)
    assert words[0].confidence == Decimal("0.955")


def test_tesseract_runner_uses_argument_vector_and_returns_metadata():
    image = ingest_image(png_bytes())
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        assert isinstance(command, list)
        assert kwargs["shell"] is False
        if command[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout=b"tesseract 5.4.1\n", stderr=b"")
        assert command[0] == "tesseract"
        assert command[-1] == "tsv"
        assert os.path.exists(command[1])
        return SimpleNamespace(returncode=0, stdout=TSV.encode(), stderr=b"")

    result = TesseractOCR(runner=runner).run(image)
    assert result.status is OcrStatus.SUCCESS
    assert result.document_id == image.document_id
    assert result.ordered_text == ("Invoice", "FF-10482", "Total")
    assert result.raw_text == "Invoice FF-10482\nTotal"
    assert result.language == "eng"
    assert result.engine == "tesseract_local"
    assert result.engine_version == "tesseract 5.4.1"
    assert result.runtime_ms >= 0
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("runner", "expected_status", "expected_code"),
    (
        (
            lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
            OcrStatus.UNAVAILABLE,
            "TESSERACT_NOT_FOUND",
        ),
        (
            lambda *args, **kwargs: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(args[0], 1)
            ),
            OcrStatus.TIMEOUT,
            "TESSERACT_TIMEOUT",
        ),
        (
            lambda *args, **kwargs: SimpleNamespace(
                returncode=2, stdout=b"", stderr=b"bad image"
            ),
            OcrStatus.FAILED,
            "TESSERACT_NONZERO_EXIT",
        ),
    ),
)
def test_tesseract_operational_failures_are_explicit(runner, expected_status, expected_code):
    result = TesseractOCR(runner=runner).run(ingest_image(png_bytes()))
    assert result.status is expected_status
    assert result.error_code == expected_code
    assert result.words == ()


def test_tesseract_malformed_tsv_fails_closed():
    def runner(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=b"not\ta\ttsv\n", stderr=b"")

    result = TesseractOCR(runner=runner).run(ingest_image(png_bytes()))
    assert result.status is OcrStatus.FAILED
    assert result.error_code == "MALFORMED_TESSERACT_TSV"


def test_tesseract_no_text_is_not_reported_as_success():
    header = TSV.splitlines()[0] + "\n"

    def runner(command, **kwargs):
        if command[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout=b"tesseract test\n", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=header.encode(), stderr=b"")

    result = TesseractOCR(runner=runner).run(ingest_image(png_bytes()))
    assert result.status is OcrStatus.NO_TEXT
    assert result.words == ()
    assert result.raw_text == ""


@pytest.mark.skipif(
    os.environ.get("RUN_TESSERACT_SMOKE") != "1" or shutil.which("tesseract") is None,
    reason="set RUN_TESSERACT_SMOKE=1 with local tesseract to run",
)
def test_real_local_tesseract_opt_in_smoke():
    result = TesseractOCR(timeout_seconds=10).run(ingest_image(png_bytes(200, 60)))
    assert result.status in {OcrStatus.SUCCESS, OcrStatus.NO_TEXT}
    assert result.engine == "tesseract_local"
    assert result.engine_version != ""
