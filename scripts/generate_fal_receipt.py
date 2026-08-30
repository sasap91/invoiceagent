#!/usr/bin/env python3
"""Generate and OCR-check the synthetic payment receipt with Fal.

The API key is read by ``fal_client`` from ``FAL_KEY`` and is never written to
the image metadata, console output, or repository.  Generated text is not
trusted: Tesseract must recover every proof field before the asset is marked
ready for the demo.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse
from urllib.request import urlopen


MODEL = "fal-ai/flux-2"
PROMPT = """A perfectly flat, front-facing scanned business payment receipt on clean
white thermal paper, high contrast black monospaced printed text, no hands, no
shadows, no perspective, no logos, no decorative graphics. Display these exact
lines with crisp legible spelling:
PAYMENT RECEIPT
Receipt ID: RCPT-FF-10482
Supplier: Fresh Farms
Supplier ID: fresh_farms
Customer: Sugar & Spice Thai Restaurant
Invoice Number: FF-10482
Amount Paid: $1,500.00
Currency: USD
Paid Date: 2026-08-30
Payment Status: PAID IN FULL
SYNTHETIC DEMO - NO AFFILIATION
NO REAL PAYMENT
Centered receipt, generous margins, 4:5 portrait composition."""
REQUIRED_OCR_VALUES = (
    "RCPT-FF-10482",
    "Fresh Farms",
    "fresh_farms",
    "Sugar & Spice Thai Restaurant",
    "FF-10482",
    "1,500.00",
    "USD",
    "2026-08-30",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic receipt with Fal and fail closed on bad OCR."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/procureagent/assets/fresh_farms_payment_receipt_fal.png"),
    )
    parser.add_argument("--seed", type=int, default=138)
    return parser.parse_args()


def extract_image_url(result: object) -> str:
    if not isinstance(result, dict):
        raise RuntimeError("Fal returned an unexpected result type")
    images = result.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], dict):
        raise RuntimeError("Fal result contains no image")
    value = images[0].get("url")
    if not isinstance(value, str) or urlparse(value).scheme != "https":
        raise RuntimeError("Fal returned an unsafe image URL")
    return value


def main() -> int:
    args = arguments()
    if not os.environ.get("FAL_KEY"):
        raise SystemExit("FAL_KEY is not set; no request was made")
    try:
        import fal_client
    except ImportError as exc:
        raise SystemExit("Install the asset extra: pip install -e '.[assets]'") from exc

    result = fal_client.subscribe(
        MODEL,
        arguments={
            "prompt": PROMPT,
            "image_size": {"width": 1024, "height": 1280},
            "num_images": 1,
            "output_format": "png",
            "seed": args.seed,
        },
    )
    url = extract_image_url(result)
    with urlopen(url, timeout=60) as response:  # noqa: S310 - HTTPS is checked above
        image_bytes = response.read(15_000_001)
    if not image_bytes or len(image_bytes) > 15_000_000:
        raise RuntimeError("generated image is empty or exceeds the 15 MB limit")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(image_bytes)
    completed = subprocess.run(
        ["tesseract", str(args.output), "stdout", "--psm", "6"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    ocr_text = completed.stdout
    missing = [value for value in REQUIRED_OCR_VALUES if value not in ocr_text]
    metadata = {
        "asset": args.output.name,
        "sha256": sha256(image_bytes).hexdigest(),
        "source": "fal_api",
        "model": MODEL,
        "seed": args.seed,
        "contains_real_payment_data": False,
        "demonstration_customer": "Sugar & Spice Thai Restaurant",
        "real_restaurant_affiliation": False,
        "restaurant_disclosure": (
            "SYNTHETIC DEMO · NO AFFILIATION. The named restaurant did not "
            "provide this receipt or any financial data."
        ),
        "ocr_engine": "tesseract",
        "ocr_required_values": list(REQUIRED_OCR_VALUES),
        "ocr_missing_values": missing,
        "ready_for_demo": completed.returncode == 0 and not missing,
        "secret_committed": False,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if missing or completed.returncode != 0:
        print(
            "Generated image was retained for inspection but was not promoted; OCR proof failed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
