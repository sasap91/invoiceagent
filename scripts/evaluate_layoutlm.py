#!/usr/bin/env python3
"""Reproduce a small, honest LayoutLMv3 invoice-number evaluation.

The script downloads only requested SROIE samples from Ryan's Hugging Face
dataset. It reports every missing prediction in the denominator. It does not
run or score the quality gate because the published OCR sidecars do not carry
measured OCR confidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from statistics import median
from time import perf_counter

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from invoiceagent import (
    LayoutLMv3InvoiceExtractor,
    OcrDocument,
    extract_anchored_identifier,
    normalize_identifier,
)


DATASET_ID = "ryanznie/SROIE_2019_with_labels"


def split_tokens(text: str) -> list[str]:
    pieces = re.split(r"([/:.#()\[\]-])", text.strip())
    tokens: list[str] = []
    for piece in pieces:
        if piece in "/-:.#[]()":
            tokens.append(piece)
        else:
            tokens.extend(piece.split())
    return [token for token in tokens if token]


def estimate_token_boxes(
    text: str, tokens: list[str], line_box: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = line_box
    width = x1 - x0
    cursor = 0
    boxes: list[tuple[int, int, int, int]] = []
    for token in tokens:
        start = text.find(token, cursor)
        if start < 0:
            start = cursor
        end = start + len(token)
        cursor = end
        left = int(x0 + (start / max(len(text), 1)) * width)
        right = max(left + 1, int(x0 + (end / max(len(text), 1)) * width))
        boxes.append((left, y0, right, y1))
    return boxes


def normalize_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    nx0 = min(999, int(1000 * x0 / width))
    ny0 = min(999, int(1000 * y0 / height))
    nx1 = min(1000, max(nx0 + 1, int(1000 * x1 / width)))
    ny1 = min(1000, max(ny0 + 1, int(1000 * y1 / height)))
    return nx0, ny0, nx1, ny1


def download_path(path: str) -> Path:
    return Path(hf_hub_download(DATASET_ID, path, repo_type="dataset"))


def load_sample(stem: str) -> tuple[Image.Image, OcrDocument]:
    image = Image.open(download_path(f"test/img/{stem}.jpg")).convert("RGB")
    box_path = download_path(f"test/box/{stem}.txt")
    lines: list[tuple[int, int, int, int, str]] = []
    for raw_line in box_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw_line.split(",")
        if len(parts) < 9:
            continue
        coordinates = list(map(int, parts[:8]))
        text = ",".join(parts[8:]).strip()
        if not text:
            continue
        xs, ys = coordinates[::2], coordinates[1::2]
        lines.append((min(xs), min(ys), max(xs), max(ys), text))
    lines.sort(key=lambda line: (line[1], line[0]))

    words: list[str] = []
    boxes: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1, text in lines:
        tokens = split_tokens(text)
        token_boxes = estimate_token_boxes(text, tokens, (x0, y0, x1, y1))
        for token, token_box in zip(tokens, token_boxes):
            words.append(token)
            boxes.append(normalize_box(token_box, image.width, image.height))

    # SROIE publishes precomputed OCR but no per-document quality probability.
    # The required value is a placeholder and is never used to report routing metrics here.
    ocr = OcrDocument(
        words=words,
        boxes=boxes,
        quality="1.0",
        raw_text="\n".join(line[4] for line in lines),
        engine="SROIE published OCR (quality unmeasured)",
    )
    return image, ocr


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", help="Run one test stem, for example X51005200931")
    parser.add_argument("--limit", type=int, default=12, help="Maximum non-ambiguous samples")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--json", action="store_true", help="Print only machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = json.loads(download_path("test/test_labels.json").read_text(encoding="utf-8"))
    if args.sample:
        stems = [args.sample.removesuffix(".jpg")]
    else:
        files = HfApi().list_repo_files(DATASET_ID, repo_type="dataset")
        stems = [
            Path(path).stem
            for path in files
            if path.startswith("test/img/")
            and path.endswith(".jpg")
            and str(labels.get(Path(path).name, "")).casefold() != "ambiguous"
        ][: max(args.limit, 0)]
    if not stems:
        raise SystemExit("no samples selected")

    extractor = LayoutLMv3InvoiceExtractor(
        device=None if args.device == "auto" else args.device
    )
    load_started = perf_counter()
    extractor.load()
    load_seconds = perf_counter() - load_started

    rows: list[dict[str, object]] = []
    for stem in stems:
        image, ocr = load_sample(stem)
        result = extractor.predict(image, ocr)
        ground_truth = str(labels.get(f"{stem}.jpg", ""))
        candidate = result.candidate
        heuristic = extract_anchored_identifier(ocr.words)
        rows.append(
            {
                "sample": stem,
                "ground_truth": ground_truth,
                "candidate": candidate,
                "strict_exact": candidate == ground_truth,
                "normalized_exact": normalize_identifier(candidate or "")
                == normalize_identifier(ground_truth),
                "missing": candidate is None,
                "candidate_spans": len(result.spans),
                "heuristic": heuristic,
                "heuristic_strict_exact": heuristic == ground_truth,
                "latency_ms": float(result.latency_ms),
            }
        )

    latencies = [float(row["latency_ms"]) for row in rows]
    total = len(rows)
    summary = {
        "dataset": DATASET_ID,
        "model": extractor.adapter_model,
        "device": extractor.device,
        "samples": total,
        "strict_exact": sum(bool(row["strict_exact"]) for row in rows),
        "normalized_exact": sum(bool(row["normalized_exact"]) for row in rows),
        "missing": sum(bool(row["missing"]) for row in rows),
        "multiple_spans": sum(int(row["candidate_spans"]) > 1 for row in rows),
        "heuristic_strict_exact": sum(bool(row["heuristic_strict_exact"]) for row in rows),
        "load_seconds": round(load_seconds, 2),
        "latency_p50_ms": round(median(latencies), 1),
        "latency_p95_ms": round(percentile_95(latencies), 1),
        "note": "Smoke slice unless the full locked test set was selected; missing predictions stay in the denominator.",
    }
    output = {"summary": summary, "rows": rows}
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        for row in rows:
            print(
                f"{row['sample']}: gold={row['ground_truth']!r} "
                f"pred={row['candidate']!r} exact={row['strict_exact']} "
                f"spans={row['candidate_spans']} latency={row['latency_ms']}ms"
            )
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
