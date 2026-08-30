"""Render the synthetic supplier-invoice images used by the /eval recording lane.

Every invoice in the demo scenario needs a document if the claim "the agent
reads the invoices" is to be literally true. Only Fresh Farms had one; this
renders the rest from one shared template.

The layout deliberately mirrors ``fresh_farms_invoice.svg`` because that layout
is known to survive Tesseract. Two anchoring rules from
``procureagent.document`` govern it and must not be broken:

* ``_LABELS = {"invoice", "inv", "bill"}`` starts a candidate, and ``No:`` is a
  qualifier -- so exactly one ``Invoice No: <ID>`` line may appear.
* ``_BOUNDARIES`` contains ``date``, so ``Invoice Date:`` terminates the scan
  and yields no second candidate. Any *other* ``Invoice <word>`` line would
  produce a second anchored candidate and trip ``AMBIGUOUS_RULE_CANDIDATES``,
  blocking the document gate.

These are synthetic fixtures for a hackathon demo. They are not real invoices
and carry no financial obligation.

Usage::

    python scripts/generate_invoice_assets.py
    python scripts/generate_invoice_assets.py --check   # verify, write nothing
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "data" / "procureagent" / "assets"

WIDTH, HEIGHT = 1000, 1280
INK = (23, 51, 45)
PAPER = (247, 245, 239)
CARD = (255, 255, 255)
MUTED = (86, 105, 99)
LEAF = (19, 121, 91)
LEAF_SOFT = (233, 245, 239)
RULE = (188, 201, 195)

FONT_CANDIDATES = {
    "sans": (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ),
    "sans_bold": (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
    "mono": (
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/Library/Fonts/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ),
    "mono_bold": (
        "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
        "/Library/Fonts/Courier New Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ),
}


@dataclass(frozen=True)
class InvoiceSpec:
    """One synthetic invoice, mirroring its record in the scenario fixture."""

    supplier_id: str
    display_name: str
    legal_name: str
    address: str
    invoice_number: str
    amount_minor: int
    category: str
    line_description: str
    invoice_date: str
    due_date: str
    footer_note: str

    @property
    def filename(self) -> str:
        return f"{self.supplier_id}_invoice.png"

    @property
    def amount_text(self) -> str:
        whole, cents = divmod(self.amount_minor, 100)
        return f"${whole:,}.{cents:02d}"


# Amounts and identities match data/procureagent/scenario_v1.json exactly. An
# invoice image that disagreed with the canonical record would be caught by the
# verifier's exact-amount check, but it would be a confusing thing to demo.
SPECS = (
    InvoiceSpec(
        supplier_id="prime_foods",
        display_name="PRIME FOODS",
        legal_name="Prime Foods Wholesale Meats",
        address="18 Harbor Provision Way · Boston, MA",
        invoice_number="PF-25031",
        amount_minor=250_000,
        category="meat",
        line_description="Butchered meat delivery",
        invoice_date="2026-08-29",
        due_date="2026-09-02",
        footer_note="Payment unlocks the next scheduled meat delivery.",
    ),
    InvoiceSpec(
        supplier_id="packright",
        display_name="PACKRIGHT",
        legal_name="PackRight Packaging Supply",
        address="7 Kiln Street · Somerville, MA",
        invoice_number="PR-15007",
        amount_minor=150_000,
        category="packaging",
        line_description="Takeout packaging restock",
        invoice_date="2026-08-25",
        due_date="2026-08-29",
        footer_note="Packaging stock is held at the depot; payment does not trigger delivery.",
    ),
    InvoiceSpec(
        supplier_id="cleanpro",
        display_name="CLEANPRO",
        legal_name="CleanPro Facility Services",
        address="240 Ridge Industrial Park · Medford, MA",
        invoice_number="CP-70019",
        amount_minor=70_000,
        category="cleaning",
        line_description="Monthly kitchen deep clean",
        invoice_date="2026-08-24",
        due_date="2026-08-30",
        footer_note="Service contract under review; confirm supplier status before payment.",
    ),
)


def _load_font(kind: str, size: int):
    from PIL import ImageFont

    for path in FONT_CANDIDATES[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit(
        f"no usable {kind} font found; install DejaVu or run on macOS. "
        f"Tried: {', '.join(FONT_CANDIDATES[kind])}"
    )


def render(spec: InvoiceSpec):
    """Draw one invoice at the same geometry as the frozen Fresh Farms asset."""

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)

    sans = _load_font("sans", 22)
    sans_small = _load_font("sans", 18)
    sans_head = _load_font("sans_bold", 21)
    mono = _load_font("mono", 22)
    mono_big = _load_font("mono_bold", 27)
    label = _load_font("sans_bold", 24)
    title = _load_font("sans_bold", 46)
    total_label = _load_font("sans_bold", 27)

    draw.rounded_rectangle((74, 54, 926, 1226), radius=18, fill=CARD, outline=INK, width=3)
    draw.rounded_rectangle((74, 54, 926, 212), radius=18, fill=INK)
    draw.rectangle((74, 180, 926, 212), fill=INK)
    draw.text((118, 96), spec.display_name, font=label, fill=(158, 215, 189))
    draw.text((118, 138), "SUPPLIER INVOICE", font=title, fill=(255, 255, 255))

    draw.text((118, 255), spec.legal_name, font=sans, fill=MUTED)
    draw.text((118, 289), spec.address, font=sans, fill=MUTED)
    draw.text((118, 323), f"Supplier ID: {spec.supplier_id}", font=sans, fill=MUTED)

    # The single anchored invoice-number line. Exactly one may exist.
    draw.rounded_rectangle((522, 264, 870, 348), radius=12, fill=LEAF_SOFT, outline=LEAF, width=2)
    draw.text((546, 296), f"Invoice No: {spec.invoice_number}", font=mono, fill=INK)

    draw.line((118, 411, 882, 411), fill=RULE, width=2)
    draw.text((118, 441), "BILL TO", font=sans_head, fill=INK)
    draw.text((118, 478), "Main Street Bistro", font=sans, fill=(55, 75, 69))
    draw.text((118, 512), "Cambridge, Massachusetts", font=sans, fill=(55, 75, 69))
    # "Date" is a scan boundary, so these lines cannot create a second candidate.
    draw.text((625, 443), f"Invoice Date: {spec.invoice_date}", font=sans_small, fill=MUTED)
    draw.text((625, 480), f"Due Date: {spec.due_date}", font=sans_small, fill=MUTED)
    draw.text((625, 517), "Currency: USD", font=sans_small, fill=MUTED)

    draw.rectangle((118, 598, 882, 656), fill=INK)
    draw.text((140, 618), "DESCRIPTION", font=sans_head, fill=(255, 255, 255))
    draw.text((638, 618), "QTY", font=sans_head, fill=(255, 255, 255))
    draw.text((782, 618), "AMOUNT", font=sans_head, fill=(255, 255, 255))

    draw.text((140, 693), spec.line_description, font=sans, fill=INK)
    draw.text((646, 693), "1", font=sans, fill=INK)
    draw.text((768, 693), spec.amount_text, font=mono, fill=INK)
    draw.line((118, 756, 882, 756), fill=(220, 228, 222), width=2)

    draw.text((630, 804), "Subtotal", font=sans, fill=MUTED)
    draw.text((768, 804), spec.amount_text, font=mono, fill=INK)
    draw.text((630, 849), "Tax", font=sans, fill=MUTED)
    draw.text((817, 849), "$0.00", font=mono, fill=INK)

    draw.rounded_rectangle((598, 905, 882, 983), radius=10, fill=LEAF_SOFT)
    draw.text((621, 930), "TOTAL", font=total_label, fill=INK)
    draw.text((750, 930), spec.amount_text, font=mono_big, fill=INK)

    draw.text((118, 1050), spec.footer_note, font=sans_head, fill=LEAF)
    draw.text(
        (118, 1096),
        "SYNTHETIC HACKATHON FIXTURE · NO REAL FINANCIAL OBLIGATION",
        font=sans_small,
        fill=(102, 122, 115),
    )
    draw.text(
        (118, 1157),
        f"Questions? demo@{spec.supplier_id.replace('_', '')}.example",
        font=sans_small,
        fill=(102, 122, 115),
    )
    return image


def check_anchors(spec: InvoiceSpec) -> list[str]:
    """Static check of the rules that decide whether the gate can see one ID."""

    problems: list[str] = []
    sys.path.insert(0, str(ROOT / "src"))
    from invoiceagent.extraction import is_valid_invoice_identifier

    if not is_valid_invoice_identifier(spec.invoice_number):
        problems.append(f"{spec.invoice_number} is not a valid invoice identifier")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the specs and report what would be written",
    )
    parser.add_argument("--out-dir", default=str(ASSET_DIR))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    failures = []
    for spec in SPECS:
        failures.extend(check_anchors(spec))
    if failures:
        for problem in failures:
            print(f"FAIL: {problem}")
        return 1

    for spec in SPECS:
        target = out_dir / spec.filename
        if args.check:
            print(f"would write {target} ({spec.invoice_number}, {spec.amount_text})")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        render(spec).save(target, format="PNG", optimize=True)
        print(f"wrote {target} ({spec.invoice_number}, {spec.amount_text})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
