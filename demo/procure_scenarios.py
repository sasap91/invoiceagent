"""Dependency-light display fallbacks for the InvoiceAgent demo UI.

The live app prefers the frozen ``procureagent`` contracts.  These mappings
keep the overview renderable if an installation is incomplete; they never
stand in for OCR, model inference, verification, or simulation output.
"""

from __future__ import annotations

from typing import Any, Final, Mapping


FIXTURE_NOTICE: Final = (
    "SYNTHETIC DEMO · NO AFFILIATION WITH SUGAR & SPICE THAI RESTAURANT. "
    "Locked-fixture overview — deterministic policy, verifier, and benchmark may run locally; "
    "OCR and the local model run only after an explicit /eval click. No procurement RL "
    "policy was trained; C6 is a separate development-only identity-router lab."
)

PRIMARY_SCENARIO: Final[dict[str, Any]] = {
    "scenario_id": "restaurant_demo_v1",
    "restaurant_name": "Sugar & Spice Thai Restaurant",
    "restaurant_context": "Cambridge, Massachusetts",
    "restaurant_disclosure": (
        "SYNTHETIC DEMO · NO AFFILIATION. The named restaurant did not provide "
        "the documents or financial data."
    ),
    "currency": "USD",
    "cash_minor": 500_000,
    "obligations_minor": 620_000,
    "funding_gap_minor": 120_000,
    "state_version": 1,
    "seed": 138,
    "day": 0,
    "source": "C0 synthetic presentation fallback fixture",
}


INVOICE_FIXTURES: Final[tuple[dict[str, Any], ...]] = (
    {
        "supplier_id": "fresh_farms",
        "supplier_name": "Fresh Farms",
        "invoice_number": "FF-10482",
        "category": "Produce",
        "amount_minor": 150_000,
        "due_in_days": 1,
        "due_label": "Tomorrow",
        "inventory_days_remaining": 2,
        "delivery_lead_days": 1,
        "supplier_criticality": "High",
        "supplier_status": "Active",
        "payment_unlocks_delivery": True,
        "record_source": "synthetic_fixture_lookup",
        "document_path": "Fixture/replay",
        "hypothesis_action": "PAY",
        "hypothesis_reasons": (
            "STOCKOUT_RISK",
            "CRITICAL_SUPPLIER",
            "DUE_SOON",
        ),
    },
    {
        "supplier_id": "prime_foods",
        "supplier_name": "Prime Foods",
        "invoice_number": "PF-25031",
        "category": "Meat",
        "amount_minor": 250_000,
        "due_in_days": 3,
        "due_label": "In 3 days",
        "inventory_days_remaining": 3,
        "delivery_lead_days": 2,
        "supplier_criticality": "High",
        "supplier_status": "Active",
        "payment_unlocks_delivery": True,
        "record_source": "synthetic_fixture_lookup",
        "document_path": "Fixture/replay",
        "hypothesis_action": "PAY",
        "hypothesis_reasons": (
            "STOCKOUT_RISK",
            "CRITICAL_SUPPLIER",
        ),
    },
    {
        "supplier_id": "packright",
        "supplier_name": "PackRight",
        "invoice_number": "PR-15007",
        "category": "Packaging",
        "amount_minor": 150_000,
        "due_in_days": -1,
        "due_label": "1 day overdue",
        "inventory_days_remaining": 20,
        "delivery_lead_days": 3,
        "supplier_criticality": "Low",
        "supplier_status": "Active",
        "payment_unlocks_delivery": False,
        "record_source": "synthetic_fixture_lookup",
        "document_path": "English fixture",
        "hypothesis_action": "DEFER",
        "hypothesis_reasons": (
            "LOW_INVENTORY_RISK",
            "BATCH_CASH_PRIORITY",
        ),
    },
    {
        "supplier_id": "cleanpro",
        "supplier_name": "CleanPro",
        "invoice_number": "CP-70019",
        "category": "Cleaning",
        "amount_minor": 70_000,
        "due_in_days": 0,
        "due_label": "Today",
        "inventory_days_remaining": 15,
        "delivery_lead_days": 1,
        "supplier_criticality": "Medium",
        "supplier_status": "Active · conflicting context field",
        "payment_unlocks_delivery": False,
        "record_source": "synthetic_fixture_lookup",
        "document_path": "Verified-ID fixture; context conflict",
        "hypothesis_action": "VERIFY",
        "hypothesis_reasons": ("CONFLICTING_SUPPLIER_STATUS",),
    },
)


DOCUMENT_EVIDENCE: Final[dict[str, dict[str, Any]]] = {
    invoice["supplier_id"]: {
        "supplier_id": invoice["supplier_id"],
        "supplier_name": invoice["supplier_name"],
        "document_id": f"doc_{invoice['supplier_id']}_{invoice['invoice_number'].lower()}",
        "proposed_invoice_number": invoice["invoice_number"],
        "evidence_tokens": (invoice["invoice_number"],),
        "ocr_status": "NOT RUN",
        "model_status": "NOT RUN",
        "document_gate_status": "NOT RUN IN STATIC OVERVIEW — use /eval",
        "identity_basis": "C0 downstream presentation fixture",
        "lookup_status": "Static fixture record displayed; /eval requires human review first",
        "disclosure": (
            "Stored presentation evidence only. This is not proof of OCR, "
            "LayoutLMv3 inference, or document-gate verification."
        ),
    }
    for invoice in INVOICE_FIXTURES
}


UNKNOWNCO_ADVERSARIAL: Final[dict[str, Any]] = {
    "supplier_id": "unknownco",
    "supplier_name": "UnknownCo",
    "document_id": "doc_unknownco_ambiguous",
    "proposed_invoice_number": "Ambiguous: UC-88102 / UC-88108",
    "evidence_tokens": ("UC-88102", "UC-88108"),
    "ocr_status": "NOT RUN",
    "model_status": "NOT RUN",
    "document_gate_status": "EXPECTED REVIEW_REQUIRED — C2 not run",
    "identity_basis": "Adversarial presentation fixture",
    "lookup_status": "BLOCKED BY DESIGN — no canonical lookup",
    "lookup_permitted": False,
    "included_in_obligations": False,
    "disclosure": (
        "Expected fail-closed example, not an executed C2 result. UnknownCo has "
        "ambiguous identity, activates no payable, and is excluded from $6,200."
    ),
}


CATEGORY_STATUS: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "C0",
        "category": "Pivot contract and locked fixtures",
        "owners": "Wilson / @skylarwooster",
        "delivery": "VERIFIED · HASH-LOCKED FIXTURE",
    },
    {
        "id": "C1",
        "category": "OCR and document ingestion",
        "owners": "David / @cheezburgerz + Ryan Nie + Dillon",
        "delivery": "INTEGRATION PASS · OWNER SIGN-OFF PENDING",
    },
    {
        "id": "C2",
        "category": "Local specialist, evidence, and document gate",
        "owners": "Ryan Nie + Dillon",
        "delivery": "INTEGRATION PASS · OWNER SIGN-OFF + ADAPTER LICENSE CLARIFICATION PENDING",
    },
    {
        "id": "C3",
        "category": "Supplier lookup and restaurant state",
        "owners": "Wilson / @skylarwooster",
        "delivery": "VERIFIED",
    },
    {
        "id": "C4",
        "category": "Recommendation, verifier, and governance",
        "owners": "Ryan Nie + Dillon",
        "delivery": "INTEGRATION PASS · OWNER SIGN-OFF PENDING",
    },
    {
        "id": "C5",
        "category": "ProcureGym, reward, and baselines",
        "owners": "Sasa P + Wilson / @skylarwooster",
        "delivery": "INTEGRATION PASS · SASA SIGN-OFF PENDING",
    },
    {
        "id": "C6",
        "category": "Contextual-bandit Router Lab",
        "owners": "David / @cheezburgerz + Ryan Nie + Dillon",
        "delivery": "DEV LAB IMPLEMENTED · NOT VERIFIED · NO FROZEN TEST",
    },
    {
        "id": "C7",
        "category": "Demo UI, orchestration, and deployment",
        "owners": "Wilson / @skylarwooster + Sasa P + Ryan Nie + Dillon + David / @cheezburgerz",
        "delivery": "INTEGRATION + PUBLIC QUICK-TUNNEL REHEARSAL PASS · PERMANENT DEPLOY/OWNER SIGN-OFF PENDING",
    },
    {
        "id": "C8",
        "category": "Evaluation, QA, and presentation proof",
        "owners": "Sasa P",
        "delivery": "9/9 OFFLINE + 10/10 LIVE + 192 PASSED/1 SKIPPED · SASA SIGN-OFF PENDING",
    },
)


def format_minor(amount_minor: int, currency: str = "USD") -> str:
    """Format integer minor units for display without financial float math."""

    if isinstance(amount_minor, bool) or not isinstance(amount_minor, int):
        raise TypeError("amount_minor must be an integer")
    sign = "-" if amount_minor < 0 else ""
    major, minor = divmod(abs(amount_minor), 100)
    if currency != "USD":
        return f"{sign}{currency} {major:,}.{minor:02d}"
    return f"{sign}${major:,}.{minor:02d}"


def invoice_by_supplier(supplier_id: str) -> Mapping[str, Any]:
    """Return a locked presentation invoice by supplier identifier."""

    for invoice in INVOICE_FIXTURES:
        if invoice["supplier_id"] == supplier_id:
            return invoice
    raise KeyError(supplier_id)


def validate_presentation_fixture() -> None:
    """Assert the C7 fallback still matches the PRD's locked headline values."""

    total = sum(invoice["amount_minor"] for invoice in INVOICE_FIXTURES)
    if len(INVOICE_FIXTURES) != 4:
        raise ValueError("restaurant_demo_v1 must contain exactly four invoices")
    if total != PRIMARY_SCENARIO["obligations_minor"]:
        raise ValueError("invoice total must match scenario obligations")
    if PRIMARY_SCENARIO["cash_minor"] != 500_000 or total != 620_000:
        raise ValueError("restaurant_demo_v1 must preserve the $5,000/$6,200 contract")
    if UNKNOWNCO_ADVERSARIAL["included_in_obligations"]:
        raise ValueError("UnknownCo must remain outside canonical obligations")


validate_presentation_fixture()
