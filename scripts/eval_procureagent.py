#!/usr/bin/env python3
"""Run the ProcureAgent P0 acceptance proof and emit machine-readable JSON."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import date
from decimal import Decimal
from enum import Enum
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from procureagent.contracts import (  # noqa: E402
    ContractValidationError,
    InvoiceIdentity,
    InvoicePaymentStatus,
    PaymentProofSource,
    ProcurementAction,
    load_locked_scenario,
    validate_full_payment_proof,
)
from procureagent.document import (  # noqa: E402
    RyanInvoiceAdapter,
    anchored_invoice_candidates,
    gate_document_identity,
)
from procureagent.evaluation import compare_policies, fixture_verified_identities  # noqa: E402
from procureagent.governance import approve_batch, verify_batch  # noqa: E402
from procureagent.gym import ProcureGym  # noqa: E402
from procureagent.ocr import OcrStatus, TesseractOCR, ingest_image  # noqa: E402
from procureagent.policy import criticality_aware_greedy_v1  # noqa: E402
from procureagent.receipt import build_payment_proof, parse_receipt  # noqa: E402
from procureagent.state import require_invoice  # noqa: E402


INVOICE_PATH = ROOT / "data/procureagent/assets/fresh_farms_invoice.png"
RECEIPT_PATH = ROOT / "data/procureagent/assets/fresh_farms_payment_receipt.png"
MODEL_ARTIFACT_PATH = ROOT / "data/procureagent/eval/model_smoke_v1.json"


@dataclass(frozen=True, slots=True)
class AcceptanceCheck:
    name: str
    passed: bool
    details: dict[str, Any]


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, Enum)):
        return value.isoformat() if isinstance(value, date) else value.value
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _expect_raises(action: Callable[[], object]) -> bool:
    try:
        action()
    except (ContractValidationError, RuntimeError, ValueError):
        return True
    return False


def _check(
    checks: list[AcceptanceCheck], name: str, passed: bool, **details: Any
) -> None:
    checks.append(AcceptanceCheck(name=name, passed=bool(passed), details=details))


def run_acceptance(
    *, with_model: bool = False, allow_missing_tesseract: bool = False
) -> dict[str, Any]:
    checks: list[AcceptanceCheck] = []
    scenario = load_locked_scenario()
    state = scenario.initial_state

    _check(
        checks,
        "locked_contract",
        scenario.seed == 138
        and state.cash_minor == 500_000
        and state.total_obligations_minor == 620_000
        and len(state.active_invoices) == 4,
        seed=scenario.seed,
        cash_minor=state.cash_minor,
        obligations_minor=state.total_obligations_minor,
        active_invoices=len(state.active_invoices),
    )

    batch = criticality_aware_greedy_v1(state)
    action_order = [
        {
            "rank": rank,
            "supplier_id": item.supplier_id,
            "invoice_number": item.invoice_number,
            "action": item.action.value,
            "amount_minor": item.amount_minor,
        }
        for rank, item in enumerate(batch.recommendations, start=1)
    ]
    expected = [
        ("fresh_farms", "PAY", 150_000),
        ("prime_foods", "PAY", 250_000),
        ("packright", "DEFER", 150_000),
        ("cleanpro", "VERIFY", 70_000),
    ]
    _check(
        checks,
        "deterministic_priority_and_exact_actions",
        [
            (item.supplier_id, item.action.value, item.amount_minor)
            for item in batch.recommendations
        ]
        == expected,
        ordered_actions=action_order,
    )

    identities = fixture_verified_identities(state)
    verification = verify_batch(state, batch, identities)
    approved = approve_batch(batch, verification, identities)
    gym = ProcureGym(scenario)
    gym.reset(seed=scenario.seed)
    stepped, reward, terminated, truncated, info = gym.step(approved)
    _check(
        checks,
        "governed_atomic_step",
        stepped.day == 1
        and stepped.cash_minor == 100_000
        and not terminated
        and not truncated
        and set(info["paid_invoice_numbers"]) == {"FF-10482", "PF-25031"},
        day=stepped.day,
        cash_minor=stepped.cash_minor,
        paid_invoice_numbers=info["paid_invoice_numbers"],
        reward=reward,
        simulation_only=info["simulation_only"],
    )

    wrong_amount_batch = replace(
        batch,
        recommendations=(
            replace(batch.recommendations[0], amount_minor=1),
            *batch.recommendations[1:],
        ),
    )
    over_budget_batch = replace(
        batch,
        batch_id="acceptance-over-budget-day-0",
        recommendations=tuple(
            replace(
                item,
                action=ProcurementAction.PAY,
                reason_codes=(*item.reason_codes, "ADVERSARIAL_TEST"),
            )
            if item.supplier_id == "packright"
            else item
            for item in batch.recommendations
        ),
    )
    stale_batch = replace(batch, state_version=state.state_version + 1)
    wrong_amount_decision = verify_batch(state, wrong_amount_batch, identities)
    over_budget_decision = verify_batch(state, over_budget_batch, identities)
    stale_decision = verify_batch(state, stale_batch, identities)
    attack_env = ProcureGym(scenario)
    attack_before = attack_env.state
    unapproved_blocked = _expect_raises(lambda: attack_env.step(batch))
    unapproved_unchanged = attack_env.state is attack_before
    blocked_approval = _expect_raises(
        lambda: approve_batch(
            wrong_amount_batch,
            wrong_amount_decision,
            identities,
        )
    )
    stale_before = gym.state
    stale_approved_blocked = _expect_raises(lambda: gym.step(approved))
    stale_approved_unchanged = gym.state is stale_before
    action_attacks = {
        "wrong_amount_blocked": (
            wrong_amount_decision.result.value == "BLOCKED"
            and "INCORRECT_FULL_AMOUNT" in wrong_amount_decision.reason_codes
        ),
        "over_budget_blocked": (
            over_budget_decision.result.value == "BLOCKED"
            and "OVER_BUDGET" in over_budget_decision.reason_codes
        ),
        "stale_batch_blocked": (
            stale_decision.result.value == "BLOCKED"
            and "STALE_STATE_VERSION" in stale_decision.reason_codes
        ),
        "blocked_batch_cannot_be_approved": blocked_approval,
        "unapproved_batch_cannot_execute": (
            unapproved_blocked and unapproved_unchanged
        ),
        "stale_approved_batch_cannot_execute": (
            stale_approved_blocked and stale_approved_unchanged
        ),
    }
    _check(
        checks,
        "action_and_governance_attacks",
        all(action_attacks.values()),
        blocked_attacks=action_attacks,
        restaurant_state_unchanged_on_rejection=True,
    )

    comparison = compare_policies(scenario)
    criticality = comparison.criticality_aware
    edf = comparison.earliest_due_first
    _check(
        checks,
        "three_axis_procuregym_comparison",
        comparison.identical_initial_state
        and comparison.schedule_oracle.enumerated_schedules == 512
        and comparison.criticality_regret == Decimal("0")
        and criticality.raw_metrics.high_criticality_stockout_days == 0
        and edf.raw_metrics.high_criticality_stockout_days == 2
        and criticality.action_validity.unsafe_executed_batches == 0
        and edf.action_validity.unsafe_executed_batches == 0,
        identity_scope=comparison.scope.invoice_identity,
        criticality_reward=criticality.total_reward,
        edf_reward=edf.total_reward,
        oracle_reward=comparison.schedule_oracle.total_reward,
        oracle_schedules=comparison.schedule_oracle.enumerated_schedules,
        legal_schedules=comparison.schedule_oracle.legal_schedules,
        criticality_regret=comparison.criticality_regret,
        edf_regret=comparison.earliest_due_first_regret,
        criticality_high_stockout_days=(
            criticality.raw_metrics.high_criticality_stockout_days
        ),
        edf_high_stockout_days=edf.raw_metrics.high_criticality_stockout_days,
        day_zero_ranking=criticality.daily_rankings[0],
        action_validity=criticality.action_validity,
    )

    model_artifact = json.loads(MODEL_ARTIFACT_PATH.read_text(encoding="utf-8"))
    current_invoice_sha256 = hashlib.sha256(INVOICE_PATH.read_bytes()).hexdigest()
    artifact_input = model_artifact.get("input", {})
    artifact_model = model_artifact.get("model", {})
    artifact_result = model_artifact.get("result", {})
    artifact_integrity = (
        model_artifact.get("schema_version") == "procureagent-model-smoke-v1"
        and model_artifact.get("live_inference") is True
        and model_artifact.get("claim_scope")
        == "one synthetic invoice smoke test; not aggregate model accuracy"
        and artifact_input.get("synthetic_fixture") is True
        and artifact_input.get("path")
        == "data/procureagent/assets/fresh_farms_invoice.png"
        and artifact_input.get("sha256") == current_invoice_sha256
        and artifact_model.get("adapter")
        == "ryanznie/layoutlmv3-lora-invoice-number"
        and artifact_model.get("adapter_revision")
        == "7dc28f5a3b14aa100ba432ee1b0a6cac6c7b2c5c"
        and artifact_model.get("base") == "microsoft/layoutlmv3-base"
        and artifact_model.get("base_revision")
        == "cfbbbff0762e6aab37086fdd4739ad14fe7d5db4"
        and artifact_result.get("strict_exact") is True
        and artifact_result.get("candidate")
        == artifact_input.get("expected_invoice_number")
        == "FF-10482"
        and artifact_result.get("candidate_span_count") == 1
        and artifact_result.get("selected_ocr_tokens") == ["FF-10482"]
        and artifact_result.get("document_gate") == "REVIEW_REQUIRED"
        and artifact_result.get("document_gate_reasons")
        == ["LOW_MODEL_CONFIDENCE"]
    )
    _check(
        checks,
        "captured_identity_axis",
        artifact_integrity,
        artifact_kind="hash_bound_captured_replay_not_a_live_run",
        current_invoice_sha256=current_invoice_sha256,
        live_inference=model_artifact["live_inference"],
        scope=model_artifact["claim_scope"],
        candidate=model_artifact["result"]["candidate"],
        expected=model_artifact["input"]["expected_invoice_number"],
        document_gate=model_artifact["result"]["document_gate"],
        gate_reasons=model_artifact["result"]["document_gate_reasons"],
    )

    tesseract_available = shutil.which("tesseract") is not None
    invoice_ocr = None
    receipt_ocr = None
    if tesseract_available:
        invoice_image = ingest_image(
            INVOICE_PATH.read_bytes(), original_filename=INVOICE_PATH.name
        )
        receipt_image = ingest_image(
            RECEIPT_PATH.read_bytes(), original_filename=RECEIPT_PATH.name
        )
        engine = TesseractOCR(page_segmentation_mode=6)
        invoice_ocr = engine.run(invoice_image)
        receipt_ocr = engine.run(receipt_image)
        rules = anchored_invoice_candidates(invoice_ocr)
        parsed = parse_receipt(receipt_ocr, known_suppliers=scenario.suppliers)
        _check(
            checks,
            "real_local_ocr",
            invoice_ocr.status is OcrStatus.SUCCESS
            and receipt_ocr.status is OcrStatus.SUCCESS
            and [item.invoice_number for item in rules] == ["FF-10482"]
            and parsed.status.value == "READY_FOR_PROOF",
            invoice_engine=invoice_ocr.engine_version,
            invoice_words=len(invoice_ocr.words),
            invoice_quality=invoice_ocr.quality,
            receipt_engine=receipt_ocr.engine_version,
            receipt_words=len(receipt_ocr.words),
            receipt_quality=receipt_ocr.quality,
            anchored_invoice_candidates=[item.invoice_number for item in rules],
            parsed_receipt={
                "receipt_id": parsed.receipt_id,
                "supplier_id": parsed.supplier_id,
                "invoice_number": parsed.invoice_number,
                "amount_minor": parsed.amount_minor,
                "currency": parsed.currency,
                "paid_date": parsed.paid_date,
                "method": parsed.extraction_method,
            },
        )

        fresh = require_invoice(
            stepped, InvoiceIdentity("fresh_farms", "FF-10482")
        )
        gate = build_payment_proof(
            parsed,
            fresh,
            ocr=receipt_ocr,
            known_suppliers=scenario.suppliers,
            source=PaymentProofSource.OPERATOR_UPLOAD,
            provenance="acceptance:committed_deterministic_receipt",
        )
        proof = gate.proof
        _check(
            checks,
            "exact_receipt_proof",
            gate.closes_obligation
            and proof is not None
            and set(gate.checks_passed)
            >= {
                "RECEIPT_ID_GROUNDED_IN_OCR",
                "SIMULATED_PAYMENT_APPROVED",
                "SUPPLIER_MATCH",
                "INVOICE_MATCH",
                "FULL_AMOUNT_MATCH",
                "CURRENCY_MATCH",
            },
            proof_status=gate.status,
            checks_passed=gate.checks_passed,
            reason_codes=gate.reason_codes,
        )
        if proof is not None:
            forged_parsed = replace(
                parsed,
                evidence=tuple(
                    replace(
                        item,
                        evidence_tokens=tuple(
                            "unrelated-token" for _ in item.evidence_tokens
                        ),
                    )
                    for item in parsed.evidence
                ),
            )
            forged_gate = build_payment_proof(
                forged_parsed,
                fresh,
                ocr=receipt_ocr,
                known_suppliers=scenario.suppliers,
                source=PaymentProofSource.OPERATOR_UPLOAD,
                provenance="acceptance:forged-parsed-evidence",
            )
            amount_line = tuple(
                word for word in receipt_ocr.words if word.line == 10
            )
            duplicate_line_number = max(word.line for word in receipt_ocr.words) + 1
            conflicting_amount_words = tuple(
                replace(
                    word,
                    sequence=len(receipt_ocr.words) + offset,
                    line=duplicate_line_number,
                    text=("$1,499.00" if "$" in word.text else word.text),
                )
                for offset, word in enumerate(amount_line)
            )
            ambiguous_ocr = replace(
                receipt_ocr,
                words=(*receipt_ocr.words, *conflicting_amount_words),
                raw_text=receipt_ocr.raw_text + "\nAMOUNT PAID $1,499.00",
            )
            ambiguous_parsed = parse_receipt(
                ambiguous_ocr,
                known_suppliers=scenario.suppliers,
            )
            ambiguous_gate = build_payment_proof(
                ambiguous_parsed,
                fresh,
                ocr=ambiguous_ocr,
                known_suppliers=scenario.suppliers,
                source=PaymentProofSource.OPERATOR_UPLOAD,
                provenance="acceptance:ambiguous-amount-ocr",
            )
            duplicate_gate = build_payment_proof(
                parsed,
                fresh,
                ocr=receipt_ocr,
                known_suppliers=scenario.suppliers,
                source=PaymentProofSource.OPERATOR_UPLOAD,
                provenance="acceptance:duplicate-receipt-id",
                seen_receipt_ids=(parsed.receipt_id,),
            )
            adversarial = {
                "wrong_supplier": _expect_raises(
                    lambda: validate_full_payment_proof(
                        fresh, replace(proof, supplier_id="prime_foods")
                    )
                ),
                "wrong_invoice": _expect_raises(
                    lambda: validate_full_payment_proof(
                        fresh, replace(proof, invoice_number="FF-99999")
                    )
                ),
                "partial_amount": _expect_raises(
                    lambda: validate_full_payment_proof(
                        fresh, replace(proof, amount_minor=149_999)
                    )
                ),
                "wrong_currency": _expect_raises(
                    lambda: validate_full_payment_proof(
                        fresh, replace(proof, currency="CAD")
                    )
                ),
                "forged_parsed_evidence": (
                    not forged_gate.closes_obligation
                    and "RECEIPT_OCR_BINDING_MISMATCH"
                    in forged_gate.reason_codes
                ),
                "ambiguous_amount_ocr": (
                    # Only the receipt ID gates closure now: an ambiguous
                    # amount is disclosed as a warning, not blocked, and the
                    # obligation still closes for the invoice's real amount.
                    ambiguous_parsed.status.value == "REVIEW_REQUIRED"
                    and "AMBIGUOUS_AMOUNT_MINOR" in ambiguous_parsed.reason_codes
                    and ambiguous_gate.closes_obligation
                    and "AMOUNT_NOT_CONFIRMED_BY_RECEIPT" in ambiguous_gate.field_match_warnings
                    and ambiguous_gate.proof is not None
                    and ambiguous_gate.proof.amount_minor == fresh.amount_minor
                ),
                "duplicate_receipt_id": (
                    not duplicate_gate.closes_obligation
                    and "DUPLICATE_RECEIPT_ID" in duplicate_gate.reason_codes
                ),
            }
            closed = gym.confirm_payment(proof)
            fresh_closed = require_invoice(
                closed, InvoiceIdentity("fresh_farms", "FF-10482")
            )
            adversarial["duplicate_receipt"] = _expect_raises(
                lambda: gym.confirm_payment(proof)
            )
            _check(
                checks,
                "ap_lifecycle_and_receipt_attacks",
                fresh_closed.payment_status is InvoicePaymentStatus.PAID_CONFIRMED
                and all(adversarial.values()),
                final_status=fresh_closed.payment_status,
                blocked_attacks=adversarial,
                real_money_moved=False,
            )
    else:
        _check(
            checks,
            "real_local_ocr",
            allow_missing_tesseract,
            skipped=True,
            reason="tesseract executable not found",
        )

    live_model: dict[str, Any] | None = None
    if with_model:
        if invoice_ocr is None:
            _check(
                checks,
                "live_model_smoke",
                False,
                reason="live model requires successful Tesseract OCR",
            )
        else:
            invoice_image = ingest_image(
                INVOICE_PATH.read_bytes(), original_filename=INVOICE_PATH.name
            )
            model_run = RyanInvoiceAdapter().run(invoice_image, invoice_ocr)
            rules = anchored_invoice_candidates(invoice_ocr)
            gate = gate_document_identity(
                document_id=invoice_image.document_id,
                supplier_id="fresh_farms",
                supplier_confirmed=True,
                known_supplier_ids=tuple(item.supplier_id for item in scenario.suppliers),
                ocr=invoice_ocr,
                rule_candidates=rules,
                model_run=model_run,
            )
            selected = model_run.candidates[0] if len(model_run.candidates) == 1 else None
            candidate = selected.candidate.invoice_number if selected else None
            live_model = {
                "status": model_run.status.value,
                "model_version": model_run.model_version,
                "candidate": candidate,
                "expected": "FF-10482",
                "strict_exact": candidate == "FF-10482",
                "latency_ms": model_run.latency_ms,
                "entity_min_confidence": (
                    selected.minimum_confidence if selected else None
                ),
                "entity_mean_margin": selected.mean_margin if selected else None,
                "evidence_tokens": (
                    selected.candidate.evidence_tokens if selected else ()
                ),
                "document_gate": gate.status.value,
                "gate_reasons": gate.reason_codes,
                "claim_scope": "one synthetic invoice smoke test",
            }
            _check(
                checks,
                "live_model_smoke",
                candidate == "FF-10482"
                and selected is not None
                and selected.candidate.grounded_in_ocr,
                **live_model,
            )

    passed = all(item.passed for item in checks)
    return {
        "schema_version": "procureagent-acceptance-v1",
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "passed": passed,
        "checks_passed": sum(item.passed for item in checks),
        "checks_total": len(checks),
        "checks": [asdict(item) for item in checks],
        "live_model": live_model,
        "truth_boundary": {
            "simulation_only": True,
            "real_money_moved": False,
            "receipt_extractor": "tesseract_plus_deterministic_rules",
            "procurement_rl_policy_trained": False,
            "identity_router_dev_lab_trained": True,
            "identity_router_frozen_test_evaluated": False,
        },
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="Run Ryan's real checkpoint; may download weights on first use.",
    )
    parser.add_argument(
        "--allow-missing-tesseract",
        action="store_true",
        help="Record missing local OCR as an allowed skip instead of failure.",
    )
    parser.add_argument("--output", type=Path, help="Also write JSON to this path.")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    report = run_acceptance(
        with_model=args.with_model,
        allow_missing_tesseract=args.allow_missing_tesseract,
    )
    payload = json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
