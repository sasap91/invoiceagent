from dataclasses import FrozenInstanceError, fields, replace
from decimal import Decimal
import json

import pytest

from procureagent import (
    AdversarialDocumentMetadata,
    BoundingBox,
    ContractValidationError,
    DailyRecommendationBatch,
    DocumentIdentityProposal,
    DocumentMethod,
    DocumentStatus,
    InvoiceIdentity,
    InvoiceNumberCandidate,
    InvoicePaymentStatus,
    OperatorDecision,
    OperatorDecisionType,
    PaymentProof,
    PaymentProofSource,
    PaymentProofStatus,
    PolicyType,
    ProcurementAction,
    Recommendation,
    RestaurantState,
    SupplierSource,
    SyntheticSupplierInvoice,
    VerifierDecision,
    VerifierResult,
    load_locked_scenario,
    load_scenario,
    make_audit_id,
    validate_full_payment_proof,
)


@pytest.fixture(scope="module")
def scenario():
    return load_locked_scenario()


def test_locked_fixture_reproduces_cash_obligations_and_identity(scenario):
    state = scenario.initial_state
    assert scenario.scenario_id == "restaurant_demo_v1"
    assert scenario.seed == 138
    assert scenario.horizon_days == 7
    assert state.day == 0
    assert state.state_version == 1
    assert state.restaurant_id == "sugar_and_spice_thai_restaurant"
    assert state.currency == "USD"
    assert state.cash_minor == 500_000
    assert state.total_obligations_minor == 620_000
    assert {
        invoice.identity: invoice.amount_minor for invoice in state.active_invoices
    } == {
        InvoiceIdentity("fresh_farms", "FF-10482"): 150_000,
        InvoiceIdentity("prime_foods", "PF-25031"): 250_000,
        InvoiceIdentity("packright", "PR-15007"): 150_000,
        InvoiceIdentity("cleanpro", "CP-70019"): 70_000,
    }


def test_locked_fixture_preserves_primary_business_context(scenario):
    invoices = {invoice.supplier_id: invoice for invoice in scenario.initial_state.invoices}
    assert (invoices["fresh_farms"].due_in_days, invoices["fresh_farms"].inventory_days_remaining) == (1, 2)
    assert (invoices["prime_foods"].due_in_days, invoices["prime_foods"].inventory_days_remaining) == (3, 3)
    assert (invoices["packright"].due_in_days, invoices["packright"].inventory_days_remaining) == (-1, 20)
    assert (invoices["cleanpro"].due_in_days, invoices["cleanpro"].inventory_days_remaining) == (0, 15)
    assert invoices["cleanpro"].context_conflict_codes == (
        "CONFLICTING_SUPPLIER_STATUS",
    )
    assert all(
        invoice.payment_status is InvoicePaymentStatus.UNPAID
        for invoice in invoices.values()
    )


def test_unknownco_is_fail_closed_and_not_an_obligation(scenario):
    assert len(scenario.adversarial_documents) == 1
    unknown = scenario.adversarial_documents[0]
    assert unknown.supplier_label == "UnknownCo"
    assert unknown.expected_status is DocumentStatus.REVIEW_REQUIRED
    assert unknown.expected_lookup_activated is False
    assert unknown.included_in_obligations is False
    assert len(unknown.candidate_invoice_numbers) == 2
    assert "unknownco" not in {invoice.supplier_id for invoice in scenario.initial_state.invoices}


def test_contract_values_are_deeply_immutable(scenario):
    with pytest.raises(FrozenInstanceError):
        scenario.initial_state.cash_minor = 0
    with pytest.raises(FrozenInstanceError):
        scenario.initial_state.invoices[0].amount_minor = 0
    assert isinstance(scenario.suppliers, tuple)
    assert isinstance(scenario.initial_state.invoices, tuple)
    assert isinstance(scenario.initial_state.invoices[-1].context_conflict_codes, tuple)


@pytest.mark.parametrize("bad_amount", [True, 1.5, Decimal("100.00"), -1, 0])
def test_invoice_amount_requires_positive_integer_minor_units(scenario, bad_amount):
    with pytest.raises(ContractValidationError):
        replace(scenario.initial_state.invoices[0], amount_minor=bad_amount)


def test_restaurant_rejects_duplicate_identity_and_stale_invoice_version(scenario):
    state = scenario.initial_state
    with pytest.raises(ContractValidationError, match="unique identities"):
        replace(state, invoices=(state.invoices[0], state.invoices[0]))
    stale = replace(state.invoices[0], state_version=2)
    with pytest.raises(ContractValidationError, match="versions must match"):
        replace(state, invoices=(stale, *state.invoices[1:]))


def test_document_confirmation_needs_one_grounded_candidate():
    grounded = InvoiceNumberCandidate(
        invoice_number="FF-10482",
        entity_confidence="0.96",
        grounded_in_ocr=True,
        evidence_tokens=("FF-10482",),
        evidence_boxes=(BoundingBox(640, 120, 820, 160),),
    )
    proposal = DocumentIdentityProposal(
        document_id="doc_fresh_farms_10482",
        supplier_id="fresh_farms",
        supplier_source=SupplierSource.OPERATOR_SELECTED,
        supplier_confirmed=True,
        candidate_spans=(grounded,),
        method=DocumentMethod.LAYOUTLMV3_LOCAL,
        model_version="layoutlmv3-invoice-number:test",
        status=DocumentStatus.CONFIRMED,
    )
    assert proposal.confirmed_identity == InvoiceIdentity("fresh_farms", "FF-10482")
    with pytest.raises(ContractValidationError, match="exactly one grounded"):
        replace(proposal, candidate_spans=())
    with pytest.raises(ContractValidationError, match="confirmed supplier"):
        replace(proposal, supplier_confirmed=False)


def test_document_evidence_is_strict_and_model_version_is_required():
    with pytest.raises(ContractValidationError, match="same length"):
        InvoiceNumberCandidate(
            "FF-10482",
            "0.9",
            True,
            ("FF-10482",),
            (),
        )
    with pytest.raises(ContractValidationError, match="Decimal"):
        InvoiceNumberCandidate("FF-10482", 0.9, False, (), ())
    with pytest.raises(ContractValidationError, match="model_version"):
        DocumentIdentityProposal(
            "doc_1",
            "fresh_farms",
            SupplierSource.OPERATOR_SELECTED,
            True,
            (),
            DocumentMethod.LAYOUTLMV3_LOCAL,
        )


def test_payment_proof_closes_only_exact_approved_full_invoice(scenario):
    invoice = replace(
        scenario.initial_state.invoices[0],
        payment_status=InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED,
    )
    proof = scenario.payment_proofs[0]
    validate_full_payment_proof(invoice, proof)

    for changed_proof in (
        replace(proof, supplier_id="prime_foods"),
        replace(proof, invoice_number="FF-99999"),
        replace(proof, amount_minor=149_999),
        replace(proof, currency="CAD"),
        replace(proof, status=PaymentProofStatus.REVIEW_REQUIRED),
    ):
        with pytest.raises(ContractValidationError):
            validate_full_payment_proof(invoice, changed_proof)

    with pytest.raises(ContractValidationError, match="simulated-payment-approved"):
        validate_full_payment_proof(scenario.initial_state.invoices[0], proof)


def test_payment_proof_contract_records_required_provenance(scenario):
    proof = scenario.payment_proofs[0]
    assert proof.receipt_id == "receipt_fresh_farms_10482"
    assert proof.source is PaymentProofSource.SYNTHETIC_FIXTURE_REPLAY
    assert proof.status is PaymentProofStatus.VERIFIED
    assert proof.paid_date.isoformat() == "2026-08-30"
    with pytest.raises(ContractValidationError):
        replace(proof, provenance="")
    with pytest.raises(ContractValidationError):
        replace(proof, paid_date="2026-02-30")


def _recommendation(invoice, action, reason):
    return Recommendation(
        supplier_id=invoice.supplier_id,
        invoice_number=invoice.invoice_number,
        action=action,
        amount_minor=invoice.amount_minor,
        reason_codes=(reason,),
    )


def test_daily_batch_supports_only_pay_defer_verify_and_unique_invoices(scenario):
    invoices = scenario.initial_state.invoices
    recommendations = (
        _recommendation(invoices[0], ProcurementAction.PAY, "STOCKOUT_RISK"),
        _recommendation(invoices[1], ProcurementAction.PAY, "CRITICAL_SUPPLIER"),
        _recommendation(invoices[2], ProcurementAction.DEFER, "BATCH_CASH_PRIORITY"),
        _recommendation(invoices[3], ProcurementAction.VERIFY, "CONFLICTING_SUPPLIER_STATUS"),
    )
    batch = DailyRecommendationBatch(
        batch_id="day-0-criticality-aware-v1",
        state_version=1,
        policy_name="criticality_aware_greedy",
        policy_version="v1",
        policy_type=PolicyType.DETERMINISTIC_RULES,
        recommendations=recommendations,
    )
    assert tuple(item.action for item in batch.recommendations) == (
        ProcurementAction.PAY,
        ProcurementAction.PAY,
        ProcurementAction.DEFER,
        ProcurementAction.VERIFY,
    )
    with pytest.raises(ContractValidationError, match="same invoice"):
        replace(batch, recommendations=(recommendations[0], recommendations[0]))
    with pytest.raises(ContractValidationError):
        replace(recommendations[0], action="TRANSFER")


def test_verifier_decision_cannot_mark_blocked_batch_as_verified():
    blocked = VerifierDecision(
        verification_id="verify-over-budget-1",
        batch_id="batch-1",
        result=VerifierResult.BLOCKED,
        reason_codes=("OVER_BUDGET",),
        checks_passed=("KNOWN_SUPPLIERS",),
    )
    assert blocked.verified_batch_id is None
    with pytest.raises(ContractValidationError, match="cannot verify"):
        replace(blocked, verified_batch_id="batch-1")
    verified = VerifierDecision(
        verification_id="verify-primary-1",
        batch_id="batch-1",
        result=VerifierResult.REQUIRES_OPERATOR,
        reason_codes=("FINANCIAL_ACTION",),
        checks_passed=("CURRENT_STATE", "BATCH_CASH_AVAILABLE"),
        verified_batch_id="batch-1",
    )
    assert verified.result is VerifierResult.REQUIRES_OPERATOR


def test_operator_decisions_preserve_approve_modify_reject_semantics():
    approved = OperatorDecision(
        decision_id="operator-decision-1",
        reviewed_batch_id="batch-1",
        decision=OperatorDecisionType.APPROVE,
        approved_batch_id="batch-1",
    )
    assert approved.approved_batch_id == approved.reviewed_batch_id
    modified = OperatorDecision(
        decision_id="operator-decision-2",
        reviewed_batch_id="batch-1",
        decision=OperatorDecisionType.MODIFY,
        replacement_batch_id="batch-2",
    )
    assert modified.approved_batch_id is None
    rejected = OperatorDecision(
        decision_id="operator-decision-3",
        reviewed_batch_id="batch-1",
        decision=OperatorDecisionType.REJECT,
    )
    assert rejected.approved_batch_id is None
    with pytest.raises(ContractValidationError, match="REJECT"):
        replace(rejected, approved_batch_id="batch-1")


def test_unknown_adversarial_metadata_cannot_activate_lookup():
    with pytest.raises(ContractValidationError, match="cannot activate"):
        AdversarialDocumentMetadata(
            document_id="doc_unknownco",
            supplier_label="UnknownCo",
            candidate_invoice_numbers=("UC-1", "UC-2"),
            expected_status=DocumentStatus.REVIEW_REQUIRED,
            reason_codes=("UNKNOWN_SUPPLIER",),
            expected_lookup_activated=True,
            included_in_obligations=False,
        )


def test_loader_rejects_unknown_schema_fields(tmp_path):
    raw = json.loads(load_locked_scenario_path_text())
    raw["restaurant"]["binary_float_balance"] = 1.1
    path = tmp_path / "unknown-field.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ContractValidationError, match="unexpected"):
        load_scenario(path)


def test_locked_loader_rejects_tampered_cash(tmp_path):
    raw = json.loads(load_locked_scenario_path_text())
    raw["restaurant"]["cash_minor"] = 499_999
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ContractValidationError, match="locked"):
        load_locked_scenario(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("due_in_days", 3650),
        ("inventory_days_remaining", 3650),
        ("delivery_lead_days", 100),
        ("supplier_criticality", "low"),
    ),
)
def test_locked_loader_rejects_tampered_policy_inputs(tmp_path, field, value):
    raw = json.loads(load_locked_scenario_path_text())
    raw["invoices"][0][field] = value
    path = tmp_path / f"tampered-{field}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ContractValidationError, match="locked"):
        load_locked_scenario(path)


def load_locked_scenario_path_text():
    from procureagent import LOCKED_SCENARIO_PATH

    return LOCKED_SCENARIO_PATH.read_text(encoding="utf-8")


def test_audit_id_is_stable_and_rejects_unsafe_values():
    assert make_audit_id("day", 0, "Criticality Aware", "v1") == (
        "day-0-criticality-aware-v1"
    )
    for unsafe in ("../batch", " batch ", "batch/id", ""):
        with pytest.raises(ContractValidationError):
            make_audit_id(unsafe, 1)


def test_c0_contract_has_no_receivable_or_fractional_payment_fields():
    contract_types = (
        SyntheticSupplierInvoice,
        RestaurantState,
        Recommendation,
        DailyRecommendationBatch,
        PaymentProof,
    )
    field_names = {field.name for kind in contract_types for field in fields(kind)}
    forbidden_fragments = ("receivable", "customer", "allocation", "partial")
    assert not any(
        fragment in name
        for name in field_names
        for fragment in forbidden_fragments
    )
