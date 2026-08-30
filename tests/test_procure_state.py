from dataclasses import FrozenInstanceError, replace

import pytest

from procureagent.contracts import (
    DocumentStatus,
    InvoiceIdentity,
    InvoicePaymentStatus,
    PaymentProofStatus,
    VerifiedInvoiceIdentity,
    load_locked_scenario,
)
from procureagent.state import (
    StateTransitionError,
    close_invoice_with_payment_proof,
    invoice_index,
    lookup_invoice,
    lookup_verified_invoice,
)


@pytest.fixture(scope="module")
def scenario():
    return load_locked_scenario()


def _verified(supplier_id="fresh_farms", invoice_number="FF-10482"):
    return VerifiedInvoiceIdentity(
        document_id=f"doc-{supplier_id}-{invoice_number}",
        supplier_id=supplier_id,
        invoice_number=invoice_number,
        status=DocumentStatus.CONFIRMED,
    )


def _approved_fresh_state(scenario):
    state = scenario.initial_state
    invoices = tuple(
        replace(
            invoice,
            payment_status=InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED,
        )
        if invoice.identity == InvoiceIdentity("fresh_farms", "FF-10482")
        else invoice
        for invoice in state.invoices
    )
    return replace(state, invoices=invoices)


def test_lookup_uses_exact_composite_key_and_has_no_side_effect(scenario):
    state = scenario.initial_state
    before = invoice_index(state)
    assert lookup_invoice(state, InvoiceIdentity("fresh_farms", "FF-10482")) is before[
        InvoiceIdentity("fresh_farms", "FF-10482")
    ]
    assert lookup_invoice(state, InvoiceIdentity("prime_foods", "FF-10482")) is None
    assert lookup_invoice(state, InvoiceIdentity("fresh_farms", "PF-25031")) is None
    assert invoice_index(state) == before
    assert state.total_obligations_minor == 620_000


def test_verified_lookup_rejects_raw_or_unknown_identity(scenario):
    state = scenario.initial_state
    assert lookup_verified_invoice(state, _verified()).invoice_number == "FF-10482"
    assert lookup_verified_invoice(
        state, _verified("fresh_farms", "FF-99999")
    ) is None
    with pytest.raises(StateTransitionError, match="VerifiedInvoiceIdentity"):
        lookup_verified_invoice(
            state, InvoiceIdentity("fresh_farms", "FF-10482")  # type: ignore[arg-type]
        )


def test_receipt_closure_is_versioned_immutable_and_exact(scenario):
    approved = _approved_fresh_state(scenario)
    proof = scenario.payment_proofs[0]
    closed = close_invoice_with_payment_proof(approved, proof)

    original = invoice_index(approved)[proof.identity]
    updated = invoice_index(closed)[proof.identity]
    assert original.payment_status is InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED
    assert updated.payment_status is InvoicePaymentStatus.PAID_CONFIRMED
    assert closed.state_version == approved.state_version + 1
    assert {item.state_version for item in closed.invoices} == {closed.state_version}
    assert closed.cash_minor == approved.cash_minor
    assert closed.day == approved.day
    with pytest.raises(FrozenInstanceError):
        updated.payment_status = InvoicePaymentStatus.UNPAID


@pytest.mark.parametrize(
    "proof_change",
    [
        {"supplier_id": "prime_foods"},
        {"invoice_number": "FF-99999"},
        {"amount_minor": 149_999},
        {"currency": "CAD"},
        {"status": PaymentProofStatus.REVIEW_REQUIRED},
    ],
)
def test_receipt_mismatch_fails_closed_without_mutation(scenario, proof_change):
    approved = _approved_fresh_state(scenario)
    changed = replace(scenario.payment_proofs[0], **proof_change)
    with pytest.raises(Exception):
        close_invoice_with_payment_proof(approved, changed)
    assert approved == _approved_fresh_state(scenario)


def test_receipt_cannot_close_unpaid_or_be_reused(scenario):
    proof = scenario.payment_proofs[0]
    with pytest.raises(Exception, match="simulated-payment-approved"):
        close_invoice_with_payment_proof(scenario.initial_state, proof)
    approved = _approved_fresh_state(scenario)
    with pytest.raises(StateTransitionError, match="already consumed"):
        close_invoice_with_payment_proof(
            approved, proof, consumed_receipt_ids=(proof.receipt_id,)
        )
    with pytest.raises(StateTransitionError, match="cannot contain duplicates"):
        close_invoice_with_payment_proof(
            approved,
            proof,
            consumed_receipt_ids=("receipt-other", "receipt-other"),
        )
