"""Truthful reward semantics for the InvoiceAgent receipt-match demo."""

from dataclasses import replace
from decimal import Decimal

import pytest

from procureagent.contracts import (
    ContractValidationError,
    InvoicePaymentStatus,
    PaymentProofSource,
    PaymentProofStatus,
    load_locked_scenario,
)
from procureagent.receipt import PaymentProofGateResult, build_payment_proof, parse_receipt
from procureagent.receipt_reward import (
    ReceiptMatchAction,
    ReceiptRewardWeights,
    score_receipt_match,
)
from tests.test_procure_receipt import receipt_ocr


def verified_gate() -> PaymentProofGateResult:
    scenario = load_locked_scenario()
    ocr = receipt_ocr()
    parsed = parse_receipt(ocr, known_suppliers=scenario.suppliers)
    invoice = replace(
        scenario.initial_state.invoices[0],
        payment_status=InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED,
    )
    return build_payment_proof(
        parsed,
        invoice,
        ocr=ocr,
        known_suppliers=scenario.suppliers,
        source=PaymentProofSource.SYNTHETIC_FIXTURE_REPLAY,
        provenance="test_fixture",
    )


def review_gate() -> PaymentProofGateResult:
    return PaymentProofGateResult(
        status=PaymentProofStatus.REVIEW_REQUIRED,
        proof=None,
        reason_codes=("FULL_AMOUNT_MISMATCH",),
        checks_passed=("SUPPLIER_MATCH", "INVOICE_MATCH"),
    )


def test_verified_full_match_receives_positive_rl_ready_signal_without_training_claim():
    result = score_receipt_match(verified_gate(), ReceiptMatchAction.ACCEPT_MATCH)

    assert result.reward == Decimal("10.0")
    assert result.outcome == "VERIFIED_FULL_MATCH"
    assert result.proof_verified
    assert result.trained_policy is False
    assert "FULL_AMOUNT_MATCH" in result.checks_passed


def test_safe_review_costs_less_than_an_unsafe_false_accept():
    review = score_receipt_match(review_gate(), ReceiptMatchAction.REQUEST_REVIEW)
    unsafe = score_receipt_match(review_gate(), ReceiptMatchAction.ACCEPT_MATCH)

    assert review.reward == Decimal("-1.0")
    assert review.outcome == "SAFE_REVIEW"
    assert unsafe.reward == Decimal("-25.0")
    assert unsafe.outcome == "UNSAFE_FALSE_ACCEPT"
    assert unsafe.reward < review.reward


def test_requesting_review_never_gets_verified_match_reward():
    result = score_receipt_match(verified_gate(), ReceiptMatchAction.REQUEST_REVIEW)

    assert result.reward == Decimal("-1.0")
    assert result.outcome == "SAFE_REVIEW"
    assert result.proof_verified


@pytest.mark.parametrize(
    "kwargs",
    (
        {"verified_match": Decimal("0")},
        {"safe_review": Decimal("1")},
        {"unsafe_accept": Decimal("0")},
    ),
)
def test_invalid_reward_schedules_are_rejected(kwargs):
    with pytest.raises(ContractValidationError):
        ReceiptRewardWeights(**kwargs)


def test_invalid_action_is_rejected():
    with pytest.raises(ContractValidationError, match="action is invalid"):
        score_receipt_match(review_gate(), "PAY_ANYWAY")
