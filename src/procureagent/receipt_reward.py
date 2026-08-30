"""RL-ready reward signal for receipt-to-invoice matching.

Ryan's LayoutLMv3 adapter remains a supervised invoice-number classifier.  This
module does not train or update that model.  It converts the existing exact
payment-proof gate into a small, auditable reward that a future contextual
bandit could optimize when deciding whether to accept a match or request human
review.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from .contracts import ContractValidationError
from .receipt import PaymentProofGateResult


class ReceiptMatchAction(str, Enum):
    """Decision taken after receipt parsing and exact proof checks."""

    ACCEPT_MATCH = "ACCEPT_MATCH"
    REQUEST_REVIEW = "REQUEST_REVIEW"


@dataclass(frozen=True, slots=True)
class ReceiptRewardWeights:
    """Declared values; raw proof checks remain visible beside the reward."""

    verified_match: Decimal = Decimal("10.0")
    safe_review: Decimal = Decimal("-1.0")
    unsafe_accept: Decimal = Decimal("-25.0")

    def __post_init__(self) -> None:
        for name in ("verified_match", "safe_review", "unsafe_accept"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ContractValidationError(f"{name} must be a finite Decimal")
        if self.verified_match <= 0:
            raise ContractValidationError("verified_match must be positive")
        if self.safe_review > 0:
            raise ContractValidationError("safe_review cannot be positive")
        if self.unsafe_accept >= self.safe_review:
            raise ContractValidationError(
                "unsafe_accept must be penalized more than safe review"
            )


@dataclass(frozen=True, slots=True)
class ReceiptRewardResult:
    action: ReceiptMatchAction
    reward: Decimal
    outcome: str
    proof_verified: bool
    checks_passed: tuple[str, ...]
    reason_codes: tuple[str, ...]
    trained_policy: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.action, ReceiptMatchAction):
            raise ContractValidationError("receipt reward action is invalid")
        if not isinstance(self.reward, Decimal) or not self.reward.is_finite():
            raise ContractValidationError("receipt reward must be a finite Decimal")
        if not isinstance(self.outcome, str) or not self.outcome:
            raise ContractValidationError("receipt reward outcome is required")
        if not isinstance(self.proof_verified, bool):
            raise ContractValidationError("proof_verified must be true or false")
        if self.trained_policy:
            raise ContractValidationError(
                "this P0 signal cannot claim that a policy was trained"
            )


def score_receipt_match(
    proof_gate: PaymentProofGateResult,
    action: ReceiptMatchAction | str,
    *,
    weights: ReceiptRewardWeights = ReceiptRewardWeights(),
) -> ReceiptRewardResult:
    """Score one receipt-routing decision without mutating model or ledger state."""

    if not isinstance(proof_gate, PaymentProofGateResult):
        raise ContractValidationError("proof_gate must be PaymentProofGateResult")
    try:
        selected = (
            action if isinstance(action, ReceiptMatchAction) else ReceiptMatchAction(action)
        )
    except (TypeError, ValueError) as exc:
        raise ContractValidationError("receipt reward action is invalid") from exc
    if not isinstance(weights, ReceiptRewardWeights):
        raise ContractValidationError("weights must be ReceiptRewardWeights")

    if selected is ReceiptMatchAction.REQUEST_REVIEW:
        reward = weights.safe_review
        outcome = "SAFE_REVIEW"
    elif proof_gate.closes_obligation:
        reward = weights.verified_match
        outcome = "VERIFIED_FULL_MATCH"
    else:
        reward = weights.unsafe_accept
        outcome = "UNSAFE_FALSE_ACCEPT"

    return ReceiptRewardResult(
        action=selected,
        reward=reward,
        outcome=outcome,
        proof_verified=proof_gate.closes_obligation,
        checks_passed=proof_gate.checks_passed,
        reason_codes=proof_gate.reason_codes,
    )


__all__ = [
    "ReceiptMatchAction",
    "ReceiptRewardResult",
    "ReceiptRewardWeights",
    "score_receipt_match",
]
