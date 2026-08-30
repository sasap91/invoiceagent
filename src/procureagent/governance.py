"""Fail-closed batch verification and explicit operator governance."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from .contracts import (
    ContractValidationError,
    DailyRecommendationBatch,
    InvoiceIdentity,
    InvoicePaymentStatus,
    OperatorDecision,
    OperatorDecisionType,
    ProcurementAction,
    Recommendation,
    RestaurantState,
    SupplierStatus,
    VerifiedInvoiceIdentity,
    VerifierDecision,
    VerifierResult,
    make_audit_id,
)
from .state import invoice_index


class GovernanceError(ContractValidationError):
    """Raised when governance inputs are invalid or approval is absent."""


def _verified_tuple(
    values: Iterable[VerifiedInvoiceIdentity],
) -> tuple[VerifiedInvoiceIdentity, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise GovernanceError("verified identities must be a sequence")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise GovernanceError("verified identities must be a sequence") from exc
    if any(not isinstance(item, VerifiedInvoiceIdentity) for item in result):
        raise GovernanceError("every document identity must be explicitly verified")
    identities = tuple(item.identity for item in result)
    if len(identities) != len(set(identities)):
        raise GovernanceError("verified document identities cannot contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class ApprovedDailyBatch:
    """A batch plus its non-blocked verification and explicit approval."""

    batch: DailyRecommendationBatch
    verification: VerifierDecision
    operator_decision: OperatorDecision
    verified_identities: tuple[VerifiedInvoiceIdentity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.batch, DailyRecommendationBatch):
            raise GovernanceError("approved batch requires DailyRecommendationBatch")
        if not isinstance(self.verification, VerifierDecision):
            raise GovernanceError("approved batch requires VerifierDecision")
        if self.verification.result is VerifierResult.BLOCKED:
            raise GovernanceError("a blocked batch cannot be approved")
        if self.verification.verified_batch_id != self.batch.batch_id:
            raise GovernanceError("verifier did not verify this batch")
        if not isinstance(self.operator_decision, OperatorDecision):
            raise GovernanceError("approved batch requires OperatorDecision")
        if self.operator_decision.decision is not OperatorDecisionType.APPROVE:
            raise GovernanceError("operator decision must be APPROVE")
        if self.operator_decision.approved_batch_id != self.batch.batch_id:
            raise GovernanceError("operator did not approve this batch")
        object.__setattr__(
            self, "verified_identities", _verified_tuple(self.verified_identities)
        )


def verify_batch(
    state: RestaurantState,
    batch: DailyRecommendationBatch,
    verified_identities: Iterable[VerifiedInvoiceIdentity],
) -> VerifierDecision:
    """Validate the entire batch, including action-specific hard constraints.

    All failures are accumulated into machine-readable reason codes.  The
    function is pure: neither blocked nor valid proposals alter restaurant
    state.
    """

    if not isinstance(state, RestaurantState):
        raise GovernanceError("state must be RestaurantState")
    if not isinstance(batch, DailyRecommendationBatch):
        raise GovernanceError("batch must be DailyRecommendationBatch")
    verified = _verified_tuple(verified_identities)
    verified_keys = {item.identity for item in verified}
    all_invoices = invoice_index(state)
    active = {invoice.identity: invoice for invoice in state.active_invoices}
    recommendation_keys = {item.identity for item in batch.recommendations}

    failures: list[str] = []
    checks: list[str] = []

    if batch.state_version != state.state_version:
        failures.append("STALE_STATE_VERSION")
    else:
        checks.append("CURRENT_STATE")

    unknown = recommendation_keys - set(all_invoices)
    if unknown:
        failures.append("UNKNOWN_INVOICE")
    else:
        checks.append("KNOWN_SUPPLIERS")

    if recommendation_keys != set(active):
        failures.append("INCOMPLETE_DAILY_BATCH")
    else:
        checks.append("COMPLETE_ACTIVE_BATCH")

    # DailyRecommendationBatch already enforces this invariant.  Keep it in
    # the verifier evidence so the UI/audit trail shows that the hard check ran.
    checks.append("UNIQUE_INVOICES")

    if recommendation_keys <= verified_keys:
        checks.append("VERIFIED_DOCUMENT_IDENTITIES")
    else:
        failures.append("UNVERIFIED_DOCUMENT")

    aggregate_pay_minor = 0
    amounts_valid = True
    currencies_valid = True
    statuses_valid = True
    actions_valid = True
    for recommendation in batch.recommendations:
        invoice = all_invoices.get(recommendation.identity)
        if invoice is None:
            continue
        if recommendation.amount_minor != invoice.amount_minor:
            amounts_valid = False
        if invoice.currency != state.currency:
            currencies_valid = False
        if invoice.payment_status is not InvoicePaymentStatus.UNPAID:
            statuses_valid = False

        if recommendation.action is ProcurementAction.PAY:
            aggregate_pay_minor += recommendation.amount_minor
            if invoice.supplier_status is not SupplierStatus.ACTIVE:
                failures.append("PAY_SUPPLIER_NOT_ACTIVE")
                actions_valid = False
            if invoice.context_conflict_codes:
                failures.append("UNRESOLVED_BUSINESS_CONTEXT")
                actions_valid = False
        elif recommendation.action is ProcurementAction.DEFER:
            if invoice.context_conflict_codes:
                failures.append("UNRESOLVED_BUSINESS_CONTEXT")
                actions_valid = False
        elif recommendation.action is ProcurementAction.VERIFY:
            # VERIFY deliberately commits zero dollars.  Its displayed amount
            # is the immutable full obligation for operator context.
            pass
        else:  # Defensive even though the frozen contract rejects this first.
            failures.append("UNSUPPORTED_ACTION")
            actions_valid = False

    if amounts_valid:
        checks.append("EXACT_AMOUNTS")
    else:
        failures.append("INCORRECT_FULL_AMOUNT")
    if currencies_valid:
        checks.append("MATCHING_CURRENCY")
    else:
        failures.append("CURRENCY_MISMATCH")
    if statuses_valid:
        checks.append("UNPAID_INVOICES")
    else:
        failures.append("INVOICE_NOT_UNPAID")
    if actions_valid:
        checks.append("ACTION_SPECIFIC_RULES")

    if aggregate_pay_minor <= state.cash_minor:
        checks.append("BATCH_CASH_AVAILABLE")
    else:
        failures.append("OVER_BUDGET")

    # Keep diagnostics deterministic and contract-valid.
    failures = list(dict.fromkeys(failures))
    checks = list(dict.fromkeys(checks))
    verification_id = make_audit_id("verify", batch.batch_id)
    if failures:
        return VerifierDecision(
            verification_id=verification_id,
            batch_id=batch.batch_id,
            result=VerifierResult.BLOCKED,
            reason_codes=tuple(failures),
            checks_passed=tuple(checks),
        )
    return VerifierDecision(
        verification_id=verification_id,
        batch_id=batch.batch_id,
        result=VerifierResult.REQUIRES_OPERATOR,
        reason_codes=(
            "FINANCIAL_ACTION"
            if aggregate_pay_minor
            else "OPERATOR_COMMIT_REQUIRED",
        ),
        checks_passed=tuple(checks),
        verified_batch_id=batch.batch_id,
    )


def approve_batch(
    batch: DailyRecommendationBatch,
    verification: VerifierDecision,
    verified_identities: Iterable[VerifiedInvoiceIdentity],
    *,
    decision_id: str | None = None,
) -> ApprovedDailyBatch:
    """Attach an explicit operator approval to a non-blocked verified batch."""

    if not isinstance(batch, DailyRecommendationBatch):
        raise GovernanceError("batch must be DailyRecommendationBatch")
    if not isinstance(verification, VerifierDecision):
        raise GovernanceError("verification must be VerifierDecision")
    if verification.result is VerifierResult.BLOCKED:
        raise GovernanceError("blocked batch cannot be approved")
    if verification.verified_batch_id != batch.batch_id:
        raise GovernanceError("verification does not belong to this batch")
    operator = OperatorDecision(
        decision_id=decision_id or make_audit_id("approve", batch.batch_id),
        reviewed_batch_id=batch.batch_id,
        decision=OperatorDecisionType.APPROVE,
        approved_batch_id=batch.batch_id,
    )
    return ApprovedDailyBatch(
        batch=batch,
        verification=verification,
        operator_decision=operator,
        verified_identities=_verified_tuple(verified_identities),
    )


def reject_batch(
    batch: DailyRecommendationBatch, *, decision_id: str | None = None
) -> OperatorDecision:
    """Record rejection; the result intentionally carries no executable batch."""

    if not isinstance(batch, DailyRecommendationBatch):
        raise GovernanceError("batch must be DailyRecommendationBatch")
    return OperatorDecision(
        decision_id=decision_id or make_audit_id("reject", batch.batch_id),
        reviewed_batch_id=batch.batch_id,
        decision=OperatorDecisionType.REJECT,
    )


def modify_batch(
    batch: DailyRecommendationBatch,
    changes: Mapping[InvoiceIdentity, ProcurementAction],
    *,
    replacement_batch_id: str,
    decision_id: str | None = None,
) -> tuple[DailyRecommendationBatch, OperatorDecision]:
    """Create an unapproved replacement that must return to ``verify_batch``."""

    if not isinstance(batch, DailyRecommendationBatch):
        raise GovernanceError("batch must be DailyRecommendationBatch")
    if not isinstance(changes, Mapping) or not changes:
        raise GovernanceError("MODIFY needs at least one action change")
    if replacement_batch_id == batch.batch_id:
        raise GovernanceError("replacement batch ID must be new")
    by_identity = {item.identity: item for item in batch.recommendations}
    unknown = set(changes) - set(by_identity)
    if unknown:
        raise GovernanceError("cannot modify an invoice outside the reviewed batch")
    for identity, action in changes.items():
        if not isinstance(identity, InvoiceIdentity):
            raise GovernanceError("change keys must be InvoiceIdentity")
        if not isinstance(action, ProcurementAction):
            raise GovernanceError("change values must be ProcurementAction")

    replacement_items: list[Recommendation] = []
    changed_any = False
    for item in batch.recommendations:
        action = changes.get(item.identity, item.action)
        changed_any = changed_any or action is not item.action
        reasons = item.reason_codes
        if action is not item.action and "OPERATOR_MODIFIED" not in reasons:
            reasons = (*reasons, "OPERATOR_MODIFIED")
        replacement_items.append(replace(item, action=action, reason_codes=reasons))
    if not changed_any:
        raise GovernanceError("MODIFY must change at least one action")

    replacement = replace(
        batch,
        batch_id=replacement_batch_id,
        recommendations=tuple(replacement_items),
    )
    decision = OperatorDecision(
        decision_id=decision_id or make_audit_id("modify", batch.batch_id),
        reviewed_batch_id=batch.batch_id,
        decision=OperatorDecisionType.MODIFY,
        replacement_batch_id=replacement.batch_id,
    )
    return replacement, decision


__all__ = [
    "ApprovedDailyBatch",
    "GovernanceError",
    "approve_batch",
    "modify_batch",
    "reject_batch",
    "verify_batch",
]
