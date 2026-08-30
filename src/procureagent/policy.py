"""Frozen deterministic procurement policies for the hackathon benchmark."""

from __future__ import annotations

from .contracts import (
    ContractValidationError,
    DailyRecommendationBatch,
    PolicyType,
    ProcurementAction,
    Recommendation,
    RestaurantState,
    SupplierCriticality,
    SupplierStatus,
    SyntheticSupplierInvoice,
    make_audit_id,
)


class PolicyError(ContractValidationError):
    """Raised when a policy cannot safely produce a complete daily batch."""


def _needs_verification(invoice: SyntheticSupplierInvoice) -> tuple[str, ...]:
    reasons = list(invoice.context_conflict_codes)
    if invoice.supplier_status is not SupplierStatus.ACTIVE:
        reasons.append("SUPPLIER_NOT_ACTIVE")
    return tuple(dict.fromkeys(reasons))


def criticality_score(invoice: SyntheticSupplierInvoice) -> int:
    """Return the exact frozen Criticality-Aware Greedy v1 score."""

    if not isinstance(invoice, SyntheticSupplierInvoice):
        raise PolicyError("invoice must be SyntheticSupplierInvoice")
    score = 0
    if invoice.inventory_days_remaining <= invoice.delivery_lead_days + 1:
        score += 100
    if invoice.supplier_criticality is SupplierCriticality.HIGH:
        score += 40
    elif invoice.supplier_criticality is SupplierCriticality.MEDIUM:
        score += 20
    if invoice.due_in_days <= 1:
        score += 30
    if invoice.due_in_days < 0:
        score += 20
    if invoice.inventory_days_remaining >= 10:
        score -= 40
    return score


def _score_reasons(invoice: SyntheticSupplierInvoice) -> tuple[str, ...]:
    reasons: list[str] = []
    if invoice.inventory_days_remaining <= invoice.delivery_lead_days + 1:
        reasons.append("STOCKOUT_RISK")
    if invoice.supplier_criticality is SupplierCriticality.HIGH:
        reasons.append("CRITICAL_SUPPLIER")
    elif invoice.supplier_criticality is SupplierCriticality.MEDIUM:
        reasons.append("MEDIUM_CRITICALITY")
    if invoice.due_in_days <= 1:
        reasons.append("DUE_SOON")
    if invoice.due_in_days < 0:
        reasons.append("OVERDUE")
    if invoice.inventory_days_remaining >= 10:
        reasons.append("INVENTORY_BUFFER")
    return tuple(reasons)


def _batch(
    state: RestaurantState,
    *,
    policy_name: str,
    policy_version: str,
    recommendations: tuple[Recommendation, ...],
) -> DailyRecommendationBatch:
    if not recommendations:
        raise PolicyError("cannot recommend a daily batch with no active invoices")
    return DailyRecommendationBatch(
        batch_id=make_audit_id(
            "day", state.day, policy_name, policy_version, state.state_version
        ),
        state_version=state.state_version,
        policy_name=policy_name,
        policy_version=policy_version,
        policy_type=PolicyType.DETERMINISTIC_RULES,
        recommendations=recommendations,
    )


def criticality_aware_greedy_v1(
    state: RestaurantState,
) -> DailyRecommendationBatch:
    """Allocate full payments by the frozen criticality-aware score.

    Conflicting or non-active supplier context is emitted as ``VERIFY`` and
    consumes no cash.  Every other active invoice receives exactly one action.
    """

    if not isinstance(state, RestaurantState):
        raise PolicyError("state must be RestaurantState")

    verifications: list[tuple[SyntheticSupplierInvoice, tuple[str, ...]]] = []
    scorable: list[SyntheticSupplierInvoice] = []
    for invoice in state.active_invoices:
        conflicts = _needs_verification(invoice)
        if conflicts:
            verifications.append((invoice, conflicts))
        else:
            scorable.append(invoice)

    scorable.sort(
        key=lambda invoice: (
            -criticality_score(invoice),
            invoice.due_in_days,
            invoice.supplier_id,
        )
    )
    verifications.sort(key=lambda item: (item[0].due_in_days, item[0].supplier_id))

    remaining_cash = state.cash_minor
    recommendations: list[Recommendation] = []
    for invoice in scorable:
        reasons = list(_score_reasons(invoice))
        if invoice.amount_minor <= remaining_cash:
            action = ProcurementAction.PAY
            remaining_cash -= invoice.amount_minor
            reasons.append("FULL_PAYMENT_FITS_BUDGET")
        else:
            action = ProcurementAction.DEFER
            reasons.append("BATCH_CASH_PRIORITY")
        recommendations.append(
            Recommendation(
                supplier_id=invoice.supplier_id,
                invoice_number=invoice.invoice_number,
                action=action,
                amount_minor=invoice.amount_minor,
                reason_codes=tuple(reasons) or ("DETERMINISTIC_PRIORITY",),
            )
        )

    for invoice, reasons in verifications:
        recommendations.append(
            Recommendation(
                supplier_id=invoice.supplier_id,
                invoice_number=invoice.invoice_number,
                action=ProcurementAction.VERIFY,
                amount_minor=invoice.amount_minor,
                reason_codes=reasons,
            )
        )

    return _batch(
        state,
        policy_name="criticality_aware_greedy",
        policy_version="v1",
        recommendations=tuple(recommendations),
    )


def earliest_due_first(state: RestaurantState) -> DailyRecommendationBatch:
    """Deterministic baseline that ignores operational criticality."""

    if not isinstance(state, RestaurantState):
        raise PolicyError("state must be RestaurantState")
    ordered = sorted(
        state.active_invoices,
        key=lambda invoice: (invoice.due_in_days, invoice.supplier_id),
    )
    remaining_cash = state.cash_minor
    recommendations: list[Recommendation] = []
    for invoice in ordered:
        conflicts = _needs_verification(invoice)
        if conflicts:
            action = ProcurementAction.VERIFY
            reasons = conflicts
        elif invoice.amount_minor <= remaining_cash:
            action = ProcurementAction.PAY
            remaining_cash -= invoice.amount_minor
            reasons = ("EARLIEST_DUE_FIRST", "FULL_PAYMENT_FITS_BUDGET")
        else:
            action = ProcurementAction.DEFER
            reasons = ("EARLIEST_DUE_FIRST", "INSUFFICIENT_DAILY_CASH")
        recommendations.append(
            Recommendation(
                supplier_id=invoice.supplier_id,
                invoice_number=invoice.invoice_number,
                action=action,
                amount_minor=invoice.amount_minor,
                reason_codes=reasons,
            )
        )
    return _batch(
        state,
        policy_name="earliest_due_first",
        policy_version="v1",
        recommendations=tuple(recommendations),
    )


# Short product-facing alias; the implementation name preserves its frozen version.
criticality_aware_greedy = criticality_aware_greedy_v1


__all__ = [
    "PolicyError",
    "criticality_aware_greedy",
    "criticality_aware_greedy_v1",
    "criticality_score",
    "earliest_due_first",
]
