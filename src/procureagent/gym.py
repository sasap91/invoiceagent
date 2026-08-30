"""A small, deterministic, approval-gated restaurant procurement simulator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from .contracts import (
    ContractValidationError,
    InvoiceIdentity,
    InvoicePaymentStatus,
    PaymentProof,
    ProcureScenario,
    ProcurementAction,
    RestaurantState,
    SupplierCriticality,
    SupplierStatus,
    load_locked_scenario,
    make_audit_id,
)
from .governance import ApprovedDailyBatch, verify_batch
from .state import close_invoice_with_payment_proof, rebuild_versioned_state


class GymTransitionError(ContractValidationError):
    """Raised when an unsafe, unapproved, stale, or terminal step is attempted."""


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Declared reward configuration; raw outcomes remain independently visible."""

    safe_day: Decimal = Decimal("1.0")
    delivery_unlocked: Decimal = Decimal("2.0")
    stockout: Decimal = Decimal("-10.0")
    high_criticality_stockout: Decimal = Decimal("-20.0")
    supplier_disruption: Decimal = Decimal("-8.0")
    negative_cash: Decimal = Decimal("-50.0")
    late_fee_per_1000_minor: Decimal = Decimal("-1.0")


@dataclass(frozen=True, slots=True)
class RawMetrics:
    stockout_days: int
    high_criticality_stockout_days: int
    late_fees_minor: int
    negative_cash_events: int
    supplier_disruptions: int
    deliveries_arrived: int
    paid_invoice_count: int


@dataclass(frozen=True, slots=True)
class AuditEvent:
    event_id: str
    event_type: str
    state_version_before: int
    state_version_after: int
    details: tuple[str, ...]


def calculate_reward(
    metrics: RawMetrics, weights: RewardWeights = RewardWeights()
) -> Decimal:
    """Compute a transparent scalar summary from raw per-step outcomes."""

    reward = weights.safe_day
    reward += weights.delivery_unlocked * metrics.deliveries_arrived
    reward += weights.stockout * metrics.stockout_days
    reward += (
        weights.high_criticality_stockout
        * metrics.high_criticality_stockout_days
    )
    reward += weights.supplier_disruption * metrics.supplier_disruptions
    reward += weights.negative_cash * metrics.negative_cash_events
    reward += (
        weights.late_fee_per_1000_minor
        * Decimal(metrics.late_fees_minor)
        / Decimal(1000)
    )
    return reward.quantize(Decimal("0.001"))


class ProcureGym:
    """Seven-day, seeded simulator with atomic approved-batch transitions."""

    def __init__(
        self,
        scenario: ProcureScenario | None = None,
        *,
        reward_weights: RewardWeights = RewardWeights(),
    ) -> None:
        self._scenario = scenario or load_locked_scenario()
        if not isinstance(self._scenario, ProcureScenario):
            raise GymTransitionError("scenario must be ProcureScenario")
        if not isinstance(reward_weights, RewardWeights):
            raise GymTransitionError("reward_weights must be RewardWeights")
        self.reward_weights = reward_weights
        self._state = self._scenario.initial_state
        self._seed = self._scenario.seed
        self._scheduled_deliveries: dict[InvoiceIdentity, int] = {}
        self._high_zero_streaks: dict[InvoiceIdentity, int] = {}
        self._consumed_receipt_ids: set[str] = set()
        self._terminated = False
        self._truncated = False
        self._audit: list[AuditEvent] = []

    @property
    def state(self) -> RestaurantState:
        return self._state

    @property
    def audit_log(self) -> tuple[AuditEvent, ...]:
        return tuple(self._audit)

    @property
    def consumed_receipt_ids(self) -> frozenset[str]:
        return frozenset(self._consumed_receipt_ids)

    @property
    def terminated(self) -> bool:
        """True once a high-criticality supplier has been starved out."""

        return self._terminated

    @property
    def truncated(self) -> bool:
        """True once the scenario horizon ended the episode without a failure."""

        return self._truncated

    @property
    def episode_complete(self) -> bool:
        """Whether ``step`` would now refuse; check this before proposing a day."""

        return self._terminated or self._truncated

    @property
    def scenario(self) -> ProcureScenario:
        return self._scenario

    @property
    def horizon_days(self) -> int:
        return self._scenario.horizon_days

    def reset(self, *, seed: int | None = None) -> tuple[RestaurantState, dict[str, Any]]:
        """Restore the exact fixture state and clear all episode-local state."""

        selected_seed = self._scenario.seed if seed is None else seed
        if isinstance(selected_seed, bool) or not isinstance(selected_seed, int):
            raise GymTransitionError("seed must be a nonnegative integer")
        if selected_seed < 0:
            raise GymTransitionError("seed must be a nonnegative integer")
        self._seed = selected_seed
        self._state = self._scenario.initial_state
        self._scheduled_deliveries = {}
        self._high_zero_streaks = {}
        self._consumed_receipt_ids = set()
        self._terminated = False
        self._truncated = False
        self._audit = []
        return self._state, {
            "scenario_id": self._scenario.scenario_id,
            "seed": self._seed,
            "horizon_days": self._scenario.horizon_days,
            "simulation_only": True,
        }

    def step(
        self, approved_daily_batch: ApprovedDailyBatch
    ) -> tuple[RestaurantState, Decimal, bool, bool, dict[str, Any]]:
        """Atomically apply one reverified, explicitly approved daily batch."""

        if self._terminated or self._truncated:
            raise GymTransitionError("episode is already complete; call reset")
        if not isinstance(approved_daily_batch, ApprovedDailyBatch):
            raise GymTransitionError("step requires an explicitly ApprovedDailyBatch")

        # Re-run hard constraints against the current state.  Approval is not a
        # capability to execute a stale or subsequently invalid proposal.
        fresh_verification = verify_batch(
            self._state,
            approved_daily_batch.batch,
            approved_daily_batch.verified_identities,
        )
        if fresh_verification.verified_batch_id != approved_daily_batch.batch.batch_id:
            raise GymTransitionError(
                "approved batch failed current-state reverification: "
                + ",".join(fresh_verification.reason_codes)
            )

        batch = approved_daily_batch.batch
        actions = {item.identity: item.action for item in batch.recommendations}
        cash_before = self._state.cash_minor
        pay_total = sum(
            item.amount_minor
            for item in batch.recommendations
            if item.action is ProcurementAction.PAY
        )
        cash_after = cash_before - pay_total
        if cash_after < 0:  # Defensive duplicate of the verifier hard constraint.
            raise GymTransitionError("approved batch would make cash negative")
        # Simulated revenue lands after the batch commits, so it can never fund
        # a payment the verifier approved against this morning's cash.
        cash_inflow_minor = self._scenario.daily_cash_inflow_minor
        cash_after += cash_inflow_minor

        day_before = self._state.day
        day_after = day_before + 1
        scheduled = dict(self._scheduled_deliveries)
        for invoice in self._state.invoices:
            if (
                actions.get(invoice.identity) is ProcurementAction.PAY
                and invoice.payment_unlocks_delivery
            ):
                scheduled[invoice.identity] = day_before + invoice.delivery_lead_days

        late_fees_minor = 0
        supplier_disruptions = 0
        deliveries_arrived = 0
        paid_invoice_count = 0
        next_invoices = []
        for invoice in self._state.invoices:
            action = actions.get(invoice.identity)
            payment_status = invoice.payment_status
            due_in_days = invoice.due_in_days
            supplier_status = invoice.supplier_status

            if action is ProcurementAction.PAY:
                payment_status = InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED
                paid_invoice_count += 1
            elif payment_status is InvoicePaymentStatus.UNPAID:
                due_in_days -= 1
                if due_in_days < 0:
                    late_fees_minor += invoice.late_fee_minor_per_day
                threshold = invoice.disruption_after_days_overdue
                if (
                    threshold is not None
                    and -due_in_days >= threshold
                    and supplier_status is not SupplierStatus.DISRUPTED
                ):
                    supplier_status = SupplierStatus.DISRUPTED
                    supplier_disruptions += 1

            inventory = max(0, invoice.inventory_days_remaining - 1)
            delivery_day = scheduled.get(invoice.identity)
            if delivery_day is not None and delivery_day <= day_after:
                inventory += invoice.delivery_inventory_days
                deliveries_arrived += 1
                del scheduled[invoice.identity]

            next_invoices.append(
                replace(
                    invoice,
                    payment_status=payment_status,
                    due_in_days=due_in_days,
                    inventory_days_remaining=inventory,
                    supplier_status=supplier_status,
                )
            )

        stockout_days = sum(
            invoice.inventory_days_remaining == 0 for invoice in next_invoices
        )
        high_stockout_days = sum(
            invoice.inventory_days_remaining == 0
            and invoice.supplier_criticality is SupplierCriticality.HIGH
            for invoice in next_invoices
        )
        streaks = dict(self._high_zero_streaks)
        for invoice in next_invoices:
            if invoice.supplier_criticality is not SupplierCriticality.HIGH:
                continue
            if invoice.inventory_days_remaining == 0:
                streaks[invoice.identity] = streaks.get(invoice.identity, 0) + 1
            else:
                streaks[invoice.identity] = 0

        metrics = RawMetrics(
            stockout_days=stockout_days,
            high_criticality_stockout_days=high_stockout_days,
            late_fees_minor=late_fees_minor,
            negative_cash_events=int(cash_after < 0),
            supplier_disruptions=supplier_disruptions,
            deliveries_arrived=deliveries_arrived,
            paid_invoice_count=paid_invoice_count,
        )
        reward = calculate_reward(metrics, self.reward_weights)

        # Construct and validate everything before committing any internal field.
        next_state = rebuild_versioned_state(
            self._state,
            invoices=tuple(next_invoices),
            cash_minor=cash_after,
            day=day_after,
        )
        terminated = any(value >= 2 for value in streaks.values())
        truncated = day_after >= self._scenario.horizon_days and not terminated
        audit = AuditEvent(
            event_id=make_audit_id("transition", batch.batch_id, next_state.state_version),
            event_type="APPROVED_DAILY_BATCH_COMMITTED",
            state_version_before=self._state.state_version,
            state_version_after=next_state.state_version,
            details=(
                f"day:{day_before}->{day_after}",
                f"cash_minor:{cash_before}->{cash_after}",
                f"cash_inflow_minor:{cash_inflow_minor}",
                f"late_fees_minor:{late_fees_minor}",
            ),
        )

        self._state = next_state
        self._scheduled_deliveries = scheduled
        self._high_zero_streaks = streaks
        self._terminated = terminated
        self._truncated = truncated
        self._audit.append(audit)

        paid = tuple(
            item.invoice_number
            for item in batch.recommendations
            if item.action is ProcurementAction.PAY
        )
        deferred = tuple(
            item.invoice_number
            for item in batch.recommendations
            if item.action is ProcurementAction.DEFER
        )
        review = tuple(
            item.invoice_number
            for item in batch.recommendations
            if item.action is ProcurementAction.VERIFY
        )
        info = {
            "scenario_id": self._scenario.scenario_id,
            "seed": self._seed,
            "batch_id": batch.batch_id,
            "day_before": day_before,
            "day_after": day_after,
            "paid_invoice_numbers": paid,
            "deferred_invoice_numbers": deferred,
            "review_invoice_numbers": review,
            "cash_before_minor": cash_before,
            "cash_after_minor": cash_after,
            "cash_inflow_minor": cash_inflow_minor,
            "raw_metrics": metrics,
            "reward": reward,
            "audit_event": audit,
            "simulation_only": True,
        }
        return self._state, reward, terminated, truncated, info

    def confirm_payment(self, proof: PaymentProof) -> RestaurantState:
        """Close one already-approved AP item with unused exact receipt proof."""

        state_before = self._state
        next_state = close_invoice_with_payment_proof(
            state_before,
            proof,
            consumed_receipt_ids=self._consumed_receipt_ids,
        )
        audit = AuditEvent(
            event_id=make_audit_id("receipt", proof.receipt_id, next_state.state_version),
            event_type="FULL_PAYMENT_PROOF_CONFIRMED",
            state_version_before=state_before.state_version,
            state_version_after=next_state.state_version,
            details=(
                f"supplier_id:{proof.supplier_id}",
                f"invoice_number:{proof.invoice_number}",
                f"amount_minor:{proof.amount_minor}",
            ),
        )
        self._state = next_state
        self._consumed_receipt_ids.add(proof.receipt_id)
        self._audit.append(audit)
        return self._state


__all__ = [
    "AuditEvent",
    "GymTransitionError",
    "ProcureGym",
    "RawMetrics",
    "RewardWeights",
    "calculate_reward",
]
