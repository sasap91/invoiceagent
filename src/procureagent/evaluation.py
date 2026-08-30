"""Controlled same-state comparison for deterministic ProcureGym policies.

The evaluation intentionally keeps three questions separate:

1. Invoice identity correctness belongs to the upstream C2 document gate.
2. Supplier prioritization is recorded as an ordered daily runway ranking.
3. Executable actions are checked for identity, exact amount, timing, uniqueness,
   and aggregate cash safety before the fixed benchmark executor applies them.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from itertools import product
from typing import Callable, Iterable

from .contracts import (
    DailyRecommendationBatch,
    DocumentStatus,
    InvoiceIdentity,
    InvoicePaymentStatus,
    PolicyType,
    ProcureScenario,
    ProcurementAction,
    Recommendation,
    RestaurantState,
    SupplierCriticality,
    SupplierStatus,
    VerifiedInvoiceIdentity,
    VerifierDecision,
    VerifierResult,
    load_locked_scenario,
    make_audit_id,
)
from .governance import approve_batch, verify_batch
from .gym import AuditEvent, ProcureGym, RawMetrics
from .policy import criticality_aware_greedy_v1, earliest_due_first
from .state import invoice_index


Policy = Callable[[RestaurantState], DailyRecommendationBatch]


class EvaluationError(RuntimeError):
    """Raised when a policy cannot pass the fixed benchmark executor."""


@dataclass(frozen=True, slots=True)
class EvaluationScope:
    invoice_identity: str = "upstream_c2_verified_input_not_rescored"
    prioritization: str = "daily_order_with_inventory_lead_due_and_criticality"
    action_correctness: str = "hard_gated_exact_full_payment_and_timing"


@dataclass(frozen=True, slots=True)
class ActionMaskEntry:
    """Legal per-invoice choices; amount is never model-generated."""

    supplier_id: str
    invoice_number: str
    exact_amount_minor: int
    allowed_actions: tuple[ProcurementAction, ...]

    @property
    def identity(self) -> InvoiceIdentity:
        return InvoiceIdentity(self.supplier_id, self.invoice_number)


@dataclass(frozen=True, slots=True)
class RankedSupplier:
    rank: int
    supplier_id: str
    invoice_number: str
    inventory_days_remaining: int
    delivery_lead_days: int
    runway_margin_days: int
    due_in_days: int
    criticality: SupplierCriticality
    proposed_action: ProcurementAction
    exact_amount_minor: int


@dataclass(frozen=True, slots=True)
class DailyRanking:
    day: int
    state_version: int
    suppliers: tuple[RankedSupplier, ...]


@dataclass(frozen=True, slots=True)
class ActionValidityMetrics:
    batches_proposed: int = 0
    batches_verified: int = 0
    blocked_batches: int = 0
    actions_proposed: int = 0
    verified_identity_actions: int = 0
    exact_full_amount_actions: int = 0
    eligible_pay_actions: int = 0
    wrong_supplier_actions: int = 0
    wrong_amount_actions: int = 0
    duplicate_actions: int = 0
    masked_action_violations: int = 0
    invalid_pay_timing_actions: int = 0
    over_budget_batches: int = 0
    unsafe_executed_batches: int = 0


@dataclass(frozen=True, slots=True)
class PolicyAction:
    day: int
    rank: int
    supplier_id: str
    invoice_number: str
    action: ProcurementAction
    exact_amount_minor: int


@dataclass(frozen=True, slots=True)
class PolicyRun:
    policy_name: str
    policy_version: str
    seed: int
    steps: int
    total_reward: Decimal
    raw_metrics: RawMetrics
    action_validity: ActionValidityMetrics
    daily_rankings: tuple[DailyRanking, ...]
    final_state: RestaurantState
    actions: tuple[PolicyAction, ...]
    audit_log: tuple[AuditEvent, ...]
    terminated: bool
    truncated: bool
    executor: str = "fixed_auto_approval_for_controlled_benchmark_only"


@dataclass(frozen=True, slots=True)
class ScheduledPayment:
    supplier_id: str
    invoice_number: str
    payment_day: int | None
    exact_amount_minor: int


@dataclass(frozen=True, slots=True)
class ScheduleOracleResult:
    """Best declared-reward schedule in the bounded four-invoice search space."""

    total_reward: Decimal
    raw_metrics: RawMetrics
    scheduled_payments: tuple[ScheduledPayment, ...]
    steps: int
    enumerated_schedules: int
    legal_schedules: int
    objective: str = "declared_procuregym_reward_configuration"
    bound: str = "at_most_4_payable_invoices_x_7_days_plus_never"


@dataclass(frozen=True, slots=True)
class ControlledComparison:
    scenario_id: str
    seed: int
    identical_initial_state: bool
    scope: EvaluationScope
    criticality_aware: PolicyRun
    earliest_due_first: PolicyRun
    schedule_oracle: ScheduleOracleResult
    criticality_regret: Decimal
    earliest_due_first_regret: Decimal


def fixture_verified_identities(
    state: RestaurantState,
) -> tuple[VerifiedInvoiceIdentity, ...]:
    """Represent the benchmark assumption that fixture identities are locked."""

    return tuple(
        VerifiedInvoiceIdentity(
            document_id=make_audit_id(
                "fixture-doc", invoice.supplier_id, invoice.invoice_number
            ),
            supplier_id=invoice.supplier_id,
            invoice_number=invoice.invoice_number,
            status=DocumentStatus.CONFIRMED,
        )
        for invoice in state.active_invoices
    )


def legal_action_mask(state: RestaurantState) -> tuple[ActionMaskEntry, ...]:
    """Expose exact outstanding amounts and bounded action enums to any policy.

    A learned P1 policy may rank or select actions, but cannot invent a supplier,
    invoice, or payment amount. Aggregate cash remains a batch-level verifier
    constraint.
    """

    entries = []
    for invoice in state.active_invoices:
        if (
            invoice.context_conflict_codes
            or invoice.supplier_status is not SupplierStatus.ACTIVE
        ):
            allowed = (ProcurementAction.VERIFY,)
        else:
            allowed = (
                ProcurementAction.PAY,
                ProcurementAction.DEFER,
                ProcurementAction.VERIFY,
            )
        entries.append(
            ActionMaskEntry(
                supplier_id=invoice.supplier_id,
                invoice_number=invoice.invoice_number,
                exact_amount_minor=invoice.amount_minor,
                allowed_actions=allowed,
            )
        )
    return tuple(entries)


def _daily_ranking(
    state: RestaurantState, batch: DailyRecommendationBatch
) -> DailyRanking:
    invoices = invoice_index(state)
    suppliers = []
    for rank, recommendation in enumerate(batch.recommendations, start=1):
        invoice = invoices[recommendation.identity]
        suppliers.append(
            RankedSupplier(
                rank=rank,
                supplier_id=invoice.supplier_id,
                invoice_number=invoice.invoice_number,
                inventory_days_remaining=invoice.inventory_days_remaining,
                delivery_lead_days=invoice.delivery_lead_days,
                runway_margin_days=(
                    invoice.inventory_days_remaining - invoice.delivery_lead_days
                ),
                due_in_days=invoice.due_in_days,
                criticality=invoice.supplier_criticality,
                proposed_action=recommendation.action,
                exact_amount_minor=invoice.amount_minor,
            )
        )
    return DailyRanking(
        day=state.day,
        state_version=state.state_version,
        suppliers=tuple(suppliers),
    )


def assess_action_validity(
    state: RestaurantState,
    batch: DailyRecommendationBatch,
    verified_identities: Iterable[VerifiedInvoiceIdentity],
    decision: VerifierDecision | None = None,
) -> ActionValidityMetrics:
    """Report action errors separately from upstream identity-model accuracy."""

    verified = tuple(verified_identities)
    verified_keys = {item.identity for item in verified}
    invoices = invoice_index(state)
    mask = {entry.identity: entry for entry in legal_action_mask(state)}
    identities = tuple(item.identity for item in batch.recommendations)
    wrong_supplier = sum(identity not in invoices for identity in identities)
    wrong_amount = sum(
        identity in invoices and item.amount_minor != invoices[identity].amount_minor
        for identity, item in zip(identities, batch.recommendations)
    )
    duplicates = len(identities) - len(set(identities))
    masked = sum(
        identity not in mask or item.action not in mask[identity].allowed_actions
        for identity, item in zip(identities, batch.recommendations)
    )
    invalid_timing = sum(
        item.action is ProcurementAction.PAY
        and (
            identity not in invoices
            or invoices[identity].payment_status is not InvoicePaymentStatus.UNPAID
            or invoices[identity].supplier_status is not SupplierStatus.ACTIVE
            or bool(invoices[identity].context_conflict_codes)
        )
        for identity, item in zip(identities, batch.recommendations)
    )
    pay_minor = sum(
        item.amount_minor
        for item in batch.recommendations
        if item.action is ProcurementAction.PAY
    )
    reviewed = decision or verify_batch(state, batch, verified)
    blocked = reviewed.result is VerifierResult.BLOCKED
    return ActionValidityMetrics(
        batches_proposed=1,
        batches_verified=int(not blocked),
        blocked_batches=int(blocked),
        actions_proposed=len(batch.recommendations),
        verified_identity_actions=sum(identity in verified_keys for identity in identities),
        exact_full_amount_actions=sum(
            identity in invoices and item.amount_minor == invoices[identity].amount_minor
            for identity, item in zip(identities, batch.recommendations)
        ),
        eligible_pay_actions=sum(
            item.action is ProcurementAction.PAY for item in batch.recommendations
        )
        - invalid_timing,
        wrong_supplier_actions=wrong_supplier,
        wrong_amount_actions=wrong_amount,
        duplicate_actions=duplicates,
        masked_action_violations=masked,
        invalid_pay_timing_actions=invalid_timing,
        over_budget_batches=int(pay_minor > state.cash_minor),
        # A blocked proposal is never executed by this evaluator.
        unsafe_executed_batches=0,
    )


def _sum_dataclass(items: tuple[object, ...], target_type: type):
    fields = target_type.__dataclass_fields__
    return target_type(
        **{name: sum(getattr(item, name) for item in items) for name in fields}
    )


def _sum_metrics(metrics: tuple[RawMetrics, ...]) -> RawMetrics:
    return _sum_dataclass(metrics, RawMetrics)


def _sum_validity(metrics: tuple[ActionValidityMetrics, ...]) -> ActionValidityMetrics:
    return _sum_dataclass(metrics, ActionValidityMetrics)


def run_controlled_policy(
    policy: Policy,
    scenario: ProcureScenario | None = None,
    *,
    seed: int | None = None,
) -> PolicyRun:
    """Run one policy with a fixed executor and no human modifications.

    Auto-approval is deliberately confined to this controlled benchmark. The
    live product path still requires an explicit operator approval.
    """

    selected = scenario or load_locked_scenario()
    selected_seed = selected.seed if seed is None else seed
    env = ProcureGym(selected)
    state, _ = env.reset(seed=selected_seed)
    actions: list[PolicyAction] = []
    rankings: list[DailyRanking] = []
    action_metrics: list[ActionValidityMetrics] = []
    step_metrics: list[RawMetrics] = []
    total_reward = Decimal("0")
    terminated = False
    truncated = False
    policy_name = ""
    policy_version = ""

    while not (terminated or truncated):
        if not state.active_invoices:
            break
        batch = policy(state)
        policy_name = batch.policy_name
        policy_version = batch.policy_version
        verified = fixture_verified_identities(state)
        decision = verify_batch(state, batch, verified)
        validity = assess_action_validity(state, batch, verified, decision)
        action_metrics.append(validity)
        if decision.result is VerifierResult.BLOCKED:
            raise EvaluationError(
                f"policy {batch.policy_name} was blocked: "
                + ",".join(decision.reason_codes)
            )
        approved = approve_batch(
            batch,
            decision,
            verified,
            decision_id=make_audit_id("benchmark-approve", batch.batch_id),
        )
        rankings.append(_daily_ranking(state, batch))
        actions.extend(
            PolicyAction(
                day=state.day,
                rank=rank,
                supplier_id=item.supplier_id,
                invoice_number=item.invoice_number,
                action=item.action,
                exact_amount_minor=item.amount_minor,
            )
            for rank, item in enumerate(batch.recommendations, start=1)
        )
        state, reward, terminated, truncated, info = env.step(approved)
        total_reward += reward
        step_metrics.append(info["raw_metrics"])

    return PolicyRun(
        policy_name=policy_name,
        policy_version=policy_version,
        seed=selected_seed,
        steps=len(step_metrics),
        total_reward=total_reward.quantize(Decimal("0.001")),
        raw_metrics=_sum_metrics(tuple(step_metrics)),
        action_validity=_sum_validity(tuple(action_metrics)),
        daily_rankings=tuple(rankings),
        final_state=state,
        actions=tuple(actions),
        audit_log=env.audit_log,
        terminated=terminated,
        truncated=truncated,
    )


def _oracle_batch(
    state: RestaurantState,
    schedule: dict[InvoiceIdentity, int | None],
    candidate_id: int,
) -> DailyRecommendationBatch | None:
    recommendations = []
    for invoice in sorted(
        state.active_invoices,
        key=lambda item: (item.supplier_id, item.invoice_number),
    ):
        payment_day = schedule.get(invoice.identity)
        unavailable = (
            invoice.context_conflict_codes
            or invoice.supplier_status is not SupplierStatus.ACTIVE
        )
        if payment_day == state.day and unavailable:
            return None
        if payment_day is not None and payment_day < state.day:
            return None
        if unavailable:
            action = ProcurementAction.VERIFY
            reasons = tuple(invoice.context_conflict_codes) or (
                "SUPPLIER_NOT_ACTIVE",
            )
        elif payment_day == state.day:
            action = ProcurementAction.PAY
            reasons = ("ORACLE_SCHEDULE_PAY",)
        else:
            action = ProcurementAction.DEFER
            reasons = ("ORACLE_SCHEDULE_DEFER",)
        recommendations.append(
            Recommendation(
                supplier_id=invoice.supplier_id,
                invoice_number=invoice.invoice_number,
                action=action,
                # The action mask supplies the canonical full amount.
                amount_minor=invoice.amount_minor,
                reason_codes=reasons,
            )
        )
    if not recommendations:
        return None
    return DailyRecommendationBatch(
        batch_id=make_audit_id(
            "oracle", candidate_id, state.day, state.state_version
        ),
        state_version=state.state_version,
        policy_name="bounded_schedule_oracle",
        policy_version="v1",
        policy_type=PolicyType.DETERMINISTIC_RULES,
        recommendations=tuple(recommendations),
    )


def _run_oracle_candidate(
    scenario: ProcureScenario,
    schedule: dict[InvoiceIdentity, int | None],
    candidate_id: int,
) -> tuple[Decimal, RawMetrics, int] | None:
    env = ProcureGym(scenario)
    state, _ = env.reset(seed=scenario.seed)
    rewards = Decimal("0")
    metrics: list[RawMetrics] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        batch = _oracle_batch(state, schedule, candidate_id)
        if batch is None:
            return None
        verified = fixture_verified_identities(state)
        decision = verify_batch(state, batch, verified)
        if decision.result is VerifierResult.BLOCKED:
            return None
        approved = approve_batch(
            batch,
            decision,
            verified,
            decision_id=make_audit_id(
                "oracle-approve", candidate_id, state.day, state.state_version
            ),
        )
        state, reward, terminated, truncated, info = env.step(approved)
        rewards += reward
        metrics.append(info["raw_metrics"])
    return rewards.quantize(Decimal("0.001")), _sum_metrics(tuple(metrics)), len(metrics)


@lru_cache(maxsize=8)
def _bounded_schedule_oracle(scenario: ProcureScenario) -> ScheduleOracleResult:
    payable = tuple(
        invoice
        for invoice in scenario.initial_state.active_invoices
        if not invoice.context_conflict_codes
        and invoice.supplier_status is SupplierStatus.ACTIVE
    )
    if len(payable) > 4 or scenario.horizon_days > 7:
        raise EvaluationError(
            "schedule oracle is intentionally bounded to four payable invoices and seven days"
        )

    choices: tuple[int | None, ...] = (*range(scenario.horizon_days), None)
    enumerated = 0
    legal = 0
    best_result: tuple[Decimal, RawMetrics, int] | None = None
    best_schedule: tuple[ScheduledPayment, ...] = ()
    best_key: tuple[Decimal, int, int, int, int, int] | None = None
    for candidate_id, payment_days in enumerate(product(choices, repeat=len(payable))):
        enumerated += 1
        committed_minor = sum(
            invoice.amount_minor
            for invoice, payment_day in zip(payable, payment_days)
            if payment_day is not None
        )
        # Admissible pre-filter only: the true per-day legality check is
        # verify_batch inside _run_oracle_candidate.  The budget must include
        # simulated revenue, otherwise a schedule that is affordable precisely
        # because it waits for inflow is discarded before it is ever simulated,
        # and the "optimum" comes back worse than a policy actually achieves.
        horizon_budget_minor = (
            scenario.initial_state.cash_minor
            + scenario.daily_cash_inflow_minor * scenario.horizon_days
        )
        if committed_minor > horizon_budget_minor:
            continue
        schedule = {
            invoice.identity: payment_day
            for invoice, payment_day in zip(payable, payment_days)
        }
        result = _run_oracle_candidate(scenario, schedule, candidate_id)
        if result is None:
            continue
        legal += 1
        reward, metrics, steps = result
        key = (
            reward,
            -metrics.high_criticality_stockout_days,
            -metrics.stockout_days,
            -metrics.supplier_disruptions,
            -metrics.late_fees_minor,
            metrics.deliveries_arrived,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_result = result
            best_schedule = tuple(
                ScheduledPayment(
                    supplier_id=invoice.supplier_id,
                    invoice_number=invoice.invoice_number,
                    payment_day=payment_day,
                    exact_amount_minor=invoice.amount_minor,
                )
                for invoice, payment_day in zip(payable, payment_days)
            )

    if best_result is None:
        raise EvaluationError("bounded schedule oracle found no legal schedule")
    reward, metrics, steps = best_result
    return ScheduleOracleResult(
        total_reward=reward,
        raw_metrics=metrics,
        scheduled_payments=best_schedule,
        steps=steps,
        enumerated_schedules=enumerated,
        legal_schedules=legal,
    )


def bounded_schedule_oracle(
    scenario: ProcureScenario | None = None,
) -> ScheduleOracleResult:
    """Exhaust every full-payment day/never schedule inside the locked bound."""

    return _bounded_schedule_oracle(scenario or load_locked_scenario())


def compare_policies(
    scenario: ProcureScenario | None = None,
) -> ControlledComparison:
    """Compare both P0 policies and bounded oracle from identical state."""

    selected = scenario or load_locked_scenario()
    criticality = run_controlled_policy(
        criticality_aware_greedy_v1, selected, seed=selected.seed
    )
    baseline = run_controlled_policy(
        earliest_due_first, selected, seed=selected.seed
    )
    oracle = bounded_schedule_oracle(selected)
    return ControlledComparison(
        scenario_id=selected.scenario_id,
        seed=selected.seed,
        # Both runners receive this same immutable object and reset with the
        # same seed; no live human modifications enter the benchmark.
        identical_initial_state=(criticality.seed == baseline.seed == selected.seed),
        scope=EvaluationScope(),
        criticality_aware=criticality,
        earliest_due_first=baseline,
        schedule_oracle=oracle,
        criticality_regret=(oracle.total_reward - criticality.total_reward).quantize(
            Decimal("0.001")
        ),
        earliest_due_first_regret=(
            oracle.total_reward - baseline.total_reward
        ).quantize(Decimal("0.001")),
    )


__all__ = [
    "ActionMaskEntry",
    "ActionValidityMetrics",
    "ControlledComparison",
    "DailyRanking",
    "EvaluationError",
    "EvaluationScope",
    "PolicyAction",
    "PolicyRun",
    "RankedSupplier",
    "ScheduleOracleResult",
    "ScheduledPayment",
    "assess_action_validity",
    "bounded_schedule_oracle",
    "compare_policies",
    "fixture_verified_identities",
    "legal_action_mask",
    "run_controlled_policy",
]
