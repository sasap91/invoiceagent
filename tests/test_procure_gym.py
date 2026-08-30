from dataclasses import replace
from decimal import Decimal

import pytest

from procureagent.contracts import (
    InvoiceIdentity,
    InvoicePaymentStatus,
    PaymentProofStatus,
    ProcurementAction,
    load_locked_scenario,
    load_scenario,
)
from procureagent.evaluation import compare_policies, fixture_verified_identities
from procureagent.governance import approve_batch, verify_batch
from procureagent.gym import GymTransitionError, ProcureGym
from procureagent.policy import criticality_aware_greedy_v1
from procureagent.state import invoice_index


@pytest.fixture(scope="module")
def scenario():
    return load_locked_scenario()


def _approve(state):
    batch = criticality_aware_greedy_v1(state)
    verified = fixture_verified_identities(state)
    result = verify_batch(state, batch, verified)
    return approve_batch(batch, result, verified)


def test_primary_step_is_atomic_approved_and_reproducible(scenario):
    env = ProcureGym(scenario)
    initial, info = env.reset(seed=138)
    approved = _approve(initial)
    next_state, reward, terminated, truncated, step_info = env.step(approved)

    assert info["simulation_only"] is True
    assert next_state.day == 1
    assert next_state.state_version == 2
    assert next_state.cash_minor == 100_000
    assert reward == Decimal("0.800")
    assert not terminated and not truncated
    assert step_info["paid_invoice_numbers"] == ("FF-10482", "PF-25031")
    assert step_info["deferred_invoice_numbers"] == ("PR-15007",)
    assert step_info["review_invoice_numbers"] == ("CP-70019",)
    by_id = invoice_index(next_state)
    assert by_id[InvoiceIdentity("fresh_farms", "FF-10482")].payment_status is (
        InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED
    )
    assert by_id[InvoiceIdentity("prime_foods", "PF-25031")].payment_status is (
        InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED
    )
    assert initial.cash_minor == 500_000
    assert initial.day == 0

    repeat = ProcureGym(scenario)
    repeated = repeat.step(_approve(repeat.state))
    assert repeated[:4] == (next_state, reward, terminated, truncated)


def test_unapproved_or_stale_batch_changes_nothing(scenario):
    env = ProcureGym(scenario)
    batch = criticality_aware_greedy_v1(env.state)
    before = env.state
    with pytest.raises(GymTransitionError, match="ApprovedDailyBatch"):
        env.step(batch)  # type: ignore[arg-type]
    assert env.state is before

    approved = _approve(env.state)
    env.step(approved)
    after_first = env.state
    with pytest.raises(GymTransitionError, match="reverification"):
        env.step(approved)
    assert env.state is after_first


def test_verify_commits_no_money_and_global_day_advances(scenario):
    env = ProcureGym(scenario)
    state, _, _, _, _ = env.step(_approve(env.state))
    before_cash = state.cash_minor
    second = _approve(state)
    assert all(
        item.action in {ProcurementAction.DEFER, ProcurementAction.VERIFY}
        for item in second.batch.recommendations
    )
    state, _, _, _, info = env.step(second)
    assert state.cash_minor == before_cash
    assert state.day == 2
    assert info["review_invoice_numbers"] == ("CP-70019",)
    # Prime Foods' explicitly scheduled two-day delivery arrived.
    prime = invoice_index(state)[InvoiceIdentity("prime_foods", "PF-25031")]
    assert prime.inventory_days_remaining == 8


def test_exact_receipt_closes_only_the_approved_simulated_payment(scenario):
    env = ProcureGym(scenario)
    env.step(_approve(env.state))
    proof = scenario.payment_proofs[0]
    cash_before = env.state.cash_minor
    day_before = env.state.day
    version_before = env.state.state_version
    closed = env.confirm_payment(proof)
    fresh = invoice_index(closed)[proof.identity]
    assert fresh.payment_status is InvoicePaymentStatus.PAID_CONFIRMED
    assert closed.cash_minor == cash_before
    assert closed.day == day_before
    assert closed.state_version == version_before + 1
    assert proof.receipt_id in env.consumed_receipt_ids
    with pytest.raises(Exception, match="already consumed"):
        env.confirm_payment(proof)


@pytest.mark.parametrize(
    "change",
    [
        {"amount_minor": 149_999},
        {"currency": "CAD"},
        {"invoice_number": "FF-99999"},
        {"status": PaymentProofStatus.REVIEW_REQUIRED},
    ],
)
def test_bad_receipt_leaves_ledger_open(scenario, change):
    env = ProcureGym(scenario)
    env.step(_approve(env.state))
    before = env.state
    with pytest.raises(Exception):
        env.confirm_payment(replace(scenario.payment_proofs[0], **change))
    assert env.state is before
    assert not env.consumed_receipt_ids


def test_reset_clears_receipts_schedule_audit_and_state(scenario):
    env = ProcureGym(scenario)
    env.step(_approve(env.state))
    env.confirm_payment(scenario.payment_proofs[0])
    state, info = env.reset(seed=138)
    assert state == scenario.initial_state
    assert info["seed"] == 138
    assert env.audit_log == ()
    assert not env.consumed_receipt_ids
    with pytest.raises(GymTransitionError):
        env.reset(seed=True)


def test_controlled_comparison_uses_same_seed_and_exposes_raw_results(scenario):
    comparison = compare_policies(scenario)
    critical = comparison.criticality_aware
    baseline = comparison.earliest_due_first
    assert comparison.identical_initial_state is True
    assert critical.seed == baseline.seed == scenario.seed
    assert critical.executor == baseline.executor
    assert comparison.scope.invoice_identity == "upstream_c2_verified_input_not_rescored"
    day_zero = critical.daily_rankings[0]
    assert [item.supplier_id for item in day_zero.suppliers] == [
        "fresh_farms",
        "prime_foods",
        "packright",
        "cleanpro",
    ]
    assert [item.runway_margin_days for item in day_zero.suppliers] == [1, 1, 17, 14]
    validity = critical.action_validity
    assert validity.actions_proposed == validity.verified_identity_actions
    assert validity.actions_proposed == validity.exact_full_amount_actions
    assert validity.blocked_batches == 0
    assert validity.wrong_supplier_actions == 0
    assert validity.wrong_amount_actions == 0
    assert validity.duplicate_actions == 0
    assert validity.over_budget_batches == 0
    assert validity.unsafe_executed_batches == 0
    assert critical.steps == 7
    assert critical.truncated is True
    assert baseline.terminated is True
    assert baseline.steps == 4
    assert critical.raw_metrics.high_criticality_stockout_days == 0
    assert baseline.raw_metrics.high_criticality_stockout_days == 2
    assert critical.raw_metrics.deliveries_arrived == 2
    assert baseline.raw_metrics.deliveries_arrived == 1
    assert critical.total_reward > baseline.total_reward
    oracle = comparison.schedule_oracle
    assert oracle.enumerated_schedules == 512
    assert 0 < oracle.legal_schedules <= oracle.enumerated_schedules
    assert comparison.criticality_regret == Decimal("0.000")
    assert comparison.earliest_due_first_regret > 0
    assert all(
        payment.exact_amount_minor > 0 for payment in oracle.scheduled_payments
    )


# ---------------------------------------------------------------------------
# Daily cash inflow and the timing decision it makes possible
# ---------------------------------------------------------------------------


CASHFLOW_SCENARIO_PATH = "data/procureagent/scenario_cashflow_v1.json"


@pytest.fixture(scope="module")
def cashflow_scenario():
    return load_scenario(CASHFLOW_SCENARIO_PATH)


def test_locked_scenario_declares_no_inflow_and_is_unchanged_by_the_field(scenario):
    """The frozen fixture cannot carry the key, so it must default to zero."""

    assert scenario.daily_cash_inflow_minor == 0

    environment = ProcureGym(scenario)
    environment.reset()
    state, _, _, _, info = environment.step(_approve(environment.state))
    assert state.cash_minor == 100_000  # $5,000 - $4,000, no revenue added
    assert info["cash_inflow_minor"] == 0


def test_inflow_is_credited_after_the_batch_commits(cashflow_scenario):
    """Revenue must not be able to fund a payment approved against today's cash."""

    environment = ProcureGym(cashflow_scenario)
    environment.reset()
    state, _, _, _, info = environment.step(_approve(environment.state))

    assert info["cash_before_minor"] == 400_000
    assert info["cash_inflow_minor"] == 30_000
    # 400,000 - 400,000 paid + 30,000 revenue
    assert state.cash_minor == 30_000
    assert info["cash_after_minor"] == 30_000


def test_inflow_makes_a_deferred_invoice_payable_on_a_later_day(cashflow_scenario):
    """The 'when' decision: PackRight is deferred until the day it is affordable."""

    environment = ProcureGym(cashflow_scenario)
    environment.reset()
    packright = InvoiceIdentity("packright", "PR-15007")
    paid_on_day = None

    while not environment.episode_complete:
        day = environment.state.day
        batch = criticality_aware_greedy_v1(environment.state)
        actions = {item.identity: item.action for item in batch.recommendations}
        environment.step(_approve(environment.state))
        if actions.get(packright) is ProcurementAction.PAY:
            paid_on_day = day
            break

    assert paid_on_day == 6, "PackRight becomes affordable only once revenue accumulates"
    invoice = invoice_index(environment.state)[packright]
    assert invoice.payment_status is InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED


def test_bounded_oracle_stays_an_upper_bound_once_revenue_exists(cashflow_scenario):
    """Regret can never be negative: a policy cannot beat the oracle.

    The oracle pre-filters schedules by total committed spend. With revenue in
    play that budget must span the horizon, otherwise a schedule that is
    affordable precisely because it waits is discarded unsimulated and the
    reported optimum comes back worse than a policy actually achieves.
    """

    comparison = compare_policies(cashflow_scenario)

    assert comparison.criticality_regret >= 0
    assert comparison.earliest_due_first_regret >= 0
    assert comparison.schedule_oracle.total_reward >= (
        comparison.criticality_aware.total_reward
    )


def test_oracle_expresses_a_real_payment_day_under_inflow(cashflow_scenario, scenario):
    """Timing is only a decision if the oracle can prefer a day over 'never'."""

    with_inflow = {
        payment.supplier_id: payment.payment_day
        for payment in compare_policies(cashflow_scenario).schedule_oracle.scheduled_payments
    }
    without_inflow = {
        payment.supplier_id: payment.payment_day
        for payment in compare_policies(scenario).schedule_oracle.scheduled_payments
    }

    assert with_inflow["packright"] is not None
    # The locked scenario is unchanged: PackRight is never affordable again.
    assert without_inflow["packright"] is None


def test_new_terminal_properties_track_the_episode(cashflow_scenario):
    environment = ProcureGym(cashflow_scenario)
    environment.reset()

    assert environment.scenario is cashflow_scenario
    assert environment.horizon_days == 7
    assert not environment.terminated and not environment.truncated
    assert not environment.episode_complete

    while not environment.episode_complete:
        environment.step(_approve(environment.state))

    assert environment.truncated and not environment.terminated
    assert environment.episode_complete
    assert environment.state.day == 7
    with pytest.raises(GymTransitionError):
        environment.step(_approve(environment.state))
