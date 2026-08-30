from dataclasses import replace

import pytest

from procureagent.contracts import (
    InvoiceIdentity,
    InvoicePaymentStatus,
    ProcurementAction,
    RestaurantState,
    VerifierResult,
    load_locked_scenario,
)
from procureagent.evaluation import (
    assess_action_validity,
    fixture_verified_identities,
    legal_action_mask,
)
from procureagent.governance import (
    GovernanceError,
    approve_batch,
    modify_batch,
    reject_batch,
    verify_batch,
)
from procureagent.policy import (
    PolicyError,
    criticality_aware_greedy_v1,
    criticality_score,
    earliest_due_first,
)


@pytest.fixture(scope="module")
def scenario():
    return load_locked_scenario()


def _actions(batch):
    return [(item.supplier_id, item.action) for item in batch.recommendations]


def test_frozen_criticality_scores_and_primary_batch(scenario):
    invoices = {item.supplier_id: item for item in scenario.initial_state.invoices}
    assert criticality_score(invoices["fresh_farms"]) == 170
    assert criticality_score(invoices["prime_foods"]) == 140
    assert criticality_score(invoices["packright"]) == 10

    batch = criticality_aware_greedy_v1(scenario.initial_state)
    assert _actions(batch) == [
        ("fresh_farms", ProcurementAction.PAY),
        ("prime_foods", ProcurementAction.PAY),
        ("packright", ProcurementAction.DEFER),
        ("cleanpro", ProcurementAction.VERIFY),
    ]
    assert sum(
        item.amount_minor
        for item in batch.recommendations
        if item.action is ProcurementAction.PAY
    ) == 400_000
    assert {item.identity for item in batch.recommendations} == {
        item.identity for item in scenario.initial_state.active_invoices
    }
    assert batch == criticality_aware_greedy_v1(scenario.initial_state)


def test_earliest_due_first_is_a_distinct_fixed_baseline(scenario):
    batch = earliest_due_first(scenario.initial_state)
    assert _actions(batch) == [
        ("packright", ProcurementAction.PAY),
        ("cleanpro", ProcurementAction.VERIFY),
        ("fresh_farms", ProcurementAction.PAY),
        ("prime_foods", ProcurementAction.DEFER),
    ]
    assert batch == earliest_due_first(scenario.initial_state)


def test_action_mask_uses_canonical_full_amounts_and_bounds_conflicts(scenario):
    mask = {entry.identity: entry for entry in legal_action_mask(scenario.initial_state)}
    assert mask[InvoiceIdentity("fresh_farms", "FF-10482")].exact_amount_minor == 150_000
    assert ProcurementAction.PAY in mask[
        InvoiceIdentity("fresh_farms", "FF-10482")
    ].allowed_actions
    assert mask[InvoiceIdentity("cleanpro", "CP-70019")].allowed_actions == (
        ProcurementAction.VERIFY,
    )


def test_policy_only_covers_active_unpaid_invoices(scenario):
    state = scenario.initial_state
    invoices = tuple(
        replace(item, payment_status=InvoicePaymentStatus.PAID_CONFIRMED)
        if item.supplier_id == "fresh_farms"
        else item
        for item in state.invoices
    )
    changed = replace(state, invoices=invoices)
    batch = criticality_aware_greedy_v1(changed)
    assert "fresh_farms" not in {item.supplier_id for item in batch.recommendations}

    empty_invoices = tuple(
        replace(item, payment_status=InvoicePaymentStatus.PAID_CONFIRMED)
        for item in state.invoices
    )
    empty = replace(state, invoices=empty_invoices)
    with pytest.raises(PolicyError, match="no active"):
        criticality_aware_greedy_v1(empty)


def test_valid_batch_passes_every_hard_check_but_still_needs_operator(scenario):
    state = scenario.initial_state
    batch = criticality_aware_greedy_v1(state)
    verified = fixture_verified_identities(state)
    result = verify_batch(state, batch, verified)
    assert result.result is VerifierResult.REQUIRES_OPERATOR
    assert result.verified_batch_id == batch.batch_id
    assert {
        "CURRENT_STATE",
        "KNOWN_SUPPLIERS",
        "COMPLETE_ACTIVE_BATCH",
        "VERIFIED_DOCUMENT_IDENTITIES",
        "EXACT_AMOUNTS",
        "BATCH_CASH_AVAILABLE",
    } <= set(result.checks_passed)
    approved = approve_batch(batch, result, verified)
    assert approved.operator_decision.approved_batch_id == batch.batch_id


def test_verifier_blocks_missing_documents_stale_amount_and_incomplete_batch(scenario):
    state = scenario.initial_state
    batch = criticality_aware_greedy_v1(state)
    verified = fixture_verified_identities(state)

    missing_document = verify_batch(state, batch, verified[:-1])
    assert missing_document.result is VerifierResult.BLOCKED
    assert "UNVERIFIED_DOCUMENT" in missing_document.reason_codes

    stale = replace(batch, state_version=state.state_version + 1)
    assert "STALE_STATE_VERSION" in verify_batch(
        state, stale, verified
    ).reason_codes

    wrong_amount_item = replace(batch.recommendations[0], amount_minor=1)
    wrong_amount = replace(
        batch, recommendations=(wrong_amount_item, *batch.recommendations[1:])
    )
    wrong_decision = verify_batch(state, wrong_amount, verified)
    assert "INCORRECT_FULL_AMOUNT" in wrong_decision.reason_codes
    wrong_metrics = assess_action_validity(
        state, wrong_amount, verified, wrong_decision
    )
    assert wrong_metrics.wrong_amount_actions == 1
    assert wrong_metrics.blocked_batches == 1
    assert wrong_metrics.unsafe_executed_batches == 0

    incomplete = replace(batch, recommendations=batch.recommendations[:-1])
    assert "INCOMPLETE_DAILY_BATCH" in verify_batch(
        state, incomplete, verified
    ).reason_codes


def test_modify_creates_unapproved_replacement_and_forces_reverification(scenario):
    state = scenario.initial_state
    batch = criticality_aware_greedy_v1(state)
    replacement, decision = modify_batch(
        batch,
        {InvoiceIdentity("packright", "PR-15007"): ProcurementAction.PAY},
        replacement_batch_id="operator-modified-day-0",
    )
    assert decision.approved_batch_id is None
    assert decision.replacement_batch_id == replacement.batch_id
    verified = fixture_verified_identities(state)
    blocked = verify_batch(state, replacement, verified)
    assert "OVER_BUDGET" in blocked.reason_codes
    metrics = assess_action_validity(state, replacement, verified, blocked)
    assert metrics.over_budget_batches == 1
    assert metrics.unsafe_executed_batches == 0
    with pytest.raises(GovernanceError):
        approve_batch(
            replacement,
            blocked,
            verified,
        )


def test_conflicting_context_cannot_be_changed_from_verify_to_pay(scenario):
    state = scenario.initial_state
    batch = criticality_aware_greedy_v1(state)
    replacement, _ = modify_batch(
        batch,
        {InvoiceIdentity("cleanpro", "CP-70019"): ProcurementAction.PAY},
        replacement_batch_id="unsafe-cleanpro-pay",
    )
    result = verify_batch(state, replacement, fixture_verified_identities(state))
    assert result.result is VerifierResult.BLOCKED
    assert "UNRESOLVED_BUSINESS_CONTEXT" in result.reason_codes
    metrics = assess_action_validity(
        state, replacement, fixture_verified_identities(state), result
    )
    assert metrics.masked_action_violations == 1
    assert metrics.invalid_pay_timing_actions == 1
    assert metrics.unsafe_executed_batches == 0


def test_reject_and_noop_modify_never_create_approval(scenario):
    batch = criticality_aware_greedy_v1(scenario.initial_state)
    rejected = reject_batch(batch)
    assert rejected.approved_batch_id is None
    with pytest.raises(GovernanceError, match="change at least one"):
        modify_batch(
            batch,
            {batch.recommendations[0].identity: batch.recommendations[0].action},
            replacement_batch_id="noop-replacement",
        )
