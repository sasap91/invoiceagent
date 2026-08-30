from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path

import pytest

from procureagent.router_lab import (
    DatasetSplit,
    IdentityRewardConfig,
    LayoutNovelty,
    OCRQualityBin,
    RouteOutcome,
    RouterAction,
    RouterContext,
    RouterDataError,
    RouterDataset,
    RouterDecision,
    RouterStateError,
    RouterTrainingRow,
    TabularRouter,
    always_review_predict,
    compare_router_baselines,
    constrain_action,
    evaluate_identity_router,
    fixed_gate_predict,
    hard_safety_reasons,
    legal_actions,
    load_router_dataset,
    run_router_lab,
)


def safe_context(**changes: object) -> RouterContext:
    context = RouterContext(
        ocr_quality_bin=OCRQualityBin.HIGH,
        rule_candidate_present=True,
        model_candidate_count=1,
        rule_model_agreement=True,
        known_supplier=True,
        grounded_evidence=True,
        layout_novelty=LayoutNovelty.KNOWN,
    )
    return replace(context, **changes)


def outcomes(
    *, rules_correct: bool, model_correct: bool
) -> dict[RouterAction, RouteOutcome]:
    return {
        RouterAction.RULES_ONLY: RouteOutcome(rules_correct, 5),
        RouterAction.RULES_PLUS_LOCAL_MODEL: RouteOutcome(model_correct, 120),
        RouterAction.HUMAN_REVIEW: RouteOutcome(None, 3000),
    }


def row(
    row_id: str,
    split: DatasetSplit,
    context: RouterContext,
    *,
    rules_correct: bool,
    model_correct: bool,
) -> RouterTrainingRow:
    return RouterTrainingRow(
        row_id=row_id,
        split=split,
        context=context,
        outcomes=outcomes(
            rules_correct=rules_correct,
            model_correct=model_correct,
        ),
    )


def test_locked_router_dataset_is_synthetic_and_split_from_ryan_test_data() -> None:
    dataset = load_router_dataset()

    assert dataset.schema_version == "procure-router-training-v1"
    assert dataset.synthetic_only is True
    assert dataset.contains_real_documents is False
    assert dataset.contains_ryan_locked_test_data is False
    assert len(dataset.rows) == 14
    assert sum(item.split is DatasetSplit.TRAIN for item in dataset.rows) == 7
    assert sum(item.split is DatasetSplit.DEVELOPMENT for item in dataset.rows) == 7
    assert all(item.synthetic for item in dataset.rows)
    assert all(
        item.origin == "hand_authored_synthetic_router_fixture"
        for item in dataset.rows
    )


def test_loader_refuses_metadata_that_mentions_locked_test_data(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "procureagent"
        / "router_training_v1.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["provenance"]["contains_ryan_locked_test_data"] = True
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RouterDataError, match="locked test"):
        load_router_dataset(unsafe)


def test_declared_identity_reward_values_and_latency_cost() -> None:
    config = IdentityRewardConfig()

    assert config.score(
        RouterAction.RULES_ONLY, RouteOutcome(True, 100)
    ) == Decimal("9.99")
    assert config.score(
        RouterAction.RULES_ONLY, RouteOutcome(False, 0)
    ) == Decimal("-50")
    assert config.score(
        RouterAction.RULES_PLUS_LOCAL_MODEL, RouteOutcome(True, 100)
    ) == Decimal("9.79")
    assert config.score(
        RouterAction.HUMAN_REVIEW, RouteOutcome(None, 100)
    ) == Decimal("-2.01")


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"known_supplier": False}, "UNKNOWN_SUPPLIER"),
        ({"grounded_evidence": False}, "UNGROUNDED_EVIDENCE"),
        (
            {"model_candidate_count": 2, "rule_model_agreement": False},
            "AMBIGUOUS_MODEL_CANDIDATES",
        ),
        ({"rule_model_agreement": False}, "RULE_MODEL_DISAGREEMENT"),
    ],
)
def test_hard_safety_conditions_force_human_review(
    changes: dict[str, object], reason: str
) -> None:
    context = safe_context(**changes)

    assert reason in hard_safety_reasons(context)
    assert legal_actions(context) == (RouterAction.HUMAN_REVIEW,)
    decision = constrain_action(
        context,
        RouterAction.RULES_PLUS_LOCAL_MODEL,
        policy_source="malicious_test_policy",
    )
    assert decision.action is RouterAction.HUMAN_REVIEW
    assert decision.safety_forced is True
    assert reason in decision.reason_codes


def test_invalid_agreement_context_is_rejected() -> None:
    with pytest.raises(RouterDataError, match="requires one rule"):
        RouterContext(
            ocr_quality_bin=OCRQualityBin.HIGH,
            rule_candidate_present=False,
            model_candidate_count=1,
            rule_model_agreement=True,
            known_supplier=True,
            grounded_evidence=True,
            layout_novelty=LayoutNovelty.KNOWN,
        )


def test_availability_mask_blocks_routes_with_missing_candidate() -> None:
    context = safe_context(
        rule_candidate_present=False,
        model_candidate_count=0,
        rule_model_agreement=False,
    )

    assert legal_actions(context) == (RouterAction.HUMAN_REVIEW,)
    decision = constrain_action(
        context,
        RouterAction.RULES_ONLY,
        policy_source="test",
    )
    assert decision.action is RouterAction.HUMAN_REVIEW
    assert "RULE_CANDIDATE_UNAVAILABLE" in decision.reason_codes


def test_fixed_gate_and_always_review_baselines_are_deterministic() -> None:
    assert fixed_gate_predict(safe_context()).action is RouterAction.RULES_ONLY
    assert (
        fixed_gate_predict(
            safe_context(
                ocr_quality_bin=OCRQualityBin.MEDIUM,
                layout_novelty=LayoutNovelty.NOVEL,
            )
        ).action
        is RouterAction.RULES_PLUS_LOCAL_MODEL
    )
    assert always_review_predict(safe_context()).action is RouterAction.HUMAN_REVIEW


def test_router_requires_fit_then_freeze_and_cannot_refit() -> None:
    dataset = load_router_dataset()
    router = TabularRouter()

    with pytest.raises(RouterStateError, match="freeze"):
        router.predict(safe_context())
    with pytest.raises(RouterStateError, match="fit"):
        router.freeze()

    router.fit(dataset.rows)
    assert router.fit_rows == 7
    assert router.is_frozen is False
    router.freeze()
    assert router.is_frozen is True
    with pytest.raises(RouterStateError, match="cannot be refit"):
        router.fit(dataset.rows)


def test_fit_ignores_development_rows_and_requires_training_data() -> None:
    dataset = load_router_dataset()
    development = [
        item for item in dataset.rows if item.split is DatasetSplit.DEVELOPMENT
    ]

    with pytest.raises(RouterDataError, match="training row"):
        TabularRouter().fit(development)


def test_frozen_router_uses_learned_routes_and_hard_override() -> None:
    dataset = load_router_dataset()
    router = TabularRouter().fit(dataset.rows).freeze()
    by_id = {item.row_id: item for item in dataset.rows}

    assert (
        router.predict(
            by_id["synthetic-development-novel-strong-rule"].context
        ).action
        is RouterAction.RULES_ONLY
    )
    assert (
        router.predict(
            by_id["synthetic-development-novel-agreement"].context
        ).action
        is RouterAction.RULES_PLUS_LOCAL_MODEL
    )
    unsafe = router.predict(
        by_id["synthetic-development-unknown-supplier"].context
    )
    assert unsafe.action is RouterAction.HUMAN_REVIEW
    assert unsafe.safety_forced is True
    assert unsafe.policy_source == "hard_safety_override_v1"


def test_unseen_context_falls_back_to_fixed_gate() -> None:
    dataset = load_router_dataset()
    router = TabularRouter().fit(dataset.rows).freeze()
    unseen = safe_context(
        ocr_quality_bin=OCRQualityBin.LOW,
        model_candidate_count=0,
        rule_model_agreement=False,
        layout_novelty=LayoutNovelty.KNOWN,
    )

    decision = router.predict(unseen)
    assert decision.action is RouterAction.HUMAN_REVIEW
    assert decision.policy_source == "fixed_gate_fallback_for_unseen_context"
    assert "UNSEEN_CONTEXT_FALLBACK" in decision.reason_codes


def test_evaluator_independently_masks_an_unsafe_custom_policy() -> None:
    unsafe_context = safe_context(known_supplier=False)
    held_out = row(
        "synthetic-development-unsafe",
        DatasetSplit.DEVELOPMENT,
        unsafe_context,
        rules_correct=False,
        model_correct=False,
    )

    def malicious(_: RouterContext) -> RouterDecision:
        return RouterDecision(
            action=RouterAction.RULES_ONLY,
            requested_action=RouterAction.RULES_ONLY,
            safety_forced=False,
            reason_codes=(),
            policy_source="malicious_test_policy",
        )

    metrics = evaluate_identity_router("malicious", malicious, [held_out])

    assert metrics.wrong_automatic_accepts == 0
    assert metrics.review_count == 1
    assert metrics.safety_forced_reviews == 1


def test_held_out_comparison_reports_all_baselines_and_scoped_claim() -> None:
    comparison = run_router_lab()

    assert comparison.evidence_scope == (
        "held_out_rows_in_repeated_hand_authored_synthetic_context_bins_only"
    )
    assert comparison.learned.fixture_count == 7
    assert comparison.learned.correct_automatic_identities == 4
    assert comparison.learned.wrong_automatic_accepts == 0
    assert comparison.learned.review_count == 3
    assert comparison.fixed_gate.correct_automatic_identities == 3
    assert comparison.fixed_gate.wrong_automatic_accepts == 0
    assert comparison.fixed_gate.review_count == 4
    assert comparison.always_review.review_count == 7
    assert comparison.development_contexts_seen_in_training == 7
    assert comparison.development_contexts_total == 7
    assert comparison.frozen_test_evaluated is False
    assert comparison.zero_additional_unsafe_accepts is True
    assert comparison.improvement_supported is True
    assert "synthetic" in comparison.claim.lower()
    assert "not evidence of generalization" in comparison.claim
    assert "real-invoice accuracy claim" in comparison.claim


def test_negative_held_out_result_makes_no_improvement_claim() -> None:
    context = safe_context()
    train = row(
        "synthetic-train-negative-case",
        DatasetSplit.TRAIN,
        context,
        rules_correct=False,
        model_correct=True,
    )
    development = row(
        "synthetic-development-negative-case",
        DatasetSplit.DEVELOPMENT,
        context,
        rules_correct=True,
        model_correct=False,
    )
    dataset = RouterDataset(
        schema_version="procure-router-training-v1",
        dataset_id="synthetic-negative-result-test",
        rows=(train, development),
        synthetic_only=True,
        contains_real_documents=False,
        contains_ryan_locked_test_data=False,
    )
    router = TabularRouter().fit(dataset.rows).freeze()

    comparison = compare_router_baselines(router, dataset)

    assert comparison.learned.wrong_automatic_accepts == 1
    assert comparison.fixed_gate.wrong_automatic_accepts == 0
    assert comparison.zero_additional_unsafe_accepts is False
    assert comparison.improvement_supported is False
    assert comparison.claim.startswith("No Router Lab improvement is claimed")


def test_metrics_keep_reviews_in_strict_exact_match_denominator() -> None:
    comparison = run_router_lab()

    assert comparison.learned.strict_exact_match_rate == Decimal(4) / Decimal(7)
    assert comparison.learned.automatic_precision == Decimal("1")
    assert comparison.learned.review_rate == Decimal(3) / Decimal(7)
    assert comparison.learned.average_reward > comparison.fixed_gate.average_reward
