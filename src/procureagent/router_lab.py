"""Constrained contextual-bandit lab for invoice-identity routing.

This module addresses only ProcureAgent evaluation question 1: should the
identity path trust an anchored rule, also use Ryan's local LayoutLMv3 token
classifier, or ask a person to review the document?

It deliberately does *not* rank suppliers or choose payment actions.  Those
sequential decisions belong to ProcureGym.  Safety constraints are applied
after policy selection and cannot be learned away.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping


class RouterDataError(ValueError):
    """Raised when router data violates the isolated synthetic-data contract."""


class RouterStateError(RuntimeError):
    """Raised when fit/freeze/predict are called out of order."""


class OCRQualityBin(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LayoutNovelty(str, Enum):
    KNOWN = "known"
    NOVEL = "novel"


class RouterAction(str, Enum):
    RULES_ONLY = "RULES_ONLY"
    RULES_PLUS_LOCAL_MODEL = "RULES_PLUS_LOCAL_MODEL"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    DEVELOPMENT = "development"


_ACTION_TIE_BREAK = (
    RouterAction.RULES_ONLY,
    RouterAction.RULES_PLUS_LOCAL_MODEL,
    RouterAction.HUMAN_REVIEW,
)


@dataclass(frozen=True, slots=True)
class RouterContext:
    """Small, auditable context visible to the identity router.

    ``rule_model_agreement`` is true only when exactly one model span exists
    and an anchored-rule candidate exists.  False also represents the cases
    where there is nothing to compare.
    """

    ocr_quality_bin: OCRQualityBin
    rule_candidate_present: bool
    model_candidate_count: int
    rule_model_agreement: bool
    known_supplier: bool
    grounded_evidence: bool
    layout_novelty: LayoutNovelty

    def __post_init__(self) -> None:
        if not isinstance(self.ocr_quality_bin, OCRQualityBin):
            raise RouterDataError("ocr_quality_bin must be an OCRQualityBin")
        if not isinstance(self.layout_novelty, LayoutNovelty):
            raise RouterDataError("layout_novelty must be a LayoutNovelty")
        for name in (
            "rule_candidate_present",
            "rule_model_agreement",
            "known_supplier",
            "grounded_evidence",
        ):
            if not isinstance(getattr(self, name), bool):
                raise RouterDataError(f"{name} must be true or false")
        if (
            isinstance(self.model_candidate_count, bool)
            or not isinstance(self.model_candidate_count, int)
            or self.model_candidate_count < 0
        ):
            raise RouterDataError("model_candidate_count must be a nonnegative integer")
        if self.rule_model_agreement and not (
            self.rule_candidate_present and self.model_candidate_count == 1
        ):
            raise RouterDataError(
                "rule_model_agreement requires one rule and one model candidate"
            )

    @property
    def key(self) -> tuple[str | bool | int, ...]:
        return (
            self.ocr_quality_bin.value,
            self.rule_candidate_present,
            self.model_candidate_count,
            self.rule_model_agreement,
            self.known_supplier,
            self.grounded_evidence,
            self.layout_novelty.value,
        )


@dataclass(frozen=True, slots=True)
class RouteOutcome:
    """Counterfactual synthetic outcome for one bounded routing action."""

    identity_correct: bool | None
    latency_ms: int

    def __post_init__(self) -> None:
        if self.identity_correct is not None and not isinstance(
            self.identity_correct, bool
        ):
            raise RouterDataError("identity_correct must be true, false, or null")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, int)
            or self.latency_ms < 0
        ):
            raise RouterDataError("latency_ms must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class RouterTrainingRow:
    row_id: str
    split: DatasetSplit
    context: RouterContext
    outcomes: Mapping[RouterAction, RouteOutcome]
    synthetic: bool = True
    origin: str = "hand_authored_synthetic_router_fixture"

    def __post_init__(self) -> None:
        if not self.row_id or self.row_id != self.row_id.strip():
            raise RouterDataError("row_id must be non-empty without surrounding space")
        if not isinstance(self.split, DatasetSplit):
            raise RouterDataError("split must be a DatasetSplit")
        if not isinstance(self.context, RouterContext):
            raise RouterDataError("context must be a RouterContext")
        if self.synthetic is not True:
            raise RouterDataError("Router Lab accepts declared synthetic rows only")
        if self.origin != "hand_authored_synthetic_router_fixture":
            raise RouterDataError("router row origin is not the allowed synthetic source")
        normalized = dict(self.outcomes)
        if set(normalized) != set(RouterAction):
            raise RouterDataError("every row must declare an outcome for all three actions")
        for action, outcome in normalized.items():
            if not isinstance(action, RouterAction) or not isinstance(
                outcome, RouteOutcome
            ):
                raise RouterDataError("outcomes must map RouterAction to RouteOutcome")
            if action is RouterAction.HUMAN_REVIEW:
                if outcome.identity_correct is not None:
                    raise RouterDataError(
                        "human review is abstention; identity_correct must be null"
                    )
            elif outcome.identity_correct is None:
                raise RouterDataError(
                    "automatic route outcomes must declare identity correctness"
                )
        object.__setattr__(self, "outcomes", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class RouterDataset:
    schema_version: str
    dataset_id: str
    rows: tuple[RouterTrainingRow, ...]
    synthetic_only: bool
    contains_real_documents: bool
    contains_ryan_locked_test_data: bool

    def __post_init__(self) -> None:
        if self.schema_version != "procure-router-training-v1":
            raise RouterDataError("unsupported router dataset schema")
        if not self.dataset_id:
            raise RouterDataError("dataset_id is required")
        if self.synthetic_only is not True:
            raise RouterDataError("Router Lab dataset must be synthetic-only")
        if self.contains_real_documents:
            raise RouterDataError("real documents are not allowed in Router Lab fitting")
        if self.contains_ryan_locked_test_data:
            raise RouterDataError("Ryan's locked test data must remain isolated")
        if not self.rows:
            raise RouterDataError("router dataset must contain rows")
        ids = [row.row_id for row in self.rows]
        if len(ids) != len(set(ids)):
            raise RouterDataError("router row_id values must be unique")
        if not any(row.split is DatasetSplit.TRAIN for row in self.rows):
            raise RouterDataError("router dataset needs a training split")
        if not any(row.split is DatasetSplit.DEVELOPMENT for row in self.rows):
            raise RouterDataError("router dataset needs a held-out development split")


@dataclass(frozen=True, slots=True)
class IdentityRewardConfig:
    """Frozen, declared reward for identity routing only.

    The latency penalty is -0.01 for every 100 ms.  It is intentionally small
    compared with the -50 wrong-auto-accept penalty.
    """

    correct_automatic_identity: Decimal = Decimal("10")
    incorrect_automatic_identity: Decimal = Decimal("-50")
    local_model_invocation: Decimal = Decimal("-0.2")
    human_review: Decimal = Decimal("-2")
    latency_cost_per_100_ms: Decimal = Decimal("-0.01")
    version: str = "identity-router-reward-v1"

    def score(self, action: RouterAction, outcome: RouteOutcome) -> Decimal:
        latency = self.latency_cost_per_100_ms * (
            Decimal(outcome.latency_ms) / Decimal(100)
        )
        if action is RouterAction.HUMAN_REVIEW:
            return self.human_review + latency
        if outcome.identity_correct is None:
            raise RouterDataError("automatic route needs a correctness outcome")
        identity = (
            self.correct_automatic_identity
            if outcome.identity_correct
            else self.incorrect_automatic_identity
        )
        model_cost = (
            self.local_model_invocation
            if action is RouterAction.RULES_PLUS_LOCAL_MODEL
            else Decimal("0")
        )
        return identity + model_cost + latency


DEFAULT_REWARD_CONFIG = IdentityRewardConfig()


def hard_safety_reasons(context: RouterContext) -> tuple[str, ...]:
    """Return non-learnable reasons that force human review."""

    reasons: list[str] = []
    if not context.known_supplier:
        reasons.append("UNKNOWN_SUPPLIER")
    if not context.grounded_evidence:
        reasons.append("UNGROUNDED_EVIDENCE")
    if context.model_candidate_count > 1:
        reasons.append("AMBIGUOUS_MODEL_CANDIDATES")
    if (
        context.rule_candidate_present
        and context.model_candidate_count == 1
        and not context.rule_model_agreement
    ):
        reasons.append("RULE_MODEL_DISAGREEMENT")
    return tuple(reasons)


def legal_actions(context: RouterContext) -> tuple[RouterAction, ...]:
    """Return the bounded actions after availability and safety masking."""

    if hard_safety_reasons(context):
        return (RouterAction.HUMAN_REVIEW,)
    actions: list[RouterAction] = []
    if context.rule_candidate_present:
        actions.append(RouterAction.RULES_ONLY)
    if context.model_candidate_count == 1 and (
        not context.rule_candidate_present or context.rule_model_agreement
    ):
        actions.append(RouterAction.RULES_PLUS_LOCAL_MODEL)
    actions.append(RouterAction.HUMAN_REVIEW)
    return tuple(actions)


@dataclass(frozen=True, slots=True)
class RouterDecision:
    action: RouterAction
    requested_action: RouterAction
    safety_forced: bool
    reason_codes: tuple[str, ...]
    policy_source: str
    estimated_reward: Decimal | None = None


def constrain_action(
    context: RouterContext,
    requested_action: RouterAction,
    *,
    policy_source: str,
    estimated_reward: Decimal | None = None,
) -> RouterDecision:
    """Apply safety and route-availability masks after any policy output."""

    reasons = list(hard_safety_reasons(context))
    allowed = legal_actions(context)
    if requested_action not in allowed:
        if requested_action is RouterAction.RULES_ONLY:
            reasons.append("RULE_CANDIDATE_UNAVAILABLE")
        elif requested_action is RouterAction.RULES_PLUS_LOCAL_MODEL:
            reasons.append("SINGLE_GROUNDED_MODEL_CANDIDATE_UNAVAILABLE")
        if not reasons:
            reasons.append("ACTION_MASKED")
        return RouterDecision(
            action=RouterAction.HUMAN_REVIEW,
            requested_action=requested_action,
            safety_forced=True,
            reason_codes=tuple(dict.fromkeys(reasons)),
            policy_source=policy_source,
            estimated_reward=estimated_reward,
        )
    return RouterDecision(
        action=requested_action,
        requested_action=requested_action,
        safety_forced=False,
        reason_codes=(),
        policy_source=policy_source,
        estimated_reward=estimated_reward,
    )


def fixed_gate_predict(context: RouterContext) -> RouterDecision:
    """Conservative deterministic baseline used by the P0 identity gate."""

    if hard_safety_reasons(context):
        return constrain_action(
            context,
            RouterAction.RULES_ONLY,
            policy_source="fixed_evidence_gate_v1",
        )
    if (
        context.ocr_quality_bin is OCRQualityBin.HIGH
        and context.rule_candidate_present
        and context.layout_novelty is LayoutNovelty.KNOWN
    ):
        requested = RouterAction.RULES_ONLY
    elif context.model_candidate_count == 1 and (
        not context.rule_candidate_present or context.rule_model_agreement
    ):
        requested = RouterAction.RULES_PLUS_LOCAL_MODEL
    else:
        requested = RouterAction.HUMAN_REVIEW
    return constrain_action(
        context, requested, policy_source="fixed_evidence_gate_v1"
    )


def always_review_predict(context: RouterContext) -> RouterDecision:
    """Maximum-abstention comparison baseline."""

    return constrain_action(
        context,
        RouterAction.HUMAN_REVIEW,
        policy_source="always_review_v1",
    )


class TabularRouter:
    """Dependency-free full-information contextual-bandit policy.

    Each exact synthetic context stores the average declared reward for every
    legal action.  Unseen contexts fall back to the fixed evidence gate.  This
    is intentionally small and interpretable; it is not a production-trained
    or generalizing model.
    """

    def __init__(self, reward_config: IdentityRewardConfig = DEFAULT_REWARD_CONFIG):
        self.reward_config = reward_config
        self._q_values: dict[
            tuple[str | bool | int, ...], Mapping[RouterAction, Decimal]
        ] = {}
        self._fit_rows = 0
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def fit_rows(self) -> int:
        return self._fit_rows

    def fit(self, rows: Iterable[RouterTrainingRow]) -> "TabularRouter":
        if self._frozen:
            raise RouterStateError("a frozen router cannot be refit")
        totals: dict[
            tuple[str | bool | int, ...], dict[RouterAction, Decimal]
        ] = defaultdict(lambda: defaultdict(Decimal))
        counts: dict[
            tuple[str | bool | int, ...], dict[RouterAction, int]
        ] = defaultdict(lambda: defaultdict(int))
        fit_rows = 0
        for row in rows:
            if not isinstance(row, RouterTrainingRow):
                raise RouterDataError("fit accepts RouterTrainingRow values only")
            if row.split is not DatasetSplit.TRAIN:
                continue
            if not row.synthetic or row.origin != "hand_authored_synthetic_router_fixture":
                raise RouterDataError("fit accepts declared synthetic fixtures only")
            fit_rows += 1
            for action in legal_actions(row.context):
                totals[row.context.key][action] += self.reward_config.score(
                    action, row.outcomes[action]
                )
                counts[row.context.key][action] += 1
        if fit_rows == 0:
            raise RouterDataError("fit needs at least one synthetic training row")
        self._q_values = {
            key: {
                action: total / Decimal(counts[key][action])
                for action, total in action_totals.items()
            }
            for key, action_totals in totals.items()
        }
        self._fit_rows = fit_rows
        return self

    def freeze(self) -> "TabularRouter":
        if not self._q_values:
            raise RouterStateError("fit the router before freezing it")
        self._q_values = {
            key: MappingProxyType(dict(values))
            for key, values in self._q_values.items()
        }
        self._frozen = True
        return self

    def predict(self, context: RouterContext) -> RouterDecision:
        if not self._frozen:
            raise RouterStateError("freeze the fitted router before prediction")
        safety = hard_safety_reasons(context)
        if safety:
            return RouterDecision(
                action=RouterAction.HUMAN_REVIEW,
                requested_action=RouterAction.HUMAN_REVIEW,
                safety_forced=True,
                reason_codes=safety,
                policy_source="hard_safety_override_v1",
                estimated_reward=None,
            )
        values = self._q_values.get(context.key)
        if not values:
            fallback = fixed_gate_predict(context)
            return RouterDecision(
                action=fallback.action,
                requested_action=fallback.requested_action,
                safety_forced=fallback.safety_forced,
                reason_codes=fallback.reason_codes + ("UNSEEN_CONTEXT_FALLBACK",),
                policy_source="fixed_gate_fallback_for_unseen_context",
                estimated_reward=None,
            )
        allowed = set(legal_actions(context))
        candidates = [action for action in _ACTION_TIE_BREAK if action in allowed]
        requested = max(
            candidates,
            key=lambda action: (values.get(action, Decimal("-Infinity")), -_ACTION_TIE_BREAK.index(action)),
        )
        return constrain_action(
            context,
            requested,
            policy_source="frozen_tabular_contextual_bandit_v1",
            estimated_reward=values.get(requested),
        )


@dataclass(frozen=True, slots=True)
class IdentityEvaluationMetrics:
    policy_name: str
    split: DatasetSplit
    fixture_count: int
    correct_automatic_identities: int
    wrong_automatic_accepts: int
    review_count: int
    local_model_invocations: int
    safety_forced_reviews: int
    total_latency_ms: int
    total_reward: Decimal

    @property
    def strict_exact_match_rate(self) -> Decimal:
        """Correct automatic identities / all rows; reviews remain denominator."""

        return Decimal(self.correct_automatic_identities) / Decimal(self.fixture_count)

    @property
    def automatic_precision(self) -> Decimal | None:
        automatic = self.correct_automatic_identities + self.wrong_automatic_accepts
        if automatic == 0:
            return None
        return Decimal(self.correct_automatic_identities) / Decimal(automatic)

    @property
    def review_rate(self) -> Decimal:
        return Decimal(self.review_count) / Decimal(self.fixture_count)

    @property
    def average_reward(self) -> Decimal:
        return self.total_reward / Decimal(self.fixture_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_name": self.policy_name,
            "split": self.split.value,
            "fixture_count": self.fixture_count,
            "correct_automatic_identities": self.correct_automatic_identities,
            "wrong_automatic_accepts": self.wrong_automatic_accepts,
            "review_count": self.review_count,
            "local_model_invocations": self.local_model_invocations,
            "safety_forced_reviews": self.safety_forced_reviews,
            "total_latency_ms": self.total_latency_ms,
            "strict_exact_match_rate": str(self.strict_exact_match_rate),
            "automatic_precision": (
                None if self.automatic_precision is None else str(self.automatic_precision)
            ),
            "review_rate": str(self.review_rate),
            "total_reward": str(self.total_reward),
            "average_reward": str(self.average_reward),
        }


Prediction = Callable[[RouterContext], RouterDecision]


def evaluate_identity_router(
    policy_name: str,
    predict: Prediction,
    rows: Iterable[RouterTrainingRow],
    *,
    split: DatasetSplit = DatasetSplit.DEVELOPMENT,
    reward_config: IdentityRewardConfig = DEFAULT_REWARD_CONFIG,
) -> IdentityEvaluationMetrics:
    """Evaluate one frozen policy on declared synthetic held-out rows."""

    selected = [row for row in rows if row.split is split]
    if not selected:
        raise RouterDataError(f"no rows found for split {split.value}")
    if any(not row.synthetic for row in selected):
        raise RouterDataError("Router Lab evaluation accepts synthetic rows only")

    correct = wrong = reviews = local = forced = latency = 0
    total_reward = Decimal("0")
    for row in selected:
        decision = predict(row.context)
        if not isinstance(decision, RouterDecision):
            raise RouterDataError("router policy must return RouterDecision")
        # Re-apply the independent mask even to a custom policy callback.
        guarded = constrain_action(
            row.context,
            decision.action,
            policy_source=decision.policy_source,
            estimated_reward=decision.estimated_reward,
        )
        if guarded.safety_forced or decision.safety_forced:
            forced += 1
        outcome = row.outcomes[guarded.action]
        latency += outcome.latency_ms
        total_reward += reward_config.score(guarded.action, outcome)
        if guarded.action is RouterAction.HUMAN_REVIEW:
            reviews += 1
        else:
            if outcome.identity_correct:
                correct += 1
            else:
                wrong += 1
            if guarded.action is RouterAction.RULES_PLUS_LOCAL_MODEL:
                local += 1

    return IdentityEvaluationMetrics(
        policy_name=policy_name,
        split=split,
        fixture_count=len(selected),
        correct_automatic_identities=correct,
        wrong_automatic_accepts=wrong,
        review_count=reviews,
        local_model_invocations=local,
        safety_forced_reviews=forced,
        total_latency_ms=latency,
        total_reward=total_reward,
    )


@dataclass(frozen=True, slots=True)
class RouterComparison:
    dataset_id: str
    evidence_scope: str
    learned: IdentityEvaluationMetrics
    fixed_gate: IdentityEvaluationMetrics
    always_review: IdentityEvaluationMetrics
    development_contexts_seen_in_training: int
    development_contexts_total: int
    frozen_test_evaluated: bool
    zero_additional_unsafe_accepts: bool
    improvement_supported: bool
    claim: str

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "evidence_scope": self.evidence_scope,
            "learned": self.learned.to_dict(),
            "fixed_gate": self.fixed_gate.to_dict(),
            "always_review": self.always_review.to_dict(),
            "development_contexts_seen_in_training": (
                self.development_contexts_seen_in_training
            ),
            "development_contexts_total": self.development_contexts_total,
            "frozen_test_evaluated": self.frozen_test_evaluated,
            "zero_additional_unsafe_accepts": self.zero_additional_unsafe_accepts,
            "improvement_supported": self.improvement_supported,
            "claim": self.claim,
        }


def compare_router_baselines(
    router: TabularRouter,
    dataset: RouterDataset,
) -> RouterComparison:
    """Compare a frozen router on held-out synthetic development fixtures.

    The function never treats these fixtures as evidence about real invoices.
    A positive result requires higher reward than both baselines and no more
    wrong auto-accepts than the fixed gate.
    """

    if not router.is_frozen:
        raise RouterStateError("freeze the router before held-out comparison")
    learned = evaluate_identity_router(
        "frozen_tabular_contextual_bandit_v1",
        router.predict,
        dataset.rows,
        reward_config=router.reward_config,
    )
    fixed = evaluate_identity_router(
        "fixed_evidence_gate_v1",
        fixed_gate_predict,
        dataset.rows,
        reward_config=router.reward_config,
    )
    review = evaluate_identity_router(
        "always_review_v1",
        always_review_predict,
        dataset.rows,
        reward_config=router.reward_config,
    )
    zero_additional = learned.wrong_automatic_accepts <= fixed.wrong_automatic_accepts
    training_contexts = {
        row.context.key for row in dataset.rows if row.split is DatasetSplit.TRAIN
    }
    development_rows = tuple(
        row for row in dataset.rows if row.split is DatasetSplit.DEVELOPMENT
    )
    seen_development_contexts = sum(
        row.context.key in training_contexts for row in development_rows
    )
    supported = (
        learned.average_reward > fixed.average_reward
        and learned.average_reward > review.average_reward
        and zero_additional
    )
    if supported:
        claim = (
            "Held-out synthetic development rows show a within-bin Router Lab reward "
            "improvement over both declared baselines with zero additional unsafe "
            "accepts. Every development context bin also appears in training, and no "
            "frozen test split was evaluated, so this is not evidence of generalization "
            "or a real-invoice accuracy claim."
        )
    else:
        claim = (
            "No Router Lab improvement is claimed: held-out synthetic development "
            "metrics did not beat both baselines without additional unsafe accepts."
        )
    return RouterComparison(
        dataset_id=dataset.dataset_id,
        evidence_scope=(
            "held_out_rows_in_repeated_hand_authored_synthetic_context_bins_only"
        ),
        learned=learned,
        fixed_gate=fixed,
        always_review=review,
        development_contexts_seen_in_training=seen_development_contexts,
        development_contexts_total=len(development_rows),
        frozen_test_evaluated=False,
        zero_additional_unsafe_accepts=zero_additional,
        improvement_supported=supported,
        claim=claim,
    )


def _context_from_json(value: object) -> RouterContext:
    if not isinstance(value, dict):
        raise RouterDataError("context must be a JSON object")
    try:
        return RouterContext(
            ocr_quality_bin=OCRQualityBin(value["ocr_quality_bin"]),
            rule_candidate_present=value["rule_candidate_present"],
            model_candidate_count=value["model_candidate_count"],
            rule_model_agreement=value["rule_model_agreement"],
            known_supplier=value["known_supplier"],
            grounded_evidence=value["grounded_evidence"],
            layout_novelty=LayoutNovelty(value["layout_novelty"]),
        )
    except (KeyError, ValueError) as exc:
        raise RouterDataError("invalid router context JSON") from exc


def _outcomes_from_json(value: object) -> Mapping[RouterAction, RouteOutcome]:
    if not isinstance(value, dict):
        raise RouterDataError("outcomes must be a JSON object")
    result: dict[RouterAction, RouteOutcome] = {}
    try:
        for action in RouterAction:
            raw = value[action.value]
            if not isinstance(raw, dict):
                raise RouterDataError("route outcome must be a JSON object")
            result[action] = RouteOutcome(
                identity_correct=raw["identity_correct"],
                latency_ms=raw["latency_ms"],
            )
    except (KeyError, ValueError) as exc:
        raise RouterDataError("invalid route outcome JSON") from exc
    return result


def load_router_dataset(path: str | Path | None = None) -> RouterDataset:
    """Load and validate the isolated synthetic Router Lab matrix."""

    source = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[2]
        / "data"
        / "procureagent"
        / "router_training_v1.json"
    )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RouterDataError(f"cannot load router dataset: {source}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("provenance"), dict):
        raise RouterDataError("router dataset root/provenance must be JSON objects")
    provenance = raw["provenance"]
    raw_rows = raw.get("rows")
    if not isinstance(raw_rows, list):
        raise RouterDataError("router dataset rows must be a JSON list")
    rows: list[RouterTrainingRow] = []
    for value in raw_rows:
        if not isinstance(value, dict):
            raise RouterDataError("each router row must be a JSON object")
        try:
            rows.append(
                RouterTrainingRow(
                    row_id=value["row_id"],
                    split=DatasetSplit(value["split"]),
                    context=_context_from_json(value["context"]),
                    outcomes=_outcomes_from_json(value["outcomes"]),
                    synthetic=value["synthetic"],
                    origin=value["origin"],
                )
            )
        except (KeyError, ValueError) as exc:
            raise RouterDataError("invalid router training row JSON") from exc
    try:
        return RouterDataset(
            schema_version=raw["schema_version"],
            dataset_id=raw["dataset_id"],
            rows=tuple(rows),
            synthetic_only=provenance["synthetic_only"],
            contains_real_documents=provenance["contains_real_documents"],
            contains_ryan_locked_test_data=provenance[
                "contains_ryan_locked_test_data"
            ],
        )
    except KeyError as exc:
        raise RouterDataError("router dataset metadata is incomplete") from exc


def run_router_lab(path: str | Path | None = None) -> RouterComparison:
    dataset = load_router_dataset(path)
    router = TabularRouter().fit(dataset.rows).freeze()
    return compare_router_baselines(router, dataset)


def main() -> None:
    print(json.dumps(run_router_lab().to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
