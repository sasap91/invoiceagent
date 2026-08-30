"""Focused tests for the UI orchestration safety gates."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

from procureagent.contracts import (
    BoundingBox,
    DocumentReviewDecision,
    InvoiceIdentity,
    InvoiceNumberCandidate,
    InvoicePaymentStatus,
    PaymentProofSource,
    ProcurementAction,
    VerifierResult,
    load_locked_scenario,
    load_scenario,
)
from procureagent.document import (
    InvoiceModelRun,
    InvoiceModelRunStatus,
    ModelInvoiceCandidate,
    RyanInvoiceAdapter,
)
from procureagent.ocr import OcrResult, OcrStatus, OcrWord, PixelBox
from procureagent.ui_adapters import (
    FIXTURE_REPLAY,
    UiFlowError,
    _reset_cached_ryan_adapter_for_tests,
    analyze_invoice_upload,
    analyze_receipt_upload,
    approve_and_simulate,
    confirm_verified_payment,
    get_cached_ryan_adapter,
    modify_and_reverify,
    prepare_procurement,
    propose_day,
    record_human_identity_decision,
    reject_day,
    start_episode,
)


ROOT = Path(__file__).resolve().parents[1]
INVOICE_ASSET = ROOT / "data/procureagent/assets/fresh_farms_invoice.png"
RECEIPT_ASSET = ROOT / "data/procureagent/assets/fresh_farms_payment_receipt.png"


def make_ocr(image, lines, *, status=OcrStatus.SUCCESS):
    if status is not OcrStatus.SUCCESS:
        return OcrResult(
            document_id=image.document_id,
            status=status,
            words=(),
            raw_text="",
            language="eng",
            engine="fake_tesseract",
            engine_version="missing",
            runtime_ms=Decimal("0.1"),
            error_code="TESSERACT_NOT_FOUND",
            error_message="not installed",
        )
    words = []
    for line_number, line in enumerate(lines, start=1):
        for word_number, token in enumerate(line, start=1):
            sequence = len(words)
            x0 = 10 + (word_number - 1) * 110
            y0 = 10 + (line_number - 1) * 45
            pixel = PixelBox(x0, y0, x0 + 90, y0 + 30)
            normalized = BoundingBox(
                1000 * pixel.x0 // image.width,
                1000 * pixel.y0 // image.height,
                (1000 * pixel.x1 + image.width - 1) // image.width,
                (1000 * pixel.y1 + image.height - 1) // image.height,
            )
            words.append(
                OcrWord(
                    sequence=sequence,
                    text=token,
                    confidence=Decimal("0.95"),
                    pixel_box=pixel,
                    normalized_box=normalized,
                    page=1,
                    block=1,
                    paragraph=1,
                    line=line_number,
                    word=word_number,
                )
            )
    return OcrResult(
        document_id=image.document_id,
        status=OcrStatus.SUCCESS,
        words=tuple(words),
        raw_text="\n".join(" ".join(line) for line in lines),
        language="eng",
        engine="fake_tesseract",
        engine_version="5.test",
        runtime_ms=Decimal("3.2"),
    )


class InvoiceOcr:
    def run(self, image):
        return make_ocr(
            image,
            (
                ("Invoice", "No:", "FF-10482"),
                ("Supplier", "ID:", "fresh_farms"),
                ("Total", "USD", "$1,500.00"),
            ),
        )


class LowScoreExactModel:
    def run(self, image, ocr):
        word = ocr.words[2]
        evidence = ModelInvoiceCandidate(
            candidate=InvoiceNumberCandidate(
                invoice_number="FF-10482",
                entity_confidence=Decimal("0.6467026472091675"),
                grounded_in_ocr=True,
                evidence_tokens=(word.text,),
                evidence_boxes=(word.normalized_box,),
            ),
            word_indices=(2,),
            minimum_confidence=Decimal("0.6467026472091675"),
            mean_confidence=Decimal("0.6467026472091675"),
            mean_margin=Decimal("0.3744482994079590"),
        )
        return InvoiceModelRun(
            document_id=image.document_id,
            status=InvoiceModelRunStatus.SUCCESS,
            candidates=(evidence,),
            model_version="ryanznie/test-fixture on cpu",
            latency_ms=Decimal("12.3"),
        )


class ReceiptOcr:
    def run(self, image):
        return make_ocr(
            image,
            (
                ("Receipt", "ID:", "RCPT-FF-10482"),
                ("Supplier:", "Fresh", "Farms"),
                ("Invoice", "Number:", "FF-10482"),
                ("Paid", "Date:", "2026-08-30"),
                ("Currency:", "USD"),
                ("Amount", "Paid:", "$1,500.00"),
            ),
        )


def analyzed_document():
    return analyze_invoice_upload(
        INVOICE_ASSET.read_bytes(),
        filename=INVOICE_ASSET.name,
        ocr_engine=InvoiceOcr(),
        model_adapter=LowScoreExactModel(),
    )


def test_low_score_exact_document_still_requires_explicit_human_review():
    analysis = analyzed_document()
    assert analysis.strict_exact is True
    assert analysis.rule_candidates[0].invoice_number == "FF-10482"
    assert analysis.gate.status.value == "REVIEW_REQUIRED"
    assert analysis.gate.reason_codes == ("LOW_MODEL_CONFIDENCE",)
    assert analysis.gate.verified_identity is None

    rejected = record_human_identity_decision(
        analysis, DocumentReviewDecision.REJECT
    )
    assert rejected.may_activate_lookup is False
    with pytest.raises(UiFlowError, match="explicit human"):
        prepare_procurement(rejected)


def test_full_controlled_flow_needs_human_then_operator_then_verified_proof():
    analysis = analyzed_document()
    human = record_human_identity_decision(
        analysis, DocumentReviewDecision.CONFIRM
    )
    prepared = prepare_procurement(human)
    assert prepared.looked_up_invoice.invoice_number == "FF-10482"
    assert prepared.verification.result is VerifierResult.REQUIRES_OPERATOR
    assert prepared.scenario.initial_state.cash_minor == 500_000
    assert prepared.scenario.initial_state.invoices[0].payment_status is (
        InvoicePaymentStatus.UNPAID
    )

    simulation = approve_and_simulate(prepared)
    assert simulation.state_after.cash_minor == 100_000
    assert simulation.info["simulation_only"] is True
    assert simulation.state_after.invoices[0].payment_status is (
        InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED
    )

    receipt = analyze_receipt_upload(
        simulation,
        RECEIPT_ASSET.read_bytes(),
        filename=RECEIPT_ASSET.name,
        source=PaymentProofSource.SYNTHETIC_FIXTURE_REPLAY,
        provenance="bundled_deterministic_svg_fixture",
        ocr_engine=ReceiptOcr(),
    )
    assert receipt.parsed.status.value == "READY_FOR_PROOF"
    assert receipt.proof_gate.closes_obligation is True
    assert receipt.simulation.environment.state.invoices[0].payment_status is (
        InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED
    )

    confirmed = confirm_verified_payment(receipt)
    assert confirmed.payment_status is InvoicePaymentStatus.PAID_CONFIRMED
    assert confirmed.state_after.cash_minor == 100_000
    assert confirmed.state_after.day == 1


def test_receipt_cannot_run_before_operator_approved_simulation():
    with pytest.raises(UiFlowError, match="approved simulated step"):
        analyze_receipt_upload(  # type: ignore[arg-type]
            object(),
            RECEIPT_ASSET.read_bytes(),
            filename=RECEIPT_ASSET.name,
            source=PaymentProofSource.OPERATOR_UPLOAD,
            provenance="test",
            ocr_engine=ReceiptOcr(),
        )


def test_failed_ocr_does_not_initialize_the_lazy_model():
    calls = []

    def factory():
        calls.append(True)
        raise AssertionError("model factory must not run after failed OCR")

    adapter = RyanInvoiceAdapter(extractor_factory=factory)

    class MissingOcr:
        def run(self, image):
            return make_ocr(image, (), status=OcrStatus.UNAVAILABLE)

    result = analyze_invoice_upload(
        INVOICE_ASSET.read_bytes(),
        filename=INVOICE_ASSET.name,
        ocr_engine=MissingOcr(),
        model_adapter=adapter,
    )
    assert calls == []
    assert result.ocr.status is OcrStatus.UNAVAILABLE
    assert result.model_run.status is InvoiceModelRunStatus.FAILED
    assert result.gate.status.value == "REVIEW_REQUIRED"


def test_cached_ryan_adapter_is_singleton_under_concurrent_access(monkeypatch):
    import procureagent.ui_adapters as adapters

    created = []

    class LightweightAdapter:
        pass

    def factory():
        created.append(True)
        return LightweightAdapter()

    _reset_cached_ryan_adapter_for_tests()
    monkeypatch.setattr(adapters, "RyanInvoiceAdapter", factory)
    with ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(lambda _: get_cached_ryan_adapter(), range(24)))
    assert len(created) == 1
    assert len({id(item) for item in instances}) == 1
    _reset_cached_ryan_adapter_for_tests()


# ---------------------------------------------------------------------------
# Governed multi-day episode (operator-gated, no auto-advance)
# ---------------------------------------------------------------------------


CASHFLOW_SCENARIO = ROOT / "data/procureagent/scenario_cashflow_v1.json"
PACKRIGHT = InvoiceIdentity("packright", "PR-15007")
CLEANPRO = InvoiceIdentity("cleanpro", "CP-70019")


@pytest.fixture
def episode():
    return start_episode(load_scenario(CASHFLOW_SCENARIO))


def actions_of(proposal):
    return {item.identity: item.action for item in proposal.batch.recommendations}


def test_prepare_procurement_without_an_environment_still_plans_day_zero():
    """The original single-step call signature must keep working untouched."""

    scenario = load_locked_scenario()
    prepared_state = scenario.initial_state
    proposal = propose_day(start_episode(scenario).environment)

    assert proposal.day == 0
    assert proposal.state_version == prepared_state.state_version
    assert len(proposal.batch.recommendations) == len(prepared_state.active_invoices)


def test_propose_day_plans_against_live_state_not_the_initial_state(episode):
    approve_and_simulate(propose_day(episode.environment))
    proposal = propose_day(episode.environment)

    assert proposal.day == 1
    assert proposal.state_version == 2
    # Fresh Farms and Prime Foods were paid on day 0 and have left the batch.
    assert set(actions_of(proposal)) == {PACKRIGHT, CLEANPRO}


def test_seven_day_episode_advances_only_on_explicit_operator_approval(episode):
    environment = episode.environment
    steps = 0
    while not episode.finished:
        before = environment.state.day
        proposal = propose_day(environment, identity_ledger=episode.identity_ledger)
        assert environment.state.day == before, "proposing must not advance time"
        episode.history.append(approve_and_simulate(proposal))
        steps += 1

    assert steps == 7
    assert episode.day == 7
    assert environment.truncated and not environment.terminated
    assert episode.steps_taken == 7
    assert len(environment.audit_log) == 7


def test_revenue_lets_the_agent_choose_a_payment_day(episode):
    """The 'when' decision: deferred while unaffordable, paid the day it fits."""

    paid_on = None
    while not episode.finished:
        proposal = propose_day(episode.environment, identity_ledger=episode.identity_ledger)
        chosen = actions_of(proposal).get(PACKRIGHT)
        approve_and_simulate(proposal)
        if chosen is ProcurementAction.PAY:
            paid_on = proposal.day
            break

    assert paid_on == 2


def test_days_without_any_pay_still_require_an_operator_commit(episode):
    approve_and_simulate(propose_day(episode.environment))
    proposal = propose_day(episode.environment)

    assert not proposal.commits_cash
    assert proposal.verification.result is VerifierResult.REQUIRES_OPERATOR
    assert "OPERATOR_COMMIT_REQUIRED" in proposal.verification.reason_codes

    run = approve_and_simulate(proposal)
    assert run.info["cash_after_minor"] == (
        run.info["cash_before_minor"] + run.info["cash_inflow_minor"]
    )


def test_reject_records_a_decision_and_changes_nothing(episode):
    proposal = propose_day(episode.environment)
    rejected = reject_day(proposal, sequence=1)

    assert not rejected.state_changed
    assert rejected.state_version_before == rejected.state_version_after
    assert rejected.cash_before_minor == rejected.cash_after_minor
    assert episode.day == 0
    assert episode.environment.audit_log == ()  # step was never called


def test_repeated_rejections_produce_distinct_decision_ids(episode):
    proposal = propose_day(episode.environment)
    first = reject_day(proposal, sequence=1)
    second = reject_day(proposal, sequence=2)

    assert first.operator_decision.decision_id != second.operator_decision.decision_id


def test_modify_mints_a_new_batch_id_that_must_be_reverified(episode):
    proposal = propose_day(episode.environment)
    modified = modify_and_reverify(proposal, {CLEANPRO: ProcurementAction.DEFER})

    assert modified.batch.batch_id != proposal.batch.batch_id
    assert modified.origin == "OPERATOR_MODIFIED"
    assert modified.revision == 1
    assert modified.operator_decisions
    cleanpro = next(
        item for item in modified.batch.recommendations if item.identity == CLEANPRO
    )
    assert "OPERATOR_MODIFIED" in cleanpro.reason_codes


def test_modify_that_breaks_a_hard_constraint_is_blocked_and_cannot_step(episode):
    proposal = propose_day(episode.environment)
    over_budget = modify_and_reverify(proposal, {PACKRIGHT: ProcurementAction.PAY})

    assert over_budget.verification.result is VerifierResult.BLOCKED
    assert "OVER_BUDGET" in over_budget.verification.reason_codes
    with pytest.raises(UiFlowError):
        approve_and_simulate(over_budget)
    assert episode.day == 0
    assert episode.environment.state.state_version == 1


def test_modify_a_conflicted_invoice_to_pay_is_blocked(episode):
    proposal = propose_day(episode.environment)
    conflicted = modify_and_reverify(proposal, {CLEANPRO: ProcurementAction.PAY})

    assert conflicted.verification.result is VerifierResult.BLOCKED
    assert "UNRESOLVED_BUSINESS_CONTEXT" in conflicted.verification.reason_codes


def test_modify_replacement_batch_id_stays_bounded_across_revisions(episode):
    proposal = propose_day(episode.environment)
    for _ in range(5):
        proposal = modify_and_reverify(proposal, {CLEANPRO: ProcurementAction.DEFER})
        proposal = modify_and_reverify(proposal, {CLEANPRO: ProcurementAction.VERIFY})

    assert len(proposal.batch.batch_id) <= 128


def test_a_stale_proposal_is_refused_before_the_gym_is_touched(episode):
    stale = propose_day(episode.environment)
    approve_and_simulate(propose_day(episode.environment))

    with pytest.raises(UiFlowError, match="no longer matches current state"):
        approve_and_simulate(stale)
    assert episode.environment.state.day == 1


def test_proposing_or_stepping_a_finished_episode_fails_closed(episode):
    while not episode.finished:
        approve_and_simulate(propose_day(episode.environment))

    with pytest.raises(UiFlowError):
        propose_day(episode.environment)


def test_identity_provenance_prefers_human_review_and_labels_the_rest(episode):
    proposal = propose_day(episode.environment)

    assert {record.provenance for record in proposal.identity_provenance} == {
        FIXTURE_REPLAY
    }
    assert all(not record.read_from_document for record in proposal.identity_provenance)
    assert len(proposal.identity_provenance) == len(proposal.batch.recommendations)


def test_a_paid_invoice_drops_out_of_the_verified_identity_set(episode):
    approve_and_simulate(propose_day(episode.environment))
    proposal = propose_day(episode.environment)

    covered = {record.identity for record in proposal.identity_provenance}
    assert InvoiceIdentity("fresh_farms", "FF-10482") not in covered
    assert covered == {PACKRIGHT, CLEANPRO}
