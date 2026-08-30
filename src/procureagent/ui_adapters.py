"""Framework-neutral adapters for ProcureAgent's controlled Streamlit flow.

The module is safe to import from render and test processes.  It performs no
OCR, model loading, network access, policy execution, or state mutation until
one of its explicit functions is called.  The Ryan model singleton is created
and invoked under a re-entrant lock because Streamlit can serve concurrent
sessions from multiple threads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from threading import RLock
from typing import Any, Mapping

from .contracts import (
    DailyRecommendationBatch,
    DocumentReviewDecision,
    DocumentStatus,
    InvoiceIdentity,
    InvoicePaymentStatus,
    OperatorDecision,
    PaymentProofSource,
    ProcurementAction,
    ProcureScenario,
    RestaurantState,
    VerifiedInvoiceIdentity,
    VerifierDecision,
    VerifierResult,
    load_locked_scenario,
    make_audit_id,
)
from .document import (
    AnchoredInvoiceCandidate,
    DocumentGateResult,
    InvoiceModelRun,
    ModelInvoiceCandidate,
    RyanInvoiceAdapter,
    anchored_invoice_candidates,
    gate_document_identity,
)
from .evaluation import (
    ControlledComparison,
    compare_policies,
    fixture_verified_identities,
)
from .governance import (
    ApprovedDailyBatch,
    approve_batch,
    modify_batch,
    reject_batch,
    verify_batch,
)
from .gym import ProcureGym
from .ocr import IngestedImage, OcrResult, TesseractOCR, ingest_image
from .policy import criticality_aware_greedy_v1
from .receipt import ParsedReceipt, PaymentProofGateResult, build_payment_proof, parse_receipt
from .state import lookup_invoice, lookup_verified_invoice, require_invoice


class UiFlowError(RuntimeError):
    """Raised when a UI stage is requested before its safety dependency."""


@dataclass(frozen=True, slots=True)
class OverviewRun:
    scenario: ProcureScenario
    batch: Any
    verified_identities: tuple[VerifiedInvoiceIdentity, ...]
    verification: Any
    comparison: ControlledComparison


@dataclass(frozen=True, slots=True)
class DocumentAnalysis:
    scenario: ProcureScenario
    image: IngestedImage
    ocr: OcrResult
    rule_candidates: tuple[AnchoredInvoiceCandidate, ...]
    model_run: InvoiceModelRun
    gate: DocumentGateResult
    supplier_id: str
    expected_invoice_number: str
    selected_model_candidate: ModelInvoiceCandidate | None
    strict_exact: bool


@dataclass(frozen=True, slots=True)
class HumanIdentityDecision:
    analysis: DocumentAnalysis
    review_id: str
    decision: DocumentReviewDecision
    reviewed_invoice_number: str | None
    verified_identity: VerifiedInvoiceIdentity | None
    gate_reason_codes: tuple[str, ...]

    @property
    def may_activate_lookup(self) -> bool:
        return self.verified_identity is not None and self.decision in {
            DocumentReviewDecision.CONFIRM,
            DocumentReviewDecision.CORRECT,
        }


HUMAN_REVIEWED_DOCUMENT = "HUMAN_REVIEWED_DOCUMENT"
FIXTURE_REPLAY = "FIXTURE_REPLAY"


@dataclass(frozen=True, slots=True)
class VerifiedIdentityRecord:
    """One verified identity plus how it was actually established.

    The distinction is the whole point: an identity a person confirmed against
    a real document that real OCR and the local model read is evidence; a
    fixture identity is a lookup convenience. Days after the first re-use the
    recorded day-0 decisions, which asserts only that this invoice number was
    verified against a document once -- never that the supplier re-sent it.
    """

    verified: VerifiedInvoiceIdentity
    provenance: str
    reviewed_on_day: int | None = None
    document_id: str | None = None
    review_id: str | None = None
    ocr_engine: str | None = None
    model_version: str | None = None
    image_sha256: str | None = None

    @property
    def identity(self) -> InvoiceIdentity:
        return self.verified.identity

    @property
    def read_from_document(self) -> bool:
        return self.provenance == HUMAN_REVIEWED_DOCUMENT


def record_from_human_decision(
    decision: HumanIdentityDecision, *, day: int
) -> VerifiedIdentityRecord:
    """Capture the perception evidence behind one human-confirmed identity."""

    verified = decision.verified_identity
    if verified is None:
        raise UiFlowError("only a CONFIRM or CORRECT decision carries an identity")
    analysis = decision.analysis
    model_run = analysis.model_run
    return VerifiedIdentityRecord(
        verified=verified,
        provenance=HUMAN_REVIEWED_DOCUMENT,
        reviewed_on_day=day,
        document_id=analysis.image.document_id,
        review_id=decision.review_id,
        ocr_engine=f"{analysis.ocr.engine}:{analysis.ocr.engine_version}",
        model_version=getattr(model_run, "model_version", None),
        image_sha256=analysis.image.sha256,
    )


def resolve_verified_identities(
    state: RestaurantState,
    ledger: Mapping[InvoiceIdentity, VerifiedIdentityRecord] | None = None,
) -> tuple[tuple[VerifiedInvoiceIdentity, ...], tuple[VerifiedIdentityRecord, ...]]:
    """Cover every active invoice, preferring a human-reviewed document.

    Iterating the fixture list rather than the ledger gives three properties for
    free: the result is exactly ``state.active_invoices`` as ``verify_batch``
    demands, a paid invoice's identity disappears the moment it leaves that set,
    and duplicates -- which ``_verified_tuple`` rejects -- are impossible.
    """

    records: list[VerifiedIdentityRecord] = []
    for fixture in fixture_verified_identities(state):
        carried = (ledger or {}).get(fixture.identity)
        records.append(
            carried
            if carried is not None
            else VerifiedIdentityRecord(verified=fixture, provenance=FIXTURE_REPLAY)
        )
    return tuple(item.verified for item in records), tuple(records)


@dataclass(frozen=True, slots=True)
class PreparedProcurement:
    scenario: ProcureScenario
    human_decision: HumanIdentityDecision
    looked_up_invoice: Any
    batch: Any
    verified_identities: tuple[VerifiedInvoiceIdentity, ...]
    verification: Any
    environment: ProcureGym | None = None
    identity_provenance: tuple[VerifiedIdentityRecord, ...] = ()
    revision: int = 0
    origin: str = "POLICY"
    origin_batch_id: str | None = None
    operator_decisions: tuple[OperatorDecision, ...] = ()

    @property
    def day(self) -> int:
        state = (
            self.environment.state if self.environment is not None
            else self.scenario.initial_state
        )
        return state.day

    @property
    def state_version(self) -> int:
        state = (
            self.environment.state if self.environment is not None
            else self.scenario.initial_state
        )
        return state.state_version


@dataclass(frozen=True, slots=True)
class DayProposal:
    """One day's proposal on a live episode, day 1 onward.

    Structurally interchangeable with :class:`PreparedProcurement` so every
    downstream reader can accept either without probing attributes. The day-0
    proposal stays a ``PreparedProcurement`` because only it carries the
    document review that activated the lookup.
    """

    scenario: ProcureScenario
    environment: ProcureGym
    day: int
    state_version: int
    batch: DailyRecommendationBatch
    verified_identities: tuple[VerifiedInvoiceIdentity, ...]
    identity_provenance: tuple[VerifiedIdentityRecord, ...]
    verification: VerifierDecision
    origin: str = "POLICY"
    origin_batch_id: str | None = None
    revision: int = 0
    operator_decisions: tuple[OperatorDecision, ...] = ()
    human_decision: HumanIdentityDecision | None = None
    looked_up_invoice: Any = None

    @property
    def commits_cash(self) -> bool:
        return any(
            item.action is ProcurementAction.PAY for item in self.batch.recommendations
        )


@dataclass(frozen=True, slots=True)
class RejectedDay:
    """Evidence that REJECT changed nothing, asserted from measured values."""

    proposal: PreparedProcurement | DayProposal
    operator_decision: OperatorDecision
    day: int
    state_version_before: int
    state_version_after: int
    cash_before_minor: int
    cash_after_minor: int

    @property
    def state_changed(self) -> bool:
        return (
            self.state_version_before != self.state_version_after
            or self.cash_before_minor != self.cash_after_minor
        )


@dataclass(slots=True)
class SimulationRun:
    prepared: PreparedProcurement | DayProposal
    environment: ProcureGym
    approved_batch: ApprovedDailyBatch
    state_before: Any
    state_after: Any
    reward: Any
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(slots=True)
class EpisodeSession:
    """One long-lived governed episode, held across Streamlit reruns.

    Deliberately mutable: this is the session ledger. It must never be a member
    of the document flow's ``FLOW_KEYS``, or editing an unrelated text input
    would silently rewind the restaurant to day 0.
    """

    environment: ProcureGym
    reset_info: dict[str, Any] = field(default_factory=dict)
    identity_ledger: dict[InvoiceIdentity, VerifiedIdentityRecord] = field(
        default_factory=dict
    )
    history: list[SimulationRun | RejectedDay] = field(default_factory=list)
    confirmed_payments: dict[InvoiceIdentity, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.environment.episode_complete

    @property
    def day(self) -> int:
        return self.environment.state.day

    @property
    def horizon_days(self) -> int:
        return self.environment.horizon_days

    @property
    def steps_taken(self) -> int:
        return sum(isinstance(item, SimulationRun) for item in self.history)

    @property
    def rejections(self) -> int:
        return sum(isinstance(item, RejectedDay) for item in self.history)

    def remember_identity(self, decision: HumanIdentityDecision) -> VerifiedIdentityRecord:
        record = record_from_human_decision(decision, day=self.day)
        self.identity_ledger[record.identity] = record
        return record


@dataclass(frozen=True, slots=True)
class ReceiptAnalysis:
    simulation: SimulationRun
    image: IngestedImage
    ocr: OcrResult
    parsed: ParsedReceipt
    proof_gate: PaymentProofGateResult
    source: PaymentProofSource
    provenance: str


@dataclass(frozen=True, slots=True)
class ConfirmedPayment:
    receipt_analysis: ReceiptAnalysis
    state_before: Any
    state_after: Any
    payment_status: InvoicePaymentStatus


_MODEL_LOCK = RLock()
_CACHED_RYAN_ADAPTER: RyanInvoiceAdapter | None = None


def get_cached_ryan_adapter() -> RyanInvoiceAdapter:
    """Return one lazy Ryan adapter without loading model weights."""

    global _CACHED_RYAN_ADAPTER
    with _MODEL_LOCK:
        if _CACHED_RYAN_ADAPTER is None:
            _CACHED_RYAN_ADAPTER = RyanInvoiceAdapter()
        return _CACHED_RYAN_ADAPTER


def _reset_cached_ryan_adapter_for_tests() -> None:
    """Clear only the lightweight singleton reference for isolated tests."""

    global _CACHED_RYAN_ADAPTER
    with _MODEL_LOCK:
        _CACHED_RYAN_ADAPTER = None


def _selected_model_candidate(run: InvoiceModelRun) -> ModelInvoiceCandidate | None:
    if not run.candidates:
        return None
    return max(
        run.candidates,
        key=lambda item: (
            item.minimum_confidence,
            item.mean_confidence,
            item.mean_margin,
            -item.word_indices[0],
        ),
    )


def _run_model(
    image: IngestedImage,
    ocr: OcrResult,
    model_adapter: Any | None,
) -> InvoiceModelRun:
    if model_adapter is not None:
        return model_adapter.run(image, ocr)
    # Loading and inference share the lock: most torch model objects are not
    # safe to mutate or invoke concurrently during a first-load race.
    with _MODEL_LOCK:
        return get_cached_ryan_adapter().run(image, ocr)


def analyze_invoice_upload(
    image_bytes: bytes,
    *,
    filename: str,
    supplier_id: str = "fresh_farms",
    expected_invoice_number: str = "FF-10482",
    scenario: ProcureScenario | None = None,
    ocr_engine: Any | None = None,
    model_adapter: Any | None = None,
) -> DocumentAnalysis:
    """Run ingestion, local OCR, Ryan's model, strict exact, and frozen gate."""

    selected = scenario or load_locked_scenario()
    if not isinstance(expected_invoice_number, str) or not expected_invoice_number:
        raise UiFlowError("expected invoice number must be non-empty text")
    image = ingest_image(image_bytes, original_filename=filename)
    ocr = (ocr_engine or TesseractOCR()).run(image)
    rules = anchored_invoice_candidates(ocr)
    model_run = _run_model(image, ocr, model_adapter)
    gate = gate_document_identity(
        document_id=image.document_id,
        supplier_id=supplier_id,
        supplier_confirmed=True,
        known_supplier_ids=(supplier.supplier_id for supplier in selected.suppliers),
        ocr=ocr,
        rule_candidates=rules,
        model_run=model_run,
        # The scenario is the immutable lookup database.  Its records have not
        # been activated by this document and therefore are not duplicates.
        active_identities=(),
    )
    model_candidate = _selected_model_candidate(model_run)
    observed = (
        model_candidate.candidate.invoice_number if model_candidate is not None else None
    )
    return DocumentAnalysis(
        scenario=selected,
        image=image,
        ocr=ocr,
        rule_candidates=rules,
        model_run=model_run,
        gate=gate,
        supplier_id=supplier_id,
        expected_invoice_number=expected_invoice_number,
        selected_model_candidate=model_candidate,
        strict_exact=observed == expected_invoice_number,
    )


def record_human_identity_decision(
    analysis: DocumentAnalysis,
    decision: DocumentReviewDecision | str,
    *,
    corrected_invoice_number: str | None = None,
) -> HumanIdentityDecision:
    """Record an explicit CONFIRM/CORRECT/REJECT decision after the gate."""

    if not isinstance(analysis, DocumentAnalysis):
        raise UiFlowError("human review requires DocumentAnalysis")
    try:
        selected_decision = (
            decision
            if isinstance(decision, DocumentReviewDecision)
            else DocumentReviewDecision(decision)
        )
    except (TypeError, ValueError) as exc:
        raise UiFlowError("unsupported human document decision") from exc

    if selected_decision is DocumentReviewDecision.REJECT:
        return HumanIdentityDecision(
            analysis=analysis,
            review_id=make_audit_id(
                "document-review", analysis.image.sha256[:16], selected_decision.value
            ),
            decision=selected_decision,
            reviewed_invoice_number=None,
            verified_identity=None,
            gate_reason_codes=analysis.gate.reason_codes,
        )

    if selected_decision is DocumentReviewDecision.CONFIRM:
        candidate = analysis.selected_model_candidate
        if candidate is None:
            raise UiFlowError("CONFIRM requires one displayed model candidate")
        invoice_number = candidate.candidate.invoice_number
        status = DocumentStatus.CONFIRMED
    else:
        if not isinstance(corrected_invoice_number, str) or not corrected_invoice_number.strip():
            raise UiFlowError("CORRECT requires a non-empty invoice number")
        if corrected_invoice_number != corrected_invoice_number.strip():
            raise UiFlowError("corrected invoice number cannot have outer whitespace")
        invoice_number = corrected_invoice_number
        status = DocumentStatus.CORRECTED

    identity = InvoiceIdentity(analysis.supplier_id, invoice_number)
    if lookup_invoice(analysis.scenario.initial_state, identity) is None:
        raise UiFlowError("human-reviewed composite identity is absent from locked lookup")
    verified = VerifiedInvoiceIdentity(
        document_id=analysis.image.document_id,
        supplier_id=identity.supplier_id,
        invoice_number=identity.invoice_number,
        status=status,
    )
    return HumanIdentityDecision(
        analysis=analysis,
        review_id=make_audit_id(
            "document-review",
            analysis.image.sha256[:16],
            selected_decision.value,
            invoice_number,
        ),
        decision=selected_decision,
        reviewed_invoice_number=invoice_number,
        verified_identity=verified,
        gate_reason_codes=analysis.gate.reason_codes,
    )


def prepare_procurement(
    decision: HumanIdentityDecision,
    *,
    environment: ProcureGym | None = None,
    identity_ledger: Mapping[InvoiceIdentity, VerifiedIdentityRecord] | None = None,
) -> PreparedProcurement:
    """Activate exact lookup and run the real policy plus batch verifier.

    ``environment=None`` plans against the scenario's initial state, which is
    the original single-day behaviour. Supplying a live environment plans
    against its current state instead.
    """

    if not isinstance(decision, HumanIdentityDecision) or not decision.may_activate_lookup:
        raise UiFlowError("lookup requires explicit human CONFIRM or CORRECT")
    verified_identity = decision.verified_identity
    if verified_identity is None:  # narrows for type checkers and stays fail closed
        raise UiFlowError("verified identity is missing")
    scenario = decision.analysis.scenario
    planning_state = (
        scenario.initial_state if environment is None else environment.state
    )
    looked_up = lookup_verified_invoice(planning_state, verified_identity)
    if looked_up is None:
        raise UiFlowError("exact composite lookup returned no invoice")

    ledger = dict(identity_ledger or {})
    ledger[verified_identity.identity] = ledger.get(
        verified_identity.identity,
        VerifiedIdentityRecord(
            verified=verified_identity,
            provenance=HUMAN_REVIEWED_DOCUMENT,
            reviewed_on_day=planning_state.day,
            document_id=decision.analysis.image.document_id,
            review_id=decision.review_id,
        ),
    )
    verified, provenance = resolve_verified_identities(planning_state, ledger)
    batch = criticality_aware_greedy_v1(planning_state)
    verification = verify_batch(planning_state, batch, verified)
    return PreparedProcurement(
        scenario=scenario,
        human_decision=decision,
        looked_up_invoice=looked_up,
        batch=batch,
        verified_identities=verified,
        verification=verification,
        environment=environment,
        identity_provenance=provenance,
        origin_batch_id=batch.batch_id,
    )


def start_episode(
    scenario: ProcureScenario | None = None, *, seed: int | None = None
) -> EpisodeSession:
    """Create one governed episode and reset it to day 0."""

    selected = scenario or load_locked_scenario()
    environment = ProcureGym(selected)
    _, info = environment.reset(seed=seed)
    return EpisodeSession(environment=environment, reset_info=dict(info))


def propose_day(
    environment: ProcureGym,
    *,
    identity_ledger: Mapping[InvoiceIdentity, VerifiedIdentityRecord] | None = None,
) -> DayProposal:
    """Plan one day against the live episode state. Mutates nothing."""

    if not isinstance(environment, ProcureGym):
        raise UiFlowError("a day proposal requires a live ProcureGym")
    if environment.episode_complete:
        raise UiFlowError("episode is complete; restart it to run another")
    state = environment.state
    if not state.active_invoices:
        raise UiFlowError("no active invoices remain to plan")

    verified, provenance = resolve_verified_identities(state, identity_ledger)
    batch = criticality_aware_greedy_v1(state)
    verification = verify_batch(state, batch, verified)
    return DayProposal(
        scenario=environment.scenario,
        environment=environment,
        day=state.day,
        state_version=state.state_version,
        batch=batch,
        verified_identities=verified,
        identity_provenance=provenance,
        verification=verification,
        origin_batch_id=batch.batch_id,
    )


def approve_and_simulate(
    prepared: PreparedProcurement | DayProposal,
    *,
    decision_id: str | None = None,
) -> SimulationRun:
    """Attach explicit operator approval and perform exactly one gym step."""

    if not isinstance(prepared, (PreparedProcurement, DayProposal)):
        raise UiFlowError("approval requires a prepared day proposal")
    if prepared.verification.result is VerifierResult.BLOCKED:
        raise UiFlowError("blocked batch cannot enter ProcureGym")
    environment = prepared.environment or ProcureGym(prepared.scenario)
    if environment.episode_complete:
        raise UiFlowError("episode is complete; restart it to run another")

    # Re-verify against live state before approving. The gym re-verifies too and
    # remains the real boundary; doing it here turns an unrecoverable
    # GymTransitionError mid-demo into a catchable, explainable UI error.
    fresh = verify_batch(environment.state, prepared.batch, prepared.verified_identities)
    if fresh.verified_batch_id != prepared.batch.batch_id:
        raise UiFlowError(
            "proposal no longer matches current state; re-propose the day: "
            + ",".join(fresh.reason_codes)
        )

    approved = approve_batch(
        prepared.batch,
        fresh,
        prepared.verified_identities,
        decision_id=decision_id
        or make_audit_id("approve", prepared.batch.batch_id, environment.state.day),
    )
    state_before = environment.state
    state_after, reward, terminated, truncated, info = environment.step(approved)
    return SimulationRun(
        prepared=prepared,
        environment=environment,
        approved_batch=approved,
        state_before=state_before,
        state_after=state_after,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


def modify_and_reverify(
    proposal: PreparedProcurement | DayProposal,
    changes: Mapping[InvoiceIdentity, ProcurementAction],
) -> DayProposal:
    """Apply an operator MODIFY and send the replacement back through the verifier.

    MODIFY changes an action enum, never an amount. The replacement gets a new
    batch ID and must clear the verifier again before it can be approved -- a
    modification that breaks a hard constraint comes back BLOCKED.
    """

    if not isinstance(proposal, (PreparedProcurement, DayProposal)):
        raise UiFlowError("MODIFY requires a prepared day proposal")
    environment = proposal.environment
    if environment is None:
        raise UiFlowError("MODIFY requires a live episode")
    if not changes:
        raise UiFlowError("MODIFY must change at least one action")

    revision = proposal.revision + 1
    # Derive from the original policy batch, not the currently displayed one, so
    # chained modifications cannot grow the ID toward the 128-character limit.
    origin_batch_id = proposal.origin_batch_id or proposal.batch.batch_id
    replacement, decision = modify_batch(
        proposal.batch,
        dict(changes),
        replacement_batch_id=make_audit_id("modify", origin_batch_id, revision),
    )
    state = environment.state
    verification = verify_batch(state, replacement, proposal.verified_identities)
    return DayProposal(
        scenario=proposal.scenario,
        environment=environment,
        day=state.day,
        state_version=state.state_version,
        batch=replacement,
        verified_identities=proposal.verified_identities,
        identity_provenance=proposal.identity_provenance,
        verification=verification,
        origin="OPERATOR_MODIFIED",
        origin_batch_id=origin_batch_id,
        revision=revision,
        operator_decisions=(*proposal.operator_decisions, decision),
        human_decision=getattr(proposal, "human_decision", None),
        looked_up_invoice=getattr(proposal, "looked_up_invoice", None),
    )


def reject_day(
    proposal: PreparedProcurement | DayProposal, *, sequence: int = 1
) -> RejectedDay:
    """Record an operator REJECT. Calls no gym method, so nothing can change.

    ``sequence`` is required for honesty rather than decoration: reject_batch's
    default decision ID is a pure function of the batch ID, so rejecting the
    same day twice would otherwise emit two audit entries with identical IDs
    and read as a single event.
    """

    if not isinstance(proposal, (PreparedProcurement, DayProposal)):
        raise UiFlowError("REJECT requires a prepared day proposal")
    environment = proposal.environment
    state = (
        environment.state if environment is not None else proposal.scenario.initial_state
    )
    decision = reject_batch(
        proposal.batch,
        decision_id=make_audit_id("reject", proposal.batch.batch_id, sequence),
    )
    after = environment.state if environment is not None else state
    return RejectedDay(
        proposal=proposal,
        operator_decision=decision,
        day=state.day,
        state_version_before=state.state_version,
        state_version_after=after.state_version,
        cash_before_minor=state.cash_minor,
        cash_after_minor=after.cash_minor,
    )


def analyze_receipt_upload(
    simulation: SimulationRun,
    image_bytes: bytes,
    *,
    filename: str,
    source: PaymentProofSource | str,
    provenance: str,
    ocr_engine: Any | None = None,
) -> ReceiptAnalysis:
    """Run receipt OCR, deterministic parsing, and the full-payment proof gate."""

    if not isinstance(simulation, SimulationRun):
        raise UiFlowError("receipt proof requires an approved simulated step")
    try:
        selected_source = (
            source if isinstance(source, PaymentProofSource) else PaymentProofSource(source)
        )
    except (TypeError, ValueError) as exc:
        raise UiFlowError("unsupported payment-proof source") from exc
    image = ingest_image(image_bytes, original_filename=filename)
    ocr = (ocr_engine or TesseractOCR()).run(image)
    parsed = parse_receipt(ocr, known_suppliers=simulation.prepared.scenario.suppliers)
    identity = simulation.prepared.human_decision.verified_identity
    if identity is None:
        raise UiFlowError("receipt proof has no human-verified invoice identity")
    invoice = require_invoice(simulation.environment.state, identity.identity)
    proof_gate = build_payment_proof(
        parsed,
        invoice,
        ocr=ocr,
        known_suppliers=simulation.prepared.scenario.suppliers,
        source=selected_source,
        provenance=provenance,
        seen_receipt_ids=simulation.environment.consumed_receipt_ids,
    )
    return ReceiptAnalysis(
        simulation=simulation,
        image=image,
        ocr=ocr,
        parsed=parsed,
        proof_gate=proof_gate,
        source=selected_source,
        provenance=provenance,
    )


def confirm_verified_payment(receipt: ReceiptAnalysis) -> ConfirmedPayment:
    """Close AP only when the proof gate produced one VERIFIED full proof."""

    if not isinstance(receipt, ReceiptAnalysis) or not receipt.proof_gate.closes_obligation:
        raise UiFlowError("PAID_CONFIRMED requires verified full payment proof")
    proof = receipt.proof_gate.proof
    if proof is None:  # defensive type narrowing
        raise UiFlowError("verified payment proof is missing")
    environment = receipt.simulation.environment
    state_before = environment.state
    state_after = environment.confirm_payment(proof)
    invoice = require_invoice(state_after, proof.identity)
    if invoice.payment_status is not InvoicePaymentStatus.PAID_CONFIRMED:
        raise UiFlowError("payment confirmation did not reach PAID_CONFIRMED")
    return ConfirmedPayment(
        receipt_analysis=receipt,
        state_before=state_before,
        state_after=state_after,
        payment_status=invoice.payment_status,
    )


@lru_cache(maxsize=1)
def load_overview_run() -> OverviewRun:
    """Run the deterministic, model-free P0 policy comparison once per process."""

    scenario = load_locked_scenario()
    verified = fixture_verified_identities(scenario.initial_state)
    batch = criticality_aware_greedy_v1(scenario.initial_state)
    verification = verify_batch(scenario.initial_state, batch, verified)
    comparison = compare_policies(scenario)
    return OverviewRun(
        scenario=scenario,
        batch=batch,
        verified_identities=verified,
        verification=verification,
        comparison=comparison,
    )


__all__ = [
    "ConfirmedPayment",
    "DocumentAnalysis",
    "HumanIdentityDecision",
    "OverviewRun",
    "PreparedProcurement",
    "ReceiptAnalysis",
    "SimulationRun",
    "UiFlowError",
    "analyze_invoice_upload",
    "analyze_receipt_upload",
    "approve_and_simulate",
    "confirm_verified_payment",
    "get_cached_ryan_adapter",
    "load_overview_run",
    "prepare_procurement",
    "record_human_identity_decision",
]
