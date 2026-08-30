"""Framework-neutral adapters for ProcureAgent's controlled Streamlit flow.

The module is safe to import from render and test processes.  It performs no
OCR, model loading, network access, policy execution, or state mutation until
one of its explicit functions is called.  The Ryan model singleton is created
and invoked under a re-entrant lock because Streamlit can serve concurrent
sessions from multiple threads.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Any

from .contracts import (
    ContractValidationError,
    DocumentReviewDecision,
    DocumentStatus,
    InvoiceIdentity,
    InvoicePaymentStatus,
    PaymentProofSource,
    ProcureScenario,
    VerifiedInvoiceIdentity,
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
    align_model_token_predictions,
    anchored_invoice_candidates,
    gate_document_identity,
)
from .evaluation import (
    ControlledComparison,
    compare_policies,
    fixture_verified_identities,
)
from .governance import ApprovedDailyBatch, approve_batch, verify_batch
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

    def __post_init__(self) -> None:
        if self.image.document_id != self.ocr.document_id:
            raise ContractValidationError("OCR document is not bound to the ingested image")
        if self.image.document_id != self.model_run.document_id:
            raise ContractValidationError("model run is not bound to the ingested image")
        align_model_token_predictions(self.model_run, self.ocr)


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


@dataclass(frozen=True, slots=True)
class PreparedProcurement:
    scenario: ProcureScenario
    human_decision: HumanIdentityDecision
    looked_up_invoice: Any
    batch: Any
    verified_identities: tuple[VerifiedInvoiceIdentity, ...]
    verification: Any


@dataclass(slots=True)
class SimulationRun:
    prepared: PreparedProcurement
    environment: ProcureGym
    approved_batch: ApprovedDailyBatch
    state_before: Any
    state_after: Any
    reward: Any
    terminated: bool
    truncated: bool
    info: dict[str, Any]


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


def prepare_procurement(decision: HumanIdentityDecision) -> PreparedProcurement:
    """Activate exact lookup and run the real policy plus batch verifier."""

    if not isinstance(decision, HumanIdentityDecision) or not decision.may_activate_lookup:
        raise UiFlowError("lookup requires explicit human CONFIRM or CORRECT")
    verified_identity = decision.verified_identity
    if verified_identity is None:  # narrows for type checkers and stays fail closed
        raise UiFlowError("verified identity is missing")
    scenario = decision.analysis.scenario
    looked_up = lookup_verified_invoice(scenario.initial_state, verified_identity)
    if looked_up is None:
        raise UiFlowError("exact composite lookup returned no invoice")

    fixture_identities = fixture_verified_identities(scenario.initial_state)
    verified = tuple(
        verified_identity if item.identity == verified_identity.identity else item
        for item in fixture_identities
    )
    batch = criticality_aware_greedy_v1(scenario.initial_state)
    verification = verify_batch(scenario.initial_state, batch, verified)
    return PreparedProcurement(
        scenario=scenario,
        human_decision=decision,
        looked_up_invoice=looked_up,
        batch=batch,
        verified_identities=verified,
        verification=verification,
    )


def approve_and_simulate(prepared: PreparedProcurement) -> SimulationRun:
    """Attach explicit operator approval and perform exactly one gym step."""

    if not isinstance(prepared, PreparedProcurement):
        raise UiFlowError("approval requires PreparedProcurement")
    if prepared.verification.result is VerifierResult.BLOCKED:
        raise UiFlowError("blocked batch cannot enter ProcureGym")
    approved = approve_batch(
        prepared.batch,
        prepared.verification,
        prepared.verified_identities,
    )
    environment = ProcureGym(prepared.scenario)
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
