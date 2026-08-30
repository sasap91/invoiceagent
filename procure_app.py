"""ProcureAgent's controlled Streamlit demo and recording surface.

Run with: ``streamlit run procure_app.py``.

The overview executes dependency-light deterministic P0 code. OCR and Ryan's
local model execute only after the /eval document button; simulation and AP
closure each require their own later operator click.
"""

from __future__ import annotations

import hashlib
import html
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demo.procure_scenarios import (  # noqa: E402
    CATEGORY_STATUS,
    DOCUMENT_EVIDENCE,
    FIXTURE_NOTICE,
    INVOICE_FIXTURES,
    PRIMARY_SCENARIO,
    UNKNOWNCO_ADVERSARIAL,
    format_minor,
)
from procureagent.contracts import (  # noqa: E402
    DocumentReviewDecision,
    PaymentProofSource,
    VerifierResult,
)
from procureagent.ui_adapters import (  # noqa: E402
    analyze_invoice_upload,
    analyze_receipt_upload,
    approve_and_simulate,
    confirm_verified_payment,
    load_overview_run,
    prepare_procurement,
    record_human_identity_decision,
)
from procureagent.router_lab import run_router_lab  # noqa: E402


ASSET_DIR = ROOT / "data" / "procureagent" / "assets"
EVAL_DIR = ROOT / "data" / "procureagent" / "eval"
SCENARIO_PATH = ROOT / "data" / "procureagent" / "scenario_v1.json"
INVOICE_PATH = ASSET_DIR / "fresh_farms_invoice.png"
RECEIPT_PATH = ASSET_DIR / "fresh_farms_payment_receipt.png"
MODEL_SMOKE_PATH = EVAL_DIR / "model_smoke_v1.json"
RECEIPT_PROVENANCE_PATH = ASSET_DIR / "receipt_provenance.json"


st.set_page_config(
    page_title="ProcureAgent · Sasa's restaurant",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
  :root {
    --ink:#17332d; --muted:#60736d; --paper:#f7f5ef; --card:#fffdf8;
    --line:#dce4de; --leaf:#13795b; --leaf-soft:#e9f5ef;
    --amber:#a85d05; --amber-soft:#fff3d9; --red:#a13d36; --red-soft:#fcebe8;
  }
  .stApp { background:var(--paper); color:var(--ink); }
  [data-testid="stSidebar"] { background:#edf3ee; border-right:1px solid var(--line); }
  [data-testid="stHeader"] { background:rgba(247,245,239,.85); }
  .block-container { max-width:1240px; padding-top:1.7rem; padding-bottom:4rem; }
  h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
  .pa-hero { position:relative; overflow:hidden; padding:2.1rem 2.3rem; border-radius:24px;
    border:1px solid var(--line); background:linear-gradient(130deg,#fffdf8 0%,#e8f5ed 62%,#fff0da 100%);
    box-shadow:0 18px 55px rgba(31,65,54,.08); margin-bottom:.85rem; }
  .pa-hero:after { content:""; position:absolute; width:230px; height:230px; border-radius:50%;
    border:28px solid rgba(19,121,91,.08); right:-70px; top:-100px; }
  .pa-kicker { color:var(--leaf); font-weight:850; letter-spacing:.15em; text-transform:uppercase; font-size:.7rem; }
  .pa-hero h1 { font-size:clamp(2.4rem,5vw,4.5rem); line-height:.95; margin:.35rem 0 .65rem; }
  .pa-hero p { color:#536963; max-width:780px; font-size:1.04rem; line-height:1.55; margin:0; }
  .pa-badge { display:inline-block; margin-top:1rem; border-radius:999px; padding:.36rem .7rem;
    background:var(--amber-soft); color:#76500b; font-weight:800; font-size:.72rem; letter-spacing:.04em; }
  .pa-section { color:var(--leaf); font-size:.68rem; font-weight:850; letter-spacing:.14em;
    text-transform:uppercase; margin:.25rem 0; }
  .pa-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.85rem; margin:.9rem 0 1.2rem; }
  .pa-card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:1.05rem 1.1rem;
    box-shadow:0 8px 24px rgba(31,65,54,.045); }
  .pa-card-top { display:flex; align-items:flex-start; justify-content:space-between; gap:.65rem; }
  .pa-card h3 { margin:0; font-size:1.08rem; }
  .pa-card .sub { color:var(--muted); font-size:.76rem; margin:.18rem 0 .8rem; }
  .pa-facts { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.45rem; }
  .pa-fact { border-radius:10px; padding:.55rem .6rem; background:#f3f6f2; }
  .pa-fact span { display:block; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-size:.58rem; }
  .pa-fact b { display:block; color:var(--ink); margin-top:.12rem; font-size:.83rem; }
  .pa-action { flex:0 0 auto; border-radius:999px; padding:.27rem .52rem; font-size:.64rem; font-weight:900; }
  .PAY { color:#0a684c; background:var(--leaf-soft); } .DEFER { color:#7b5007; background:var(--amber-soft); }
  .VERIFY { color:#893d36; background:var(--red-soft); }
  .pa-source { color:var(--muted); font-size:.68rem; margin-top:.72rem; border-top:1px solid #edf0ec; padding-top:.6rem; }
  .pa-evidence { background:#172f2b; color:#e9f4ef; border-radius:16px; padding:1.25rem; min-height:170px; }
  .pa-evidence span { color:#9cc7b7; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; }
  .pa-evidence strong { color:white; display:block; font-size:1.7rem; margin:.7rem 0; }
  .pa-evidence mark { background:#f4cb67; color:#38290b; padding:.18rem .38rem; border-radius:4px; }
  .pa-evidence p { color:#bfd2ca; font-size:.82rem; line-height:1.5; margin:.7rem 0 0; }
  .pa-batch { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.72rem; margin:.9rem 0 1.2rem; }
  .pa-batch-card { background:var(--card); border:1px solid var(--line); border-top:4px solid #9aaaa3;
    border-radius:14px; padding:.9rem; }
  .pa-batch-card.PAY { border-top-color:var(--leaf); } .pa-batch-card.DEFER { border-top-color:#d5942f; }
  .pa-batch-card.VERIFY { border-top-color:#c76b61; }
  .pa-batch-card span { font-size:.62rem; letter-spacing:.09em; text-transform:uppercase; font-weight:900; }
  .pa-batch-card b { display:block; color:var(--ink); margin:.38rem 0 .15rem; }
  .pa-batch-card small { color:var(--muted); line-height:1.4; }
  .pa-status { border-left:4px solid var(--leaf); padding:.7rem .85rem; background:var(--card); margin:.45rem 0; border-radius:8px; }
  .pa-status b { color:var(--ink); } .pa-status span { color:var(--muted); font-size:.78rem; }
  .pa-stage { display:flex; gap:.6rem; align-items:flex-start; padding:.7rem .85rem; margin:.5rem 0;
    background:var(--card); border:1px solid var(--line); border-radius:12px; }
  .pa-stage .dot { width:.7rem; height:.7rem; border-radius:50%; margin-top:.28rem; background:#9aaaa3; flex:0 0 auto; }
  .pa-stage.ok .dot { background:var(--leaf); } .pa-stage.review .dot { background:#d5942f; }
  .pa-stage.stop .dot { background:#c8574d; }
  .pa-stage b { color:var(--ink); } .pa-stage span { color:var(--muted); font-size:.78rem; }
  .pa-table-wrap { overflow-x:auto; margin:.55rem 0 1rem; border:1px solid var(--line); border-radius:12px; }
  .pa-table { width:100%; border-collapse:collapse; background:var(--card); font-size:.78rem; }
  .pa-table th { color:#44625a; background:#edf3ee; text-align:left; font-size:.64rem; letter-spacing:.04em; text-transform:uppercase; }
  .pa-table th,.pa-table td { padding:.58rem .68rem; border-bottom:1px solid #edf0ec; white-space:nowrap; }
  .pa-table tr:last-child td { border-bottom:0; }
  @media (max-width:850px) { .pa-grid {grid-template-columns:1fr;} .pa-batch {grid-template-columns:repeat(2,1fr);} }
  @media (max-width:520px) { .pa-batch,.pa-facts {grid-template-columns:1fr;} .pa-hero {padding:1.4rem;} }
</style>
"""


FLOW_KEYS = (
    "eval-document-analysis",
    "eval-human-decision",
    "eval-prepared",
    "eval-simulation",
    "eval-receipt-analysis",
    "eval-confirmed-payment",
)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def render_responsive_image(image_bytes: bytes, *, caption: str) -> None:
    """Use the image-width API supported by both CI and model environments."""

    supports_stretch = "use_container_width" in inspect.signature(st.image).parameters
    st.image(image_bytes, caption=caption, width="stretch" if supports_stretch else 900)


def clear_flow(after: str | None = None) -> None:
    start = 0 if after is None else FLOW_KEYS.index(after) + 1
    for key in FLOW_KEYS[start:]:
        st.session_state.pop(key, None)


def stage_badge(label: str, detail: str, tone: str = "pending") -> None:
    st.markdown(
        f'<div class="pa-stage {esc(tone)}"><span class="dot"></span><div><b>{esc(label)}</b><br><span>{esc(detail)}</span></div></div>',
        unsafe_allow_html=True,
    )


def render_table(rows: list[dict[str, Any]]) -> None:
    """Render small audit tables without a pandas/Arrow runtime dependency."""

    if not rows:
        st.caption("No rows")
        return
    headers = list(rows[0])
    heading = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(row.get(item, ''))}</td>" for item in headers) + "</tr>"
        for row in rows
    )
    st.markdown(
        f'<div class="pa-table-wrap"><table class="pa-table"><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>',
        unsafe_allow_html=True,
    )


try:
    OVERVIEW = load_overview_run()
    OVERVIEW_ERROR = None
except Exception as exc:
    OVERVIEW = None
    OVERVIEW_ERROR = f"{type(exc).__name__}: {exc}"

try:
    ROUTER_RESULT = run_router_lab()
    ROUTER_ERROR = None
except Exception as exc:
    ROUTER_RESULT = None
    ROUTER_ERROR = f"{type(exc).__name__}: {exc}"


def supplier_names() -> dict[str, str]:
    if OVERVIEW is None:
        return {item["supplier_id"]: item["supplier_name"] for item in INVOICE_FIXTURES}
    return {item.supplier_id: item.display_name for item in OVERVIEW.scenario.suppliers}


def render_invoice_cards() -> None:
    names = supplier_names()
    if OVERVIEW is None:
        invoices = INVOICE_FIXTURES
        actions = {item["supplier_id"]: item["hypothesis_action"] for item in invoices}
        cards = []
        for invoice in invoices:
            action = actions[invoice["supplier_id"]]
            cards.append(
                f'<article class="pa-card"><div class="pa-card-top"><div><h3>{esc(invoice["supplier_name"])}</h3>'
                f'<div class="sub">{esc(invoice["category"])} · {esc(invoice["invoice_number"])}</div></div>'
                f'<span class="pa-action {esc(action)}">FALLBACK · {esc(action)}</span></div>'
                f'<div class="pa-facts"><div class="pa-fact"><span>Amount</span><b>{esc(format_minor(invoice["amount_minor"]))}</b></div>'
                f'<div class="pa-fact"><span>Due</span><b>{esc(invoice["due_label"])}</b></div>'
                f'<div class="pa-fact"><span>Inventory</span><b>{invoice["inventory_days_remaining"]} days</b></div></div>'
                '<div class="pa-source">Dependency-light fallback · no backend claim</div></article>'
            )
    else:
        actions = {(item.supplier_id, item.invoice_number): item.action.value for item in OVERVIEW.batch.recommendations}
        cards = []
        for invoice in OVERVIEW.scenario.initial_state.invoices:
            action = actions[(invoice.supplier_id, invoice.invoice_number)]
            due = "Today" if invoice.due_in_days == 0 else (
                f"{abs(invoice.due_in_days)} day overdue" if invoice.due_in_days < 0 else f"In {invoice.due_in_days} days"
            )
            cards.append(
                f'<article class="pa-card"><div class="pa-card-top"><div><h3>{esc(names[invoice.supplier_id])}</h3>'
                f'<div class="sub">{esc(invoice.category.title())} · {esc(invoice.invoice_number)}</div></div>'
                f'<span class="pa-action {esc(action)}">ACTUAL POLICY · {esc(action)}</span></div>'
                f'<div class="pa-facts"><div class="pa-fact"><span>Looked-up amount</span><b>{esc(format_minor(invoice.amount_minor))}</b></div>'
                f'<div class="pa-fact"><span>Due</span><b>{esc(due)}</b></div>'
                f'<div class="pa-fact"><span>Inventory</span><b>{invoice.inventory_days_remaining} days</b></div>'
                f'<div class="pa-fact"><span>Criticality</span><b>{esc(invoice.supplier_criticality.value.title())}</b></div>'
                f'<div class="pa-fact"><span>Lead time</span><b>{invoice.delivery_lead_days} days</b></div>'
                f'<div class="pa-fact"><span>Status</span><b>{esc(invoice.supplier_status.value)}</b></div></div>'
                '<div class="pa-source">synthetic_fixture_lookup · policy computed locally · not OCR extraction</div></article>'
            )
    st.markdown(f'<div class="pa-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_document_evidence() -> None:
    st.markdown('<div class="pa-section">Screen 2 · Static evidence boundary</div>', unsafe_allow_html=True)
    st.subheader("Inspect the boundary before business lookup")
    choices = {item["supplier_name"]: item["supplier_id"] for item in INVOICE_FIXTURES}
    choices[UNKNOWNCO_ADVERSARIAL["supplier_name"]] = UNKNOWNCO_ADVERSARIAL["supplier_id"]
    selected_name = st.selectbox("Document fixture", tuple(choices), key="document-fixture-selector")
    supplier_id = choices[selected_name]
    evidence = UNKNOWNCO_ADVERSARIAL if supplier_id == "unknownco" else DOCUMENT_EVIDENCE[supplier_id]
    if supplier_id == "unknownco":
        st.error("Fail-closed fixture boundary: UnknownCo never reaches canonical lookup, activates no payable, and is excluded from the $6,200 obligations. This static card did not run C2.")
    else:
        st.warning("Static stored evidence only. For actual Tesseract + Ryan model evidence, use the /eval recording tab.")
    left, right = st.columns([1.05, 1])
    with left:
        tokens = " &nbsp; ".join(f"<mark>{esc(token)}</mark>" for token in evidence["evidence_tokens"])
        st.markdown(
            f'<div class="pa-evidence"><span>Stored preview · not OCR output</span><strong>{esc(evidence["supplier_name"])}</strong>{tokens}'
            f'<p>Candidate shown: {esc(evidence["proposed_invoice_number"])}<br>{esc(evidence["disclosure"])}</p></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("**Provenance and gates**")
        st.write(f"Document ID: `{evidence['document_id']}`")
        st.write(f"OCR: **{evidence['ocr_status']}**")
        st.write(f"Local model: **{evidence['model_status']}**")
        st.write(f"Document gate: **{evidence['document_gate_status']}**")
        st.write(f"Lookup: **{evidence['lookup_status']}**")


def render_batch_cards(batch: Any, prefix: str) -> None:
    names = supplier_names()
    cards = []
    for item in batch.recommendations:
        action = item.action.value
        cards.append(
            f'<div class="pa-batch-card {esc(action)}"><span>{esc(action)} · {esc(prefix)}</span>'
            f'<b>{esc(names.get(item.supplier_id, item.supplier_id))}</b><small>{esc(item.invoice_number)} · '
            f'{esc(format_minor(item.amount_minor))}<br>{esc(" · ".join(item.reason_codes))}</small></div>'
        )
    st.markdown(f'<div class="pa-batch">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_batch() -> None:
    st.markdown('<div class="pa-section">Screen 3 · Actual deterministic proposal</div>', unsafe_allow_html=True)
    st.subheader("One daily batch, three possible actions")
    if OVERVIEW is None:
        st.error(f"P0 backend unavailable; no policy/verifier claim. {OVERVIEW_ERROR}")
        return
    st.success("Criticality-Aware Greedy v1 and the batch verifier ran on the locked state. This overview did not approve or commit the batch.")
    render_batch_cards(OVERVIEW.batch, "policy output")
    st.write(f"Policy: `{OVERVIEW.batch.policy_name}:{OVERVIEW.batch.policy_version}` · `{OVERVIEW.batch.policy_type.value}`")
    st.write(f"Verifier: **{OVERVIEW.verification.result.value}** · batch `{OVERVIEW.verification.batch_id}`")
    st.caption("Checks passed: " + " · ".join(OVERVIEW.verification.checks_passed))
    st.warning("Operator approval: NOT RECORDED here · state unchanged · use /eval for the governed mutation path")
    controls = st.columns(3)
    controls[0].radio("Operator decision (overview disabled)", ("APPROVE", "MODIFY", "REJECT"), key="operator-decision-preview", disabled=True, horizontal=True)
    controls[1].button("Verifier already ran", key="run-verifier", disabled=True, use_container_width=True)
    controls[2].button("Commit only in /eval", key="commit-procuregym", disabled=True, use_container_width=True)


def raw_metric_rows(run: Any) -> list[dict[str, Any]]:
    metrics = run.raw_metrics
    return [
        {"Metric": "Reward", "Value": str(run.total_reward)},
        {"Metric": "High-criticality stockout days", "Value": str(metrics.high_criticality_stockout_days)},
        {"Metric": "All stockout days", "Value": str(metrics.stockout_days)},
        {"Metric": "Late fees", "Value": format_minor(metrics.late_fees_minor)},
        {"Metric": "Supplier disruptions", "Value": str(metrics.supplier_disruptions)},
        {"Metric": "Deliveries arrived", "Value": str(metrics.deliveries_arrived)},
        {"Metric": "Unsafe executed batches", "Value": str(run.action_validity.unsafe_executed_batches)},
    ]


def render_router_lab() -> None:
    st.markdown("#### C6 · Router Lab (development evidence only)")
    if ROUTER_RESULT is None:
        st.warning(f"Router Lab result unavailable; no C6 claim. {ROUTER_ERROR}")
        return
    result = ROUTER_RESULT
    st.warning(
        "Narrow scope: 7 hand-authored synthetic development rows, and all 7 context bins "
        "also appeared in training. There is no frozen test or generalization claim. This "
        "routes synthetic invoice identity only—not this uploaded invoice, supplier "
        "prioritization, or payment actions."
    )
    rows = []
    for label, metrics in (
        ("Learned tabular router", result.learned),
        ("Fixed evidence gate", result.fixed_gate),
        ("Always review", result.always_review),
    ):
        rows.append(
            {
                "Development policy": label,
                "Correct automatic": metrics.correct_automatic_identities,
                "Wrong automatic": metrics.wrong_automatic_accepts,
                "Human reviews": metrics.review_count,
                "Local model calls": metrics.local_model_invocations,
                "Total reward": str(metrics.total_reward),
            }
        )
    render_table(rows)
    st.caption(
        f"Dataset `{result.dataset_id}` · development bins seen in training "
        f"{result.development_contexts_seen_in_training}/{result.development_contexts_total} · "
        f"frozen_test_evaluated={result.frozen_test_evaluated}."
    )
    if result.improvement_supported:
        st.info(
            "Within these repeated synthetic context bins only, learned reward exceeds the "
            "two declared baselines with zero additional unsafe accepts."
        )
    else:
        st.info("This development comparison does not support a within-bin reward improvement.")


def render_gym() -> None:
    st.markdown('<div class="pa-section">Screen 4 · Controlled C5 comparison</div>', unsafe_allow_html=True)
    st.subheader("ProcureGym, EDF, and the bounded oracle")
    if OVERVIEW is None:
        st.error(f"Controlled comparison unavailable. {OVERVIEW_ERROR}")
        return
    comparison = OVERVIEW.comparison
    st.success("Actual deterministic benchmark complete from the same locked state and seed. Its fixed auto-approval executor is evaluation-only, not the live operator workflow.")
    cols = st.columns(3)
    cols[0].metric("Criticality-aware reward", str(comparison.criticality_aware.total_reward), f"regret {comparison.criticality_regret}")
    cols[1].metric("EDF reward", str(comparison.earliest_due_first.total_reward), f"regret {comparison.earliest_due_first_regret}")
    cols[2].metric("Oracle reward", str(comparison.schedule_oracle.total_reward), f"{comparison.schedule_oracle.legal_schedules} legal schedules")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Criticality-Aware Greedy v1")
        render_table(raw_metric_rows(comparison.criticality_aware))
    with right:
        st.markdown("#### Earliest Due First")
        render_table(raw_metric_rows(comparison.earliest_due_first))
    st.caption(f"Seed {comparison.seed} · identical state: {comparison.identical_initial_state} · oracle enumerated {comparison.schedule_oracle.enumerated_schedules} bounded schedules.")
    st.info("Three axes stay separate: invoice identity belongs upstream to C2; daily supplier ranking is ordered; payment actions are checked for exact identity, amount, timing, and cash.")
    render_router_lab()


def render_ocr_result(ocr: Any, heading: str) -> None:
    st.markdown(f"##### {heading}")
    cols = st.columns(4)
    cols[0].metric("OCR status", ocr.status.value)
    cols[1].metric("Words", str(len(ocr.words)))
    cols[2].metric("Mean quality", f"{ocr.quality:.3f}")
    cols[3].metric("Runtime", f"{ocr.runtime_ms} ms")
    st.caption(f"Actual module: procureagent.ocr.TesseractOCR · engine: {ocr.engine}:{ocr.engine_version}")
    if ocr.error_code:
        st.error(f"{ocr.error_code}: {ocr.error_message}")
        return
    with st.expander(f"{heading} · all OCR tokens and boxes"):
        render_table(
            [{"#": word.sequence, "token": word.text, "confidence": str(word.confidence),
              "pixel_box": f"({word.pixel_box.x0},{word.pixel_box.y0},{word.pixel_box.x1},{word.pixel_box.y1})",
              "layout_box": f"({word.normalized_box.x0},{word.normalized_box.y0},{word.normalized_box.x1},{word.normalized_box.y1})"}
             for word in ocr.words]
        )
        st.code(ocr.raw_text or "<no OCR text>")


def render_document_analysis(analysis: Any) -> None:
    stage_badge("1 · Image ingested", f"{analysis.image.image_format.value} · {analysis.image.width}×{analysis.image.height} · SHA-256 {analysis.image.sha256}", "ok")
    render_ocr_result(analysis.ocr, "Invoice OCR evidence")
    if analysis.rule_candidates:
        st.success("Anchored rule candidate: " + " · ".join(item.invoice_number for item in analysis.rule_candidates))
        st.caption("Rule evidence: " + " | ".join(" ".join(item.evidence_tokens) for item in analysis.rule_candidates))
    else:
        st.error("Anchored rule returned no candidate")
    run = analysis.model_run
    selected = analysis.selected_model_candidate
    st.markdown("##### Ryan local invoice-number specialist")
    st.caption(f"Actual module: procureagent.document.RyanInvoiceAdapter · model: {run.model_version}")
    if selected is None:
        st.error(f"Model status: {run.status.value} · {run.error_code or 'no candidate'} · {run.error_message or ''}")
    else:
        cols = st.columns(4)
        cols[0].metric("Selected token", selected.candidate.invoice_number)
        cols[1].metric("Entity score", str(selected.minimum_confidence))
        cols[2].metric("Mean margin", str(selected.mean_margin))
        cols[3].metric("Latency", f"{run.latency_ms} ms")
        st.write("Selected OCR evidence: **" + " · ".join(selected.candidate.evidence_tokens) + "**")
        st.caption("Evidence boxes: " + " · ".join(f"({b.x0},{b.y0},{b.x1},{b.y1})" for b in selected.candidate.evidence_boxes))
    if analysis.strict_exact:
        st.success(f"Strict exact: YES · model token equals operator expected `{analysis.expected_invoice_number}`")
    else:
        observed = selected.candidate.invoice_number if selected is not None else "<missing>"
        st.error(f"Strict exact: NO · observed `{observed}` vs expected `{analysis.expected_invoice_number}`")
    gate = analysis.gate
    detail = gate.status.value + (" · " + " · ".join(gate.reason_codes) if gate.reason_codes else "")
    stage_badge("2 · Frozen document gate", detail, "ok" if gate.may_activate_lookup else "review")
    if not gate.may_activate_lookup:
        st.warning("The gate did not verify identity. Exact match and agreement never override frozen score thresholds; a human decision is required.")


def render_prepared(prepared: Any) -> None:
    invoice = prepared.looked_up_invoice
    stage_badge("4 · Composite lookup activated", f"{invoice.supplier_id} + {invoice.invoice_number} · synthetic_fixture_lookup", "ok")
    cols = st.columns(4)
    cols[0].metric("Looked-up AP amount", format_minor(invoice.amount_minor))
    cols[1].metric("Inventory", f"{invoice.inventory_days_remaining} days")
    cols[2].metric("Due", f"{invoice.due_in_days} days")
    cols[3].metric("Criticality", invoice.supplier_criticality.value)
    st.caption("Business context came from exact synthetic lookup after human-reviewed identity; LayoutLMv3 did not extract it.")
    render_batch_cards(prepared.batch, "actual live-lane proposal")
    verification = prepared.verification
    stage_badge("5 · Batch verifier", f"{verification.result.value} · {' · '.join(verification.reason_codes)}", "stop" if verification.result is VerifierResult.BLOCKED else "review")
    st.caption("Checks passed: " + " · ".join(verification.checks_passed))
    st.info("Identity provenance: Fresh Farms uses this session's human-reviewed document. Prime Foods, PackRight, and CleanPro are locked fixture/replay identities for the canonical four-item demo batch.")


def render_simulation(simulation: Any) -> None:
    stage_badge("6 · Operator APPROVE + ProcureGym", f"{simulation.approved_batch.operator_decision.decision.value} · day {simulation.info['day_before']}→{simulation.info['day_after']} · simulation_only=True", "ok")
    cols = st.columns(4)
    cols[0].metric("Cash before", format_minor(simulation.info["cash_before_minor"]))
    cols[1].metric("Cash after", format_minor(simulation.info["cash_after_minor"]))
    cols[2].metric("Step reward", str(simulation.reward))
    cols[3].metric("State version", str(simulation.state_after.state_version))
    st.write("Simulated paid: **" + ", ".join(simulation.info["paid_invoice_numbers"]) + "**")
    st.caption(f"Actual modules: approve_batch → ProcureGym.step · decision `{simulation.approved_batch.operator_decision.decision_id}` · no real money moved.")


def render_axis_scorecard(analysis: Any | None) -> None:
    if OVERVIEW is None:
        return
    comparison = OVERVIEW.comparison
    ranking = comparison.criticality_aware.daily_rankings[0]
    st.markdown("#### Three-axis C5 scorecard")
    axes = st.columns(3)
    with axes[0]:
        st.markdown("**1 · Invoice identity**")
        st.metric("Live exact result", "NOT RUN" if analysis is None else ("STRICT EXACT" if analysis.strict_exact else "NOT EXACT"))
        st.caption("One live document only; gate and human review remain separate.")
    with axes[1]:
        st.markdown("**2 · Prioritization ranking**")
        st.write(" → ".join(item.supplier_id for item in ranking.suppliers))
        st.caption("Actual day-0 ordered ranking; runway and criticality retained below.")
    with axes[2]:
        validity = comparison.criticality_aware.action_validity
        st.markdown("**3 · Payment action**")
        st.metric("Verified / proposed", f"{validity.batches_verified}/{validity.batches_proposed}")
        st.caption(f"Wrong amount {validity.wrong_amount_actions} · over-budget {validity.over_budget_batches} · unsafe executed {validity.unsafe_executed_batches}")
    render_table(
        [{"rank": item.rank, "supplier": item.supplier_id, "invoice": item.invoice_number,
          "runway margin": item.runway_margin_days, "due days": item.due_in_days,
          "criticality": item.criticality.value, "action": item.proposed_action.value,
          "exact amount": format_minor(item.exact_amount_minor)} for item in ranking.suppliers],
    )


def render_receipt_result(receipt: Any) -> None:
    render_ocr_result(receipt.ocr, "Receipt OCR evidence")
    parsed = receipt.parsed
    stage_badge("7 · Deterministic receipt parser", f"{parsed.status.value} · {parsed.extraction_method}", "ok" if parsed.status.value == "READY_FOR_PROOF" else "review")
    fields = {"Receipt ID": parsed.receipt_id, "Supplier": parsed.supplier_name, "Supplier ID": parsed.supplier_id,
              "Invoice": parsed.invoice_number, "Amount": format_minor(parsed.amount_minor) if parsed.amount_minor is not None else None,
              "Currency": parsed.currency, "Paid date": parsed.paid_date}
    render_table([{"Field": key, "Parsed value": value} for key, value in fields.items()])
    gate = receipt.proof_gate
    detail = gate.status.value + (" · " + " · ".join(gate.reason_codes) if gate.reason_codes else "")
    stage_badge("8 · Full-payment proof gate", detail, "ok" if gate.closes_obligation else "stop")
    st.caption("Checks passed: " + (" · ".join(gate.checks_passed) or "none"))
    st.caption(f"Source: {receipt.source.value} · provenance: {receipt.provenance}")
    if gate.closes_obligation:
        st.success("Verified full proof is ready. AP is still SIMULATED_PAYMENT_APPROVED until the separate confirmation click.")
    else:
        st.error("Proof did not satisfy every exact full-payment check. AP remains open.")


def render_recording_kit() -> None:
    st.markdown("#### Recording kit & fixture downloads")
    cols = st.columns(4)
    cols[0].download_button("Download invoice PNG", read_bytes(INVOICE_PATH), INVOICE_PATH.name, "image/png", key="download-invoice")
    cols[1].download_button("Download receipt PNG", read_bytes(RECEIPT_PATH), RECEIPT_PATH.name, "image/png", key="download-receipt")
    cols[2].download_button("Download scenario JSON", read_bytes(SCENARIO_PATH), SCENARIO_PATH.name, "application/json", key="download-scenario")
    cols[3].download_button("Download model smoke JSON", read_bytes(MODEL_SMOKE_PATH), MODEL_SMOKE_PATH.name, "application/json", key="download-model-smoke")
    st.caption("All bundled artifacts are synthetic and safe for demo recording.")


def render_code_provenance() -> None:
    smoke = read_json(MODEL_SMOKE_PATH)
    receipt_meta = read_json(RECEIPT_PROVENANCE_PATH)
    with st.expander("Code, provenance, and deployment", expanded=False):
        st.markdown("**Exact implementation chain**")
        st.code("ingest_image → TesseractOCR.run → RyanInvoiceAdapter.run\n→ gate_document_identity → human CONFIRM/CORRECT/REJECT\n→ lookup_verified_invoice → criticality_aware_greedy_v1 → verify_batch\n→ operator APPROVE → ProcureGym.step\n→ receipt TesseractOCR → parse_receipt → build_payment_proof → ProcureGym.confirm_payment")
        st.markdown("**Model and artifact provenance**")
        st.json({"model": smoke.get("model", {}), "recorded_smoke": smoke.get("result", {}),
                 "smoke_claim_scope": smoke.get("claim_scope", "unavailable"),
                 "receipt_source": receipt_meta.get("source", "unavailable"),
                 "receipt_sha256": receipt_meta.get("sha256", "unavailable"),
                 "fal_generation": receipt_meta.get("fal_generation", {})})
        actual_invoice = read_bytes(INVOICE_PATH)
        recorded_input = smoke.get("input", {})
        recorded_hash = recorded_input.get("sha256") if isinstance(recorded_input, dict) else None
        actual_hash = hashlib.sha256(actual_invoice).hexdigest() if actual_invoice else None
        if recorded_hash and actual_hash and recorded_hash != actual_hash:
            st.warning("Recorded smoke hash differs from the current bundled PNG; this session's live evidence is authoritative for any recording claim.")
        st.markdown("**Deployment**")
        st.code("streamlit run procure_app.py\n# or\ndocker build -t procureagent . && docker run -p 8501:8501 procureagent")
        st.caption("Missing Tesseract or model extras fail closed and remain visible in stage output.")


def render_eval() -> None:
    st.markdown('<div class="pa-section">Controlled recording lane · /eval</div>', unsafe_allow_html=True)
    st.subheader("Document → human review → governed simulation → payment proof")
    st.warning("Nothing heavy runs when this page renders. Tesseract and Ryan's local model run only after the document button. Restaurant mutation and AP closure require separate later clicks.")
    st.info("**AP in plain English:** a supplier invoice is Accounts Payable—money the restaurant owes. ProcureGym records simulated approval; exact receipt proof can then close only that demo obligation. No bank is connected.")
    if st.button("Reset recording flow", key="eval-reset-flow"):
        clear_flow()
        st.session_state.pop("eval-document-input-key", None)
        st.session_state.pop("eval-receipt-input-key", None)
    render_recording_kit()

    st.markdown("#### 1 · Choose and analyze the Fresh Farms invoice")
    source_choice = st.radio("Invoice source", ("Bundled Fresh Farms PNG", "Upload PNG/JPEG"), key="eval-invoice-source", horizontal=True)
    st.selectbox("Confirmed supplier selection", ("fresh_farms",), key="eval-supplier", disabled=True)
    expected = st.text_input("Operator expected invoice number", "FF-10482", key="eval-expected-reference")
    if source_choice == "Upload PNG/JPEG":
        upload = st.file_uploader("Invoice PNG/JPEG", type=("png", "jpg", "jpeg"), key="eval-invoice-upload")
        invoice_bytes = upload.getvalue() if upload is not None else b""
        invoice_name = upload.name if upload is not None else "invoice-upload"
    else:
        invoice_bytes = read_bytes(INVOICE_PATH)
        invoice_name = INVOICE_PATH.name
        if invoice_bytes:
            render_responsive_image(
                invoice_bytes,
                caption="Bundled synthetic Fresh Farms invoice",
            )
        else:
            st.error("Bundled invoice asset is missing; choose upload.")
    input_key = hashlib.sha256(invoice_bytes).hexdigest() + "|" + expected if invoice_bytes else ""
    if st.session_state.get("eval-document-input-key") != input_key:
        clear_flow()
        st.session_state["eval-document-input-key"] = input_key
    if st.button("Run real invoice OCR + Ryan model + frozen gate", key="eval-run-document-adapter", disabled=not invoice_bytes or not expected):
        clear_flow()
        with st.spinner("Running local Tesseract and lazy local model…"):
            try:
                analysis = analyze_invoice_upload(invoice_bytes, filename=invoice_name, supplier_id="fresh_farms", expected_invoice_number=expected)
            except Exception as exc:
                st.error(f"Document pipeline failed closed: {type(exc).__name__}: {exc}")
            else:
                st.session_state["eval-document-analysis"] = analysis
    analysis = st.session_state.get("eval-document-analysis")
    if analysis is None:
        stage_badge("1–2 · Document pipeline", "NOT RUN · click required", "review")
    else:
        render_document_analysis(analysis)

    st.markdown("#### 2 · Explicit human document decision")
    review_choice = st.radio("Document review decision", ("CONFIRM", "CORRECT", "REJECT"), key="eval-document-review-choice", horizontal=True, disabled=analysis is None)
    correction = st.text_input("Correct invoice number", value="FF-10482", key="eval-corrected-reference", disabled=analysis is None or review_choice != "CORRECT")
    existing_human = st.session_state.get("eval-human-decision")
    if existing_human is not None and (existing_human.decision.value != review_choice or (review_choice == "CORRECT" and existing_human.reviewed_invoice_number != correction)):
        st.session_state.pop("eval-human-decision", None)
        clear_flow("eval-human-decision")
    if st.button("Record human document decision", key="eval-record-human-review", disabled=analysis is None):
        clear_flow("eval-document-analysis")
        try:
            human = record_human_identity_decision(analysis, DocumentReviewDecision(review_choice), corrected_invoice_number=correction if review_choice == "CORRECT" else None)
            st.session_state["eval-human-decision"] = human
            if human.may_activate_lookup:
                st.session_state["eval-prepared"] = prepare_procurement(human)
        except Exception as exc:
            st.error(f"Human review failed closed: {type(exc).__name__}: {exc}")
    human = st.session_state.get("eval-human-decision")
    prepared = st.session_state.get("eval-prepared")
    if human is None:
        stage_badge("3 · Human identity review", "WAITING · lookup blocked", "review")
    elif not human.may_activate_lookup:
        stage_badge("3 · Human identity review", f"{human.decision.value} · lookup blocked · review {human.review_id}", "stop")
    else:
        stage_badge("3 · Human identity review", f"{human.decision.value} {human.reviewed_invoice_number} · review {human.review_id}", "ok")
    if prepared is not None:
        render_prepared(prepared)

    st.markdown("#### 3 · Explicit operator approval")
    simulation = st.session_state.get("eval-simulation")
    can_approve = prepared is not None and prepared.verification.result is not VerifierResult.BLOCKED and simulation is None
    if prepared is not None and simulation is None:
        st.warning("Verifier ran, but state is unchanged. Only the button below records APPROVE and advances one simulated day.")
    if st.button("APPROVE batch & advance ProcureGym one day", key="eval-approve-batch", disabled=not can_approve):
        clear_flow("eval-prepared")
        try:
            simulation = approve_and_simulate(prepared)
        except Exception as exc:
            st.error(f"Approval/simulation failed closed: {type(exc).__name__}: {exc}")
        else:
            st.session_state["eval-simulation"] = simulation
    simulation = st.session_state.get("eval-simulation")
    if simulation is None:
        stage_badge("6 · Operator + ProcureGym", "NOT COMMITTED · restaurant state unchanged", "review")
    else:
        render_simulation(simulation)
    render_axis_scorecard(analysis)

    st.markdown("#### 4 · Receipt OCR, proof gate, and AP closure")
    confirmed = st.session_state.get("eval-confirmed-payment")
    receipt_source = st.radio(
        "Receipt source",
        ("Bundled deterministic receipt PNG", "Upload PNG/JPEG"),
        key="eval-receipt-source",
        horizontal=True,
        disabled=confirmed is not None,
    )
    if receipt_source == "Upload PNG/JPEG":
        receipt_upload = st.file_uploader(
            "Receipt PNG/JPEG",
            type=("png", "jpg", "jpeg"),
            key="eval-receipt-upload",
            disabled=confirmed is not None,
        )
        receipt_bytes = receipt_upload.getvalue() if receipt_upload is not None else b""
        receipt_name = receipt_upload.name if receipt_upload is not None else "receipt-upload"
        proof_source = PaymentProofSource.OPERATOR_UPLOAD
        provenance = f"operator_upload:{receipt_name}"
    else:
        receipt_bytes = read_bytes(RECEIPT_PATH)
        receipt_name = RECEIPT_PATH.name
        proof_source = PaymentProofSource.SYNTHETIC_FIXTURE_REPLAY
        provenance = "bundled_deterministic_svg_fixture; see receipt_provenance.json"
        if receipt_bytes:
            render_responsive_image(
                receipt_bytes,
                caption="Bundled deterministic synthetic receipt · Fal attempt did not complete",
            )
        else:
            st.error("Bundled receipt asset is missing; choose upload.")
    receipt_key = hashlib.sha256(receipt_bytes).hexdigest() if receipt_bytes else ""
    if confirmed is None and st.session_state.get("eval-receipt-input-key") != receipt_key:
        st.session_state.pop("eval-receipt-analysis", None)
        st.session_state["eval-receipt-input-key"] = receipt_key
    if simulation is None:
        st.info("Receipt proof is locked until operator-approved ProcureGym marks Fresh Farms SIMULATED_PAYMENT_APPROVED.")
    run_receipt = st.button(
        "Run real receipt OCR + deterministic proof gate",
        key="eval-run-receipt-adapter",
        disabled=simulation is None or not receipt_bytes or confirmed is not None,
    )
    if run_receipt and confirmed is None and simulation is not None and receipt_bytes:
        try:
            receipt = analyze_receipt_upload(simulation, receipt_bytes, filename=receipt_name, source=proof_source, provenance=provenance)
        except Exception as exc:
            st.session_state.pop("eval-receipt-analysis", None)
            st.error(f"Receipt pipeline failed closed: {type(exc).__name__}: {exc}")
        else:
            st.session_state["eval-receipt-analysis"] = receipt
    receipt = st.session_state.get("eval-receipt-analysis")
    if receipt is None:
        stage_badge("7–8 · Receipt and proof", "NOT RUN · AP remains open", "review")
    else:
        render_receipt_result(receipt)
    confirm_payment = st.button(
        "Confirm verified full proof → PAID_CONFIRMED",
        key="eval-confirm-payment",
        disabled=receipt is None or not receipt.proof_gate.closes_obligation or confirmed is not None,
    )
    if (
        confirm_payment
        and confirmed is None
        and receipt is not None
        and receipt.proof_gate.closes_obligation
    ):
        try:
            confirmed = confirm_verified_payment(receipt)
        except Exception as exc:
            st.error(f"Payment confirmation failed closed: {type(exc).__name__}: {exc}")
        else:
            st.session_state["eval-confirmed-payment"] = confirmed
            # Refresh immediately so receipt controls visibly lock in the same
            # operator interaction that closes the simulated obligation.
            st.rerun()
    confirmed = st.session_state.get("eval-confirmed-payment")
    if confirmed is None:
        stage_badge("9 · AP lifecycle", "OPEN or SIMULATED_PAYMENT_APPROVED · no real money", "review")
    else:
        stage_badge("9 · AP lifecycle", f"{confirmed.payment_status.value} · state version {confirmed.state_after.state_version}", "ok")
        st.success("PAID_CONFIRMED in the simulated AP ledger · verified full proof consumed · no real bank payment")
    render_code_provenance()


def render_task_status() -> None:
    st.markdown('<div class="pa-section">Delivery map</div>', unsafe_allow_html=True)
    st.subheader("Claimed task categories")
    st.caption("Claims identify responsibility; delivery labels still require their category done tests.")
    for item in CATEGORY_STATUS:
        st.markdown(f'<div class="pa-status"><b>{esc(item["id"])} · {esc(item["category"])}</b><br><span>{esc(item["owners"])} · {esc(item["delivery"])}</span></div>', unsafe_allow_html=True)
    if OVERVIEW is None:
        st.error(f"P0 backend unavailable: {OVERVIEW_ERROR}")
    else:
        st.success("C0/C3 contracts, C4 policy/governance, and C5 comparison imported and executed locally.")
        st.info("C1/C2 OCR and model adapters are installed but stay lazy until the /eval click.")
    if ROUTER_RESULT is None:
        st.warning(f"C6 Router Lab unavailable; no result claimed. {ROUTER_ERROR}")
    else:
        st.info(
            "C6 Router Lab executed on 7 synthetic development rows in repeated training "
            "context bins; no frozen test, generalization, or live-document claim."
        )


st.markdown(CSS, unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## ProcureAgent")
    st.success("CONTROLLED P0 DEMO")
    st.caption(FIXTURE_NOTICE)
    st.markdown("**Render-time behavior**")
    st.write("✓ Locked deterministic comparison")
    st.write("✗ No OCR/model until click")
    st.write("✗ No mutation until APPROVE")
    st.write("✗ No AP closure without verified proof")
    st.caption("Simulation only. No bank, ERP, POS, or accounting system is connected.")

st.markdown(
    """
    <section class="pa-hero"><div class="pa-kicker">Restaurant procurement · controlled P0</div>
      <h1>Choose what to pay<br>before the kitchen feels it.</h1>
      <p>Sasa's restaurant has limited cash and four supplier obligations. The overview runs deterministic policy comparisons; /eval records real local evidence and keeps every human and simulation gate explicit.</p>
      <span class="pa-badge">MODEL ONLY ON CLICK · SIMULATION ONLY · NO RL CLAIM</span>
    </section>
    """,
    unsafe_allow_html=True,
)
st.warning(FIXTURE_NOTICE)
headline = st.columns(4)
cash = OVERVIEW.scenario.initial_state.cash_minor if OVERVIEW is not None else PRIMARY_SCENARIO["cash_minor"]
obligations = OVERVIEW.scenario.initial_state.total_obligations_minor if OVERVIEW is not None else PRIMARY_SCENARIO["obligations_minor"]
version = OVERVIEW.scenario.initial_state.state_version if OVERVIEW is not None else PRIMARY_SCENARIO["state_version"]
headline[0].metric("Cash available", format_minor(cash), "Day 0")
headline[1].metric("Supplier obligations", format_minor(obligations), "4 invoices")
headline[2].metric("Funding gap", format_minor(obligations - cash), "Needs prioritization")
headline[3].metric("State version", str(version), f"Seed {PRIMARY_SCENARIO['seed']}")

tabs = st.tabs(("1 · Restaurant state", "2 · Document evidence", "3 · Daily batch", "4 · ProcureGym", "5 · /eval recording", "6 · Task status"))
with tabs[0]:
    st.markdown('<div class="pa-section">Screen 1 · Locked synthetic state</div>', unsafe_allow_html=True)
    st.subheader("Four bills compete for $5,000")
    st.caption("Business fields are exact synthetic lookup data. Action badges are actual deterministic policy output.")
    render_invoice_cards()
    st.info("The four canonical invoices total $6,200. UnknownCo is separate, activates no payable, and is excluded.")
with tabs[1]:
    render_document_evidence()
with tabs[2]:
    render_batch()
with tabs[3]:
    render_gym()
with tabs[4]:
    render_eval()
with tabs[5]:
    render_task_status()
