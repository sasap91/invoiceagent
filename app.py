"""InvoiceAgent — a transparent small-first reconciliation demo.

Run with:
    streamlit run app.py

The bundled scenarios are synthetic fixture replays. Manual entry uses only
structured fields and the dependency-light reconciliation core when available.
"""

from __future__ import annotations

import html
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from demo import (
    CASH_SUMMARY,
    LEDGER,
    PAYMENT_STORY,
    SCENARIOS,
    SCENARIO_ORDER,
    route_structured_entry,
)


SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from invoiceagent import (
        ExtractionMetadata,
        ExtractionSource,
        Invoice,
        LayoutLMv3InvoiceExtractor,
        LedgerSide,
        MatchMethod,
        OcrDocument,
        PaymentReceipt,
        ReceiptStatus,
        ValidationError,
        decide_small_first_route,
        extract_anchored_identifier,
        parse_money,
        reconcile,
    )

    CORE_AVAILABLE = True
except (ImportError, AttributeError):
    CORE_AVAILABLE = False


@st.cache_resource(show_spinner=False)
def load_local_invoice_extractor() -> Any:
    """Load Ryan's adapter once per Streamlit process, only on explicit request."""

    extractor = LayoutLMv3InvoiceExtractor()
    extractor.load()
    return extractor


st.set_page_config(
    page_title="InvoiceAgent · Small-first reconciliation",
    page_icon="◫",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
  :root {
    --ia-ink: #172126;
    --ia-muted: #5c6a70;
    --ia-paper: #f6f4ee;
    --ia-card: #ffffff;
    --ia-line: #dce2df;
    --ia-teal: #087f73;
    --ia-teal-soft: #e6f4f1;
    --ia-coral: #e56f51;
    --ia-coral-soft: #fff0ea;
    --ia-blue: #376e9f;
    --ia-blue-soft: #edf4fa;
    --ia-warn: #9a5b08;
    --ia-warn-soft: #fff5dc;
  }
  .stApp { background: var(--ia-paper); color: var(--ia-ink); }
  [data-testid="stHeader"] { background: rgba(246,244,238,.86); }
  [data-testid="stSidebar"] { background: #edf1ee; border-right: 1px solid var(--ia-line); }
  .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem; }
  h1, h2, h3 { color: var(--ia-ink); letter-spacing: -.02em; }
  p, li { color: var(--ia-muted); }
  .ia-hero {
    position: relative; overflow: hidden; padding: 2rem 2.2rem; margin-bottom: 1rem;
    border: 1px solid var(--ia-line); border-radius: 22px;
    background: linear-gradient(132deg, #fff 0%, #eef7f3 58%, #fff0e8 100%);
    box-shadow: 0 16px 44px rgba(35,57,58,.08);
  }
  .ia-hero:after {
    content: ""; position: absolute; width: 220px; height: 220px; right: -70px; top: -90px;
    border: 26px solid rgba(229,111,81,.12); border-radius: 50%;
  }
  .ia-kicker { color: var(--ia-teal); font-size: .72rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
  .ia-hero h1 { margin: .35rem 0 .55rem; font-size: clamp(2.2rem, 5vw, 4.1rem); line-height: .98; max-width: 850px; }
  .ia-hero p { margin: 0; max-width: 780px; font-size: 1.06rem; line-height: 1.6; }
  .ia-fixture {
    display: flex; gap: .7rem; align-items: flex-start; padding: .8rem 1rem; margin: .8rem 0 1.4rem;
    background: var(--ia-warn-soft); border: 1px solid #ecd39c; border-radius: 12px;
    color: #6f4b13; font-size: .86rem;
  }
  .ia-fixture b { color: #6f4b13; }
  .ia-section-label { color: var(--ia-teal); font-size: .7rem; letter-spacing: .14em; font-weight: 800; text-transform: uppercase; margin-bottom: .25rem; }
  .ia-explainer-grid, .ia-kpi-grid, .ia-check-grid {
    display: grid; gap: .8rem; grid-template-columns: repeat(4, minmax(0,1fr)); margin: .8rem 0 1.4rem;
  }
  .ia-explainer-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .ia-explainer, .ia-kpi, .ia-check {
    background: var(--ia-card); border: 1px solid var(--ia-line); border-radius: 14px; padding: 1rem 1.05rem;
  }
  .ia-explainer b, .ia-kpi b, .ia-check b { color: var(--ia-ink); }
  .ia-explainer p, .ia-check p { margin: .28rem 0 0; font-size: .84rem; line-height: 1.45; }
  .ia-kpi span { display: block; color: var(--ia-muted); font-size: .73rem; text-transform: uppercase; letter-spacing: .08em; }
  .ia-kpi strong { display: block; color: var(--ia-ink); font-size: 1.55rem; margin: .15rem 0; }
  .ia-kpi small { color: var(--ia-muted); }
  .ia-kpi.accent { border-top: 4px solid var(--ia-teal); }
  .ia-scenario {
    background: var(--ia-card); border: 1px solid var(--ia-line); border-radius: 18px;
    padding: 1.35rem 1.5rem; margin: .5rem 0 1rem; box-shadow: 0 10px 32px rgba(35,57,58,.05);
  }
  .ia-scenario h2 { margin: .25rem 0 .5rem; font-size: 1.75rem; }
  .ia-scenario p { margin: 0; line-height: 1.55; }
  .ia-decision { display: inline-flex; align-items: center; gap: .42rem; margin-top: .8rem; padding: .38rem .65rem; border-radius: 999px; font-size: .76rem; font-weight: 800; }
  .ia-decision.good { color: #086150; background: var(--ia-teal-soft); }
  .ia-decision.model { color: #2f608d; background: var(--ia-blue-soft); }
  .ia-decision.warn { color: #80500c; background: var(--ia-warn-soft); }
  .ia-route {
    display: grid; grid-template-columns: minmax(0,1fr) auto minmax(0,1fr) auto minmax(0,1fr);
    gap: .55rem; align-items: stretch; margin: 1rem 0 1.25rem;
  }
  .ia-route-step { background: rgba(255,255,255,.68); border: 1px solid var(--ia-line); border-radius: 14px; padding: .9rem; }
  .ia-route-step.active.local { border: 2px solid var(--ia-teal); background: var(--ia-teal-soft); }
  .ia-route-step.active.small_model { border: 2px solid var(--ia-blue); background: var(--ia-blue-soft); }
  .ia-route-step.active.escalate { border: 2px solid var(--ia-coral); background: var(--ia-coral-soft); }
  .ia-route-step span { color: var(--ia-muted); font-size: .68rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
  .ia-route-step b { display: block; color: var(--ia-ink); margin: .18rem 0; }
  .ia-route-step p { margin: 0; font-size: .77rem; line-height: 1.4; }
  .ia-arrow { display: grid; place-items: center; color: #8c9997; font-size: 1.3rem; }
  .ia-records { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: .8rem; margin: 1rem 0; }
  .ia-record { border: 1px solid var(--ia-line); border-radius: 14px; padding: 1rem; background: #fbfcfb; }
  .ia-record h4 { margin: 0 0 .65rem; color: var(--ia-ink); }
  .ia-record dl { display: grid; grid-template-columns: 1fr auto; margin: 0; gap: .35rem .8rem; }
  .ia-record dt { color: var(--ia-muted); font-size: .79rem; }
  .ia-record dd { color: var(--ia-ink); font-size: .79rem; font-weight: 700; margin: 0; text-align: right; }
  .ia-check-grid { grid-template-columns: repeat(3,minmax(0,1fr)); }
  .ia-check { border-left: 4px solid var(--ia-teal); }
  .ia-check.fail { border-left-color: var(--ia-coral); }
  .ia-model-replay { padding: .9rem 1rem; background: var(--ia-blue-soft); border: 1px solid #c7dbea; border-radius: 12px; font-size: .82rem; }
  .ia-model-replay code { color: #204a70; background: rgba(255,255,255,.55); padding: .12rem .3rem; border-radius: 4px; }
  .ia-steps { counter-reset: step; list-style: none; padding: 0; margin: .9rem 0; }
  .ia-steps li { counter-increment: step; display: flex; gap: .7rem; align-items: flex-start; margin: .55rem 0; }
  .ia-steps li:before { content: counter(step); flex: 0 0 1.55rem; height: 1.55rem; display: grid; place-items: center; border-radius: 50%; background: var(--ia-teal-soft); color: var(--ia-teal); font-size: .72rem; font-weight: 800; }
  .ia-table-wrap { overflow-x: auto; border: 1px solid var(--ia-line); border-radius: 14px; background: #fff; }
  .ia-table { width: 100%; border-collapse: collapse; font-size: .79rem; }
  .ia-table th { color: var(--ia-muted); background: #f2f5f3; font-size: .66rem; text-align: left; letter-spacing: .08em; text-transform: uppercase; padding: .75rem; white-space: nowrap; }
  .ia-table td { color: var(--ia-ink); border-top: 1px solid #edf0ee; padding: .72rem .75rem; white-space: nowrap; }
  .ia-pill { display: inline-block; border-radius: 999px; padding: .18rem .45rem; font-size: .68rem; font-weight: 800; background: #eef1ef; }
  .ia-pill.AP { color: #8b482e; background: var(--ia-coral-soft); }
  .ia-pill.AR { color: #07665b; background: var(--ia-teal-soft); }
  .ia-result { padding: 1rem 1.1rem; border-radius: 14px; border: 1px solid var(--ia-line); background: #fff; margin-top: .8rem; }
  .ia-result.local { border-left: 5px solid var(--ia-teal); }
  .ia-result.small_model { border-left: 5px solid var(--ia-blue); }
  .ia-result.escalate { border-left: 5px solid var(--ia-coral); }
  .ia-result b { color: var(--ia-ink); }
  .ia-result p { margin: .25rem 0 0; font-size: .86rem; }
  @media (max-width: 800px) {
    .ia-kpi-grid { grid-template-columns: repeat(2,minmax(0,1fr)); }
    .ia-route { grid-template-columns: 1fr; }
    .ia-arrow { transform: rotate(90deg); }
    .ia-records, .ia-explainer-grid, .ia-check-grid { grid-template-columns: 1fr; }
    .ia-hero { padding: 1.4rem; }
  }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def money(value: float) -> str:
    return f"${value:,.0f}"


def render_explainers() -> None:
    st.markdown(
        """
        <div class="ia-explainer-grid">
          <div class="ia-explainer"><b>AP · Accounts Payable</b><p>Bills we owe suppliers. Think <em>cash going out</em>.</p></div>
          <div class="ia-explainer"><b>AR · Accounts Receivable</b><p>Invoices customers owe us. Think <em>cash coming in</em>.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cash_summary() -> None:
    cards = [
        ("Cash on hand", money(CASH_SUMMARY["cash_on_hand"]), "known today", "accent"),
        ("Coming in · 30 days", money(CASH_SUMMARY["receivables_30d"]), "accounts receivable", ""),
        ("Going out · 30 days", money(CASH_SUMMARY["payables_30d"]), "accounts payable", ""),
        ("Expected net change", f"+{money(CASH_SUMMARY['net_30d'])}", "in minus out", "accent"),
    ]
    body = "".join(
        f'<div class="ia-kpi {kind}"><span>{esc(label)}</span><strong>{esc(value)}</strong><small>{esc(note)}</small></div>'
        for label, value, note, kind in cards
    )
    st.markdown(f'<div class="ia-kpi-grid">{body}</div>', unsafe_allow_html=True)


def render_payment_story() -> None:
    st.markdown("### Watch one invoice become paid")
    st.caption(
        "Choose a moment in the synthetic story. Exact decimal accounting turns a $500 supplier bill into $200 outstanding, then $0."
    )
    stage_index = st.radio(
        "Payment timeline",
        options=tuple(range(len(PAYMENT_STORY))),
        format_func=lambda index: PAYMENT_STORY[index]["label"],
        horizontal=True,
        key="payment-story-stage",
        label_visibility="collapsed",
    )
    stage = PAYMENT_STORY[stage_index]
    columns = st.columns(4)
    columns[0].metric("Invoice", stage["invoice_id"])
    columns[1].metric("Bill total", money(stage["total"]))
    columns[2].metric("Paid so far", money(stage["paid"]))
    columns[3].metric("Still owed", money(stage["outstanding"]))
    progress = stage["paid"] / stage["total"]
    st.progress(progress, text=f"{stage['status']} · {progress:.0%} settled")
    st.info(stage["explanation"])
    if stage["receipts"]:
        receipt_text = " + ".join(
            f"{receipt_id} ({money(amount)})" for receipt_id, amount in stage["receipts"]
        )
        st.write(f"**Payment evidence:** {receipt_text}")
    else:
        st.write("**Payment evidence:** No receipt yet")


def render_route(active: str) -> None:
    steps = [
        ("local", "1 · Rules", "Exact match", "IDs, amounts, and known parties"),
        ("small_model", "2 · Small model", "Read the ID", "Invoice-number specialist"),
        ("escalate", "3 · Human", "Escalate", "When evidence remains ambiguous"),
    ]
    parts = []
    for index, (route, label, title, copy) in enumerate(steps):
        active_class = f"active {route}" if active == route else ""
        parts.append(
            f'<div class="ia-route-step {active_class}"><span>{esc(label)}</span><b>{esc(title)}</b><p>{esc(copy)}</p></div>'
        )
        if index < len(steps) - 1:
            parts.append('<div class="ia-arrow" aria-hidden="true">→</div>')
    st.markdown(f'<div class="ia-route">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_record(title: str, record: Mapping[str, Any], invoice: bool) -> str:
    if invoice:
        rows = [
            ("ID", record["invoice_id"]),
            ("Party", record["party"]),
            ("Amount", money(record["amount"])),
            ("Due", record["due"]),
            ("Book", record["direction"]),
        ]
    else:
        rows = [
            ("Reference", record["reference"]),
            ("Party", record["party"]),
            ("Amount", money(record["amount"])),
            ("Date", record["date"]),
        ]
    terms = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows)
    return f'<div class="ia-record"><h4>{esc(title)}</h4><dl>{terms}</dl></div>'


def render_scenario(scenario: Mapping[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="ia-scenario">
          <div class="ia-section-label">{esc(scenario['eyebrow'])}</div>
          <h2>{esc(scenario['title'])}</h2>
          <p>{esc(scenario['summary'])}</p>
          <div class="ia-decision {esc(scenario['decision_tone'])}">● {esc(scenario['decision'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_route(scenario["route"])
    st.markdown(
        '<div class="ia-records">'
        + render_record("Invoice", scenario["invoice"], True)
        + render_record("Payment evidence", scenario["payment"], False)
        + "</div>",
        unsafe_allow_html=True,
    )

    checks = "".join(
        f'<div class="ia-check {"" if check["passed"] else "fail"}"><b>{esc(check["label"])}</b><p>{esc(check["value"])}</p></div>'
        for check in scenario["checks"]
    )
    st.markdown(f'<div class="ia-check-grid">{checks}</div>', unsafe_allow_html=True)

    replay = scenario.get("model_replay")
    if replay:
        st.markdown(
            f"""
            <div class="ia-model-replay">
              <b>Recorded model suggestion · fixture replay</b><br>
              <code>{esc(replay['label'])}</code> · score {esc(replay['confidence'])}<br>
              {esc(replay['evidence'])}<br><small>{esc(replay['provenance'])}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    step_items = "".join(f"<li>{esc(step)}</li>" for step in scenario["steps"])
    st.markdown(f'<ol class="ia-steps">{step_items}</ol>', unsafe_allow_html=True)
    st.info(scenario["why"], icon="💡")


def render_ledger() -> None:
    rows = []
    for row in LEDGER:
        rows.append(
            "<tr>"
            f'<td><span class="ia-pill {esc(row["kind"])}">{esc(row["kind"])}</span></td>'
            f'<td><b>{esc(row["invoice_id"])}</b></td>'
            f'<td>{esc(row["party"])}</td>'
            f'<td>{esc(row["due"])}</td>'
            f'<td>{esc(money(row["amount"]))}</td>'
            f'<td>{esc(row["payment_ref"])}</td>'
            f'<td>{esc(row["reconciliation"])}</td>'
            f'<td>{esc(row["cash_status"])}</td>'
            "</tr>"
        )
    st.markdown(
        """
        <div class="ia-table-wrap"><table class="ia-table">
          <thead><tr><th>Book</th><th>Invoice</th><th>Party</th><th>Due</th><th>Amount</th><th>Payment ref</th><th>Reconciliation</th><th>Cash status</th></tr></thead>
          <tbody>"""
        + "".join(rows)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def reconcile_manual(payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Use the conservative core API when present; otherwise remain deterministic."""

    if CORE_AVAILABLE and payload.get("known_party") is True:
        try:
            side = (
                LedgerSide.ACCOUNTS_PAYABLE
                if payload["direction"] == "AP"
                else LedgerSide.ACCOUNTS_RECEIVABLE
            )
            manual_provenance = ExtractionMetadata(
                source=ExtractionSource.MANUAL,
                grounded=True,
                note="Entered in the Streamlit structured fallback form.",
            )
            invoice = Invoice(
                invoice_number=str(payload["invoice_id"]),
                counterparty=str(payload["party"]),
                amount=parse_money(str(payload["invoice_amount"]).strip()),
                issue_date=str(payload["issue_date"]),
                due_date=str(payload["due_date"]),
                side=side,
                extraction=manual_provenance,
                approved=True,
            )
            receipt = PaymentReceipt(
                receipt_number=str(payload["payment_id"]),
                counterparty=str(payload["party"]),
                amount=parse_money(str(payload["payment_amount"]).strip()),
                payment_date=str(payload["payment_date"]),
                side=side,
                invoice_reference=str(payload["payment_reference"]).strip() or None,
                extraction=manual_provenance,
                approved=True,
            )
            report = reconcile([invoice], [receipt])
            receipt_result = report.receipts[0]
            if receipt_result.status == ReceiptStatus.MATCHED:
                allocation = receipt_result.allocations[0]
                method = (
                    "exact reference"
                    if allocation.method == MatchMethod.EXPLICIT_REFERENCE
                    else "weighted local evidence"
                )
                return {
                    "route": "local",
                    "decision": "Accepted by the core",
                    "accepted": True,
                    "model_called": False,
                    "reason": f"Matched by {method}: {receipt_result.reason}.",
                }, "core"
            return {
                "route": "escalate",
                "decision": "Core sent this to review",
                "accepted": False,
                "model_called": False,
                "reason": receipt_result.reason,
            }, "core"
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            return {
                "route": "escalate",
                "decision": "Structured input needs correction",
                "accepted": False,
                "model_called": False,
                "reason": str(exc),
            }, "core"
        except Exception as exc:
            st.warning(f"Core adapter could not process this entry; using the deterministic fallback. ({type(exc).__name__})")
    return route_structured_entry(payload), "fallback"


def render_manual_entry() -> None:
    st.markdown("### Try a structured entry")
    st.caption(
        "This form does not read a PDF, run OCR, or invent a model response. It evaluates only the fields you provide."
    )
    with st.form("manual-entry", clear_on_submit=False):
        left, middle, right = st.columns(3)
        with left:
            direction = st.selectbox("Book", ("AP · we owe", "AR · owed to us"))
            invoice_id = st.text_input("Invoice ID", value="DEMO-101", max_chars=64)
            party = st.text_input("Counterparty", value="Demo Supply Co.", max_chars=100)
        with middle:
            invoice_amount = st.text_input("Invoice amount", value="1250.00", max_chars=24)
            issue = st.date_input("Invoice issue date", value=date(2026, 8, 20))
            due = st.date_input("Due date", value=date(2026, 9, 15))
            known_party = st.checkbox("Known counterparty", value=True)
        with right:
            payment_id = st.text_input("Payment ID", value="PAY-DEMO-101", max_chars=64)
            payment_reference = st.text_input("Payment reference", value="DEMO-101", max_chars=64)
            payment_amount = st.text_input("Payment amount", value="1250.00", max_chars=24)
            payment_date = st.date_input("Payment date", value=date(2026, 8, 30))
        submitted = st.form_submit_button("Route this entry", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "direction": "AP" if direction.startswith("AP") else "AR",
            "invoice_id": invoice_id,
            "party": party,
            "invoice_amount": invoice_amount,
            "issue_date": issue.isoformat(),
            "due_date": due.isoformat(),
            "known_party": known_party,
            "payment_id": payment_id,
            "payment_reference": payment_reference,
            "payment_amount": payment_amount,
            "payment_date": payment_date.isoformat(),
        }
        result, source = reconcile_manual(payload)
        st.session_state["manual_result"] = result
        st.session_state["manual_source"] = source

    result = st.session_state.get("manual_result")
    if result:
        source = st.session_state.get("manual_source", "fallback")
        source_label = "InvoiceAgent core adapter" if source == "core" else "deterministic fallback"
        model_note = "A model was called." if result["model_called"] else "No model call occurred."
        st.markdown(
            f"""
            <div class="ia-result {esc(result['route'])}">
              <div class="ia-section-label">{esc(source_label)}</div>
              <b>{esc(result['decision'])}</b>
              <p>{esc(result['reason'])} {esc(model_note)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_live_extraction() -> None:
    st.markdown("### Run Ryan's local invoice-number specialist")
    st.caption(
        "This is the real model path—not a fixture. LayoutLMv3 needs both an invoice image "
        "and OCR words with 0–1000 bounding boxes because the model does not perform OCR."
    )
    with st.expander("OCR sidecar format"):
        st.code(
            json.dumps(
                {
                    "words": ["Invoice", "No", "INV-204"],
                    "boxes": [[80, 60, 180, 100], [190, 60, 240, 100], [250, 60, 390, 100]],
                    "quality": "0.92",
                    "raw_text": "Invoice No INV-204",
                    "engine": "your-ocr-engine",
                },
                indent=2,
            ),
            language="json",
        )

    image_file = st.file_uploader(
        "Invoice image", type=("png", "jpg", "jpeg"), key="live-invoice-image"
    )
    ocr_file = st.file_uploader(
        "OCR JSON sidecar", type=("json",), key="live-invoice-ocr"
    )
    run_model = st.button(
        "Run the local specialist",
        type="primary",
        disabled=image_file is None or ocr_file is None,
        use_container_width=True,
    )
    if not run_model:
        st.info(
            "First run downloads the Microsoft base model and Ryan's LoRA adapter. "
            "Install the optional runtime with `pip install -e '.[model]'`.",
            icon="ℹ️",
        )
        return

    try:
        from PIL import Image

        payload = json.loads(ocr_file.getvalue().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValidationError("OCR JSON must contain an object")
        ocr = OcrDocument(
            words=payload.get("words", ()),
            boxes=payload.get("boxes", ()),
            quality=str(payload.get("quality", "0")),
            raw_text=str(payload.get("raw_text", "")),
            engine=str(payload.get("engine", "uploaded-sidecar")),
        )
        image = Image.open(image_file).convert("RGB")
        heuristic = extract_anchored_identifier(ocr.words)
        with st.spinner("Loading and running the local document specialist…"):
            result = load_local_invoice_extractor().predict(image, ocr)
        signals = result.routing_signals(
            ocr,
            heuristic_candidate=heuristic,
            escalation_available=False,
        )
        decision = decide_small_first_route(signals)
    except (ValidationError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        st.error(f"The image/OCR input is invalid: {exc}")
        return
    except RuntimeError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(
            f"Local inference failed safely ({type(exc).__name__}). No ledger record was changed."
        )
        return

    selected = result.selected
    metrics = st.columns(4)
    metrics[0].metric("Candidate", result.candidate or "Not found")
    metrics[1].metric(
        "Entity confidence",
        f"{selected.minimum_confidence:.1%}" if selected else "—",
        help="Minimum probability across selected invoice-number tokens only; background words are excluded.",
    )
    metrics[2].metric("Latency", f"{result.latency_ms:.1f} ms")
    metrics[3].metric("Route", decision.action.value.replace("_", " ").title())

    if decision.action.value == "accept":
        st.success("Accepted locally by the quality gate. No remote model was called.")
    else:
        st.warning("Not accepted automatically. The safe next step is human review.")
    st.write("**Gate reasons:**", "; ".join(decision.reasons))
    st.write("**Anchored rule candidate:**", heuristic or "No rule candidate")
    if selected:
        evidence_rows = [
            {
                "OCR token": ocr.words[index],
                "box": list(ocr.boxes[index]),
                "token probability": f"{confidence:.4f}",
                "top-two margin": f"{margin:.4f}",
            }
            for index, confidence, margin in zip(
                selected.word_indices,
                selected.token_confidences,
                selected.token_margins,
            )
        ]
        st.dataframe(evidence_rows, use_container_width=True, hide_index=True)
    if result.ambiguous:
        st.warning(f"The model returned {len(result.spans)} candidate spans, so ambiguity failed closed.")


with st.sidebar:
    st.markdown("## InvoiceAgent")
    st.caption("Small-first AP/AR reconciliation")
    scenario_id = st.radio(
        "Demo replay",
        options=SCENARIO_ORDER,
        format_func=lambda key: SCENARIOS[key]["short_label"],
        index=0,
        key="scenario-selector",
    )
    st.divider()
    st.markdown("**Runtime truth**")
    st.caption("Fixtures: preloaded\n\nOCR: sidecar required\n\nLive model: optional local tab")
    if not CORE_AVAILABLE:
        st.caption("Core adapter: fallback mode")
    else:
        st.caption("Core adapter: deterministic core connected")


st.markdown(
    """
    <div class="ia-hero">
      <div class="ia-kicker">InvoiceAgent · cash operations without the guessing</div>
      <h1>Let rules do the obvious. Let a small model read. Let people decide the ambiguous.</h1>
      <p>One transparent queue for invoices, payments, and the evidence connecting them—designed to spend model compute only where words are actually fuzzy.</p>
    </div>
    <div class="ia-fixture"><b>Demo integrity</b><span>Every scenario below is a synthetic fixture replay. No document was scanned, no OCR ran, and no live model produced these examples.</span></div>
    """,
    unsafe_allow_html=True,
)

render_explainers()

st.markdown('<div class="ia-section-label">Cash-flow snapshot · synthetic ledger</div>', unsafe_allow_html=True)
render_cash_summary()

tab_story, tab_decision, tab_ledger, tab_manual, tab_live, tab_method = st.tabs(
    [
        "$500 → $200 → $0",
        "Routing replay",
        "Invoice & payment ledger",
        "Structured fallback",
        "Live local model",
        "How routing works",
    ]
)

with tab_story:
    render_payment_story()

with tab_decision:
    render_scenario(SCENARIOS[scenario_id])

with tab_ledger:
    st.markdown("### What is reconciled—and what still needs attention")
    st.caption("Synthetic ledger as of 30 Aug 2026. AP is money out; AR is money in.")
    matched = sum(
        row["reconciliation"] in {"Matched", "Matched after local extraction"}
        for row in LEDGER
    )
    st.progress(matched / len(LEDGER), text=f"{matched} of {len(LEDGER)} ledger items reconciled")
    render_ledger()
    st.warning(
        f"Attention: {money(CASH_SUMMARY['overdue_receivables'])} is overdue and ambiguous. "
        "It remains unposted until a reviewer chooses the correct payment.",
        icon="⚠️",
    )

with tab_manual:
    render_manual_entry()

with tab_live:
    render_live_extraction()

with tab_method:
    st.markdown("### Why small-first routing?")
    render_route("small_model")
    st.markdown(
        """
        - **Rules first:** exact IDs and amounts are database work, not language-model work.
        - **Small model second:** Ryan's local LayoutLMv3 specialist finds invoice-number tokens when a layout defeats the label rule.
        - **Verification before posting:** an extracted identifier must still match grounded OCR and satisfy amount, party, date, and policy checks.
        - **Escalate on ambiguity:** the system creates a compact review packet instead of silently guessing.

        The demo's model-shaped outputs are replays, so you can inspect the interaction even with every AI service disconnected.
        """
    )

st.caption("InvoiceAgent demo · synthetic data only · no banking action is performed")
