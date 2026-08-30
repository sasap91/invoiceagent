"""ProcureAgent's controlled Streamlit demo and recording surface.

Run with: ``streamlit run procure_app.py``.

The overview executes dependency-light deterministic P0 code. OCR and Ryan's
local model execute only after the Guided demo document button; simulation and AP
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
from procureagent.token_labels import (  # noqa: E402
    TokenLabel,
    label_invoice_tokens,
    label_receipt_tokens,
)


ASSET_DIR = ROOT / "data" / "procureagent" / "assets"
EVAL_DIR = ROOT / "data" / "procureagent" / "eval"
SCENARIO_PATH = ROOT / "data" / "procureagent" / "scenario_v1.json"
INVOICE_PATH = ASSET_DIR / "fresh_farms_invoice.png"
RECEIPT_PATH = ASSET_DIR / "fresh_farms_payment_receipt.png"
MODEL_SMOKE_PATH = EVAL_DIR / "model_smoke_v1.json"
RECEIPT_PROVENANCE_PATH = ASSET_DIR / "receipt_provenance.json"


st.set_page_config(
    page_title="ProcureAgent · Guided accounts payable demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Raleway:wght@400;500;600;700&display=swap');

  :root {
    --pa-bg:#f5f5f0;
    --pa-surface:#ffffff;
    --pa-surface-subtle:#f6f7f5;
    --pa-ink:#102a2a;
    --pa-muted:#4c635f;
    --pa-line:#dce4df;
    --pa-line-strong:#bdcbc4;
    --pa-accent:#08778a;
    --pa-accent-dark:#075c6a;
    --pa-accent-soft:#e4f3f5;
    --pa-success:#176b52;
    --pa-success-soft:#e7f4ee;
    --pa-warning:#8b5a12;
    --pa-warning-soft:#fff4d9;
    --pa-danger:#9b3e37;
    --pa-danger-soft:#fcecea;
    --pa-code:#132d2b;
    --pa-radius-sm:10px;
    --pa-radius-md:16px;
    --pa-radius-lg:24px;
    --pa-shadow-sm:0 1px 2px rgba(16,42,42,.05);
    --pa-shadow-md:0 12px 34px rgba(16,42,42,.08);
    --pa-space-1:.25rem;
    --pa-space-2:.5rem;
    --pa-space-3:.75rem;
    --pa-space-4:1rem;
    --pa-space-5:1.5rem;
    --pa-space-6:2rem;
  }
  html { scroll-behavior:smooth; }
  body, .stApp, [class*="css"] { font-family:'Raleway', sans-serif; }
  .stApp { background:var(--pa-bg); color:var(--pa-ink); }
  [data-testid="stSidebar"] { background:#edf2ee; border-right:1px solid var(--pa-line); }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:var(--pa-muted); }
  [data-testid="stHeader"] { display:none; }
  .block-container { max-width:1180px; padding-top:1.45rem; padding-bottom:4rem; }
  h1,h2,h3,h4,h5 { color:var(--pa-ink); font-family:'Lora', serif; letter-spacing:-.025em; text-wrap:balance; }
  p, li { line-height:1.58; }
  a { color:var(--pa-accent-dark); text-underline-offset:3px; }

  .pa-hero { position:relative; overflow:hidden; padding:clamp(1.4rem,4vw,2.35rem); border-radius:var(--pa-radius-lg);
    border:1px solid var(--pa-line); background:var(--pa-surface); box-shadow:var(--pa-shadow-md); margin-bottom:var(--pa-space-4); }
  .pa-hero:before { content:""; position:absolute; inset:0 auto 0 0; width:7px; background:var(--pa-accent); }
  .pa-hero:after { content:""; position:absolute; width:190px; height:190px; border-radius:46% 54% 64% 36%;
    background:#d9ece7; right:-72px; top:-103px; transform:rotate(18deg); opacity:.72; }
  .pa-kicker { color:var(--pa-accent-dark); font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:.72rem; }
  .pa-hero h1 { max-width:18ch; font-size:clamp(2rem,5vw,3.75rem); line-height:1.02; margin:.4rem 0 .65rem; }
  .pa-hero p { color:var(--pa-muted); max-width:740px; font-size:1.04rem; line-height:1.58; margin:0; }
  .pa-badge { display:inline-flex; align-items:center; margin-top:1rem; border-radius:999px; padding:.42rem .72rem;
    background:var(--pa-accent-soft); color:var(--pa-accent-dark); font-weight:700; font-size:.75rem; letter-spacing:.035em; }
  .pa-section { color:var(--pa-accent-dark); font-size:.7rem; font-weight:700; letter-spacing:.12em;
    text-transform:uppercase; margin:.25rem 0; }

  .pa-trust-strip { position:static; display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
    gap:1px; margin:.7rem 0 1rem; border:1px solid var(--pa-line-strong); border-radius:var(--pa-radius-md);
    overflow:hidden; background:var(--pa-line); box-shadow:var(--pa-shadow-sm); }
  .pa-trust-item { background:#f8faf8; padding:.72rem .85rem; }
  .pa-trust-item b { display:block; color:var(--pa-ink); font-size:.8rem; }
  .pa-trust-item span { color:var(--pa-muted); font-size:.72rem; }

  .stApp .pa-progress { display:grid !important; width:100% !important; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.55rem; list-style:none;
    margin:0 0 1rem; padding:0; }
  .pa-progress li { position:relative; min-width:0; padding:.72rem .68rem .72rem 2.55rem; border:1px solid var(--pa-line);
    border-radius:var(--pa-radius-sm); background:var(--pa-surface); color:var(--pa-muted); }
  .pa-progress .step-no { position:absolute; left:.68rem; top:.69rem; display:grid; place-items:center; width:1.42rem; height:1.42rem;
    border-radius:50%; border:1px solid var(--pa-line-strong); font-size:.68rem; font-weight:700; }
  .pa-progress b { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:inherit; font-size:.76rem; }
  .pa-progress small { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.63rem; margin-top:.08rem; }
  .pa-progress li.done { background:var(--pa-success-soft); border-color:#a9cdbf; color:var(--pa-success); }
  .pa-progress li.done .step-no { background:var(--pa-success); color:#fff; border-color:var(--pa-success); }
  .pa-progress li.current { border:2px solid var(--pa-accent); padding-top:calc(.72rem - 1px); padding-bottom:calc(.72rem - 1px);
    background:var(--pa-accent-soft); color:var(--pa-accent-dark); box-shadow:0 0 0 3px rgba(8,119,138,.08); }
  .pa-progress li.current .step-no { background:var(--pa-accent); border-color:var(--pa-accent); color:#fff; }

  .pa-step-panel { margin:0 0 1.25rem; padding:clamp(1rem,3vw,1.65rem); background:var(--pa-surface);
    border:1px solid var(--pa-line); border-radius:var(--pa-radius-lg); box-shadow:var(--pa-shadow-sm); }
  .pa-step-head { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; padding-bottom:1rem;
    margin-bottom:1rem; border-bottom:1px solid var(--pa-line); }
  .pa-step-head .number { display:block; color:var(--pa-accent-dark); font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
  .pa-step-head h2 { margin:.15rem 0 .25rem; font-size:clamp(1.55rem,3vw,2.15rem); }
  .pa-step-head p { color:var(--pa-muted); margin:0; max-width:64ch; }
  .pa-step-status { flex:0 0 auto; border-radius:999px; padding:.42rem .68rem; background:var(--pa-warning-soft);
    color:var(--pa-warning); font-size:.7rem; font-weight:700; white-space:nowrap; }

  .pa-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.85rem; margin:.9rem 0 1.2rem; }
  .pa-card { background:var(--pa-surface); border:1px solid var(--pa-line); border-radius:var(--pa-radius-md); padding:1.05rem 1.1rem;
    box-shadow:var(--pa-shadow-sm); }
  .pa-card-top { display:flex; align-items:flex-start; justify-content:space-between; gap:.65rem; }
  .pa-card h3 { margin:0; font-size:1.08rem; }
  .pa-card .sub { color:var(--pa-muted); font-size:.76rem; margin:.18rem 0 .8rem; }
  .pa-facts { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.45rem; }
  .pa-fact { border-radius:var(--pa-radius-sm); padding:.55rem .6rem; background:var(--pa-surface-subtle); }
  .pa-fact span { display:block; color:var(--pa-muted); text-transform:uppercase; letter-spacing:.06em; font-size:.58rem; }
  .pa-fact b { display:block; color:var(--pa-ink); margin-top:.12rem; font-size:.83rem; }
  .pa-action { flex:0 0 auto; border-radius:999px; padding:.27rem .52rem; font-size:.64rem; font-weight:900; }
  .PAY { color:var(--pa-success); background:var(--pa-success-soft); } .DEFER { color:var(--pa-warning); background:var(--pa-warning-soft); }
  .VERIFY { color:var(--pa-danger); background:var(--pa-danger-soft); }
  .pa-source { color:var(--pa-muted); font-size:.68rem; margin-top:.72rem; border-top:1px solid #edf0ec; padding-top:.6rem; }
  .pa-evidence { background:var(--pa-code); color:#e9f4ef; border-radius:var(--pa-radius-md); padding:1.25rem; min-height:170px; }
  .pa-evidence span { color:#9cc7b7; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; }
  .pa-evidence strong { color:white; display:block; font-size:1.7rem; margin:.7rem 0; }
  .pa-evidence mark { background:#f4cb67; color:#38290b; padding:.18rem .38rem; border-radius:4px; }
  .pa-evidence p { color:#bfd2ca; font-size:.82rem; line-height:1.5; margin:.7rem 0 0; }

  .pa-token-legend { display:flex; flex-wrap:wrap; gap:.55rem; margin:.55rem 0; color:var(--pa-muted); font-size:.72rem; }
  .pa-token-legend i { width:.75rem; height:.75rem; border-radius:3px; border:1px solid var(--pa-line-strong); background:var(--pa-surface-subtle); }
  .pa-token-legend i.target { background:#d9f0f3; border-color:#5aa6b1; }
  .pa-token-map { display:flex; flex-wrap:wrap; align-items:flex-start; gap:.4rem; padding:.8rem; max-height:21rem; overflow-y:auto;
    background:var(--pa-code); border:1px solid #254844; border-radius:var(--pa-radius-md); }
  .pa-token { display:inline-flex; flex-direction:column; gap:.05rem; max-width:16rem; padding:.36rem .48rem; border:1px solid #3d5b57;
    border-radius:7px; color:#ecf5f2; background:#1b3935; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .pa-token b { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.78rem; font-weight:600; }
  .pa-token small { color:#9eb9b3; font-size:.55rem; line-height:1.15; }
  .pa-token.target { border-color:#70c7d2; background:#123f45; box-shadow:inset 0 -2px 0 #70c7d2; }
  .pa-token.target small { color:#a9e2e8; }
  .pa-token.invoice-number { border-color:#67c6d2; background:#123f45; box-shadow:inset 0 -3px 0 #67c6d2; }
  .pa-token.amount { border-color:#f2b84b; background:#403416; box-shadow:inset 0 -3px 0 #f2b84b; }
  .pa-token.receipt-id { border-color:#b99ce7; background:#302743; box-shadow:inset 0 -3px 0 #b99ce7; }
  .pa-token.supplier { border-color:#78c59c; background:#183d2d; box-shadow:inset 0 -3px 0 #78c59c; }
  .pa-token.paid-date { border-color:#ec946f; background:#482b23; box-shadow:inset 0 -3px 0 #ec946f; }
  .pa-token.currency { border-color:#94abc9; background:#233344; box-shadow:inset 0 -3px 0 #94abc9; }
  .pa-token.invoice-number small,.pa-token.amount small,.pa-token.receipt-id small,
  .pa-token.supplier small,.pa-token.paid-date small,.pa-token.currency small { color:#eef8f5; }
  .pa-token-legend i.invoice-number { background:#67c6d2; border-color:#298a97; }
  .pa-token-legend i.amount { background:#f2b84b; border-color:#a46d09; }
  .pa-token-legend i.receipt-id { background:#b99ce7; border-color:#7656aa; }
  .pa-token-legend i.supplier { background:#78c59c; border-color:#39805a; }
  .pa-token-legend i.paid-date { background:#ec946f; border-color:#a95131; }
  .pa-token-legend i.currency { background:#94abc9; border-color:#526d91; }

  .pa-decision { display:grid; grid-template-columns:1.15fr .85fr; gap:1rem; margin:1rem 0; }
  .pa-decision-card { padding:1rem; border:1px solid var(--pa-line); border-radius:var(--pa-radius-md); background:var(--pa-surface-subtle); }
  .pa-decision-card .label { color:var(--pa-muted); font-size:.68rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
  .pa-decision-card strong { display:block; margin:.3rem 0; color:var(--pa-ink); font-family:'Lora',serif; font-size:1.55rem; }
  .pa-decision-card p { color:var(--pa-muted); font-size:.8rem; margin:.2rem 0 0; }

  .pa-journal { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1px; margin:1rem 0;
    border:1px solid var(--pa-line-strong); border-radius:var(--pa-radius-md); overflow:hidden; background:var(--pa-line); }
  .pa-journal-entry { background:var(--pa-surface); padding:1rem; }
  .pa-journal-entry span { color:var(--pa-muted); text-transform:uppercase; letter-spacing:.1em; font-size:.65rem; font-weight:700; }
  .pa-journal-entry b { display:block; color:var(--pa-ink); font-size:1rem; margin:.35rem 0; }
  .pa-journal-entry strong { color:var(--pa-accent-dark); font-family:'Lora',serif; font-size:1.4rem; }
  .pa-journal-note { grid-column:1/-1; background:#f3f7f4; padding:.75rem 1rem; color:var(--pa-muted); font-size:.78rem; }

  .pa-batch { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.72rem; margin:.9rem 0 1.2rem; }
  .pa-batch-card { background:var(--pa-surface); border:1px solid var(--pa-line); border-top:4px solid #9aaaa3;
    border-radius:14px; padding:.9rem; }
  .pa-batch-card.PAY { border-top-color:var(--pa-success); } .pa-batch-card.DEFER { border-top-color:#d5942f; }
  .pa-batch-card.VERIFY { border-top-color:#c76b61; }
  .pa-batch-card span { font-size:.62rem; letter-spacing:.09em; text-transform:uppercase; font-weight:900; }
  .pa-batch-card b { display:block; color:var(--pa-ink); margin:.38rem 0 .15rem; }
  .pa-batch-card small { color:var(--pa-muted); line-height:1.4; }
  .pa-status { border-left:4px solid var(--pa-accent); padding:.7rem .85rem; background:var(--pa-surface); margin:.45rem 0; border-radius:8px; }
  .pa-status b { color:var(--pa-ink); } .pa-status span { color:var(--pa-muted); font-size:.78rem; }
  .pa-stage { display:flex; gap:.6rem; align-items:flex-start; padding:.7rem .85rem; margin:.5rem 0;
    background:var(--pa-surface); border:1px solid var(--pa-line); border-radius:12px; }
  .pa-stage .dot { width:.7rem; height:.7rem; border-radius:50%; margin-top:.28rem; background:#9aaaa3; flex:0 0 auto; }
  .pa-stage.ok .dot { background:var(--pa-success); } .pa-stage.review .dot { background:#d5942f; }
  .pa-stage.stop .dot { background:#c8574d; }
  .pa-stage b { color:var(--pa-ink); } .pa-stage span { color:var(--pa-muted); font-size:.78rem; }
  .pa-table-wrap { overflow-x:auto; margin:.55rem 0 1rem; border:1px solid var(--pa-line); border-radius:12px; }
  .pa-table { width:100%; border-collapse:collapse; background:var(--pa-surface); font-size:.78rem; }
  .pa-table th { color:#44625a; background:#edf3ee; text-align:left; font-size:.64rem; letter-spacing:.04em; text-transform:uppercase; }
  .pa-table th,.pa-table td { padding:.58rem .68rem; border-bottom:1px solid #edf0ec; white-space:nowrap; }
  .pa-table tr:last-child td { border-bottom:0; }

  [data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
  [data-testid="stFileUploaderDropzone"] button { min-height:44px; border-radius:10px; font-family:'Raleway',sans-serif;
    font-weight:700; transition:background-color .18s ease,border-color .18s ease,box-shadow .18s ease,transform .18s ease; cursor:pointer; }
  [data-testid="stButton"] button[kind="primary"] { background:var(--pa-accent); color:#fff; border-color:var(--pa-accent); }
  [data-testid="stButton"] button[kind="primary"]:hover { background:var(--pa-accent-dark); border-color:var(--pa-accent-dark); }
  [data-testid="stButton"] button:disabled,[data-testid="stDownloadButton"] button:disabled {
    cursor:not-allowed; opacity:1; background:#e5ebe7 !important; color:#5a6d68 !important; border-color:#cbd6d0 !important; }
  [data-baseweb="radio"] label, [data-baseweb="checkbox"] label { min-height:44px; cursor:pointer; }
  input, textarea, [data-baseweb="select"] > div { min-height:44px; }
  button:focus-visible, input:focus-visible, textarea:focus-visible, [tabindex]:focus-visible {
    outline:3px solid rgba(8,119,138,.48) !important; outline-offset:2px !important; box-shadow:none !important; }
  [data-testid="stExpander"] { border:1px solid var(--pa-line); border-radius:var(--pa-radius-sm); background:rgba(255,255,255,.7); }
  [data-testid="stExpander"] summary { min-height:44px; cursor:pointer; }

  @media (hover:hover) {
    [data-testid="stButton"] button:not(:disabled):hover, [data-testid="stDownloadButton"] button:not(:disabled):hover { transform:translateY(-1px); }
  }
  @media (max-width:900px) {
    .pa-grid,.pa-decision { grid-template-columns:1fr; }
    .pa-batch { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .pa-progress { gap:.35rem; }
    .pa-progress li { padding:.62rem .45rem .62rem 2.15rem; }
    .pa-progress .step-no { left:.48rem; top:.59rem; }
    .pa-progress small { display:none; }
  }
  @media (max-width:640px) {
    .block-container { padding-left:1rem; padding-right:1rem; padding-top:1rem; }
    .pa-trust-strip { position:static; grid-template-columns:1fr; }
    .pa-progress { grid-template-columns:repeat(4,1fr); }
    .pa-progress li { min-height:3rem; padding:.55rem .3rem; text-align:center; }
    .pa-progress .step-no { position:static; margin:0 auto .2rem; }
    .pa-progress b { font-size:.62rem; }
    .pa-step-head { flex-direction:column; }
    .pa-journal,.pa-batch,.pa-facts { grid-template-columns:1fr; }
    .pa-journal-note { grid-column:auto; }
  }
  @media (max-width:375px) {
    .pa-progress b { font-size:.56rem; }
    .pa-token-map { max-height:17rem; }
  }
  @media (prefers-reduced-motion:reduce) {
    html { scroll-behavior:auto; }
    *, *::before, *::after { animation-duration:.01ms !important; animation-iteration-count:1 !important;
      transition-duration:.01ms !important; scroll-behavior:auto !important; }
  }
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

GUIDED_WIDGET_KEYS = (
    "eval-document-review-choice",
    "eval-corrected-reference",
    "eval-operator-confirmation",
    "eval-proof-confirmation",
)

ROUTES = ("Guided demo", "Overview", "Evidence & methods")
GITHUB_BLOB = "https://github.com/sasap91/invoiceagent/blob/main"

REVIEW_DECISIONS = {
    "Confirm the displayed invoice number": "CONFIRM",
    "Correct the invoice number": "CORRECT",
    "Reject this document": "REJECT",
}

STEP_META = (
    (1, "Read invoice", "OCR + model"),
    (2, "Confirm & plan", "Human review"),
    (3, "Approve simulation", "Operator gate"),
    (4, "Match receipt", "Close with proof"),
)

RECEIPT_FIELD_LABELS = {
    "receipt_id": "Receipt ID",
    "supplier": "Supplier",
    "invoice_number": "Invoice number",
    "amount_minor": "Amount",
    "currency": "Currency",
    "paid_date": "Paid date",
}

TOKEN_DISPLAY_LABELS = {
    TokenLabel.RECEIPT_ID: "Receipt ID",
    TokenLabel.INVOICE_NUMBER: "Invoice number",
    TokenLabel.AMOUNT: "Amount",
    TokenLabel.CURRENCY: "Currency",
    TokenLabel.DATE: "Paid date",
    TokenLabel.SUPPLIER: "Supplier",
}

TOKEN_CSS_CLASSES = {
    "Invoice number": "invoice-number",
    "Amount": "amount",
    "Receipt ID": "receipt-id",
    "Supplier": "supplier",
    "Paid date": "paid-date",
    "Currency": "currency",
}


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


def reset_guided_flow() -> None:
    """Reset demo results and explicit decisions without running any adapter."""

    clear_flow()
    for key in (
        *GUIDED_WIDGET_KEYS,
        "eval-document-input-key",
        "eval-receipt-input-key",
    ):
        st.session_state.pop(key, None)


def current_guided_step() -> int:
    if st.session_state.get("eval-document-analysis") is None:
        return 1
    if st.session_state.get("eval-prepared") is None:
        return 2
    if st.session_state.get("eval-simulation") is None:
        return 3
    return 4


def render_progress_rail(current_step: int) -> None:
    items = []
    confirmed = st.session_state.get("eval-confirmed-payment") is not None
    for number, label, detail in STEP_META:
        state = "done" if number < current_step or (number == 4 and confirmed) else (
            "current" if number == current_step else "upcoming"
        )
        aria = ' aria-current="step"' if state == "current" else ""
        state_label = "Complete" if state == "done" else ("Current" if state == "current" else "Not started")
        items.append(
            f'<li class="{state}"{aria}><span class="step-no">{number}</span>'
            f'<b>{esc(label)}</b><small>{esc(state_label)} · {esc(detail)}</small></li>'
        )
    st.markdown(
        f'<ol class="pa-progress" aria-label="Demo progress">{"".join(items)}</ol>',
        unsafe_allow_html=True,
    )


def render_trust_strip() -> None:
    st.markdown(
        """
        <aside class="pa-trust-strip" aria-label="Demo safety boundaries">
          <div class="pa-trust-item"><b>Simulation only</b><span>No real payment is sent</span></div>
          <div class="pa-trust-item"><b>Human approval required</b><span>No invoice is accepted silently</span></div>
          <div class="pa-trust-item"><b>No bank or ERP connected</b><span>All records stay inside this demo</span></div>
        </aside>
        """,
        unsafe_allow_html=True,
    )


def render_step_header(number: int, title: str, description: str, status: str) -> None:
    st.markdown(
        f'<header class="pa-step-head"><div><span class="number">Step {number} of 4</span>'
        f'<h2>{esc(title)}</h2><p>{esc(description)}</p></div>'
        f'<span class="pa-step-status">{esc(status)}</span></header>',
        unsafe_allow_html=True,
    )


def source_excerpt(relative_path: str, start: int, end: int) -> str:
    """Return a short, numbered excerpt from the real repository source."""

    try:
        lines = (ROOT / relative_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return "Source unavailable in this runtime."
    clipped_start = max(1, start)
    clipped_end = min(len(lines), end)
    return "\n".join(
        f"{number:>4}  {lines[number - 1]}"
        for number in range(clipped_start, clipped_end + 1)
    )


def render_technical_evidence(
    *,
    sources: tuple[tuple[str, str, int, int], ...],
    runtime: dict[str, Any] | None = None,
    answer_key_note: bool = False,
) -> None:
    """Progressively disclose real code and runtime provenance for one step."""

    with st.expander("How this works · Technical evidence", expanded=False):
        st.caption(
            "These are live repository excerpts, not pseudocode. Open a source link to inspect the full implementation."
        )
        if answer_key_note:
            st.info(
                "The evaluation answer key is hidden from this workflow. OCR receives the image; "
                "the local model receives only the image plus OCR words and boxes."
            )
        for label, relative_path, start, end in sources:
            href = f"{GITHUB_BLOB}/{relative_path}#L{start}-L{end}"
            st.markdown(f"**{esc(label)}** · [`{esc(relative_path)}`]({href})")
            st.code(source_excerpt(relative_path, start, end), language="python")
        if runtime:
            st.markdown("**This session's runtime provenance**")
            st.json(runtime)


def build_token_label_rows(
    words: Any,
    label_by_sequence: dict[int, str] | None = None,
    *,
    default_label: str = "Other text",
) -> list[dict[str, Any]]:
    """Pure UI adapter for any future backend token-label dictionary.

    The OCR/model contracts currently expose grounded word indices instead of
    one label per OCR token. This helper is the explicit integration seam: it
    never invents a model output and labels every unmapped token as other text.
    """

    labels = dict(label_by_sequence or {})
    return [
        {
            "sequence": word.sequence,
            "token": word.text,
            "label": labels.get(word.sequence, default_label),
            "confidence": str(word.confidence),
            "pixel_box": (
                f"({word.pixel_box.x0},{word.pixel_box.y0},"
                f"{word.pixel_box.x1},{word.pixel_box.y1})"
            ),
            "layout_box": (
                f"({word.normalized_box.x0},{word.normalized_box.y0},"
                f"{word.normalized_box.x1},{word.normalized_box.y1})"
            ),
        }
        for word in words
    ]


def invoice_token_labels(analysis: Any) -> dict[int, str]:
    tokens = label_invoice_tokens(analysis.ocr, gate=analysis.gate)
    return {
        token.index: (
            f"{TOKEN_DISPLAY_LABELS[token.label]} · "
            f"{token.source.value.replace('_', ' ').lower()}"
        )
        for token in tokens
        if token.label is not TokenLabel.OTHER
    }


def receipt_token_labels(ocr: Any, parsed: Any) -> dict[int, str]:
    tokens = label_receipt_tokens(ocr, parsed)
    return {
        token.index: (
            f"{TOKEN_DISPLAY_LABELS[token.label]} · "
            f"{token.source.value.replace('_', ' ').lower()}"
        )
        for token in tokens
        if token.label is not TokenLabel.OTHER
    }


def render_token_map(
    words: Any,
    label_by_sequence: dict[int, str],
    *,
    default_label: str = "Other text",
) -> list[dict[str, Any]]:
    rows = build_token_label_rows(words, label_by_sequence, default_label=default_label)
    tokens = []
    visible_kinds: list[str] = []
    for row in rows:
        target = row["label"] != default_label
        label_kind = str(row["label"]).split(" · ", 1)[0]
        category_class = TOKEN_CSS_CLASSES.get(label_kind, "") if target else ""
        if category_class and label_kind not in visible_kinds:
            visible_kinds.append(label_kind)
        token_class = (
            f"pa-token target {category_class}" if category_class else (
                "pa-token target" if target else "pa-token"
            )
        )
        tokens.append(
            f'<span class="{token_class}" title="OCR confidence {esc(row["confidence"])}">'
            f'<b>{esc(row["token"])}</b><small>{esc(row["label"])}</small></span>'
        )
    legend = "".join(
        f'<i class="{TOKEN_CSS_CLASSES[kind]}"></i><span>{esc(kind)}</span>'
        for kind in TOKEN_CSS_CLASSES
        if kind in visible_kinds
    )
    st.markdown(
        f'<div class="pa-token-legend">{legend}'
        f'<i></i><span>{esc(default_label)}</span></div>'
        f'<div class="pa-token-map" aria-label="OCR tokens and labels">{"".join(tokens)}</div>',
        unsafe_allow_html=True,
    )
    return rows


def stage_badge(label: str, detail: str, tone: str = "pending") -> None:
    st.markdown(
        f'<div class="pa-stage {esc(tone)}"><span class="dot"></span><div><b>{esc(label)}</b><br><span>{esc(detail)}</span></div></div>',
        unsafe_allow_html=True,
    )


def render_table(rows: list[dict[str, Any]], *, highlight: Any = None) -> None:
    """Render small audit tables without a pandas/Arrow runtime dependency.

    ``highlight`` is an optional ``row -> bool`` predicate; matching rows get
    a visible highlight class so entity predictions stand out from background.
    """

    if not rows:
        st.caption("No rows")
        return
    headers = list(rows[0])
    heading = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "".join(
        f'<tr class="{"pa-row-highlight" if highlight and highlight(row) else ""}">'
        + "".join(f"<td>{esc(row.get(item, ''))}</td>" for item in headers) + "</tr>"
        for row in rows
    )
    st.markdown(
        '<style>.pa-row-highlight{background:#fff3b0!important;font-weight:600;}</style>'
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
        st.warning("Static stored evidence only. For actual Tesseract + Ryan model evidence, use the Guided demo.")
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
    st.warning("Operator approval: NOT RECORDED here · state unchanged · use the Guided demo for the governed mutation path")
    controls = st.columns(3)
    controls[0].radio("Operator decision (overview disabled)", ("APPROVE", "MODIFY", "REJECT"), key="operator-decision-preview", disabled=True, horizontal=True)
    controls[1].button("Verifier already ran", key="run-verifier", disabled=True, use_container_width=True)
    controls[2].button("Commit in Guided demo", key="commit-procuregym", disabled=True, use_container_width=True)


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


def render_ocr_result(
    ocr: Any,
    heading: str,
    *,
    labels: dict[int, str] | None = None,
    default_label: str = "Other text",
) -> None:
    st.markdown(f"#### {heading}")
    cols = st.columns(4)
    cols[0].metric("OCR status", ocr.status.value)
    cols[1].metric("Words", str(len(ocr.words)))
    cols[2].metric("Mean quality", f"{ocr.quality:.3f}")
    cols[3].metric("Runtime", f"{ocr.runtime_ms} ms")
    if ocr.error_code:
        st.error(f"{ocr.error_code}: {ocr.error_message}")
        return
    st.caption(
        f"Read locally by Tesseract {ocr.engine_version} · no document text left this app"
    )
    rows = render_token_map(
        ocr.words,
        labels or {},
        default_label=default_label,
    )
    with st.expander(f"Inspect all {len(rows)} token boxes and confidence scores"):
        render_table(rows)
        st.code(ocr.raw_text or "<no OCR text>")


def _is_entity_label(label: str) -> bool:
    return label not in ("O", "LABEL_0")


def render_annotated_invoice_image(image_bytes: bytes, words: tuple, token_predictions: tuple) -> None:
    """Draw the model's per-word boxes on the invoice image: red = predicted entity, blue = background."""

    try:
        from io import BytesIO
        from PIL import Image, ImageDraw
    except ImportError:
        st.warning("Pillow is not installed; cannot draw model bounding boxes on the image.")
        return
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        st.warning(f"Could not open the invoice image for annotation: {exc}")
        return
    draw = ImageDraw.Draw(image)
    for word, token in zip(words, token_predictions):
        entity = _is_entity_label(token.label)
        box = word.pixel_box
        draw.rectangle(
            [box.x0, box.y0, box.x1, box.y1],
            outline="red" if entity else "deepskyblue",
            width=4 if entity else 1,
        )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    render_responsive_image(
        buffer.getvalue(),
        caption="Model bounding boxes · red = predicted entity token, blue = background",
    )


def render_document_analysis(analysis: Any) -> None:
    stage_badge(
        "Image read",
        f"{analysis.image.image_format.value} · {analysis.image.width}×{analysis.image.height} · content hash recorded",
        "ok",
    )
    run = analysis.model_run
    selected = analysis.selected_model_candidate
    rule_value = (
        " · ".join(item.invoice_number for item in analysis.rule_candidates)
        if analysis.rule_candidates
        else "No candidate"
    )
    model_value = selected.candidate.invoice_number if selected is not None else "No candidate"
    labeled_tokens = label_invoice_tokens(analysis.ocr, gate=analysis.gate)
    amount_tokens = tuple(
        token.text for token in labeled_tokens if token.label is TokenLabel.AMOUNT
    )
    amount_value = " ".join(amount_tokens) if amount_tokens else "No unique total"
    st.markdown(
        '<div class="pa-decision">'
        f'<div class="pa-decision-card"><span class="label">Anchored rule read</span><strong>{esc(rule_value)}</strong>'
        '<p>Looks only beside invoice labels such as Invoice No.</p></div>'
        f'<div class="pa-decision-card"><span class="label">Local model read</span><strong>{esc(model_value)}</strong>'
        '<p>Grounded back to the OCR token and its layout box.</p></div></div>',
        unsafe_allow_html=True,
    )
    if selected is None:
        st.error(f"Model status: {run.status.value} · {run.error_code or 'no candidate'} · {run.error_message or ''}")
    else:
        cols = st.columns(4)
        cols[0].metric("Invoice number", selected.candidate.invoice_number)
        cols[1].metric("OCR total candidate", amount_value)
        cols[2].metric("Model confidence", str(selected.minimum_confidence))
        fixture_source = st.session_state.get("eval-invoice-source") == "Use sample invoice"
        fixture_result = (
            "Exact match" if analysis.strict_exact else "Not exact"
        ) if fixture_source else "Not scored"
        cols[3].metric("Fixture evaluation", fixture_result)
        st.caption(
            f"Rule/model agreement: {'yes' if rule_value == model_value else 'no'} · model runtime "
            f"{run.latency_ms} ms. Confidence is not accuracy. The fixture answer key is used "
            "only after inference and was not provided to OCR or the model."
        )
    if run.token_predictions:
        render_annotated_invoice_image(analysis.image.image_bytes, analysis.ocr.words, run.token_predictions)
        with st.expander("Ryan model · every word's raw label and confidence", expanded=True):
            render_table(
                [{"#": index, "word": token.word, "label": token.label,
                  "confidence": str(token.confidence), "margin": str(token.margin),
                  "box": f"({token.box[0]},{token.box[1]},{token.box[2]},{token.box[3]})"}
                 for index, token in enumerate(run.token_predictions)],
                highlight=lambda row: _is_entity_label(row["label"]),
            )
    gate = analysis.gate
    detail = gate.status.value + (" · " + " · ".join(gate.reason_codes) if gate.reason_codes else "")
    stage_badge("Safety gate", detail, "ok" if gate.may_activate_lookup else "review")
    if not gate.may_activate_lookup:
        st.warning(
            "This invoice needs a person to decide. Rule/model agreement never overrides "
            "the frozen score thresholds."
        )
    st.info(
        "Ryan's local model reads only the invoice number. The highlighted total is a separate "
        "OCR + anchored-rule candidate; the payment plan still uses the exact Accounts Payable "
        "record retrieved only after human confirmation."
    )
    render_ocr_result(
        analysis.ocr,
        "Every invoice token",
        labels=invoice_token_labels(analysis),
        default_label="Not invoice number",
    )


def render_prepared(prepared: Any) -> None:
    invoice = prepared.looked_up_invoice
    stage_badge(
        "Exact invoice found",
        f"{invoice.supplier_id} + {invoice.invoice_number} · locked synthetic business record",
        "ok",
    )
    cols = st.columns(4)
    cols[0].metric("Accounts Payable", format_minor(invoice.amount_minor))
    cols[1].metric("Inventory left", f"{invoice.inventory_days_remaining} days")
    cols[2].metric("Due in", f"{invoice.due_in_days} days")
    cols[3].metric("Supplier importance", invoice.supplier_criticality.value.title())
    st.caption(
        "The amount and business context came from exact lookup after human-reviewed identity; "
        "the invoice-number model did not extract them."
    )
    st.markdown("#### Today's proposed plan")
    render_batch_cards(prepared.batch, "verified proposal")
    verification = prepared.verification
    stage_badge(
        "Plan safety check",
        f"{verification.result.value} · {' · '.join(verification.reason_codes)}",
        "stop" if verification.result is VerifierResult.BLOCKED else "review",
    )
    st.caption("Checks passed: " + " · ".join(verification.checks_passed))
    st.info(
        "Fresh Farms uses the invoice you reviewed. The other three suppliers use locked "
        "fixture identities so this four-invoice demo stays reproducible."
    )


def render_simulation(simulation: Any) -> None:
    stage_badge(
        "Operator approval recorded",
        f"{simulation.approved_batch.operator_decision.decision.value} · day {simulation.info['day_before']}→{simulation.info['day_after']} · simulation_only=True",
        "ok",
    )
    cols = st.columns(4)
    cols[0].metric("Cash before", format_minor(simulation.info["cash_before_minor"]))
    cols[1].metric("Cash after", format_minor(simulation.info["cash_after_minor"]))
    cols[2].metric("Step reward", str(simulation.reward))
    cols[3].metric("State version", str(simulation.state_after.state_version))
    st.write("Simulated as paid: **" + ", ".join(simulation.info["paid_invoice_numbers"]) + "**")
    st.caption(
        f"Decision `{simulation.approved_batch.operator_decision.decision_id}` · "
        "no real money moved."
    )


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
    parsed = receipt.parsed
    render_ocr_result(
        receipt.ocr,
        "Every receipt token",
        labels=receipt_token_labels(receipt.ocr, parsed),
    )
    stage_badge(
        "Receipt fields read",
        f"{parsed.status.value} · {parsed.extraction_method}",
        "ok" if parsed.status.value == "READY_FOR_PROOF" else "review",
    )
    fields = {"Receipt ID": parsed.receipt_id, "Supplier": parsed.supplier_name, "Supplier ID": parsed.supplier_id,
              "Invoice": parsed.invoice_number, "Amount": format_minor(parsed.amount_minor) if parsed.amount_minor is not None else None,
              "Currency": parsed.currency, "Paid date": parsed.paid_date}
    render_table([{"Field": key, "Parsed value": value} for key, value in fields.items()])
    gate = receipt.proof_gate
    detail = gate.status.value + (" · " + " · ".join(gate.reason_codes) if gate.reason_codes else "")
    stage_badge("Exact payment-proof check", detail, "ok" if gate.closes_obligation else "stop")
    st.caption("Checks passed: " + (" · ".join(gate.checks_passed) or "none"))
    st.caption(f"Source: {receipt.source.value} · provenance: {receipt.provenance}")
    if gate.closes_obligation:
        st.success(
            "The supplier, invoice number, amount, currency and receipt ID all pass. "
            "Accounts Payable stays SIMULATED_PAYMENT_APPROVED until your separate confirmation."
        )
    else:
        st.error("The receipt did not pass every exact check. Accounts Payable remains open.")


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


def render_journal_entry(prepared: Any) -> None:
    """Explain the Fresh Farms portion of the simulated payment entry."""

    amount = format_minor(prepared.looked_up_invoice.amount_minor)
    st.markdown(
        '<div class="pa-journal" aria-label="Simulated accounting entry">'
        '<div class="pa-journal-entry"><span>Debit</span><b>Accounts Payable — Fresh Farms</b>'
        f'<strong>{esc(amount)}</strong><br><small>Liability decreases</small></div>'
        '<div class="pa-journal-entry"><span>Credit</span><b>Cash</b>'
        f'<strong>{esc(amount)}</strong><br><small>Asset decreases</small></div>'
        '<div class="pa-journal-note">This entry belongs to the approved simulated payment. '
        'Receipt confirmation later adds evidence and changes status to PAID_CONFIRMED; '
        'it does not post this entry or deduct cash a second time.</div></div>',
        unsafe_allow_html=True,
    )


def render_completed_summaries(current_step: int) -> None:
    analysis = st.session_state.get("eval-document-analysis")
    human = st.session_state.get("eval-human-decision")
    prepared = st.session_state.get("eval-prepared")
    simulation = st.session_state.get("eval-simulation")
    if current_step > 1 and analysis is not None:
        selected = analysis.selected_model_candidate
        value = selected.candidate.invoice_number if selected is not None else "No candidate"
        with st.expander(f"Completed · Invoice read · {value}", expanded=False):
            st.write(
                f"{len(analysis.ocr.words)} OCR words · OCR {analysis.ocr.status.value} · "
                f"document gate {analysis.gate.status.value}"
            )
            st.caption(
                f"Document `{analysis.image.document_id}` · SHA-256 `{analysis.image.sha256}`"
            )
    if current_step > 2 and human is not None and prepared is not None:
        with st.expander(
            f"Completed · Human review · {human.decision.value} {human.reviewed_invoice_number}",
            expanded=False,
        ):
            st.write(
                f"Exact AP lookup: {format_minor(prepared.looked_up_invoice.amount_minor)} · "
                f"plan verifier: {prepared.verification.result.value}"
            )
            st.caption(f"Review `{human.review_id}` · lookup was blocked until this decision.")
    if current_step > 3 and simulation is not None:
        with st.expander("Completed · Operator-approved simulation", expanded=False):
            st.write(
                f"Cash {format_minor(simulation.info['cash_before_minor'])} → "
                f"{format_minor(simulation.info['cash_after_minor'])} · day "
                f"{simulation.info['day_before']} → {simulation.info['day_after']}"
            )
            st.caption("Simulation only · no real bank payment was sent.")


def render_step_read_invoice() -> None:
    with st.container(border=True):
        render_step_header(
            1,
            "Read the invoice",
            "Use the included Fresh Farms sample or upload a PNG/JPEG. Nothing runs until you ask it to.",
            "Waiting for you",
        )
        st.info(
            "Accounts Payable is money the business owes a supplier. First, we read only "
            "the invoice identity—not an instruction to pay."
        )
        controls, preview = st.columns([0.82, 1.18])
        with controls:
            source_choice = st.radio(
                "Choose an invoice",
                ("Use sample invoice", "Upload my invoice"),
                key="eval-invoice-source",
            )
            supplier_choice = st.selectbox(
                "Which supplier sent it?",
                ("Fresh Farms",),
                index=None,
                placeholder="Choose the supplier",
                key="eval-supplier",
                help="This explicit supplier choice is part of the identity safety boundary.",
            )
            supplier_id = "fresh_farms" if supplier_choice == "Fresh Farms" else None
            if source_choice == "Upload my invoice":
                upload = st.file_uploader(
                    "Invoice image",
                    type=("png", "jpg", "jpeg"),
                    key="eval-invoice-upload",
                    help="PNG or JPEG. The file is processed inside this app runtime.",
                )
                invoice_bytes = upload.getvalue() if upload is not None else b""
                invoice_name = upload.name if upload is not None else "invoice-upload"
            else:
                invoice_bytes = read_bytes(INVOICE_PATH)
                invoice_name = INVOICE_PATH.name
                st.caption("Included sample: `data/procureagent/assets/fresh_farms_invoice.png`")
                st.download_button(
                    "Download sample invoice",
                    invoice_bytes,
                    INVOICE_PATH.name,
                    "image/png",
                    key="guided-download-invoice",
                    use_container_width=True,
                )
        with preview:
            if invoice_bytes:
                render_responsive_image(
                    invoice_bytes,
                    caption=(
                        "Synthetic Fresh Farms sample invoice"
                        if source_choice == "Use sample invoice"
                        else f"Uploaded invoice · {invoice_name}"
                    ),
                )
            else:
                st.warning("Choose an image to continue.")

        input_key = (
            f"{hashlib.sha256(invoice_bytes).hexdigest()}|{supplier_id or ''}"
            if invoice_bytes
            else ""
        )
        if st.session_state.get("eval-document-input-key") != input_key:
            clear_flow()
            for key in GUIDED_WIDGET_KEYS:
                st.session_state.pop(key, None)
            st.session_state["eval-document-input-key"] = input_key

        run_document = st.button(
            "Read invoice and find its number",
            key="eval-run-document-adapter",
            type="primary",
            disabled=not invoice_bytes or supplier_id is None,
            use_container_width=True,
        )
        if run_document and invoice_bytes and supplier_id is not None:
            clear_flow()
            with st.spinner("Reading OCR words, token boxes and the local invoice-number model…"):
                try:
                    analysis = analyze_invoice_upload(
                        invoice_bytes,
                        filename=invoice_name,
                        supplier_id=supplier_id,
                    )
                except Exception as exc:
                    st.error(f"Document pipeline stopped safely: {type(exc).__name__}: {exc}")
                else:
                    st.session_state["eval-document-analysis"] = analysis
                    st.rerun()
        stage_badge("Invoice pipeline", "NOT RUN · click required", "review")
        render_technical_evidence(
            sources=(("OCR, model and safety gate", "src/procureagent/ui_adapters.py", 201, 215),),
            answer_key_note=True,
        )


def render_step_confirm_and_plan(analysis: Any) -> None:
    with st.container(border=True):
        render_step_header(
            2,
            "Confirm the invoice, then build a plan",
            "Review every token and make an explicit identity decision before any business lookup can run.",
            "Human decision required",
        )
        render_document_analysis(analysis)
        st.markdown("#### Your decision")
        st.caption("No choice is preselected. Confirm, correct, or reject what the model displayed.")
        review_choice = st.radio(
            "Document review decision",
            tuple(REVIEW_DECISIONS),
            index=None,
            key="eval-document-review-choice",
        )
        review_decision = REVIEW_DECISIONS.get(review_choice)
        correction = st.text_input(
            "Correct invoice number",
            value="",
            placeholder="Enter the exact invoice number",
            key="eval-corrected-reference",
            disabled=review_decision != "CORRECT",
        )
        existing_human = st.session_state.get("eval-human-decision")
        if existing_human is not None and (
            existing_human.decision.value != review_decision
            or (
                review_decision == "CORRECT"
                and existing_human.reviewed_invoice_number != correction
            )
        ):
            st.session_state.pop("eval-human-decision", None)
            clear_flow("eval-human-decision")
            existing_human = None

        can_record = review_decision is not None and (
            review_decision != "CORRECT" or bool(correction.strip())
        )
        if st.button(
            "Record my invoice decision",
            key="eval-record-human-review",
            type="primary",
            disabled=not can_record,
            use_container_width=True,
        ):
            clear_flow("eval-document-analysis")
            try:
                human = record_human_identity_decision(
                    analysis,
                    DocumentReviewDecision(review_decision),
                    corrected_invoice_number=(
                        correction if review_decision == "CORRECT" else None
                    ),
                )
                st.session_state["eval-human-decision"] = human
                if human.may_activate_lookup:
                    with st.spinner("Looking up the exact AP record and checking today's plan…"):
                        st.session_state["eval-prepared"] = prepare_procurement(human)
                    st.rerun()
            except Exception as exc:
                st.error(f"Human review stopped safely: {type(exc).__name__}: {exc}")

        human = st.session_state.get("eval-human-decision")
        if human is None:
            stage_badge("Human review", "WAITING · business lookup blocked", "review")
        elif not human.may_activate_lookup:
            stage_badge(
                "Human review",
                f"{human.decision.value} · lookup blocked · review {human.review_id}",
                "stop",
            )
            st.error("This document was rejected. No payable was activated and no plan was changed.")

        st.button(
            "Choose a different invoice",
            key="eval-back-to-invoice",
            on_click=reset_guided_flow,
            use_container_width=True,
        )
        selected = analysis.selected_model_candidate
        render_technical_evidence(
            sources=(
                ("Explicit human identity gate", "src/procureagent/ui_adapters.py", 235, 282),
                ("Exact lookup and plan verifier", "src/procureagent/ui_adapters.py", 304, 324),
            ),
            runtime={
                "document_id": analysis.image.document_id,
                "sha256": analysis.image.sha256,
                "ocr": f"{analysis.ocr.engine}:{analysis.ocr.engine_version}",
                "ocr_words": len(analysis.ocr.words),
                "ocr_runtime_ms": str(analysis.ocr.runtime_ms),
                "model": analysis.model_run.model_version,
                "model_runtime_ms": str(analysis.model_run.latency_ms),
                "displayed_candidate": (
                    selected.candidate.invoice_number if selected is not None else None
                ),
                "document_gate": analysis.gate.status.value,
                "gate_reasons": list(analysis.gate.reason_codes),
            },
            answer_key_note=True,
        )


def render_step_approve(prepared: Any) -> None:
    with st.container(border=True):
        render_step_header(
            3,
            "Approve the simulation",
            "Inspect the verified plan and accounting effect. The restaurant state remains unchanged until you approve.",
            "Operator approval required",
        )
        render_prepared(prepared)
        st.markdown("#### Fresh Farms accounting entry inside this plan")
        render_journal_entry(prepared)
        blocked = prepared.verification.result is VerifierResult.BLOCKED
        operator_confirmed = st.checkbox(
            "I approve this verified batch and understand it advances the demo by one simulated day.",
            value=False,
            key="eval-operator-confirmation",
            disabled=blocked,
        )
        st.warning(
            "The plan verifier has run, but nothing has been committed. This approval is "
            "simulation-only and cannot send a bank payment."
        )
        if st.button(
            "Approve batch and run one simulated day",
            key="eval-approve-batch",
            type="primary",
            disabled=blocked or not operator_confirmed,
            use_container_width=True,
        ):
            clear_flow("eval-prepared")
            with st.spinner("Applying the approved plan to the isolated ProcureGym state…"):
                try:
                    simulation = approve_and_simulate(prepared)
                except Exception as exc:
                    st.error(f"Approval/simulation stopped safely: {type(exc).__name__}: {exc}")
                else:
                    st.session_state["eval-simulation"] = simulation
                    st.rerun()
        stage_badge("Operator + simulation", "NOT COMMITTED · restaurant state unchanged", "review")

        if st.button("Back to invoice review", key="eval-back-to-review"):
            clear_flow("eval-document-analysis")
            for key in ("eval-document-review-choice", "eval-corrected-reference"):
                st.session_state.pop(key, None)
            st.rerun()
        render_technical_evidence(
            sources=(
                ("Approval then one isolated step", "src/procureagent/ui_adapters.py", 334, 348),
                ("Simulation-only state transition", "src/procureagent/gym.py", 250, 282),
            ),
            runtime={
                "batch_id": prepared.batch.batch_id,
                "policy": f"{prepared.batch.policy_name}:{prepared.batch.policy_version}",
                "verifier": prepared.verification.result.value,
                "checks_passed": list(prepared.verification.checks_passed),
                "state_before": prepared.scenario.initial_state.state_version,
            },
        )


def render_step_match_receipt(simulation: Any) -> None:
    confirmed = st.session_state.get("eval-confirmed-payment")
    receipt = st.session_state.get("eval-receipt-analysis")
    status = "Complete" if confirmed is not None else (
        "Proof ready" if receipt is not None and receipt.proof_gate.closes_obligation else "Receipt required"
    )
    with st.container(border=True):
        render_step_header(
            4,
            "Match the receipt",
            "Read payment proof, match every field to Fresh Farms, then explicitly close only that demo payable.",
            status,
        )
        render_simulation(simulation)
        st.markdown("#### Accounting entry already created by the simulated payment")
        render_journal_entry(simulation.prepared)
        st.info(
            "Receipt confirmation supplies evidence and changes the demo status to "
            "PAID_CONFIRMED. It does not deduct cash again."
        )

        receipt_source = st.radio(
            "Choose a receipt",
            ("Use sample receipt", "Upload my receipt"),
            key="eval-receipt-source",
            disabled=confirmed is not None,
        )
        image_col, input_col = st.columns([1.1, .9])
        if receipt_source == "Upload my receipt":
            with input_col:
                receipt_upload = st.file_uploader(
                    "Receipt image",
                    type=("png", "jpg", "jpeg"),
                    key="eval-receipt-upload",
                    disabled=confirmed is not None,
                )
                receipt_bytes = (
                    receipt_upload.getvalue() if receipt_upload is not None else b""
                )
                receipt_name = (
                    receipt_upload.name if receipt_upload is not None else "receipt-upload"
                )
                proof_source = PaymentProofSource.OPERATOR_UPLOAD
                provenance = f"operator_upload:{receipt_name}"
        else:
            receipt_bytes = read_bytes(RECEIPT_PATH)
            receipt_name = RECEIPT_PATH.name
            proof_source = PaymentProofSource.SYNTHETIC_FIXTURE_REPLAY
            provenance = "bundled_deterministic_svg_fixture; see receipt_provenance.json"
            with input_col:
                st.caption(
                    "Included sample: `data/procureagent/assets/fresh_farms_payment_receipt.png`"
                )
                st.download_button(
                    "Download sample receipt",
                    receipt_bytes,
                    RECEIPT_PATH.name,
                    "image/png",
                    key="guided-download-receipt",
                    disabled=confirmed is not None,
                    use_container_width=True,
                )
        with image_col:
            if receipt_bytes:
                render_responsive_image(
                    receipt_bytes,
                    caption=(
                        "Synthetic Fresh Farms payment receipt"
                        if receipt_source == "Use sample receipt"
                        else f"Uploaded receipt · {receipt_name}"
                    ),
                )
            else:
                st.warning("Choose a receipt image to continue.")

        receipt_key = hashlib.sha256(receipt_bytes).hexdigest() if receipt_bytes else ""
        if (
            confirmed is None
            and st.session_state.get("eval-receipt-input-key") != receipt_key
        ):
            st.session_state.pop("eval-receipt-analysis", None)
            st.session_state.pop("eval-proof-confirmation", None)
            st.session_state["eval-receipt-input-key"] = receipt_key
            receipt = None

        if st.button(
            "Read receipt and match payment proof",
            key="eval-run-receipt-adapter",
            type="primary",
            disabled=not receipt_bytes or confirmed is not None,
            use_container_width=True,
        ):
            with st.spinner("Reading receipt tokens and checking every payment field…"):
                try:
                    receipt = analyze_receipt_upload(
                        simulation,
                        receipt_bytes,
                        filename=receipt_name,
                        source=proof_source,
                        provenance=provenance,
                    )
                except Exception as exc:
                    st.session_state.pop("eval-receipt-analysis", None)
                    st.error(f"Receipt pipeline stopped safely: {type(exc).__name__}: {exc}")
                else:
                    st.session_state["eval-receipt-analysis"] = receipt

        receipt = st.session_state.get("eval-receipt-analysis")
        if receipt is None:
            stage_badge("Receipt proof", "NOT RUN · Accounts Payable remains open", "review")
        else:
            render_receipt_result(receipt)

        proof_ready = receipt is not None and receipt.proof_gate.closes_obligation
        proof_confirmed = st.checkbox(
            "I confirm this exact receipt is valid payment evidence for Fresh Farms invoice FF-10482 at $1,500.00.",
            value=False,
            key="eval-proof-confirmation",
            disabled=not proof_ready or confirmed is not None,
        )
        confirm_payment = st.button(
            "Confirm proof and mark Accounts Payable PAID_CONFIRMED",
            key="eval-confirm-payment",
            type="primary",
            disabled=not proof_ready or not proof_confirmed or confirmed is not None,
            use_container_width=True,
        )
        if confirm_payment and confirmed is None and receipt is not None and proof_ready:
            with st.spinner("Recording the verified receipt and closing this demo payable…"):
                try:
                    confirmed = confirm_verified_payment(receipt)
                except Exception as exc:
                    st.error(f"Payment confirmation stopped safely: {type(exc).__name__}: {exc}")
                else:
                    st.session_state["eval-confirmed-payment"] = confirmed
                    st.rerun()

        confirmed = st.session_state.get("eval-confirmed-payment")
        if confirmed is None:
            stage_badge(
                "Accounts Payable status",
                "SIMULATED_PAYMENT_APPROVED · waiting for verified receipt confirmation",
                "review",
            )
        else:
            stage_badge(
                "Accounts Payable status",
                f"{confirmed.payment_status.value} · state version {confirmed.state_after.state_version}",
                "ok",
            )
            before_cash = confirmed.state_before.cash_minor
            after_cash = confirmed.state_after.cash_minor
            cols = st.columns(3)
            cols[0].metric("AP status", confirmed.payment_status.value)
            cols[1].metric("Cash at receipt confirmation", format_minor(after_cash))
            cols[2].metric("Second cash deduction", format_minor(before_cash - after_cash))
            st.success(
                "PAID_CONFIRMED in the simulated AP ledger. Verified proof was consumed "
                "once; no real bank payment was sent and cash was not deducted again."
            )

        runtime: dict[str, Any] = {
            "simulation_only": simulation.info["simulation_only"],
            "operator_decision_id": simulation.approved_batch.operator_decision.decision_id,
            "cash_after_simulation": format_minor(simulation.info["cash_after_minor"]),
            "ap_before_receipt": "SIMULATED_PAYMENT_APPROVED",
        }
        if receipt is not None:
            runtime.update(
                {
                    "receipt_document_id": receipt.image.document_id,
                    "receipt_sha256": receipt.image.sha256,
                    "ocr": f"{receipt.ocr.engine}:{receipt.ocr.engine_version}",
                    "ocr_words": len(receipt.ocr.words),
                    "ocr_runtime_ms": str(receipt.ocr.runtime_ms),
                    "parser": receipt.parsed.extraction_method,
                    "proof_gate": receipt.proof_gate.status.value,
                    "proof_checks": list(receipt.proof_gate.checks_passed),
                }
            )
        if confirmed is not None:
            runtime["ap_after_receipt"] = confirmed.payment_status.value
            runtime["cash_deducted_again"] = before_cash != after_cash
        render_technical_evidence(
            sources=(
                ("Receipt OCR, parse and exact proof gate", "src/procureagent/ui_adapters.py", 381, 396),
                ("Evidence-only AP confirmation", "src/procureagent/ui_adapters.py", 408, 421),
            ),
            runtime=runtime,
        )


def render_eval() -> None:
    """Render the default four-step guided demo (legacy name kept for tests)."""

    step = current_guided_step()
    render_progress_rail(step)
    render_completed_summaries(step)
    if step == 1:
        render_step_read_invoice()
    elif step == 2:
        render_step_confirm_and_plan(st.session_state["eval-document-analysis"])
    elif step == 3:
        render_step_approve(st.session_state["eval-prepared"])
    else:
        render_step_match_receipt(st.session_state["eval-simulation"])


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
        st.info("C1/C2 OCR and model adapters are installed but stay lazy until the Guided demo click.")
    if ROUTER_RESULT is None:
        st.warning(f"C6 Router Lab unavailable; no result claimed. {ROUTER_ERROR}")
    else:
        st.info(
            "C6 Router Lab executed on 7 synthetic development rows in repeated training "
            "context bins; no frozen test, generalization, or live-document claim."
        )


def render_route_hero(route: str) -> None:
    copy = {
        "Guided demo": (
            "Small-business accounts payable · guided demo",
            "Turn one invoice into a closed payable—with proof.",
            "Read an invoice, confirm its identity, approve a simulated plan, then match the receipt. "
            "Each step explains the business outcome first and keeps the real code one click away.",
            "Four clear steps · sample files included",
        ),
        "Overview": (
            "Restaurant cash planning · locked scenario",
            "See what is owed and what the plan protects.",
            "Four supplier invoices compete for limited cash. This view explains the deterministic "
            "recommendation without accepting a document or changing state.",
            "Read-only overview · exact synthetic records",
        ),
        "Evidence & methods": (
            "Engineering evidence · bounded claims",
            "Inspect the code, tests and evaluation boundaries.",
            "Review the real source chain, OCR/model provenance, deterministic comparisons, "
            "adversarial boundary and downloadable demo fixtures.",
            "Technical detail · no hidden mutation",
        ),
    }[route]
    kicker, title, description, badge = copy
    st.markdown(
        f'<section class="pa-hero"><div class="pa-kicker">{esc(kicker)}</div>'
        f'<h1>{esc(title)}</h1><p>{esc(description)}</p>'
        f'<span class="pa-badge">{esc(badge)}</span></section>',
        unsafe_allow_html=True,
    )


def render_headline_metrics() -> None:
    cash = (
        OVERVIEW.scenario.initial_state.cash_minor
        if OVERVIEW is not None
        else PRIMARY_SCENARIO["cash_minor"]
    )
    obligations = (
        OVERVIEW.scenario.initial_state.total_obligations_minor
        if OVERVIEW is not None
        else PRIMARY_SCENARIO["obligations_minor"]
    )
    version = (
        OVERVIEW.scenario.initial_state.state_version
        if OVERVIEW is not None
        else PRIMARY_SCENARIO["state_version"]
    )
    headline = st.columns(4)
    headline[0].metric("Cash available", format_minor(cash), "Day 0")
    headline[1].metric("Supplier obligations", format_minor(obligations), "4 invoices")
    headline[2].metric("Funding gap", format_minor(obligations - cash), "Needs prioritization")
    headline[3].metric("State version", str(version), f"Seed {PRIMARY_SCENARIO['seed']}")


def render_overview_route() -> None:
    render_headline_metrics()
    st.markdown('<div class="pa-section">Current supplier bills</div>', unsafe_allow_html=True)
    st.subheader("Four bills compete for $5,000")
    st.caption(
        "Business fields are exact synthetic lookup data. Action labels are actual deterministic policy output."
    )
    render_invoice_cards()
    st.info(
        "The four canonical invoices total $6,200. UnknownCo is separate, activates no "
        "payable, and is excluded."
    )
    render_batch()


def render_evidence_route() -> None:
    st.warning(FIXTURE_NOTICE)
    render_recording_kit()
    render_code_provenance()
    st.divider()
    render_document_evidence()
    st.divider()
    render_gym()
    st.divider()
    render_task_status()


st.markdown(CSS, unsafe_allow_html=True)
nav_col, reset_col = st.columns([3.2, .8])
with nav_col:
    route = st.radio(
        "Choose a view",
        ROUTES,
        key="top-route",
        horizontal=True,
    )
with reset_col:
    if st.button("Restart demo", key="eval-reset-flow", use_container_width=True):
        reset_guided_flow()
        st.rerun()

render_route_hero(route)
render_trust_strip()
if route == "Guided demo":
    st.caption(
        "Synthetic restaurant scenario · OCR and the local invoice-number model run only "
        "after you click · every financial action stays simulated."
    )
    render_eval()
elif route == "Overview":
    render_overview_route()
else:
    render_evidence_route()
