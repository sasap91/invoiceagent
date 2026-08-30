"""InvoiceAgent's controlled Streamlit demo and recording surface.

Run with: ``streamlit run procure_app.py``.

The overview executes dependency-light deterministic P0 code. OCR and the LayoutLMv3
local model execute only after the Guided demo document button; simulation and AP
closure each require their own later operator click.
"""

from __future__ import annotations

import base64
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
    InvoicePaymentStatus,
    PaymentProofSource,
    ProcurementAction,
    VerifierResult,
)
from procureagent.document import align_model_token_predictions  # noqa: E402
from procureagent.ui_adapters import (  # noqa: E402
    UiFlowError,
    analyze_invoice_upload,
    analyze_receipt_upload,
    approve_and_simulate,
    confirm_verified_payment,
    load_overview_run,
    prepare_procurement,
    record_human_identity_decision,
)
from procureagent.router_lab import run_router_lab  # noqa: E402
from procureagent.receipt_reward import (  # noqa: E402
    ReceiptMatchAction,
    score_receipt_match,
)
from procureagent.token_labels import (  # noqa: E402
    TokenLabel,
    TokenSource,
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
RESTAURANT_HERO_PATH = ASSET_DIR / "invoiceagent-restaurant-hero.jpg"


st.set_page_config(
    page_title="InvoiceAgent · Invoice-to-receipt demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');

  :root {
    --pa-bg:#f8f7f3;
    --pa-surface:#ffffff;
    --pa-surface-subtle:#f5f4ef;
    --pa-ink:#183029;
    --pa-muted:#52645f;
    --pa-line:#dce2dc;
    --pa-line-strong:#bac8c0;
    --pa-accent:#a6421f;
    --pa-accent-dark:#78301b;
    --pa-accent-soft:#fff0e8;
    --pa-brand:#153f35;
    --pa-proof:#08778a;
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
  body, .stApp, [class*="css"] { font-family:'Manrope', sans-serif; }
  .stApp { background:var(--pa-bg); color:var(--pa-ink); }
  [data-testid="stSidebar"] { background:#edf2ee; border-right:1px solid var(--pa-line); }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:var(--pa-muted); }
  [data-testid="stHeader"] { display:none; }
  .block-container { max-width:1180px; padding-top:1.45rem; padding-bottom:4rem; }
  h1,h2,h3,h4,h5 { color:var(--pa-ink); font-family:'Fraunces', serif; letter-spacing:-.025em; text-wrap:balance; }
  p, li { line-height:1.58; }
  a { color:var(--pa-accent-dark); text-underline-offset:3px; }

  .pa-brandbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:0 0 .75rem;
    padding:.35rem .1rem; }
  .pa-wordmark { display:flex; align-items:center; gap:.7rem; color:var(--pa-brand); font-weight:800; letter-spacing:-.025em; }
  .pa-monogram { display:grid; place-items:center; width:2.2rem; height:2.2rem; border-radius:9px; background:var(--pa-brand);
    color:#fff; font-family:'Fraunces',serif; font-size:.88rem; letter-spacing:-.03em; }
  .pa-brandbar small { color:var(--pa-muted); text-align:right; line-height:1.35; }
  .pa-byline { color:var(--pa-accent-dark); font-size:.68rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
  .pa-hero { position:relative; overflow:hidden; padding:clamp(1.25rem,3vw,1.85rem); border-radius:var(--pa-radius-lg);
    border:1px solid var(--pa-line); border-left:7px solid var(--pa-accent); background:var(--pa-surface); margin-bottom:var(--pa-space-4); }
  .pa-hero--visual { min-height:330px; display:flex; align-items:center; background-image:
    linear-gradient(90deg, rgba(248,247,243,.99) 0%, rgba(248,247,243,.96) 36%, rgba(248,247,243,.7) 57%, rgba(248,247,243,.08) 78%),
    var(--pa-hero-image); background-size:cover; background-position:center; }
  .pa-hero-copy { position:relative; z-index:1; max-width:58%; }
  .pa-kicker { color:var(--pa-accent-dark); font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:.72rem; }
  .pa-hero h1 { max-width:24ch; font-size:clamp(2rem,4vw,3.1rem); line-height:1.02; margin:.35rem 0 .55rem; }
  .pa-hero p { color:var(--pa-muted); max-width:740px; font-size:1.04rem; line-height:1.58; margin:0; }
  .pa-badge { display:inline-flex; align-items:center; margin-top:1rem; border-radius:999px; padding:.5rem .78rem;
    border:1px solid #e8c3b3; background:var(--pa-accent-soft); color:var(--pa-accent-dark); font-weight:700; font-size:.75rem; letter-spacing:.015em; }
  .pa-hero-note { display:block; margin-top:.65rem; color:var(--pa-muted); font-size:.64rem; line-height:1.4; }
  .pa-section { color:var(--pa-accent-dark); font-size:.7rem; font-weight:700; letter-spacing:.12em;
    text-transform:uppercase; margin:.25rem 0; }

  .pa-trust-strip { position:static; display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
    gap:1px; margin:.7rem 0 1rem; border:1px solid var(--pa-line-strong); border-radius:var(--pa-radius-md);
    overflow:hidden; background:var(--pa-line); box-shadow:var(--pa-shadow-sm); }
  .pa-trust-item { background:#f8faf8; padding:.72rem .85rem; }
  .pa-trust-item b { display:block; color:var(--pa-ink); font-size:.8rem; }
  .pa-trust-item span { color:var(--pa-muted); font-size:.72rem; }

  .pa-tech-line { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1px; margin:.9rem 0 1.15rem;
    border:1px solid var(--pa-line-strong); border-radius:var(--pa-radius-md); overflow:hidden; background:var(--pa-line); }
  .pa-tech-item { min-width:0; padding:.82rem .9rem; background:var(--pa-surface-subtle); }
  .pa-tech-item span { display:block; color:var(--pa-accent-dark); font-size:.64rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
  .pa-tech-item b { display:block; color:var(--pa-ink); margin:.18rem 0; font-size:.82rem; }
  .pa-tech-item small { display:block; color:var(--pa-muted); line-height:1.35; font-size:.68rem; }

  .pa-reward { margin:.9rem 0; padding:1rem 1.05rem; border:1px solid #b5d8ca; border-left:5px solid var(--pa-success);
    border-radius:var(--pa-radius-md); background:var(--pa-success-soft); }
  .pa-reward span { display:block; color:var(--pa-success); font-size:.65rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; }
  .pa-reward b { display:block; margin:.24rem 0; color:var(--pa-ink); font-size:.95rem; }
  .pa-reward p { margin:.15rem 0 0; color:var(--pa-muted); font-size:.76rem; }

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
    background:var(--pa-accent-soft); color:var(--pa-accent-dark); box-shadow:0 0 0 3px rgba(166,66,31,.08); }
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
  .pa-token.muted { opacity:.28; filter:saturate(.2); border-color:#647873; background:#17302d; }
  .pa-token-map.focus-preview { max-height:none; overflow:visible; background:#eef1ed; border-color:#d7ddd8; }
  .pa-token-map.focus-preview .pa-token.muted { color:#52625e; background:#e5e9e5; border-color:#cbd3ce; }
  .pa-token-map.focus-preview .pa-token.muted small { color:#71807b; }
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
  .pa-decision-card strong { display:block; margin:.3rem 0; color:var(--pa-ink); font-family:'Fraunces',serif; font-size:1.55rem; }
  .pa-decision-card p { color:var(--pa-muted); font-size:.8rem; margin:.2rem 0 0; }

  .pa-journal { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1px; margin:1rem 0;
    border:1px solid var(--pa-line-strong); border-radius:var(--pa-radius-md); overflow:hidden; background:var(--pa-line); }
  .pa-journal-entry { background:var(--pa-surface); padding:1rem; }
  .pa-journal-entry span { color:var(--pa-muted); text-transform:uppercase; letter-spacing:.1em; font-size:.65rem; font-weight:700; }
  .pa-journal-entry b { display:block; color:var(--pa-ink); font-size:1rem; margin:.35rem 0; }
  .pa-journal-entry strong { color:var(--pa-accent-dark); font-family:'Fraunces',serif; font-size:1.4rem; }
  .pa-journal-note { grid-column:1/-1; background:#f3f7f4; padding:.75rem 1rem; color:var(--pa-muted); font-size:.78rem; }

  .pa-history-head { margin:1.1rem 0 .85rem; padding:1.15rem 1.25rem; border-radius:var(--pa-radius-lg);
    border:1px solid #a9cdbf; border-left:6px solid var(--pa-success); background:var(--pa-success-soft); }
  .pa-history-head span { color:var(--pa-success); font-size:.66rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
  .pa-history-head h2 { margin:.28rem 0 .25rem; font-size:clamp(1.55rem,3vw,2.2rem); }
  .pa-history-head p { margin:0; color:var(--pa-muted); max-width:72ch; }
  .pa-ledger-list { display:grid; gap:.55rem; margin:.65rem 0 1rem; }
  .pa-ledger-row { display:grid; grid-template-columns:1.25fr .9fr .8fr 1fr; align-items:center; gap:.8rem;
    padding:.85rem .95rem; border:1px solid var(--pa-line); border-radius:var(--pa-radius-sm); background:var(--pa-surface); }
  .pa-ledger-row b { display:block; color:var(--pa-ink); }
  .pa-ledger-row small { display:block; margin-top:.12rem; color:var(--pa-muted); line-height:1.35; }
  .pa-ledger-amount { color:var(--pa-ink); font-weight:800; font-variant-numeric:tabular-nums; }
  .pa-ledger-status { justify-self:end; padding:.35rem .52rem; border-radius:999px; font-size:.62rem; font-weight:800;
    letter-spacing:.04em; text-transform:uppercase; text-align:center; }
  .pa-ledger-status.open { color:var(--pa-warning); background:var(--pa-warning-soft); }
  .pa-ledger-status.awaiting { color:var(--pa-accent-dark); background:var(--pa-accent-soft); }
  .pa-ledger-status.completed { color:var(--pa-success); background:var(--pa-success-soft); }
  .pa-history-empty { padding:1rem; border:1px dashed var(--pa-line-strong); border-radius:var(--pa-radius-sm);
    color:var(--pa-muted); background:var(--pa-surface-subtle); }

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
  [data-testid="stFileUploaderDropzone"] button { min-height:44px; border-radius:10px; font-family:'Manrope',sans-serif;
    font-weight:700; transition:background-color .18s ease,border-color .18s ease,box-shadow .18s ease,transform .18s ease; cursor:pointer; }
  [data-testid="stButton"] button[kind="primary"] { background:var(--pa-accent); color:#fff; border-color:var(--pa-accent); }
  [data-testid="stButton"] button[kind="primary"]:hover { background:var(--pa-accent-dark); border-color:var(--pa-accent-dark); }
  [data-testid="stButton"] button:disabled,[data-testid="stDownloadButton"] button:disabled {
    cursor:not-allowed; opacity:1; background:#e5ebe7 !important; color:#5a6d68 !important; border-color:#cbd6d0 !important; }
  [data-baseweb="radio"] label, [data-baseweb="checkbox"] label { min-height:44px; cursor:pointer; }
  input, textarea, [data-baseweb="select"] > div { min-height:44px; }
  button:focus-visible, input:focus-visible, textarea:focus-visible, [tabindex]:focus-visible {
    outline:3px solid rgba(166,66,31,.48) !important; outline-offset:2px !important; box-shadow:none !important; }
  [data-testid="stExpander"] { border:1px solid var(--pa-line); border-radius:var(--pa-radius-sm); background:rgba(255,255,255,.7); }
  [data-testid="stExpander"] summary { min-height:44px; cursor:pointer; }

  @media (hover:hover) {
    [data-testid="stButton"] button:not(:disabled):hover, [data-testid="stDownloadButton"] button:not(:disabled):hover { transform:translateY(-1px); }
  }
  @media (max-width:900px) {
    .pa-grid,.pa-decision { grid-template-columns:1fr; }
    .pa-tech-line { grid-template-columns:1fr; }
    .pa-batch { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .pa-progress { gap:.35rem; }
    .pa-progress li { padding:.62rem .45rem .62rem 2.15rem; }
    .pa-progress .step-no { left:.48rem; top:.59rem; }
    .pa-progress small { display:none; }
  }
  @media (max-width:640px) {
    .block-container { padding-left:1rem; padding-right:1rem; padding-top:1rem; }
    .pa-trust-strip { position:static; grid-template-columns:repeat(2,minmax(0,1fr)); }
    .pa-trust-item { padding:.58rem .62rem; }
    .pa-brandbar { align-items:flex-start; }
    .pa-brandbar small { max-width:11rem; font-size:.67rem; }
    .pa-hero { padding:1.05rem 1.15rem; border-left-width:5px; }
    .pa-hero--visual { min-height:350px; align-items:flex-end; background-image:
      linear-gradient(180deg, rgba(248,247,243,.18) 0%, rgba(248,247,243,.82) 43%, rgba(248,247,243,.99) 72%),
      var(--pa-hero-image); background-position:68% center; }
    .pa-hero-copy { max-width:100%; }
    .pa-hero h1 { font-size:2rem; line-height:1.04; }
    .pa-hero p { font-size:1rem; line-height:1.5; }
    .pa-badge { border-radius:12px; font-size:.68rem; }
    .pa-progress { grid-template-columns:repeat(4,1fr); }
    .pa-progress li { min-height:3rem; padding:.55rem .3rem; text-align:center; }
    .pa-progress .step-no { position:static; margin:0 auto .2rem; }
    .pa-progress b { font-size:.62rem; }
    .pa-step-head { flex-direction:column; }
    .pa-journal,.pa-batch,.pa-facts { grid-template-columns:1fr; }
    .pa-journal-note { grid-column:auto; }
    .pa-ledger-row { grid-template-columns:1fr 1fr; }
    .pa-ledger-status { justify-self:start; }
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

# Streamlit preserves Python objects across source hot reloads. Contract classes
# can therefore look identical while failing ``isinstance`` after a class was
# redefined. Bump this value whenever workflow object schemas change; unrelated
# preferences such as the selected route remain untouched.
APP_SESSION_SCHEMA = "invoiceagent-guided-v3-layoutlm-tokens-20260830"
if st.session_state.get("invoiceagent-session-schema") != APP_SESSION_SCHEMA:
    for stale_key in (
        *FLOW_KEYS,
        *GUIDED_WIDGET_KEYS,
        "eval-document-input-key",
        "eval-receipt-input-key",
        "eval-document-kind-error",
        "eval-ap-history-visible",
    ):
        st.session_state.pop(stale_key, None)
    st.session_state["invoiceagent-session-schema"] = APP_SESSION_SCHEMA

ROUTES = ("Guided demo", "Overview", "Evidence & methods")
GITHUB_BLOB = "https://github.com/sasap91/invoiceagent/blob/main"

REVIEW_DECISIONS = {
    "Confirm the displayed invoice number": "CONFIRM",
    "Correct the invoice number": "CORRECT",
    "Reject this document": "REJECT",
}

STEP_META = (
    (1, "Scan invoice", "Tesseract + model"),
    (2, "Verify fields", "Human review"),
    (3, "Simulate payment", "Dr AP · Cr Cash"),
    (4, "Match receipt", "Close the loop"),
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

TOKEN_SOURCE_DISPLAY_LABELS = {
    TokenSource.OCR_ONLY: "ocr only",
    TokenSource.INVOICE_ANCHORED_RULE: "invoice anchored rule",
    TokenSource.RYAN_INVOICE_NUMBER_MODEL: "LayoutLMv3 invoice-number model",
    TokenSource.INVOICE_RULE_AND_RYAN_MODEL: "invoice rule and LayoutLMv3 model",
    TokenSource.INVOICE_AMOUNT_RULE: "invoice amount rule",
    TokenSource.RECEIPT_FIELD_RULE: "receipt field rule",
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


def looks_like_payment_receipt(ocr: Any) -> bool:
    """Return true only when at least two strong receipt markers are present."""

    raw_text = str(getattr(ocr, "raw_text", "") or "")
    if not raw_text:
        raw_text = " ".join(
            str(getattr(word, "text", "")) for word in getattr(ocr, "words", ())
        )
    normalized = " ".join(raw_text.upper().split())
    markers = ("PAYMENT RECEIPT", "RECEIPT ID", "PAID IN FULL")
    return sum(marker in normalized for marker in markers) >= 2


def render_responsive_image(image_bytes: bytes, *, caption: str) -> None:
    """Use the image-width API supported by both CI and model environments."""

    supports_stretch = "use_container_width" in inspect.signature(st.image).parameters
    st.image(image_bytes, caption=caption, width="stretch" if supports_stretch else 900)


def clear_flow(after: str | None = None) -> None:
    st.session_state.pop("eval-ap-history-visible", None)
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
        "eval-document-kind-error",
        "eval-ap-history-visible",
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
          <div class="pa-trust-item"><b>Synthetic documents</b><span>No restaurant records are used</span></div>
          <div class="pa-trust-item"><b>No affiliation</b><span>Sugar &amp; Spice is the demo setting only</span></div>
          <div class="pa-trust-item"><b>Human controlled</b><span>Review before every state change</span></div>
          <div class="pa-trust-item"><b>Simulation only</b><span>No bank or ERP is connected</span></div>
        </aside>
        """,
        unsafe_allow_html=True,
    )


def render_ai_lineup() -> None:
    """Show the narrow AI boundary before anyone runs the invoice reader."""

    st.markdown(
        """
        <div class="pa-tech-line" aria-label="What is AI and what is deterministic">
          <div class="pa-tech-item"><span>Local OCR</span><b>Tesseract reads every word</b>
          <small>Text, confidence and bounding boxes stay in this app runtime.</small></div>
          <div class="pa-tech-item"><span>Small local AI</span><b>LayoutLMv3 + LoRA · adapted by Ryan</b>
          <small>The supervised specialist identifies the invoice number only.</small></div>
          <div class="pa-tech-item"><span>Deterministic checks</span><b>Amount and receipt matching</b>
          <small>Rules verify exact fields; they are not presented as model predictions.</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reward_signal(proof_gate: Any) -> Any:
    """Render the tested reward adapter without implying online model training."""

    action = (
        ReceiptMatchAction.ACCEPT_MATCH
        if proof_gate.closes_obligation
        else ReceiptMatchAction.REQUEST_REVIEW
    )
    result = score_receipt_match(proof_gate, action)
    explanation = (
        "This exact outcome can train a future router to choose accept, retry OCR, or human "
        "review. The supervised LayoutLMv3 invoice-number specialist was not trained or updated here."
    )
    st.markdown(
        f'<aside class="pa-reward"><span>RL-ready evaluation signal · no policy/model was trained</span>'
        f'<b>{esc(result.outcome)} · reward {esc(result.reward)} · action {esc(result.action.value)}</b>'
        f'<p>{esc(explanation)}</p></aside>',
        unsafe_allow_html=True,
    )
    return result


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
            f"{TOKEN_SOURCE_DISPLAY_LABELS[token.source]}"
        )
        for token in tokens
        if token.label is not TokenLabel.OTHER
    }


def receipt_token_labels(ocr: Any, parsed: Any) -> dict[int, str]:
    tokens = label_receipt_tokens(ocr, parsed)
    return {
        token.index: (
            f"{TOKEN_DISPLAY_LABELS[token.label]} · "
            f"{TOKEN_SOURCE_DISPLAY_LABELS[token.source]}"
        )
        for token in tokens
        if token.label is not TokenLabel.OTHER
    }


def render_token_map(
    words: Any,
    label_by_sequence: dict[int, str],
    *,
    default_label: str = "Other text",
    muted_default: bool = False,
    preview: bool = False,
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
                "pa-token target" if target else (
                    "pa-token muted" if muted_default else "pa-token"
                )
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
        f'<div class="pa-token-map{" focus-preview" if preview else ""}" '
        f'aria-label="OCR tokens and labels">{"".join(tokens)}</div>',
        unsafe_allow_html=True,
    )
    return rows


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


def ap_history_buckets(confirmed: Any) -> dict[str, tuple[Any, ...]]:
    """Group the confirmed restaurant state by its authoritative payment enum."""

    buckets: dict[str, list[Any]] = {
        "open": [],
        "awaiting_proof": [],
        "completed": [],
    }
    for invoice in confirmed.state_after.invoices:
        if invoice.payment_status is InvoicePaymentStatus.UNPAID:
            buckets["open"].append(invoice)
        elif invoice.payment_status is InvoicePaymentStatus.SIMULATED_PAYMENT_APPROVED:
            buckets["awaiting_proof"].append(invoice)
        elif invoice.payment_status is InvoicePaymentStatus.PAID_CONFIRMED:
            buckets["completed"].append(invoice)
    return {name: tuple(items) for name, items in buckets.items()}


def due_label(days: int) -> str:
    if days == 0:
        return "Due today"
    if days < 0:
        count = abs(days)
        return f"{count} day{'s' if count != 1 else ''} overdue"
    return f"Due in {days} day{'s' if days != 1 else ''}"


def render_ap_rows(
    invoices: tuple[Any, ...],
    *,
    names: dict[str, str],
    actions: dict[tuple[str, str], str],
    bucket: str,
) -> None:
    """Render state-derived AP rows with words as well as status color."""

    if not invoices:
        st.markdown(
            '<div class="pa-history-empty">No invoices in this category.</div>',
            unsafe_allow_html=True,
        )
        return
    status_copy = {
        "open": lambda invoice: f"OPEN · {actions.get((invoice.supplier_id, invoice.invoice_number), 'REVIEW')}",
        "awaiting_proof": lambda _invoice: "PAID · AWAITING PROOF",
        "completed": lambda _invoice: "COMPLETED · PROOF MATCHED",
    }
    status_class = {
        "open": "open",
        "awaiting_proof": "awaiting",
        "completed": "completed",
    }[bucket]
    rows = []
    for invoice in invoices:
        status = status_copy[bucket](invoice)
        rows.append(
            '<div class="pa-ledger-row">'
            f'<div><b>{esc(names.get(invoice.supplier_id, invoice.supplier_id))}</b>'
            f'<small>{esc(invoice.invoice_number)} · {esc(invoice.category.title())}</small></div>'
            f'<div><b>{esc(due_label(invoice.due_in_days))}</b>'
            f'<small>{esc(invoice.payment_status.value)}</small></div>'
            f'<div class="pa-ledger-amount">{esc(format_minor(invoice.amount_minor))}</div>'
            f'<div class="pa-ledger-status {status_class}">{esc(status)}</div></div>'
        )
    st.markdown(
        f'<div class="pa-ledger-list" aria-label="{esc(bucket.replace("_", " "))} invoices">'
        f'{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def render_ap_history_dashboard(confirmed: Any) -> None:
    """Render a read-only AP lifecycle view from the post-proof state."""

    simulation = confirmed.receipt_analysis.simulation
    scenario = simulation.prepared.scenario
    state = confirmed.state_after
    buckets = ap_history_buckets(confirmed)
    names = {supplier.supplier_id: supplier.display_name for supplier in scenario.suppliers}
    actions = {
        (item.supplier_id, item.invoice_number): item.action.value
        for item in simulation.prepared.batch.recommendations
    }
    open_ap_minor = sum(invoice.amount_minor for invoice in buckets["open"])
    awaiting_minor = sum(
        invoice.amount_minor for invoice in buckets["awaiting_proof"]
    )

    st.markdown(
        '<section class="pa-history-head"><span>Accounts Payable history · state-derived</span>'
        '<h2>Every supplier bill now has a clear next state.</h2>'
        '<p>InvoiceAgent separates bills still open, simulated payments waiting for proof, '
        'and completed payments with an auditable receipt trail.</p></section>',
        unsafe_allow_html=True,
    )
    metrics = st.columns(4)
    metrics[0].metric("Cash after batch", format_minor(state.cash_minor))
    metrics[1].metric("Remaining open AP", format_minor(open_ap_minor))
    metrics[2].metric("Paid · awaiting proof", format_minor(awaiting_minor))
    metrics[3].metric("Completed invoices", str(len(buckets["completed"])))
    st.info(
        "Working-capital support view—not full Net Working Capital (NWC). Accounts "
        "Receivable, inventory valuation, other current assets, and other current "
        "liabilities are outside this demo's scope."
    )

    paid_identities = {
        (item.supplier_id, item.invoice_number)
        for item in simulation.approved_batch.batch.recommendations
        if item.action is ProcurementAction.PAY
    }
    paid_components = tuple(
        invoice
        for invoice in state.invoices
        if (invoice.supplier_id, invoice.invoice_number) in paid_identities
    )
    batch_total_minor = sum(invoice.amount_minor for invoice in paid_components)
    with st.expander(
        "Journal entry interpretation · full simulated batch",
        expanded=True,
    ):
        st.warning(
            "Accounting interpretation of the isolated simulated transition—not a journal "
            "posted to a bank, ERP, or general ledger. Both paid invoice components explain "
            "the batch cash change."
        )
        batch_rows = [
            {
                "Side": "Debit",
                "Account": f"Accounts Payable — {names.get(invoice.supplier_id, invoice.supplier_id)}",
                "Invoice": invoice.invoice_number,
                "Amount": format_minor(invoice.amount_minor),
            }
            for invoice in paid_components
        ]
        batch_rows.append(
            {
                "Side": "Credit",
                "Account": "Cash",
                "Invoice": "Batch total",
                "Amount": format_minor(batch_total_minor),
            }
        )
        render_table(batch_rows)
        st.caption(
            f"Cash {format_minor(simulation.info['cash_before_minor'])} → "
            f"{format_minor(simulation.info['cash_after_minor'])} · balanced batch "
            f"{format_minor(batch_total_minor)}."
        )

    open_tab, awaiting_tab, completed_tab = st.tabs(
        (
            f"Open invoices ({len(buckets['open'])})",
            f"Paid · awaiting proof ({len(buckets['awaiting_proof'])})",
            f"Completed ({len(buckets['completed'])})",
        )
    )
    with open_tab:
        st.caption("Still unpaid after the simulated batch; no receipt can close these yet.")
        render_ap_rows(buckets["open"], names=names, actions=actions, bucket="open")
    with awaiting_tab:
        st.caption(
            "Payment was simulated, but exact receipt proof has not been attached. The AP "
            "record remains SIMULATED_PAYMENT_APPROVED."
        )
        render_ap_rows(
            buckets["awaiting_proof"],
            names=names,
            actions=actions,
            bucket="awaiting_proof",
        )
    with completed_tab:
        st.caption("Verified receipt proof completed this AP lifecycle without another cash entry.")
        render_ap_rows(
            buckets["completed"],
            names=names,
            actions=actions,
            bucket="completed",
        )
        proof = confirmed.receipt_analysis.proof_gate.proof
        if proof is None:
            st.error("Completed history stopped safely: verified receipt proof is missing.")
            return
        completed_invoice = next(
            (
                invoice
                for invoice in buckets["completed"]
                if invoice.supplier_id == proof.supplier_id
                and invoice.invoice_number == proof.invoice_number
            ),
            None,
        )
        if completed_invoice is None:
            st.error("Completed history stopped safely: proof does not match a completed invoice.")
            return
        completed_name = names.get(
            completed_invoice.supplier_id, completed_invoice.supplier_id
        )
        awaiting_summary = " · ".join(
            f"{names.get(invoice.supplier_id, invoice.supplier_id)} "
            f"{format_minor(invoice.amount_minor)}"
            for invoice in buckets["awaiting_proof"]
        ) or "none"
        st.markdown(f"#### Auditable {completed_name} journal component")
        st.caption(
            f"This is the {completed_name} component of the earlier "
            f"{format_minor(batch_total_minor)} simulated batch interpretation; "
            f"{awaiting_summary} remains Paid · awaiting proof."
        )
        render_table(
            [
                {
                    "Entry line": "Debit",
                    "Account": f"Accounts Payable — {completed_name}",
                    "Debit": format_minor(completed_invoice.amount_minor),
                    "Credit": "—",
                },
                {
                    "Entry line": "Credit",
                    "Account": "Cash",
                    "Debit": "—",
                    "Credit": format_minor(completed_invoice.amount_minor),
                },
            ]
        )
        render_table(
            [
                {
                    "Supplier": names.get(proof.supplier_id, proof.supplier_id),
                    "Invoice": proof.invoice_number,
                    "Receipt": proof.receipt_id,
                    "Proof date": proof.paid_date.isoformat(),
                    "Source": proof.source.value,
                }
            ]
        )
        second_cash_hit = confirmed.state_before.cash_minor - confirmed.state_after.cash_minor
        no_second_hit = second_cash_hit == 0
        st.success(
            f"Receipt confirmation cash impact: {format_minor(second_cash_hit)}. "
            f"Second cash hit: {'NO' if no_second_hit else 'YES'}. The receipt added proof "
            "and PAID_CONFIRMED status only; it posted no second journal entry."
        )
        st.caption(
            f"Simulation day {simulation.info['day_before']} → {simulation.info['day_after']} · "
            f"simulation_only={simulation.info['simulation_only']} · state version {state.state_version}."
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
        st.warning("Static stored evidence only. For actual Tesseract + LayoutLMv3 evidence, use the Guided demo.")
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
    compact_targets: bool = False,
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
    selected_labels = labels or {}
    rows = build_token_label_rows(
        ocr.words,
        selected_labels,
        default_label=default_label,
    )
    if compact_targets:
        target_sequences = tuple(
            row["sequence"] for row in rows if row["label"] != default_label
        )
        if target_sequences:
            visible_sequences = {
                sequence
                for target in target_sequences
                for sequence in range(max(0, target - 4), target + 5)
            }
            preview_words = tuple(
                word for word in ocr.words if word.sequence in visible_sequences
            )
            st.caption(
                f"Focused preview · {len(target_sequences)} bookkeeping token(s) highlighted "
                f"from {len(rows)} OCR tokens"
            )
            render_token_map(
                preview_words,
                selected_labels,
                default_label=default_label,
                muted_default=True,
                preview=True,
            )
        else:
            st.warning("OCR completed, but no target bookkeeping token was grounded.")
    else:
        render_token_map(
            ocr.words,
            selected_labels,
            default_label=default_label,
        )
    with st.expander(
        f"Inspect all {len(rows)} token boxes and confidence scores",
        expanded=False,
    ):
        if compact_targets:
            render_token_map(
                ocr.words,
                selected_labels,
                default_label=default_label,
                muted_default=True,
            )
        render_table(rows)
        st.code(ocr.raw_text or "<no OCR text>")


def _is_layoutlm_entity_label(label: str) -> bool:
    return label not in {"O", "LABEL_0", "NOT_EVALUATED"}


def layoutlm_token_rows(analysis: Any) -> list[dict[str, Any]]:
    """Return every OCR word with honest model-evaluation provenance."""

    aligned = align_model_token_predictions(analysis.model_run, analysis.ocr)
    rows: list[dict[str, Any]] = []
    for index, (word, prediction) in enumerate(zip(analysis.ocr.words, aligned)):
        rows.append(
            {
                "OCR #": index,
                "Word": word.text,
                "LayoutLMv3 label": (
                    prediction.label if prediction is not None else "NOT_EVALUATED"
                ),
                "Confidence": (
                    str(prediction.confidence) if prediction is not None else "—"
                ),
                "Margin": str(prediction.margin) if prediction is not None else "—",
                "Status": "MODEL_EVALUATED" if prediction is not None else "TOKENIZER_TRUNCATED",
                "Box": (
                    f"({word.normalized_box.x0},{word.normalized_box.y0},"
                    f"{word.normalized_box.x1},{word.normalized_box.y1})"
                ),
            }
        )
    return rows


def render_annotated_invoice_image(analysis: Any) -> None:
    """Draw only aligned model evidence; truncated words remain visibly distinct."""

    try:
        from io import BytesIO

        from PIL import Image, ImageDraw
    except ImportError:
        st.warning("Pillow is unavailable, so the model-box overlay cannot be drawn.")
        return
    aligned = align_model_token_predictions(analysis.model_run, analysis.ocr)
    try:
        image = Image.open(BytesIO(analysis.image.image_bytes)).convert("RGB")
    except Exception as exc:
        st.warning(f"Could not render the model-box overlay: {type(exc).__name__}")
        return
    draw = ImageDraw.Draw(image)
    for word, prediction in zip(analysis.ocr.words, aligned):
        if prediction is None:
            color, width = "#9ca3af", 1
        elif _is_layoutlm_entity_label(prediction.label):
            color, width = "#dc2626", 4
        else:
            color, width = "#0284c7", 1
        box = word.pixel_box
        draw.rectangle((box.x0, box.y0, box.x1, box.y1), outline=color, width=width)
    output = BytesIO()
    image.save(output, format="PNG")
    render_responsive_image(
        output.getvalue(),
        caption=(
            "LayoutLMv3 evidence overlay · red = invoice-number token · blue = evaluated "
            "background · gray = tokenizer did not evaluate"
        ),
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
        token_rows = layoutlm_token_rows(analysis)
        evaluated = sum(row["Status"] == "MODEL_EVALUATED" for row in token_rows)
        render_annotated_invoice_image(analysis)
        with st.expander(
            f"LayoutLMv3 token evidence · {evaluated}/{len(token_rows)} OCR words evaluated",
            expanded=True,
        ):
            st.caption(
                "These are raw, OCR-index-aligned model outputs. NOT_EVALUATED means the "
                "tokenizer truncated that word; it is never presented as model-predicted background."
            )
            render_table(token_rows)
    else:
        st.caption("Per-token LayoutLMv3 evidence is unavailable for this model run.")
    gate = analysis.gate
    detail = gate.status.value + (" · " + " · ".join(gate.reason_codes) if gate.reason_codes else "")
    stage_badge("Safety gate", detail, "ok" if gate.may_activate_lookup else "review")
    if not gate.may_activate_lookup:
        st.warning(
            "This invoice needs a person to decide. Rule/model agreement never overrides "
            "the frozen score thresholds."
        )
    st.info(
        "The small local LayoutLMv3 document specialist reads only the invoice number. The highlighted total "
        "comes from Tesseract plus a separate anchored rule. The exact Accounts Payable record "
        "stays locked until a person confirms the invoice identity."
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
    cols[0].metric("Vendor", "Fresh Farms")
    cols[1].metric("Invoice", invoice.invoice_number)
    cols[2].metric("Amount due", format_minor(invoice.amount_minor))
    cols[3].metric("Due in", f"{invoice.due_in_days} days")
    st.caption(
        "After human review, the exact synthetic Accounts Payable record supplies the amount. "
        "LayoutLMv3 did not extract the payable amount."
    )
    verification = prepared.verification
    stage_badge(
        "Payment safety check",
        f"{verification.result.value} · {' · '.join(verification.reason_codes)}",
        "stop" if verification.result is VerifierResult.BLOCKED else "review",
    )
    st.caption("Checks passed: " + " · ".join(verification.checks_passed))
    with st.expander("Advanced cash-priority policy · optional technical context"):
        st.caption(
            "The guided story follows Fresh Farms only. Underneath, the deterministic restaurant-day "
            "simulator evaluates four locked synthetic bills so its cash constraints remain reproducible."
        )
        render_batch_cards(prepared.batch, "verified proposal")


def render_simulation(simulation: Any) -> None:
    stage_badge(
        "Operator approval recorded",
        f"{simulation.approved_batch.operator_decision.decision.value} · day {simulation.info['day_before']}→{simulation.info['day_after']} · simulation_only=True",
        "ok",
    )
    selected = simulation.prepared.looked_up_invoice
    cols = st.columns(4)
    cols[0].metric("Invoice", selected.invoice_number)
    cols[1].metric("Payment amount", format_minor(selected.amount_minor))
    cols[2].metric("AP status", "PAYMENT SIMULATED")
    cols[3].metric("Next proof", "Receipt")
    st.success("The Fresh Farms accounting entry is now simulated. No bank payment was sent.")
    with st.expander("Advanced restaurant-day simulation output"):
        details = st.columns(4)
        details[0].metric("Cash before", format_minor(simulation.info["cash_before_minor"]))
        details[1].metric("Cash after", format_minor(simulation.info["cash_after_minor"]))
        details[2].metric("Policy reward", str(simulation.reward))
        details[3].metric("State version", str(simulation.state_after.state_version))
        st.write("All simulated payments: **" + ", ".join(simulation.info["paid_invoice_numbers"]) + "**")
        st.caption(
            f"Decision `{simulation.approved_batch.operator_decision.decision_id}` · "
            "the policy reward describes the restaurant-day simulator, not receipt matching."
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


def render_receipt_result(receipt: Any) -> Any:
    parsed = receipt.parsed
    render_ocr_result(
        receipt.ocr,
        "Receipt ID evidence",
        labels=receipt_token_labels(receipt.ocr, parsed),
        compact_targets=True,
    )
    gate = receipt.proof_gate
    receipt_id_grounded = (
        "PARSED_RECEIPT_BOUND_TO_OCR" in gate.checks_passed
        and parsed.receipt_id is not None
        and any(
            item.field_name == "receipt_id" and item.value == parsed.receipt_id
            for item in parsed.evidence
        )
    )
    if receipt_id_grounded:
        stage_badge(
            "Receipt ID captured",
            f"{parsed.receipt_id} · grounded in the uploaded OCR evidence",
            "ok",
        )
        primary = st.columns(3)
        primary[0].metric("Receipt ID", parsed.receipt_id)
        primary[1].metric("OCR tokens processed", str(len(receipt.ocr.words)))
        primary[2].metric("Grounding", "OCR MATCH")
    else:
        stage_badge("Receipt ID captured", "MISSING · manual review required", "stop")
    stage_badge(
        "Receipt fields read",
        f"{parsed.status.value} · {parsed.extraction_method}",
        "ok" if parsed.status.value == "READY_FOR_PROOF" else "review",
    )
    fields = {"Receipt ID": parsed.receipt_id, "Supplier": parsed.supplier_name, "Supplier ID": parsed.supplier_id,
              "Invoice": parsed.invoice_number, "Amount": format_minor(parsed.amount_minor) if parsed.amount_minor is not None else None,
              "Currency": parsed.currency, "Paid date": parsed.paid_date}
    fields = {key: value if value is not None else "Not found" for key, value in fields.items()}
    field_rows = [{"Field": key, "Parsed value": value} for key, value in fields.items()]
    missing_payment_fields = sum(
        value is None
        for value in (
            parsed.supplier_id,
            parsed.invoice_number,
            parsed.amount_minor,
            parsed.currency,
            parsed.paid_date,
        )
    )
    if gate.closes_obligation:
        render_table(field_rows)
    else:
        with st.expander("Manual matching details · why this ID cannot close AP yet"):
            render_table(field_rows)
            st.caption(
                "A receipt ID identifies the uploaded document, but it does not by itself "
                "prove which supplier invoice or payment amount it belongs to."
            )
            st.caption(
                "Technical reason codes: "
                + (" · ".join(gate.reason_codes) if gate.reason_codes else "none")
            )
    detail = (
        gate.status.value
        if gate.closes_obligation
        else (
            f"Payment proof incomplete · {missing_payment_fields} required fields missing"
            if missing_payment_fields
            else "Payment proof incomplete · exact fields do not match"
        )
    )
    stage_badge("Exact payment-proof check", detail, "ok" if gate.closes_obligation else "review")
    st.caption("Checks passed: " + (" · ".join(gate.checks_passed) or "none"))
    st.caption(f"Source: {receipt.source.value} · provenance: {receipt.provenance}")
    if gate.closes_obligation:
        st.success(
            "The receipt ID, supplier, invoice number, full amount, currency, and paid date all pass. "
            "Accounts Payable stays SIMULATED_PAYMENT_APPROVED until your separate confirmation."
        )
    else:
        if receipt_id_grounded:
            st.warning(
                f"Receipt ID {parsed.receipt_id} was captured from OCR. Supplier, invoice "
                "number, full amount, currency, and paid date are still required, so this "
                "is not payment proof. Accounts Payable remains SIMULATED_PAYMENT_APPROVED; "
                "no second cash entry or PAID_CONFIRMED status was created."
            )
        else:
            st.error(
                "No grounded receipt ID was captured. Payment proof remains pending and "
                "the lifecycle stays SIMULATED_PAYMENT_APPROVED."
            )
    reward = render_reward_signal(gate)
    if not gate.closes_obligation:
        st.caption("Receipt-ID capture alone receives SAFE_REVIEW · reward -1.0, never the +10 full-match reward.")
    return reward


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
        st.code("streamlit run procure_app.py\n# or\ndocker build -t invoiceagent . && docker run -p 8501:8501 invoiceagent")
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
        'Debit AP reduces what the restaurant owes; Credit Cash reduces its cash asset. '
        'Receipt confirmation later adds proof and changes status to PAID_CONFIRMED; '
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
                f"payment verifier: {prepared.verification.result.value}"
            )
            st.caption(f"Review `{human.review_id}` · lookup was blocked until this decision.")
    if current_step > 3 and simulation is not None:
        with st.expander("Completed · Payment simulated", expanded=False):
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
            "Scan the vendor invoice",
            "Start with the bundled sample or upload a PNG/JPEG. Nothing runs until you ask it to.",
            "Waiting for you",
        )
        st.info(
            "Sugar & Spice receives a paper bill from Fresh Farms. That bill becomes Accounts "
            "Payable: money the restaurant owes its vendor. This is a synthetic walkthrough "
            "with no affiliation to Sugar & Spice and no real business records."
        )
        render_ai_lineup()
        controls, preview = st.columns([0.82, 1.18])
        with controls:
            source_choice = st.radio(
                "Choose an invoice",
                ("Use sample invoice", "Upload my invoice"),
                key="eval-invoice-source",
            )
            supplier_choice = st.selectbox(
                "Confirm the vendor shown on the invoice",
                ("Fresh Farms",),
                index=None,
                placeholder="Select Fresh Farms",
                key="eval-supplier",
                help="A person must confirm the vendor before the demo can activate any payable.",
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
                st.caption("Bundled synthetic file: `data/procureagent/assets/fresh_farms_invoice.png`")
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
            st.session_state.pop("eval-document-kind-error", None)
            st.session_state["eval-document-input-key"] = input_key

        run_document = st.button(
            "Scan invoice with Tesseract + LayoutLMv3",
            key="eval-run-document-adapter",
            type="primary",
            disabled=not invoice_bytes or supplier_id is None,
            use_container_width=True,
        )
        if run_document and invoice_bytes and supplier_id is not None:
            clear_flow()
            with st.spinner("Tesseract is reading every word; LayoutLMv3 is finding the invoice number…"):
                try:
                    analysis = analyze_invoice_upload(
                        invoice_bytes,
                        filename=invoice_name,
                        supplier_id=supplier_id,
                    )
                except Exception as exc:
                    st.error(f"Document pipeline stopped safely: {type(exc).__name__}: {exc}")
                else:
                    if looks_like_payment_receipt(analysis.ocr):
                        st.session_state["eval-document-kind-error"] = (
                            "This appears to be a payment receipt—upload the supplier invoice first."
                        )
                    else:
                        st.session_state.pop("eval-document-kind-error", None)
                        st.session_state["eval-document-analysis"] = analysis
                        st.rerun()
        kind_error = st.session_state.get("eval-document-kind-error")
        if kind_error:
            st.error(kind_error)
            stage_badge("Invoice reader", "STOPPED SAFELY · receipt detected in invoice step", "stop")
        else:
            stage_badge("Invoice reader", "NOT RUN · click required", "review")
        render_technical_evidence(
            sources=(("OCR, model and safety gate", "src/procureagent/ui_adapters.py", 201, 215),),
            answer_key_note=True,
        )


def render_step_confirm_and_plan(analysis: Any) -> None:
    with st.container(border=True):
        render_step_header(
            2,
            "Verify what InvoiceAgent found",
            "See every OCR token, then confirm, correct, or reject the invoice number before bookkeeping continues.",
            "Human decision required",
        )
        render_document_analysis(analysis)
        st.markdown("#### Your decision")
        selected_model = analysis.selected_model_candidate
        if selected_model is not None:
            st.info(
                "LayoutLMv3 suggested invoice number: "
                f"**{esc(selected_model.candidate.invoice_number)}**"
            )
        elif analysis.rule_candidates:
            st.warning(
                "LayoutLMv3 found no invoice number. The separate anchored rule found "
                f"**{esc(analysis.rule_candidates[0].invoice_number)}**; use **Correct** "
                "to enter it after checking the image, or reject the document."
            )
        else:
            st.warning(
                "Neither LayoutLMv3 nor the anchored rule found an invoice number. "
                "Enter a checked correction or reject the document."
            )
        available_decisions = dict(REVIEW_DECISIONS)
        if selected_model is None:
            available_decisions.pop("Confirm the displayed invoice number")
            st.caption(
                "Confirm is unavailable because LayoutLMv3 found no candidate. "
                "Correct the invoice number after checking the image, or reject the document."
            )
        else:
            st.caption("No choice is preselected. Confirm, correct, or reject what the model displayed.")
        review_choice = st.radio(
            "Document review decision",
            tuple(available_decisions),
            index=None,
            key="eval-document-review-choice",
        )
        review_decision = available_decisions.get(review_choice)
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

        can_record = (
            review_decision == "REJECT"
            or (review_decision == "CORRECT" and bool(correction.strip()))
            or (review_decision == "CONFIRM" and selected_model is not None)
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
                    with st.spinner("Opening the exact synthetic AP record and running the payment safety check…"):
                        st.session_state["eval-prepared"] = prepare_procurement(human)
                    st.rerun()
            except UiFlowError as exc:
                if "absent from locked lookup" in str(exc):
                    st.error(
                        "That reviewed invoice number is not in this demo's locked Accounts "
                        "Payable records, so no obligation was created. Check the number or "
                        "choose a different invoice."
                    )
                else:
                    st.error(f"Human review stopped safely: {type(exc).__name__}: {exc}")
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
            st.error("This document was rejected. No payable was activated and no bookkeeping state changed.")

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
            "Simulate the payment",
            "Review the exact bookkeeping entry. The demo ledger stays unchanged until you explicitly approve.",
            "Operator approval required",
        )
        render_prepared(prepared)
        st.markdown("#### What paying Fresh Farms does to the books")
        render_journal_entry(prepared)
        blocked = prepared.verification.result is VerifierResult.BLOCKED
        operator_confirmed = st.checkbox(
            "I approve this verified demo payment run and understand no real money will move.",
            value=False,
            key="eval-operator-confirmation",
            disabled=blocked,
        )
        st.warning(
            "The safety verifier has run, but nothing has been recorded yet. This approval "
            "changes only the isolated demo ledger and cannot send a bank payment."
        )
        if st.button(
            "Approve and simulate payment",
            key="eval-approve-batch",
            type="primary",
            disabled=blocked or not operator_confirmed,
            use_container_width=True,
        ):
            clear_flow("eval-prepared")
            with st.spinner("Recording the approved payment in the isolated demo ledger…"):
                try:
                    simulation = approve_and_simulate(prepared)
                except Exception as exc:
                    st.error(f"Approval/simulation stopped safely: {type(exc).__name__}: {exc}")
                else:
                    st.session_state["eval-simulation"] = simulation
                    st.rerun()
        stage_badge("Payment simulation", "NOT RECORDED · demo ledger unchanged", "review")

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
        "Proof ready" if receipt is not None and receipt.proof_gate.closes_obligation else (
            "Receipt ID captured · proof incomplete"
            if receipt is not None and receipt.parsed.receipt_id is not None
            else "Receipt required"
        )
    )
    with st.container(border=True):
        render_step_header(
            4,
            "Match the payment receipt",
            "A receipt can confirm the simulated payment only when its receipt ID, supplier, invoice number, full amount, currency, and paid date all match.",
            status,
        )
        render_simulation(simulation)
        st.markdown("#### Accounting entry already created by the simulated payment")
        render_journal_entry(simulation.prepared)
        st.info(
            "The receipt proves the payment already simulated in step 3. Confirming it changes "
            "the AP status to PAID_CONFIRMED; it does not deduct cash again."
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
                    "Bundled synthetic file: `data/procureagent/assets/fresh_farms_payment_receipt.png`"
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
            "Scan receipt and compare exact fields",
            key="eval-run-receipt-adapter",
            type="primary",
            disabled=not receipt_bytes or confirmed is not None,
            use_container_width=True,
        ):
            with st.spinner("Tesseract is reading the receipt; exact rules are comparing every payment field…"):
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
            stage_badge("Receipt proof", "NOT RUN · scan a receipt to continue", "review")
        else:
            render_receipt_result(receipt)

        proof_ready = receipt is not None and receipt.proof_gate.closes_obligation
        proof_confirmed = st.checkbox(
            "I confirm this exact receipt proves payment of Fresh Farms invoice FF-10482 for $1,500.00.",
            value=False,
            key="eval-proof-confirmation",
            disabled=not proof_ready or confirmed is not None,
        )
        st.caption("Confirmation is unavailable until all six receipt fields match and the receipt ID is unused by a confirmed proof.")
        confirm_payment = st.button(
            "Confirm match and close Accounts Payable",
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
                "Complete: invoice FF-10482 and its receipt match. Accounts Payable is "
                "PAID_CONFIRMED in the simulated ledger; cash was not deducted a second time."
            )

        runtime: dict[str, Any] = {
            "simulation_only": simulation.info["simulation_only"],
            "operator_decision_id": simulation.approved_batch.operator_decision.decision_id,
            "cash_after_simulation": format_minor(simulation.info["cash_after_minor"]),
            "ap_before_receipt": "SIMULATED_PAYMENT_APPROVED",
        }
        if receipt is not None:
            receipt_reward = score_receipt_match(
                receipt.proof_gate,
                ReceiptMatchAction.ACCEPT_MATCH
                if receipt.proof_gate.closes_obligation
                else ReceiptMatchAction.REQUEST_REVIEW,
            )
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
                    "rl_ready_reward": str(receipt_reward.reward),
                    "rl_ready_outcome": receipt_reward.outcome,
                    "trained_policy": receipt_reward.trained_policy,
                }
            )
        if confirmed is not None:
            runtime["ap_after_receipt"] = confirmed.payment_status.value
            runtime["cash_deducted_again"] = before_cash != after_cash
        render_technical_evidence(
            sources=(
                ("Receipt OCR, parse and exact proof gate", "src/procureagent/ui_adapters.py", 381, 396),
                ("RL-ready receipt reward · evaluation only", "src/procureagent/receipt_reward.py", 75, 109),
                ("Evidence-only AP confirmation", "src/procureagent/ui_adapters.py", 408, 421),
            ),
            runtime=runtime,
        )

    confirmed = st.session_state.get("eval-confirmed-payment")
    if confirmed is not None:
        history_visible = bool(st.session_state.get("eval-ap-history-visible"))
        if not history_visible:
            if st.button(
                "Done — view AP history",
                key="eval-view-ap-history",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["eval-ap-history-visible"] = True
                st.rerun()
        else:
            render_ap_history_dashboard(confirmed)


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
            "InvoiceAgent · Cambridge restaurant walkthrough",
            "Paper invoice in. Paid proof out.",
            "Follow one synthetic Sugar & Spice Thai Restaurant bill from Fresh Farms invoice "
            "to matched receipt, with a person approving every bookkeeping step.",
            "Synthetic scenario · no affiliation with Sugar & Spice",
        ),
        "Overview": (
            "InvoiceAgent · advanced restaurant view",
            "See what is owed and what the policy protects.",
            "Four supplier invoices compete for limited cash. This view explains the deterministic "
            "recommendation without accepting a document or changing state.",
            "Read-only overview · exact synthetic records",
        ),
        "Evidence & methods": (
            "InvoiceAgent · engineering evidence",
            "Inspect the code, tests and evaluation boundaries.",
            "Review the real source chain, OCR/model provenance, deterministic comparisons, "
            "adversarial boundary and downloadable demo fixtures.",
            "Technical detail · no hidden mutation",
        ),
    }[route]
    kicker, title, description, badge = copy
    if route == "Guided demo" and RESTAURANT_HERO_PATH.exists():
        encoded = base64.b64encode(read_bytes(RESTAURANT_HERO_PATH)).decode("ascii")
        st.markdown(
            f'<section class="pa-hero pa-hero--visual" '
            f'style="--pa-hero-image:url(data:image/jpeg;base64,{encoded})">'
            f'<div class="pa-hero-copy"><div class="pa-kicker">{esc(kicker)}</div>'
            f'<h1>{esc(title)}</h1><p>{esc(description)}</p>'
            f'<span class="pa-badge">{esc(badge)}</span>'
            '<small class="pa-hero-note">Illustrative original image · not a photograph of Sugar &amp; Spice · '
            '<a href="https://sugarspices.com/" target="_blank" rel="noreferrer">restaurant context</a></small>'
            '</div></section>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<section class="pa-hero"><div class="pa-kicker">{esc(kicker)}</div>'
            f'<h1>{esc(title)}</h1><p>{esc(description)}</p>'
            f'<span class="pa-badge">{esc(badge)}</span></section>',
            unsafe_allow_html=True,
        )


def render_brandbar() -> None:
    st.markdown(
        '<header class="pa-brandbar"><div class="pa-wordmark">'
        '<span class="pa-monogram" aria-hidden="true">IA</span>'
        '<span>InvoiceAgent<br><span class="pa-byline">by Sundai</span></span></div>'
        '<small>Sugar &amp; Spice demo workspace<br>Porter Square · Cambridge</small></header>',
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
render_brandbar()
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
    render_eval()
elif route == "Overview":
    render_overview_route()
else:
    render_evidence_route()
