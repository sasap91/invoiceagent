# InvoiceAgent demo and recording runbook

This is the operator checklist for the 2–3 minute hackathon demo. It assumes the repository root is the working directory and no real banking or accounting integration exists.

## 1. Preflight before the room arrives

Install Python 3.12, Tesseract, and the combined demo/model environment:

```bash
python3.12 -m venv .venv-model
.venv-model/bin/pip install -e ".[demo,test,model]"
```

Run the complete offline proof:

```bash
.venv-model/bin/python scripts/eval_procureagent.py \
  --output data/procureagent/eval/acceptance_v1.json
```

Warm Ryan's checkpoint once while internet access is reliable:

```bash
HF_HUB_DISABLE_PROGRESS_BARS=1 \
  .venv-model/bin/python scripts/eval_procureagent.py \
  --with-model \
  --output data/procureagent/eval/acceptance_live_v1.json
```

Expected result: offline acceptance is **9/9** and live acceptance is **10/10**; the live candidate is `FF-10482`. The rule and model agree, but the document gate deliberately requests review because the entity score is below the frozen `0.80` threshold. That is the safeguard to show, not an error to hide.

The pinned adapter revision is `7dc28f5a3b14aa100ba432ee1b0a6cac6c7b2c5c`; the pinned base-model revision is `cfbbbff0762e6aab37086fdd4739ad14fe7d5db4`.

## 2. Launch locally or publicly

Local recording:

```bash
./scripts/run_demo.sh
```

Public room demo through the Mac:

```bash
./scripts/run_demo.sh --public
```

Keep the terminal open and copy the printed `https://…trycloudflare.com` URL. A Quick Tunnel is temporary, changes when restarted, and is a public unauthenticated endpoint with no access-control or uptime guarantee. It is appropriate for synthetic presentation fixtures only; never upload confidential invoices.

Useful assets:

- Invoice: `data/procureagent/assets/fresh_farms_invoice.png`
- Receipt: `data/procureagent/assets/fresh_farms_payment_receipt.png`
- Receipt provenance: `data/procureagent/assets/receipt_provenance.json`
- Captured model proof: `data/procureagent/eval/model_smoke_v1.json`
- Acceptance result: `data/procureagent/eval/acceptance_live_v1.json`
- Original illustrative restaurant hero: `data/procureagent/assets/invoiceagent-restaurant-hero.jpg` with adjacent provenance JSON — not a real Sugar & Spice photograph; no affiliation
- Dashboard screenshot: `docs/assets/invoiceagent-dashboard.png`
- Completed AP-history screenshot: `docs/assets/invoiceagent-eval-complete.png`
- Sundai thumbnail: `docs/assets/sundai-thumbnail.png` (1200×630)

## 3. Four-step guided talk track

### 0:00–0:20 — Set the problem

“This restaurant has $5,000, but four verified supplier bills total $6,200. Paying only the oldest invoice can leave the kitchen without produce or meat.”

Point out that this is Accounts Payable: money the restaurant owes suppliers.

### Step 1 · 0:20–0:55 — Read the invoice

Open the guided demo, keep **Use sample invoice** selected (or upload PNG/JPEG), choose **Fresh Farms** as the supplier, and run the document analysis. The bundled source is `data/procureagent/assets/fresh_farms_invoice.png`; its `FF-10482` answer key is evaluated only after inference and is never sent to OCR or the model.

Call out each boundary:

- Tesseract produced every word, OCR confidence, and normalized box.
- Every token remains inspectable with its label and provenance; ordinary words remain honestly labeled OCR-only.
- The local LayoutLMv3 token classifier adapted by Ryan selected only the invoice-number token `FF-10482`.
- The displayed invoice amount is grounded by OCR plus a separate deterministic total-anchor rule. LayoutLMv3 did not extract it.
- Strict exact match on this document is yes.
- Canonical AP amount, due date, inventory, and criticality come from the synthetic lookup after identity confirmation, not from the model.
- The score is below the auto-confirm threshold, so the gate asks for a person even though rule and model agree.

Use the **Technical evidence** expander only when the audience asks for depth. It exposes raw OCR, all token boxes/confidences/sources, document hash, anchored-rule evidence, LayoutLMv3 version/latency/scores, gate reasons, verifier details, and artifact provenance without crowding the guided story.

### Step 2 · 0:55–1:20 — Confirm identity and review the plan

Click the explicit identity **CONFIRM** action. **CORRECT** and **REJECT** are the fail-closed alternatives. Do not call the one-document exact result aggregate model accuracy.

Reveal the synthetic business lookup, then the policy result:

1. **Identity:** was `FF-10482` read correctly? Every supplier now has a bundled
   invoice image, so you can read all four documents rather than one. Each is
   verified to yield exactly one anchored candidate under real Tesseract; each
   still needs its own human CONFIRM, because the frozen 0.80 confidence
   threshold routes them to review.
2. **Priority:** first Fresh Farms, second Prime Foods, third PackRight, then CleanPro review. Show inventory runway and lead time beside the ranking.
3. **Action:** Fresh Farms $1,500 and Prime Foods $2,500 are exact verified full-payment actions. The agent cannot invent a supplier or amount.

Explain the plan before approving it: the deterministic policy proposes actions; the verifier checks identity, exact canonical amount, currency, cash, duplicates, and state version. Nothing has changed yet.

### Step 3 · 1:20–1:45 — Approve the simulated payment

Explicitly approve the reverified batch. ProcureGym should move cash from $5,000 to $1,000 and advance one day. The $4,000 total contains Fresh Farms $1,500 plus Prime Foods $2,500. No real system is touched.

State the accounting boundary precisely: the approved synthetic batch is interpreted as **Dr Accounts Payable—Fresh Farms $1,500 + Dr Accounts Payable—Prime Foods $2,500 / Cr Cash $4,000**, once. This is not a real general-ledger posting. The later receipt links proof to the Fresh Farms component and closes its lifecycle status; it is not another payment or journal entry.

### Optional — run the whole governed week

The `/eval` lane now runs all seven days, one explicit operator decision each.
Three beats worth recording, all verified:

- **REJECT day 0.** The badge asserts `ProcureGym.step was never called`, the day
  stays at 0 and the state version is unchanged. Rejecting costs nothing.
- **MODIFY PackRight `DEFER` to `PAY` on day 1.** Re-verification returns
  `BLOCKED · OVER_BUDGET` — $1,500 against $1,000 of cash — and APPROVE stays
  disabled. Modifying CleanPro to `PAY` returns
  `BLOCKED · UNRESOLVED_BUSINESS_CONTEXT` instead.
- **Switch the scenario picker to `restaurant_cashflow_v1`.** Identical to the
  locked fixture except for $250/day of simulated revenue. PackRight is deferred
  on days 0 and 1 because it is unaffordable, then paid on **day 2**, the day it
  fits. That is the agent choosing *when*, and the bounded oracle independently
  agrees on day 2.

On the locked fixture, days 1–6 are deliberately six identical no-op batches:
cash is stuck at $1,000, the oracle says PackRight should be paid `never`, and
regret is `0.000`. Say that plainly rather than clicking through it in silence —
it is why the cash-flow scenario exists.

Show the controlled comparison: Criticality-Aware Greedy matches the bounded 512-schedule oracle with `0.000` regret and zero high-criticality stockout days; Earliest Due First has `56.400` regret and reaches two high-criticality stockout days. Reward remains visible beside raw outcomes.

### Step 4 · 1:45–2:15 — Verify receipt proof and close status

Select the bundled payment receipt at `data/procureagent/assets/fresh_farms_payment_receipt.png` or upload PNG/JPEG, then run receipt proof. Its bundled-source metadata is `data/procureagent/assets/receipt_provenance.json`.

Say explicitly:

- Every receipt OCR token remains inspectable with its box, confidence, field label, and deterministic provenance.
- Receipt supplier, invoice number, amount, currency, paid date, and receipt ID came from Tesseract plus the deterministic receipt parser—not LayoutLMv3.
- Supplier, invoice, full amount, and currency must match exactly; the receipt ID must be unused, and the paid date must be present, valid, and grounded in OCR.
- The proof is synthetic and no money moved.

When every check passes, show Fresh Farms move from `SIMULATED_PAYMENT_APPROVED` to `PAID_CONFIRMED` in the demo ledger. Cash remains $1,000: proof closes the status and consumes the receipt ID without a second deduction. The exact receipt-routing reward is `+10`; call it an RL-ready evaluation signal, not a trained model or policy.

### 2:15–2:35 — View the AP history result

Select **Done — view AP history** and verify the three category labels and all four records:

| Category | Expected result |
|---|---|
| **Open invoices (2)** | PackRight `PR-15007` · $1,500 · DEFER; CleanPro `CP-70019` · $700 · VERIFY |
| **Paid · awaiting proof (1)** | Prime Foods `PF-25031` · $2,500 · `SIMULATED_PAYMENT_APPROVED` |
| **Completed (1)** | Fresh Farms `FF-10482` · $1,500 · `PAID_CONFIRMED` · receipt `RCPT-FF-10482` |

Point to the **$0 second cash impact**. Prime Foods is not mislabeled complete: it was paid in the simulation but still needs receipt proof. PackRight and CleanPro remain open for different reasons.

Explain the finance boundary: this dashboard supports working-capital discipline by organizing cash, supplier obligations, timing, and proof. It is **not** a full net-working-capital calculation because Accounts Receivable, balance-sheet inventory valuation, and other current assets/liabilities are outside the demo. Inventory days is an operational runway signal, not a balance-sheet valuation.

### 2:35–2:50 — Close

“The small model reads one narrow field. The live P0 evidence gate decides whether a person must review it; the deterministic policy ranks legal actions. C6 separately demonstrates a development-only identity router. Deterministic rules and a person govern the financial boundary. ProcureGym lets us measure consequences, and AP history keeps each obligation and its proof organized before any real integration.”

If showing the Router Lab inside **Evidence & methods**, state the exact scope: seven synthetic development rows; all seven context bins were also seen in training; no frozen test was evaluated. The learned router's `4 correct / 0 wrong / 3 review` result versus the fixed gate's `3 / 0 / 4` is a within-bin development result, not a live-invoice or generalization claim. C6 does not route the uploaded invoice, rank suppliers, or make payment actions in this recording flow.

## 4. Permanent URL

The preferred non-Streamlit-hosting path is the committed Docker container on Google Cloud Run. It supports the Python, Tesseract, Torch, and WebSocket runtime; use a dedicated approved Google Cloud/Firebase project and follow [`docs/CLOUD_RUN_DEPLOY.md`](CLOUD_RUN_DEPLOY.md). The owner must approve the exact billing account before project creation or deployment.

Streamlit Community Cloud remains an optional fastest path if the team changes its preference; Sasa's repository-admin checklist remains in [`docs/SASA_STREAMLIT_DEPLOY.md`](SASA_STREAMLIT_DEPLOY.md). Treat the Quick Tunnel only as a temporary fallback: it is public and unauthenticated, its URL changes when restarted, the host Mac must stay awake, and it has no uptime guarantee.

## 5. Fal receipt status

Fal generation was attempted through `fal-ai/flux-2`, but the configured account returned `403` because its balance is exhausted. No key was exposed. The committed deterministic receipt is the active OCR-safe fixture and its provenance says so.

After the balance is replenished:

```bash
.venv-model/bin/pip install -e ".[assets]"
.venv-model/bin/python scripts/generate_fal_receipt.py
```

The generator refuses to promote a Fal image unless Tesseract recovers every required proof field. Never expose `FAL_KEY` in Streamlit, browser JavaScript, logs, or Git.

## 6. Truth-language checklist

Say:

- “One live model inference” or “captured replay,” whichever the screen reports.
- “Model score,” not “probability the answer is correct.”
- “Synthetic looked-up amount and restaurant state.”
- “Deterministic P0 policy; contextual bandit is a separately labeled lab.”
- “Recommend PAY,” “simulated paid,” and “receipt proof confirmed.”
- “The $4,000 simulated batch is interpreted as Dr Accounts Payable for Fresh Farms and Prime Foods / Cr Cash once; receipt proof closes Fresh Farms status without deducting cash again.”
- “AP history ends at 2 open, 1 paid awaiting proof, and 1 completed.”
- “This supports working-capital discipline; it is not a full net-working-capital calculation.”
- “No real money moved.”

Never say:

- LayoutLMv3 performed OCR, read the receipt, extracted every field, or chose the supplier payment.
- LayoutLMv3 extracted the invoice amount; that displayed token is OCR plus a deterministic amount rule.
- A replay was a live run.
- A learned policy beat a baseline unless a frozen held-out artifact proves it.
- `PAID_CONFIRMED` means a bank payment occurred.
- Receipt verification paid the invoice or reduced cash a second time.
- The AP-history screen calculates net working capital or values inventory.

## 7. Release checklist

Latest evidence: **262 tests passed with two intentionally opt-in real-Tesseract smokes skipped** in the default run; the enabled focused Tesseract/UI/reward subset passes **56/56**. Offline acceptance is **9/9**, live acceptance is **10/10**, all **6/6** action/governance attacks are blocked, and all **8/8** receipt/proof attacks are blocked.

- [ ] `scripts/eval_procureagent.py` passes offline.
- [ ] `scripts/eval_procureagent.py --with-model` passes after warmup.
- [ ] Full pytest suite passes with only the declared opt-in skip.
- [ ] Invoice and receipt fixture images open and OCR correctly.
- [ ] The document gate visibly requests review on the current Fresh Farms run.
- [ ] No ProcureGym state changes before explicit operator approval.
- [ ] The approved simulation deducts cash exactly once and the receipt confirmation leaves cash unchanged.
- [ ] **Done — view AP history** shows exactly 2 open / 1 paid awaiting proof / 1 completed with the locked supplier mapping.
- [ ] The final screen shows $0 second cash impact and distinguishes working-capital support from complete NWC reporting.
- [ ] Wrong amount/supplier/invoice/currency and duplicate proof remain blocked.
- [ ] Clean browser reaches the app and completes the recording path.
- [ ] Public tunnel, Cloud Run URL, or optional Streamlit URL works from a second device.
- [ ] GitHub `main` contains the exact tested commit.
- [ ] Fal provenance is truthful and no secret is committed.
- [ ] Upload `docs/assets/sundai-thumbnail.png` to the Sundai card and add the public URL before the event cutoff.
- [ ] Ryan confirms adapter/base/dataset usage terms; until then say only that repository code is MIT.
