# ProcureAgent demo and recording runbook

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
- Dashboard screenshot: `docs/assets/procureagent-dashboard.png`
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
- Ryan's local LayoutLMv3 token classifier selected only the invoice-number token `FF-10482`.
- The displayed invoice amount is grounded by OCR plus a separate deterministic total-anchor rule. Ryan's model did not extract it.
- Strict exact match on this document is yes.
- Canonical AP amount, due date, inventory, and criticality come from the synthetic lookup after identity confirmation, not from the model.
- The score is below the auto-confirm threshold, so the gate asks for a person even though rule and model agree.

Use the **Technical evidence** expander only when the audience asks for depth. It exposes raw OCR, all token boxes/confidences/sources, document hash, anchored-rule evidence, LayoutLMv3 version/latency/scores, gate reasons, verifier details, and artifact provenance without crowding the guided story.

### Step 2 · 0:55–1:20 — Confirm identity and review the plan

Click the explicit identity **CONFIRM** action. **CORRECT** and **REJECT** are the fail-closed alternatives. Do not call the one-document exact result aggregate model accuracy.

Reveal the synthetic business lookup, then the policy result:

1. **Identity:** was `FF-10482` read correctly?
2. **Priority:** first Fresh Farms, second Prime Foods, third PackRight, then CleanPro review. Show inventory runway and lead time beside the ranking.
3. **Action:** Fresh Farms $1,500 and Prime Foods $2,500 are exact verified full-payment actions. The agent cannot invent a supplier or amount.

Explain the plan before approving it: the deterministic policy proposes actions; the verifier checks identity, exact canonical amount, currency, cash, duplicates, and state version. Nothing has changed yet.

### Step 3 · 1:20–1:45 — Approve the simulated payment

Explicitly approve the reverified batch. ProcureGym should move cash from $5,000 to $1,000 and advance one day. No real system is touched.

State the accounting boundary precisely: at this simulated payment step, the demo economically records **Dr Accounts Payable / Cr Cash** for the approved full payments. Cash is deducted here, once. The later receipt proves and closes lifecycle status; it is not another payment or journal entry.

Show the controlled comparison: Criticality-Aware Greedy matches the bounded 512-schedule oracle with `0.000` regret and zero high-criticality stockout days; Earliest Due First has `56.400` regret and reaches two high-criticality stockout days. Reward remains visible beside raw outcomes.

### Step 4 · 1:45–2:15 — Verify receipt proof and close status

Select the bundled payment receipt at `data/procureagent/assets/fresh_farms_payment_receipt.png` or upload PNG/JPEG, then run receipt proof. Its bundled-source metadata is `data/procureagent/assets/receipt_provenance.json`.

Say explicitly:

- Every receipt OCR token remains inspectable with its box, confidence, field label, and deterministic provenance.
- Receipt supplier, invoice number, amount, currency, paid date, and receipt ID came from Tesseract plus the deterministic receipt parser—not Ryan's model.
- Supplier, invoice, full amount, and currency must match exactly; the receipt ID must be unused, and the paid date must be present, valid, and grounded in OCR.
- The proof is synthetic and no money moved.

When every check passes, show Fresh Farms move from `SIMULATED_PAYMENT_APPROVED` to `PAID_CONFIRMED` in the demo ledger. Cash remains $1,000: proof closes the status and consumes the receipt ID without a second deduction.

### 2:15–2:30 — Close

“The small model reads one narrow field. The live P0 evidence gate decides whether a person must review it; the deterministic policy ranks legal actions. C6 separately demonstrates a development-only identity router. Deterministic rules and a person govern the financial boundary. ProcureGym lets us measure the consequences before any real integration.”

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
- “The simulated payment records Dr Accounts Payable / Cr Cash once; receipt proof closes status without deducting cash again.”
- “No real money moved.”

Never say:

- Ryan's model performed OCR, read the receipt, extracted every field, or chose the supplier payment.
- Ryan's model extracted the invoice amount; that displayed token is OCR plus a deterministic amount rule.
- A replay was a live run.
- A learned policy beat a baseline unless a frozen held-out artifact proves it.
- `PAID_CONFIRMED` means a bank payment occurred.
- Receipt verification paid the invoice or reduced cash a second time.

## 7. Release checklist

Latest evidence: **208 tests passed with two intentionally opt-in real-Tesseract smokes skipped** in the default run; the enabled focused smoke subset passes **32/32**. Offline acceptance is **9/9**, live acceptance is **10/10**, all **6/6** action/governance attacks are blocked, and all **8/8** receipt/proof attacks are blocked.

- [ ] `scripts/eval_procureagent.py` passes offline.
- [ ] `scripts/eval_procureagent.py --with-model` passes after warmup.
- [ ] Full pytest suite passes with only the declared opt-in skip.
- [ ] Invoice and receipt fixture images open and OCR correctly.
- [ ] The document gate visibly requests review on the current Fresh Farms run.
- [ ] No ProcureGym state changes before explicit operator approval.
- [ ] The approved simulation deducts cash exactly once and the receipt confirmation leaves cash unchanged.
- [ ] Wrong amount/supplier/invoice/currency and duplicate proof remain blocked.
- [ ] Clean browser reaches the app and completes the recording path.
- [ ] Public tunnel, Cloud Run URL, or optional Streamlit URL works from a second device.
- [ ] GitHub `main` contains the exact tested commit.
- [ ] Fal provenance is truthful and no secret is committed.
- [ ] Upload `docs/assets/sundai-thumbnail.png` to the Sundai card and add the public URL before the event cutoff.
- [ ] Ryan confirms adapter/base/dataset usage terms; until then say only that repository code is MIT.
