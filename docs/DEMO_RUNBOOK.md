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
- Captured model proof: `data/procureagent/eval/model_smoke_v1.json`
- Acceptance result: `data/procureagent/eval/acceptance_live_v1.json`
- Dashboard screenshot: `docs/assets/procureagent-dashboard.png`
- Sundai thumbnail: `docs/assets/sundai-thumbnail.png` (1200×630)

## 3. Two-to-three-minute talk track

### 0:00–0:20 — Set the problem

“This restaurant has $5,000, but four verified supplier bills total $6,200. Paying only the oldest invoice can leave the kitchen without produce or meat.”

Point out that this is Accounts Payable: money the restaurant owes suppliers.

### 0:20–0:55 — Run the small document specialist

Open **/eval lab**, select or upload the Fresh Farms invoice, enter expected invoice number `FF-10482`, and run the document analysis.

Call out each boundary:

- Tesseract produced the words and positions.
- Ryan's local LayoutLMv3 token classifier selected only `FF-10482`.
- Strict exact match on this document is yes.
- The model did not extract the amount, due date, inventory, or criticality.
- The score is below the auto-confirm threshold, so the gate asks for a person even though rule and model agree.

Click the explicit identity confirmation. Do not call this aggregate model accuracy; it is one live document result.

### 0:55–1:30 — Show the three RL/evaluation questions

Reveal the synthetic business lookup, then the policy result:

1. **Identity:** was `FF-10482` read correctly? Every supplier now has a bundled
   invoice image, so you can read all four documents rather than one. Each is
   verified to yield exactly one anchored candidate under real Tesseract; each
   still needs its own human CONFIRM, because the frozen 0.80 confidence
   threshold routes them to review.
2. **Priority:** first Fresh Farms, second Prime Foods, third PackRight, then CleanPro review. Show inventory runway and lead time beside the ranking.
3. **Action:** Fresh Farms $1,500 and Prime Foods $2,500 are exact verified full-payment actions. The agent cannot invent a supplier or amount.

Run the verifier and explicitly approve the simulated batch. ProcureGym should move cash from $5,000 to $1,000 and advance one day.

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

### 1:30–2:10 — Close the AP transaction with proof

Select or upload the payment receipt and run receipt proof.

Say explicitly:

- Receipt fields came from Tesseract plus deterministic anchored rules—not Ryan's model.
- Supplier, invoice, full amount, and currency must match exactly; the receipt ID must be unused, and the paid date must be present, valid, and grounded in OCR.
- The proof is synthetic and no money moved.

When every check passes, show Fresh Farms move from `SIMULATED_PAYMENT_APPROVED` to `PAID_CONFIRMED` in the demo ledger.

### 2:10–2:30 — Close

“The small model reads one narrow field. The live P0 evidence gate decides whether a person must review it; the deterministic policy ranks legal actions. C6 separately demonstrates a development-only identity router. Deterministic rules and a person govern the financial boundary. ProcureGym lets us measure the consequences before any real integration.”

If showing the Router Lab tab, state the exact scope: seven synthetic development rows; all seven context bins were also seen in training; no frozen test was evaluated. The learned router's `4 correct / 0 wrong / 3 review` result versus the fixed gate's `3 / 0 / 4` is a within-bin development result, not a live-invoice or generalization claim. C6 does not route the uploaded invoice, rank suppliers, or make payment actions in this recording flow.

## 4. Permanent Streamlit URL

The GitHub source is prepared with `requirements.txt`, `packages.txt`, Python 3.12 configuration, and `procure_app.py`. Streamlit Community Cloud requires repository **admin** permission for the initial deployment, so Sasa must perform this one-time action:

1. Sign in at <https://share.streamlit.io> with GitHub.
2. Choose **Create app**.
3. Repository: `sasap91/invoiceagent`.
4. Branch: `main`.
5. Entrypoint: `procure_app.py`.
6. Advanced settings: Python `3.12`.
7. Deploy, then send the resulting `streamlit.app` URL to the team.

Subsequent pushes to `main` trigger redeployment. If Community Cloud cannot fit the model runtime, keep it as the fixture/replay presentation path and use the local Cloudflare tunnel for the live checkpoint path.

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
- “No real money moved.”

Never say:

- Ryan's model performed OCR, read the receipt, extracted every field, or chose the supplier payment.
- A replay was a live run.
- A learned policy beat a baseline unless a frozen held-out artifact proves it.
- `PAID_CONFIRMED` means a bank payment occurred.

## 7. Release checklist

Latest frozen evidence: **192 tests passed with one intentionally opt-in Tesseract smoke skipped**, offline acceptance **9/9**, live acceptance **10/10**, all **6/6** action/governance attacks blocked, and all **8/8** receipt/proof attacks blocked.

- [ ] `scripts/eval_procureagent.py` passes offline.
- [ ] `scripts/eval_procureagent.py --with-model` passes after warmup.
- [ ] Full pytest suite passes with only the declared opt-in skip.
- [ ] Invoice and receipt fixture images open and OCR correctly.
- [ ] The document gate visibly requests review on the current Fresh Farms run.
- [ ] No ProcureGym state changes before explicit operator approval.
- [ ] Wrong amount/supplier/invoice/currency and duplicate proof remain blocked.
- [ ] Clean browser reaches the app and completes the recording path.
- [ ] Public tunnel or Streamlit URL works from a second device.
- [ ] GitHub `main` contains the exact tested commit.
- [ ] Fal provenance is truthful and no secret is committed.
- [ ] Upload `docs/assets/sundai-thumbnail.png` to the Sundai card and add the public URL before the event cutoff.
- [ ] Ryan confirms adapter/base/dataset usage terms; until then say only that repository code is MIT.
