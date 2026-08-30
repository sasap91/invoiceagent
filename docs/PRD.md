# InvoiceAgent — Hackathon MVP Product Requirements Document

**One-line promise:** Help a small restaurant turn a paper supplier invoice and its payment receipt into a clear, verified Accounts Payable record.

**Status:** Integrated reference path implemented and tested; named-owner sign-off and permanent deployment remain open

**Scope decision:** Sasa's restaurant procurement use case is the source of truth; the public product name is **InvoiceAgent**. Internal `procureagent` package and data paths remain stable implementation details.

**Team:** Sasa, Ryan, David, Wilson, and Dillon

**Environment:** ProcureGym
**Primary workflow:** 1 · Read invoice → 2 · Confirm identity and plan → 3 · Approve simulated payment → 4 · Verify receipt proof and close AP status → **Done — view AP history**

> **Current checkpoint:** The integrated InvoiceAgent reference path is implemented. Remaining release work is named-owner sign-off, permanent deployment, and presentation freeze.

> **Guided-demo contract:** The primary screen tells this workflow in four progressive steps and then opens one AP-history completion dashboard. Compact business outcomes stay visible by default; every OCR token, evidence source, model/rule detail, gate reason, hash, and deployment provenance remains accessible under technical evidence.

---

## 1. The story in 30 seconds — STAR

### Situation

Small restaurants still receive paper and image-based supplier invoices. Manually finding each invoice number, recording the amount, and later proving that the bill was paid is slow and error-prone.

### Task & action

InvoiceAgent turns one supplier invoice into a guarded Accounts Payable workflow:

1. Tesseract OCR reads every visible token and bounding box locally.
2. The supervised LayoutLMv3 specialist adapted by Ryan proposes the invoice-number token only.
3. A deterministic total rule highlights the apparent invoice amount; the verified canonical amount comes from the synthetic AP record.
4. A person confirms the document identity before the payable becomes active.
5. The restaurant reviews the recommendation and approves a simulated payment.
6. Receipt OCR and deterministic rules read the payment receipt, then an exact gate compares supplier, invoice number, full amount, and currency.
7. The match produces an auditable, **RL-ready reward signal**: `+10` for a verified full match, `-1` for safe review, and `-25` for an unsafe automatic acceptance. This MVP does **not** claim that LayoutLMv3 or a routing policy was trained with that reward.

### Result

A clearly labeled synthetic Sugar & Spice Thai Restaurant walkthrough records the Fresh Farms supplier invoice and its matching payment receipt, closes that simulated Accounts Payable obligation as `PAID_CONFIRMED`, and leaves a human-readable evidence trail. The final dashboard resolves the four synthetic invoices into **2 open / 1 paid awaiting proof / 1 completed**. The fixture is a no-affiliation demo and does not represent a real Sugar & Spice transaction.

### One-screen product graph

~~~mermaid
flowchart LR
    A[Paper supplier invoice] --> B[Tesseract OCR<br/>every word + box]
    B --> C[LayoutLMv3<br/>adapted by Ryan · invoice number only]
    B --> D[Deterministic rule<br/>displayed total]
    C --> E{Human confirms<br/>document identity}
    D --> E
    E --> F[Synthetic AP record<br/>supplier + amount + due date]
    F --> G[Recommendation +<br/>safety verifier]
    G --> H{Operator approves<br/>simulation}
    H --> I[Dr Accounts Payable<br/>Cr Cash — once]
    I --> J[Payment receipt upload]
    J --> K[Tesseract + exact rules]
    K --> L{Supplier + invoice +<br/>amount + currency match?}
    L -->|yes| M[PAID_CONFIRMED<br/>RL-ready reward +10]
    L -->|uncertain| N[Human review<br/>reward -1]
    L -->|unsafe auto-accept| O[Blocked<br/>reward -25]
    M --> P[AP history dashboard<br/>Open 2 · Awaiting proof 1 · Completed 1]
~~~

The approved $4,000 synthetic batch is interpreted as **Dr Accounts Payable—Fresh Farms $1,500 + Dr Accounts Payable—Prime Foods $2,500 / Cr Cash $4,000** once. No real general-ledger entry is posted. Receipt `RCPT-FF-10482` links proof to the Fresh Farms component and changes its lifecycle status only; its second cash impact is exactly **$0**. Prime Foods remains paid in the simulation but awaiting proof.

The product supports **working-capital discipline** by putting cash, supplier obligations, timing, and proof status in one operating view. It does not calculate complete net working capital. Accounts Receivable, balance-sheet inventory valuation, and other current assets and liabilities are outside P0; the inventory-days field is an operational runway input, not an accounting valuation.

The restaurant narrative continues into Sasa's cash-prioritization use case. A small restaurant receives bills from several suppliers and may not have enough cash to pay every bill immediately. Paying only the oldest bill can look sensible while accidentally allowing essential produce or meat to run out.

InvoiceAgent helps the owner reason through that choice:

1. The owner uploads a supplier invoice.
2. OCR supplies the words and locations on the page.
3. Ryan's local, approximately 133M-parameter LayoutLMv3 specialist proposes the invoice number.
4. Evidence checks or a person verify that identity.
5. The verified number retrieves a clearly labeled **synthetic** supplier record containing amount, due date, inventory coverage, and supplier criticality.
6. A procurement policy recommends **PAY**, **DEFER**, or **VERIFY** and explains why.
7. Deterministic rules check that the recommendation is safe and valid.
8. A human operator approves, modifies, or rejects it.
9. ProcureGym shows what the approved choice would do to simulated cash, inventory, fees, and supplier status.
10. For a simulated full payment, the operator uploads a payment receipt; deterministic receipt extraction and an exact proof gate match supplier, invoice, amount, and currency before the demo marks the Accounts Payable transaction complete.

The model does not move money and does not run procurement by itself.

The UI groups those mechanics into four audience-facing steps:

1. **Read invoice:** Tesseract exposes every token and box; Ryan's LayoutLMv3 adapter proposes the invoice-number token only; a separate deterministic OCR rule identifies the displayed invoice total.
2. **Confirm and plan:** explicit human document review unlocks the synthetic AP record and an explained daily plan.
3. **Approve simulated payment:** the verifier and operator gate precede one ProcureGym mutation. Economically, the $4,000 synthetic batch is interpreted as the two supplier AP debits and one Cash credit described above, once; no real ledger is posted.
4. **Verify receipt proof:** receipt OCR plus the deterministic parser grounds all required receipt fields. Exact proof closes lifecycle status without a second cash deduction.

> **Core claim:** A narrow local model can serve as a low-cost perception component inside a larger decision system, while deterministic checks, human control, and simulation govern consequential actions.

### ELI5 example

Imagine a restaurant has **$5,000** but owes suppliers **$6,200**:

- Produce is due tomorrow and only two days remain in stock.
- Meat is due in three days and only three days remain.
- Packaging is already late, but the restaurant has twenty days of packaging left.
- A cleaning invoice is difficult to identify confidently.

InvoiceAgent may recommend paying produce and meat, deferring packaging, and verifying the cleaning invoice. The owner still decides. The app then advances a pretend restaurant by one day so everyone can see the tradeoff.

---

## 2. User and problem

### Primary user

An independent restaurant owner or manager working with approximately five to ten recurring suppliers and a limited daily cash balance.

### Job to be done

> Given the cash available today, which supplier invoices should I pay now, which can wait, and which require review?

### Why existing approaches fall short

- **Earliest Due First** ignores whether a delayed supplier will stop restaurant operations.
- **Largest Bill First** can consume cash needed for several critical smaller suppliers.
- **A language model alone** can recommend an unsafe or impossible action.
- **Document extraction alone** identifies the bill but does not reason about restaurant consequences.
- **Fully automated payment** is inappropriate for a hackathon prototype involving financial decisions.

---

## 3. Product goals

### P0 goals

1. Use the existing local invoice-number model only for the task it currently supports.
2. Make OCR, model evidence, synthetic lookup, policy output, rule checks, and human actions visibly distinct.
3. Connect a verified supplier invoice to restaurant cash, inventory, due-date, and criticality context.
4. Recommend **PAY**, **DEFER**, or **VERIFY** with structured reasons.
5. Require an explicit operator decision before any simulated financial state changes.
6. Implement a deterministic, seeded ProcureGym environment.
7. Compare the same scenario against **Earliest Due First**.
8. Fail closed when document identity, supplier context, or financial constraints are uncertain.
9. Demonstrate the full Accounts Payable lifecycle with a full-payment receipt whose supplier, invoice, amount, and currency match exactly before the transaction is marked complete.
10. Instrument the three learning targets separately: invoice-number correctness, multi-day supplier-ranking quality, and exact payment-action correctness.
11. End the guided flow with an AP-history dashboard that reconciles every synthetic obligation into mutually exclusive open, paid-awaiting-proof, or completed categories.

### P1 stretch goals

1. Add a named learned ProcureGym procurement policy with frozen training configuration, masked legal actions, and structured output.
2. Generate an offline action-outcome matrix for extraction routes.
3. Train and evaluate a constrained contextual-bandit router in a sandbox.
4. Add an experimental non-English fixture that routes to review unless separately validated.
5. Run several seeded restaurant scenarios rather than one.
6. Train a procurement policy in ProcureGym over the legal masked action space and compare it with deterministic baselines plus a bounded oracle where feasible.

### Demo success statement

If at least one actual model run is captured with artifact and runtime metadata, a judge should be able to say:

> I saw a small local model propose an invoice number, saw the evidence and human safeguard, watched a policy prioritize limited restaurant cash, and saw an exact receipt proof close the simulated Accounts Payable transaction.

If only a replay is available, the presentation must instead say that it showed a recorded proposal and that no model ran during that interaction. A replay alone cannot support a live-model claim.

---

## 4. Scope

### In scope for P0

- One synthetic restaurant
- Approximately six synthetic suppliers
- Four locked primary invoices, expandable to ten to fifteen fixtures
- Supplier invoices and Accounts Payable only
- One full-payment receipt-proof path for an approved simulated payment
- One inspected synthetic receipt fixture plus user-uploaded receipt images, with generation provenance visible; use Fal when the account is available and a deterministic OCR-safe fallback otherwise
- PNG/JPEG input plus live or visibly labeled precomputed OCR
- Existing LayoutLMv3 invoice-number specialist
- Composite supplier-and-invoice lookup
- Synthetic amount, due date, inventory, lead-time, and criticality records
- Recommendations: **PAY**, **DEFER**, **VERIFY**
- Operator decisions: **APPROVE**, **MODIFY**, **REJECT**
- Deterministic risk verifier
- Seven-day, seeded ProcureGym simulation
- Earliest Due First baseline
- Exact integer-minor-unit or Decimal arithmetic
- Audit trail from document proposal through simulated transition
- A final AP-history view covering all four synthetic obligations

### Explicitly out of scope for P0

- Accounts Receivable and customer invoices
- Full net-working-capital or balance-sheet reporting, including inventory valuation and other current assets or liabilities
- Partial payments
- General receipt-number extraction, arbitrary receipt reconciliation, or receipt learning
- Real banking, payment, POS, ERP, or supplier integrations
- Automatic financial authorization
- Production authentication or compliance
- Universal invoice or line-item extraction
- A claim that amount, due date, inventory, or supplier criticality came from LayoutLMv3
- A claim that the current specialist is multilingual
- PPO, DQN, GRPO, or end-to-end generative-model RL
- Online exploration on real invoices
- Training from implicit approval alone

P0 receipt proof is deliberately narrow: it closes one simulated **Accounts Payable** obligation only after an exact full-payment match. It does not revive the earlier Accounts Receivable product, infer partial allocations, or claim that the LayoutLMv3 invoice-number specialist reads receipts.

---

## 5. Terms that must remain distinct

| Term | Exact meaning |
|---|---|
| **Document review** | Confirm whether OCR/model proposed the correct supplier and invoice number |
| **Procurement VERIFY** | Business context or the proposed decision requires review; it is not permission to use an unknown document |
| **PAY** | Recommend full simulated payment of one outstanding invoice in today's batch |
| **DEFER** | Recommend no payment for that invoice in today's batch |
| **VERIFY** | Queue business-context review and make no direct payment; only an explicitly committed daily batch advances time |
| **APPROVE** | Operator confirms the proposed action for the simulator |
| **MODIFY** | Operator changes PAY, DEFER, or VERIFY; the entire batch must then pass the verifier again |
| **REJECT** | Operator rejects the proposal; no state changes |
| **Looked up** | Loaded from immutable synthetic fixture data after document identity was confirmed |
| **Extracted** | Produced from OCR, a deterministic parser, or a named model with evidence |
| **Simulated accounting entry** | Economic interpretation of the approved ProcureGym PAY transition: debit Accounts Payable and credit Cash exactly once; no external ledger is posted |
| **Payment proof** | Receipt fields extracted by OCR plus deterministic rules and matched to one simulated AP obligation by supplier ID, invoice number, full amount, and currency |
| **PAID_CONFIRMED** | Demo lifecycle status reached only after an approved simulated PAY and a verified full-payment proof; confirmation consumes proof and changes status without deducting cash again, and it does not mean a bank moved money |

The UI must say **Simulated paid** or **Receipt proof confirmed**, never imply that a bank payment occurred.

---

## 6. End-to-end architecture

The system separates perception, business context, recommendation, governance, and consequence.

~~~mermaid
flowchart TD
    A[Supplier invoice] --> B[OCR words and boxes]
    B --> C[Anchored invoice-number rule]
    B --> D[Local LayoutLMv3 specialist]
    B --> U[Deterministic invoice-total rule for displayed evidence]
    C --> E{Document evidence gate}
    D --> E
    E -->|Uncertain| F[Document review]
    F -->|Confirmed identity| G[Composite synthetic lookup]
    E -->|Verified identity| G
    G --> H[Restaurant state]
    H --> I[Procurement policy]
    I --> J[Daily action batch]
    J --> K{Deterministic batch verifier}
    K --> L[Operator review]
    L -->|Modify| K
    L -->|Approve reverified batch| M[ProcureGym step: Dr AP / Cr Cash once]
    L -->|Reject or do not commit| N[No state change]
    M --> O[Next synthetic state and metrics]
    M --> P[Upload full-payment receipt]
    P --> Q[Receipt OCR and deterministic parser]
    Q --> R{Exact payment-proof gate}
    R -->|Supplier + invoice + amount + currency match| S[PAID_CONFIRMED status; cash unchanged]
    R -->|Missing, ambiguous, or mismatched| T[Receipt review; payment proof remains pending]
~~~

### Non-negotiable boundaries

- An unverified invoice identity cannot activate a payable.
- LayoutLMv3 proposes invoice-number tokens; it does not choose PAY or DEFER.
- Invoice-total highlighting is an OCR-plus-deterministic-rule result, never LayoutLMv3 output. The canonical payable amount still comes from the confirmed synthetic AP lookup.
- Synthetic lookup fields must be labeled **Looked up**, not **Extracted**.
- The recommendation is a proposal, not authorization.
- The verifier can block an action regardless of model output or reward.
- Only an explicit operator-approved and reverified daily batch can enter ProcureGym.
- LayoutLMv3 is used on the invoice-number path only. Receipt fields come from OCR plus a deterministic parser and must be labeled that way.
- A receipt cannot close an AP obligation unless it matches the approved simulated full payment on the exact composite identity, amount, and currency.
- Approved simulated PAY performs the single cash deduction and is presented as **Dr Accounts Payable / Cr Cash**. Receipt confirmation consumes proof and closes status only; it cannot reduce cash a second time.
- ProcureGym never changes a real bank or accounting system.

---

## 7. Where learning belongs

InvoiceAgent contains different learning and policy components. They must not be blended into one vague “AI” claim.

### 7.0 The three RL/evaluation questions

The team agreed that learning and reward design must answer three separate questions:

1. **Did we identify the invoice number correctly?** Compare the proposed invoice number with declared ground truth using strict exact match, and heavily penalize any incorrect identity that is automatically accepted. Ryan's token reader remains supervised; the RL component here is the extraction router deciding whether to trust cheap rules, invoke the local specialist, or request review.
2. **Did we prioritize suppliers in the right order?** At each simulated day, rank whom to pay first, second, third, and so on using the restaurant's operational runway: inventory days remaining relative to delivery lead time, due days, supplier criticality, payment-unlocked delivery, and available cash. Record **who to pay, what invoice and amount to pay, and when**, then score whether that ordered plan protects the restaurant's remaining operating runway through its downstream ProcureGym outcomes—not only by agreement with a hand-written ranking.
3. **Did we take the right payment action?** Check that the chosen action targets the right verified supplier and invoice, at the right simulated time, for the exact full outstanding amount. The learned policy may select among legal actions; it may not invent a supplier, invoice, currency, or amount.

These are reported as three reward/metric components rather than one opaque score:

| Component | Positive evidence | Failure evidence | Hard safety boundary |
|---|---|---|---|
| Identity correctness | Strict exact invoice-number match or appropriate review | Wrong automatic acceptance, missing identity, unnecessary model/review cost | Unknown, ambiguous, or ungrounded identity cannot activate a payable |
| Priority quality | Critical restaurant runway protected, fewer stockout/disruption days, sensible ordered schedule | Critical stockout, avoidable supplier disruption, excessive late fees, poor schedule regret | Ranking cannot bypass document verification or current state |
| Payment action correctness | Right composite identity, exact full amount, legal timing, nonnegative cash | Wrong supplier/invoice, wrong amount/currency, duplicate, stale, or over-budget action | Invalid actions are masked during learning and independently blocked by the verifier |

For the locked four-invoice scenario, C5 compares the policy with Earliest Due First and the implemented bounded exhaustive legal-schedule oracle. The oracle is an evaluation reference, not a production policy. Any reported aggregate reward must be shown beside all three component scores and raw outcomes.

### 7.1 Supervised document reader

Ryan's existing LayoutLMv3 LoRA adapter is a supervised token classifier. It consumes an image plus externally generated OCR words and normalized boxes. It may propose zero, one, or multiple invoice-number candidate spans; the document gate decides whether one can be confirmed or review is required.

For the guided evidence display, an independent deterministic total-anchor rule may label one invoice-amount span from Tesseract OCR. That amount label is not a LayoutLMv3 output and does not replace the canonical full payable amount returned by the post-confirmation synthetic lookup.

It currently does **not**:

- Perform OCR
- Extract total, due date, inventory, or supplier criticality
- Make procurement recommendations
- Learn from ProcureGym reward

Human-corrected invoice numbers are valuable supervised labels for a future training batch.

### 7.1.1 Receipt-match reward — implemented RL-ready signal

The exact receipt proof gate now emits a small, inspectable reward through `src/procureagent/receipt_reward.py`. It scores the **routing decision after deterministic proof checks**; it does not update LayoutLMv3 weights and it is not evidence that a policy was trained.

| Decision and proof outcome | Reward | Why |
|---|---:|---|
| Accept a receipt only after exact supplier + invoice + full amount + currency proof | `+10` | A grounded full match safely closes the simulated obligation |
| Request human review | `-1` | Review has a small cost but is safer than guessing |
| Automatically accept a receipt whose exact proof did not pass | `-25` | A false financial match must be much worse than review |

The UI must display the raw proof checks beside the reward and say **RL-ready evaluation signal — no policy/model was trained**. A future contextual bandit may learn when to accept or request review from offline, labeled episodes; until that training and held-out evaluation exist, the current decision remains deterministic.

### 7.2 Extraction router — constrained contextual bandit, P1

The current safe baseline is a fixed evidence gate. The first learned-RL experiment should be a small contextual bandit that chooses how much extraction effort to spend.

| RL concept | InvoiceAgent mapping |
|---|---|
| Context | OCR availability/quality, anchored-rule result, candidate count, known supplier, layout novelty |
| Actions | Rules only, rules plus local document specialist, or human review |
| Reward | Correct verified identity minus compute, latency, and review cost |
| Severe penalty | Incorrect invoice identity automatically accepted |
| Ground truth | Locked dataset label or explicit human correction |

Illustrative reward configuration:

- **+10:** correct automatic identity
- **-50:** incorrect identity automatically accepted
- **-0.2:** local model invocation
- **-2:** human review
- Small normalized latency penalty

Reward is subordinate to hard constraints. The router may never override document grounding, candidate ambiguity, evidence, or auto-accept safety. The downstream procurement verifier independently enforces supplier, duplicate, amount, cash, and state-version rules.

Fitting and threshold selection use only declared training/development data or synthetic training fixtures. The locked test split remains untouched until the policy is frozen and evaluated once. Until a trained policy beats calibrated fixed thresholds on held-out data, the product must say **bandit-ready** rather than claim an RL improvement.

**30 August 2026 Router Lab checkpoint:** the implemented tabular contextual bandit was evaluated on seven held-out synthetic development rows. It produced four correct automatic identities, zero wrong automatic accepts, three reviews, and average reward `4.6621`; the fixed evidence gate produced three correct automatic identities, zero wrong automatic accepts, four reviews, and average reward `2.9029`; always-review produced seven reviews and average reward `-2.3179`. All seven development context bins also occur in training, and no frozen test split has been evaluated. This supports only a within-bin development result—not generalization, live-invoice accuracy, or any claim about supplier ranking or payment actions.

### 7.3 Procurement policy and ProcureGym

ProcureGym is a sequential decision environment:

| RL element | MVP definition |
|---|---|
| State | Cash, outstanding invoices, inventory days, delivery lead days, due days, supplier criticality/status, payment-unlocked delivery, and document verification status |
| Action | An ordered daily batch containing PAY, DEFER, or VERIFY for every active invoice; PAY is masked to one verified composite identity and its exact full outstanding amount |
| Transition | One reverified, operator-approved batch commits atomically; then time changes inventory, invoice age, fees, deliveries, and supplier status |
| Reward | Reports identity correctness, ordered-priority quality, and payment-action correctness separately, then summarizes continuity and safe allocation while penalizing stockouts, fees, disruption, negative cash, and unsafe proposals |
| Horizon | Seven simulated restaurant days |
| Terminal conditions | Seven days reached; or any high-criticality inventory remains at zero for two consecutive simulated days |

P0 implements and evaluates the environment; it does not claim that a sophisticated procurement RL policy was trained. Human APPROVE, MODIFY, and REJECT events are logged as feedback, but approval is not proof that a recommendation was objectively correct.

### P0 recommendation policy: Criticality-Aware Greedy v1

P0 uses a deterministic policy, not an unnamed AI model:

1. Return VERIFY when required business context is missing or contradictory.
2. Otherwise compute a frozen priority score:
   - `+100` when inventory days are at or below delivery lead days plus one
   - `+40` for a high-criticality supplier; `+20` for medium
   - `+30` when due in one day or less
   - `+20` when already overdue
   - `-40` when at least ten inventory days remain
3. Sort descending by score, then ascending due days, then supplier ID for a deterministic tie-break.
4. Recommend PAY in that order while the full amount fits the remaining daily cash budget.
5. Recommend DEFER for a valid invoice that does not fit the remaining budget.
6. Return one recommendation for every active invoice in one daily batch.

Weights freeze before the evaluation scenario runs. A malformed or unavailable future learned policy falls back to VERIFY. Any P1 policy must name its algorithm/model, runtime, training data/scenarios, reward version, random seeds, action mask, configuration, and structured output. A generative policy must additionally disclose its prompt and sampling settings. The UI must never imply that the 133M document extractor made the procurement decision.

> **Honest presentation line:** The supervised model reads. The router decides how much extraction effort to spend. A procurement policy ranks legal actions. ProcureGym scores identity, priority, and payment action separately; deterministic checks and a person remain in control.

---

## 8. Canonical data contracts

### 8.1 Document identity proposal

~~~json
{
  "document_id": "doc_fresh_farms_10482",
  "supplier_id": "fresh_farms",
  "supplier_source": "operator_selected",
  "supplier_confirmed": true,
  "candidate_spans": [{
    "invoice_number": "FF-10482",
    "entity_confidence": 0.96,
    "grounded_in_ocr": true,
    "evidence_tokens": ["FF-10482"],
    "evidence_boxes": [[640, 120, 820, 160]]
  }],
  "method": "layoutlmv3_local",
  "model_version": "layoutlmv3-invoice-number:<artifact-id>",
  "status": "PROPOSED"
}
~~~

Confidence is a model score, not a probability that the answer is correct.

In P0 the operator selects the supplier from the restaurant's known-supplier list before upload. The selection and invoice-number proposal are confirmed separately. A model hint never becomes a verified supplier ID.

Document statuses are **PROPOSED**, **REVIEW_REQUIRED**, **CONFIRMED**, **CORRECTED**, and **REJECTED**. Review decisions are **CONFIRM**, **CORRECT**, or **REJECT**. Only CONFIRMED or CORRECTED composite identity may query the lookup.

### 8.2 Canonical synthetic supplier invoice

Lookup uses the composite key **supplier ID plus invoice number**.

~~~json
{
  "record_source": "synthetic_fixture_lookup",
  "supplier_id": "fresh_farms",
  "invoice_number": "FF-10482",
  "category": "produce",
  "amount_minor": 150000,
  "currency": "USD",
  "due_in_days": 1,
  "inventory_days_remaining": 2,
  "delivery_lead_days": 1,
  "payment_unlocks_delivery": true,
  "supplier_criticality": "high",
  "supplier_status": "active",
  "payment_status": "unpaid",
  "state_version": 1
}
~~~

The demo fixture explicitly states when payment unlocks a pending replenishment. ProcureGym must not assume that paying an arbitrary historical invoice creates inventory.

### 8.3 Structured daily recommendation batch

~~~json
{
  "batch_id": "day-0-criticality-aware-v1",
  "state_version": 1,
  "policy_name": "criticality_aware_greedy",
  "policy_version": "v1",
  "policy_type": "deterministic_rules",
  "recommendations": [
    {
      "supplier_id": "fresh_farms",
      "invoice_number": "FF-10482",
      "action": "PAY",
      "amount_minor": 150000,
      "reason_codes": ["STOCKOUT_RISK", "CRITICAL_SUPPLIER", "DUE_SOON"]
    },
    {
      "supplier_id": "prime_foods",
      "invoice_number": "PF-25031",
      "action": "PAY",
      "amount_minor": 250000,
      "reason_codes": ["STOCKOUT_RISK", "CRITICAL_SUPPLIER"]
    },
    {
      "supplier_id": "packright",
      "invoice_number": "PR-15007",
      "action": "DEFER",
      "amount_minor": 150000,
      "reason_codes": ["LOW_INVENTORY_RISK", "BATCH_CASH_PRIORITY"]
    },
    {
      "supplier_id": "cleanpro",
      "invoice_number": "CP-70019",
      "action": "VERIFY",
      "amount_minor": 70000,
      "reason_codes": ["CONFLICTING_SUPPLIER_STATUS"]
    }
  ]
}
~~~

### 8.4 Verifier and operator decision

~~~json
{
  "verifier": {
    "result": "REQUIRES_OPERATOR",
    "verified_batch_id": "day-0-criticality-aware-v1",
    "reason_codes": ["FINANCIAL_ACTION"],
    "checks_passed": ["KNOWN_SUPPLIERS", "EXACT_AMOUNTS", "CURRENT_STATE", "BATCH_CASH_AVAILABLE"]
  },
  "operator": {
    "decision": "APPROVE",
    "approved_batch_id": "day-0-criticality-aware-v1"
  }
}
~~~

Verifier results are **BLOCKED**, **REQUIRES_OPERATOR**, and **VERIFIED**. Operator decisions are **APPROVE**, **MODIFY**, and **REJECT**. A blocked or rejected batch has no approved batch ID. MODIFY creates a new batch ID and state snapshot, then loops through the verifier again.

### 8.5 ProcureGym transition

~~~json
{
  "scenario_id": "restaurant_demo_v1",
  "batch_id": "day-0-criticality-aware-v1",
  "seed": 138,
  "day_before": 0,
  "day_after": 1,
  "paid_invoice_numbers": ["FF-10482", "PF-25031"],
  "deferred_invoice_numbers": ["PR-15007"],
  "review_invoice_numbers": ["CP-70019"],
  "cash_before_minor": 500000,
  "cash_after_minor": 100000,
  "reward": 4.2,
  "raw_metrics": {
    "stockout_days": 0,
    "late_fees_minor": 0,
    "negative_cash_events": 0,
    "supplier_disruptions": 0
  }
}
~~~

All financial arithmetic uses integer minor units or Decimal. Binary floating point must not determine balances.

### 8.6 Full-payment receipt proof

~~~json
{
  "receipt_id": "receipt_fresh_farms_ff10482",
  "supplier_id": "fresh_farms",
  "invoice_number": "FF-10482",
  "amount_minor": 150000,
  "currency": "USD",
  "paid_date": "2026-08-30",
  "extraction_method": "ocr_plus_deterministic_rules",
  "source": "synthetic_fixture_replay",
  "status": "VERIFIED",
  "matched_payment_status": "PAID_CONFIRMED"
}
~~~

The receipt proof is valid only for a full simulated payment that was present in an approved, reverified batch. The gate requires one exact supplier-and-invoice composite match, the full approved amount, and matching currency. A missing, partial, duplicated, ambiguous, or mismatched proof routes to review and leaves the AP obligation open. The approved ProcureGym PAY already performed the one cash deduction (economically, Dr Accounts Payable / Cr Cash); proof confirmation changes status and consumes the receipt ID without changing cash or day. `PAID_CONFIRMED` is a demo-ledger state, not evidence of a real bank transaction.

---

## 9. Core workflow

1. Operator selects one known supplier, then uploads its invoice.
2. Ingestion assigns an immutable document ID and duplicate hash.
3. OCR returns ordered words, normalized boxes, raw text, language metadata, and an honest failure state. The UI retains every token rather than showing selected evidence alone.
4. An anchored rule and Ryan's local specialist produce zero or more invoice-number candidates. Independently, a deterministic total-anchor rule may label the displayed invoice amount from OCR; each token carries its business label and rule/model/OCR provenance.
5. Document gate checks grounding, ambiguity, agreement, entity margin, confirmed supplier selection, and OCR status.
6. Uncertain identity goes to document review before lookup.
7. Confirmed supplier ID plus invoice number retrieves one immutable synthetic record.
8. Restaurant state adds cash and all other active supplier obligations.
9. Each comparison policy receives an identical state snapshot.
10. Policy proposes one daily batch containing PAY, DEFER, or VERIFY for every active invoice.
11. Verifier checks the entire batch against exact financial and state constraints.
12. Operator approves, modifies, or rejects the batch.
13. A modified batch gets a new ID and state snapshot, then returns to the verifier.
14. Rejecting or not committing the batch changes nothing and does not advance time.
15. ProcureGym atomically applies one reverified, approved daily batch and advances exactly one day. Each approved simulated PAY deducts cash exactly once—the demo's economic **Dr Accounts Payable / Cr Cash** entry.
16. VERIFY items receive no direct payment; if the daily batch commits, they remain in review while global time advances.
17. UI shows raw outcomes and reward beside the baseline.
18. For a simulated PAY item, the operator uploads a full-payment receipt image or selects the visibly labeled synthetic fixture with its actual generation provenance.
19. Receipt OCR returns raw text and every token; the deterministic parser grounds supplier, invoice number, amount, currency, paid date, and receipt ID where present. LayoutLMv3 is not invoked for receipt fields.
20. The payment-proof gate compares those fields with the approved simulated payment and canonical AP record.
21. An exact, unambiguous full match moves the demo-ledger lifecycle from `SIMULATED_PAYMENT_APPROVED` to `PAID_CONFIRMED`, consumes the receipt ID, and leaves day and cash unchanged. Proof closes status; it does not post a second payment.
22. Any mismatch or incomplete evidence routes to receipt review; it never silently closes the obligation.
23. After exact Fresh Farms proof, **Done — view AP history** opens a four-record dashboard: **Open invoices (2)** contains PackRight `PR-15007` as DEFER and CleanPro `CP-70019` as VERIFY; **Paid · awaiting proof (1)** contains Prime Foods `PF-25031` as `SIMULATED_PAYMENT_APPROVED`; **Completed (1)** contains Fresh Farms `FF-10482` as `PAID_CONFIRMED` with receipt `RCPT-FF-10482`.
24. The dashboard states that these cash/AP controls support working-capital operations but are not a full net-working-capital calculation because AR, inventory valuation, and other current assets/liabilities are excluded.

---

## 10. Safety and human governance

Document confidence and financial-action risk are separate.

### Document gate

The identity cannot proceed when:

- No candidate is grounded in OCR
- More than one plausible invoice number remains
- Rule and model disagree without human confirmation
- Supplier identity is unknown or conflicting
- Candidate duplicates an active record unexpectedly
- OCR or model processing fails

### Procurement verifier

Before a batch can reach the operator, apply checks by action:

**Every batch item:**

- Composite supplier and invoice key exists
- Document identity was verified
- State version is current
- Same invoice appears only once in the batch

**PAY:**

- Supplier is known and active
- Invoice is unpaid
- Amount exactly equals the full outstanding P0 amount
- Currency matches the scenario
- Aggregate PAY amounts fit available cash and leave nonnegative cash

**DEFER:**

- Invoice is known and unpaid
- No payment amount enters the committed total

**VERIFY:**

- Verified document identity may carry missing or conflicting business context
- It contributes zero to committed payments and remains queued for review
- Its context conflict does not block otherwise valid PAY or DEFER items in the atomic daily batch

An unknown or unverified document still fails upstream and never becomes a VERIFY batch item.

### Operator gate

- Every daily batch requires an explicit operator click before simulation.
- MODIFY changes an action enum, not the amount, and always loops through the verifier again.
- REJECT or no commit makes no state change and does not advance time.
- VERIFY makes no direct payment; it remains queued if the operator commits the rest of the daily batch.
- UI controls are sandbox confirmation, not authenticated financial authorization.

### Payment-proof gate

- The AP record must already be `SIMULATED_PAYMENT_APPROVED` from the current approved batch.
- Supplier ID plus invoice number must identify exactly one canonical obligation.
- Receipt amount must equal the complete approved amount; P0 rejects partial or excess amounts.
- Currency must match and the receipt evidence must be grounded in OCR text.
- Duplicate receipt IDs or a receipt already consumed by another obligation are blocked.
- Missing or ambiguous fields route to receipt review and leave the AP record open.
- Ryan's LayoutLMv3 adapter is not credited with receipt extraction; the UI identifies OCR and deterministic rules as the active components.
- Proof confirmation must preserve the cash balance produced by the approved ProcureGym step; repeated or forged proof cannot create another debit or credit.

### Audit log

Record every invoice and receipt OCR token with confidence, normalized box, business label, and source; model/rule evidence; document-gate result; synthetic record/version; recommendation/reasons; verifier checks; operator decision; ProcureGym transition and accounting interpretation; proof-gate checks; and AP lifecycle transition.

---

## 11. ProcureGym specification

### Interface

~~~python
state, info = env.reset(seed=138)
next_state, reward, terminated, truncated, info = env.step(approved_daily_batch)
~~~

### Deterministic transition rules

- Each step represents one restaurant day.
- Inventory coverage decreases by one day unless a scheduled delivery arrives.
- PAY subtracts the full invoice amount exactly once; the UI describes the simulated economic entry as **Dr Accounts Payable / Cr Cash**. It schedules delivery only when the synthetic fixture explicitly says payment unlocks a pending replenishment.
- DEFER leaves the invoice unpaid and ages it one day.
- VERIFY makes no direct payment and remains queued for review while a committed batch advances the restaurant day.
- Past-due invoices accrue the configured synthetic fee.
- Critical suppliers may become disrupted after a deterministic fixture threshold.
- The episode terminates if any high-criticality inventory remains at zero for two consecutive days.
- The complete batch passes aggregate cash validation and commits atomically or not at all.
- Receipt proof happens after this transition. A verified proof changes the matching obligation from `SIMULATED_PAYMENT_APPROVED` to `PAID_CONFIRMED` without changing cash, day, or the earlier accounting amount.
- Identical seed, state, and actions must produce identical results.

### Reward design

Reward is a summary for policy comparison, not the only result. Raw metrics must always appear beside it.

ProcureGym reward should favor operational continuity and penalize critical stockouts, supplier disruption, late fees, and negative cash. Unsafe or unapproved proposals are scored separately by the pre-verifier safety harness because they never enter ProcureGym. Reward weights are configuration, not objective truth. A policy cannot be declared better solely because the team chose weights that favor it.

### Policies compared on the same state

1. **Earliest Due First:** sort valid invoices by due days ascending, then supplier ID; PAY each full invoice that fits remaining cash, skip an invoice that does not fit, and continue; incomplete business context returns VERIFY.
2. **Criticality-Aware Greedy v1, P0:** apply the frozen scoring and batch-allocation algorithm in Section 7.
3. **Learned ProcureGym Agent, P1:** use the same structured state and masked legal actions; disclose algorithm/model, training scenarios, reward, seeds, runtime, configuration, and version.

Controlled evaluation compares raw proposals from identical seeded states. Every policy uses the same document set, batch verifier, and fixed executor that applies a valid batch without human edits. The live operator workflow demonstrates governance separately and is not mixed into the policy benchmark.

---

## 12. Primary demo scenario

| Supplier | Category | Amount | Due | Inventory | Document path |
|---|---:|---:|---:|---:|---|
| Fresh Farms | Produce | $1,500 | Tomorrow | 2 days | English live model or labeled replay |
| Prime Foods | Meat | $2,500 | 3 days | 3 days | English live model or labeled replay |
| PackRight | Packaging | $1,500 | 1 day overdue | 20 days | English fixture |
| CleanPro | Cleaning | $700 | Today | 15 days | Verified English ID; conflicting supplier-status context |

**Cash available:** $5,000

**Obligations shown:** $6,200

Illustrative hypothesis:

- Recommend PAY Fresh Farms
- Recommend PAY Prime Foods
- Recommend DEFER PackRight
- Recommend VERIFY CleanPro

This is a hypothesis, not a hardcoded success result. The UI must show actual policy output. The operator may change it before ProcureGym runs.

For the locked approved-demo path, the final AP-history dashboard reconciles all four records exactly:

| Exact final category | Records |
|---|---|
| **Open invoices (2)** | PackRight `PR-15007` · $1,500 · DEFER; CleanPro `CP-70019` · $700 · VERIFY |
| **Paid · awaiting proof (1)** | Prime Foods `PF-25031` · $2,500 · `SIMULATED_PAYMENT_APPROVED` |
| **Completed (1)** | Fresh Farms `FF-10482` · $1,500 · `PAID_CONFIRMED` with receipt `RCPT-FF-10482` |

The approved batch reduces simulated cash from $5,000 to $1,000 once for Fresh Farms plus Prime Foods. Fresh Farms receipt confirmation changes proof/status only and leaves cash at $1,000.

A separate **UnknownCo adversarial document** has ambiguous identity, never reaches canonical lookup, is excluded from the $6,200 obligations, and demonstrates document-level fail-closed behavior. CleanPro demonstrates the different procurement-level VERIFY path after its identity is already verified.

### Multilingual experiment

Spanish or another language may appear only as a clearly labeled experiment. The current model card describes an English/SROIE-like specialization. A non-English document routes to review unless the team publishes a locked per-language sample count and result. Precomputed OCR is not proof of multilingual model accuracy.

---

## 13. Functional requirements

| ID | Priority | Requirement |
|---|---|---|
| FR-01 | P0 | Operator selects a known supplier, uploads its invoice image, and ingestion records supplier-selection provenance plus immutable document ID/hash. |
| FR-02 | P0 | OCR returns every word, confidence, normalized box, raw text, metadata, and explicit failure state; UI preserves one visible label/source record per token. |
| FR-03 | P0 | Local specialist returns invoice-number proposal, entity evidence, latency, and version. Invoice amount evidence is labeled separately by the deterministic OCR total-anchor rule and never attributed to the specialist. |
| FR-04 | P0 | Uncertain or ungrounded identity fails closed before canonical lookup. |
| FR-05 | P0 | Confirmed composite identity retrieves exactly one visibly synthetic supplier record. |
| FR-06 | P0 | Restaurant state uses exact financial arithmetic and a versioned snapshot. |
| FR-07 | P0 | Criticality-Aware Greedy v1 returns one strict daily batch with PAY/DEFER/VERIFY and reason codes for every active invoice. |
| FR-08 | P0 | Verifier checks the entire batch for identity, state, duplicates, exact amounts, currency, suppliers, and aggregate cash. |
| FR-09 | P0 | Operator can APPROVE, MODIFY, or REJECT; every modified batch returns through verification. |
| FR-10 | P0 | Only an approved, reverified daily batch mutates ProcureGym; no action changes a real system. |
| FR-11 | P0 | ProcureGym implements seeded reset/step, seven-day horizon, reward, raw outcomes, and audit trace. |
| FR-12 | P0 | Identical initial state runs under Criticality-Aware Greedy v1 and Earliest Due First using the fixed evaluation executor. |
| FR-13 | P0 | Guided UI labels OCR-only, deterministic-rule, LayoutLMv3, looked-up, recommended, verified, human-confirmed, simulated, and proof-confirmed data separately; technical evidence remains expandable. |
| FR-14 | P0 | Fixture/replay path is visibly distinguished from live OCR or model execution. |
| FR-15 | P0 | Any statement that the model ran is backed by one actual recorded inference with artifact/version metadata; otherwise UI says replay and no model ran. |
| FR-16 | P0 | Guided UI accepts a payment-receipt image after an approved simulated PAY and shows every OCR token, deterministic parsed fields, provenance, and proof checks. |
| FR-17 | P0 | Payment-proof gate requires exact supplier, invoice, full amount, and currency match; ambiguous, duplicate, partial, excess, or mismatched proof leaves lifecycle status unchanged and payment proof pending. |
| FR-18 | P0 | Approved simulated PAY deducts cash once and is explained as Dr Accounts Payable / Cr Cash; verified proof moves only the matching obligation to `PAID_CONFIRMED` without a second cash deduction and is never labeled as a bank transaction. |
| FR-19 | P0 | Repository includes one inspected synthetic receipt fixture and metadata without exposing `FAL_KEY`; Fal status or deterministic fallback provenance is explicit. |
| FR-20 | P0 | Evaluation and UI report invoice-identity correctness, daily ordered-priority quality, and exact payment-action correctness as separate components with raw outcomes. |
| FR-21 | P1 | Generate offline extraction action-outcome matrix with accuracy, latency, and review cost. |
| FR-22 | P1 | Train constrained contextual-bandit router without using locked test split. |
| FR-23 | P1 | Compare frozen bandit with calibrated threshold and always-review baselines. |
| FR-24 | P1 | Train a named ProcureGym policy only over masked legal actions, then compare identical seeded scenarios with deterministic baselines and a bounded oracle when available. |
| FR-25 | P1 | Add separately reported non-English experiments without generalized claims. |
| FR-26 | P0 | After exact Fresh Farms proof, **Done — view AP history** shows all four obligations once across `Open invoices (2)`, `Paid · awaiting proof (1)`, and `Completed (1)`, including exact supplier, invoice, amount, action/status, and receipt ID where available. |

---

## 14. Evaluation and success metrics

### Three-axis RL scorecard

The evaluation artifact must present these columns independently for every run:

| Axis | Required metrics | What “right” means |
|---|---|---|
| **1 · Invoice identity** | Strict exact match, wrong-auto-accept count, review rate, missing/ambiguous count, latency, route cost | Matches a frozen label exactly, or safely requests review when the label cannot be supported |
| **2 · Prioritization ranking** | Ordered supplier list for every day, critical suppliers protected in top positions, stockout/disruption days, late fees, cumulative reward, and schedule regret when an oracle exists | Produces the best verified downstream restaurant outcome under the same starting state and cash—not merely the ranking the team expected |
| **3 · Payment action** | Correct composite supplier/invoice, exact full amount/currency, timing, duplicates, stale actions, over-budget attempts, and blocked-invalid count | Chooses a currently legal verified action; the verifier independently enforces identity, exact amount, and cash constraints |

The UI must never collapse these into a single “accuracy” percentage. A policy can identify an invoice correctly and still rank it poorly; it can rank a supplier correctly and still emit an invalid payment. Those failures must remain visible.

### Document perception

- Invoice-number exact match
- Missing and ambiguous-result rates
- Wrong-auto-accept rate
- Accepted coverage and selective accuracy
- Review rate
- Latency and model-call count

### Procurement safety

- Unsafe proposals intercepted
- Unapproved mutations: target **zero**
- Stale-state, duplicate, and negative-cash proposals blocked
- Operator approve/modify/reject counts

### AP lifecycle proof

- Exact receipt-proof match rate and review rate
- Partial, duplicate, wrong-invoice, wrong-supplier, wrong-amount, and wrong-currency proofs blocked
- AP obligations closed without an approved simulated payment: target **zero**
- AP obligations closed without verified full-payment proof: target **zero**

### Synthetic restaurant outcomes

- Critical stockout days
- Supplier disruptions
- Late fees and ending cash in minor units
- Negative-cash events
- Invoices paid/deferred/verified
- Cumulative reward

These measures support operational working-capital discipline but do not constitute a full net-working-capital calculation. The demo covers cash and selected Accounts Payable obligations; it excludes Accounts Receivable, balance-sheet inventory valuation, and other current assets and liabilities.

### Comparison rules

- Run policies from identical seeded states.
- Show raw outcomes beside reward.
- Keep dataset test split untouched until policy freeze.
- Keep missing predictions in the denominator.
- Report fixture count beside each accuracy percentage.
- Do not call a hardcoded expected action an evaluation result.
- Do not call ProcureGym-trained until a policy was actually trained.

---

## 15. User interface requirements

The primary recording experience is a single progressive four-step demo, not a collection of peer navigation tabs. Completed steps remain summarized, the current step owns the primary action, and future actions stay visibly locked.

### Step 1 — Read invoice

- Select `data/procureagent/assets/fresh_farms_invoice.png` or upload PNG/JPEG.
- Show source, immutable hash, dimensions, OCR runtime/status, and live-versus-fixture provenance.
- Preserve and expose every OCR token with text, confidence, normalized box, business label, and source.
- Label Ryan's LayoutLMv3 output as **invoice number only**. Label invoice amount evidence as **Tesseract OCR + deterministic total-anchor rule**.
- For a bundled evaluation fixture, show post-inference strict exact match against a hidden answer key; never send that key to OCR/model, and label custom uploads **not scored**. Keep fixture evaluation separate from document-gate acceptance.

### Step 2 — Confirm identity and review plan

- Require explicit **CONFIRM**, **CORRECT**, or **REJECT** before composite lookup.
- Show the synthetic AP lookup and restaurant context only after confirmation, and distinguish the canonical looked-up payable amount from displayed OCR amount evidence.
- Explain the ordered PAY/DEFER/VERIFY plan, reason codes, exact amounts, verifier result, and fixture/replay identity provenance for the other canonical invoices.

### Step 3 — Approve simulated payment

- Require an explicit operator **APPROVE** after verification and before any ProcureGym mutation.
- Show before/after cash, state version, day, raw outcomes, reward, ordered ranking, and baseline/oracle comparison.
- Explain each approved PAY as a simulated **Dr Accounts Payable / Cr Cash** entry. Cash is deducted here exactly once; no bank or external ledger is connected.

### Step 4 — Verify receipt proof

- Select `data/procureagent/assets/fresh_farms_payment_receipt.png` or upload PNG/JPEG; expose `data/procureagent/assets/receipt_provenance.json`.
- Show every receipt OCR token and identify supplier, invoice number, amount, currency, paid date, and receipt ID as **OCR + deterministic parser** output, never LayoutLMv3 output.
- Require exact, grounded, unused full-payment proof before `PAID_CONFIRMED`.
- State that receipt confirmation closes lifecycle status and consumes proof without a second cash deduction.

### Completion — View AP history

- Show the exact call to action **Done — view AP history** after Fresh Farms reaches `PAID_CONFIRMED`.
- Reconcile every synthetic obligation once across **Open invoices (2)**, **Paid · awaiting proof (1)**, and **Completed (1)** using the locked mapping in section 12.
- Explain the approved $4,000 simulation as Dr AP—Fresh Farms $1,500 + Dr AP—Prime Foods $2,500 / Cr Cash $4,000 once. Fresh Farms receipt proof has $0 second cash impact; Prime Foods remains awaiting proof.
- Label the display as operational working-capital support, not a complete net-working-capital statement. Do not infer AR, inventory valuation, or unmodeled current assets/liabilities.

### Technical evidence and secondary views

The guided outcome stays readable at presentation distance. Expandable technical evidence provides raw OCR, every labeled token/box/confidence/source, document and artifact hashes, rule/model candidates, model version/latency/scores, frozen-gate reasons, verifier checks, audit IDs, source files, receipt parser/proof checks, and live-versus-fixture provenance. Restaurant overview, failure examples, C5 comparisons, C6 development-only results, task ownership, downloads, and deployment information may remain secondary views; they must not interrupt the four-step happy path.

---

## 16. Team member roster

| Member | Current recorded status |
|---|---|
| **[Sasa Phanitsombat](https://www.linkedin.com/in/sasakorn-p/)** | Co-owner: C5 and C7; owner: C8 |
| **[Ryan Nie](https://www.linkedin.com/in/ryanznie/)** | Co-owner: C1, C2, C4, C6, and C7; owner of the pre-existing invoice-number model asset |
| **[David Lee](https://www.linkedin.com/in/authordavidlee/) / @cheezburgerz** | Co-owner: C1, C6, and C7 |
| **[Wilson Wu](https://www.linkedin.com/in/wilson1wu/) / @skylarwooster** | Owner: C0, C3, and C5; co-owner: C7 |
| **[Dillon Johnson](https://www.linkedin.com/in/dillonqjohnson/)** | Co-owner: C1, C2, C4, C6, and C7 |

Public background summaries are maintained in the repository [README](../README.md#team). These assignments were confirmed by Wilson on 30 August 2026. Multiple names indicate shared ownership; the co-owners must agree who signs off the category's done test. To protect the hackathon deadline, Wilson is also implementing the integrated reference path across C0–C8. This does not erase category ownership: compatible teammate work is merged against the frozen contracts, and the listed owner or co-owners still review and sign off their category.

---

## 17. Task-category claim board

### How to claim work

1. Change **Claim state** from OPEN to CLAIMED and add your name under **Owner / delivery**.
2. A claim does not mean completion; delivery remains NOT VERIFIED until the done test passes.
3. Tell the team with: **CLAIM**, **PAIR**, **HELPING**, or **BLOCKED**, followed by the category and exact dependency.
4. One person owns the final done test even when several people help.
5. No owner may weaken safety or truthfulness requirements to finish faster.

| Claim state | Category | Scope | Dependencies | Deliverable / done test | Owner / delivery |
|---|---|---|---|---|---|
| **CLAIMED** | **C0 — Pivot contract and locked fixtures** | Freeze schemas, enums, reasons, primary invoices, restaurant, lookup, seed, AP lifecycle, and exact full-payment proof. Keep AR and partial payments out. | All component owners | One invoice and receipt fixture validate end to end; no P0 contract requires AR or partial payments. | **Wilson / @skylarwooster · VERIFIED · fixture hash and mutation tests pass** |
| **CLAIMED** | **C1 — OCR and document ingestion** | Invoice/receipt image intake, ID/hash, OCR words, boxes, raw text, metadata, fallback, and failure behavior. | Locked images and OCR schema | One command produces contract-valid OCR for both document types; missing OCR yields labeled replay or review, never invented identity. | **David / @cheezburgerz + Ryan Nie + Dillon · Wilson reference implementation · INTEGRATION PASS · owner sign-off pending** |
| **CLAIMED** | **C2 — Local specialist, evidence, and document gate** | Package Ryan's invoice adapter; return value, entity tokens/boxes/scores, latency, version, failures, and anchored-rule candidate; merge invoice proposals and decide verified identity or document review. Receipt fields remain OCR-plus-rules. | C1; Ryan's artifact; license review | Another member runs correct, wrong, missing, ungrounded, ambiguous, and rule/model-disagreement fixtures outside a notebook; unsafe identity never reaches C3 and every failure remains in metrics. | **Ryan Nie + Dillon · Wilson reference implementation · INTEGRATION PASS · owner sign-off and model-license clarification pending** |
| **CLAIMED** | **C3 — Supplier lookup and restaurant state** | Composite lookup, synthetic invoices, exact cash, inventory, due dates, criticality, status, versioning, and AP lifecycle through verified receipt proof. | C0 contract; C2 verified identity for live integration | $5,000 cash and $6,200 obligations reproduce exactly; unknown IDs activate nothing; only exact full-payment proof closes an obligation. | **Wilson / @skylarwooster · VERIFIED · exact lookup, state, and AP proof tests pass** |
| **CLAIMED** | **C4 — Recommendation, verifier, and governance** | Implement Criticality-Aware Greedy v1, daily batch schema, reasons, hard checks, operator controls, reverification, and audit events. | C3; document gate | Four-invoice batch is deterministic; unsafe, modified-unverified, unapproved, or stale batches cannot reach ProcureGym. | **Ryan Nie + Dillon · Wilson reference implementation · INTEGRATION PASS · owner sign-off pending** |
| **CLAIMED** | **C5 — ProcureGym, reward, and baselines** | Seeded batch reset/step, transitions, horizon, three-axis scorecard, raw metrics, reward, Earliest Due First, legal-schedule oracle when available, and fixed evaluation executor. | C0, C2 identity result, C3, C4 approved-batch contract | Same state runs reproducibly under both P0 policies; identity, ordered ranking, and exact action correctness remain separate; only approved batches change state. | **Sasa P + Wilson / @skylarwooster · INTEGRATION PASS · Sasa co-owner sign-off pending** |
| **CLAIMED** | **C6 — Contextual-bandit Router Lab** | Action matrix, constrained reward, training/development split, frozen test, and fixed-gate comparison. | C1/C2 outputs; locked labels | Report learned router only if it beats declared baselines without more unsafe accepts; otherwise show negative result. | **David / @cheezburgerz + Ryan Nie + Dillon · Wilson reference implementation · DEV LAB IMPLEMENTED · NOT VERIFIED · P1 · no frozen-test/generalization claim** |
| **CLAIMED** | **C7 — Demo UI, orchestration, and deployment** | Four-step guided demo, three-axis RL scorecard, every-token model/rule/OCR provenance, AP and accounting explanation, invoice-to-receipt lifecycle, expandable technical evidence, container/health path, live/replay labels, public URL, and offline backup. | Stable C0–C5 contracts | Clean-browser 2–3 minute invoice→confirm/plan→simulated approval→receipt-proof demo works; public and offline paths pass rehearsal. | **Wilson / @skylarwooster + Sasa P + Ryan Nie + Dillon + David / @cheezburgerz · INTEGRATION AND PUBLIC QUICK-TUNNEL REHEARSAL PASS · permanent Cloud Run deployment and owner sign-off pending** |
| **CLAIMED** | **C8 — Evaluation, QA, and presentation proof** | Locked three-axis evaluation, invoice/action/receipt adversarial attacks, results card, runbook, talk track, and release checklist. | All P0 outputs | One command proves identity correctness, priority ranking/outcomes, exact action validity, and zero unapproved or unproved lifecycle mutations; each public claim is reproducible or labeled planned. | **Sasa P · Wilson reference implementation · 219 passed, 2 opt-in skips; enabled Tesseract/UI/reward subset 49/49; offline acceptance 9/9; live acceptance 10/10 · Sasa owner sign-off pending** |

### 30 August 2026 integrated implementation checkpoint

The Wilson reference path now runs invoice PNG/JPEG ingestion, real Tesseract OCR with every-token provenance, the revision-pinned Ryan LayoutLMv3 invoice-number adapter, deterministic OCR amount labeling, fail-closed document review, exact synthetic lookup, deterministic first/second/third supplier ranking, independently verified exact payment actions, explicit operator approval, a seeded ProcureGym transition, receipt OCR/parser evidence, exact full-payment proof through simulated `PAID_CONFIRMED`, and a final 2/1/1 AP-history reconciliation. The guided screen keeps technical evidence accessible while separating the one $4,000 Dr AP / Cr Cash simulation interpretation from proof-only status closure. The clean-browser path completed locally and through the public Quick Tunnel. The default full suite reports **219 passed and two intentionally opt-in real-Tesseract smokes skipped**; the enabled focused Tesseract/UI/reward subset passes **49/49**, the offline acceptance artifact reports **9/9**, and the real-model artifact reports **10/10**.

Those integration results do not replace the named owners' review. They also do not mean real money moved, do not establish aggregate LayoutLMv3 accuracy, and do not establish Router Lab generalization. Permanent Cloud Run hosting still requires approval of a dedicated billed Google Cloud/Firebase project.

Deployment handoff: follow the [Cloud Run checklist](CLOUD_RUN_DEPLOY.md). [Sasa's Streamlit checklist](SASA_STREAMLIT_DEPLOY.md) remains an optional fallback. Until a permanent URL is created and rehearsed, the Cloudflare Quick Tunnel is a temporary, public, unauthenticated fallback whose URL changes on restart and has no uptime guarantee. Only synthetic fixtures may cross it.

### Parallel-work rule

After C0 freezes contracts, C1 and fixture-backed scaffolds for C2–C5 and C7 may build in parallel. Their final done tests follow the real integration edges below. C8 writes tests early and verifies integration. C6 remains isolated from P0 so it cannot destabilize the live demo.

---

## 18. Dependency graph and merge gates

~~~mermaid
flowchart LR
    C0[C0 contracts and fixtures] --> C1[C1 OCR]
    C0 --> C7[C7 UI shell]
    C1 --> C2[C2 specialist and document gate]
    C2 --> C3[C3 restaurant state]
    C3 --> C4[C4 policy and verifier]
    C3 --> C5[C5 ProcureGym]
    C1 --> R[Receipt OCR and proof gate]
    C3 --> R
    C4 --> C5
    C2 --> I[Integrated vertical slice]
    C5 --> I
    R --> I
    C7 --> I
    I --> C8[C8 independent QA]
    C1 --> C6[C6 Router Lab]
    C2 --> C6
    C8 --> F[Demo freeze]
~~~

### Gate 0 — Contract freeze

- InvoiceAgent is the only active P0 product.
- JSON contracts, enums, and fixture/replay labels are frozen.
- Categories have explicit owners.

### Gate 1 — Thin vertical slice

- One English invoice follows OCR or labeled replay through identity review.
- Lookup returns one synthetic record.
- One proposal passes verifier and operator gate.
- One approved action advances ProcureGym.
- One full-payment receipt passes OCR/rules and the exact proof gate, then closes only the matching demo AP obligation.

### Gate 2 — Policy comparison

- Earliest Due First and Criticality-Aware Greedy v1 receive identical seeded state.
- Each emits one daily batch and passes through the same verifier and fixed evaluation executor.
- Raw proposals, outcomes, and reward reproduce.
- Rejecting or withholding commit changes nothing; VERIFY makes no direct payment inside a committed batch.

### Gate 3 — Safety proof

- Confident wrong ID, unknown supplier, duplicate PAY, stale state, and over-budget batch are blocked.
- Partial, duplicate, ambiguous, wrong-invoice, wrong-supplier, wrong-amount, and wrong-currency receipt proofs are blocked.

### Gate 4 — Demo freeze

- Public URL and offline fallback pass.
- Live and fixture lanes are labeled.
- Results card matches the latest reproducible run.
- No feature work after freeze without a team decision.

---

## 19. Live demo script

1. **Set the problem:** Sasa's restaurant has $5,000 and $6,200 of supplier bills.
2. **Step 1 — Read invoice:** select the bundled Fresh Farms image, run actual OCR/model or show an explicit replay label, and show that every OCR token has a box, confidence, label, and provenance. LayoutLMv3 labels only `FF-10482`; invoice amount evidence comes from OCR plus a deterministic rule.
3. **Step 2 — Confirm and plan:** show why the frozen gate requests review, explicitly confirm/correct/reject identity, then reveal the synthetic AP context and explained PAY/DEFER/VERIFY plan.
4. **Step 3 — Approve simulation:** show verifier checks and require APPROVE. Advance ProcureGym, show raw outcomes/baselines, and describe the $4,000 synthetic batch interpretation: Dr AP—Fresh Farms $1,500 + Dr AP—Prime Foods $2,500 / Cr Cash $4,000 once.
5. **Step 4 — Verify receipt:** use the bundled receipt, show every OCR token and deterministic parser field, run exact proof checks, and move Fresh Farms to `PAID_CONFIRMED` while proving cash did not change again.
6. **Finish on AP history:** click **Done — view AP history** and show `Open invoices (2)` for PackRight/CleanPro, `Paid · awaiting proof (1)` for Prime Foods, and `Completed (1)` for Fresh Farms. State that the receipt had $0 second cash impact.
7. **Use technical evidence on demand:** expand hashes, token boxes/sources, model/runtime/scores, rule evidence, verifier/audit records, proof checks, and artifact provenance when a judge asks.
8. **Show failure boundaries if time permits:** UnknownCo fails document identity before lookup; verified CleanPro receives procurement VERIFY for conflicting business context; a mismatched or duplicate receipt leaves payment proof pending.
9. **Close honestly:** the small specialist proposes only invoice identity; deterministic invoice/receipt rules label other grounded fields; policy reasons over structured context; rules and people govern; ProcureGym measures consequences before any real integration. The AP dashboard supports working-capital discipline but is not a complete net-working-capital calculation.

---

## 20. Acceptance criteria

| ID | Given | When | Then |
|---|---|---|---|
| AC-01 | Valid English fixture and OCR | Actual specialist runs | Number, evidence, artifact/version, and measured latency are captured |
| AC-02 | Ungrounded or ambiguous identity | Document gate runs | Lookup does not run and item enters review |
| AC-03 | Confirmed composite identity | Lookup runs | Exactly one visibly synthetic supplier record returns |
| AC-04 | Unknown supplier or invoice key | Lookup runs | No payable activates |
| AC-05 | Versioned restaurant state | Criticality-Aware Greedy v1 runs | One deterministic daily batch covers every active invoice with actions and reasons |
| AC-06 | Invalid amount, stale state, duplicate, or over-budget batch | Verifier runs | Entire batch is blocked atomically |
| AC-07 | Valid daily batch without operator commit | Integration runs | ProcureGym state and day do not change |
| AC-08 | Operator approves reverified primary batch | ProcureGym steps | Fresh Farms and Prime Foods deduct exactly once, cash becomes $1,000, one day advances, and UI explains simulated Dr AP / Cr Cash |
| AC-09 | Operator modifies DEFER to an unaffordable PAY | Batch returns to verifier | Modified batch is blocked and ProcureGym does not step |
| AC-10 | Operator rejects or does not commit | Integration runs | No state change or simulated-day advance occurs |
| AC-11 | Approved batch contains VERIFY | ProcureGym steps | VERIFY invoice receives no payment, remains queued, and the globally committed day advances once |
| AC-12 | Same initial state | Criticality-Aware Greedy and Earliest Due First run | Raw proposals and governed outcomes appear side by side under the fixed protocol |
| AC-13 | Fixture/replay used | UI renders | It is visibly labeled; model-run claims are suppressed unless backed by recorded real inference |
| AC-14 | Non-English experiment lacks results | Document arrives | It routes to review and no accuracy claim appears |
| AC-15 | Full adversarial suite | QA runs | Zero unsafe or unapproved state mutations |
| AC-16 | Approved simulated Fresh Farms PAY and exact full-payment receipt | Proof gate runs | Every receipt token/provenance and all exact checks show; only Fresh Farms becomes `PAID_CONFIRMED`, with day and cash unchanged from the approved step |
| AC-17 | Receipt is partial, duplicated, ambiguous, wrong supplier, wrong invoice, wrong amount, or wrong currency | Proof gate runs | Lifecycle status remains `SIMULATED_PAYMENT_APPROVED`, proof remains pending, and the reason is visible |
| AC-18 | Guided demo runs in a clean session | User completes the four progressive steps | Every-token OCR/model/rule/provenance labels remain distinct, technical evidence is accessible, future actions stay gated, and no real-payment claim appears |
| AC-19 | Frozen labeled invoice | Identity evaluation runs | Strict exact result is shown independently from routing cost and downstream procurement reward |
| AC-20 | Same versioned restaurant state | Each policy runs | Daily first/second/third supplier order, runway inputs, exact actions, and downstream outcomes are recorded |
| AC-21 | Policy proposes wrong supplier, invoice, amount, currency, duplicate, stale, or over-budget action | Action mask or verifier runs | Action never mutates state; blocked-invalid count increases and remains visible |
| AC-22 | Three-axis evaluation completes | UI/results artifact renders | Identity, ranking, and payment-action results appear separately; no single accuracy number hides a failure |
| AC-23 | Receipt proof completes or requires review | Receipt routing reward is scored | Exact proof yields `+10`, safe review yields `-1`, unsafe acceptance yields `-25`, raw checks remain visible, and the UI makes no trained-policy claim |
| AC-24 | Exact Fresh Farms proof is confirmed | Operator selects **Done — view AP history** | Dashboard shows exactly 2 open, 1 paid awaiting proof, and 1 completed record with the locked suppliers/statuses; cash stays $1,000 and the display disclaims full NWC calculation |

---

## 21. Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Permanent deployment and owner sign-off remain open | A tested reference path is not the same as team acceptance or durable hosting | Keep the tested commit, acceptance artifacts, temporary/offline fallback, and named-owner sign-offs visible |
| Model supports only invoice number | Other fields could be falsely attributed to AI | Label invoice amount as OCR + deterministic total-anchor rule; label canonical AP and restaurant fields as synthetic lookup |
| OCR missing or slow | LayoutLMv3 needs external words and boxes | David owns adapter; retain labeled precomputed fixtures |
| Unsupported multilingual claim | Current model is English/SROIE-like | P0 English; route experiments to review; report counts before claims |
| Reward circularity | Chosen weights can manufacture a winner | Show raw metrics and identical seeds |
| Policy provenance is omitted | Audience may mistake rules for a model or the extractor for the decision-maker | Display policy type, name, and version with every proposal |
| Unsafe payment proposal | Financial consequence is high | Deterministic verifier plus mandatory operator gate |
| Actions exceed batch cash | Per-action checks miss aggregate risk | Validate entire approved batch |
| Stale recommendation | Cash or invoices may change | Require state version |
| Fixture presented as live | Damages credibility | Prominent live/replay badge and provenance |
| PAY sounds real | Audience may infer banking integration | Say Recommend PAY and Simulated paid |
| Model licensing differs from code | MIT repo does not relicense weights | Verify base, adapter, and dataset terms separately |
| Stretch RL destabilizes demo | Router work may consume integration time | Isolate C6; fixed gate remains P0 |
| Receipt is mistaken for model output | LayoutLMv3 does not read receipt fields | Label receipt OCR and deterministic rules at every step |
| Synthetic receipt is mistaken for real payment | Demo could overclaim financial integration | Show its actual Fal or deterministic generation provenance and “simulation only”; never expose the API key |
| Receipt proof is mistaken for a second payment | Audience could think cash was deducted twice | Show Dr AP / Cr Cash at approved ProcureGym PAY only; receipt confirmation changes status and consumes proof while cash stays unchanged |
| AP-history dashboard is mistaken for complete net working capital | Cash and AP alone omit material balance-sheet accounts | Say it supports working-capital discipline; explicitly exclude AR, inventory valuation, and other current assets/liabilities from any NWC claim |
| Temporary tunnel is mistaken for durable hosting | Public URL can change, expose synthetic demo traffic, or disappear | Link the Cloud Run deployment handoff; label Quick Tunnel public, unauthenticated, temporary, and synthetic-only |

---

## 22. Future work

- Accounts Receivable and general receipt reconciliation
- Complete net-working-capital reporting after AR, inventory valuation, and all other current assets/liabilities are modeled
- Partial payments
- Full amount/date/supplier document extraction
- Learned receipt-number or receipt-field model and annotations
- Calibrated multilingual evaluation
- Contextual-bandit router after baselines
- Procurement-policy learning inside ProcureGym
- Vendor/template novelty detection
- Supervised continual-learning batches from corrections
- Real integrations only after authentication, authorization, audit, and compliance design

---

## 23. Source material

- Sasa's **ProcureAgent_Hackathon_MVP_PRD.docx**, merged on 30 August 2026
- Ryan's implementation: <https://github.com/ryanznie/invoice>
- Ryan's labeled dataset: <https://huggingface.co/datasets/ryanznie/SROIE_2019_with_labels>
- Ryan's invoice-number adapter: <https://huggingface.co/ryanznie/layoutlmv3-lora-invoice-number>
- LayoutLMv3 paper: <https://arxiv.org/abs/2204.08387>
- Offline contextual-bandit replay evaluation: <https://arxiv.org/abs/1003.5956>
- Doubly robust contextual-bandit evaluation: <https://arxiv.org/abs/1103.4601>

Dataset, code, base-model, and adapter licenses must be verified separately. The repository's MIT license does not automatically relicense model weights or datasets. At the 30 August freeze, the adapter card declares MIT while the Microsoft base model declares CC BY-NC-SA 4.0 and Ryan's dataset card lists its license as unknown; Ryan must clarify the intended weight/data usage before anyone makes a broader licensing claim.

---

## 24. Final definition of done

InvoiceAgent P0 is complete only when:

- The task board records an owner for every P0 category.
- All four primary invoices reproduce the $5,000 cash versus $6,200 obligations scenario.
- At least one actual model inference is captured with artifact/version and runtime metadata before any model-run claim is made; every replay remains labeled.
- Invoice-number strict exact correctness, wrong-auto-accepts, route/review cost, and missing predictions are reported as the identity component.
- UnknownCo fails document identity before lookup, while verified CleanPro reaches the separate procurement VERIFY path.
- Every verified composite identity retrieves exactly one immutable synthetic record.
- Criticality-Aware Greedy v1 produces one deterministic daily batch covering all four active invoices.
- Modified batches are reverified; the verifier and operator gate prevent every unapproved or over-budget mutation.
- ProcureGym atomically advances exact, seeded synthetic state by one day for the approved batch.
- Earliest Due First runs the same four-invoice batch from the same starting state under the fixed comparison protocol.
- Every simulated day records the ordered supplier ranking and restaurant-runway inputs that produced first/second/third priority.
- Every PAY action names one verified composite identity and its exact full amount/currency; no learned or deterministic policy can invent these values.
- Identity correctness, priority-ranking outcomes, and payment-action correctness reproduce as separate scorecard components beside raw metrics and aggregate reward.
- UnknownCo activates no payable; CleanPro receives no direct payment while queued for review.
- The inspected synthetic receipt fixture and a user-uploaded receipt can exercise the payment-proof gate without exposing credentials; generation provenance names the deterministic fallback until Fal balance is available.
- Exact full-payment proof closes only the matching simulated AP obligation; every partial, duplicate, ambiguous, or mismatched proof cannot advance lifecycle status or consume proof.
- The receipt decision emits the declared `+10 / -1 / -25` RL-ready signal with raw proof checks, while the product states that no receipt policy or model was reinforcement-trained.
- **Done — view AP history** reconciles all four obligations exactly as 2 open / 1 paid awaiting proof / 1 completed, and labels the result working-capital support rather than a full NWC calculation.
- Public URL and offline fallback both work.
- Every presentation claim is backed by an artifact or labeled future work.
