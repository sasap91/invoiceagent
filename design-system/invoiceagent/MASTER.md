# InvoiceAgent design system

**Public signature:** InvoiceAgent by Sundai

**Personalized demo context:** Sugar & Spice Thai Restaurant, Porter Square, Cambridge. Every use is an illustrative synthetic workflow with no affiliation and no real restaurant data.

**Hero asset:** `data/procureagent/assets/invoiceagent-restaurant-hero.jpg` is an original generated illustration. It must be labeled as illustrative and must never be described as a photograph of Sugar & Spice.

Generated with the UI UX Pro Max workflow and then corrected against the implemented demo.

## Product character

- Audience: restaurant owner first, engineer second.
- Tone: calm operational storytelling with visible proof.
- Differentiator: one guided invoice-to-receipt journey; technical detail is progressively disclosed.
- Density: low in the active task, high only inside evidence expanders.

## Foundations

| Role | Value |
|---|---|
| Background | `#F8F7F3` |
| Surface | `#FFFFFF` |
| Ink | `#183029` |
| Muted ink | `#52645F` |
| Brand green | `#153F35` |
| Primary terracotta | `#A6421F` |
| Proof teal | `#08778A` |
| Success | `#176B52` |
| Warning | `#8B5A12` |
| Danger | `#9B3E37` |
| Border | `#DCE4DF` |

- Headings: Fraunces.
- Body and controls: Manrope.
- Spacing follows a 4/8/12/16/24/32px rhythm.
- Corners are soft but restrained: 10px controls, 16px cards, 24px hero/step shells.
- Motion is optional and subtle; honor `prefers-reduced-motion`.

## Interaction rules

- Minimum control height: 44px; minimum 8px spacing between adjacent targets.
- Visible 3px focus ring and keyboard-operable flow.
- Never rely on color alone; status always includes words.
- Disable asynchronous actions while running and show plain-language progress.
- Errors sit next to the failed step and state that the pipeline stopped safely.
- Sticky elements must never cover focused controls; the trust strip is static.

## Product rules

- Show simulation, human-control, and no-bank boundaries before the workflow.
- Never call model confidence “accuracy.” Fixture exact-match evaluation happens only after inference.
- Attribute invoice number to Ryan's LayoutLMv3 specialist, amount evidence to OCR + deterministic rule, and receipt fields to OCR + deterministic parser.
- Keep one primary action per step.
- Collapse completed steps into summaries; show only the current step in full.
- Keep actual repository excerpts and runtime provenance in **How this works** expanders.

## Anti-patterns

- No chat-style primary interface, fake charts, gradients, glass effects, or decorative card grids.
- No preselected human decision, hidden state mutation, or fixture data presented as live output.
- No tiny structural icons, emoji navigation, hover-only meaning, or low-contrast disabled states.
- No receipt-triggered second cash deduction.

## Release checklist

- Desktop and 375px mobile layouts are readable.
- Every OCR word is escaped before HTML rendering.
- Invoice-number, amount, receipt ID, supplier, date, and currency tokens have text labels and distinct colors.
- All financial transitions require the declared human/operator gate.
- Default route tells the four-step story without opening an engineering panel.
