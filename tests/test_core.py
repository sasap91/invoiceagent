from datetime import datetime
from decimal import Decimal
import unittest

from invoiceagent import (
    ExtractionMetadata,
    ExtractionSource,
    Invoice,
    InvoiceStatus,
    LedgerSide,
    MatchMethod,
    PaymentReceipt,
    ReceiptStatus,
    RoutingAction,
    RoutingSignals,
    ValidationError,
    decide_small_first_route,
    parse_iso_date,
    parse_money,
    reconcile,
    summarize_cash_flow,
)


AP = LedgerSide.ACCOUNTS_PAYABLE
AR = LedgerSide.ACCOUNTS_RECEIVABLE
MANUAL = ExtractionMetadata(
    source=ExtractionSource.MANUAL,
    grounded=True,
    note="Confirmed test fixture",
)


def invoice(
    number="INV-100",
    counterparty="Acme Supplies",
    amount="100.00",
    side=AP,
    issue="2026-08-01",
    due="2026-08-31",
):
    return Invoice(
        number,
        counterparty,
        amount,
        issue,
        due,
        side,
        extraction=MANUAL,
        approved=True,
    )


def receipt(
    number="RCPT-100",
    counterparty="Acme Supplies",
    amount="100.00",
    side=AP,
    paid="2026-08-20",
    reference=None,
):
    return PaymentReceipt(
        number,
        counterparty,
        amount,
        paid,
        side,
        reference,
        extraction=MANUAL,
        approved=True,
    )


class ValidationTests(unittest.TestCase):
    def test_money_is_decimal_and_strict(self):
        self.assertEqual(parse_money("12.3"), Decimal("12.30"))
        self.assertEqual(parse_money(Decimal("12.30")), Decimal("12.30"))
        for bad in (12.3, 12, "12.345", "$12.00", "1e2", " 12.00", "-1.00"):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                parse_money(bad)  # type: ignore[arg-type]

    def test_date_is_exact_and_real(self):
        self.assertEqual(str(parse_iso_date("2026-08-30")), "2026-08-30")
        for bad in ("08/30/2026", "2026-02-30", datetime(2026, 8, 30, 12)):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                parse_iso_date(bad)  # type: ignore[arg-type]

    def test_invoice_rejects_zero_and_backwards_due_date(self):
        with self.assertRaises(ValidationError):
            invoice(amount="0.00")
        with self.assertRaises(ValidationError):
            invoice(issue="2026-08-02", due="2026-08-01")

    def test_model_provenance_requires_name_and_confidence(self):
        with self.assertRaises(ValidationError):
            ExtractionMetadata(source=ExtractionSource.SMALL_MODEL, model_name="tiny")
        with self.assertRaises(ValidationError):
            ExtractionMetadata(
                source=ExtractionSource.SMALL_MODEL,
                confidence="0.9",
            )
        metadata = ExtractionMetadata(
            source=ExtractionSource.SMALL_MODEL,
            confidence="0.94",
            grounded=True,
            model_name="layoutlmv3-invoice",
        )
        self.assertEqual(metadata.confidence, Decimal("0.94"))


class ReconciliationTests(unittest.TestCase):
    def test_unverified_documents_cannot_enter_reconciliation(self):
        unverified = Invoice(
            "INV-X",
            "Acme",
            "10.00",
            "2026-08-01",
            "2026-08-31",
            AP,
            approved=True,
        )
        with self.assertRaises(ValidationError):
            reconcile([unverified], [])

    def test_unapproved_model_result_cannot_change_a_balance(self):
        model_metadata = ExtractionMetadata(
            source=ExtractionSource.SMALL_MODEL,
            confidence="0.99",
            grounded=True,
            model_name="test-model",
        )
        unapproved = Invoice(
            "INV-X",
            "Acme",
            "10.00",
            "2026-08-01",
            "2026-08-31",
            AP,
            extraction=model_metadata,
            approved=False,
        )
        with self.assertRaises(ValidationError):
            reconcile([unapproved], [])

    def test_explicit_reference_matches_before_unreferenced_receipt(self):
        invoices = [invoice(amount="100.00")]
        receipts = [
            receipt(number="NO-REF", amount="100.00"),
            receipt(number="WITH-REF", amount="100.00", reference="inv100"),
        ]
        report = reconcile(invoices, receipts)
        self.assertEqual(report.receipts[0].status, ReceiptStatus.NEEDS_REVIEW)
        self.assertEqual(report.receipts[1].status, ReceiptStatus.MATCHED)
        self.assertEqual(
            report.receipts[1].allocations[0].method,
            MatchMethod.EXPLICIT_REFERENCE,
        )

    def test_multiple_partial_payments_close_one_invoice(self):
        report = reconcile(
            [invoice(amount="100.00")],
            [
                receipt(number="R-1", amount="40.00", reference="INV-100"),
                receipt(number="R-2", amount="60.00", reference="INV-100"),
            ],
        )
        result = report.invoices[0]
        self.assertEqual(result.status, InvoiceStatus.PAID)
        self.assertEqual(result.paid_amount, Decimal("100.00"))
        self.assertEqual(len(result.allocations), 2)

    def test_partial_payment_leaves_invoice_open_balance(self):
        report = reconcile(
            [invoice(amount="100.00")],
            [receipt(amount="40.00", reference="INV-100")],
        )
        result = report.invoices[0]
        self.assertEqual(result.status, InvoiceStatus.PARTIALLY_PAID)
        self.assertEqual(result.outstanding_amount, Decimal("60.00"))

    def test_overpayment_is_not_allocated_or_marked_paid(self):
        report = reconcile(
            [invoice(amount="100.00")],
            [receipt(amount="125.00", reference="INV-100")],
        )
        result = report.receipts[0]
        self.assertEqual(result.status, ReceiptStatus.NEEDS_REVIEW)
        self.assertEqual(result.allocated_amount, Decimal("0.00"))
        self.assertEqual(result.unallocated_amount, Decimal("125.00"))
        self.assertEqual(report.invoices[0].status, InvoiceStatus.OPEN)
        self.assertEqual(report.invoices[0].outstanding_amount, Decimal("100.00"))

    def test_payment_before_invoice_issue_fails_closed(self):
        report = reconcile(
            [invoice(issue="2026-08-10")],
            [receipt(paid="2026-08-09", reference="INV-100")],
        )
        self.assertEqual(report.receipts[0].status, ReceiptStatus.NEEDS_REVIEW)
        self.assertEqual(report.invoices[0].paid_amount, Decimal("0.00"))

    def test_weighted_fallback_accepts_one_strong_candidate(self):
        report = reconcile(
            [
                invoice(number="A", counterparty="Acme Supplies", amount="100.00"),
                invoice(number="B", counterparty="Different Vendor", amount="250.00"),
            ],
            [receipt(reference=None)],
        )
        result = report.receipts[0]
        self.assertEqual(result.status, ReceiptStatus.MATCHED)
        self.assertEqual(result.allocations[0].invoice_number, "A")
        self.assertEqual(result.allocations[0].method, MatchMethod.WEIGHTED_FALLBACK)

    def test_unreferenced_partial_amount_requires_review(self):
        report = reconcile(
            [invoice(amount="100.00")],
            [receipt(amount="1.00", reference=None)],
        )
        self.assertEqual(report.receipts[0].status, ReceiptStatus.NEEDS_REVIEW)
        self.assertEqual(report.invoices[0].paid_amount, Decimal("0.00"))

    def test_ambiguous_fallback_fails_closed(self):
        report = reconcile(
            [
                invoice(number="A", counterparty="Acme Supplies", amount="100.00"),
                invoice(number="B", counterparty="Acme Supplies", amount="100.00"),
            ],
            [receipt(reference=None)],
        )
        self.assertEqual(report.receipts[0].status, ReceiptStatus.NEEDS_REVIEW)
        self.assertEqual(len(report.allocations), 0)
        self.assertIn("ambiguity", report.receipts[0].reason)

    def test_reference_counterparty_conflict_fails_closed(self):
        report = reconcile(
            [invoice()],
            [receipt(counterparty="Unrelated Company", reference="INV-100")],
        )
        self.assertEqual(report.receipts[0].status, ReceiptStatus.NEEDS_REVIEW)

    def test_duplicate_receipt_cannot_double_count_cash(self):
        with self.assertRaises(ValidationError):
            reconcile(
                [invoice()],
                [receipt(number="R-1"), receipt(number="r1")],
            )


class CashFlowTests(unittest.TestCase):
    def test_missing_due_date_is_not_overdue(self):
        report = reconcile([invoice(due=None)], [])
        summary = summarize_cash_flow(report, as_of="2026-08-30")
        self.assertEqual(summary.overdue_payables, Decimal("0.00"))

    def test_ap_ar_summary_for_small_business(self):
        invoices = [
            invoice(number="BILL-1", amount="100.00", side=AP, due="2026-08-25"),
            invoice(
                number="SALE-1",
                counterparty="Corner Customer",
                amount="250.00",
                side=AR,
                due="2026-09-15",
            ),
        ]
        receipts = [
            receipt(number="PAID-1", amount="40.00", side=AP, reference="BILL-1"),
            receipt(
                number="COLLECTED-1",
                counterparty="Corner Customer",
                amount="250.00",
                side=AR,
                reference="SALE-1",
            ),
        ]
        summary = summarize_cash_flow(
            reconcile(invoices, receipts), as_of="2026-08-30"
        )
        self.assertEqual(summary.cash_paid, Decimal("40.00"))
        self.assertEqual(summary.cash_received, Decimal("250.00"))
        self.assertEqual(summary.net_cash_movement, Decimal("210.00"))
        self.assertEqual(summary.outstanding_payables, Decimal("60.00"))
        self.assertEqual(summary.overdue_payables, Decimal("60.00"))
        self.assertEqual(summary.outstanding_receivables, Decimal("0.00"))

    def test_future_payment_does_not_change_as_of_balance(self):
        report = reconcile(
            [invoice(amount="100.00", issue="2026-08-01", due="2026-09-30")],
            [receipt(amount="100.00", paid="2026-09-10", reference="INV-100")],
        )
        summary = summarize_cash_flow(report, as_of="2026-08-30")
        self.assertEqual(summary.cash_paid, Decimal("0.00"))
        self.assertEqual(summary.outstanding_payables, Decimal("100.00"))


class RoutingTests(unittest.TestCase):
    def metadata(self, confidence="0.95", grounded=True):
        return ExtractionMetadata(
            source=ExtractionSource.SMALL_MODEL,
            confidence=confidence,
            grounded=grounded,
            model_name="layoutlmv3-invoice",
        )

    def test_accepts_grounded_agreed_local_result(self):
        decision = decide_small_first_route(
            RoutingSignals(
                extraction=self.metadata(),
                ocr_quality="0.90",
                validation_passed=True,
                heuristic_model_agreement=True,
            )
        )
        self.assertEqual(decision.action, RoutingAction.ACCEPT)
        self.assertTrue(decision.stayed_local)

    def test_disagreement_escalates_when_available(self):
        decision = decide_small_first_route(
            RoutingSignals(
                extraction=self.metadata(),
                ocr_quality="0.90",
                validation_passed=True,
                heuristic_model_agreement=False,
            )
        )
        self.assertEqual(decision.action, RoutingAction.ESCALATE)
        self.assertFalse(decision.stayed_local)

    def test_failed_gate_without_escalation_goes_to_human(self):
        decision = decide_small_first_route(
            RoutingSignals(
                extraction=self.metadata(grounded=False),
                ocr_quality="0.90",
                validation_passed=True,
                heuristic_model_agreement=True,
                escalation_available=False,
            )
        )
        self.assertEqual(decision.action, RoutingAction.HUMAN_REVIEW)

    def test_no_heuristic_opinion_requires_stricter_confidence(self):
        decision = decide_small_first_route(
            RoutingSignals(
                extraction=self.metadata(confidence="0.95"),
                ocr_quality="0.95",
                validation_passed=True,
                heuristic_model_agreement=None,
            )
        )
        self.assertEqual(decision.action, RoutingAction.ESCALATE)

    def test_manual_validated_entry_does_not_pretend_model_ran(self):
        decision = decide_small_first_route(
            RoutingSignals(
                extraction=ExtractionMetadata(source=ExtractionSource.MANUAL),
                ocr_quality=None,
                validation_passed=True,
                heuristic_model_agreement=None,
            )
        )
        self.assertEqual(decision.action, RoutingAction.ACCEPT)
        self.assertEqual(decision.accepted_source, ExtractionSource.MANUAL)


if __name__ == "__main__":
    unittest.main()
