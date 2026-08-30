from decimal import Decimal
import unittest

from invoiceagent import (
    OcrDocument,
    RoutingAction,
    decide_small_first_route,
)
from invoiceagent.extraction import (
    EntitySpan,
    InvoiceNumberResult,
    decode_bio_spans,
    extract_anchored_identifier,
    is_valid_invoice_identifier,
    normalize_pixel_boxes,
)


class OcrTests(unittest.TestCase):
    def test_pixel_boxes_are_normalized(self):
        self.assertEqual(
            normalize_pixel_boxes([(10, 20, 30, 60)], 100, 100),
            ((100, 200, 300, 600),),
        )

    def test_ocr_requires_aligned_words_and_boxes(self):
        with self.assertRaises(ValueError):
            OcrDocument(["Invoice"], [], "0.9")


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.words = ("Invoice", "No", "INV", "-", "204", "Total", "500.00")
        self.boxes = tuple((index * 100, 10, index * 100 + 80, 60) for index in range(7))

    def test_anchored_rule_and_identifier_validation(self):
        self.assertEqual(extract_anchored_identifier(self.words), "INV-204")
        self.assertTrue(is_valid_invoice_identifier("INV-204"))
        self.assertFalse(is_valid_invoice_identifier("2026/08/30"))
        self.assertFalse(is_valid_invoice_identifier("500.00"))

    def test_bio_decoder_scores_only_entity_tokens(self):
        labels = ("O", "O", "LABEL_1", "LABEL_2", "LABEL_2", "O", "O")
        confidences = ("0.999", "0.999", "0.96", "0.93", "0.95", "0.999", "0.999")
        margins = ("0.99", "0.99", "0.70", "0.60", "0.65", "0.99", "0.99")
        spans = decode_bio_spans(self.words, self.boxes, labels, confidences, margins)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].value, "INV-204")
        self.assertEqual(spans[0].minimum_confidence, Decimal("0.93"))
        self.assertNotEqual(spans[0].mean_confidence, Decimal("0.999"))

    def test_multiple_disjoint_spans_are_ambiguous(self):
        labels = ("O", "O", "B-INVOICE", "O", "O", "B-INVOICE", "O")
        spans = decode_bio_spans(
            self.words,
            self.boxes,
            labels,
            ("0.9",) * 7,
            ("0.5",) * 7,
        )
        self.assertEqual(len(spans), 2)

    def test_orphan_inside_tag_is_rejected(self):
        spans = decode_bio_spans(
            self.words,
            self.boxes,
            ("O", "O", "I-INVOICE", "O", "O", "O", "O"),
            ("0.9",) * 7,
            ("0.5",) * 7,
        )
        self.assertEqual(spans, ())

    def test_unrelated_claim_is_not_grounded_by_valid_indices(self):
        fake = EntitySpan(
            value="FAKE-999",
            word_indices=(2, 3, 4),
            boxes=self.boxes[2:5],
            token_confidences=(Decimal("0.99"),) * 3,
            token_margins=(Decimal("0.8"),) * 3,
        )
        result = InvoiceNumberResult((fake,), "ryan-adapter", Decimal("42.0"))
        ocr = OcrDocument(self.words, self.boxes, "0.92")
        signals = result.routing_signals(
            ocr, heuristic_candidate=None, escalation_available=False
        )
        self.assertFalse(signals.extraction.grounded)
        self.assertEqual(
            decide_small_first_route(signals).action,
            RoutingAction.HUMAN_REVIEW,
        )

    def test_result_routes_grounded_agreement(self):
        span = EntitySpan(
            value="INV-204",
            word_indices=(2, 3, 4),
            boxes=self.boxes[2:5],
            token_confidences=(Decimal("0.96"), Decimal("0.95"), Decimal("0.97")),
            token_margins=(Decimal("0.7"), Decimal("0.6"), Decimal("0.7")),
        )
        result = InvoiceNumberResult((span,), "ryan-adapter", Decimal("42.0"))
        ocr = OcrDocument(self.words, self.boxes, "0.92")
        signals = result.routing_signals(
            ocr, heuristic_candidate="inv204", escalation_available=True
        )
        decision = decide_small_first_route(signals)
        self.assertEqual(decision.action, RoutingAction.ACCEPT)

    def test_missing_span_goes_to_human_when_escalation_unavailable(self):
        result = InvoiceNumberResult((), "ryan-adapter", Decimal("41.0"))
        ocr = OcrDocument(self.words, self.boxes, "0.92")
        decision = decide_small_first_route(
            result.routing_signals(
                ocr, heuristic_candidate=None, escalation_available=False
            )
        )
        self.assertEqual(decision.action, RoutingAction.HUMAN_REVIEW)


if __name__ == "__main__":
    unittest.main()
